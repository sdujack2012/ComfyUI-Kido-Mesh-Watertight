"""Submit an API-format prompt to ComfyUI and report node_errors + outputs honestly.

Usage: python run_api_prompt.py <prompt.json> [host] [timeout_s]
"""
import json, sys, time, urllib.request, urllib.error

path = sys.argv[1]
host = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8188"
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0


def get_json(url, t=30):
    with urllib.request.urlopen(url, timeout=t) as f:
        return json.load(f)


def status_now():
    try:
        q = get_json(f"{host}/queue", 10)
        return f"running={len(q.get('queue_running', []))} pending={len(q.get('queue_pending', []))}"
    except Exception as exc:
        return f"queue unreachable ({exc})"


prompt = json.load(open(path))
payload = json.dumps({"prompt": prompt, "client_id": "kido-watertight"}).encode()

try:
    req = urllib.request.Request(f"{host}/prompt", data=payload,
                                 headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=60))
except urllib.error.HTTPError as e:
    print("SUBMIT REJECTED", e.code)
    print(e.read().decode(errors="replace")[:4000])
    sys.exit(2)

pid = resp.get("prompt_id")
print("submitted prompt_id:", pid)
print("node_errors:", json.dumps(resp.get("node_errors", {}))[:2000])
expected = [n for n in prompt
            if prompt[n]["class_type"].startswith("Save") or "Save" in prompt[n]["class_type"]]
print("expected save nodes:", expected)

t0 = time.time()
last = None
while time.time() - t0 < timeout:
    hist = {}
    try:
        hist = get_json(f"{host}/history/{pid}", 30)
    except urllib.error.HTTPError:
        pass
    if not hist:
        try:
            allh = get_json(f"{host}/history", 60)
            hist = {pid: allh[pid]} if pid in allh else {}
        except Exception:
            hist = {}
    if hist:
        entry = hist[pid]
        status = entry.get("status", {})
        print("status:", json.dumps(status)[:900])
        produced = sorted(entry.get("outputs", {}).keys())
        print("output nodes with results:", produced)
        missing = [n for n in expected if n not in produced]
        if missing:
            print("!!! branches that produced NO output:", missing)
        for n in produced:
            print(f"  {n}: {json.dumps(entry['outputs'][n])[:600]}")
        print(f"done in {time.time() - t0:.1f}s")
        sys.exit(0 if not missing else 3)
    cur = status_now()
    if cur != last:
        last = cur
        print(f"  [{time.time() - t0:6.1f}s] {cur}", flush=True)
    time.sleep(10)
print("TIMEOUT waiting for history")
sys.exit(4)
