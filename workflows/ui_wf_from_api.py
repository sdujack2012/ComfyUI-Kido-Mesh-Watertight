#!/usr/bin/env python3
"""Build a ComfyUI UI-format workflow from an API-format prompt, using /object_info for
widget order, input/output names and link types.

Usage: ui_wf_from_api.py <api_prompt.json> <object_info.json> <out_ui.json>
"""
import json
import sys

SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN"}
WIDGET_TYPES = SCALARS | {"COMBO", "COLOR"}


def spec_of(objinfo, ct):
    return objinfo[ct].get("input", {}) or {}


def widget_names(objinfo, ct):
    out = []
    for section in ("required", "optional"):
        for name, s in (spec_of(objinfo, ct).get(section) or {}).items():
            t = s[0] if isinstance(s, (list, tuple)) and s else s
            if isinstance(t, list) or (isinstance(t, str) and t in WIDGET_TYPES):
                out.append(name)
    return out


def input_type(objinfo, ct, name):
    for section in ("required", "optional"):
        s = (spec_of(objinfo, ct).get(section) or {}).get(name)
        if s is not None:
            t = s[0] if isinstance(s, (list, tuple)) and s else s
            return "COMBO[]" if isinstance(t, list) else t
    return "*"


def default_of(objinfo, ct, name):
    for section in ("required", "optional"):
        s = (spec_of(objinfo, ct).get(section) or {}).get(name)
        if isinstance(s, (list, tuple)) and len(s) > 1 and isinstance(s[1], dict):
            return s[1].get("default")
    return None


def main():
    api = json.load(open(sys.argv[1]))
    objinfo = json.load(open(sys.argv[2]))

    nodes = {k: v for k, v in api.items()}
    ids = {k: i + 1 for i, k in enumerate(nodes)}
    # topological-ish order: producers before consumers
    order, seen = [], set()

    def visit(k):
        if k in seen:
            return
        seen.add(k)
        for v in nodes[k]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) in nodes:
                visit(str(v[0]))
        order.append(k)

    for k in nodes:
        visit(k)

    ui_nodes, ui_links = [], []
    link_id = 0
    # map producer (node, slot) -> list of link ids, filled as we go
    produced = {}
    x, y = 60, 60
    for idx, key in enumerate(order):
        n = nodes[key]
        ct = n["class_type"]
        node_id = ids[key]
        inputs, widgets_seen = [], {}
        for name, val in n["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and str(val[0]) in nodes:
                src = str(val[0])
                slot = int(val[1])
                link_id += 1
                t = input_type(objinfo, ct, name)
                inputs.append({"name": name, "type": t, "link": link_id,
                               "widget": {"name": name} if name in widget_names(objinfo, ct) else None})
                if inputs[-1]["widget"] is None:
                    inputs[-1].pop("widget")
                ui_links.append([link_id, ids[src], slot, node_id, len(inputs) - 1, t])
                produced.setdefault((src, slot), []).append(link_id)
                widgets_seen[name] = val
        wnames = widget_names(objinfo, ct)
        widgets_values = []
        for wn in wnames:
            if wn in n["inputs"] and not isinstance(n["inputs"][wn], list):
                widgets_values.append(n["inputs"][wn])
            elif wn in widgets_seen:
                widgets_values.append(widgets_seen[wn])
            else:
                widgets_values.append(default_of(objinfo, ct, wn))
        outs = objinfo[ct].get("output") or []
        onames = objinfo[ct].get("output_name") or [f"out{i}" for i in range(len(outs))]
        if isinstance(outs, str):
            outs, onames = [outs], [onames]
        outputs = [{"name": onames[i] if i < len(onames) else f"out{i}",
                    "type": outs[i], "links": produced.get((key, i), []) or None}
                   for i in range(len(outs))]

        col, row = idx % 4, idx // 4
        ui_nodes.append({
            "id": node_id, "type": ct, "pos": [60 + col * 330, 60 + row * 190],
            "size": [300, 110], "flags": {}, "order": idx, "mode": 0,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": ct, "cnr_id": ct},
            "widgets_values": widgets_values,
        })

    ui = {
        "id": "kido-watertight", "revision": 0,
        "last_node_id": max(ids.values()), "last_link_id": link_id,
        "nodes": ui_nodes, "links": ui_links, "groups": [],
        "config": {}, "extra": {"ds": {"scale": 1.0, "offset": [0, 0]}}, "version": 0.4,
    }
    out = sys.argv[3]
    json.dump(ui, open(out, "w"), indent=2)
    print(f"wrote {out}: {len(ui_nodes)} nodes, {len(ui_links)} links")


if __name__ == "__main__":
    main()
