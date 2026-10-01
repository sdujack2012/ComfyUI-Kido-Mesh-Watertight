#!/usr/bin/env python3
"""Convert a ComfyUI UI-format workflow JSON into API-format prompt JSON.

ComfyUI's /prompt endpoint only accepts API format ({node_id: {class_type, inputs}}), while
published workflows ship in UI format (nodes/links/widgets_values). The fiddly part is widget
name mapping, so it is derived from the server's own /object_info rather than guessed:

  * widget-eligible inputs = required then optional, in declared order, keeping only types that
    are actually widgets (scalar types or COMBO lists). Real link types (IMAGE, MESH, MODEL,
    VAE, LATENT, ...) are never widgets.
  * widgets_values align to that list in order. A widget that the graph converted to an input
    KEEPS its value in widgets_values, so alignment must still consume it - and then the link
    overrides it (this is exactly the Trellis2MeshEncoder.mesh_filename case).
  * extra trailing widgets_values (e.g. LoadImage's hidden 'upload' widget) are ignored.

Usage: ui_to_api.py <workflow.json> <object_info.json> [out.json]
"""
import json
import sys

SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN"}
# ComfyUI does not report every widget as a scalar: combo inputs arrive as the bare string
# "COMBO" (choices live outside the spec) and colour pickers as "COLOR". Missing these silently
# drops the value, and the server then rejects the node with "required input is missing" - so a
# plain SCALARS allowlist is not enough.
WIDGET_TYPES = SCALARS | {"COMBO", "COLOR"}


def widget_names(spec):
    """Ordered (name, spec_entry) for widget-eligible inputs of a node class."""
    out = []
    for section in ("required", "optional"):
        for name, s in (spec.get(section) or {}).items():
            t = s[0] if isinstance(s, (list, tuple)) and s else s
            if isinstance(t, list):          # COMBO of choices -> widget
                out.append((name, s))
            elif isinstance(t, str) and t in WIDGET_TYPES:
                out.append((name, s))
    return out


def default_of(entry):
    if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict):
        return entry[1].get("default", None)
    return None


# Older workflows store KSampler-style seed controls ("randomize"/"fixed"/...) as a widget value.
# This ComfyUI no longer exposes that as an input, so leaving it in place shifts EVERY later value
# by one - steps became "randomize", cfg became steps, and so on. Drop them.
CONTROL_VALUES = {"randomize", "fixed", "increment", "decrement"}


def convert(ui, objinfo):
    links = {}
    for l in ui.get("links", []) or []:
        if isinstance(l, list) and len(l) >= 5:
            links[l[0]] = (str(l[1]), l[2])          # link_id -> (origin_node, origin_slot)
    prompt, skipped = {}, []
    for n in ui.get("nodes", []):
        ct = n.get("type")
        if ct not in objinfo:
            skipped.append((n.get("id"), ct))
            continue
        spec = objinfo[ct].get("input", {}) or {}
        names = widget_names(spec)
        wv = n.get("widgets_values")
        wv = wv if isinstance(wv, list) else []
        wv = [v for v in wv if not (isinstance(v, str) and v in CONTROL_VALUES)]
        inputs = {}
        for i, (name, entry) in enumerate(names):
            if i < len(wv):
                inputs[name] = wv[i]
            else:
                # Inputs added to a node AFTER the workflow was authored are absent from
                # widgets_values; fill the declared default so validation does not fail on
                # "required input is missing".
                d = default_of(entry)
                if d is not None:
                    inputs[name] = d
        for inp in n.get("inputs", []) or []:
            lid = inp.get("link")
            if lid is None:
                continue
            if lid in links:
                inputs[inp["name"]] = list(links[lid])
                inputs.pop(inp["name"] + "_widget", None)
        # a link wins over the widget value it converted
        for inp in n.get("inputs", []) or []:
            if inp.get("link") is None and inp.get("widget"):
                pass
        prompt[str(n["id"])] = {"class_type": ct, "inputs": inputs,
                                "_meta": {"title": n.get("title") or ct}}
    return prompt, skipped


def main():
    ui = json.load(open(sys.argv[1]))
    obj = json.load(open(sys.argv[2]))
    prompt, skipped = convert(ui, obj)
    if len(sys.argv) > 3:
        json.dump(prompt, open(sys.argv[3], "w"), indent=1)
        print(f"wrote {sys.argv[3]}: {len(prompt)} nodes")
    print(f"converted {len(prompt)} nodes; skipped non-backend nodes: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
