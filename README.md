# ComfyUI-Kido-Mesh-Watertight

A **watertight mesh reconstruction node for ComfyUI that runs on Linux** — no Windows
prebuilt `.pyd`, no native build step.

It takes the triangle soup that AI 3D generators (TRELLIS 2 / Pixal3D / Hunyuan-style
pipelines) emit and returns a **closed surface**: every edge is shared by exactly two
faces, i.e. `0 boundary edges` after a weld-by-position audit. That is the property
decimation, UV unwrapping, baking, rigging and game-engine import all depend on.

```
input soup (fwizard2, 692,754 tris)      output (this node, res 512)
  RAW     boundary 216,636                  RAW     boundary 0
  WELDED  boundary     432                  WELDED  boundary 0
  WELDED  nonmanifold 5,207                 WELDED  nonmanifold 258
  WELDED  shells         31                 WELDED  shells         1
  winding inconsistent                      winding consistent
  => NOT watertight                         => WATERTIGHT
```

...and it **stays closed after CuMesh decimation** (1,760,866 → 296,534 faces,
boundary 0, non-manifold 99), which is where most repair pipelines fall apart.

---

## Why this exists

The popular `WTiVo - Mesh Watertight` node (`WTiVoNativeMeshToMesh`) is Windows-only:
its `CMakeLists.txt` hard-fails on non-Windows (`if(NOT WIN32) message(FATAL_ERROR …)`)
and the repo ships only `*.cp312-win_amd64.pyd` + `.dll` binaries. On a Linux box the
node cannot load, ComfyUI silently drops the branch, and the workflow reports success
while producing a mesh that is still not watertight.

This pack implements the **same node class name with the same widgets**, so a workflow
built around the Windows node validates and runs unchanged — the reconstruction just
happens with tooling that exists on Linux.

## How it works

```
mesh
 └─ (optional) cumesh CUDA repair ........ dedupe / degenerate / non-manifold faces,
                                           non-manifold edges, fill holes, unify winding
 └─ cuBVH unsigned distance field ........ UDF sampled on a resolution^3 grid over [-1,1]^3
 └─ shell band + border flood fill ....... band = {udf < iso}; everything unreachable from
                                           the grid border through free space = SOLID
 └─ level set ........................... field = udf - iso everywhere, EXCEPT a flat
                                           plateau (-iso/4) deep inside the solid
 └─ marching cubes ...................... one zero crossing => one closed surface
```

The plateau is the important part. If you sign-flip the distance by region
(`-(udf - iso)` inside the solid), the band's *inner* wall becomes a second zero crossing
and you get a double-walled shell — twice the triangles, self-intersections, and an
inflated volume. Keeping the deep interior at a constant negative value leaves exactly
one crossing: the boundary of the solid region.

## Requirements

| | |
|---|---|
| ComfyUI | recent build with typed `MESH` (`comfy_api.latest.Types.MESH`) |
| Python packages | `numpy`, `scipy`, `scikit-image`, `torch` |
| GPU path | `cumesh` (CUDA) — used for the UDF and repair. Built from source here: [visualbruno/CuMesh](https://github.com/visualbruno/CuMesh) |
| CPU fallback | `trimesh` — used automatically when CUDA/`cumesh` is unavailable (slower, coarser) |

No compilation of this node is required. If `cumesh` is missing the node still runs
through the CPU voxelise path, so it installs on any box.

## Install

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/<you>/ComfyUI-Kido-Mesh-Watertight.git
# restart ComfyUI
```

Or copy the two files (`__init__.py`, `nodes.py`) into
`<ComfyUI>/custom_nodes/ComfyUI-Kido-Mesh-Watertight/` and restart.
Verify registration:

```bash
curl -s localhost:8188/object_info | python3 -c \
 "import json,sys; d=json.load(sys.stdin); print([k for k in d if 'Kido' in k or k=='WTiVoNativeMeshToMesh'])"
# ['KidoMeshTopologyAudit', 'KidoMeshWatertight', 'WTiVoNativeMeshToMesh']
```

## Nodes

### `KidoMeshWatertight` — "Kido - Mesh Watertight (level set)"

| widget | default | meaning |
|---|---|---|
| `resolution` | 512 | voxel grid resolution. ~15 s at 512³; 768/1024 resolve finer detail (more RAM/VRAM) |
| `iso_voxels` | 1.0 | how far outside the source surface the band sits, in voxels |
| `seal_iterations` | 0 | morphological closing of the band. Raise to 2–3 when the source is an **open surface** (cloth/paper-style meshes), so the flood fill cannot leak inside |
| `smooth_sigma` | 0.0 | cheap `[1,2,1]` smoothing of the level set (`1`–`2` softens voxel-scale crinkle) |
| `auto_repair` | true | run the cumesh repair pass first |
| `max_hole_perimeter` | 0.05 | largest hole (normalized units) cumesh may fill |
| `min_component_voxels` | 0 | drop solid components smaller than this (0 = keep everything, e.g. hair cards) |
| `unload_models` | true | free ComfyUI models/tensors before allocating |

### `KidoMeshTopologyAudit` — "Kido - Mesh Topology Audit"

Passthrough mesh + the numbers, so the graph can prove its own claim instead of
asserting it:

outputs `mesh, boundary_edges, nonmanifold_edges, shells, watertight (BOOLEAN), report (STRING)`

Counts are computed after welding vertices by position (rounded to `weld_precision`
decimals) — never on the raw glTF index buffer, which splits every UV seam and makes
any mesh look like a triangle soup.

### `KidoMeshLoader` — "Kido - Load Mesh (MESH from file)"

`.glb/.gltf/.obj/.ply/.stl` → native `MESH`, geometry only (no UVs/materials — the
reconstruction replaces both anyway). Exists so a watertight run needs **no** BrainDead /
Trellis2 stack installed: a clean ComfyUI plus this pack is enough.

### `KidoSaveMesh` — "Kido - Save Mesh / File3D"

Writes a `MESH` **or any runtime `FILE_3D` object** to `output/<prefix>_NNNNN_.glb`.

Core `SaveGLB` assumes a MESH: anything else falls into its mesh branch and dies with
`AttributeError: '_BakedFile3D' object has no attribute 'vertices'` (see
`comfy_extras/nodes_save_3d.py`). That is why the author's texturing/baking workflows never
write a file — the bake node's output has no route to disk without this node.

### `WTiVoNativeMeshToMesh` — "WTiVo - Mesh Watertight (Kido/Linux)"

Drop-in for the Windows node: identical class name, identical widget names
(`input_res`, `final_res`, `proxy_points`, `proxy_eps_scale`, `proxy_feature_weight`,
`lambda_fill`, `threads`, `thin_iso_vox`, `faithc_component_mode`). The widget mapping:

| WTiVo widget | maps to |
|---|---|
| `input_res` | grid `resolution` (clamped to 1024) |
| `proxy_eps_scale` + `thin_iso_vox` | `iso_voxels` |
| `lambda_fill` | `seal_iterations` (÷10) |
| `faithc_component_mode="largest"` | drop solid components < 0.01 % of the grid |

## Run it

Two workflows ship in [`workflows/`](workflows/):

- `kido_watertight.json` — UI format. Open it in the ComfyUI browser interface
  (`Workflow → Open`), point `BD_LoadMesh.file_path` at a mesh, hit Queue.
- `kido_watertight_api.json` — API format, same graph, for headless runs:

```bash
python workflows/run_api_prompt.py workflows/kido_watertight_api.json   # submit + poll + report
```

`workflows/ui_wf_from_api.py` and `workflows/ui_to_api.py` convert between the two
formats using the server's own `/object_info`, so widget order is never guessed.

Graph: `BD_LoadMesh → BD_TrimeshToMesh → KidoMeshWatertight → KidoMeshTopologyAudit
→ CuMeshGeometryDecimate → SaveGLB` (both the raw watertight and the decimated mesh
are saved).

## Verified on

- Linux, 2× RTX 3090 (24 GB), ComfyUI 0.3xx, Python 3.13, `cumesh` built from source
- 692,754-triangle female-wizard soup: **boundary 0 / non-manifold 258 / shells 1 /
  winding consistent**, ~15 s at res 512 in-process
- decimation afterwards keeps it closed (296,534 faces, boundary 0)
- `iso=1.0, seal=0, smooth=0` preserves the silhouette best; `iso=1.5, seal=2` erodes
  the hem slightly

## Known limits

- **Geometry only — no UVs, no textures.** The reconstruction replaces the mesh, so any
  atlas is gone by construction (same as the Windows node). Re-texture afterwards with
  an unwrap + bake pass, or transfer UVs from the source by proximity.
- Watertight means *closed*, not *manifold*: a handful of non-manifold edges
  (≈100–260 on a 1.7 M-triangle output) survive where marching cubes touches itself.
  They are reported by the audit node and are tolerable for engines; decimate low if
  you need fewer.
- Very thin features (hair cards, cloth layers) are sealed into slabs rather than
  solidified — expected, and how voxel/level-set reconstruction behaves generally.
- `resolution` 1024 needs ≈4.3 GB of system RAM for the distance volume; the UDF is
  computed in slabs so VRAM stays modest.

## License

MIT — see [LICENSE](LICENSE).

Not affiliated with WTiVo / CelloCut / MostAadTech. The `WTiVoNativeMeshToMesh` class
name is re-implemented here only so existing workflows can load on Linux; no WTiVo
binaries or code are redistributed.
