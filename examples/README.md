# Example renders

Same character, same orbit camera, rendered by the pipeline's plain-Blender renderer.

| file | what it is |
|---|---|
| `render_fw2_src.png` | the **source** stage-2 GLB (692,754 tris) as produced by multi-view 3D fusion — 432 boundary edges survive welding, 5,207 non-manifold edges, 31 shells, inconsistent winding |
| `render_fw2_master_wt.png` | the earlier **Blender voxel-remesh** repair of that same source, kept for comparison (textured via a proximity UV transfer, hence the mottled atlas) |
| `render_c1_kido_512_1.0_0.png` | **this node**, `resolution 512, iso 1.0, seal 0, smooth 0` — untextured by construction (the reconstruction replaces geometry, so the atlas is gone) |

The node's output is geometrically identical in silhouette to the source (hat, hair, sash,
boots, both arms intact) while being a closed surface. Texture has to be re-applied
afterwards — that is a separate unwrap/bake stage, and it is the same limitation the
Windows `WTiVo - Mesh Watertight` node has.
