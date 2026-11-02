# TRELLIS+ : quality mods for TRELLIS.2 / Pixal3D in ComfyUI

**Live sample viewer: https://sh1ftmaker.github.io/trellis-plus/** (input images vs. generated models, baseline vs. mods)

A ComfyUI node pack plus a headless pipeline that sits on top of ComfyUI's native TRELLIS.2 / Pixal3D nodes and
targets the three weak spots of image-to-3D output: mesh cleanliness, topology, and texture fidelity.
Install: clone into `ComfyUI/custom_nodes/` (or symlink `trellis_plus/`), then `pip install pymeshlab pynanoinstantmeshes`.

## Local layout (author's machine)

```
models/            weights (single copy, ~24 GB), registered in ComfyUI via extra_model_paths.yaml
  diffusion_models/  pixal3d_bf16, trellis_2_bf16
  vae/               trellis_2_shape_vae_bf16, trellis_2_texture_vae_bf16
  clip_vision/       dino_v3_L_naf_fp32
  geometry_estimation/ moge_2_vitl_normal_fp16      background_removal/ birefnet
trellis_plus/      ComfyUI node pack (symlinked as ComfyUI/custom_nodes/trellis_plus)
tools/pipeline.py  headless runner (baseline / plus presets, optional quad retopo)
tools/evaluate.py  topology metrics + turntable render sheet for GLB/OBJ outputs
tools/gen_test_image.py  make a test input with the local Qwen-Image 2.1
workflows/         API-format prompts (drag into ComfyUI to load)
outputs/ reports/  results and comparison sheets
```

ComfyUI lives at `/bigdata/agent-tom/qwen-image/ComfyUI` (TRELLIS.2 is native there, no extra install).
`extra_model_paths.yaml` in that folder points ComfyUI at `models/` here.

## Running

```bash
# start ComfyUI (if the qwen-image service isn't already running it)
sbg start comfy --mem 20G -- /bigdata/agent-tom/qwen-image/venv/bin/python /bigdata/agent-tom/qwen-image/ComfyUI/main.py --listen 127.0.0.1 --port 8188 --reserve-vram 2 --cache-none   # cache-none: dense meshes in the output cache exceed RAM

PY=/bigdata/agent-tom/qwen-image/venv/bin/python
$PY tools/pipeline.py photo.png                    # quality preset (Pixal3D)
$PY tools/pipeline.py photo.png --quad --quad-faces 20000   # + quad retopology, OBJ with quads
$PY tools/pipeline.py photo.png --preset baseline  # stock Comfy-Org template settings
$PY tools/pipeline.py photo.png --model trellis2   # TRELLIS.2 instead of Pixal3D (no photo projection)
$PY tools/evaluate.py outputs/<run>/*.glb --sheet reports/x.png
```

## Nodes (category `3d/trellis_plus`)

| Node | What it fixes |
|---|---|
| **Mesh Repair** | Raw decoder output has inconsistent winding, duplicate and non-manifold faces and floating fragments. pymeshlab repair pass; makes winding consistent so the native SDF remesh / QEF sharp-feature mode becomes usable. |
| **Isotropic Remesh** | QEM decimation leaves slivers and fans. Feature-preserving isotropic remesh (split / collapse / flip / relax + reprojection) at about the same face count gives even, near-equilateral triangles that unwrap, bake and deform better. |
| **Quad Remesh (Instant Meshes)** | Pure-quad retopology with a 4-RoSy/4-PoSy field, vertices snapped back onto the dense surface. Optional crease alignment for hard-surface. Output is triangulated for glTF in quad order. |
| **Save Quad OBJ** | Restores the exact quads (triangle pairs survive UnwrapMesh in order, verified per pair; greedy fallback otherwise), never across a UV seam. Writes MTL + base color / normal / ORM textures. |
| **Project Input Photo onto Texture** | Pixal3D is pixel-aligned with its input camera, so the full-res photo is projected onto visible texels: ray-cast occlusion, facing-angle and edge feathering, silhouette-IoU auto-alignment (skips if the fit is poor). `detail` mode transfers photo high-frequencies; `replace` blends in the colour-matched photo. |
| **Texture Enhance** | Chart-aware unsharp mask (coverage-normalised blur so gutters don't leak in) + JFA gutter re-fill. |

The `plus` preset also changes sampling (more steps, Heun on the 1536 refinement and texture stages) and
remeshing (1024 grid, QEF, inverted/enclosed shell removal, pole fixing, light Taubin smoothing, project-back)
and decimates with feature-edge-weighted QEM.

## Samples

`samples/` holds web versions of pipeline runs (meshopt-compressed GLB, 2K WebP textures, full topology), exported
with `tools/publish.py`; full-res and raw meshes are not committed. Test inputs: `viking_wolf_rune_axe.png` from
[Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates) (MIT); `goblin_merchant.png`
generated locally with Qwen-Image 2.1.

Built on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT), Pixal3D, and ComfyUI's native integration.
