# Local3D+: run Pixal3D locally, with quality mods

**Live sample viewer: https://sh1ftmaker.github.io/local3d-plus/** (input photos vs generated models, stock vs mods,
low-poly LODs; works on phones)

[**Pixal3D**](https://github.com/TencentARC/Pixal3D) (SIGGRAPH 2026, Tsinghua University & Tencent ARC Lab) is the
best open image-to-3D model we have tested. It lifts pixel features into 3D by back-projection, so the generated
asset is *pixel-aligned* with the input photo. Local3D+ is an **unofficial** community toolkit for running Pixal3D
locally in ComfyUI, through ComfyUI's native Pixal3D / TRELLIS.2 nodes. It adds a node pack aimed at what the raw
output still lacks: clean meshes, game-friendly topology and triangle budgets, and textures that match the photo.

> Not affiliated with the Pixal3D or TRELLIS.2 authors. All credit for the models goes to them; see Credits.

## What it adds

| Node (category `3d/local3d_plus`) | What it fixes |
|---|---|
| **Mesh Repair** | Raw decoder output has inconsistent winding, duplicate and non-manifold faces, and floating fragments (one asset had 325k non-manifold edges). A pymeshlab repair pass makes the winding consistent so SDF / QEF remeshing works. |
| **Isotropic Remesh** | Turns the slivers and fans left by decimation into even, near-equilateral triangles (mean triangle quality 0.81 → 0.86). |
| **Low-Poly LOD** | Shrink-wraps the model into one watertight skin (CGAL alpha wrap), decimates to an exact triangle budget, re-unwraps, and re-bakes base color, roughness/metallic/AO and a normal map from the full-resolution model. At 5k triangles (about 1% of the original) photo fidelity drops by 0.1 dB or less on our samples. |
| **Quad Remesh (Instant Meshes)** + **Save Quad OBJ** | Pure-quad retopology snapped back onto the surface. The OBJ export restores the exact quads and never merges across a UV seam. |
| **Project Input Photo onto Texture** | Uses Pixal3D's pixel alignment to project the full-resolution photo onto visible texels, with ray-cast occlusion, facing and edge feathering, and silhouette-IoU auto-alignment. `replace` mode measurably improves photo fidelity. |
| **Texture Enhance** | Chart-aware unsharp mask plus gutter re-fill. Off by default: it did not improve photo fidelity in tests. |

Measured results, including those where a mod did *not* help, are on the viewer's stats tab and in
`samples/*/*/run.json`.

## Use

Requirements: ComfyUI ≥ 0.34 with the native Pixal3D nodes, plus these weights from
[Comfy-Org/Pixal3D](https://huggingface.co/Comfy-Org/Pixal3D), [Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe)
and [Comfy-Org/BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet): `pixal3d_bf16` (or `_int8_convrot`),
`trellis_2_shape_vae_bf16`, `trellis_2_texture_vae_bf16`, `dino_v3_L_naf_fp32`, `moge_2_vitl_normal_fp16` and
`birefnet`. Put them in ComfyUI's model folders or in any folder registered in `extra_model_paths.yaml`.

```bash
git clone https://github.com/sh1ftmaker/local3d-plus
ln -s "$PWD/local3d-plus/trellis_plus" ComfyUI/custom_nodes/local3d_plus
pip install pymeshlab pynanoinstantmeshes trimesh

# headless runs against a local ComfyUI on :8188
python tools/pipeline.py photo.png                          # Local3D+ preset
python tools/pipeline.py photo.png --preset baseline        # stock template settings
python tools/pipeline.py photo.png --quad --quad-faces 20000
python tools/pipeline.py photo.png --set project_mode=replace --set sharpen=0   # override any preset value
python tools/lowpoly.py outputs/<run> --budgets 5000 20000  # low-poly LODs from a finished run
python tools/split_sheets.py sheets/*.png --out inputs/   # cut multi-subject sheets into single inputs
python tools/evaluate.py outputs/<run>/*.glb --sheet sheet.png --front front/
python tools/publish.py                                     # export runs to samples/ for the viewer
```

Memory: bf16 Pixal3D plus the dense intermediate meshes need about 20 GB of system RAM. Run ComfyUI with
`--cache-none`, or use the INT8 checkpoint on smaller machines.

## Repo layout

```
trellis_plus/    ComfyUI node pack (folder name kept for compatibility; node ids are TrellisPlus*)
tools/           pipeline.py, lowpoly.py, split_sheets.py, gen_sheets.sh, evaluate.py, publish.py, gen_test_image.py, web/ (gltf-transform)
site/            static viewer (GitHub Pages)
samples/         web-optimised sample runs + index.json (meshopt geometry, WebP textures)
```

## Credits

- **Pixal3D** by Dong-Yang Li, Wang Zhao, Yuxin Chen, Wenbo Hu, Meng-Hao Guo, Fang-Lue Zhang, Ying Shan and Shi-Min Hu:
  [code](https://github.com/TencentARC/Pixal3D) (MIT), [project page](https://ldyang694.github.io/projects/pixal3d/)
- **TRELLIS.2** (Microsoft, MIT), the backbone Pixal3D builds on: https://github.com/microsoft/TRELLIS.2
- **ComfyUI / Comfy-Org** for the native Pixal3D + TRELLIS.2 integration and the repackaged weights.
- Instant Meshes (via pynanoinstantmeshes), PyMeshLab, MoGe, BiRefNet, three.js and glTF-Transform.
- Input sheets (`samples/reports/sheet*.jpg`): generated with Codex's image tool; `tools/split_sheets.py` cuts them into single-subject inputs.
- Test inputs: `viking_wolf_rune_axe.png` from [Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates)
  (MIT); `goblin_merchant.png` was generated locally with Qwen-Image 2.1.

MIT licensed.
