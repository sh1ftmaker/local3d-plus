#!/usr/bin/env python3
"""Make low-poly LODs from finished runs: outputs/<run>/ -> outputs/<run>_lod<N>k/ (picked up by publish.py).

    tools/lowpoly.py outputs/goblin_merchant_plus_pixal3d_default --budgets 5000 20000
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/bigdata/agent-tom/qwen-image/ComfyUI")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import trimesh  # noqa: E402
from comfy_extras.nodes_save_3d import save_glb  # noqa: E402
from PIL import Image  # noqa: E402
from trellis_plus.lod import make_lod  # noqa: E402


def load_textured(glb):
    sc = trimesh.load(glb, process=False)
    m = sc.to_geometry() if hasattr(sc, "to_geometry") else sc
    mat = m.visual.material
    uv = np.asarray(m.visual.uv, np.float32).copy()
    uv[:, 1] = 1.0 - uv[:, 1]                       # trimesh flips V on load; back to file / MESH convention
    t = lambda im: torch.from_numpy(np.asarray(im.convert("RGB"), np.float32) / 255.0)
    maps = {"base_color": t(mat.baseColorTexture)}
    if getattr(mat, "metallicRoughnessTexture", None) is not None:
        mr = t(mat.metallicRoughnessTexture)
        maps.update({"roughness": mr[..., 1:2], "metallic": mr[..., 2:3]})
        if getattr(mat, "occlusionTexture", None) is not None:
            maps["occlusion"] = t(mat.occlusionTexture)[..., 0:1]
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64), uv, maps


def to_pil(t):
    return Image.fromarray((t[0, ..., :3].clamp(0, 1).numpy() * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--budgets", nargs="+", type=int, default=[5000, 20000])
    ap.add_argument("--tex", type=int, default=0, help="texture size (0 = 1024 below 10k faces, else 2048)")
    a = ap.parse_args()
    for run in a.runs:
        glb = max((p for p in run.glob("*.glb") if "_raw_" not in p.name), key=lambda p: p.stat().st_mtime)
        v, f, uv, maps = load_textured(glb)
        for b in a.budgets:
            t0 = time.time()
            tex = a.tex or (1024 if b < 10000 else 2048)
            mesh, st = make_lod(v, f, uv, maps, b, tex)
            dest = run.parent / f"{run.name}_lod{b // 1000}k"
            dest.mkdir(exist_ok=True)
            save_glb(mesh.vertices[0], mesh.faces[0], str(dest / f"lod{b // 1000}k_00001_.glb"),
                     uvs=mesh.uvs[0], texture_image=to_pil(mesh.texture),
                     metallic_roughness_image=to_pil(mesh.metallic_roughness) if mesh.metallic_roughness is not None else None,
                     normals=mesh.normals[0], normal_map_image=to_pil(mesh.normal_map), tangents=mesh.tangents[0],
                     occlusion_in_mr=bool(mesh.occlusion_in_mr))
            for side in ("crop.png", "mask.png"):
                if (run / side).exists():
                    shutil.copy2(run / side, dest / side)
            meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
            meta.update({"preset": "lod", "source_run": run.name, "source_preset": meta.get("preset"),
                         "lod": st, "seconds": round(time.time() - t0, 1)})
            (dest / "meta.json").write_text(json.dumps(meta, indent=1))
            print(json.dumps({"lod": dest.name, **st, "seconds": meta["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
