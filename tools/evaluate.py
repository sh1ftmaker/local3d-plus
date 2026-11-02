#!/usr/bin/env python3
"""Compare GLB / OBJ outputs: topology stats + textured and shaded turntable renders.

    tools/evaluate.py outputs/axe_baseline_pixal3d/*.glb outputs/axe_plus_pixal3d/*.glb --sheet cmp.png
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/bigdata/agent-tom/qwen-image/ComfyUI")
import trimesh  # noqa: E402

from comfy_extras.nodes_mesh_postprocess import _build_triangle_bvh, _render_view, _smooth_vertex_normals  # noqa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trellis_plus.mesh_ops import mesh_stats  # noqa: E402


def load(path):
    path = Path(path)
    if path.suffix == ".obj":
        txt = path.read_text().splitlines()
        polys = [l.split()[1:] for l in txt if l.startswith("f ")]
        quads = sum(1 for p in polys if len(p) == 4)
        m = trimesh.load(path, force="mesh", process=False)
        return m, {"quads_pct": 100.0 * quads / max(1, len(polys)), "polys": len(polys)}
    sc = trimesh.load(path, process=False)
    m = sc.to_geometry() if hasattr(sc, "to_geometry") else (
        trimesh.util.concatenate(list(sc.geometry.values())) if isinstance(sc, trimesh.Scene) else sc)
    return m, {}


def render(m, size=512, views=(0, 90, 180, 270), elev=15):
    dev = "cuda"
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev)
    f = torch.tensor(np.asarray(m.faces), dtype=torch.long, device=dev)
    c = (v.max(0).values + v.min(0).values) / 2
    rad = float((v - c).norm(dim=1).max())
    tri = v[f]
    bvh = _build_triangle_bvh(tri)
    tex = uv = None
    vis = getattr(m, "visual", None)
    if vis is not None and getattr(vis, "uv", None) is not None:
        mat = vis.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if img is not None:
            tex = torch.tensor(np.asarray(img.convert("RGB")), dtype=torch.float32, device=dev) / 255.0
            uvn = np.asarray(vis.uv, dtype=np.float32).copy()
            uvn[:, 1] = 1.0 - uvn[:, 1]  # glTF/trimesh flips V on load
            uv = torch.tensor(uvn, device=dev)
    vn = torch.tensor(_smooth_vertex_normals(np.asarray(m.vertices, np.float32), np.asarray(m.faces)), device=dev)
    fov = math.radians(35)
    dist = rad / math.sin(fov / 2) * 1.05
    rows = []
    for mode in ("texture", "normal"):
        row = []
        for az in views:
            a, e = math.radians(az), math.radians(elev)
            eye = c + dist * torch.tensor([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)],
                                          device=dev)
            fw = torch.nn.functional.normalize(c - eye, dim=0)
            r = torch.nn.functional.normalize(torch.linalg.cross(fw, torch.tensor([0., 1., 0.], device=dev)), dim=0)
            u = torch.linalg.cross(r, fw)
            if mode == "texture" and tex is not None:
                img, hit, _ = _render_view(tri, bvh, uv, f, tex, eye, fw, r, u, fov, size, size)
            else:
                img, hit, _ = _render_view(tri, bvh, None, f, None, eye, fw, r, u, fov, size, size,
                                           vertex_normals=vn, render_normal=True)
                # convert view-space normal to simple shading for readability
                n = img * 2 - 1
                shade = (n[..., 2:3] * 0.75 + n[..., 1:2] * 0.25).clamp(0, 1) * 0.85 + 0.1
                img = shade.expand_as(img)
            img = torch.where(hit.reshape(size, size, 1) if hit.ndim == 1 else hit[..., None], img,
                              torch.full_like(img, 0.12))
            row.append((img.reshape(size, size, 3).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        rows.append(np.concatenate(row, 1))
    return np.concatenate(rows, 0)


def _prep(m):
    dev = "cuda"
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev)
    f = torch.tensor(np.asarray(m.faces), dtype=torch.long, device=dev)
    vis = m.visual
    img = getattr(vis.material, "baseColorTexture", None) or getattr(vis.material, "image", None)
    tex = torch.tensor(np.asarray(img.convert("RGB")), dtype=torch.float32, device=dev) / 255.0
    uvn = np.asarray(vis.uv, dtype=np.float32).copy()
    uvn[:, 1] = 1.0 - uvn[:, 1]
    return v, f, tex, torch.tensor(uvn, device=dev)


def front_fidelity(m, run_dir, size=1024):
    """Render from the Pixal3D input camera (mesh frame: eye (0,0,d), looking -Z, +Y up) and compare with the
    conditioning crop inside the object mask. Returns metrics and a side-by-side image."""
    meta = json.loads((run_dir / "meta.json").read_text())
    fov = math.radians(meta["fov"])
    d = 0.5 / math.tan(fov / 2)
    v, f, tex, uv = _prep(m)
    tri = v[f]
    bvh = _build_triangle_bvh(tri)
    dev = v.device
    eye = torch.tensor([0.0, 0.0, d], device=dev)
    fw = torch.tensor([0.0, 0.0, -1.0], device=dev)
    up = torch.tensor([0.0, 1.0, 0.0], device=dev)
    r = torch.tensor([1.0, 0.0, 0.0], device=dev)
    img, hit, _ = _render_view(tri, bvh, uv, f, tex, eye, fw, r, up, fov, size, size)
    img = img.reshape(size, size, 3).cpu().numpy()
    hit = hit.reshape(size, size).cpu().numpy()
    photo = np.asarray(Image.open(run_dir / "crop.png").convert("RGB").resize((size, size)), np.float32) / 255
    mask = np.asarray(Image.open(run_dir / "mask.png").convert("L").resize((size, size)), np.float32) / 255 > 0.5
    import scipy.ndimage as ndi
    sel = ndi.binary_erosion(mask & hit, iterations=3)
    mse = float(((img - photo) ** 2)[sel].mean())
    lap = lambda x: ndi.laplace(ndi.gaussian_filter(x.mean(-1), 0.7))
    a, b = lap(img)[sel], lap(photo)[sel]
    hf_corr = float(np.corrcoef(a, b)[0, 1])
    iou = float((mask & hit).sum() / max(1, (mask | hit).sum()))
    side = np.concatenate([photo, np.where(hit[..., None], img, 0.0)], 1)
    return {"psnr": 10 * math.log10(1 / max(mse, 1e-10)), "hf_corr": hf_corr, "silhouette_iou": iou}, \
        (side * 255).clip(0, 255).astype(np.uint8)


def compare_renders(m_lod, m_src, size=512, views=(0, 45, 90, 135, 180, 225, 270, 315), elev=15):
    """Render LOD and source textured from the same cameras (framed on the source). Returns PSNR over pixels
    covered by either, and mean silhouette IoU."""
    out = {}
    imgs = {}
    c = rad = None
    for key, m in (("src", m_src), ("lod", m_lod)):
        v, f, tex, uv = _prep(m)
        if c is None:
            c = (v.max(0).values + v.min(0).values) / 2
            rad = float((v - c).norm(dim=1).max())
        tri = v[f]
        bvh = _build_triangle_bvh(tri)
        fov = math.radians(35)
        dist = rad / math.sin(fov / 2) * 1.05
        res = []
        for az in views:
            a, e = math.radians(az), math.radians(elev)
            eye = c + dist * torch.tensor([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)],
                                          device=v.device)
            fw = torch.nn.functional.normalize(c - eye, dim=0)
            r = torch.nn.functional.normalize(torch.linalg.cross(fw, torch.tensor([0., 1., 0.], device=v.device)), dim=0)
            u = torch.linalg.cross(r, fw)
            img, hit, _ = _render_view(tri, bvh, uv, f, tex, eye, fw, r, u, fov, size, size)
            res.append((img.reshape(size, size, 3).cpu().numpy(), hit.reshape(size, size).cpu().numpy()))
        imgs[key] = res
    mse, ious = [], []
    for (a, ha), (b, hb) in zip(imgs["src"], imgs["lod"]):
        u = ha | hb
        a = np.where(ha[..., None], a, 0.12)
        b = np.where(hb[..., None], b, 0.12)
        mse.append(((a - b) ** 2)[u].mean())
        ious.append((ha & hb).sum() / max(1, u.sum()))
    return {"render_psnr_vs_src": 10 * math.log10(1 / max(float(np.mean(mse)), 1e-10)),
            "silhouette_iou_vs_src": float(np.mean(ious))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--front", type=Path, help="write input-camera comparison crops (side by side) here")
    ap.add_argument("--compare-to", type=Path, help="source GLB: score each file's renders against it")
    a = ap.parse_args()
    report, tiles = {}, []
    for p in a.files:
        m, extra = load(p)
        v = np.asarray(m.vertices, np.float64)
        # weld UV-seam duplicates so topology metrics see the real surface
        _, pid = np.unique(np.round(v / np.ptp(v, axis=0).max() * 1e6).astype(np.int64), axis=0, return_inverse=True)
        pid = pid.reshape(-1)
        vw = np.zeros((pid.max() + 1, 3))
        vw[pid] = v
        st = mesh_stats(vw, pid[np.asarray(m.faces, np.int64)])
        st.update(extra)
        if a.front is not None and (Path(p).parent / "meta.json").exists() and "fov" in json.loads(
                (Path(p).parent / "meta.json").read_text()):
            fm, side = front_fidelity(m, Path(p).parent)
            st.update(fm)
            a.front.mkdir(parents=True, exist_ok=True)
            Image.fromarray(side).save(a.front / f"{Path(p).parent.name}.png")
        if a.compare_to is not None:
            st.update(compare_renders(m, load(a.compare_to)[0]))
        report[str(p)] = st
        if a.sheet:
            im = Image.fromarray(render(m, a.size))
            ImageDraw.Draw(im).text((8, 8), f"{Path(p).parent.name}/{Path(p).name}  faces={st['faces']}  "
                                            f"q={st['tri_quality_mean']:.2f}", fill=(255, 255, 0))
            tiles.append(np.asarray(im))
    print(json.dumps(report, indent=1))
    if a.sheet and tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]
        Image.fromarray(np.concatenate(tiles, 0)).save(a.sheet)
        print("sheet:", a.sheet)


if __name__ == "__main__":
    main()
