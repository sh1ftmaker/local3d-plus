#!/usr/bin/env python3
"""Export pipeline runs (outputs/) as web samples (samples/) for the GitHub Pages viewer.

Per run: meshopt-compressed GLB with 2K WebP textures (full topology kept, no decimation), input / crop / mask
images, a turntable render, topology stats and input-camera fidelity. Incremental: a run is re-exported only when
its source GLB changes. Raw decoder meshes and full-res GLBs stay local.

    tools/publish.py                 # export new/changed runs, rebuild samples/index.json
    tools/publish.py --only goblin   # runs whose folder name contains "goblin"
"""
import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402  (loads ComfyUI helpers + trellis_plus.mesh_ops)

GLTF = ROOT / "tools/web/node_modules/.bin/gltf-transform"
SAMPLES = ROOT / "samples"
OUTPUTS = ROOT / "outputs"
INPUTS = ROOT / "test_inputs"

# Human labels for run tags (folder suffix after "<stem>_").
LABELS = {
    "baseline_pixal3d": "Stock template (baseline)",
    "plus_pixal3d": "TRELLIS+ (first plus run)",
    "plus_pixal3d_A_gen": "TRELLIS+ mesh mods, no photo projection",
    "plus_pixal3d_B_detail": "TRELLIS+ photo projection: detail",
    "plus_pixal3d_C_replace": "TRELLIS+ photo projection: replace",
    "plus_pixal3d_D_detail_sharp": "TRELLIS+ detail projection + sharpen",
    "plus_pixal3d_default": "TRELLIS+ defaults",
    "plus_quad_pixal3d": "TRELLIS+ quad retopology",
    "baseline_pixal3d_F": "Stock template (baseline)",
}


def _input_for(run_name):
    stems = sorted((p.stem for p in INPUTS.glob("*.png")), key=len, reverse=True)
    for s in stems:
        if run_name.startswith(s + "_"):
            return s
    return None


def _final_glb(run_dir):
    glbs = [p for p in run_dir.glob("*.glb") if "_raw_" not in p.name]
    return max(glbs, key=lambda p: p.stat().st_mtime) if glbs else None


def _jpg(src, dst, size=None, quality=88):
    im = Image.open(src).convert("RGB")
    if size:
        im.thumbnail((size, size), Image.LANCZOS)
    im.save(dst, "JPEG", quality=quality, optimize=True)


def export_run(run_dir, stem, force=False):
    glb = _final_glb(run_dir)
    if glb is None:
        return None
    rid = run_dir.name
    out = SAMPLES / stem / rid
    stamp = out / ".source"
    src_key = f"{glb.name}:{glb.stat().st_size}:{int(glb.stat().st_mtime)}"
    if not force and stamp.exists() and stamp.read_text() == src_key and (out / "run.json").exists():
        return json.loads((out / "run.json").read_text())
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print(f"[publish] {rid}: {glb.name} ({glb.stat().st_size / 1e6:.1f} MB)", flush=True)

    subprocess.run([str(GLTF), "optimize", str(glb), str(out / "model.glb"), "--compress", "meshopt",
                    "--texture-compress", "webp", "--texture-size", "2048", "--simplify", "false",
                    "--weld", "false"], check=True, capture_output=True)

    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    m, extra = evaluate.load(glb)
    v = np.asarray(m.vertices, np.float64)
    _, pid = np.unique(np.round(v / np.ptp(v, axis=0).max() * 1e6).astype(np.int64), axis=0, return_inverse=True)
    pid = pid.reshape(-1)
    vw = np.zeros((pid.max() + 1, 3))
    vw[pid] = v
    stats = evaluate.mesh_stats(vw, pid[np.asarray(m.faces, np.int64)])
    stats.update(extra)
    crop = mask = None
    if (run_dir / "crop.png").exists():
        _jpg(run_dir / "crop.png", out / "crop.jpg", 1024)
        crop = f"{stem}/{rid}/crop.jpg"
    if (run_dir / "mask.png").exists():
        Image.open(run_dir / "mask.png").convert("L").resize((512, 512)).save(out / "mask.png", optimize=True)
        mask = f"{stem}/{rid}/mask.png"
    if crop and "fov" in meta:
        fm, side = evaluate.front_fidelity(m, run_dir)
        stats.update({k: round(v, 4) for k, v in fm.items()})
        Image.fromarray(side).save(out / "front_compare.jpg", quality=86)
    Image.fromarray(evaluate.render(m, 384)).save(out / "turntable.jpg", quality=86)

    obj = None
    objs = sorted(run_dir.glob("*.obj"))
    if objs:
        o = objs[-1]
        txt = o.read_text()
        mtl_src = o.with_suffix(".mtl")
        (out / "quad.obj").write_text(txt.replace(f"mtllib {mtl_src.name}", "mtllib quad.mtl"))
        if mtl_src.exists():
            mtl = mtl_src.read_text()
            for png in run_dir.glob(o.stem + "*.png"):
                kind = png.stem.replace(o.stem, "") or "tex"
                _jpg(png, out / f"quad_{kind}.jpg", 2048)
                mtl = mtl.replace(png.name, f"quad_{kind}.jpg")
            (out / "quad.mtl").write_text(mtl)
        obj = f"{stem}/{rid}/quad.obj"

    tag = rid[len(stem) + 1:]
    label = LABELS.get(tag, tag.replace("_", " "))
    if "_lod" in tag:
        base, lod = tag.rsplit("_lod", 1)
        label = f"Low-poly {lod} tris (from {LABELS.get(base, base.replace('_', ' '))})"
    run = {
        "id": rid, "label": label, "tag": tag,
        "glb": f"{stem}/{rid}/model.glb", "crop": crop, "mask": mask, "meta": meta,
        "stats": {**stats, "full_res_faces": stats["faces"]},
        "renders": [f"{stem}/{rid}/turntable.jpg"] + ([f"{stem}/{rid}/front_compare.jpg"]
                                                     if (out / "front_compare.jpg").exists() else []),
        "obj": obj, "size_bytes": (out / "model.glb").stat().st_size,
        "created": dt.datetime.fromtimestamp(glb.stat().st_mtime, dt.timezone.utc).isoformat(),
    }
    (out / "run.json").write_text(json.dumps(run, indent=1))
    stamp.write_text(src_key)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    SAMPLES.mkdir(exist_ok=True)
    inputs = {}
    for run_dir in sorted(p for p in OUTPUTS.iterdir() if p.is_dir()):
        stem = _input_for(run_dir.name)
        if stem is None:
            continue
        (SAMPLES / stem).mkdir(exist_ok=True)
        if not (SAMPLES / stem / "input.jpg").exists():
            _jpg(INPUTS / f"{stem}.png", SAMPLES / stem / "input.jpg", 1024)
            _jpg(INPUTS / f"{stem}.png", SAMPLES / stem / "thumb.jpg", 256, 82)
        run = None
        if a.only in run_dir.name:
            run = export_run(run_dir, stem, a.force)
        elif (SAMPLES / stem / run_dir.name / "run.json").exists():
            run = json.loads((SAMPLES / stem / run_dir.name / "run.json").read_text())
        if run:
            inputs.setdefault(stem, []).append(run)
    reports = []
    (SAMPLES / "reports").mkdir(exist_ok=True)
    for rp in sorted((ROOT / "reports").glob("*.png")):
        dst = SAMPLES / "reports" / (rp.stem + ".jpg")
        _jpg(rp, dst, 2048)
        reports.append({"title": rp.stem.replace("_", " "), "image": f"reports/{dst.name}"})
    for sh in sorted((INPUTS / "sheets").glob("*.png")):
        dst = SAMPLES / "reports" / (sh.stem + ".jpg")
        if not dst.exists():
            _jpg(sh, dst, 2048)
        title = sh.stem.split("_", 1)[1].replace("_", " ").title() if "_" in sh.stem else sh.stem
        reports.append({"title": f"Input sheet: {title}", "image": f"reports/{dst.name}"})
    index = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": [{"id": s, "title": s.replace("_", " ").title(), "image": f"{s}/input.jpg",
                    "thumb": f"{s}/thumb.jpg", "runs": sorted(r, key=lambda x: x["created"])}
                   for s, r in sorted(inputs.items())],
        "reports": reports,
    }
    (SAMPLES / "index.json").write_text(json.dumps(index, indent=1))
    total = sum(f.stat().st_size for f in SAMPLES.rglob("*") if f.is_file())
    print(f"[publish] {sum(len(r) for r in inputs.values())} runs, samples/ = {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
