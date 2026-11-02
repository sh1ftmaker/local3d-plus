#!/usr/bin/env python3
"""Headless TRELLIS.2 / Pixal3D image-to-3D runner against a local ComfyUI.

Builds an API-format prompt (mirroring Comfy-Org's 3d_pixal3d_trellis2_image_to_model
template) for a preset, submits it, waits, and copies the GLB(s) into outputs/.

    tools/pipeline.py test_inputs/axe.png --preset baseline
    tools/pipeline.py test_inputs/axe.png --preset plus --quad
"""
import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFY_OUT = Path("/bigdata/agent-tom/qwen-image/ComfyUI/output")


class Graph:
    def __init__(self):
        self.nodes, self._n = {}, 0

    def add(self, cls, **inputs):
        self._n += 1
        nid = str(self._n)
        self.nodes[nid] = {"class_type": cls, "inputs": inputs}
        return Ref(nid)


class Ref:
    def __init__(self, nid):
        self.nid = nid

    def __getitem__(self, i):
        return [self.nid, i]


def _flat(d, prefix):
    """DynamicCombo: {'sign_mode': 'sdf', 'qef': True} -> {'sign_mode': 'sdf', 'sign_mode.qef': True}."""
    key = d.pop("_key")
    out = {prefix: key}
    out.update({f"{prefix}.{k}": v for k, v in d.items()})
    return out


PRESETS = {
    # Comfy-Org template defaults (bf16 weights).
    "baseline": dict(
        ss_steps=12, shape_steps=20, up_steps=12, tex_steps=12,
        sampler="euler", upsample_res="1536",
        remesh=dict(resolution=768, sign={"_key": "udf", "qef": False, "drop_inverted_components": False,
                                          "drop_enclosed_components": False},
                    band=1.0, project_back=0.0, fix_poles=False, smooth_iters=0,
                    drop_small_components=0.01),
        decimate=700_000, atlas=4096, ao=True,
    ),
    # Quality preset: more geometry sampling steps + Heun on the 1536 shape refinement, sharper remesh, mesh repair,
    # isotropic remesh and photo projection. Texture sampling stays at the stock 12 Euler steps: 20 Heun steps
    # scored 2.2 dB worse against the input photo (14.7 vs 16.9 PSNR, axe ablation 2026-11-02).
    "plus": dict(
        ss_steps=20, shape_steps=30, up_steps=20, tex_steps=12,
        sampler="euler", up_sampler="heun", tex_sampler="euler", upsample_res="1536",
        remesh=dict(resolution=1024, sign={"_key": "udf", "qef": True, "drop_inverted_components": True,
                                           "drop_enclosed_components": True},
                    band=1.0, project_back=0.5, fix_poles=True, smooth_iters=2,
                    drop_small_components=0.01),
        decimate=400_000, atlas=4096, ao=True, plus=True,
        decimate_mode="qem", isotropic=True, repair=True,
        project_mode="replace", project_strength=1.0, sharpen=0.0,   # replace: +1.5 dB; detail/sharpen: no gain
    ),
}


def _coerce(v):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def build(image_name, preset, model="pixal3d", seed=42, quad=False, quad_faces=20_000, project=True, overrides=None,
          tag_suffix=""):
    p = {**PRESETS[preset], **(overrides or {})}
    if "remesh_sign" in p:
        p["remesh"] = {**p["remesh"], "sign": ({"_key": "sdf", "qef": True, "manifold": False} if p["remesh_sign"] == "sdf"
                                               else p["remesh"]["sign"])}
    g = Graph()
    img = g.add("LoadImage", image=image_name)
    bg = g.add("LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")
    mask = g.add("RemoveBackground", bg_removal_model=bg[0], image=img[0])
    crop = g.add("ImageCropToMask", images=img[0], masks=mask[0], width=1024, height=1024,
                 pad_factor=1.1, grow_mask=0, background="#000000")
    clipv = g.add("CLIPVisionLoader", clip_name="dino_v3_L_naf_fp32.safetensors")
    fov = None
    if model == "pixal3d":
        moge = g.add("LoadMoGeModel", model_name="moge_2_vitl_normal_fp16.safetensors")
        geo = g.add("MoGeInference", moge_model=moge[0], image=crop[0], resolution_level=9, fov_x_degrees=0,
                    batch_size=4, force_projection=True, apply_mask=True, refine_steps=0)
        fov = g.add("MoGeGeometryToFOV", moge_geometry=geo[0], axis="horizontal", unit="degrees")
        cond = g.add("Pixal3DConditioning", clip_vision_model=clipv[0], image=crop[0], camera_angle_x=fov[0])
        unet_name = "pixal3d_bf16.safetensors"
    else:
        cond = g.add("Trellis2Conditioning", clip_vision_model=clipv[0], image=crop[0])
        unet_name = "trellis_2_bf16.safetensors"
    unet = g.add("UNETLoader", unet_name=unet_name, weight_dtype="default")
    shape_vae = g.add("VAELoader", vae_name="trellis_2_shape_vae_bf16.safetensors")
    tex_vae = g.add("VAELoader", vae_name="trellis_2_texture_vae_bf16.safetensors")

    # Sampler model chains (CFG override / rescale match the upstream pipeline).
    m_ss = g.add("ModelSamplingSD3", shift=5,
                 model=g.add("RescaleCFG", multiplier=0.7,
                             model=g.add("CFGOverride", model=unet[0], cfg=1, start_percent=0.667, end_percent=1)[0])[0])
    m_shape = g.add("RescaleCFG", multiplier=0.5,
                    model=g.add("CFGOverride", model=unet[0], cfg=1, start_percent=0.769, end_percent=1)[0])

    def ks(model, pos, neg, lat, steps, cfg, s, sampler="euler", sched="normal"):
        return g.add("KSampler", model=model, seed=s, steps=steps, cfg=cfg, sampler_name=sampler,
                     scheduler=sched, positive=pos, negative=neg, latent_image=lat, denoise=1.0)

    empty = g.add("EmptyTrellis2LatentStructure", batch_size=1)
    ss = ks(m_ss[0], cond[0], cond[1], empty[0], p["ss_steps"], 7.5, seed + 14)
    vox = g.add("VaeDecodeStructureTrellis2", samples=ss[0], vae=shape_vae[0], resolution="32")
    st = g.add("Trellis2ShapeStage", positive=cond[0], negative=cond[1], voxel=vox[0])
    shape = ks(m_shape[0], st[0], st[1], st[2], p["shape_steps"], 7.5, seed)
    up = g.add("Trellis2UpsampleStage", positive=st[0], negative=st[1], shape_latent=shape[0], vae=shape_vae[0],
               target_resolution=p["upsample_res"])
    shape_hr = ks(m_shape[0], up[0], up[1], up[2], p["up_steps"], 7.5, seed, p.get("up_sampler", "euler"), "simple")
    dec = g.add("VaeDecodeShapeTrellis", samples=shape_hr[0], vae=shape_vae[0])
    ts = g.add("Trellis2TextureStage", positive=up[0], negative=up[1], shape_latent=shape_hr[0])
    tex = ks(unet[0], ts[0], ts[1], ts[2], p["tex_steps"], 1, seed + 1, p.get("tex_sampler", "euler"))
    voxc = g.add("VaeDecodeTextureTrellis", samples=tex[0], vae=tex_vae[0], shape_subdivides=dec[1])

    r = dict(p["remesh"])
    remesh_in = dec[0]
    if p.get("plus") and p.get("repair", True):
        # Repair the raw dual-contoured surface before remeshing (non-manifold, duplicate, flipped faces).
        remesh_in = g.add("TrellisPlusMeshRepair", mesh=dec[0], remove_small_components=0.002,
                          close_holes_max_edges=100)[0]
    remesh = g.add("RemeshMesh", mesh=remesh_in, resolution=r["resolution"], **_flat(dict(r["sign"]), "sign_mode"),
                   band=r["band"], project_back=r["project_back"], fix_poles=r["fix_poles"],
                   smooth_iters=r["smooth_iters"], drop_small_components=r["drop_small_components"],
                   precluster_max_verts=p.get("precluster_max_verts", 4_000_000))
    hi = remesh[0]
    if quad:
        low = g.add("TrellisPlusQuadRemesh", mesh=hi, target_faces=quad_faces, crease_angle=0.0,
                    smooth_iters=2, align_to_boundaries=True, project_to_source=True)[0]
    else:
        if p.get("decimate_mode", "midpoint") == "qem":
            dm = _flat({"_key": "qem", "line_quadric_weight": 1.0, "feature_edge_quadric_weight": 10.0,
                        "feature_edge_min_dihedral_deg": 30.0, "clamp_v_to_edge": True}, "placement_mode")
        else:
            dm = {"placement_mode": "midpoint"}
        low = g.add("DecimateMesh", mesh=hi, target_face_count=p["decimate"], **dm)[0]
        if p.get("isotropic"):
            low = g.add("TrellisPlusIsotropicRemesh", mesh=low, target_edge_length=0.0, iterations=3,
                        feature_angle=35.0, adaptive=False)[0]
    low = g.add("MeshSmoothNormals", mesh=low, crease_angle=180.0)[0]
    uv = g.add("UnwrapMesh", mesh=low, segmenter="pec", resolution=p["atlas"], padding=4 if p.get("plus") else 1,
               weld_distance=0.0002)
    bake_res = p["atlas"]
    bake = g.add("BakeTextureFromVoxel", mesh=uv[0], voxel_colors=voxc[0], texture_size=bake_res,
                 reference_mesh=dec[0])
    base = bake[0]
    mask_crop = g.add("ImageCropToMask", images=g.add("MaskToImage", mask=mask[0])[0], masks=mask[0],
                      width=1024, height=1024, pad_factor=1.1, grow_mask=0, background="#000000")
    if p.get("plus") and project and model == "pixal3d" and p.get("project_mode", "off") != "off":
        # Pixal3D is pixel-aligned with the input view: project the full-res photo onto visible texels.
        base = g.add("TrellisPlusProjectImageTexture", mesh=uv[0], base_color=bake[0], image=crop[0],
                     mask=mask_crop[0],
                     fov=fov[0], mode=p["project_mode"], strength=p.get("project_strength", 1.0),
                     facing_power=2.0, auto_align=True,
                     feather_px=6.0, detail_sigma=2.0)[0]
    if p.get("plus") and p.get("sharpen", 0) > 0:
        base = g.add("TrellisPlusTextureEnhance", mesh=uv[0], image=base, sharpen=p["sharpen"], sharpen_radius=1.5,
                     dilate=True)[0]
    nrm = g.add("BakeNormalMapFromMesh", low_poly=uv[0], high_poly=hi, resolution=2048, cage_distance=0.05,
                ignore_backfaces=True)
    extra = {}
    if p["ao"]:
        ao = g.add("BakeAmbientOcclusion", low_poly=uv[0], high_poly=hi, resolution=1024, samples=64,
                   max_distance=0.71, strength=1, bias=0.01)
        extra["occlusion"] = ao[0]
    final = g.add("ApplyTextureToMesh", mesh=uv[0], base_color=base, metallic=bake[1], roughness=bake[2],
                  normal_map=nrm[0], **extra)
    final = g.add("MeshSmoothNormals", mesh=final[0], crease_angle=180.0)
    tag = f"{preset}{'_quad' if quad else ''}_{model}" + (f"_{tag_suffix}" if tag_suffix else "")
    # evaluation sidecars: the conditioning crop, its mask and the camera FOV
    g.add("SaveImage", images=crop[0], filename_prefix=f"trellis/{tag}_crop")
    g.add("SaveImage", images=mask_crop[0], filename_prefix=f"trellis/{tag}_mask")
    if fov is not None:
        g.add("PreviewAny", source=fov[0])
    g.add("SaveGLB", mesh=final[0], filename_prefix=f"trellis/{tag}")
    if quad:
        g.add("TrellisPlusSaveQuadOBJ", mesh=final[0], filename_prefix=f"trellis/{tag}")
    g.add("SaveGLB", mesh=dec[0], filename_prefix=f"trellis/{tag}_raw")
    return g.nodes, tag


def api(path, data=None, host="127.0.0.1:8188"):
    req = urllib.request.Request(f"http://{host}{path}",
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code}: {e.read().decode()[:4000]}") from None


def upload(path: Path, host="127.0.0.1:8188"):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + path.read_bytes() + \
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"http://{host}/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["name"]


def free_models():
    """Ask ComfyUI to unload all models and drop cached memory (keeps an idle server small)."""
    try:
        api("/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass


def run(prompt, timeout=3600):
    try:
        return _run(prompt, timeout)
    finally:
        free_models()


def _run(prompt, timeout=3600):
    pid = api("/prompt", {"prompt": prompt, "client_id": "trellis-cli"})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = api(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(json.dumps(msgs, indent=1)[:4000])
            return h[pid], time.time() - t0
        time.sleep(3)
    raise TimeoutError(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--preset", default="plus", choices=list(PRESETS))
    ap.add_argument("--model", default="pixal3d", choices=["pixal3d", "trellis2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quad", action="store_true", help="quad-dominant retopology (Instant Meshes)")
    ap.add_argument("--quad-faces", type=int, default=20_000)
    ap.add_argument("--no-project", action="store_true", help="skip photo->texture projection")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--dump", action="store_true", help="print the API prompt and exit")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a preset value, e.g. --set tex_sampler=euler --set project_mode=off")
    ap.add_argument("--tag", default="", help="suffix for the output folder / files")
    a = ap.parse_args()

    name = a.image.name if a.dump else upload(a.image)
    overrides = {k: _coerce(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    prompt, tag = build(name, a.preset, a.model, a.seed, a.quad, a.quad_faces, not a.no_project, overrides, a.tag)
    if a.dump:
        print(json.dumps(prompt, indent=1))
        return
    hist, dt = run(prompt)
    dest = a.out / f"{a.image.stem}_{tag}"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    meta = {"preset": a.preset, "model": a.model, "seed": a.seed, "overrides": overrides, "seconds": round(dt, 1)}
    for out in hist["outputs"].values():
        if "text" in out:
            try:
                meta["fov"] = float(out["text"][0])
            except (ValueError, TypeError, IndexError):
                pass
        for key in ("3d", "result", "files", "meshes", "images"):
            for item in out.get(key, []) or []:
                if isinstance(item, dict) and "filename" in item:
                    src = COMFY_OUT / item.get("subfolder", "") / item["filename"]
                    if src.exists() and src.suffix.lower() in (".glb", ".obj", ".mtl", ".png"):
                        shutil.copy2(src, dest / src.name)
                        copied.append(src.name)
    # MTL/texture companions of an OBJ export.
    for obj in [c for c in copied if c.endswith(".obj")]:
        for comp in (COMFY_OUT / "trellis").glob(Path(obj).stem + "*"):
            if comp.name not in copied:
                shutil.copy2(comp, dest / comp.name)
                copied.append(comp.name)
    for c in list(copied):
        for kind in ("crop", "mask"):
            if f"_{kind}_" in c:
                (dest / c).replace(dest / f"{kind}.png")
                copied.remove(c)
                copied.append(f"{kind}.png")
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"tag": tag, "seconds": round(dt, 1), "dir": str(dest), "files": sorted(copied)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
