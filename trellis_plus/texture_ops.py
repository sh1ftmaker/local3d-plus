"""Texture nodes.

- TrellisPlusProjectImageTexture : projects the full-resolution input photo onto the UV atlas (Pixal3D is
  pixel-aligned with its input camera), with occlusion, facing and edge-feathered weights.
- TrellisPlusTextureEnhance      : chart-aware unsharp mask + gutter dilation for baked atlases.
"""
import itertools
import logging
import math

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

import comfy.model_management
from comfy_api.latest import IO

from comfy_extras.nodes_save_3d import get_mesh_batch_item
from comfy_extras.nodes_mesh_postprocess import (
    _rasterize_uv_barycentric, _interp_vertex_attr, _normalize_uvs_to_unit, _smooth_vertex_normals,
    _build_triangle_bvh, _closest_hit_rays_bvh, _seam_fill,
)

log = logging.getLogger("trellis_plus")


# ------------------------------------------------------------------------------------- camera model
# Pixal3D rig (comfy/ldm/trellis2/model.py): world is Z-up, camera at (0, -d, 0) looking +Y, d = 0.5/tan(fov/2),
# pinhole with focal_px = res / (2 tan(fov/2)). Mesh vertices come out of the VAE in a Y-up frame; the exact
# axis map is recovered by silhouette matching (auto_align) rather than hard-coded.
def _rotations24():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                m[r, c] = s
            if np.linalg.det(m) > 0:
                mats.append(m)
    return mats


def _project(Pw, d, fpx, res):
    """Pw [N,3] world -> pixel x, y, depth (torch)."""
    depth = Pw[:, 1] + d
    x = fpx * Pw[:, 0] / depth.clamp_min(1e-6) + res / 2.0
    y = -fpx * Pw[:, 2] / depth.clamp_min(1e-6) + res / 2.0
    return x, y, depth


def _silhouette(Pw, faces_t, d, fpx, res, sres=256):
    """Approximate silhouette by splatting vertices + face centroids, then closing."""
    pts = torch.cat([Pw, Pw[faces_t].mean(1)], 0)
    x, y, _ = _project(pts, d, fpx, res)
    xi = (x * sres / res).long()
    yi = (y * sres / res).long()
    ok = (xi >= 0) & (xi < sres) & (yi >= 0) & (yi < sres)
    m = torch.zeros(sres * sres, dtype=torch.bool, device=Pw.device)
    m[(yi[ok] * sres + xi[ok])] = True
    m = m.reshape(sres, sres).cpu().numpy()
    m = ndi.binary_closing(m, iterations=2)
    return ndi.binary_fill_holes(m)


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def align_to_view(v_t, faces_t, mask_np, d, fpx, res, refine=True):
    """Find rotation R (24 axis maps), scale s and world shift t such that the projected silhouette of
    (R v) * s + t best matches mask. Returns (R [3,3] np, s, t [3] np, iou)."""
    sres = 256
    target = np.array(torch.nn.functional.interpolate(
        torch.from_numpy(mask_np.astype(np.float32))[None, None], size=(sres, sres), mode="area")[0, 0] > 0.5)
    dev = v_t.device
    best = (None, 1.0, np.zeros(3), -1.0)
    for R in _rotations24():
        Pw = v_t @ torch.from_numpy(R.T).to(dev, v_t.dtype)
        iou = _iou(_silhouette(Pw, faces_t, d, fpx, res, sres), target)
        if iou > best[3]:
            best = (R, 1.0, np.zeros(3), iou)
    R, s, t, iou0 = best
    if refine:
        base = v_t @ torch.from_numpy(R.T).to(dev, v_t.dtype)
        # coordinate descent over scale and the two image-plane shifts (world x, z)
        steps = [(0.04, 0.02), (0.015, 0.006), (0.005, 0.002)]
        for ds, dt in steps:
            improved = True
            while improved:
                improved = False
                for cand in ((s + ds, t), (s - ds, t),
                             (s, t + np.array([dt, 0, 0])), (s, t - np.array([dt, 0, 0])),
                             (s, t + np.array([0, 0, dt])), (s, t - np.array([0, 0, dt]))):
                    cs, ct = cand
                    Pw = base * cs + torch.from_numpy(ct).to(dev, v_t.dtype)
                    ci = _iou(_silhouette(Pw, faces_t, d, fpx, res, sres), target)
                    if ci > best[3] + 1e-4:
                        best = (R, cs, ct, ci)
                        s, t = cs, ct
                        improved = True
    log.info(f"[ProjectImageTexture] alignment IoU {iou0:.4f} -> {best[3]:.4f} (scale {best[1]:.3f}, "
             f"shift {np.round(best[2], 4).tolist()}, axes {best[0].astype(int).tolist()})")
    return best


def _depth_buffer(tri_w, d, fpx, res):
    """Ray-cast depth (camera-axis distance) per image pixel; inf on miss."""
    dev = tri_w.device
    bvh = _build_triangle_bvh(tri_w)
    px = torch.arange(res, device=dev, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(px, px, indexing="ij")
    xc = (xx - res / 2.0) / fpx
    yc = -(yy - res / 2.0) / fpx
    dirs = torch.stack([xc, torch.ones_like(xc), yc], -1).reshape(-1, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    orig = torch.tensor([0.0, -d, 0.0], device=dev).expand_as(dirs)
    depth = torch.full((res * res,), float("inf"), device=dev)
    for s in range(0, res * res, 1 << 20):
        e = s + (1 << 20)
        t, _f, hit = _closest_hit_rays_bvh(orig[s:e], dirs[s:e], tri_w, bvh, tmin=1e-5)
        dd = t * dirs[s:e, 1]                      # distance along the camera axis
        depth[s:e] = torch.where(hit, dd, depth[s:e])
    return depth.reshape(res, res)


def _gauss(img_chw, sigma):
    if sigma <= 0:
        return img_chw
    r = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-r, r + 1, device=img_chw.device, dtype=img_chw.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    C = img_chw.shape[0]
    y = F.conv2d(F.pad(img_chw[None], (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1),
                 groups=C)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return y[0]


def _sample(img_chw, x, y):
    """Bilinear sample img [C,H,W] at pixel coords x, y [K] -> [K,C]."""
    C, H, W = img_chw.shape
    gx = (x / W) * 2 - 1
    gy = (y / H) * 2 - 1
    grid = torch.stack([gx, gy], -1).view(1, -1, 1, 2)
    return F.grid_sample(img_chw[None], grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0, :, :, 0].T


def project_image_to_texture(v, f, uv, base, photo, mask, fov_deg, mode="detail", strength=1.0,
                             facing_power=2.0, feather_px=6.0, detail_sigma=2.0, auto_align=True,
                             return_weight=False):
    """v [N,3], f [M,3], uv [N,2] numpy; base [H,W,3] float torch; photo [R,R,3], mask [R,R] float torch.
    Returns new base [H,W,3] (and the per-texel weight map if return_weight)."""
    dev = comfy.model_management.get_torch_device()
    res = int(photo.shape[0])
    if photo.shape[1] != res:
        raise ValueError("ProjectImageTexture expects the square ImageCropToMask output used for conditioning")
    fov = math.radians(float(fov_deg))
    d = 0.5 / math.tan(fov / 2.0)
    fpx = res / (2.0 * math.tan(fov / 2.0))
    v_t = torch.from_numpy(v.astype(np.float32)).to(dev)
    f_t = torch.from_numpy(f.astype(np.int64)).to(dev)
    mask_np = (mask.cpu().numpy() > 0.5)

    if auto_align:
        R, s, t, iou = align_to_view(v_t, f_t, mask_np, d, fpx, res)
    else:  # Y-up mesh -> Z-up rig, front faces the camera
        R, s, t, iou = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), 1.0, np.zeros(3), float("nan")
    if auto_align and iou < 0.85:
        log.warning(f"[ProjectImageTexture] silhouette IoU {iou:.3f} < 0.85: the mesh does not match the input "
                    f"view well (TRELLIS.2 is not pixel-aligned; use Pixal3D). Projection skipped.")
        return (base, torch.zeros(base.shape[:2])) if return_weight else base
    Rt = torch.from_numpy(R.T.astype(np.float32)).to(dev)
    tt = torch.from_numpy(t.astype(np.float32)).to(dev)
    Pw_v = v_t @ Rt * s + tt
    tri_w = Pw_v[f_t]

    # per-texel world position + normal
    H, W = base.shape[:2]
    uv_n = _normalize_uvs_to_unit(uv)
    face_idx, bary, cov = _rasterize_uv_barycentric(f.astype(np.int64), uv_n, H)
    n_v = torch.from_numpy(_smooth_vertex_normals(v.astype(np.float32), f.astype(np.int64))).to(dev) @ Rt
    P = _interp_vertex_attr(Pw_v, f_t, face_idx, bary, cov)[cov]
    N = F.normalize(_interp_vertex_attr(n_v, f_t, face_idx, bary, cov)[cov], dim=-1)

    x, y, depth = _project(P, d, fpx, res)
    inside = (x >= 0) & (x < res) & (y >= 0) & (y < res) & (depth > 0)

    zbuf = _depth_buffer(tri_w, d, fpx, res)
    xi = x.long().clamp(0, res - 1)
    yi = y.long().clamp(0, res - 1)
    # conservative: compare against the max depth of the 3x3 neighbourhood so texels near grazing edges survive
    zmax = -F.max_pool2d(-torch.where(torch.isinf(zbuf), torch.full_like(zbuf, 1e9), zbuf)[None, None], 3, 1, 1)[0, 0]
    zmax = torch.maximum(zmax, F.max_pool2d(torch.where(torch.isinf(zbuf), torch.zeros_like(zbuf), zbuf)[None, None],
                                            3, 1, 1)[0, 0])
    tol = 4.0 / max(H, 1024)
    visible = inside & (depth <= zbuf[yi, xi] + tol)

    view = F.normalize(torch.tensor([0.0, -d, 0.0], device=dev) - P, dim=-1)
    facing = (N * view).sum(-1).clamp_min(0) ** float(facing_power)

    # feather near the photo's mask edge and near depth discontinuities (occlusion boundaries)
    zf = torch.where(torch.isinf(zbuf), torch.full_like(zbuf, 1e3), zbuf)
    disc = ((F.max_pool2d(zf[None, None], 3, 1, 1) + F.max_pool2d(-zf[None, None], 3, 1, 1))[0, 0] > 0.02)
    ok = mask_np & ~disc.cpu().numpy()
    feather = torch.from_numpy(np.minimum(ndi.distance_transform_edt(ok) / max(feather_px, 1e-3), 1.0)
                               .astype(np.float32)).to(dev)
    w = facing * feather[yi, xi] * visible.float() * float(strength)
    w = w.clamp(0, 1)

    photo_chw = photo[..., :3].to(dev).float().permute(2, 0, 1)
    base_t = base[..., :3].to(dev).float()
    b = base_t[cov]
    c = _sample(photo_chw, x, y)
    if mode == "detail":
        hp = photo_chw - _gauss(photo_chw, detail_sigma)
        out = b + _sample(hp, x, y) * w[:, None]
    else:  # replace with per-channel colour transfer photo -> baked albedo over confidently-seen texels
        sel = w > 0.5
        if sel.sum() > 100:
            cm, cs = c[sel].mean(0), c[sel].std(0).clamp_min(1e-3)
            bm, bs = b[sel].mean(0), b[sel].std(0).clamp_min(1e-3)
            c = (c - cm) / cs * bs + bm
        out = b * (1 - w[:, None]) + c * w[:, None]
    new = base_t.clone()
    new[cov] = out.clamp(0, 1)
    covered = cov.cpu().numpy()
    filled = torch.from_numpy(_seam_fill(new.cpu().numpy(), covered))
    wmap = torch.zeros((H, W), device=dev)
    wmap[cov] = w
    log.info(f"[ProjectImageTexture] mode={mode}: {int((w > 0.05).sum())}/{int(cov.sum())} texels "
             f"({100.0 * float((w > 0.05).float().mean()):.1f}%) received photo detail")
    return (filled, wmap.cpu()) if return_weight else filled


class TrellisPlusProjectImageTexture(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TrellisPlusProjectImageTexture",
            display_name="Project Input Photo onto Texture (Trellis+)",
            category="3d/trellis_plus",
            description="Pixal3D generates geometry pixel-aligned to its input camera, so the full-resolution "
                        "input photo can be projected back onto the visible part of the UV atlas. Handles "
                        "occlusion (ray-cast depth), grazing angles and silhouette/occlusion-edge feathering. "
                        "'detail' adds the photo's high frequencies on top of the baked albedo (robust to "
                        "lighting in the photo); 'replace' blends in the colour-matched photo. Auto-align "
                        "verifies the fit by silhouette IoU and skips projection when it is poor.",
            inputs=[
                IO.Mesh.Input("mesh", tooltip="The UV-unwrapped mesh the base_color was baked for."),
                IO.Image.Input("base_color", tooltip="Baked base color atlas (BakeTextureFromVoxel)."),
                IO.Image.Input("image", tooltip="The exact square image fed to Pixal3D Conditioning."),
                IO.Image.Input("mask", tooltip="Object mask cropped identically (ImageCropToMask of MaskToImage)."),
                IO.Float.Input("fov", default=49.13, min=1.0, max=170.0, step=0.01,
                               tooltip="Same horizontal FOV (degrees) given to Pixal3D Conditioning."),
                IO.Combo.Input("mode", options=["detail", "replace"], default="detail"),
                IO.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05),
                IO.Float.Input("facing_power", default=2.0, min=0.0, max=8.0, step=0.1,
                               tooltip="Weight = cos(view angle)^power; higher fades out faster at grazing angles."),
                IO.Boolean.Input("auto_align", default=True),
                IO.Float.Input("feather_px", default=6.0, min=0.0, max=64.0, step=0.5, advanced=True),
                IO.Float.Input("detail_sigma", default=2.0, min=0.5, max=16.0, step=0.1, advanced=True,
                               tooltip="'detail' mode: blur radius (photo pixels) separating detail from colour."),
            ],
            outputs=[IO.Image.Output(display_name="base_color"), IO.Image.Output(display_name="weight")],
        )

    @classmethod
    def execute(cls, mesh, base_color, image, mask, fov, mode, strength, facing_power, auto_align,
                feather_px=6.0, detail_sigma=2.0):
        outs, ws = [], []
        for i in range(int(mesh.vertices.shape[0])):
            v, f, _c, uv, _n = get_mesh_batch_item(mesh, i)
            if uv is None:
                raise ValueError("ProjectImageTexture: mesh has no UVs (connect the unwrapped mesh)")
            base = base_color[min(i, base_color.shape[0] - 1)]
            ph = image[min(i, image.shape[0] - 1)]
            mk = mask[min(i, mask.shape[0] - 1)]
            mk = mk[..., 0] if mk.ndim == 3 else mk
            o, w = project_image_to_texture(
                v.cpu().numpy(), f.cpu().numpy(), uv.cpu().numpy(), base, ph, mk, fov, mode, strength,
                facing_power, feather_px, detail_sigma, auto_align, return_weight=True)
            outs.append(o.float())
            ws.append(w.float()[..., None].repeat(1, 1, 3))
        return IO.NodeOutput(torch.stack(outs), torch.stack(ws))


# ------------------------------------------------------------------------------------- enhance
def _uv_coverage(f, uv, size):
    _fi, _b, cov = _rasterize_uv_barycentric(f.astype(np.int64), _normalize_uvs_to_unit(uv), size)
    return cov


class TrellisPlusTextureEnhance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TrellisPlusTextureEnhance",
            display_name="Texture Enhance (Trellis+)",
            category="3d/trellis_plus",
            description="Chart-aware unsharp mask for a baked atlas (counters the trilinear voxel-sampling blur), "
                        "then re-fills the UV gutters from the nearest chart texel so seams and mip-maps don't "
                        "bleed. Sharpening only affects texels inside UV charts.",
            inputs=[
                IO.Mesh.Input("mesh", tooltip="UV-unwrapped mesh the atlas belongs to (for the chart mask)."),
                IO.Image.Input("image"),
                IO.Float.Input("sharpen", default=0.4, min=0.0, max=3.0, step=0.05),
                IO.Float.Input("sharpen_radius", default=1.5, min=0.3, max=8.0, step=0.1),
                IO.Boolean.Input("dilate", default=True, tooltip="Re-fill UV gutters after sharpening."),
            ],
            outputs=[IO.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, mesh, image, sharpen, sharpen_radius, dilate):
        dev = comfy.model_management.get_torch_device()
        outs = []
        for i in range(int(image.shape[0])):
            img = image[i].to(dev).float()
            H = img.shape[0]
            mi = min(i, int(mesh.vertices.shape[0]) - 1)
            v, f, _c, uv, _n = get_mesh_batch_item(mesh, mi)
            cov = _uv_coverage(f.cpu().numpy(), uv.cpu().numpy(), H) if uv is not None else \
                torch.ones(img.shape[:2], dtype=torch.bool, device=dev)
            chw = img[..., :3].permute(2, 0, 1)
            if sharpen > 0:
                # Blur with a coverage-normalised kernel so gutter texels don't leak into chart edges.
                m = cov.float()[None]
                blur = _gauss(chw * m, sharpen_radius) / _gauss(m, sharpen_radius).clamp_min(1e-4)
                chw = torch.where(cov[None], (chw + sharpen * (chw - blur)).clamp(0, 1), chw)
            out = chw.permute(1, 2, 0)
            if dilate:
                out = torch.from_numpy(_seam_fill(out.cpu().numpy(), cov.cpu().numpy()))
            outs.append(out.cpu().float())
        return IO.NodeOutput(torch.stack(outs))
