"""Low-poly LOD generation: simplify a textured model to a triangle budget and re-bake its look.

Pipeline (per mesh):
  1. weld UV seams, quadric-simplify to the budget (pymeshlab: normal/topology preserving, optimal placement);
  2. re-unwrap the low-poly (native UV unwrapper, same convention as UnwrapMesh);
  3. transfer base color / roughness / metallic / AO texel by texel: a short ray along the low-poly normal (both
     directions) finds the matching point on the full-resolution surface, whose UVs are sampled in its textures;
  4. bake a tangent-space normal map from the full-resolution geometry (native BakeNormalMapFromMesh) and attach
     everything with the native ApplyTextureToMesh, so tangents/normals match what SaveGLB writes.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
from comfy_api.latest import IO, Types

from comfy_extras.nodes_save_3d import get_mesh_batch_item
from comfy_extras.nodes_mesh_postprocess import (
    _rasterize_uv_barycentric, _interp_vertex_attr, _smooth_vertex_normals, _build_triangle_bvh,
    _closest_hit_rays_bvh, _closest_points_on_mesh_bvh, _barycentric, _uv_unwrap, _seam_fill,
    _normalize_uvs_to_unit, BakeNormalMapFromMesh, ApplyTextureToMesh,
)
from .mesh_ops import _ms, _ms_out, mesh_stats

log = logging.getLogger("trellis_plus")


def weld(v, f, uv=None, tol=1e-6):
    """Merge UV-seam duplicates (same position). Returns welded v, f."""
    span = max(float(np.ptp(v, axis=0).max()), 1e-12)
    _, pid = np.unique(np.round(v / span / tol).astype(np.int64), axis=0, return_inverse=True)
    pid = pid.reshape(-1)
    vw = np.zeros((pid.max() + 1, 3))
    vw[pid] = v
    fw = pid[f]
    keep = (fw[:, 0] != fw[:, 1]) & (fw[:, 1] != fw[:, 2]) & (fw[:, 2] != fw[:, 0])
    return vw, fw[keep]


def simplify(v, f, budget, alpha_pct=0.35, offset_pct=0.08):
    """Watertight single-skin low-poly at exactly `budget` faces.

    Generated meshes are thin double shells (outward skin + inward skin one voxel apart, joined at seams), with
    cracks and non-manifold strands. Decimating them directly interleaves the skins (black, inside-out patches)
    and stalls on non-manifold edges. CGAL alpha wrap (via pymeshlab) first shrink-wraps the outside into one
    closed 2-manifold surface (alpha / offset as % of the bbox diagonal); topology-preserving quadric decimation
    then hits the budget exactly. Detail lost to the wrap comes back through the baked normal map."""
    import pymeshlab
    ms = _ms(v, f)
    ms.generate_alpha_wrap(alpha=pymeshlab.PercentageValue(alpha_pct), offset=pymeshlab.PercentageValue(offset_pct))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(budget), qualitythr=0.5, preservenormal=True, preservetopology=True,
        optimalplacement=True, planarquadric=True, autoclean=True)
    v, f = compact(*_ms_out(ms))
    return v, orient(v, f)


def orient(v, f):
    """Make winding consistent across manifold edges (BFS) and point each connected piece outward (signed
    volume). Generated + decimated meshes often carry flipped patches that shade black in viewers."""
    from collections import deque
    F = len(f)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fid = np.tile(np.arange(F), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    ok = same & ~np.r_[False, same[:-1]] & ~np.r_[same[1:], False]          # edges shared by exactly 2 faces
    a, b = order[:-1][ok], order[1:][ok]
    fa, fb = fid[a], fid[b]
    # same direction on both faces -> the pair is inconsistent (needs one flipped)
    flip_rel = (e[a, 0] == e[b, 0])
    adj = [[] for _ in range(F)]
    for x, y, r in zip(fa.tolist(), fb.tolist(), flip_rel.tolist()):
        adj[x].append((y, r))
        adj[y].append((x, r))
    flip = np.zeros(F, bool)
    comp = np.full(F, -1)
    c = 0
    for s0 in range(F):
        if comp[s0] >= 0:
            continue
        comp[s0] = c
        dq = deque([s0])
        while dq:
            x = dq.popleft()
            for y, r in adj[x]:
                if comp[y] < 0:
                    comp[y] = c
                    flip[y] = flip[x] ^ r
                    dq.append(y)
        c += 1
    f = np.where(flip[:, None], f[:, ::-1], f)
    tri = v[f]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    comp_vol = np.bincount(comp, weights=vol, minlength=c)
    f = np.where((comp_vol[comp] < 0)[:, None], f[:, ::-1], f)
    return f


def compact(v, f):
    """Drop faces touching non-finite or duplicate-index vertices, then unreferenced vertices."""
    bad_v = ~np.isfinite(v).all(1)
    keep = ~bad_v[f].any(1) & (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
    f = f[keep]
    used, inv = np.unique(f, return_inverse=True)
    return v[used], inv.reshape(-1, 3)


def unwrap(v, f, resolution, padding=4):
    dev = comfy.model_management.get_torch_device()
    vmap, idx, uvs = _uv_unwrap(torch.from_numpy(v.astype(np.float32)).to(dev),
                                torch.from_numpy(f.astype(np.int64)).to(dev), "pec", int(resolution),
                                int(padding), 0.0)
    uvs = uvs.copy()
    uvs[:, 1] = 1.0 - uvs[:, 1]          # same flip as UnwrapMesh
    return v[vmap], np.asarray(idx, np.int64), uvs.astype(np.float32)


def _sample(img_hwc, uv01):
    """Bilinear sample img [H,W,C] at uv in [0,1] (texel row = v * H, as the native rasterizer)."""
    H, W = img_hwc.shape[:2]
    g = torch.stack([uv01[:, 0] * 2 - 1, uv01[:, 1] * 2 - 1], -1).view(1, -1, 1, 2)
    return F.grid_sample(img_hwc.permute(2, 0, 1)[None], g, mode="bilinear", padding_mode="border",
                         align_corners=False)[0, :, :, 0].T


def transfer_textures(lo_v, lo_f, lo_uv, hi_v, hi_f, hi_uv, hi_maps, size, cage_rel=0.03):
    """Re-sample hi-res texture maps (dict name -> [H,W,C] float tensor, hi_uv convention) into the low-poly UV
    layout. Returns dict name -> [size,size,C] numpy (gutters seam-filled) and the hit-rate."""
    dev = comfy.model_management.get_torch_device()
    face_idx, bary, cov = _rasterize_uv_barycentric(lo_f, _normalize_uvs_to_unit(lo_uv), size)
    lv = torch.from_numpy(lo_v.astype(np.float32)).to(dev)
    lf = torch.from_numpy(lo_f.astype(np.int64)).to(dev)
    ln = torch.from_numpy(_smooth_vertex_normals(lo_v.astype(np.float32), lo_f)).to(dev)
    P = _interp_vertex_attr(lv, lf, face_idx, bary, cov)[cov]
    N = F.normalize(_interp_vertex_attr(ln, lf, face_idx, bary, cov)[cov], dim=-1)

    hv = torch.from_numpy(hi_v.astype(np.float32)).to(dev)
    hf = torch.from_numpy(hi_f.astype(np.int64)).to(dev)
    tri = hv[hf]
    fn = F.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1)
    bvh = _build_triangle_bvh(tri)
    cage = cage_rel * float(np.linalg.norm(np.ptp(hi_v, axis=0)))

    K = P.shape[0]
    best_t = torch.full((K,), float("inf"), device=dev)
    best_f = torch.full((K,), -1, dtype=torch.long, device=dev)
    Q = torch.zeros_like(P)
    for sgn in (1.0, -1.0):        # cast outward and inward from the low-poly surface
        for s in range(0, K, 1 << 20):
            e = min(K, s + (1 << 20))
            d = N[s:e] * sgn
            t, fh, hit = _closest_hit_rays_bvh(P[s:e], d, tri, bvh, tmin=0.0, tmax=cage)
            ok = hit & ((fn[fh.clamp_min(0)] * N[s:e]).sum(-1) > 0.2)   # same-facing surface only
            better = ok & (t < best_t[s:e])
            best_t[s:e] = torch.where(better, t, best_t[s:e])
            best_f[s:e] = torch.where(better, fh, best_f[s:e])
            hit_pt = P[s:e] + d * t[:, None]
            Q[s:e] = torch.where(better[:, None], hit_pt, Q[s:e])
    miss = best_f < 0
    if bool(miss.any()):
        cp, cf = _closest_points_on_mesh_bvh(P[miss], tri, bvh, return_face=True)
        Q[miss] = cp
        best_f[miss] = cf
    hit_rate = 1.0 - float(miss.float().mean())
    b = _barycentric(Q, tri[best_f])
    huv = torch.from_numpy(hi_uv.astype(np.float32)).to(dev)
    uvq = (b[:, :, None] * huv[hf[best_f]]).sum(1)
    out = {}
    covered = cov.cpu().numpy()
    for name, img in hi_maps.items():
        img = img.to(dev).float()
        full = torch.zeros((size, size, img.shape[-1]), device=dev)
        full[cov] = _sample(img, uvq)
        out[name] = _seam_fill(full.cpu().numpy(), covered)
    return out, hit_rate


def make_lod(src_v, src_f, src_uv, src_maps, budget, tex_size=2048, normal_size=None, cage_rel=0.03):
    """src_* : full-res textured model (unwelded, uv in MESH/glTF convention), src_maps: dict with any of
    base_color [H,W,3], metallic, roughness, occlusion ([H,W,1] each). Returns a textured Types.MESH and stats."""
    normal_size = normal_size or tex_size
    wv, wf = weld(src_v, src_f)
    lv, lf = simplify(wv, wf, budget)
    uv_v, uv_f, uv = unwrap(lv, lf, tex_size)
    maps, hit_rate = transfer_textures(uv_v, uv_f, uv, src_v, src_f, src_uv, src_maps, tex_size, cage_rel)

    low = Types.MESH(vertices=torch.from_numpy(uv_v.astype(np.float32))[None],
                     faces=torch.from_numpy(uv_f.astype(np.int32))[None],
                     uvs=torch.from_numpy(uv)[None])
    high = Types.MESH(vertices=torch.from_numpy(wv.astype(np.float32))[None],
                      faces=torch.from_numpy(wf.astype(np.int32))[None])
    nmap = BakeNormalMapFromMesh.execute(low, high, int(normal_size), cage_distance=cage_rel,
                                         ignore_backfaces=True).args[0]
    t = lambda k: torch.from_numpy(np.ascontiguousarray(maps[k]))[None].float() if k in maps else None
    out = ApplyTextureToMesh.execute(low, t("base_color"), metallic=t("metallic"), roughness=t("roughness"),
                                     occlusion=t("occlusion"), normal_map=nmap).args[0]
    st = mesh_stats(*weld(uv_v, uv_f))
    st.update({"budget": int(budget), "texel_hit_rate": round(hit_rate, 4), "source_faces": int(len(wf))})
    log.info(f"[LowPolyLOD] {len(wf)} -> {len(lf)} faces (budget {budget}), {len(uv_v)} verts after unwrap, "
             f"transfer hit-rate {hit_rate:.3f}")
    return out, st


class TrellisPlusLowPolyLOD(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TrellisPlusLowPolyLOD",
            display_name="Low-Poly LOD (Local3D+)",
            category="3d/local3d_plus",
            description="Simplifies a textured model to a triangle budget, re-unwraps it and re-bakes base color, "
                        "roughness/metallic/AO and a tangent-space normal map from the full-resolution model, so "
                        "the low-poly still looks like the original. Feed the final textured mesh (after Apply "
                        "Texture To Mesh); output goes straight to SaveGLB.",
            inputs=[
                IO.Mesh.Input("mesh", tooltip="Full-resolution textured mesh (UVs + base color texture)."),
                IO.Int.Input("target_faces", default=20000, min=500, max=1_000_000, step=500),
                IO.Int.Input("texture_size", default=2048, min=256, max=8192, step=256),
                IO.Float.Input("cage_distance", default=0.03, min=0.002, max=0.2, step=0.001,
                               tooltip="Max surface search distance (fraction of bbox diagonal) when transferring "
                                       "textures and normals. Raise for very low budgets."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, target_faces, texture_size, cage_distance):
        if mesh.uvs is None or mesh.texture is None:
            raise ValueError("Low-Poly LOD needs a textured mesh (connect Apply Texture To Mesh's output)")
        v, f, _c, uv, _n = get_mesh_batch_item(mesh, 0)
        maps = {"base_color": mesh.texture[0, ..., :3].float()}
        if mesh.metallic_roughness is not None:
            mr = mesh.metallic_roughness[0].float()
            maps.update({"roughness": mr[..., 1:2], "metallic": mr[..., 2:3]})
            if mesh.occlusion_in_mr:
                maps["occlusion"] = mr[..., 0:1]
        out, _st = make_lod(v.cpu().numpy().astype(np.float64), f.cpu().numpy().astype(np.int64),
                            _normalize_uvs_to_unit(uv.cpu().numpy()), maps, target_faces, texture_size,
                            cage_rel=cage_distance)
        return IO.NodeOutput(out)
