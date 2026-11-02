"""Mesh-quality and topology nodes for TRELLIS.2 / Pixal3D output.

- TrellisPlusMeshRepair      : pymeshlab repair pass (duplicates, null / non-manifold faces, winding, fragments).
- TrellisPlusIsotropicRemesh : feature-preserving isotropic remesh (even, well-shaped triangles) at ~constant count.
- TrellisPlusQuadRemesh      : Instant Meshes quad-dominant retopology, re-projected onto the source surface.
- TrellisPlusSaveQuadOBJ     : OBJ export that recombines triangle pairs into quads (+ MTL / PBR textures).
"""
import logging
import os

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import IO, Types

from comfy_extras.nodes_save_3d import get_mesh_batch_item, pack_variable_mesh_batch
from comfy_extras.nodes_mesh_postprocess import (
    _build_triangle_bvh, _closest_points_on_mesh_bvh, _normalize_uvs_to_unit,
)
from comfy_extras.mesh3d.postprocess.qem_decimate import qem_decimate_simplify, QEMConfig

log = logging.getLogger("trellis_plus")


# ----------------------------------------------------------------------------------------- helpers
def _items(mesh):
    """Yield (vertices, faces) numpy per batch item."""
    for i in range(int(mesh.vertices.shape[0])):
        v, f, *_ = get_mesh_batch_item(mesh, i)
        yield v.detach().cpu().numpy().astype(np.float64), f.detach().cpu().numpy().astype(np.int64)


def _pack(vs, fs, like):
    vt = [torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for v in vs]
    ft = [torch.from_numpy(np.ascontiguousarray(f, dtype=np.int64)).to(like.faces.dtype) for f in fs]
    if all(v.shape == vt[0].shape for v in vt) and all(f.shape == ft[0].shape for f in ft):
        return Types.MESH(vertices=torch.stack(vt), faces=torch.stack(ft))
    return pack_variable_mesh_batch(vt, ft)


def _bbox_diag(v):
    return float(np.linalg.norm(v.max(0) - v.min(0))) if len(v) else 1.0


def _ms(v, f):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f.astype(np.int32)))
    return ms


def _ms_out(ms):
    m = ms.current_mesh()
    return m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int64)


def _project_to_surface(q, src_v, src_f):
    """Snap points q [N,3] onto the closest point of mesh (src_v, src_f) using the native GPU BVH."""
    dev = comfy.model_management.get_torch_device()
    tri = torch.from_numpy(src_v[src_f].astype(np.float32)).to(dev)
    bvh = _build_triangle_bvh(tri)
    out = []
    qt = torch.from_numpy(q.astype(np.float32)).to(dev)
    for s in range(0, qt.shape[0], 1 << 18):
        out.append(_closest_points_on_mesh_bvh(qt[s:s + (1 << 18)], tri, bvh).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def mesh_stats(v, f):
    """Topology / quality stats used by the evaluation tool and node logs."""
    f = np.asarray(f, np.int64)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    l = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], 1)
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    # Normalized triangle quality: 4*sqrt(3)*A / sum(l^2) (1 = equilateral, 0 = sliver).
    q = 4 * np.sqrt(3) * area / np.maximum((l ** 2).sum(1), 1e-30)
    return {
        "verts": int(len(v)), "faces": int(len(f)),
        "boundary_edges": int((cnt == 1).sum()), "nonmanifold_edges": int((cnt > 2).sum()),
        "tri_quality_mean": float(q.mean()) if len(q) else 0.0,
        "slivers_pct": float((q < 0.1).mean() * 100) if len(q) else 0.0,
    }


# ----------------------------------------------------------------------------------------- repair
def repair_fn(v, f, remove_small_components=0.002, close_holes_max_edges=0):
    ms = _ms(v, f)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_edges(method="Remove Faces")
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0.0)
    if remove_small_components > 0:
        n = ms.current_mesh().face_number()
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=max(1, int(n * remove_small_components)))
    if close_holes_max_edges > 0:
        try:
            ms.meshing_close_holes(maxholesize=int(close_holes_max_edges), newfaceselected=False)
        except Exception as e:  # close_holes refuses non-manifold leftovers; repair is still useful without it
            log.warning(f"close_holes skipped: {e}")
    try:
        ms.meshing_re_orient_faces_coherently()
    except Exception as e:
        log.warning(f"re-orient skipped: {e}")
    ms.meshing_remove_unreferenced_vertices()
    return _ms_out(ms)


class TrellisPlusMeshRepair(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TrellisPlusMeshRepair",
            display_name="Mesh Repair (Trellis+)",
            category="3d/trellis_plus",
            description="pymeshlab repair: merges duplicate vertices/faces, removes zero-area and non-manifold "
                        "faces, drops floating fragments, optionally closes small holes, and makes winding "
                        "consistent. Run on the raw decoded mesh before Remesh (SDF mode needs consistent winding). "
                        "Geometry only: drops colors/UVs.",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("remove_small_components", default=0.002, min=0.0, max=0.5, step=0.0005,
                               tooltip="Drop connected pieces smaller than this fraction of total faces. 0 = off."),
                IO.Int.Input("close_holes_max_edges", default=0, min=0, max=10000,
                             tooltip="Close holes bounded by up to this many edges. 0 = off."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, remove_small_components, close_holes_max_edges):
        vs, fs = [], []
        for v, f in _items(mesh):
            before = mesh_stats(v, f)
            v, f = repair_fn(v, f, remove_small_components, close_holes_max_edges)
            after = mesh_stats(v, f)
            log.info(f"[MeshRepair] faces {before['faces']}->{after['faces']}, non-manifold edges "
                     f"{before['nonmanifold_edges']}->{after['nonmanifold_edges']}, boundary edges "
                     f"{before['boundary_edges']}->{after['boundary_edges']}")
            vs.append(v)
            fs.append(f)
        return IO.NodeOutput(_pack(vs, fs, mesh))


# ----------------------------------------------------------------------------------------- isotropic
def isotropic_fn(v, f, target_edge_rel=0.0, iterations=3, feature_angle=35.0, adaptive=False):
    ms = _ms(v, f)
    diag = _bbox_diag(v)
    if target_edge_rel > 0:
        target = target_edge_rel * diag
    else:  # keep the current density: mean edge length
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        target = float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).mean())
    import pymeshlab
    ms.meshing_isotropic_explicit_remeshing(
        iterations=int(iterations), adaptive=bool(adaptive),
        targetlen=pymeshlab.PureValue(target),
        featuredeg=float(feature_angle), checksurfdist=True,
        maxsurfdist=pymeshlab.PureValue(target * 0.25),
        splitflag=True, collapseflag=True, swapflag=True, smoothflag=True, reprojectflag=True)
    return _ms_out(ms)


class TrellisPlusIsotropicRemesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TrellisPlusIsotropicRemesh",
            display_name="Isotropic Remesh (Trellis+)",
            category="3d/trellis_plus",
            description="Feature-preserving isotropic remesh (split/collapse/flip/relax + reprojection). Turns "
                        "the slivers and fans of a decimated mesh into even, near-equilateral triangles, "
                        "which bake, unwrap, subdivide and deform better. Keeps creases above feature_angle.",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("target_edge_length", default=0.0, min=0.0, max=0.1, step=0.0005,
                               tooltip="Target edge length as a fraction of the bbox diagonal. 0 = keep current mean "
                                       "edge length (face count stays about the same)."),
                IO.Int.Input("iterations", default=3, min=1, max=20),
                IO.Float.Input("feature_angle", default=35.0, min=0.0, max=180.0, step=1.0,
                               tooltip="Dihedral angle (degrees) above which edges are preserved as creases."),
                IO.Boolean.Input("adaptive", default=False,
                                 tooltip="Curvature-adaptive edge length: denser in detailed areas, sparser on flats."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, target_edge_length, iterations, feature_angle, adaptive):
        vs, fs = [], []
        for v, f in _items(mesh):
            b = mesh_stats(v, f)
            v, f = isotropic_fn(v, f, target_edge_length, iterations, feature_angle, adaptive)
            a = mesh_stats(v, f)
            log.info(f"[IsotropicRemesh] faces {b['faces']}->{a['faces']}, tri quality "
                     f"{b['tri_quality_mean']:.3f}->{a['tri_quality_mean']:.3f}, slivers "
                     f"{b['slivers_pct']:.2f}%->{a['slivers_pct']:.2f}%")
            vs.append(v)
            fs.append(f)
        return IO.NodeOutput(_pack(vs, fs, mesh))


# ----------------------------------------------------------------------------------------- quads
def _triangulate_quads(qv, qf):
    """Instant Meshes returns [F,4] faces (triangles repeat the last index). Split each quad along its shorter
    diagonal and emit the two halves consecutively (quad i -> triangles 2i, 2i+1, triangles last); UnwrapMesh
    and the other native nodes keep face order, so Save Quad OBJ can restore the exact quads."""
    qf = np.asarray(qf, np.int64)
    is_tri = qf[:, 2] == qf[:, 3]
    q = qf[~is_tri]
    d02 = np.linalg.norm(qv[q[:, 0]] - qv[q[:, 2]], axis=1)
    d13 = np.linalg.norm(qv[q[:, 1]] - qv[q[:, 3]], axis=1)
    a = (d02 <= d13)[:, None]
    t1 = np.where(a, q[:, [0, 1, 2]], q[:, [0, 1, 3]])
    t2 = np.where(a, q[:, [0, 2, 3]], q[:, [1, 2, 3]])
    pairs = np.stack([t1, t2], 1).reshape(-1, 3)
    return np.concatenate([pairs, qf[is_tri][:, :3]]), int(len(q)), int(is_tri.sum())


class _mute_stdout:
    def __enter__(self):
        import sys
        sys.stdout.flush()
        self.saved = os.dup(1)
        self.null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.null, 1)

    def __exit__(self, *a):
        os.dup2(self.saved, 1)
        os.close(self.saved)
        os.close(self.null)


def quad_remesh_fn(v, f, target_faces=20000, crease_angle=0.0, smooth_iters=2, align_to_boundaries=True,
                   project_to_source=True, pre_decimate=400_000):
    import pynanoinstantmeshes as nim
    src_v, src_f = v, f
    if len(f) > pre_decimate > 0:  # Instant Meshes' field solve is O(n); keep it interactive
        dev = comfy.model_management.get_torch_device()
        rv, rf, *_ = qem_decimate_simplify(torch.from_numpy(v.astype(np.float32)).to(dev),
                                          torch.from_numpy(f).to(dev), int(pre_decimate), config=QEMConfig())
        v, f = rv.cpu().numpy().astype(np.float64), rf.cpu().numpy().astype(np.int64)
    # Instant Meshes ends with a regular subdivision of the quad-dominant mesh into pure quads (x4 faces),
    # so ask for a quarter of the target. Its C++ progress log goes to stdout: mute fd 1 for the call.
    with _mute_stdout():
        qv, qf = nim.remesh(v.astype(np.float32), f.astype(np.uint32), -1,
                            face_count=max(50, int(target_faces) // 4), rosy=4, posy=4,
                            creaseAngle=float(crease_angle), align_to_boundaries=bool(align_to_boundaries),
                            extrinsic=True, smooth_iter=int(smooth_iters), deterministic=True)
    qv = np.asarray(qv, np.float64)
    tris, nq, nt = _triangulate_quads(qv, qf)
    if project_to_source:
        qv = _project_to_surface(qv, src_v, src_f)
    log.info(f"[QuadRemesh] {len(f)} tris -> {nq} quads + {nt} tris ({100 * nq / max(1, nq + nt):.1f}% quads)")
    return qv, tris


class TrellisPlusQuadRemesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TrellisPlusQuadRemesh",
            display_name="Quad Remesh (Instant Meshes, Trellis+)",
            category="3d/trellis_plus",
            description="Quad-dominant retopology with a 4-RoSy/4-PoSy field (Instant Meshes), then vertices are "
                        "snapped back onto the dense surface. Output is triangulated (glTF has no quads) with each "
                        "quad split along its short diagonal; Save Quad OBJ (Trellis+) recombines them. Bake normal "
                        "maps from the dense mesh to recover detail.",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Int.Input("target_faces", default=20000, min=200, max=2_000_000, step=100,
                             tooltip="Approximate number of quads."),
                IO.Float.Input("crease_angle", default=0.0, min=0.0, max=180.0, step=1.0,
                               tooltip="Align edge flow to creases sharper than this (degrees). 0 = off; ~30-60 "
                                       "for hard-surface."),
                IO.Int.Input("smooth_iters", default=2, min=0, max=10, tooltip="Field smoothing iterations."),
                IO.Boolean.Input("align_to_boundaries", default=True),
                IO.Boolean.Input("project_to_source", default=True,
                                 tooltip="Snap output vertices onto the input surface (removes Instant Meshes' "
                                         "slight shrink)."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, target_faces, crease_angle, smooth_iters, align_to_boundaries, project_to_source):
        vs, fs = [], []
        for v, f in _items(mesh):
            qv, qt = quad_remesh_fn(v, f, target_faces, crease_angle, smooth_iters, align_to_boundaries,
                                    project_to_source)
            vs.append(qv)
            fs.append(qt)
        return IO.NodeOutput(_pack(vs, fs, mesh))


def _pair_to_quad(A, B):
    """Triangles A, B (3 ids each, same winding) sharing one edge -> quad cycle, or None."""
    for k in range(3):
        a0, a1, a2 = A[k], A[(k + 1) % 3], A[(k + 2) % 3]
        for j in range(3):
            if B[j] == a1 and B[(j + 1) % 3] == a0:
                return (a1, a2, a0, B[(j + 2) % 3])
    return None


def recombine_quads(pos, faces, max_angle_dev=40.0):
    """Triangle-pair -> quad recombination. pos [N,3], faces [F,3]; returns faces as tuples of 3 or 4 ids.

    1) Sequential: if most consecutive pairs (2i, 2i+1) share an edge (Quad Remesh output, order kept by the
       native nodes), those pairs are the original quads and are restored exactly.
    2) Greedy for the rest: merge the pair whose quad is convex and closest to right angles first."""
    faces = np.asarray(faces, np.int64)
    out = []
    used = np.zeros(len(faces), bool)
    n2 = len(faces) // 2 * 2
    seq = [_pair_to_quad(faces[i], faces[i + 1]) for i in range(0, n2, 2)]
    if seq and sum(q is not None for q in seq) >= 0.5 * len(seq):
        for i, q in zip(range(0, n2, 2), seq):
            if q is not None:
                out.append(q)
                used[i] = used[i + 1] = True
    rest = np.nonzero(~used)[0]
    if len(rest):
        out += [tuple(int(x) for x in p) for p in _greedy_recombine(pos, faces[rest], max_angle_dev)]
    return out


def _greedy_recombine(pos, faces, max_angle_dev):
    F = len(faces)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])   # directed edges
    fid = np.tile(np.arange(F), 3)
    k = np.repeat(np.arange(3), F)                                                # edge slot in face
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    # only edges shared by exactly two faces (manifold): a pair not chained to a neighbouring equal pair
    ok = same & ~np.r_[False, same[:-1]] & ~np.r_[same[1:], False]
    i0 = order[:-1][ok]
    i1 = order[1:][ok]
    fa, fb = fid[i0], fid[i1]
    ka = k[i0]
    # quad corners: tri A = (p_k, p_k+1, p_k+2), shared edge p_k->p_k+1, opposite of B = q
    A = faces[fa]
    pk = A[np.arange(len(fa)), ka]
    pk1 = A[np.arange(len(fa)), (ka + 1) % 3]
    pk2 = A[np.arange(len(fa)), (ka + 2) % 3]
    B = faces[fb]
    q = B.sum(1) - pk - pk1
    quad = np.stack([pk1, pk2, pk, q], 1)                                         # winding-consistent cycle
    P = pos[quad]                                                                 # [E,4,3]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) + np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])
    dev = np.zeros(len(quad))
    convex = np.ones(len(quad), bool)
    for c in range(4):
        u = P[:, (c + 1) % 4] - P[:, c]
        w = P[:, (c - 1) % 4] - P[:, c]
        cosang = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-30)
        dev = np.maximum(dev, np.abs(np.degrees(np.arccos(np.clip(cosang, -1, 1))) - 90))
        convex &= (np.cross(-w, u) * nrm).sum(1) > 0   # incoming x outgoing edge agrees with normal
    good = convex & (dev <= max_angle_dev) & (fa != fb)
    # A split quad shares its diagonal, which is the longest edge of both triangles: prefer those pairs.
    def _longest(tris):
        T = pos[tris]
        return np.stack([np.linalg.norm(T[:, 1] - T[:, 0], axis=1), np.linalg.norm(T[:, 2] - T[:, 1], axis=1),
                         np.linalg.norm(T[:, 0] - T[:, 2], axis=1)], 1).max(1)
    shared = np.linalg.norm(pos[pk] - pos[pk1], axis=1)
    diag = (shared >= _longest(A) * (1 - 1e-6)) & (shared >= _longest(B) * (1 - 1e-6))
    gi = np.nonzero(good)[0]
    cand = gi[np.lexsort((dev[gi], ~diag[gi]))]
    used = np.zeros(F, bool)
    out = []
    for c in cand:
        a_, b_ = fa[c], fb[c]
        if used[a_] or used[b_]:
            continue
        used[a_] = used[b_] = True
        out.append(tuple(int(x) for x in quad[c]))
    out += [tuple(int(x) for x in t) for t in faces[~used]]
    return out


class TrellisPlusSaveQuadOBJ(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TrellisPlusSaveQuadOBJ",
            display_name="Save Quad OBJ (Trellis+)",
            category="3d/trellis_plus",
            description="Writes an OBJ with triangle pairs recombined into quads (positions welded across UV "
                        "seams, UVs kept per-corner), plus an MTL with base color / roughness-metallic / normal "
                        "maps when present. Use after Quad Remesh for a DCC-friendly quad mesh.",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.String.Input("filename_prefix", default="trellis/quad"),
                IO.Float.Input("max_quad_angle_deviation", default=40.0, min=0.0, max=90.0, step=1.0,
                               tooltip="Merge a triangle pair only if every quad corner is within this many "
                                       "degrees of 90."),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, mesh, filename_prefix, max_quad_angle_deviation=40.0):
        out_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        results = []
        for i in range(int(mesh.vertices.shape[0])):
            v, f, _c, uv, nrm = get_mesh_batch_item(mesh, i)
            v = v.detach().cpu().numpy().astype(np.float64)
            f = f.detach().cpu().numpy().astype(np.int64)
            uv_np = None
            if uv is not None:
                uv_np = _normalize_uvs_to_unit(uv.detach().cpu().numpy())
            # Weld positions for the OBJ "v" list; pair triangles on the original (per-UV-chart) vertex ids so a
            # quad never spans a UV seam. Corners then index v by welded id and vt by original id.
            key = np.round(v / max(_bbox_diag(v), 1e-12) * 1e6).astype(np.int64)
            _, pid = np.unique(key, axis=0, return_inverse=True)
            pid = pid.reshape(-1)
            upos = np.zeros((pid.max() + 1, 3))
            upos[pid] = v
            pair_faces = f if uv_np is not None else pid[f]
            pair_pos = v if uv_np is not None else upos
            polys = recombine_quads(pair_pos, pair_faces, max_quad_angle_deviation)
            base = f"{fname}_{counter + i:05}_"
            obj_path = os.path.join(out_dir, base + ".obj")
            mtl_name = base + ".mtl"
            tex_files = {}
            b = lambda t: None if t is None else (t[i] if t.ndim == 4 else t)
            for attr, suffix in (("texture", "basecolor"), ("metallic_roughness", "mr"), ("normal_map", "normal")):
                img = b(getattr(mesh, attr, None))
                if img is not None:
                    arr = (img[..., :3].detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                    Image.fromarray(arr).save(os.path.join(out_dir, base + suffix + ".png"))
                    tex_files[suffix] = base + suffix + ".png"
            n_quads = sum(1 for p in polys if len(p) == 4)
            with open(obj_path, "w") as fh:
                fh.write(f"# TRELLIS+ quad export: {n_quads} quads, {len(polys) - n_quads} tris\n")
                if tex_files:
                    fh.write(f"mtllib {mtl_name}\n")
                for p in upos:
                    fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                if uv_np is not None:
                    for t in uv_np:
                        fh.write(f"vt {t[0]:.6f} {1.0 - t[1]:.6f}\n")
                if tex_files:
                    fh.write("usemtl material0\n")
                for j, p in enumerate(polys):
                    if uv_np is not None:
                        fh.write("f " + " ".join(f"{pid[a] + 1}/{a + 1}" for a in p) + "\n")
                    else:
                        fh.write("f " + " ".join(str(a + 1) for a in p) + "\n")
            if tex_files:
                with open(os.path.join(out_dir, mtl_name), "w") as fh:
                    fh.write("newmtl material0\nKd 1 1 1\n")
                    if "basecolor" in tex_files:
                        fh.write(f"map_Kd {tex_files['basecolor']}\n")
                    if "normal" in tex_files:
                        fh.write(f"norm {tex_files['normal']}\nmap_Bump -bm 1 {tex_files['normal']}\n")
                    if "mr" in tex_files:
                        fh.write(f"# glTF metallicRoughness packing: G=roughness B=metallic\nmap_Pr {tex_files['mr']}\n")
            log.info(f"[SaveQuadOBJ] {obj_path}: {n_quads} quads, {len(polys) - n_quads} tris")
            results.append({"filename": base + ".obj", "subfolder": subfolder, "type": "output"})
        return IO.NodeOutput(ui={"3d": results})

