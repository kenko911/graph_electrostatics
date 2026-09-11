###########################################################################################
# Warp-accelerated reciprocal-space Ewald (structure factor + back-projection)
#
# Replaces the two O(K*N) hot spots of the GTO k-space path with fused Warp kernels
# that never materialize the dense [K, N] `cosines` / `sines` matrices:
#
#   assemble_fourier_series_batch  ->  structure factor   S[k,c] = sum_i cos(k.r_i) feat[i,c]
#                                                          (+ sin analogue)
#   project_to_features_batch      ->  back-projection    P[i,c] = sum_k A[k,c] cos(k.r_i)
#                                                                        + B[k,c] sin(k.r_i)
#
# These two maps are mutual adjoints and are IDENTICAL for monopole/dipole/quadrupole
# (rank 0/1/2) source features -- the (l,m) channel index `c` is just carried along.  All
# the GTO physics (the complex Fourier basis `density_basis_fs`, the 1/k^2 Coulomb kernel,
# the self / slab / molecule corrections) is O(K) or O(N*sm) and stays in PyTorch.
#
# Numerics: kernels accumulate in float64 (as in the reference nvalchemiops Ewald kernels)
# regardless of the torch dtype; inputs/outputs round-trip through the caller's dtype so the
# module is a drop-in.  Each thread OWNS its output row (k-major for S, atom-major for P), so
# there are no atomics.  First-order autograd (positions, k-vectors, and the linear feature /
# potential argument) is hand-written -> energy, forces, stress, and feature/charge gradients
# all flow.  Double-backward (Hessian / BEC *through the k-space term*) is NOT yet provided.
#
# STATUS: written from the reference recip kernels + the torch k-space path; NOT yet validated
# on GPU (author env has no warp/CUDA).  Validate with benchmarks_pme/benchmark_kspace_warp.py
# before relying on it.  Convention risks are flagged inline with `# VERIFY`.
###########################################################################################

from __future__ import annotations

from typing import Optional, Tuple

import torch
from scipy.constants import pi

try:
    import warp as wp

    wp.init()
    _HAS_WARP = True
except Exception:  # pragma: no cover - warp optional
    wp = None
    _HAS_WARP = False

_TWO_PI_CUBED = (2.0 * pi) ** 3

# ---------------------------------------------------------------------------------------
# Warp kernels (all arrays float64; scalar `sm` = number of (sigma, l, m) channels)
# ---------------------------------------------------------------------------------------
if _HAS_WARP:

    @wp.kernel
    def _sf_fwd(
        r: wp.array2d(dtype=wp.float64),        # [N, 3] cartesian positions
        k: wp.array2d(dtype=wp.float64),        # [K, 3] reciprocal vectors
        feat: wp.array2d(dtype=wp.float64),     # [N, sm] source features
        kbatch: wp.array(dtype=wp.int32),       # [K] graph id of each k
        astart: wp.array(dtype=wp.int32),       # [G] first atom of graph
        aend: wp.array(dtype=wp.int32),         # [G] last atom (exclusive)
        sm: wp.int32,
        cos_c: wp.array2d(dtype=wp.float64),    # [K, sm] OUT sum_i cos(k.r_i) feat
        sin_c: wp.array2d(dtype=wp.float64),    # [K, sm] OUT sum_i sin(k.r_i) feat
    ):
        """Structure factor, k-major (thread owns row k -> no atomics)."""
        j = wp.tid()
        g = kbatch[j]
        a0 = astart[g]
        a1 = aend[g]
        kx = k[j, 0]
        ky = k[j, 1]
        kz = k[j, 2]
        for c in range(sm):
            cos_c[j, c] = wp.float64(0.0)
            sin_c[j, c] = wp.float64(0.0)
        for i in range(a0, a1):
            phase = kx * r[i, 0] + ky * r[i, 1] + kz * r[i, 2]
            cph = wp.cos(phase)
            sph = wp.sin(phase)
            for c in range(sm):
                f = feat[i, c]
                cos_c[j, c] = cos_c[j, c] + cph * f
                sin_c[j, c] = sin_c[j, c] + sph * f

    @wp.kernel
    def _sf_bwd_atom(
        r: wp.array2d(dtype=wp.float64),
        k: wp.array2d(dtype=wp.float64),
        feat: wp.array2d(dtype=wp.float64),
        gcos: wp.array2d(dtype=wp.float64),     # [K, sm] dL/dcos_c
        gsin: wp.array2d(dtype=wp.float64),     # [K, sm] dL/dsin_c
        nbatch: wp.array(dtype=wp.int32),       # [N] graph id of each atom
        kstart: wp.array(dtype=wp.int32),       # [G] first k of graph
        kend: wp.array(dtype=wp.int32),         # [G] last k (exclusive)
        sm: wp.int32,
        grad_feat: wp.array2d(dtype=wp.float64),  # [N, sm] OUT
        grad_r: wp.array2d(dtype=wp.float64),     # [N, 3] OUT
    ):
        """dL/dfeat and dL/dr for the structure factor, atom-major (thread owns atom i)."""
        i = wp.tid()
        g = nbatch[i]
        k0 = kstart[g]
        k1 = kend[g]
        rx = r[i, 0]
        ry = r[i, 1]
        rz = r[i, 2]
        for c in range(sm):
            grad_feat[i, c] = wp.float64(0.0)
        grx = wp.float64(0.0)
        gry = wp.float64(0.0)
        grz = wp.float64(0.0)
        for j in range(k0, k1):
            kx = k[j, 0]
            ky = k[j, 1]
            kz = k[j, 2]
            phase = kx * rx + ky * ry + kz * rz
            cph = wp.cos(phase)
            sph = wp.sin(phase)
            s = wp.float64(0.0)  # sum_c feat[i,c] * (-sin gcos + cos gsin)
            for c in range(sm):
                gc = gcos[j, c]
                gs = gsin[j, c]
                grad_feat[i, c] = grad_feat[i, c] + cph * gc + sph * gs
                s = s + feat[i, c] * (cph * gs - sph * gc)
            grx = grx + s * kx
            gry = gry + s * ky
            grz = grz + s * kz
        grad_r[i, 0] = grx
        grad_r[i, 1] = gry
        grad_r[i, 2] = grz

    @wp.kernel
    def _sf_bwd_k(
        r: wp.array2d(dtype=wp.float64),
        k: wp.array2d(dtype=wp.float64),
        feat: wp.array2d(dtype=wp.float64),
        gcos: wp.array2d(dtype=wp.float64),
        gsin: wp.array2d(dtype=wp.float64),
        kbatch: wp.array(dtype=wp.int32),
        astart: wp.array(dtype=wp.int32),
        aend: wp.array(dtype=wp.int32),
        sm: wp.int32,
        grad_k: wp.array2d(dtype=wp.float64),   # [K, 3] OUT
    ):
        """dL/dk for the structure factor, k-major (thread owns k)."""
        j = wp.tid()
        g = kbatch[j]
        a0 = astart[g]
        a1 = aend[g]
        kx = k[j, 0]
        ky = k[j, 1]
        kz = k[j, 2]
        gkx = wp.float64(0.0)
        gky = wp.float64(0.0)
        gkz = wp.float64(0.0)
        for i in range(a0, a1):
            rx = r[i, 0]
            ry = r[i, 1]
            rz = r[i, 2]
            phase = kx * rx + ky * ry + kz * rz
            cph = wp.cos(phase)
            sph = wp.sin(phase)
            s = wp.float64(0.0)  # sum_c feat[i,c] * (-sin gcos + cos gsin)
            for c in range(sm):
                s = s + feat[i, c] * (cph * gsin[j, c] - sph * gcos[j, c])
            gkx = gkx + s * rx
            gky = gky + s * ry
            gkz = gkz + s * rz
        grad_k[j, 0] = gkx
        grad_k[j, 1] = gky
        grad_k[j, 2] = gkz

    @wp.kernel
    def _bp_fwd(
        r: wp.array2d(dtype=wp.float64),
        k: wp.array2d(dtype=wp.float64),
        a: wp.array2d(dtype=wp.float64),        # [K, sm]
        b: wp.array2d(dtype=wp.float64),        # [K, sm]
        nbatch: wp.array(dtype=wp.int32),
        kstart: wp.array(dtype=wp.int32),
        kend: wp.array(dtype=wp.int32),
        sm: wp.int32,
        out: wp.array2d(dtype=wp.float64),      # [N, sm] OUT sum_k a cos + b sin
    ):
        """Back-projection, atom-major (thread owns atom i -> no atomics)."""
        i = wp.tid()
        g = nbatch[i]
        k0 = kstart[g]
        k1 = kend[g]
        rx = r[i, 0]
        ry = r[i, 1]
        rz = r[i, 2]
        for c in range(sm):
            out[i, c] = wp.float64(0.0)
        for j in range(k0, k1):
            phase = k[j, 0] * rx + k[j, 1] * ry + k[j, 2] * rz
            cph = wp.cos(phase)
            sph = wp.sin(phase)
            for c in range(sm):
                out[i, c] = out[i, c] + a[j, c] * cph + b[j, c] * sph

    @wp.kernel
    def _bp_bwd_k(
        r: wp.array2d(dtype=wp.float64),
        k: wp.array2d(dtype=wp.float64),
        a: wp.array2d(dtype=wp.float64),
        b: wp.array2d(dtype=wp.float64),
        gout: wp.array2d(dtype=wp.float64),     # [N, sm] dL/dout
        kbatch: wp.array(dtype=wp.int32),
        astart: wp.array(dtype=wp.int32),
        aend: wp.array(dtype=wp.int32),
        sm: wp.int32,
        grad_a: wp.array2d(dtype=wp.float64),   # [K, sm] OUT
        grad_b: wp.array2d(dtype=wp.float64),   # [K, sm] OUT
        grad_k: wp.array2d(dtype=wp.float64),   # [K, 3] OUT
    ):
        """dL/dA, dL/dB, dL/dk for the back-projection, k-major (thread owns k)."""
        j = wp.tid()
        g = kbatch[j]
        a0 = astart[g]
        a1 = aend[g]
        kx = k[j, 0]
        ky = k[j, 1]
        kz = k[j, 2]
        for c in range(sm):
            grad_a[j, c] = wp.float64(0.0)
            grad_b[j, c] = wp.float64(0.0)
        gkx = wp.float64(0.0)
        gky = wp.float64(0.0)
        gkz = wp.float64(0.0)
        for i in range(a0, a1):
            rx = r[i, 0]
            ry = r[i, 1]
            rz = r[i, 2]
            phase = kx * rx + ky * ry + kz * rz
            cph = wp.cos(phase)
            sph = wp.sin(phase)
            s = wp.float64(0.0)  # sum_c (-a sin + b cos) gout
            for c in range(sm):
                go = gout[i, c]
                grad_a[j, c] = grad_a[j, c] + cph * go
                grad_b[j, c] = grad_b[j, c] + sph * go
                s = s + (b[j, c] * cph - a[j, c] * sph) * go
            gkx = gkx + s * rx
            gky = gky + s * ry
            gkz = gkz + s * rz
        grad_k[j, 0] = gkx
        grad_k[j, 1] = gky
        grad_k[j, 2] = gkz

    @wp.kernel
    def _bp_bwd_atom(
        r: wp.array2d(dtype=wp.float64),
        k: wp.array2d(dtype=wp.float64),
        a: wp.array2d(dtype=wp.float64),
        b: wp.array2d(dtype=wp.float64),
        gout: wp.array2d(dtype=wp.float64),
        nbatch: wp.array(dtype=wp.int32),
        kstart: wp.array(dtype=wp.int32),
        kend: wp.array(dtype=wp.int32),
        sm: wp.int32,
        grad_r: wp.array2d(dtype=wp.float64),   # [N, 3] OUT
    ):
        """dL/dr for the back-projection, atom-major (thread owns atom i)."""
        i = wp.tid()
        g = nbatch[i]
        k0 = kstart[g]
        k1 = kend[g]
        rx = r[i, 0]
        ry = r[i, 1]
        rz = r[i, 2]
        grx = wp.float64(0.0)
        gry = wp.float64(0.0)
        grz = wp.float64(0.0)
        for j in range(k0, k1):
            kx = k[j, 0]
            ky = k[j, 1]
            kz = k[j, 2]
            phase = kx * rx + ky * ry + kz * rz
            cph = wp.cos(phase)
            sph = wp.sin(phase)
            s = wp.float64(0.0)
            for c in range(sm):
                s = s + (b[j, c] * cph - a[j, c] * sph) * gout[i, c]
            grx = grx + s * kx
            gry = gry + s * ky
            grz = grz + s * kz
        grad_r[i, 0] = grx
        grad_r[i, 1] = gry
        grad_r[i, 2] = grz


# ---------------------------------------------------------------------------------------
# torch <-> warp helpers
#
# Lifetime: kernels run async on the torch stream (via `_stream`), so torch's stream-aware
# caching allocator keeps the f64/i32 staging tensors alive until the kernel consumes them.
# We therefore MUST keep python references to the staged tensors until after `wp.launch`
# returns -- callers below bind them to locals (`_stage`) before launching.
# ---------------------------------------------------------------------------------------
def _stream(device: torch.device):
    if device.type == "cuda":
        return wp.stream_from_torch(torch.cuda.current_stream(device))
    return None


def _stage_f(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().to(torch.float64)


def _stage_i(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().to(torch.int32)


def _wp(t: torch.Tensor):
    """Warp ctype view of an ALREADY-staged (contiguous, correct-dtype) torch tensor."""
    return wp.from_torch(t, return_ctype=True)


def graph_ranges(index: torch.Tensor, num_graphs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contiguous [start, end) per graph for a graph-grouped index (batch or k_vector_batch).

    Assumes ``index`` is sorted/grouped by graph (true for PyG node batches and for the
    flattened k-vectors from ``compute_k_vectors_flat``).
    """
    counts = torch.bincount(index, minlength=num_graphs).to(torch.int32)
    ends = torch.cumsum(counts, dim=0).to(torch.int32)
    starts = ends - counts
    return starts, ends


# ---------------------------------------------------------------------------------------
# autograd Functions
# ---------------------------------------------------------------------------------------
class _StructureFactor(torch.autograd.Function):
    """(r, feat, k) -> (S_cos[K,sm], S_sin[K,sm]), S_*[k,c] = sum_{i in g(k)} {cos,sin}(k.r_i) feat[i,c]."""

    @staticmethod
    def forward(ctx, r, feat, k, kbatch, nbatch, astart, aend, kstart, kend):
        K, sm, N = k.shape[0], feat.shape[1], r.shape[0]
        dev, stream = wp.device_from_torch(r.device), _stream(r.device)
        rf, kf, ff = _stage_f(r), _stage_f(k), _stage_f(feat)          # keep alive past launch
        kbi, asi, aei = _stage_i(kbatch), _stage_i(astart), _stage_i(aend)
        cos_c = torch.empty((K, sm), dtype=torch.float64, device=r.device)
        sin_c = torch.empty((K, sm), dtype=torch.float64, device=r.device)
        wp.launch(_sf_fwd, dim=K, device=dev, stream=stream,
                  inputs=[_wp(rf), _wp(kf), _wp(ff), _wp(kbi), _wp(asi), _wp(aei), int(sm),
                          _wp(cos_c), _wp(sin_c)])
        ctx.save_for_backward(r, feat, k, kbatch, nbatch, astart, aend, kstart, kend)
        ctx.dtypes = (r.dtype, feat.dtype, k.dtype)
        return cos_c.to(feat.dtype), sin_c.to(feat.dtype)

    @staticmethod
    def backward(ctx, gcos, gsin):
        r, feat, k, kbatch, nbatch, astart, aend, kstart, kend = ctx.saved_tensors
        rdt, fdt, kdt = ctx.dtypes
        K, sm, N = k.shape[0], feat.shape[1], r.shape[0]
        dev, stream = wp.device_from_torch(r.device), _stream(r.device)
        need_r, need_f, need_k = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        rf, kf, ff = _stage_f(r), _stage_f(k), _stage_f(feat)
        gcf, gsf = _stage_f(gcos), _stage_f(gsin)
        grad_feat = torch.empty((N, sm), dtype=torch.float64, device=r.device)
        grad_r = torch.empty((N, 3), dtype=torch.float64, device=r.device)
        if need_r or need_f:
            nbi, ksi, kei = _stage_i(nbatch), _stage_i(kstart), _stage_i(kend)
            wp.launch(_sf_bwd_atom, dim=N, device=dev, stream=stream,
                      inputs=[_wp(rf), _wp(kf), _wp(ff), _wp(gcf), _wp(gsf), _wp(nbi),
                              _wp(ksi), _wp(kei), int(sm), _wp(grad_feat), _wp(grad_r)])
        gk = None
        if need_k:
            kbi, asi, aei = _stage_i(kbatch), _stage_i(astart), _stage_i(aend)
            grad_k = torch.empty((K, 3), dtype=torch.float64, device=r.device)
            wp.launch(_sf_bwd_k, dim=K, device=dev, stream=stream,
                      inputs=[_wp(rf), _wp(kf), _wp(ff), _wp(gcf), _wp(gsf), _wp(kbi),
                              _wp(asi), _wp(aei), int(sm), _wp(grad_k)])
            gk = grad_k.to(kdt)
        return (grad_r.to(rdt) if need_r else None,
                grad_feat.to(fdt) if need_f else None,
                gk, None, None, None, None, None, None)


class _BackProject(torch.autograd.Function):
    """(r, A, B, k) -> P[N,sm], P[i,c] = sum_{k in g(i)} A[k,c] cos(k.r_i) + B[k,c] sin(k.r_i)."""

    @staticmethod
    def forward(ctx, r, a, b, k, kbatch, nbatch, astart, aend, kstart, kend):
        N, sm, K = r.shape[0], a.shape[1], k.shape[0]
        dev, stream = wp.device_from_torch(r.device), _stream(r.device)
        rf, kf, af, bf = _stage_f(r), _stage_f(k), _stage_f(a), _stage_f(b)
        nbi, ksi, kei = _stage_i(nbatch), _stage_i(kstart), _stage_i(kend)
        out = torch.empty((N, sm), dtype=torch.float64, device=r.device)
        wp.launch(_bp_fwd, dim=N, device=dev, stream=stream,
                  inputs=[_wp(rf), _wp(kf), _wp(af), _wp(bf), _wp(nbi), _wp(ksi), _wp(kei),
                          int(sm), _wp(out)])
        ctx.save_for_backward(r, a, b, k, kbatch, nbatch, astart, aend, kstart, kend)
        ctx.dtypes = (r.dtype, a.dtype, b.dtype, k.dtype)
        return out.to(a.dtype)

    @staticmethod
    def backward(ctx, gout):
        r, a, b, k, kbatch, nbatch, astart, aend, kstart, kend = ctx.saved_tensors
        rdt, adt, bdt, kdt = ctx.dtypes
        N, sm, K = r.shape[0], a.shape[1], k.shape[0]
        dev, stream = wp.device_from_torch(r.device), _stream(r.device)
        need_r = ctx.needs_input_grad[0]
        need_ab = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        need_k = ctx.needs_input_grad[3]
        rf, kf, af, bf, gof = _stage_f(r), _stage_f(k), _stage_f(a), _stage_f(b), _stage_f(gout)
        ga = gb = gk = gr = None
        if need_ab or need_k:
            kbi, asi, aei = _stage_i(kbatch), _stage_i(astart), _stage_i(aend)
            grad_a = torch.empty((K, sm), dtype=torch.float64, device=r.device)
            grad_b = torch.empty((K, sm), dtype=torch.float64, device=r.device)
            grad_k = torch.empty((K, 3), dtype=torch.float64, device=r.device)
            wp.launch(_bp_bwd_k, dim=K, device=dev, stream=stream,
                      inputs=[_wp(rf), _wp(kf), _wp(af), _wp(bf), _wp(gof), _wp(kbi),
                              _wp(asi), _wp(aei), int(sm), _wp(grad_a), _wp(grad_b), _wp(grad_k)])
            ga = grad_a.to(adt) if ctx.needs_input_grad[1] else None
            gb = grad_b.to(bdt) if ctx.needs_input_grad[2] else None
            gk = grad_k.to(kdt) if need_k else None
        if need_r:
            nbi, ksi, kei = _stage_i(nbatch), _stage_i(kstart), _stage_i(kend)
            grad_r = torch.empty((N, 3), dtype=torch.float64, device=r.device)
            wp.launch(_bp_bwd_atom, dim=N, device=dev, stream=stream,
                      inputs=[_wp(rf), _wp(kf), _wp(af), _wp(bf), _wp(gof), _wp(nbi),
                              _wp(ksi), _wp(kei), int(sm), _wp(grad_r)])
            gr = grad_r.to(rdt)
        return gr, ga, gb, gk, None, None, None, None, None, None


# ---------------------------------------------------------------------------------------
# Public drop-in replacements (same numerics as features.assemble_/project_ but no [K,N])
# ---------------------------------------------------------------------------------------
def _ranges_from(k_vector_batch, batch, num_graphs):
    kstart, kend = graph_ranges(k_vector_batch, num_graphs)
    astart, aend = graph_ranges(batch, num_graphs)
    return kbatch_int(k_vector_batch), nbatch_int(batch), astart, aend, kstart, kend


def kbatch_int(x):
    return x.to(torch.int32)


def nbatch_int(x):
    return x.to(torch.int32)


def assemble_fourier_series_batch_warp(
    source_feats: torch.Tensor,     # [N, (1,) sm]
    node_positions: torch.Tensor,   # [N, 3]
    k_vectors: torch.Tensor,        # [K, 3]
    k_vector_batch: torch.Tensor,   # [K]
    batch: torch.Tensor,            # [N]
    density_basis_fs: torch.Tensor,  # [K, n_sigma, m_dim, 2]
    volume_per_k: torch.Tensor,     # [K]
    num_graphs: int,
) -> torch.Tensor:
    """Warp equivalent of ``features.assemble_fourier_series_batch`` (returns density [K, 2])."""
    K = k_vectors.shape[0]
    n_sigma, m_dim = density_basis_fs.shape[1], density_basis_fs.shape[2]
    sm = n_sigma * m_dim
    feat2d = source_feats.reshape(source_feats.shape[0], sm)
    kb, nb, astart, aend, kstart, kend = _ranges_from(k_vector_batch, batch, num_graphs)

    coeff_cos, coeff_sin = _StructureFactor.apply(
        node_positions, feat2d, k_vectors, kb, nb, astart, aend, kstart, kend)

    b_r = density_basis_fs[..., 0].reshape(K, sm)
    b_i = density_basis_fs[..., 1].reshape(K, sm)
    rho_real = (b_r * coeff_cos).sum(-1) + (b_i * coeff_sin).sum(-1)
    rho_imag = (b_i * coeff_cos).sum(-1) - (b_r * coeff_sin).sum(-1)
    density = torch.stack([rho_real, rho_imag], dim=-1)
    return _TWO_PI_CUBED * density / volume_per_k.unsqueeze(-1)


def project_to_features_batch_warp(
    potential: torch.Tensor,        # [K, 2]
    feature_basis_fs: torch.Tensor,  # [K, n_sigma, m_dim, 2]
    node_positions: torch.Tensor,   # [N, 3]
    k_vectors: torch.Tensor,        # [K, 3]
    k_vector_batch: torch.Tensor,   # [K]
    batch: torch.Tensor,            # [N]
    num_graphs: int,
    k_factor_proj: Optional[torch.Tensor] = None,  # [K]
) -> torch.Tensor:
    """Warp equivalent of ``features.project_to_features_batch`` (returns [N, n_sigma, m_dim])."""
    K = k_vectors.shape[0]
    n_sigma, m_dim = feature_basis_fs.shape[1], feature_basis_fs.shape[2]
    sm = n_sigma * m_dim
    proj_r = feature_basis_fs[..., 0].reshape(K, sm)
    proj_i = feature_basis_fs[..., 1].reshape(K, sm)
    p_r = potential[:, 0].unsqueeze(-1)
    p_i = potential[:, 1].unsqueeze(-1)
    a = p_r * proj_r + p_i * proj_i
    b = p_r * proj_i - p_i * proj_r
    if k_factor_proj is not None:
        a = a * k_factor_proj.unsqueeze(-1)
        b = b * k_factor_proj.unsqueeze(-1)

    kb, nb, astart, aend, kstart, kend = _ranges_from(k_vector_batch, batch, num_graphs)
    out = _BackProject.apply(node_positions, a, b, k_vectors, kb, nb, astart, aend, kstart, kend)
    proj_total = 2.0 * out
    return (proj_total / _TWO_PI_CUBED).reshape(node_positions.shape[0], n_sigma, m_dim)


def reconstruct_esps_batch_warp(
    potential: torch.Tensor,        # [K, 2]
    node_positions: torch.Tensor,   # [N, 3]
    k_vectors: torch.Tensor,        # [K, 3]
    k_vector_batch: torch.Tensor,   # [K]
    batch: torch.Tensor,            # [N]
    k0_mask: torch.Tensor,          # [K]
    num_graphs: int,
) -> torch.Tensor:
    """Warp equivalent of ``features.reconstruct_esps_batch`` (returns ESP [N])."""
    k0_factor = torch.ones_like(k0_mask)
    k0_factor[k0_mask > 0.0] = 0.5
    a = (2.0 * potential[:, 0] * k0_factor).unsqueeze(-1)   # [K, 1]
    b = (-2.0 * potential[:, 1] * k0_factor).unsqueeze(-1)
    kb, nb, astart, aend, kstart, kend = _ranges_from(k_vector_batch, batch, num_graphs)
    out = _BackProject.apply(node_positions, a, b, k_vectors, kb, nb, astart, aend, kstart, kend)
    return out.squeeze(-1) / _TWO_PI_CUBED
