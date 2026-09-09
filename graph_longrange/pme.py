###########################################################################################
# PME-based electrostatic energy and features
# Wraps the reference compute_pme implementation (charge/dipole/quadrupole,
# rank 0/1/2) to provide drop-in replacements for the k-space Ewald modules.
#
# Scaling convention
# ------------------
# compute_pme uses a Gaussian-unit Coulomb constant of 4π (same as Ck_1 = 4π/V/k²).
# The GTO k-space code uses FIELD_CONSTANT as its Coulomb constant.
# The GTO Fourier basis for l=0 is f_{00}(k) = 2√π exp(−k²σ²/2) (not just exp).
# Tracing through the energy formula gives:
#
#   E_GTO = FIELD_CONSTANT/(4π) * E_PME_rank0_no_self
#
# where E_PME_rank0_no_self = sum_{i≠j} q_i q_j erf(alpha*r_ij)/r_ij  (Gaussian units)
# and alpha = 1 / (2 * sigma).
#
# For features the same scaling applies per radial channel with
# alpha_s = 1 / (2 * w_s),  w_s = sqrt((sigma_src² + sigma_proj_s²) / 2).
###########################################################################################

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
from scipy.constants import pi

from .gto_utils import GTOSelfInteractionBlock, get_Cl_sigma
from .realspace_electrostatics import (
    RealSpaceAnalyticalEnergy,
    RealSpaceAnalyticalElectrostaticFeatures,
    batch_complete_graph_excluding_self_duplicates_vector,
    _l2_source_to_cartesian,
)

# SH(l=2)->Cartesian convention factor that matches the GTO k-space quadrupole
# normalization (energy is quadratic in Q, so 2/3 here == (2/3)^2 in the energy).
_PME_QUAD_SH_TO_CART = 2.0 / 3.0
from .slabs import slab_dipole_correction_energy, MonopoleDipoleCorrectionBlock
from .utils import FIELD_CONSTANT

# Coulomb scaling: PME energies are in Gaussian-unit (4π) convention;
# multiply by this to get code-unit energies.
_PME_TO_CODE = FIELD_CONSTANT / (4.0 * pi)


# ---------------------------------------------------------------------------
# PME reference kernels (pure PyTorch, no Warp dependency)
# ---------------------------------------------------------------------------

def _bspline(u: torch.Tensor, order: int = 6) -> torch.Tensor:
    u2, u3, u4, u5 = u**2, u**3, u**4, u**5
    m1, m2, m3 = u - 1, u - 2, u - 3
    c = [
        torch.logical_and(u >= 0, u < 1),
        torch.logical_and(u >= 1, u < 2),
        torch.logical_and(u >= 2, u < 3),
        torch.logical_and(u >= 3, u < 4),
        torch.logical_and(u >= 4, u < 5),
        torch.logical_and(u >= 5, u < 6),
    ]
    v = [
        u5 / 120,
        u5 / 120 - m1**5 / 20,
        u5 / 120 + m2**5 / 8 - m1**5 / 20,
        u5 / 120 - m3**5 / 6 + m2**5 / 8 - m1**5 / 20,
        u5 / 24 - u4 + 19 * u3 / 2 - 89 * u2 / 2 + 409 * u / 4 - 1829 / 20,
        -u5 / 120 + u4 / 4 - 3 * u3 + 18 * u2 - 54 * u + 324 / 5,
    ]
    return (
        c[0].to(u.dtype) * v[0] + c[1].to(u.dtype) * v[1] + c[2].to(u.dtype) * v[2]
        + c[3].to(u.dtype) * v[3] + c[4].to(u.dtype) * v[4] + c[5].to(u.dtype) * v[5]
    )


def _bspline_prime(u: torch.Tensor) -> torch.Tensor:
    u2, u3, u4 = u**2, u**3, u**4
    m1, m2 = u - 1, u - 2
    c = [
        torch.logical_and(u >= 0, u < 1),
        torch.logical_and(u >= 1, u < 2),
        torch.logical_and(u >= 2, u < 3),
        torch.logical_and(u >= 3, u < 4),
        torch.logical_and(u >= 4, u < 5),
        torch.logical_and(u >= 5, u < 6),
    ]
    v = [
        u4 / 24,
        u4 / 24 - m1**4 / 4,
        u4 / 24 + 5 * m2**4 / 8 - m1**4 / 4,
        -5 * u4 / 12 + 6 * u3 - 63 * u2 / 2 + 71 * u - 231 / 4,
        5 * u4 / 24 - 4 * u3 + 57 * u2 / 2 - 89 * u + 409 / 4,
        -u4 / 24 + u3 - 9 * u2 + 36 * u - 54,
    ]
    return (
        c[0].to(u.dtype) * v[0] + c[1].to(u.dtype) * v[1] + c[2].to(u.dtype) * v[2]
        + c[3].to(u.dtype) * v[3] + c[4].to(u.dtype) * v[4] + c[5].to(u.dtype) * v[5]
    )


def _bspline_double_prime(u: torch.Tensor) -> torch.Tensor:
    """Second derivative of the order-6 B-spline (for quadrupole spreading / EFG interp)."""
    u2, u3 = u**2, u**3
    m1, m2 = u - 1, u - 2
    c = [
        torch.logical_and(u >= 0, u < 1),
        torch.logical_and(u >= 1, u < 2),
        torch.logical_and(u >= 2, u < 3),
        torch.logical_and(u >= 3, u < 4),
        torch.logical_and(u >= 4, u < 5),
        torch.logical_and(u >= 5, u < 6),
    ]
    v = [
        u3 / 6,
        u3 / 6 - m1**3,
        u3 / 6 + 5 * m2**3 / 2 - m1**3,
        -5 * u3 / 3 + 18 * u2 - 63 * u + 71,
        5 * u3 / 6 - 12 * u2 + 57 * u - 89,
        -u3 / 6 + 3 * u2 - 18 * u + 36,
    ]
    return (
        c[0].to(u.dtype) * v[0] + c[1].to(u.dtype) * v[1] + c[2].to(u.dtype) * v[2]
        + c[3].to(u.dtype) * v[3] + c[4].to(u.dtype) * v[4] + c[5].to(u.dtype) * v[5]
    )


def _grid_hessian_weights(M: torch.Tensor, dM: torch.Tensor, ddM: torch.Tensor) -> torch.Tensor:
    """Hessian of the separable B-spline product W=Mx·My·Mz w.r.t. grid coords. -> [Na, S, 3, 3]."""
    H = torch.stack([
        ddM[:, :, 0] * M[:, :, 1] * M[:, :, 2],   # xx
        dM[:, :, 0] * dM[:, :, 1] * M[:, :, 2],    # xy
        dM[:, :, 0] * M[:, :, 1] * dM[:, :, 2],    # xz
        M[:, :, 0] * dM[:, :, 1] * dM[:, :, 2],    # yz
        M[:, :, 0] * ddM[:, :, 1] * M[:, :, 2],    # yy
        M[:, :, 0] * M[:, :, 1] * ddM[:, :, 2],    # zz
    ], dim=-1)
    xx, xy, xz, yz, yy, zz = H.unbind(-1)
    return torch.stack([
        torch.stack([xx, xy, xz], -1),
        torch.stack([xy, yy, yz], -1),
        torch.stack([xz, yz, zz], -1),
    ], dim=-2)


def _make_stencil(order: int, device, dtype) -> torch.Tensor:
    half = order // 2
    r = torch.arange(-half, half, device=device, dtype=dtype)
    shifts = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1)
    return shifts.reshape(1, order**3, 3)


def _get_recip_vectors(N: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """N_j * A_ji^{-1}: maps fractional grid coords to k-space."""
    return (N.reshape(1, 3) * torch.linalg.inv(box)).T


def _get_u_reference(coords: torch.Tensor, Nj_Aji_star: torch.Tensor, order: int = 6):
    R = torch.einsum("ij,kj->ki", Nj_Aji_star, coords)
    m_u0 = torch.ceil(R).to(torch.int64)
    u0 = (m_u0 - R) + order / 2
    return m_u0, u0


def _scatter_to_mesh(W: torch.Tensor, m_u0: torch.Tensor,
                      N: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    N = N.int()
    idx = (m_u0[:, None, :] + shifts) % N[None, None, :]
    idx = idx.to(torch.int64)
    mesh = torch.zeros(N.tolist(), dtype=W.dtype, device=W.device)
    mesh.index_put_(
        (idx[:, :, 0].flatten(), idx[:, :, 1].flatten(), idx[:, :, 2].flatten()),
        W.flatten(), accumulate=True,
    )
    return mesh


def _spread_charges(positions: torch.Tensor, box: torch.Tensor,
                    q: torch.Tensor, p: Optional[torch.Tensor],
                    N: torch.Tensor, order: int = 6,
                    Q: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Spread monopoles (+ optional dipoles p, + optional Cartesian quadrupoles Q) via B-splines.

    Density Taylor expansion: ρ = q·δ − p·∇δ + ½ Q:∇∇δ, so the mesh weights are
    W = q·W − p·∇W + ½ Q:∇∇W  (∇ w.r.t. grid coords; vectors/tensors mapped to grid by Nj)."""
    Nj = _get_recip_vectors(N, box)
    m_u0, u0 = _get_u_reference(positions, Nj, order)
    shifts = _make_stencil(order, positions.device, positions.dtype)

    u = u0[:, None, :] + shifts                      # [Na, S, 3]
    M = _bspline(u)                                   # [Na, S, 3]
    W = q.reshape(-1, 1) * M.prod(dim=2)             # [Na, S]

    if p is not None or Q is not None:
        dM = _bspline_prime(u)

    if p is not None:
        p_u = torch.matmul(p, Nj.T)                  # [Na, 3] in grid basis
        grad_W = torch.stack([
            dM[:, :, 0] * M[:, :, 1] * M[:, :, 2],
            M[:, :, 0] * dM[:, :, 1] * M[:, :, 2],
            M[:, :, 0] * M[:, :, 1] * dM[:, :, 2],
        ], dim=2)                                     # [Na, S, 3]
        W = W - (p_u[:, None, :] * grad_W).sum(dim=2)

    if Q is not None:
        ddM = _bspline_double_prime(u)
        ggW = _grid_hessian_weights(M, dM, ddM)       # [Na, S, 3, 3] grid Hessian
        Q_u = torch.einsum("ab,nbc,dc->nad", Nj, Q, Nj)   # quad in grid basis: Nj Q Njᵀ
        W = W + 0.5 * (Q_u[:, None, :, :] * ggW).sum(dim=(-1, -2))

    return _scatter_to_mesh(W, m_u0, N, shifts[0])


_THETA_CACHE: dict = {}


def _theta_k(N: torch.Tensor, order: int, device, dtype) -> torch.Tensor:
    """B-spline structure factor on the full FFT grid (flattened).

    Depends only on (mesh dims, order, device, dtype) -- a position/charge-independent
    geometric constant -- so it is computed once and cached, then reused across all
    subsequent forward calls (mirrors the GTO path's static k-space cache).
    """
    N_list = N.int().tolist()
    key = (tuple(N_list), int(order), str(device), dtype)
    cached = _THETA_CACHE.get(key)
    if cached is not None:
        return cached

    kpts_int = torch.stack(torch.meshgrid(
        *[torch.arange(n, device=device, dtype=dtype) for n in N_list],
        indexing="ij",
    ), dim=-1).reshape(-1, 3)

    half = order // 2
    m = torch.arange(-half, half, device=device, dtype=dtype).reshape(-1, 1, 1)
    bsp = _bspline(m + half)                          # [order, 1, 1]
    exp_arg = (2 * math.pi * m * kpts_int.unsqueeze(0)
               / torch.tensor(N_list, device=device, dtype=dtype))
    theta = (bsp * torch.cos(exp_arg)).sum(0).prod(1)  # [N_total]
    _THETA_CACHE[key] = theta
    return theta


def _interpolate_potential(phi_grid: torch.Tensor, positions: torch.Tensor,
                            box: torch.Tensor, N: torch.Tensor,
                            want_field: bool, order: int = 6, want_hessian: bool = False):
    """Gather potential (and optionally field) at atom positions via B-spline interpolation."""
    Nj = _get_recip_vectors(N, box)
    m_u0, u0 = _get_u_reference(positions, Nj, order)
    shifts = _make_stencil(order, positions.device, positions.dtype)

    Nx, Ny, Nz = N.int().tolist()
    idx = (m_u0[:, None, :] + shifts[0]) % N.int()[None, None, :]
    flat_idx = (idx[:, :, 0] * Ny * Nz + idx[:, :, 1] * Nz + idx[:, :, 2]).long()
    phi_loc = phi_grid.reshape(-1)[flat_idx]          # [Na, S]

    u = u0[:, None, :] + shifts                       # [Na, S, 3]
    B = _bspline(u)
    phi_atoms = (phi_loc * B.prod(dim=2)).sum(dim=1)  # [Na]

    E_atoms = None
    H_atoms = None
    if want_field or want_hessian:
        dB = _bspline_prime(u)
    if want_field:
        grad_W = torch.stack([
            dB[:, :, 0] * B[:, :, 1] * B[:, :, 2],
            B[:, :, 0] * dB[:, :, 1] * B[:, :, 2],
            B[:, :, 0] * B[:, :, 1] * dB[:, :, 2],
        ], dim=2)
        grad_phi_u = (phi_loc.unsqueeze(-1) * grad_W).sum(dim=1)  # [Na, 3]
        E_atoms = torch.matmul(grad_phi_u, Nj.T)      # [Na, 3]
    if want_hessian:
        ddB = _bspline_double_prime(u)
        Hgrid = (phi_loc[:, :, None, None] * _grid_hessian_weights(B, dB, ddB)).sum(dim=1)  # [Na,3,3] grid-frame
        H_atoms = torch.einsum("ac,ncd,bd->nab", Nj, Hgrid, Nj)   # [Na,3,3] physical ∇∇φ

    return phi_atoms, E_atoms, H_atoms


def compute_pme_single(
    coords: torch.Tensor,          # [Na, 3]
    box: torch.Tensor,             # [3, 3]
    q: torch.Tensor,               # [Na]
    p: Optional[torch.Tensor],     # [Na, 3] or None
    alpha: float,
    K: int,
    rank: int,
    want_field: bool = False,
    Q: Optional[torch.Tensor] = None,   # [Na, 3, 3] traceless Cartesian quadrupole
    want_hessian: bool = False,
) -> tuple:
    """
    PME reciprocal-space potential (and optionally field/Hessian) for one system.

    Returns (phi_atoms, E_atoms, H_atoms) in Gaussian units (Coulomb constant = 4π).
    E_atoms = ∇φ (positive gradient, not the electric field −∇φ);
    H_atoms = ∇∇φ (physical Hessian) or None when rank < 2.
    Self-corrections are NOT applied here; callers handle them.

    Parameters
    ----------
    rank : int
        0 = charges only, 1 = + dipoles, 2 = + quadrupoles (spread onto mesh).
    want_field : bool
        If True, also compute ∇φ at atom positions even for rank=0.
    Q : Optional[Tensor]
        Traceless Cartesian quadrupole [Na,3,3], spread when rank >= 2.
    """
    device, dtype = coords.device, coords.dtype
    N = torch.tensor([K, K, K], device=device, dtype=dtype)

    mesh = _spread_charges(coords, box, q, p if rank >= 1 else None, N,
                           Q=Q if rank >= 2 else None)

    # k-space grid
    N_int = [K, K, K]
    kx = torch.fft.fftfreq(K, d=1.0 / K, device=device, dtype=dtype)
    ky = torch.fft.fftfreq(K, d=1.0 / K, device=device, dtype=dtype)
    kz = torch.fft.fftfreq(K, d=1.0 / K, device=device, dtype=dtype)
    kpts_int = torch.stack(
        torch.meshgrid(kx, ky, kz, indexing="ij"), dim=-1
    ).reshape(-1, 3)

    box_inv = torch.linalg.inv(box).T
    kpts = 2 * math.pi * torch.matmul(kpts_int, box_inv)  # [N³, 3]
    ksq = (kpts**2).sum(-1)                               # [N³]

    V = torch.linalg.det(box).abs()

    # Structure factor & Green's function
    S_k = torch.fft.fftn(mesh).reshape(-1)               # [N³]
    theta = _theta_k(N, order=6, device=device, dtype=dtype)  # [N³]

    C_k = torch.zeros_like(ksq)
    mask = ksq > 1e-10
    C_k[mask] = (4.0 * math.pi / V) * torch.exp(-ksq[mask] / (4.0 * alpha**2)) / ksq[mask]

    theta_safe = theta.abs().clamp(min=1e-10)
    Phi_k = torch.zeros_like(S_k)
    Phi_k[mask] = C_k[mask] * S_k[mask] / theta_safe[mask].pow(2)
    Phi_real = torch.fft.ifftn(Phi_k.reshape(K, K, K), norm="forward").real

    want_field_actual = want_field or (rank >= 1)
    want_hessian_actual = want_hessian or (rank >= 2)
    phi_atoms, E_atoms, H_atoms = _interpolate_potential(
        Phi_real, coords, box, N,
        want_field=want_field_actual, want_hessian=want_hessian_actual)
    return phi_atoms, E_atoms, H_atoms


# ---------------------------------------------------------------------------
# PMEElectrostaticEnergy
# ---------------------------------------------------------------------------

class PMEElectrostaticEnergy(nn.Module):
    """
    PME-based electrostatic energy for GTO multipole densities.

    Replaces the explicit k-vector Ewald sum in GTOElectrostaticEnergy
    for periodic systems.  Scales as O(N log N) vs O(N × N_k).

    Parameters
    ----------
    density_max_l : int
        Maximum angular momentum of source density
        (0 = charges, 1 = + dipoles, 2 = + quadrupoles).
    density_smearing_width : float
        GTO smearing width σ (Å).  Sets Ewald parameter α = 1/(2σ).
    mesh_size : int
        PME mesh edge length K (K³ grid).  Rule of thumb: K ≥ 2 × k_cutoff_Å × L_Å / (2π).
    include_self_interaction : bool
        If False (default) subtract the GTO self-interaction.
    include_pbc_corrections : bool
        Apply molecule/slab geometry corrections.
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        mesh_size: int,
        include_self_interaction: bool = False,
        include_pbc_corrections: bool = True,
    ):
        if density_max_l > 2:
            raise ValueError("PMEElectrostaticEnergy supports density_max_l <= 2.")
        super().__init__()
        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.mesh_size = mesh_size
        self.include_self_interaction = include_self_interaction
        self.include_pbc_corrections = include_pbc_corrections
        self.alpha = 1.0 / (2.0 * density_smearing_width)

        self.self_interaction_terms = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=density_max_l,
            sigmas_receive=[density_smearing_width],
            normalize_source="multipoles",
            normalize_receive="multipoles",
        )
        self.realspace_energy = RealSpaceAnalyticalEnergy(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            include_self_interaction=include_self_interaction,
        )
        self.monopole_dipole_correction = MonopoleDipoleCorrectionBlock(density_max_l)

    def forward(
        self,
        source_feats: torch.Tensor,    # [n_nodes, (l+1)^2] or [n_nodes, 1, (l+1)^2]
        node_positions: torch.Tensor,  # [n_nodes, 3]
        batch: torch.Tensor,           # [n_nodes]
        box: torch.Tensor,             # [n_graphs, 3, 3]
        volume: torch.Tensor,          # [n_graphs]
        pbc: torch.Tensor,             # [n_graphs, 3] bool
    ) -> torch.Tensor:
        if torch.any(pbc):
            return self._pme_energy(source_feats, node_positions, batch, box, volume, pbc)
        return self.realspace_energy(
            source_feats=source_feats,
            positions=node_positions,
            batch=batch,
        )

    def _pme_energy(self, source_feats, node_positions, batch, box, volume, pbc):
        feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats

        n_graphs = int(volume.shape[0])
        energies = torch.zeros(n_graphs, dtype=feats.dtype, device=feats.device)

        for g in range(n_graphs):
            mask = batch == g
            pos_g = node_positions[mask]
            q_g = feats[mask, 0]
            p_g = feats[mask][:, [3, 1, 2]] if self.density_max_l >= 1 else None  # (x,y,z) Cartesian
            Q_g = None
            if self.density_max_l >= 2:
                # SH l=2 coeffs -> traceless Cartesian, rescaled to the GTO k-space convention
                Q_g = _l2_source_to_cartesian(feats[mask][:, 4:9]) * _PME_QUAD_SH_TO_CART

            phi, E_field, H = compute_pme_single(
                pos_g, box[g], q_g, p_g, self.alpha, self.mesh_size,
                rank=self.density_max_l, Q=Q_g,
            )

            # Self-correction (PME Gaussian convention)
            alpha = self.alpha
            phi_corr = phi - 2.0 * alpha / math.sqrt(math.pi) * q_g
            term_q = 0.5 * (q_g * phi_corr).sum()

            term_p = torch.tensor(0.0, dtype=feats.dtype, device=feats.device)
            if self.density_max_l >= 1 and p_g is not None and E_field is not None:
                field_corr = E_field + alpha * (4.0 * alpha**2 / 3.0) / math.sqrt(math.pi) * p_g
                term_p = -0.5 * (p_g * field_corr).sum()

            term_Q = torch.tensor(0.0, dtype=feats.dtype, device=feats.device)
            if self.density_max_l >= 2 and Q_g is not None and H is not None:
                # E_quad = 1/4 Q:∇∇φ  −  (2/(5√π)) α⁵ Q:Q   (reciprocal − Ewald self)
                self_quad = (2.0 / (5.0 * math.sqrt(math.pi))) * alpha**5 * (Q_g * Q_g).sum()
                term_Q = 0.25 * (Q_g * H).sum() - self_quad

            energies[g] = (term_q + term_p + term_Q) * _PME_TO_CODE

        # PME phi_corr already removes the Ewald self-interaction, so
        # energies is already the inter-atomic energy (no self).
        # Only ADD self-interaction back when include_self_interaction=True.
        if self.include_self_interaction:
            self_fields = self.self_interaction_terms(feats)
            node_energies = torch.einsum("nb,nb->n", feats, self_fields)
            self_energy = torch.zeros(n_graphs, dtype=node_energies.dtype,
                                      device=node_energies.device)
            self_energy.index_add_(0, batch, node_energies)
            energies = energies + self_energy * 0.5

        if self.include_pbc_corrections:
            slab = torch.tensor([0, 0, 1], dtype=torch.bool, device=pbc.device)
            is_molecule = torch.all(~pbc, dim=1)
            is_slab = torch.all(torch.logical_xor(slab, pbc), dim=1)
            if is_molecule.any() or is_slab.any():
                mol_corr = self.monopole_dipole_correction(
                    feats, node_positions, volume, batch
                )
                slab_corr = slab_dipole_correction_energy(
                    feats, node_positions, volume, batch
                )
                corr = torch.zeros_like(mol_corr)
                corr = torch.where(is_molecule, mol_corr, corr)
                corr = torch.where(is_slab, slab_corr, corr)
                energies = energies + corr

        return energies


# ---------------------------------------------------------------------------
# PMEElectrostaticFeatures
# ---------------------------------------------------------------------------

class PMEElectrostaticFeatures(nn.Module):
    """
    PME-based GTO electrostatic features for periodic systems.

    Replaces the explicit k-vector sum in GTOElectrostaticFeatures._pbc_forward_dynamic.
    For each projection width σ_proj_s, runs one PME call with effective Ewald
    parameter α_s = 1/(2 w_s),  w_s = sqrt((σ_src² + σ_proj_s²)/2).

    Parameters
    ----------
    density_max_l, density_smearing_width : source multipole parameters.
    feature_max_l, feature_smearing_widths : projection GTO parameters.
    mesh_size : int  PME mesh edge length K.
    include_self_interaction, integral_normalization : same as GTOElectrostaticFeatures.
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        feature_max_l: int,
        feature_smearing_widths: List[float],
        mesh_size: int,
        include_self_interaction: bool = False,
        integral_normalization: str = "receiver",
    ):
        if density_max_l > 2 or feature_max_l > 2:
            raise ValueError("PMEElectrostaticFeatures supports l <= 2 only.")
        super().__init__()
        self.density_max_l = density_max_l
        self.feature_max_l = feature_max_l
        self.mesh_size = mesh_size
        self.include_self_interaction = include_self_interaction
        self.num_radial = len(feature_smearing_widths)

        sig_src = density_smearing_width
        self.alphas = [
            1.0 / (2.0 * math.sqrt((sig_src**2 + s**2) / 2.0))
            for s in feature_smearing_widths
        ]

        l0_factors = [
            get_Cl_sigma(0, s, integral_normalization) / get_Cl_sigma(0, s, "multipoles")
            for s in feature_smearing_widths
        ]
        self.register_buffer("l0_factors",
                             torch.tensor(l0_factors, dtype=torch.get_default_dtype()))

        if feature_max_l >= 1:
            l1_weight = [
                3**0.5 * s**2 * get_Cl_sigma(1, s, integral_normalization)
                / get_Cl_sigma(0, s, "multipoles")
                for s in feature_smearing_widths
            ]
            self.register_buffer("l1_weight",
                                 torch.tensor(l1_weight, dtype=torch.get_default_dtype()))
        else:
            self.register_buffer("l1_weight", None)

        if feature_max_l >= 2:
            # EFG (∇∇φ, traceless) -> l=2 GTO projection. Constant 3√5/2 pinned
            # against the k-space evaluator (mirrors √3 for the l=1 field weight).
            l2_weight = [
                (3.0 * 5**0.5 / 2.0) * s**4 * get_Cl_sigma(2, s, integral_normalization)
                / get_Cl_sigma(0, s, "multipoles")
                for s in feature_smearing_widths
            ]
            self.register_buffer("l2_weight",
                                 torch.tensor(l2_weight, dtype=torch.get_default_dtype()))
        else:
            self.register_buffer("l2_weight", None)

        self.self_interaction_terms = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=feature_max_l,
            sigmas_receive=feature_smearing_widths,
            normalize_source="multipoles",
            normalize_receive=integral_normalization,
        )
        self.realspace_features = RealSpaceAnalyticalElectrostaticFeatures(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            projection_max_l=feature_max_l,
            projection_smearing_widths=feature_smearing_widths,
            include_self_interaction=include_self_interaction,
            integral_normalization=integral_normalization,
        )

    def forward(
        self,
        source_feats: torch.Tensor,    # [n_nodes, (l+1)^2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        box: torch.Tensor,             # [n_graphs, 3, 3]
        pbc: torch.Tensor,             # [n_graphs, 3] bool
    ) -> torch.Tensor:
        if torch.any(pbc):
            return self._pme_features(source_feats, node_positions, batch, box)
        feats_out, _, _ = self.realspace_features(source_feats, node_positions, batch)
        return feats_out

    def _pme_features(self, source_feats, node_positions, batch, box):
        feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats

        n_out = (self.feature_max_l + 1) ** 2 * self.num_radial
        result = torch.zeros(
            feats.shape[0], n_out, dtype=feats.dtype, device=feats.device
        )

        # box is [n_graphs, 3, 3]; use its shape (a guarded symint) instead of
        # batch.max().item() (an unbacked symint) so this compiles without a graph break.
        n_graphs = int(box.shape[0])
        s3 = math.sqrt(3.0)

        for s_idx, alpha_s in enumerate(self.alphas):
            for g in range(n_graphs):
                mask = batch == g
                pos_g = node_positions[mask]
                q_g = feats[mask, 0]
                # For feature computation we always treat charges as rank-1 if
                # density_max_l >= 1 so dipole sources contribute to l=0 features.
                p_g = feats[mask][:, [3, 1, 2]] if self.density_max_l >= 1 else None  # (x,y,z) Cartesian
                Q_g = None
                if self.density_max_l >= 2:
                    Q_g = _l2_source_to_cartesian(feats[mask][:, 4:9]) * _PME_QUAD_SH_TO_CART

                phi, E_field, H = compute_pme_single(
                    pos_g, box[g], q_g, p_g, alpha_s, self.mesh_size,
                    rank=self.density_max_l, Q=Q_g,
                    want_field=(self.feature_max_l >= 1),
                    want_hessian=(self.feature_max_l >= 2),
                )

                # Self-correction for the potential: remove on-site contribution.
                # phi_corr_i = Σ_{j≠i} q_j T(r_ij) (no self-potential at i)
                phi_corr = phi - 2.0 * alpha_s / math.sqrt(math.pi) * q_g

                # l=0 feature for channel s_idx
                l0_f = self.l0_factors[s_idx] * _PME_TO_CODE * phi_corr
                result[mask, s_idx] += l0_f

                if self.feature_max_l >= 1 and E_field is not None:
                    # E_field = ∇φ at each atom (positive gradient, Gaussian units).
                    # For charge density: the self-gradient is zero by spherical symmetry.
                    # For dipole density: subtract the Ewald dipole self-field.
                    if self.density_max_l >= 1 and p_g is not None:
                        E_use = E_field + (4.0 * alpha_s**3 / 3.0) / math.sqrt(math.pi) * p_g
                    else:
                        E_use = E_field  # charge-only: self-gradient = 0

                    l1_w = self.l1_weight[s_idx]
                    # E_use = -∇φ (electric field, NOT +∇φ): sign from ∂u/∂r = -Nⱼ.
                    # Feature requires +∂V/∂r = -E_use, so subtract.
                    # Layout: [n_radial::3]=y, [n_radial+1::3]=z, [n_radial+2::3]=x
                    result[mask, self.num_radial + s_idx * 3 + 0] -= l1_w * _PME_TO_CODE * E_use[:, 1]
                    result[mask, self.num_radial + s_idx * 3 + 1] -= l1_w * _PME_TO_CODE * E_use[:, 2]
                    result[mask, self.num_radial + s_idx * 3 + 2] -= l1_w * _PME_TO_CODE * E_use[:, 0]

                if self.feature_max_l >= 2 and H is not None:
                    # EFG = ∇∇φ (physical Hessian). The charge/dipole on-site EFG is
                    # traceless-zero by symmetry; only the quadrupole source needs a
                    # self-EFG removal (H_self = (8/(5√π)) α⁵ Q from the energy self-term).
                    H_use = H
                    if self.density_max_l >= 2 and Q_g is not None:
                        H_use = H - (8.0 / (5.0 * math.sqrt(math.pi))) * alpha_s**5 * Q_g
                    tr = (H_use[:, 0, 0] + H_use[:, 1, 1] + H_use[:, 2, 2]) / 3.0
                    l2_w = self.l2_weight[s_idx] * _PME_TO_CODE
                    base = 4 * self.num_radial
                    # real-SH order m=-2,-1,0,+1,+2 = xy, yz, z², xz, x²-y²
                    result[mask, base + s_idx * 5 + 0] += l2_w * (2.0 / s3) * H_use[:, 0, 1]
                    result[mask, base + s_idx * 5 + 1] += l2_w * (2.0 / s3) * H_use[:, 1, 2]
                    result[mask, base + s_idx * 5 + 2] += l2_w * (H_use[:, 2, 2] - tr)
                    result[mask, base + s_idx * 5 + 3] += l2_w * (2.0 / s3) * H_use[:, 0, 2]
                    result[mask, base + s_idx * 5 + 4] += l2_w * (1.0 / s3) * (H_use[:, 0, 0] - H_use[:, 1, 1])

        # The PME potential already excludes the on-site self-interaction
        # (removed by the alpha/sqrt(pi) correction in compute_pme_single).
        # Only add self-interaction when explicitly requested.
        if self.include_self_interaction:
            si_terms = self.self_interaction_terms(feats)
            result = result + si_terms

        return result
