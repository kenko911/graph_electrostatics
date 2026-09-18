###########################################################################################
# Warp-backed Ewald + PME electrostatics via NVIDIA nvalchemiops
#
# `nvalchemi-toolkit-ops` (repo: nvalchemi-toolkit-ops, package `nvalchemiops`) ships Warp
# GPU implementations that were built for parity with THIS package ("customer reference"):
#
#   graph_longrange (torch reference)          nvalchemiops (Warp)
#   -----------------------------------------  ------------------------------------------
#   GTOElectrostaticEnergy  (k-space Ewald)    multipole_electrostatic_energy
#   GTOElectrostaticFeatures (k-space Ewald)   multipole_electrostatic_features
#   PMEElectrostaticEnergy   (B-spline mesh)   multipole_pme_reciprocal_space
#   PMEElectrostaticFeatures (B-spline mesh)   multipole_electrostatic_features (direct-k)
#
# Shared conventions (verified in source): FIELD_CONSTANT = 1/5.526349406e-3 (e, V, Å),
# e3nn moment packing [q, mu_y, mu_z, mu_x, 5 x l=2], GTO exp(-sigma^2 k^2) Green's factor,
# F/(4pi) output scaling, NormMode MULTIPOLES / RECEIVER = the reference's normalization
# strings, half-space-with-origin k conventions, `include_self_interaction` semantics.
# Their docstrings claim bit-for-bit parity at l_max <= 1 under matched inputs; l=2 parity
# and all gradients must be CONFIRMED numerically with
# benchmarks_pme/benchmark_toolkit_electrostatics.py before production use.
#
# Install (user venv):  pip install -e ~/projects/aip-aspuru/kenko/repos/nvalchemi-toolkit-ops
#
# l=2 conventions (RESOLVED, verified numerically on CPU 2026-07-06):
#  * The five l=2 channels are interpreted differently by the two packages. The verified
#    source-side map is applied automatically in `_moments()`:
#        f5_toolkit = (2/3) * pinv(C_tk) @ C_ref @ f5_ref     (see _l2_source_transform)
#    With it, energies match to ~4e-7 and features to ~1e-8 per l-block (probe script:
#    benchmarks_pme/probe_l2_transform.py). l<=1 needs no transform (parity ~1e-7 raw).
#  * The toolkit's l=2 FEATURE self-interaction constants differ from the reference's, so
#    the feature adapters always call the toolkit with include_self_interaction=True and
#    subtract the reference GTOSelfInteractionBlock torch-side.
#
# Remaining notes:
#  * PMEElectrostaticEnergyWarp uses the toolkit's background correction (non-neutral
#    systems); the torch reference has no explicit background term. Neutral systems agree.
#  * PMEElectrostaticFeaturesWarp replaces the mesh-PME feature approximation with the
#    toolkit's *direct k-space* Warp features (needs `k_cutoff`). That is the EXACT quantity
#    the mesh path approximates, so accuracy goes UP; cost is O(N*K) with Warp instead of
#    O(N log N) torch mesh -- benchmark decides which wins at your (N, K).
###########################################################################################

from __future__ import annotations

from functools import partial

import torch

from .pme import PMEElectrostaticEnergy, PMEElectrostaticFeatures
from .pme_optimized import _pme_energy_impl, _pme_features_impl

try:
    from nvalchemiops.torch.interactions.electrostatics import (
        multipole_electrostatic_energy,
        multipole_electrostatic_features,
    )
    from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
        multipole_pme_reciprocal_space,
    )

    HAS_NVALCHEMIOPS = True
except Exception:  # pragma: no cover - toolkit optional
    HAS_NVALCHEMIOPS = False

_INSTALL_MSG = (
    "nvalchemiops is not installed. Install it into the active venv with:\n"
    "  pip install -e ~/projects/aip-aspuru/kenko/repos/nvalchemi-toolkit-ops"
)


def _require_toolkit():
    if not HAS_NVALCHEMIOPS:
        raise ImportError(_INSTALL_MSG)


_L2_S_CACHE: dict = {}


def _l2_source_transform(dtype, device) -> torch.Tensor:
    """5x5 map from graph_longrange l=2 channels to nvalchemiops l=2 channels.

    The two packages interpret the five l=2 slots differently (the toolkit uses e3nn
    component normalization in e3nn's internal axis convention; graph_longrange uses
    [sqrt(3)xy, sqrt(3)yz, (3z^2-r^2)/2, sqrt(3)xz, sqrt(3)(x^2-y^2)/2] in standard axes).
    Both convert to traceless Cartesian with known linear maps C_ref / C_tk, and the
    k-space energy functionals agree under

        f5_toolkit = (2/3) * pinv(C_tk) @ C_ref @ f5_ref

    VERIFIED numerically (benchmarks_pme/probe_l2_transform.py): energy bilinear forms
    match to ~5e-8 and random-input energies to ~4e-7 (k-truncation level); with this
    transform features match to ~1e-8 per l-block.  The 2/3 is the same constant as
    pme._PME_QUAD_SH_TO_CART.
    """
    key = (dtype, str(device))
    S = _L2_S_CACHE.get(key)
    if S is None:
        from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
            e3nn_to_cartesian_quadrupole,
        )

        from .realspace_electrostatics import _l2_source_to_cartesian

        eye = torch.eye(5, dtype=torch.float64)
        c_ref = _l2_source_to_cartesian(eye).reshape(5, 9).T
        c_tk = e3nn_to_cartesian_quadrupole(eye).reshape(5, 9).T
        S = ((2.0 / 3.0) * torch.linalg.pinv(c_tk) @ c_ref).to(dtype=dtype, device=device)
        _L2_S_CACHE[key] = S
    return S


def _moments(feats: torch.Tensor, density_max_l: int) -> torch.Tensor:
    """Packed e3nn moments for the toolkit: slice to (N, (l+1)^2), remap the l=2 block."""
    feats = feats.squeeze(-2) if feats.dim() == 3 else feats
    m = feats[:, : (density_max_l + 1) ** 2]
    if density_max_l >= 2:
        S = _l2_source_transform(m.dtype, m.device)
        m = torch.cat([m[:, :4], m[:, 4:9] @ S.T], dim=1)
    return m


# ---------------------------------------------------------------------------------------
# Ewald (direct k-space) functional adapters — mirror the GTO* classes' periodic core.
# The molecule/slab corrections and the non-periodic evaluator are cheap torch code and
# intentionally stay with the reference classes.
# ---------------------------------------------------------------------------------------
def ewald_energy_warp(
    source_feats: torch.Tensor,     # [N, (l+1)^2] e3nn-packed (or [N, 1, m])
    node_positions: torch.Tensor,   # [N, 3]
    cell: torch.Tensor,             # [3, 3] or [B, 3, 3]
    *,
    density_max_l: int,
    density_smearing_width: float,
    k_cutoff: float,
    batch: torch.Tensor | None = None,   # [N] int, sorted; None for single system
    include_self_interaction: bool = False,
    k_vectors: torch.Tensor | None = None,  # pass the model's k-set for exact parity (single)
) -> torch.Tensor:
    """Warp GTO k-space Ewald energy (per-atom, FIELD_CONSTANT units).

    Parity target: ``GTOElectrostaticEnergy._pbc_energy_batch`` minus the molecule/slab
    correction terms (add those in torch if needed).  Sum over atoms of one graph equals
    the reference per-graph energy.  With the toolkit's own ``k_cutoff`` grid the k-set
    differs slightly from ``compute_k_vectors_flat`` (both truncate the same sum; ~1e-6
    rel); pass ``k_vectors=`` (single-system) for exact-k parity.
    """
    _require_toolkit()
    return multipole_electrostatic_energy(
        node_positions,
        _moments(source_feats, density_max_l),
        cell,
        batch_idx=None if batch is None else batch.to(torch.int32),
        sigma=density_smearing_width,
        k_cutoff=None if k_vectors is not None else k_cutoff,
        k_vectors=k_vectors,
        normalize="multipoles",
        include_self_interaction=include_self_interaction,
    )


def ewald_features_warp(
    source_feats: torch.Tensor,
    node_positions: torch.Tensor,
    cell: torch.Tensor,
    *,
    density_max_l: int,
    density_smearing_width: float,
    feature_max_l: int,
    feature_smearing_widths: list[float],
    k_cutoff: float,
    batch: torch.Tensor | None = None,
    include_self_interaction: bool = False,
    integral_normalization: str = "receiver",
    k_vectors: torch.Tensor | None = None,  # pass the model's k-set for exact parity (single)
) -> torch.Tensor:
    """Warp GTO k-space Ewald features [N, n_sigma * (feature_max_l+1)^2].

    Parity target: ``GTOElectrostaticFeatures._pbc_forward_dynamic`` (same permuted-flat
    output layout), minus the non-periodic correction terms.

    Self-interaction is ALWAYS subtracted torch-side with the reference
    ``GTOSelfInteractionBlock`` (the toolkit's own l=2 self constants differ; verified);
    the toolkit is called with ``include_self_interaction=True`` (subtract nothing).
    """
    _require_toolkit()
    feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
    out = multipole_electrostatic_features(
        node_positions,
        _moments(feats, density_max_l),
        cell,
        batch_idx=None if batch is None else batch.to(torch.int32),
        sigma=density_smearing_width,
        receiver_sigmas=list(feature_smearing_widths),
        k_cutoff=None if k_vectors is not None else k_cutoff,
        k_vectors=k_vectors,
        feature_max_l=feature_max_l,
        density_normalize="multipoles",
        feature_normalize=integral_normalization,
        include_self_interaction=True,   # subtract reference-side below instead
    ).to(feats.dtype)
    if not include_self_interaction:
        from .gto_utils import GTOSelfInteractionBlock

        block = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=feature_max_l,
            sigmas_receive=list(feature_smearing_widths),
            normalize_source="multipoles",
            normalize_receive=integral_normalization,
        ).to(feats.device)
        out = out - block(feats[:, : (density_max_l + 1) ** 2])
    return out


# ---------------------------------------------------------------------------------------
# PME drop-in modules — subclass the torch reference and override only the periodic core.
# Realspace fallback (pbc=False), self-interaction add-back, and molecule/slab corrections
# are inherited from the reference implementation unchanged.
# ---------------------------------------------------------------------------------------
class PMEElectrostaticEnergyWarp(PMEElectrostaticEnergy):
    """PMEElectrostaticEnergy with the per-graph torch PME replaced by nvalchemiops.

    One batched Warp/FFT call replaces the python loop over graphs; energies come back
    per-atom, already self/background-corrected, in FIELD_CONSTANT units (same as the
    reference's ``* _PME_TO_CODE`` scaling).  ``spline_order=6`` matches the reference
    B-spline order.

    alpha convention: the toolkit PME is an Ewald-SPLIT reciprocal — its Green's function
    damps with ``exp(-(1/(4a^2) + sigma^2) k^2)`` and its corrections use the combined
    width ``sigma_c = sqrt(sigma^2 + 1/(4a^2))``.  The graph_longrange reference is the
    FULL smeared-GTO energy (no real-space complement), i.e. the a -> infinity limit, so
    we pass ``SPLIT_ALPHA`` (1/(4a^2) ~ 2.5e-17, negligible vs sigma^2) instead of the
    reference's ``self.alpha = 1/(2 sigma)`` (which would double-damp).
    """

    # The toolkit JIT-compiles a SEPARATE kernel per (order, l_max, dtype); order 6 + l=2 is
    # the single heaviest corner in the library (216-pt stencil + quadrupole Hessian gather)
    # and takes minutes to compile the FIRST time (then cached in ~/.cache/warp -> instant).
    # Order 4 is the PME standard, compiles ~10x faster, and only trades a little mesh
    # accuracy (NOT a convention change -- we validate against the exact Ewald oracle).
    SPLINE_ORDER = 4
    SPLIT_ALPHA = 1.0e8   # effectively alpha = infinity (all interaction in k-space)

    def __init__(self, *args, spline_order: int | None = None, **kwargs):
        _require_toolkit()
        super().__init__(*args, **kwargs)
        if spline_order is not None:
            self.SPLINE_ORDER = int(spline_order)

    def _pme_energy(self, source_feats, node_positions, batch, box, volume, pbc):
        feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        moments = _moments(feats, self.density_max_l)
        n_graphs = int(volume.shape[0])
        single = n_graphs == 1

        e_atoms = multipole_pme_reciprocal_space(
            node_positions,
            moments,
            box.squeeze(0) if single else box,
            sigma=self.density_smearing_width,
            alpha=self.SPLIT_ALPHA,   # NOT self.alpha: reference = alpha->infinity limit
            mesh_dimensions=(self.mesh_size, self.mesh_size, self.mesh_size),
            spline_order=self.SPLINE_ORDER,
            batch_idx=None if single else batch.to(torch.int32),
            volume=volume.reshape(-1),      # tensor keeps cell autograd (stress) alive
        )
        energies = torch.zeros(n_graphs, dtype=e_atoms.dtype, device=e_atoms.device)
        energies.index_add_(0, batch, e_atoms)
        energies = energies.to(feats.dtype)

        # --- identical tail to the torch reference (self add-back + pbc corrections) ---
        if self.include_self_interaction:
            self_fields = self.self_interaction_terms(feats)
            node_energies = torch.einsum("nb,nb->n", feats, self_fields)
            self_energy = torch.zeros(n_graphs, dtype=node_energies.dtype,
                                      device=node_energies.device)
            self_energy.index_add_(0, batch, node_energies)
            energies = energies + self_energy * 0.5

        if self.include_pbc_corrections:
            from .slabs import slab_dipole_correction_energy

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


def _compute_pme_single_fused_warp(
    coords: torch.Tensor,
    box: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor | None,
    alpha: float,
    K: int,
    rank: int,
    want_field: bool = False,
    Q: torch.Tensor | None = None,
    want_hessian: bool = False,
    reciprocal_cache: dict | None = None,
    geometry_cache: dict | None = None,
    *,
    spline_order: int = 6,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused Warp spread/gather with the reference PME Green function.

    Keeping graph_longrange's reciprocal factors makes this a backend change,
    rather than a change to the mesh approximation. Fractional moments let the
    unified spread-transpose gather return potential, field, and Hessian in one
    differentiable operation while retaining live-cell NPT gradients.
    """
    del geometry_cache  # The Warp primitive builds and fuses its own stencil.
    from .pme import _precompute_pme_reciprocal

    if reciprocal_cache is None:
        reciprocal_cache = _precompute_pme_reciprocal(box, alpha, K)

    n_atoms = coords.shape[0]
    cell = box.unsqueeze(0) if box.dim() == 2 else box
    cell_inv_batched = torch.linalg.inv_ex(cell)[0]
    cell_inv_t = cell_inv_batched.transpose(-1, -2).contiguous()
    atom_batch = torch.zeros(n_atoms, dtype=torch.int32, device=coords.device)
    dipoles = p if rank >= 1 and p is not None else torch.zeros_like(coords)
    quadrupoles = (
        Q
        if rank >= 2 and Q is not None
        else torch.zeros((n_atoms, 3, 3), dtype=coords.dtype, device=coords.device)
    )
    p_frac, d_frac, q_frac = torch.ops.nvalchemiops.multipole_pme_fractionalize(
        coords,
        cell_inv_t,
        dipoles,
        quadrupoles,
        atom_batch,
    )
    identity_cell = torch.eye(3, dtype=coords.dtype, device=coords.device).unsqueeze(0)
    rho_grid = torch.ops.nvalchemiops.multipole_pme_spread_unified(
        p_frac,
        q,
        d_frac,
        q_frac,
        identity_cell,
        K,
        K,
        K,
        spline_order,
        rank,
    )

    # Use rFFT for the real mesh but retain the exact reference Green factors.
    # This deliberately avoids nvalchemiops' sinc approximation, which defines
    # a measurably different PME discretization at order 6.
    rho_k = torch.fft.rfftn(rho_grid, norm="backward").contiguous()
    nz_rfft = K // 2 + 1
    green = reciprocal_cache["coulomb"].reshape(K, K, K)[:, :, :nz_rfft]
    theta = reciprocal_cache["theta_safe"].reshape(K, K, K)[:, :, :nz_rfft]
    phi_k = rho_k * green / theta.pow(2)
    phi_grid = torch.fft.irfftn(phi_k, s=(K, K, K), norm="forward").to(coords.dtype)

    phi, gathered_d, gathered_Q = torch.ops.nvalchemiops.multipole_pme_gather_via_spread_t(
        phi_grid,
        p_frac,
        identity_cell,
        K,
        K,
        K,
        spline_order,
        rank,
    )
    want_field_actual = want_field or rank >= 1
    want_hessian_actual = want_hessian or rank >= 2
    cell_inv = cell_inv_batched[0]
    # Fractionalization uses s = r @ A^-1 (cell_inv_t passed to multipole_pme_fractionalize),
    # so ds_i/dr_j = A^-1[j,i] and the back-transform of a fractional-frame gradient is A^-T,
    # with the Hessian going as A^-1 H A^-T.  Both were transposed.  Identical for an orthogonal
    # cell (A diagonal), wrong on a triclinic one -- same class of bug as the torch gather.
    field = (-gathered_d @ cell_inv.T) if want_field_actual else None
    hessian = (
        torch.einsum("ac,ncd,bd->nab", cell_inv, 2.0 * gathered_Q, cell_inv)
        if want_hessian_actual
        else None
    )
    return phi, field, hessian


class PMEElectrostaticFeaturesWarpPME(PMEElectrostaticFeatures):
    """Experimental fused mesh-PME feature backend.

    The Torch PME remains the default. This backend is selected explicitly by
    TensorNetPolar ``use_warp_pme=True`` and keeps the same mesh size, order-6
    spline, Green function, feature layout, and molecular fallback.
    """

    def __init__(self, *args, spline_order: int = 6, **kwargs):
        _require_toolkit()
        super().__init__(*args, **kwargs)
        self.spline_order = int(spline_order)

    def _pme_features(self, source_feats, node_positions, batch, box):
        return _pme_features_impl(
            self,
            source_feats,
            node_positions,
            batch,
            box,
            partial(_compute_pme_single_fused_warp, spline_order=self.spline_order),
        )


class PMEElectrostaticEnergyWarpPME(PMEElectrostaticEnergy):
    """Experimental exact-Green fused mesh-PME energy backend.

    Unlike :class:`PMEElectrostaticEnergyWarp`, this class keeps the reference
    graph_longrange PME discretization and only replaces spread/gather with the
    differentiable unified Warp operations. This is the energy counterpart of
    :class:`PMEElectrostaticFeaturesWarpPME`.
    """

    def __init__(self, *args, spline_order: int = 6, **kwargs):
        _require_toolkit()
        super().__init__(*args, **kwargs)
        self.spline_order = int(spline_order)

    def _pme_energy(self, source_feats, node_positions, batch, box, volume, pbc):
        return _pme_energy_impl(
            self,
            source_feats,
            node_positions,
            batch,
            box,
            volume,
            pbc,
            partial(_compute_pme_single_fused_warp, spline_order=self.spline_order),
        )


class PMEElectrostaticFeaturesWarp(PMEElectrostaticFeatures):
    """High-accuracy DIRECT-k Warp features -- an ACCURACY option, NOT a large-N speedup.

    Replaces the mesh-PME feature loop with the toolkit's exact k-space projection
    (``multipole_electrostatic_features``).  It is the exact quantity the mesh approximates,
    so accuracy goes UP (l=2 rel ~1e-6 vs the mesh's ~1e-5).

    SCALING WARNING (measured on L40S): the toolkit has NO mesh-based feature path -- this is
    O(N*K) direct k-space.  Benchmarked at N=3375 it was ~20x SLOWER than the torch mesh
    features, and it runs OUT OF MEMORY at N>=8000.  The parent torch ``PMEElectrostaticFeatures``
    is O(N log N) mesh and scales fine.  So:
        * small / medium N where accuracy matters and it fits  -> this class.
        * large N (production periodic)                        -> keep the torch mesh parent.
    A mesh-PME Warp features path could be built from the toolkit's ``multipole_pme_gather_*``
    kernels (phi / field / hessian), but that is a separate implementation, not done here.
    Requires ``k_cutoff`` (reciprocal cutoff, same meaning as the model's ``kspace_cutoff``).
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        feature_max_l: int,
        feature_smearing_widths: list[float],
        mesh_size: int,                       # unused by the direct-k path; kept for API parity
        include_self_interaction: bool = False,
        integral_normalization: str = "receiver",
        *,
        k_cutoff: float,
    ):
        _require_toolkit()
        super().__init__(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            feature_max_l=feature_max_l,
            feature_smearing_widths=feature_smearing_widths,
            mesh_size=mesh_size,
            include_self_interaction=include_self_interaction,
            integral_normalization=integral_normalization,
        )
        self.k_cutoff = float(k_cutoff)
        self._integral_normalization = integral_normalization
        self._density_smearing_width = float(density_smearing_width)
        self._feature_smearing_widths = list(feature_smearing_widths)

    def _pme_features(self, source_feats, node_positions, batch, box):
        feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        moments = _moments(feats, self.density_max_l)
        single = int(box.shape[0]) == 1

        out = multipole_electrostatic_features(
            node_positions,
            moments,
            box.squeeze(0) if single else box,
            batch_idx=None if single else batch.to(torch.int32),
            sigma=self._density_smearing_width,
            receiver_sigmas=list(self._feature_smearing_widths),
            k_cutoff=self.k_cutoff,
            feature_max_l=self.feature_max_l,
            density_normalize="multipoles",
            feature_normalize=self._integral_normalization,
            include_self_interaction=True,   # subtract reference-side below instead
        ).to(feats.dtype)
        if not self.include_self_interaction:
            # parent's block has exactly the reference l<=2 self constants
            out = out - self.self_interaction_terms(feats[:, : (self.density_max_l + 1) ** 2])
        return out
