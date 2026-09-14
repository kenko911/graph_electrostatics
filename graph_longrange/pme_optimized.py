"""Opt-in pure-PyTorch PME implementations.

The reference implementation in :mod:`graph_longrange.pme` is intentionally
left unchanged.  Optimizations live here so their value and numerical behavior
can be tested independently before any are considered for the default path.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .pme import (
    PMEElectrostaticEnergy,
    PMEElectrostaticFeatures,
    _PME_QUAD_SH_TO_CART,
    _PME_TO_CODE,
    _bspline,
    _bspline_double_prime,
    _bspline_prime,
    _cached_pme_reciprocal,
    _fixed_pme_grid,
    _get_recip_vectors,
    _get_u_reference,
    _grid_hessian_weights,
    _l2_source_to_cartesian,
    _make_stencil,
)
from .slabs import slab_dipole_correction_energy


def build_pme_geometry_optimized(
    coords: torch.Tensor,
    box: torch.Tensor,
    K: int,
    rank: int = 2,
    Nj: torch.Tensor | None = None,
) -> dict:
    """Build the shared order-6 spread/gather stencil for one MD structure."""
    fixed_grid = _fixed_pme_grid(K, coords.device, coords.dtype)
    N = fixed_grid["N"]
    if Nj is None:
        Nj = _get_recip_vectors(N, box)
    m_u0, u0 = _get_u_reference(coords, Nj, order=6)
    shifts = _make_stencil(6, coords.device, coords.dtype)
    grid_shape = fixed_grid["grid_shape"]
    _, ny, nz = grid_shape
    indices = (m_u0[:, None, :] + shifts[0]) % N.int()[None, None, :]
    flat_indices = (
        indices[:, :, 0] * ny * nz + indices[:, :, 1] * nz + indices[:, :, 2]
    ).long()
    u = u0[:, None, :] + shifts
    basis = _bspline(u)
    basis_product = basis.prod(dim=2)
    gradient_weights = None
    hessian_weights = None
    if rank >= 1:
        basis_gradient = _bspline_prime(u)
        gradient_weights = torch.stack(
            [
                basis_gradient[:, :, 0] * basis[:, :, 1] * basis[:, :, 2],
                basis[:, :, 0] * basis_gradient[:, :, 1] * basis[:, :, 2],
                basis[:, :, 0] * basis[:, :, 1] * basis_gradient[:, :, 2],
            ],
            dim=2,
        )
    if rank >= 2:
        basis_hessian = _bspline_double_prime(u)
        hessian_weights = _grid_hessian_weights(basis, basis_gradient, basis_hessian)
    return {
        "N": N,
        "Nj": Nj,
        "grid_shape": grid_shape,
        "flat_indices": flat_indices,
        "basis_product": basis_product,
        "gradient_weights": gradient_weights,
        "hessian_weights": hessian_weights,
    }


def _spread_with_geometry(
    geometry: dict,
    q: torch.Tensor,
    p: torch.Tensor | None,
    Q: torch.Tensor | None,
) -> torch.Tensor:
    weights = q[:, None] * geometry["basis_product"]
    gradient_weights = geometry["gradient_weights"]
    if p is not None:
        p_grid = torch.matmul(p, geometry["Nj"].T)
        weights = weights - (p_grid[:, None, :] * gradient_weights).sum(dim=2)
    if Q is not None:
        Nj = geometry["Nj"]
        Q_grid = torch.einsum("ab,nbc,dc->nad", Nj, Q, Nj)
        weights = weights + 0.5 * (
            Q_grid[:, None, :, :] * geometry["hessian_weights"]
        ).sum(dim=(-1, -2))
    mesh = torch.zeros(
        math.prod(geometry["grid_shape"]), dtype=weights.dtype, device=weights.device
    )
    mesh.index_add_(0, geometry["flat_indices"].reshape(-1), weights.reshape(-1))
    return mesh.reshape(geometry["grid_shape"])


def _gather_with_geometry(
    potential_grid: torch.Tensor,
    geometry: dict,
    want_field: bool,
    want_hessian: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    local = potential_grid.reshape(-1)[geometry["flat_indices"]]
    phi = (local * geometry["basis_product"]).sum(dim=1)
    field = None
    hessian = None
    if want_field:
        field_grid = (local.unsqueeze(-1) * geometry["gradient_weights"]).sum(dim=1)
        field = torch.matmul(field_grid, geometry["Nj"].T)
    if want_hessian:
        hessian_grid = (
            local[:, :, None, None] * geometry["hessian_weights"]
        ).sum(dim=1)
        Nj = geometry["Nj"]
        hessian = torch.einsum("ac,ncd,bd->nab", Nj, hessian_grid, Nj)
    return phi, field, hessian


def compute_pme_single_optimized(
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
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Single-structure PME using rFFT and one shared spread/gather stencil.

    The reference discretization and order-6 spline are retained.  The real
    mesh uses the nonredundant Hermitian half-spectrum, while precomputed
    interpolation weights are reused for spreading and gathering.
    """
    from .pme import _precompute_pme_reciprocal

    if reciprocal_cache is None:
        reciprocal_cache = _precompute_pme_reciprocal(box, alpha, K)
    if geometry_cache is None:
        geometry_cache = build_pme_geometry_optimized(
            coords, box, K, rank=max(rank, int(want_field), 2 * int(want_hessian)),
            Nj=reciprocal_cache["Nj"],
        )
    mesh = _spread_with_geometry(
        geometry_cache,
        q,
        p if rank >= 1 else None,
        Q if rank >= 2 else None,
    )
    structure_factor = torch.fft.rfftn(mesh)
    half_z = K // 2 + 1
    coulomb = reciprocal_cache["coulomb"].reshape(K, K, K)[:, :, :half_z]
    theta = reciprocal_cache["theta_safe"].reshape(K, K, K)[:, :, :half_z]
    potential_k = coulomb * structure_factor / theta.pow(2)
    potential_grid = torch.fft.irfftn(
        potential_k, s=(K, K, K), norm="forward"
    )
    return _gather_with_geometry(
        potential_grid,
        geometry_cache,
        want_field=want_field or rank >= 1,
        want_hessian=want_hessian or rank >= 2,
    )


PMECompute = Callable[..., tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]


def _pme_energy_impl(
    module,
    source_feats,
    node_positions,
    batch,
    box,
    volume,
    pbc,
    compute: PMECompute,
    geometry_cache: list[dict] | None = None,
):
    feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
    n_graphs = int(volume.shape[0])
    energy_parts = []
    reciprocal = (
        _cached_pme_reciprocal(module, box, [module.alpha], module.mesh_size)[0]
        if geometry_cache is None
        else None
    )

    for graph_index in range(n_graphs):
        mask = batch == graph_index
        positions = node_positions[mask]
        q = feats[mask, 0]
        dipoles = feats[mask][:, [3, 1, 2]] if module.density_max_l >= 1 else None
        quadrupoles = None
        if module.density_max_l >= 2:
            quadrupoles = _l2_source_to_cartesian(feats[mask][:, 4:9]) * _PME_QUAD_SH_TO_CART

        phi, field, hessian = compute(
            positions,
            box[graph_index],
            q,
            dipoles,
            module.alpha,
            module.mesh_size,
            rank=module.density_max_l,
            Q=quadrupoles,
            reciprocal_cache=(
                reciprocal[graph_index]
                if reciprocal is not None
                else geometry_cache[graph_index]["reciprocal"][module.alpha]
            ),
            geometry_cache=None if geometry_cache is None else geometry_cache[graph_index],
        )
        alpha = module.alpha
        phi_corr = phi - 2.0 * alpha / math.sqrt(math.pi) * q
        energy = 0.5 * (q * phi_corr).sum()
        if dipoles is not None and field is not None:
            field_corr = field + alpha * (4.0 * alpha**2 / 3.0) / math.sqrt(math.pi) * dipoles
            energy = energy - 0.5 * (dipoles * field_corr).sum()
        if quadrupoles is not None and hessian is not None:
            self_quad = (
                2.0 / (5.0 * math.sqrt(math.pi))
            ) * alpha**5 * (quadrupoles * quadrupoles).sum()
            energy = energy + 0.25 * (quadrupoles * hessian).sum() - self_quad
        energy_parts.append(energy * _PME_TO_CODE)

    energies = torch.stack(energy_parts)
    if module.include_self_interaction:
        self_fields = module.self_interaction_terms(feats)
        node_energies = torch.einsum("nb,nb->n", feats, self_fields)
        self_energy = torch.zeros(n_graphs, dtype=feats.dtype, device=feats.device)
        self_energy.index_add_(0, batch, node_energies)
        energies = energies + 0.5 * self_energy

    # Fully periodic systems need neither correction. Avoid constructing both
    # correction graphs merely to discard them with torch.where.
    if module.include_pbc_corrections and not torch.all(pbc):
        slab = torch.tensor([0, 0, 1], dtype=torch.bool, device=pbc.device)
        is_molecule = torch.all(~pbc, dim=1)
        is_slab = torch.all(torch.logical_xor(slab, pbc), dim=1)
        mol_corr = module.monopole_dipole_correction(feats, node_positions, volume, batch)
        slab_corr = slab_dipole_correction_energy(feats, node_positions, volume, batch)
        correction = torch.zeros_like(mol_corr)
        correction = torch.where(is_molecule, mol_corr, correction)
        correction = torch.where(is_slab, slab_corr, correction)
        energies = energies + correction
    return energies


def _pme_features_impl(
    module,
    source_feats,
    node_positions,
    batch,
    box,
    compute: PMECompute,
    geometry_cache: list[dict] | None = None,
):
    feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
    n_out = (module.feature_max_l + 1) ** 2 * module.num_radial
    result = torch.zeros(feats.shape[0], n_out, dtype=feats.dtype, device=feats.device)
    n_graphs = int(box.shape[0])
    reciprocal = (
        _cached_pme_reciprocal(module, box, module.alphas, module.mesh_size)
        if geometry_cache is None
        else None
    )
    sqrt_three = math.sqrt(3.0)

    for sigma_index, alpha in enumerate(module.alphas):
        for graph_index in range(n_graphs):
            mask = batch == graph_index
            positions = node_positions[mask]
            q = feats[mask, 0]
            dipoles = feats[mask][:, [3, 1, 2]] if module.density_max_l >= 1 else None
            quadrupoles = None
            if module.density_max_l >= 2:
                quadrupoles = _l2_source_to_cartesian(feats[mask][:, 4:9]) * _PME_QUAD_SH_TO_CART
            phi, field, hessian = compute(
                positions,
                box[graph_index],
                q,
                dipoles,
                alpha,
                module.mesh_size,
                rank=module.density_max_l,
                Q=quadrupoles,
                want_field=module.feature_max_l >= 1,
                want_hessian=module.feature_max_l >= 2,
                reciprocal_cache=(
                    reciprocal[sigma_index][graph_index]
                    if reciprocal is not None
                    else geometry_cache[graph_index]["reciprocal"][alpha]
                ),
                geometry_cache=None if geometry_cache is None else geometry_cache[graph_index],
            )

            phi_corr = phi - 2.0 * alpha / math.sqrt(math.pi) * q
            result[mask, sigma_index] += module.l0_factors[sigma_index] * _PME_TO_CODE * phi_corr
            if module.feature_max_l >= 1 and field is not None:
                field_use = field
                if dipoles is not None:
                    field_use = field + (4.0 * alpha**3 / 3.0) / math.sqrt(math.pi) * dipoles
                weight = module.l1_weight[sigma_index] * _PME_TO_CODE
                base = module.num_radial + sigma_index * 3
                result[mask, base + 0] -= weight * field_use[:, 1]
                result[mask, base + 1] -= weight * field_use[:, 2]
                result[mask, base + 2] -= weight * field_use[:, 0]
            if module.feature_max_l >= 2 and hessian is not None:
                hessian_use = hessian
                if quadrupoles is not None:
                    hessian_use = hessian - (
                        8.0 / (5.0 * math.sqrt(math.pi))
                    ) * alpha**5 * quadrupoles
                trace = hessian_use.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
                weight = module.l2_weight[sigma_index] * _PME_TO_CODE
                base = 4 * module.num_radial + sigma_index * 5
                result[mask, base + 0] += weight * (2.0 / sqrt_three) * hessian_use[:, 0, 1]
                result[mask, base + 1] += weight * (2.0 / sqrt_three) * hessian_use[:, 1, 2]
                result[mask, base + 2] += weight * (hessian_use[:, 2, 2] - trace)
                result[mask, base + 3] += weight * (2.0 / sqrt_three) * hessian_use[:, 0, 2]
                result[mask, base + 4] += weight * (1.0 / sqrt_three) * (
                    hessian_use[:, 0, 0] - hessian_use[:, 1, 1]
                )
    if module.include_self_interaction:
        result = result + module.self_interaction_terms(feats)
    return result


class PMEElectrostaticEnergyOptimized(PMEElectrostaticEnergy):
    """Opt-in pure-PyTorch PME energy; reference class remains unchanged."""

    def _pme_energy(self, source_feats, node_positions, batch, box, volume, pbc):
        return _pme_energy_impl(
            self, source_feats, node_positions, batch, box, volume, pbc, compute_pme_single_optimized
        )

    def forward_from_geometry(
        self, source_feats, node_positions, batch, box, volume, pbc, geometry_cache
    ):
        if torch.any(pbc):
            return _pme_energy_impl(
                self,
                source_feats,
                node_positions,
                batch,
                box,
                volume,
                pbc,
                compute_pme_single_optimized,
                geometry_cache,
            )
        return self.realspace_energy(source_feats=source_feats, positions=node_positions, batch=batch)


class PMEElectrostaticFeaturesOptimized(PMEElectrostaticFeatures):
    """Opt-in pure-PyTorch PME features; reference class remains unchanged."""

    def _pme_features(self, source_feats, node_positions, batch, box):
        return _pme_features_impl(
            self, source_feats, node_positions, batch, box, compute_pme_single_optimized
        )

    def forward_from_geometry(
        self, source_feats, node_positions, batch, box, pbc, geometry_cache
    ):
        if torch.any(pbc):
            return _pme_features_impl(
                self,
                source_feats,
                node_positions,
                batch,
                box,
                compute_pme_single_optimized,
                geometry_cache,
            )
        feats_out, _, _ = self.realspace_features(source_feats, node_positions, batch)
        return feats_out


__all__ = [
    "PMEElectrostaticEnergyOptimized",
    "PMEElectrostaticFeaturesOptimized",
    "build_pme_geometry_optimized",
    "compute_pme_single_optimized",
]
