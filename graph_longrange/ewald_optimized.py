"""Opt-in pure-PyTorch direct-k Ewald implementations.

The reference energy/features classes remain unchanged.  This module provides
a single-periodic-graph fast path that avoids constructing a dense all-ones
graph-membership mask.
"""

from __future__ import annotations

import torch

from .energy import GTOElectrostaticEnergy, energy_product_batch
from .features import (
    GTOElectrostaticFeatures,
    apply_coulomb_kernel_batch,
    assemble_fourier_series_batch,
)
from .slabs import slab_dipole_correction_energy


class GTOElectrostaticFeaturesOptimized(GTOElectrostaticFeatures):
    """Reference-compatible features without an all-ones single-graph mask."""

    def _pbc_precompute_static(
        self,
        k_vectors,
        k_norm2,
        k_vector_batch,
        k0_mask,
        volume,
        pbc,
        batch,
    ):
        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        feature_basis_fs = self.feature_basis(k_vectors, k_norm2, k0_mask)
        volume_per_k = volume.reshape(-1)[k_vector_batch]
        k0_mask_bool = k0_mask > 0.0
        k_factor_coulomb = torch.zeros_like(k_norm2)
        k_factor_coulomb[~k0_mask_bool] = 1.0 / k_norm2[~k0_mask_bool]
        k_factor_proj = torch.ones_like(k_norm2)
        k_factor_proj[k0_mask_bool] = 0.5
        single_graph = int(volume.shape[0]) == 1
        cache = {
            "mode": "pbc",
            "k_vectors": k_vectors,
            "k_norm2": k_norm2,
            "k_vector_batch": k_vector_batch,
            "k0_mask": k0_mask,
            "volume_per_k": volume_per_k,
            "k_factor_coulomb": k_factor_coulomb,
            "k_factor_proj": k_factor_proj,
            "density_basis_fs": density_basis_fs,
            "feature_basis_fs": feature_basis_fs,
            "volumes": volume.reshape(-1),
            "batch": batch,
            "pbc": pbc,
            "single_graph": single_graph,
            **self._build_correction_cache(pbc=pbc, batch=batch),
        }
        if not single_graph:
            cache["mask_f"] = (
                k_vector_batch[:, None] == batch[None, :]
            ).to(dtype=density_basis_fs.dtype)
        return cache

    def _pbc_update_positions(self, node_positions: torch.Tensor, static_cache: dict) -> dict:
        if self._warp_kspace_inference_enabled(node_positions):
            return {**static_cache, "node_positions": node_positions}
        phases = torch.matmul(static_cache["k_vectors"], node_positions.t())
        if static_cache.get("single_graph", False):
            cosines = torch.cos(phases)
            sines = torch.sin(phases)
        else:
            cosines = torch.cos(phases) * static_cache["mask_f"]
            sines = torch.sin(phases) * static_cache["mask_f"]
        return {**static_cache, "node_positions": node_positions, "cosines": cosines, "sines": sines}


class GTOElectrostaticEnergyOptimized(GTOElectrostaticEnergy):
    """Reference-compatible direct-k energy with reusable reciprocal geometry."""

    def _finish_energy(
        self,
        density,
        k_norm2,
        k_vector_batch,
        k_factor_coulomb,
        source_feats,
        node_positions,
        batch,
        volume,
        pbc,
    ):
        potential = apply_coulomb_kernel_batch(
            k_norm2=k_norm2,
            density=density,
            k_factor_coulomb=k_factor_coulomb,
        )
        energy = energy_product_batch(
            density=density,
            potential=potential,
            volume=volume,
            k_vector_batch=k_vector_batch,
        )
        feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        if not self.include_self_interaction:
            self_fields = self.self_interaction_terms(feats)
            node_energies = torch.einsum("nb,nb->n", feats, self_fields)
            self_energy = torch.zeros(
                int(volume.shape[0]), dtype=node_energies.dtype, device=node_energies.device
            )
            self_energy.index_add_(0, batch, node_energies)
            energy = energy - self_energy * 0.5

        if self.include_pbc_corrections:
            molecule_correction = self.monopole_dipole_correction(
                feats, node_positions, volume, batch
            )
            slab_correction = slab_dipole_correction_energy(
                feats, node_positions, volume, batch
            )
            slab = torch.tensor([0, 0, 1], dtype=torch.bool, device=pbc.device)
            is_molecule = torch.all(torch.logical_not(pbc), dim=1)
            is_slab = torch.all(torch.logical_xor(slab, pbc), dim=1)
            correction = torch.zeros_like(molecule_correction)
            correction = torch.where(is_molecule, molecule_correction, correction)
            correction = torch.where(is_slab, slab_correction, correction)
            energy = energy + correction
        return energy

    def _pbc_energy_batch(
        self,
        k_vectors,
        k_norm2,
        k_vector_batch,
        k0_mask,
        source_feats,
        node_positions,
        batch,
        volume,
        pbc,
    ):
        phases = torch.matmul(k_vectors, node_positions.t())
        if int(volume.shape[0]) == 1:
            cosines = torch.cos(phases)
            sines = torch.sin(phases)
        else:
            mask = (k_vector_batch[:, None] == batch[None, :]).to(phases.dtype)
            cosines = torch.cos(phases) * mask
            sines = torch.sin(phases) * mask
        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        volume_per_k = volume.reshape(-1)[k_vector_batch]
        density = assemble_fourier_series_batch(
            source_feats, cosines, sines, density_basis_fs, volume_per_k
        )
        k0 = k0_mask > 0.0
        k_factor_coulomb = torch.zeros_like(k_norm2)
        k_factor_coulomb[~k0] = 1.0 / k_norm2[~k0]
        return self._finish_energy(
            density,
            k_norm2,
            k_vector_batch,
            k_factor_coulomb,
            source_feats,
            node_positions,
            batch,
            volume,
            pbc,
        )

__all__ = ["GTOElectrostaticEnergyOptimized", "GTOElectrostaticFeaturesOptimized"]
