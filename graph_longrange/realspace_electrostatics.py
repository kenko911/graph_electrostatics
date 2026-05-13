import torch
from scipy.constants import e, epsilon_0, pi
from mace.tools.scatter import scatter_sum
from .utils import FIELD_CONSTANT
from typing import List, Optional, Tuple
import warnings
from .gto_utils import (
    GTOSelfInteractionBlock,
    get_Cl_sigma,
)

@torch.no_grad()
def batch_complete_graph_excluding_self_duplicates_vector(
    batch: torch.Tensor, N: int
) -> torch.Tensor:
    """
    Duplicate each node N times, then for each graph build directed
    edges between every pair of duplicates *unless* they share the same
    original node ID.

    Args:
        batch (LongTensor): shape [M], graph ID of each original node.
        N (int): number of duplicates per node.

    Returns:
        edge_index (LongTensor[2, E])
    """
    batch = batch.long()
    orig = torch.arange(batch.size(0), device=batch.device)
    # duplicated per-node graph ID and original-ID
    batch2 = batch.repeat_interleave(N)  # [M*N]
    orig2 = orig.repeat_interleave(N)  # [M*N]

    G = int(batch2.max().item()) + 1
    edges = []

    for g in range(G):
        # pick out all duplicates in graph g
        mask = batch2 == g
        nodes = mask.nonzero(as_tuple=False).view(-1)  # [D]
        if nodes.numel() <= 1:
            continue

        # 1 big mesh of every pair in this graph
        D = nodes.size(0)
        row = nodes.view(-1, 1).expand(-1, D).reshape(-1)
        col = nodes.view(1, -1).expand(D, -1).reshape(-1)

        # mask out pairs where orig2 is the same
        orig_row = orig2[mask].view(-1, 1).expand(-1, D).reshape(-1)
        orig_col = orig2[mask].view(1, -1).expand(D, -1).reshape(-1)
        keep = orig_row != orig_col

        edges.append(torch.stack([row[keep], col[keep]], dim=0))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=batch.device)
    return torch.cat(edges, dim=1)


def charges_energy_from_graph(
    charges,  # [n_atoms]
    positions,
    edge_index,
    batch,
    density_smearing_width,
):
    """
    Computes the energy of a collection of charges considering only specifed edges.
    normalization of the charges is multipoles.
    """
    sender, receiver = edge_index

    R_ij = positions[receiver] - positions[sender]  # [N_edges,3]
    d_ij = torch.linalg.norm(R_ij, dim=-1)  # [N_edges,1]
    smooth_reciprocal = torch.erf(d_ij * 0.5 / density_smearing_width) / (
        torch.abs(d_ij) + 1e-6
    )

    # charge part
    edge_energy = (
        0.5
        * FIELD_CONSTANT
        * smooth_reciprocal
        * charges[sender]
        * charges[receiver]
        / (4 * pi)
    )
    # handle the case with no edges
    if edge_energy.numel() == 0:
        return torch.zeros(
            (batch.max() + 1,), dtype=charges.dtype, device=charges.device
        )
    node_energies = scatter_sum(src=edge_energy.squeeze(-1), index=receiver, dim=-1)
    return scatter_sum(src=node_energies, index=batch, dim=-1)  # [n_graphs]


class RealSpaceFiniteDiffereneEnergy(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        include_self_interaction: bool = False,
        offset=0.02,
    ):
        if density_max_l > 1:
            raise ValueError(
                "RealSpaceFiniteDiffereneEnergy only supports l=0 and l=1."
            )

        super().__init__()
        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.include_self_interaction = include_self_interaction
        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            density_max_l,
            [density_smearing_width],
            "multipoles",
            "multipoles",
        )

        self.offset = offset
        self.register_buffer(
            "x", torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "y", torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "z", torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )

    def energy_l0(
        self,
        source_feats: torch.Tensor,  # [n_node, 1]
        positions: torch.Tensor,  # [n_node, 3]
        batch: torch.Tensor,  # [n_node]
    ) -> torch.Tensor:

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)

        energy = charges_energy_from_graph(
            source_feats.squeeze(-1),
            positions,
            edge_index,
            batch,
            density_smearing_width=self.density_smearing_width,
        )

        # self interaction
        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)  # [n_node, (l+1)^2]
            node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            self_energy = scatter_sum(src=node_energies, index=batch, dim=-1)
            energy += self_energy * 0.5

        return energy

    def energy_l1(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,  # [n_node, 3]
        batch: torch.Tensor,  # [n_node]
    ) -> torch.Tensor:
        extended_positions = positions.repeat_interleave(7, dim=0)
        extended_positions[1::7] += self.x
        extended_positions[2::7] += self.y
        extended_positions[3::7] += self.z
        extended_positions[4::7] -= self.x
        extended_positions[5::7] -= self.y
        extended_positions[6::7] -= self.z

        extended_batch = batch.repeat_interleave(7)
        charges = torch.zeros_like(extended_positions[:, 0])

        two_offset = 2.0 * self.offset
        charges[0::7] = source_feats[:, 0]
        charges[1::7] = source_feats[:, 3] / two_offset
        charges[2::7] = source_feats[:, 1] / two_offset
        charges[3::7] = source_feats[:, 2] / two_offset
        charges[4::7] = -source_feats[:, 3] / two_offset
        charges[5::7] = -source_feats[:, 1] / two_offset
        charges[6::7] = -source_feats[:, 2] / two_offset

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 7)

        energy = charges_energy_from_graph(
            charges,
            extended_positions,
            edge_index,
            extended_batch,
            density_smearing_width=self.density_smearing_width,
        )

        # self interaction
        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)  # [n_node, (l+1)^2]
            node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            self_energy = scatter_sum(src=node_energies, index=batch, dim=-1)
            energy += self_energy * 0.5

        return energy

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.density_max_l == 0:
            return self.energy_l0(source_feats, positions, batch)
        else:
            return self.energy_l1(source_feats, positions, batch)


def charges_features_from_graph(
    charges,  # [n_atoms]
    positions,
    edge_index,
    batch,
    total_width_factors,  # [1, n_radial]
):
    """
    Computes the features from a collection of charges, on set of scalar features, considering only specified edges.
    normalization of the charges is multipoles.
    """
    num_nodes = positions.shape[0]
    sender, receiver = edge_index
    R_ij = positions[sender] - positions[receiver]  # [N_edges,3]
    d_ij = torch.norm(R_ij, dim=-1, keepdim=True)  # [N_edges,1]
    smooth_reciprocal = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)

    features = scatter_sum(
        charges[sender].unsqueeze(-1) * smooth_reciprocal,
        receiver,
        dim=0,
        dim_size=num_nodes,
    )  # [n_nodes, n_radial]

    features = FIELD_CONSTANT * features / (4 * pi)
    return features


class RealSpaceFiniteDifferenceElectrostaticFeatures(torch.nn.Module):
    """Computes field features for L=0,1 charges and features.
    vector charges and features are represented by displaced scalars."""

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        include_self_interaction=False,
        integral_normalization="receiver",
        offset: float = 0.1,
    ):
        super().__init__()

        self.density_max_l = density_max_l
        self.projection_max_l = projection_max_l
        self.include_self_interaction = include_self_interaction
        self.density_smearing_width = density_smearing_width
        self.projection_smearing_widths = projection_smearing_widths
        self.num_radial = len(projection_smearing_widths)

        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            projection_max_l,
            projection_smearing_widths,
            "multipoles",
            integral_normalization,
        )

        projection_smearing_widths_tensor = torch.tensor(
            projection_smearing_widths, dtype=torch.get_default_dtype()
        )
        total_width_factors = torch.pow(
            (density_smearing_width**2 + projection_smearing_widths_tensor**2) / 2, 0.5
        )
        self.register_buffer("total_width_factors", total_width_factors)

        self.offset = offset
        self.register_buffer(
            "x", torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "y", torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "z", torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )

        l0_factors = [
            get_Cl_sigma(0, sigma, normalize=integral_normalization)
            / get_Cl_sigma(0, sigma, normalize="multipoles")
            for sigma in projection_smearing_widths
        ]
        self.register_buffer(
            "l0_factors", torch.tensor(l0_factors, dtype=torch.get_default_dtype())
        )
        l1_factors = [
            3**0.5
            * sigma**2
            * (
                get_Cl_sigma(1, sigma, normalize=integral_normalization)
                / get_Cl_sigma(0, sigma, normalize="multipoles")
            )
            / self.offset
            for sigma in projection_smearing_widths
        ]
        self.register_buffer(
            "l1_factors", torch.tensor(l1_factors, dtype=torch.get_default_dtype())
        )

    def call_density_0_feats_0(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        edge_long_index = batch_complete_graph_excluding_self_duplicates_vector(
            batch, 1
        )
        feats = charges_features_from_graph(
            charges=source_feats[:, 0],
            positions=positions,
            edge_index=edge_long_index,
            batch=batch,
            total_width_factors=self.total_width_factors.unsqueeze(0),
        )  # [n_atoms, n_radial]
        return self.l0_factors * feats

    def call_density_1_feats_1(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # 7 positions per atom: center(0), +x(1), +y(2), +z(3), -x(4), -y(5), -z(6).
        # Centered finite differences make the probe sets of symmetric atoms exact
        # mirror images of each other, preserving molecular point-group symmetry.
        extended_positions = positions.repeat_interleave(7, dim=0)
        extended_positions[1::7] += self.x
        extended_positions[2::7] += self.y
        extended_positions[3::7] += self.z
        extended_positions[4::7] -= self.x
        extended_positions[5::7] -= self.y
        extended_positions[6::7] -= self.z

        extended_batch = batch.repeat_interleave(7)
        charges = torch.zeros_like(extended_positions[:, 0])

        two_offset = 2.0 * self.offset
        charges[0::7] = source_feats[:, 0]                          # q at center
        charges[1::7] = source_feats[:, 3] / two_offset             # +x: +mu_x/(2δ)
        charges[2::7] = source_feats[:, 1] / two_offset             # +y: +mu_y/(2δ)
        charges[3::7] = source_feats[:, 2] / two_offset             # +z: +mu_z/(2δ)
        charges[4::7] = -source_feats[:, 3] / two_offset            # -x: -mu_x/(2δ)
        charges[5::7] = -source_feats[:, 1] / two_offset            # -y: -mu_y/(2δ)
        charges[6::7] = -source_feats[:, 2] / two_offset            # -z: -mu_z/(2δ)

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 7)

        scalar_features = charges_features_from_graph(
            charges=charges,
            positions=extended_positions,
            edge_index=edge_index,
            batch=extended_batch,
            total_width_factors=self.total_width_factors.unsqueeze(0),
        )  # [7*n_nodes, num_radial]

        all_features = torch.zeros(
            batch.size(0),
            4 * self.num_radial,
            dtype=torch.get_default_dtype(),
            device=batch.device,
        )

        # l=0: potential at center.
        all_features[:, : self.num_radial] = self.l0_factors * scalar_features[0::7]
        # l=1 (SH [y,z,x] order): centered difference (V_+ - V_-) / 2.
        # Leading order is δ·E, same as one-sided, so l1_factors is unchanged.
        all_features[:, self.num_radial :: 3] = self.l1_factors * (
            scalar_features[2::7] - scalar_features[5::7]
        ) / 2
        all_features[:, self.num_radial + 1 :: 3] = self.l1_factors * (
            scalar_features[3::7] - scalar_features[6::7]
        ) / 2
        all_features[:, self.num_radial + 2 :: 3] = self.l1_factors * (
            scalar_features[1::7] - scalar_features[4::7]
        ) / 2

        return all_features

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_nodes, 1, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.density_max_l == 0 and self.projection_max_l == 0:
            features = self.call_density_0_feats_0(
                source_feats.squeeze(-2), node_positions, batch
            )
        elif self.density_max_l == 1 and self.projection_max_l == 0:
            all_feats = self.call_density_1_feats_1(
                source_feats.squeeze(-2), node_positions, batch
            )
            features = all_feats[:, : self.num_radial]
        elif self.density_max_l == 0 and self.projection_max_l == 1:
            padded_source_feats = torch.zeros(
                source_feats.shape[0],
                4,
                dtype=source_feats.dtype,
                device=source_feats.device,
            )
            padded_source_feats[:, 0] = source_feats[:, 0, 0]
            features = self.call_density_1_feats_1(
                padded_source_feats, node_positions, batch
            )
        else:
            features = self.call_density_1_feats_1(
                source_feats.squeeze(-2), node_positions, batch
            )

        self_interaction_terms = self.self_interaction(source_feats.squeeze(-2))
        if self.include_self_interaction:
            features += self_interaction_terms

        return features, self_interaction_terms, None


# ---------------------------------------------------------------------------
# Analytical real-space multipole energy (replaces finite-difference approach)
# ---------------------------------------------------------------------------

def _smeared_coulomb_kernels(
    r: torch.Tensor, sigma: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Kernel functions for GTO-smeared Coulomb potential T(r) = erf(r/(2σ))/r.

    Returns T0, fp, fpp where:
      T0  = erf(r/(2σ)) / r
      fp  = T'(r)  = (g - T0) / r,     g = exp(-r²/(4σ²)) / (σ√π)
      fpp = T''(r) = -g/(2σ²) - 2g/r² + 2T0/r²
    """
    r_safe = r.clamp(min=1e-10)
    g = torch.exp(-r_safe.pow(2) / (4.0 * sigma ** 2)) / (sigma * math.sqrt(math.pi))
    T0 = torch.erf(r_safe / (2.0 * sigma)) / r_safe
    fp = (g - T0) / r_safe
    fpp = -g / (2.0 * sigma ** 2) - 2.0 * g / r_safe.pow(2) + 2.0 * T0 / r_safe.pow(2)
    return T0, fp, fpp


def multipole_energy_from_graph(
    source_feats: torch.Tensor,  # [n_nodes, 4]: [q, μ_y, μ_z, μ_x] (e3nn SH order)
    positions: torch.Tensor,     # [n_nodes, 3]
    edge_index: torch.Tensor,    # [2, n_edges]
    batch: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Analytical energy for charge + dipole densities (l=0,1) with GTO smearing.

    Pair energy between sender i (charge q_s, dipole μ_s) and receiver j (q_r, μ_r):
      E_ij = K/(4π) * [
          q_s*q_r*T0
        + (q_s*(μ_r·R̂) - q_r*(μ_s·R̂)) * T'
        + (μ_s·μ_r) * T'/r
        + (μ_s·R̂)*(μ_r·R̂) * (T'' - T'/r)
      ]
    with R = r_j - r_i, r = |R|, R̂ = R/r.
    """
    sender, receiver = edge_index
    if sender.numel() == 0:
        n_graphs = int(batch.max().item()) + 1
        return torch.zeros(n_graphs, dtype=source_feats.dtype, device=source_feats.device)

    R = positions[receiver] - positions[sender]       # [n_edges, 3]
    r = torch.linalg.norm(R, dim=-1)                  # [n_edges]
    r_safe = r.clamp(min=1e-10)
    Rhat = R / r_safe.unsqueeze(-1)                   # [n_edges, 3]

    T0, fp, fpp = _smeared_coulomb_kernels(r_safe, sigma)
    fp_over_r = fp / r_safe

    q_s = source_feats[sender, 0]
    q_r = source_feats[receiver, 0]
    # e3nn SH l=1 order is [m=-1,0,+1] = [y,z,x]; reorder to Cartesian [x,y,z]
    idx = source_feats.new_tensor([3, 1, 2], dtype=torch.long)
    mu_s = source_feats[sender][:, idx]               # [n_edges, 3]: (μ_x, μ_y, μ_z)
    mu_r = source_feats[receiver][:, idx]

    mu_s_Rhat = (mu_s * Rhat).sum(-1)
    mu_r_Rhat = (mu_r * Rhat).sum(-1)
    mu_dot = (mu_s * mu_r).sum(-1)

    pair_energy = (
        q_s * q_r * T0
        + (q_s * mu_r_Rhat - q_r * mu_s_Rhat) * fp
        - fp_over_r * mu_dot
        + (fp_over_r - fpp) * mu_s_Rhat * mu_r_Rhat
    )

    edge_energy = 0.5 * FIELD_CONSTANT / (4.0 * pi) * pair_energy
    node_energies = scatter_sum(src=edge_energy, index=receiver, dim=0,
                                out=torch.zeros(positions.shape[0],
                                                dtype=edge_energy.dtype,
                                                device=edge_energy.device))
    return scatter_sum(src=node_energies, index=batch, dim=0)


class RealSpaceAnalyticalEnergy(torch.nn.Module):
    """
    Analytical real-space electrostatic energy for l=0,1 GTO charge densities.

    Replaces RealSpaceFiniteDiffereneEnergy: no finite-difference offset, no
    ghost atoms — computes charge/charge, charge/dipole, and dipole/dipole
    interactions directly from the smeared Coulomb kernel and its derivatives.
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        include_self_interaction: bool = False,
    ):
        if density_max_l > 1:
            raise ValueError("RealSpaceAnalyticalEnergy only supports l=0 and l=1.")
        super().__init__()
        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.include_self_interaction = include_self_interaction
        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            density_max_l,
            [density_smearing_width],
            "multipoles",
            "multipoles",
        )

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (l+1)^2] or [n_nodes, 1, (l+1)^2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)

        if self.density_max_l == 0:
            energy = charges_energy_from_graph(
                feats.squeeze(-1),
                positions,
                edge_index,
                batch,
                density_smearing_width=self.density_smearing_width,
            )
        else:
            energy = multipole_energy_from_graph(
                feats,
                positions,
                edge_index,
                batch,
                sigma=self.density_smearing_width,
            )

        if self.include_self_interaction:
            self_fields = self.self_interaction(feats)
            node_energies = torch.einsum("nb,nb->n", feats, self_fields)
            self_energy = scatter_sum(src=node_energies, index=batch, dim=0)
            energy = energy + self_energy * 0.5

        return energy


# ---------------------------------------------------------------------------
# Analytical real-space electrostatic features
# ---------------------------------------------------------------------------

def multipole_features_from_graph(
    source_feats: torch.Tensor,     # [n_nodes, 1] (l=0) or [n_nodes, 4] (l=1)
    positions: torch.Tensor,        # [n_nodes, 3]
    edge_index: torch.Tensor,       # [2, n_edges]
    total_width_factors: torch.Tensor,  # [n_radial]  w_s = sqrt((s_src^2+s_proj_s^2)/2)
    l0_factors: torch.Tensor,       # [n_radial]
    l1_weight: Optional[torch.Tensor],  # [n_radial] or None if projection_max_l==0
    density_max_l: int,
    projection_max_l: int,
) -> torch.Tensor:
    """
    Analytical l=0,1 feature projection for GTO densities.

    For each (sender=j, receiver=i) edge with R = r_i - r_j:

      l=0 from q_j:   +K/4pi * q_j * T_s
      l=0 from mu_j:  -K/4pi * (mu_j . Rhat) * fp_s

      l=1_a from q_j:   -K/4pi * q_j * fp_s * Rhat_a
      l=1_a from mu_j:  -K/4pi * [fp_over_r*mu_j^a - (fp_over_r-fpp)*(mu_j.Rhat)*Rhat_a]

    Output shape: [n_nodes, n_radial] (proj l=0) or [n_nodes, 4*n_radial] (proj l=1).
    """
    num_radial = total_width_factors.shape[0]
    n_nodes = positions.shape[0]
    n_out = num_radial if projection_max_l == 0 else 4 * num_radial

    sender, receiver = edge_index
    if sender.numel() == 0:
        return torch.zeros(n_nodes, n_out, dtype=source_feats.dtype, device=source_feats.device)

    # R points from sender j to receiver i
    R = positions[receiver] - positions[sender]     # [n_edges, 3]
    r = torch.linalg.norm(R, dim=-1)               # [n_edges]
    r_e = r.clamp(min=1e-10).unsqueeze(-1)         # [n_edges, 1]
    Rhat = R / r_e                                  # [n_edges, 3]

    # Smeared Coulomb kernels per radial channel  [n_edges, n_radial]
    w = total_width_factors.unsqueeze(0)            # [1, n_radial]
    g_s = torch.exp(-r_e.pow(2) / (4.0 * w.pow(2))) / (w * math.sqrt(math.pi))
    T_s = torch.erf(r_e / (2.0 * w)) / r_e
    fp_s = (g_s - T_s) / r_e

    q_j = source_feats[sender, 0]                  # [n_edges]

    # l=0 contributions per edge  [n_edges, n_radial]
    contrib_l0 = q_j.unsqueeze(-1) * T_s

    if density_max_l >= 1:
        idx = source_feats.new_tensor([3, 1, 2], dtype=torch.long)  # (x,y,z) from e3nn
        mu_j = source_feats[sender][:, idx]         # [n_edges, 3]
        mu_Rhat = (mu_j * Rhat).sum(-1)             # [n_edges]
        contrib_l0 = contrib_l0 - mu_Rhat.unsqueeze(-1) * fp_s

    feat_l0 = scatter_sum(
        contrib_l0, receiver, dim=0,
        out=torch.zeros(n_nodes, num_radial, dtype=contrib_l0.dtype, device=contrib_l0.device),
    )
    feat_l0 = FIELD_CONSTANT / (4.0 * pi) * l0_factors.unsqueeze(0) * feat_l0

    if projection_max_l == 0:
        return feat_l0

    # l=1 gradient contributions per edge  [n_edges, n_radial, 3]
    fp_over_r = fp_s / r_e
    fpp_s = -g_s / (2.0 * w.pow(2)) - 2.0 * g_s / r_e.pow(2) + 2.0 * T_s / r_e.pow(2)

    # from charge q_j:  dV/dr_i = fp_s * Rhat_a * q_j  → [n_edges, n_radial, 3]
    contrib_l1 = fp_s.unsqueeze(-1) * Rhat.unsqueeze(-2) * q_j.unsqueeze(-1).unsqueeze(-1)

    if density_max_l >= 1:
        # isotropic: -fp_over_r * mu_j^a  → [n_edges, n_radial, 3]
        dip_iso = -fp_over_r.unsqueeze(-1) * mu_j.unsqueeze(-2)
        # anisotropic: (fp_over_r - fpp) * (mu.Rhat) * Rhat_a  → [n_edges, n_radial, 3]
        dip_aniso = (
            ((fp_over_r - fpp_s) * mu_Rhat.unsqueeze(-1)).unsqueeze(-1)
            * Rhat.unsqueeze(-2)
        )
        contrib_l1 = contrib_l1 + dip_iso + dip_aniso

    feat_l1 = scatter_sum(
        contrib_l1.reshape(contrib_l1.shape[0], -1),
        receiver, dim=0,
        out=torch.zeros(n_nodes, num_radial * 3, dtype=contrib_l1.dtype, device=contrib_l1.device),
    ).reshape(n_nodes, num_radial, 3)   # [n_nodes, n_radial, 3]: last dim is Cartesian (x,y,z)

    feat_l1 = FIELD_CONSTANT / (4.0 * pi) * l1_weight.unsqueeze(0).unsqueeze(-1) * feat_l1

    # Assemble output [n_nodes, 4*n_radial]
    # Layout: [:n_radial]=l0, then per radial: (y, z, x) matching FD convention
    out = torch.zeros(n_nodes, n_out, dtype=source_feats.dtype, device=source_feats.device)
    out[:, :num_radial] = feat_l0
    out[:, num_radial::3]     = feat_l1[:, :, 1]   # y (e3nn m=-1)
    out[:, num_radial + 1::3] = feat_l1[:, :, 2]   # z (e3nn m=0)
    out[:, num_radial + 2::3] = feat_l1[:, :, 0]   # x (e3nn m=+1)
    return out


class RealSpaceAnalyticalElectrostaticFeatures(torch.nn.Module):
    """
    Analytical drop-in for RealSpaceFiniteDifferenceElectrostaticFeatures.

    Replaces the 7-ghost-atom FD scheme with direct evaluation of the
    smeared Coulomb potential and its gradient at each receiver site.
    Reduces edges from O(49 N^2) to O(2 N^2) with no offset hyperparameter.
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        include_self_interaction: bool = False,
        integral_normalization: str = "receiver",
    ):
        if density_max_l > 1 or projection_max_l > 1:
            raise ValueError("RealSpaceAnalyticalElectrostaticFeatures supports l<=1 only.")
        super().__init__()
        self.density_max_l = density_max_l
        self.projection_max_l = projection_max_l
        self.include_self_interaction = include_self_interaction
        self.num_radial = len(projection_smearing_widths)

        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            projection_max_l,
            projection_smearing_widths,
            "multipoles",
            integral_normalization,
        )

        w_t = torch.tensor(projection_smearing_widths, dtype=torch.get_default_dtype())
        total_width_factors = ((density_smearing_width**2 + w_t**2) / 2).pow(0.5)
        self.register_buffer("total_width_factors", total_width_factors)

        l0_factors = torch.tensor(
            [get_Cl_sigma(0, s, integral_normalization) / get_Cl_sigma(0, s, "multipoles")
             for s in projection_smearing_widths],
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("l0_factors", l0_factors)

        if projection_max_l >= 1:
            l1_weight = torch.tensor(
                [3**0.5 * s**2 * get_Cl_sigma(1, s, integral_normalization)
                 / get_Cl_sigma(0, s, "multipoles")
                 for s in projection_smearing_widths],
                dtype=torch.get_default_dtype(),
            )
            self.register_buffer("l1_weight", l1_weight)
        else:
            self.register_buffer("l1_weight", None)

    def forward(
        self,
        source_feats: torch.Tensor,   # [n_nodes, 1, lm_dim] or [n_nodes, lm_dim]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        # For l=0 density with l=1 projection, pad to 4 components
        if self.density_max_l == 0 and self.projection_max_l == 1 and feats.shape[-1] == 1:
            padded = torch.zeros(
                feats.shape[0], 4, dtype=feats.dtype, device=feats.device
            )
            padded[:, 0] = feats[:, 0]
            feats = padded

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)
        features = multipole_features_from_graph(
            source_feats=feats,
            positions=node_positions,
            edge_index=edge_index,
            total_width_factors=self.total_width_factors,
            l0_factors=self.l0_factors,
            l1_weight=self.l1_weight,
            density_max_l=self.density_max_l,
            projection_max_l=self.projection_max_l,
        )

        si_terms = self.self_interaction(feats if self.density_max_l >= 1 else source_feats.squeeze(-2))
        if self.include_self_interaction:
            features = features + si_terms

        return features, si_terms, None

