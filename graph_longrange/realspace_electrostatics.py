import math
import torch
from scipy.constants import e, epsilon_0, pi
from .utils import FIELD_CONSTANT, scatter_sum
from typing import List, Optional, Tuple
import warnings


def _load_gto_utils():
    from .gto_utils import GTOSelfInteractionBlock, get_Cl_sigma

    return GTOSelfInteractionBlock, get_Cl_sigma

@torch.no_grad()
def batch_complete_graph_excluding_self_duplicates_vector(
    batch: torch.Tensor, N: int, n_graphs: int | None = None
) -> torch.Tensor:
    """
    Duplicate each node N times, then for each graph build directed
    edges between every pair of duplicates *unless* they share the same
    original node ID.

    Fully vectorized — no Python loop over graphs.

    Args:
        batch (LongTensor): shape [M], graph ID of each original node.
        N (int): number of duplicates per node.
        n_graphs: number of graphs.  When provided as a plain Python int
            (e.g. from ``data.num_graphs``), avoids a ``batch.max().item()``
            call that would create an unbacked symint inside compiled code.

    Returns:
        edge_index (LongTensor[2, E])
    """
    batch = batch.long()
    M = batch.shape[0]
    device = batch.device

    if M == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    orig = torch.arange(M, device=device)
    batch2 = batch.repeat_interleave(N)   # [M*N]
    orig2 = orig.repeat_interleave(N)     # [M*N]
    M2 = batch2.shape[0]

    G = n_graphs if n_graphs is not None else int(batch2.max().item()) + 1

    # Count duplicated nodes per graph, build offsets into the sorted array
    counts = torch.zeros(G, dtype=torch.long, device=device)
    counts.scatter_add_(0, batch2, torch.ones(M2, dtype=torch.long, device=device))
    offsets = torch.zeros(G + 1, dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)

    # Sort duplicated nodes by graph for contiguous per-graph blocks
    sort_perm = torch.argsort(batch2, stable=True)   # [M2] → sorted global indices
    sorted_graph = batch2[sort_perm]                  # [M2] graph id in sorted order
    sorted_orig2 = orig2[sort_perm]                   # [M2] original node id in sorted order

    # Each sorted node i in graph g contributes (counts[g] - 1) edges
    edge_count_per_node = counts[sorted_graph] - 1    # [M2]
    total_edges = int(edge_count_per_node.sum().item())

    if total_edges == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # Row indices (in sorted space): repeat each node index by its edge count
    row_sorted = torch.arange(M2, device=device).repeat_interleave(edge_count_per_node)

    # Local position of each sorted node within its graph: i - offsets[g]
    local_row_pos = torch.arange(M2, device=device) - offsets[sorted_graph]

    # Sequential column index within each row node's block: 0, 1, ..., count_g-2
    edge_start = edge_count_per_node.cumsum(0) - edge_count_per_node  # exclusive prefix sum
    local_col_base = torch.arange(total_edges, device=device) - edge_start[row_sorted]

    # Skip the row node itself: shift by 1 wherever local_col_base >= row's local pos
    local_col = local_col_base + (local_col_base >= local_row_pos[row_sorted]).long()

    # Map (graph, local_col) → sorted-array position → global index
    col_sorted_pos = offsets[sorted_graph[row_sorted]] + local_col
    row_global = sort_perm[row_sorted]
    col_global = sort_perm[col_sorted_pos]

    # For N > 1: also exclude edges between different duplicates of the same original node
    if N > 1:
        keep = sorted_orig2[row_sorted] != sorted_orig2[col_sorted_pos]
        row_global = row_global[keep]
        col_global = col_global[keep]

    return torch.stack([row_global, col_global], dim=0)


def charges_energy_from_graph(
    charges,  # [n_atoms]
    positions,
    edge_index,
    batch,
    density_smearing_width,
    n_graphs: int | None = None,
):
    """
    Computes the energy of a collection of charges considering only specifed edges.
    normalization of the charges is multipoles.
    """
    sender, receiver = edge_index

    if n_graphs is None:
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

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
            (n_graphs,), dtype=charges.dtype, device=charges.device
        )
    node_energies = scatter_sum(src=edge_energy.squeeze(-1), index=receiver, dim=-1,
                                out=torch.zeros(positions.shape[0], dtype=edge_energy.dtype, device=edge_energy.device))
    return scatter_sum(src=node_energies, index=batch, dim=0, dim_size=n_graphs)


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
        GTOSelfInteractionBlock, _ = _load_gto_utils()
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
        GTOSelfInteractionBlock, get_Cl_sigma = _load_gto_utils()

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


def _smeared_coulomb_third_derivative(
    r: torch.Tensor,
    sigma: torch.Tensor,
    g: torch.Tensor,
    T0: torch.Tensor,
) -> torch.Tensor:
    """Third radial derivative of T(r) = erf(r/(2 sigma)) / r."""
    return (
        r * g / (4.0 * sigma.pow(4))
        + g / (sigma.pow(2) * r)
        + 6.0 * g / r.pow(3)
        - 6.0 * T0 / r.pow(3)
    )


def _smeared_coulomb_fourth_derivative(
    r: torch.Tensor,
    sigma: torch.Tensor,
    g: torch.Tensor,
    T0: torch.Tensor,
) -> torch.Tensor:
    """Fourth radial derivative of T(r) = erf(r/(2 sigma)) / r."""
    return (
        -r.pow(2) * g / (8.0 * sigma.pow(6))
        - g / (4.0 * sigma.pow(4))
        - 4.0 * g / (sigma.pow(2) * r.pow(2))
        - 24.0 * g / r.pow(4)
        + 24.0 * T0 / r.pow(4)
    )


def _l2_source_to_cartesian(source_feats: torch.Tensor) -> torch.Tensor:
    """Convert graph_longrange l=2 real-SH coefficients to Cartesian traceless tensors.

    The input ordering is the same real-harmonic order used by graph_longrange / e3nn:
    [sqrt(3)xy, sqrt(3)yz, (3z^2-r^2)/2, sqrt(3)xz, sqrt(3)(x^2-y^2)/2].
    """
    q_m2, q_m1, q_0, q_p1, q_p2 = source_feats.unbind(dim=-1)
    s3_over_2 = math.sqrt(3.0) / 2.0

    t_xx = -0.5 * q_0 + s3_over_2 * q_p2
    t_yy = -0.5 * q_0 - s3_over_2 * q_p2
    t_zz = q_0
    t_xy = s3_over_2 * q_m2
    t_yz = s3_over_2 * q_m1
    t_xz = s3_over_2 * q_p1

    return torch.stack(
        [
            torch.stack([t_xx, t_xy, t_xz], dim=-1),
            torch.stack([t_xy, t_yy, t_yz], dim=-1),
            torch.stack([t_xz, t_yz, t_zz], dim=-1),
        ],
        dim=-2,
    )


def multipole_energy_from_graph(
    source_feats: torch.Tensor,  # [n_nodes, 1], [n_nodes, 4], or [n_nodes, 9]
    positions: torch.Tensor,     # [n_nodes, 3]
    edge_index: torch.Tensor,    # [2, n_edges]
    batch: torch.Tensor,
    sigma: float,
    n_graphs: int | None = None,
) -> torch.Tensor:
    """
    Analytical energy for charge, dipole, and quadrupole densities with GTO smearing.

    Pair energy between sender i and receiver j with source multipoles
    (q_s, μ_s, Q_s) and receiver multipoles (q_r, μ_r, Q_r):
      E_ij = K/(4π) * [
          q_s*q_r*T0
        + (q_s*(μ_r·R̂) - q_r*(μ_s·R̂)) * T'
        + (μ_s·μ_r) * T'/r
        + (μ_s·R̂)*(μ_r·R̂) * (T'' - T'/r)
        + charge-quadrupole terms
        + dipole-quadrupole terms
        + quadrupole-quadrupole terms
      ]
    with R = r_j - r_i, r = |R|, R̂ = R/r.
    """
    sender, receiver = edge_index
    if n_graphs is None:
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    if sender.numel() == 0:
        return torch.zeros(n_graphs, dtype=source_feats.dtype, device=source_feats.device)

    R = positions[receiver] - positions[sender]       # [n_edges, 3]
    r = torch.linalg.norm(R, dim=-1)                  # [n_edges]
    r_safe = r.clamp(min=1e-10)
    Rhat = R / r_safe.unsqueeze(-1)                   # [n_edges, 3]

    T0, fp, fpp = _smeared_coulomb_kernels(r_safe, sigma)
    fp_over_r = fp / r_safe
    g = torch.exp(-r_safe.pow(2) / (4.0 * sigma ** 2)) / (sigma * math.sqrt(math.pi))
    fppp = _smeared_coulomb_third_derivative(r_safe, torch.as_tensor(sigma, dtype=r_safe.dtype, device=r_safe.device), g, T0)
    f4 = _smeared_coulomb_fourth_derivative(r_safe, torch.as_tensor(sigma, dtype=r_safe.dtype, device=r_safe.device), g, T0)

    q_s = source_feats[sender, 0]
    q_r = source_feats[receiver, 0]
    pair_energy = q_s * q_r * T0

    if source_feats.shape[-1] >= 4:
        # e3nn SH l=1 order is [m=-1,0,+1] = [y,z,x]; reorder to Cartesian [x,y,z]
        idx = source_feats.new_tensor([3, 1, 2], dtype=torch.long)
        mu_s = source_feats[sender][:, idx]
        mu_r = source_feats[receiver][:, idx]

        mu_s_Rhat = (mu_s * Rhat).sum(-1)
        mu_r_Rhat = (mu_r * Rhat).sum(-1)
        mu_dot = (mu_s * mu_r).sum(-1)

        pair_energy = pair_energy + (
            (q_s * mu_r_Rhat - q_r * mu_s_Rhat) * fp
            - fp_over_r * mu_dot
            + (fp_over_r - fpp) * mu_s_Rhat * mu_r_Rhat
        )
    else:
        mu_s = mu_r = mu_s_Rhat = mu_r_Rhat = None

    if source_feats.shape[-1] >= 9:
        quad_s = _l2_source_to_cartesian(source_feats[sender][:, 4:9])
        quad_r = _l2_source_to_cartesian(source_feats[receiver][:, 4:9])

        quad_s_Rhat = torch.einsum("eab,eb->ea", quad_s, Rhat)
        quad_r_Rhat = torch.einsum("eab,eb->ea", quad_r, Rhat)
        quad_s_nn = torch.einsum("ea,ea->e", quad_s_Rhat, Rhat)
        quad_r_nn = torch.einsum("ea,ea->e", quad_r_Rhat, Rhat)

        hess_aniso = fpp - fp_over_r
        third_radial = fppp - 3.0 * fpp / r_safe + 3.0 * fp / r_safe.pow(2)
        third_mixed = fpp / r_safe - fp / r_safe.pow(2)
        fourth_radial = (
            f4
            - 6.0 * fppp / r_safe
            + 15.0 * fpp / r_safe.pow(2)
            - 15.0 * fp / r_safe.pow(3)
        )

        pair_energy = pair_energy + hess_aniso * (
            q_r * quad_s_nn + q_s * quad_r_nn
        )

        if mu_s is not None and mu_r is not None:
            pair_energy = pair_energy + third_radial * (
                mu_r_Rhat * quad_s_nn - mu_s_Rhat * quad_r_nn
            ) + 2.0 * third_mixed * (
                torch.einsum("ea,ea->e", mu_r, quad_s_Rhat)
                - torch.einsum("ea,ea->e", mu_s, quad_r_Rhat)
            )

        pair_energy = pair_energy + fourth_radial * quad_s_nn * quad_r_nn
        pair_energy = pair_energy + 4.0 * third_radial / r_safe * torch.einsum(
            "ea,eab,eb->e",
            Rhat,
            torch.matmul(quad_r, quad_s),
            Rhat,
        )
        pair_energy = pair_energy + 2.0 * hess_aniso / r_safe.pow(2) * torch.einsum(
            "eab,eab->e",
            quad_r,
            quad_s,
        )

    edge_energy = 0.5 * FIELD_CONSTANT / (4.0 * pi) * pair_energy
    node_energies = scatter_sum(src=edge_energy, index=receiver, dim=0,
                                out=torch.zeros(positions.shape[0],
                                                dtype=edge_energy.dtype,
                                                device=edge_energy.device))
    return scatter_sum(src=node_energies, index=batch, dim=0, dim_size=n_graphs)


class RealSpaceAnalyticalEnergy(torch.nn.Module):
    """
    Analytical real-space electrostatic energy for l=0,1,2 GTO charge densities.

    Replaces RealSpaceFiniteDiffereneEnergy: no finite-difference offset, no
    ghost atoms — computes charge, dipole, and quadrupole interactions
    directly from the smeared Coulomb kernel and its derivatives.
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        include_self_interaction: bool = False,
    ):
        if density_max_l > 2:
            raise ValueError("RealSpaceAnalyticalEnergy only supports l=0, l=1, and l=2.")
        super().__init__()
        GTOSelfInteractionBlock, _ = _load_gto_utils()
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
        n_graphs: int | None = None,
    ) -> torch.Tensor:
        feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        if n_graphs is None:
            n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1, n_graphs=n_graphs)

        if self.density_max_l == 0:
            energy = charges_energy_from_graph(
                feats.squeeze(-1),
                positions,
                edge_index,
                batch,
                density_smearing_width=self.density_smearing_width,
                n_graphs=n_graphs,
            )
        else:
            energy = multipole_energy_from_graph(
                feats,
                positions,
                edge_index,
                batch,
                sigma=self.density_smearing_width,
                n_graphs=n_graphs,
            )

        if self.include_self_interaction:
            self_fields = self.self_interaction(feats)
            node_energies = torch.einsum("nb,nb->n", feats, self_fields)
            self_energy = scatter_sum(src=node_energies, index=batch, dim=0, dim_size=n_graphs)
            energy = energy + self_energy * 0.5

        return energy


# ---------------------------------------------------------------------------
# Analytical real-space electrostatic features
# ---------------------------------------------------------------------------

def multipole_features_from_graph(
    source_feats: torch.Tensor,     # [n_nodes, 1] (l=0), [n_nodes, 4] (l=1), or [n_nodes, 9] (l=2 source)
    positions: torch.Tensor,        # [n_nodes, 3]
    edge_index: torch.Tensor,       # [2, n_edges]
    total_width_factors: torch.Tensor,  # [n_radial]  w_s = sqrt((s_src^2+s_proj_s^2)/2)
    l0_factors: torch.Tensor,       # [n_radial]
    l1_weight: Optional[torch.Tensor],  # [n_radial] or None if projection_max_l==0
    density_max_l: int,
    projection_max_l: int,
    l2_weight: Optional[torch.Tensor] = None,  # [n_radial] or None if projection_max_l<2
) -> torch.Tensor:
    """
    Analytical l=0,1,2 feature projection for GTO densities.

    For each (sender=j, receiver=i) edge with R = r_i - r_j:

      l=0 from q_j:   +K/4pi * q_j * T_s
      l=0 from mu_j:  -K/4pi * (mu_j . Rhat) * fp_s
      l=0 from Q_j:   +K/4pi * (Q_j : RhatRhat) * (fpp - fp/r)

      l=1_a from q_j: fp_s * Rhat_a * q_j
      l=1_a from mu_j: -fp_over_r * mu_j^a + (fp_over_r-fpp)*(mu.Rhat)*Rhat_a
      l=1_a from Q_j:  2*(fpp/r-fp/r^2)*(Q_j Rhat)_a + (f'''-3f''/r+3f'/r^2)*(Q_j:RR)*Rhat_a

      l=2_m from q_j: hess_aniso * rsh_l2(Rhat) * q_j
      l=2_m from mu_j: K1 * mu_Rhat * rsh_l2 + K2 * sym_l2(mu_j, Rhat)
      l=2_m from Q_j: Λ4/2 * quad_nn * rsh_l2 + Λm * sym_l2(quad_Rhat, Rhat) + Λi * SH(Q_j)

    Output shape: [n_nodes, n_radial] (proj l=0), [n_nodes, 4*n_radial] (proj l=1),
    or [n_nodes, 9*n_radial] (proj l=2).
    """
    num_radial = total_width_factors.shape[0]
    n_nodes = positions.shape[0]
    if projection_max_l == 0:
        n_out = num_radial
    elif projection_max_l == 1:
        n_out = 4 * num_radial
    else:
        n_out = 9 * num_radial

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
    fp_over_r = fp_s / r_e
    fpp_s = -g_s / (2.0 * w.pow(2)) - 2.0 * g_s / r_e.pow(2) + 2.0 * T_s / r_e.pow(2)
    fppp_s = _smeared_coulomb_third_derivative(r_e, w, g_s, T_s)

    q_j = source_feats[sender, 0]                  # [n_edges]

    # l=0 contributions per edge  [n_edges, n_radial]
    contrib_l0 = q_j.unsqueeze(-1) * T_s

    if density_max_l >= 1:
        idx = source_feats.new_tensor([3, 1, 2], dtype=torch.long)  # (x,y,z) from e3nn
        mu_j = source_feats[sender][:, idx]         # [n_edges, 3]
        mu_Rhat = (mu_j * Rhat).sum(-1)             # [n_edges]
        contrib_l0 = contrib_l0 - mu_Rhat.unsqueeze(-1) * fp_s

    if density_max_l >= 2:
        quad_j = _l2_source_to_cartesian(source_feats[sender][:, 4:9])   # [n_edges, 3, 3]
        quad_Rhat = torch.einsum("eab,eb->ea", quad_j, Rhat)              # [n_edges, 3]
        quad_nn = torch.einsum("ea,ea->e", quad_Rhat, Rhat)               # [n_edges]
        hess_aniso = fpp_s - fp_over_r
        contrib_l0 = contrib_l0 + quad_nn.unsqueeze(-1) * hess_aniso

    feat_l0 = scatter_sum(
        contrib_l0, receiver, dim=0,
        out=torch.zeros(n_nodes, num_radial, dtype=contrib_l0.dtype, device=contrib_l0.device),
    )
    feat_l0 = FIELD_CONSTANT / (4.0 * pi) * l0_factors.unsqueeze(0) * feat_l0

    if projection_max_l == 0:
        return feat_l0

    # l=1 gradient contributions per edge  [n_edges, n_radial, 3]

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

    if density_max_l >= 2:
        quad_mix = 2.0 * (fpp_s / r_e - fp_s / r_e.pow(2))               # [n_edges, n_radial]
        quad_radial = fppp_s - 3.0 * fpp_s / r_e + 3.0 * fp_s / r_e.pow(2)
        quad_vec = (
            quad_mix.unsqueeze(-1) * quad_Rhat.unsqueeze(-2)
            + (quad_radial * quad_nn.unsqueeze(-1)).unsqueeze(-1) * Rhat.unsqueeze(-2)
        )
        contrib_l1 = contrib_l1 + quad_vec

    feat_l1 = scatter_sum(
        contrib_l1.reshape(contrib_l1.shape[0], -1),
        receiver, dim=0,
        out=torch.zeros(n_nodes, num_radial * 3, dtype=contrib_l1.dtype, device=contrib_l1.device),
    ).reshape(n_nodes, num_radial, 3)   # [n_nodes, n_radial, 3]: last dim is Cartesian (x,y,z)

    feat_l1 = FIELD_CONSTANT / (4.0 * pi) * l1_weight.unsqueeze(0).unsqueeze(-1) * feat_l1

    if projection_max_l == 1:
        # Assemble output [n_nodes, 4*n_radial]
        # Layout: [:n_radial]=l0, then per radial: (y, z, x) matching FD/e3nn convention
        out = torch.zeros(n_nodes, n_out, dtype=source_feats.dtype, device=source_feats.device)
        out[:, :num_radial] = feat_l0
        out[:, num_radial::3]     = feat_l1[:, :, 1]   # y (e3nn m=-1)
        out[:, num_radial + 1::3] = feat_l1[:, :, 2]   # z (e3nn m=0)
        out[:, num_radial + 2::3] = feat_l1[:, :, 0]   # x (e3nn m=+1)
        return out

    # -------------------------------------------------------------------------
    # l=2 EFG contributions  (projection_max_l == 2)
    # -------------------------------------------------------------------------
    # Geometric factors: rsh_l2 follows the same e3nn convention as
    # _l2_source_to_cartesian, i.e. components [m=-2,-1,0,+1,+2] map to
    #   m=-2: (2/√3)*rx*ry,  m=-1: (2/√3)*ry*rz,  m=0: rz²-1/3,
    #   m=+1: (2/√3)*rx*rz,  m=+2: (rx²-ry²)/√3
    INV_SQRT3 = 1.0 / math.sqrt(3.0)
    TWO_OVER_SQRT3 = 2.0 * INV_SQRT3
    rx, ry, rz = Rhat[:, 0], Rhat[:, 1], Rhat[:, 2]

    rsh_l2 = torch.stack([
        TWO_OVER_SQRT3 * rx * ry,
        TWO_OVER_SQRT3 * ry * rz,
        rz * rz - 1.0 / 3.0,
        TWO_OVER_SQRT3 * rx * rz,
        (rx * rx - ry * ry) * INV_SQRT3,
    ], dim=-1)  # [n_edges, 5]

    # Charge → l=2: hess_aniso * q_j * rsh_l2(R̂)
    if density_max_l >= 2:
        hess_aniso_l2 = hess_aniso  # already computed above
    else:
        hess_aniso_l2 = fpp_s - fp_over_r

    contrib_l2 = (
        hess_aniso_l2 * q_j.unsqueeze(-1)
    ).unsqueeze(-1) * rsh_l2.unsqueeze(-2)
    # [n_edges, n_radial, 5]

    if density_max_l >= 1:
        # K1 = fppp - 3*fpp/r + 3*fp/r²,  K2 = fpp/r - fp/r²
        K1 = fppp_s - 3.0 * fpp_s / r_e + 3.0 * fp_s / r_e.pow(2)   # [n_edges, n_radial]
        K2 = fpp_s / r_e - fp_s / r_e.pow(2)                           # [n_edges, n_radial]

        # Dipole → l=2, K1 term: K1 * (mu·R̂) * rsh_l2
        dip_K1 = (K1 * mu_Rhat.unsqueeze(-1)).unsqueeze(-1) * rsh_l2.unsqueeze(-2)

        # Dipole → l=2, K2 term: K2 * sym_l2(mu_j, R̂)
        mx, my, mz = mu_j[:, 0], mu_j[:, 1], mu_j[:, 2]
        sym_mu = torch.stack([
            TWO_OVER_SQRT3 * (mx * ry + rx * my),
            TWO_OVER_SQRT3 * (my * rz + ry * mz),
            2.0 * mz * rz - (2.0 / 3.0) * mu_Rhat,
            TWO_OVER_SQRT3 * (mx * rz + rx * mz),
            2.0 * (mx * rx - my * ry) * INV_SQRT3,
        ], dim=-1)  # [n_edges, 5]

        dip_K2 = K2.unsqueeze(-1) * sym_mu.unsqueeze(-2)
        contrib_l2 = contrib_l2 + dip_K1 + dip_K2

    if density_max_l >= 2:
        # Fourth derivative for quadrupole → l=2
        f4_s = _smeared_coulomb_fourth_derivative(r_e, w, g_s, T_s)   # [n_edges, n_radial]
        fourth_radial = (
            f4_s
            - 6.0 * fppp_s / r_e
            + 15.0 * fpp_s / r_e.pow(2)
            - 15.0 * fp_s / r_e.pow(3)
        )   # Λ_4/2 coefficient

        # K1 already computed in density_max_l>=1 block
        lam_m = 2.0 * K1 / r_e           # [n_edges, n_radial]
        lam_i = 2.0 * hess_aniso_l2 / r_e.pow(2)  # [n_edges, n_radial]

        # Λ_4/2 * quad_nn * rsh_l2
        quad_l2_radial = (fourth_radial * quad_nn.unsqueeze(-1)).unsqueeze(-1) * rsh_l2.unsqueeze(-2)

        # Λ_m * sym_l2(quad_Rhat, R̂)
        qx, qy, qz = quad_Rhat[:, 0], quad_Rhat[:, 1], quad_Rhat[:, 2]
        sym_quad = torch.stack([
            TWO_OVER_SQRT3 * (qx * ry + rx * qy),
            TWO_OVER_SQRT3 * (qy * rz + ry * qz),
            2.0 * qz * rz - (2.0 / 3.0) * quad_nn,
            TWO_OVER_SQRT3 * (qx * rz + rx * qz),
            2.0 * (qx * rx - qy * ry) * INV_SQRT3,
        ], dim=-1)  # [n_edges, 5]
        quad_l2_mixed = lam_m.unsqueeze(-1) * sym_quad.unsqueeze(-2)

        # Λ_i * Q_j  (Q already in SH form at positions 4:9 in source_feats)
        q_sh = source_feats[sender, 4:9]  # [n_edges, 5]
        quad_l2_iso = lam_i.unsqueeze(-1) * q_sh.unsqueeze(-2)

        contrib_l2 = contrib_l2 + quad_l2_radial + quad_l2_mixed + quad_l2_iso

    feat_l2 = scatter_sum(
        contrib_l2.reshape(contrib_l2.shape[0], -1),
        receiver, dim=0,
        out=torch.zeros(n_nodes, num_radial * 5, dtype=contrib_l2.dtype, device=contrib_l2.device),
    ).reshape(n_nodes, num_radial, 5)   # [n_nodes, n_radial, 5]

    feat_l2 = FIELD_CONSTANT / (4.0 * pi) * l2_weight.unsqueeze(0).unsqueeze(-1) * feat_l2

    # Assemble output [n_nodes, 9*n_radial]
    # Layout: [:n_radial]=l0, [n_radial:4*n_radial]=l1, [4*n_radial:9*n_radial]=l2
    # l=2 in e3nn order [m=-2,-1,0,+1,+2]
    l1_end = 4 * num_radial
    l2_end = 9 * num_radial
    out = torch.zeros(n_nodes, n_out, dtype=source_feats.dtype, device=source_feats.device)
    out[:, :num_radial] = feat_l0
    out[:, num_radial:l1_end:3]     = feat_l1[:, :, 1]   # y (e3nn m=-1)
    out[:, num_radial + 1:l1_end:3] = feat_l1[:, :, 2]   # z (e3nn m=0)
    out[:, num_radial + 2:l1_end:3] = feat_l1[:, :, 0]   # x (e3nn m=+1)
    out[:, l1_end:l2_end:5]     = feat_l2[:, :, 0]   # m=-2
    out[:, l1_end + 1:l2_end:5] = feat_l2[:, :, 1]   # m=-1
    out[:, l1_end + 2:l2_end:5] = feat_l2[:, :, 2]   # m=0
    out[:, l1_end + 3:l2_end:5] = feat_l2[:, :, 3]   # m=+1
    out[:, l1_end + 4:l2_end:5] = feat_l2[:, :, 4]   # m=+2
    return out


class RealSpaceAnalyticalElectrostaticFeatures(torch.nn.Module):
    """
    Analytical drop-in for RealSpaceFiniteDifferenceElectrostaticFeatures.

    Replaces the 7-ghost-atom FD scheme with direct evaluation of the
    smeared Coulomb potential and its gradient at each receiver site.
    Reduces edges from O(49 N^2) to O(2 N^2) with no offset hyperparameter.

    Supports source multipoles up to l=2 (charges, dipoles, quadrupoles) and
    receiver projections up to l=1 (potential + electric-field-like features).
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
        if density_max_l > 2 or projection_max_l > 2:
            raise ValueError(
                "RealSpaceAnalyticalElectrostaticFeatures supports density_max_l<=2 and projection_max_l<=2."
            )
        super().__init__()
        GTOSelfInteractionBlock, get_Cl_sigma = _load_gto_utils()
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

        if projection_max_l >= 2:
            l2_weight = torch.tensor(
                [5**0.5 * s**4 * get_Cl_sigma(2, s, integral_normalization)
                 / get_Cl_sigma(0, s, "multipoles")
                 for s in projection_smearing_widths],
                dtype=torch.get_default_dtype(),
            )
            self.register_buffer("l2_weight", l2_weight)
        else:
            self.register_buffer("l2_weight", None)

    def forward(
        self,
        source_feats: torch.Tensor,   # [n_nodes, 1, lm_dim] or [n_nodes, lm_dim]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int | None = None,
    ) -> torch.Tensor:
        feats = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        # For l=0 density with l=1 projection, pad to 4 components
        if self.density_max_l == 0 and self.projection_max_l == 1 and feats.shape[-1] == 1:
            padded = torch.zeros(
                feats.shape[0], 4, dtype=feats.dtype, device=feats.device
            )
            padded[:, 0] = feats[:, 0]
            feats = padded

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1, n_graphs=n_graphs)
        features = multipole_features_from_graph(
            source_feats=feats,
            positions=node_positions,
            edge_index=edge_index,
            total_width_factors=self.total_width_factors,
            l0_factors=self.l0_factors,
            l1_weight=self.l1_weight,
            density_max_l=self.density_max_l,
            projection_max_l=self.projection_max_l,
            l2_weight=self.l2_weight,
        )

        si_terms = self.self_interaction(feats if self.density_max_l >= 1 else source_feats.squeeze(-2))
        if self.include_self_interaction:
            features = features + si_terms

        return features, si_terms, None
