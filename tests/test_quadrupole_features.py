import unittest

import torch

from graph_longrange.realspace_electrostatics import (
    RealSpaceAnalyticalElectrostaticFeatures,
    multipole_features_from_graph,
)
import math


torch.set_default_dtype(torch.float64)


def _quadrupole_source_features() -> torch.Tensor:
    # Real-SH l=2 order is [xy, yz, z2, xz, x2-y2]. Setting only m=0 gives an
    # axis-aligned traceless quadrupole diag(-1/2, -1/2, 1).
    source_feats = torch.zeros((2, 9), dtype=torch.get_default_dtype())
    source_feats[0, 6] = 1.0
    return source_feats


def _pair_edge_index() -> torch.Tensor:
    return torch.tensor([[0], [1]], dtype=torch.long)


def _evaluate_pair_features(
    receiver_position: torch.Tensor,
    projection_max_l: int,
) -> torch.Tensor:
    positions = torch.zeros((2, 3), dtype=torch.get_default_dtype())
    positions[1] = receiver_position

    num_radial = 1
    total_width_factors = torch.full((num_radial,), 0.9, dtype=positions.dtype)
    l0_factors = torch.ones(num_radial, dtype=positions.dtype)
    l1_weight = (
        torch.ones(num_radial, dtype=positions.dtype)
        if projection_max_l >= 1
        else None
    )

    return multipole_features_from_graph(
        source_feats=_quadrupole_source_features(),
        positions=positions,
        edge_index=_pair_edge_index(),
        total_width_factors=total_width_factors,
        l0_factors=l0_factors,
        l1_weight=l1_weight,
        density_max_l=2,
        projection_max_l=projection_max_l,
    )[1]


def _finite_difference_gradient(
    receiver_position: torch.Tensor,
    step: float = 1e-5,
) -> torch.Tensor:
    grad = []
    for axis in range(3):
        pos_plus = receiver_position.clone()
        pos_minus = receiver_position.clone()
        pos_plus[axis] += step
        pos_minus[axis] -= step
        v_plus = _evaluate_pair_features(pos_plus, projection_max_l=0)[0]
        v_minus = _evaluate_pair_features(pos_minus, projection_max_l=0)[0]
        grad.append((v_plus - v_minus) / (2.0 * step))
    return torch.stack(grad)


def test_quadrupole_field_matches_finite_difference_of_potential():
    receiver_position = torch.tensor([1.2, -0.7, 0.9], dtype=torch.get_default_dtype())

    features = _evaluate_pair_features(receiver_position, projection_max_l=1)
    field_xyz = torch.stack([features[3], features[1], features[2]])
    grad_xyz = _finite_difference_gradient(receiver_position)

    torch.testing.assert_close(field_xyz, grad_xyz, rtol=2e-5, atol=5e-7)


def test_axis_aligned_quadrupole_has_no_transverse_field_on_axis():
    receiver_position = torch.tensor([0.0, 0.0, 1.8], dtype=torch.get_default_dtype())

    features = _evaluate_pair_features(receiver_position, projection_max_l=1)
    field_xyz = torch.stack([features[3], features[1], features[2]])

    assert features[0].abs() > 1e-8
    assert field_xyz[2].abs() > 1e-8
    assert field_xyz[0].abs() < 1e-12
    assert field_xyz[1].abs() < 1e-12


def _evaluate_pair_efg(
    receiver_position: torch.Tensor,
) -> torch.Tensor:
    """l=2 EFG features with unit weights at the receiver."""
    positions = torch.zeros((2, 3), dtype=torch.get_default_dtype())
    positions[1] = receiver_position

    num_radial = 1
    total_width_factors = torch.full((num_radial,), 0.9, dtype=positions.dtype)
    l0_factors = torch.ones(num_radial, dtype=positions.dtype)
    l1_weight = torch.ones(num_radial, dtype=positions.dtype)
    l2_weight = torch.ones(num_radial, dtype=positions.dtype)

    return multipole_features_from_graph(
        source_feats=_quadrupole_source_features(),
        positions=positions,
        edge_index=_pair_edge_index(),
        total_width_factors=total_width_factors,
        l0_factors=l0_factors,
        l1_weight=l1_weight,
        density_max_l=2,
        projection_max_l=2,
        l2_weight=l2_weight,
    )[1]


def _finite_difference_efg(
    receiver_position: torch.Tensor,
    step: float = 1e-4,
) -> torch.Tensor:
    """Symmetric FD gradient of field features [y, z, x] → 3×3 EFG in e3nn SH order."""
    # e3nn l=1 field indices: m=-1→y (idx 1), m=0→z (idx 2), m=+1→x (idx 3)
    field_indices = [3, 1, 2]  # Cartesian x, y, z from features
    # e3nn l=2 SH order: m=-2→(2/√3)xy, m=-1→(2/√3)yz, m=0→z²-1/3, m=+1→(2/√3)xz, m=+2→(x²-y²)/√3
    # Gradient of field[a] w.r.t. axis b gives EFG[b,a]
    s3 = math.sqrt(3.0)

    # We need ∂E_x/∂x, ∂E_y/∂y, ∂E_z/∂z, and cross terms
    grad_field = torch.zeros(3, 3, dtype=receiver_position.dtype)
    for b in range(3):
        for a, feat_idx in enumerate(field_indices):
            pos_p = receiver_position.clone(); pos_p[b] += step
            pos_m = receiver_position.clone(); pos_m[b] -= step
            fp = _evaluate_pair_features(pos_p, projection_max_l=1)[feat_idx]
            fm = _evaluate_pair_features(pos_m, projection_max_l=1)[feat_idx]
            grad_field[b, a] = (fp - fm) / (2.0 * step)  # ∂E_a/∂r_b

    # Convert symmetric traceless 3×3 → SH l=2 using inverse of _l2_source_to_cartesian
    # (grad_field is symmetric: ∂E_a/∂r_b = ∂E_b/∂r_a for conservative field)
    H = (grad_field + grad_field.T) / 2.0  # symmetrize full Hessian
    # The l=2 features project onto the TRACELESS part of the Hessian.
    # For the smeared Coulomb kernel T_s(r), ∇²T_s ≠ 0 inside the source, so
    # we subtract the trace to get the traceless EFG.
    trace = H[0, 0] + H[1, 1] + H[2, 2]
    H = H - (trace / 3.0) * torch.eye(3, dtype=H.dtype)
    return torch.stack([
        (2.0 / s3) * H[0, 1],              # m=-2: (2/√3)*xy
        (2.0 / s3) * H[1, 2],              # m=-1: (2/√3)*yz
        H[2, 2],                            # m=0:  H_zz (traceless)
        (2.0 / s3) * H[0, 2],              # m=+1: (2/√3)*xz
        (H[0, 0] - H[1, 1]) / s3,          # m=+2: (x²-y²)/√3
    ])


def test_efg_matches_gradient_of_field():
    receiver_position = torch.tensor([1.2, -0.7, 0.9], dtype=torch.get_default_dtype())

    features = _evaluate_pair_efg(receiver_position)
    efg_analytic = features[4:9]   # l=2 block: 5 SH components

    efg_fd = _finite_difference_efg(receiver_position)

    torch.testing.assert_close(efg_analytic, efg_fd, rtol=1e-4, atol=1e-6)


def test_efg_axis_aligned_quadrupole_symmetry():
    """On the z-axis the EFG from an m=0 quadrupole is axially symmetric."""
    receiver_position = torch.tensor([0.0, 0.0, 1.8], dtype=torch.get_default_dtype())
    features = _evaluate_pair_efg(receiver_position)
    efg = features[4:9]  # [m=-2, m=-1, m=0, m=+1, m=+2]

    # By symmetry on z-axis: m≠0 components should vanish
    assert efg[2].abs() > 1e-8, "m=0 EFG should be nonzero on z-axis"
    assert efg[0].abs() < 1e-12, "m=-2 EFG should vanish on z-axis"
    assert efg[1].abs() < 1e-12, "m=-1 EFG should vanish on z-axis"
    assert efg[3].abs() < 1e-12, "m=+1 EFG should vanish on z-axis"
    assert efg[4].abs() < 1e-12, "m=+2 EFG should vanish on z-axis"


def test_gto_electrostatic_features_support_quadrupole_sources_in_realspace():
    try:
        from graph_longrange.features import GTOElectrostaticFeatures
    except ImportError as exc:
        raise unittest.SkipTest(
            "GTOElectrostaticFeatures requires the full optional dependency stack."
        ) from exc

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, -0.5, 1.1]],
        dtype=torch.get_default_dtype(),
    )
    batch = torch.zeros(2, dtype=torch.long)
    pbc = torch.tensor([[False, False, False]], dtype=torch.bool)
    source_feats = _quadrupole_source_features().unsqueeze(1)

    model = GTOElectrostaticFeatures(
        density_max_l=2,
        density_smearing_width=1.1,
        feature_max_l=1,
        feature_smearing_widths=[0.7],
        include_self_interaction=False,
        kspace_cutoff=3.0,
    )
    reference = RealSpaceAnalyticalElectrostaticFeatures(
        density_max_l=2,
        density_smearing_width=1.1,
        projection_max_l=1,
        projection_smearing_widths=[0.7],
        include_self_interaction=False,
    )

    features = model(
        k_vectors=torch.zeros((0, 3), dtype=positions.dtype),
        k_norm2=torch.zeros(0, dtype=positions.dtype),
        k_vector_batch=torch.zeros(0, dtype=torch.long),
        k0_mask=torch.zeros(0, dtype=positions.dtype),
        source_feats=source_feats,
        node_positions=positions,
        batch=batch,
        volume=torch.ones(1, dtype=positions.dtype),
        pbc=pbc,
    )
    reference_features, _, _ = reference(
        source_feats=source_feats,
        node_positions=positions,
        batch=batch,
    )

    torch.testing.assert_close(features, reference_features, rtol=1e-7, atol=1e-9)
