"""Strict parity tests for the opt-in pure-PyTorch electrostatics backend."""

import pytest
import torch

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.ewald_optimized import (
    GTOElectrostaticEnergyOptimized,
    GTOElectrostaticFeaturesOptimized,
)
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.pme import (
    PMEElectrostaticEnergy,
    PMEElectrostaticFeatures,
    _precompute_pme_reciprocal,
    compute_pme_single,
)
from graph_longrange.pme_optimized import (
    PMEElectrostaticEnergyOptimized,
    PMEElectrostaticFeaturesOptimized,
    build_pme_geometry_optimized,
    compute_pme_single_optimized,
)


@pytest.fixture(autouse=True)
def _stable_default_dtype():
    """Isolate these tests from modules that change PyTorch's process default."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_optimized_pme_l2_matches_in_forward_and_backward(dtype):
    torch.manual_seed(71)
    variables = (
        torch.rand(7, 3, dtype=dtype).requires_grad_(),
        (6.0 * torch.eye(3, dtype=dtype)).requires_grad_(),
        torch.randn(7, dtype=dtype).requires_grad_(),
        torch.randn(7, 3, dtype=dtype).requires_grad_(),
        torch.randn(7, 3, 3, dtype=dtype).requires_grad_(),
    )
    positions, box, charges, dipoles, quadrupoles = variables
    reciprocal = _precompute_pme_reciprocal(box, 1.0 / 2.4, 8)
    args = (positions, box, charges, dipoles, 1.0 / 2.4, 8, 2, True, quadrupoles, True, reciprocal)
    expected = compute_pme_single(*args)
    actual = compute_pme_single_optimized(*args)
    weights = tuple(torch.randn_like(value) for value in expected)
    expected_gradients = torch.autograd.grad(expected, variables, weights, retain_graph=True)
    actual_gradients = torch.autograd.grad(actual, variables, weights)
    forward_tolerance = 1e-7 if dtype == torch.float32 else 1e-13
    gradient_tolerance = 3e-4 if dtype == torch.float32 else 1e-10
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_value, expected_value, atol=forward_tolerance, rtol=forward_tolerance
        )
    for actual_value, expected_value in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(
            actual_value, expected_value, atol=gradient_tolerance, rtol=gradient_tolerance
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_optimized_pme_modules_l2_match_forward_and_backward(dtype):
    torch.manual_seed(73)
    positions = torch.rand(7, 3, dtype=dtype, requires_grad=True)
    box = (6.0 * torch.eye(3, dtype=dtype)).unsqueeze(0).requires_grad_()
    source = torch.randn(7, 9, dtype=dtype, requires_grad=True)
    batch = torch.zeros(7, dtype=torch.long)
    pbc = torch.ones(1, 3, dtype=torch.bool)
    volume = torch.linalg.det(box)
    feature_kwargs = {
        "density_max_l": 2,
        "density_smearing_width": 1.2,
        "feature_max_l": 2,
        "feature_smearing_widths": [1.2],
        "mesh_size": 8,
        "include_self_interaction": False,
        "integral_normalization": "receiver",
    }
    reference_features = PMEElectrostaticFeatures(**feature_kwargs).to(dtype)
    optimized_features = PMEElectrostaticFeaturesOptimized(**feature_kwargs).to(dtype)
    reference_energy = PMEElectrostaticEnergy(2, 1.2, 8, include_self_interaction=True).to(dtype)
    optimized_energy = PMEElectrostaticEnergyOptimized(2, 1.2, 8, include_self_interaction=True).to(dtype)
    expected = (
        reference_features(source, positions, batch, box, pbc),
        reference_energy(source, positions, batch, box, volume, pbc),
    )
    reciprocal = _precompute_pme_reciprocal(
        box[0], optimized_features.alphas[0], optimized_features.mesh_size
    )
    geometry = build_pme_geometry_optimized(
        positions,
        box[0],
        optimized_features.mesh_size,
        rank=2,
        Nj=reciprocal["Nj"],
    )
    geometry["reciprocal"] = {optimized_features.alphas[0]: reciprocal}
    shared_geometry = [geometry]
    actual = (
        optimized_features.forward_from_geometry(
            source, positions, batch, box, pbc, shared_geometry
        ),
        optimized_energy.forward_from_geometry(
            source, positions, batch, box, volume, pbc, shared_geometry
        ),
    )
    weights = tuple(torch.randn_like(value) for value in expected)
    variables = (positions, box, source)
    expected_gradients = torch.autograd.grad(expected, variables, weights, retain_graph=True)
    actual_gradients = torch.autograd.grad(actual, variables, weights)
    forward_tolerance = 1e-5 if dtype == torch.float32 else 1e-13
    gradient_tolerance = 3e-3 if dtype == torch.float32 else 2e-9
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_value, expected_value, atol=forward_tolerance, rtol=forward_tolerance
        )
    for actual_value, expected_value in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(
            actual_value, expected_value, atol=gradient_tolerance, rtol=gradient_tolerance
        )
    assert optimized_features.state_dict().keys() == reference_features.state_dict().keys()
    assert optimized_energy.state_dict().keys() == reference_energy.state_dict().keys()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_optimized_single_graph_ewald_l2_is_exact_in_forward_and_backward(dtype):
    torch.manual_seed(72)
    positions = torch.rand(7, 3, dtype=dtype, requires_grad=True)
    cell = (6.0 * torch.eye(3, dtype=dtype)).unsqueeze(0).requires_grad_()
    source = torch.randn(7, 9, dtype=dtype, requires_grad=True)
    batch = torch.zeros(7, dtype=torch.long)
    pbc = torch.ones(1, 3, dtype=torch.bool)
    volume = torch.linalg.det(cell)
    reciprocal_cell = 2.0 * torch.pi * torch.linalg.inv(cell).transpose(-1, -2)
    k_vectors, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(3.0, cell, reciprocal_cell)
    feature_kwargs = {
        "density_max_l": 2,
        "density_smearing_width": 1.2,
        "feature_max_l": 2,
        "feature_smearing_widths": [1.2],
        "include_self_interaction": False,
        "kspace_cutoff": 3.0,
        "quadrupole_feature_corrections": True,
        "integral_normalization": "receiver",
        "use_warp_kspace": False,
    }
    reference_features = GTOElectrostaticFeatures(**feature_kwargs).to(dtype)
    optimized_features = GTOElectrostaticFeaturesOptimized(**feature_kwargs).to(dtype)
    reference_energy = GTOElectrostaticEnergy(2, 1.2, 3.0, include_self_interaction=True).to(dtype)
    optimized_energy = GTOElectrostaticEnergyOptimized(2, 1.2, 3.0, include_self_interaction=True).to(dtype)
    args = (k_vectors, k_norm2, k_batch, k0_mask, source, positions, batch, volume, pbc)
    expected = (
        reference_features(*args, force_pbc_evaluator=True),
        reference_energy(*args, force_pbc_evaluator=True),
    )
    actual = (
        optimized_features(*args, force_pbc_evaluator=True),
        optimized_energy(*args, force_pbc_evaluator=True),
    )
    weights = tuple(torch.randn_like(value) for value in expected)
    variables = (positions, cell, source)
    expected_gradients = torch.autograd.grad(expected, variables, weights, retain_graph=True)
    actual_gradients = torch.autograd.grad(actual, variables, weights)
    for actual_value, expected_value in zip((*actual, *actual_gradients), (*expected, *expected_gradients), strict=True):
        tolerance = 1e-7 if dtype == torch.float32 else 0.0
        torch.testing.assert_close(actual_value, expected_value, atol=tolerance, rtol=tolerance)
    assert optimized_features.state_dict().keys() == reference_features.state_dict().keys()
    assert optimized_energy.state_dict().keys() == reference_energy.state_dict().keys()
