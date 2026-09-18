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
    _bspline,
    _bspline_prime,
    _bspline_double_prime,
    _get_u_reference,
    _get_recip_vectors,
    _make_stencil,
    PMEElectrostaticEnergy,
    PMEElectrostaticFeatures,
    _precompute_pme_reciprocal,
    compute_pme_single,
)
from graph_longrange.pme_optimized import (
    _expand_axis_weights,
    _pme_rfft_factors,
    _precompute_pme_reciprocal_rfft,
    _cached_pme_reciprocal_optimized,
    PMEElectrostaticEnergyOptimized,
    PMEElectrostaticFeaturesOptimized,
    build_pme_geometry_optimized,
    compute_pme_single_optimized,
)


@pytest.mark.parametrize("mesh", [8, 9, 10, 32])
def test_float32_half_spectrum_matches_full_grid_and_cell_derivatives(mesh):
    box = torch.tensor([[6., .2, .1], [0., 5.8, .3], [.1, 0., 6.2]],
                       dtype=torch.float32, requires_grad=True)
    full = _precompute_pme_reciprocal(box, 1 / 2.4, mesh)
    half = _precompute_pme_reciprocal_rfft(box, 1 / 2.4, mesh)
    assert "coulomb" not in half and "theta_safe" not in half
    fg, ft = _pme_rfft_factors(full, mesh)
    hg, ht = _pme_rfft_factors(half, mesh)
    torch.testing.assert_close(hg, fg, atol=1e-7, rtol=1e-6)
    # A shorter SIMD tail can round a few non-power-of-two entries differently.
    # This is factor parity, not bitwise identity; output tolerances stay unchanged.
    torch.testing.assert_close(ht, ft, atol=1e-7, rtol=1e-6)
    weights = torch.linspace(.1, 1., fg.numel()).reshape_as(fg)
    expected = torch.autograd.grad((fg * weights).sum(), box, retain_graph=True)[0]
    actual = torch.autograd.grad((hg * weights).sum(), box)[0]
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def test_half_spectrum_cache_invalidates_on_cell_change():
    module = torch.nn.Module()
    box = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 6
    first = _cached_pme_reciprocal_optimized(module, box, [.4], 8)
    assert _cached_pme_reciprocal_optimized(module, box.clone(), [.4], 8) is first
    box[0, 0, 0] += .1
    second = _cached_pme_reciprocal_optimized(module, box, [.4], 8)
    assert second is not first
    live = box.clone().requires_grad_()
    assert _cached_pme_reciprocal_optimized(module, live, [.4], 8) is not second
    with torch.inference_mode():
        assert _cached_pme_reciprocal_optimized(module, box.clone(), [.4], 8) is second


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mesh", [8, 10])
def test_rfft_factor_reuse_preserves_live_cell_gradients(dtype, mesh):
    box = torch.tensor(
        [[6.0, 0.2, 0.1], [0.0, 5.8, 0.3], [0.1, 0.0, 6.2]],
        dtype=dtype, requires_grad=True,
    )
    cache = _precompute_pme_reciprocal(box, 1.0 / 2.4, mesh)
    green, theta2 = _pme_rfft_factors(cache, mesh)
    reused_green, reused_theta2 = _pme_rfft_factors(cache, mesh)
    assert green is reused_green and theta2 is reused_theta2
    assert green.is_contiguous() and theta2.is_contiguous()
    expected_green = cache["coulomb"].reshape(mesh, mesh, mesh)[:, :, :mesh // 2 + 1]
    expected_theta2 = cache["theta_safe"].reshape(mesh, mesh, mesh)[:, :, :mesh // 2 + 1].square()
    torch.testing.assert_close(green, expected_green, atol=0, rtol=0)
    torch.testing.assert_close(theta2, expected_theta2, atol=0, rtol=0)
    expected_grad = torch.autograd.grad(expected_green.sum(), box, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(green.sum(), box)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_atoms", [0, 1, 17])
def test_separable_spline_stencil_preserves_values_and_derivatives(dtype, n_atoms):
    """Axis expansion retains the original meshgrid order and autograd paths."""
    torch.manual_seed(82)
    coords = (torch.rand(n_atoms, 3, dtype=dtype) * 5).requires_grad_()
    box = torch.tensor(
        [[6.0, 0.2, 0.1], [0.0, 5.8, 0.3], [0.1, 0.0, 6.2]],
        dtype=dtype,
        requires_grad=True,
    )
    Nj = _get_recip_vectors(torch.full((3,), 32, dtype=dtype), box)
    _, u0 = _get_u_reference(coords, Nj, 6)
    shifts = _make_stencil(6, coords.device, dtype)
    axis_shifts = torch.arange(-3, 3, dtype=dtype)
    for spline in (_bspline, _bspline_prime, _bspline_double_prime):
        expected = spline(u0[:, None, :] + shifts)
        actual = _expand_axis_weights(spline(u0[:, None, :] + axis_shifts[None, :, None]))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # One stencil entry isolates derivative parity from summation-order
        # differences introduced when 36 repeated axis contributions combine.
        expected_loss = expected[:, 73, :].sum()
        actual_loss = actual[:, 73, :].sum()
        expected_grads = torch.autograd.grad(expected_loss, (coords, box), retain_graph=True)
        actual_grads = torch.autograd.grad(actual_loss, (coords, box), retain_graph=True)
        for a, e in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(a, e, atol=0, rtol=0)


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
@pytest.mark.parametrize("mesh_size", [8, 10], ids=["power-of-two", "five-smooth"])
def test_optimized_pme_l2_matches_in_forward_and_backward(dtype, mesh_size):
    torch.manual_seed(71)
    variables = (
        torch.rand(7, 3, dtype=dtype).requires_grad_(),
        (6.0 * torch.eye(3, dtype=dtype)).requires_grad_(),
        torch.randn(7, dtype=dtype).requires_grad_(),
        torch.randn(7, 3, dtype=dtype).requires_grad_(),
        torch.randn(7, 3, 3, dtype=dtype).requires_grad_(),
    )
    positions, box, charges, dipoles, quadrupoles = variables
    reciprocal = _precompute_pme_reciprocal(box, 1.0 / 2.4, mesh_size)
    args = (
        positions,
        box,
        charges,
        dipoles,
        1.0 / 2.4,
        mesh_size,
        2,
        True,
        quadrupoles,
        True,
        reciprocal,
    )
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
    assert geometry["hessian_weights"].shape[-1] == 6
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
