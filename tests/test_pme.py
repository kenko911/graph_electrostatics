"""
Tests comparing PME-based modules against the reference k-space Ewald implementation.

Key convention note
-------------------
The production code (matgl/_tensornet_polar_v3_1.py) uses:
    rcell = 2π * torch.linalg.inv(cell).transpose(-1,-2)

This gives the PHYSICAL reciprocal lattice vectors k = 2π·n·B⁻¹ (in Å⁻¹).
Older tests used torch.inverse(cell) without 2π, giving k = n/L, which is
NOT the correct Brillouin-zone k-vector convention.

PME internally uses k_phys = 2π · (fftfreq · N) · B⁻¹, matching the production
convention. Both methods agree to ~0.3% for the mesh sizes tested here.

Run with:
    pytest tests/test_pme.py -v
"""

import math

import pytest
import torch

import graph_longrange.pme as pme_module
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.pme import PMEElectrostaticEnergy, PMEElectrostaticFeatures

torch.set_default_dtype(torch.float64)

MESH = 32
KCUT = 6.0
BOX_L = 10.0
SIGMA_SRC = 1.2
SIGMA_PROJ = [1.0, 1.5]

PME_RTOL = 5e-3   # 0.5 %
PME_ATOL = 0.1    # absolute, for small features


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_npt_grid_cache_preserves_l2_values_forces_and_cell_gradients(dtype):
    """Fixed-grid reuse must not alter l=2 PME arithmetic for a live NPT cell."""
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(321)
        n_atoms, mesh = 4, 8
        base_coords = torch.rand(n_atoms, 3, dtype=dtype) * 4.0
        base_box = torch.tensor(
            [[6.1, 0.2, 0.0], [0.1, 5.8, 0.3], [0.0, 0.2, 6.3]], dtype=dtype
        )
        base_q = torch.randn(n_atoms, dtype=dtype)
        base_p = torch.randn(n_atoms, 3, dtype=dtype)
        base_Q = torch.randn(n_atoms, 3, 3, dtype=dtype)
        weights = (
            torch.randn(n_atoms, dtype=dtype),
            torch.randn(n_atoms, 3, dtype=dtype),
            torch.randn(n_atoms, 3, 3, dtype=dtype),
        )

        def evaluate(use_python_grid_shape):
            variables = tuple(
                value.clone().requires_grad_()
                for value in (base_coords, base_box, base_q, base_p, base_Q)
            )
            coords, box, q, p, Q = variables
            reciprocal = pme_module._precompute_pme_reciprocal(box, 1.0 / 2.4, mesh)
            if not use_python_grid_shape:
                reciprocal = {key: value for key, value in reciprocal.items() if key != "grid_shape"}
            output = pme_module.compute_pme_single(
                coords, box, q, p, 1.0 / 2.4, mesh, rank=2, Q=Q,
                want_field=True, want_hessian=True, reciprocal_cache=reciprocal,
            )
            gradients = torch.autograd.grad(output, variables, weights)
            return output, gradients

        cached_output, cached_gradients = evaluate(True)
        legacy_output, legacy_gradients = evaluate(False)
        for cached, legacy in zip(cached_output + cached_gradients, legacy_output + legacy_gradients):
            torch.testing.assert_close(cached, legacy, atol=0, rtol=0)
    finally:
        torch.set_default_dtype(previous_default)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cell(L: float):
    """Returns (cell [1,3,3], rcell [1,3,3]) using the PHYSICAL 2π convention."""
    cell = (torch.eye(3) * L).unsqueeze(0)
    # Production convention: rcell = 2π * inv(cell).T
    rcell = 2.0 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)
    return cell, rcell


def _kspace_inputs(cell, rcell, kcut=KCUT):
    return compute_k_vectors_flat(kcut, cell, rcell)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def neutral_l0_system():
    torch.manual_seed(42)
    n = 4
    positions = torch.rand(n, 3) * (BOX_L * 0.6) + BOX_L * 0.2
    batch = torch.zeros(n, dtype=torch.long)
    q = torch.tensor([1.0, -1.0, 0.5, -0.5])
    source_feats = q.unsqueeze(-1)
    cell, rcell = _make_cell(BOX_L)
    volume = torch.tensor([BOX_L**3])
    pbc = torch.tensor([[True, True, True]])
    box = cell   # [1, 3, 3]
    k_vecs, k_norm2, k_batch, k0_mask = _kspace_inputs(cell, rcell)
    return dict(positions=positions, batch=batch, source_feats=source_feats,
                volume=volume, pbc=pbc, box=box,
                k_vecs=k_vecs, k_norm2=k_norm2, k_batch=k_batch, k0_mask=k0_mask)


@pytest.fixture(scope="module")
def neutral_l1_system():
    torch.manual_seed(7)
    n = 4
    positions = torch.rand(n, 3) * (BOX_L * 0.6) + BOX_L * 0.2
    batch = torch.zeros(n, dtype=torch.long)
    source_feats = torch.tensor([
        [ 1.0,  0.1,  0.2,  0.3],
        [-1.0,  0.05, -0.1,  0.2],
        [ 0.5, -0.15,  0.1, -0.1],
        [-0.5,  0.0,  -0.2, -0.4],
    ])
    cell, rcell = _make_cell(BOX_L)
    volume = torch.tensor([BOX_L**3])
    pbc = torch.tensor([[True, True, True]])
    box = cell
    k_vecs, k_norm2, k_batch, k0_mask = _kspace_inputs(cell, rcell)
    return dict(positions=positions, batch=batch, source_feats=source_feats,
                volume=volume, pbc=pbc, box=box,
                k_vecs=k_vecs, k_norm2=k_norm2, k_batch=k_batch, k0_mask=k0_mask)


@pytest.fixture(scope="module")
def batch_l0_system():
    torch.manual_seed(99)
    n_per = 3
    pos0 = torch.rand(n_per, 3) * (BOX_L * 0.5) + BOX_L * 0.25
    pos1 = torch.rand(n_per, 3) * (BOX_L * 0.5) + BOX_L * 0.25
    positions = torch.cat([pos0, pos1], dim=0)
    batch = torch.cat([torch.zeros(n_per, dtype=torch.long),
                       torch.ones(n_per, dtype=torch.long)])
    q = torch.tensor([1.0, -0.5, -0.5,  0.8, -0.4, -0.4])
    source_feats = q.unsqueeze(-1)
    cell, rcell = _make_cell(BOX_L)
    cell = cell.expand(2, 3, 3).contiguous()
    rcell = rcell.expand(2, 3, 3).contiguous()
    volume = torch.tensor([BOX_L**3, BOX_L**3])
    pbc = torch.tensor([[True, True, True], [True, True, True]])
    box = cell
    k_vecs, k_norm2, k_batch, k0_mask = _kspace_inputs(cell, rcell)
    return dict(positions=positions, batch=batch, source_feats=source_feats,
                volume=volume, pbc=pbc, box=box,
                k_vecs=k_vecs, k_norm2=k_norm2, k_batch=k_batch, k0_mask=k0_mask)


@pytest.fixture(scope="module")
def neutral_l2_system():
    """Full multipole source [q, p(3), Q(5)] -> source_feats [n, 9]."""
    torch.manual_seed(2024)
    n = 5
    positions = torch.rand(n, 3) * (BOX_L * 0.6) + BOX_L * 0.2
    batch = torch.zeros(n, dtype=torch.long)
    q = torch.tensor([1.0, -1.0, 0.5, -0.8, 0.3])
    q = q - q.mean()                                   # neutral
    p = torch.randn(n, 3) * 0.3                         # l=1 (real-SH order m=-1,0,+1)
    quad = torch.randn(n, 5) * 0.4                      # l=2 (real-SH order m=-2..+2)
    source_feats = torch.cat([q.unsqueeze(-1), p, quad], dim=-1)  # [n, 9]
    cell, rcell = _make_cell(BOX_L)
    volume = torch.tensor([BOX_L**3])
    pbc = torch.tensor([[True, True, True]])
    box = cell
    k_vecs, k_norm2, k_batch, k0_mask = _kspace_inputs(cell, rcell)
    return dict(positions=positions, batch=batch, source_feats=source_feats,
                volume=volume, pbc=pbc, box=box,
                k_vecs=k_vecs, k_norm2=k_norm2, k_batch=k_batch, k0_mask=k0_mask)


# ── energy tests ──────────────────────────────────────────────────────────────

class TestPMEEnergy:

    def _ref(self, l, si):
        return GTOElectrostaticEnergy(
            density_max_l=l, density_smearing_width=SIGMA_SRC,
            kspace_cutoff=KCUT, include_self_interaction=si,
            include_pbc_corrections=False,
        )

    def _pme(self, l, si):
        return PMEElectrostaticEnergy(
            density_max_l=l, density_smearing_width=SIGMA_SRC,
            mesh_size=MESH, include_self_interaction=si,
            include_pbc_corrections=False,
        )

    def _run_ref(self, model, sys):
        return model(
            sys["k_vecs"], sys["k_norm2"], sys["k_batch"], sys["k0_mask"],
            sys["source_feats"], sys["positions"], sys["batch"],
            sys["volume"], sys["pbc"],
        )

    def _run_pme(self, model, sys):
        return model(
            sys["source_feats"], sys["positions"], sys["batch"],
            sys["box"], sys["volume"], sys["pbc"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("include_si", [False, True])
    def test_l0_single_graph(self, neutral_l0_system, include_si):
        E_ref = self._run_ref(self._ref(0, include_si), neutral_l0_system)
        E_pme = self._run_pme(self._pme(0, include_si), neutral_l0_system)
        torch.testing.assert_close(E_pme, E_ref, rtol=PME_RTOL, atol=PME_ATOL)

    @pytest.mark.parametrize("include_si", [False, True])
    def test_l1_single_graph(self, neutral_l1_system, include_si):
        E_ref = self._run_ref(self._ref(1, include_si), neutral_l1_system)
        E_pme = self._run_pme(self._pme(1, include_si), neutral_l1_system)
        torch.testing.assert_close(E_pme, E_ref, rtol=PME_RTOL, atol=PME_ATOL)

    @pytest.mark.parametrize("include_si", [False, True])
    def test_l2_single_graph(self, neutral_l2_system, include_si):
        """Quadrupole path: PME(density_max_l=2) matches k-space Ewald incl. all
        cross-terms (q-q, q-p, q-Q, p-p, p-Q, Q-Q)."""
        E_ref = self._run_ref(self._ref(2, include_si), neutral_l2_system)
        E_pme = self._run_pme(self._pme(2, include_si), neutral_l2_system)
        torch.testing.assert_close(E_pme, E_ref, rtol=PME_RTOL, atol=PME_ATOL)

    def test_l0_batch(self, batch_l0_system):
        E_ref = self._run_ref(self._ref(0, False), batch_l0_system)
        E_pme = self._run_pme(self._pme(0, False), batch_l0_system)
        torch.testing.assert_close(E_pme, E_ref, rtol=PME_RTOL, atol=PME_ATOL)

    def test_energy_negative_for_opposite_charges(self):
        """Attractive interaction: opposite charges → negative energy."""
        sigma, L = 1.0, 12.0
        pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        batch = torch.zeros(2, dtype=torch.long)
        sf = torch.tensor([[1.0], [-1.0]])
        box = (torch.eye(3) * L).unsqueeze(0)
        pmc = PMEElectrostaticEnergy(0, sigma, MESH, include_pbc_corrections=False)
        E = pmc(sf, pos, batch, box, torch.tensor([L**3]),
                torch.tensor([[True, True, True]]))
        assert E.item() < 0.0, f"Expected E < 0, got {E.item()}"

    def test_energy_zero_for_zero_charges(self):
        pos = torch.rand(3, 3) * 8.0 + 1.0
        batch = torch.zeros(3, dtype=torch.long)
        sf = torch.zeros(3, 1)
        box = (torch.eye(3) * 10.0).unsqueeze(0)
        pmc = PMEElectrostaticEnergy(0, SIGMA_SRC, MESH, include_pbc_corrections=False)
        E = pmc(sf, pos, batch, box, torch.tensor([1000.0]),
                torch.tensor([[True, True, True]]))
        assert abs(E.item()) < 1e-8

    def test_non_pbc_uses_realspace(self, neutral_l0_system):
        """When pbc=False, PME falls back to real-space (finite but ~-4 to -6 range)."""
        pmc = self._pme(0, False)
        sys = neutral_l0_system
        E = pmc(sys["source_feats"], sys["positions"], sys["batch"],
                sys["box"], sys["volume"],
                torch.zeros(1, 3, dtype=torch.bool))
        assert torch.isfinite(E).all()

    def test_energy_matches_analytic_pair(self):
        """
        For a well-separated +1/-1 pair, energy ≈ K/(4π)·q₁q₂·erf(r/2σ)/r.
        PBC corrections negligible for large box.
        """
        from graph_longrange.utils import FIELD_CONSTANT
        sigma, L, r = 1.0, 30.0, 4.0
        pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        batch = torch.zeros(2, dtype=torch.long)
        sf = torch.tensor([[1.0], [-1.0]])
        box = (torch.eye(3) * L).unsqueeze(0)
        E_analytic = FIELD_CONSTANT / (4*math.pi) * (-1.0) * math.erf(r / (2*sigma)) / r

        pmc = PMEElectrostaticEnergy(0, sigma, 48, include_pbc_corrections=False)
        E = pmc(sf, pos, batch, box, torch.tensor([L**3]),
                torch.tensor([[True, True, True]]))
        # PME is the periodic sum; for large neutral box it ≈ isolated pair
        assert abs(E.item() - E_analytic) / abs(E_analytic) < 0.02, \
            f"PME={E.item():.6f}, analytic={E_analytic:.6f}"


# ── feature tests ─────────────────────────────────────────────────────────────

class TestPMEFeatures:

    def _ref(self, dl, pl, norm="receiver"):
        return GTOElectrostaticFeatures(
            density_max_l=dl, density_smearing_width=SIGMA_SRC,
            feature_max_l=pl, feature_smearing_widths=SIGMA_PROJ,
            include_self_interaction=False, kspace_cutoff=KCUT,
            integral_normalization=norm,
        )

    def _pme(self, dl, pl, norm="receiver"):
        return PMEElectrostaticFeatures(
            density_max_l=dl, density_smearing_width=SIGMA_SRC,
            feature_max_l=pl, feature_smearing_widths=SIGMA_PROJ,
            mesh_size=MESH, include_self_interaction=False,
            integral_normalization=norm,
        )

    def _run_ref(self, model, sys):
        return model(
            sys["k_vecs"], sys["k_norm2"], sys["k_batch"], sys["k0_mask"],
            sys["source_feats"], sys["positions"], sys["batch"],
            sys["volume"], sys["pbc"],
        )

    def _run_pme(self, model, sys):
        sf = sys["source_feats"]
        if sf.dim() == 2:
            sf = sf.unsqueeze(1)
        return model(sf, sys["positions"], sys["batch"], sys["box"], sys["pbc"])

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _pick(dl, l0, l1, l2):
        return {0: l0, 1: l1, 2: l2}[dl]

    @pytest.mark.parametrize("dl,pl", [(0, 0), (1, 0), (0, 1), (1, 1),
                                       (0, 2), (1, 2), (2, 2)])
    def test_feature_shapes(self, neutral_l0_system, neutral_l1_system,
                            neutral_l2_system, dl, pl):
        sys = self._pick(dl, neutral_l0_system, neutral_l1_system, neutral_l2_system)
        out = self._run_pme(self._pme(dl, pl), sys)
        n = sys["positions"].shape[0]
        expected = (pl + 1) ** 2 * len(SIGMA_PROJ)
        assert out.shape == (n, expected)

    @pytest.mark.parametrize("dl,pl,norm", [
        (0, 0, "receiver"),
        (1, 0, "receiver"),
        (0, 1, "receiver"),
        (1, 1, "receiver"),
        (1, 1, "multipoles"),
        # quadrupole feature path (EFG projection)
        (0, 2, "receiver"),
        (1, 2, "receiver"),
        (2, 2, "receiver"),
        (2, 2, "multipoles"),
    ])
    def test_features_match_kspace(self, neutral_l0_system, neutral_l1_system,
                                   neutral_l2_system, dl, pl, norm):
        sys = self._pick(dl, neutral_l0_system, neutral_l1_system, neutral_l2_system)
        F_ref = self._run_ref(self._ref(dl, pl, norm), sys)
        F_pme = self._run_pme(self._pme(dl, pl, norm), sys)
        torch.testing.assert_close(F_pme, F_ref, rtol=PME_RTOL, atol=PME_ATOL)

    def test_zero_charges_zero_features(self):
        pos = torch.rand(3, 3) * 8.0 + 1.0
        batch = torch.zeros(3, dtype=torch.long)
        sf = torch.zeros(3, 1, 1)
        box = (torch.eye(3) * 10.0).unsqueeze(0)
        pme = PMEElectrostaticFeatures(0, SIGMA_SRC, 0, SIGMA_PROJ, MESH)
        out = pme(sf, pos, batch, box, torch.tensor([[True, True, True]]))
        assert out.abs().max().item() < 1e-8

    def test_fixed_cell_reciprocal_cache_is_reused_and_invalidated(self, neutral_l2_system):
        sys = neutral_l2_system
        model = self._pme(2, 2)
        first = self._run_pme(model, sys)
        cache = model._pme_reciprocal_cache
        second = self._run_pme(model, sys)
        assert model._pme_reciprocal_cache is cache
        torch.testing.assert_close(second, first, atol=0, rtol=0)

        changed = dict(sys)
        changed["box"] = sys["box"].clone()
        changed["box"][0, 0, 0] += 0.01
        third = self._run_pme(model, changed)
        assert model._pme_reciprocal_cache is not cache
        fresh = self._run_pme(self._pme(2, 2), changed)
        torch.testing.assert_close(third, fresh, atol=0, rtol=0)

    def test_non_pbc_falls_back(self, neutral_l1_system):
        sys = neutral_l1_system
        pme = self._pme(1, 1)
        out = pme(sys["source_feats"].unsqueeze(1), sys["positions"], sys["batch"],
                  sys["box"], torch.zeros(1, 3, dtype=torch.bool))
        assert torch.isfinite(out).all()


# ── regression (pinned values) ────────────────────────────────────────────────

class TestRegressions:
    """
    Pinned regression: PME must agree with k-space within 0.5 % (both use
    physical 2π-scaled rcell).
    """

    def test_l0_energy_regression(self, neutral_l0_system):
        sys = neutral_l0_system
        ref = GTOElectrostaticEnergy(0, SIGMA_SRC, KCUT, include_self_interaction=False,
                                     include_pbc_corrections=False)
        E_ref = ref(sys["k_vecs"], sys["k_norm2"], sys["k_batch"], sys["k0_mask"],
                    sys["source_feats"], sys["positions"], sys["batch"],
                    sys["volume"], sys["pbc"])

        pme = PMEElectrostaticEnergy(0, SIGMA_SRC, MESH, include_pbc_corrections=False)
        E_pme = pme(sys["source_feats"], sys["positions"], sys["batch"],
                    sys["box"], sys["volume"], sys["pbc"])

        assert torch.sign(E_pme) == torch.sign(E_ref) or E_ref.abs() < 1e-6
        rel = (E_pme - E_ref).abs() / E_ref.abs().clamp(min=1e-8)
        assert rel.item() < 0.005, f"Regression: rel_diff={rel.item():.4f}"

    def test_l1_feature_regression(self, neutral_l1_system):
        sys = neutral_l1_system
        ref = GTOElectrostaticFeatures(
            1, SIGMA_SRC, 1, SIGMA_PROJ, False, KCUT,
        )
        F_ref = ref(sys["k_vecs"], sys["k_norm2"], sys["k_batch"], sys["k0_mask"],
                    sys["source_feats"], sys["positions"], sys["batch"],
                    sys["volume"], sys["pbc"])
        pme = PMEElectrostaticFeatures(1, SIGMA_SRC, 1, SIGMA_PROJ, MESH)
        F_pme = pme(sys["source_feats"].unsqueeze(1), sys["positions"], sys["batch"],
                    sys["box"], sys["pbc"])

        # Check max absolute error relative to max feature magnitude
        abs_err = (F_pme - F_ref).abs().max()
        scale = F_ref.abs().max().clamp(min=1e-8)
        assert (abs_err / scale).item() < 0.005, \
            f"Feature regression: rel_err={abs_err/scale:.4f}"
