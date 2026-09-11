"""
Tests for the reciprocal-space (k-space Ewald) implementations:

  - GTOElectrostaticEnergy._pbc_energy_batch
  - GTOElectrostaticFeatures._pbc_forward_dynamic
  - assemble_fourier_series_batch / apply_coulomb_kernel_batch / project_to_features_batch
  - k-vector computation (compute_k_vectors_flat)

All tests use the PHYSICAL k-vector convention:
    rcell = 2π × inv(cell).transpose(-1,-2)
as used by the production MATGL models (see _tensornet_polar_v3_1.py).

Tolerance rationale
-------------------
k-space energies converge as the cutoff increases; with kcut ≥ 4 Å⁻¹ the
dominant GTO modes (k < 1/σ ≈ 1 Å⁻¹) are well resolved.  Comparisons against
PME (mesh K=48) expect ≤ 0.5 % relative agreement for the chosen geometry.
Analytic identities (symmetry, sign, zero-charge) should hold to machine
precision (~1e-12) after the k-space self-correction.
"""

import math

import pytest
import torch

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.pme import PMEElectrostaticEnergy, PMEElectrostaticFeatures
from graph_longrange.realspace_electrostatics import RealSpaceAnalyticalEnergy

torch.set_default_dtype(torch.float64)

# ── shared constants ─────────────────────────────────────────────────────────

KCUT = 6.0          # k-space cutoff [Å⁻¹], physical convention
SIGMA = 1.2         # GTO smearing width [Å]
SIGMA_PROJ = [1.0, 1.5]
BOX_L = 10.0        # cubic box side [Å]
MESH = 48           # PME mesh for reference comparisons
PME_RTOL = 5e-3     # 0.5 % relative tolerance vs PME
PME_ATOL = 0.1


# ── helpers ──────────────────────────────────────────────────────────────────

def _phys_rcell(cell):
    """Physical reciprocal lattice: rcell = 2π × inv(cell).T"""
    return 2.0 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)


def _make_cubic(L: float, n_graphs: int = 1):
    cell = (torch.eye(3) * L).unsqueeze(0).expand(n_graphs, 3, 3).contiguous()
    rcell = _phys_rcell(cell)
    volume = torch.full((n_graphs,), L ** 3)
    return cell, rcell, volume


def _kvecs(cell, rcell, kcut=KCUT):
    return compute_k_vectors_flat(kcut, cell, rcell)


def _neutral_l0(n, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(n)
    q -= q.mean()
    return q.unsqueeze(-1)


def _neutral_l1(n, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(n); q -= q.mean()
    mu = torch.randn(n, 3) * 0.3
    # [q, μ_y, μ_z, μ_x]  (e3nn SH order)
    return torch.cat([q.unsqueeze(-1), mu[:, 1:2], mu[:, 2:3], mu[:, 0:1]], dim=1)


def _positions(n, L, seed=0):
    torch.manual_seed(seed)
    return torch.rand(n, 3) * (L * 0.6) + L * 0.2


def _run_energy(model, k_vecs, k_norm2, k_batch, k0_mask,
                sf, pos, batch, volume, pbc):
    return model(k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc)


def _run_features(model, k_vecs, k_norm2, k_batch, k0_mask,
                  sf, pos, batch, volume, pbc):
    return model(k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc)


# ── k-vector tests ────────────────────────────────────────────────────────────

class TestKVectors:
    """Tests for compute_k_vectors_flat with the physical 2π rcell convention."""

    def test_k_min_equals_2pi_over_L(self):
        """Smallest nonzero |k| must be 2π/L for a cubic box."""
        L = 8.0
        cell, rcell, _ = _make_cubic(L)
        k_vecs, k_norm2, _, _ = _kvecs(cell, rcell, kcut=5.0)
        # k=0 is included (first vector); smallest nonzero k ≈ 2π/L
        nonzero = k_norm2[k_norm2 > 1e-10]
        k_min = nonzero.min().sqrt().item()
        assert abs(k_min - 2 * math.pi / L) / (2 * math.pi / L) < 1e-6

    def test_all_k_within_cutoff(self):
        cell, rcell, _ = _make_cubic(BOX_L)
        k_vecs, k_norm2, _, _ = _kvecs(cell, rcell, kcut=KCUT)
        assert (k_norm2 > KCUT ** 2 + 1e-8).sum().item() == 0

    def test_k0_mask_at_first_entry(self):
        """k=0 vector is always the first entry and flagged by k0_mask=1."""
        cell, rcell, _ = _make_cubic(BOX_L)
        k_vecs, _, _, k0_mask = _kvecs(cell, rcell, kcut=KCUT)
        assert k_vecs[0].norm().item() < 1e-10
        assert k0_mask[0].item() == 1.0
        assert k0_mask[1:].sum().item() == 0.0

    def test_kvec_count_scales_with_cutoff(self):
        """More k-vectors for larger cutoff."""
        cell, rcell, _ = _make_cubic(BOX_L)
        n3, _, _, _ = _kvecs(cell, rcell, kcut=3.0)
        n6, _, _, _ = _kvecs(cell, rcell, kcut=6.0)
        assert n6.shape[0] > n3.shape[0]

    def test_batch_k_vectors(self):
        """Batched call returns separate k-vector sets per graph."""
        n_graphs = 2
        cell, rcell, _ = _make_cubic(BOX_L, n_graphs)
        k_vecs, _, k_batch, _ = _kvecs(cell, rcell, kcut=KCUT)
        assert (k_batch == 0).sum().item() > 0
        assert (k_batch == 1).sum().item() > 0
        # Same box → same count
        assert (k_batch == 0).sum().item() == (k_batch == 1).sum().item()


# ── energy tests ─────────────────────────────────────────────────────────────

class TestKSpaceEnergy:

    def _ref_model(self, dl, si=False, pbc_corr=False):
        return GTOElectrostaticEnergy(dl, SIGMA, KCUT, si, pbc_corr)

    def _pme_model(self, dl, si=False):
        return PMEElectrostaticEnergy(dl, SIGMA, MESH, si, include_pbc_corrections=False)

    # ── sign and nullity ──────────────────────────────────────────────────────

    def test_energy_zero_for_zero_charges(self):
        n = 4
        cell, rcell, volume = _make_cubic(BOX_L)
        pos = _positions(n, BOX_L, seed=0)
        batch = torch.zeros(n, dtype=torch.long)
        sf = torch.zeros(n, 1)
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)
        E = _run_energy(self._ref_model(0), k_vecs, k_norm2, k_batch, k0_mask,
                        sf, pos, batch, volume, pbc)
        assert E.abs().item() < 1e-10

    def test_energy_negative_for_opposite_charges(self):
        """Attractive dominant interaction → energy < 0."""
        cell, rcell, volume = _make_cubic(BOX_L)
        pos = torch.tensor([[2., 2., 2.], [5., 2., 2.]])
        batch = torch.zeros(2, dtype=torch.long)
        sf = torch.tensor([[1.], [-1.]])
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)
        E = _run_energy(self._ref_model(0, pbc_corr=False), k_vecs, k_norm2, k_batch,
                        k0_mask, sf, pos, batch, volume, pbc)
        assert E.item() < 0

    def test_energy_invariant_under_charge_inversion(self):
        """
        E(−q, r) = E(q, r): negating all charges leaves the energy unchanged
        because E ∝ Σᵢⱼ qᵢqⱼ V(rᵢⱼ) is even in q.
        """
        cell, rcell, volume = _make_cubic(BOX_L)
        pos = _positions(4, BOX_L, seed=1)
        batch = torch.zeros(4, dtype=torch.long)
        sf_pos = torch.tensor([[1.], [-1.], [0.5], [-0.5]])
        sf_neg = -sf_pos
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)
        model = self._ref_model(0, pbc_corr=False)
        Ep = _run_energy(model, k_vecs, k_norm2, k_batch, k0_mask, sf_pos, pos, batch, volume, pbc)
        En = _run_energy(model, k_vecs, k_norm2, k_batch, k0_mask, sf_neg, pos, batch, volume, pbc)
        assert abs(Ep.item() - En.item()) < 1e-10

    def test_l1_zero_dipoles_equals_l0(self):
        """l=1 energy with zero dipoles must equal l=0 energy."""
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 5
        pos = _positions(n, BOX_L, seed=2)
        batch = torch.zeros(n, dtype=torch.long)
        q = torch.tensor([1., -1., 0.5, -0.5, 0.])
        sf_l0 = q.unsqueeze(-1)
        sf_l1 = torch.zeros(n, 4); sf_l1[:, 0] = q
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)
        E0 = _run_energy(self._ref_model(0, pbc_corr=False), k_vecs, k_norm2, k_batch,
                         k0_mask, sf_l0, pos, batch, volume, pbc)
        E1 = _run_energy(self._ref_model(1, pbc_corr=False), k_vecs, k_norm2, k_batch,
                         k0_mask, sf_l1, pos, batch, volume, pbc)
        assert abs(E0.item() - E1.item()) / abs(E0.item()) < 1e-6

    # ── convergence with kcut ─────────────────────────────────────────────────

    def test_energy_converges_with_kcut(self):
        """Energy should plateau as kcut increases past 1/sigma."""
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 4
        pos = _positions(n, BOX_L, seed=3)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _neutral_l0(n, seed=3)
        pbc = torch.tensor([[True, True, True]])

        energies = []
        for kcut in [2.0, 4.0, 6.0, 8.0]:
            cell2, rcell2, _ = _make_cubic(BOX_L)
            k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell2, rcell2, kcut=kcut)
            model = GTOElectrostaticEnergy(0, SIGMA, kcut, False, False)
            E = _run_energy(model, k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc)
            energies.append(E.item())

        # Last two values should be stable (< 0.01 % change)
        assert abs(energies[-1] - energies[-2]) / max(abs(energies[-1]), 1e-8) < 1e-4

    # ── self-interaction flag ─────────────────────────────────────────────────

    def test_si_flag_changes_energy(self):
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 4
        pos = _positions(n, BOX_L, seed=4)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _neutral_l0(n, seed=4)
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)
        E_no = _run_energy(self._ref_model(0, si=False, pbc_corr=False),
                           k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc)
        E_si = _run_energy(self._ref_model(0, si=True, pbc_corr=False),
                           k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc)
        assert abs(E_si.item() - E_no.item()) > 1e-6

    # ── non-PBC fallback ──────────────────────────────────────────────────────

    def test_non_pbc_uses_realspace(self):
        """When pbc=False, GTOElectrostaticEnergy uses the real-space path."""
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 4
        pos = _positions(n, BOX_L, seed=5)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _neutral_l0(n, seed=5)
        pbc_false = torch.zeros(1, 3, dtype=torch.bool)
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)
        E = _run_energy(self._ref_model(0, pbc_corr=False), k_vecs, k_norm2, k_batch,
                        k0_mask, sf, pos, batch, volume, pbc_false)
        assert torch.isfinite(E).all()

    def test_non_pbc_matches_realspace_model(self):
        """Non-PBC path must agree with RealSpaceAnalyticalEnergy to < 1e-8."""
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 4
        pos = _positions(n, BOX_L, seed=6)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _neutral_l0(n, seed=6)
        pbc_false = torch.zeros(1, 3, dtype=torch.bool)
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)

        E_kspace = _run_energy(GTOElectrostaticEnergy(0, SIGMA, KCUT, False, False),
                               k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc_false)
        E_rs = RealSpaceAnalyticalEnergy(0, SIGMA, False)(sf, pos, batch)
        assert abs(E_kspace.item() - E_rs.item()) < 1e-8

    # ── batch ─────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("dl", [0, 1])
    def test_batch_energy(self, dl):
        n_graphs = 2
        n_per = 4
        cell, rcell, volume = _make_cubic(BOX_L, n_graphs)
        torch.manual_seed(10 + dl)
        pos = torch.rand(n_per * n_graphs, 3) * (BOX_L * 0.6) + BOX_L * 0.2
        batch = torch.cat([torch.zeros(n_per, dtype=torch.long),
                           torch.ones(n_per, dtype=torch.long)])
        sf = (_neutral_l1 if dl == 1 else _neutral_l0)(n_per * n_graphs, seed=10)
        pbc = torch.tensor([[True, True, True], [True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)
        E = _run_energy(self._ref_model(dl, pbc_corr=False), k_vecs, k_norm2, k_batch,
                        k0_mask, sf, pos, batch, volume, pbc)
        assert E.shape == (2,)
        assert torch.isfinite(E).all()

    def test_batch_additivity(self):
        """Energy of a 2-graph batch equals sum of individually computed energies."""
        n_per = 4
        cell, rcell, volume = _make_cubic(BOX_L)
        torch.manual_seed(20)
        pos0 = torch.rand(n_per, 3) * (BOX_L * 0.6) + BOX_L * 0.2
        pos1 = torch.rand(n_per, 3) * (BOX_L * 0.6) + BOX_L * 0.2
        q0 = torch.tensor([1., -1., 0.5, -0.5])
        q1 = torch.tensor([0.8, -0.3, -0.3, -0.2])
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)

        model = GTOElectrostaticEnergy(0, SIGMA, KCUT, False, False)
        E0 = _run_energy(model, k_vecs, k_norm2, k_batch, k0_mask,
                         q0.unsqueeze(-1), pos0, torch.zeros(n_per, dtype=torch.long), volume, pbc)
        E1 = _run_energy(model, k_vecs, k_norm2, k_batch, k0_mask,
                         q1.unsqueeze(-1), pos1, torch.zeros(n_per, dtype=torch.long), volume, pbc)

        # Batched call
        cell2, rcell2, volume2 = _make_cubic(BOX_L, 2)
        k_vecs2, k_norm2_2, k_batch2, k0_mask2 = _kvecs(cell2, rcell2)
        batch2 = torch.cat([torch.zeros(n_per, dtype=torch.long),
                            torch.ones(n_per, dtype=torch.long)])
        E_batch = _run_energy(model, k_vecs2, k_norm2_2, k_batch2, k0_mask2,
                              torch.cat([q0, q1]).unsqueeze(-1),
                              torch.cat([pos0, pos1]),
                              batch2, volume2,
                              torch.tensor([[True,True,True],[True,True,True]]))
        assert abs(E_batch[0].item() - E0.item()) / abs(E0.item()) < 1e-6
        assert abs(E_batch[1].item() - E1.item()) / abs(E1.item()) < 1e-6

    # ── PME agreement ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("dl,si", [(0, False), (0, True), (1, False)])
    def test_agrees_with_pme(self, dl, si):
        """k-space energy must match PME to 0.5 %."""
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 5
        pos = _positions(n, BOX_L, seed=30 + dl)
        batch = torch.zeros(n, dtype=torch.long)
        sf = (_neutral_l1 if dl == 1 else _neutral_l0)(n, seed=31)
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)

        E_k = _run_energy(self._ref_model(dl, si, pbc_corr=False),
                          k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc)
        E_pme = self._pme_model(dl, si)(sf, pos, batch, cell, volume, pbc)
        torch.testing.assert_close(E_k, E_pme, rtol=PME_RTOL, atol=PME_ATOL)

    # ── analytic pair ─────────────────────────────────────────────────────────

    def test_analytic_pair_energy(self):
        """
        For a well-separated +1/-1 pair (PBC images negligible),
        the k-space energy must match FIELD_CONSTANT/(4π) × q₁q₂ × erf(r/2σ)/r.
        """
        from graph_longrange.utils import FIELD_CONSTANT
        from scipy.constants import pi as SPI

        sigma, L, r = 1.5, 30.0, 4.0
        cell, rcell, volume = _make_cubic(L)
        cell = cell.contiguous(); rcell = _phys_rcell(cell)
        pos = torch.tensor([[0., 0., 0.], [r, 0., 0.]])
        batch = torch.zeros(2, dtype=torch.long)
        sf = torch.tensor([[1.], [-1.]])
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(4.0, cell, rcell)

        model = GTOElectrostaticEnergy(0, sigma, 4.0, False, False)
        E_k = _run_energy(model, k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch,
                          torch.tensor([L**3]), pbc)

        E_analytic = FIELD_CONSTANT / (4 * SPI) * (-1.) * math.erf(r / (2 * sigma)) / r
        assert abs(E_k.item() - E_analytic) / abs(E_analytic) < 0.02


# ── feature tests ─────────────────────────────────────────────────────────────

class TestKSpaceFeatures:

    def _ref_model(self, dl, pl, norm="receiver"):
        return GTOElectrostaticFeatures(
            dl, SIGMA, pl, SIGMA_PROJ, False, KCUT, integral_normalization=norm,
        )

    def _pme_model(self, dl, pl, norm="receiver"):
        return PMEElectrostaticFeatures(
            dl, SIGMA, pl, SIGMA_PROJ, MESH, False, norm,
        )

    # ── shape ─────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("dl,pl", [(0,0),(1,0),(0,1),(1,1)])
    def test_feature_shapes(self, dl, pl):
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 5
        pos = _positions(n, BOX_L, seed=40)
        batch = torch.zeros(n, dtype=torch.long)
        sf = (_neutral_l1 if dl==1 else _neutral_l0)(n, seed=41)
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)

        out = _run_features(self._ref_model(dl, pl), k_vecs, k_norm2, k_batch,
                            k0_mask, sf, pos, batch, volume, pbc)
        n_rad = len(SIGMA_PROJ)
        expected = n_rad if pl == 0 else 4 * n_rad
        assert out.shape == (n, expected), f"d{dl}p{pl}: {out.shape}"

    # ── zero / null inputs ────────────────────────────────────────────────────

    def test_zero_charges_zero_features(self):
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 4
        pos = _positions(n, BOX_L)
        batch = torch.zeros(n, dtype=torch.long)
        sf = torch.zeros(n, 1)
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)
        out = _run_features(self._ref_model(0, 0), k_vecs, k_norm2, k_batch,
                            k0_mask, sf, pos, batch, volume, pbc)
        assert out.abs().max().item() < 1e-10

    def test_l1_zero_dipoles_matches_l0(self):
        """l=1 density with zero dipoles → same l=0 features as l=0 density."""
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 4
        pos = _positions(n, BOX_L, seed=50)
        batch = torch.zeros(n, dtype=torch.long)
        q = torch.tensor([1., -1., 0.5, -0.5])
        sf_l0 = q.unsqueeze(-1)
        sf_l1 = torch.zeros(n, 4); sf_l1[:, 0] = q
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)

        out0 = _run_features(self._ref_model(0, 0), k_vecs, k_norm2, k_batch,
                              k0_mask, sf_l0, pos, batch, volume, pbc)
        out1 = _run_features(
            GTOElectrostaticFeatures(1, SIGMA, 0, SIGMA_PROJ, False, KCUT),
            k_vecs, k_norm2, k_batch, k0_mask, sf_l1, pos, batch, volume, pbc
        )
        assert (out0 - out1).abs().max().item() < 1e-8

    # ── non-PBC fallback ──────────────────────────────────────────────────────

    def test_non_pbc_falls_back_to_realspace(self):
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 4
        pos = _positions(n, BOX_L, seed=60)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _neutral_l1(n, seed=60)
        pbc_false = torch.zeros(1, 3, dtype=torch.bool)
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)
        out = _run_features(self._ref_model(1, 1), k_vecs, k_norm2, k_batch,
                            k0_mask, sf, pos, batch, volume, pbc_false)
        assert torch.isfinite(out).all()

    # ── convergence ───────────────────────────────────────────────────────────

    def test_features_converge_with_kcut(self):
        """L=0 features should stabilise once kcut > 1/sigma."""
        cell, _, volume = _make_cubic(BOX_L)
        n = 4
        pos = _positions(n, BOX_L, seed=70)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _neutral_l0(n, seed=70)
        pbc = torch.tensor([[True, True, True]])

        feats = []
        for kcut in [2.0, 4.0, 6.0, 8.0]:
            c, rc, _ = _make_cubic(BOX_L)
            kv, kn, kb, km = _kvecs(c, rc, kcut=kcut)
            model = GTOElectrostaticFeatures(0, SIGMA, 0, [SIGMA_PROJ[0]], False, kcut)
            out = _run_features(model, kv, kn, kb, km, sf, pos, batch, volume, pbc)
            feats.append(out)

        delta = (feats[-1] - feats[-2]).abs().max().item()
        assert delta / feats[-1].abs().max().item() < 1e-4

    # ── PME agreement ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("dl,pl,norm", [
        (0, 0, "receiver"),
        (1, 0, "receiver"),
        (0, 1, "receiver"),
        (1, 1, "receiver"),
        (1, 1, "multipoles"),
    ])
    def test_features_agree_with_pme(self, dl, pl, norm):
        """k-space features must match PME to 0.5 %."""
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 5
        pos = _positions(n, BOX_L, seed=80 + dl * 10 + pl)
        batch = torch.zeros(n, dtype=torch.long)
        sf = (_neutral_l1 if dl==1 else _neutral_l0)(n, seed=81)
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)

        F_k = _run_features(
            GTOElectrostaticFeatures(dl, SIGMA, pl, SIGMA_PROJ, False, KCUT,
                                     integral_normalization=norm),
            k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc,
        )
        F_pme = self._pme_model(dl, pl, norm)(
            sf.unsqueeze(1), pos, batch, cell, pbc
        )
        torch.testing.assert_close(F_k, F_pme, rtol=PME_RTOL, atol=PME_ATOL)

    # ── energy-feature self-consistency ──────────────────────────────────────

    def test_energy_equals_half_charge_dot_potential(self):
        """
        E = 0.5 × Σᵢ qᵢ × φᵢ  where φᵢ = raw l=0 feature / l0_factor.
        Verified by comparing GTOElectrostaticEnergy (si=False) against
        the l=0 feature channel of GTOElectrostaticFeatures (si=False)
        projected with σ_proj = σ_src.
        """
        from graph_longrange.utils import FIELD_CONSTANT
        from graph_longrange.gto_utils import get_Cl_sigma
        from scipy.constants import pi as SPI

        sigma_proj = SIGMA  # same as source: potential = feature / l0_factor
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 5
        pos = _positions(n, BOX_L, seed=90)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _neutral_l0(n, seed=91)
        pbc = torch.tensor([[True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)

        # k-space energy (no si, no pbc correction)
        E_k = _run_energy(
            GTOElectrostaticEnergy(0, SIGMA, KCUT, False, False),
            k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc,
        )
        # l=0 feature with σ_proj = σ_src (single channel)
        feat_model = GTOElectrostaticFeatures(0, SIGMA, 0, [sigma_proj], False, KCUT)
        F = _run_features(feat_model, k_vecs, k_norm2, k_batch, k0_mask,
                          sf, pos, batch, volume, pbc)  # [n, 1]

        l0_fac = get_Cl_sigma(0, sigma_proj, "receiver") / get_Cl_sigma(0, sigma_proj, "multipoles")
        phi = F[:, 0] / l0_fac          # recover raw potential
        E_from_feat = 0.5 * (sf[:, 0] * phi).sum()

        assert abs(E_from_feat.item() - E_k.item()) / abs(E_k.item()) < 1e-5

    # ── batch ─────────────────────────────────────────────────────────────────

    def test_features_batch(self):
        n_graphs = 2
        n_per = 4
        cell, rcell, volume = _make_cubic(BOX_L, n_graphs)
        torch.manual_seed(100)
        pos = torch.rand(n_per * n_graphs, 3) * (BOX_L * 0.6) + BOX_L * 0.2
        batch = torch.cat([torch.zeros(n_per, dtype=torch.long),
                           torch.ones(n_per, dtype=torch.long)])
        sf = _neutral_l1(n_per * n_graphs, seed=101)
        pbc = torch.tensor([[True, True, True], [True, True, True]])
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)

        out = _run_features(self._ref_model(1, 1), k_vecs, k_norm2, k_batch,
                            k0_mask, sf, pos, batch, volume, pbc)
        assert out.shape == (n_per * n_graphs, 4 * len(SIGMA_PROJ))
        assert torch.isfinite(out).all()

    # ── geometry corrections ──────────────────────────────────────────────────

    def test_molecule_correction_changes_features(self):
        """For a non-periodic (molecule) system, the PBC correction must matter."""
        cell, rcell, volume = _make_cubic(BOX_L)
        n = 4
        pos = _positions(n, BOX_L, seed=110)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _neutral_l1(n, seed=111)
        pbc_false = torch.zeros(1, 3, dtype=torch.bool)  # molecule
        k_vecs, k_norm2, k_batch, k0_mask = _kvecs(cell, rcell)

        out_corr = _run_features(
            GTOElectrostaticFeatures(1, SIGMA, 1, SIGMA_PROJ, False, KCUT,
                                     quadrupole_feature_corrections=False),
            k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume, pbc_false,
        )
        out_nocorr = _run_features(
            GTOElectrostaticFeatures(1, SIGMA, 1, SIGMA_PROJ, False, KCUT),
            k_vecs, k_norm2, k_batch, k0_mask, sf, pos, batch, volume,
            torch.tensor([[True, True, True]]),  # PBC → no molecule correction
        )
        # Molecule and bulk PBC give different features
        assert (out_corr - out_nocorr).abs().max().item() > 1e-6


# ── k-space building-block functions ─────────────────────────────────────────

class TestKSpaceFunctions:
    """Tests for assemble_fourier_series_batch, apply_coulomb_kernel_batch, etc."""

    def test_density_k0_is_total_charge(self):
        """
        ρ(k=0) ∝ Σᵢ qᵢ.  For a neutral system the k=0 density must vanish.
        """
        from graph_longrange.features import assemble_fourier_series_batch
        from graph_longrange.gto_utils import GTOBasis
        from graph_longrange.kspace import compute_k_vectors_flat

        L, sigma = 8.0, 1.2
        cell = (torch.eye(3) * L).unsqueeze(0)
        rcell = _phys_rcell(cell)
        k_vecs, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(5.0, cell, rcell)

        n = 4
        pos = _positions(n, L, seed=200)
        batch = torch.zeros(n, dtype=torch.long)
        q_neutral = torch.tensor([1., -1., 0.5, -0.5]).unsqueeze(-1)
        q_charged = torch.tensor([1., 1., 0.5, 0.5]).unsqueeze(-1)

        inner = torch.cos(torch.matmul(k_vecs, pos.t()))
        sines = torch.sin(torch.matmul(k_vecs, pos.t()))
        mask_f = (k_batch[:, None] == batch[None, :]).float()
        cosines = inner * mask_f
        sines   = sines  * mask_f

        basis = GTOBasis(0, [sigma], 5.0, "multipoles")
        db = basis(k_vecs, k_norm2, k0_mask)
        vol_per_k = torch.full((k_vecs.shape[0],), L ** 3)

        rho_neutral = assemble_fourier_series_batch(q_neutral, cosines, sines, db, vol_per_k)
        rho_charged = assemble_fourier_series_batch(q_charged, cosines, sines, db, vol_per_k)

        k0_idx = k0_mask.bool()
        # For neutral system: ρ(k=0) ≈ 0 (up to GTO normalization factors)
        assert rho_neutral[k0_idx].abs().max().item() < 1e-10
        # For charged system: ρ(k=0) ≠ 0
        assert rho_charged[k0_idx].abs().max().item() > 1e-6

    def test_coulomb_kernel_zeroes_k0(self):
        """apply_coulomb_kernel_batch must set the k=0 potential to zero."""
        from graph_longrange.features import apply_coulomb_kernel_batch

        k_norm2 = torch.tensor([0.0, 1.0, 2.0, 4.0])
        density = torch.rand(4, 2)
        k0_mask_bool = k_norm2 == 0.0

        k_factor = torch.zeros_like(k_norm2)
        k_factor[~k0_mask_bool] = 1.0 / k_norm2[~k0_mask_bool]

        potential = apply_coulomb_kernel_batch(k_norm2, density, k_factor)
        assert potential[0].abs().max().item() < 1e-12   # k=0 → 0
        assert potential[1:].abs().max().item() > 1e-6   # k≠0 → nonzero

    def test_project_features_shape(self):
        """project_to_features_batch output shape is [n_nodes, n_sigma, lm_dim]."""
        from graph_longrange.features import (
            assemble_fourier_series_batch,
            apply_coulomb_kernel_batch,
            project_to_features_batch,
        )
        from graph_longrange.gto_utils import GTOBasis

        L, sigma_src, sigma_proj = 8.0, 1.2, 1.0
        cell = (torch.eye(3) * L).unsqueeze(0)
        rcell = _phys_rcell(cell)
        k_vecs, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(5.0, cell, rcell)

        n = 4
        pos = _positions(n, L, seed=300)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _neutral_l0(n, seed=301)

        cos_ = torch.cos(torch.matmul(k_vecs, pos.t())) * (k_batch[:, None] == batch[None, :]).float()
        sin_ = torch.sin(torch.matmul(k_vecs, pos.t())) * (k_batch[:, None] == batch[None, :]).float()

        db = GTOBasis(0, [sigma_src], 5.0, "multipoles")(k_vecs, k_norm2, k0_mask)
        fb = GTOBasis(0, [sigma_proj], 5.0, "receiver")(k_vecs, k_norm2, k0_mask)
        vol = torch.full((k_vecs.shape[0],), L**3)

        density   = assemble_fourier_series_batch(sf, cos_, sin_, db, vol)
        k_factor  = torch.where(k_norm2 > 0, 1.0 / k_norm2.clamp(min=1e-12), torch.zeros_like(k_norm2))
        potential = apply_coulomb_kernel_batch(k_norm2, density, k_factor)
        features  = project_to_features_batch(potential, fb, cos_, sin_)

        # [n_nodes, n_sigma=1, lm_dim=1] for l=0
        assert features.shape == (n, 1, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_warp_kspace_reductions_match_dense_reference(dtype, tmp_path):
    warp = pytest.importorskip("warp")
    warp.config.kernel_cache_dir = str(tmp_path / "warp-kspace-cache")
    from graph_longrange.features import assemble_fourier_series_batch, project_to_features_batch
    from graph_longrange.kspace_warp import assemble_fourier_series_batch_warp, project_to_features_batch_warp

    torch.manual_seed(82)
    n_atoms, n_k, channels = 6, 11, 9
    positions = torch.randn(n_atoms, 3, dtype=dtype)
    k_vectors = torch.randn(n_k, 3, dtype=dtype)
    source = torch.randn(n_atoms, 1, channels, dtype=dtype)
    density_basis = torch.randn(n_k, 1, channels, 2, dtype=dtype)
    feature_basis = torch.randn_like(density_basis)
    volume_per_k = torch.rand(n_k, dtype=dtype) + 2
    k_batch = torch.zeros(n_k, dtype=torch.long)
    batch = torch.zeros(n_atoms, dtype=torch.long)
    phase = k_vectors @ positions.T

    expected_density = assemble_fourier_series_batch(
        source, phase.cos(), phase.sin(), density_basis, volume_per_k
    )
    actual_density = assemble_fourier_series_batch_warp(
        source, positions, k_vectors, k_batch, batch, density_basis, volume_per_k, 1
    )
    potential = torch.randn(n_k, 2, dtype=dtype)
    factor = torch.rand(n_k, dtype=dtype)
    expected_features = project_to_features_batch(
        potential, feature_basis, phase.cos(), phase.sin(), factor
    )
    actual_features = project_to_features_batch_warp(
        potential, feature_basis, positions, k_vectors, k_batch, batch, 1, factor
    )

    tolerance = 3e-4 if dtype == torch.float32 else 5e-13
    torch.testing.assert_close(actual_density, expected_density, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_features, expected_features, atol=tolerance, rtol=tolerance)
    normalized_density_error = (actual_density - expected_density).abs().max() / expected_density.abs().max()
    assert normalized_density_error < (4e-7 if dtype == torch.float32 else 1e-14)
