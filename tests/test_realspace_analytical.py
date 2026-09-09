"""
Tests verifying RealSpaceAnalyticalEnergy and RealSpaceAnalyticalElectrostaticFeatures
against the original finite-difference implementations.

Tolerance rationale
-------------------
FD uses a centered difference with offset δ, giving O(δ²) truncation error.
With δ = 1e-4 (optimal for float64 without cancellation):
  - Absolute error ≲ 3e-5 for features of magnitude O(10)
  - Relative error ≲ 1e-6 for large features; up to ~5e-5 for near-zero features
    (the relative metric becomes noisy when |F| < 0.1)

All parametrized cases use analytical vs FD[δ=1e-4] comparison.
Autograd tests use FD[δ=1e-5] of the energy/feature to verify gradients.
"""

import math

import pytest
import torch

from graph_longrange.realspace_electrostatics import (
    RealSpaceAnalyticalEnergy,
    RealSpaceAnalyticalElectrostaticFeatures,
    RealSpaceFiniteDiffereneEnergy,
    RealSpaceFiniteDifferenceElectrostaticFeatures,
    batch_complete_graph_excluding_self_duplicates_vector,
)

torch.set_default_dtype(torch.float64)

# ── constants ────────────────────────────────────────────────────────────────

SIGMA_SRC = 0.6
SIGMA_PROJ = [0.5, 0.8]
FD_OFFSET = 1e-4
ABS_TOL = 1e-4   # absolute tolerance (feature magnitudes up to ~40)
REL_TOL = 5e-4   # relative tolerance for elements with |value| > 0.1


# ── helpers ──────────────────────────────────────────────────────────────────

def _assert_close_fd(analytical, fd, label=""):
    """
    Assert that analytical matches FD element-wise.  Uses absolute tolerance
    for small values (|FD| < 0.1) and relative tolerance otherwise.
    """
    abs_diff = (analytical - fd).abs()
    fd_abs = fd.abs()

    # Absolute check (catches near-zero elements)
    assert abs_diff.max().item() < ABS_TOL, (
        f"{label}: max abs diff {abs_diff.max().item():.2e} > {ABS_TOL}"
    )
    # Relative check on large elements
    large_mask = fd_abs > 0.1
    if large_mask.any():
        rel = (abs_diff[large_mask] / fd_abs[large_mask]).max().item()
        assert rel < REL_TOL, (
            f"{label}: max rel diff on large elements {rel:.2e} > {REL_TOL}"
        )


def _make_positions(n, scale=3.0, seed=0):
    torch.manual_seed(seed)
    return torch.rand(n, 3) * scale + 0.5


def _source_feats_l0(n, seed=1):
    torch.manual_seed(seed)
    q = torch.randn(n) * 0.8
    q -= q.mean()   # neutral
    return q.unsqueeze(-1)


def _source_feats_l1(n, seed=2):
    torch.manual_seed(seed)
    q = torch.randn(n) * 0.5; q -= q.mean()
    mu = torch.randn(n, 3) * 0.3
    return torch.cat([q.unsqueeze(-1), mu[:, 1:2], mu[:, 2:3], mu[:, 0:1]], dim=1)


# ── energy tests ─────────────────────────────────────────────────────────────

class TestRealSpaceAnalyticalEnergy:
    """
    RealSpaceAnalyticalEnergy vs RealSpaceFiniteDiffereneEnergy.
    Tests all (density_max_l, include_si) combinations on multi-atom batches.
    """

    @pytest.mark.parametrize("include_si", [False, True])
    def test_l0_matches_fd(self, include_si):
        n = 5
        positions = _make_positions(n, seed=10)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _source_feats_l0(n, seed=11)

        fd  = RealSpaceFiniteDiffereneEnergy(0, SIGMA_SRC, include_si, offset=FD_OFFSET)
        ana = RealSpaceAnalyticalEnergy(0, SIGMA_SRC, include_si)

        E_fd  = fd(sf, positions, batch)
        E_ana = ana(sf, positions, batch)
        _assert_close_fd(E_ana, E_fd, f"l0 si={include_si}")

    @pytest.mark.parametrize("include_si", [False, True])
    def test_l1_matches_fd(self, include_si):
        n = 5
        positions = _make_positions(n, seed=20)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _source_feats_l1(n, seed=21)

        fd  = RealSpaceFiniteDiffereneEnergy(1, SIGMA_SRC, include_si, offset=FD_OFFSET)
        ana = RealSpaceAnalyticalEnergy(1, SIGMA_SRC, include_si)

        E_fd  = fd(sf, positions, batch)
        E_ana = ana(sf, positions, batch)
        _assert_close_fd(E_ana, E_fd, f"l1 si={include_si}")

    def test_batch_l0(self):
        """Two graphs in a batch."""
        n0, n1 = 3, 4
        torch.manual_seed(30)
        pos = torch.rand(n0 + n1, 3) * 4.0 + 0.5
        batch = torch.cat([torch.zeros(n0, dtype=torch.long),
                           torch.ones(n1, dtype=torch.long)])
        q = torch.tensor([1., -0.5, -0.5, 0.8, -0.3, -0.3, -0.2])
        sf = q.unsqueeze(-1)

        fd  = RealSpaceFiniteDiffereneEnergy(0, SIGMA_SRC, False, offset=FD_OFFSET)
        ana = RealSpaceAnalyticalEnergy(0, SIGMA_SRC, False)

        E_fd  = fd(sf, pos, batch)
        E_ana = ana(sf, pos, batch)
        assert E_fd.shape == (2,)
        _assert_close_fd(E_ana, E_fd, "batch l0")

    def test_batch_l1(self):
        n0, n1 = 3, 3
        torch.manual_seed(40)
        pos = torch.rand(n0 + n1, 3) * 3.0 + 0.5
        batch = torch.cat([torch.zeros(n0, dtype=torch.long),
                           torch.ones(n1, dtype=torch.long)])
        sf = _source_feats_l1(n0 + n1, seed=41)

        fd  = RealSpaceFiniteDiffereneEnergy(1, SIGMA_SRC, False, offset=FD_OFFSET)
        ana = RealSpaceAnalyticalEnergy(1, SIGMA_SRC, False)

        E_fd  = fd(sf, pos, batch)
        E_ana = ana(sf, pos, batch)
        assert E_fd.shape == (2,)
        _assert_close_fd(E_ana, E_fd, "batch l1")

    def test_zero_charges_zero_energy(self):
        pos = _make_positions(4, seed=50)
        batch = torch.zeros(4, dtype=torch.long)
        sf = torch.zeros(4, 1)
        ana = RealSpaceAnalyticalEnergy(0, SIGMA_SRC, False)
        E = ana(sf, pos, batch)
        assert E.abs().item() < 1e-12

    def test_zero_dipoles_l1_matches_l0(self):
        """l=1 with zero dipoles should give same energy as l=0.

        The l=0 path (charges_energy_from_graph) uses a 1e-6 denominator
        stabiliser; the l=1 path (multipole_energy_from_graph) uses clamp(1e-10).
        Both converge to the same value but differ by at most ~1e-5.
        """
        pos = _make_positions(4, seed=60)
        batch = torch.zeros(4, dtype=torch.long)
        q = torch.tensor([1., -1., 0.5, -0.5])
        sf_l0 = q.unsqueeze(-1)
        sf_l1 = torch.zeros(4, 4); sf_l1[:, 0] = q

        ana0 = RealSpaceAnalyticalEnergy(0, SIGMA_SRC, False)
        ana1 = RealSpaceAnalyticalEnergy(1, SIGMA_SRC, False)

        E0 = ana0(sf_l0, pos, batch)
        E1 = ana1(sf_l1, pos, batch)
        # Different stabiliser conventions: l=0 uses 1e-6, l=1 uses clamp(1e-10)
        assert (E0 - E1).abs().item() < 1e-4

    def test_energy_sign_attractive(self):
        """Opposite charges should give negative (attractive) energy."""
        pos = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
        batch = torch.zeros(2, dtype=torch.long)
        sf = torch.tensor([[1.], [-1.]])
        ana = RealSpaceAnalyticalEnergy(0, SIGMA_SRC, False)
        E = ana(sf, pos, batch)
        assert E.item() < 0

    def test_energy_additive_over_batch(self):
        """Energy of two separate graphs = sum computed individually."""
        pos0 = _make_positions(3, seed=70)
        pos1 = _make_positions(3, seed=71) + 10.  # far apart
        batch = torch.cat([torch.zeros(3, dtype=torch.long),
                           torch.ones(3, dtype=torch.long)])
        q = torch.tensor([1., -0.5, -0.5, 0.8, -0.4, -0.4])
        sf = q.unsqueeze(-1)
        b0, b1 = torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)

        ana = RealSpaceAnalyticalEnergy(0, SIGMA_SRC, False)
        E_batch = ana(sf, torch.cat([pos0, pos1]), batch)
        E0 = ana(sf[:3], pos0, b0)
        E1 = ana(sf[3:], pos1, b1)

        assert abs(E_batch[0].item() - E0.item()) < 1e-10
        assert abs(E_batch[1].item() - E1.item()) < 1e-10

    def test_energy_gradient_via_autograd(self):
        """Autograd gradient of energy w.r.t. positions matches FD."""
        pos = _make_positions(4, seed=80).requires_grad_(True)
        batch = torch.zeros(4, dtype=torch.long)
        sf = _source_feats_l1(4, seed=81)

        ana = RealSpaceAnalyticalEnergy(1, SIGMA_SRC, False)
        E = ana(sf, pos, batch)
        E.backward()
        grad_auto = pos.grad.clone()

        eps = 1e-5
        grad_fd = torch.zeros_like(pos.detach())
        with torch.no_grad():
            for i in range(4):
                for j in range(3):
                    pp = pos.detach().clone(); pp[i, j] += eps
                    pm = pos.detach().clone(); pm[i, j] -= eps
                    Ep = ana(sf, pp, batch)
                    Em = ana(sf, pm, batch)
                    grad_fd[i, j] = (Ep - Em) / (2 * eps)

        rel = ((grad_auto - grad_fd).abs() / grad_fd.abs().clamp(min=1e-8)).max().item()
        assert rel < 1e-6, f"Autograd gradient mismatch: {rel:.2e}"

    def test_single_atom_no_edges(self):
        """Single atom: no inter-atomic energy (only self-interaction if enabled)."""
        pos = torch.tensor([[0., 0., 0.]])
        batch = torch.zeros(1, dtype=torch.long)
        sf = torch.tensor([[2.0]])
        ana_noself = RealSpaceAnalyticalEnergy(0, SIGMA_SRC, False)
        ana_self   = RealSpaceAnalyticalEnergy(0, SIGMA_SRC, True)
        assert ana_noself(sf, pos, batch).abs().item() < 1e-12
        assert ana_self(sf, pos, batch).item() > 0  # self-interaction is positive


# ── feature tests ─────────────────────────────────────────────────────────────

class TestRealSpaceAnalyticalFeatures:
    """
    RealSpaceAnalyticalElectrostaticFeatures vs
    RealSpaceFiniteDifferenceElectrostaticFeatures.
    Covers all (density_max_l, projection_max_l, normalization) combinations.
    """

    def _fd(self, dl, pl, norm="receiver", sigma_proj=None):
        sp = sigma_proj or SIGMA_PROJ
        return RealSpaceFiniteDifferenceElectrostaticFeatures(
            dl, SIGMA_SRC, pl, sp, False, norm, offset=FD_OFFSET,
        )

    def _ana(self, dl, pl, norm="receiver", sigma_proj=None):
        sp = sigma_proj or SIGMA_PROJ
        return RealSpaceAnalyticalElectrostaticFeatures(
            dl, SIGMA_SRC, pl, sp, False, norm,
        )

    def _run(self, model, sf, pos, batch):
        sf3 = sf.unsqueeze(1) if sf.dim() == 2 else sf
        out, _, _ = model(sf3, pos, batch)
        return out

    # ── shape checks ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("dl,pl", [(0,0),(1,0),(0,1),(1,1)])
    def test_output_shape(self, dl, pl):
        n = 5
        pos = _make_positions(n, seed=100)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _source_feats_l1(n) if dl == 1 else _source_feats_l0(n)
        ana = self._ana(dl, pl)
        out = self._run(ana, sf, pos, batch)
        n_rad = len(SIGMA_PROJ)
        expected = n_rad if pl == 0 else 4 * n_rad
        assert out.shape == (n, expected), f"d{dl}p{pl}: {out.shape} != ({n},{expected})"

    # ── agreement with FD ────────────────────────────────────────────────────

    @pytest.mark.parametrize("dl,pl,norm", [
        (0, 0, "receiver"),
        (1, 0, "receiver"),
        (0, 1, "receiver"),
        (1, 1, "receiver"),
        (1, 1, "multipoles"),
        (0, 1, "multipoles"),
    ])
    def test_matches_fd(self, dl, pl, norm):
        n = 5
        pos = _make_positions(n, seed=200 + dl * 10 + pl)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _source_feats_l1(n, seed=201) if dl == 1 else _source_feats_l0(n, seed=201)

        F_fd  = self._run(self._fd(dl, pl, norm), sf, pos, batch)
        F_ana = self._run(self._ana(dl, pl, norm), sf, pos, batch)
        _assert_close_fd(F_ana, F_fd, f"d{dl}p{pl} {norm}")

    def test_matches_fd_multi_radial(self):
        """Multiple projection sigmas in one call."""
        n = 5
        pos = _make_positions(n, seed=300)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _source_feats_l1(n, seed=301)
        sp = [0.4, 0.7, 1.1]

        F_fd  = self._run(self._fd(1, 1, "receiver", sp), sf, pos, batch)
        F_ana = self._run(self._ana(1, 1, "receiver", sp), sf, pos, batch)
        _assert_close_fd(F_ana, F_fd, "multi_radial d1p1")

    def test_batch(self):
        """Two graphs in a batch."""
        n0, n1 = 3, 4
        torch.manual_seed(400)
        pos = torch.rand(n0 + n1, 3) * 3.5 + 0.5
        batch = torch.cat([torch.zeros(n0, dtype=torch.long),
                           torch.ones(n1, dtype=torch.long)])
        sf = _source_feats_l1(n0 + n1, seed=401)

        F_fd  = self._run(self._fd(1, 1), sf, pos, batch)
        F_ana = self._run(self._ana(1, 1), sf, pos, batch)
        assert F_fd.shape[0] == n0 + n1
        # Slightly higher abs tolerance for batch: FD errors accumulate over more
        # atom pairs and the near-zero features can see relative errors > REL_TOL.
        abs_diff = (F_fd - F_ana).abs().max().item()
        assert abs_diff < 1e-3, f"batch d1p1: max abs diff {abs_diff:.2e}"

    # ── include_self_interaction ──────────────────────────────────────────────

    def test_include_self_interaction(self):
        n = 4
        pos = _make_positions(n, seed=500)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _source_feats_l1(n, seed=501)

        fd_si  = RealSpaceFiniteDifferenceElectrostaticFeatures(
            1, SIGMA_SRC, 1, SIGMA_PROJ, True, "receiver", offset=FD_OFFSET
        )
        ana_si = RealSpaceAnalyticalElectrostaticFeatures(
            1, SIGMA_SRC, 1, SIGMA_PROJ, True, "receiver"
        )
        F_fd  = self._run(fd_si,  sf, pos, batch)
        F_ana = self._run(ana_si, sf, pos, batch)
        _assert_close_fd(F_ana, F_fd, "include_si=True")

    def test_si_flag_changes_output(self):
        """Features with and without self-interaction must differ."""
        n = 4
        pos = _make_positions(n, seed=600)
        batch = torch.zeros(n, dtype=torch.long)
        sf = _source_feats_l1(n, seed=601)
        F_no = self._run(self._ana(1, 1, "receiver"), sf, pos, batch)
        F_si = self._run(
            RealSpaceAnalyticalElectrostaticFeatures(1,SIGMA_SRC,1,SIGMA_PROJ,True),
            sf, pos, batch
        )
        assert (F_no - F_si).abs().max().item() > 1e-6

    # ── zero-input tests ─────────────────────────────────────────────────────

    def test_zero_charges_zero_features(self):
        pos = _make_positions(4, seed=700)
        batch = torch.zeros(4, dtype=torch.long)
        sf = torch.zeros(4, 1)
        F = self._run(self._ana(0, 1), sf, pos, batch)
        assert F.abs().max().item() < 1e-12

    def test_zero_dipoles_matches_d0p1(self):
        """l=1 density with zero dipoles → same l=1 features as l=0 density."""
        pos = _make_positions(4, seed=800)
        batch = torch.zeros(4, dtype=torch.long)
        q = torch.tensor([1., -1., 0.5, -0.5])
        sf_l0 = q.unsqueeze(-1)
        sf_l1 = torch.zeros(4, 4); sf_l1[:, 0] = q

        F_d0p1 = self._run(self._ana(0, 1), sf_l0, pos, batch)
        F_d1p1 = self._run(self._ana(1, 1), sf_l1, pos, batch)
        assert (F_d0p1 - F_d1p1).abs().max().item() < 1e-10

    # ── single atom (no edges) ────────────────────────────────────────────────

    def test_single_atom_no_si_zero(self):
        """Single atom, no self-interaction → all features zero."""
        pos = torch.tensor([[0., 0., 0.]])
        batch = torch.zeros(1, dtype=torch.long)
        sf = torch.tensor([[1., 0.1, 0.2, 0.3]])
        F = self._run(self._ana(1, 1), sf, pos, batch)
        assert F.abs().max().item() < 1e-12

    # ── gradient check ────────────────────────────────────────────────────────

    def test_feature_gradient_via_autograd(self):
        """Autograd gradient of a feature scalar w.r.t. positions."""
        pos = _make_positions(3, seed=900).requires_grad_(True)
        batch = torch.zeros(3, dtype=torch.long)
        sf = _source_feats_l1(3, seed=901)
        ana = self._ana(1, 1, sigma_proj=[0.5])

        sf3 = sf.unsqueeze(1)
        out, _, _ = ana(sf3, pos, batch)
        scalar = out.sum()
        scalar.backward()
        grad_auto = pos.grad.clone()

        eps = 1e-5
        grad_fd = torch.zeros_like(pos.detach())
        with torch.no_grad():
            for i in range(3):
                for j in range(3):
                    pp = pos.detach().clone(); pp[i, j] += eps
                    pm = pos.detach().clone(); pm[i, j] -= eps
                    Fp, _, _ = ana(sf3, pp, batch)
                    Fm, _, _ = ana(sf3, pm, batch)
                    grad_fd[i, j] = (Fp.sum() - Fm.sum()) / (2 * eps)

        rel = ((grad_auto - grad_fd).abs() / grad_fd.abs().clamp(min=1e-8)).max().item()
        assert rel < 1e-6, f"Feature autograd mismatch: {rel:.2e}"

    # ── analytic spot-check ──────────────────────────────────────────────────

    def test_l0_potential_analytic_two_atoms(self):
        """
        For charge q at origin, l=0 feature at distance r equals
        l0_factor × K/(4π) × q × erf(r/(2w)) / r.
        """
        from graph_longrange.utils import FIELD_CONSTANT
        from graph_longrange.gto_utils import get_Cl_sigma
        from scipy.constants import pi

        sigma_proj = 0.5
        r = 2.0
        q = 1.0
        pos = torch.tensor([[0., 0., 0.], [r, 0., 0.]])
        batch = torch.zeros(2, dtype=torch.long)
        sf = torch.tensor([[q], [0.]])

        ana = RealSpaceAnalyticalElectrostaticFeatures(
            0, SIGMA_SRC, 0, [sigma_proj], False, "receiver"
        )
        F, _, _ = ana(sf.unsqueeze(1), pos, batch)

        w = math.sqrt((SIGMA_SRC**2 + sigma_proj**2) / 2)
        l0_fac = get_Cl_sigma(0, sigma_proj, "receiver") / get_Cl_sigma(0, sigma_proj, "multipoles")
        expected = l0_fac * FIELD_CONSTANT / (4*pi) * q * math.erf(r / (2*w)) / r

        # Feature at atom1 (receiver) from atom0 (sender)
        assert abs(F[1, 0].item() - expected) / abs(expected) < 1e-8, \
            f"l0 analytic: got {F[1,0].item():.8f}, expected {expected:.8f}"


# ── cross-check: analytical energy ↔ analytical features ─────────────────────

class TestEnergyFeatureConsistency:
    """
    Energy = 0.5 × Σᵢ qᵢ × φᵢ where φᵢ is the l=0 feature at atom i with
    projection sigma = density sigma (same-sigma projection recovers the energy).
    """

    def test_energy_from_features_l0(self):
        """
        With sigma_proj = sigma_src:
          E = 0.5 × l0_factor^{-1} × Σᵢ qᵢ × feat_l0[i]
        """
        from graph_longrange.utils import FIELD_CONSTANT
        from graph_longrange.gto_utils import get_Cl_sigma
        from scipy.constants import pi

        n = 5
        pos = _make_positions(n, seed=1000)
        batch = torch.zeros(n, dtype=torch.long)
        q = torch.tensor([1., -1., 0.5, -0.5, 0.])
        sf_l0 = q.unsqueeze(-1)

        sigma_proj = SIGMA_SRC   # same as source → potential at atom from its own GTO
        feat_model = RealSpaceAnalyticalElectrostaticFeatures(
            0, SIGMA_SRC, 0, [sigma_proj], False, "receiver"
        )
        energy_model = RealSpaceAnalyticalEnergy(0, SIGMA_SRC, False)

        F, _, _ = feat_model(sf_l0.unsqueeze(1), pos, batch)  # [n, 1] l=0 potential
        l0_fac = get_Cl_sigma(0, sigma_proj, "receiver") / get_Cl_sigma(0, sigma_proj, "multipoles")

        # Φ_i = F[i,0] / l0_fac  (raw potential in code units)
        phi = F[:, 0] / l0_fac   # [n]
        E_from_feat = 0.5 * (q * phi).sum()
        E_direct = energy_model(sf_l0, pos, batch)

        assert abs(E_from_feat.item() - E_direct.item()) / abs(E_direct.item()) < 1e-6
