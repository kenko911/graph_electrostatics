import math

import torch

from graph_longrange.realspace_electrostatics import (
    _l2_source_to_cartesian,
    multipole_energy_from_graph,
)
from graph_longrange.utils import FIELD_CONSTANT


torch.set_default_dtype(torch.float64)


def _kernel(r: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.erf(r / (2.0 * sigma)) / r


def _autodiff_pair_energy(
    R0: torch.Tensor,
    sigma: float,
    q_s: torch.Tensor,
    mu_s: torch.Tensor,
    quad_s: torch.Tensor,
    q_r: torch.Tensor,
    mu_r: torch.Tensor,
    quad_r: torch.Tensor,
) -> torch.Tensor:
    R = R0.clone().requires_grad_(True)
    r = torch.linalg.norm(R)
    T = _kernel(r, sigma)

    grad_T = torch.autograd.grad(T, R, create_graph=True)[0]
    hess_T = torch.stack(
        [torch.autograd.grad(grad_T[i], R, create_graph=True)[0] for i in range(3)]
    )

    V_source = q_s * T - torch.dot(mu_s, grad_T) + torch.sum(quad_s * hess_T)
    grad_V = torch.autograd.grad(V_source, R, create_graph=True)[0]
    hess_V = torch.stack(
        [torch.autograd.grad(grad_V[i], R, create_graph=True)[0] for i in range(3)]
    )

    return q_r * V_source + torch.dot(mu_r, grad_V) + torch.sum(quad_r * hess_V)


def _random_traceless(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    raw = torch.randn(3, 3, dtype=torch.get_default_dtype())
    sym = 0.5 * (raw + raw.T)
    return sym - torch.trace(sym) * torch.eye(3, dtype=sym.dtype) / 3.0


def test_quadrupole_energy_matches_autodiff_pair_formula():
    sigma = 0.9
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, -0.8, 0.6]],
        dtype=torch.get_default_dtype(),
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)

    source_feats = torch.tensor(
        [
            [0.7, -0.3, 0.5, 0.2, 0.1, -0.4, 0.8, 0.6, -0.2],
            [-0.2, 0.9, -0.1, 0.4, -0.5, 0.3, 0.7, -0.6, 0.2],
        ],
        dtype=torch.get_default_dtype(),
    )

    energy = multipole_energy_from_graph(
        source_feats=source_feats,
        positions=positions,
        edge_index=edge_index,
        batch=batch,
        sigma=sigma,
    )[0]

    q_s = source_feats[0, 0]
    q_r = source_feats[1, 0]
    mu_s = source_feats[0, [3, 1, 2]]
    mu_r = source_feats[1, [3, 1, 2]]
    quad_s = _l2_source_to_cartesian(source_feats[0:1, 4:9])[0]
    quad_r = _l2_source_to_cartesian(source_feats[1:2, 4:9])[0]

    pair = _autodiff_pair_energy(
        R0=positions[1] - positions[0],
        sigma=sigma,
        q_s=q_s,
        mu_s=mu_s,
        quad_s=quad_s,
        q_r=q_r,
        mu_r=mu_r,
        quad_r=quad_r,
    )
    expected = FIELD_CONSTANT / (4.0 * math.pi) * pair

    torch.testing.assert_close(energy, expected, rtol=2e-7, atol=1e-9)


def test_quadrupole_energy_reduces_to_dipole_case_when_quadrupoles_vanish():
    sigma = 1.1
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.2, -0.7]],
        dtype=torch.get_default_dtype(),
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)

    source_feats_l1 = torch.tensor(
        [
            [0.3, 0.5, -0.2, 0.1],
            [-0.4, 0.2, 0.6, -0.3],
        ],
        dtype=torch.get_default_dtype(),
    )
    source_feats_l2 = torch.cat(
        [source_feats_l1, torch.zeros(2, 5, dtype=source_feats_l1.dtype)],
        dim=-1,
    )

    energy_l1 = multipole_energy_from_graph(
        source_feats=source_feats_l1,
        positions=positions,
        edge_index=edge_index,
        batch=batch,
        sigma=sigma,
    )
    energy_l2 = multipole_energy_from_graph(
        source_feats=source_feats_l2,
        positions=positions,
        edge_index=edge_index,
        batch=batch,
        sigma=sigma,
    )

    torch.testing.assert_close(energy_l1, energy_l2, rtol=1e-9, atol=1e-11)
