# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Graph-owned optimized Warp primitives for single-system order-6, l=2 PME.

The general-purpose spread and higher-order formulas remain provided by
``nvalchemiops``.  This module owns the two MD-critical specializations so a
graph_longrange installation does not depend on an unpublished toolkit patch:

* a gather forward that omits discarded position/cell derivatives and a fused
  gather backward that produces the mesh and position adjoints in one stencil;
* a contention-free single-system CUDA cell gradient for fractionalization.

Unsupported dtype/order/rank/batch combinations continue through the toolkit.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import warp as wp

from nvalchemiops.torch._warp_op_helpers import register_warp_op_chain
from nvalchemiops.torch.interactions.electrostatics import pme_multipole as _pme


_VEC6F = wp.types.vector(length=6, dtype=wp.float32)
_REDUCED_CELL_DEVICE_TYPES = {"cuda"}


@wp.func
def _bspline6(u: wp.float32, derivative: wp.int32) -> wp.float32:
    """Stable local quintic spline and derivatives through third order."""
    zero = wp.float32(0.0)
    if u < zero or u >= wp.float32(6.0):
        return zero
    v = u
    sign = wp.float32(1.0)
    if u > wp.float32(3.0):
        v = wp.float32(6.0) - u
        if derivative == 1 or derivative == 3:
            sign = -sign
    result = zero
    if v < wp.float32(1.0):
        if derivative == 0:
            result = v * v * v * v * v / wp.float32(120.0)
        elif derivative == 1:
            result = v * v * v * v / wp.float32(24.0)
        elif derivative == 2:
            result = v * v * v / wp.float32(6.0)
        else:
            result = v * v / wp.float32(2.0)
    elif v < wp.float32(2.0):
        t = v - wp.float32(1.0)
        if derivative == 0:
            result = (
                wp.float32(1.0)
                + t
                * (
                    wp.float32(5.0)
                    + t
                    * (
                        wp.float32(10.0)
                        + t
                        * (
                            wp.float32(10.0)
                            + t * (wp.float32(5.0) - wp.float32(5.0) * t)
                        )
                    )
                )
            ) / wp.float32(120.0)
        elif derivative == 1:
            result = (
                wp.float32(5.0)
                + t
                * (
                    wp.float32(20.0)
                    + t
                    * (wp.float32(30.0) + t * (wp.float32(20.0) - wp.float32(25.0) * t))
                )
            ) / wp.float32(120.0)
        elif derivative == 2:
            result = (
                wp.float32(20.0)
                + t
                * (wp.float32(60.0) + t * (wp.float32(60.0) - wp.float32(100.0) * t))
            ) / wp.float32(120.0)
        else:
            result = (
                wp.float32(60.0) + t * (wp.float32(120.0) - wp.float32(300.0) * t)
            ) / wp.float32(120.0)
    else:
        t = v - wp.float32(2.0)
        if derivative == 0:
            result = (
                wp.float32(26.0)
                + t
                * (
                    wp.float32(50.0)
                    + t
                    * (
                        wp.float32(20.0)
                        + t
                        * (
                            -wp.float32(20.0)
                            + t * (-wp.float32(20.0) + wp.float32(10.0) * t)
                        )
                    )
                )
            ) / wp.float32(120.0)
        elif derivative == 1:
            result = (
                wp.float32(50.0)
                + t
                * (
                    wp.float32(40.0)
                    + t
                    * (
                        -wp.float32(60.0)
                        + t * (-wp.float32(80.0) + wp.float32(50.0) * t)
                    )
                )
            ) / wp.float32(120.0)
        elif derivative == 2:
            result = (
                wp.float32(40.0)
                + t
                * (
                    -wp.float32(120.0)
                    + t * (-wp.float32(240.0) + wp.float32(200.0) * t)
                )
            ) / wp.float32(120.0)
        else:
            result = (
                -wp.float32(120.0) + t * (-wp.float32(480.0) + wp.float32(600.0) * t)
            ) / wp.float32(120.0)
    return sign * result


@wp.func
def _wrap(index: wp.int32, extent: wp.int32) -> wp.int32:
    return (index % extent + extent) % extent


@wp.kernel(enable_backward=False)
def _gather_l2_order6_f32(
    phi: wp.array3d(dtype=wp.float32),
    positions: wp.array(dtype=wp.vec3f),
    out_q: wp.array(dtype=wp.float32),
    out_d: wp.array(dtype=wp.vec3f),
    out_Q: wp.array(dtype=wp.mat33f),
):
    atom = wp.tid()
    pos = positions[atom]
    nx = wp.int32(phi.shape[0])
    ny = wp.int32(phi.shape[1])
    nz = wp.int32(phi.shape[2])
    sx = pos[0] * wp.float32(nx)
    sy = pos[1] * wp.float32(ny)
    sz = pos[2] * wp.float32(nz)
    bx = wp.int32(wp.floor(sx))
    by = wp.int32(wp.floor(sy))
    bz = wp.int32(wp.floor(sz))
    tx = sx - wp.float32(bx)
    ty = sy - wp.float32(by)
    tz = sz - wp.float32(bz)
    ox = wp.int32(wp.floor(tx - wp.float32(2.0)))
    oy = wp.int32(wp.floor(ty - wp.float32(2.0)))
    oz = wp.int32(wp.floor(tz - wp.float32(2.0)))
    wx = _VEC6F()
    wy = _VEC6F()
    wz = _VEC6F()
    dx = _VEC6F()
    dy = _VEC6F()
    dz = _VEC6F()
    d2x = _VEC6F()
    d2y = _VEC6F()
    d2z = _VEC6F()
    for a in range(6):
        ux = wp.float32(3.0) + tx - wp.float32(ox + a)
        uy = wp.float32(3.0) + ty - wp.float32(oy + a)
        uz = wp.float32(3.0) + tz - wp.float32(oz + a)
        wx[a] = _bspline6(ux, 0)
        wy[a] = _bspline6(uy, 0)
        wz[a] = _bspline6(uz, 0)
        dx[a] = _bspline6(ux, 1) * wp.float32(nx)
        dy[a] = _bspline6(uy, 1) * wp.float32(ny)
        dz[a] = _bspline6(uz, 1) * wp.float32(nz)
        d2x[a] = _bspline6(ux, 2) * wp.float32(nx * nx)
        d2y[a] = _bspline6(uy, 2) * wp.float32(ny * ny)
        d2z[a] = _bspline6(uz, 2) * wp.float32(nz * nz)
    aq = wp.float32(0.0)
    ax = wp.float32(0.0)
    ay = wp.float32(0.0)
    az = wp.float32(0.0)
    hxx = wp.float32(0.0)
    hyy = wp.float32(0.0)
    hzz = wp.float32(0.0)
    hxy = wp.float32(0.0)
    hxz = wp.float32(0.0)
    hyz = wp.float32(0.0)
    for i in range(6):
        gx = _wrap(bx + ox + i, nx)
        for j in range(6):
            gy = _wrap(by + oy + j, ny)
            for k in range(6):
                gz = _wrap(bz + oz + k, nz)
                value = phi[gx, gy, gz]
                w = wx[i] * wy[j] * wz[k]
                aq += value * w
                ax += value * dx[i] * wy[j] * wz[k]
                ay += value * wx[i] * dy[j] * wz[k]
                az += value * wx[i] * wy[j] * dz[k]
                hxx += value * d2x[i] * wy[j] * wz[k]
                hyy += value * wx[i] * d2y[j] * wz[k]
                hzz += value * wx[i] * wy[j] * d2z[k]
                hxy += value * dx[i] * dy[j] * wz[k]
                hxz += value * dx[i] * wy[j] * dz[k]
                hyz += value * wx[i] * dy[j] * dz[k]
    out_q[atom] = aq
    out_d[atom] = wp.vec3f(ax, ay, az)
    half = wp.float32(0.5)
    out_Q[atom] = wp.mat33f(
        half * hxx,
        half * hxy,
        half * hxz,
        half * hxy,
        half * hyy,
        half * hyz,
        half * hxz,
        half * hyz,
        half * hzz,
    )


@wp.kernel(enable_backward=False)
def _gather_l2_order6_f32_backward(
    phi: wp.array3d(dtype=wp.float32),
    positions: wp.array(dtype=wp.vec3f),
    cq: wp.array(dtype=wp.float32),
    cd: wp.array(dtype=wp.vec3f),
    cQ: wp.array(dtype=wp.mat33f),
    d_phi: wp.array3d(dtype=wp.float32),
    d_positions: wp.array(dtype=wp.vec3f),
):
    atom = wp.tid()
    pos = positions[atom]
    nx = wp.int32(phi.shape[0])
    ny = wp.int32(phi.shape[1])
    nz = wp.int32(phi.shape[2])
    sx = pos[0] * wp.float32(nx)
    sy = pos[1] * wp.float32(ny)
    sz = pos[2] * wp.float32(nz)
    bx = wp.int32(wp.floor(sx))
    by = wp.int32(wp.floor(sy))
    bz = wp.int32(wp.floor(sz))
    tx = sx - wp.float32(bx)
    ty = sy - wp.float32(by)
    tz = sz - wp.float32(bz)
    ox = wp.int32(wp.floor(tx - wp.float32(2.0)))
    oy = wp.int32(wp.floor(ty - wp.float32(2.0)))
    oz = wp.int32(wp.floor(tz - wp.float32(2.0)))
    wx = _VEC6F()
    wy = _VEC6F()
    wz = _VEC6F()
    dx = _VEC6F()
    dy = _VEC6F()
    dz = _VEC6F()
    d2x = _VEC6F()
    d2y = _VEC6F()
    d2z = _VEC6F()
    d3x = _VEC6F()
    d3y = _VEC6F()
    d3z = _VEC6F()
    for a in range(6):
        ux = wp.float32(3.0) + tx - wp.float32(ox + a)
        uy = wp.float32(3.0) + ty - wp.float32(oy + a)
        uz = wp.float32(3.0) + tz - wp.float32(oz + a)
        wx[a] = _bspline6(ux, 0)
        wy[a] = _bspline6(uy, 0)
        wz[a] = _bspline6(uz, 0)
        dx[a] = _bspline6(ux, 1) * wp.float32(nx)
        dy[a] = _bspline6(uy, 1) * wp.float32(ny)
        dz[a] = _bspline6(uz, 1) * wp.float32(nz)
        d2x[a] = _bspline6(ux, 2) * wp.float32(nx * nx)
        d2y[a] = _bspline6(uy, 2) * wp.float32(ny * ny)
        d2z[a] = _bspline6(uz, 2) * wp.float32(nz * nz)
        d3x[a] = _bspline6(ux, 3) * wp.float32(nx * nx * nx)
        d3y[a] = _bspline6(uy, 3) * wp.float32(ny * ny * ny)
        d3z[a] = _bspline6(uz, 3) * wp.float32(nz * nz * nz)
    acc_fx = wp.float32(0.0)
    acc_fy = wp.float32(0.0)
    acc_fz = wp.float32(0.0)
    acc_hxx = wp.float32(0.0)
    acc_hyy = wp.float32(0.0)
    acc_hzz = wp.float32(0.0)
    acc_hxy = wp.float32(0.0)
    acc_hxz = wp.float32(0.0)
    acc_hyz = wp.float32(0.0)
    acc_txxx = wp.float32(0.0)
    acc_tyyy = wp.float32(0.0)
    acc_tzzz = wp.float32(0.0)
    acc_txxy = wp.float32(0.0)
    acc_txxz = wp.float32(0.0)
    acc_tyyx = wp.float32(0.0)
    acc_tyyz = wp.float32(0.0)
    acc_tzzx = wp.float32(0.0)
    acc_tzzy = wp.float32(0.0)
    acc_txyz = wp.float32(0.0)
    source_q = cq[atom]
    source_d = cd[atom]
    source_Q = cQ[atom]
    for i in range(6):
        gx = _wrap(bx + ox + i, nx)
        for j in range(6):
            gy = _wrap(by + oy + j, ny)
            for k in range(6):
                gz = _wrap(bz + oz + k, nz)
                value = phi[gx, gy, gz]
                w = wx[i] * wy[j] * wz[k]
                gxw = dx[i] * wy[j] * wz[k]
                gyw = wx[i] * dy[j] * wz[k]
                gzw = wx[i] * wy[j] * dz[k]
                hxx = d2x[i] * wy[j] * wz[k]
                hyy = wx[i] * d2y[j] * wz[k]
                hzz = wx[i] * wy[j] * d2z[k]
                hxy = dx[i] * dy[j] * wz[k]
                hxz = dx[i] * wy[j] * dz[k]
                hyz = wx[i] * dy[j] * dz[k]
                contribution = (
                    source_q * w
                    + source_d[0] * gxw
                    + source_d[1] * gyw
                    + source_d[2] * gzw
                    + wp.float32(0.5)
                    * (
                        source_Q[0, 0] * hxx
                        + source_Q[1, 1] * hyy
                        + source_Q[2, 2] * hzz
                        + wp.float32(2.0)
                        * (
                            source_Q[0, 1] * hxy
                            + source_Q[0, 2] * hxz
                            + source_Q[1, 2] * hyz
                        )
                    )
                )
                wp.atomic_add(d_phi, gx, gy, gz, contribution)
                txxx = d3x[i] * wy[j] * wz[k]
                tyyy = wx[i] * d3y[j] * wz[k]
                tzzz = wx[i] * wy[j] * d3z[k]
                txxy = d2x[i] * dy[j] * wz[k]
                txxz = d2x[i] * wy[j] * dz[k]
                tyyx = dx[i] * d2y[j] * wz[k]
                tyyz = wx[i] * d2y[j] * dz[k]
                tzzx = dx[i] * wy[j] * d2z[k]
                tzzy = wx[i] * dy[j] * d2z[k]
                txyz = dx[i] * dy[j] * dz[k]
                acc_fx += value * gxw
                acc_fy += value * gyw
                acc_fz += value * gzw
                acc_hxx += value * hxx
                acc_hyy += value * hyy
                acc_hzz += value * hzz
                acc_hxy += value * hxy
                acc_hxz += value * hxz
                acc_hyz += value * hyz
                acc_txxx += value * txxx
                acc_tyyy += value * tyyy
                acc_tzzz += value * tzzz
                acc_txxy += value * txxy
                acc_txxz += value * txxz
                acc_tyyx += value * tyyx
                acc_tyyz += value * tyyz
                acc_tzzx += value * tzzx
                acc_tzzy += value * tzzy
                acc_txyz += value * txyz
    two = wp.float32(2.0)
    half = wp.float32(0.5)
    qfx = (
        source_Q[0, 0] * acc_txxx
        + source_Q[1, 1] * acc_tyyx
        + source_Q[2, 2] * acc_tzzx
        + two
        * (
            source_Q[0, 1] * acc_txxy
            + source_Q[0, 2] * acc_txxz
            + source_Q[1, 2] * acc_txyz
        )
    )
    qfy = (
        source_Q[0, 0] * acc_txxy
        + source_Q[1, 1] * acc_tyyy
        + source_Q[2, 2] * acc_tzzy
        + two
        * (
            source_Q[0, 1] * acc_tyyx
            + source_Q[0, 2] * acc_txyz
            + source_Q[1, 2] * acc_tyyz
        )
    )
    qfz = (
        source_Q[0, 0] * acc_txxz
        + source_Q[1, 1] * acc_tyyz
        + source_Q[2, 2] * acc_tzzz
        + two
        * (
            source_Q[0, 1] * acc_txyz
            + source_Q[0, 2] * acc_tzzx
            + source_Q[1, 2] * acc_tzzy
        )
    )
    fx = (
        source_q * acc_fx
        + source_d[0] * acc_hxx
        + source_d[1] * acc_hxy
        + source_d[2] * acc_hxz
        + half * qfx
    )
    fy = (
        source_q * acc_fy
        + source_d[0] * acc_hxy
        + source_d[1] * acc_hyy
        + source_d[2] * acc_hyz
        + half * qfy
    )
    fz = (
        source_q * acc_fz
        + source_d[0] * acc_hxz
        + source_d[1] * acc_hyz
        + source_d[2] * acc_hzz
        + half * qfz
    )
    d_positions[atom] = wp.vec3f(fx, fy, fz)


@wp.kernel(enable_backward=False)
def _fractionalize_backward_per_atom_cell_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell_inv_t: wp.array(dtype=wp.mat33f),
    dipoles: wp.array(dtype=wp.vec3f),
    quadrupoles: wp.array(dtype=wp.mat33f),
    gp: wp.array(dtype=wp.vec3f),
    gdf: wp.array(dtype=wp.vec3f),
    gQf: wp.array(dtype=wp.mat33f),
    grad_positions: wp.array(dtype=wp.vec3f),
    grad_cell_per_atom: wp.array(dtype=wp.mat33f),
    grad_dipoles: wp.array(dtype=wp.vec3f),
    grad_quadrupoles: wp.array(dtype=wp.mat33f),
):
    i = wp.tid()
    m = cell_inv_t[0]
    mt = wp.transpose(m)
    grad_positions[i] = mt * gp[i]
    grad_dipoles[i] = mt * gdf[i]
    grad_quadrupoles[i] = mt * gQf[i] * m
    grad_cell_per_atom[i] = (
        wp.outer(gp[i], positions[i])
        + wp.outer(gdf[i], dipoles[i])
        + gQf[i] * m * wp.transpose(quadrupoles[i])
        + wp.transpose(gQf[i]) * m * quadrupoles[i]
    )


def _stream(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))


def _wpt(tensor: torch.Tensor, dtype):
    return wp.from_torch(tensor.detach(), dtype=dtype, requires_grad=False)


def _gather_forward(
    phi: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = positions.shape[0]
    out_q = torch.empty(n, dtype=phi.dtype, device=phi.device)
    out_d = torch.empty((n, 3), dtype=phi.dtype, device=phi.device)
    out_Q = torch.empty((n, 3, 3), dtype=phi.dtype, device=phi.device)
    with _stream(phi.device):
        wp.launch(
            _gather_l2_order6_f32,
            dim=n,
            inputs=[
                _wpt(phi.contiguous(), wp.float32),
                _wpt(positions.contiguous(), wp.vec3f),
            ],
            outputs=[
                _wpt(out_q, wp.float32),
                _wpt(out_d, wp.vec3f),
                _wpt(out_Q, wp.mat33f),
            ],
            device=str(wp.device_from_torch(phi.device)),
        )
    return out_q, out_d, out_Q


def _gather_forward_fake(phi, positions):  # pragma: no cover
    n = positions.shape[0]
    return (
        torch.empty(n, dtype=phi.dtype, device=phi.device),
        torch.empty((n, 3), dtype=phi.dtype, device=phi.device),
        torch.empty((n, 3, 3), dtype=phi.dtype, device=phi.device),
    )


def _gather_backward(cq, cd, cQ, phi, positions):  # pragma: no cover
    cQ = 0.5 * (cQ + cQ.transpose(-1, -2))
    d_phi = torch.zeros_like(phi)
    d_positions = torch.empty_like(positions)
    with _stream(phi.device):
        wp.launch(
            _gather_l2_order6_f32_backward,
            dim=positions.shape[0],
            inputs=[
                _wpt(phi.contiguous(), wp.float32),
                _wpt(positions.contiguous(), wp.vec3f),
                _wpt(cq.contiguous(), wp.float32),
                _wpt(cd.contiguous(), wp.vec3f),
                _wpt(cQ.contiguous(), wp.mat33f),
            ],
            outputs=[_wpt(d_phi, wp.float32), _wpt(d_positions, wp.vec3f)],
            device=str(wp.device_from_torch(phi.device)),
        )
    return d_phi, d_positions


def _gather_backward_fake(cq, cd, cQ, phi, positions):  # pragma: no cover
    del cq, cd, cQ
    return torch.empty_like(phi), torch.empty_like(positions)


def _gather_double_backward(
    gg_phi, gg_positions, cq, cd, cQ, phi, positions
):  # pragma: no cover
    identity = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    nx, ny, nz = phi.shape
    return _pme._multipole_pme_gather_via_spread_t_double_backward(
        gg_phi,
        gg_positions,
        cq,
        cd,
        cQ,
        phi,
        positions,
        identity,
        nx,
        ny,
        nz,
        6,
        2,
    )


def _gather_double_backward_fake(
    gg_phi, gg_positions, cq, cd, cQ, phi, positions
):  # pragma: no cover
    del gg_phi, gg_positions
    return tuple(torch.empty_like(value) for value in (cq, cd, cQ, phi, positions))


register_warp_op_chain(
    name="graph_longrange::pme_gather_l2_order6_f32",
    forward=_gather_forward,
    forward_fake=_gather_forward_fake,
    forward_return_arity=3,
    backward=_gather_backward,
    backward_fake=_gather_backward_fake,
    backward_return_arity=2,
    diff_input_positions=(0, 1),
    n_forward_inputs=2,
    double_backward=_gather_double_backward,
    double_backward_fake=_gather_double_backward_fake,
    double_backward_return_arity=5,
    second_order_diff_positions=(0, 1, 2, 3, 4),
    n_backward_inputs=5,
    forward_schema="(Tensor phi, Tensor positions) -> (Tensor, Tensor, Tensor)",
    backward_schema=(
        "(Tensor cq, Tensor cd, Tensor cQ, Tensor phi, Tensor positions) "
        "-> (Tensor, Tensor)"
    ),
    double_backward_schema=(
        "(Tensor? gg_phi, Tensor? gg_positions, Tensor cq, Tensor cd, Tensor cQ, "
        "Tensor phi, Tensor positions) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
)


def _fractionalize_forward(positions, cell_inv_t, dipoles, quadrupoles, batch_idx):
    return _pme._pme_fractionalize_forward(
        positions, cell_inv_t, dipoles, quadrupoles, batch_idx
    )


def _fractionalize_forward_fake(
    positions, cell_inv_t, dipoles, quadrupoles, batch_idx
):  # pragma: no cover
    del cell_inv_t, batch_idx
    return (
        torch.empty_like(positions),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


def _fractionalize_backward(
    gp, gdf, gQf, positions, cell_inv_t, dipoles, quadrupoles, batch_idx
):  # pragma: no cover
    if (
        positions.dtype != torch.float32
        or cell_inv_t.shape[0] != 1
        or positions.device.type not in _REDUCED_CELL_DEVICE_TYPES
    ):
        return _pme._pme_fractionalize_backward(
            gp, gdf, gQf, positions, cell_inv_t, dipoles, quadrupoles, batch_idx
        )
    n = positions.shape[0]
    grad_positions = torch.empty_like(positions)
    grad_cell_atoms = torch.empty(
        (n, 3, 3), dtype=positions.dtype, device=positions.device
    )
    grad_dipoles = torch.empty_like(dipoles)
    grad_quadrupoles = torch.empty_like(quadrupoles)
    with _stream(positions.device):
        wp.launch(
            _fractionalize_backward_per_atom_cell_f32,
            dim=n,
            inputs=[
                _wpt(positions.contiguous(), wp.vec3f),
                _wpt(cell_inv_t.contiguous(), wp.mat33f),
                _wpt(dipoles.contiguous(), wp.vec3f),
                _wpt(quadrupoles.contiguous(), wp.mat33f),
                _wpt(gp.contiguous(), wp.vec3f),
                _wpt(gdf.contiguous(), wp.vec3f),
                _wpt(gQf.contiguous(), wp.mat33f),
            ],
            outputs=[
                _wpt(grad_positions, wp.vec3f),
                _wpt(grad_cell_atoms, wp.mat33f),
                _wpt(grad_dipoles, wp.vec3f),
                _wpt(grad_quadrupoles, wp.mat33f),
            ],
            device=str(wp.device_from_torch(positions.device)),
        )
    return (
        grad_positions,
        grad_cell_atoms.sum(dim=0, keepdim=True),
        grad_dipoles,
        grad_quadrupoles,
    )


def _fractionalize_backward_fake(
    gp, gdf, gQf, positions, cell_inv_t, dipoles, quadrupoles, batch_idx
):  # pragma: no cover
    del gp, gdf, gQf, batch_idx
    return tuple(
        torch.empty_like(value)
        for value in (positions, cell_inv_t, dipoles, quadrupoles)
    )


def _fractionalize_double_backward(
    g_pos,
    g_cell,
    g_dip,
    g_quad,
    gp,
    gdf,
    gQf,
    positions,
    cell_inv_t,
    dipoles,
    quadrupoles,
    batch_idx,
):  # pragma: no cover
    return _pme._pme_fractionalize_double_backward(
        g_pos,
        g_cell,
        g_dip,
        g_quad,
        gp,
        gdf,
        gQf,
        positions,
        cell_inv_t,
        dipoles,
        quadrupoles,
        batch_idx,
    )


register_warp_op_chain(
    name="graph_longrange::pme_fractionalize",
    forward=_fractionalize_forward,
    forward_fake=_fractionalize_forward_fake,
    forward_return_arity=3,
    backward=_fractionalize_backward,
    backward_fake=_fractionalize_backward_fake,
    backward_return_arity=4,
    diff_input_positions=(0, 1, 2, 3),
    n_forward_inputs=5,
    double_backward=_fractionalize_double_backward,
    second_order_diff_positions=(0, 1, 2, 3, 4, 5, 6),
    n_backward_inputs=8,
    double_backward_return_arity=7,
    forward_schema=(
        "(Tensor positions, Tensor cell_inv_t, Tensor dipoles, Tensor quadrupoles, "
        "Tensor batch_idx) -> (Tensor, Tensor, Tensor)"
    ),
    backward_schema=(
        "(Tensor gp, Tensor gdf, Tensor gQf, Tensor positions, Tensor cell_inv_t, "
        "Tensor dipoles, Tensor quadrupoles, Tensor batch_idx) "
        "-> (Tensor, Tensor, Tensor, Tensor)"
    ),
    double_backward_schema=(
        "(Tensor? g_pos, Tensor? g_cell, Tensor? g_dip, Tensor? g_quad, Tensor gp, "
        "Tensor gdf, Tensor gQf, Tensor positions, Tensor cell_inv_t, Tensor dipoles, "
        "Tensor quadrupoles, Tensor batch_idx) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
)


def fractionalize(positions, cell_inv_t, dipoles, quadrupoles, batch_idx):
    """Graph-owned fractionalization with reduced single-system cell gradient."""
    return torch.ops.graph_longrange.pme_fractionalize(
        positions, cell_inv_t, dipoles, quadrupoles, batch_idx
    )


def gather_l2_order6_f32(phi, positions):
    """Graph-owned order-6 l=2 gather with fused analytical backward."""
    return torch.ops.graph_longrange.pme_gather_l2_order6_f32(phi, positions)
