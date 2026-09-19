# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Graph-owned optimized Warp primitives for single-system order-6, l=2 PME.

This module owns the MD-critical float32 specialization so graph_longrange does
not depend on an unpublished toolkit patch:

* stable order-6 spread and its analytical backward;
* a gather forward that omits discarded position/cell derivatives and a fused
  gather backward that produces the mesh and position adjoints in one stencil;
* a contention-free single-system CUDA cell gradient for fractionalization.

The rarely used double-backward recomputes the same stable stencil with Torch;
unsupported dtype/order/rank/batch combinations continue through the toolkit.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import warp as wp

from nvalchemiops.torch._warp_op_helpers import register_warp_op_chain
from nvalchemiops.torch.interactions.electrostatics import pme_multipole as _pme

from .pme import _bspline, _bspline_double_prime, _bspline_prime


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
                    + t * (wp.float32(10.0) + t * (wp.float32(10.0) + t * (wp.float32(5.0) - wp.float32(5.0) * t)))
                )
            ) / wp.float32(120.0)
        elif derivative == 1:
            result = (
                wp.float32(5.0)
                + t * (wp.float32(20.0) + t * (wp.float32(30.0) + t * (wp.float32(20.0) - wp.float32(25.0) * t)))
            ) / wp.float32(120.0)
        elif derivative == 2:
            result = (
                wp.float32(20.0) + t * (wp.float32(60.0) + t * (wp.float32(60.0) - wp.float32(100.0) * t))
            ) / wp.float32(120.0)
        else:
            result = (wp.float32(60.0) + t * (wp.float32(120.0) - wp.float32(300.0) * t)) / wp.float32(120.0)
    else:
        t = v - wp.float32(2.0)
        if derivative == 0:
            result = (
                wp.float32(26.0)
                + t
                * (
                    wp.float32(50.0)
                    + t * (wp.float32(20.0) + t * (-wp.float32(20.0) + t * (-wp.float32(20.0) + wp.float32(10.0) * t)))
                )
            ) / wp.float32(120.0)
        elif derivative == 1:
            result = (
                wp.float32(50.0)
                + t * (wp.float32(40.0) + t * (-wp.float32(60.0) + t * (-wp.float32(80.0) + wp.float32(50.0) * t)))
            ) / wp.float32(120.0)
        elif derivative == 2:
            result = (
                wp.float32(40.0) + t * (-wp.float32(120.0) + t * (-wp.float32(240.0) + wp.float32(200.0) * t))
            ) / wp.float32(120.0)
        else:
            result = (-wp.float32(120.0) + t * (-wp.float32(480.0) + wp.float32(600.0) * t)) / wp.float32(120.0)
    return sign * result


@wp.func
def _wrap(index: wp.int32, extent: wp.int32) -> wp.int32:
    return (index % extent + extent) % extent


@wp.kernel(enable_backward=False)
def _spread_l2_order6_f32(
    positions: wp.array(dtype=wp.vec3f),
    charges: wp.array(dtype=wp.float32),
    dipoles: wp.array(dtype=wp.vec3f),
    quadrupoles: wp.array(dtype=wp.mat33f),
    mesh: wp.array3d(dtype=wp.float32),
):
    atom = wp.tid()
    pos = positions[atom]
    nx = wp.int32(mesh.shape[0])
    ny = wp.int32(mesh.shape[1])
    nz = wp.int32(mesh.shape[2])
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
    charge = charges[atom]
    dipole = dipoles[atom]
    quad = quadrupoles[atom]
    half = wp.float32(0.5)
    two = wp.float32(2.0)
    for i in range(6):
        gx = _wrap(bx + ox + i, nx)
        for j in range(6):
            gy = _wrap(by + oy + j, ny)
            for k in range(6):
                gz = _wrap(bz + oz + k, nz)
                w = wx[i] * wy[j] * wz[k]
                grad_x = dx[i] * wy[j] * wz[k]
                grad_y = wx[i] * dy[j] * wz[k]
                grad_z = wx[i] * wy[j] * dz[k]
                hxx = d2x[i] * wy[j] * wz[k]
                hyy = wx[i] * d2y[j] * wz[k]
                hzz = wx[i] * wy[j] * d2z[k]
                hxy = dx[i] * dy[j] * wz[k]
                hxz = dx[i] * wy[j] * dz[k]
                hyz = wx[i] * dy[j] * dz[k]
                contribution = (
                    charge * w
                    + dipole[0] * grad_x
                    + dipole[1] * grad_y
                    + dipole[2] * grad_z
                    + half
                    * (
                        quad[0, 0] * hxx
                        + quad[1, 1] * hyy
                        + quad[2, 2] * hzz
                        + two * (quad[0, 1] * hxy + quad[0, 2] * hxz + quad[1, 2] * hyz)
                    )
                )
                wp.atomic_add(mesh, gx, gy, gz, contribution)


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
    write_mesh: wp.int32,
    d_phi: wp.array3d(dtype=wp.float32),
    d_positions: wp.array(dtype=wp.vec3f),
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
    acc_q = wp.float32(0.0)
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
                        + wp.float32(2.0) * (source_Q[0, 1] * hxy + source_Q[0, 2] * hxz + source_Q[1, 2] * hyz)
                    )
                )
                if write_mesh != 0:
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
                acc_q += value * w
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
        + two * (source_Q[0, 1] * acc_txxy + source_Q[0, 2] * acc_txxz + source_Q[1, 2] * acc_txyz)
    )
    qfy = (
        source_Q[0, 0] * acc_txxy
        + source_Q[1, 1] * acc_tyyy
        + source_Q[2, 2] * acc_tzzy
        + two * (source_Q[0, 1] * acc_tyyx + source_Q[0, 2] * acc_txyz + source_Q[1, 2] * acc_tyyz)
    )
    qfz = (
        source_Q[0, 0] * acc_txxz
        + source_Q[1, 1] * acc_tyyz
        + source_Q[2, 2] * acc_tzzz
        + two * (source_Q[0, 1] * acc_txyz + source_Q[0, 2] * acc_tzzx + source_Q[1, 2] * acc_tzzy)
    )
    fx = source_q * acc_fx + source_d[0] * acc_hxx + source_d[1] * acc_hxy + source_d[2] * acc_hxz + half * qfx
    fy = source_q * acc_fy + source_d[0] * acc_hxy + source_d[1] * acc_hyy + source_d[2] * acc_hyz + half * qfy
    fz = source_q * acc_fz + source_d[0] * acc_hxz + source_d[1] * acc_hyz + source_d[2] * acc_hzz + half * qfz
    d_positions[atom] = wp.vec3f(fx, fy, fz)
    out_q[atom] = acc_q
    out_d[atom] = wp.vec3f(acc_fx, acc_fy, acc_fz)
    out_Q[atom] = wp.mat33f(
        half * acc_hxx,
        half * acc_hxy,
        half * acc_hxz,
        half * acc_hxy,
        half * acc_hyy,
        half * acc_hyz,
        half * acc_hxz,
        half * acc_hyz,
        half * acc_hzz,
    )


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


def _stable_stencil_torch(positions: torch.Tensor, shape: tuple[int, int, int]):
    """Build the stable order-6 identity-cell stencil for double backward."""
    dims = positions.new_tensor(shape)
    scaled = positions * dims
    base = torch.floor(scaled).to(torch.int64)
    frac = scaled - base.to(positions.dtype)
    offset = torch.floor(frac - 2.0).to(torch.int64)
    steps = torch.arange(6, dtype=torch.int64, device=positions.device)
    u = 3.0 + frac[:, None, :] - (offset[:, None, :] + steps[None, :, None]).to(positions.dtype)
    weights = _bspline(u)
    first = _bspline_prime(u) * dims
    second = _bspline_double_prime(u) * dims.square()

    ix = (base[:, 0, None] + offset[:, 0, None] + steps) % shape[0]
    iy = (base[:, 1, None] + offset[:, 1, None] + steps) % shape[1]
    iz = (base[:, 2, None] + offset[:, 2, None] + steps) % shape[2]
    indices = (
        ix[:, :, None, None].expand(-1, 6, 6, 6).reshape(-1),
        iy[:, None, :, None].expand(-1, 6, 6, 6).reshape(-1),
        iz[:, None, None, :].expand(-1, 6, 6, 6).reshape(-1),
    )
    return indices, weights, first, second


def _stable_stencil_products(positions: torch.Tensor, shape: tuple[int, int, int]):
    indices, weights, first, second = _stable_stencil_torch(positions, shape)
    wx, wy, wz = (weights[:, :, axis] for axis in range(3))
    dx, dy, dz = (first[:, :, axis] for axis in range(3))
    d2x, d2y, d2z = (second[:, :, axis] for axis in range(3))
    w = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    gx = dx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    gy = wx[:, :, None, None] * dy[:, None, :, None] * wz[:, None, None, :]
    gz = wx[:, :, None, None] * wy[:, None, :, None] * dz[:, None, None, :]
    hxx = d2x[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    hyy = wx[:, :, None, None] * d2y[:, None, :, None] * wz[:, None, None, :]
    hzz = wx[:, :, None, None] * wy[:, None, :, None] * d2z[:, None, None, :]
    hxy = dx[:, :, None, None] * dy[:, None, :, None] * wz[:, None, None, :]
    hxz = dx[:, :, None, None] * wy[:, None, :, None] * dz[:, None, None, :]
    hyz = wx[:, :, None, None] * dy[:, None, :, None] * dz[:, None, None, :]
    return indices, (w, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz)


def _stable_spread_torch(positions, charges, dipoles, quadrupoles, shape):
    indices, products = _stable_stencil_products(positions, shape)
    w, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz = products
    quad = 0.5 * (quadrupoles + quadrupoles.transpose(-1, -2))
    contribution = (
        charges[:, None, None, None] * w
        + dipoles[:, 0, None, None, None] * gx
        + dipoles[:, 1, None, None, None] * gy
        + dipoles[:, 2, None, None, None] * gz
        + 0.5
        * (
            quad[:, 0, 0, None, None, None] * hxx
            + quad[:, 1, 1, None, None, None] * hyy
            + quad[:, 2, 2, None, None, None] * hzz
            + 2.0
            * (
                quad[:, 0, 1, None, None, None] * hxy
                + quad[:, 0, 2, None, None, None] * hxz
                + quad[:, 1, 2, None, None, None] * hyz
            )
        )
    )
    mesh = positions.new_zeros(shape)
    return mesh.index_put(indices, contribution.reshape(-1), accumulate=True)


def _stable_gather_torch(phi, positions):
    indices, products = _stable_stencil_products(positions, tuple(phi.shape))
    w, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz = products
    values = phi[indices].reshape(positions.shape[0], 6, 6, 6)
    q = (values * w).sum((1, 2, 3))
    d = torch.stack(
        tuple((values * component).sum((1, 2, 3)) for component in (gx, gy, gz)),
        dim=-1,
    )
    half = 0.5
    Q = torch.stack(
        (
            half * (values * hxx).sum((1, 2, 3)),
            half * (values * hxy).sum((1, 2, 3)),
            half * (values * hxz).sum((1, 2, 3)),
            half * (values * hxy).sum((1, 2, 3)),
            half * (values * hyy).sum((1, 2, 3)),
            half * (values * hyz).sum((1, 2, 3)),
            half * (values * hxz).sum((1, 2, 3)),
            half * (values * hyz).sum((1, 2, 3)),
            half * (values * hzz).sum((1, 2, 3)),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    return q, d, Q


def _optional_cotangent(value, like):
    return torch.zeros_like(like) if value is None else value


def _bspline_third_torch(u):
    reflected = u > 3.0
    v = torch.where(reflected, 6.0 - u, u)
    value = torch.where(
        v < 1.0,
        0.5 * v.square(),
        torch.where(
            v < 2.0,
            0.5 + (v - 1.0) - 2.5 * (v - 1.0).square(),
            -1.0 - 4.0 * (v - 2.0) + 5.0 * (v - 2.0).square(),
        ),
    )
    value = torch.where(reflected, -value, value)
    return torch.where((u >= 0.0) & (u < 6.0), value, torch.zeros_like(value))


def _bspline_fourth_torch(u):
    v = torch.where(u > 3.0, 6.0 - u, u)
    value = torch.where(
        v < 1.0,
        v,
        torch.where(v < 2.0, 1.0 - 5.0 * (v - 1.0), -4.0 + 10.0 * (v - 2.0)),
    )
    return torch.where((u >= 0.0) & (u < 6.0), value, torch.zeros_like(value))


def _stable_high_stencil_torch(positions, shape):
    dims = positions.new_tensor(shape)
    scaled = positions * dims
    base = torch.floor(scaled).to(torch.int64)
    frac = scaled - base.to(positions.dtype)
    offset = torch.floor(frac - 2.0).to(torch.int64)
    steps = torch.arange(6, dtype=torch.int64, device=positions.device)
    u = 3.0 + frac[:, None, :] - (offset[:, None, :] + steps[None, :, None]).to(positions.dtype)
    derivatives = (
        _bspline(u),
        _bspline_prime(u) * dims,
        _bspline_double_prime(u) * dims.square(),
        _bspline_third_torch(u) * dims.pow(3),
        _bspline_fourth_torch(u) * dims.pow(4),
    )
    ix = (base[:, 0, None] + offset[:, 0, None] + steps) % shape[0]
    iy = (base[:, 1, None] + offset[:, 1, None] + steps) % shape[1]
    iz = (base[:, 2, None] + offset[:, 2, None] + steps) % shape[2]
    indices = (
        ix[:, :, None, None].expand(-1, 6, 6, 6).reshape(-1),
        iy[:, None, :, None].expand(-1, 6, 6, 6).reshape(-1),
        iz[:, None, None, :].expand(-1, 6, 6, 6).reshape(-1),
    )
    return indices, derivatives


def _octupole_double_terms_torch(positions, quadrupoles, gg_positions, grad_mesh):
    shape = tuple(grad_mesh.shape)
    indices, derivatives = _stable_high_stencil_torch(positions, shape)
    w, d1, d2, d3, d4 = derivatives
    wx, wy, wz = (w[:, :, axis] for axis in range(3))
    dx, dy, dz = (d1[:, :, axis] for axis in range(3))
    d2x, d2y, d2z = (d2[:, :, axis] for axis in range(3))
    d3x, d3y, d3z = (d3[:, :, axis] for axis in range(3))
    d4x, d4y, d4z = (d4[:, :, axis] for axis in range(3))

    def product(x, y, z):
        return x[:, :, None, None] * y[:, None, :, None] * z[:, None, None, :]

    third = (
        product(d3x, wy, wz),
        product(wx, d3y, wz),
        product(wx, wy, d3z),
        product(d2x, dy, wz),
        product(d2x, wy, dz),
        product(dx, d2y, wz),
        product(wx, d2y, dz),
        product(dx, wy, d2z),
        product(wx, dy, d2z),
        product(dx, dy, dz),
    )
    txxx, tyyy, tzzz, txxy, txxz, txyy, tyyz, txzz, tyzz, txyz = third
    quad = 0.5 * (quadrupoles + quadrupoles.transpose(-1, -2))
    sx = (
        quad[:, 0, 0, None, None, None] * txxx
        + quad[:, 1, 1, None, None, None] * txyy
        + quad[:, 2, 2, None, None, None] * txzz
        + 2.0
        * (
            quad[:, 0, 1, None, None, None] * txxy
            + quad[:, 0, 2, None, None, None] * txxz
            + quad[:, 1, 2, None, None, None] * txyz
        )
    )
    sy = (
        quad[:, 0, 0, None, None, None] * txxy
        + quad[:, 1, 1, None, None, None] * tyyy
        + quad[:, 2, 2, None, None, None] * tyzz
        + 2.0
        * (
            quad[:, 0, 1, None, None, None] * txyy
            + quad[:, 0, 2, None, None, None] * txyz
            + quad[:, 1, 2, None, None, None] * tyyz
        )
    )
    sz = (
        quad[:, 0, 0, None, None, None] * txxz
        + quad[:, 1, 1, None, None, None] * tyyz
        + quad[:, 2, 2, None, None, None] * tzzz
        + 2.0
        * (
            quad[:, 0, 1, None, None, None] * txyz
            + quad[:, 0, 2, None, None, None] * txzz
            + quad[:, 1, 2, None, None, None] * tyzz
        )
    )
    oct_values = 0.5 * (
        gg_positions[:, 0, None, None, None] * sx
        + gg_positions[:, 1, None, None, None] * sy
        + gg_positions[:, 2, None, None, None] * sz
    )
    oct_mesh = grad_mesh.new_zeros(shape).index_put(indices, oct_values.reshape(-1), accumulate=True)

    mesh_values = grad_mesh[indices].reshape(positions.shape[0], 6, 6, 6)
    a3 = tuple((mesh_values * value).sum((1, 2, 3)) for value in third)
    a3xxx, a3yyy, a3zzz, a3xxy, a3xxz, a3xyy, a3yyz, a3xzz, a3yzz, a3xyz = a3
    gx, gy, gz = gg_positions.unbind(-1)
    r00 = gx * a3xxx + gy * a3xxy + gz * a3xxz
    r11 = gx * a3xyy + gy * a3yyy + gz * a3yyz
    r22 = gx * a3xzz + gy * a3yzz + gz * a3zzz
    r01 = gx * a3xxy + gy * a3xyy + gz * a3xyz
    r02 = gx * a3xxz + gy * a3xyz + gz * a3xzz
    r12 = gx * a3xyz + gy * a3yyz + gz * a3yzz
    oct_Q = 0.5 * torch.stack((r00, r01, r02, r01, r11, r12, r02, r12, r22), dim=-1).reshape(-1, 3, 3)

    fourth = (
        product(d4x, wy, wz),
        product(wx, d4y, wz),
        product(wx, wy, d4z),
        product(d3x, dy, wz),
        product(d3x, wy, dz),
        product(dx, d3y, wz),
        product(wx, d3y, dz),
        product(dx, wy, d3z),
        product(wx, dy, d3z),
        product(d2x, d2y, wz),
        product(d2x, wy, d2z),
        product(wx, d2y, d2z),
        product(d2x, dy, dz),
        product(dx, d2y, dz),
        product(dx, dy, d2z),
    )
    a4 = tuple((mesh_values * value).sum((1, 2, 3)) for value in fourth)
    (
        a4xxxx,
        a4yyyy,
        a4zzzz,
        a4xxxy,
        a4xxxz,
        a4xyyy,
        a4yyyz,
        a4xzzz,
        a4yzzz,
        a4xxyy,
        a4xxzz,
        a4yyzz,
        a4xxyz,
        a4xyyz,
        a4xyzz,
    ) = a4
    q00, q11, q22 = quad[:, 0, 0], quad[:, 1, 1], quad[:, 2, 2]
    q01, q02, q12 = quad[:, 0, 1], quad[:, 0, 2], quad[:, 1, 2]
    p00 = q00 * a4xxxx + q11 * a4xxyy + q22 * a4xxzz + 2 * (q01 * a4xxxy + q02 * a4xxxz + q12 * a4xxyz)
    p11 = q00 * a4xxyy + q11 * a4yyyy + q22 * a4yyzz + 2 * (q01 * a4xyyy + q02 * a4xyyz + q12 * a4yyyz)
    p22 = q00 * a4xxzz + q11 * a4yyzz + q22 * a4zzzz + 2 * (q01 * a4xyzz + q02 * a4xzzz + q12 * a4yzzz)
    p01 = q00 * a4xxxy + q11 * a4xyyy + q22 * a4xyzz + 2 * (q01 * a4xxyy + q02 * a4xxyz + q12 * a4xyyz)
    p02 = q00 * a4xxxz + q11 * a4xyyz + q22 * a4xzzz + 2 * (q01 * a4xxyz + q02 * a4xxzz + q12 * a4xyzz)
    p12 = q00 * a4xxyz + q11 * a4yyyz + q22 * a4yzzz + 2 * (q01 * a4xyyz + q02 * a4xyzz + q12 * a4yyzz)
    oct_pos = 0.5 * torch.stack(
        (
            p00 * gx + p01 * gy + p02 * gz,
            p01 * gx + p11 * gy + p12 * gz,
            p02 * gx + p12 * gy + p22 * gz,
        ),
        dim=-1,
    )
    return oct_mesh, oct_pos, oct_Q


def _spread_forward(positions, charges, dipoles, quadrupoles, nx, ny, nz):
    mesh = torch.zeros((nx, ny, nz), dtype=positions.dtype, device=positions.device)
    with _stream(positions.device):
        wp.launch(
            _spread_l2_order6_f32,
            dim=positions.shape[0],
            inputs=[
                _wpt(positions.contiguous(), wp.vec3f),
                _wpt(charges.contiguous(), wp.float32),
                _wpt(dipoles.contiguous(), wp.vec3f),
                _wpt(quadrupoles.contiguous(), wp.mat33f),
            ],
            outputs=[_wpt(mesh, wp.float32)],
            device=str(wp.device_from_torch(positions.device)),
        )
    return mesh


def _spread_forward_fake(positions, charges, dipoles, quadrupoles, nx, ny, nz):
    del charges, dipoles, quadrupoles
    return torch.empty((nx, ny, nz), dtype=positions.dtype, device=positions.device)


def _spread_backward(grad_mesh, positions, charges, dipoles, quadrupoles, nx, ny, nz):  # pragma: no cover
    del nx, ny, nz
    n = positions.shape[0]
    grad_positions = torch.empty_like(positions)
    grad_charges = torch.empty_like(charges)
    grad_dipoles = torch.empty_like(dipoles)
    grad_quadrupoles = torch.empty_like(quadrupoles)
    unused_mesh = torch.empty((1, 1, 1), dtype=grad_mesh.dtype, device=grad_mesh.device)
    with _stream(positions.device):
        wp.launch(
            _gather_l2_order6_f32_backward,
            dim=n,
            inputs=[
                _wpt(grad_mesh.contiguous(), wp.float32),
                _wpt(positions.contiguous(), wp.vec3f),
                _wpt(charges.contiguous(), wp.float32),
                _wpt(dipoles.contiguous(), wp.vec3f),
                _wpt(quadrupoles.contiguous(), wp.mat33f),
                wp.int32(0),
            ],
            outputs=[
                _wpt(unused_mesh, wp.float32),
                _wpt(grad_positions, wp.vec3f),
                _wpt(grad_charges, wp.float32),
                _wpt(grad_dipoles, wp.vec3f),
                _wpt(grad_quadrupoles, wp.mat33f),
            ],
            device=str(wp.device_from_torch(positions.device)),
        )
    return grad_positions, grad_charges, grad_dipoles, grad_quadrupoles


def _spread_backward_fake(grad_mesh, positions, charges, dipoles, quadrupoles, nx, ny, nz):  # pragma: no cover
    del grad_mesh, nx, ny, nz
    return tuple(torch.empty_like(value) for value in (positions, charges, dipoles, quadrupoles))


def _spread_double_backward(
    gg_positions,
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    grad_mesh,
    positions,
    charges,
    dipoles,
    quadrupoles,
    nx,
    ny,
    nz,
):  # pragma: no cover
    gg_pos = _optional_cotangent(gg_positions, positions)
    gg_q = _optional_cotangent(gg_charges, charges)
    gg_d = _optional_cotangent(gg_dipoles, dipoles)
    gg_Q = _optional_cotangent(gg_quadrupoles, quadrupoles)
    gg_Q = 0.5 * (gg_Q + gg_Q.transpose(-1, -2))

    eff_d = gg_d + charges[:, None] * gg_pos
    outer = gg_pos[:, :, None] * dipoles[:, None, :]
    eff_Q = gg_Q + outer + outer.transpose(-1, -2)
    d_grad_mesh = _spread_forward(positions, gg_q, eff_d, eff_Q, nx, ny, nz)
    d_positions, _, gd2, gQ2 = _spread_backward(grad_mesh, positions, gg_q, eff_d, eff_Q, nx, ny, nz)
    d_charges = (gg_pos * gd2).sum(-1)
    d_dipoles = 2.0 * torch.matmul(gQ2, gg_pos.unsqueeze(-1)).squeeze(-1)
    oct_mesh, oct_positions, d_quadrupoles = _octupole_double_terms_torch(positions, quadrupoles, gg_pos, grad_mesh)
    return (
        d_grad_mesh + oct_mesh,
        d_positions + oct_positions,
        d_charges,
        d_dipoles,
        d_quadrupoles,
    )


def _spread_double_backward_fake(
    gg_positions,
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    grad_mesh,
    positions,
    charges,
    dipoles,
    quadrupoles,
    nx,
    ny,
    nz,
):  # pragma: no cover
    del gg_positions, gg_charges, gg_dipoles, gg_quadrupoles, nx, ny, nz
    return tuple(torch.empty_like(value) for value in (grad_mesh, positions, charges, dipoles, quadrupoles))


register_warp_op_chain(
    name="graph_longrange::pme_spread_l2_order6_f32",
    forward=_spread_forward,
    forward_fake=_spread_forward_fake,
    backward=_spread_backward,
    backward_fake=_spread_backward_fake,
    backward_return_arity=4,
    diff_input_positions=(0, 1, 2, 3),
    n_forward_inputs=7,
    double_backward=_spread_double_backward,
    double_backward_fake=_spread_double_backward_fake,
    double_backward_return_arity=5,
    second_order_diff_positions=(0, 1, 2, 3, 4),
    n_backward_inputs=8,
    forward_schema=(
        "(Tensor positions, Tensor charges, Tensor dipoles, Tensor quadrupoles, " "int nx, int ny, int nz) -> Tensor"
    ),
    backward_schema=(
        "(Tensor grad_mesh, Tensor positions, Tensor charges, Tensor dipoles, "
        "Tensor quadrupoles, int nx, int ny, int nz) "
        "-> (Tensor, Tensor, Tensor, Tensor)"
    ),
    double_backward_schema=(
        "(Tensor? gg_positions, Tensor? gg_charges, Tensor? gg_dipoles, "
        "Tensor? gg_quadrupoles, Tensor grad_mesh, Tensor positions, "
        "Tensor charges, Tensor dipoles, Tensor quadrupoles, int nx, int ny, "
        "int nz) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
)


def _gather_forward(phi: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    unused_q = torch.empty(positions.shape[0], dtype=phi.dtype, device=phi.device)
    unused_d = torch.empty_like(positions)
    unused_Q = torch.empty((positions.shape[0], 3, 3), dtype=phi.dtype, device=phi.device)
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
                wp.int32(1),
            ],
            outputs=[
                _wpt(d_phi, wp.float32),
                _wpt(d_positions, wp.vec3f),
                _wpt(unused_q, wp.float32),
                _wpt(unused_d, wp.vec3f),
                _wpt(unused_Q, wp.mat33f),
            ],
            device=str(wp.device_from_torch(phi.device)),
        )
    return d_phi, d_positions


def _gather_backward_fake(cq, cd, cQ, phi, positions):  # pragma: no cover
    del cq, cd, cQ
    return torch.empty_like(phi), torch.empty_like(positions)


def _gather_double_backward(gg_phi, gg_positions, cq, cd, cQ, phi, positions):  # pragma: no cover
    gg_phi = _optional_cotangent(gg_phi, phi)
    gg_positions = _optional_cotangent(gg_positions, positions)
    cQ = 0.5 * (cQ + cQ.transpose(-1, -2))

    # The gather forward is the transpose of charge-only spreading.  The two
    # Hessian-vector-product groups below are therefore expressible entirely
    # through the graph-owned spread chain; no nested autograd (which is
    # disabled inside a torch.library backend) or toolkit spline is needed.
    dcq, dcd, dcQ = _gather_forward(gg_phi, positions)
    _, a_dpositions = _gather_backward(cq, cd, cQ, gg_phi, positions)
    b_dphi, b_dpositions, b_dcq, b_dcd, b_dcQ = _spread_double_backward(
        gg_positions,
        None,
        None,
        None,
        phi,
        positions,
        cq,
        cd,
        cQ,
        *phi.shape,
    )
    return (
        dcq + b_dcq,
        dcd + b_dcd,
        dcQ + b_dcQ,
        b_dphi,
        a_dpositions + b_dpositions,
    )


def _gather_double_backward_fake(gg_phi, gg_positions, cq, cd, cQ, phi, positions):  # pragma: no cover
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
    backward_schema=("(Tensor cq, Tensor cd, Tensor cQ, Tensor phi, Tensor positions) " "-> (Tensor, Tensor)"),
    double_backward_schema=(
        "(Tensor? gg_phi, Tensor? gg_positions, Tensor cq, Tensor cd, Tensor cQ, "
        "Tensor phi, Tensor positions) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
)


def _fractionalize_forward(positions, cell_inv_t, dipoles, quadrupoles, batch_idx):
    return _pme._pme_fractionalize_forward(positions, cell_inv_t, dipoles, quadrupoles, batch_idx)


def _fractionalize_forward_fake(positions, cell_inv_t, dipoles, quadrupoles, batch_idx):  # pragma: no cover
    del cell_inv_t, batch_idx
    return (
        torch.empty_like(positions),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


def _fractionalize_backward(gp, gdf, gQf, positions, cell_inv_t, dipoles, quadrupoles, batch_idx):  # pragma: no cover
    if (
        positions.dtype != torch.float32
        or cell_inv_t.shape[0] != 1
        or positions.device.type not in _REDUCED_CELL_DEVICE_TYPES
    ):
        return _pme._pme_fractionalize_backward(gp, gdf, gQf, positions, cell_inv_t, dipoles, quadrupoles, batch_idx)
    n = positions.shape[0]
    grad_positions = torch.empty_like(positions)
    grad_cell_atoms = torch.empty((n, 3, 3), dtype=positions.dtype, device=positions.device)
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
    return tuple(torch.empty_like(value) for value in (positions, cell_inv_t, dipoles, quadrupoles))


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
    return torch.ops.graph_longrange.pme_fractionalize(positions, cell_inv_t, dipoles, quadrupoles, batch_idx)


def spread_l2_order6_f32(positions, charges, dipoles, quadrupoles, nx, ny, nz):
    """Graph-owned stable order-6 l=2 spread with analytical first backward."""
    return torch.ops.graph_longrange.pme_spread_l2_order6_f32(positions, charges, dipoles, quadrupoles, nx, ny, nz)


def gather_l2_order6_f32(phi, positions):
    """Graph-owned order-6 l=2 gather with fused analytical backward."""
    return torch.ops.graph_longrange.pme_gather_l2_order6_f32(phi, positions)
