"""Fused CUDA inference kernel for molecular l=0/1/2 electrostatic features."""

import torch
import warp as wp

wp.init()


def _make_kernel(dtype, dtype_name):
    class vec3(wp.types.vector(length=3, dtype=dtype)):
        pass

    class vec5(wp.types.vector(length=5, dtype=dtype)):
        pass

    def kernel(
        source: wp.array(ndim=2, dtype=dtype),
        positions: wp.array(ndim=2, dtype=dtype),
        batch: wp.array(ndim=1, dtype=wp.int32),
        starts: wp.array(ndim=1, dtype=wp.int32),
        ends: wp.array(ndim=1, dtype=wp.int32),
        width: dtype,
        l0_factor: dtype,
        l1_weight: dtype,
        l2_weight: dtype,
        scale: dtype,
        out: wp.array(ndim=2, dtype=dtype),
    ):
        i = wp.tid()
        graph = batch[i]
        l0 = dtype(0.0)
        l1 = vec3(dtype(0.0))
        l2 = vec5(dtype(0.0))
        inv_sqrt3 = dtype(0.57735026918962576451)
        two_inv_sqrt3 = dtype(1.1547005383792515290)
        sqrt3_over2 = dtype(0.86602540378443864676)
        sqrt_pi = dtype(1.7724538509055160273)
        width2 = width * width
        width4 = width2 * width2
        width6 = width4 * width2

        for j in range(starts[graph], ends[graph]):
            if j == i:
                continue
            rx = positions[i, 0] - positions[j, 0]
            ry = positions[i, 1] - positions[j, 1]
            rz = positions[i, 2] - positions[j, 2]
            r = wp.max(wp.sqrt(rx * rx + ry * ry + rz * rz), dtype(1.0e-10))
            r2 = r * r
            inv_r = dtype(1.0) / r
            nx = rx * inv_r
            ny = ry * inv_r
            nz = rz * inv_r
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r2 * inv_r
            inv_r4 = inv_r2 * inv_r2
            gaussian = wp.exp(-r2 / (dtype(4.0) * width2)) / (width * sqrt_pi)
            potential = wp.erf(r / (dtype(2.0) * width)) * inv_r
            fp = (gaussian - potential) * inv_r
            fp_over_r = fp * inv_r
            fpp = -gaussian / (dtype(2.0) * width2) - dtype(2.0) * gaussian * inv_r2 + dtype(2.0) * potential * inv_r2
            fppp = r * gaussian / (dtype(4.0) * width4) + gaussian / (width2 * r) + dtype(6.0) * gaussian * inv_r3 - dtype(6.0) * potential * inv_r3
            f4 = -r2 * gaussian / (dtype(8.0) * width6) - gaussian / (dtype(4.0) * width4) - dtype(4.0) * gaussian / (width2 * r2) - dtype(24.0) * gaussian * inv_r4 + dtype(24.0) * potential * inv_r4

            charge = source[j, 0]
            mx = source[j, 3]
            my = source[j, 1]
            mz = source[j, 2]
            mu_n = mx * nx + my * ny + mz * nz

            qm2 = source[j, 4]
            qm1 = source[j, 5]
            q0 = source[j, 6]
            qp1 = source[j, 7]
            qp2 = source[j, 8]
            qxx = -dtype(0.5) * q0 + sqrt3_over2 * qp2
            qyy = -dtype(0.5) * q0 - sqrt3_over2 * qp2
            qzz = q0
            qxy = sqrt3_over2 * qm2
            qyz = sqrt3_over2 * qm1
            qxz = sqrt3_over2 * qp1
            qnx = qxx * nx + qxy * ny + qxz * nz
            qny = qxy * nx + qyy * ny + qyz * nz
            qnz = qxz * nx + qyz * ny + qzz * nz
            qnn = qnx * nx + qny * ny + qnz * nz

            hessian_aniso = fpp - fp_over_r
            l0 += charge * potential - mu_n * fp + qnn * hessian_aniso

            dip_coeff = (fp_over_r - fpp) * mu_n
            quad_mix = dtype(2.0) * (fpp * inv_r - fp * inv_r2)
            quad_radial = fppp - dtype(3.0) * fpp * inv_r + dtype(3.0) * fp * inv_r2
            common = fp * charge + dip_coeff + quad_radial * qnn
            l1[0] += common * nx - fp_over_r * mx + quad_mix * qnx
            l1[1] += common * ny - fp_over_r * my + quad_mix * qny
            l1[2] += common * nz - fp_over_r * mz + quad_mix * qnz

            rsh = vec5()
            rsh[0] = two_inv_sqrt3 * nx * ny
            rsh[1] = two_inv_sqrt3 * ny * nz
            rsh[2] = nz * nz - dtype(1.0 / 3.0)
            rsh[3] = two_inv_sqrt3 * nx * nz
            rsh[4] = (nx * nx - ny * ny) * inv_sqrt3
            k1 = quad_radial
            k2 = fpp * inv_r - fp * inv_r2
            sym_mu = vec5()
            sym_mu[0] = two_inv_sqrt3 * (mx * ny + nx * my)
            sym_mu[1] = two_inv_sqrt3 * (my * nz + ny * mz)
            sym_mu[2] = dtype(2.0) * mz * nz - dtype(2.0 / 3.0) * mu_n
            sym_mu[3] = two_inv_sqrt3 * (mx * nz + nx * mz)
            sym_mu[4] = dtype(2.0) * (mx * nx - my * ny) * inv_sqrt3
            fourth = f4 - dtype(6.0) * fppp * inv_r + dtype(15.0) * fpp * inv_r2 - dtype(15.0) * fp * inv_r3
            lam_m = dtype(2.0) * k1 * inv_r
            lam_i = dtype(2.0) * hessian_aniso * inv_r2
            sym_q = vec5()
            sym_q[0] = two_inv_sqrt3 * (qnx * ny + nx * qny)
            sym_q[1] = two_inv_sqrt3 * (qny * nz + ny * qnz)
            sym_q[2] = dtype(2.0) * qnz * nz - dtype(2.0 / 3.0) * qnn
            sym_q[3] = two_inv_sqrt3 * (qnx * nz + nx * qnz)
            sym_q[4] = dtype(2.0) * (qnx * nx - qny * ny) * inv_sqrt3
            for m in range(5):
                l2[m] += (hessian_aniso * charge + k1 * mu_n + fourth * qnn) * rsh[m]
                l2[m] += k2 * sym_mu[m] + lam_m * sym_q[m] + lam_i * source[j, 4 + m]

        out[i, 0] = scale * l0_factor * l0
        out[i, 1] = scale * l1_weight * l1[1]
        out[i, 2] = scale * l1_weight * l1[2]
        out[i, 3] = scale * l1_weight * l1[0]
        for m in range(5):
            out[i, 4 + m] = scale * l2_weight * l2[m]

    return wp.Kernel(kernel, key=f"realspace_features_l2_{dtype_name}")


_KERNELS = {
    torch.float32: _make_kernel(wp.float32, "float32"),
    torch.float64: _make_kernel(wp.float64, "float64"),
}


def fused_multipole_features_l2(source, positions, batch, starts, ends, width, l0_factor, l1_weight, l2_weight, scale):
    """Evaluate single-radial l=2 molecular features without edge intermediates."""
    out = torch.empty((positions.shape[0], 9), dtype=source.dtype, device=source.device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device)) if positions.is_cuda else None
    scalar = wp.float32 if source.dtype == torch.float32 else wp.float64
    wp.launch(
        _KERNELS[source.dtype],
        dim=positions.shape[0],
        inputs=[
            wp.from_torch(source.detach().contiguous(), return_ctype=True),
            wp.from_torch(positions.detach().contiguous(), return_ctype=True),
            wp.from_torch(batch.to(torch.int32).contiguous(), return_ctype=True),
            wp.from_torch(starts.to(torch.int32).contiguous(), return_ctype=True),
            wp.from_torch(ends.to(torch.int32).contiguous(), return_ctype=True),
            scalar(width), scalar(l0_factor), scalar(l1_weight), scalar(l2_weight), scalar(scale),
        ],
        outputs=[wp.from_torch(out, return_ctype=True)],
        stream=stream,
    )
    return out
