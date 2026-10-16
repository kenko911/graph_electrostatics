#!/usr/bin/env python3
"""Benchmark PME vs GTO/Ewald electrostatic-feature evaluators across supercell sizes.

Times the two evaluators the TensorNetPolar SCF wraps:
    GTOElectrostaticFeatures   -- reciprocal-space Ewald sum,  cost ~ O(N * N_k) ~ O(N^2)
    PMEElectrostaticFeatures   -- particle-mesh Ewald (FFT),   cost ~ O(N log N)

Both are run in "steady state" (k-vector table and PME theta cache pre-warmed), which is
what training sees when the cell is fixed.  The PME mesh is scaled with the cell edge so the
accuracy (mesh spacing) is held constant -> a fair equal-accuracy comparison.  ``relerr`` is
the max abs difference of the PME features vs the exact Ewald features, normalised.

GPU timing uses CUDA events with explicit synchronisation.

Examples
--------
    python benchmark_pme_vs_ewald.py                       # auto GPU, n=1..6
    python benchmark_pme_vs_ewald.py --ns 1 2 3 4 5 6 7 8  # push toward the crossover
    python benchmark_pme_vs_ewald.py --dtype float64 --mesh-per-ang 5
    python benchmark_pme_vs_ewald.py --device cpu
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch

from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.pme import PMEElectrostaticFeatures
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
from graph_longrange.kspace import compute_k_vectors_flat


def time_fn(fn, device: torch.device, warmup: int, repeats: int) -> float:
    """Median wall-time of ``fn`` in milliseconds, with correct GPU synchronisation."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
    else:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def make_supercell(n: int, base_pos, base_sf, base_cell: float, device, dtype):
    """n x n x n replication of the base cell -> (positions, source_feats, box, vol, pbc, batch)."""
    L = base_cell * n
    reps = torch.tensor(
        [[a, b, c] for a in range(n) for b in range(n) for c in range(n)],
        device=device, dtype=dtype,
    )
    pos = (base_pos[None] + (reps * base_cell)[:, None]).reshape(-1, 3)
    sf = base_sf.repeat(n**3, 1)
    box = (torch.eye(3, device=device, dtype=dtype) * L).unsqueeze(0)
    vol = torch.tensor([L**3], device=device, dtype=dtype)
    pbc = torch.tensor([[True, True, True]], device=device)
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
    return pos, sf, box, vol, pbc, batch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    help="supercell replication factors n (system = n^3 * base atoms)")
    ap.add_argument("--base-cell", type=float, default=5.0, help="base cubic cell edge (Angstrom)")
    ap.add_argument("--base-atoms", type=int, default=4, help="atoms in the base cell")
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian smearing width (Angstrom)")
    ap.add_argument("--mesh-per-ang", type=float, default=4.0,
                    help="PME mesh points per Angstrom (constant spacing across sizes)")
    ap.add_argument("--max-l", type=int, default=2, choices=[1, 2],
                    help="source/feature max angular momentum (2 = with quadrupole/EFG)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    L = args.max_l
    K = (L + 1) ** 2  # source components: 1 (l0) [+3 l1] [+5 l2]

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"device={device}  dtype={args.dtype}  max_l={L}  sigma={args.sigma}  "
          f"mesh={args.mesh_per_ang}/A  warmup={args.warmup} repeats={args.repeats}")

    kcut = 1.5 * gto_basis_kspace_cutoff([args.sigma, args.sigma], L)
    base_pos = torch.rand(args.base_atoms, 3, device=device, dtype=dtype) * args.base_cell
    base_sf = torch.randn(args.base_atoms, K, device=device, dtype=dtype) * 0.3
    base_sf[:, 0] -= base_sf[:, 0].mean()  # ~charge-neutral l0

    gto = GTOElectrostaticFeatures(
        L, args.sigma, L, [args.sigma], False, kcut,
        quadrupole_feature_corrections=(L >= 2), integral_normalization="receiver",
    ).to(device)

    print(f"\n{'n':>2} {'N':>7} {'N_k':>8} {'mesh':>5} | "
          f"{'Ewald_ms':>10} {'PME_ms':>10} {'speedup':>8} {'relerr':>9}", flush=True)
    for n in args.ns:
        try:
            pos, sf, box, vol, pbc, batch = make_supercell(
                n, base_pos, base_sf, args.base_cell, device, dtype)
            N = pos.shape[0]
            mesh = int(round(args.mesh_per_ang * args.base_cell * n))

            # k-vectors (cached across forwards in the model -> compute once, outside timing)
            rcell = 2 * math.pi * torch.linalg.inv(box).transpose(-1, -2)
            kvec, knorm, kbatch, k0 = compute_k_vectors_flat(kcut, box, rcell)

            pme = PMEElectrostaticFeatures(
                L, args.sigma, L, [args.sigma], mesh,
                include_self_interaction=False, integral_normalization="receiver",
            ).to(device)

            fg = lambda: gto(kvec, knorm, kbatch, k0, sf, pos, batch, vol, pbc)  # noqa: E731
            fp = lambda: pme(sf, pos, batch, box, pbc)                          # noqa: E731

            with torch.no_grad():
                Fg = fg()
                Fp = fp()
                rel = float((Fg - Fp).abs().max() / Fg.abs().max().clamp(min=1e-12))
                tg = time_fn(fg, device, args.warmup, args.repeats)
                tp = time_fn(fp, device, args.warmup, args.repeats)

            print(f"{n:>2} {N:>7} {kvec.shape[0]:>8} {mesh:>5} | "
                  f"{tg:>10.2f} {tp:>10.2f} {tg / tp:>7.2f}x {rel:>9.1e}", flush=True)
        except RuntimeError as e:
            msg = "OOM" if "out of memory" in str(e).lower() else type(e).__name__
            print(f"{n:>2} {'-':>7} {'-':>8} {'-':>5} |  skipped ({msg})", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nEwald ~ O(N * N_k) ~ O(N^2);  PME ~ O(N log N).  PME wins once N is large enough")
    print("that Ewald's quadratic k-sum dominates PME's fixed FFT/spreading cost.")


if __name__ == "__main__":
    main()
