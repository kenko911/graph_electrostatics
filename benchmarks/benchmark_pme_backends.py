"""Benchmark PME phi/field/Hessian and one VJP (not full TensorNetPolar MD).

Example: python benchmarks/benchmark_pme_backends.py --device cuda --atoms 3645 --mesh 64
Defaults to float32. Timings measure module calls, not full MD steps.
"""

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import statistics
import time

import torch

from graph_longrange.pme import _precompute_pme_reciprocal, compute_pme_single
from graph_longrange.pme_optimized import compute_pme_single_optimized, _precompute_pme_reciprocal_rfft
import graph_longrange.pme_optimized as optimized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--compile-stencils", action="store_true",
                        help="also benchmark isolated Inductor spread/gather fusion; defaults unchanged")
    parser.add_argument("--check-parity", action="store_true",
                        help="check module outputs and VJP against uncompiled optimized Torch before timing")
    parser.add_argument("--parity-atol", type=float, default=2e-5)
    parser.add_argument("--parity-rtol", type=float, default=2e-5)
    parser.add_argument("--profile", action="store_true", help="print an operator-level profile for each mode")
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--trace-dir", type=Path, help="save Chrome traces (requires --profile)")
    parser.add_argument("--backends", nargs="+", choices=["reference", "torch", "warp"],
                        default=["reference", "torch", "warp"])
    args = parser.parse_args()
    if min(args.atoms, args.mesh, args.repeats, args.threads, args.profile_repeats) < 1:
        parser.error("sizes, repeats, and threads must be positive")
    if args.trace_dir is not None and not args.profile:
        parser.error("--trace-dir requires --profile")
    torch.set_num_threads(args.threads)
    torch.manual_seed(74)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    # Toolkit backward factories must use the same default dtype as inputs.
    torch.set_default_dtype(dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    print(json.dumps({"metadata": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                  "device": str(device), "dtype": args.dtype, "threads": args.threads,
                                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                                  "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                  "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                  "seed": 74}}), flush=True)
    box = torch.tensor([[20., 0., 0.], [.3, 19., 0.], [.1, .2, 21.]], device=device, dtype=dtype)
    coords = torch.rand(args.atoms, 3, device=device, dtype=dtype) @ box
    q = torch.randn(args.atoms, device=device, dtype=dtype) * .1
    q -= q.mean()
    p = torch.randn_like(coords) * .1
    Q = torch.randn(args.atoms, 3, 3, device=device, dtype=dtype) * .1
    Q = (Q + Q.transpose(-1, -2)) / 2
    Q -= torch.eye(3, device=device, dtype=dtype) * Q.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] / 3
    weights = (torch.randn_like(q), torch.randn_like(p), torch.randn_like(Q))
    methods = {"reference": compute_pme_single, "torch": compute_pme_single_optimized}
    original_stencils = (optimized._stencil_weights, optimized._spread_with_geometry,
                         optimized._gather_with_geometry)
    compiled_stencils = None
    if args.compile_stencils:
        if "torch" not in args.backends:
            parser.error("--compile-stencils requires the torch backend")
        compiled_stencils = tuple(torch.compile(fn, fullgraph=True) for fn in original_stencils)
        methods["torch_compiled"] = compute_pme_single_optimized
        args.backends.append("torch_compiled")

    @contextmanager
    def stencil_mode(compiled):
        if compiled:
            optimized._stencil_weights, optimized._spread_with_geometry, optimized._gather_with_geometry = compiled_stencils
        try:
            yield
        finally:
            optimized._stencil_weights, optimized._spread_with_geometry, optimized._gather_with_geometry = original_stencils
    if "warp" in args.backends:
        from graph_longrange.warp_electrostatics import _compute_pme_single_fused_warp
        methods["warp"] = _compute_pme_single_fused_warp

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for mode in ("forward_fixed_cell", "forward_vjp_live_cell"):
        backward = mode == "forward_vjp_live_cell"
        # Static reciprocal preparation is excluded only in fixed-cell forward.
        for backend in args.backends:
            prepare = _precompute_pme_reciprocal if backend == "reference" else _precompute_pme_reciprocal_rfft
            static = prepare(box, 1 / 3., args.mesh) if not backward else None
            def call(profiling=False, compiled=None, method=None):
                def scope(name):
                    return torch.profiler.record_function(name) if profiling else nullcontext()

                use_compiled = backend == "torch_compiled" if compiled is None else compiled
                with torch.set_grad_enabled(backward), stencil_mode(use_compiled):
                    variables = tuple(v.detach().requires_grad_(backward) for v in (coords, box, q, p, Q))
                    x, cell, charges, dipoles, quads = variables
                    with scope("PME_reciprocal_preparation"):
                        reciprocal = prepare(cell, 1 / 3., args.mesh) if backward else static
                    with scope("PME_forward"):
                        out = (methods[backend] if method is None else method)(x, cell, charges, dipoles, 1 / 3., args.mesh, 2,
                                               want_field=True, Q=quads, want_hessian=True,
                                               reciprocal_cache=reciprocal)
                    if backward:
                        with scope("PME_backward_VJP"):
                            gradients = torch.autograd.grad(out, variables, weights)
                    return tuple(v.detach() for v in out) + (gradients if backward else ())
            if args.check_parity:
                reference = call(compiled=False, method=compute_pme_single_optimized)
                candidate = call()
                metrics = [{"max_absolute_error": float((a - b).abs().max()),
                            "relative_to_max_reference": float((a - b).abs().max() /
                                                               b.abs().max().clamp_min(1e-30))}
                           for a, b in zip(candidate, reference)]
                print(json.dumps({"backend": backend, "mode": mode,
                                  "parity_order": ["phi", "field", "hessian"] +
                                  (["dpositions", "dcell", "dq", "dp", "dQ"] if backward else []),
                                  "parity_metrics": metrics}), flush=True)
                for a, b in zip(candidate, reference):
                    torch.testing.assert_close(a, b, atol=args.parity_atol, rtol=args.parity_rtol)
            if backend == "torch_compiled":
                reference = call(compiled=False)
                candidate = call(compiled=True)
                errors = [float((a - b).abs().max()) for a, b in zip(candidate, reference)]
                for a, b in zip(candidate, reference):
                    torch.testing.assert_close(a, b, rtol=3e-4 if dtype == torch.float32 else 1e-9,
                                               atol=3e-5 if dtype == torch.float32 else 1e-10)
                print(json.dumps({"backend": backend, "mode": mode,
                                  "compile_parity_max_absolute_errors": errors,
                                  "compilation_excluded_from_timing": True}), flush=True)
            for _ in range(3):
                call()
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in range(args.repeats):
                sync()
                start = time.perf_counter()
                call()
                sync()
                timings.append(1000 * (time.perf_counter() - start))
            print(json.dumps({"backend": backend, "mode": mode, "atoms": args.atoms,
                              "mesh": args.mesh, "device": str(device), "dtype": args.dtype,
                              "median_ms": statistics.median(timings), "min_ms": min(timings),
                              "max_ms": max(timings), "peak_allocated_bytes":
                              torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}),
                  flush=True)
            if args.profile:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities, record_shapes=True,
                                            profile_memory=True) as profiler:
                    with torch.profiler.record_function(f"PME_{backend}_{mode}"):
                        for _ in range(args.profile_repeats):
                            call(profiling=True)
                    sync()
                sort_by = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
                print(profiler.key_averages().table(sort_by=sort_by, row_limit=20), flush=True)
                if args.trace_dir is not None:
                    args.trace_dir.mkdir(parents=True, exist_ok=True)
                    trace = args.trace_dir / f"pme_{backend}_{mode}_{args.atoms}_{args.mesh}_{args.dtype}.json"
                    if trace.exists():
                        raise FileExistsError(f"Trace exists; choose a fresh --trace-dir: {trace}")
                    profiler.export_chrome_trace(str(trace))
                    print(json.dumps({"trace": str(trace), "profile_repeats": args.profile_repeats}), flush=True)


if __name__ == "__main__":
    main()
