"""Compare independent versus shared Torch PME transforms, not full model MD."""

import argparse
import json
import statistics
import time

import torch

from graph_longrange.pme_optimized import (
    PMEElectrostaticFeaturesOptimized,
    _pme_features_impl,
    compute_pme_single_optimized,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--widths", type=float, nargs="+", default=[.8, 1.2, 1.6])
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.atoms, args.mesh, args.repeats, args.threads, *args.widths) <= 0:
        parser.error("sizes, widths, repeats and threads must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(814)
    positions = torch.rand(args.atoms, 3, device=device) * 12
    cell = torch.eye(3, device=device).unsqueeze(0) * 14
    source = torch.randn(args.atoms, 9, device=device) * .1
    batch = torch.zeros(args.atoms, dtype=torch.long, device=device)
    module = PMEElectrostaticFeaturesOptimized(
        density_max_l=2, density_smearing_width=1.2, feature_max_l=2,
        feature_smearing_widths=args.widths, mesh_size=args.mesh,
        include_self_interaction=False,
    ).to(device)
    methods = {
        # Wrapper bypasses only transform sharing, not other Torch optimizations.
        "independent": lambda *a, **kw: compute_pme_single_optimized(*a, **kw),
        "shared": compute_pme_single_optimized,
    }

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    print(json.dumps({"metadata": vars(args), "dtype": "float32", "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}))
    for backward in (False, True):
        reference = None
        for name, compute in methods.items():
            def call():
                variables = tuple(t.detach().requires_grad_(backward) for t in (positions, cell, source))
                with torch.set_grad_enabled(backward):
                    out = _pme_features_impl(module, variables[2], variables[0], batch, variables[1], compute)
                    grads = torch.autograd.grad(out.square().sum(), variables) if backward else ()
                return (out.detach(),) + grads

            # CPU indexed reductions can vary across threads even without the
            # optimization. Isolate arithmetic parity, then restore timed threads.
            if device.type == "cpu":
                torch.set_num_threads(1)
            candidate = call()
            if reference is None:
                reference = candidate
            errors = [float((a - b).abs().max()) for a, b in zip(candidate, reference)]
            print(json.dumps({"method": name, "backward": backward, "parity_errors": errors,
                              "reference_magnitudes": [float(v.abs().max()) for v in reference]}), flush=True)
            for a, b in zip(candidate, reference):
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
            if device.type == "cpu":
                torch.set_num_threads(args.threads)
            for _ in range(4):
                call()
            sync()
            timings = []
            for _ in range(args.repeats):
                sync()
                start = time.perf_counter()
                call()
                sync()
                timings.append((time.perf_counter() - start) * 1000)
            print(json.dumps({"method": name, "mode": "live_cell_vjp" if backward else "fixed_cell_forward",
                              "median_ms": statistics.median(timings), "max_absolute_errors": errors,
                              "sharing_enabled": name == "shared" and not backward}), flush=True)


if __name__ == "__main__":
    main()
