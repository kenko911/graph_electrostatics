# Float32 PME profiling baseline

Run from this repository with its `graph_longrange` package installed (or set
`PYTHONPATH` to this checkout). Use a fresh directory for each run; existing trace
files are never overwritten. Shell failures must stop the benchmark job.

```bash
set -euo pipefail
python benchmarks/benchmark_pme_backends.py \
  --device cuda --dtype float32 --atoms 1080 --mesh 32 \
  --backends torch warp --repeats 30 --profile \
  --trace-dir results/pme_1080_mesh32_run1
python benchmarks/benchmark_pme_backends.py \
  --device cuda --dtype float32 --atoms 3645 --mesh 32 \
  --backends torch warp --repeats 30 --profile \
  --trace-dir results/pme_3645_mesh32_run1
```

Use the checkpoint's actual mesh instead of 32 if different; compare identical
meshes, widths, precision settings and hardware. Repeat runs with
`--backends warp torch` to expose backend-order drift. Save stdout/stderr and the
MatGL and graph_electrostatics commit IDs alongside the traces.

The fixture has a fixed 20 x 19 x 21 A triclinic cell, random neutral charges,
dipoles and traceless quadrupoles. It is a controlled kernel workload, NOT a
physical density-preserving scaling test or a trained TensorNetPolar benchmark.

The synchronized wall-time medians exclude kernel compilation and profiling.
`forward_fixed_cell` excludes reciprocal preparation after warmup;
`forward_vjp_live_cell` includes live-cell reciprocal preparation and one VJP of
potential, field and field gradient with respect to positions, cell and sources.
This VJP is not a physical E+F+S calculation and is not three-component BEC.

Chrome traces contain separate `PME_reciprocal_preparation`, `PME_forward` and
`PME_backward_VJP` regions. Inspect GPU work associated with the backward region
before choosing a Warp stencil/atomic optimization. Table totals include all
profile repetitions; do not compare them directly with per-call wall time.

## Accuracy gates

Torch float32 is checked against the stable reference in CPU regressions. Two
strict expected-failure tests explicitly track the pre-existing float32
Warp-versus-stable-Torch discrepancy; they are not an accuracy endorsement.
Separate Warp tests check that the half-spectrum change preserves its existing
full-cache outputs and derivatives.

Before enabling a new kernel for production, rerun trained-checkpoint GPU
comparisons of energy, forces, stress, charges, dipole, polarizability and BEC
on both orthogonal and triclinic cells. Report absolute and relative errors;
implementation self-consistency does not establish PME accuracy against Ewald.
