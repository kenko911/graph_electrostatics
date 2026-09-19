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

Torch float32 is checked against the stable reference in CPU regressions. Warp
must use the toolkit's stable order-6 local spline polynomials, including third
and fourth derivatives. Tests report expected failures for older toolkits;
those expected failures are not an accuracy endorsement.
Separate Warp tests check that the half-spectrum change preserves its existing
full-cache outputs and derivatives.

Before enabling a new kernel for production, rerun trained-checkpoint GPU
comparisons of energy, forces, stress, charges, dipole, polarizability and BEC
on both orthogonal and triclinic cells. Report absolute and relative errors;
implementation self-consistency does not establish PME accuracy against Ewald.

## Isolated compilation and Warp precision checks

The benchmark-only `--compile-stencils` mode compiles spline tensor products,
spread, and gather separately with Inductor. It does not compile the FFT or
change model defaults. Compilation is excluded from timings, and outputs plus
the first VJP must pass parity checks before compiled timings are printed.
This does not establish compiled higher-derivative/BEC compatibility.

```bash
python benchmarks/benchmark_pme_backends.py \
  --device cuda --dtype float32 --atoms 3645 --mesh 32 \
  --backends torch --compile-stencils --repeats 30 --profile
```

For Warp checks, install the updated `nvalchemiops` toolkit, or explicitly put
both source checkouts on `PYTHONPATH` (substitute your actual paths):

```bash
set -euo pipefail
export PYTHONPATH=/path/to/nvalchemi-toolkit-ops:/path/to/graph_electrostatics
python -c 'import nvalchemiops.math.spline as s; print(s.__file__); assert hasattr(s, "_bspline6_local")'
python benchmarks/benchmark_pme_backends.py \
  --device cuda --dtype float32 --atoms 1080 --mesh 32 \
  --backends torch warp --check-parity --repeats 30 --profile
```

`--check-parity` prints absolute and max-reference-normalized errors for phi,
field, Hessian and position/cell/source VJPs, then checks elementwise tolerances
(`--parity-atol` and `--parity-rtol`, both default to 2e-5). These are module
units and diagnostic thresholds, not physical force/stress acceptance targets.
Do not loosen thresholds merely to make a failed production check pass.

Optimized Torch features now share spreading and the forward rFFT across
multiple Gaussian widths within one no-gradient feature call. All dtypes
retain independent graphs for forces/stress/BEC: sharing backward has been
removed after merging float32 stencil contributions failed the larger accuracy
gate despite identical forward outputs. Sharing requires `torch.no_grad()` or
`torch.inference_mode()`; the low-level shared-spectrum argument rejects use
when gradient recording is enabled.
The inverse transforms remain
separate because their Green functions differ. A single-width call is unchanged;
no spectrum is cached across model calls, MD steps, or changing SCF multipoles.

Measure that sharing independently with:

```bash
python benchmarks/benchmark_pme_multiwidth.py \
  --device cuda --atoms 3645 --mesh 32 --widths .8 1.2 1.6 --repeats 30
```

CPU smoke results (float32, two threads, 512 atoms, mesh 32; not CUDA or MD):
three-width sharing was about 2.1-2.3x faster forward. A diagnostic merged live-cell
VJP was about 2x faster but failed strict float32 gradient parity at 512 atoms;
that derivative optimization is disabled, not a deployed speedup.
Isolated stencil compilation was slower: 8.25 -> 11.83 ms forward and
19.87 -> 21.98 ms with the live-cell VJP. Keep compilation benchmark-only until
CUDA measurements justify deployment.
