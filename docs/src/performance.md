# Performance

All numbers below are produced by the repository's own benchmark scripts;
re-run them on your machine — absolute seconds are hardware- and load-dependent
(the atmospheric tutorial shows 3-6x swings on a shared box).

## Pipeline-stage decomposition

`examples/perf/bench_pipeline_stages.jl` (median of 3; Julia 1.12.7; const-V
adiabatic CH4-air ignition, FBDF @ reltol=1e-8/abstol=1e-12, reaction-sharded
analytic Jacobian, KLU) — the same table as the README:

| Mechanism | build | JIT compile | cold solve | warm solve |
|---|---:|---:|---:|---:|
| GRI-Mech 3.0 (53 sp / 325 rxn) | 2.5 s | 9.9 s | 10.2 s | 0.27 s |
| FFCM 2.0 (96 sp / 1054 rxn) | 8.4 s | 31.2 s | 31.7 s | 0.55 s |
| Aramco 3.0 (581 sp / 3037 rxn) | 47.5 s | 197.9 s | 201.5 s | **3.55 s** |

How to read it:

- **Cold is compile, not solve.** Cold solves are dominated by the one-time LLVM
  JIT compilation of the generated code; warm solves reuse the compiled functions.
  Any cold-vs-cold linear-solver comparison must account for this (the linsolver
  bench below splits compile from warm precisely because a naive first-run ordering
  once made KLU look 35x slower than it is).
- **Warm, ChemMechSim beats Cantera on the largest mechanism** at the same
  tolerance: 3.55 s vs `IdealGasReactor` 3.87 s, with roughly half the integration
  steps (898 vs 1783).

```bash
julia --project=. examples/perf/bench_pipeline_stages.jl     # → output/bench_pipeline.csv
python3 examples/perf/plot_pipeline.py                       # figures from the CSVs
```

## Linear-solver matrix

`examples/perf/bench_linsolver_matrix.jl` benchmarks mechanism x linear solver
(GRI30/FFCM2/Aramco x KLU/UMFPACK/Sparspak/Pardiso/MUMPS) end to end, with a
standalone linear-solve micro-benchmark and trajectory-accuracy checks against a
reference:

```bash
julia --project=. examples/perf/bench_linsolver_matrix.jl --repeats 5
python3 examples/perf/plot_bench.py
```

Findings the bench encodes (run it for your own numbers):

- KLU is the best general choice at full scale.
- MUMPS requires `MPI.Init()` before use — the bench gates on it, because without
  it MUMPS fails *silently* (the solve returns Unstable with T stuck at T₀, not an
  error).
- When the Jacobian is dense-ish (the atmospheric box: 76% dense), a dense
  `LUFactorization()` can beat every sparse solver — FBDF's default sparse path pays
  a per-linear-solve `dropzeros` copy of the Newton matrix, measured at 69 GiB/day
  of pure allocation on that box (see the atmospheric tutorial).

## Jacobian strategy is problem-dependent

The analytic reaction-sharded Jacobian costs several times more to build but wins
at tight tolerance and long spans; at loose tolerance with many forced solver
restarts, the cheap-to-form FD Jacobian amortizes better. The atmospheric example
measured both back to back (8-day box): frozen at reltol 1e-6 — 820 s FD vs 61 s
analytic; diurnal at reltol 1e-4 with 11520 forced 60-s ticks — the analytic path
*loses* 6.6x on total time. `simulate`'s `jac`/`jac_strategy` options select the
path (see the API reference); the atmospheric driver picks per mode.

## Sparsity

`examples/perf/jacobian_sparsity_figure.jl` documents the Jacobian nonzero-pattern
workflow (`--sources sharded|mtk|bench`).
