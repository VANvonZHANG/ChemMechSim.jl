# Solver guide

How to pick the solver, Jacobian path, linear solver and tolerances for a
ChemMechSim problem. Everything here reflects measured behavior from the
repository's own benchmarks and examples (`examples/perf/`, the atmospheric box);
re-measure on your problem — the trade-offs are problem-dependent.

## The solver

```julia
sol = simulate(reactor, (0.0, 5e-3); u0 = u0, solver = FBDF(),
               reltol = 1e-8, abstol = 1e-12)
```

- **FBDF** is the workhorse for real mechanisms — every validation and benchmark
  run in this repository uses it. Any SciML ODE solver can be passed via `solver`;
  the small demos use Rodas5P.
- `reltol`/`abstol` and all other keywords forward to `solve`.
- When you supply an analytic Jacobian (below), disable ForwardDiff so it does not
  run anyway: `FBDF(autodiff = false)`.

## Jacobian options

`simulate` / `build_problem` accept:

| option | meaning |
|---|---|
| `jac = true` | assemble the analytic (reaction-sharded) Jacobian |
| `jac_strategy = :auto` | pick `:reaction_sharded` when the problem supports it, else fall back to `:none` (with a warning) |
| `jac_strategy = :reaction_sharded` | force the analytic path |
| `jac_strategy = :none` | finite-difference Jacobian |
| `jac_strategy = :mtk` | intentionally disabled — throws (unsuitable for large mechanisms) |
| `chunk_size`, `cse_chunk_size`, `write_chunk_size` | codegen chunking knobs for the analytic path |

**Analytic vs FD is problem-dependent, and the repository has both winners
measured back to back** (8-day atmospheric box, same mechanism):

- frozen mode, reltol 1e-6: analytic wins big — 820 s FD vs 61 s analytic
  (2-10x per-span solve speedups at ~11x build cost);
- diurnal mode, reltol 1e-4 with 11520 forced 60-s ticks: the analytic path
  *loses* ~6.6x on total time — the loose tolerance needs so few Newton
  iterations that the cheap-to-form FD Jacobian amortizes better.

Rule of thumb: tight tolerance and long uninterrupted spans favor the analytic
Jacobian; loose tolerance with many forced restarts favors FD.

## Linear solvers

Pass via the solver: `FBDF(linsolve = KLU())`, `FBDF(linsolve = LUFactorization())`,
… (any `LinearSolve.jl` algorithm; `LinearSolve` is **not** a package dependency —
`Pkg.add("LinearSolve")` it into your environment, likewise `MUMPS`/`Pardiso` when
benchmarking those). Measured guidance:

- **KLU** is the best general choice at full mechanism scale (581 sp) — the
  default in the pipeline benchmarks.
- **Dense LU** (`LUFactorization()`) wins when the Jacobian is dense-ish: on the
  atmospheric box (76% dense) FBDF's default sparse path pays a per-linear-solve
  `dropzeros` copy of the Newton matrix — measured 69 GiB/day of pure allocation —
  which dense LU eliminates while also running fastest.
- **UMFPACK** is what the large-mechanism species-export script uses
  (`FBDF(linsolve=UMFPACKFactorization())`): required for Aramco (581 sp) where
  the default *dense* Jacobian OOMs.
- **MUMPS** requires `MPI.Init()` **before** use — without it MUMPS fails
  *silently*: the solve returns `Unstable` with T stuck at T₀, not an error. The
  linsolver benchmark gates on this.
- The full mechanism x solver matrix (KLU/UMFPACK/Sparspak/Pardiso/MUMPS) with
  compile-vs-warm separation: `examples/perf/bench_linsolver_matrix.jl`.

## Tolerances

- Flat `abstol` hides trace species: anything below `abstol` is unresolved — the
  solver may return anything up to ~`abstol` there. The atmospheric example draws
  such species at the resolution line rather than trusting them.
- Resolving night-time radical troughs (~1e-17 mol/m³) needed per-state `abstol`
  of 1e-20..1e-22 there — which cost >94 min vs 7.5 min at flat 1e-12 and was
  abandoned. Decide deliberately what you need to resolve.
- Comparing against a reference code? Run both at the *same* tolerances — the
  validation workflow does exactly this so the comparison isolates the model, not
  the integrator.

## `u0` completeness (a real failure mode)

`build_problem` does not default omitted species to zero: partial `u0` makes MTK
solve an underdetermined initialization system by **least squares**, handing
unlisted species arbitrary values in ~[0, 1) — on one large mechanism this produced
`NaN` on the first RHS call. Build the full dict:

```julia
u0 = Dict(String(sp.name) => get(X_INIT, String(sp.name), 0.0) * c_air
          for sp in mech.species)
```

## Entry points

`simulate` (one call) → `build_problem` (SciML `ODEProblem`) → `extract_system`
(inspectable `ODESystem`) → `lower_to_mtk` (bare MTK primitives). Signatures and
keyword defaults: the [API reference](@ref).
