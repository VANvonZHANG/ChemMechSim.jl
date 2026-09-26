# ChemMechSim.jl

**English** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/VANvonZHANG/ChemMechSim.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/VANvonZHANG/ChemMechSim.jl/actions/workflows/CI.yml)
[![Julia](https://img.shields.io/badge/Julia-1.12%2B-9558B2.svg)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)

An MTK-first, symbolically transparent, reactor-composable framework for gas-phase
chemical-kinetics modeling.

> **Status:** Usable. Cantera-YAML (subset) mechanism parsing, per-reaction lowering to
> unit-carrying ModelingToolkit `ODESystem`s, reaction-sharded analytic Jacobians, and the
> SciML solve chain are implemented and validated against Cantera. For a first tour of the
> API see `examples/demos/brusselator.jl`.

## Quick start

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using ChemMechSim, OrdinaryDiffEq

mech    = load_mechanism("examples/mechanism/gri30.yaml")   # Cantera-YAML → Mechanism
reactor = BatchReactor(mech; mode=:adiabatic_constV)        # convenience preset → MTK ODESystem

# Stoichiometric CH4-air at 1500 K, 1 atm (same setup as examples/validation/gri30_ignition.jl)
R, T0, P0 = 8.314, 1500.0, 101325.0
c_tot = P0 / (R * T0)
u0 = Dict("CH4" => c_tot / 10.52, "O2" => 2c_tot / 10.52, "N2" => 7.52c_tot / 10.52, "T" => T0)

sol = simulate(reactor, (0.0, 5e-3); u0 = u0, solver = FBDF(), reltol = 1e-8, abstol = 1e-12)
```

## Features

- **Data layer** (pure Julia): `SpeciesData`, `ReactionData`, `Mechanism`, and the
  `AbstractKinetics` hierarchy (elementary / third-body / Lindemann / Troe / PLOG +
  thermodynamic reverse rates)
- **Unit system**: `ChemUnits` (DynamicQuantities, SI/mol) with build-time dimensional checks
- **Layered interface**: `simulate` (one call) → `build_problem` (SciML `ODEProblem`) →
  `extract_system` (inspectable `ODESystem`) → `lower_to_mtk` (bare MTK primitives)
- **Rate-law extension protocol**: `struct + body + paramspec + needs_T` — write the formula
  once; the numeric and symbolic paths share it
- **Reactor modes**: `:kinetic` / `:fixedT` / `:adiabatic_constV` / `:adiabatic_constP`

## Performance

`examples/perf/bench_pipeline_stages.jl` (median of 3; Julia 1.12.7; const-V adiabatic
CH4-air ignition, FBDF @ reltol=1e-8/abstol=1e-12, reaction-sharded analytic Jacobian, KLU):

| Mechanism | build | JIT compile | cold solve | warm solve |
|---|---:|---:|---:|---:|
| GRI-Mech 3.0 (53 sp / 325 rxn) | 2.5 s | 9.9 s | 10.2 s | 0.27 s |
| FFCM 2.0 (96 sp / 1054 rxn) | 8.4 s | 31.2 s | 31.7 s | 0.55 s |
| Aramco 3.0 (581 sp / 3037 rxn) | 47.5 s | 197.9 s | 201.5 s | **3.55 s** |

- On the largest mechanism the **warm solve is faster than Cantera** (same tolerance:
  `IdealGasReactor` 3.87 s), with roughly half the integration steps (898 vs 1783).
- Cold solves are dominated by the one-time LLVM JIT compilation of generated code; warm
  solves reuse the compiled functions.
- Linear-solver comparison (KLU / UMFPACK / Sparspak / Pardiso / MUMPS):
  `examples/perf/bench_linsolver_matrix.jl`.

## Documentation

- [`examples/README.md`](examples/README.md) — guided tour: demo learning path, validation
  workflows vs Cantera, performance benchmarks
- User documentation site: landing with this release

## License

[MIT](LICENSE)
