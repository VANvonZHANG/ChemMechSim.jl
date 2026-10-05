# ChemMechSim.jl

**English** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/VANvonZHANG/ChemMechSim.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/VANvonZHANG/ChemMechSim.jl/actions/workflows/CI.yml)
[![Docs](https://github.com/VANvonZHANG/ChemMechSim.jl/actions/workflows/Documenter.yml/badge.svg)](https://vanvonzhang.github.io/ChemMechSim.jl/)
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
using Pkg; Pkg.add(["ChemMechSim", "OrdinaryDiffEq"])  # OrdinaryDiffEq provides FBDF
using ChemMechSim, OrdinaryDiffEq

# GRI-Mech 3.0 ships inside the package tarball:
mech    = load_mechanism(joinpath(pkgdir(ChemMechSim), "examples", "mechanism", "gri30.yaml"))  # Cantera-YAML → Mechanism
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

## Related packages

[ReactionMechanismSimulator.jl](https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl) (RMS) is a registered, actively maintained Julia package that also simulates large gas-phase mechanisms read from Cantera YAML — if it already fits your workflow, keep using it. ChemMechSim.jl exists because we wanted the mechanism itself to be a ModelingToolkit/Catalyst `ODESystem` as the primary artifact — inspectable, symbolically transformable equations that compose with the wider SciML ecosystem — and that representation is architectural, so we built a new package rather than rearchitecting an existing one. On that foundation ChemMechSim.jl offers:

- a unit-aware data layer (DynamicQuantities) with dimension checks at construction time;
- an extensible rate-law protocol (`rate_type_handlers`) — custom rate types without forking the package;
- explicit reactor-physics presets (`:kinetic` / `:fixedT` / `:adiabatic_constV` / `:adiabatic_constP`) with conserved invariants (U, H);
- selectable Jacobian strategies for stiff, large mechanisms (validated to Aramco 3.0, 581 species).

v1.0.0 targets gas-phase chemistry; multiphase and multi-state reactor models are planned extensions.

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
- User documentation: <https://vanvonzhang.github.io/ChemMechSim.jl> — getting started
  and the full API reference

## Third-party data

`examples/mechanism/` redistributes third-party reaction mechanisms **unmodified and with
attribution, for validation and example use**. The MIT license of this repository covers
its original code and documentation only — the mechanism files below are excluded from
that grant and remain subject to their respective sources. All other fixtures
(e.g. `brusselator.yaml`) are original toy mechanisms of this repository.

| Files | Source | Attribution |
|---|---|---|
| `gri30.yaml` | GRI-Mech 3.0, vendored via Cantera's `ck2yaml` conversion (Cantera is BSD-3-Clause) | Cite as the [GRI-Mech site](http://combustion.berkeley.edu/gri-mech/version30/text30.html) requests: *Gregory P. Smith, David M. Golden, Michael Frenklach, Nigel W. Moriarty, Boris Eiteneer, Mikhail Goldenberg, C. Thomas Bowman, Ronald K. Hanson, Soonho Song, William C. Gardiner, Jr., Vitali V. Lissianski, and Zhiwei Qin*, http://www.me.berkeley.edu/gri_mech/ |
| `h2o2.yaml`, `test/data/h2o2.yaml` | Cantera `data/h2o2.yaml` (H2-O2 submechanism of GRI-Mech 3.0) | BSD-3-Clause (Cantera); distributed unchanged |
| `FFCM2.yaml` | [FFCM-2](https://web.stanford.edu/group/haiwanglab/FFCM2/), July 2023 release, © Stanford Foundational Fuel Chemistry Model Initiative | Citation required: *Y. Zhang, W. Dong, L. Vandewalle, R. Xu, G.P. Smith and H. Wang, "Foundational Fuel Chemistry Model Version 2.0 (FFCM-2)", https://web.stanford.edu/group/haiwanglab/FFCM2, 2023* (also embedded in the file header) |
| `AramcoMech3.0.{MECH,THERM,TRAN}` (+ `AramcoMech3.0.yaml`, our `ck2yaml` conversion) | [Combustion Chemistry Centre, University of Galway](https://c3.universityofgalway.ie/combustionchemistrycentre/mechanismdownloads/), AramcoMech 3.0 (2018) | The download site provides no license for these files; they are redistributed here unmodified with attribution. Cite the source publication: *C-W. Zhou, Y. Li, U. Burke, C. Banyon, K.P. Somers, S. Khan, J.W. Hargis, T. Sikes, E.L. Petersen, M. AlAbbad, A. Farooq, Y. Pan, Y. Zhang, Z. Huang, J. Lopez, Z. Loparo, S.S. Vasu, H.J. Curran, "An experimental and chemical kinetic modeling study of 1,3-butadiene combustion: Ignition delay time and laminar flame speed measurements", Combustion and Flame 197 (2018) 423-438* ([doi](https://doi.org/10.1016/j.combustflame.2018.08.006)) |

## License

[MIT](LICENSE)
