# ChemMechSim.jl

An MTK-first, symbolically transparent, reactor-composable framework for gas-phase
chemical-kinetics modeling, built on ModelingToolkit, Catalyst, DynamicQuantities, and
the SciML solve chain.

> **Status:** Usable. Cantera-YAML (subset) mechanism parsing, per-reaction lowering to
> unit-carrying ModelingToolkit `ODESystem`s, reaction-sharded analytic Jacobians, and the
> SciML solve chain are implemented and validated against Cantera.

## Related packages

[ReactionMechanismSimulator.jl](https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl) (RMS) is a registered, actively maintained Julia package that also simulates large gas-phase mechanisms read from Cantera YAML — if it already fits your workflow, keep using it. ChemMechSim.jl differs architecturally: the mechanism lowers to a ModelingToolkit/Catalyst `ODESystem` as the primary artifact — composable with the SciML ecosystem, on a unit-aware data layer with an extensible rate-law protocol.

## What you can do

- **Parse** published mechanisms from Cantera-YAML — GRI-Mech 3.0, FFCM-2, Aramco 3.0,
  H2-O2 and custom subsets.
- **Inspect what you solve**: every reaction lowers to a symbolic rate law with unit
  metadata attached at build time; extract the `ODESystem` and read it.
- **Compose reactor modes**: `:kinetic`, `:fixedT`, `:adiabatic_constV`,
  `:adiabatic_constP` via `BatchReactor` / `convenience_config`.
- **Extend the rate laws**: a new `AbstractKinetics` subtype lowers with zero framework
  edits once `struct + body + paramspec + needs_T` are defined.
- **Validate against Cantera** with the workflows in `examples/validation/`.

## Performance snapshot

`examples/perf/bench_pipeline_stages.jl` (median of 3; Julia 1.12.7; const-V adiabatic
CH4-air ignition, FBDF @ reltol=1e-8/abstol=1e-12, reaction-sharded analytic Jacobian,
KLU):

| Mechanism | build | JIT compile | cold solve | warm solve |
|---|---:|---:|---:|---:|
| GRI-Mech 3.0 (53 sp / 325 rxn) | 2.5 s | 9.9 s | 10.2 s | 0.27 s |
| FFCM 2.0 (96 sp / 1054 rxn) | 8.4 s | 31.2 s | 31.7 s | 0.55 s |
| Aramco 3.0 (581 sp / 3037 rxn) | 47.5 s | 197.9 s | 201.5 s | **3.55 s** |

On the largest mechanism the warm solve is faster than Cantera (`IdealGasReactor` 3.87 s)
with roughly half the integration steps.

## First steps

- [Getting started](@ref) — installation and a first CH4-air ignition run.
- [API reference](@ref) — every exported symbol, grouped as in the source layout.
- `examples/README.md` (in the repository) — the guided tour: demo learning path,
  validation workflows vs Cantera, performance benchmarks.

MIT licensed. See `LICENSE` in the repository.
