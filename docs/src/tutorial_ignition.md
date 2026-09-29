# Ignition tutorial

From a fresh clone to a measured CH4-air ignition delay on GRI-Mech 3.0 — the same
condition the validation workflow (`examples/validation/gri30_ignition.jl`) uses.

## The setup

Stoichiometric CH4-air at 1500 K, 1 atm in an adiabatic constant-volume batch
reactor. Stoichiometric CH4-air is 1 part CH4 per 2 O2 + 7.52 N2, i.e. per 10.52
parts mixture:

```julia
using ChemMechSim, OrdinaryDiffEq

mech    = load_mechanism("examples/mechanism/gri30.yaml")   # Cantera-YAML → Mechanism
reactor = BatchReactor(mech; mode=:adiabatic_constV)        # preset → MTK ODESystem

R, T0, P0 = 8.314, 1500.0, 101325.0
c_tot = P0 / (R * T0)                                       # total concentration [mol/m³]
u0 = Dict("CH4" => c_tot / 10.52, "O2" => 2c_tot / 10.52,
          "N2" => 7.52c_tot / 10.52, "T" => T0)             # "T" seeds the energy state
```

## Solve

```julia
sol = simulate(reactor, (0.0, 5e-3); u0 = u0, solver = FBDF(),
               reltol = 1e-8, abstol = 1e-12)
```

`:adiabatic_constV` makes temperature a *state*: the mechanism's reactions release
heat, T rises past ignition, and the internal energy `V·Σ cᵢ·ūᵢ(T)` is the conserved
invariant. The solve takes seconds warm; the first run pays one-time JIT compilation
of the generated code — `examples/perf/bench_pipeline_stages.jl` measures that split
per mechanism.

## Reading the ignition delay

The standard metric (used by every script in `examples/validation/`) is the time of
maximum `|dT/dt|` — robust against solver noise:

```julia
using ModelingToolkit: getname, unknowns
sys  = extract_system(reactor)
Tvar = unknowns(sys)[findfirst(s -> String(getname(s)) == "T", unknowns(sys))]
Ts   = [sol(t; idxs=Tvar) for t in sol.t]
dTdt = diff(Ts) ./ diff(sol.t)
t_ign = sol.t[argmax(dTdt) + 1]
```

On this condition ChemMechSim measures **t_ign ≈ 1.1 ms** (the test suite pins the
same run: GRI30 CH4-air const-V, FBDF @ reltol 1e-8 / abstol 1e-12, with relative
internal-energy drift ~1e-7).

> The comparison against Cantera on the same condition lives in
> `examples/validation/` — workflow A measures exactly this ignition delay on both
> codes at the same tolerances (see that directory's README).

## What happened under the hood

One call, four layers, each inspectable:

1. `load_mechanism` parsed the Cantera-YAML into a pure-Julia `Mechanism`
   (no MTK types).
2. `BatchReactor(mech; mode=:adiabatic_constV)` wrapped a `convenience_config`
   preset into a `ChemPhaseSystem`.
3. `simulate` called `build_problem` → `lower_to_mtk` — every reaction lowered to a
   symbolic rate law with unit metadata, energy equation and ideal-gas EOS attached.
4. SciML solved the resulting `ODEProblem` with FBDF.

Swap step 3 for `extract_system(reactor)` to read the actual `ODESystem` equations —
see the [API reference](@ref) for every entry point.

## Building the mechanism inline instead

For toy mechanisms you can skip YAML entirely and construct the data layer
directly — `examples/demos/h2o2_subset.jl` builds an H2-O2 subset with an
elementary and a Troe-falloff reaction in ~10 lines:

```julia
mech = Mechanism(species=[H2, O2, H2O, H, O, OH, HO2], reactions=[
    ReactionData(reactants=Dict(1=>1.0, 2=>1.0), products=Dict(6=>2.0),
                 kinetics=ElementaryArrhenius(1.0e6, 0.0, 0.0)),
    ReactionData(reactants=Dict(4=>1.0, 2=>1.0), products=Dict(7=>1.0),
                 kinetics=TroeFalloff(ElementaryArrhenius(1.0e9, 0.0, 0.0),   # low-P
                                      ElementaryArrhenius(1.0e6, 0.0, 0.0),   # high-P
                                      allM, TroeParams(0.5, 1e-30, 1e30, 1e30))),
])
```

## Next

- [Reactor modes](@ref) — what each of the four presets does.
- [Mechanism format](@ref) — what `load_mechanism` reads.
- `examples/demos/` — the full learning path, one demo per framework feature.
