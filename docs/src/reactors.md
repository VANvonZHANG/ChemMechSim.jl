# Reactor modes

`BatchReactor(mech; mode=...)` presets select which physics layers
(`MechanismConfig`) are attached during lowering. Four modes:

| mode | energy | constraint | EOS | state basis | what you get |
|---|---|---|---|---|---|
| `:kinetic` | isothermal | none | off | concentration | pure kinetics — T is a parameter |
| `:fixedT` | isothermal | none | ideal-gas | concentration | T a parameter, P an observed output |
| `:adiabatic_constV` | adiabatic | constant volume | ideal-gas | concentration | T a state, U = V·Σcᵢūᵢ(T) conserved |
| `:adiabatic_constP` | adiabatic | constant pressure | ideal-gas | moles | nᵢ + T states, V observed, H conserved |

The same table as keyword form: `MechanismConfig(energy=…, constraint=…, eos=…,
thermo_data=…, reverse_rate=…, state_basis=…)` — see the [API reference](@ref).

## `:kinetic` — the zero point

```julia
mech = Mechanism(
    species=[SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
    reactions=[ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                            kinetics=ElementaryArrhenius(2.0, 0.0, 0.0))])

reactor = BatchReactor(mech; mode=:kinetic, name=:decay)
sol = simulate(reactor, (0.0, 2.0); u0=Dict("A" => 1.0, "B" => 0.0))
```

No energy equation, no EOS: the ODEs are exactly the reaction rates. The demo
(`examples/demos/batch_reactor.jl`) prints the generated equations and checks the
analytic answer `A(2) = exp(-4)`.

## `:fixedT` — isothermal with pressure output

T stays a **parameter** (set it via `params`), and the ideal-gas EOS adds pressure
as an *observed* variable — no extra ODE:

```julia
r   = BatchReactor(mech; mode=:fixedT)
sys = extract_system(r)
Tp  = parameters(sys)[findfirst(p -> String(getname(p)) == "T", parameters(sys))]
sol = simulate(r, (0.0, 1e-3); u0=u0, params=[Tp => 1200.0], solver=Rodas5P(),
               reltol=1e-9, abstol=1e-12)
# P is observed: read it off the solution like any state
```

Source: `examples/demos/fixedT_reactor.jl` (H2 + OH ⇌ H + H2O with an explicit
reverse rate, P printed at the end).

## `:adiabatic_constV` — T as a state, U conserved

Temperature becomes an unknown driven by the energy balance; with no heat loss and
fixed V, `U/V = Σ cᵢ·ūᵢ(T)` is the conserved invariant (ū = h̄ − RT from the NASA7
thermo):

```julia
reactor = BatchReactor(mech; mode=:adiabatic_constV)
sol = simulate(reactor, (0.0, 10.0);
               u0=Dict("A"=>1.0, "B"=>0.0, "T"=>300.0),   # "T" seeds the state
               solver=Rodas5P(), reltol=1e-8, abstol=1e-10)
```

The demo (`examples/demos/adiabatic_reactor.jl`) builds an exothermic A → B with a
hand-written NASA7 pair and verifies ΔU/V ≈ 0.

## `:adiabatic_constP` — moles as states, H conserved

Constant pressure is lowered as a pure ODE: species states are **moles** `nᵢ`
(`u0` keys are `"n_A"`, `"n_B"`, …, plus `"T"`), volume is an observed output, and
the enthalpy `H = Σ nᵢ·h̄ᵢ(T)` is the conserved invariant:

```julia
reactor = BatchReactor(mech; mode=:adiabatic_constP)
sol = simulate(reactor, (0.0, 5.0);
               u0=Dict("n_A"=>1.0, "n_B"=>0.0, "T"=>800.0),
               solver=Rodas5P(), reltol=1e-8, abstol=1e-10)
```

Source: `examples/demos/adiabatic_constP_reactor.jl`.

## `u0` keys per mode

| mode | `u0` keys |
|---|---|
| `:kinetic` / `:fixedT` | species names → concentrations [mol/m³] |
| `:adiabatic_constV` | species names → concentrations, plus `"T"` → T₀ [K] |
| `:adiabatic_constP` | `"n_<species>"` → moles, plus `"T"` → T₀ [K] |

Pass a **complete** `u0`: `build_problem` does not default omitted species to zero —
missing initial conditions are filled by least squares and can hand unlisted species
arbitrary values (on one large mechanism this produced `NaN` on the first RHS call).
