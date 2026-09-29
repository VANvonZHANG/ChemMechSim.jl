# Rate-law extensibility

A new rate law lowers into a simulable MTK system with **zero framework edits**:
define the struct, the formula, and the parameter roles. The formula is written
**once** — the numeric and symbolic paths share it.

## The contract

Four definitions, then it behaves like any built-in law:

```julia
using ChemMechSim: afactor, ktemp, plain, paramspec, body, needs_T

# 1. The type and its formula
struct MyArrhenius <: AbstractKinetics
    A::Float64; b::Float64; Ea::Float64; f::Float64        # k(T) = f·A·T^b·exp(-Ea/RT)
end
_my_arrhenius_body(A, b, θ, f, T) = f * A * T^b * exp(-θ / T)

# 2. Parameter roles + body + T-dependence — the whole "lowering" contract.
#    These extend ChemMechSim's generic functions, so qualify them:
ChemMechSim.paramspec(kin::MyArrhenius) = (afactor(:A, "", kin.b), plain(:b),
                                           ktemp(:Ea, ""), plain(:f))
ChemMechSim.body(kin::MyArrhenius)      = _my_arrhenius_body
ChemMechSim.needs_T(kin::MyArrhenius)   = true
```

## Parameter roles

`paramspec` declares how each parameter enters the lowering's unit bookkeeping:

| role | meaning |
|---|---|
| `afactor(name, unit, b)` | the Arrhenius A-factor — log-converted at lowering, carrying its exponent `b` |
| `ktemp(name, unit)` | an activation energy expressed as θ = Ea/R in K |
| `kvalue(name, unit)` | a plain temperature value in K (e.g. Troe's T1/T2/T3) |
| `plain(name)` | dimensionless |

`rate_param(name, default, unit)` is the lowering-side declaration that pairs with
these roles. Full signatures: the [API reference](@ref).

## Use it like any built-in

```julia
mech = Mechanism(;
    species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
    reactions = [ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                              kinetics=MyArrhenius(1.0, 0.5, 5000.0, 2.0),
                              reverse_policy=Irreversible())])
phase = ChemMechSim.ChemPhaseSystem(mech)
sol = simulate(phase, (0.0, 1.0); u0=Dict("A"=>1.0, "B"=>0.0),
               params=[Tparam => 1000.0], reltol=1e-9, abstol=1e-12)
```

`needs_T=true` auto-creates T as a *parameter* (default 300 K) — override it via
`params`, exactly like `:fixedT`. Worked example, runnable end to end:
`examples/demos/custom_ratelaw.jl`.

## How it lowers

The generic fallback `symbolic_kf(::AbstractKinetics, ctx)` (the "L2 generic
materializer") reads `paramspec` to create symbolic parameters with the right unit
metadata, then calls `body` to build the rate expression — the same `body` function
serves the numeric path (`rate_constant`). No dispatch table to register, no
lowering code to touch: declare the contract and the machinery applies.

## When to use a custom law vs `rate_type_handlers`

- Custom **law type** (this page): the physics is new — a functional form the
  built-ins do not cover.
- Custom **parser** (`rate_type_handlers`, see [Mechanism format](@ref)): the YAML
  carries a type string the parser does not know, but the law itself is expressible
  with existing kinetics types (or a small custom type — the MCM example combines
  both).
