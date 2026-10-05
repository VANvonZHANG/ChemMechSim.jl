# Getting started

## Prerequisites

- Julia **1.12** or later.

## Installation

```julia
using Pkg
Pkg.add(["ChemMechSim", "OrdinaryDiffEq"])  # OrdinaryDiffEq provides the solvers (FBDF, Rodas5P, …)
```

To work on the package itself, use a development clone instead:

```bash
git clone https://github.com/VANvonZHANG/ChemMechSim.jl
cd ChemMechSim.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## First run: CH4-air ignition delay

Mechanism fixtures ship inside the package tarball, so the example works identically
from an installed package and from a development clone:

```julia
using ChemMechSim, OrdinaryDiffEq

mech    = load_mechanism(joinpath(pkgdir(ChemMechSim), "examples", "mechanism", "gri30.yaml"))  # Cantera-YAML → Mechanism
reactor = BatchReactor(mech; mode=:adiabatic_constV)        # convenience preset → MTK ODESystem

# Stoichiometric CH4-air at 1500 K, 1 atm (same setup as examples/validation/gri30_ignition.jl)
R, T0, P0 = 8.314, 1500.0, 101325.0
c_tot = P0 / (R * T0)
u0 = Dict("CH4" => c_tot / 10.52, "O2" => 2c_tot / 10.52, "N2" => 7.52c_tot / 10.52, "T" => T0)

sol = simulate(reactor, (0.0, 5e-3); u0 = u0, solver = FBDF(), reltol = 1e-8, abstol = 1e-12)
```

The returned `sol` is a standard SciML solution — index states by name (`sol[sys.CH4]`),
plot with your favourite backend, or read the ignition delay as the time of maximum
`|dT/dt|` (the validation scripts in `examples/validation/` do exactly this).

## Where to go next

All `examples/` paths below are relative to the repository root; installed copies
live under `pkgdir(ChemMechSim)`.

- [API reference](@ref) for every exported symbol.
- `examples/demos/brusselator.jl` — a first tour of the three input routes
  (programmatic / Catalyst import / YAML) on a tiny toy mechanism.
- `examples/README.md` — the full layout: demos learning path, validation workflows
  vs Cantera, performance benchmarks, mechanism fixtures.
- Custom rate laws: `examples/demos/custom_ratelaw.jl` (the `struct + body + paramspec
  + needs_T` protocol, worked end to end).
