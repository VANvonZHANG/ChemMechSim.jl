# Atmospheric box tutorial

An atmospheric box model on the MCM alkanes/alkenes subset: 1843 species (one a
phantom `M`), 5600 reactions, isothermal at 298 K, fixed pressure — integrated for
8 days in two photolysis modes, both reading the 24 MB source mechanism **directly**
(no preprocessor, no derived mechanism, no sidecar).

## Why `:kinetic` is the only correct mode here

```julia
reactor = BatchReactor(mech; mode=:kinetic, checks=false)
```

| `:kinetic` layer | why it matches an atmospheric box |
|---|---|
| `energy = :isothermal` | T is given by the scenario (298 K), never solved |
| `constraint = :none` | no volume to solve for |
| `eos = :off` | P is given (102858 Pa); state basis is concentration |
| `reverse_rate = :irreversible` | MCM emits only forward rates; its thermo is uniformly zero |
| `state_basis = :concentration` | chemistry parameterised in number density |

`:fixedT`, `:adiabatic_constV`, `:adiabatic_constP` each inject an energy equation
and/or an EOS this problem does not have. `checks=false` is **required**, not an
optimisation: MTK's unit validator cannot fold this mechanism's equations (they are
dimensionally correct; the checker cannot prove it in reasonable time).

## Run it

The source mechanism is not committed — copy it in from the converter, then:

```bash
julia --project=examples examples/atmospheric/mcm_box.jl frozen     # J at overhead sun (perpetual day)
julia --project=examples examples/atmospheric/mcm_box.jl diurnal    # photolysis driven by the zenith clock
julia --project=examples examples/atmospheric/tools/budget.jl       # O3/HOx rate budgets
julia --project=examples examples/atmospheric/tools/bench_jac.jl    # Jacobian-strategy A/B
python3 examples/atmospheric/tools/figures/fig_series.py     # series_frozen.png + series_diurnal.png
```

The two MCM rate types ChemMechSim's parser does not know (`zenith-angle-photolysis`,
`sigmoid-branching`) are parsed by example-side kinetics types handed to
`load_mechanism` through `rate_type_handlers` (see [Mechanism format](@ref)). Each
photolysis A-factor materializes as the symbolic parameter `k_{j}_A` — that one
parameter is what makes a single mechanism serve both modes: frozen solves with its
default (J at overhead sun), diurnal drives it from a `DiscreteCallback` on a 60-s
grid through `setp` setters, solver never restarted.

## What the runs show

- **frozen**: the box relaxes toward a photochemical steady state — O3 falls
  monotonically 30 → 4.7 ppbv over 8 days, NO2 exhausts, the radical pool builds to
  a steady level. Perpetual noon, so no night-only chemistry (NO3 / N2O5) appears.
- **diurnal**: the same box with photolysis on the zenith clock — O3 30 → 18.9 ppbv
  with a daily sawtooth (perpetual noon eats O3 visibly faster than 12 h/day), CH4
  declines in daylight-only staircase steps, OH/HO2 pulse sun-synchronously.

## The limits — read before quoting numbers

1. Frozen is a perpetual noon; everything from `output/frozen/` is a perpetual-day
   result.
2. Every species has `molecular_weight == 0` (converter emits empty compositions) —
   concentrations only, no mass-based output.
3. The scenario initialises only N2/O2/H2O/O3/NO2/CH4 — the box is effectively a
   CH4-NOx-O3 system; the organic degradation chemistry stays largely idle.
   Initialise a VOC (C2H6, C3H8, BUT1ENE) to exercise it.
4. The diurnal run uses reltol 1e-4 / flat abstol 1e-12: night-time radical troughs
   and all of NO3 sit **below** abstol — those parts of the figure are bounds at the
   resolution line, not physics. Resolving them needs per-state abstol ~1e-22.

## Cost (order of magnitude — a shared box, seconds move 3-6x with load)

| stage | measured |
|---|---|
| parse the source directly (incl. both MCM types) | ~22 s |
| lower (`checks=false`) | ~52 s |
| `build_problem` frozen (analytic Jacobian) | ~268 s |
| solve 8 days frozen (reltol 1e-6, analytic + dense LU) | ~56 s |
| solve 8 days diurnal (reltol 1e-4, 11520 forced ticks, FD) | ~81 s |

Full story, the API traps, and the de-risking evidence:
`examples/atmospheric/README.md` (the file this page condenses).
