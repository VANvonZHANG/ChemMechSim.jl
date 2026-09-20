# Atmospheric chemistry example — MCM alkanes/alkenes

An atmospheric box model built on the MCM alkanes/alkenes subset: 1842 species, 5600
reactions, isothermal at 298 K, fixed pressure, integrated for 3 days.

## Why `:kinetic` mode

The reactor is built with `BatchReactor(mech; mode=:kinetic, checks=false)`. `:kinetic` is the
`MechanismConfig()` zero point, and for an atmospheric box model it is not merely sufficient —
the other modes are wrong:

| `:kinetic` field | why it matches an atmospheric box |
|---|---|
| `energy = :isothermal` | T is given by the scenario (298 K), never solved from an energy balance |
| `constraint = :none` | there is no volume to solve for; no constraint layer is wanted |
| `eos = :off` | P is given (102858 Pa) and the state basis is concentration, so no pressure is ever derived |
| `reverse_rate = :irreversible` | MCM emits only forward rates (every reaction is `=>`), and its thermo data is uniformly zero, so `K_c` could not be formed anyway |
| `state_basis = :concentration` | the chemistry is parameterised in number density |

`:fixedT`, `:adiabatic_constV` and `:adiabatic_constP` each inject an energy equation and/or an
EOS that this problem does not have.

## How to run it

The 24 MB source mechanism is **not committed**. Copy it in from the converter:

```bash
cp <kpp-cantera-converter>/examples/mcm/mcm_alkanes_alkenes_converted.yaml \
   examples/atmospheric/
```

Then:

```bash
julia --project=. examples/atmospheric/tools/flatten_photolysis.jl   # ~20 s
julia --project=. examples/atmospheric/mcm_box.jl                    # ~6 min
```

`flatten_photolysis.jl` takes an optional solar zenith in degrees (default `0`, overhead sun)
and writes the derived mechanism to `output/` (gitignored, like the source).

**Expected cost, so you don't think it hung** (measured on this machine):

| stage | quiet machine | busy machine |
|---|---|---|
| parse the derived mechanism | ~24 s | ~100 s |
| lower it (`checks=false`) | ~56 s, peak ~3.5 GiB | ~340 s, ~3.7 GiB |
| integrate 3 days | ~300 s | ~1080 s |

The two columns are the same code on the same machine — this box is shared, and another user
running WRF benchmarks at load average 70–100 inflates every stage by 3–6×. Treat these as an
order of magnitude, not a benchmark.

`checks=false` is **required**, not an optimisation: with `checks=true` MTK's unit validator
cannot fold this mechanism's equations and lowering did not finish in 16 minutes. The equations
are dimensionally correct; the checker just cannot prove it.

## The limits — read these before drawing conclusions

1. **Photolysis is frozen, so this is a perpetual day.** The 1041 `zenith-angle-photolysis`
   reactions are evaluated once at a fixed zenith χ₀ and emitted as constant Arrhenius rates.
   The diurnal cycle is not represented, so **night-only chemistry does not appear** — in
   particular NO₃ and N₂O₅ will not accumulate overnight, and the species settle toward a
   steady daytime state instead of cycling. This is the single biggest fidelity caveat here and
   is the direct cost of the preprocessor approach. Re-running with a different `χ₀` gives a
   different frozen-noon box, not a better one.

2. **The 2 `sigmoid-branching` entries are exact only at T₀ = 298 K.** Each is flattened to a
   constant, evaluated at T₀. That is exact at the operating point, but it is not a temperature
   law — if you change T, re-run the preprocessor.

   Note these two entries are *not* a standalone pair. They are the signed correction and the
   σ channel of `CH3O2 + HO2`; the third, plain-Arrhenius entry carrying `k_total` is passed
   through untouched, and the three sum to MCM's own rate. (An earlier version of the
   preprocessor paired the two sigmoids and invented a `k_total` term, running that reaction at
   2× MCM's rate. See the warning in `_flatten_sigmoid`.)

3. **Every species has `molecular_weight == 0`** — the converter emits `composition: {}` for all
   of them. Harmless in `state_basis=:concentration`, but it means mass-based or mixing-ratio
   output is unavailable. Concentrations only.

4. **The derived mechanism is a build artifact, not a source of truth.** It is gitignored, and
   it drops the source file's ~16 800 provenance comment lines. Regenerate it rather than
   editing it.

5. **The scenario contains no alkanes or alkenes.** `X_INIT` is copied verbatim from the
   reference config, and it initialises only N2/O2/H2O/O3/NO2/CH4 — so despite the mechanism's
   name, the box is effectively a CH4–NOx–O3 system and the mechanism's organic degradation
   chemistry stays largely idle. This is faithful to the reference scenario (which is why it is
   kept), but it is not a demonstration of the alkane/alkene chemistry. Initialise a VOC (e.g.
   `C2H6`, `C3H8`, `BUT1ENE`) to exercise that.

   What the run *does* demonstrate, and is checked: OH/HO2 rise from zero and NO appears from NO2
   photolysis — so the flattened photolysis really is driving the radical chemistry rather than
   being a silent no-op.

   O3 also **falls** (1.245e-6 → 6.188e-7 mol/m³ over 3 days; OH peaks at 2.26e-11). That is
   *not* NOx titration: total NOx is 1e-10 mole fraction ≈ 4.2e-9 mol/m³, some 150× smaller than
   the O3 decline. It is the HOx cycle — O3 photolysis (`O3 => O1D` at 3.78e-5 s⁻¹, ~12% of
   which gives OH) plus the 1% H2O, then OH + O3. Photolysis-driven loss, which is the point the
   example is making.

## Unrelated known trap: pass a *complete* `u0`

`build_problem` does not default omitted species to zero. If you pass a partial `u0`, MTK treats
the missing initial conditions as an underdetermined initialization system and fills them by
**least squares**, handing unlisted species arbitrary values in ~[0, 1). On this mechanism that
produced a `NaN` RHS and `retcode = DtNaN` on the first solve.

Build the full dict, as the other large-mechanism examples already do
(`gri30_ignition.jl:28`, `aramco_ignition.jl:39`, `ffcm2_ignition.jl:28`):

```julia
u0 = Dict(String(sp.name) => get(X_INIT, String(sp.name), 0.0) * c_air for sp in mech.species)
```

## Upstream dependencies

This example needs two fixes that are not on `main` yet:

- **`atmospheric-parser-and-jac`** (PR #36) — without it `load_mechanism` cannot read the
  KPP/MCM YAML dialect at all.
- **`meff-sharing`** (PR #35) — without it lowering this mechanism takes ~33 minutes instead of
  under a minute, because the 1269 third-body/falloff reactions each get their own `M_eff`
  algebraic variable instead of sharing one per distinct efficiency vector.

The branch this example lives on has both merged in.
