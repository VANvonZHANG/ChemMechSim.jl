# ChemMechSim.jl examples

Run any example from the repo root (`ChemMechSim.jl/`) with:

    julia --project=. examples/<dir>/<script>.jl

## Layout

| Dir | Purpose |
|---|---|
| [`demos/`](demos/) | Small toy-mechanism demos, one per framework feature. The recommended learning path (Phase 1 → extensibility). |
| [`validation/`](validation/) | Real-mechanism accuracy validation vs Cantera — Julia ignition scripts + the Python reference/figure workflow. |
| [`perf/`](perf/) | Performance benchmarks and large-mechanism coverage probes. |
| [`mechanism/`](mechanism/) | Mechanism fixtures (GRI30, H2-O2, FFCM2, Aramco 3.0). |

## demos/ — learning path

Each demo builds a tiny mechanism inline (no YAML) and exercises one layer of the
framework. Read in this order; each assumes the one before it.

| Script | Phase | Demonstrates |
|---|---|---|
| `brusselator.jl` | 1 | MVP: ODE system, limit cycle, optional plotting |
| `batch_reactor.jl` | 2 | `BatchReactor` script API + convenience modes |
| `mixed_lowering.jl` | 2.5a | Mixed Catalyst / direct MTK lowering, T-dependent Arrhenius |
| `h2o2_subset.jl` | 2.5b | H2-O2 subset, unit-aware mixed lowering |
| `fixedT_reactor.jl` | 3 | Isothermal reactor, thermodynamic reverse rate, EOS pressure output |
| `adiabatic_reactor.jl` | 4a | Const-V adiabatic, internal-energy conservation, T-as-state |
| `adiabatic_constP_reactor.jl` | 4b | Const-P adiabatic (path A, pure ODE), enthalpy conservation |
| `custom_ratelaw.jl` | ext. | User-defined rate law lowered with zero framework edits (L2 generic materializer) |

## validation/ — ChemMechSim vs Cantera

Two complementary workflows; both need `pip install cantera` (Cantera is **not** a CI
dependency — only the generated CSVs are committed/used).

**A. Per-mechanism ignition (interactive, single mechanism, CairoMakie plot):**

    cd examples/validation && python3 <mech>_ref.py && cd ../..   # generate the Cantera ref CSV
    julia --project=. examples/validation/<mech>_ignition.jl       # ChemMechSim solve + comparison plot

Scripts: `h2o2_ignition.jl` (P5a, const-V + const-P), `gri30_ignition.jl` (P5b),
`ffcm2_ignition.jl`, `aramco_ignition.jl`. Metric: ignition-delay relative diff at
max |dT/dt|.

**B. Combined-species validation (publication figures + error table):**

    python3 examples/validation/gen_ref_species.py                 # Cantera T + 5 species for all 3 mechs
    julia --project=. examples/validation/export_species.jl        # ChemMechSim side (jac=true + UMFPACK)
    python3 examples/validation/plot_validation.py                 # 3 single-mech figures + 1 combined + validation_errors.txt

See [`validation/README.md`](validation/README.md) for details.

> **PLOG rate validation** lives in the test suite, not here: `test/test_plog.jl`
> ("Phase 6 T5") compares `plog_rate` against Cantera kf committed in
> `test/data/plog_ref_rates.csv`, regenerable via `test/tools/plog_ref.py`.

## perf/

| Script | Phase | Measures |
|---|---|---|
| `gri30_benchmark.jl` | 5b | GRI30 lowering + Jacobian codegen + solve timing (cold vs warm) |
| `aramco_ffcm2_plog_load.jl` | 6 | Large-mechanism (Aramco/FFCM2) parse + lower coverage, all PLOG reactions |

## mechanism/

Published mechanisms used by the validation and perf scripts:

- `gri30.yaml` — GRI-30 (also mirrored in `test/data/` for the test suite)
- `h2o2.yaml` — H2-O2
- `FFCM2.yaml` — FFCM-2
- `AramcoMech3.0.{yaml,MECH,THERM,TRAN}` — Aramco 3.0 (Cantera YAML + original CHEMKIN files)
