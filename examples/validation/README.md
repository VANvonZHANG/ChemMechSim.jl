# Validation — ChemMechSim.jl vs Cantera

Scripts that validate ChemMechSim against [Cantera](https://cantera.org/) on real
mechanisms (GRI30, H2-O2, FFCM2, Aramco 3.0). **Cantera is not a CI dependency** —
the Python scripts here generate reference CSVs that are then compared (offline) with
ChemMechSim output. All generated artifacts (CSVs, figures, `validation_errors.txt`)
are written under `output/` and gitignored; `output/` is created on first run.

## Setup

    pip install cantera

## Two workflows

### A. Per-mechanism ignition (interactive, single mechanism)

Each `<mech>_ref.py` generates a Cantera const-V (and for H2-O2, const-P) ignition
CSV; the matching `<mech>_ignition.jl` runs ChemMechSim on the same condition, reports
the ignition-delay relative difference, and saves a CairoMakie comparison plot.

    python3 examples/validation/h2o2_ref.py        # or gri30_ref.py / ffcm2_ref.py / aramco_ref.py
    julia --project=. examples/validation/h2o2_ignition.jl

| Mechanism | Ref script | Julia script | Tol (Δt_ign) |
|---|---|---|---|
| H2-O2 (`h2o2.yaml`) | `h2o2_ignition.py` | `h2o2_ignition.jl` | const-V 5%, const-P 8% |
| GRI-30 | `gri30_ref.py` | `gri30_ignition.jl` | 10% |
| FFCM-2 | `ffcm2_ref.py` | `ffcm2_ignition.jl` | 2% |
| Aramco 3.0 | `aramco_ref.py` | `aramco_ignition.jl` | 2% |

**Metric:** `t_ignition` = time of maximum |dT/dt| (robust against noise).

### B. Combined-species validation (publication figures + error table)

A three-step pipeline over GRI30 / FFCM2 / Aramco producing publication-grade figures
and a quantitative error table (`validation_errors.txt`):

    # 1. Cantera side: T + 5 species (CH4, O2, CO2, OH, H2O) trajectories
    python3 examples/validation/gen_ref_species.py
    # 2. ChemMechSim side: same species on the same time grid
    julia --project=. examples/validation/export_species.jl
    # 3. Figures + error table
    python3 examples/validation/plot_validation.py

Outputs (gitignored, under `output/`): `fig_{gri30,ffcm2,aramco,combined}_validation.{svg,pdf,png}`
and `validation_errors.txt` (Δt_ign % + max ΔX per species, per mechanism). Workflow A's
per-mech ref CSVs and ignition PNGs also land in `output/`.

`export_species.jl` uses `jac=true` + `FBDF(linsolve=UMFPACKFactorization())` uniformly
across all mechanisms — required for Aramco (581 sp) where the default dense Jacobian
OOMs; see `examples/perf/` and the linsolve probe for the 35× KLU→UMFPACK finding.

## Files

```
gen_ref_species.py     Cantera T + 5 species for all 3 mechs (workflow B, step 1)
{gri30,ffcm2,aramco}_ref.py   Per-mech Cantera const-V ignition CSV (workflow A)
h2o2_ignition.py       Cantera H2-O2 const-V + const-P ignition CSVs (workflow A)
export_species.jl      ChemMechSim species export (workflow B, step 2)
{h2o2,gri30,ffcm2,aramco}_ignition.jl   ChemMechSim per-mech ignition + plot (workflow A)
plot_validation.py     Combined figures + error table (workflow B, step 3)
```

## PLOG rate validation

Not in this directory — PLOG `kf` is validated in the **test suite**
(`test/test_plog.jl`, "Phase 6 T5: PLOG rate vs Cantera") against
`test/data/plog_ref_rates.csv`. The CSV is regenerable via `test/tools/plog_ref.py`
(run manually with Cantera installed).
