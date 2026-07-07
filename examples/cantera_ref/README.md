# Cantera Reference Solutions (Phase 5a)

These scripts generate Cantera reference ignition curves for validating
ChemMechSim's H2-O2 simulation. **Not run in CI** (requires Python + Cantera).

## Setup

    pip install cantera

## Generate reference CSVs

    cd examples/cantera_ref
    python h2o2_ignition.py

Produces `h2o2_ref_constV.csv` and `h2o2_ref_constP.csv` (ignored by git).

## Compare with ChemMechSim

    cd ../..   # back to ChemMechSim.jl/
    julia --project=. examples/h2o2_ignition.jl

This reads the YAML, simulates both reactors, loads the reference CSVs (if present),
and reports ignition-delay comparison + plots.

## Ignition-delay metric

`t_ignition` = time of maximum `dT/dt` (most robust against noise).
Tolerance: const-V < 5%, const-P < 8% (spec §9).
