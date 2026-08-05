#!/usr/bin/env python3
"""Cantera reference solution for H2-O2 ignition (Phase 5a validation).

Outputs:
  h2o2_ref_constV.csv — constant-volume ignition (ct.IdealGasReactor)
  h2o2_ref_constP.csv — constant-pressure ignition (ct.IdealGasConstPressureReactor)

Columns: t, T, X_<species> for each species in the mechanism.

Run:  pip install cantera && python h2o2_ignition.py
"""
import cantera as ct
import numpy as np


# Use the same h2o2.yaml shipped with Cantera (identical schema to
# ChemMechSim/test/data/h2o2.yaml — both sourced from Cantera's data repo).
gas = ct.Solution("h2o2.yaml", "ohmech")

# IC matching the Julia test: H2:O2:N2 = 2:1:4, T=1000 K, P=1 atm.
gas.TPX = 1000.0, ct.one_atm, "H2:2, O2:1, N2:4"


def run_constV(t_end=1e-3, n=2001):
    """Constant-volume adiabatic ignition: ct.IdealGasReactor (energy='on')."""
    gas.TPX = 1000.0, ct.one_atm, "H2:2, O2:1, N2:4"
    reac = ct.IdealGasReactor(gas, energy="on")
    net = ct.ReactorNet([reac])
    times = np.linspace(0, t_end, n)
    cols = ["t", "T"] + ["X_" + s for s in gas.species_names]
    out = np.zeros((n, len(cols)))
    for i, t in enumerate(times):
        net.advance(t)
        out[i, 0] = t
        out[i, 1] = reac.T
        out[i, 2:] = gas.X
    return cols, out


def run_constP(t_end=1e-3, n=2001):
    """Constant-pressure adiabatic ignition: ct.IdealGasConstPressureReactor."""
    gas.TPX = 1000.0, ct.one_atm, "H2:2, O2:1, N2:4"
    reac = ct.IdealGasConstPressureReactor(gas, energy="on")
    net = ct.ReactorNet([reac])
    times = np.linspace(0, t_end, n)
    cols = ["t", "T"] + ["X_" + s for s in gas.species_names]
    out = np.zeros((n, len(cols)))
    for i, t in enumerate(times):
        net.advance(t)
        out[i, 0] = t
        out[i, 1] = reac.T
        out[i, 2:] = gas.X
    return cols, out


# Write both reference CSVs
colsV, outV = run_constV()
colsP, outP = run_constP()
np.savetxt("h2o2_ref_constV.csv", outV, delimiter=",", header=",".join(colsV), comments="")
np.savetxt("h2o2_ref_constP.csv", outP, delimiter=",", header=",".join(colsP), comments="")
print("Wrote h2o2_ref_constV.csv and h2o2_ref_constP.csv")
