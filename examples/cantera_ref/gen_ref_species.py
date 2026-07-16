#!/usr/bin/env python3
"""Generate Cantera reference CSVs with T + 5 species for validation figures.
Outputs: gri30_ref_species.csv, ffcm2_ref_species.csv
Run: python3 examples/cantera_ref/gen_ref_species.py"""
import cantera as ct
import numpy as np
import os

DIR = os.path.dirname(os.path.abspath(__file__))
SPECIES = ["CH4", "O2", "CO2", "OH", "H2O"]
HEADER = "time_s,T_K," + ",".join(f"X_{s}" for s in SPECIES)

def run(yaml_path, out_csv, label):
    gas = ct.Solution(yaml_path)
    gas.TP = 1500.0, ct.one_atm
    gas.set_equivalence_ratio(1.0, "CH4", "O2:1,N2:3.76")
    sidx = [gas.species_index(s) for s in SPECIES]

    reactor = ct.IdealGasReactor(gas, energy="on")
    reactor.volume = 1.0
    sim = ct.ReactorNet([reactor])

    rows = []
    t_end = 5.0e-3
    while sim.time < t_end:
        sim.step()
        Xs = gas.X[sidx]
        rows.append([sim.time, reactor.T] + list(Xs))

    out = np.array(rows)
    np.savetxt(out_csv, out, delimiter=",", header=HEADER, comments="", fmt="%.9e")
    print(f"wrote {out_csv} ({len(rows)} rows, T_max={max(out[:,1]):.1f} K)")

# GRI30 (Cantera built-in)
run("test/data/gri30.yaml", f"{DIR}/gri30_ref_species.csv", "GRI30")
# FFCM2 (project fixture)
run("examples/data/FFCM2.yaml", f"{DIR}/ffcm2_ref_species.csv", "FFCM2")
print("Cantera species refs generated.")
