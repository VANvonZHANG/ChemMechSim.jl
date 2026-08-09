#!/usr/bin/env python3
"""Cantera external baseline: cold (first) vs warm (second) solve time.

Runs the SAME ignition problem (const-V adiabatic CH4-air, T0=1500 K, P0=1 atm,
phi=1, 5 ms) through Cantera's ReactorNet for GRI30 and Aramco 3.0. Reports
cold + warm wall-clock time so the paper can compare:
  - Cantera cold (C++ pre-compiled — just load + solve)
  - Cantera warm (re-stepping — cheap)
  - ChemMechSim cold (Julia JIT compile + solve — the compile cost)
  - ChemMechSim warm (cached native code — integrate only)

Usage: python3 examples/perf/cantera_baseline.py
Outputs: examples/perf/output/bench_cantera.csv
"""
import cantera as ct
import time, csv, os, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
T0, P0 = 1500.0, ct.one_atm
T_END = 5.0e-3

MECHS = [
    ("gri30",  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mechanism", "gri30.yaml")),
    ("aramco", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mechanism", "AramcoMech3.0.yaml")),
]


def run_once(yaml_path):
    """Run const-V adiabatic ignition once, return (wall_s, n_steps, T_end, n_species)."""
    gas = ct.Solution(yaml_path)
    nsp = gas.n_species
    gas.TP = T0, P0
    gas.set_equivalence_ratio(1.0, "CH4", "O2:1,N2:3.76")
    reactor = ct.IdealGasReactor(gas, energy="on")
    reactor.volume = 1.0
    sim = ct.ReactorNet([reactor])
    n_steps = 0
    t0 = time.perf_counter()
    while sim.time < T_END:
        sim.step()
        n_steps += 1
    wall = time.perf_counter() - t0
    return wall, n_steps, reactor.T, nsp


os.makedirs(OUT_DIR, exist_ok=True)
out_csv = os.path.join(OUT_DIR, "bench_cantera.csv")
rows = []
for name, yaml in MECHS:
    print(f"=== {name} ===")
    cold_s, n_steps, T_end, nsp = run_once(yaml)
    print(f"  cold: {cold_s:.3f}s  {n_steps} steps  T_end={T_end:.1f} K")
    warm_s, _, _, _ = run_once(yaml)
    print(f"  warm: {warm_s:.3f}s")
    rows.append({"mech": name, "n_species": nsp, "cold_s": round(cold_s, 4),
                 "warm_s": round(warm_s, 4), "steps": n_steps, "T_end_K": round(T_end, 1)})

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mech", "n_species", "cold_s", "warm_s", "steps", "T_end_K"])
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {out_csv}")
print("\nCompare: Cantera cold ≈ warm (C++ pre-compiled, no JIT).")
print("ChemMechSim cold >> warm (Julia JIT compile of MTK-generated code).")
print("ChemMechSim warm ≈ Cantera warm (both run native code).")
