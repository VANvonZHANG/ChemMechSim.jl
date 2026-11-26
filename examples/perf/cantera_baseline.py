#!/usr/bin/env python3
"""Cantera external baseline: cold (first) vs warm (second) solve time.

Runs the SAME ignition problem (const-V adiabatic CH4-air, T0=1500 K, P0=1 atm,
phi=1, 5 ms) through Cantera's ReactorNet for GRI30, FFCM 2.0 and Aramco 3.0. Reports
cold + warm wall-clock time so the paper can compare:
  - Cantera cold (C++ pre-compiled — just load + solve)
  - Cantera warm (re-stepping — cheap)
  - ChemMechSim cold (Julia JIT compile + solve — the compile cost)
  - ChemMechSim warm (cached native code — integrate only)

Usage: python3 examples/perf/cantera_baseline.py \
           [--runs 3] [--reltol 1e-8 --abstol 1e-12] [--out-dir examples/perf/output]
Each run is a cold (first) + warm (second) solve per mechanism.  Both timers wrap
only the integration loop; the mechanism is loaded with ct.Solution() beforehand,
so neither number includes the yaml load.  --runs N repeats that N times in the
SAME process; for a median over independent processes, launch N processes with
--runs 1 (as run_bench_j1134_bound.sh does).
Tolerances: omitted → Cantera/ReactorNet defaults; given → sim.rtol/atol are set
(to mirror ChemMechSim's reltol/abstol).
Outputs: examples/perf/output/bench_cantera.csv (one row per mech × run × cold/warm)
"""
import cantera as ct
import time, csv, os, sys, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args(argv):
    cfg = dict(runs=1, reltol=None, abstol=None,
               out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--runs" and i + 1 < len(argv):
            cfg["runs"] = int(argv[i + 1]); i += 1
        elif a == "--reltol" and i + 1 < len(argv):
            cfg["reltol"] = float(argv[i + 1]); i += 1
        elif a == "--abstol" and i + 1 < len(argv):
            cfg["abstol"] = float(argv[i + 1]); i += 1
        elif a == "--out-dir" and i + 1 < len(argv):
            cfg["out_dir"] = argv[i + 1]; i += 1
        else:
            sys.exit(f"unknown arg {a}")
        i += 1
    return cfg

_CFG = parse_args(sys.argv)
RUNS, OUT_DIR = _CFG["runs"], _CFG["out_dir"]
RELTOL, ABSTOL = _CFG["reltol"], _CFG["abstol"]
T0, P0 = 1500.0, ct.one_atm
T_END = 5.0e-3

MECHS = [
    ("gri30",  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mechanism", "gri30.yaml")),
    ("ffcm2",  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mechanism", "FFCM2.yaml")),
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
    if RELTOL is not None:
        sim.rtol = RELTOL
    if ABSTOL is not None:
        sim.atol = ABSTOL
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
for run_idx in range(1, RUNS + 1):
    for name, yaml in MECHS:
        print(f"=== run {run_idx} {name} ===")
        cold_s, n_steps, T_end, nsp = run_once(yaml)
        print(f"  cold: {cold_s:.3f}s  {n_steps} steps  T_end={T_end:.1f} K")
        warm_s, _, _, _ = run_once(yaml)
        print(f"  warm: {warm_s:.3f}s")
        rows.append({"mech": name, "run": run_idx, "n_species": nsp,
                     "cold_s": round(cold_s, 4), "warm_s": round(warm_s, 4),
                     "steps": n_steps, "T_end_K": round(T_end, 1), "cantera": ct.__version__})

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mech", "run", "n_species", "cold_s",
                                      "warm_s", "steps", "T_end_K", "cantera"])
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {out_csv}")
print("\nCompare: Cantera cold ≈ warm (C++ pre-compiled, no JIT).")
print("ChemMechSim cold >> warm (Julia JIT compile of MTK-generated code).")
print("ChemMechSim warm ≈ Cantera warm (both run native code).")
