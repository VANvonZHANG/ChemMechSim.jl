"""Cantera PLOG rate reference for Phase 6 T5.

Generates test/data/plog_ref_rates.csv: rows of (T_K, P_Pa, k_fwd) for a
1-PLOG-reaction mechanism, across clamp + interpolation + extrapolation pressures.

Run: python3 examples/cantera_ref/plog_ref.py
(writes test/data/plog_ref_rates.csv so it's committed + CI-readable; the CSV is
the CI interface — Cantera itself is NOT a CI dep.)
"""
import cantera as ct
import csv
import os

# A 1-PLOG-reaction mechanism committed alongside this script. Its 3 rate-constants
# EXACTLY match test/data/plog_minimal.yaml (same P/A/b/Ea), so the kf Cantera computes
# here is directly comparable to ChemMechSim's plog_rate on the minimal fixture.
gas = ct.Solution("examples/cantera_ref/plog_mech.yaml")

# Temperatures spanning the NASA7 mid/high range.
Ts = [800.0, 1000.0, 1500.0, 2000.0, 2500.0]

# Pressures in Pa spanning below-lowest (clamp), interpolation (all 3 segments),
# and above-highest (clamp). The 3 PLOG points are at 0.1/1.0/10.0 atm.
Ppoints_Pa = [
    0.005 * 101325.0,   # below 0.1 atm → low clamp
    0.05  * 101325.0,   # below 0.1 atm → low clamp (different magnitude)
    0.5   * 101325.0,   # between 0.1 and 1.0 → interpolation segment 1
    5.0   * 101325.0,   # between 1.0 and 10.0 → interpolation segment 2
    50.0  * 101325.0,   # above 10.0 → high clamp
    500.0 * 101325.0,   # above 10.0 → high clamp (different magnitude)
]

rows = []
for T in Ts:
    for P in Ppoints_Pa:
        gas.TP = T, P
        k = float(gas.forward_rate_constants[0])   # the single PLOG reaction's kf
        rows.append((T, P, k))

# Write to test/data/ (committed, CI-readable) — not examples/cantera_ref/ (gitignored *.csv).
out_path = "test/data/plog_ref_rates.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["T_K", "P_Pa", "k_fwd"])
    w.writerows(rows)
print(f"wrote {out_path} ({len(rows)} rows)")
