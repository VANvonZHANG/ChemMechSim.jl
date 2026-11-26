#!/usr/bin/env python3
"""Assemble + compare the pipeline / linear-solver benchmarks measured with
Julia 1.13.4 on Slurm-bound cores vs the 2026-09-16 Julia 1.12.7 baseline.

- Per-column median of the 3 pipeline runs → bench_pipeline_julia1134_bound_median.csv
  (same assembly as the 1.12 baseline's bench_pipeline_plogfix_median.csv).
- Per-(mech,solver) median of warm reps (retcode Success) from j1134_bound/bench_matrix.csv.
- Side-by-side deltas vs baseline.
"""
import csv, statistics as st
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"
NEW = OUT / "j1134_bound"
MECHS = ["gri30", "ffcm2", "aramco"]
COLS = ["parse_s", "lower_s", "build_problem_s", "build_s", "cold_s", "warm_s", "jit_compile_s"]


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def median_pipeline(runs):
    rows = []
    by_mech = defaultdict(list)
    for r in runs:
        for row in r:
            by_mech[row["mech"]].append(row)
    for m in MECHS:
        rs = by_mech[m]
        assert len(rs) == len(runs), f"{m}: expected {len(runs)} runs, got {len(rs)}"
        out = dict(rs[0])
        for c in COLS:
            out[c] = round(st.median(float(x[c]) for x in rs), 2)
        rows.append(out)
    return rows


def main():
    # ---- pipeline (Table 7) ----
    runs = [read_csv(OUT / f"bench_pipeline_julia1134_bound_run{i}.csv") for i in (1, 2, 3)]
    new = median_pipeline(runs)
    with open(OUT / "bench_pipeline_julia1134_bound_median.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(new[0].keys()))
        w.writeheader(); w.writerows(new)

    old = {r["mech"]: r for r in read_csv(OUT / "bench_pipeline_plogfix_median.csv")}
    print("== Table 7 (pipeline, median of 3) — 1.13.4 bound  vs  1.12.7 baseline ==")
    hdr = f"{'mech':8s} {'stage':14s} {'1.12.7':>8s} {'1.13.4':>8s} {'ratio':>6s}"
    print(hdr); print("-" * len(hdr))
    for r in new:
        for c in COLS:
            o, n = float(old[r["mech"]][c]), float(r[c])
            print(f"{r['mech']:8s} {c:14s} {o:8.2f} {n:8.2f} {n/o:6.2f}")
        print()

    # ---- linsolver matrix (Table 8) ----
    mm = defaultdict(list); steps = defaultdict(list)
    for row in read_csv(NEW / "bench_matrix.csv"):
        if row["retcode"] == "Success":
            mm[(row["mech"], row["linsolve"])].append(float(row["wall_s"]))
            steps[(row["mech"], row["linsolve"])].append(int(row["steps"]))
    oldm = defaultdict(list)
    for row in read_csv(OUT / "bench_matrix_plogfix_3rep.csv"):
        if row["retcode"] == "Success":
            oldm[(row["mech"], row["linsolve"])].append(float(row["wall_s"]))
    print("== Table 8 (warm solve, median of 3 reps) ==")
    print(f"{'mech':8s} {'solver':9s} {'1.12.7':>8s} {'1.13.4':>8s} {'ratio':>6s}  steps(med)")
    for m in MECHS:
        for s in ["klu", "umfpack", "sparspak", "pardiso", "mumps"]:
            k = (m, s)
            if k in mm:
                o = st.median(oldm[k]) if oldm.get(k) else float("nan")
                n = st.median(mm[k])
                print(f"{m:8s} {s:9s} {o:8.2f} {n:8.2f} {n/o:6.2f}  {st.median(steps[k]):.0f}")
        print()

    # ---- compile probe (cold first-solve per mech) ----
    cp = NEW / "bench_compile.csv"
    if cp.exists():
        print("== compile probe (first solve in fresh process) ==")
        for row in read_csv(cp):
            print(row)


if __name__ == "__main__":
    main()
