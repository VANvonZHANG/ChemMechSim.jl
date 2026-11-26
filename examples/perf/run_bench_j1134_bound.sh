#!/bin/bash
# run_bench_j1134_bound.sh — pipeline-stage + linear-solver benchmarks on
# Slurm-bound cores with a pinned Julia, 1 Julia thread + 1 BLAS thread
# (identical threading to a single-threaded reference run, so the Julia version
# and the core binding are the only variables).
#
# Run inside an allocation, e.g.:
#   JBIN=/path/to/julia-1.13.4/bin/julia \
#       srun -p <partition> --ntasks=1 -c 4 --mem=48G --cpu-bind=cores \
#       bash perf/run_bench_j1134_bound.sh
#
# JBIN overrides the Julia binary (default: julia on PATH). It is used as a
# single path (no word splitting), so pass a full binary path rather than
# "julia +1.13.4". Pinning the version matters: a bare `julia` resolves to
# whatever juliaup's default channel is at the time.
# JULIA_DEPOT_PATH is left untouched so the caller's depot is used; exporting
# an EMPTY value makes Julia fail at startup, so it is never set here.
set -u
JBIN=${JBIN:-julia}
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."   # examples/

OUT=perf/output
NEW=perf/output/j1134_bound
mkdir -p "$NEW"

echo "=== bench start $(date -Is)  host=$(hostname)  loadavg: $(cat /proc/loadavg)"
$JBIN --version
grep Cpus_allowed_list /proc/self/status
echo "JULIA_NUM_THREADS=$JULIA_NUM_THREADS  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"

# ---- pipeline stages: 3 fresh-process runs (take the per-column median) ----
for r in 1 2 3; do
    echo "=== pipeline run $r  $(date -Is)"
    $JBIN --project=. perf/bench_pipeline_stages.jl || echo "RUN $r FAILED"
    mv "$OUT/bench_pipeline.csv"      "$OUT/bench_pipeline_julia1134_bound_run${r}.csv"
    mv "$OUT/bench_pipeline_meta.yaml" "$OUT/bench_pipeline_meta_julia1134_bound_run${r}.yaml"
    echo "=== pipeline run $r done $(date -Is)  loadavg: $(cat /proc/loadavg)"
done

# ---- linear-solver matrix (klu,umfpack,sparspak,mumps,pardiso × 3 warm reps,
#      compile probe per mech, microbench + accuracy), fresh out-dir ----
echo "=== linsolver matrix  $(date -Is)"
$JBIN --project=. perf/bench_linsolver_matrix.jl --repeats 3 --out-dir "$NEW" || echo "LINSOLVER BENCH FAILED"

echo "=== ALL DONE $(date -Is)  loadavg: $(cat /proc/loadavg)"
