# Phase 6 T5: ChemMechSim PLOG rate vs Cantera.
# Compares plog_rate(kin, T, P) against Cantera's forward_rate_constants[0] across
# clamp + interpolation + extrapolation (T,P) points. The reference CSV is generated
# by examples/cantera_ref/plog_ref.py against examples/cantera_ref/plog_mech.yaml
# (whose 3 rate-constants EXACTLY match test/data/plog_minimal.yaml).
#
# Run: julia --project=. examples/plog_validation.jl
using ChemMechSim
using ChemMechSim: PlogRate, plog_rate
using DelimitedFiles

mech = load_mechanism("test/data/plog_minimal.yaml")          # same points as the Cantera ref mechanism
kin = mech.reactions[1].kinetics::PlogRate

data = readdlm("test/data/plog_ref_rates.csv", ',')[2:end, :] # skip header; cols: T_K, P_Pa, k_fwd
maxrel = 0.0
worst = (0.0, 0.0)
for i in 1:size(data, 1)
    k_cms = plog_rate(kin, data[i, 1], data[i, 2])
    k_can = data[i, 3]
    rel = abs(k_cms - k_can) / max(abs(k_can), 1e-30)
    if rel > maxrel
        global maxrel = rel
        global worst = (data[i, 1], data[i, 2])
    end
end
println("PLOG rate vs Cantera: max relative diff = $maxrel")
println("  worst point: T=$(worst[1]) K, P=$(worst[2]) Pa")
@assert maxrel < 1e-6 "PLOG rate mismatch vs Cantera: $maxrel"
println("PASS")
