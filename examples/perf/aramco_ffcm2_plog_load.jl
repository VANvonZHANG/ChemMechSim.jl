# Phase 6 T5: coverage — full Aramco 3.0 + FFCM2 parse + lower (all PLOG reactions).
# Local coverage script (not a CI test). Mechanism fixtures live in examples/mechanism/.
# Run: julia --project=. examples/perf/aramco_ffcm2_plog_load.jl
#
# Coverage behavior (this script lowers with default checks=true):
#  - Both mechanisms PARSE cleanly, including all PLOG reactions. FFCM2's 6 PLOG
#    reactions that carry duplicate pressure points within one entry (Cantera sums
#    channels at each node) are accepted — same-pressure sum-at-pressure support
#    landed in commit 23f9d68 (large-mech T3). Aramco's 504 PLOG reactions all have
#    distinct pressure nodes.
#  - Both mechanisms LOWER-FAIL here with ModelingToolkitBase.ValidationError. This
#    is NOT PLOG-related and NOT parse-related: the inlined NASA7 K_c reverse-rate
#    terms trip MTK's unit validator on the full energy ODE (a known large-mech
#    K_c-unit-fold limitation). The validation/ ignition scripts bypass it with
#    `checks=false` and lower + solve both mechanisms fine (see aramco_ignition.jl
#    header). This script deliberately keeps checks=true to surface the limitation.
using ChemMechSim
using ChemMechSim: PlogRate

for path in ["examples/mechanism/AramcoMech3.0.yaml", "examples/mechanism/FFCM2.yaml"]
    println("=" ^ 60)
    println("Loading: $path")
    local mech
    try
        mech = load_mechanism(path)
    catch e
        println("  PARSE FAILED: $(typeof(e))")
        println("  ", split(sprint(showerror, e), "\n")[1])
        println("  (see header comment for known limitations)")
        println("  result: PARSE_FAIL")
        continue
    end
    nplog = count(r -> r.kinetics isa PlogRate, mech.reactions)
    println("  parsed: $(length(mech.species)) species, $(length(mech.reactions)) reactions ($nplog PLOG)")
    local ok = false
    try
        phase = ChemPhaseSystem(mech; config=convenience_config(:fixedT))
        sys = ChemMechSim.extract_system(phase)
        println("  lowered: $(length(unknowns(sys))) states, $(length(parameters(sys))) params — OK")
        ok = true
    catch e
        println("  LOWERING FAILED: $(typeof(e))")
        println("  ", split(sprint(showerror, e), "\n")[1])
        println("  (see header comment for known limitations)")
    end
    println("  result: $(ok ? "OK" : "LOWER_FAIL")")
end
