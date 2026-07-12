# Phase 6 T5: coverage — full Aramco 3.0 + FFCM2 parse + lower (all PLOG reactions).
# Local only (the fixtures in examples/data/ are untracked). NOT a CI test.
# Run: julia --project=. examples/aramco_ffcm2_plog_load.jl
#
# Known limitations surfaced by this coverage run (documented, not blocking T5):
#  - FFCM2 has 6 PLOG reactions with duplicate pressure points within a single entry
#    (Cantera sums channels at each node). The T4 parser rejects duplicates per spec §5.1
#    (out-of-scope for Phase 6 T1-T5; flagged as a known edge). FFCM2 parse aborts at the
#    first such reaction. A future task could extend PlogPoint to hold multi-channel (A,b,Ea)
#    sets per pressure (Cantera sum-then-interp semantics).
#  - Aramco parses fully (504 PLOG) but lowering hits pre-existing MTK dimension warnings
#    on some non-PLOG reactions (three-body/elementary with exotic species). These are
#    unrelated to PLOG (PLOG lowers cleanly in isolation — see test_plog.jl T3).
using ChemMechSim
using ChemMechSim: PlogRate

for path in ["examples/data/AramcoMech3.0.yaml", "examples/data/FFCM2_model.yaml"]
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
