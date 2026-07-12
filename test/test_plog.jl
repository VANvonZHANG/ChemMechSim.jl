using Test, ModelingToolkit
using ChemMechSim
using ChemMechSim: PlogPoint, PlogRate, plog_rate, symbolic_kf, RateCtx, needs_P

@testset "Phase 6 T3: PLOG symbolic_kf + needs_P" begin
    # needs_P(::PlogRate) = true
    kin = PlogRate([PlogPoint(1e4, 1e9, 0.0, 0.0), PlogPoint(1e6, 1e7, 0.0, 0.0)])
    @test needs_P(kin) == true

    # PLOG lowers under :fixedT (eos=:ideal_gas → P observed)
    sp = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")]
    rx = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0), kinetics=kin)
    mech = Mechanism(species=sp, reactions=[rx])
    phase = ChemMechSim.ChemPhaseSystem(mech; config=convenience_config(:fixedT))
    sys = ChemMechSim.extract_system(phase)
    # 2 concentration states
    @test length(unknowns(sys)) == 2
    # PLOG params named per convention: k_1_p1_A, k_1_p1_theta, k_1_p2_A, k_1_p2_theta
    pnames = sort([String(ModelingToolkit.getname(p)) for p in parameters(sys)])
    @test "k_1_p1_A" in pnames
    @test "k_1_p1_theta" in pnames
    @test "k_1_p2_A" in pnames
    @test "k_1_p2_theta" in pnames

    # PLOG under :kinetic (eos=:off, no P) → clear error from T2 early validation
    @test_throws ErrorException ChemMechSim.ChemPhaseSystem(mech; config=MechanismConfig())

    # numeric consistency: materialize kf at a sample (T,P) and compare to plog_rate
    # (build a RateCtx the way the test_lowering T6 test does, or lower + read the rate)
    # Simplest: the lowered system's RHS, evaluated at a fixed (T,P,c), tracks plog_rate.
    # (Covered structurally by the param-name checks above; deep numeric check is T5.)
end

@testset "Phase 6 T5: PLOG rate vs Cantera" begin
    using DelimitedFiles
    mech = load_mechanism(joinpath(@__DIR__, "data", "plog_minimal.yaml"))
    kin = mech.reactions[1].kinetics
    data = readdlm(joinpath(@__DIR__, "data", "plog_ref_rates.csv"), ',')[2:end, :]  # skip header; T_K,P_Pa,k_fwd
    maxrel = 0.0
    for i in 1:size(data, 1)
        k_cms = plog_rate(kin::PlogRate, data[i, 1], data[i, 2])
        k_can = data[i, 3]
        maxrel = max(maxrel, abs(k_cms - k_can) / max(abs(k_can), 1e-30))
    end
    @test maxrel < 1e-6                                   # PLOG math must match Cantera
end
