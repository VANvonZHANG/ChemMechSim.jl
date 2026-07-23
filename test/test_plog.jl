using Test, ModelingToolkit
using ChemMechSim
using DynamicQuantities
using ChemMechSim: PlogPoint, PlogRate, plog_rate, symbolic_kf, RateCtx, needs_P,
                   plog_kf, plog_kf_dT, plog_kf_dP
import ModelingToolkit: substitute, value, get_variables, getname, getdefault

@testset "PLOG registered numeric helpers accept Real-valued ids" begin
    kin = PlogRate([PlogPoint(1e4, 1e3, 0, 0), PlogPoint(1e6, 1e2, 0, 0)])
    sp = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")]
    mech = Mechanism(species=sp, reactions=[
        ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0), kinetics=kin)
    ])
    @parameters T P
    ctx = RateCtx(mech, Dict{Int,Any}(), T, 1, 1.0, nothing, nothing,
                  Dict{Int,Any}(), P, Any[])
    symbolic_kf(kin, ctx)
    id = ChemMechSim.PLOG_NEXT_ID[]
    id_float = Float64(id)
    T_sample, P_sample = 1000.0, 101325.0

    @test plog_kf(T_sample, P_sample, id_float) ≈ plog_kf(T_sample, P_sample, id)
    @test plog_kf_dT(T_sample, P_sample, id_float) ≈ plog_kf_dT(T_sample, P_sample, id)
    @test plog_kf_dP(T_sample, P_sample, id_float) ≈ plog_kf_dP(T_sample, P_sample, id)
end

@testset "PLOG registered Quantity helpers accept Int and unitless Quantity ids" begin
    kin = PlogRate([PlogPoint(1e4, 1e3, 0, 0), PlogPoint(1e6, 1e2, 0, 0)])
    sp = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")]
    mech = Mechanism(species=sp, reactions=[
        ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0), kinetics=kin)
    ])
    @parameters T P
    ctx = RateCtx(mech, Dict{Int,Any}(), T, 1, 1.0, nothing, nothing,
                  Dict{Int,Any}(), P, Any[])
    symbolic_kf(kin, ctx)
    id = ChemMechSim.PLOG_NEXT_ID[]
    id_quantity = id * u"1"
    T_quantity, P_quantity = 1000.0u"K", 101325.0u"Pa"

    @test plog_kf(T_quantity, P_quantity, id_quantity) ≈ plog_kf(T_quantity, P_quantity, id)
    @test plog_kf_dT(T_quantity, P_quantity, id_quantity) ≈ plog_kf_dT(T_quantity, P_quantity, id)
    @test plog_kf_dP(T_quantity, P_quantity, id_quantity) ≈ plog_kf_dP(T_quantity, P_quantity, id)
end

@testset "Phase 6 T3: PLOG symbolic_kf (opaque call node) + needs_P" begin
    kin = PlogRate([PlogPoint(1e4, 1e9, 0.0, 0.0), PlogPoint(1e6, 1e7, 0.0, 0.0)])
    @test needs_P(kin) == true
    sp = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")]
    rx = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0), kinetics=kin)
    mech = Mechanism(species=sp, reactions=[rx])
    phase = ChemMechSim.ChemPhaseSystem(mech; config=convenience_config(:fixedT))
    sys = ChemMechSim.extract_system(phase)
    unk_names = Set(String(ModelingToolkit.getname(u)) for u in unknowns(sys))
    @test "A" in unk_names && "B" in unk_names   # species present (P count differs by config)
    # RHS references plog_kf call node (opaque), NOT inlined ifelse
    rhs_str = string(equations(sys)[1].rhs)
    @test occursin("plog_kf", rhs_str)
    @test !occursin("ifelse", rhs_str)          # no inlined interpolation tree
    # Numeric check (task-2 brief ambiguity #3): substitute fixed (T, P, c_A) on the RHS
    # equation and confirm the opaque plog_kf call node evaluates to plog_rate(kin, T, P)
    # times the mass-action factor (c_A^1 = c_A). Verifies the registered call returns the
    # correct VALUE, not just that a node exists.
    rhs = equations(sys)[1].rhs
    T_sample, P_sample, cA_sample = 1000.0, 1e5, 0.5
    sub = Dict{Any,Float64}()
    for v in get_variables(rhs)
        n = getname(v)
        if n === :T
            sub[v] = T_sample
        elseif n === :P
            sub[v] = P_sample
        elseif n === :A
            sub[v] = cA_sample
        else
            sub[v] = getdefault(v)
        end
    end
    rhs_num = Float64(value(substitute(rhs, sub; fold=Val(true))))
    @test rhs_num ≈ plog_rate(kin, T_sample, P_sample) * cA_sample  rtol=1e-10
    @test_throws ErrorException ChemMechSim.ChemPhaseSystem(mech; config=MechanismConfig())
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

@testset "Large-mech C: same-pressure PLOG sums at pressure" begin
    using ChemMechSim: PlogPoint, PlogRate, plog_rate
    # 2 points at P=1e5 (A=1e9,b=0,Ea=0 and A=3e9,b=0,Ea=0 → sum 4e9), 1 at P=1e6 (A=1e7)
    kin = PlogRate([PlogPoint(1e5,1e9,0,0), PlogPoint(1e5,3e9,0,0), PlogPoint(1e6,1e7,0,0)])
    @test plog_rate(kin, 1000.0, 1e5) ≈ 4e9          # at P=1e5 → sum of the two (4e9), not interp
    @test plog_rate(kin, 1000.0, 1e6) ≈ 1e7          # at P=1e6
    # between: f=0.5 at log-mid → 4e9·(1e7/4e9)^0.5
    @test plog_rate(kin, 1000.0, sqrt(1e5*1e6)) ≈ 4e9 * (1e7/4e9)^0.5
end
