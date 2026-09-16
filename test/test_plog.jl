using Test, ModelingToolkit
using ChemMechSim
using DynamicQuantities
using ChemMechSim: PlogPoint, PlogRate, plog_rate, plog_dkdT, plog_dkdP,
                   symbolic_kf, RateCtx, needs_P,
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

@testset "fixedT PLOG reaction-sharded sparse Jacobian executes" begin
    kin = PlogRate([PlogPoint(1e4, 1e3, 0, 0), PlogPoint(1e6, 1e2, 0, 0)])
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = kin,
            ),
        ],
    )
    phase = ChemPhaseSystem(mech; config=convenience_config(:fixedT), checks=false)
    sys = ChemMechSim.extract_system(phase)
    prob = build_problem(phase, Dict("A"=>1.0, "B"=>0.0, "P"=>101325.0), (0.0, 0.1))

    jac!, J_proto = ChemMechSim.build_reaction_sharded_jac(
        mech; config=phase.config, checks=false, sys=sys)
    J = copy(J_proto)
    fill!(J.nzval, NaN)
    jac!(J, prob.u0, prob.p, 0.0)

    @test all(isfinite, J.nzval)
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
    # RHS references plog_kf call node (opaque), NOT inlined ifelse.
    # Look up the species-B equation by name (Task 3: P is now also a state under :fixedT
    # const-V, so equation ordering is [P, B, A] — positional indexing would hit D(P)~...).
    eqs = equations(sys)
    b_eq = first(eq for eq in eqs if operation(eq.lhs) isa Differential &&
                                     getname(arguments(eq.lhs)[1]) === :B)
    rhs_str = string(b_eq.rhs)
    @test occursin("plog_kf", rhs_str)
    @test !occursin("ifelse", rhs_str)          # no inlined interpolation tree
    # Numeric check (task-2 brief ambiguity #3): substitute fixed (T, P, c_A) on the RHS
    # equation and confirm the opaque plog_kf call node evaluates to plog_rate(kin, T, P)
    # times the mass-action factor (c_A^1 = c_A). Verifies the registered call returns the
    # correct VALUE, not just that a node exists.
    rhs = b_eq.rhs
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

# —— 2026-09-16 零分配重写的保真网：oracle = 旧（分配型）实现的逐字拷贝 ————————————
# 逐位对照 + @allocated==0。旧实现细节见 git 历史本 testset 引入前的 src/data/kinetics.jl。

# Shared 6-channel fixture: duplicate pressure at 1e4 (same-P summing path),
# nonzero b/Ea; used by the oracle testset and the alloc testset below.
const _PLOG_TEST_KIN6 = PlogRate([PlogPoint(1e3, 1e12, -0.5, 2e4),
                                  PlogPoint(1e4, 3e15,  0.3, 5e4),
                                  PlogPoint(1e4, 7e14,  0.2, 4e4),
                                  PlogPoint(1e5, 2e13, -0.1, 1e5),
                                  PlogPoint(1e6, 5e11,  0.0, 6e4),
                                  PlogPoint(1e7, 8e10,  0.5, 3e4)])

_orr_arr(A, b, θ, T) = A * T^b * exp(-θ / T)
function _orr_k_dkT(A, b, θ, T)
    k = A * T^b * exp(-θ / T)
    return (k, k * (b / T + θ / T^2))
end
const _ORR_R = 8.314
const _ORR_PSTD = 1.0e5
_orr_seg(k_lo, k_hi, f) = k_lo * (k_hi / k_lo)^f
function _orr_group(ks, log_Pi)
    out_ks = Float64[]; out_lp = Float64[]
    i = 1
    while i ≤ length(ks)
        j = i; s = ks[i]
        while j + 1 ≤ length(ks) && log_Pi[j + 1] == log_Pi[i]
            j += 1; s += ks[j]
        end
        push!(out_ks, s); push!(out_lp, log_Pi[i])
        i = j + 1
    end
    return (out_ks, out_lp)
end
function _orr_rate(kin, T, P)
    pts = kin.points
    ks = [_orr_arr(p.A, p.b, p.Ea / _ORR_R, T) for p in pts]
    lp = [log(p.P / _ORR_PSTD) for p in pts]
    s_ks, s_lp = _orr_group(ks, lp)
    log_P = log(P / _ORR_PSTD); n = length(s_ks)
    n == 1 && return s_ks[1]
    result = s_ks[n]
    for i in (n - 1):-1:1
        f = (log_P - s_lp[i]) / (s_lp[i + 1] - s_lp[i])
        seg = _orr_seg(s_ks[i], s_ks[i + 1], f)
        result = ifelse(log_P <= s_lp[i], s_ks[i],
                        ifelse(log_P <= s_lp[i + 1], seg, result))
    end
    return result
end
function _orr_dkdT(kin, T, P)
    pts = kin.points
    kd = [_orr_k_dkT(p.A, p.b, p.Ea / _ORR_R, T) for p in pts]
    ks = [first(x) for x in kd]; dks = [last(x) for x in kd]
    lp = [log(p.P / _ORR_PSTD) for p in pts]
    s_ks, s_lp = _orr_group(ks, lp); s_dks, _ = _orr_group(dks, lp)
    log_P = log(P / _ORR_PSTD); n = length(s_ks)
    n == 1 && return s_dks[1]
    out = s_dks[n]
    for i in (n - 1):-1:1
        f = (log_P - s_lp[i]) / (s_lp[i + 1] - s_lp[i])
        klo, khi = s_ks[i], s_ks[i + 1]
        seg_k = klo^(1 - f) * khi^f
        seg_d = seg_k * ((1 - f) * s_dks[i] / klo + f * s_dks[i + 1] / khi)
        out = ifelse(log_P <= s_lp[i], s_dks[i],
                     ifelse(log_P <= s_lp[i + 1], seg_d, out))
    end
    return out
end
function _orr_dkdP(kin, T, P)
    pts = kin.points
    ks = [_orr_arr(p.A, p.b, p.Ea / _ORR_R, T) for p in pts]
    lp = [log(p.P / _ORR_PSTD) for p in pts]
    s_ks, s_lp = _orr_group(ks, lp)
    log_P = log(P / _ORR_PSTD); n = length(s_ks)
    n == 1 && return 0.0
    out = 0.0
    for i in (n - 1):-1:1
        klo, khi = s_ks[i], s_ks[i + 1]
        f = (log_P - s_lp[i]) / (s_lp[i + 1] - s_lp[i])
        seg_k = klo^(1 - f) * khi^f
        seg_d = seg_k * log(khi / klo) * (1 / P) / (s_lp[i + 1] - s_lp[i])
        out = ifelse(log_P <= s_lp[i], 0.0,
                     ifelse(log_P <= s_lp[i + 1], seg_d, out))
    end
    return out
end

@testset "PLOG zero-alloc rewrite is bit-identical to the old implementation" begin
    kin6 = _PLOG_TEST_KIN6
    Ts = [300.0, 800.0, 1500.0, 2500.0]
    # below range, log grid through range, exact nodes, above range
    Ps = vcat([1e2], 10.0 .^ (2.0:0.25:7.0), [1e3, 1e4, 1e5, 1e6, 1e7, 3e7])
    # nested @testset puts "T=… P=…" into the failure summary (plain @test
    # messages are dropped on 1.12); assertions stay byte-identical.
    for T in Ts, P in Ps
        @testset "T=$T P=$P" begin
            @test isequal(plog_rate(kin6, T, P),  _orr_rate(kin6, T, P))
            @test isequal(plog_dkdT(kin6, T, P),  _orr_dkdT(kin6, T, P))
            @test isequal(plog_dkdP(kin6, T, P),  _orr_dkdP(kin6, T, P))
        end
    end
    # degenerate single-channel PlogRate (programmatic-only; parser forbids it)
    kin1 = PlogRate([PlogPoint(5e4, 1e13, 0.25, 3e4)])
    for T in Ts, P in Ps
        @testset "T=$T P=$P" begin
            @test isequal(plog_rate(kin1, T, P), _orr_rate(kin1, T, P))
            @test isequal(plog_dkdT(kin1, T, P), _orr_dkdT(kin1, T, P))
            @test isequal(plog_dkdP(kin1, T, P), _orr_dkdP(kin1, T, P))
        end
    end
end

@testset "PLOG runtime calls allocate nothing" begin
    kin6 = _PLOG_TEST_KIN6
    plog_rate(kin6, 1500.0, 101325.0)   # warm-up: compile outside the measurement
    plog_dkdT(kin6, 1500.0, 101325.0)
    plog_dkdP(kin6, 1500.0, 101325.0)
    @test (@allocated plog_rate(kin6, 1500.0, 101325.0)) == 0
    @test (@allocated plog_dkdT(kin6, 1500.0, 101325.0)) == 0
    @test (@allocated plog_dkdP(kin6, 1500.0, 101325.0)) == 0
end
