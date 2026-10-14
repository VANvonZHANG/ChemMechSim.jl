using Test
using ChemMechSim
using ChemMechSim: SpeciesData, ReactionData, ElementaryArrhenius, ThirdBodyArrhenius,
                   Irreversible, ExplicitReverse, Mechanism, BatchReactor, extract_system,
                   build_param_vjp
using ModelingToolkit: parameters, getname, unknowns
using ForwardDiff: Dual, ForwardDiff

@testset "param_vjp: θ-block plan" begin
    mech = ab_toy()
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:pvjp1)
    sys = extract_system(r)
    ps = parameters(sys)
    blocks, covered = ChemMechSim._param_vjp_blocks(mech, sys, r.phase.config)

    # 2 irreversible reactions, each kf = A (b=Ea=0) → one θ each
    @test length(blocks) == 2
    @test [b.rx_idx for b in blocks] == [1, 2]
    # each block covers exactly its own k_A, ascending flat order
    @test sort(vcat([b.pidx for b in blocks]...)) == covered
    @test length(covered) == 2
    # dep of rx1 = state col of A; of rx2 = col of B (toy state order [B, A])
    si = Dict(String(getname(s)) => i for (i, s) in enumerate(unknowns(sys)))
    @test blocks[1].dep_cols == [si["A"]]
    @test blocks[2].dep_cols == [si["B"]]
    # θ symbols are the actual parameter objects
    @test Set(blocks[1].θ_syms) == Set([ps[findfirst(p -> String(getname(p)) == "k_1_A", ps)]])
    @test Set(blocks[2].θ_syms) == Set([ps[findfirst(p -> String(getname(p)) == "k_2_A", ps)]])
end

@testset "param_vjp: third-body block is all-species-dense and multi-θ" begin
    # N2 + A -> N2 + B with enhanced N2: kf = k·[M]_eff, deps = all species
    spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B"); spN = SpeciesData(id=3, name="N2")
    rx = ReactionData(reactants=Dict(1 => 1.0, 3 => 1.0), products=Dict(2 => 1.0, 3 => 1.0),
                      kinetics=ThirdBodyArrhenius(ElementaryArrhenius(2.0, 0.5, 100.0),
                                                   Dict(3 => 2.0)),
                      reverse_policy=Irreversible())
    mech = Mechanism(species=[spA, spB, spN], reactions=[rx])
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:pvjptb)
    sys = extract_system(r)
    blocks, covered = ChemMechSim._param_vjp_blocks(mech, sys, r.phase.config)
    @test length(blocks) == 1
    b = blocks[1]
    nsp = length(unknowns(sys))
    @test length(b.dep_cols) == nsp          # Σα·c inlined → every species is an input
    # exactly {k_A, k_theta} — b is a literal (never a param), T/P are zero-column-excluded
    names = sort!([String(getname(s)) for s in b.θ_syms])
    @test names == ["k_1_A", "k_1_theta"]
    @test issorted(b.pidx)
    @test sort(covered) == sort(b.pidx)
    @test length(covered) == 2
end

@testset "param_vjp: reversible block captures the reverse law's θ; T column excluded" begin
    # A ⇄ B, forward ElementaryArrhenius(0.8,0,1000), explicit reverse (0.5,0,3000):
    # net = k_1_A·exp(-k_1_theta/T)·[A] − k_1_rev_A·exp(-k_1_rev_theta/T)·[B]; T IS in the
    # rate expr (both Arrhenius factors) but must stay OUT of θ (zero-column policy, v1).
    # Forward Ea≠0 also creates the T parameter the reverse law needs.
    spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B")
    rx = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                      kinetics=ElementaryArrhenius(0.8, 0.0, 1000.0),
                      reverse_policy=ExplicitReverse(ElementaryArrhenius(0.5, 0.0, 3000.0)))
    mech = Mechanism(species=[spA, spB], reactions=[rx])
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:pvjprev)
    sys = extract_system(r)
    blocks, covered = ChemMechSim._param_vjp_blocks(mech, sys, r.phase.config)
    @test length(blocks) == 1
    names = sort!([String(getname(s)) for s in blocks[1].θ_syms])
    @test names == ["k_1_A", "k_1_rev_A", "k_1_rev_theta", "k_1_theta"]  # forward + reverse tunables
    @test !("T" in names)
    @test length(covered) == 4
end

# ---- zero-tunable law (paramspec with only Plain roles) → skip path ----
struct _ConstRateLaw <: AbstractKinetics
    k::Float64
end
ChemMechSim.paramspec(kin::_ConstRateLaw) = (ChemMechSim.plain(:k),)
ChemMechSim.body(kin::_ConstRateLaw) = (k, T) -> k            # generic materializer appends ctx.T

@testset "param_vjp: no tunables → no blocks, empty covered" begin
    spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B")
    rx = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                      kinetics=_ConstRateLaw(0.5), reverse_policy=Irreversible())
    mech = Mechanism(species=[spA, spB], reactions=[rx])
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:pvjpconst)
    sys = extract_system(r)
    blocks, covered = ChemMechSim._param_vjp_blocks(mech, sys, r.phase.config)
    @test isempty(blocks)
    @test isempty(covered)
end

@testset "param_vjp: θ-explicit rate kernel values + dual propagation" begin
    mech = ab_toy()
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:pvjp2)
    sys = extract_system(r)
    blocks, _ = ChemMechSim._param_vjp_blocks(mech, sys, r.phase.config)
    ChemMechSim._materialize_param_vjp_blocks!(blocks, mech, sys, r.phase.config)
    si = Dict(String(getname(s)) => i for (i, s) in enumerate(unknowns(sys)))
    u = zeros(length(si)); u[si["A"]] = 0.4; u[si["B"]] = 0.6   # state order [B, A]
    # p shaped like prob.p (the toy is T-independent, so its kernels never read p —
    # the reversible testset below exercises the positional-p contract)
    prob = build_problem(r, Dict("A" => 0.4, "B" => 0.6), (0.0, 1.0); jac=true)
    p = prob.p
    for (k, b) in enumerate(blocks)
        @test b.rx_idx == k
        θv = k == 1 ? [0.8] : [0.3]               # θ segment length == length(pidx) (1 here)
        buf = [0.0]
        x = vcat(u[b.dep_cols], θv)
        b.fn(buf, x, p, 0.0)
        @test buf[1] ≈ (k == 1 ? 0.8 * 0.4 : 0.3 * 0.6)   # r1 = k1·[A], r2 = k2·[B]
        # dual propagation: tags MUST enter through the θ segment (∂r/∂θ = dep state value)
        bufD = [Dual{Nothing}(0.0, 0.0)]  # Dual-typed buffer: the kernel returns a Dual
        xD = vcat(u[b.dep_cols], [Dual{Nothing}(θv[1], 1.0)])
        b.fn(bufD, xD, p, 0.0)
        @test ForwardDiff.partials(bufD[1], 1) ≈ (k == 1 ? 0.4 : 0.6)
    end
    # double materialization is loud
    @test_throws ErrorException ChemMechSim._materialize_param_vjp_blocks!(
        blocks, mech, sys, r.phase.config)
end

@testset "param_vjp: multi-θ kernel gradient (reversible) + positional p read" begin
    # net = kf·[A] − kr·[B], kf = k_1_A·exp(-k_1_theta/T), kr = k_1_rev_A·exp(-k_1_rev_theta/T).
    # T enters ONLY via the kernel's positional p read (T ∉ θ — zero-column policy), so this
    # is the suite's first exercise of that contract; partials hand-derived, recomputed inline.
    spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B")
    rx = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                      kinetics=ElementaryArrhenius(0.8, 0.0, 1000.0),
                      reverse_policy=ExplicitReverse(ElementaryArrhenius(0.5, 0.0, 3000.0)))
    mech = Mechanism(species=[spA, spB], reactions=[rx])
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:pvjp5)
    sys = extract_system(r)
    blocks, _ = ChemMechSim._param_vjp_blocks(mech, sys, r.phase.config)
    ChemMechSim._materialize_param_vjp_blocks!(blocks, mech, sys, r.phase.config)
    prob = build_problem(r, Dict("A" => 0.4, "B" => 0.6), (0.0, 1.0); jac=true)
    pv = ChemMechSim._parameter_vector(prob.p)          # FLAT vector — the kernels' p contract
    ps = parameters(sys)
    Tval = pv[findfirst(p -> String(getname(p)) == "T", ps)]
    θfwd, θrev = 1000.0 / R_GAS, 3000.0 / R_GAS         # k_theta defaults are Ea/R
    kf, kr = 0.8 * exp(-θfwd / Tval), 0.5 * exp(-θrev / Tval)
    u = zeros(2)                                         # state order [B, A]
    u[1] = 0.6; u[2] = 0.4
    b = blocks[1]
    g = ForwardDiff.gradient(θv -> begin
            buf = Vector{promote_type(Float64, eltype(θv))}(undef, 1)
            b.fn(buf, vcat(u[b.dep_cols], θv), pv, 0.0)
            buf[1]
        end, pv[b.pidx])
    expected = Dict("k_1_A"         =>  exp(-θfwd / Tval) * 0.4,
                    "k_1_theta"     => -kf * 0.4 / Tval,
                    "k_1_rev_A"     => -exp(-θrev / Tval) * 0.6,
                    "k_1_rev_theta" =>  kr * 0.6 / Tval)
    @test length(g) == 4
    for (i, s) in enumerate(b.θ_syms)
        @test g[i] ≈ expected[String(getname(s))] rtol = 1e-10
    end
end

@testset "param_vjp: Sᵀλ · local-FD assembly vs hand calculation" begin
    mech = ab_toy()
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:pvjp3)
    sys = extract_system(r)
    prob = build_problem(r, Dict("A" => 0.4, "B" => 0.6), (0.0, 1.0); jac=true)
    vjpp, gaps = build_param_vjp(mech; config=r.phase.config, checks=false, sys=sys)
    @test vjpp !== nothing && isempty(gaps)
    ps = parameters(sys)
    # name-exact lookup: the toy's flat p order is [k_2_A, k_1_A] (pinned by
    # test_adjoint_sensitivity.jl), so findfirst/findlast over "_A" would cross them
    ikA = findfirst(p -> String(getname(p)) == "k_1_A", ps)   # k_1_A
    ikB = findfirst(p -> String(getname(p)) == "k_2_A", ps)    # k_2_A
    si = Dict(String(getname(s)) => i for (i, s) in enumerate(unknowns(sys)))
    u = zeros(2); u[si["A"]] = 0.4; u[si["B"]] = 0.6
    λ = [0.7, -0.3]
    dgrad = fill(NaN, length(ps))
    vjpp(dgrad, λ, u, prob.p, 0.0)
    @test dgrad[ikA] ≈ 0.4 atol = 1e-12     # A·(λ_B − λ_A) = 0.4·(0.7+0.3)
    @test dgrad[ikB] ≈ -0.6 atol = 1e-12    # B·(λ_A − λ_B) = 0.6·(−0.3−0.7)
    # T parameter column is ZERO by design (v1 spec amendment)
    iT = findfirst(p -> String(getname(p)) == "T", ps)
    iT === nothing || (@test iszero(dgrad[iT]))
    # Float64 λ (plain) and Dual λ must both work — eltype-generic assembly:
    # (testset-local `using` is illegal; the file head already imports Dual)
    dgradD = fill(Dual{Nothing}(0.0, 0.0), length(ps))
    vjpp(dgradD, Dual{Nothing}.(λ, 0.0), u, prob.p, 0.0)
    @test ForwardDiff.value(dgradD[ikA]) ≈ 0.4 atol = 1e-12
end
