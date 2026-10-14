using Test
using ChemMechSim
using ChemMechSim: SpeciesData, ReactionData, ElementaryArrhenius, ThirdBodyArrhenius,
                   Irreversible, ExplicitReverse, Mechanism, BatchReactor, extract_system
using ModelingToolkit: parameters, getname, unknowns

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
