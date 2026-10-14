using Test
using ChemMechSim
using ChemMechSim: SpeciesData, ReactionData, ElementaryArrhenius, ThirdBodyArrhenius,
                   Irreversible, Mechanism, BatchReactor, extract_system
using ModelingToolkit: parameters, getname, unknowns

@testset "param_vjp: θ-block plan" begin
    mech = ab_toy()
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:pvjp1)
    sys = extract_system(r)
    ps = parameters(sys)
    blocks, covered = ChemMechSim._param_vjp_blocks(mech, sys, r.phase.config)

    # 2 irreversible reactions, each kf = A (b=Ea=0) → one θ each
    @test length(blocks) == 2
    @test [b.rx for b in blocks] == [1, 2]
    # each block covers exactly its own k_A, ascending flat order
    @test sort(vcat([b.pidx for b in blocks]...)) == covered
    @test length(covered) == 2
    # dep of rx1 = state col of A; of rx2 = col of B (toy state order [B, A])
    si = Dict(String(getname(s)) => i for (i, s) in enumerate(unknowns(sys)))
    @test blocks[1].dep_cols == [si["A"]]
    @test blocks[2].dep_cols == [si["B"]]
    # θ symbols are the actual parameter objects
    @test Set(blocks[1].θ_syms) == Set([ps[findfirst(p -> String(getname(p)) == "k_1_A", ps)]])
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
    @test length(b.pidx) >= 2                # k_A (+ k_theta for Ea≠0) (+ b if b≠0)
    @test issorted(b.pidx)
    @test sort(covered) == sort(b.pidx)
end
