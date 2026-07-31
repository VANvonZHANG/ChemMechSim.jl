using Test
using ChemMechSim
using ModelingToolkit
using OrdinaryDiffEq: ReturnCode
using SparseArrays

@testset "shared CSE Jacobian AST rewrite" begin
    c1 = Symbol("##cse#1")
    c2 = Symbol("##cse#2")
    slotmap = Dict(c1 => 1, c2 => 2)

    ex = :(($c1 + 2 * $c2) / local_rate)
    rewritten = ChemMechSim._rewrite_cse_refs(ex, slotmap, :work)

    @test rewritten == :((work[1] + 2 * work[2]) / local_rate)
    @test ChemMechSim._rewrite_cse_refs(:local_rate, slotmap, :work) == :local_rate
    @test ChemMechSim._rewrite_cse_refs(c1, slotmap, :work) == :(work[1])

    setup = Any[
        :(rate_scale = 2.0),
        Expr(:(=), c1, :(rate_scale + 1.0)),
        :(local_rate = $c1 * 3.0),
        Expr(:(=), c2, :($c1 + local_rate)),
    ]
    rewritten_setup = ChemMechSim._rewrite_setup_prefix(setup, slotmap, :work)
    @test rewritten_setup == Any[
        :(rate_scale = 2.0),
        :(work[1] = rate_scale + 1.0),
        :(local_rate = work[1] * 3.0),
        :(work[2] = work[1] + local_rate),
    ]

    aliases = ChemMechSim._sparse_write_aliases(
        Any[:(__nz = J.nzval), :(__idx = 7), :(local_rate = $c1)],
        :J,
    )
    rewritten_write = ChemMechSim._rewrite_sparse_write(
        :(__nz[__idx] = $c2 + local_rate),
        aliases,
        slotmap,
        :work,
    )
    @test rewritten_write == :(J.nzval[7] = work[2] + local_rate)
end

@testset "shared CSE Jacobian matches full MTK Jacobian on a small mechanism" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(2.0, 0.5, 1000.0),
            ),
        ],
    )
    phase = ChemPhaseSystem(mech; config=convenience_config(:fixedT), checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 3.0, "B" => 0.25), (0.0, 0.1))

    jac_shared!, J_proto = ChemMechSim.build_shared_cse_jac(
        sys; cse_chunk_size=1, write_chunk_size=1)
    J_shared = copy(J_proto)
    fill!(J_shared.nzval, NaN)
    jac_shared!(J_shared, prob.u0, prob.p, 0.0)

    jac_full! = ModelingToolkit.generate_jacobian(
        sys; sparse=true, expression=Val{false}, wrap_gfw=Val{false})[2]
    J_full = similar(J_proto)
    fill!(J_full.nzval, NaN)
    jac_full!(J_full, prob.u0, prob.p, 0.0)

    @test J_shared isa SparseMatrixCSC{Float64}
    @test J_proto.rowval == J_full.rowval
    @test J_proto.colptr == J_full.colptr
    @test J_shared.nzval ≈ J_full.nzval
end

@testset "build_problem jac=true uses shared CSE sparse Jacobian" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(2.0, 0.5, 1000.0),
            ),
        ],
    )
    phase = ChemPhaseSystem(mech; config=convenience_config(:fixedT), checks=false)
    prob = build_problem(phase, Dict("A" => 3.0, "B" => 0.25), (0.0, 0.1); jac=true)

    @test prob.f.jac !== nothing
    @test prob.f.jac_prototype isa SparseMatrixCSC{Float64}

    J = copy(prob.f.jac_prototype)
    fill!(J.nzval, NaN)
    prob.f.jac(J, prob.u0, prob.p, 0.0)

    @test all(isfinite, J.nzval)
end

@testset "build_problem honors jac_strategy keyword" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(1.0, 0.0, 0.0),
            ),
        ],
    )
    phase = ChemPhaseSystem(mech; config=convenience_config(:fixedT), checks=false)
    u0 = Dict("A" => 1.0, "B" => 0.0)

    prob_shared = build_problem(phase, u0, (0.0, 0.1);
                                jac=true, jac_strategy=:shared_cse)
    @test prob_shared.f.jac !== nothing
    @test prob_shared.f.jac_prototype isa SparseMatrixCSC{Float64}

    prob_none = build_problem(phase, u0, (0.0, 0.1);
                              jac=false, jac_strategy=:none)
    @test prob_none.f.jac === nothing

    @test_throws ArgumentError build_problem(phase, u0, (0.0, 0.1);
                                             jac=true, jac_strategy=:unknown)
    @test_throws ArgumentError("jac_strategy=:mtk is intentionally disabled for large mechanisms; use :shared_cse, :reaction_sharded, or :none") build_problem(
        phase, u0, (0.0, 0.1); jac=true, jac_strategy=:mtk)
end

@testset "simulate forwards jac=true to shared CSE problem build" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(1.0, 0.0, 0.0),
            ),
        ],
    )
    phase = ChemPhaseSystem(mech; config=convenience_config(:fixedT), checks=false)

    sol = simulate(phase, (0.0, 0.01);
                   u0=Dict("A" => 1.0, "B" => 0.0),
                   jac=true,
                   save_everystep=false)

    @test sol.retcode == ReturnCode.Success
end

@testset "shared CSE Jacobian preserves integer CSE ids on reversible thermo mechanism" begin
    mech = load_mechanism(joinpath(@__DIR__, "data", "h2o2.yaml"))
    phase = ChemPhaseSystem(mech; config=convenience_config(:fixedT), checks=false)
    sys = extract_system(phase)
    ctot = 101325.0 / (8.314 * 1000.0)
    u0 = Dict(sp.name => 0.0 for sp in mech.species)
    u0["H2"] = 2 / 7 * ctot
    u0["O2"] = 1 / 7 * ctot
    u0["N2"] = 4 / 7 * ctot
    prob = build_problem(phase, u0, (0.0, 1e-6))

    jac_shared!, J_proto = ChemMechSim.build_shared_cse_jac(
        sys; cse_chunk_size=20, write_chunk_size=20)
    J_shared = copy(J_proto)
    fill!(J_shared.nzval, NaN)
    jac_shared!(J_shared, prob.u0, prob.p, 0.0)

    jac_full! = ModelingToolkit.generate_jacobian(
        sys; sparse=true, expression=Val{false}, wrap_gfw=Val{false})[2]
    J_full = similar(J_proto)
    fill!(J_full.nzval, NaN)
    jac_full!(J_full, prob.u0, prob.p, 0.0)

    @test J_proto.rowval == J_full.rowval
    @test J_proto.colptr == J_full.colptr
    @test J_shared.nzval ≈ J_full.nzval
end

@testset "shared CSE Jacobian rejects invalid chunk sizes" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(2.0, 0.5, 1000.0),
            ),
        ],
    )
    phase = ChemPhaseSystem(mech; config=convenience_config(:fixedT), checks=false)
    sys = extract_system(phase)

    @test_throws ArgumentError ChemMechSim.build_shared_cse_jac(
        sys; cse_chunk_size=0, write_chunk_size=1)
    @test_throws ArgumentError ChemMechSim.build_shared_cse_jac(
        sys; cse_chunk_size=1, write_chunk_size=0)
end
