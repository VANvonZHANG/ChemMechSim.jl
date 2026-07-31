using Test
using ChemMechSim
using SparseArrays

function _shared_cse_stats_phase()
    mech = Mechanism(
        species = [
            SpeciesData(id=1, name="A"),
            SpeciesData(id=2, name="B"),
            SpeciesData(id=3, name="C"),
        ],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(2.0, 0.5, 1000.0),
            ),
            ReactionData(
                reactants = Dict(2 => 1.0),
                products = Dict(3 => 1.0),
                kinetics = ElementaryArrhenius(3.0, 0.0, 0.0),
            ),
        ],
    )
    return ChemPhaseSystem(mech; config=convenience_config(:fixedT), checks=false)
end

_shared_cse_stats_system() = extract_system(_shared_cse_stats_phase())

@testset "AST node budget helpers" begin
    @test ChemMechSim._expr_node_count(:x) == 1
    @test ChemMechSim._expr_node_count(:(a + b)) <
          ChemMechSim._expr_node_count(:((a + b) * (c + d) + exp(e / f)))

    deep = :x
    for _ in 1:90
        deep = :($deep + x)
    end
    @test ChemMechSim._expr_node_count(deep) == 271

    stmts = Any[:(a = b + c), :(d = exp(e / f)), :(g = h * i)]
    @test ChemMechSim._chunk_by_node_budget(stmts, 1_000_000) == [stmts]
    @test ChemMechSim._chunk_by_node_budget(stmts, 3) == Any[
        Any[:(a = b + c)],
        Any[:(d = exp(e / f))],
        Any[:(g = h * i)],
    ]
    @test_throws ArgumentError ChemMechSim._chunk_by_node_budget(stmts, 0)
    @test_throws ArgumentError ChemMechSim._chunk_by_node_budget(stmts, -1)
end

@testset "_needed_setup_prefix traces write dependencies" begin
    c1 = Symbol("##cse#1")
    c2 = Symbol("##cse#2")
    c3 = Symbol("##cse#3")
    setup = Any[
        Expr(:(=), c1, :(a + b)),
        Expr(:(=), c2, :($c1 * 2.0)),
        Expr(:(=), c3, :(unused + 1.0)),
        :(local_rate = $c2 + k),
    ]
    writes = Any[:(nz[idx] = local_rate)]

    pruned = ChemMechSim._needed_setup_prefix(setup, writes)

    @test pruned == Any[
        Expr(:(=), c1, :(a + b)),
        Expr(:(=), c2, :($c1 * 2.0)),
        :(local_rate = $c2 + k),
    ]
end

@testset "_needed_setup_prefix preserves sparse alias setup" begin
    c1 = Symbol("##cse#1")
    c2 = Symbol("##cse#2")
    setup = Any[
        :(nz = J.nzval),
        :(idx = 1),
        Expr(:(=), c1, :(a + b)),
        Expr(:(=), c2, :(unused + 1.0)),
    ]
    writes = Any[:(nz[idx] = $c1)]

    pruned = ChemMechSim._needed_setup_prefix(setup, writes)

    @test pruned == Any[
        :(nz = J.nzval),
        :(idx = 1),
        Expr(:(=), c1, :(a + b)),
    ]
end

@testset "_needed_setup_prefix preserves dependencies from evaluated setup lhs" begin
    c1 = Symbol("##cse#1")
    c2 = Symbol("##cse#2")
    setup = Any[
        Expr(:(=), c1, :(idx + offset)),
        Expr(:(=), c2, :(unused + 1.0)),
        Expr(:(=), Expr(:ref, :buf, c1), 0.0),
    ]
    writes = Any[:(nz[out_idx] = value)]

    pruned = ChemMechSim._needed_setup_prefix(setup, writes)

    @test pruned == Any[
        Expr(:(=), c1, :(idx + offset)),
        Expr(:(=), Expr(:ref, :buf, c1), 0.0),
    ]
end

@testset "_rewrite_setup_prefix rewrites evaluated setup lhs refs" begin
    c1 = Symbol("##cse#1")
    setup = Any[
        Expr(:(=), Expr(:ref, :buf, c1), 0.0),
    ]

    rewritten = ChemMechSim._rewrite_setup_prefix(setup, Dict(c1 => 1), :work)

    @test rewritten == Any[
        :(buf[work[1]] = 0.0),
    ]
end

@testset "build_shared_cse_jac returns stats when requested" begin
    sys = _shared_cse_stats_system()

    jac!, J_proto, stats = ChemMechSim.build_shared_cse_jac(
        sys; cse_chunk_size=1, write_chunk_size=1, return_stats=true)

    @test jac! isa Function
    @test J_proto isa SparseMatrixCSC{Float64}
    @test stats.n_setup_stmts > 0
    @test stats.n_sparse_writes == length(nonzeros(J_proto))
    @test stats.n_skipped_sparse_trailers == 1
    @test stats.n_cse_float > 0
    @test stats.n_cse_bool == 0
    @test stats.n_env_symbols > 0
    @test stats.n_setup_chunks > 0
    @test stats.n_write_chunks == stats.n_sparse_writes
    @test stats.cse_chunk_size == 1
    @test stats.write_chunk_size == 1
    @test stats.cse_node_budget == 80_000
    @test stats.write_node_budget == 80_000
end

@testset "build_shared_cse_jac legacy chunk sizes still split with large node budgets" begin
    phase = _shared_cse_stats_phase()
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 1.0, "B" => 0.2, "C" => 0.0), (0.0, 0.1))

    jac_large!, J_large_proto, large_stats = ChemMechSim.build_shared_cse_jac(
        sys;
        cse_chunk_size=1_000_000,
        write_chunk_size=1_000_000,
        cse_node_budget=1_000_000,
        write_node_budget=1_000_000,
        return_stats=true,
    )
    jac_legacy!, J_legacy_proto, legacy_stats = ChemMechSim.build_shared_cse_jac(
        sys;
        cse_chunk_size=1,
        write_chunk_size=1,
        cse_node_budget=1_000_000,
        write_node_budget=1_000_000,
        return_stats=true,
    )

    J_large = copy(J_large_proto)
    J_legacy = copy(J_legacy_proto)
    fill!(J_large.nzval, NaN)
    fill!(J_legacy.nzval, NaN)
    jac_large!(J_large, prob.u0, prob.p, 0.0)
    jac_legacy!(J_legacy, prob.u0, prob.p, 0.0)

    @test J_legacy_proto.rowval == J_large_proto.rowval
    @test J_legacy_proto.colptr == J_large_proto.colptr
    @test J_legacy.nzval ≈ J_large.nzval
    @test legacy_stats.n_setup_chunks > large_stats.n_setup_chunks
    @test legacy_stats.n_write_chunks > large_stats.n_write_chunks
    @test legacy_stats.n_setup_chunks == legacy_stats.n_cse_float + legacy_stats.n_cse_bool
    @test legacy_stats.n_write_chunks == legacy_stats.n_sparse_writes
    @test legacy_stats.cse_chunk_size == 1
    @test legacy_stats.write_chunk_size == 1
    @test legacy_stats.cse_node_budget == 1_000_000
    @test legacy_stats.write_node_budget == 1_000_000
end

@testset "build_shared_cse_jac chunks by node budget" begin
    phase = _shared_cse_stats_phase()
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 1.0, "B" => 0.2, "C" => 0.0), (0.0, 0.1))

    jac_default!, J_default_proto, default_stats = ChemMechSim.build_shared_cse_jac(
        sys; return_stats=true)
    jac_small!, J_small_proto, small_stats = ChemMechSim.build_shared_cse_jac(
        sys; cse_node_budget=3, write_node_budget=3, return_stats=true)

    J_default = copy(J_default_proto)
    J_small = copy(J_small_proto)
    fill!(J_default.nzval, NaN)
    fill!(J_small.nzval, NaN)
    jac_default!(J_default, prob.u0, prob.p, 0.0)
    jac_small!(J_small, prob.u0, prob.p, 0.0)

    @test J_small_proto.rowval == J_default_proto.rowval
    @test J_small_proto.colptr == J_default_proto.colptr
    @test J_small.nzval ≈ J_default.nzval
    @test small_stats.n_setup_chunks > default_stats.n_setup_chunks
    @test small_stats.n_write_chunks > default_stats.n_write_chunks
    @test default_stats.cse_node_budget == 80_000
    @test default_stats.write_node_budget == 80_000
    @test small_stats.cse_node_budget == 3
    @test small_stats.write_node_budget == 3
end

@testset "build_shared_cse_jac keeps old two-value return by default" begin
    sys = _shared_cse_stats_system()

    result = ChemMechSim.build_shared_cse_jac(sys)

    @test result isa Tuple
    @test length(result) == 2
end

@testset "build_shared_cse_jac rejects invalid chunk budgets" begin
    sys = _shared_cse_stats_system()

    @test_throws ArgumentError ChemMechSim.build_shared_cse_jac(sys; cse_chunk_size=0)
    @test_throws ArgumentError ChemMechSim.build_shared_cse_jac(sys; write_chunk_size=0)
    @test_throws ArgumentError ChemMechSim.build_shared_cse_jac(sys; cse_node_budget=0)
    @test_throws ArgumentError ChemMechSim.build_shared_cse_jac(sys; write_node_budget=0)
end
