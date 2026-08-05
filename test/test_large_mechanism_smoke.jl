using Test
using ChemMechSim
using SparseArrays

# Large-mechanism CONSTRUCTION smoke test for the reaction-sharded Jacobian: verifies the
# shard-batching build path stays tractable (no full symbolic Jacobian is ever materialized)
# on realistic mechanisms. Guarded on local fixture availability. End-to-end large-mechanism
# jac=! solve timing is covered separately (examples + the Aramco validation).

function _fixture(path...)
    p = joinpath(@__DIR__, "..", "examples", "mechanism", path...)
    return isfile(p) ? p : nothing
end

"Subset of a mechanism restricted to reaction-sharded-supported kinetics + reverse policies,
 capped at `limit` reactions (keeps all species/thermo so reversible rates have NASA7)."
function _supported_subset(mech::Mechanism; limit::Int=300)
    supported = ReactionData[]
    for rx in mech.reactions
        kin_ok = rx.kinetics isa ElementaryArrhenius ||
                 rx.kinetics isa ThirdBodyArrhenius ||
                 rx.kinetics isa PlogRate ||
                 rx.kinetics isa LindemannFalloff ||
                 rx.kinetics isa TroeFalloff
        rev_ok = rx.reverse_policy isa Irreversible ||
                 rx.reverse_policy isa ExplicitReverse ||
                 rx.reverse_policy isa ThermoReverse
        kin_ok && rev_ok && push!(supported, rx)
        length(supported) >= limit && break
    end
    return Mechanism(species=mech.species, reactions=supported,
                     thermo=mech.thermo, elements=mech.elements)
end

@testset "large mechanism reaction-sharded smoke" begin
    # FFCM2 (96 sp) keeps the smoke fast while exercising the shard-batching build at a
    # realistic scale in BOTH energy regimes. Full AramcoMech3.0 (581 sp) end-to-end jac
    # timing is validated separately — its lowering is too slow to lower 4× in a unit test.
    path = _fixture("FFCM2.yaml")
    path === nothing && (@test_skip false; return)
    mech = load_mechanism(path)
    sub = _supported_subset(mech; limit=300)
    @test length(sub.reactions) > 0
    shard = 50

    for mode in (:fixedT, :adiabatic_constV)
        config = convenience_config(mode)
        jac!, J_proto, stats = ChemMechSim.build_reaction_sharded_jac(
            sub; config=config, checks=false,
            reaction_shard_size=shard, return_stats=true)

        # Shards are partitioned: Path S (simple) + Path M (M_eff), each batched independently.
        n_simple = count(rx -> !ChemMechSim._reaction_uses_meff(rx), sub.reactions)
        n_meff = length(sub.reactions) - n_simple
        expected_shards = cld(n_simple, shard) + cld(n_meff, shard)
        @test J_proto isa SparseMatrixCSC{Float64}
        @test stats.n_reactions == length(sub.reactions)
        @test stats.n_reaction_shards == expected_shards
        @test stats.n_compiled_derivative_functions == stats.n_reaction_shards
        @test stats.n_nonzeros == length(nonzeros(J_proto))
        @test jac! !== nothing
    end
end
