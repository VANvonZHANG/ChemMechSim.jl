using Test
using ChemMechSim
using ChemMechSim: PlogPoint, PlogRate
using ModelingToolkit
using SparseArrays

function _full_sparse_jacobian(sys, u, p, t)
    jac_full! = ModelingToolkit.generate_jacobian(
        sys; sparse=true, expression=Val{false}, wrap_gfw=Val{false})[2]
    J_full = SparseArrays.similar(ModelingToolkit.calculate_jacobian(sys; sparse=true), Float64)
    fill!(J_full.nzval, NaN)
    jac_full!(J_full, u, p, t)
    return J_full
end

_stored_pattern(J) = Set(zip(findnz(J)[1], findnz(J)[2]))

function _expected_pattern(sys, row_col_groups)
    state_idx = ChemMechSim._state_name_index(sys)
    return Set((state_idx[row_name], state_idx[col_name])
               for (row_names, col_names) in row_col_groups
               for row_name in row_names
               for col_name in col_names)
end

@testset "reaction-sharded Jacobian builds sparse template from reaction graph" begin
    mech = Mechanism(
        species = [
            SpeciesData(id=1, name="A"),
            SpeciesData(id=2, name="B"),
            SpeciesData(id=3, name="C"),
        ],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0, 2 => 1.0),
                products = Dict(3 => 1.0),
                kinetics = ElementaryArrhenius(5.0, 0.0, 0.0),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)

    _, J_proto, stats = ChemMechSim.build_reaction_sharded_jac(
        mech; config=config, checks=false, return_stats=true)

    observed = _stored_pattern(J_proto)
    expected = _expected_pattern(sys, [(("A", "B", "C", "P"), ("A", "B"))])

    @test observed == expected
    @test stats.n_nonzeros == length(observed)
end

@testset "reaction-sharded Jacobian prototype matches full MTK Jacobian on fixedT chain" begin
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
                kinetics = ElementaryArrhenius(2.0, 0.0, 0.0),
            ),
            ReactionData(
                reactants = Dict(2 => 1.0),
                products = Dict(3 => 1.0),
                kinetics = ElementaryArrhenius(3.0, 0.0, 0.0),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 2.0, "B" => 0.25, "C" => 0.0), (0.0, 0.1))

    jac_sharded!, J_proto, stats = ChemMechSim.build_reaction_sharded_jac(
        mech; config=config, checks=false, reaction_shard_size=1, return_stats=true)
    J_sharded = copy(J_proto)
    fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)

    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test J_sharded isa SparseMatrixCSC{Float64}
    @test stats.n_reactions == 2
    @test stats.n_states == length(ModelingToolkit.unknowns(sys))
    @test stats.n_nonzeros == length(nonzeros(J_proto))
    @test stats.n_reaction_shards == 2
    @test _stored_pattern(J_proto) == _expected_pattern(
        sys, [
            (("A", "B", "P"), ("A",)),
            (("B", "C", "P"), ("B",)),
        ])
    @test Matrix(J_sharded) ≈ Matrix(J_full)
end

@testset "reaction-sharded Jacobian assembles branching elementary fixedT stoichiometry" begin
    mech = Mechanism(
        species = [
            SpeciesData(id=1, name="A"),
            SpeciesData(id=2, name="B"),
            SpeciesData(id=3, name="C"),
        ],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0, 2 => 1.0),
                products = Dict(3 => 1.0),
                kinetics = ElementaryArrhenius(5.0, 0.0, 0.0),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 2.0, "B" => 3.0, "C" => 0.0), (0.0, 0.1))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(
        mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)

    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test _stored_pattern(J_proto) == _expected_pattern(
        sys, [(("A", "B", "C", "P"), ("A", "B"))])
    @test Matrix(J_sharded) ≈ Matrix(J_full)
end

@testset "reaction-sharded Jacobian honors fixedT parameter overrides" begin
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
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    Tparam = only(filter(p -> String(ModelingToolkit.getname(p)) == "T",
                         ModelingToolkit.parameters(sys)))
    prob = build_problem(
        phase, Dict("A" => 2.0, "B" => 0.0), (0.0, 0.1); params=[Tparam => 700.0])

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(
        mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)

    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test _stored_pattern(J_proto) == _expected_pattern(
        sys, [(("A", "B", "P"), ("A",))])
    @test Matrix(J_sharded) ≈ Matrix(J_full)
end

@testset "reaction-sharded Jacobian handles zero concentration product-rule partials" begin
    mech = Mechanism(
        species = [
            SpeciesData(id=1, name="A"),
            SpeciesData(id=2, name="B"),
            SpeciesData(id=3, name="C"),
        ],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0, 2 => 1.0),
                products = Dict(3 => 1.0),
                kinetics = ElementaryArrhenius(5.0, 0.0, 0.0),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 0.0, "B" => 3.0, "C" => 0.0), (0.0, 0.1))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(
        mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)

    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test _stored_pattern(J_proto) == _expected_pattern(
        sys, [(("A", "B", "C", "P"), ("A", "B"))])
    @test Matrix(J_sharded) ≈ Matrix(J_full)
end

@testset "reaction-sharded Jacobian stays finite for fractional order at zero concentration" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 0.5),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(2.0, 0.0, 0.0),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 0.0, "B" => 0.0), (0.0, 0.1))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(
        mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)

    # At zero concentration a fractional-order rate has a singular Symbolics derivative
    # (0^(ν-1) = Inf, and 0·Inf = NaN). The sharded Jacobian sanitizes these to 0 — the
    # reaction is inactive (rate = 0) so the true contribution is 0 — so the Jacobian is
    # finite and usable by the ODE solver. (Symbolics/MTK's raw _full_sparse_jacobian is NOT
    # finite here, so we assert finiteness rather than agreement with it.)
    @test _stored_pattern(J_proto) == _expected_pattern(
        sys, [(("A", "B", "P"), ("A",))])
    @test all(isfinite, J_sharded.nzval)
end

@testset "reaction-sharded Jacobian rejects reversible reactions in prototype scope" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(1.0, 0.0, 0.0),
                reverse_policy = ThermoReverse(),
            ),
        ],
    )

    err = try
        ChemMechSim.build_reaction_sharded_jac(
            mech; config=convenience_config(:fixedT), checks=false)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("reaction 1", sprint(showerror, err))
end

@testset "reaction-sharded Jacobian supports fixedT third-body reactions" begin
    mech = Mechanism(
        species = [
            SpeciesData(id=1, name="A"),
            SpeciesData(id=2, name="B"),
            SpeciesData(id=3, name="M"),
        ],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ThirdBodyArrhenius(
                    ElementaryArrhenius(2.0, 0.0, 0.0), Dict(3 => 2.0)),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 2.0, "B" => 0.0, "M" => 3.0), (0.0, 0.1))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(
        mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)

    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test _stored_pattern(J_proto) == _expected_pattern(
        sys, [(("A", "B", "P"), ("A", "B", "M"))])
    @test Matrix(J_sharded) ≈ Matrix(J_full)
end

@testset "reaction-sharded Jacobian supports fixedT PLOG pressure derivatives" begin
    plog = PlogRate([
        PlogPoint(101325.0, 1.0, 0.0, 0.0),
        PlogPoint(1013250.0, 3.0, 0.0, 0.0),
    ])
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = plog,
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(
        phase, Dict("A" => 2.0, "B" => 0.0, "P" => 202650.0), (0.0, 0.1))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(
        mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)

    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test _stored_pattern(J_proto) == _expected_pattern(
        sys, [(("A", "B", "P"), ("A", "P"))])
    @test Matrix(J_sharded) ≈ Matrix(J_full)
end

@testset "reaction-sharded Jacobian rejects invalid shard sizes" begin
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

    @test_throws ArgumentError ChemMechSim.build_reaction_sharded_jac(
        mech; config=convenience_config(:fixedT), checks=false, reaction_shard_size=0)
end

@testset "reaction-sharded Jacobian rejects unsupported configs in prototype scope" begin
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

    err = try
        ChemMechSim.build_reaction_sharded_jac(mech; config=MechanismConfig(), checks=false)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("unsupported config", sprint(showerror, err))
end

@testset "build_problem exposes opt-in reaction_sharded Jacobian strategy" begin
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
    reactor = BatchReactor(mech; mode=:fixedT, checks=false)

    prob = build_problem(
        reactor, Dict("A" => 1.0, "B" => 0.0), (0.0, 0.1);
        jac_strategy=:reaction_sharded)

    @test prob.f.jac !== nothing
    @test prob.f.jac_prototype isa SparseMatrixCSC

    J = copy(prob.f.jac_prototype)
    prob.f.jac(J, prob.u0, prob.p, 0.0)

    @test all(isfinite, nonzeros(J))
    @test ChemMechSim._normalize_jac_strategy(true, false, :auto) === :shared_cse
end
