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

@testset "reaction-sharded pattern includes explicit reverse products as dependencies" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(1.0, 0.0, 0.0),
                reverse_policy = ExplicitReverse(ElementaryArrhenius(0.5, 0.0, 0.0)),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    _, J_proto = ChemMechSim.build_reaction_sharded_jac(mech; config=config, checks=false)

    @test _stored_pattern(J_proto) == _expected_pattern(sys, [(("A", "B", "P"), ("A", "B"))])
end

@testset "reaction-sharded ThermoReverse reuses opaque keq" begin
    mech = load_mechanism(joinpath(@__DIR__, "data", "gri30.yaml"))
    rxidx = findfirst(r -> r.reverse_policy isa ChemMechSim.ThermoReverse &&
                           r.kinetics isa ChemMechSim.ElementaryArrhenius,
                      mech.reactions)
    @test rxidx !== nothing
    sub = Mechanism(species=mech.species, reactions=[mech.reactions[rxidx]],
                    thermo=mech.thermo, elements=mech.elements)
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(sub; config=config, checks=false)
    sys = extract_system(phase)
    u0 = Dict(String(sp.name) => 1.0e-6 for sp in sub.species)
    for sid in keys(sub.reactions[1].reactants)
        u0[String(ChemMechSim.species_by_id(sub, sid).name)] = 1.0
    end
    prob = build_problem(phase, u0, (0.0, 1.0e-6))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(sub; config=config, checks=false)
    J_sharded = copy(J_proto)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)
    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test Matrix(J_sharded) ≈ Matrix(J_full) rtol=1e-8 atol=1e-8
end

@testset "reaction-sharded net rate uses lowering protocol" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0),
                products = Dict(2 => 1.0),
                kinetics = ElementaryArrhenius(2.0, 0.0, 0.0),
                reverse_policy = ExplicitReverse(ElementaryArrhenius(0.5, 0.0, 0.0)),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 2.0, "B" => 3.0), (0.0, 0.1))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(
        mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)

    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)
    @test Matrix(J_sharded) ≈ Matrix(J_full)
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

@testset "reaction-sharded third-body expands M_eff to all species" begin
    # A + B + M -> C  (M is an inert third body with non-default efficiency).
    # [M_eff] couples A, B, C, M. A correct Jacobian has ∂rate/∂c_m ≠ 0 for ALL of them.
    # This test specifically guards the runtime [M_eff] → all-species expansion: a regression
    # to "differentiate only reactants" would miss the C and M columns.
    mech = Mechanism(
        species = [
            SpeciesData(id=1, name="A"),
            SpeciesData(id=2, name="B"),
            SpeciesData(id=3, name="C"),
            SpeciesData(id=4, name="M"),
        ],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0, 2 => 1.0),
                products = Dict(3 => 1.0),
                kinetics = ThirdBodyArrhenius(
                    ElementaryArrhenius(2.0, 0.0, 0.0), Dict(4 => 3.0)),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A" => 1.0, "B" => 0.5, "C" => 0.0, "M" => 2.0), (0.0, 0.1))
    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)
    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)
    # Pattern: rows are the reactant/product species (A, B, C) plus the pressure row (P).
    # M is the third body (NOT a reactant/product → zero net stoich → no row).
    # Columns include EVERY species (A, B, C, M) because [M_eff] couples them all.
    @test _stored_pattern(J_proto) == _expected_pattern(
        sys, [(("A", "B", "C", "P"), ("A", "B", "C", "M"))])
    @test Matrix(J_sharded) ≈ Matrix(J_full)
end

@testset "reaction-sharded supports fixedT Lindemann falloff" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B"), SpeciesData(id=3, name="C")],
        reactions = [
            ReactionData(
                reactants=Dict(1=>1.0, 2=>1.0),
                products=Dict(3=>1.0),
                kinetics=LindemannFalloff(
                    ElementaryArrhenius(1.0e6, 0.0, 0.0),
                    ElementaryArrhenius(2.0, 0.0, 0.0),
                    Dict(2=>2.0)),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A"=>1.0, "B"=>2.0, "C"=>0.0), (0.0, 0.1))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)
    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test _stored_pattern(J_proto) == _expected_pattern(sys, [(("A", "B", "C", "P"), ("A", "B", "C"))])
    @test Matrix(J_sharded) ≈ Matrix(J_full)
end

@testset "reaction-sharded supports fixedT Troe falloff" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B"), SpeciesData(id=3, name="C")],
        reactions = [
            ReactionData(
                reactants=Dict(1=>1.0, 2=>1.0),
                products=Dict(3=>1.0),
                kinetics=TroeFalloff(
                    ElementaryArrhenius(1.0e6, 0.0, 0.0),
                    ElementaryArrhenius(2.0, 0.0, 0.0),
                    Dict(2=>2.0),
                    TroeParams(0.5, 1000.0, 1.0e30, 100.0)),
            ),
        ],
    )
    config = convenience_config(:fixedT)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A"=>1.0, "B"=>2.0, "C"=>0.0), (0.0, 0.1))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)
    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test Matrix(J_sharded) ≈ Matrix(J_full) rtol=1e-8 atol=1e-8
end

@testset "reaction-sharded supports adiabatic constV Jacobian" begin
    # Distinct NASA7 (Δu ≠ 0), A→2B (Δν ≠ 0), and a T-dependent rate: together these exercise
    # the species rows incl. the T column, the dense energy (T) row (cvsum couples all species),
    # and the pressure (P) row (coupled to dT/dt). Compares the full (N+2)×(N+2) Jacobian.
    nasa_A = NASA7((4.0,0,0,0,0,1000.0,0.0),(4.0,0,0,0,0,1000.0,0.0), 200.0, 1000.0, 3500.0)
    nasa_B = NASA7((4.0,0,0,0,0,3000.0,0.0),(4.0,0,0,0,0,3000.0,0.0), 200.0, 1000.0, 3500.0)
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A", thermo=nasa_A), SpeciesData(id=2, name="B", thermo=nasa_B)],
        reactions = [ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>2.0),
                                  kinetics=ElementaryArrhenius(2.0, 0.5, 500.0))],
    )
    config = convenience_config(:adiabatic_constV)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A"=>1.0, "B"=>0.5, "T"=>1000.0, "P"=>101325.0), (0.0, 0.1))

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(mech; config=config, checks=false)
    J_sharded = copy(J_proto)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)
    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    @test Matrix(J_sharded) ≈ Matrix(J_full) rtol=1e-8 atol=1e-8
end

@testset "reaction-sharded supports fixedT PLOG" begin
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

@testset "reaction_shard_size controls compiled derivative shards" begin
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B"), SpeciesData(id=3, name="C")],
        reactions = [
            ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                         kinetics=ElementaryArrhenius(1.0, 0.0, 0.0)),
            ReactionData(reactants=Dict(2=>1.0), products=Dict(3=>1.0),
                         kinetics=ElementaryArrhenius(2.0, 0.0, 0.0)),
            ReactionData(reactants=Dict(3=>1.0), products=Dict(1=>1.0),
                         kinetics=ElementaryArrhenius(3.0, 0.0, 0.0)),
        ],
    )
    jac!, J_proto, stats = ChemMechSim.build_reaction_sharded_jac(
        mech; config=convenience_config(:fixedT), checks=false,
        reaction_shard_size=2, return_stats=true)

    @test stats.n_reactions == 3
    @test stats.n_reaction_shards == 2
    @test stats.n_compiled_derivative_functions == 2
    @test stats.n_compiled_derivative_functions == stats.n_reaction_shards
    @test stats.max_shard_reactions == 2
    @test stats.n_nonzeros == length(nonzeros(J_proto))
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
    @test ChemMechSim._normalize_jac_strategy(true, false, :auto) === :auto
end

@testset "reaction-sharded jac reads prob.p in correct parameter order (build_problem path)" begin
    # Regression (2026-07-24): build_reaction_sharded_jac used to construct a FRESH
    # ChemPhaseSystem internally, but ChemPhaseSystem lowering orders parameters
    # non-deterministically (hashed-container iteration). For mechanisms with many
    # parameters (GRI30: 1390) the fresh system's parameter order desynced from the
    # ODEProblem's p (built from a different system instance), so every shard function
    # misread its rate parameters and produced a Jacobian off by 20-50 orders of
    # magnitude. The fix threads build_problem's own `sys` into build_reaction_sharded_jac
    # so the shard functions and prob.p share one parameter layout. This test builds via
    # build_problem(jac=true) — the real entry point — and finite-difference checks the
    # factored T-row, which is exquisitely sensitive to the rate-param misalignment.
    yaml = joinpath(@__DIR__, "..", "examples", "data", "gri30.yaml")
    isfile(yaml) || return   # skip when mechanism data isn't shipped
    mech = load_mechanism(yaml)
    reactor = BatchReactor(mech; mode=:adiabatic_constV, checks=false)
    u0d = Dict(sp.name => 1e-3 for sp in mech.species); u0d["T"] = 1500.0
    prob = build_problem(reactor, u0d, (0.0, 1e-3); jac=true)
    @test prob.f.jac !== nothing
    u = copy(prob.u0); p = prob.p
    du = similar(u); prob.f(du, u, p, 0.0)
    J = copy(prob.f.jac_prototype); prob.f.jac(J, u, p, 0.0)
    @test all(isfinite, nonzeros(J))
    @test maximum(abs, nonzeros(J)) < 1e16   # misalignment produced 1e40+ entries
    nm2i = ChemMechSim._state_name_index(extract_system(reactor))
    T_idx = nm2i["T"]; P_idx = nm2i["P"]
    Ja = Matrix(J)
    # forward-FD the CH4 and T columns (active, well-conditioned) and check the T and P rows.
    for cname in ("CH4", "T")
        c = nm2i[cname]
        h = 1e-8 * abs(u[c])
        dup = similar(u); u[c] += h; prob.f(dup, u, p, 0.0); u[c] -= h
        fd_T = (dup[T_idx] - du[T_idx]) / h
        fd_P = (dup[P_idx] - du[P_idx]) / h
        @test Ja[T_idx, c] ≈ fd_T rtol=1e-4 atol=1e-8
        @test Ja[P_idx, c] ≈ fd_P rtol=1e-4 atol=1e-8
    end
end

# Regression (2026-07-27): the Path M branch (third-body / falloff M_eff reactions) was built
# for Irreversible rates and omitted (a) product-species derivative columns for reversible M_eff
# rates (so d(reverse rate)/d(c_product) was never differentiated) and (b) the T-column
# species-row entries J[species_row, T] for every M_eff reaction. The Troe test above is
# Irreversible (no products), and the ThermoReverse test is Elementary (Path S, correct); the
# falloff/third-body × reversible intersection was untested. GRI30 has 41 such reactions, which
# caused GRI30 jac=true to take 314k FBDF steps vs the exact symbolic jac's 997.

@testset "reaction-sharded TroeFalloff + ThermoReverse matches exact symbolic jac" begin
    # GRI30 rxn 241: CH + N2 -> HCNN, TroeFalloff | ThermoReverse.
    yaml = joinpath(@__DIR__, "..", "examples", "data", "gri30.yaml")
    isfile(yaml) || return   # skip when mechanism data isn't shipped
    mech = load_mechanism(yaml)
    # Find CH + N2 -> HCNN (TroeFalloff + ThermoReverse) — rxn index 241 in canonical GRI30.
    id_by_name = Dict(String(sp.name) => sp.id for sp in mech.species)
    rxidx = findfirst(r -> r.kinetics isa ChemMechSim.TroeFalloff &&
                           r.reverse_policy isa ChemMechSim.ThermoReverse &&
                           haskey(r.reactants, id_by_name["CH"]) &&
                           haskey(r.reactants, id_by_name["N2"]) &&
                           haskey(r.products,  id_by_name["HCNN"]),
                      mech.reactions)
    @test rxidx !== nothing
    rx = mech.reactions[rxidx]
    sub = Mechanism(species=mech.species, reactions=[rx],
                    thermo=mech.thermo, elements=mech.elements)
    config = convenience_config(:adiabatic_constV)
    phase = ChemPhaseSystem(sub; config=config, checks=false)
    sys = extract_system(phase)
    # Well-conditioned state: all involved species present, others small, T = 1500.
    u0d = Dict(sp.name => 0.1 for sp in sub.species)
    for sid in keys(rx.reactants); u0d[String(ChemMechSim.species_by_id(sub, sid).name)] = 1.0; end
    for sid in keys(rx.products);  u0d[String(ChemMechSim.species_by_id(sub, sid).name)] = 1.0; end
    u0d["T"] = 1500.0
    prob = build_problem(phase, u0d, (0.0, 1e-3); jac=true)

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(sub; config=config, checks=false)
    J_sharded = copy(J_proto); fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)
    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    # Spot-check the previously-broken entries: J[CH, HCNN] (product col) and J[CH, T].
    nm2i = ChemMechSim._state_name_index(sys)
    As = Matrix(J_sharded); Af = Matrix(J_full)
    @test As[nm2i["CH"], nm2i["HCNN"]] ≈ Af[nm2i["CH"], nm2i["HCNN"]] rtol=1e-8 atol=1e-8
    @test As[nm2i["CH"], nm2i["T"]]    ≈ Af[nm2i["CH"], nm2i["T"]]    rtol=1e-8 atol=1e-8
    # And the full matrix.
    @test Matrix(J_sharded) ≈ Matrix(J_full) rtol=1e-8 atol=1e-8
end

@testset "reaction-sharded ThirdBody + ThermoReverse matches exact symbolic jac" begin
    # Synthetic A + B + M -> C with third-body M and ThermoReverse: covers the
    # third-body × reversible intersection (GRI30 has 12 such reactions). Adiabatic so the
    # T-column species-row scatter is exercised.
    nasa_A = NASA7((4.0,0,0,0,0,1000.0,0.0),(4.0,0,0,0,0,1000.0,0.0), 200.0, 1000.0, 3500.0)
    nasa_B = NASA7((4.0,0,0,0,0,3000.0,0.0),(4.0,0,0,0,0,3000.0,0.0), 200.0, 1000.0, 3500.0)
    nasa_C = NASA7((5.0,0,0,0,0,2000.0,0.0),(5.0,0,0,0,0,2000.0,0.0), 200.0, 1000.0, 3500.0)
    nasa_M = NASA7((3.5,0,0,0,0, 500.0,0.0),(3.5,0,0,0,0, 500.0,0.0), 200.0, 1000.0, 3500.0)
    mech = Mechanism(
        species = [SpeciesData(id=1, name="A", thermo=nasa_A),
                   SpeciesData(id=2, name="B", thermo=nasa_B),
                   SpeciesData(id=3, name="C", thermo=nasa_C),
                   SpeciesData(id=4, name="M", thermo=nasa_M)],
        reactions = [
            ReactionData(
                reactants = Dict(1 => 1.0, 2 => 1.0),
                products = Dict(3 => 1.0),
                kinetics = ThirdBodyArrhenius(
                    ElementaryArrhenius(2.0, 0.5, 500.0), Dict(4 => 3.0)),
                reverse_policy = ThermoReverse(),
            ),
        ],
    )
    config = convenience_config(:adiabatic_constV)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    sys = extract_system(phase)
    prob = build_problem(phase, Dict("A"=>1.0, "B"=>0.5, "C"=>0.3, "M"=>2.0, "T"=>1200.0),
                         (0.0, 0.1); jac=true)

    jac_sharded!, J_proto = ChemMechSim.build_reaction_sharded_jac(mech; config=config, checks=false)
    J_sharded = copy(J_proto); fill!(J_sharded.nzval, NaN)
    jac_sharded!(J_sharded, prob.u0, prob.p, 0.0)
    J_full = _full_sparse_jacobian(sys, prob.u0, prob.p, 0.0)

    # C is the product — its d(reverse rate)/d(c_C) column was missing before the fix.
    nm2i = ChemMechSim._state_name_index(sys)
    As = Matrix(J_sharded); Af = Matrix(J_full)
    @test As[nm2i["A"], nm2i["C"]] ≈ Af[nm2i["A"], nm2i["C"]] rtol=1e-8 atol=1e-8
    @test As[nm2i["A"], nm2i["T"]] ≈ Af[nm2i["A"], nm2i["T"]] rtol=1e-8 atol=1e-8
    @test Matrix(J_sharded) ≈ Matrix(J_full) rtol=1e-8 atol=1e-8
end
