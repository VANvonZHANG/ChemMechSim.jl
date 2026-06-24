using Test
using ChemMechSim

@testset "stubs" begin
    a = SpeciesData(id=1, name="A")
    b = SpeciesData(id=2, name="B")
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=[a, b], reactions=[rxn])

    # Each stub errors with a clear "framework" message referencing the phase plan.
    for f in (lower_to_mtk, simulate, build_problem, extract_system,
              generate_function, generate_jacobian, validate, import_from_catalyst)
        @test_throws ErrorException f(mech)
    end
    @test_throws ErrorException lower_reaction(rxn, mech, nothing, MechanismConfig())
    @test_throws ErrorException ChemPhaseSystem(mech)
    @test_throws ErrorException BatchReactor(mech)

    # ValidationReport is a concrete data type (no computation needed)
    rep = ValidationReport()
    @test isempty(rep.errors) && isempty(rep.warnings) && isempty(rep.info)
end
