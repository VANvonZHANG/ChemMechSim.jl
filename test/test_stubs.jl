using Test
using ChemMechSim

# After Phase 1, only these interface functions remain unimplemented stubs:
#   generate_jacobian (Phase 3), validate (Phase 2.5), BatchReactor (Phase 2).
# Every other former stub is now real and covered by its own test file.
@testset "remaining stubs" begin
    a = SpeciesData(id=1, name="A")
    b = SpeciesData(id=2, name="B")
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=[a, b], reactions=[rxn])

    for f in (generate_jacobian, validate)
        @test_throws ErrorException f(mech)
    end
    @test_throws ErrorException BatchReactor(mech)

    # ValidationReport is a concrete data type (no computation needed)
    rep = ValidationReport()
    @test isempty(rep.errors) && isempty(rep.warnings) && isempty(rep.info)
end
