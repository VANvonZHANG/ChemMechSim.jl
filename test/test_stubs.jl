using Test
using ChemMechSim

# After Phase 2.5b, only generate_jacobian remains an unimplemented stub (Phase 3).
# validate is now real (covered by test_validation.jl).
@testset "remaining stubs" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=[a, b], reactions=[rxn])
    @test_throws ErrorException generate_jacobian(mech)
    rep = ValidationReport()
    @test isempty(rep.errors) && isempty(rep.warnings) && isempty(rep.info)
end
