using Test
using ChemMechSim

# generate_jacobian is now real (Phase 3). validate is covered by test_validation.jl.
@testset "generate_jacobian: exports standalone Jacobian code" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=[a, b], reactions=[rxn])
    sys = extract_system(BatchReactor(mech))
    jac_code = generate_jacobian(sys)
    @test jac_code isa String || jac_code isa Expr
    @test occursin("function", string(jac_code))         # emits a function
    jac_sparse = generate_jacobian(sys; sparse=true)
    @test jac_sparse isa String || jac_sparse isa Expr   # sparse path also produces code
    # BatchReactor dispatch
    @test typeof(generate_jacobian(BatchReactor(mech))) <: Union{String,Expr}
end

@testset "ValidationReport default" begin
    rep = ValidationReport()
    @test isempty(rep.errors) && isempty(rep.warnings) && isempty(rep.info)
end
