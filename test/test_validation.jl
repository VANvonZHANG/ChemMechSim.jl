using Test
using ChemMechSim

@testset "validate: clean mechanism passes" begin
    H2  = SpeciesData(id=1, name="H2",  elements=Dict("H" => 2), molecular_weight=0.002016)
    O2  = SpeciesData(id=2, name="O2",  elements=Dict("O" => 2), molecular_weight=0.031998)
    H2O = SpeciesData(id=3, name="H2O", elements=Dict("H" => 2, "O" => 1), molecular_weight=0.018015)
    rxn = ReactionData(reactants=Dict(1 => 2.0, 2 => 1.0), products=Dict(3 => 2.0),   # 2H2+O2->2H2O
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=[H2, O2, H2O], reactions=[rxn], elements=["H", "O"])
    rep = validate(mech)
    @test isempty(rep.errors)
    @test isempty(rep.warnings)
end

@testset "validate: unbalanced reaction -> error" begin
    H2  = SpeciesData(id=1, name="H2",  elements=Dict("H" => 2), molecular_weight=0.002016)
    O2  = SpeciesData(id=2, name="O2",  elements=Dict("O" => 2), molecular_weight=0.031998)
    H2O = SpeciesData(id=3, name="H2O", elements=Dict("H" => 2, "O" => 1), molecular_weight=0.018015)
    rxn = ReactionData(reactants=Dict(1 => 1.0, 2 => 1.0), products=Dict(3 => 1.0),   # H2+O2->H2O (unbalanced)
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=[H2, O2, H2O], reactions=[rxn], elements=["H", "O"])
    rep = validate(mech)
    @test !isempty(rep.errors)
    @test occursin("element", lowercase(first(rep.errors)))
end

@testset "validate: missing molecular weight -> warning" begin
    H2  = SpeciesData(id=1, name="H2", elements=Dict("H" => 2), molecular_weight=0.002016)
    O2  = SpeciesData(id=2, name="O2", elements=Dict("O" => 2), molecular_weight=NaN)   # missing MW
    OH  = SpeciesData(id=3, name="OH", elements=Dict("O" => 1, "H" => 1), molecular_weight=0.017007)
    rxn = ReactionData(reactants=Dict(1 => 1.0, 2 => 1.0), products=Dict(3 => 2.0),     # H2+O2->2OH (balanced)
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=[H2, O2, OH], reactions=[rxn], elements=["H", "O"])
    rep = validate(mech)
    @test isempty(rep.errors)
    @test any(w -> occursin("O2", w), rep.warnings)
end

@testset "validate: empty element list -> skip-warning, no crash" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    mech = Mechanism(species=[a, b],
        reactions=[ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                                kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))])
    rep = validate(mech)
    @test isempty(rep.errors)
    @test !isempty(rep.warnings)
end
