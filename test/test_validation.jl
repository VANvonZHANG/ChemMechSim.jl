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

@testset "validate: NASA temp-range coverage" begin
    n = NASA7((2.5,0,0,0,0,-2500.0,0.0),(2.5,0,0,0,0,-2500.0,0.0),200.0,1000.0,3500.0)
    sp = SpeciesData(id=1,name="A",thermo=n,elements=Dict("X"=>1),molecular_weight=0.001)
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(1=>1.0), kinetics=ElementaryArrhenius(1.0,0.0,0.0))
    mech_ok  = Mechanism(species=[sp], reactions=[rxn], elements=["X"])
    @test isempty(validate(mech_ok; T_range=(300.0,900.0)).warnings)         # within [200,3500]
    w = validate(mech_ok; T_range=(300.0,4000.0)).warnings                    # 4000 > Thigh=3500
    @test !isempty(w) && any(occursin("range", l) for l in w)
    @test isempty(validate(mech_ok).warnings)                                  # T_range=nothing → skip
end

@testset "validate: third-body efficiency species exist" begin
    a = SpeciesData(id=1,name="A"); b = SpeciesData(id=2,name="B"); m = SpeciesData(id=3,name="M")
    good = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                        kinetics=ThirdBodyArrhenius(ElementaryArrhenius(1.0,0.0,0.0), Dict(3=>1.0)))
    mech_ok = Mechanism(species=[a,b,m], reactions=[good])
    @test isempty(validate(mech_ok).errors)
    bad = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ThirdBodyArrhenius(ElementaryArrhenius(1.0,0.0,0.0), Dict(99=>1.0)))  # id 99 absent
    mech_bad = Mechanism(species=[a,b,m], reactions=[bad])
    @test !isempty(validate(mech_bad).errors) && any(occursin("third-body", l) for l in validate(mech_bad).errors)
end

@testset "validate: duplicate-reaction marking" begin
    a = SpeciesData(id=1,name="A"); b = SpeciesData(id=2,name="B")
    r1 = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0), kinetics=ElementaryArrhenius(1.0,0.0,0.0))
    r2 = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0), kinetics=ElementaryArrhenius(2.0,0.0,0.0))
    # identical stoichiometry, neither marked duplicate → warning
    mech = Mechanism(species=[a,b], reactions=[r1,r2])
    @test !isempty(validate(mech).warnings) && any(occursin("duplicate", l) for l in validate(mech).warnings)
end

@testset "validate: reverse-policy consistency" begin
    n = NASA7((2.5,0,0,0,0,-2500.0,0.0),(2.5,0,0,0,0,-2500.0,0.0),200.0,1000.0,3500.0)
    a = SpeciesData(id=1,name="A",thermo=n); b = SpeciesData(id=2,name="B",thermo=n)
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(1.0,0.0,0.0), reverse_policy=ThermoReverse())
    @test isempty(validate(Mechanism(species=[a,b], reactions=[rxn])).errors)   # thermo present → ok
    a_no = SpeciesData(id=1,name="A")                                            # no thermo
    mech_bad = Mechanism(species=[a_no,b], reactions=[rxn])
    errs = validate(mech_bad).errors
    @test !isempty(errs) && any(occursin("thermo", l) for l in errs)
end
