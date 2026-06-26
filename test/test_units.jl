using Test
using ChemMechSim
using DynamicQuantities
const CU = ChemMechSim.ChemUnits

@testset "ChemUnits" begin
    @test CU.conc == u"mol/m^3"
    @test CU.molmass == u"kg/mol"

    # canonical: Real passes through unchanged
    @test CU.canonical(0.018015, CU.molmass) == 0.018015

    # canonical: Quantity converted to reference unit, stripped to number
    @test CU.canonical(18.015u"g/mol", CU.molmass) ≈ 0.018015 atol = 1e-9

    # NaN Real passes through (unspecified value)
    @test isnan(CU.canonical(NaN, CU.molmass))

    # Wrong-dimension quantity is rejected
    @test_throws ErrorException CU.canonical(1.0u"m", CU.molmass)
end

import ModelingToolkit
using OrdinaryDiffEq
import Catalyst: @species
import ModelingToolkit: @named

@testset "unit-aware lowering: dimension check passes + solves with k default" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    mech = Mechanism(species=[a, b],
        reactions=[ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                                kinetics=ElementaryArrhenius(0.5, 0.0, 0.0))])   # 1st order, k=0.5 [1/s]
    phase = ChemPhaseSystem(mech)
    sys = extract_system(phase)
    # k is a parameter with derived unit s^-1 and default 0.5; system builds => dim check passed
    @test any(p -> String(ModelingToolkit.getname(p)) == "k_1_A", ModelingToolkit.parameters(sys))
    # solve uses k's default (0.5) -> A(1)=exp(-0.5)
    A = _var(sys, "A")
    sol = simulate(phase, (0.0, 1.0); u0=Dict("A" => 1.0, "B" => 0.0))
    @test sol(1.0; idxs=A) ≈ exp(-0.5)  rtol = 1e-6
end

@testset "unit-aware lowering: mismatched k unit is rejected (ValidationError)" begin
    # Build a system by hand where k has a WRONG unit; System construction must reject it.
    # (Dimension check fires at @named sys = System(eqs, t), not at mtkcompile.)
    t = ModelingToolkit.t; D = ModelingToolkit.D
    A = ModelingToolkit.setmetadata(only(@species A(t)), ModelingToolkit.VariableUnit, CU.conc)
    kbad = ChemMechSim.rate_param(:kbad, 0.5, u"m/s")   # wrong unit for a 1st-order rate
    @test_throws Exception (@named sys = ODESystem([D(A) ~ -kbad * A], t))
end
