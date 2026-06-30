using Test
using ChemMechSim
using ModelingToolkit
using ModelingToolkit: unknowns, getname
using Catalyst
import ChemMechSim: catalyst_native, catalyst_lowering, direct_mtk_lowering

@testset "lower_to_mtk: first-order A -> B" begin
    a = SpeciesData(id=1, name="A")
    b = SpeciesData(id=2, name="B")
    rxn = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(2.0, 0.0, 0.0))
    mech = Mechanism(species=[a, b], reactions=[rxn])

    sys = lower_to_mtk(mech)
    @test length(unknowns(sys)) == 2

    # RHS at A=3, B=5: dA = -2*3 = -6, dB = +2*3 = +6.
    idx = _state_index(sys)
    u = zeros(2); u[idx["A"]] = 3.0; u[idx["B"]] = 5.0
    du = zeros(2)
    ODEFunction(sys)(du, u, _pvals(sys), 0.0)
    @test du[idx["A"]] ≈ -6.0
    @test du[idx["B"]] ≈  6.0
end

@testset "lower_to_mtk: Brusselator RHS (A=1, B=3)" begin
    # dX = 1 + X^2*Y - 4X ; dY = 3X - X^2*Y
    X = SpeciesData(id=1, name="X")
    Y = SpeciesData(id=2, name="Y")
    rxns = [
        ReactionData(reactants=Dict{Int,Float64}(),   products=Dict(1=>1.0), kinetics=ElementaryArrhenius(1.0,0,0)),  # ∅ -> X
        ReactionData(reactants=Dict(1=>2.0, 2=>1.0),  products=Dict(1=>3.0), kinetics=ElementaryArrhenius(1.0,0,0)),  # 2X+Y -> 3X
        ReactionData(reactants=Dict(1=>1.0),          products=Dict(2=>1.0), kinetics=ElementaryArrhenius(3.0,0,0)),  # X -> Y  (rate = B)
        ReactionData(reactants=Dict(1=>1.0),          products=Dict{Int,Float64}(), kinetics=ElementaryArrhenius(1.0,0,0)),  # X -> ∅
    ]
    mech = Mechanism(species=[X, Y], reactions=rxns)
    sys = lower_to_mtk(mech)

    # At X=2, Y=1: dX = 1 - 8 + 4 = -3 ; dY = 6 - 4 = 2.
    idx = _state_index(sys)
    u = zeros(2); u[idx["X"]] = 2.0; u[idx["Y"]] = 1.0
    du = zeros(2)
    ODEFunction(sys)(du, u, _pvals(sys), 0.0)
    @test du[idx["X"]] ≈ -3.0
    @test du[idx["Y"]] ≈  2.0
end

@testset "lower_to_mtk: rejects non-zero-point config (zero-point only so far)" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    mech = Mechanism(species=[a, b],
        reactions=[ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                                kinetics=ElementaryArrhenius(1.0,0,0))])
    @test_throws ErrorException lower_to_mtk(mech; config=MechanismConfig(energy=:adiabatic))
    @test lower_to_mtk(mech; config=MechanismConfig()) !== nothing   # default zero-point ok
end

@testset "lower_to_mtk: constant-rate mechanism has no T parameter" begin
    sys = lower_to_mtk(_brusselator_mech())     # all rates constant
    @test !any(p -> String(ModelingToolkit.getname(p)) == "T", ModelingToolkit.parameters(sys))
end

@testset "lower_to_mtk: T-dependent Arrhenius solves at the given T" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    # A=2, b=1, Ea=8314 J/mol  (Ea/R = 1000 K; R = 8.314 J/(mol·K))
    rxn = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(2.0, 1.0, 8314.0))
    phase = ChemPhaseSystem(Mechanism(species=[a, b], reactions=[rxn]))
    sys = extract_system(phase)
    # a T parameter was introduced
    @test any(p -> String(ModelingToolkit.getname(p)) == "T", ModelingToolkit.parameters(sys))
    Tparam = ModelingToolkit.parameters(sys)[findfirst(p -> String(ModelingToolkit.getname(p)) == "T",
                                                       ModelingToolkit.parameters(sys))]
    sol = simulate(phase, (0.0, 0.01); u0=Dict("A" => 3.0, "B" => 0.0),
                   params=[Tparam => 500.0], reltol=1e-10, abstol=1e-10)
    k500 = 2.0 * 500.0^1.0 * exp(-8314.0 / (8.314 * 500.0))   # = 1000*exp(-2) ≈ 135.335
    a_idx = findfirst(s -> String(ModelingToolkit.getname(s)) == "A", ModelingToolkit.unknowns(sys))
    @test sol.u[end][a_idx] ≈ 3 * exp(-k500 * 0.01) atol=1e-4
    @test all(isfinite, sol.u[end])
end

@testset "lower_to_mtk: species are Catalyst @species (backend-ready)" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    mech = Mechanism(species=[a, b],
        reactions=[ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                                kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))])
    sys = lower_to_mtk(mech)
    A = unknowns(sys)[findfirst(s -> String(getname(s)) == "A", unknowns(sys))]
    B = unknowns(sys)[findfirst(s -> String(getname(s)) == "B", unknowns(sys))]
    # Catalyst accepts @species (but rejects plain @variables) as Reaction substrates.
    # isequal: symbolic == returns a non-boolean Equation, so use structural isequal.
    rx = Catalyst.Reaction(2.0, [A], [B])
    @test isequal(Catalyst.oderatelaw(rx; combinatoric_ratelaw=false), 2.0 * A)
end

@testset "catalyst_native: ElementaryArrhenius is Catalyst-native" begin
    rxn = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    @test catalyst_native(rxn, MechanismConfig()) == true
end

@testset "catalyst_lowering: rate equals the direct path (shared @species)" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    rxn = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(2.0, 0.0, 0.0))
    mech = Mechanism(species=[a, b], reactions=[rxn])
    sys = lower_to_mtk(mech)
    A = unknowns(sys)[findfirst(s -> String(getname(s)) == "A", unknowns(sys))]
    B = unknowns(sys)[findfirst(s -> String(getname(s)) == "B", unknowns(sys))]
    cvar = Dict(1 => A, 2 => B)
    # Under units, k is a rate_param (default = stored A-factor); both paths
    # produce the same symbolic k·A. isequal: symbolic == returns a non-boolean Equation.
    crate = catalyst_lowering(rxn, mech, cvar, nothing, 1)
    drate = direct_mtk_lowering(rxn, mech, cvar, nothing, 1)
    @test isequal(crate, drate)
    # The k parameter carries the stored default (2.0).
    kparam = ModelingToolkit.parameters(sys)[findfirst(p -> String(ModelingToolkit.getname(p)) == "k_1_A",
                                                            ModelingToolkit.parameters(sys))]
    @test ModelingToolkit.getdefault(kparam) == 2.0
end

@testset "catalyst path: Brusselator RHS is unchanged" begin
    # After Task 3 every elementary reaction routes through catalyst_lowering;
    # the species-conservation RHS must be identical to the direct path.
    sys = lower_to_mtk(_brusselator_mech())
    idx = _state_index(sys)
    u = zeros(2); u[idx["X"]] = 2.0; u[idx["Y"]] = 1.0
    du = zeros(2)
    ODEFunction(sys)(du, u, _pvals(sys), 0.0)
    @test du[idx["X"]] ≈ -3.0                    # 1 - 4·2 + (2²)·1 = -3
    @test du[idx["Y"]] ≈  2.0                    # 3·2 - (2²)·1 = 2
end

@testset "M1: no spurious theta param when b!=0, Ea=0" begin
    # k(T) = A·T^b with b=0.5, Ea=0 → power-law, no exp term, no theta param.
    x = SpeciesData(id=1, name="X"); y = SpeciesData(id=2, name="Y")
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(1.0e6, 0.5, 0.0))
    mech = Mechanism(species=[x,y], reactions=[rxn])
    sys = extract_system(ChemPhaseSystem(mech))
    pnames = [String(getname(p)) for p in parameters(sys)]
    @test !any(occursin("theta", n) for n in pnames)   # no theta param (was spurious before)
    @test any(occursin("_A", n)   for n in pnames)      # A param present
    @test "T" in pnames                                  # T retained (T-dependent)
end

@testset ":fixedT mode — isothermal species ODE + EOS pressure observed" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(1.0,0.0,0.0),
                       reverse_policy=ExplicitReverse(ElementaryArrhenius(2.0,0.0,0.0)))
    mech = Mechanism(species=[a,b], reactions=[rxn])
    phase = ChemPhaseSystem(mech; config=convenience_config(:fixedT))
    sys = extract_system(phase)
    @test :P in [getname(o.lhs) for o in observed(sys)]                     # P observed
    Av = unknowns(sys)[findfirst(s -> String(getname(s))=="A", unknowns(sys))]
    Bv = unknowns(sys)[findfirst(s -> String(getname(s))=="B", unknowns(sys))]
    Tp  = parameters(sys)[findfirst(p -> String(getname(p))=="T", parameters(sys))]  # T retained
    Pvar = [o.lhs for o in observed(sys) if getname(o.lhs)==:P][1]
    Tv = 1000.0
    sol = simulate(phase, (0.0,5.0); u0=Dict("A"=>1.0,"B"=>0.0), params=[Tp=>Tv], reltol=1e-9, abstol=1e-12)
    csum = sol(2.0; idxs=Av) + sol(2.0; idxs=Bv)
    @test sol(2.0; idxs=Pvar) ≈ csum * 8.314 * Tv  rtol=1e-4               # P = (Σc)·R·T
    @test csum ≈ 1.0  atol=1e-6                                            # mass conserved
end
