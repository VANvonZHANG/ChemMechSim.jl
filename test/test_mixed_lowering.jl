using Test
using ChemMechSim
using Catalyst
using ModelingToolkit
using ModelingToolkit: unknowns, getname, equations, parameters
using OrdinaryDiffEq
# catalyst_lowering / direct_mtk_lowering are the two lowering paths under test;
# they are not in the public export list, so import them explicitly.
import ChemMechSim: catalyst_lowering, direct_mtk_lowering

@testset "§3.4 #3: catalyst + direct rates share @species, unify into one ODESystem" begin
    # Three species, shared via lower_to_mtk's @species declaration.
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B"); m = SpeciesData(id=3, name="M")
    rxA = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    rxB = ReactionData(reactants=Dict(1 => 1.0, 3 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(0.5, 0.0, 0.0))
    mech = Mechanism(species=[a, b, m], reactions=[rxA, rxB])
    sys = lower_to_mtk(mech)        # rxA and rxB are both ElementaryArrhenius → both via catalyst_lowering
    A, B, Mv = _var(sys, "A"), _var(sys, "B"), _var(sys, "M")
    cvar = Dict(1 => A, 2 => B, 3 => Mv)

    # Prove the two lowering PATH FUNCTIONS agree on shared @species and unify into
    # one ODESystem: rxA's rate via the Catalyst path, rxB's via the direct path.
    # Under units (§5.6) k is a unit-bearing rate_param (default = stored A-factor).
    rate_cat = catalyst_lowering(rxA, mech, cvar, nothing, 1)    # k_1_A * A    (Catalyst path)
    rate_dir = direct_mtk_lowering(rxB, mech, cvar, nothing, 2)  # k_2_A * A*M  (direct path)
    t = ModelingToolkit.t; D = ModelingToolkit.D
    eqs = [D(A) ~ -rate_cat - rate_dir,
           D(B) ~  rate_cat + rate_dir,
           D(Mv) ~ 0.0]
    @named osys = System(eqs, t)
    simp = mtkcompile(osys)

    @test length(equations(simp)) == 3                       # one unified ODESystem
    # Structural check: each path's rate is the mass-action form k_param · ∏species
    # (§3.4 #3). The k-params created by the standalone lowering calls are symbolically
    # equal to the same-named params in `sys`, so isequal verifies the symbolic structure.
    k1A = _param(sys, "k_1_A")          # reaction 1 (catalyst path, rxA: A -> B)
    k2A = _param(sys, "k_2_A")          # reaction 2 (direct path,   rxB: A + M -> B)
    @test isequal(rate_cat, k1A * A)         # catalyst path output: k·A
    @test isequal(rate_dir, k2A * A * Mv)    # direct path output:   k·A·M
    # The k parameters carry their stored defaults (1.0 and 0.5).
    @test ModelingToolkit.getdefault(k1A) == 1.0
    @test ModelingToolkit.getdefault(k2A) == 0.5
    sol = solve(ODEProblem(simp, [A => 1.0, B => 0.0, Mv => 2.0,
                                  k1A => 1.0, k2A => 0.5], (0.0, 1.0)), Tsit5())
    @test all(isfinite, sol.u[end])                          # and it solves
    @test sol(1.0; idxs=B) > 0.0
end

@testset "§3.4 #2: T-dependent catalyst rate equals the direct path" begin
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
    rxn = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(2.0, 1.0, 8314.0))   # T-dependent
    mech = Mechanism(species=[a, b], reactions=[rxn])
    sys = lower_to_mtk(mech)
    A, B = _var(sys, "A"), _var(sys, "B")
    T = parameters(sys)[findfirst(p -> String(getname(p)) == "T", parameters(sys))]
    cvar = Dict(1 => A, 2 => B)
    # Both paths build the same T-dependent symbolic rate k(T)·A on the shared @species.
    # Under units, k = A_param · T^b · exp(-θ/T) where θ = Ea/R (dimensionless exponent).
    rc = catalyst_lowering(rxn, mech, cvar, T, 1)
    rd = direct_mtk_lowering(rxn, mech, cvar, T, 1)
    @test isequal(rc, rd)
    # Structural check: the T-dependent rate is k_A · T^b · exp(-θ/T) · A (§3.4 #2).
    # Under units, k = A_param · T^b · exp(-θ/T) where θ = Ea/R (dimensionless exponent).
    kA  = _param(sys, "k_1_A")
    kθ  = _param(sys, "k_1_theta")
    @test isequal(rc, kA * T^1.0 * exp(-kθ / T) * A)
    # The A-factor and θ parameters carry their stored defaults.
    @test ModelingToolkit.getdefault(kA) == 2.0
    @test ModelingToolkit.getdefault(kθ) ≈ 8314.0 / 8.314
end

@testset "§3.4 #3 (flow): a mixed mechanism lowers & solves end-to-end" begin
    # A normal lower_to_mtk run (all elementary → catalyst path) on a 3-species network.
    a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B"); c = SpeciesData(id=3, name="C")
    mech = Mechanism(species=[a, b, c], reactions=[
        ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0), kinetics=ElementaryArrhenius(1.0, 0, 0)),
        ReactionData(reactants=Dict(2 => 2.0), products=Dict(3 => 1.0), kinetics=ElementaryArrhenius(0.5, 0, 0)),
    ])
    r = BatchReactor(mech)
    sol = simulate(r, (0.0, 5.0); u0=Dict("A" => 1.0, "B" => 0.0, "C" => 0.0), reltol=1e-9, abstol=1e-9)
    @test all(isfinite, sol.u[end])
    @test length(equations(extract_system(r))) == 3
end
