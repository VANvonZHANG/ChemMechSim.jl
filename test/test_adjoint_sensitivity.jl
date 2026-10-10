using Test
using ChemMechSim
using ChemMechSim: SpeciesData, ReactionData, ElementaryArrhenius, Irreversible
using ModelingToolkit
using ModelingToolkit: getname, parameters
using OrdinaryDiffEq, LinearSolve, ADTypes, SciMLSensitivity, Zygote, ForwardDiff

const SOLVER = FBDF(autodiff = ADTypes.AutoFiniteDiff(), linsolve = LUFactorization())

function ab_toy()
    spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B")
    rx1 = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(0.8, 0.0, 0.0), reverse_policy=Irreversible())
    rx2 = ReactionData(reactants=Dict(2 => 1.0), products=Dict(1 => 1.0),
                       kinetics=ElementaryArrhenius(0.3, 0.0, 0.0), reverse_policy=Irreversible())
    return Mechanism(species=[spA, spB], reactions=[rx1, rx2])
end

# Closed form (state order [B, A]; N = total mass). GOLD is computed independently with
# ForwardDiff over these expressions — a different channel than the adjoint under test.
# RULE: helpers as multi-line `function` blocks, never compact one-liners with chained
# semicolons (the 2026-10-09 probe scripts ParseError'd on those repeatedly).
function A_of(t, k1, k2; A0 = 1.0, N = 1.0)
    s = k1 + k2
    ss = k2 / s * N
    return ss + (A0 - ss) * exp(-s * t)
end
function A_of_u0(t, u0)                     # u0 in state order [B0, A0]; k = (0.8, 0.3)
    N = sum(u0)
    return A_of(t, 0.8, 0.3; A0 = u0[2], N = N)
end

const TF = 3.0
gold_u0 = ForwardDiff.gradient(u0 -> A_of_u0(TF, u0), [0.0, 1.0])
gold_p  = ForwardDiff.gradient(v -> A_of(TF, v[2], v[1]), [0.3, 0.8])  # v = [k2, k1]

mech = ab_toy()
r = BatchReactor(mech; mode = :kinetic, checks = false, name = :adjgold)
sys = extract_system(r)
iA = ChemMechSim.state_index(sys, "A")                                # == 2
probs = (("fd(MTK)", build_problem(r, Dict("A"=>1.0, "B"=>0.0), (0.0, TF); jac = false)),
         ("sharded", build_problem(r, Dict("A"=>1.0, "B"=>0.0), (0.0, TF); jac = true)))

@testset "adjoint golden: u₀ gradient" begin
    for (name, prob) in probs
        G(u0v) = (sol = solve(remake(prob, u0 = u0v), SOLVER;
                              reltol = 1e-10, abstol = 1e-12, saveat = [TF],
                              sensealg = InterpolatingAdjoint());
                  sol.u[end][iA])
        g = Zygote.gradient(G, [0.0, 1.0])[1]
        @test isapprox(g, ForwardDiff.gradient(u0 -> A_of_u0(TF, u0), [0.0, 1.0]); rtol = 1e-5)
        # one pinned literal from the 2026-10-08 probe (fd path, 4 digits):
        name == "fd(MTK)" && (@test g[1] ≈ 0.2626 atol = 1e-3; @test g[2] ≈ 0.2997 atol = 1e-3)
    end
end

@testset "adjoint golden: parameter gradient via flat channel" begin
    # gold in FLAT order [k2, k1]; pinned literal from probe round 3 (fd path, 8 digits):
    for (name, prob) in probs
        v0 = ChemMechSim.flat_params(prob)
        G(v) = (sol = solve(remake(prob, p = ChemMechSim.flat_to_mtk(prob, v)), SOLVER;
                            reltol = 1e-10, abstol = 1e-12, saveat = [TF],
                            sensealg = InterpolatingAdjoint());
                sol.u[end][iA])
        g = Zygote.gradient(G, v0)[1]
        @test isapprox(g, gold_p; rtol = 1e-5)
        name == "fd(MTK)" && (@test g[1] ≈ 0.556299094 atol = 1e-6;
                              @test g[2] ≈ -0.319261663 atol = 1e-6)
    end
end
