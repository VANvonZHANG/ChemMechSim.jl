# Adjoint-based parameter fitting (inverse problem) on the toy A --k1--> B --k2--> A.
# Run:  julia --project=examples examples/demos/fit_parameters.jl
#
# Recovers rate constants from noisy "experimental" data with the PR #61 gradient
# channel (flat_params / flat_to_mtk + InterpolatingAdjoint) and Optimization.jl's
# Adam — the optimization LOOP lives in the library, the gradient is one forward +
# one adjoint solve per iteration, independent of parameter count.
#
# Workflow:
#   1. truth (k2,k1) = (0.3,0.8) generates A(t) at 12 points; add 2% relative noise
#   2. minimize a RELATIVE-residual loss over θ = log k: exp keeps k positive and
#      balances the 10-orders-of-magnitude span of real A-factors; relative residuals
#      because concentrations span orders of magnitude
#   3. Adam converges in ~100 iterations to k within ~3% of truth — the statistical
#      uncertainty of 2% noise on 12 points (the loss bottoms out at the irreducible
#      noise floor, so this is a converged fit, not a stalled one)
#
# Optimization.jl v5 API notes (verified 2026-10-11):
#   * the objective must take TWO arguments f(θ, p); single-argument f trips SciMLBase's
#     "too few arguments" check — pass p = nothing to OptimizationProblem
#   * the callback receives an OptimizationState (.iter / .objective / .u), not (θ, loss)
#   * verbose=false silences the harmless ReverseDiffVBP→ZygoteVJP fallback notice
#   * to box-bound instead: OptimizationLBFGSB with lb = [log(0.1), ...], ub = [...]
using Random
using ChemMechSim
using ModelingToolkit
using OrdinaryDiffEq, LinearSolve
using ADTypes, SciMLSensitivity, Zygote
using Optimization, OptimizationOptimisers

const SOLVER = FBDF(autodiff = ADTypes.AutoFiniteDiff(), linsolve = LUFactorization())

# ---- the toy (linear, closed-form-checkable): A --k1--> B --k2--> A ----
spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B")
mech = Mechanism(species=[spA, spB], reactions=[
    ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                 kinetics=ElementaryArrhenius(0.8, 0.0, 0.0), reverse_policy=Irreversible()),
    ReactionData(reactants=Dict(2 => 1.0), products=Dict(1 => 1.0),
                 kinetics=ElementaryArrhenius(0.3, 0.0, 0.0), reverse_policy=Irreversible()),
])
r = BatchReactor(mech; mode=:kinetic, checks=false, name=:fitdemo)
sys = extract_system(r)
prob = build_problem(r, Dict("A" => 1.0, "B" => 0.0), (0.0, 3.0); jac=true)
iA = state_index(sys, "A")
v_true = flat_params(prob)               # [k_2_A, k_1_A] = [0.3, 0.8] — parameters(sys) order

# ---- synthetic experiment: truth + 2% relative noise at 12 points ----
tsave = collect(0.25:0.25:3.0)
sol = solve(remake(prob, p=flat_to_mtk(prob, v_true)), SOLVER;
            reltol=1e-10, abstol=1e-12, saveat=tsave)
Random.seed!(20261011)
data = [u[iA] for u in sol.u] .* (1 .+ 0.02 .* randn(length(tsave)))

# ---- loss over θ = log k ----
function loss(θ, _)
    v = exp.(θ)
    s = solve(remake(prob, p=flat_to_mtk(prob, v)), SOLVER;
              reltol=1e-8, abstol=1e-10, saveat=tsave,     # saveat = data times ONLY:
              sensealg=InterpolatingAdjoint(), verbose=false)  # adjoint memory ~ saved points
    y = [u[iA] for u in s.u]
    return sum(abs2, (y .- data) ./ max.(abs.(data), 0.1))
end

θ0 = log.([0.5, 1.5])                    # deliberately wrong start
println("truth  k = ", v_true)
println("start  k = ", exp.(θ0), "  loss = ", round(loss(θ0, nothing), sigdigits=4))

optf = OptimizationFunction(loss, AutoZygote())
optprob = OptimizationProblem(optf, θ0, nothing)
res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05); maxiters = 300,
    callback = (state, kwargs...) -> begin
        state.iter % 50 == 0 && println("iter ", state.iter, ": loss = ",
            round(state.objective, sigdigits=6), "  k = ", round.(exp.(state.u), sigdigits=5))
        return false
    end)

kfit = exp.(res.u)
println("fitted k = ", round.(kfit, sigdigits=6))
println("truth  k = ", v_true)
println("deviation = ", round.((kfit .- v_true) ./ v_true .* 100, digits=2), " %",
        "  (~noise level: 2% on 12 points)")
