# MCM adjoint sensitivity ranking: which reactions drive O3 under perpetual noon?
# Run:  julia --project=examples examples/atmospheric/mcm_sensitivity.jl [hours]
#
# ONE adjoint solve yields d O3(TF) / d ALL ~5600 rate parameters simultaneously —
# the few-outputs × many-parameters regime where reverse mode costs ~2-3 solve
# equivalents regardless of parameter count (finite differences: #params+1 solves).
# Gradient channel: PR #61's flat_params / flat_to_mtk + remake + InterpolatingAdjoint.
#
# FROZEN photolysis (parameter defaults = J at overhead sun, no runtime callbacks):
# the solve is AD-transparent. Differentiating through the diurnal setp! callback
# (affect! mutating parameters) is a separate, unverified problem — not attempted here.
#
# Ranking by |d ln O3 / d ln k| — a dimensionless elasticity (dO3/O3 per dk/k).
# Raw dO3/dk compares incomparable units across rate constants spanning 15 orders
# of magnitude; the elasticity is the standard mechanism-analysis quantity.
#
# Costs, CPU seconds on a shared box (1843 sp / 5600 rxn, 6 h, FBDF + analytic sharded
# Jacobian, measured 2026-10-11/12): parse ~30, lower ~110, build ~390, forward ~82.
# The FIRST gradient is a one-time multi-CPU-hour compile + reverse pass (~11 GiB
# resident) — GRI30 dress rehearsal (53 sp / 325 rxn): first gradient 1913 CPU s,
# FD-verified to 0.06-0.26%; warm passes are far cheaper (the toy fit demo runs 300
# forward+adjoint iterations in ~2 min). Run the first MCM gradient as a batch job.
using ChemMechSim
using ModelingToolkit
using ModelingToolkit: getname, parameters
using OrdinaryDiffEq
using LinearSolve
using SciMLSensitivity
using ReverseDiff
using Zygote
using Printf

include(joinpath(@__DIR__, "tools", "mcm_rate_types.jl"))

const HOURS = isempty(ARGS) ? 6.0 : parse(Float64, ARGS[1])
const TF = HOURS * 3600.0
const T0 = 298.0; const P0 = 102858.0; const R_GAS = 8.314
const X_INIT = Dict("N2" => 0.78, "O2" => 0.21, "H2O" => 0.01,
                    "O3" => 3.0e-8, "NO2" => 1.0e-10, "CH4" => 1.8e-6)

const SRC = joinpath(@__DIR__, "mcm_alkanes_alkenes_converted.yaml")
isfile(SRC) || error("mcm_sensitivity: source mechanism not found at\n  $SRC\n" *
                     "It is not committed (24 MB). Copy it from the converter:\n" *
                     "  cp <kpp-cantera-converter>/examples/mcm/mcm_alkanes_alkenes_converted.yaml \\\n" *
                     "     examples/atmospheric/\nSee examples/atmospheric/README.md.")

# --- load + lower + build (same shape invariants as mcm_box.jl) --------------------------
t = @elapsed mech = load_mechanism(SRC; rate_type_handlers = mcm_rate_handlers())
@printf("load   : %.0f s   (%d species, %d reactions)\n", t, length(mech.species), length(mech.reactions))
t = @elapsed r = BatchReactor(mech; mode = :kinetic, checks = false, name = :mcmsens)
@printf("lower  : %.0f s   (peak %.2f GiB; checks=false is REQUIRED for MCM)\n", t, Sys.maxrss() / 2^30)
sys = extract_system(r); ps = parameters(sys)
iO3 = state_index(sys, "O3")
C_AIR = P0 / (R_GAS * T0)
u0 = Dict(String(sp.name) => get(X_INIT, String(sp.name), 0.0) * C_AIR for sp in mech.species)
Tparam = ps[findfirst(p -> String(getname(p)) == "T", ps)]
t = @elapsed prob = build_problem(r, u0, (0.0, TF); params = [Tparam => T0], jac = true)
@printf("build  : %.0f s   (reaction-sharded analytic Jacobian)\n", t)

v0 = flat_params(prob)                     # [k_j_A..., T, ...] in parameters(sys) order
pnames = String.(getname.(ps))
println("params : ", length(v0))

const SOLVER = FBDF(autodiff = false, linsolve = LUFactorization())
# ReverseDiffVJP: a runtime TAPE, not per-operator reverse codegen — avoids the
# pathological Zygote compile over a 5600-reaction generated RHS.
const SENSE = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false))

function G(v)
    s = solve(remake(prob, p = flat_to_mtk(prob, v)), SOLVER;
              reltol = 1e-6, abstol = 1e-12, saveat = [TF],      # adjoint memory ~ saved points
              sensealg = SENSE, verbose = false)
    return s.u[end][iO3]
end
G0(v) = solve(remake(prob, p = flat_to_mtk(prob, v)), SOLVER;
              reltol = 1e-6, abstol = 1e-12, saveat = [TF]).u[end][iO3]

println("forward: O3(", HOURS, " h) = ", G0(v0), " mol/m^3")

t = @elapsed g = Zygote.gradient(G, v0)[1]
@printf("adjoint: %.0f s   (one gradient w.r.t. all %d params; peak %.2f GiB)\n",
        t, length(v0), Sys.maxrss() / 2^30)

# --- FD cross-check: the top-|g| param (strong signal) + one photolysis param -------------
# The trust anchor for any adjoint result: a central difference on the plain forward.
top_i = argmax(abs.(g))
for i in (top_i, findfirst(==(Symbol("k_1_A")), Symbol.(pnames)))
    i === nothing && continue
    eps = 1e-4 * v0[i]
    vp = copy(v0); vp[i] += eps
    vm = copy(v0); vm[i] -= eps
    fd = (G0(vp) - G0(vm)) / (2eps)
    @printf("check  : %s  adjoint=%.4e  fd=%.4e  rel.err=%.1e\n",
            pnames[i], g[i], fd, abs(g[i] - fd) / max(abs(fd), abs(g[i]), 1e-30))
end

# --- ranking by elasticity |d ln O3 / d ln k| = |k * dO3/dk| -----------------------------
const ID2N = Dict(sp.id => String(sp.name) for sp in mech.species)
terms(d) = join((c == 1.0 ? ID2N[s] : "$(c) $(ID2N[s])" for (s, c) in sort(collect(d), by = first)), " + ")
rows = (name = String[], k = Float64[], elast = Float64[], idx = Int[], eq = String[])
for (i, name) in enumerate(pnames)
    m = match(r"^k_(\d+)_A$", name)
    m === nothing && continue                    # T / k_j_B... params: not rate A-factors
    j = parse(Int, m.captures[1])
    kin = mech.reactions[j].kinetics
    kin isa ZenithPhotolysis && continue         # photolysis rates are scenario inputs,
    push!(rows.name, name)                       # not chemistry to rank (see README)
    push!(rows.k, v0[i])
    push!(rows.elast, abs(v0[i] * g[i]))
    push!(rows.idx, i)
    push!(rows.eq, terms(mech.reactions[j].reactants) * " => " * terms(mech.reactions[j].products))
end
perm = sortperm(rows.elast; rev = true)

@printf("\nTop-20 reactions by |d ln O3 / d ln k| after %.0f h of perpetual noon:\n\n", HOURS)
println("rank  param          elasticity   reaction")
for (rank, idx) in enumerate(perm[1:min(20, end)])
    @printf("%3d   %-12s %10.3e   %s\n", rank, rows.name[idx], rows.elast[idx], rows.eq[idx])
end

# --- first-order uncertainty propagation (assumed factor-2 on every thermal k) ------------
sigma_ln = log(2.0)
sig = sqrt(sum((rows.k .* g[rows.idx] .* sigma_ln) .^ 2))
@printf("\nfirst-order sigma(ln O3) with factor-2 uncertainty on every thermal k: %.3f\n", sig)

OUTDIR = joinpath(@__DIR__, "output", "sensitivity")
mkpath(OUTDIR)
open(joinpath(OUTDIR, @sprintf("frozen_%gh.csv", HOURS)), "w") do io
    println(io, "rank,param,k,abs_dlnO3_dlnk,reaction")
    for (rank, idx) in enumerate(perm)
        println(io, rank, ",", rows.name[idx], ",", rows.k[idx], ",", rows.elast[idx],
                ",\"", rows.eq[idx], "\"")
    end
end
println("wrote ", joinpath(OUTDIR, @sprintf("frozen_%gh.csv", HOURS)))
