# Linear-solver probe for Aramco 3.0 CH4-air ignition under `jac=true`.
#
# Run:   julia --project=. examples/perf/aramco_linsolve_probe.jl [t_end_ms]
#        (default t_end_ms = 5.0 → matches examples/validation/aramco_ignition.jl)
#
# PURPOSE. The `jac=true` Aramco solve (~582 states, reaction-sharded analytic
# Jacobian → sparse jac_prototype) spends ~97% of its wall time inside the
# sparse linear solve of FBDF's Newton iteration (W = I/h − γJ). With no
# explicit `linsolve`, FBDF/LinearSolve defaults to sparse LU (KLU/PureKLU).
# This probe runs the SAME problem through 5 different linear solvers and
# reports wall time / step count / retcode so we can see whether the linear
# solver is the lever, and which one wins.
#
#   KLU            — sparse LU, the current default (the 457 s / 97 % baseline)
#   UMFPACK        — sparse LU (SuiteSparse), column-filling — often faster than
#                    KLU on denser patterns, higher memory
#   Sparspak       — pure-Julia envelope/profile sparse LU — no binary dep
#   GMRES          — Krylov, matrix-free W-operator, NO preconditioner (baseline
#                    for whether naked GMRES can survive stiff chemistry)
#   GMRES + ILU(0) — Krylov with zero-fill incomplete LU preconditioner (the
#                    recommended next try — avoids the per-step refactor that
#                    makes KLU expensive)
#
# DEPENDENCIES. KLU/UMFPACK/GMRES are already available (PureKLU + Krylov ship
# with LinearSolve, UMFPACK via SuiteSparse). Sparspak and IncompleteLU are OPTIONAL —
# `] add Sparspak IncompleteLU` to enable those two rows; the probe auto-skips whatever
# is missing. Both are lightweight pure-Julia packages.
#
# HOW TO READ THE OUTPUT.
#   • `steps` is the key diagnostic. If steps are ~equal across solvers, then
#     per-step linear-solve cost drives `time` and the fastest solver wins
#     outright (the scenario where GMRES+ILU should beat KLU).
#   • If steps differ a lot, Newton convergence differs — a better/worse
#     preconditioner changes step count. GMRES+ILU should cut steps vs naked
#     GMRES; if it doesn't, the preconditioner is too weak (raise ILU fill).
#   • If ALL solvers show ~1e5–1e6 steps, the Jacobian accuracy (not the linear
#     solver) is the real bottleneck — see memory sharded-jac notes.
#
# EXPECTED RUNTIME. Each solve is minutes; full 5-solver pass can be 30–60 min.
# Pass a short t_end (e.g. `... 1.0`) for a faster smoke run. Problem build
# (lowering + sharded Jacobian) happens ONCE and is reused for every solver.
using ChemMechSim
using OrdinaryDiffEq: FBDF
using SciMLBase: solve                          # not re-exported by ChemMechSim
using LinearSolve: KLUFactorization, UMFPACKFactorization, SparspakFactorization,
                   KrylovJL_GMRES
using ModelingToolkit: unknowns, getname
using SparseArrays: nnz
using Printf: @printf

# Optional solvers — gated so the probe still runs if these aren't installed.
const HAS_SPARSPAK = Ref(false)
try
    @eval using Sparspak
    HAS_SPARSPAK[] = true
catch
    println("(Sparspak not loaded — `] add Sparspak` to enable that row)")
end
const HAS_ILU = Ref(false)
try
    @eval using IncompleteLU
    HAS_ILU[] = true
catch
    println("(IncompleteLU not loaded — `] add IncompleteLU` to enable the GMRES+ILU row)")
end

# ---- ILU(0) preconditioner callback (LinearSolve/OrdinaryDiffEq `precs` API) ----
# Signature: Pl, Pr = precs(W, du, u, p, t, newW, Plprev, Prprev, solverdata)
# Recompute the ILU only when W changed (newW === true); reuse Plprev otherwise.
# The newW === nothing dispatch is the integrator setup phase → return nothing.
# τ=0.0 → zero-fill (ILU(0), most stable, most precond work); raise toward 1e-2
# to drop more entries (cheaper precond, weaker — tune if GMRES step count is high).
function precs_ilu0(W, du, u, p, t, newW, Plprev, Prprev, solverdata)
    # Canonical SciML pattern: fold setup (newW===nothing) into the compute branch so the
    # integrator's precond cache types consistently as the ILU object (a separate setup
    # branch returning (nothing,nothing) caused a setfield! TypeError on InvPreconditioner).
    # Needs `concrete_jac=true` on the alg + convert(AbstractMatrix, W) (not SparseMatrixCSC,
    # which has no WOperator method). (Diagnosed 2026-08-01.)
    if newW === nothing || newW
        return IncompleteLU.ilu(convert(AbstractMatrix, W); τ = 0.0), nothing
    else
        return Plprev, nothing                        # reuse cached
    end
end

# ---- mechanism + initial condition (identical to aramco_ignition.jl) ----
const YAML_PATH = joinpath(@__DIR__, "..", "mechanism", "AramcoMech3.0.yaml")
const R, T0, P0 = 8.314, 1500.0, 101325.0
const c_tot = P0 / (R * T0)
const X0 = Dict("CH4" => 1.0 / 10.52, "O2" => 2.0 / 10.52, "N2" => 7.52 / 10.52)

t_end_s = parse(Float64, get(ARGS, 1, "5.0")) * 1e-3     # ms → s
TSPAN = (0.0, t_end_s)

mech = load_mechanism(YAML_PATH)
println("Loaded Aramco: $(length(mech.species)) species, $(length(mech.reactions)) reactions")
reactor = BatchReactor(mech; mode=:adiabatic_constV, checks=false)
sys = extract_system(reactor)
T_idx = findfirst(s -> String(getname(s)) == "T", unknowns(sys))
u0 = Dict(sp.name => get(X0, sp.name, 0.0) * c_tot for sp in mech.species)
u0["T"] = T0

# ---- build the problem ONCE with jac=true (reaction-sharded analytic sparse Jac) ----
println("\nBuilding ODEProblem (jac=true, :reaction_sharded) …")
# NOTE: build_problem signature is (phase, u0, tspan) — u0 positional BEFORE tspan
# (unlike simulate, which takes (phase, tspan; u0=...)).
t_build = @elapsed prob = build_problem(reactor, u0, TSPAN; jac=true,
                                        jac_strategy=:reaction_sharded)
J_proto = isdefined(prob.f, :jac_prototype) ? prob.f.jac_prototype : nothing
println("Built in $(round(t_build, digits=1)) s. " *
        "States=$(length(prob.u0)), nnz(jac)=$(J_proto === nothing ? "?" : nnz(J_proto))")
println("tspan = $TSPAN   reltol=1e-8  abstol=1e-12\n")

# ---- solver configurations ----
# Each entry: name => the FBDF algorithm with its linsolve/precs baked in.
configs = Pair{String,Any}[
    "KLU (default)"  => FBDF(linsolve=KLUFactorization()),
    "UMFPACK"        => FBDF(linsolve=UMFPACKFactorization()),
    "GMRES (bare)"   => FBDF(linsolve=KrylovJL_GMRES()),
]
HAS_SPARSPAK[] && push!(configs,
    "Sparspak"       => FBDF(linsolve=SparspakFactorization()))
HAS_ILU[] && push!(configs,
    # concrete_jac=true is REQUIRED for ILU: under a Krylov linsolve FBDF defaults to
    # matrix-free (concrete_jac=false), so W is a lazy operator that no convert() can
    # materialize for ilu. Forcing concrete_jac=true makes FBDF build a real sparse W
    # each step, which both gives ilu a matrix and lets GMRES matvec on the concrete W.
    # STATUS 2026-08-02: this config WORKS (gets past all earlier convert/cache crashes)
    # but is SLOW — IncompleteLU.jl's pure-Julia Crout ILU is refactored every newW=true
    # step, which costs more than UMFPACK's 12s direct solve. Not competitive; left in
    # for completeness. Run solo via `...jl 5.0 "GMRES + ILU"` (don't block the full pass).
    "GMRES + ILU(0)" => FBDF(linsolve=KrylovJL_GMRES(), concrete_jac=true, precs=precs_ilu0))
if !HAS_SPARSPAK[]
    println("[skip] Sparspak        — run `] add Sparspak`")
end
if !HAS_ILU[]
    println("[skip] GMRES + ILU(0)  — run `] add IncompleteLU`")
end

# Optional: filter to one solver via ARGS[2] (substring of a config name), e.g.
#   julia ...jl 5.0 "GMRES + ILU"     → build once, run only the matching solver.
if length(ARGS) >= 2
    filt = ARGS[2]
    configs = Pair{String,Any}[c for c in configs if occursin(filt, first(c))]
    isempty(configs) && error("no solver name matches '$filt'")
    println("(running only configs matching \"$filt\")")
end

# ---- run + measure ----
results = []   # Vector of NamedTuples (name, retcode, time, steps, T_end, note)
for (name, alg) in configs
    print("running $name …")
    flush(stdout)
    GC.gc()
    local sol, t
    try
        t = @elapsed sol = solve(prob, alg; reltol=1e-8, abstol=1e-12)
    catch e
        msg = first(split(sprint(showerror, e), '\n'))
        println(" CRASH")
        push!(results, (name=name, retcode="CRASH", time=NaN, steps=0,
                        T_end=NaN, note=msg[1:min(end, 60)]))
        continue
    end
    T_end = (T_idx === nothing || isempty(sol.u)) ? NaN : Float64(sol.u[end][T_idx])
    println(" retcode=$(sol.retcode)  time=$(round(t, digits=1))s  steps=$(length(sol))" *
            (isfinite(T_end) ? "  T_end=$(round(T_end, digits=1))K" : ""))
    push!(results, (name=name, retcode=string(sol.retcode), time=t,
                    steps=length(sol), T_end=T_end, note=""))
end

# ---- results table ----
println("\n" * "="^86)
@printf("%-18s %-10s %9s %9s %10s  %s\n", "solver", "retcode", "time(s)", "steps", "T_end(K)", "note")
println("-"^86)
for r in results
    @printf("%-18s %-10s %9s %9d %10s  %s\n", r.name, r.retcode,
            isnan(r.time) ? "—" : round(r.time, digits=1), r.steps,
            isnan(r.T_end) ? "—" : round(r.T_end, digits=1), r.note)
end
println("="^86)

println("""
Interpretation:
  • steps ~equal across solvers → per-step linear-solve cost drives time (fastest wins).
  • steps differ a lot       → Newton convergence differs (preconditioner quality).
  • ALL solvers ~1e5–1e6 steps → Jacobian accuracy, not the linear solver, is the bottleneck.
""")
