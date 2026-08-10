#!/usr/bin/env julia
# bench_matrix.jl — mechanism × linear-solver benchmark matrix (paper-grade perf data).
#
# For each mechanism × each linear solver × N repeats: end-to-end FBDF(jac=true) solve,
# PLUS a standalone linear-solve micro-benchmark (isolates W\b per-call cost from the
# ODE), PLUS trajectory accuracy vs a reference solver. Streams tidy CSVs + a
# reproducibility metadata YAML for plotting.
#
# Run (framework only — run on the target machine for paper numbers):
#   julia --project=. examples/perf/bench_matrix.jl \
#       [--mechs gri30,ffcm2,aramco] [--solvers klu,umfpack,sparspak,mumps,pardiso] \
#       [--repeats 1] [--no-warmup] [--reltol 1e-8] [--abstol 1e-12] [--tspan-ms 5.0] \
#       [--no-microbench] [--no-accuracy] [--quick] [--out-dir DIR]
#
# Outputs (examples/perf/output/ unless --out-dir):
#   bench_matrix.csv          end-to-end: one row per (mech, linsolve, run)
#   bench_linsolve_micro.csv  per-call linsolve cost (factorize+solve on the sampled J)
#   bench_accuracy.csv        trajectory vs reference (umfpack): Δt_ign, max ΔT
#   bench_meta.yaml           machine + package versions + config (reproducibility)
#
# --quick = repeats=1, no warmup (minimal solve count; still full tspan unless --tspan-ms).
# Default repeats is 1 (smoke / first pass); bump to 5–10 for the paper run.

using ChemMechSim
using OrdinaryDiffEq: FBDF
using SciMLBase: solve
using LinearSolve: KLUFactorization, UMFPACKFactorization, SparspakFactorization,
                   MUMPSFactorization, PardisoJL, LinearProblem
using ModelingToolkit: unknowns, getname
using SparseArrays: nnz
using LinearAlgebra: BLAS, I
using Statistics: median
using Printf: @printf
using YAML
using Dates
using Pkg
using Random

# ---- optional direct-sparse backends (auto-skip whatever isn't installed) ----
const HAS_SPARSPAK = Ref(false)
try
    @eval using Sparspak
    HAS_SPARSPAK[] = true
catch
    println("[gate] Sparspak not loaded — `] add Sparspak` to enable that row")
end
const HAS_MUMPS = Ref(false)
try
    @eval using MUMPS
    @eval MUMPS.MPI.Initialized() || MUMPS.MPI.Init()   # MUMPS.jl requires MPI be initialized before any factorize; it never auto-inits (mumps_struc.jl:158 throws otherwise), which silently made every MUMPS solve return zero → FBDF Unstable.
    HAS_MUMPS[] = true
catch e
    println("[gate] MUMPS not loaded — `] add MUMPS` to enable that row ($(first(split(sprint(showerror,e),'\n'))))")
end
const HAS_PARDISO = Ref(false)
try
    @eval using Pardiso
    HAS_PARDISO[] = true
catch
    println("[gate] Pardiso not loaded — `] add Pardiso` (or MKL) to enable that row")
end

# ---- mechanism configs (each its own natural ignition problem; @__DIR__-relative) ----
const R_GAS = 8.314
const MICRO_ALPHA = 1.0e6   # linsolve micro-bench: W = αI − J (representative BDF 1/(hγ); factorize cost is pattern-driven so α is non-critical)
struct MechConfig
    name::String; yaml::String; T0::Float64; P0::Float64
    X0::Dict{String,Float64}; tspan::Tuple{Float64,Float64}
end
const MECH_CONFIGS = Dict{String,MechConfig}(
    "gri30"  => MechConfig("gri30",  joinpath(@__DIR__, "..", "mechanism", "gri30.yaml"), 1500.0, 101325.0,
                           Dict("CH4" => 1.0/10.52, "O2" => 2.0/10.52, "N2" => 7.52/10.52), (0.0, 5.0e-3)),
    "ffcm2"  => MechConfig("ffcm2",  joinpath(@__DIR__, "..", "mechanism", "FFCM2.yaml"), 1500.0, 101325.0,
                           Dict("CH4" => 1.0/10.52, "O2" => 2.0/10.52, "N2" => 7.52/10.52), (0.0, 5.0e-3)),
    "aramco" => MechConfig("aramco", joinpath(@__DIR__, "..", "mechanism", "AramcoMech3.0.yaml"), 1500.0, 101325.0,
                           Dict("CH4" => 1.0/10.52, "O2" => 2.0/10.52, "N2" => 7.52/10.52), (0.0, 5.0e-3)),
    "h2o2"   => MechConfig("h2o2",   joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"), 1000.0, 101325.0,
                           Dict("H2" => 2.0/7, "O2" => 1.0/7, "N2" => 4.0/7), (0.0, 1.0e-3)),
)

# ---- solver configs: (name, FBDF alg, standalone-linsolve alg | nothing) ----
struct SolverConfig
    name::String; fbdf; ls    # ls=nothing → micro-bench skipped (precs not isolated standalone)
end
function solver_configs(selected)
    sel = lowercase.(selected)
    avail = SolverConfig[
        SolverConfig("klu",     FBDF(linsolve=KLUFactorization()),     KLUFactorization()),
        SolverConfig("umfpack", FBDF(linsolve=UMFPACKFactorization()), UMFPACKFactorization()),
    ]
    HAS_SPARSPAK[] && push!(avail, SolverConfig("sparspak", FBDF(linsolve=SparspakFactorization()), SparspakFactorization()))
    HAS_MUMPS[]    && push!(avail, SolverConfig("mumps",    FBDF(linsolve=MUMPSFactorization()),    MUMPSFactorization()))
    HAS_PARDISO[]  && push!(avail, SolverConfig("pardiso",  FBDF(linsolve=PardisoJL()),             PardisoJL()))
    out = [c for c in avail if c.name in sel]
    avail_names = getfield.(avail, :name)
    isempty(out) && error("no solvers selected (got $sel); available: $(join(avail_names, ','))")
    missing = setdiff(sel, avail_names)
    isempty(missing) || println("[gate] requested but unavailable (dep missing): $(join(missing, ','))")
    return out
end

# ---- CLI ----
function parse_cli(args::Vector{String})
    cfg = (mechs="gri30,ffcm2,aramco", solvers="klu,umfpack,sparspak,mumps,pardiso",
           repeats=1, warmup=true, reltol=1e-8, abstol=1e-12, tspan_ms=nothing,
           microbench=true, accuracy=true, out_dir=joinpath(@__DIR__, "output"))
    i = 1
    while i ≤ length(args)
        a = args[i]
        if     a == "--quick";         cfg = merge(cfg, (repeats=1, warmup=false))
        elseif a == "--no-warmup";     cfg = merge(cfg, (warmup=false,))
        elseif a == "--no-microbench"; cfg = merge(cfg, (microbench=false,))
        elseif a == "--no-accuracy";   cfg = merge(cfg, (accuracy=false,))
        elseif startswith(a, "--")
            i+1 ≤ length(args) || error("missing value for $a")
            v = args[i+1]; i += 1
            if     a == "--mechs";     cfg = merge(cfg, (mechs=v,))
            elseif a == "--solvers";  cfg = merge(cfg, (solvers=v,))
            elseif a == "--repeats";  cfg = merge(cfg, (repeats=parse(Int, v),))
            elseif a == "--reltol";   cfg = merge(cfg, (reltol=parse(Float64, v),))
            elseif a == "--abstol";   cfg = merge(cfg, (abstol=parse(Float64, v),))
            elseif a == "--tspan-ms"; cfg = merge(cfg, (tspan_ms=parse(Float64, v),))
            elseif a == "--out-dir";  cfg = merge(cfg, (out_dir=v,))
            else error("unknown arg $a")
            end
        else
            error("unexpected positional arg $a")
        end
        i += 1
    end
    cfg.repeats ≥ 1 || error("--repeats must be ≥ 1")
    return cfg
end

# ---- build one mechanism's problem (jac=true, reaction-sharded) + sample the Jacobian ----
function build_for_mech(mc::MechConfig, tspan::Tuple{Float64,Float64})
    mech = load_mechanism(mc.yaml)
    c_tot = mc.P0 / (R_GAS * mc.T0)
    u0 = Dict(sp.name => get(mc.X0, sp.name, 0.0) * c_tot for sp in mech.species)
    u0["T"] = mc.T0
    reactor = BatchReactor(mech; mode=:adiabatic_constV, checks=false)   # checks=false: K_c unit-fold
    sys = extract_system(reactor)
    T_idx = findfirst(s -> String(getname(s)) == "T", unknowns(sys))
    t_build = @elapsed prob = build_problem(reactor, u0, tspan; jac=true, jac_strategy=:reaction_sharded)
    J_proto = isdefined(prob.f, :jac_prototype) ? prob.f.jac_prototype : nothing
    n_states = length(prob.u0)
    nz = J_proto === nothing ? 0 : nnz(J_proto)
    # sample J at u0, then form the representative BDF W = αI − J for the standalone micro-bench.
    # Raw J is singular/ill-conditioned (solving J\b "fails"); W is well-posed. Factorize cost —
    # the KLU-vs-UMFPACK lever — is pattern-driven, so α is non-critical.
    W_sample = nothing
    if J_proto !== nothing && isdefined(prob.f, :jac) && prob.f.jac !== nothing
        try
            J = copy(J_proto)
            prob.f.jac(J, prob.u0, prob.p, tspan[1])
            W_sample = MICRO_ALPHA * I - J
        catch e
            println("  [microbench] $(mc.name): jac! sampling failed — $(first(split(sprint(showerror, e), '\n')))")
            W_sample = nothing
        end
    end
    return (mech=mech, prob=prob, T_idx=T_idx, t_build=t_build, n_states=n_states, nnz=nz, W_sample=W_sample)
end

# ---- one-time compile probe: the FIRST solve in a fresh process for this mech (designated ----
# solver). Dominated by Julia compilation of the reaction-sharded Jacobian codegen (scales with
# mechanism size; ~570 s for Aramco). Paid once per process, ~solver-independent — NOT a per-
# solver metric. This is the single-shot user experience; the per-solver comparison is the warm
# `run_endtoend` below (the shared jac compile is already done by this probe).
function compile_probe(prob, alg, reltol, abstol, T_idx)
    GC.gc()
    try
        sol, t, alloc, _gc, _mem = @timed solve(prob, alg; reltol=reltol, abstol=abstol)
        @printf(" compile+1st solve: %.1fs  %d steps  %s\n", t, length(sol), sol.retcode)
        return (first_solve_s=t, steps=length(sol), retcode=string(sol.retcode))
    catch e
        println(" compile probe CRASH: $(first(split(sprint(showerror, e), '\n')))")
        return (first_solve_s=NaN, steps=0, retcode="COMPILE_CRASH")
    end
end

# ---- end-to-end solve (warmup absorbs each solver's own linsolve compile; shared jac compile ----
# is done by the compile probe). Returns warm rows + the last successful solution.
function run_endtoend(prob, alg, repeats::Int, warmup::Bool, reltol, abstol, T_idx)
    if warmup
        try
            solve(prob, alg; reltol=reltol, abstol=abstol)   # discard (this solver's linsolve compile)
        catch e
            println(" warmup CRASH: $(first(split(sprint(showerror, e), '\n')))")
            return (rows=[(run_idx=1, wall_s=NaN, alloc_bytes=0, steps=0, retcode="WARMUP_CRASH", T_end=NaN)], last_sol=nothing)
        end
    end
    rows = []; last_sol = nothing
    for r in 1:repeats
        GC.gc()
        try
            sol, t, alloc, _gc, _mem = @timed solve(prob, alg; reltol=reltol, abstol=abstol)
            T_end = (T_idx === nothing || isempty(sol.u)) ? NaN : Float64(sol.u[end][T_idx])
            push!(rows, (run_idx=r, wall_s=t, alloc_bytes=alloc, steps=length(sol), retcode=string(sol.retcode), T_end=T_end))
            last_sol = sol
            @printf(" warm %d: %.2fs  %d steps  %s\n", r, t, length(sol), sol.retcode)
        catch e
            println(" warm $r CRASH: $(first(split(sprint(showerror, e), '\n')))")
            push!(rows, (run_idx=r, wall_s=NaN, alloc_bytes=0, steps=0, retcode="CRASH", T_end=NaN))
        end
    end
    return (rows=rows, last_sol=last_sol)
end

# ---- standalone linear-solve micro-benchmark: time solve(LinearProblem) per call on W = αI − J ----
# solve(LinearProblem(W,b), alg) does init (symbolic + numeric factorize, for direct) + triangular
# solve each call — the per-Newton-step linear-solve cost (warm; the one-time symbolic is compiled
# away by the compile probe + the micro-bench's own warmup). Direct-sparse solvers only now.
function run_microbench(W_sample, ls_alg, repeats::Int)
    (W_sample === nothing) && return (ok=false, per_call_s=NaN, alloc=0, note="no W sample")
    n = size(W_sample, 1)
    Random.seed!(0)
    b = rand(n)
    try                                       # warmup the path
        solve(LinearProblem(W_sample, b), ls_alg)
    catch e
        return (ok=false, per_call_s=NaN, alloc=0, note="solve failed: $(first(split(sprint(showerror, e), '\n')))")
    end
    ts = Float64[]; allocs = Float64[]
    for _ in 1:repeats
        GC.gc()
        try
            _, t, alloc, _gc, _mem = @timed solve(LinearProblem(W_sample, b), ls_alg)
            push!(ts, t); push!(allocs, Float64(alloc))
        catch e
            return (ok=false, per_call_s=NaN, alloc=0, note="timed run failed: $(first(split(sprint(showerror, e), '\n')))")
        end
    end
    return (ok=true, per_call_s=median(ts), alloc=median(allocs), note="")
end

# ---- trajectory accuracy vs reference (numeric state indexing — sharded path drops sol(t;idxs=)) ----
function _t_ignition(sol, T_idx, n=2001)
    isempty(sol.t) && return NaN
    ts = range(0, stop=sol.t[end], length=n)
    Ts = [Float64(sol(t)[T_idx]) for t in ts]
    dTdt = diff(Ts) ./ diff(ts)
    return ts[argmax(abs.(dTdt)) + 1]
end
function accuracy_vs_ref(sol, ref_sol, T_idx, t_end)
    (sol === nothing || ref_sol === nothing) && return (max_dT=NaN, dt_ign_rel=NaN, t_ign=NaN)
    ts = range(0, stop=t_end, length=200)
    dT = 0.0
    for t in ts
        dT = max(dT, abs(Float64(sol(t)[T_idx]) - Float64(ref_sol(t)[T_idx])))
    end
    tign = _t_ignition(sol, T_idx)
    tign_ref = _t_ignition(ref_sol, T_idx)
    dt_rel = isnan(tign_ref) || tign_ref == 0 ? NaN : abs(tign - tign_ref) / tign_ref
    return (max_dT=dT, dt_ign_rel=dt_rel, t_ign=tign)
end

# ---- reproducibility metadata ----
function write_meta(path, cfg, mech_names, solver_names)
    deps = Pkg.dependencies()
    pkgv(name) = begin
        for (_uuid, info) in deps
            info.name == name && return string(info.version)
        end
        return "n/a"
    end
    blas_threads = try; BLAS.get_num_threads(); catch; "?" end
    cpu_model = try; Sys.cpu_info()[1].model; catch; "?" end
    meta = Dict(
        "timestamp"    => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "hostname"     => gethostname(),
        "cpu"          => cpu_model,
        "cpu_threads"  => Sys.CPU_THREADS,
        "ram_GB"       => round(Sys.total_memory() / 2^30, digits=1),
        "julia"        => string(VERSION),
        "blas_threads" => blas_threads,
        "julia_threads"=> Threads.nthreads(),
        "packages"     => Dict(n => pkgv(n) for n in
                              ("ChemMechSim", "OrdinaryDiffEq", "LinearSolve", "ModelingToolkit",
                               "Sparspak", "MUMPS", "Pardiso", "Catalyst", "SciMLBase")),
        "git_sha"      => try; readchomp(`git -C $(@__DIR__) rev-parse HEAD`); catch; "unknown"; end,
        "config"       => Dict("mechs" => mech_names, "solvers" => solver_names,
                               "repeats" => cfg.repeats, "warmup" => cfg.warmup,
                               "reltol" => cfg.reltol, "abstol" => cfg.abstol,
                               "tspan_ms" => cfg.tspan_ms === nothing ? "per-mech" : cfg.tspan_ms),
    )
    YAML.write_file(path, meta)
    return meta
end

# ---- main ----
const REF_NAME = "umfpack"   # reference solver for the accuracy column

function main()
    cfg = parse_cli(ARGS)
    mkpath(cfg.out_dir)
    mech_names  = [strip(lowercase(String(m))) for m in split(cfg.mechs, ",")]
    solver_names= [strip(lowercase(String(s))) for s in split(cfg.solvers, ",")]
    sconfigs = solver_configs(solver_names)
    println("bench_matrix: mechs=$mech_names  solvers=$(getfield.(sconfigs, :name))  " *
            "repeats=$(cfg.repeats)  warmup=$(cfg.warmup)  microbench=$(cfg.microbench)  accuracy=$(cfg.accuracy)")
    have_ref = any(c.name == REF_NAME for c in sconfigs)
    cfg.accuracy && !have_ref && println("[accuracy] '$REF_NAME' not in solver set — accuracy column will be empty")

    fm     = open(joinpath(cfg.out_dir, "bench_matrix.csv"), "w")
    println(fm, "mech,n_species,n_reactions,n_states,nnz_jac,density_pct,linsolve,run_idx,wall_s,alloc_bytes,steps,retcode,T_end")
    fcompile = open(joinpath(cfg.out_dir, "bench_compile.csv"), "w")
    println(fcompile, "mech,n_states,first_solve_s,solver_used,steps,retcode")
    fmicro = cfg.microbench ? open(joinpath(cfg.out_dir, "bench_linsolve_micro.csv"), "w") : nothing
    fmicro !== nothing && println(fmicro, "mech,n_states,nnz_jac,linsolve,per_call_s,alloc_bytes,note")
    facc   = cfg.accuracy   ? open(joinpath(cfg.out_dir, "bench_accuracy.csv"), "w") : nothing
    facc   !== nothing && println(facc, "mech,linsolve,dt_ign_rel_pct,max_dT_K,t_ign_s")
    write_meta(joinpath(cfg.out_dir, "bench_meta.yaml"), cfg, mech_names, getfield.(sconfigs, :name))

    for mname in mech_names
        haskey(MECH_CONFIGS, mname) || (println("\n[skip] unknown mech '$mname'"); continue)
        mc = MECH_CONFIGS[mname]
        tspan = cfg.tspan_ms === nothing ? mc.tspan : (0.0, cfg.tspan_ms * 1e-3)
        println("\n=== $(mc.name) ($(mc.yaml))  tspan=$tspan ===")
        b = try; build_for_mech(mc, tspan)
              catch e; println("  BUILD FAIL: $(first(split(sprint(showerror, e), '\n')))"); flush(fm); continue; end
        nsp, nrx = length(b.mech.species), length(b.mech.reactions)
        density = b.n_states > 0 ? b.nnz / b.n_states^2 * 100 : 0.0
        println("  $nsp sp, $nrx rxn, $(b.n_states) states, nnz(jac)=$(b.nnz) ($(round(density, digits=1))% dense), build=$(round(b.t_build, digits=1))s")

        # one-time compile probe (mech-level, first selected solver) — separate from per-solver warm.
        # Dominated by the reaction-sharded Jacobian codegen compile; ~solver-independent.
        cp = compile_probe(b.prob, sconfigs[1].fbdf, cfg.reltol, cfg.abstol, b.T_idx)
        println(fcompile, join(Any[mc.name, b.n_states, isnan(cp.first_solve_s) ? "" : round(cp.first_solve_s, digits=2),
                             sconfigs[1].name, cp.steps, cp.retcode], ","))
        flush(fcompile)

        function write_rows(res, sc)
            for r in res.rows
                println(fm, join(Any[mc.name, nsp, nrx, b.n_states, b.nnz, round(density, digits=1),
                                     sc.name, r.run_idx, isnan(r.wall_s) ? "" : round(r.wall_s, digits=4),
                                     r.alloc_bytes, r.steps, r.retcode, isnan(r.T_end) ? "" : round(r.T_end, digits=2)], ","))
            end
            flush(fm)
        end

        ref_sol = nothing
        order = have_ref ? vcat([c for c in sconfigs if c.name == REF_NAME], [c for c in sconfigs if c.name != REF_NAME]) : sconfigs
        for sc in order
            print("  [$(sc.name)] "); flush(stdout)
            res = run_endtoend(b.prob, sc.fbdf, cfg.repeats, cfg.warmup, cfg.reltol, cfg.abstol, b.T_idx)
            write_rows(res, sc)
            if sc.name == REF_NAME
                ref_sol = res.last_sol
            elseif cfg.accuracy && facc !== nothing
                a = accuracy_vs_ref(res.last_sol, ref_sol, b.T_idx, tspan[2])
                println(facc, join(Any[mc.name, sc.name,
                                   isnan(a.dt_ign_rel) ? "" : round(a.dt_ign_rel * 100, digits=4),
                                   isnan(a.max_dT) ? "" : round(a.max_dT, digits=4),
                                   isnan(a.t_ign) ? "" : round(a.t_ign, digits=6)], ","))
                flush(facc)
            end
        end

        if cfg.microbench && fmicro !== nothing
            println("  [microbench]")
            for sc in sconfigs
                mb = run_microbench(b.W_sample, sc.ls, max(1, cfg.repeats))
                println(fmicro, join(Any[mc.name, b.n_states, b.nnz, sc.name,
                                     mb.ok ? round(mb.per_call_s, digits=6) : "",
                                     mb.ok ? mb.alloc : "", mb.note], ","))
                println("    $(sc.name): ", mb.ok ? "$(round(mb.per_call_s * 1000, digits=3)) ms/call" : "skip ($(mb.note))")
            end
            flush(fmicro)
        end
    end

    close(fm); close(fcompile); fmicro !== nothing && close(fmicro); facc !== nothing && close(facc)
    println("\nDone. CSVs + meta in $(cfg.out_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
