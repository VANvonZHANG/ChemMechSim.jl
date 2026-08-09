# Pipeline-cost decomposition across mechanism sizes (compile/warm principle).
# Run: julia --project=. examples/perf/gri30_benchmark.jl
#
# Demonstrates that the cold (first) solve is dominated by Julia JIT compilation of the
# MTK-generated RHS + reaction-sharded Jacobian; the warm solve reuses that native code.
# The gap (cold − warm = JIT compile) scales with mechanism size (reaction count) and is
# the real perf bottleneck for large mechanisms — NOT the linear-solver choice.
#
# Stages per mechanism:
#   build   = build_problem(jac=true, reaction_sharded) → lowering + mtkcompile + Jac codegen
#   cold    = first solve → Julia JIT compiles the generated code + integrates
#   warm    = second solve → native code cached, integrate only
#   jit_compile = cold − warm → the one-time Julia compilation cost
using ChemMechSim
using OrdinaryDiffEq: FBDF
using SciMLBase: solve
using Printf
using YAML
using Dates
using Pkg

const R = 8.314
const P0 = 101325.0
const TSPAN = (0.0, 5.0e-3)
const MECHS = [
    ("gri30",  joinpath(@__DIR__, "..", "mechanism", "gri30.yaml"),       1500.0, Dict("CH4"=>1.0/10.52,"O2"=>2.0/10.52,"N2"=>7.52/10.52)),
    ("ffcm2",  joinpath(@__DIR__, "..", "mechanism", "FFCM2.yaml"),       1500.0, Dict("CH4"=>1.0/10.52,"O2"=>2.0/10.52,"N2"=>7.52/10.52)),
    ("aramco", joinpath(@__DIR__, "..", "mechanism", "AramcoMech3.0.yaml"), 1500.0, Dict("CH4"=>1.0/10.52,"O2"=>2.0/10.52,"N2"=>7.52/10.52)),
]

mkpath(joinpath(@__DIR__, "output"))
const FCSV = open(joinpath(@__DIR__, "output", "bench_pipeline.csv"), "w")
println(FCSV, "mech,n_species,n_reactions,n_states,build_s,cold_s,warm_s,jit_compile_s")

@printf("%-8s %5s %5s %6s %8s %8s %8s %11s  %s\n", "mech", "sp", "rxn", "st", "build", "cold", "warm", "jit_compile", "retcode")
println("-"^78)

for (name, yaml, T0, X0) in MECHS
    mech = load_mechanism(yaml)
    nsp, nrx = length(mech.species), length(mech.reactions)
    c_tot = P0 / (R * T0)
    u0 = Dict(sp.name => get(X0, sp.name, 0.0) * c_tot for sp in mech.species); u0["T"] = T0
    reactor = BatchReactor(mech; mode=:adiabatic_constV, checks=false)
    t_build = @elapsed prob = build_problem(reactor, u0, TSPAN; jac=true, jac_strategy=:reaction_sharded)
    n_st = length(prob.u0)
    t_cold = @elapsed sol = solve(prob, FBDF(); reltol=1e-8, abstol=1e-12)
    t_warm = @elapsed solve(prob, FBDF(); reltol=1e-8, abstol=1e-12)
    jit = t_cold - t_warm
    @printf("%-8s %5d %5d %6d %8.2f %8.2f %8.2f %11.2f  %s\n", name, nsp, nrx, n_st, t_build, t_cold, t_warm, jit, sol.retcode)
    println(FCSV, join([name, nsp, nrx, n_st, round(t_build, digits=2), round(t_cold, digits=2), round(t_warm, digits=2), round(jit, digits=2)], ","))
    flush(FCSV)
end
close(FCSV)

# reproducibility metadata
deps = Pkg.dependencies()
pkgv(name) = begin
    for (_u, info) in deps
        info.name == name && return string(info.version)
    end
    return "n/a"
end
meta = Dict(
    "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
    "hostname" => gethostname(),
    "cpu" => try; Sys.cpu_info()[1].model; catch; "?" end,
    "cpu_threads" => Sys.CPU_THREADS,
    "ram_GB" => round(Sys.total_memory() / 2^30, digits=1),
    "julia" => string(VERSION),
    "packages" => Dict(n => pkgv(n) for n in
                      ("ChemMechSim", "OrdinaryDiffEq", "LinearSolve", "ModelingToolkit", "SciMLBase")),
    "git_sha" => try; readchomp(`git -C $(@__DIR__) rev-parse HEAD`); catch; "unknown"; end,
)
YAML.write_file(joinpath(@__DIR__, "output", "bench_pipeline_meta.yaml"), meta)
println("\nWrote $(joinpath(@__DIR__, "output", "bench_pipeline.csv"))")
println("\njit_compile = cold − warm = one-time Julia JIT compilation of MTK-generated RHS + Jacobian.")
println("Optimization lever: shrink generated code via opaque registered functions (less to compile).")
