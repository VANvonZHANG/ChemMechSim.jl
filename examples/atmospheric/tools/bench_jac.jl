# Confirm that solve time scales linearly with simulated span for both Jacobian strategies.
#
#   julia --project=. examples/atmospheric/tools/bench_jac.jl
#
# One build_problem per strategy, then SOLVES at increasing spans on the same problem via
# remake — so the per-span numbers cost only solve time, not a rebuild.
#
# The build cost is a single number per strategy and does not depend on the span.

using ChemMechSim, ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: getname, parameters

const HERE = dirname(@__DIR__)                   # tools/ -> examples/atmospheric/
const T0, P0, R = 298.0, 102858.0, 8.314
C_AIR = P0 / (R * T0)
const X_INIT = Dict("N2"=>0.78,"O2"=>0.21,"H2O"=>0.01,"O3"=>3.0e-8,"NO2"=>1.0e-10,"CH4"=>1.8e-6)
const SPANS = (0.25, 0.5, 1.0)                   # simulated days

mech = load_mechanism(joinpath(HERE, "output", "mcm_alkanes_alkenes_frozen.yaml"))
u0 = Dict(String(sp.name) => get(X_INIT, String(sp.name), 0.0) * C_AIR for sp in mech.species)
r = BatchReactor(mech; mode=:kinetic, checks=false)
sys = extract_system(r)
Tp = parameters(sys)[findfirst(p -> String(getname(p)) == "T", parameters(sys))]

# remake MUST preserve the analytic Jacobian: the one-build/many-solves design of this
# bench rests on it. If it does not, fall back to build_problem per span (solve_s still
# measures only the solve) and say so on stdout + in the CSV footer comment, so the
# figure/table scripts and the commit message can report it.
remake_keeps_jac = Ref(true)

open(joinpath(HERE, "output", "bench_jac.csv"), "w") do io
    println(io, "strategy,span_days,build_s,solve_s,retcode")
    for (label, use_jac) in (("finite-difference", false), ("reaction-sharded", true))
        t_build = @elapsed prob = build_problem(r, u0, (0.0, SPANS[1]*86400.0);
                                                params = [Tp => T0], jac = use_jac)
        for days in SPANS
            p2 = remake(prob, tspan = (0.0, days * 86400.0))
            if use_jac && p2.f.jac === nothing
                remake_keeps_jac[] = false
                println(stderr, "WARNING: remake dropped the analytic jac (span=",
                        days, " d) — falling back to build_problem per span")
                p2 = build_problem(r, u0, (0.0, days * 86400.0);
                                   params = [Tp => T0], jac = true)
            end
            t_solve = @elapsed sol = solve(p2, FBDF(autodiff = false);
                                           reltol = 1e-6, abstol = 1e-12, saveat = 1200.0)
            println(io, label, ",", days, ",", round(t_build, digits = 1), ",",
                    round(t_solve, digits = 1), ",", sol.retcode)
            flush(io)
        end
    end
    if !remake_keeps_jac[]
        # Footer comment (starts with '#'): pandas read_csv(comment='#') skips it.
        println(io, "# remake dropped the analytic jac: reaction-sharded rows were",
                " produced with a per-span build_problem fallback")
    end
end
println("wrote ", joinpath(HERE, "output", "bench_jac.csv"),
        remake_keeps_jac[] ? "" : "  (remake dropped the analytic jac — used fallback)")
