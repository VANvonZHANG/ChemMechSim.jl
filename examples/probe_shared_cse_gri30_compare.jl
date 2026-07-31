# Compare shared-CSE symbolic Jacobian against full MTK Jacobian on GRI30.
#
# Run:
#   julia --project=. examples/probe_shared_cse_gri30_compare.jl

using ChemMechSim
using ModelingToolkit
using Printf

function _state_index(sys)
    Dict(String(ModelingToolkit.getname(s)) => i for (i, s) in enumerate(ModelingToolkit.unknowns(sys)))
end

yaml = joinpath(@__DIR__, "data", "gri30.yaml")
isfile(yaml) || (yaml = joinpath(@__DIR__, "..", "test", "data", "gri30.yaml"))

const R = 8.314
const T0 = 1500.0
const P0 = 101325.0
const X0 = Dict("CH4" => 0.095, "O2" => 0.190, "N2" => 0.715)

println("=" ^ 70)
println("Shared-CSE GRI30 Jacobian comparison")
println("=" ^ 70)

const CSE_CHUNK_SIZE = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
const WRITE_CHUNK_SIZE = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
@printf("Chunks: cse=%d, write=%d\n", CSE_CHUNK_SIZE, WRITE_CHUNK_SIZE)

mech = load_mechanism(yaml)
@printf("Loaded: %d species, %d reactions\n", length(mech.species), length(mech.reactions))

reactor = BatchReactor(mech; mode=:adiabatic_constV, checks=false)
sys = extract_system(reactor)
@printf("System: %d unknowns, %d parameters\n",
        length(ModelingToolkit.unknowns(sys)), length(ModelingToolkit.parameters(sys)))

c_tot = P0 / (R * T0)
u0 = Dict(sp.name => get(X0, sp.name, 0.0) * c_tot for sp in mech.species)
u0["T"] = T0
prob = build_problem(reactor, u0, (0.0, 5.0e-3))

GC.gc()
m0 = Sys.maxrss() / 1e9
t0 = time()
jac_shared!, J_proto = ChemMechSim.build_shared_cse_jac(
    sys; cse_chunk_size=CSE_CHUNK_SIZE, write_chunk_size=WRITE_CHUNK_SIZE)
@printf("build_shared_cse_jac: %.2fs, RSS delta %.2f GB\n",
        time() - t0, Sys.maxrss() / 1e9 - m0)

J_shared = copy(J_proto)
fill!(J_shared.nzval, NaN)
GC.gc()
m0 = Sys.maxrss() / 1e9
t0 = time()
jac_shared!(J_shared, prob.u0, prob.p, 0.0)
@printf("first shared jac call: %.2fs, RSS delta %.2f GB\n",
        time() - t0, Sys.maxrss() / 1e9 - m0)

GC.gc()
m0 = Sys.maxrss() / 1e9
t0 = time()
jac_full! = ModelingToolkit.generate_jacobian(
    sys; sparse=true, expression=Val{false}, wrap_gfw=Val{false})[2]
@printf("generate full MTK jac RGF: %.2fs, RSS delta %.2f GB\n",
        time() - t0, Sys.maxrss() / 1e9 - m0)

J_full = similar(J_proto)
fill!(J_full.nzval, NaN)
GC.gc()
m0 = Sys.maxrss() / 1e9
t0 = time()
jac_full!(J_full, prob.u0, prob.p, 0.0)
@printf("first full jac call: %.2fs, RSS delta %.2f GB\n",
        time() - t0, Sys.maxrss() / 1e9 - m0)

max_abs = maximum(abs.(J_shared.nzval .- J_full.nzval))
max_rel = maximum(abs.(J_shared.nzval .- J_full.nzval) ./ max.(abs.(J_full.nzval), 1.0))
@printf("max_abs_diff: %.6e\n", max_abs)
@printf("max_rel_diff: %.6e\n", max_rel)

if max_abs <= 1e-8 || max_rel <= 1e-10
    println("PROBE PASS")
else
    println("PROBE FAIL")
    exit(1)
end
