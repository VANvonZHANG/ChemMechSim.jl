# Diagnostic probe for large-mechanism symbolic Jacobian pipeline stages.
#
# Run:
#   julia --project=. examples/probe_large_mech_jac_pipeline.jl GRI30 false
#   julia --project=. examples/probe_large_mech_jac_pipeline.jl FFCM2 false
#   julia --project=. examples/probe_large_mech_jac_pipeline.jl Aramco false

using ChemMechSim
using ModelingToolkit
using Printf
using SparseArrays

const CASE = length(ARGS) >= 1 ? ARGS[1] : "FFCM2"
const RUN_FULL_MTK = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : false

const YAML_PATHS = Dict(
    "GRI30" => joinpath(@__DIR__, "data", "gri30.yaml"),
    "FFCM2" => joinpath(@__DIR__, "data", "FFCM2.yaml"),
    "Aramco" => joinpath(@__DIR__, "data", "AramcoMech3.0.yaml"),
)

const R = 8.314
const T0 = 1500.0
const P0 = 101325.0
const X0 = Dict(
    "CH4" => 1.0 / 10.52,
    "O2" => 2.0 / 10.52,
    "N2" => 7.52 / 10.52,
)

maxrss_gb() = Sys.maxrss() / 1.0e9

function timed_stage(f, name)
    GC.gc()
    maxrss0 = maxrss_gb()
    t0 = time()
    @printf("\n--- %s ---\n", name)
    result = f()
    elapsed = time() - t0
    maxrss1 = maxrss_gb()
    @printf("%-34s %.3f s, maxrss_delta %+7.3f GB, maxrss %.3f GB\n",
            name * ":", elapsed, maxrss1 - maxrss0, maxrss1)
    return result
end

function print_stats(stats)
    println("stats propertynames:")
    for name in propertynames(stats)
        @printf("  %-32s %s\n", String(name) * ":", getproperty(stats, name))
    end
end

function unsupported_return_stats_method_error(err)
    err isa MethodError || return false
    err.f === Core.kwcall || return false
    length(err.args) >= 2 || return false
    kwargs = err.args[1]
    return kwargs isa NamedTuple &&
        haskey(kwargs, :return_stats) &&
        err.args[2] === ChemMechSim.build_shared_cse_jac
end

function build_shared_cse_jac_with_optional_stats(sys)
    try
        result = ChemMechSim.build_shared_cse_jac(sys; return_stats=true)
        if result isa Tuple && length(result) == 3
            jac!, J_proto, stats = result
            return jac!, J_proto, stats
        elseif result isa Tuple && length(result) == 2
            jac!, J_proto = result
            stats = (; return_stats_supported=false,
                     message="return_stats=true accepted but no stats were returned")
            return jac!, J_proto, stats
        else
            error("unexpected build_shared_cse_jac return value: $(typeof(result))")
        end
    catch err
        if unsupported_return_stats_method_error(err)
            println("build_shared_cse_jac return_stats=true unsupported; falling back without stats.")
            jac!, J_proto = ChemMechSim.build_shared_cse_jac(sys)
            stats = (; return_stats_supported=false,
                     fallback="MethodError for return_stats=true")
            return jac!, J_proto, stats
        end
        rethrow()
    end
end

function initial_state(mech)
    c_tot = P0 / (R * T0)
    u0 = Dict(sp.name => get(X0, sp.name, 0.0) * c_tot for sp in mech.species)
    u0["T"] = T0
    return u0
end

function finite_report(label, J)
    vals = SparseArrays.nonzeros(J)
    nfinite = count(isfinite, vals)
    ok = nfinite == length(vals)
    @printf("%-34s finite=%s (%d/%d), nnz=%d\n",
            label * ":", ok, nfinite, length(vals), SparseArrays.nnz(J))
    return ok
end

haskey(YAML_PATHS, CASE) ||
    error("unknown CASE=$(repr(CASE)); choose one of $(sort(collect(keys(YAML_PATHS))))")

yaml_path = YAML_PATHS[CASE]
isfile(yaml_path) || error("mechanism file not found: $yaml_path")

println("=" ^ 78)
println("Large-mechanism Jacobian pipeline probe")
println("=" ^ 78)
println("CASE=$CASE")
println("RUN_FULL_MTK=$RUN_FULL_MTK")
println("YAML=$yaml_path")

mech = timed_stage("load_mechanism") do
    load_mechanism(yaml_path)
end
@printf("Mechanism counts: species=%d, reactions=%d\n",
        length(mech.species), length(mech.reactions))

reactor = timed_stage("BatchReactor lower") do
    BatchReactor(mech; mode=:adiabatic_constV, checks=false)
end

sys = extract_system(reactor)
unknowns_sys = ModelingToolkit.unknowns(sys)
params_sys = ModelingToolkit.parameters(sys)
eqs_sys = ModelingToolkit.equations(sys)
@printf("System counts: unknowns=%d, parameters=%d, equations=%d\n",
        length(unknowns_sys), length(params_sys), length(eqs_sys))

J_sparse = timed_stage("calculate_jacobian sparse") do
    ModelingToolkit.calculate_jacobian(sys; sparse=true)
end
rows, cols = size(J_sparse)
density = SparseArrays.nnz(J_sparse) / (rows * cols)
@printf("Sparse Jacobian: size=%dx%d, nnz=%d, density=%.6e\n",
        rows, cols, SparseArrays.nnz(J_sparse), density)

if RUN_FULL_MTK
    timed_stage("generate_jacobian full Expr") do
        ModelingToolkit.generate_jacobian(
            sys; sparse=true, expression=Val{true}, wrap_gfw=Val{false})
    end
else
    println("\n--- generate_jacobian full Expr ---")
    println("SKIP: RUN_FULL_MTK=false; standalone comparison Expr generation skipped. " *
            "build_shared_cse_jac may still generate Expr internally.")
end

jac_shared!, J_proto, stats = timed_stage("build_shared_cse_jac") do
    build_shared_cse_jac_with_optional_stats(sys)
end
print_stats(stats)

u0 = initial_state(mech)
prob = timed_stage("build_problem no jac") do
    build_problem(reactor, u0, (0.0, 1.0e-6); jac=false)
end
@printf("Problem counts: u0=%d, p_type=%s\n", length(prob.u0), typeof(prob.p))

J_first = copy(J_proto)
fill!(SparseArrays.nonzeros(J_first), NaN)
first_ok = timed_stage("first shared jac call") do
    jac_shared!(J_first, prob.u0, prob.p, 0.0)
    finite_report("first shared jac call", J_first)
end

J_second = copy(J_proto)
fill!(SparseArrays.nonzeros(J_second), NaN)
second_ok = timed_stage("second shared jac call") do
    jac_shared!(J_second, prob.u0, prob.p, 0.0)
    finite_report("second shared jac call", J_second)
end

println("\nProbe finite summary: first=$(first_ok), second=$(second_ok)")
