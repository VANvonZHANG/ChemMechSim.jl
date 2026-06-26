# Layered API: extract_system / build_problem / simulate / generate_function.
# simulate/build_problem operate on a ChemPhaseSystem (the Phase 1 "reactor").
# generate_function returns standalone Julia code (spec §6, §7).

"Extract the underlying MTK ODESystem from a ChemPhaseSystem."
extract_system(phase::ChemPhaseSystem) = phase.sys

"Resolve a speciesname => value initial-condition map to state => value pairs
 (the mtkcompile'd system may have reordered its states)."
function _u0_pairs(phase::ChemPhaseSystem, u0::AbstractDict)
    byname = Dict(String(ModelingToolkit.getname(s)) => s
                  for s in ModelingToolkit.unknowns(phase.sys))
    return [byname[k] => v for (k, v) in u0]
end

"Build an ODEProblem from a ChemPhaseSystem. `u0` is a Dict(speciesname => value);
 `params` is an optional Vector of Pair(parameter => value) (e.g. `[T => 500.0]`)."
function build_problem(phase::ChemPhaseSystem, u0::AbstractDict, tspan; params=Pair[])
    return ODEProblem(phase.sys, [_u0_pairs(phase, u0); params], tspan)
end

"Simulate a ChemPhaseSystem over `tspan`. `u0` is a Dict(speciesname => value);
 `params` sets parameter values (e.g. `[T => 500.0]`). Default solver Tsit5()
 (non-stiff); stiff mechanisms (Phase 5) should pass Rodas5P/CVODE_BDF."
function simulate(phase::ChemPhaseSystem, tspan=(0.0, 1.0); u0,
                  solver=Tsit5(), params=Pair[], kwargs...)
    return solve(build_problem(phase, u0, tspan; params=params), solver; kwargs...)
end

"Generate standalone RHS Julia code (an out-of-place function Expr) from an MTK system."
function generate_function(sys)
    rhss = [eq.rhs for eq in equations(sys)]
    return first(ModelingToolkit.build_function(rhss, ModelingToolkit.unknowns(sys),
                                                ModelingToolkit.parameters(sys),
                                                [ModelingToolkit.t_nounits]))
end

# —— BatchReactor dispatch (Phase 2): delegate to the wrapped ChemPhaseSystem ——

"Extract the underlying MTK ODESystem from a BatchReactor."
extract_system(r::BatchReactor) = extract_system(r.phase)

"Build an ODEProblem from a BatchReactor. `u0` is a Dict(speciesname => value);
 `params` is an optional Pair vector."
build_problem(r::BatchReactor, u0::AbstractDict, tspan; params=Pair[]) =
    build_problem(r.phase, u0, tspan; params=params)

"Simulate a BatchReactor over `tspan`. `u0` is a Dict(speciesname => value);
 `params` sets parameter values. Default solver Tsit5()."
function simulate(r::BatchReactor, tspan=(0.0, 1.0); u0, solver=Tsit5(), params=Pair[], kwargs...)
    return simulate(r.phase, tspan; u0=u0, solver=solver, params=params, kwargs...)
end

"Generate standalone RHS Julia code from a BatchReactor's system."
generate_function(r::BatchReactor) = generate_function(extract_system(r))

"Generate standalone Jacobian Julia code. (stub — Phase 3)"
generate_jacobian(sys; kwargs...) =
    error("generate_jacobian: not implemented in Phase 1; see the Phase 3 plan.")
