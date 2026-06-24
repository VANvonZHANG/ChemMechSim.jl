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

"Build an ODEProblem from a ChemPhaseSystem. `u0` is a Dict(speciesname => value)."
function build_problem(phase::ChemPhaseSystem, u0::AbstractDict, tspan)
    return ODEProblem(phase.sys, _u0_pairs(phase, u0), tspan)
end

"Simulate a ChemPhaseSystem over `tspan`. `u0` is a Dict(speciesname => value).
 Default solver Tsit5() (non-stiff); stiff mechanisms (Phase 5) should pass Rodas5P/CVODE_BDF."
function simulate(phase::ChemPhaseSystem, tspan=(0.0, 1.0); u0,
                  solver=Tsit5(), kwargs...)
    return solve(build_problem(phase, u0, tspan), solver; kwargs...)
end

"Generate standalone RHS Julia code (an in-place function Expr) from an MTK system."
function generate_function(sys)
    rhss = [eq.rhs for eq in equations(sys)]
    return first(ModelingToolkit.build_function(rhss, ModelingToolkit.unknowns(sys),
                                                ModelingToolkit.parameters(sys),
                                                [ModelingToolkit.t_nounits]))
end

"Generate standalone Jacobian Julia code. (stub — Phase 3)"
generate_jacobian(sys; kwargs...) =
    error("generate_jacobian: not implemented in Phase 1; see the Phase 3 plan.")
