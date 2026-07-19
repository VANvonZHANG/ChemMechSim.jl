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
 `params` is an optional Vector of Pair(parameter => value) (e.g. `[T => 500.0]`).
 When P is a differential state (const-V P-ODE, Task 4 + Task 3) and `u0` omits `\"P\"`, P0 is
 auto-filled as (Σ species c0)·R·T0 — the EOS initial pressure consistent with the supplied
 composition/T. T0 is resolved from `u0[\"T\"]` when present (the :adiabatic_constV case, where
 T is a state); otherwise the T PARAMETER's default value is used (the :fixedT case, where T is
 a parameter and the caller typically sets it via `params=` rather than `u0=`). Callers who
 override the T parameter via `params=` (e.g. `[T => 500.0]`) should pass `u0[\"P\"]` explicitly
 for a precise P0 — the auto-fill falls back to the T-param default, not the overridden value."
function build_problem(phase::ChemPhaseSystem, u0::AbstractDict, tspan; params=Pair[])
    sys = phase.sys
    unks = ModelingToolkit.unknowns(sys)
    byname = Dict(String(ModelingToolkit.getname(s)) => s for s in unks)
    pairs = [_u0_pairs(phase, u0); params]
    # P0 auto-fill (const-V P differential, Task 4 + Task 3): P0 = (Σ species c0)·R·T0 when P
    # is a state and the caller did not supply it. Excludes T and P from the concentration sum.
    if haskey(byname, "P") && !haskey(u0, "P")
        csum = sum(v for (k, v) in u0 if haskey(byname, k) && k != "T" && k != "P")
        # T0: from u0["T"] when present (:adiabatic_constV — T is a state); else from the T
        # parameter's default (:fixedT — T is a parameter the caller sets via `params=`).
        T0 = get(u0, "T", nothing)
        if T0 === nothing
            Tparam_idx = findfirst(p -> String(ModelingToolkit.getname(p)) == "T",
                                   ModelingToolkit.parameters(sys))
            T0 = Tparam_idx === nothing ? 300.0 :
                 ModelingToolkit.getdefault(ModelingToolkit.parameters(sys)[Tparam_idx])
        end
        push!(pairs, byname["P"] => R_GAS * csum * Float64(T0))
    end
    return ODEProblem(sys, pairs, tspan)
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

"Generate standalone Jacobian code from a BatchReactor's system."
generate_jacobian(r::BatchReactor; kwargs...) = generate_jacobian(extract_system(r); kwargs...)

"Generate standalone Jacobian Julia code from an MTK system (mirror of generate_function).
 `sparse=true` emits SparseMatrixCSC codegen (for large mechanisms — GRI-30 prep, Phase 5)."
function generate_jacobian(sys; sparse::Bool=false)
    jac = ModelingToolkit.calculate_jacobian(sys; sparse=sparse)
    return first(ModelingToolkit.build_function(jac,
                ModelingToolkit.unknowns(sys), ModelingToolkit.parameters(sys),
                [ModelingToolkit.t_nounits]))
end
