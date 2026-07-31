# Layered API: extract_system / build_problem / simulate / generate_function.
# simulate/build_problem operate on a ChemPhaseSystem (the Phase 1 "reactor").
# generate_function returns standalone Julia code (spec §6, §7).
using SciMLBase: NoSpecialize, ODEFunction
using ModelingToolkit: generate_rhs

"Extract the underlying MTK ODESystem from a ChemPhaseSystem."
extract_system(phase::ChemPhaseSystem) = phase.sys

"Resolve a speciesname => value initial-condition map to state => value pairs
 (the mtkcompile'd system may have reordered its states)."
function _u0_pairs(phase::ChemPhaseSystem, u0::AbstractDict)
    byname = Dict(String(ModelingToolkit.getname(s)) => s
                  for s in ModelingToolkit.unknowns(phase.sys))
    return [byname[k] => v for (k, v) in u0]
end

function _normalize_jac_strategy(jac::Bool, jac_chunked::Bool, jac_strategy::Symbol)
    # `jac_chunked` is kept as a deprecated no-op alias for `jac` (formerly selected the
    # shared-CSE Jacobian; that path was removed — see build_problem). It only contributes to
    # the "user wants a Jacobian" predicate below and no longer selects any specific strategy.
    jac_strategy === :shared_cse &&
        throw(ArgumentError(
            "jac_strategy=:shared_cse was removed (its Expr-surgery was fragile across " *
            "Symbolics versions). Use :auto (preferred — routes to :reaction_sharded when " *
            "supported) or :reaction_sharded / :none."))
    jac_strategy in (:auto, :reaction_sharded, :mtk, :none) ||
        throw(ArgumentError(
            "jac_strategy must be one of :auto, :reaction_sharded, :mtk, :none"))
    jac_strategy === :auto && return (jac || jac_chunked) ? :auto : :none
    jac_strategy === :mtk &&
        throw(ArgumentError("jac_strategy=:mtk is intentionally disabled for large mechanisms; use :reaction_sharded or :none"))
    return jac_strategy
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
function build_problem(phase::ChemPhaseSystem, u0::AbstractDict, tspan;
                        params=Pair[], jac::Bool=false, jac_chunked::Bool=false,
                        jac_strategy::Symbol=:auto,
                        chunk_size::Int=200, cse_chunk_size::Int=chunk_size,
                        write_chunk_size::Int=500)
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
    strategy = _normalize_jac_strategy(jac, jac_chunked, jac_strategy)
    # Resolve :auto: prefer :reaction_sharded (the only analytic path that scales to large
    # mechanisms) when the mechanism + config are fully supported; otherwise fall back to
    # :none (the ODE solver's default ForwardDiff). The shared-CSE path was removed (its
    # Expr-surgery broke across Symbolics versions); :auto no longer needs the cse_chunk_size
    # /write_chunk_size kwargs, but they remain in the signature for call-site compatibility.
    if strategy === :auto
        strategy = _reaction_sharded_supports_mechconfig(phase.mech, phase.config) ?
                   :reaction_sharded : :none
    end
    if strategy === :reaction_sharded
        prob_baseline = ODEProblem(sys, pairs, tspan)
        rhs_iip = generate_rhs(sys; expression=Val{false}, wrap_gfw=Val{false})[2]
        jac!, J_proto = build_reaction_sharded_jac(
            phase.mech; config=phase.config, checks=false, sys=sys)
        ofn = ODEFunction{true, NoSpecialize}(rhs_iip; jac=jac!, jac_prototype=J_proto)
        return ODEProblem(ofn, prob_baseline.u0, tspan, prob_baseline.p)
    else
        return ODEProblem(sys, pairs, tspan)
    end
end

"Simulate a ChemPhaseSystem over `tspan`. `u0` is a Dict(speciesname => value);
 `params` sets parameter values (e.g. `[T => 500.0]`). Default solver Tsit5()
 (non-stiff); stiff mechanisms (Phase 5) should pass Rodas5P/CVODE_BDF."
function simulate(phase::ChemPhaseSystem, tspan=(0.0, 1.0); u0, solver=Tsit5(),
                  params=Pair[], jac::Bool=false, jac_chunked::Bool=false,
                  jac_strategy::Symbol=:auto,
                  chunk_size::Int=200, cse_chunk_size::Int=chunk_size,
                  write_chunk_size::Int=500, kwargs...)
    prob = build_problem(phase, u0, tspan; params=params, jac=jac,
                         jac_chunked=jac_chunked, jac_strategy=jac_strategy,
                         chunk_size=chunk_size, cse_chunk_size=cse_chunk_size,
                         write_chunk_size=write_chunk_size)
    return solve(prob, solver; kwargs...)
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
build_problem(r::BatchReactor, u0::AbstractDict, tspan; params=Pair[], jac::Bool=false,
              jac_chunked::Bool=false, jac_strategy::Symbol=:auto, chunk_size::Int=200,
              cse_chunk_size::Int=chunk_size, write_chunk_size::Int=500) =
    build_problem(r.phase, u0, tspan; params=params, jac=jac,
                  jac_chunked=jac_chunked, jac_strategy=jac_strategy,
                  chunk_size=chunk_size, cse_chunk_size=cse_chunk_size,
                  write_chunk_size=write_chunk_size)

"Simulate a BatchReactor over `tspan`. `u0` is a Dict(speciesname => value);
 `params` sets parameter values. Default solver Tsit5()."
function simulate(r::BatchReactor, tspan=(0.0, 1.0); u0, solver=Tsit5(),
                  params=Pair[], jac::Bool=false, jac_chunked::Bool=false,
                  jac_strategy::Symbol=:auto,
                  chunk_size::Int=200, cse_chunk_size::Int=chunk_size,
                  write_chunk_size::Int=500, kwargs...)
    return simulate(r.phase, tspan; u0=u0, solver=solver, params=params, jac=jac,
                    jac_chunked=jac_chunked, jac_strategy=jac_strategy,
                    chunk_size=chunk_size, cse_chunk_size=cse_chunk_size,
                    write_chunk_size=write_chunk_size, kwargs...)
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
