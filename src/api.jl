# Layered API: extract_system / build_problem / simulate / generate_function.
# simulate/build_problem operate on a ChemPhaseSystem (the Phase 1 "reactor").
# generate_function returns standalone Julia code (spec §6, §7).
using SciMLBase: NoSpecialize, ODEFunction
using ModelingToolkit: generate_rhs

"""Extract the underlying ModelingToolkit `ODESystem` from a `ChemPhaseSystem`.

The returned system is `mtkcompile`d, so its states may be reordered relative to
the mechanism; inspect with `ModelingToolkit.unknowns` / `equations`. A
`BatchReactor` is accepted too (delegates to its wrapped phase).

# Example
sys = extract_system(reactor)   # then: ModelingToolkit.equations(sys)
"""
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

"""Build an `ODEProblem` from a `ChemPhaseSystem` (a `BatchReactor` too).

    build_problem(phase, u0, tspan; params=Pair[], jac=false, jac_chunked=false,
                  jac_strategy=:auto, chunk_size=200, cse_chunk_size=chunk_size,
                  write_chunk_size=500)

`u0` is a `Dict(speciesname => value)` mapping species names to concentrations
[mol/m³], plus `"T"` for the initial temperature [K] in adiabatic modes; `params`
is an optional vector of `Pair(parameter => value)` (e.g. `[T => 500.0]`).
`jac=true` enables the analytic Jacobian: `jac_strategy` is `:auto` (default —
routes to `:reaction_sharded` when the mechanism and config support it, else
`:none` with a warning), `:reaction_sharded`, or `:none` (the solver ForwardDiff
Jacobian). `jac_chunked` is a deprecated alias of `jac`; the chunk-size kwargs
are retained for call-site compatibility. When P is a differential state
(const-V P-ODE) and `u0` omits `"P"`, P0 is auto-filled as
(Σ species c0)·R·T0 — the EOS initial pressure consistent with the supplied
composition/T. T0 comes from `u0["T"]` when present (the `:adiabatic_constV`
case, where T is a state); otherwise from the T PARAMETER default (the
`:fixedT` case). Callers overriding T via `params=` should pass `u0["P"]`
explicitly for a precise P0 — the auto-fill falls back to the T-param default,
not the overridden value.

# Example
prob = build_problem(reactor, u0, (0.0, 5e-3); jac=true)
"""
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
        supported = _reaction_sharded_supports_mechconfig(phase.mech, phase.config)
        if !supported && (jac || jac_chunked)
            # An EXPLICIT analytic-Jacobian request must not degrade silently (the
            # 2026-09-20 lesson): name the blocking kinetics types and the outcome.
            bad = unique(typeof(rx.kinetics) for rx in phase.mech.reactions
                         if !_reaction_sharded_supports(rx))
            @warn "build_problem: jac=true requested, but the reaction-sharded Jacobian does " *
                  "not support these kinetics types: $(join(bad, ", ")) — falling back to " *
                  ":none (the solver's ForwardDiff). Pass jac_strategy=:none to request the " *
                  "fallback explicitly."
        end
        strategy = supported ? :reaction_sharded : :none
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

"""Simulate over `tspan` — one-call convenience over `build_problem` + `solve`.

    simulate(x, tspan=(0.0, 1.0); u0, solver=Tsit5(), params=Pair[], jac=false,
             jac_chunked=false, jac_strategy=:auto, chunk_size=200,
             cse_chunk_size=chunk_size, write_chunk_size=500, kwargs...)

`x` is a `ChemPhaseSystem` or a `BatchReactor`. `u0` maps species names to
concentrations [mol/m³] and `"T"` to the initial temperature [K]; `params` sets
parameter values (e.g. `[T => 500.0]`); `kwargs` forward to `solve` (`reltol`,
`abstol`, callbacks, ...). The default solver `Tsit5()` suits non-stiff toy
mechanisms; pass a stiff solver for real chemistry (e.g. `Rodas5P()` or
`FBDF()`). Jacobian options are those of `build_problem` (`jac=true` and
friends).

# Example
reactor = BatchReactor(mech; mode=:adiabatic_constV)
sol = simulate(reactor, (0.0, 5e-3); u0=u0, solver=FBDF(), reltol=1e-8, abstol=1e-12)
"""
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

"""Generate standalone RHS Julia code (an out-of-place function `Expr`) from an
MTK system — or from a `BatchReactor`/`ChemPhaseSystem` via its system."""
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

"""Simulate a `BatchReactor` over `tspan`; see the `ChemPhaseSystem` method
for the full argument documentation (the kwargs are identical)."""
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
 `sparse=true` emits SparseMatrixCSC codegen (for large mechanisms). Accepts a
 `BatchReactor` via its system."
function generate_jacobian(sys; sparse::Bool=false)
    jac = ModelingToolkit.calculate_jacobian(sys; sparse=sparse)
    return first(ModelingToolkit.build_function(jac,
                ModelingToolkit.unknowns(sys), ModelingToolkit.parameters(sys),
                [ModelingToolkit.t_nounits]))
end
