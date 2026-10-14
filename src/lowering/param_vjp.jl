# —— Parameter VJP (θ-explicit rate kernels + local ForwardDiff) ————————————————————
# Spec: docs/superpowers/specs/2026-10-13-vjpp-local-forwarddiff-design.md
# dgrad[block_j] = (λ·ν_j) · ∂r_j/∂θ_j — assembled per reverse-pass node; the ∂f/∂p
# matrix is NEVER materialized (the vjp_p SciMLBase hook exists for exactly this).

"""One reaction's θ-block: dependency state columns, flat tunable indices (ascending),
the parameter symbols matching `pidx` order, and (Task 2) a generated rate kernel
`fn(buf, x, p, t)` with `x = [u[dep_cols]…; θ…]` — parameters as EXPLICIT inputs, the
only door through which ForwardDiff can inject tags."""
mutable struct ParamVjpBlock
    rx_idx::Int                  # reaction index into mech.reactions (the object stays `rx`)
    dep_cols::Vector{Int}        # dependency state columns into u
    pidx::Vector{Int}            # flat tunable indices (parameters(sys) order), ascending
    θ_syms::Vector{Any}          # parameter symbols matching pidx order
    fn::Any                      # filled by _materialize_param_vjp_blocks; nothing until then
end

# v1 gradient columns that are ZERO BY DESIGN (spec amendment 2026-10-13): T/P reach the
# opaque Float64 evaluators (keq in ThermoReverse, PLOG tables) that cannot carry
# ForwardDiff tags. They stay OUT of θ blocks — rate kernels read them from `p` (Float64),
# so their dgrad columns are simply never written.
const _VJP_ZERO_COLUMN_PARAMS = ("T", "P")

"""Order the parameter symbols appearing in `expr` by their flat index (ascending) —
deterministic block layout independent of hashing."""
function _ordered_param_syms(expr, flat_of::Dict)
    syms = Any[]
    for v in ModelingToolkit.get_variables(expr)
        if haskey(flat_of, v) && !any(isequal(v, s) for s in syms)  # `==` on symbolics is SymBool
            push!(syms, v)
        end
    end
    sort!(syms; by = s -> flat_of[s])
    return syms
end

"""Per-reaction θ-blocks. Reuses the lowering net-rate symbolic body (`_reaction_rate_expr`,
which routes every built-in and paramspec law through the same formula-once `symbolic_kf`);
the θ block is the tunable `parameters(sys)` members appearing in it, minus the zero-column
set (T/P — see `_VJP_ZERO_COLUMN_PARAMS`). `sys` must be the exact structural instance the
ODEProblem is built from: `pidx` indexes ITS flat parameter vector, and parameter ordering
is not stable across re-derivations of the system. Reactions whose only "parameters" are
the excluded ones get no block (e.g. PLOG: coefficients live in PLOG_TABLE)."""
function _param_vjp_blocks(mech::Mechanism, sys, config::MechanismConfig)
    param_syms = ModelingToolkit.parameters(sys)
    flat_of = Dict(p => i for (i, p) in enumerate(param_syms))
    state_idx = _state_name_index(sys)
    @assert !haskey(state_idx, "T") && !haskey(state_idx, "P")  # caller gate: isothermal, no P-ODE (v1)
    sid_to_idx = _species_id_to_state_index(mech, sys)
    blocks = ParamVjpBlock[]
    for (j, rx) in enumerate(mech.reactions)
        rate_expr = _reaction_rate_expr(rx, mech, sys, config, j)
        θ = filter(s -> !(String(ModelingToolkit.getname(s)) in _VJP_ZERO_COLUMN_PARAMS),
                   _ordered_param_syms(rate_expr, flat_of))
        isempty(θ) && continue
        pidx = Int[flat_of[s] for s in θ]
        # unique: the shared helper concatenates sids + extra states without dedupe, and a
        # duplicate column would bind two kernel inputs to the same state variable.
        dep_cols = unique(_reaction_dependency_state_indices(rx, mech, state_idx, sid_to_idx, false))
        push!(blocks, ParamVjpBlock(j, dep_cols, pidx, θ, nothing))
    end
    covered = sort!(unique(Int[i for b in blocks for i in b.pidx]))
    return blocks, covered
end

# Shared θ-dummy pool: each block's kernel binds its slice from its own xvars. Toplevel
# array variables (NOT runtime symbol creation — Symbolics isn't a direct dep).
@variables _pvjp_θ[1:12]

"""Materialize each block's rate kernel: the SAME symbolic net-rate body the sharded
Jacobian uses (`_reaction_rate_expr`), built as a function of x = [dep states…; θ…].
Each θ symbol is first substituted by a dummy from `_pvjp_θ`: a symbol appearing in BOTH
xvars and the parameter list gets bound from `p` (verified 2026-10-14 — the kernel would
read Float64 θ and ForwardDiff tags would never enter). Dummies keep the full param_syms
list positionally intact, so shared T/R/P_std still read from `p` in parameters(sys) order.
Mirrors the build_function call at jac_reaction_sharded.jl:402 (IIP, [2] slot, cse)."""
function _materialize_param_vjp_blocks!(blocks::Vector{ParamVjpBlock}, mech::Mechanism,
                                        sys, config::MechanismConfig)
    state_syms = ModelingToolkit.unknowns(sys)
    param_syms = ModelingToolkit.parameters(sys)
    for b in blocks
        dummies = Any[_pvjp_θ[i] for i in eachindex(b.θ_syms)]
        rate_expr = _reaction_rate_expr(mech.reactions[b.rx_idx], mech, sys, config, b.rx_idx)
        rate_expr = ModelingToolkit.substitute(rate_expr, Dict(zip(b.θ_syms, dummies)))
        xvars = vcat(Any[state_syms[c] for c in b.dep_cols], dummies)
        fn = ModelingToolkit.build_function(
            [rate_expr], xvars, param_syms, [ModelingToolkit.t_nounits];
            expression=Val{false}, wrap_gfw=Val{false}, cse=true)[2]
        b.fn === nothing || error("block $(b.rx_idx) already materialized")
        b.fn = fn
    end
    return blocks
end
