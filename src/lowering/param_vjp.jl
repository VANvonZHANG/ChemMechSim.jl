# —— Parameter VJP (θ-explicit rate kernels + local ForwardDiff) ————————————————————
# Spec: docs/superpowers/specs/2026-10-13-vjpp-local-forwarddiff-design.md
# dgrad[block_j] = (λ·ν_j) · ∂r_j/∂θ_j — assembled per reverse-pass node; the 2900×7915
# ∂f/∂p matrix is NEVER materialized (the vjp_p SciMLBase hook exists for exactly this).

"""One reaction's θ-block: dependency state columns, flat tunable indices (ascending),
the parameter symbols matching `pidx` order, and (Task 2) a generated rate kernel
`fn(buf, x, p, t)` with `x = [u[dep_cols]…; θ…]` — parameters as EXPLICIT inputs, the
only door through which ForwardDiff can inject tags."""
mutable struct ParamVjpBlock
    rx::Int                       # reaction index into mech.reactions
    dep_cols::Vector{Int}         # dependency state columns into u
    pidx::Vector{Int}             # flat tunable indices (parameters(sys) order), ascending
    θ_syms::Vector{Any}           # parameter symbols matching pidx order
    fn::Any                       # filled by _materialize_param_vjp_blocks; nothing until then
end

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
the θ block is simply the parameters(sys) members appearing in it. PLOG rates keep their
params in PLOG_TABLE (not tunables) → no block, no dgrad column — correct by construction."""
function _param_vjp_blocks(mech::Mechanism, sys, config::MechanismConfig)
    param_syms = ModelingToolkit.parameters(sys)
    flat_of = Dict(p => i for (i, p) in enumerate(param_syms))
    state_idx = _state_name_index(sys)
    sid_to_idx = _species_id_to_state_index(mech, sys)
    blocks = ParamVjpBlock[]
    for (j, rx) in enumerate(mech.reactions)
        rate_expr = _reaction_rate_expr(rx, mech, sys, config, j)
        θ = _ordered_param_syms(rate_expr, flat_of)
        isempty(θ) && continue
        pidx = Int[flat_of[s] for s in θ]
        dep_cols = _reaction_dependency_state_indices(rx, mech, state_idx, sid_to_idx, false)
        push!(blocks, ParamVjpBlock(j, dep_cols, pidx, θ, nothing))
    end
    covered = sort!(unique(Int[i for b in blocks for i in b.pidx]))
    return blocks, covered
end
