# —— Parameter VJP (θ-explicit rate kernels + local ForwardDiff) ————————————————————
# Spec: docs/superpowers/specs/2026-10-13-vjpp-local-forwarddiff-design.md
# dgrad[block_j] = (λ·ν_j) · ∂r_j/∂θ_j — assembled per reverse-pass node; the ∂f/∂p
# matrix is NEVER materialized (the vjp_p SciMLBase hook exists for exactly this).

using ForwardDiff       # local gradient over each block's θ-explicit kernel (Task 3)
using SparseArrays      # _stoich_csc net-stoichiometry (jac_reaction_sharded.jl precedent)

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
    # caller gate (structural, mirrors build_param_vjp): species-concentration states only —
    # a non-species state (energy T, P-ODE) would get a silently zero adjoint row.
    @assert isempty(setdiff(Set(keys(state_idx)), Set(String(sp.name) for sp in mech.species)))
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
# array variables (NOT runtime symbol creation — Symbolics isn't a direct dep). The dummies
# are POSITIONAL, per generated function — never combine two blocks' substituted bodies
# symbolically (block 1's and block 5's _pvjp_θ[1] would alias).
@variables _pvjp_θ[1:12]

"""Materialize each block's rate kernel: the SAME symbolic net-rate body the sharded
Jacobian uses (`_reaction_rate_expr`), built as a function of x = [dep states…; θ…].
Each θ symbol is first substituted by a dummy from `_pvjp_θ`: a symbol appearing in BOTH
xvars and the parameter list gets bound from `p` (verified 2026-10-14 — the kernel would
read Float64 θ and ForwardDiff tags would never enter). Dummies keep the full param_syms
list positionally intact, so remaining params (T/R/P_std) read from `p` positionally, in
`parameters(sys)` order — callers must pass the FLAT vector (`_parameter_vector`), never a
raw MTKParameters (its `p[j]` returns the j-th component, not the j-th scalar).
Mirrors the build_function call at jac_reaction_sharded.jl:402 (IIP, [2] slot, cse)."""
function _materialize_param_vjp_blocks!(blocks::Vector{ParamVjpBlock}, mech::Mechanism,
                                        sys, config::MechanismConfig)
    state_syms = ModelingToolkit.unknowns(sys)
    param_syms = ModelingToolkit.parameters(sys)
    for b in blocks
        b.fn === nothing || error("block $(b.rx_idx) already materialized")
        nθ = length(b.θ_syms)
        nθ > length(_pvjp_θ) && error(
            "param_vjp: θ pool exhausted — reaction $(b.rx_idx) has $nθ θ params (pool $(length(_pvjp_θ))). Enlarge _pvjp_θ.")
        dummies = Any[_pvjp_θ[i] for i in eachindex(b.θ_syms)]
        rate_expr = _reaction_rate_expr(mech.reactions[b.rx_idx], mech, sys, config, b.rx_idx)
        rate_expr = ModelingToolkit.substitute(rate_expr, Dict(zip(b.θ_syms, dummies)))
        xvars = vcat(Any[state_syms[c] for c in b.dep_cols], dummies)
        fn = ModelingToolkit.build_function(
            [rate_expr], xvars, param_syms, [ModelingToolkit.t_nounits];
            expression=Val{false}, wrap_gfw=Val{false}, cse=true)[2]
        b.fn = fn
    end
    return blocks
end

"""Net-stoichiometry CSC over the block reactions: S[i, col] = ν_{i, rx(col)}
(products − reactants, duplicates summed by sparse()). s = Sᵀλ gives λ·ν_j per block."""
function _stoich_csc(mech::Mechanism, sys, blocks::Vector{ParamVjpBlock})
    sid_to_idx = _species_id_to_state_index(mech, sys)
    n = length(ModelingToolkit.unknowns(sys))
    I = Int[]; J = Int[]; V = Float64[]
    for (col, b) in enumerate(blocks)
        rx = mech.reactions[b.rx_idx]
        for (sid, ν) in rx.products
            push!(I, sid_to_idx[sid]); push!(J, col); push!(V, ν)
        end
        for (sid, ν) in rx.reactants
            push!(I, sid_to_idx[sid]); push!(J, col); push!(V, -ν)
        end
    end
    return SparseArrays.sparse(I, J, V, n, length(blocks))
end

"""Build the parameter-VJP function, or (nothing, reasons) when not attachable.
Gate: the system's states must be species concentrations only (v1 is isothermal, no
pressure-ODE — a STRUCTURAL check rejects any non-species state, whose adjoint row would
otherwise be silently dropped; energy/pressure states get a mode hint, and a species
legitimately named "T"/"P" is not rejected). `sys` is REQUIRED and must be the exact
structural instance the ODEProblem is built from — a separately lowered sys can order
parameters differently (the 8277716 lesson) and lengths alone cannot detect it. The T
PARAMETER stays a zero column by design — the opaque keq node is not Dual-generic
(spec amendment 2026-10-13); temperature sensitivity remains the default channel's job."""
function build_param_vjp(mech::Mechanism; config::MechanismConfig=MechanismConfig(),
                         checks::Bool=false, sys)
    state_idx = _state_name_index(sys)
    # structural (not name-based): any state that is not a species would get a silently
    # zero adjoint row; a species legitimately named "T"/"P" must NOT be rejected.
    species_names = Set(String(sp.name) for sp in mech.species)
    extras = sort!(collect(setdiff(Set(keys(state_idx)), species_names)))
    if !isempty(extras)
        hint = "T" in extras ? " (energy mode?)" : "P" in extras ? " (pressure-ODE mode?)" : ""
        return (nothing, ["states beyond species ($(join(extras, ", ")))$hint — vjp_p v1 covers concentration states only"])
    end
    blocks, covered = _param_vjp_blocks(mech, sys, config)
    _materialize_param_vjp_blocks!(blocks, mech, sys, config)
    param_syms = ModelingToolkit.parameters(sys)
    covered_set = Set(covered)
    expected_zero = ["T"]  # (extend with "R"/"P_std" here ONLY if Task 5 Step 2 shows they are
                          #  tunables absent from every rate expression; then document the columns)
    gaps = String[]
    for (i, p) in enumerate(param_syms)
        name = String(ModelingToolkit.getname(p))
        (i in covered_set || name in expected_zero) || push!(gaps, name)
    end
    isempty(gaps) || return (nothing, ["uncovered tunables: $(join(gaps, ", "))"])
    S = _stoich_csc(mech, sys, blocks)
    return (_make_param_vjp_p!(blocks, S, length(param_syms)), String[])
end

"""The in-place vjp_p handed to ODEFunction. dgrad is FILLED (zeroed) each call, then
accumulated per block: dgrad[pidx] .+= s_col .* ∂r/∂θ (local ForwardDiff over the θ-explicit
kernel — no tape, no matrix). λ may arrive Dual-typed from the reverse pass: everything is
eltype-generic; only θ (from the parameter vector) carries ForwardDiff tags."""
function _make_param_vjp_p!(blocks::Vector{ParamVjpBlock}, S, nparam_expected::Int)
    return function vjp_p!(dgrad, λ, u, p, t)
        length(dgrad) == nparam_expected || error(
            "vjp_p: dgrad length $(length(dgrad)) ≠ flat param count $nparam_expected — " *
            "the problem's sys must be the SAME instance the blocks were materialized from")
        fill!(dgrad, zero(eltype(dgrad)))
        # Task 0 probe (2026-10-13, verified): the vjp_p callback receives p as a RAW
        # MTKParameters (λ and dgrad as SubArray views, dgrad length = flat param count) —
        # flatten BEFORE the kernels read it.
        pv = _parameter_vector(p)      # flat Vector (MTKParameters-aware), parameters(sys) order
        length(pv) == nparam_expected || error("vjp_p: p length mismatch ($(length(pv)) vs $nparam_expected)")
        s = S' * λ                     # s[col] = λ·ν_j, eltype follows λ (Dual-safe)
        for (col, b) in enumerate(blocks)
            θ0 = pv[b.pidx]
            g = ForwardDiff.gradient(θv -> begin
                    x = vcat(u[b.dep_cols], θv)
                    buf = Vector{promote_type(eltype(u), eltype(θv))}(undef, 1)
                    b.fn(buf, x, pv, t)   # ① pv, NOT p: kernels read p[j] positionally over
                    buf[1]                #    parameters(sys); raw MTKParameters p[j] reads the
                end, θ0)                 #    j-th COMPONENT (tunables blob), silently wrong (rev Q3)
            dgrad[b.pidx] .+= s[col] .* g
        end
        return nothing
    end
end
