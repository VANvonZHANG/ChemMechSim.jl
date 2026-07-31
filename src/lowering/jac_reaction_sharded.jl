using SparseArrays

struct ReactionShardedJacStats
    n_reactions::Int
    n_states::Int
    n_nonzeros::Int
    n_reaction_shards::Int
    n_compiled_derivative_functions::Int
    max_shard_reactions::Int
    max_shard_derivative_outputs::Int
end

function _state_name_index(sys)
    return Dict(String(ModelingToolkit.getname(s)) => i
                for (i, s) in enumerate(ModelingToolkit.unknowns(sys)))
end

function _same_sparsity_template(sys)
    J_sym = ModelingToolkit.calculate_jacobian(sys; sparse=true)
    return SparseArrays.similar(J_sym, Float64)
end

_reaction_row_sids(rx::ReactionData) = union(keys(rx.reactants), keys(rx.products))

function _reaction_dependency_sids(rx::ReactionData, mech::Mechanism)
    deps = Set{SpeciesID}(keys(rx.reactants))
    # Reversible rates also depend on product concentrations (reverse mass-action / K_c).
    rx.reverse_policy isa Irreversible || union!(deps, keys(rx.products))
    # Third-body and falloff rates depend on every species concentration via [M]_eff.
    (rx.kinetics isa ThirdBodyArrhenius || rx.kinetics isa AbstractFalloff) &&
        union!(deps, (sp.id for sp in mech.species))
    return deps
end

function _reaction_extra_dependency_states(rx::ReactionData, state_idx)
    rx.kinetics isa PlogRate || return Int[]
    haskey(state_idx, "P") ||
        throw(ArgumentError("build_reaction_sharded_jac: PlogRate reaction requires pressure " *
                            "state P in the lowered fixedT system."))
    return [state_idx["P"]]
end

"State-column indices a reaction's rate depends on. Under adiabatic, T is a state and every
 rate's T-dependence (Arrhenius) plus the energy contribution's T-dependence make T a column;
 PLOG adds P. These are the columns the shard differentiates against."
function _reaction_dependency_state_indices(rx::ReactionData, mech::Mechanism, state_idx,
                                            sid_to_idx, is_adiabatic::Bool=false)
    cols = Int[sid_to_idx[sid] for sid in _reaction_dependency_sids(rx, mech)]
    append!(cols, _reaction_extra_dependency_states(rx, state_idx))
    is_adiabatic && haskey(state_idx, "T") && push!(cols, state_idx["T"])
    return cols
end

_reaction_sharded_supported_kinetics(kin) =
    kin isa ElementaryArrhenius || kin isa ThirdBodyArrhenius || kin isa PlogRate ||
    kin isa LindemannFalloff || kin isa TroeFalloff

"Reverse policies the reaction-sharded Jacobian can route through the lowering `_net_rate`.
 ExplicitReverse reuses the policy's own rate law; ThermoReverse reuses the opaque keq node."
_reaction_sharded_supported_reverse(policy::ReverseRatePolicy) =
    policy isa Irreversible || policy isa ExplicitReverse || policy isa ThermoReverse

_reaction_sharded_supports(rx::ReactionData) =
    _reaction_sharded_supported_kinetics(rx.kinetics) &&
    _reaction_sharded_supported_reverse(rx.reverse_policy)

function _assert_reaction_sharded_supported(mech::Mechanism)
    for (i, rx) in enumerate(mech.reactions)
        _reaction_sharded_supports(rx) && continue
        throw(ArgumentError(
            "build_reaction_sharded_jac: unsupported reaction $i; supports " *
            "ElementaryArrhenius, ThirdBodyArrhenius, PlogRate, LindemannFalloff, and " *
            "TroeFalloff kinetics with Irreversible, ExplicitReverse, or ThermoReverse policy " *
            "in fixedT mode. Got kinetics=$(typeof(rx.kinetics)), " *
            "reverse_policy=$(typeof(rx.reverse_policy))."))
    end
    return nothing
end

"Config guard: the sharded Jacobian supports concentration-basis, ideal-gas, constant-volume
 mechanisms in either :isothermal (fixedT — T is a parameter) or :adiabatic (T is a state)
 energy regimes."
_reaction_sharded_config_ok(config::MechanismConfig) =
    config.state_basis === :concentration &&
    config.constraint === :constant_volume &&
    config.eos === :ideal_gas &&
    config.energy in (:isothermal, :adiabatic)

function _assert_reaction_sharded_config(config::MechanismConfig)
    _reaction_sharded_config_ok(config) ||
        throw(ArgumentError(
            "build_reaction_sharded_jac: unsupported config; supports concentration, ideal-gas, " *
            "constant-volume :isothermal or :adiabatic. Got energy=$(config.energy) " *
            "constraint=$(config.constraint) eos=$(config.eos) basis=$(config.state_basis)."))
    return nothing
end

"True iff the mechanism + config are fully supported by the reaction-sharded Jacobian, so the
 :auto strategy can prefer it over the shared_cse symbolic path (whose full Jacobian codegen
 does not scale past ~100 species). Unsupported kinetics (Chebyshev/SRI/...) fall back to
 shared_cse."
_reaction_sharded_supports_mechconfig(mech::Mechanism, config::MechanismConfig) =
    _reaction_sharded_config_ok(config) && all(_reaction_sharded_supports, mech.reactions)

function _species_id_to_state_index(mech::Mechanism, sys)
    state_idx = _state_name_index(sys)
    sid_to_idx = Dict{SpeciesID,Int}()
    for sp in mech.species
        name = String(sp.name)
        haskey(state_idx, name) ||
            throw(ArgumentError("build_reaction_sharded_jac: species $(sp.id) ($(sp.name)) " *
                                "is not a state in the lowered fixedT system."))
        sid_to_idx[sp.id] = state_idx[name]
    end
    return sid_to_idx
end

"Build the Jacobian sparsity template. fixedT: species rows + P row over reaction dependency
 columns. adiabatic: additionally the T (energy) and P rows are DENSE in all species (cvsum
 couples every concentration into dT/dt, and dP/dt couples to dT/dt), and T is a dependency
 column for every rate. PLOG adds P as a dependency column."
function _reaction_sharded_sparsity_template(mech::Mechanism, sys, is_adiabatic::Bool)
    state_idx = _state_name_index(sys)
    sid_to_idx = _species_id_to_state_index(mech, sys)
    n = length(ModelingToolkit.unknowns(sys))
    pressure_idx = get(state_idx, "P", nothing)
    T_idx = get(state_idx, "T", nothing)
    species_state_idxs = collect(values(sid_to_idx))
    rows = Int[]
    cols = Int[]

    for rx in mech.reactions
        row_idxs = [sid_to_idx[sid] for sid in _reaction_row_sids(rx)]
        pressure_idx === nothing || push!(row_idxs, pressure_idx)
        col_idxs = Int[sid_to_idx[sid] for sid in _reaction_dependency_sids(rx, mech)]
        append!(col_idxs, _reaction_extra_dependency_states(rx, state_idx))
        is_adiabatic && T_idx !== nothing && push!(col_idxs, T_idx)
        for col in col_idxs
            for row in row_idxs
                push!(rows, row); push!(cols, col)
            end
        end
    end

    if is_adiabatic
        # Energy (T) and pressure (P) rows are dense in species (+ T, P columns).
        for prow in (T_idx, pressure_idx)
            prow === nothing && continue
            for col in species_state_idxs
                push!(rows, prow); push!(cols, col)
            end
            T_idx === nothing || (push!(rows, prow); push!(cols, T_idx))
            pressure_idx === nothing || (push!(rows, prow); push!(cols, pressure_idx))
        end
    end

    J = SparseArrays.sparse(rows, cols, ones(Float64, length(rows)), n, n)
    fill!(J.nzval, 0.0)
    return J
end

_netstoich(rx::ReactionData, sid::SpeciesID) =
    get(rx.products, sid, 0.0) - get(rx.reactants, sid, 0.0)

"Build a (row, col) → nzval-slot lookup from a sparse CSC template, so shard assembly can
 scatter each derivative directly into its slot in O(1) (no per-write column scan — essential
 for dense-column large mechanisms where FBDF calls jac! every step)."
function _slot_map(J::SparseArrays.SparseMatrixCSC)
    m = Dict{Tuple{Int,Int},Int}()
    for col in 1:size(J, 2)
        for slot in J.colptr[col]:(J.colptr[col + 1] - 1)
            m[(J.rowval[slot], col)] = slot
        end
    end
    return m
end

function _parameter_vector(p)
    try
        candidate = p[1]
        candidate isa AbstractVector && return candidate
    catch
    end
    return p
end

function _parameter_value_by_index(p, idx::Int)
    try
        return Float64(p[1][idx])
    catch
    end
    try
        return Float64(p[idx])
    catch
    end
    return nothing
end

function _temperature_value(sys, u, p, state_idx)
    haskey(state_idx, "T") && return Float64(u[state_idx["T"]])
    params = ModelingToolkit.parameters(sys)
    Tidx = findfirst(par -> String(ModelingToolkit.getname(par)) == "T", params)
    if Tidx !== nothing
        pval = _parameter_value_by_index(p, Tidx)
        pval === nothing || return pval
        default = ModelingToolkit.getdefault(params[Tidx])
        default === nothing || return Float64(default)
    end
    return 300.0
end

function _species_id_to_state_symbol(mech::Mechanism, sys)
    state_syms = ModelingToolkit.unknowns(sys)
    state_by_name = Dict(String(ModelingToolkit.getname(s)) => s for s in state_syms)
    cvar = Dict{SpeciesID,Any}()
    for sp in mech.species
        name = String(sp.name)
        haskey(state_by_name, name) ||
            throw(ArgumentError("build_reaction_sharded_jac: species $(sp.id) ($(sp.name)) " *
                                "is not a state in the lowered fixedT system."))
        cvar[sp.id] = state_by_name[name]
    end
    return cvar
end

function _fixedT_parameter_symbol(sys)
    param_syms = ModelingToolkit.parameters(sys)
    Tidx = findfirst(par -> String(ModelingToolkit.getname(par)) == "T", param_syms)
    Tidx === nothing &&
        throw(ArgumentError("build_reaction_sharded_jac: fixedT lowered system has no T parameter."))
    return param_syms[Tidx]
end

function _reaction_sharded_direct_meff(ctx::RateCtx,
                                       efficiencies::Dict{SpeciesID,Float64})
    meff = 0.0
    for sp in ctx.mech.species
        alpha = get(efficiencies, sp.id, 1.0)
        meff += alpha * ctx.cvar[sp.id]
    end
    return meff
end

function _reaction_sharded_symbolic_kf(kin::ElementaryArrhenius, ctx::RateCtx)
    return symbolic_kf(kin, ctx)
end

function _reaction_sharded_symbolic_kf(kin::ThirdBodyArrhenius, ctx::RateCtx)
    A = rate_param(Symbol("k_", ctx.j, "_A"), kin.base.A,
                   _k_unit(ctx.order + 1, kin.base.b))
    base = iszero(kin.base.b) && iszero(kin.base.Ea) ? A :
           iszero(kin.base.Ea) ? A * ctx.T^kin.base.b :
           _arrhenius_body(A, kin.base.b, _kparam(ctx, "", kin.base.Ea), ctx.T)
    return base * _reaction_sharded_direct_meff(ctx, kin.efficiencies)
end

function _reaction_sharded_symbolic_kf(kin::PlogRate, ctx::RateCtx)
    return symbolic_kf(kin, ctx)
end

"Troe/Lindemann falloff: reuse the lowering symbolic_kf (the Troe/Lindemann formula + the
 [M]_eff term). The M_eff_j(t) variables it creates via _meff are inlined into the rate
 expression by _reaction_rate_expr (no formula duplication here, no M_eff free variable)."
_reaction_sharded_symbolic_kf(kin::AbstractFalloff, ctx::RateCtx) = symbolic_kf(kin, ctx)

"True for rate laws whose forward rate couples every species via [M_eff] (third-body, falloff).
 These are the reactions that do not scale under per-species symbolic differentiation — Path M
 keeps [M_eff] as a single symbolic shard input and expands the dense coupling at runtime."
_reaction_uses_meff(rx::ReactionData) =
    rx.kinetics isa ThirdBodyArrhenius || rx.kinetics isa AbstractFalloff

"Efficiencies dict (species id → enhanced collision efficiency) for a third-body/falloff rate law."
_meff_efficiencies(kin::ThirdBodyArrhenius) = kin.efficiencies
_meff_efficiencies(kin::AbstractFalloff) = kin.efficiencies

"Like _reaction_rate_expr, but for [M_eff]-coupled rates it KEEPS the M_eff_j symbol in the
 expression (does NOT substitute Σα·c) and returns the symbol + efficiencies so the caller can
 differentiate w.r.t. M_eff (few outputs) and expand to all species at runtime. For non-M_eff
 rates it returns (rate_expr, nothing, nothing) identical to _reaction_rate_expr.

 For Irreversible third-body/falloff, symbolic_kf routes through _meff which creates M_eff_j and
 pushes M_eff_j ~ Σα·c into meff_eqs; we KEEP M_eff_j (do not substitute). For non-M_eff rates
 (Elementary/PLOG) meff_eqs is empty and this is identical to _reaction_rate_expr. The old
 _reaction_sharded_direct_meff inline path is no longer used here."
function _reaction_rate_expr_with_meff(rx::ReactionData, mech::Mechanism, sys,
                                        config::MechanismConfig, j::Int)
    cvar = _species_id_to_state_symbol(mech, sys)
    state_syms = ModelingToolkit.unknowns(sys)
    state_by_name = Dict(String(ModelingToolkit.getname(s)) => s for s in state_syms)
    P = get(state_by_name, "P", nothing)
    T = haskey(state_by_name, "T") ? state_by_name["T"] : _fixedT_parameter_symbol(sys)
    tcx = make_thermo_ctx(T)
    meff_eqs = Any[]
    ctx = RateCtx(mech, cvar, T, j, sum(values(rx.reactants)),
                  tcx.R, tcx.P_std, tcx.coeff_cache, P, meff_eqs)
    rate_expr = rx.reverse_policy isa Irreversible ?
        symbolic_kf(rx.kinetics, ctx) * _mass_action(rx.reactants, cvar) :
        _net_rate(rx, mech, cvar, T, j, ctx)
    if isempty(meff_eqs)
        return (rate_expr, nothing, nothing)
    end
    @assert length(meff_eqs) == 1 "expected one M_eff eq per reaction, got $(length(meff_eqs))"
    meff_sym = meff_eqs[1].lhs
    effs = _meff_efficiencies(rx.kinetics)
    return (rate_expr, meff_sym, effs)
end

"Symbolic net rate for one reaction, reusing the lowering protocol (design §5).
 Irreversible: _reaction_sharded_symbolic_kf · mass_action(reactants). Reversible: routes
 through thermo.jl's `_net_rate`, which forms the reverse rate from the policy (ExplicitReverse
 own rate law, or the opaque keq call node) WITHOUT duplicating any rate-law formula here."
function _reaction_rate_expr(rx::ReactionData, mech::Mechanism, sys,
                             config::MechanismConfig, j::Int)
    cvar = _species_id_to_state_symbol(mech, sys)
    state_syms = ModelingToolkit.unknowns(sys)
    state_by_name = Dict(String(ModelingToolkit.getname(s)) => s for s in state_syms)
    P = get(state_by_name, "P", nothing)
    T = haskey(state_by_name, "T") ? state_by_name["T"] : _fixedT_parameter_symbol(sys)
    tcx = make_thermo_ctx(T)
    meff_eqs = Any[]
    ctx = RateCtx(mech, cvar, T, j, sum(values(rx.reactants)),
                  tcx.R, tcx.P_std, tcx.coeff_cache, P, meff_eqs)
    rate_expr = rx.reverse_policy isa Irreversible ?
        _reaction_sharded_symbolic_kf(rx.kinetics, ctx) * _mass_action(rx.reactants, cvar) :
        _net_rate(rx, mech, cvar, T, j, ctx)
    # Falloff forward rates (symbolic_kf) introduce M_eff_j(t) algebraic variables via _meff.
    # The sharded path runs no MTK tearing to eliminate them, so inline each M_eff_j → its
    # Σα·c sum here (reuses the _meff expression; no rate-law formula duplication, and no
    # M_eff free variable reaches build_function). Direct third-body kf uses
    # _reaction_sharded_direct_meff (no M_eff_j), so meff_eqs stays empty for it.
    if !isempty(meff_eqs)
        rate_expr = ModelingToolkit.substitute(
            rate_expr, Dict(eq.lhs => eq.rhs for eq in meff_eqs))
    end
    return rate_expr
end

"One compiled derivative function for a batch (shard) of reactions. Differentiates each
 reaction's net rate w.r.t. ONLY its dependency columns (small expressions), not every state.
 `fn` is the IIP build_function output (writes into `buf`); `buf` is reused across jac! calls
 to avoid per-call allocation. `columns_by_rx`/`output_offsets` map local reaction → its
 dependency columns and the slice of `buf` holding those derivatives. Under adiabatic,
 `rate_offsets` additionally marks each reaction's rate-value output (consumed by the energy/
 pressure row assembly)."
struct ReactionDerivativeShard
    rx_indices::Vector{Int}
    columns_by_rx::Vector{Vector{Int}}
    output_offsets::Vector{UnitRange{Int}}
    rate_offsets::Vector{Int}
    fn::Any
    buf::Vector{Float64}
end

function _reaction_derivative_shard(mech::Mechanism, sys, config::MechanismConfig,
                                    rx_indices::Vector{Int}, dep_cols_by_rx::Vector{Vector{Int}},
                                    is_adiabatic::Bool)
    state_syms = ModelingToolkit.unknowns(sys)
    param_syms = ModelingToolkit.parameters(sys)
    outputs = Any[]
    ranges = UnitRange{Int}[]
    rate_offsets = Int[]
    for (lr, j) in pairs(rx_indices)
        rx = mech.reactions[j]
        rate_expr = _reaction_rate_expr(rx, mech, sys, config, j)
        if is_adiabatic
            push!(rate_offsets, length(outputs) + 1)
            push!(outputs, rate_expr)
        end
        first_idx = length(outputs) + 1
        for col in dep_cols_by_rx[lr]
            push!(outputs, ModelingToolkit.expand_derivatives(
                ModelingToolkit.Differential(state_syms[col])(rate_expr)))
        end
        push!(ranges, first_idx:length(outputs))
    end
    fn = ModelingToolkit.build_function(
        outputs, state_syms, param_syms, [ModelingToolkit.t_nounits];
        expression=Val{false}, wrap_gfw=Val{false}, cse=true)[2]
    buf = Vector{Float64}(undef, length(outputs))
    return ReactionDerivativeShard(rx_indices, dep_cols_by_rx, ranges, rate_offsets, fn, buf)
end

"Precomputed O(1) scatter plan for one shard: for each derivative output, the exact nzval
 slots it accumulates into and the constant coefficient. Species-row writes carry the net
 stoichiometric coefficient; pressure-row writes carry Δν (multiplied by R·T at runtime)."
struct ReactionShardSlotMap
    slots::Vector{Int}
    outidx::Vector{Int}
    coefs::Vector{Float64}
    pressure_slots::Vector{Int}
    pressure_outidx::Vector{Int}
    pressure_coefs::Vector{Float64}
end

function _build_shard_slotmap(mech::Mechanism, shard::ReactionDerivativeShard,
                              sid_to_idx, pressure_idx, slotmap, is_adiabatic::Bool)
    slots = Int[]; outidx = Int[]; coefs = Float64[]
    pressure_slots = Int[]; pressure_outidx = Int[]; pressure_coefs = Float64[]
    for (lr, j) in pairs(shard.rx_indices)
        rx = mech.reactions[j]
        row_sids = union(keys(rx.reactants), keys(rx.products))
        dnu = sum(values(rx.products)) - sum(values(rx.reactants))
        outbase = shard.output_offsets[lr]
        for (k, col) in enumerate(shard.columns_by_rx[lr])
            oi = outbase[k]
            for sid in row_sids
                net = _netstoich(rx, sid)
                iszero(net) && continue
                push!(slots, slotmap[(sid_to_idx[sid], col)]); push!(outidx, oi); push!(coefs, net)
            end
            # fixedT pressure row (dP/dt = R·T·ΣΔν·rate). Adiabatic assembles P row at runtime.
            if !is_adiabatic && pressure_idx !== nothing && !iszero(dnu)
                push!(pressure_slots, slotmap[(pressure_idx, col)])
                push!(pressure_outidx, oi); push!(pressure_coefs, dnu)
            end
        end
    end
    return ReactionShardSlotMap(slots, outidx, coefs, pressure_slots, pressure_outidx, pressure_coefs)
end

# —— Path M: [M_eff]-symbol shard (third-body/falloff) ——————————————————————
# One derivative function for a shard of [M_eff]-coupled reactions. Differentiates each rate
# w.r.t. ONLY {reactants, M_eff_j, T} (few outputs → small generated function → LLVM JITs fast).
# `meff_syms` are appended to the state inputs; at runtime the caller passes [u; meff_values].
# Layout per local reaction lr:
#   [rate (if adiabatic)] ++ [∂rate/∂col for col in columns_by_rx[lr]]
# where columns_by_rx[lr] = [reactant_state_cols..., meff_col(nstates+lr), T_col(if adiabatic)]
struct MeffShard
    rx_indices::Vector{Int}              # global reaction indices in this shard
    meff_syms::Vector{Any}               # M_eff_j symbols, one per local reaction (combined-input order)
    columns_by_rx::Vector{Vector{Int}}   # combined-input column indices differentiated against
    output_offsets::Vector{UnitRange{Int}}  # slice of buf per local reaction (its outputs)
    rate_offsets::Vector{Int}            # buf index of each local reaction's rate value (0 if not adiabatic)
    fn::Any
    buf::Vector{Float64}
end

function _build_meff_shard(mech::Mechanism, sys, config::MechanismConfig,
                           rx_indices::Vector{Int}, is_adiabatic::Bool,
                           sid_to_idx::Dict{SpeciesID,Int})
    state_syms = ModelingToolkit.unknowns(sys)
    param_syms = ModelingToolkit.parameters(sys)
    nstates = length(state_syms)
    T_idx = findfirst(s -> String(ModelingToolkit.getname(s)) == "T", state_syms)
    meff_syms = Any[]
    outputs = Any[]
    ranges = UnitRange{Int}[]
    rate_offsets = Int[]
    columns_by_rx = Vector{Int}[]
    for (lr, j) in pairs(rx_indices)
        rx = mech.reactions[j]
        rate_expr, meff_sym, _ = _reaction_rate_expr_with_meff(rx, mech, sys, config, j)
        @assert meff_sym !== nothing "MeffShard built for a non-M_eff reaction $j"
        # Register this reaction's M_eff symbol. In the combined input [state_syms; meff_syms],
        # it occupies position nstates + lr (1-based local index).
        push!(meff_syms, meff_sym)
        meff_col = nstates + lr          # this reaction's M_eff column in combined input
        # Per-reaction output layout: [rate (if adiabatic)] ++ [deriv for each col]
        if is_adiabatic
            push!(rate_offsets, length(outputs) + 1)
            push!(outputs, rate_expr)
        else
            push!(rate_offsets, 0)
        end
        first_idx = length(outputs) + 1
        cols = Int[]
        # Reactant state columns (direct mass-action derivatives, M_eff held symbolic)
        for sid in keys(rx.reactants)
            push!(cols, sid_to_idx[sid])
        end
        # Reversible rates also depend on product concentrations (reverse mass-action / K_c).
        # Path S includes these via _reaction_dependency_sids; Path M must too, or the reverse
        # rate's d(rate)/d(c_product) is never differentiated.
        if !(rx.reverse_policy isa Irreversible)
            for sid in keys(rx.products)
                push!(cols, sid_to_idx[sid])
            end
        end
        # M_eff column
        push!(cols, meff_col)
        # T column (adiabatic only — T is a state; third-body/falloff rates are T-dependent)
        T_idx === nothing || push!(cols, T_idx)
        combined_inputs = [state_syms; meff_syms]
        for col in cols
            push!(outputs, ModelingToolkit.expand_derivatives(
                ModelingToolkit.Differential(combined_inputs[col])(rate_expr)))
        end
        push!(ranges, first_idx:length(outputs))
        push!(columns_by_rx, cols)
    end
    combined_inputs = [state_syms; meff_syms]
    fn = ModelingToolkit.build_function(
        outputs, combined_inputs, param_syms, [ModelingToolkit.t_nounits];
        expression=Val{false}, wrap_gfw=Val{false}, cse=true)[2]
    buf = Vector{Float64}(undef, length(outputs))
    return MeffShard(rx_indices, meff_syms, columns_by_rx, ranges, rate_offsets, fn, buf)
end

"Runtime expansion plan for one M_eff reaction: precomputed nzval slots for (species_row, col)
 over ALL species columns (the [M_eff] coupling is dense), the α_m vector, the reactant set
 (with their direct-derivative buf positions), and buf positions of the M_eff derivative + rate.
 The chain rule: ∂(ν·rate)/∂c_m = ν·[(∂rate/∂[M_eff])·α_m + Σ_reactants (∂rate/∂c_r)·δ_{rm}]."
struct MeffReactionPlan
    rx_index::Int
    row_sids::Vector{SpeciesID}                 # species rows (reactants ∪ products) with nonzero netstoich
    netstoich::Vector{Float64}                  # parallel to row_sids
    slots::Matrix{Int}                          # slots[row_k, species_m] = nzval slot for (row_state, species_state)
    alpha::Vector{Float64}                      # α_m for each species position m (default 1.0)
    meff_deriv_pos::Int                         # buf position of ∂rate/∂[M_eff]
    direct_species_positions::Vector{Tuple{Int,Int}}  # (species_position_m, buf_position_of_direct_derivative) — reactants AND products (for reversible M_eff)
    rate_pos::Int                               # buf position of rate (0 if not adiabatic)
    # Adiabatic energy/pressure row coupling: buf positions of T-column derivative (0 if no T col)
    t_deriv_pos::Int
    # fixedT pressure row: Δν for dP/dt = R·T·ΣΔν·rate coupling
    dnu::Float64
end

function _build_meff_plan(mech::Mechanism, shard::MeffShard, lr::Int,
                          sid_to_idx, slotmap, nstates::Int, T_idx)
    j = shard.rx_indices[lr]
    rx = mech.reactions[j]
    sp_state_idx = [sid_to_idx[sp.id] for sp in mech.species]
    effs = _meff_efficiencies(rx.kinetics)
    alpha = [get(effs, sp.id, 1.0) for sp in mech.species]
    row_sids = SpeciesID[]; netstoich = Float64[]
    for sid in union(keys(rx.reactants), keys(rx.products))
        net = _netstoich(rx, sid)
        iszero(net) && continue
        push!(row_sids, sid); push!(netstoich, net)
    end
    slots = zeros(Int, length(row_sids), length(mech.species))
    for (rk, sid) in enumerate(row_sids)
        rcol = sid_to_idx[sid]
        for m in eachindex(mech.species)
            slots[rk, m] = slotmap[(rcol, sp_state_idx[m])]
        end
    end
    cols = shard.columns_by_rx[lr]
    outbase = shard.output_offsets[lr]
    meff_col_expected = nstates + lr           # this reaction's M_eff column in combined input
    meff_deriv_pos = 0
    direct_species_positions = Tuple{Int,Int}[]
    t_deriv_pos = 0
    for (k, col) in enumerate(cols)
        buf_pos = outbase[k]
        if col == meff_col_expected
            meff_deriv_pos = buf_pos
        elseif T_idx !== nothing && col == T_idx
            t_deriv_pos = buf_pos
        else
            for (m, sp) in enumerate(mech.species)
                if sid_to_idx[sp.id] == col
                    push!(direct_species_positions, (m, buf_pos))
                    break
                end
            end
        end
    end
    rate_pos = shard.rate_offsets[lr]
    dnu = sum(values(rx.products)) - sum(values(rx.reactants))
    return MeffReactionPlan(j, row_sids, netstoich, slots, alpha,
                            meff_deriv_pos, direct_species_positions, rate_pos, t_deriv_pos, dnu)
end

"Replace non-finite shard outputs with 0. A non-finite rate or derivative is a symbolic
 artifact (0·NaN or 0·Inf) that arises only when a reactant is absent — i.e. the reaction is
 inactive, so the true Jacobian contribution is 0. Sanitizing keeps the Jacobian finite so the
 ODE solver's Newton iteration converges; it never alters a nonzero physical value (the value
 tests all run at nonzero reactant concentration and are unaffected)."
@inline function _sanitize_shard_buf!(buf)
    @inbounds for i in eachindex(buf)
        isfinite(buf[i]) || (buf[i] = 0.0)
    end
    return buf
end

function build_reaction_sharded_jac(mech::Mechanism;
                                    config::MechanismConfig=MechanismConfig(),
                                    checks::Bool=false,
                                    reaction_shard_size::Int=100,
                                    return_stats::Bool=false,
                                    sys=nothing)
    reaction_shard_size > 0 || throw(ArgumentError("reaction_shard_size must be positive"))
    _assert_reaction_sharded_config(config)
    _assert_reaction_sharded_supported(mech)
    is_adiabatic = config.energy === :adiabatic
    # The shard functions are generated against this system's parameter ordering, so `sys`
    # MUST be the same system object that produced the runtime parameter vector `prob.p`
    # (the ODEProblem's p). ChemPhaseSystem lowering orders parameters non-deterministically
    # (hashed-container iteration), so reconstructing it here would desync the shard's
    # parameter layout from `prob.p` and silently misread every rate parameter. When called
    # from build_problem we pass the problem's own `sys`; standalone callers may omit it and
    # get a freshly-constructed system (whose param order then must match whatever p they use).
    if sys === nothing
        phase = ChemPhaseSystem(mech; config=config, checks=checks)
        sys = extract_system(phase)
    end
    J_proto = _reaction_sharded_sparsity_template(mech, sys, is_adiabatic)
    state_idx = _state_name_index(sys)
    sid_to_idx = _species_id_to_state_index(mech, sys)
    pressure_idx = get(state_idx, "P", nothing)
    T_idx = get(state_idx, "T", nothing)
    slotmap = _slot_map(J_proto)

    # Partition reactions into Path S (simple: Elementary/PLOG — few deps, unchanged shard path)
    # and Path M (M_eff-coupled: third-body/falloff — keep [M_eff] symbolic, expand at runtime).
    simple_idxs = [j for (j, rx) in enumerate(mech.reactions) if !_reaction_uses_meff(rx)]
    meff_idxs   = [j for (j, rx) in enumerate(mech.reactions) if  _reaction_uses_meff(rx)]

    nrx = length(mech.reactions)
    nstates = length(ModelingToolkit.unknowns(sys))
    nsp = length(mech.species)

    # ---- Path S: simple reactions (existing shard + slotmap machinery) ----
    # Dependency columns per simple reaction (only its direct deps; no all-species branch).
    dep_cols_simple = [_reaction_dependency_state_indices(mech.reactions[j], mech, state_idx,
                                                          sid_to_idx, is_adiabatic)
                       for j in simple_idxs]
    rx_groups_s = [collect(i:min(i + reaction_shard_size - 1, length(simple_idxs)))
                   for i in 1:reaction_shard_size:length(simple_idxs)]
    simple_shards = ReactionDerivativeShard[]
    simple_slotmaps = ReactionShardSlotMap[]
    for group in rx_groups_s
        group_global = simple_idxs[group]
        group_dep_cols = [dep_cols_simple[g] for g in group]
        shard = _reaction_derivative_shard(mech, sys, config, group_global, group_dep_cols, is_adiabatic)
        sm = _build_shard_slotmap(mech, shard, sid_to_idx, pressure_idx, slotmap, is_adiabatic)
        push!(simple_shards, shard)
        push!(simple_slotmaps, sm)
    end

    # ---- Path M: [M_eff]-coupled reactions (new shard + dense scatter plan) ----
    rx_groups_m = [collect(i:min(i + reaction_shard_size - 1, length(meff_idxs)))
                   for i in 1:reaction_shard_size:length(meff_idxs)]
    meff_shards = MeffShard[]
    meff_plans = Vector{MeffReactionPlan}[]
    for group in rx_groups_m
        group_global = meff_idxs[group]
        shard = _build_meff_shard(mech, sys, config, group_global, is_adiabatic, sid_to_idx)
        plans = [_build_meff_plan(mech, shard, lr, sid_to_idx, slotmap, nstates, T_idx)
                 for lr in eachindex(group_global)]
        push!(meff_shards, shard)
        push!(meff_plans, plans)
    end

    is_adiabatic && T_idx === nothing &&
        error("build_reaction_sharded_jac: adiabatic config requires T as a state.")
    # Adiabatic-only precompute (energy-row assembly). Computed unconditionally; the fixedT
    # jac! branch never touches these, and the arrays are small.
    sid_to_pos = Dict(sp.id => m for (m, sp) in enumerate(mech.species))
    species_state_idx = [sid_to_idx[sp.id] for sp in mech.species]
    product_pos_nu = [[(sid_to_pos[sid], nu) for (sid, nu) in rx.products] for rx in mech.reactions]
    reactant_pos_nu = [[(sid_to_pos[sid], nu) for (sid, nu) in rx.reactants] for rx in mech.reactions]
    dnu_vec = [sum(values(rx.products)) - sum(values(rx.reactants)) for rx in mech.reactions]
    u_vec = Vector{Float64}(undef, nsp)
    cv_vec = Vector{Float64}(undef, nsp)
    dcv_dT_vec = Vector{Float64}(undef, nsp)
    delta_u_vec = Vector{Float64}(undef, nrx)
    delta_cv_vec = Vector{Float64}(undef, nrx)
    dG = Vector{Float64}(undef, nstates)   # ∂(ΣΔν·rate)/∂x  (pressure-row coupling)
    Pn = Vector{Float64}(undef, nstates)   # ∂(Σ rate·Δu)/∂x  (energy-row direct part)
    thermo = [sp.thermo for sp in mech.species]

    # Single jac! branching on energy regime. fixedT: species rows + the isothermal pressure
    # row (dP/dt = R·T·ΣΔν·rate). adiabatic: species rows (incl. T,P columns) + the energy (T)
    # and pressure (P) rows assembled in factored form.
    # Species rows are assembled from BOTH paths:
    #   Path S (simple): precomputed slot map scatter (per-derivative → nzval slot × netstoich coef).
    #   Path M (M_eff): dense runtime scatter via the chain rule
    #     ∂(ν·rate)/∂c_m = ν·[(∂rate/∂[M_eff])·α_m + Σ_reactants (∂rate/∂c_r)·δ_{rm}]
    # The fixedT pressure row and the adiabatic energy/pressure rows consume rate + derivatives
    # from BOTH paths.
    function jac!(J::SparseArrays.SparseMatrixCSC, u, p, t)
        fill!(J.nzval, 0.0)
        pvec = _parameter_vector(p)
        if !is_adiabatic
            T = _temperature_value(sys, u, p, state_idx)
            RT = R_GAS * T
            # ---- Path S (simple reactions) ----
            for (shard, sm) in zip(simple_shards, simple_slotmaps)
                shard.fn(shard.buf, u, pvec, t)
                _sanitize_shard_buf!(shard.buf)
                vals = shard.buf
                @inbounds for k in eachindex(sm.slots)
                    J.nzval[sm.slots[k]] += sm.coefs[k] * vals[sm.outidx[k]]
                end
                if pressure_idx !== nothing
                    @inbounds for k in eachindex(sm.pressure_slots)
                        J.nzval[sm.pressure_slots[k]] += (RT * sm.pressure_coefs[k]) * vals[sm.pressure_outidx[k]]
                    end
                end
            end
            # ---- Path M ([M_eff]-coupled reactions) ----
            for (shard, plans) in zip(meff_shards, meff_plans)
                # Compute M_eff values for this shard's local reactions: meff_vals[lr] = Σ_m α_m·c_m.
                meff_vals = Vector{Float64}(undef, length(plans))
                for lr in eachindex(plans)
                    plan = plans[lr]
                    s = 0.0
                    for m in eachindex(mech.species)
                        s += plan.alpha[m] * u[species_state_idx[m]]
                    end
                    meff_vals[lr] = s
                end
                shard.fn(shard.buf, [u; meff_vals], pvec, t)
                _sanitize_shard_buf!(shard.buf)
                vals = shard.buf
                for (lr, plan) in pairs(plans)
                    # Species rows: chain-rule dense expansion.
                    dR_dMeff = vals[plan.meff_deriv_pos]
                    @inbounds for rk in eachindex(plan.row_sids)
                        coef = plan.netstoich[rk]
                        for m in eachindex(mech.species)
                            J.nzval[plan.slots[rk, m]] += coef * dR_dMeff * plan.alpha[m]
                        end
                    end
                    @inbounds for (m, dpos) in plan.direct_species_positions
                        direct = vals[dpos]
                        for rk in eachindex(plan.row_sids)
                            J.nzval[plan.slots[rk, m]] += plan.netstoich[rk] * direct
                        end
                    end
                    # fixedT pressure row (dP/dt = R·T·ΣΔν·rate): chain-rule expansion.
                    if pressure_idx !== nothing && !iszero(plan.dnu)
                        prate = dR_dMeff   # reuse for pressure row: d(Δν·rate)/dc_m = Δν·dR_dMeff·α_m + ...
                        @inbounds for m in eachindex(mech.species)
                            # (pressure_idx, species_state_idx[m]) slot
                            J.nzval[slotmap[(pressure_idx, species_state_idx[m])]] +=
                                RT * plan.dnu * prate * plan.alpha[m]
                        end
                        @inbounds for (m, dpos) in plan.direct_species_positions
                            direct = vals[dpos]
                            J.nzval[slotmap[(pressure_idx, species_state_idx[m])]] +=
                                RT * plan.dnu * direct
                        end
                    end
                end
            end
            return nothing
        end
        # ---- adiabatic ----
        T = Float64(u[T_idx])
        cvsum = 0.0; dcvsum_dT = 0.0; csum = 0.0
        @inbounds for m in 1:nsp
            a = _nasa7_coeffs(thermo[m], T)
            cpR = a[1] + a[2]*T + a[3]*T^2 + a[4]*T^3 + a[5]*T^4
            cv_vec[m] = (cpR - 1) * R_GAS
            dcv_dT_vec[m] = R_GAS * (a[2] + 2*a[3]*T + 3*a[4]*T^2 + 4*a[5]*T^3)
            hRT = a[1] + a[2]*T/2 + a[3]*T^2/3 + a[4]*T^3/4 + a[5]*T^4/5 + a[6]/T
            u_vec[m] = (hRT - 1) * R_GAS * T
            cm = u[species_state_idx[m]]
            cvsum += cm * cv_vec[m]
            dcvsum_dT += cm * dcv_dT_vec[m]
            csum += cm
        end
        @inbounds for j in 1:nrx
            du = 0.0; dcv = 0.0
            for (pos, nu) in product_pos_nu[j]
                du += nu * u_vec[pos]; dcv += nu * cv_vec[pos]
            end
            for (pos, nu) in reactant_pos_nu[j]
                du -= nu * u_vec[pos]; dcv -= nu * cv_vec[pos]
            end
            delta_u_vec[j] = du; delta_cv_vec[j] = dcv
        end
        fill!(dG, 0.0); fill!(Pn, 0.0)
        S = 0.0; G = 0.0
        # ---- Path S (simple reactions): species rows + energy/pressure accumulators ----
        for (shard, sm) in zip(simple_shards, simple_slotmaps)
            shard.fn(shard.buf, u, pvec, t)
            _sanitize_shard_buf!(shard.buf)
            vals = shard.buf
            @inbounds for k in eachindex(sm.slots)
                J.nzval[sm.slots[k]] += sm.coefs[k] * vals[sm.outidx[k]]
            end
            @inbounds for (lr, j) in pairs(shard.rx_indices)
                rate = vals[shard.rate_offsets[lr]]
                du = delta_u_vec[j]; dcv = delta_cv_vec[j]; dnur = dnu_vec[j]
                S += rate * du
                G += dnur * rate
                outbase = shard.output_offsets[lr]
                for (k, col) in enumerate(shard.columns_by_rx[lr])
                    drate = vals[outbase[k]]
                    dG[col] += dnur * drate
                    Pn[col] += drate * du
                    col == T_idx && (Pn[col] += rate * dcv)   # dΔu/dT = Δcv term
                end
            end
        end
        # ---- Path M ([M_eff]-coupled reactions): species rows + energy/pressure accumulators ----
        for (shard, plans) in zip(meff_shards, meff_plans)
            meff_vals = Vector{Float64}(undef, length(plans))
            for lr in eachindex(plans)
                plan = plans[lr]
                s = 0.0
                for m in eachindex(mech.species)
                    s += plan.alpha[m] * u[species_state_idx[m]]
                end
                meff_vals[lr] = s
            end
            shard.fn(shard.buf, [u; meff_vals], pvec, t)
            _sanitize_shard_buf!(shard.buf)
            vals = shard.buf
            for (lr, plan) in pairs(plans)
                j = plan.rx_index
                du = delta_u_vec[j]; dcv = delta_cv_vec[j]; dnur = dnu_vec[j]
                # Species rows: chain-rule dense expansion.
                dR_dMeff = vals[plan.meff_deriv_pos]
                rate = plan.rate_pos > 0 ? vals[plan.rate_pos] : 0.0
                @inbounds for rk in eachindex(plan.row_sids)
                    coef = plan.netstoich[rk]
                    for m in eachindex(mech.species)
                        J.nzval[plan.slots[rk, m]] += coef * dR_dMeff * plan.alpha[m]
                    end
                end
                @inbounds for (m, dpos) in plan.direct_species_positions
                    direct = vals[dpos]
                    for rk in eachindex(plan.row_sids)
                        J.nzval[plan.slots[rk, m]] += plan.netstoich[rk] * direct
                    end
                end
                # T-column species-row entries: J[species_row, T] += nu_row * d(rate)/dT.
                # (The dG[T]/Pn[T] accumulators below already consume t_deriv_pos for the T/P rows;
                #  this scatter fills the species-row T-column slots the chain-rule + direct-species
                #  loops above do not reach.) Adiabatic-only — fixedT has no T state.
                if plan.t_deriv_pos > 0
                    drate_T = vals[plan.t_deriv_pos]
                    @inbounds for rk in eachindex(plan.row_sids)
                        J.nzval[slotmap[(sid_to_idx[plan.row_sids[rk]], T_idx)]] +=
                            plan.netstoich[rk] * drate_T
                    end
                end
                # Energy/pressure accumulators: fold rate + derivatives (same identities as Path S).
                # For species columns, dG[col] += Δν·∂rate/∂col and Pn[col] += ∂rate/∂col·Δu.
                # Path M's ∂rate/∂c_m = (∂rate/∂[M_eff])·α_m + direct; expand into the accumulators.
                S += rate * du
                G += dnur * rate
                # M_eff chain-rule contribution to dG/Pn for every species col.
                @inbounds for m in eachindex(mech.species)
                    col = species_state_idx[m]
                    drate_m = dR_dMeff * plan.alpha[m]
                    dG[col] += dnur * drate_m
                    Pn[col] += drate_m * du
                end
                # Reactant direct-derivative contribution to dG/Pn.
                @inbounds for (m, dpos) in plan.direct_species_positions
                    col = species_state_idx[m]
                    direct = vals[dpos]
                    dG[col] += dnur * direct
                    Pn[col] += direct * du
                end
                # T-column contribution: ∂rate/∂T (from the shard's T derivative) + Δcv term.
                if plan.t_deriv_pos > 0
                    drate_T = vals[plan.t_deriv_pos]
                    dG[T_idx] += dnur * drate_T
                    Pn[T_idx] += drate_T * du
                    Pn[T_idx] += rate * dcv   # dΔu/dT = Δcv term
                end
            end
        end
        inv_cv = 1.0 / cvsum
        inv_cv2 = inv_cv * inv_cv
        @inbounds begin
            # Energy (T) row: J[T,x] = -Pn[x]/cvsum + S·(∂cvsum/∂x)/cvsum²
            for m in 1:nsp
                col = species_state_idx[m]
                J.nzval[slotmap[(T_idx, col)]] = -Pn[col]*inv_cv + S*cv_vec[m]*inv_cv2
            end
            JTT = -Pn[T_idx]*inv_cv + S*dcvsum_dT*inv_cv2
            J.nzval[slotmap[(T_idx, T_idx)]] = JTT
            pressure_idx === nothing || (J.nzval[slotmap[(T_idx, pressure_idx)]] = -Pn[pressure_idx]*inv_cv)
            # Pressure (P) row: J[P,x] = R·(δ_Tx·G + T·dG[x] + δ_cx·dTdt + (Σc)·J[T,x])
            dTdt = -S * inv_cv
            for m in 1:nsp
                col = species_state_idx[m]
                JTc = -Pn[col]*inv_cv + S*cv_vec[m]*inv_cv2
                J.nzval[slotmap[(pressure_idx, col)]] = R_GAS*(T*dG[col] + dTdt + csum*JTc)
            end
            J.nzval[slotmap[(pressure_idx, T_idx)]] = R_GAS*(G + T*dG[T_idx] + csum*JTT)
            pressure_idx === nothing ||
                (J.nzval[slotmap[(pressure_idx, pressure_idx)]] = R_GAS*(T*dG[pressure_idx] + csum*(-Pn[pressure_idx]*inv_cv)))
        end
        return nothing
    end

    n_simple_shards = length(simple_shards)
    n_meff_shards = length(meff_shards)
    n_total_shards = n_simple_shards + n_meff_shards
    all_group_sizes = [length(g) for g in rx_groups_s]
    isempty(rx_groups_m) || append!(all_group_sizes, [length(g) for g in rx_groups_m])
    simple_max_outputs = maximum((sum(length(r) for r in shard.output_offsets)
                                  for shard in simple_shards); init=0)
    meff_max_outputs = maximum((sum(length(r) for r in shard.output_offsets)
                                for shard in meff_shards); init=0)
    stats = ReactionShardedJacStats(
        nrx,
        length(ModelingToolkit.unknowns(sys)),
        length(SparseArrays.nonzeros(J_proto)),
        n_total_shards,
        n_total_shards,
        maximum(all_group_sizes; init=0),
        max(simple_max_outputs, meff_max_outputs),
    )
    return return_stats ? (jac!, J_proto, stats) : (jac!, J_proto)
end
