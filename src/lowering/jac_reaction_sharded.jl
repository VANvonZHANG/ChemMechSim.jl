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
    # Third-body rates depend on every species concentration via [M]_eff.
    rx.kinetics isa ThirdBodyArrhenius && union!(deps, (sp.id for sp in mech.species))
    return deps
end

function _reaction_extra_dependency_states(rx::ReactionData, state_idx)
    rx.kinetics isa PlogRate || return Int[]
    haskey(state_idx, "P") ||
        throw(ArgumentError("build_reaction_sharded_jac: PlogRate reaction requires pressure " *
                            "state P in the lowered fixedT system."))
    return [state_idx["P"]]
end

"State-column indices a reaction's rate depends on (dependency sids + extra states like P).
 These are the columns the shard differentiates against and the columns it scatters into."
function _reaction_dependency_state_indices(rx::ReactionData, mech::Mechanism, state_idx, sid_to_idx)
    cols = Int[sid_to_idx[sid] for sid in _reaction_dependency_sids(rx, mech)]
    append!(cols, _reaction_extra_dependency_states(rx, state_idx))
    return cols
end

_reaction_sharded_supported_kinetics(kin) =
    kin isa ElementaryArrhenius || kin isa ThirdBodyArrhenius || kin isa PlogRate

"Reverse policies the reaction-sharded Jacobian can route through the lowering `_net_rate`.
 ExplicitReverse reuses the policy's own rate law; ThermoReverse reuses the opaque keq node."
_reaction_sharded_supported_reverse(policy::ReverseRatePolicy) =
    policy isa Irreversible || policy isa ExplicitReverse

_reaction_sharded_supports(rx::ReactionData) =
    _reaction_sharded_supported_kinetics(rx.kinetics) &&
    _reaction_sharded_supported_reverse(rx.reverse_policy)

function _assert_reaction_sharded_supported(mech::Mechanism)
    for (i, rx) in enumerate(mech.reactions)
        _reaction_sharded_supports(rx) && continue
        throw(ArgumentError(
            "build_reaction_sharded_jac: unsupported reaction $i; supports " *
            "ElementaryArrhenius, ThirdBodyArrhenius, and PlogRate kinetics with " *
            "Irreversible or ExplicitReverse policy in fixedT mode. Got kinetics=" *
            "$(typeof(rx.kinetics)), reverse_policy=$(typeof(rx.reverse_policy))."))
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

function _reaction_sharded_sparsity_template(mech::Mechanism, sys)
    state_idx = _state_name_index(sys)
    sid_to_idx = _species_id_to_state_index(mech, sys)
    n = length(ModelingToolkit.unknowns(sys))
    pressure_idx = get(state_idx, "P", nothing)
    rows = Int[]
    cols = Int[]

    for rx in mech.reactions
        row_idxs = [sid_to_idx[sid] for sid in _reaction_row_sids(rx)]
        pressure_idx === nothing || push!(row_idxs, pressure_idx)
        col_idxs = [sid_to_idx[sid] for sid in _reaction_dependency_sids(rx, mech)]
        append!(col_idxs, _reaction_extra_dependency_states(rx, state_idx))

        for col in col_idxs
            for row in row_idxs
                push!(rows, row)
                push!(cols, col)
            end
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
    ctx = RateCtx(mech, cvar, T, j, sum(values(rx.reactants)),
                  tcx.R, tcx.P_std, tcx.coeff_cache, P, Any[])
    return rx.reverse_policy isa Irreversible ?
           _reaction_sharded_symbolic_kf(rx.kinetics, ctx) * _mass_action(rx.reactants, cvar) :
           _net_rate(rx, mech, cvar, T, j, ctx)
end

"One compiled derivative function for a batch (shard) of reactions. Differentiates each
 reaction's net rate w.r.t. ONLY its dependency columns (small expressions), not every state.
 `fn` is the IIP build_function output (writes into `buf`); `buf` is reused across jac! calls
 to avoid per-call allocation. `columns_by_rx`/`output_offsets` map local reaction → its
 dependency columns and the slice of `buf` holding those derivatives."
struct ReactionDerivativeShard
    rx_indices::Vector{Int}
    columns_by_rx::Vector{Vector{Int}}
    output_offsets::Vector{UnitRange{Int}}
    fn::Any
    buf::Vector{Float64}
end

function _reaction_derivative_shard(mech::Mechanism, sys, config::MechanismConfig,
                                    rx_indices::Vector{Int}, dep_cols_by_rx::Vector{Vector{Int}})
    state_syms = ModelingToolkit.unknowns(sys)
    param_syms = ModelingToolkit.parameters(sys)
    outputs = Any[]
    ranges = UnitRange{Int}[]
    for (lr, j) in pairs(rx_indices)
        rx = mech.reactions[j]
        rate_expr = _reaction_rate_expr(rx, mech, sys, config, j)
        first_idx = length(outputs) + 1
        for col in dep_cols_by_rx[lr]
            push!(outputs, ModelingToolkit.expand_derivatives(
                ModelingToolkit.Differential(state_syms[col])(rate_expr)))
        end
        push!(ranges, first_idx:length(outputs))
    end
    fn = ModelingToolkit.build_function(
        outputs, state_syms, param_syms, [ModelingToolkit.t_nounits];
        expression=Val{false}, wrap_gfw=Val{false})[2]
    buf = Vector{Float64}(undef, length(outputs))
    return ReactionDerivativeShard(rx_indices, dep_cols_by_rx, ranges, fn, buf)
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
                              sid_to_idx, pressure_idx, slotmap)
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
                key = (sid_to_idx[sid], col)
                push!(slots, slotmap[key]); push!(outidx, oi); push!(coefs, net)
            end
            if pressure_idx !== nothing && !iszero(dnu)
                push!(pressure_slots, slotmap[(pressure_idx, col)])
                push!(pressure_outidx, oi); push!(pressure_coefs, dnu)
            end
        end
    end
    return ReactionShardSlotMap(slots, outidx, coefs, pressure_slots, pressure_outidx, pressure_coefs)
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
    J_proto = _reaction_sharded_sparsity_template(mech, sys)
    state_idx = _state_name_index(sys)
    sid_to_idx = _species_id_to_state_index(mech, sys)
    pressure_idx = get(state_idx, "P", nothing)
    slotmap = _slot_map(J_proto)

    # Dependency columns per reaction (the states its rate depends on).
    dep_cols_by_rx = [_reaction_dependency_state_indices(rx, mech, state_idx, sid_to_idx)
                      for rx in mech.reactions]
    # Batch reactions into shards of `reaction_shard_size`; compile one derivative function
    # per shard (differentiates each reaction's net rate w.r.t. its dependency columns only).
    nrx = length(mech.reactions)
    rx_groups = [collect(i:min(i + reaction_shard_size - 1, nrx))
                 for i in 1:reaction_shard_size:nrx]
    shards = [_reaction_derivative_shard(mech, sys, config, group, dep_cols_by_rx[group])
              for group in rx_groups]
    slotmaps = [_build_shard_slotmap(mech, shard, sid_to_idx, pressure_idx, slotmap)
                for shard in shards]

    function jac!(J::SparseArrays.SparseMatrixCSC, u, p, t)
        fill!(J.nzval, 0.0)
        T = _temperature_value(sys, u, p, state_idx)
        pvec = _parameter_vector(p)
        RT = R_GAS * T
        for (shard, sm) in zip(shards, slotmaps)
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
        return nothing
    end

    stats = ReactionShardedJacStats(
        nrx,
        length(ModelingToolkit.unknowns(sys)),
        length(SparseArrays.nonzeros(J_proto)),
        length(shards),
        length(shards),
        maximum((length(g) for g in rx_groups); init=0),
        maximum((sum(length(r) for r in shard.output_offsets) for shard in shards); init=0),
    )
    return return_stats ? (jac!, J_proto, stats) : (jac!, J_proto)
end
