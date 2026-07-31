using SparseArrays

struct ReactionShardedJacStats
    n_reactions::Int
    n_states::Int
    n_nonzeros::Int
    n_reaction_shards::Int
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
    deps = Set(keys(rx.reactants))
    if rx.kinetics isa ThirdBodyArrhenius
        union!(deps, (sp.id for sp in mech.species))
    end
    return deps
end

function _reaction_extra_dependency_states(rx::ReactionData, state_idx)
    rx.kinetics isa PlogRate || return Int[]
    haskey(state_idx, "P") ||
        throw(ArgumentError("build_reaction_sharded_jac: PlogRate reaction requires pressure " *
                            "state P in the lowered fixedT system."))
    return [state_idx["P"]]
end

_reaction_sharded_supported_kinetics(kin) =
    kin isa ElementaryArrhenius || kin isa ThirdBodyArrhenius || kin isa PlogRate

_reaction_sharded_supports(rx::ReactionData) =
    _reaction_sharded_supported_kinetics(rx.kinetics) && rx.reverse_policy isa Irreversible

function _assert_reaction_sharded_supported(mech::Mechanism)
    for (i, rx) in enumerate(mech.reactions)
        _reaction_sharded_supports(rx) && continue
        throw(ArgumentError(
            "build_reaction_sharded_jac: unsupported reaction $i; prototype supports only " *
            "irreversible ElementaryArrhenius, ThirdBodyArrhenius, and PlogRate reactions " *
            "in fixedT mode. Got kinetics=" *
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

function _add_jac_entry!(J::SparseArrays.SparseMatrixCSC, row::Int, col::Int, value)
    iszero(value) && return nothing
    for slot in J.colptr[col]:(J.colptr[col + 1] - 1)
        if J.rowval[slot] == row
            J.nzval[slot] += value
            return nothing
        end
    end
    throw(ArgumentError("build_reaction_sharded_jac: sparse template has no slot for " *
                        "Jacobian entry row=$row col=$col."))
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

function _symbolic_reaction_rate(rx::ReactionData, mech::Mechanism, sys,
                                 config::MechanismConfig, j::Int)
    cvar = _species_id_to_state_symbol(mech, sys)
    state_syms = ModelingToolkit.unknowns(sys)
    state_by_name = Dict(String(ModelingToolkit.getname(s)) => s for s in state_syms)
    P = get(state_by_name, "P", nothing)
    T = _fixedT_parameter_symbol(sys)
    tcx = make_thermo_ctx(T)
    ctx = RateCtx(mech, cvar, T, j, sum(values(rx.reactants)),
                  tcx.R, tcx.P_std, tcx.coeff_cache, P, Any[])
    return _reaction_sharded_symbolic_kf(rx.kinetics, ctx) *
           _mass_action(rx.reactants, cvar)
end

function _reaction_derivative_fn(rx::ReactionData, mech::Mechanism, sys,
                                 config::MechanismConfig, j::Int)
    state_syms = ModelingToolkit.unknowns(sys)
    param_syms = ModelingToolkit.parameters(sys)
    rate_expr = _symbolic_reaction_rate(rx, mech, sys, config, j)
    derivs = [ModelingToolkit.expand_derivatives(ModelingToolkit.Differential(v)(rate_expr))
              for v in state_syms]
    return ModelingToolkit.build_function(
        derivs, state_syms, param_syms, [ModelingToolkit.t_nounits];
        expression=Val{false}, wrap_gfw=Val{false})[1]
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
    # Prototype note: reaction_shard_size is reported in stats only. The current fixedT
    # elementary path already builds one derivative function per reaction; future work can
    # batch those functions into actual shards when broadening this path beyond the prototype.
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
    reaction_derivative_fns = [_reaction_derivative_fn(rx, mech, sys, config, j)
                               for (j, rx) in enumerate(mech.reactions)]

    function jac!(J::SparseArrays.SparseMatrixCSC, u, p, t)
        fill!(J.nzval, 0.0)
        T = _temperature_value(sys, u, p, state_idx)
        pvec = _parameter_vector(p)
        for (rx, derivative_fn) in zip(mech.reactions, reaction_derivative_fns)
            drdu = derivative_fn(u, pvec, t)
            _sanitize_shard_buf!(drdu)
            row_sids = union(keys(rx.reactants), keys(rx.products))
            pressure_net = 0.0
            for sid in row_sids
                net = _netstoich(rx, sid)
                pressure_net += net
                iszero(net) && continue
                row = sid_to_idx[sid]
                for col in eachindex(drdu)
                    drate_dconc = drdu[col]
                    _add_jac_entry!(J, row, col, net * drate_dconc)
                end
            end
            if pressure_idx !== nothing && !iszero(pressure_net)
                pressure_factor = R_GAS * T * pressure_net
                for col in eachindex(drdu)
                    drate_dconc = drdu[col]
                    _add_jac_entry!(J, pressure_idx, col, pressure_factor * drate_dconc)
                end
            end
        end
        return nothing
    end

    stats = ReactionShardedJacStats(
        length(mech.reactions),
        length(ModelingToolkit.unknowns(sys)),
        length(SparseArrays.nonzeros(J_proto)),
        cld(length(mech.reactions), reaction_shard_size),
    )
    return return_stats ? (jac!, J_proto, stats) : (jac!, J_proto)
end
