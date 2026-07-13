# Lowering context: explicit replacement for the former module-level Ref singletons
# (the old P°/R/coeff-cache refs). One RateCtx per reaction (carries the
# per-reaction naming index `j`, stoich `order`, species vars); one ThermoCtx shared
# across a lower_to_mtk call (R/P°/coeff-cache, T). Threaded explicitly — thread-safe
# (the old Refs were not). All fields are MTK symbolic objects (or plain Dict/Int).

"Per-reaction lowering context for the symbolic rate path."
struct RateCtx
    mech::Mechanism                         # for _meff / _species_by_id
    cvar::Dict{SpeciesID,Any}               # species concentration symbols
    T::Any                                  # temperature symbol (Num) or nothing
    j::Int                                  # reaction index — unique param naming
    order::Real                             # Σ reactant stoich — k unit derivation
    R::Any                                  # shared R_gas param
    P_std::Any                              # shared P° param
    coeff_cache::Dict{Int,Any}              # per-species NASA coeff cache (ThermoCtx shares this Dict)
    P::Any                                  # pressure symbol (Num) under eos=:ideal_gas configs; nothing otherwise
    meff_eqs::Vector{Any}                   # M_eff algebraic equations (one per third-body/falloff reaction),
                                            # collected by lower_to_mtk and appended to eqs; MTK tearing
                                            # eliminates M_eff_j → observed (state+algebraic pattern, §7.1)
end

"Shared thermo/energy lowering context (R/P°/coeff-cache/T). Built once per lower_to_mtk."
struct ThermoCtx
    T::Any
    R::Any
    P_std::Any
    coeff_cache::Dict{Int,Any}
end

"Build a fresh ThermoCtx (new shared R/P° params + empty coeff cache) for one lower_to_mtk call."
function make_thermo_ctx(T)
    R = rate_param(:R_gas, R_GAS, u"J/(mol*K)")
    P = rate_param(:P_std, P_STD, u"Pa")
    ThermoCtx(T, R, P, Dict{Int,Any}())
end
