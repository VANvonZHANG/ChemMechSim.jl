# Lowering-scoped shared state (§5.6, Phase 4a): the R / P° thermo-constant singletons
# and the per-species NASA7-coefficient cache, each held in a module-level Ref so every
# K_c / energy-equation use in one lower_to_mtk call references the SAME parameter
# (no duplicate-name params). All three are reset at the start of each lower_to_mtk call
# (see core.jl). NOT thread-safe — one lowering at a time.

# Lowering-scoped singletons for the thermo constants R and P° (shared across all K_c uses, so
# every reaction's (P°/RT)^Δν factor references the SAME parameter — no duplicate-name params).
# Reset at the start of each lower_to_mtk call.
const _PSTD_PARAM = Ref{Any}(nothing)
const _RGAS_PARAM = Ref{Any}(nothing)
_p_std_param() = _PSTD_PARAM[] === nothing ? (_PSTD_PARAM[] = rate_param(:P_std, P_STD, u"Pa")) : _PSTD_PARAM[]
_r_param()     = _RGAS_PARAM[] === nothing ? (_RGAS_PARAM[] = rate_param(:R_gas, R_GAS, u"J/(mol*K)")) : _RGAS_PARAM[]

# Per-species NASA7 coefficient cache (Phase 4a). cp/R, h/RT and g/RT for one species share
# ONE coefficient set; without this, calling each separately creates duplicate-name params
# (sp{sid}_a1l, …). Reset at the start of each lower_to_mtk call.
const _COEFF_CACHE = Ref{Dict{Int,Any}}(Dict{Int,Any}())
