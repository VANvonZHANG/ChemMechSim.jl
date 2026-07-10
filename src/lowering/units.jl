# Unit-bearing symbolic parameter construction (spec §5.6): attaching DynamicQuantities
# units to variables/parameters via setmetadata, the rate-constant parameter builder
# rate_param, and the stoichiometrically-derived rate-constant unit _k_unit.

"Attach a DynamicQuantities unit to a symbolic variable/parameter (the @species/@parameters
 macros reject interpolated names with [unit=...], so units are attached via setmetadata)."
_attach_unit(sym, unit) = ModelingToolkit.setmetadata(sym, ModelingToolkit.VariableUnit, unit)

"Build a rate-constant parameter `name` with `default` value and a derived `unit` (§5.6.5).
 The @parameters macro only accepts a LITERAL default with interpolation, so create with a
 placeholder then setdefault + setmetadata."
function rate_param(name::Symbol, default, unit)
    kp = only(@parameters ($(name)) = 1.0)
    kp = ModelingToolkit.setdefault(kp, default)
    return _attach_unit(kp, unit)
end

"Expected unit of a rate constant for overall reaction order `order` (Σ reactant stoich,
 incl. the third-body/[M] factor where applicable) and Arrhenius exponent `b`:
 [k] = conc^(1-order)·s⁻¹ ; the A-factor absorbs T^b -> [A] = [k] / K^b."
_k_unit(order, b) = ChemUnits.conc^(1 - order) * u"s^-1" / (u"K"^b)

# —— direct param-materialization helpers (reproduce current param naming verbatim) ——
# Used by built-in symbolic_kf methods (Task 4) AND by materialize (Task 6). The naming
# Symbol("k_", j, tag, "_A") matches the former _arrhenius_k_param("k_$j"*tag, ...) output:
# tag="" → k_{j}_A, tag="_high" → k_{j}_high_A. θ/Troe-T names likewise.
_aparam(ctx, tag, A, b)  = rate_param(Symbol("k_", ctx.j, tag, "_A"),     A,         _k_unit(ctx.order, b))
_kparam(ctx, tag, Ea)    = rate_param(Symbol("k_", ctx.j, tag, "_theta"), Ea / R_GAS, u"K")
_tvparam(ctx, tag, T)    = rate_param(Symbol("k_", ctx.j, "_troe", tag),  T,         u"K")
