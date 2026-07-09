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
