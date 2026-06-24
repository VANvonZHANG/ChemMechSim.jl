# Thermodynamic model types. Pure Julia (no MTK/unit dependency).
#
# Concrete NASA7/NASA9 polynomial models and their cp/h/s/g evaluations are
# deferred to a later phase plan. Here we define only the abstract parent
# (so SpeciesData.thermo can be typed) and a minimal database container.

"Abstract parent of thermodynamic models (e.g. NASA7, NASA9)."
abstract type ThermoModel end

"A species-keyed collection of ThermoModel entries (the NASA-coefficient store)."
struct ThermoDatabase
    entries::Dict{String,ThermoModel}
end

"Construct an empty ThermoDatabase."
ThermoDatabase() = ThermoDatabase(Dict{String,ThermoModel}())
