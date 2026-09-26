# Species data. Pure Julia struct, but its keyword constructor canonicalizes the
# molar mass at the boundary via the ChemUnits submodule (DynamicQuantities). The
# stored value is a bare Float64 (kg/mol) so the data layer stays unit-free.

"""
    SpeciesData(; id, name, elements=Dict(), molecular_weight=NaN, thermo=nothing, role=:dynamic)

One chemical species. `elements` is the elemental composition (`Dict{String,Int}`,
element symbol → atom count); `molecular_weight` is canonicalized to kg/mol at the
constructor (NaN if unspecified); `thermo` is an optional `ThermoModel` (nothing for
pure kinetics); `role` is a `SpeciesRole`.

# Example
sp = SpeciesData(id=1, name="A", elements=Dict("A" => 1), thermo=nA)
"""
struct SpeciesData
    id::SpeciesID                         # stable integer index
    name::String
    elements::Dict{String,Int}            # elemental composition
    molecular_weight::Float64             # kg/mol (canonical; NaN if unspecified)
    thermo::Union{ThermoModel,Nothing}    # optional; nothing for pure kinetics
    role::SpeciesRole                     # :dynamic | :algebraic_qssa | :constant_pool | :bath_gas
end

function SpeciesData(; id::SpeciesID,
                       name::AbstractString,
                       elements::Dict{String,Int}=Dict{String,Int}(),
                       molecular_weight=NaN,
                       thermo::Union{ThermoModel,Nothing}=nothing,
                       role::SpeciesRole=:dynamic)
    mw = ChemUnits.canonical(molecular_weight, ChemUnits.molmass)
    SpeciesData(id, String(name), elements, mw, thermo, role)
end
