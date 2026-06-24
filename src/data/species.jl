# Species data. Pure Julia struct, but its keyword constructor canonicalizes the
# molar mass at the boundary via the ChemUnits submodule (DynamicQuantities). The
# stored value is a bare Float64 (kg/mol) so the data layer stays unit-free.

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
