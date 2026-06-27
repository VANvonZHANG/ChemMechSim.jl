# Mechanism: top-level pure-data aggregate. Pure Julia (no MTK dependency).
# Short name (not *Data) per the §14.2 three-tier convention: it is the main
# type users interact with. NOT the same abstraction as Catalyst's ReactionSystem
# (a symbolic system); this is a data record.

struct Mechanism
    species::Vector{SpeciesData}
    reactions::Vector{ReactionData}
    thermo::ThermoDatabase           # NASA-coefficient store (may be empty)
    elements::Vector{String}         # element list (for element-conservation checks)
end

function Mechanism(; species::Vector{SpeciesData},
                     reactions::Vector{ReactionData},
                     thermo::ThermoDatabase=ThermoDatabase(),
                     elements::Vector{String}=String[])
    Mechanism(species, reactions, thermo, elements)
end

"Robust species lookup by id. Uses findfirst rather than assuming species.id == 1-based
 index, so non-contiguous/non-1-based ids cannot silently look up the wrong species (review I2)."
function species_by_id(mech::Mechanism, sid::SpeciesID)
    idx = findfirst(sp -> sp.id == sid, mech.species)
    idx === nothing && error("species_by_id: no species with id $sid in this Mechanism.")
    return mech.species[idx]
end
