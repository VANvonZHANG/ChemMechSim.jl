# Catalyst interoperability: import a Catalyst ReactionSystem (mass-action,
# numeric-rate subset) as a ChemMechSim Mechanism (spec §3.3, §4).
#
# Catalyst v16 exports no accessor functions for Reaction fields, so we read the
# documented fields directly: r.rate, r.substrates, r.substoich, r.products,
# r.prodstoich, r.only_use_rate. Only reactions with a plain numeric rate law are
# converted (→ ElementaryArrhenius(rate, 0, 0)); symbolic/parameter rates and
# only_use_rate reactions are deferred to a later phase.

using Catalyst
using ModelingToolkit: getname

"Name of a Catalyst/MTK species symbol, as a String."
_species_name(s) = String(getname(s))

"Import a Catalyst ReactionSystem as a Mechanism (mass-action, numeric-rate subset)."
function import_from_catalyst(rn)
    sps = Catalyst.species(rn)
    species_data = [SpeciesData(id=i, name=_species_name(s)) for (i, s) in enumerate(sps)]
    name_to_id = Dict(_species_name(s) => i for (i, s) in enumerate(sps))
    reactions = ReactionData[]
    for r in Catalyst.reactions(rn)
        r.only_use_rate &&
            error("import_from_catalyst: only_use_rate reactions are not supported in Phase 1.")
        kval = try
            Float64(r.rate)
        catch _
            error("import_from_catalyst: non-numeric rate law ($(r.rate)) is not supported in Phase 1; " *
                  "use plain numeric rate constants.")
        end
        reactants = Dict{SpeciesID,Float64}(
            name_to_id[_species_name(s)] => Float64(nu)
            for (s, nu) in zip(r.substrates, r.substoich))
        products = Dict{SpeciesID,Float64}(
            name_to_id[_species_name(s)] => Float64(nu)
            for (s, nu) in zip(r.products, r.prodstoich))
        push!(reactions, ReactionData(reactants=reactants, products=products,
                                      kinetics=ElementaryArrhenius(kval, 0.0, 0.0)))
    end
    return Mechanism(species=species_data, reactions=reactions)
end
