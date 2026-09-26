# Reaction data types. Pure Julia (no MTK/unit dependency).
# Species referenced by SpeciesID integer keys, not nested objects.

# —— Reversibility policy (replaces reversible::Bool) ——

"""
    ReverseRatePolicy

Abstract supertype of the reversibility strategies: `Irreversible` (forward only),
`ExplicitReverse` (an explicit reverse rate law), `ThermoReverse` (reverse rate from
the thermodynamic equilibrium constant K_c(T)).
"""
abstract type ReverseRatePolicy end

"Reaction proceeds only forward."
struct Irreversible <: ReverseRatePolicy end

"Reaction has an explicit reverse rate law."
struct ExplicitReverse{R<:AbstractKinetics} <: ReverseRatePolicy
    rate::R
end

"Reverse rate derived from thermodynamic equilibrium constant K_c(T)."
struct ThermoReverse <: ReverseRatePolicy end

# —— Reaction metadata ——

"""
    ReactionMeta(; duplicate=false, orders=Dict())

Per-reaction bookkeeping: `duplicate` flags CHEMKIN-style duplicate reactions, and
`orders` holds non-mass-action reaction orders keyed by `SpeciesID` (empty = plain
mass action).
"""
struct ReactionMeta
    duplicate::Bool
    orders::Dict{SpeciesID,Float64}   # non-mass-action reaction orders
end
ReactionMeta(; duplicate::Bool=false,
               orders::Dict{SpeciesID,Float64}=Dict{SpeciesID,Float64}()) =
    ReactionMeta(duplicate, orders)

# —— Reaction data ——

"""
    ReactionData(; reactants, products, kinetics, reverse_policy=Irreversible(), meta=ReactionMeta())

One reaction. `reactants`/`products` are `Dict{SpeciesID,Float64}` stoichiometric
coefficient maps, `kinetics` the forward rate law (`AbstractKinetics`), `reverse_policy`
the reversibility strategy, and `meta` optional bookkeeping.

# Example
rxn = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                   kinetics=ElementaryArrhenius(0.5, 0.0, 0.0))   # k = 0.5 s⁻¹
"""
struct ReactionData
    reactants::Dict{SpeciesID,Float64}    # stoichiometric coefficients
    products::Dict{SpeciesID,Float64}
    kinetics::AbstractKinetics            # rate law (see kinetics.jl)
    reverse_policy::ReverseRatePolicy     # reversibility strategy
    meta::ReactionMeta
end

function ReactionData(; reactants::Dict{SpeciesID,Float64},
                        products::Dict{SpeciesID,Float64},
                        kinetics::AbstractKinetics,
                        reverse_policy::ReverseRatePolicy=Irreversible(),
                        meta::ReactionMeta=ReactionMeta())
    ReactionData(reactants, products, kinetics, reverse_policy, meta)
end
