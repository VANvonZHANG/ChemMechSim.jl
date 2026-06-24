# Reaction data types. Pure Julia (no MTK/unit dependency).
# Species referenced by SpeciesID integer keys, not nested objects.

# —— Reversibility policy (replaces reversible::Bool) ——

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

struct ReactionMeta
    duplicate::Bool
    orders::Dict{SpeciesID,Float64}   # non-mass-action reaction orders
end
ReactionMeta(; duplicate::Bool=false,
               orders::Dict{SpeciesID,Float64}=Dict{SpeciesID,Float64}()) =
    ReactionMeta(duplicate, orders)

# —— Reaction data ——

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
