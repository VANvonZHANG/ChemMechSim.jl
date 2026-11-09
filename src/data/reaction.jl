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
    orders::Dict{SpeciesID,Float64}   # per-species kinetic-order overrides (the YAML
                                      # `kinetic-order` key); empty = pure stoichiometry
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

"Per-species rate-law powers: `meta.orders` overrides the stoichiometric coefficient
 per reactant, unlisted reactants keep ν. THE single source both the mass-action product
 and the effective total order consume — `load_mechanism` validates orders keys ⊆
 reactants; the belt-and-braces check below catches hand-built `ReactionData` that
 bypassed it. Returns `rx.reactants` itself when no override exists (zero-cost,
 bit-identical lowering for override-free mechanisms)."
function rate_powers(rx::ReactionData)
    orders = rx.meta.orders
    isempty(orders) && return rx.reactants
    powers = Dict{SpeciesID,Float64}()
    for (sid, ν) in rx.reactants
        powers[sid] = get(orders, sid, ν)
    end
    for sid in keys(orders)
        haskey(powers, sid) ||
            error("rate_powers: ReactionMeta.orders species $sid is not a reactant of the " *
                  "reaction — per-species orders must reference reactants (load_mechanism " *
                  "enforces this at parse time; hand-built ReactionData bypassed it).")
    end
    return powers
end

"Effective total reaction order = Σ per-species powers (`meta.orders` override, ν
 default). By construction N can never disagree with the mass-action powers — there is
 no way to state N directly (the scalar `kinetic-order` form was evaluated and rejected:
 P3 grilling, 2026-11-09)."
effective_order(rx::ReactionData) = sum(values(rate_powers(rx)))
