# Cantera YAML mechanism parser → Mechanism (spec §5.1, Phase 5a).
# Depends on: YAML.jl, the data layer (SpeciesData/ReactionData/Mechanism/NASA7/kinetics).
# Entry point `load_mechanism` arrives in Task 4.

using YAML
using ..ChemMechSim: SpeciesData, SpeciesID, ReactionData, ReactionMeta,
                     ElementaryArrhenius, ThirdBodyArrhenius,
                     TroeFalloff, LindemannFalloff, TroeParams,
                     Irreversible, ThermoReverse,
                     NASA7, ThermoDatabase, Mechanism, molecular_weight

# —— Equation string parser ———————————————————————————————————

"Parse one side of a reaction equation (e.g. \"2 OH + H2O\") into Dict(name=>coeff)."
function _parse_terms(s::AbstractString)
    result = Dict{String,Float64}()
    for term in split(s, "+")
        term = strip(term)
        isempty(term) && continue
        # optional leading coefficient (int/float) then species name (letter-led)
        m = match(r"^(\d+\.?\d*|\.\d+)?\s*([A-Za-z][A-Za-z0-9]*)$", term)
        m === nothing && error("_parse_terms: cannot parse term \"$term\"")
        coef = isnothing(m.captures[1]) ? 1.0 : parse(Float64, m.captures[1])
        name = m.captures[2]
        result[name] = get(result, name, 0.0) + coef
    end
    return result
end

"Parse a Cantera equation string into (reactants, products, reversible, third_body).
 Handles: elementary `A + B <=> C`, irreversible `A => B`, three-body `A + M <=> C + M`,
 falloff `2 OH (+M) <=> H2O2 (+M)`."
function _parse_equation(eq::AbstractString)
    s = strip(eq)
    third_body = false
    # falloff (+M) modifier — strip it, flag third_body
    if occursin(r"\(\+\s*M\)", s)
        third_body = true
        s = replace(s, r"\(\+\s*M\)" => "")
    end
    # split reactants / products
    rev = true
    if occursin("<=>", s)
        left, right = split(s, "<=>", limit=2)
    elseif occursin("=>", s)
        rev = false
        left, right = split(s, "=>", limit=2)
    elseif occursin("=", s)
        left, right = split(s, "=", limit=2)
    else
        error("_parse_equation: no reaction arrow in \"$eq\"")
    end
    reactants = _parse_terms(left)
    products  = _parse_terms(right)
    # three-body + M — remove M from both sides, flag third_body
    if haskey(reactants, "M")
        third_body = true
        delete!(reactants, "M")
        delete!(products, "M")
    end
    return (reactants=reactants, products=products, reversible=rev, third_body=third_body)
end
