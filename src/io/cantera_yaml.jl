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

# —— Unit conversion context (spec §5.6.3) ——————————————————————————

struct _UnitCtx
    length_m::Float64      # length unit in meters (cm→0.01, m→1.0)
    ea_J_per_mol::Float64  # activation-energy factor to J/mol (cal/mol→4.184, J/mol→1.0)
end

"Parse the YAML `units:` block into a conversion context (defaults to SI)."
function _parse_units(units_dict::Union{Dict,Nothing})
    units_dict === nothing && return _UnitCtx(1.0, 1.0)
    length_unit = get(units_dict, "length", "m")
    length_m = length_unit == "cm" ? 0.01 :
               length_unit == "m"  ? 1.0  :
               error("_parse_units: unsupported length unit \"$length_unit\"")
    ea_unit = get(units_dict, "activation-energy", "J/mol")
    ea_factor = ea_unit == "cal/mol" ? 4.184 :
                ea_unit == "J/mol"   ? 1.0   :
                error("_parse_units: unsupported activation-energy unit \"$ea_unit\"")
    return _UnitCtx(length_m, ea_factor)
end

"A-factor conversion factor: Cantera → canonical (m-mol-s).
 A_canon = A_cantera × (1/length_m)^(3·(1−order));  order = reactant stoichiometric sum."
_a_factor(ctx::_UnitCtx, order::Real) = (1.0 / ctx.length_m)^(3 * (1 - order))

"Convert a Cantera A-factor value to canonical m-mol-s units given reaction order."
_convert_A(A::Real, ctx::_UnitCtx, order::Real) = A * _a_factor(ctx, order)

# —— Phase selection ——————————————————————————————————————————————

"Select the first ideal-gas phase (or the named one). Errors on non-ideal-gas."
function _select_phase(phases_list, phase_name::Union{Nothing,String})
    for ph in phases_list
        if ph["thermo"] == "ideal-gas" && (phase_name === nothing || ph["name"] == phase_name)
            return ph
        end
    end
    error("load_mechanism: no ideal-gas phase found" *
          (phase_name === nothing ? "" : " named \"$phase_name\""))
end

# —— Species / thermo parsing ——————————————————————————————————————

"Parse a Cantera thermo block (NASA7 only) into a ThermoModel."
function _parse_thermo(thermo_dict)
    model = thermo_dict["model"]
    model == "NASA7" ||
        error("_parse_thermo: unsupported thermo model \"$model\" (only NASA7 in Phase 5a)")
    ranges = thermo_dict["temperature-ranges"]
    data   = thermo_dict["data"]
    low  = NTuple{7,Float64}(Float64(x) for x in data[1])
    high = NTuple{7,Float64}(Float64(x) for x in data[2])
    return NASA7(low, high, Float64(ranges[1]), Float64(ranges[2]), Float64(ranges[3]))
end

"Parse the species list. Returns (Vector{SpeciesData}, ThermoDatabase).
 Only species in name_to_id (phase-declared) are kept; others skipped."
function _parse_species(species_list, name_to_id::Dict{String,SpeciesID})
    species = SpeciesData[]
    thermo_entries = Dict{String,ThermoModel}()
    for sp_dict in species_list
        name = String(sp_dict["name"])
        haskey(name_to_id, name) || continue
        id = name_to_id[name]
        comp_raw = sp_dict["composition"]                  # Dict{Any,Any} from YAML.jl
        elements = Dict{String,Int}(String(k) => Int(v) for (k, v) in comp_raw)
        mw = molecular_weight(elements)
        thermo = haskey(sp_dict, "thermo") ? _parse_thermo(sp_dict["thermo"]) : nothing
        if thermo !== nothing
            thermo_entries[name] = thermo
        end
        push!(species, SpeciesData(id=id, name=name, elements=elements,
                                   molecular_weight=mw, thermo=thermo))
    end
    return species, ThermoDatabase(thermo_entries)
end
