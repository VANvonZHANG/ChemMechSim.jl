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
        # optional leading coefficient (int/float) then species name (letter-led).
        # Parens allowed for labelled/excited states (e.g. GRI30's CH2(S) singlet methylene).
        m = match(r"^(\d+\.?\d*|\.\d+)?\s*([A-Za-z][A-Za-z0-9()]*)$", term)
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

# —— Reaction parsing (dispatch on `type`) ——————————————————————————

"Convert Dict(name=>coeff) to Dict(SpeciesID=>coeff). Errors on unknown species."
function _names_to_ids(name_coeff::Dict{String,Float64}, name_to_id::Dict{String,SpeciesID})
    out = Dict{SpeciesID,Float64}()
    for (name, coeff) in name_coeff
        haskey(name_to_id, name) ||
            error("load_mechanism: reaction references unknown species \"$name\"")
        out[name_to_id[name]] = coeff
    end
    return out
end

"Parse efficiencies dict (species name -> α). Defaults to 1.0 for unlisted species."
function _parse_efficiencies(eff_dict::Union{Dict,Nothing}, name_to_id::Dict{String,SpeciesID})
    out = Dict{SpeciesID,Float64}()
    eff_dict === nothing && return out
    for (name, α) in eff_dict
        haskey(name_to_id, String(name)) ||
            error("load_mechanism: efficiency references unknown species \"$name\"")
        out[name_to_id[String(name)]] = Float64(α)
    end
    return out
end

"Build an ElementaryArrhenius from a Cantera rate-constant dict, converting units."
function _arrhenius_from_rc(rc, ctx::_UnitCtx, order::Real)
    A  = _convert_A(Float64(rc["A"]), ctx, order)
    b  = Float64(rc["b"])
    Ea = Float64(rc["Ea"]) * ctx.ea_J_per_mol
    return ElementaryArrhenius(A, b, Ea)
end

"Parse one reaction dict into ReactionData. Returns nothing (skipped + warning) for
 unsupported types (PLOG/Chebyshev/etc., Phase 6)."
function _parse_reaction(rxn_dict, name_to_id::Dict{String,SpeciesID}, ctx::_UnitCtx)
    eq = String(rxn_dict["equation"])
    parsed = _parse_equation(eq)
    reactants = _names_to_ids(parsed.reactants, name_to_id)
    products  = _names_to_ids(parsed.products,  name_to_id)
    rtype = get(rxn_dict, "type", "elementary")

    if rtype == "elementary"
        order = sum(values(reactants))
        kin = _arrhenius_from_rc(rxn_dict["rate-constant"], ctx, order)
    elseif rtype == "three-body"
        order = sum(values(reactants)) + 1                  # +1 for [M]
        base = _arrhenius_from_rc(rxn_dict["rate-constant"], ctx, order)
        eff  = _parse_efficiencies(get(rxn_dict, "efficiencies", nothing), name_to_id)
        kin = ThirdBodyArrhenius(base, eff)
    elseif rtype == "falloff"
        base_order = sum(values(reactants))                 # excludes (+M)
        high_rate = _arrhenius_from_rc(rxn_dict["high-P-rate-constant"], ctx, base_order)
        low_rate  = _arrhenius_from_rc(rxn_dict["low-P-rate-constant"],  ctx, base_order + 1)
        eff = _parse_efficiencies(get(rxn_dict, "efficiencies", nothing), name_to_id)
        if haskey(rxn_dict, "Troe")
            t = rxn_dict["Troe"]
            # Cantera {A,T3,T1,T2} -> TroeParams(α=A, T1, T2, T3) — field-aligned, NO reorder (spec T1;
            # lowering.jl:141 formula Fcent=(1-α)exp(-T/T3)+α·exp(-T/T1)+exp(-T/T2) confirmed)
            tp = TroeParams(Float64(t["A"]), Float64(t["T1"]), Float64(t["T2"]), Float64(t["T3"]))
            kin = TroeFalloff(low_rate, high_rate, eff, tp)
        else
            kin = LindemannFalloff(low_rate, high_rate, eff)
        end
    else
        @warn "load_mechanism: skipping $rtype reaction (Phase 6): $eq"
        return nothing
    end

    # reversibility: <=> → ThermoReverse (default); => → Irreversible
    reverse_policy = parsed.reversible ? ThermoReverse() : Irreversible()
    duplicate = Bool(get(rxn_dict, "duplicate", false))
    meta = ReactionMeta(duplicate=duplicate)
    return ReactionData(reactants=reactants, products=products,
                        kinetics=kin, reverse_policy=reverse_policy, meta=meta)
end

# —— Entry point ———————————————————————————————————————————————————

"Load a Cantera YAML mechanism file into a Mechanism (spec §5.1, Phase 5a).
 Covers: elementary / three-body / falloff(Troe/Lindemann). PLOG/Chebyshev/etc.
 are skipped with a warning (Phase 6). Selects the first ideal-gas phase (or the
 named one via `phase`)."
function load_mechanism(path::AbstractString; phase::Union{Nothing,String}=nothing)::Mechanism
    dict = YAML.load_file(path)
    ctx = _parse_units(get(dict, "units", nothing))
    phase_dict = _select_phase(dict["phases"], phase)
    # name -> SpeciesID (1-based by declaration order in the phase)
    species_names = String.(phase_dict["species"])
    name_to_id = Dict{String,SpeciesID}(name => SpeciesID(i) for (i, name) in enumerate(species_names))
    elements = String.(phase_dict["elements"])
    species, thermo_db = _parse_species(dict["species"], name_to_id)
    reactions = ReactionData[]
    for rxn_dict in dict["reactions"]
        r = _parse_reaction(rxn_dict, name_to_id, ctx)
        r === nothing || push!(reactions, r)
    end
    return Mechanism(; species=species, reactions=reactions, thermo=thermo_db, elements=elements)
end
