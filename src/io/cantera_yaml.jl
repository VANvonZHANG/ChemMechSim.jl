# Cantera YAML mechanism parser → Mechanism (spec §5.1, Phase 5a).
# Depends on: YAML.jl, the data layer (SpeciesData/ReactionData/Mechanism/NASA7/kinetics).
# Entry point `load_mechanism` arrives in Task 4.

using YAML
using ..ChemMechSim: SpeciesData, SpeciesID, ReactionData, ReactionMeta,
                     ElementaryArrhenius, ThirdBodyArrhenius,
                     TroeFalloff, LindemannFalloff, TroeParams,
                     Irreversible, ThermoReverse,
                     NASA7, ThermoDatabase, Mechanism, molecular_weight,
                     PlogPoint, PlogRate, N_AVOGADRO

# —— Equation string parser ———————————————————————————————————

"Parse one side of a reaction equation (e.g. \"2 OH + H2O\") into Dict(name=>coeff)."
function _parse_terms(s::AbstractString)
    result = Dict{String,Float64}()
    for term in split(s, "+")
        term = strip(term)
        isempty(term) && continue
        # optional leading coefficient (int/float) then species name (letter-led).
        # Parens allowed for labelled/excited states (e.g. GRI30's CH2(S) singlet methylene).
        # Hyphens, asterisks, #, commas allowed for Aramco/FFCM2 isomer labels
        # (e.g. C2H4O1-2, CH*, C4H5-2, 1,3-C3H6-style — note no species name starts with a digit).
        m = match(r"^(\d+\.?\d*|\.\d+)?\s*([A-Za-z][A-Za-z0-9()\-*,#]*)$", term)
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
    length_m::Float64        # length unit in meters (cm→0.01, m→1.0)
    ea_J_per_mol::Float64    # activation-energy factor to J/mol (cal/mol→4.184, J/mol→1.0, K→8.314)
    amount_per_mol::Float64  # declared amount units per mole (mol→1.0, kmol→1e-3, molec→N_A)
end

# 2-arg convenience: callers that only care about length/Ea keep working, amount defaults to mol.
_UnitCtx(length_m::Float64, ea_J_per_mol::Float64) = _UnitCtx(length_m, ea_J_per_mol, 1.0)

"Parse the YAML `units:` block into a conversion context (defaults to SI).
 `activation-energy: K` is the MCM/KPP convention — Ea is already divided by R, so
 exp(-Ea/T) holds directly and the SI factor is R itself (Ea_SI = Ea_K · R).
 `quantity: molec` (also spelled `molecule`) declares A in molecules, not moles — the
 dominant convention in KPP-derived atmospheric mechanisms (MCM, GEOS-Chem fullchem)."
function _parse_units(units_dict::Union{Dict,Nothing})
    units_dict === nothing && return _UnitCtx(1.0, 1.0)
    length_unit = get(units_dict, "length", "m")
    length_m = length_unit == "cm" ? 0.01 :
               length_unit == "m"  ? 1.0  :
               error("_parse_units: unsupported length unit \"$length_unit\"")
    ea_unit = get(units_dict, "activation-energy", "J/mol")
    ea_factor = ea_unit == "cal/mol" ? 4.184 :
                ea_unit == "J/mol"   ? 1.0   :
                ea_unit == "K"       ? 8.314 :
                error("_parse_units: unsupported activation-energy unit \"$ea_unit\"")
    amount_unit = get(units_dict, "quantity", "mol")
    amount_per_mol = amount_unit == "mol"     ? 1.0        :
                     amount_unit == "kmol"    ? 1.0e-3     :
                     amount_unit == "molec"   ? N_AVOGADRO :
                     amount_unit == "molecule" ? N_AVOGADRO :
                     error("_parse_units: unsupported quantity unit \"$amount_unit\"")
    return _UnitCtx(length_m, ea_factor, amount_per_mol)
end

"A-factor conversion factor: Cantera → canonical (m-mol-s).
 A_canon = A_cantera × (1/length_m)^(3·(1−order)) × amount_per_mol^(order−1);
 order = reactant stoichiometric sum (incl. the +1 for a third body's [M]).
 The amount factor is 1.0 for `quantity: mol`, so mol-declared mechanisms (GRI30, FFCM2,
 AramcoMech3.0) are numerically unchanged; it is N_A for `quantity: molec`."
_a_factor(ctx::_UnitCtx, order::Real) =
    (1.0 / ctx.length_m)^(3 * (1 - order)) * ctx.amount_per_mol^(order - 1)

"Convert a Cantera A-factor value to canonical m-mol-s units given reaction order."
_convert_A(A::Real, ctx::_UnitCtx, order::Real) = A * _a_factor(ctx, order)

"Handler-facing: convert a raw YAML A-factor to canonical m-mol-s units given reaction
 order. For use inside rate_type_handlers callbacks, where the ctx is passed opaquely."
convert_afactor(A::Real, ctx, order::Real) = _convert_A(A, ctx, order)

"Handler-facing: activation energy from the file's declared unit to J/mol
 (K→8.314, cal/mol→4.184, J/mol→1.0)."
ea_to_J_per_mol(Ea::Real, ctx) = Ea * ctx.ea_J_per_mol

"Parse a Cantera pressure quantity (e.g. \"0.001 atm\", \"986.9 atm\", or a bare number)
 to Pa. atm ×101325, bar ×1e5, Pa ×1. Default Pa if no unit suffix."
function _parse_pressure(p, ::_UnitCtx)
    p isa Number && return Float64(p)                    # bare number → assume Pa
    s = strip(string(p))
    m = match(r"^\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)\s*(atm|bar|Pa)?\s*$", s)
    m === nothing && error("_parse_pressure: cannot parse \"$s\"")
    val = parse(Float64, m.captures[1])
    unit = m.captures[2]
    return unit == "atm" ? val * 101325.0 :
           unit == "bar" ? val * 1.0e5 :
           val                                           # Pa (or unspecified → Pa)
end

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

"Parse a Cantera thermo block (NASA7 only) into a ThermoModel.
 Handles both the canonical 2-block form (3 temperature-ranges, 2 data rows) and
 the single-block form (2 temperature-ranges, 1 data row — e.g. noble gases in
 AramcoMech3.0). For the single-block form the same coefficients are used for both
 low and high ranges (the midpoint is taken as the upper bound)."
function _parse_thermo(thermo_dict)
    model = thermo_dict["model"]
    # `constant-cp` carries no polynomial. In KPP/MCM converter output its data is uniformly
    # zero (h0=s0=cp0=0), so there is nothing to represent — return `nothing`: the species
    # still parses and joins the mechanism, and `:kinetic` + reverse_rate=:irreversible never
    # consults thermo. Any other unknown model stays loud.
    model == "constant-cp" && return nothing
    model == "NASA7" ||
        error("_parse_thermo: unsupported thermo model \"$model\" (only NASA7 in Phase 5a)")
    ranges = thermo_dict["temperature-ranges"]
    data   = thermo_dict["data"]
    low  = NTuple{7,Float64}(Float64(x) for x in data[1])
    if length(data) >= 2
        # canonical 2-block form: ranges = [Tlow, Tmid, Thigh], data = [low, high]
        high = NTuple{7,Float64}(Float64(x) for x in data[2])
        return NASA7(low, high, Float64(ranges[1]), Float64(ranges[2]), Float64(ranges[3]))
    else
        # single-block form: ranges = [Tlow, Thigh], data = [only]; same coeffs for both ranges
        high = low
        return NASA7(low, high, Float64(ranges[1]), Float64(ranges[2]), Float64(ranges[2]))
    end
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

# —— rate-type parsers (uniform registry contract) ————————————————————————————
# Each parser: (rxn_dict, reactants, name_to_id, ctx) -> AbstractKinetics.
# The reaction loop dispatches the YAML `type` string through a Dict merged from
# DEFAULT_RATE_PARSERS and the caller's rate_type_handlers — user entries override
# built-ins. Nothing here knows about any specific non-Cantera dialect.

function _parse_elementary_rc(rxn_dict, reactants, name_to_id, ctx::_UnitCtx)
    order = sum(values(reactants))
    return _arrhenius_from_rc(rxn_dict["rate-constant"], ctx, order)
end

function _parse_three_body_rc(rxn_dict, reactants, name_to_id, ctx::_UnitCtx)
    order = sum(values(reactants)) + 1                  # +1 for [M]
    base = _arrhenius_from_rc(rxn_dict["rate-constant"], ctx, order)
    eff  = _parse_efficiencies(get(rxn_dict, "efficiencies", nothing), name_to_id)
    return ThirdBodyArrhenius(base, eff)
end

function _parse_falloff_rc(rxn_dict, reactants, name_to_id, ctx::_UnitCtx)
    base_order = sum(values(reactants))                 # excludes (+M)
    high_rate = _arrhenius_from_rc(rxn_dict["high-P-rate-constant"], ctx, base_order)
    low_rate  = _arrhenius_from_rc(rxn_dict["low-P-rate-constant"],  ctx, base_order + 1)
    eff = _parse_efficiencies(get(rxn_dict, "efficiencies", nothing), name_to_id)
    if haskey(rxn_dict, "Troe")
        t = rxn_dict["Troe"]
        # Cantera {A,T3,T1,T2} -> TroeParams(α=A, T1, T2, T3) — field-aligned, NO reorder (spec T1;
        # lowering.jl _troe_F formula Fcent=(1-α)exp(-T/T3)+α·exp(-T/T1)+exp(-T2/T) confirmed).
        # T2 is optional in Cantera (omitted in e.g. AramcoMech3.0 for some reactions); when
        # absent, exp(-T2/T) → 0, so we use a huge T2 (1e30) to underflow that term to zero.
        T2 = Float64(get(t, "T2", 1e30))
        tp = TroeParams(Float64(t["A"]), Float64(t["T1"]), T2, Float64(t["T3"]))
        return TroeFalloff(low_rate, high_rate, eff, tp)
    else
        return LindemannFalloff(low_rate, high_rate, eff)
    end
end

function _parse_plog_rc(rxn_dict, reactants, name_to_id, ctx::_UnitCtx)
    order = sum(values(reactants))
    pts = PlogPoint[]
    for rc in rxn_dict["rate-constants"]
        P_Pa = _parse_pressure(rc["P"], ctx)
        A    = _convert_A(Float64(rc["A"]), ctx, order)
        b    = Float64(rc["b"])
        Ea   = Float64(rc["Ea"]) * ctx.ea_J_per_mol
        push!(pts, PlogPoint(P_Pa, A, b, Ea))
    end
    sort!(pts, by = p -> p.P)                        # defensive: ensure ascending
    eq = String(rxn_dict["equation"])
    length(pts) >= 2 ||
        error("load_mechanism: PLOG reaction needs ≥2 pressure points; got $(length(pts)) in \"$eq\".")
    length(unique(round.(p.P, sigdigits=12) for p in pts)) >= 2 ||
        error("load_mechanism: PLOG reaction needs ≥2 distinct pressures; all same in \"$eq\".")
    return PlogRate(pts)
end

"The built-in Cantera reaction types, keyed by their YAML `type` string."
const DEFAULT_RATE_PARSERS = Dict{String,Function}(
    "elementary"                   => _parse_elementary_rc,
    "three-body"                   => _parse_three_body_rc,
    "falloff"                      => _parse_falloff_rc,
    "pressure-dependent-Arrhenius" => _parse_plog_rc,
)

"Parse one reaction dict into ReactionData, dispatching the YAML `type` string through
 `parsers`. Returns `nothing` for a type with no parser — the caller counts and
 aggregates the skip into one warning."
function _build_reaction(rxn_dict, parsers::Dict{String,Function},
                         name_to_id::Dict{String,SpeciesID}, ctx::_UnitCtx)
    eq = String(rxn_dict["equation"])
    parsed = _parse_equation(eq)
    reactants = _names_to_ids(parsed.reactants, name_to_id)
    products  = _names_to_ids(parsed.products,  name_to_id)
    rtype = get(rxn_dict, "type", "elementary")
    parser = get(parsers, rtype, nothing)
    parser === nothing && return nothing
    kin = parser(rxn_dict, reactants, name_to_id, ctx)

    # reversibility: <=> → ThermoReverse (default); => → Irreversible
    reverse_policy = parsed.reversible ? ThermoReverse() : Irreversible()
    duplicate = Bool(get(rxn_dict, "duplicate", false))
    meta = ReactionMeta(duplicate=duplicate)
    return ReactionData(reactants=reactants, products=products,
                        kinetics=kin, reverse_policy=reverse_policy, meta=meta)
end

# —— Entry point ———————————————————————————————————————————————————

"""Load a Cantera-YAML mechanism.

`rate_type_handlers` maps YAML reaction `type` strings to parser functions with the
signature `(rxn_dict, reactants, name_to_id, ctx) -> AbstractKinetics`; entries here
OVERRIDE the built-in parsers for the same type. Use `convert_afactor` / `ea_to_J_per_mol`
on `ctx` for unit conversions. Reactions whose type has no parser are skipped, with one
aggregated warning at the end.

# Example
mech = load_mechanism("examples/mechanism/gri30.yaml")
mech = load_mechanism("mcm.yaml";
                      rate_type_handlers=Dict("Arrhenius-Photo" => my_parser))
"""
function load_mechanism(path::AbstractString;
                        phase::Union{Nothing,String}=nothing,
                        rate_type_handlers::Dict{String,Function}=Dict{String,Function}())::Mechanism
    dict = YAML.load_file(path)
    ctx = _parse_units(get(dict, "units", nothing))
    phase_dict = _select_phase(dict["phases"], phase)
    # name -> SpeciesID (1-based by declaration order in the phase)
    species_names = String.(phase_dict["species"])
    name_to_id = Dict{String,SpeciesID}(name => SpeciesID(i) for (i, name) in enumerate(species_names))
    # `elements` is optional in Cantera YAML — KPP/MCM converter output omits it.
    # Nothing downstream needs it (species carry their own composition), so default empty.
    elements = String.(get(phase_dict, "elements", String[]))
    species, thermo_db = _parse_species(dict["species"], name_to_id)
    # User handlers override built-ins (merge order matters).
    parsers = merge(DEFAULT_RATE_PARSERS, rate_type_handlers)
    reactions = ReactionData[]
    skipped = Dict{String,Int}()
    for rxn_dict in dict["reactions"]
        r = _build_reaction(rxn_dict, parsers, name_to_id, ctx)
        if r === nothing
            rtype = get(rxn_dict, "type", "elementary")
            skipped[rtype] = get(skipped, rtype, 0) + 1
        else
            push!(reactions, r)
        end
    end
    if !isempty(skipped)
        total = sum(values(skipped))
        detail = join(["$(k) ×$(v)" for (k, v) in sort(collect(skipped))], ", ")
        @warn "load_mechanism: skipped $total reactions with unsupported rate types: $detail. " *
              "Pass rate_type_handlers=Dict(\"<type>\" => (rxn, reactants, name_to_id, ctx) -> kin) to handle them."
    end
    return Mechanism(; species=species, reactions=reactions, thermo=thermo_db, elements=elements)
end
