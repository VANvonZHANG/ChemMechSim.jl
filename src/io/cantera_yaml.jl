# Cantera YAML mechanism parser → Mechanism.
# Depends on: YAML.jl, the data layer (SpeciesData/ReactionData/Mechanism/NASA7/kinetics).
# Entry point: `load_mechanism`.

using YAML
using ..ChemMechSim: SpeciesData, SpeciesID, ReactionData, ReactionMeta,
                     ElementaryArrhenius, ThirdBodyArrhenius,
                     TroeFalloff, LindemannFalloff, TroeParams,
                     Irreversible, ThermoReverse,
                     NASA7, ThermoDatabase, Mechanism, molecular_weight,
                     PlogPoint, PlogRate, N_AVOGADRO

# —— Equation string parser ———————————————————————————————————

"Parse one side of a reaction equation (e.g. \"2 OH + H2O\") into Dict(name=>coeff).
 Negative coefficients are accepted (the KPP/MCM-dialect ensemble-bookkeeping extension,
 e.g. NCAR MM's `1.0 A - 2.0 XAOO`): a whitespace-bounded `-` separates terms exactly
 like `+` does, and each term may carry its own leading sign (`-X`, `-1.0 X`, `- 1.0 X`).
 Species names contain no spaces, so a space-bounded ` - ` can never occur inside a name
 like `C4H5-2` — the separator normalization below cannot split a name. Repeated terms
 accumulate with their signs."
function _parse_terms(s::AbstractString)
    result = Dict{String,Float64}()
    # ` - ` → ` + -`: normalize the spaced-minus separator into an explicit per-term sign,
    # so `B - 1.0 C` and `B + -1.0 C` reach the term regex in the same shape.
    s = replace(s, r"\s-\s" => " + -")
    for term in split(s, "+")
        term = strip(term)
        isempty(term) && continue
        # optional leading sign (negative PRODUCT coefficients, ensemble-bookkeeping
        # dialect), then optional coefficient (int/float), then species name (letter-led).
        # Parens allowed for labelled/excited states (e.g. GRI30's CH2(S) singlet methylene).
        # Hyphens, asterisks, #, commas allowed for Aramco/FFCM2 isomer labels
        # (e.g. C2H4O1-2, CH*, C4H5-2, 1,3-C3H6-style — note no species name starts with a
        # digit, and the sign group at term start cannot shadow an in-name hyphen: names
        # never start with `-`). Underscores allowed for WRF-style names (e.g. O1D_CB4);
        # a strict superset of the previous charset — underscore names were simply
        # unparseable before, so no previously-parseable term changes meaning.
        m = match(r"^(-)?\s*((?:\d+\.?\d*|\.\d+))?\s*([A-Za-z][A-Za-z0-9_()\-*,#]*)$", term)
        m === nothing && error("_parse_terms: cannot parse term \"$term\"")
        mag  = isnothing(m.captures[2]) ? 1.0 : parse(Float64, m.captures[2])
        coef = isnothing(m.captures[1]) ? mag : -mag
        name = m.captures[3]
        result[name] = get(result, name, 0.0) + coef
    end
    return result
end

"Parse a Cantera equation string into (reactants, products, reversible, third_body).
 Handles: elementary `A + B <=> C`, irreversible `A => B`, three-body `A + M <=> C + M`,
 falloff `2 OH (+M) <=> H2O2 (+M)`, and negative PRODUCT coefficients
 (`A => B - 1.0 C`, the ensemble-bookkeeping dialect — see `_parse_terms`)."
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
    # Negative stoichiometry is a PRODUCT-side extension. Reactant coefficients must stay
    # positive: the forward rate law is mass-action over the reactants and the A-factor
    # unit conversion uses order = Σν_reactants, so a negative reactant exponent would
    # silently divide by a concentration and corrupt the unit conversion. Fail loudly here.
    any(<(0.0), values(reactants)) &&
        error("_parse_equation: negative reactant coefficient in \"$eq\" " *
              "(negative stoichiometry is supported on the product side only)")
    # three-body + M — remove M from both sides, flag third_body
    if haskey(reactants, "M")
        third_body = true
        delete!(reactants, "M")
        delete!(products, "M")
    end
    return (reactants=reactants, products=products, reversible=rev, third_body=third_body)
end

# —— Unit conversion context ———————————————————————————————————

struct _UnitCtx
    length_m::Float64        # length unit in meters (cm→0.01, m→1.0)
    ea_J_per_mol::Float64    # activation-energy factor to J/mol (cal/mol→4.184, J/mol→1.0, K→8.314)
    amount_per_mol::Float64  # declared amount units per mole (mol→1.0, kmol→1e-3, molec→N_A)
    effective_order::Float64 # per-row effective total order (Σ kinetic-order overrides, ν
                             # default) — set by _build_reaction before each parser/handler
                             # dispatch, so nobody downstream recomputes it by hand
end

# 3-arg convenience: file-level callers (the units block) keep working, order unset (0.0).
_UnitCtx(length_m::Float64, ea_J_per_mol::Float64, amount_per_mol::Float64) =
    _UnitCtx(length_m, ea_J_per_mol, amount_per_mol, 0.0)

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

"Convert a raw YAML A-factor value to canonical m-mol-s units given reaction order.
 Handler-facing: for use inside rate_type_handlers callbacks, where the ctx is passed
 opaquely."
convert_afactor(A::Real, ctx, order::Real) = A * _a_factor(ctx, order)

"Convert a raw YAML A-factor using the ctx's precomputed effective order.
 `load_mechanism` sets `ctx.effective_order` per row (Σ of the `kinetic-order` overrides,
 ν default) before any parser or handler runs, so handlers stop hand-computing the
 order — the historically omitted-conversion bug class lived exactly there. Equivalent to
 `convert_afactor(A, ctx, ctx.effective_order)`; the 3-arg form remains for handlers
 that genuinely need an explicit order."
convert_afactor(A::Real, ctx) = A * _a_factor(ctx, ctx.effective_order)

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
        error("_parse_thermo: unsupported thermo model \"$model\" (only NASA7 and constant-cp are supported)")
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

"""Parse efficiencies dict (species name -> α) honoring the row's `default-efficiency`
 (Cantera semantics: unlisted third bodies take this efficiency; the converter emits
 `default-efficiency: 0.0` on catalyst-preservation rows).

 Returns a SPARSE dict (only listed species present) when `default_α == 1.0`, and a
 MATERIALIZED dict (every phase species, unlisted → `default_α`) otherwise. Sparseness
 matters: lowering's `_meff` (src/lowering/kinetics.jl) resolves unlisted species to
 α=1.0 at lower time and memoizes M_eff variables by the RESOLVED α vector, so returning
 the sparse dict when the default is 1.0 preserves that M_eff sharing across reactions.
 Materializing only when `default_α != 1.0` yields resolved α vectors identical to the
 runner-override behavior being adopted here."""
function _parse_efficiencies(eff_dict::Union{Dict,Nothing}, default_α::Float64,
                             name_to_id::Dict{String,SpeciesID})
    out = Dict{SpeciesID,Float64}()
    eff_dict === nothing && default_α == 1.0 && return out
    if eff_dict !== nothing
        for (name, α) in eff_dict
            haskey(name_to_id, String(name)) ||
                error("load_mechanism: efficiency references unknown species \"$name\"")
            out[name_to_id[String(name)]] = Float64(α)
        end
    end
    default_α == 1.0 && return out          # sparse: unlisted species stay absent
    for (name, id) in name_to_id            # materialize: unlisted → default_α
        haskey(out, id) || (out[id] = default_α)
    end
    return out
end

"Build an ElementaryArrhenius from a Cantera rate-constant dict, converting units."
function _arrhenius_from_rc(rc, ctx::_UnitCtx, order::Real)
    A  = convert_afactor(Float64(rc["A"]), ctx, order)
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
    # ctx.effective_order: precomputed by _build_reaction (Σ kinetic-order overrides, ν
    # default) — the ONE order source; identical to sum(values(reactants)) when the row
    # carries no `kinetic-order` key.
    return _arrhenius_from_rc(rxn_dict["rate-constant"], ctx, ctx.effective_order)
end

function _parse_three_body_rc(rxn_dict, reactants, name_to_id, ctx::_UnitCtx)
    order = ctx.effective_order + 1                     # +1 for [M]
    base = _arrhenius_from_rc(rxn_dict["rate-constant"], ctx, order)
    # `default-efficiency`: efficiency of unlisted third bodies (Cantera semantics; the
    # converter emits 0.0 on catalyst-preservation rows). Key absent → 1.0 → the
    # historical sparse dict, bit-identically.
    default_α = Float64(get(rxn_dict, "default-efficiency", 1.0))
    eff  = _parse_efficiencies(get(rxn_dict, "efficiencies", nothing), default_α, name_to_id)
    return ThirdBodyArrhenius(base, eff)
end

function _parse_falloff_rc(rxn_dict, reactants, name_to_id, ctx::_UnitCtx)
    base_order = ctx.effective_order                    # excludes (+M)
    high_rate = _arrhenius_from_rc(rxn_dict["high-P-rate-constant"], ctx, base_order)
    low_rate  = _arrhenius_from_rc(rxn_dict["low-P-rate-constant"],  ctx, base_order + 1)
    # `default-efficiency`: efficiency of unlisted third bodies (Cantera semantics; the
    # converter emits 0.0 on catalyst-preservation rows). Key absent → 1.0 → the
    # historical sparse dict, bit-identically.
    default_α = Float64(get(rxn_dict, "default-efficiency", 1.0))
    eff = _parse_efficiencies(get(rxn_dict, "efficiencies", nothing), default_α, name_to_id)
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
    order = ctx.effective_order
    pts = PlogPoint[]
    for rc in rxn_dict["rate-constants"]
        P_Pa = _parse_pressure(rc["P"], ctx)
        A    = convert_afactor(Float64(rc["A"]), ctx, order)
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

"Parse the row-level `kinetic-order` override (P3): a species→apparent-order mapping.
 Only the mapping form exists — the effective total order is ALWAYS the per-species sum
 (override where given, stoichiometric ν otherwise), never stated directly, so N and the
 mass-action powers cannot disagree by construction. All failures are loud
 `load_mechanism:` errors: unsupported on `three-body`/`falloff` rows (the [M]_eff
 factor is not a per-species power), keys must name reactants of the row, values must
 be finite and ≥ 0 (a negative power would divide by a concentration; 0 is legal
 zero-order kinetics — the species leaves the rate law but stays in the ledger).
 An empty mapping is treated as no key."
function _parse_kinetic_order(rxn_dict, rtype::AbstractString,
                              reactants::Dict{SpeciesID,Float64},
                              name_to_id::Dict{String,SpeciesID})
    ko = get(rxn_dict, "kinetic-order", nothing)
    ko === nothing && return Dict{SpeciesID,Float64}()
    eq = String(rxn_dict["equation"])
    (rtype == "three-body" || rtype == "falloff") &&
        error("load_mechanism: kinetic-order is not supported on $rtype rows — the " *
              "third-body [M]_eff factor is not a per-species power. Row: \"$eq\".")
    ko isa AbstractDict ||
        error("load_mechanism: kinetic-order must be a species→order mapping (e.g. " *
              "{ICNOO: 1.0}); the effective total order is the per-species sum and is " *
              "never stated directly. Got: $ko. Row: \"$eq\".")
    out = Dict{SpeciesID,Float64}()
    for (name, ν) in ko
        sname = String(name)
        haskey(name_to_id, sname) ||
            error("load_mechanism: kinetic-order references unknown species \"$sname\". Row: \"$eq\".")
        sid = name_to_id[sname]
        haskey(reactants, sid) ||
            error("load_mechanism: kinetic-order species \"$sname\" is not a reactant of \"$eq\".")
        ν isa Real && isfinite(ν) && ν >= 0 ||
            error("load_mechanism: kinetic-order for \"$sname\" must be a finite number " *
                  "≥ 0 (a negative power would divide by a concentration; 0 is legal " *
                  "zero-order kinetics). Got: $ν. Row: \"$eq\".")
        out[sid] = Float64(ν)
    end
    return out
end

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
    orders = _parse_kinetic_order(rxn_dict, rtype, reactants, name_to_id)
    # Reversible rows reject overrides: the thermodynamic reverse (kf/K_c) requires
    # mass-action stoichiometry — with N_eff ≠ Σν one kf would need two different units
    # (conc^(1-N_eff)·s⁻¹ forward vs conc^(1-Σν)·s⁻¹ reverse), which the symbolic unit
    # check rejects. P3 grilling amendment, 2026-11-09.
    if !isempty(orders) && parsed.reversible
        error("load_mechanism: `kinetic-order` on a reversible (`<=>`) row is not supported — " *
              "the thermodynamic reverse rate (kf/K_c) requires mass-action stoichiometry " *
              "(one kf cannot carry both branch units). Use an irreversible row (`=>`) or " *
              "remove the override. Equation: $eq")
    end
    # Per-row ctx: the effective order (Σ overrides, ν default) rides the ctx so built-in
    # parsers and custom handlers consume ONE order source (see `convert_afactor(A, ctx)`).
    # When no override exists this is bitwise the former sum(values(reactants)).
    # init=0.0: an empty reactants side (source term `=> A`) is order 0 — the bare sum
    # would throw on the empty collection (source rows never reached this sum before).
    eff_order = sum(get(orders, sid, ν) for (sid, ν) in reactants; init=0.0)
    rowctx = _UnitCtx(ctx.length_m, ctx.ea_J_per_mol, ctx.amount_per_mol, eff_order)
    kin = parser(rxn_dict, reactants, name_to_id, rowctx)

    # reversibility: <=> → ThermoReverse (default); => → Irreversible
    reverse_policy = parsed.reversible ? ThermoReverse() : Irreversible()
    duplicate = Bool(get(rxn_dict, "duplicate", false))
    meta = ReactionMeta(duplicate=duplicate, orders=orders)
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
