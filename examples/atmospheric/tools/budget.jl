# Rank the reactions driving O3 and HOx at the final state of the box.
#
#   julia --project=. examples/atmospheric/tools/budget.jl
#
# WHY A BUDGET AND NOT A TIME SERIES: the box reaches a near-steady state, so the interesting
# statement is not "O3 falls" but "O3 falls because of THESE reactions". This evaluates each
# reaction's net rate at the final time and signs it by whether it makes or destroys O3.
#
# THE RATE LAW AND ITS UNITS — read this before changing anything below. The law is
#     k = A·T^b·exp(-Ea/(R·T))
# The mechanism's YAML header says `activation-energy: K`, which is true OF THE FILE — but the
# parser converts on load: `_arrhenius_from_rc` (src/io/cantera_yaml.jl) stores
# `Ea = Ea_yaml · ea_J_per_mol`, and `_parse_units` maps the "K" unit to 8.314. So in the
# PARSED `ElementaryArrhenius` struct this script reads, Ea is in J/mol and must be divided by
# R. Every numeric rate path in ChemMechSim does exactly that (`_kparam` in
# src/lowering/units.jl; `numeric_value(::KTemp, Ea)` and `PlogRate` in src/data/kinetics.jl).
# Writing `exp(-rc.Ea / T0)` — the form that IS correct for the raw YAML dict, as
# test/test_atmospheric_preprocessor.jl uses on it — is wrong by a factor of R in the exponent
# here, which for a reaction with Ea = -600 K inflates its rate by ~2.5e6.
#
# CAVEAT this script cannot escape: photolysis rates are FROZEN (see the README), so a budget
# computed here describes a perpetual-noon box, not a diurnal average.

using ChemMechSim
using Printf

const HERE = dirname(@__DIR__)                   # tools/ -> examples/atmospheric/
const MECH = joinpath(HERE, "output", "mcm_alkanes_alkenes_frozen.yaml")
const STATE = joinpath(HERE, "output", "final_state.csv")
const OUT = joinpath(HERE, "output", "budget.csv")

const T0 = 298.0
const R_GAS = 8.314                              # J/(mol·K) — see the units note above

isfile(STATE) || error("budget: run mcm_box.jl first — $STATE not found")
isfile(MECH)  || error("budget: run tools/flatten_photolysis.jl first — $MECH not found")

mech = load_mechanism(MECH)

# Final-state concentrations, keyed by species name. The COLUMN POSITIONS come from the file's
# own header, so reordering the columns upstream cannot silently mis-index them.
lines = readlines(STATE)
header = strip.(split(lines[1], ","))
i_name = findfirst(==("species"), header)
i_conc = findfirst(==("concentration_mol_m3"), header)
(i_name === nothing || i_conc === nothing) &&
    error("budget: $STATE must have columns species,concentration_mol_m3; " *
          "found $(join(header, ","))")
c = Dict{String,Float64}()
negatives = Float64[]        # `push!` mutates, so this needs no top-level-loop scope dance
for line in lines[2:end]
    isempty(strip(line)) && continue
    parts = split(line, ",")
    name = strip(parts[i_name])
    v = parse(Float64, parts[i_conc])
    # Physically-zero species come back from the integrator as tiny NEGATIVE residual: 461 of
    # the 1842 values here, most negative ~-4e-23, against a solver abstol of 1e-12 (the largest
    # concentration in the file is 32.4 = N2). A negative base is not harmless to a rate law:
    # c^nu raises a DomainError for a non-integer nu — and `_parse_terms` does accept fractional
    # coefficients — while for integer nu it silently flips the sign of the rate, so a small
    # negative squared becomes a spurious POSITIVE contribution. Below tolerance there is no
    # physics left to preserve, so clamp to exactly 0.0 and count what was clamped.
    v < 0.0 && push!(negatives, v)
    # Magnitude gate (review): clamp only tolerable residual. A genuinely unstable run would
    # hand back negatives far above abstol; clamping those would emit a plausible-looking budget
    # from garbage, so fail loudly instead.
    v < -1e-9 &&
        error("budget: $name = $v is far below abstol=1e-12 — the run diverged; not clamping")
    c[name] = max(v, 0.0)
end
@printf("concentrations: %d species read from %s\n", length(c), basename(STATE))
@printf("  %d negative values clamped to 0 (most negative %.3e; solver residual, abstol = 1e-12)\n",
        length(negatives), isempty(negatives) ? 0.0 : minimum(negatives))

# The rate laws need EVERY species. A partial vector would evaluate silently against zeros, so
# fail loudly instead — this is the failure the 7-species series.csv produced.
missing_species = [String(sp.name) for sp in mech.species if !haskey(c, String(sp.name))]
isempty(missing_species) ||
    error("budget: $STATE is missing $(length(missing_species)) of $(length(mech.species)) " *
          "species, e.g. $(join(missing_species[1:min(3, end)], ", "))")

# Concentrations indexed by species id (mech.species order) so the M_eff sum below does not do
# a string-keyed lookup per term.
conc_by_id = [c[String(sp.name)] for sp in mech.species]

"Third-body effective concentration [M]_eff = Σ α_i·c_i over ALL species, unlisted α → 1.0.
 Mirrors ChemMechSim's `_meff` (src/lowering/kinetics.jl)."
function meff(efficiencies)
    s = 0.0
    for (i, sp) in enumerate(mech.species)
        s += get(efficiencies, sp.id, 1.0) * conc_by_id[i]
    end
    return s
end

"Arrhenius k(T0) = A·T0^b·exp(-Ea/(R·T0)). The struct's Ea is J/mol — see the units note."
arrhenius_k(kin::ElementaryArrhenius) = kin.A * T0^kin.b * exp(-kin.Ea / (R_GAS * T0))

"Effective forward rate constant at T0 EXCLUDING the mass-action term — i.e. what
 `symbolic_kf` returns in ChemMechSim's own lowering. Returns nothing for kinetics this budget
 cannot evaluate; the caller counts those and reports them, so coverage is visible rather than
 implied."
effective_k(kin::ElementaryArrhenius) = arrhenius_k(kin)

effective_k(kin::ThirdBodyArrhenius) = arrhenius_k(kin.base) * meff(kin.efficiencies)

"Troe: k = kinf·(Pr/(1+Pr))·F with Pr = k0·[M]_eff/kinf. F comes from ChemMechSim's own
 `_troe_F_body` (src/data/kinetics.jl), which that file documents as the numeric-entry-point
 form of the formula the symbolic lowering uses — so the budget cannot drift from the
 integrator's own Fcent/F arithmetic."
function effective_k(kin::TroeFalloff)
    kinf = arrhenius_k(kin.high_rate)
    k0   = arrhenius_k(kin.low_rate)
    Pr   = k0 * meff(kin.efficiencies) / kinf
    tp = kin.troe
    F = ChemMechSim._troe_F_body(tp.α, tp.T1, tp.T2, tp.T3, Pr, T0)
    return kinf * (Pr / (1.0 + Pr)) * F
end

"Lindemann: as Troe with F ≡ 1 (no center broadening)."
function effective_k(kin::LindemannFalloff)
    kinf = arrhenius_k(kin.high_rate)
    k0   = arrhenius_k(kin.low_rate)
    Pr   = k0 * meff(kin.efficiencies) / kinf
    return kinf * (Pr / (1.0 + Pr))
end

effective_k(::AbstractKinetics) = nothing        # e.g. PLOG — counted below, never silent

"Rate of one reaction at T0 given the final-state concentrations, in mol/(m^3 s)."
function reaction_rate(rx)
    k = effective_k(rx.kinetics)
    k === nothing && return nothing
    prod = 1.0
    for (sid, nu) in rx.reactants
        prod *= c[String(mech.species[sid].name)]^nu
    end
    return k * prod
end

# Species that define each budget. Resolved once — `findfirst` per reaction would be 5600 linear
# scans of a 1842-vector.
const O3_SPECIES = "O3"
const HOX_SPECIES = ("OH", "HO2")
id_of(name) = let i = findfirst(sp -> String(sp.name) == name, mech.species)
    i === nothing && error("budget: no $name species in the mechanism")
    mech.species[i].id
end
const o3_id = id_of(O3_SPECIES)
const hox_ids = [id_of(n) for n in HOX_SPECIES]

"Net stoichiometric change in a species group over one reaction: Σ (products − reactants)."
net_change(products, reactants, ids) =
    sum(get(products, sid, 0.0) - get(reactants, sid, 0.0) for sid in ids)

"One side of a reaction as text, terms sorted by species name so the rendering is deterministic:
 `2 HO2 + O2`. The parser strips the literal `M` from every equation, so a third-body reaction
 shows without it — its third-body nature lives in the kinetics type, not in the equation text."
function side_str(stoich)
    parts = String[]
    for sid in sort(collect(keys(stoich)), by = s -> String(mech.species[s].name))
        nu = stoich[sid]
        name = String(mech.species[sid].name)
        push!(parts, nu == 1.0 ? name : string(nu % 1 == 0 ? Int(nu) : nu, " ", name))
    end
    return join(parts, " + ")
end

"Kinetics classification, spelled the way the mechanism file spells it in its own `type:` field.
 Reports `typeof(rx.kinetics)` — nothing is reconstructed."
kinetics_tag(::ElementaryArrhenius) = "elementary"
kinetics_tag(::ThirdBodyArrhenius)  = "three-body"
kinetics_tag(::Union{TroeFalloff,LindemannFalloff}) = "falloff"
kinetics_tag(kin::AbstractKinetics) = string(nameof(typeof(kin)))   # anything else names itself

"Full equation text, e.g. `O3 => O1D [elementary]`. This is the column that disambiguates
 reactions whose reactant-only label collides: the two `O3` photolysis channels, and the duplicate
 entries the converter splits out of one summed rate.
 The trailing tag is load-bearing, not decoration. The parser strips the literal `M` from every
 equation, so `2 HO2 => H2O2` (elementary) and `2 HO2 + M => H2O2 + M` (three-body) otherwise
 render identically while being physically different — the second is scaled by [M]_eff ≈ 41.7.
 Tagging reports the parser's own classification; it does NOT reconstruct the `M`, because 136 of
 this mechanism's three-body reactions never wrote one in their source equation."
equation_str(rx) = side_str(rx.reactants) * " => " * side_str(rx.products) *
                   " [" * kinetics_tag(rx.kinetics) * "]"

# BUDGET TAGGING IS BY NET STOICHIOMETRY, not by "does this species appear on this side":
#
#     net = Σ_{s ∈ group} (products[s] − reactants[s])
#       net > 0  →  the reaction is a SOURCE for that group
#       net < 0  →  it is a SINK
#       net = 0  →  it belongs in NEITHER list (counted below, never silently dropped)
#
# For O3 the two rules coincide — no reaction in this mechanism has O3 on both sides (verified) —
# but net is what makes that structural rather than a property of the mechanism: a reaction that
# both made and consumed O3 would otherwise be counted into both lists.
#
# For HOx the difference is the whole point. OH and HO2 are ONE family, so tagging by appearance
# put every OH↔HO2 interconversion into BOTH lists at an identical rate (92 such pairs in the
# first version of this script), ranking net-neutral chemistry such as
# `HCHO + OH => HO2 + CO + H2O` above genuine sources. Net tagging makes the two lists disjoint
# and makes `HOx_source` answer the question actually being asked: where does NEW HOx come from?
#
# The rate column is the HOx-MOLECULE flux, rate × |net|, not the reaction rate. `O1D + H2O => 2 OH`
# therefore reports twice its reaction rate, because it makes two HOx. Each list consequently sums
# to the total HOx production / destruction rate in mol/(m^3 s), and the two sums differ by the
# box's net HOx tendency.

rows = Tuple{String,String,String,Float64}[]     # reaction label, equation, kind, rate
skipped = Dict{String,Int}()                     # kinetics type -> reactions not evaluated
for rx in mech.reactions
    rate = reaction_rate(rx)
    if rate === nothing
        t = string(nameof(typeof(rx.kinetics)))
        skipped[t] = get(skipped, t, 0) + 1      # Dict index-assign mutates; not a soft-scope hit
        continue
    end
    iszero(rate) && continue
    # `reaction` keeps the brief's reactant-species-only form because Task 3 consumes that column;
    # `equation` carries the disambiguating full text. Both are emitted on purpose.
    # Sorted (not Dict order) so the label is deterministic across regenerations — a consumer
    # string-matching this column must not see "CH4 + OH" flip to "OH + CH4". side_str sorts
    # for the same reason; this is its reactants-only equivalent.
    label = join(sort([String(mech.species[sid].name) for sid in keys(rx.reactants)]), " + ")
    eq = equation_str(rx)
    net_o3 = net_change(rx.products, rx.reactants, (o3_id,))
    if net_o3 > 0.0
        push!(rows, (label, eq, "O3_production", rate * net_o3))
    elseif net_o3 < 0.0
        push!(rows, (label, eq, "O3_loss", rate * -net_o3))
    end
    net_hox = net_change(rx.products, rx.reactants, hox_ids)
    if net_hox > 0.0
        push!(rows, (label, eq, "HOx_source", rate * net_hox))
    elseif net_hox < 0.0
        push!(rows, (label, eq, "HOx_sink", rate * -net_hox))
    end
end

# Reactions that involve OH/HO2 but shift net HOx by zero. They are now in neither HOx list; the
# old appear-on-either-side rule put each of them in BOTH. Counted, not silently dropped.
touches_hox(rx) = any(sid -> haskey(rx.reactants, sid) || haskey(rx.products, sid), hox_ids)
n_hox_neutral = count(rx -> touches_hox(rx) &&
                           net_change(rx.products, rx.reactants, hox_ids) == 0.0,
                      mech.reactions)

# Every reaction either landed in `skipped` or was evaluated, so the coverage count is derived
# rather than accumulated — which also keeps it out of the top-level loop's soft scope.
n_evaluated = length(mech.reactions) - sum(values(skipped); init = 0)

open(OUT, "w") do io
    println(io, "reaction,equation,kind,rate_mol_m3_s")
    for (rxn, eq, kind, rate) in sort(rows, by = r -> -r[4])
        println(io, replace(rxn, "," => ";"), ",", replace(eq, "," => ";"), ",", kind, ",", rate)
    end
end

# Coverage, stated rather than implied: a budget that silently dropped a quarter of the
# mechanism would otherwise look exactly like a complete one.
@printf("\nreaction coverage: %d of %d evaluated (%.1f%%)",
        n_evaluated, length(mech.reactions), 100 * n_evaluated / length(mech.reactions))
if isempty(skipped)
    println("  — none skipped")
else
    @printf("  — %d skipped:\n", sum(values(skipped)))
    for (t, n) in sort(collect(skipped), by = x -> -x[2])
        @printf("      %-22s %5d   no rate law in this budget\n", t, n)
    end
end

@printf("\nHOx net-neutral: %d reactions involve OH/HO2 but leave net HOx unchanged, so they are\n",
        n_hox_neutral)
println("  in NEITHER HOx list. Both HOx lists report the HOx-molecule flux, rate × |net|, not")
println("  the reaction rate — so the two columns are directly comparable and summable.")

# Print the dominant terms per budget so a human can sanity-check without opening the CSV.
for kind in ("O3_production", "O3_loss", "HOx_source", "HOx_sink")
    sub = sort([r for r in rows if r[3] == kind], by = r -> -r[4])
    isempty(sub) && (println("\n", kind, " — NO ROWS"); continue)
    @printf("\n%s — top 5 of %d (mol/m^3/s)\n", kind, length(sub))
    for (rxn, eq, _, rate) in sub[1:min(5, end)]
        @printf("  %-24s %.4e   %s\n", rxn, rate, eq)
    end
end

println("\nwrote ", OUT)
