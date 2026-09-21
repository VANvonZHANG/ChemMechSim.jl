# Rewrite a KPP/MCM converter mechanism into one ChemMechSim can simulate.
#
#   julia --project=. examples/atmospheric/tools/flatten_photolysis.jl [chi0_degrees]
#
# Two transforms. Both are exact AT THE OPERATING POINT rather than approximations of the
# kinetics, and both are stated plainly in the README's limits section:
#
#  1. `zenith-angle-photolysis` -> `elementary` Arrhenius. The MCM rate law is
#        J = l·cos(χ)^m·exp(−n/cos χ)
#     which is evaluated at a FIXED zenith χ0 and frozen. This is the central approximation
#     of this example: it discards the diurnal cycle, so night-only chemistry (NO3 / N2O5
#     accumulation) does not appear. χ0 is a CLI argument, default 0 (overhead sun).
#  2. `sigmoid-branching` -> one constant PER ENTRY, evaluated at T0 and PRESERVING THE SIGN
#     OF `A`. Temperature is fixed in this scenario, so this is exact at T0 — it is not a
#     temperature law. Sign preservation matters: the converter splits a sum into one entry per
#     term, so a negative `A` is a deliberate signed correction, and entries sharing an equation
#     are meant to be summed downstream. See `_flatten_sigmoid`.
#
# It also drops the literal `M` species. This is hygiene, not a correctness fix: the parser
# strips `M` from reaction equations before the species lookup, and `u0` leaves `M` at 0 with no
# reaction producing it, so its `[M]_eff` term is exactly zero either way. Dropping it just
# avoids a phantom all-zero state.
#
# The output is a DERIVED artifact (examples/atmospheric/output/, gitignored). The YAML dict
# round-trip drops the source file's ~16800 provenance comment lines; that is accepted — the
# derived file's only consumer is `load_mechanism`, and the comments' one load-bearing role
# (the sigmoid ground-truth expressions) is consumed here at transform time.

using YAML

"J = l·cos(χ)^m·exp(−n/cos χ), and 0 for cos χ ≤ 1e-10 (night) — never NaN/Inf."
function _photolysis_J(l, m, n, χ)
    c = cos(χ)
    c <= 1e-10 && return 0.0
    return l * c^m * exp(-n / c)
end

"One `zenith-angle-photolysis` reaction -> an `elementary` one carrying a frozen J."
function _flatten_photolysis(rx, χ)
    J = _photolysis_J(Float64(rx["l"]), Float64(rx["m"]), Float64(rx["n"]), χ)
    out = Dict{Any,Any}(
        "equation" => rx["equation"],
        "order" => rx["order"],
        # No "type" key: `_parse_reaction` treats an absent type as elementary.
        "rate-constant" => Dict{Any,Any}("type" => "arrhenius", "A" => J,
                                         "b" => 0.0, "Ea" => 0.0),
    )
    haskey(rx, "duplicate") && (out["duplicate"] = rx["duplicate"])
    return out
end

"""
One `sigmoid-branching` entry -> an `elementary` constant at `T0`, evaluating the converter's
own formula and PRESERVING THE SIGN OF `A`:

    F = 1 + C·exp(D/T)
    k = A·exp(B/T)·(1/F)        normally
    k = A·exp(B/T)·(1 − 1/F)    when is_complement

Sign preservation is the whole point, not an oversight. A negative `A` is a deliberate *signed
correction term*: the converter splits a sum into one entry per term. MCM's CH3O2+HO2 arrives as
THREE entries — a plain Arrhenius carrying `k_total`, plus two signed sigmoids — and
ChemMechSim sums duplicate equations natively (`src/lowering/core.jl`), which is exactly how the
several other signed-correction reactions in this file already work.

    <53>   CH3O2 + HO2 = CH3OOH : 3.8E-13*EXP(780./TEMP)*(1.-1./(1.+498.*EXP(-1160./TEMP)))
    <3778> CH3O2 + HO2 = HCHO  : 3.8E-13*EXP(780./TEMP)*(1./(1.+498.*EXP(-1160./TEMP)))

⚠ An earlier version of this transform instead PAIRED the two sigmoids, treated `|A|` as
`k_total`, and emitted σ / (1−σ) channels. That invented a base term which already existed as the
plain sibling, running the reaction at 2.0× MCM's total rate and 2.1× CH3OOH. Do not
reintroduce it — and note the test that "covered" it passed, because its fixture omitted the
plain sibling and so encoded the same misreading.
"""
function _flatten_sigmoid(rx, T0)
    A = Float64(rx["A"])
    F = 1.0 + Float64(rx["C"]) * exp(Float64(rx["D"]) / T0)
    k = A * exp(Float64(rx["B"]) / T0) * (get(rx, "is_complement", false) ? (1.0 - 1.0 / F) : (1.0 / F))
    out = Dict{Any,Any}(
        "equation" => rx["equation"],
        "order" => rx["order"],
        "rate-constant" => Dict{Any,Any}("type" => "arrhenius", "A" => k,
                                         "b" => 0.0, "Ea" => 0.0))
    haskey(rx, "duplicate") && (out["duplicate"] = rx["duplicate"])
    return out
end

"Transform every reaction of the mechanism dict. MUTATES the dict's reaction list and returns
 the rewritten list alongside the per-type counts, so a silent no-op is visible."
function _flatten_mechanism!(mech_dict, χ0, T0)
    out = Any[]
    n_photo = 0
    n_sig = 0
    for rx in mech_dict["reactions"]
        ty = get(rx, "type", nothing)
        if ty == "zenith-angle-photolysis"
            push!(out, _flatten_photolysis(rx, χ0)); n_photo += 1
        elseif ty == "sigmoid-branching"
            # Per-entry, sign-preserving. Entries that share an equation are deliberately left as
            # duplicates for ChemMechSim to sum — see _flatten_sigmoid.
            push!(out, _flatten_sigmoid(rx, T0)); n_sig += 1
        else
            push!(out, rx)
        end
    end
    mech_dict["reactions"] = out
    return (n_photolysis = n_photo, n_sigmoid = n_sig, reactions = out)
end

"Drop the converter's literal `M` species from the phase list and the species list."
function _drop_M!(mech_dict)
    removed = 0
    for ph in mech_dict["phases"]
        sp = ph["species"]
        n = count(==("M"), sp)
        removed += n
        ph["species"] = [s for s in sp if s != "M"]
    end
    removed += count(sp -> sp["name"] == "M", mech_dict["species"])
    mech_dict["species"] = [sp for sp in mech_dict["species"] if sp["name"] != "M"]
    return removed
end

# —— script body (skipped when the file is `include`d by the test suite) ———————————————
if abspath(PROGRAM_FILE) == @__FILE__
    const χ0 = isempty(ARGS) ? 0.0 : deg2rad(parse(Float64, ARGS[1]))
    const T0 = 298.0
    const HERE = dirname(@__DIR__)                   # tools/ -> examples/atmospheric/
    const SRC  = joinpath(HERE, "mcm_alkanes_alkenes_converted.yaml")
    const DST  = joinpath(HERE, "output", "mcm_alkanes_alkenes_frozen.yaml")

    isfile(SRC) ||
        error("flatten_photolysis: source mechanism not found at\n  $SRC\n" *
              "It is not committed (24 MB). Copy it from the converter:\n" *
              "  cp <kpp-cantera-converter>/examples/mcm/mcm_alkanes_alkenes_converted.yaml \\\n" *
              "     examples/atmospheric/\n" *
              "See examples/atmospheric/README.md.")

    println("loading ", SRC, " ...")
    d = YAML.load_file(SRC)

    n_M = _drop_M!(d)
    stats = _flatten_mechanism!(d, χ0, T0)

    println("  zenith-angle-photolysis -> elementary : ", stats.n_photolysis,
            "  (frozen at χ0 = ", round(rad2deg(χ0), digits = 3), "°)")
    println("  sigmoid-branching entries flattened   : ", stats.n_sigmoid,
            "  (per-entry, sign of A preserved; at T0 = ", T0, " K)")
    println("  literal `M` species dropped           : ", n_M)

    # A silent no-op is the failure mode that matters: assert nothing was left behind.
    left_p = count(r -> get(r, "type", nothing) == "zenith-angle-photolysis", d["reactions"])
    left_s = count(r -> get(r, "type", nothing) == "sigmoid-branching", d["reactions"])
    (left_p == 0 && left_s == 0) ||
        error("flatten_photolysis: $left_p photolysis / $left_s sigmoid reactions remain")
    any(sp -> sp["name"] == "M", d["species"]) &&
        error("flatten_photolysis: `M` still present after _drop_M!")

    # Mechanism-shape invariants. The 24 MB source is gitignored and copied by hand, so a wrong
    # or stale copy is the likeliest fresh-clone failure — and it would otherwise fail SILENTLY
    # (the box would still run, and the OH>0 check would still pass). Change these numbers only
    # when the source mechanism changes.
    length(d["species"]) == 1842 ||
        error("flatten_photolysis: expected 1842 species after dropping M, got ",
              length(d["species"]), " — wrong or stale source file?")
    stats.n_photolysis == 1041 ||
        error("flatten_photolysis: expected 1041 photolysis reactions, got ", stats.n_photolysis)
    stats.n_sigmoid == 2 ||
        error("flatten_photolysis: expected 2 sigmoid-branching entries, got ", stats.n_sigmoid)
    length(d["reactions"]) == 5600 ||
        error("flatten_photolysis: expected 5600 reactions, got ", length(d["reactions"]),
              " — transform dropped or duplicated some")

    mkpath(dirname(DST))
    YAML.write_file(DST, d)
    println("wrote ", DST, "  (", length(d["reactions"]), " reactions, ",
            length(d["species"]), " species)")
end
