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
#  2. `sigmoid-branching` -> two constants evaluated at T0. Temperature is fixed in this
#     scenario, so this is exact at T0 — it is not a temperature law.
#
# It also drops the literal `M` species. The converter declares `M` as a real species, and
# `_meff` sums over ALL declared species, so leaving it in would double-count it against
# N2/O2 in [M]_eff.
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
The two `sigmoid-branching` channels of one reaction -> two elementary constants at `T0`.

    k_total = |A|·exp(B/T)          σ = 1/(1 + C·exp(D/T))
    negative-A entry -> k_total·(1 − σ)      positive-A entry -> k_total·σ

The A-sign convention was verified against the source file's own `# Original Rate
Expression` comments for MCM's CH3O2+HO2 pair:
    CH3OOH = 3.8E-13*EXP(780./TEMP)*(1.-1./(1.+498.*EXP(-1160./TEMP)))
    HCHO   = 3.8E-13*EXP(780./TEMP)*(1./(1.+498.*EXP(-1160./TEMP)))

Note the converter's own `SigmoidBranchingRate.eval` does NOT reproduce this: it branches on
`is_complement`, which is `false` for both entries here, so it would return a NEGATIVE rate
for the (1−σ) channel. We deliberately do not copy that evaluator.

Asserts the pairing (opposite-sign A, identical B/C/D) so a malformed group errors loudly
instead of silently mis-assigning a channel.
"""
function _flatten_sigmoid_pair(group, T0)
    length(group) == 2 ||
        error("_flatten_sigmoid_pair: expected exactly 2 channels, got $(length(group))")
    rxs = sort(group; by = r -> Float64(r["A"]))
    neg, pos = rxs[1], rxs[2]
    aneg, apos = Float64(neg["A"]), Float64(pos["A"])
    aneg < 0 < apos ||
        error("_flatten_sigmoid_pair: expected one negative and one positive A, got " *
              "$aneg / $apos")
    for k in ("B", "C", "D")
        Float64(neg[k]) == Float64(pos[k]) ||
            error("_flatten_sigmoid_pair: $k mismatch ($(neg[k]) vs $(pos[k]))")
    end

    ktot = abs(apos) * exp(Float64(pos["B"]) / T0)
    σ = 1.0 / (1.0 + Float64(pos["C"]) * exp(Float64(pos["D"]) / T0))
    mk(r, k) = Dict{Any,Any}(
        "equation" => r["equation"],
        "order" => r["order"],
        "rate-constant" => Dict{Any,Any}("type" => "arrhenius", "A" => k,
                                         "b" => 0.0, "Ea" => 0.0))
    return [mk(neg, ktot * (1 - σ)), mk(pos, ktot * σ)]
end

"Reactant side of an equation string — the sigmoid grouping key (the two channels of one
 reaction share reactants and differ only in products)."
_sigmoid_key(rx) = (strip(split(String(rx["equation"]), "=>")[1]),
                    Float64(rx["B"]), Float64(rx["C"]), Float64(rx["D"]))

"Transform the whole mechanism dict in place-ish, returning the rewritten reaction list.
 Pure w.r.t. the input dict so it can be unit-tested without file I/O."
function _flatten_mechanism!(mech_dict, χ0, T0)
    rxs = mech_dict["reactions"]

    # Group sigmoid channels by (reactants, B, C, D) so each pair is flattened as a unit.
    sig_groups = Dict{Any,Vector{Any}}()
    for rx in rxs
        get(rx, "type", nothing) == "sigmoid-branching" || continue
        push!(get!(sig_groups, _sigmoid_key(rx), Any[]), rx)
    end

    out = Any[]
    n_photo = 0
    n_sig = 0
    for rx in rxs
        ty = get(rx, "type", nothing)
        if ty == "zenith-angle-photolysis"
            push!(out, _flatten_photolysis(rx, χ0)); n_photo += 1
        elseif ty == "sigmoid-branching"
            n_sig += 1                       # replaced wholesale by the group pass below
        else
            push!(out, rx)
        end
    end
    n_groups = 0
    for (_, g) in sig_groups
        append!(out, _flatten_sigmoid_pair(g, T0)); n_groups += 1
    end

    mech_dict["reactions"] = out
    return (n_photolysis = n_photo, n_sigmoid_channels = n_sig, n_sigmoid_groups = n_groups)
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
    println("  sigmoid-branching channels flattened  : ", stats.n_sigmoid_channels,
            " in ", stats.n_sigmoid_groups, " pair(s)  (at T0 = ", T0, " K)")
    println("  literal `M` species dropped           : ", n_M)

    # A silent no-op is the failure mode that matters: assert nothing was left behind.
    left_p = count(r -> get(r, "type", nothing) == "zenith-angle-photolysis", d["reactions"])
    left_s = count(r -> get(r, "type", nothing) == "sigmoid-branching", d["reactions"])
    (left_p == 0 && left_s == 0) ||
        error("flatten_photolysis: $left_p photolysis / $left_s sigmoid reactions remain")
    stats.n_photolysis > 0 ||
        error("flatten_photolysis: no photolysis reactions found — wrong input file?")
    any(sp -> sp["name"] == "M", d["species"]) &&
        error("flatten_photolysis: `M` still present after _drop_M!")

    mkpath(dirname(DST))
    YAML.write_file(DST, d)
    println("wrote ", DST, "  (", length(d["reactions"]), " reactions, ",
            length(d["species"]), " species)")
end
