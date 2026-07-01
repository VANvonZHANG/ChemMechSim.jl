# Validation module. ValidationReport is concrete (data); the `validate` checks
# are implemented starting Phase 2.5b (spec §8).

"A structured validation report: errors (must fix), warnings, info."
struct ValidationReport
    errors::Vector{String}
    warnings::Vector{String}
    info::Vector{String}
end
ValidationReport() = ValidationReport(String[], String[], String[])

"Run scientific-reliability checks on a Mechanism. Returns a ValidationReport."
function validate(mech::Mechanism; T_range::Union{Tuple{Float64,Float64},Nothing}=nothing)
    rep = ValidationReport()
    _check_element_conservation(mech, rep)
    _check_molecular_weights(mech, rep)
    _check_nasa_temp_range(mech, rep, T_range)
    _check_thirdbody_efficiencies(mech, rep)
    _check_duplicate_reactions(mech, rep)
    _check_reverse_consistency(mech, rep)
    return rep
end

function _check_element_conservation(mech::Mechanism, rep::ValidationReport)
    if isempty(mech.elements)
        push!(rep.warnings,
              "element-conservation check skipped: mechanism has no element list " *
              "(provide Mechanism(elements=[...]) to enable it).")
        return
    end
    for (j, rx) in enumerate(mech.reactions)
        lhs = _element_totals(rx.reactants, mech, rep)
        rhs = _element_totals(rx.products,  mech, rep)
        _dicts_approx_equal(lhs, rhs) ||
            push!(rep.errors,
                  "reaction #$j is not element-balanced: " *
                  "reactant elements $lhs ≠ product elements $rhs.")
    end
end

function _element_totals(stoich::Dict{SpeciesID,Float64}, mech::Mechanism, rep::ValidationReport)
    counts = Dict{String,Float64}()
    for (sid, nu) in stoich
        sp = species_by_id(mech, sid)
        if isempty(sp.elements)
            push!(rep.warnings,
                  "species $(sp.name) (id $sid) has no elemental composition; " *
                  "element-conservation check cannot verify it.")
            continue
        end
        for (el, n) in sp.elements
            counts[el] = get(counts, el, 0.0) + nu * n
        end
    end
    return counts
end

function _check_molecular_weights(mech::Mechanism, rep::ValidationReport)
    for sp in mech.species
        isnan(sp.molecular_weight) &&
            push!(rep.warnings,
                  "species $(sp.name) (id $(sp.id)) has no molecular weight " *
                  "(required for EOS / mass-fraction state basis; spec §5.3.4).")
    end
end

function _check_nasa_temp_range(mech::Mechanism, rep::ValidationReport, T_range)
    T_range === nothing && return
    Tmin, Tmax = T_range
    for sp in mech.species
        sp.thermo isa NASA7 || continue
        n = sp.thermo
        (n.Tlow <= Tmin && Tmax <= n.Thigh) ||
            push!(rep.warnings,
                  "species $(sp.name) (id $(sp.id)) NASA range [$(n.Tlow),$(n.Thigh)] does not " *
                  "cover the simulation T range [$Tmin,$Tmax]; results outside the fit are unreliable.")
    end
end

function _check_thirdbody_efficiencies(mech::Mechanism, rep::ValidationReport)
    ids = Set(sp.id for sp in mech.species)
    for (j, rx) in enumerate(mech.reactions)
        eff = _efficiencies_of(rx.kinetics)
        eff === nothing && continue
        for sid in keys(eff)
            sid in ids ||
                push!(rep.errors,
                      "reaction #$j third-body efficiency references unknown species id $sid.")
        end
    end
end
_efficiencies_of(::ElementaryArrhenius) = nothing
_efficiencies_of(k::ThirdBodyArrhenius) = k.efficiencies
_efficiencies_of(k::AbstractFalloff) = k.efficiencies
_efficiencies_of(::AbstractKinetics) = nothing

function _check_duplicate_reactions(mech::Mechanism, rep::ValidationReport)
    sigs = [_stoich_signature(rx) for rx in mech.reactions]
    for i in eachindex(mech.reactions)
        rx = mech.reactions[i]
        has_twin = any(j != i && sigs[j] == sigs[i] for j in eachindex(mech.reactions))
        if has_twin && !rx.meta.duplicate
            push!(rep.warnings,
                  "reaction #$i has the same stoichiometry as another reaction but is not marked " *
                  "`duplicate` (ReactionMeta); CHEMKIN requires duplicate pairs to both carry the flag.")
        end
    end
end
_stoich_signature(rx::ReactionData) =
    (sort(collect(rx.reactants)), sort(collect(rx.products)))

function _check_reverse_consistency(mech::Mechanism, rep::ValidationReport)
    ids = Set(sp.id for sp in mech.species)
    for (j, rx) in enumerate(mech.reactions)
        rx.reverse_policy isa ThermoReverse || continue
        for sid in union(keys(rx.reactants), keys(rx.products))
            sid in ids || continue                      # dangling id: skip (not this check's concern)
            _species_by_id(mech, sid).thermo === nothing &&
                push!(rep.errors,
                      "reaction #$j is ThermoReverse but species id $sid lacks NASA thermo " *
                      "(K_c needs thermo on all reactants/products).")
        end
    end
end
_species_by_id(mech::Mechanism, sid) = species_by_id(mech, sid)

"Float-robust Dict{String,Float64} equality for element-count comparison (review M5):
 same keys and all values isapprox within atol=1e-9."
function _dicts_approx_equal(a::Dict{String,Float64}, b::Dict{String,Float64})
    sort(collect(keys(a))) == sort(collect(keys(b))) || return false
    return all(isapprox(a[k], b[k]; atol=1e-9) for k in keys(a))
end
