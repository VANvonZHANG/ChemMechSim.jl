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
function validate(mech::Mechanism)
    rep = ValidationReport()
    _check_element_conservation(mech, rep)
    _check_molecular_weights(mech, rep)
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
        lhs == rhs ||
            push!(rep.errors,
                  "reaction #$j is not element-balanced: " *
                  "reactant elements $lhs ≠ product elements $rhs.")
    end
end

function _element_totals(stoich::Dict{SpeciesID,Float64}, mech::Mechanism, rep::ValidationReport)
    counts = Dict{String,Float64}()
    for (sid, nu) in stoich
        sp = mech.species[sid]
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
