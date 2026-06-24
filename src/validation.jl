# Validation module. ValidationReport is concrete (data); the `validate` checks
# are implemented starting Phase 2.5. (spec §8)

"A structured validation report: errors (must fix), warnings, info."
struct ValidationReport
    errors::Vector{String}
    warnings::Vector{String}
    info::Vector{String}
end
ValidationReport() = ValidationReport(String[], String[], String[])

"Run scientific-reliability checks on a Mechanism. (stub — Phase 2.5)"
validate(mech) =
    error("validate: not implemented in framework scaffold; see the Phase 2.5 plan.")
