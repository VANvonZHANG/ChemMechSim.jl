# High-level API stubs (layered: simulate / build_problem / extract_system).
# Implementation arrives with the MVP (Phase 1) / Phase 3 plans. (spec §6, §7)

_apinotimpl(what, phase) =
    error("$what: not implemented in framework scaffold; see the $phase plan.")

"Simulate a reactor over a time span. (stub — Phase 1)"
simulate(reactor; kwargs...) = _apinotimpl("simulate", "MVP (Phase 1)")
simulate(reactor, tspan; kwargs...) = _apinotimpl("simulate", "MVP (Phase 1)")

"Build an ODEProblem from a reactor. (stub — Phase 1)"
build_problem(reactor, args...; kwargs...) =
    _apinotimpl("build_problem", "MVP (Phase 1)")

"Extract the underlying MTK ODESystem from a ChemPhaseSystem."
extract_system(phase::ChemPhaseSystem) = phase.sys

"Generate standalone RHS Julia code from a system. (stub — Phase 3)"
generate_function(sys; kwargs...) =
    _apinotimpl("generate_function", "Phase 3")

"Generate standalone Jacobian Julia code from a system. (stub — Phase 3)"
generate_jacobian(sys; kwargs...) =
    _apinotimpl("generate_jacobian", "Phase 3")
