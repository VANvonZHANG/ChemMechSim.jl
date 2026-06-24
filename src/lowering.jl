# Lowering pipeline stubs (Mechanism + config → MTK System).
# Implementation deferred to the MVP (Phase 1) plan, which adds the
# ModelingToolkit dependency and the real lowering logic (spec §5.4).

_notimpl(what, phase) =
    error("$what: not implemented in framework scaffold; see the $phase plan.")

"Lower a Mechanism into an MTK ODESystem/DAE. (stub — Phase 1)"
lower_to_mtk(mech::Mechanism; config::MechanismConfig=MechanismConfig()) =
    _notimpl("lower_to_mtk", "MVP (Phase 1)")

"Generate the symbolic rate expression for one reaction. (stub — Phase 1)"
lower_reaction(rx, mech, c, config) = _notimpl("lower_reaction", "MVP (Phase 1)")

"Whether a reaction can use the Catalyst mass-action path. (stub — Phase 1)"
catalyst_native(rx, config) = _notimpl("catalyst_native", "MVP (Phase 1)")

"Catalyst mass-action lowering path. (stub — Phase 1)"
catalyst_lowering(rx, mech, c) = _notimpl("catalyst_lowering", "MVP (Phase 1)")

"Direct-MTK lowering path (third-body/falloff/PLOG/Chebyshev). (stub — Phase 1)"
direct_mtk_lowering(rx, mech, c) = _notimpl("direct_mtk_lowering", "MVP (Phase 1)")

"Append constraint layers (energy/EOS/reactor) to the equation set. (stub — Phase 1)"
append_constraint_layers!(eqs, mech, config) =
    _notimpl("append_constraint_layers!", "MVP (Phase 1)")
