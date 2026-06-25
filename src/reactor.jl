# Reactor assembly. Phase 1 ships a minimal ChemPhaseSystem: the lowering entry
# point / wrapper around a Mechanism (+ optional Catalyst import). The full
# @mtkmodel reactor family (BatchReactor) arrives in Phase 2 (spec §5.5).

"A ChemPhaseSystem wraps a lowered MTK ODESystem with its source Mechanism and config."
struct ChemPhaseSystem
    sys::Any          # an mtkcompile'd ModelingToolkit.ODESystem
    mech::Mechanism
    config::MechanismConfig
end

"Build a ChemPhaseSystem from a Mechanism (lowers with the given config)."
function ChemPhaseSystem(mech::Mechanism; config::MechanismConfig=MechanismConfig())
    return ChemPhaseSystem(lower_to_mtk(mech; config=config), mech, config)
end

"Build a ChemPhaseSystem from a Catalyst ReactionSystem (imports, then lowers)."
function ChemPhaseSystem(rn; config::MechanismConfig=MechanismConfig())
    return ChemPhaseSystem(import_from_catalyst(rn); config=config)
end

"Convenience batch-reactor constructor. (stub — Phase 2)"
BatchReactor(args...; kwargs...) =
    error("BatchReactor: not implemented in Phase 1; see the Phase 2 reactor plan.")
