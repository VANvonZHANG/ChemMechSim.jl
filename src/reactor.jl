# Reactor assembly stubs. The @mtkmodel reactors require ModelingToolkit, which
# arrives in the Phase 2 plan. (spec §5.5)

"Build a ChemPhaseSystem (the lowering product) from a Mechanism. (stub — Phase 2)"
ChemPhaseSystem(args...; kwargs...) =
    error("ChemPhaseSystem: not implemented in framework scaffold; see the Phase 2 plan.")

"Convenience batch-reactor constructor. (stub — Phase 2)"
BatchReactor(args...; kwargs...) =
    error("BatchReactor: not implemented in framework scaffold; see the Phase 2 plan.")
