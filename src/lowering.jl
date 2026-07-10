# Lowering pipeline: Mechanism + config → MTK ODESystem (spec §5.4, §5.6).
# Unit-aware: @species carry [unit=conc], T [unit=K], and each reaction's rate
# constant k is a unit-bearing parameter (stoichiometrically derived unit), so
# MTK's dimension check fires at System construction. The Catalyst mass-action
# backend (catalyst_lowering via oderatelaw) shares the same unit-bearing k.
#
# Split into focused files under src/lowering/, included here in dependency order
# into the ChemMechSim module (no submodules — just file organization, matching
# the src/data/ pattern). All names stay module-private (prefixed `_`) except the
# public exports declared in src/ChemMechSim.jl. Include order: units → state →
# kinetics → thermo → energy → core (function-to-function references resolve at
# call time; only const/struct definitions need ordered includes, and those live
# in src/data/, loaded before this file).
#
# R_GAS / P_STD live in src/data/types.jl (visible here because types.jl is
# included before lowering.jl in src/ChemMechSim.jl).

using DynamicQuantities     # u"..." unit literals used across the lowering files

include("lowering/units.jl")
include("lowering/state.jl")
include("lowering/kinetics.jl")
include("lowering/thermo.jl")
include("lowering/core.jl")
