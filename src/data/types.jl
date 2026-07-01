# Cross-cutting data-layer aliases. Pure Julia (no MTK/unit dependency).

# Stable integer index referencing a species inside a Mechanism.
# Reactants/products/efficiencies reference species by this id (not by nested
# SpeciesData objects), avoiding duplication.
const SpeciesID = Int

# Role of a species within a simulation. Valid values:
#   :dynamic         — evolved by the ODE (default)
#   :algebraic_qssa  — quasi-steady-state algebraic variable (reserved; Phase 7)
#   :constant_pool   — held at a fixed concentration
#   :bath_gas        — third-body bath gas
const SpeciesRole = Symbol

# Molar gas constant (spec §5.6.2: R = 8.314 J/(mol·K)).
# Lives in the data layer so src/data/thermo.jl can use it WITHOUT `using` MTK;
# types.jl is included before both thermo.jl and lowering.jl.
const R_GAS = 8.314   # J/(mol·K)

# Standard-state pressure for NASA-polynomial thermo (spec §4.2 / §5.6).
# 1 bar = 1e5 Pa (modern Cantera/GRI30 ideal-gas convention; CHEMKIN historical = 1 atm = 101325).
# Revisit at Phase 5 (Cantera validation) if ignition-delay comparison shows a systematic offset.
const P_STD = 1.0e5   # Pa
