module ChemMechSim

# Framework scaffold for ChemMechSim — MTK-first 气相化学机理建模框架.
# Design spec: docs/superpowers/specs/2026-06-23-chemmechsim-design.md

include("units.jl")
using .ChemUnits: canonical
export canonical

# —— Data layer (pure Julia, no MTK dependency) ——
include("data/types.jl")         # SpeciesID, SpeciesRole
export SpeciesID, SpeciesRole

include("data/thermo.jl")         # ThermoModel (abstract), ThermoDatabase
export ThermoModel, ThermoDatabase

include("data/kinetics.jl")       # AbstractKinetics hierarchy
export AbstractKinetics, AbstractFalloff,
       ElementaryArrhenius, ThirdBodyArrhenius,
       TroeFalloff, SRIFalloff, LindemannFalloff, PlogRate, ChebyshevRate,
       TroeParams, SRIParams

include("data/reaction.jl")       # ReactionData, ReverseRatePolicy
export ReverseRatePolicy, Irreversible, ExplicitReverse, ThermoReverse,
       ReactionData, ReactionMeta

include("data/species.jl")        # SpeciesData
export SpeciesData

include("data/mechanism.jl")      # Mechanism
export Mechanism

end # module
