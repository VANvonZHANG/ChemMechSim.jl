module ChemMechSim

# Framework scaffold for ChemMechSim — MTK-first 气相化学机理建模框架.
# Design spec: docs/superpowers/specs/2026-06-23-chemmechsim-design.md

include("units.jl")
using .ChemUnits: canonical
export canonical

# —— Data layer (pure Julia, no MTK dependency) ——
include("data/types.jl")         # SpeciesID, SpeciesRole, R_GAS, P_STD
export SpeciesID, SpeciesRole, R_GAS, P_STD

include("data/thermo.jl")         # ThermoModel (abstract), ThermoDatabase, NASA7
export ThermoModel, ThermoDatabase,
       NASA7,
       cp_over_R, h_over_RT, s_over_R, g_over_RT,
       cp_molar, h_molar, s_molar, g_molar,
       u_molar, cv_molar, u_over_RT, cv_over_R,
       KcData, equilibrium_constant, equilibrium_constant_dT

include("data/kinetics.jl")       # AbstractKinetics hierarchy
export AbstractKinetics, AbstractFalloff,
       ElementaryArrhenius, ThirdBodyArrhenius,
       TroeFalloff, SRIFalloff, LindemannFalloff, PlogRate, ChebyshevRate,
       TroeParams, SRIParams,
       plog_rate, plog_dkdT, plog_dkdP

include("data/reaction.jl")       # ReactionData, ReverseRatePolicy
export ReverseRatePolicy, Irreversible, ExplicitReverse, ThermoReverse,
       ReactionData, ReactionMeta

include("data/species.jl")        # SpeciesData
export SpeciesData

include("data/elements.jl")      # atomic masses + molecular_weight
export molecular_weight, ATOMIC_MASSES

include("data/mechanism.jl")      # Mechanism
export Mechanism

include("config.jl")             # MechanismConfig
export MechanismConfig

# —— MTK / Catalyst / solver (arrive in Phase 1; data layer above stays pure) ——
using ModelingToolkit, Catalyst, OrdinaryDiffEq

# —— Interface stubs (implemented in later phase plans) ——
include("lowering.jl")
include("catalyst_interop.jl")
include("reactor.jl")
include("validation.jl")
include("io/cantera_yaml.jl")
include("api.jl")

export lower_to_mtk, lower_reaction, rate_param, import_from_catalyst,
       ChemPhaseSystem, BatchReactor, convenience_config,
       validate, ValidationReport,
       simulate, build_problem, extract_system, generate_function, generate_jacobian,
       load_mechanism,
       symbolic_kf, symbolic_rate, needs_T, needs_P, paramspec, body, rate_constant,
       ParamRole, AFactor, KTemp, KValue, Plain, afactor, ktemp, kvalue, plain

end # module
