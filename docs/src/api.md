# API reference

Every exported symbol, grouped by source layout. The ten entry points most users need:
`load_mechanism`, `BatchReactor`, `simulate`, `build_problem`, `extract_system`,
`lower_to_mtk`, `MechanismConfig`, `convenience_config`, `validate` (plus the
`rate_type_handlers` keyword of `load_mechanism` for parsing custom rate types).

## Data layer

```@docs
SpeciesID
SpeciesRole
R_GAS
P_STD
ATOMIC_MASSES
molecular_weight
SpeciesData
ReactionData
ReactionMeta
ReverseRatePolicy
Irreversible
ExplicitReverse
ThermoReverse
Mechanism
```

## Thermodynamics

```@docs
ThermoModel
NASA7
KcData
ThermoDatabase
cp_over_R
h_over_RT
s_over_R
g_over_RT
u_over_RT
cv_over_R
cp_molar
h_molar
s_molar
g_molar
u_molar
cv_molar
equilibrium_constant
equilibrium_constant_dT
```

## Kinetics hierarchy

```@docs
AbstractKinetics
ElementaryArrhenius
ThirdBodyArrhenius
AbstractFalloff
LindemannFalloff
TroeFalloff
TroeParams
SRIFalloff
SRIParams
PlogRate
plog_rate
plog_dkdT
plog_dkdP
ChebyshevRate
```

## Units

```@docs
ChemMechSim.ChemUnits
ChemMechSim.ChemUnits.canonical
```

## Configuration

```@docs
MechanismConfig
convenience_config
```

## Solve API

```@docs
simulate
build_problem
extract_system
generate_function
generate_jacobian
ChemPhaseSystem
BatchReactor
```
