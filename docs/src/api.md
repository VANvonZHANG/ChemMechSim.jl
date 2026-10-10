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

## Adjoint gradients

The flat parameter channel and the symbol-indexed objective helper — see the README
"Adjoint gradients" recipe for the full pinned configuration.

```@docs
flat_params
flat_to_mtk
state_index
```

## Lowering & the rate-law protocol

```@docs
lower_to_mtk
lower_reaction
symbolic_kf
symbolic_rate
rate_constant
needs_T
needs_P
rate_param
import_from_catalyst
ParamRole
AFactor
KTemp
KValue
Plain
paramspec
body
afactor
ktemp
kvalue
plain
```

## I/O

```@docs
load_mechanism
convert_afactor
ea_to_J_per_mol
```

## Validation

```@docs
ValidationReport
validate
```
