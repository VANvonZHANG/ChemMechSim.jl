# Mechanism format

`load_mechanism(path)` reads the **Cantera-YAML** subset that ChemMechSim lowers —
the same dialect Cantera's `ck2yaml` emits, plus the KPP/MCM converter's dialect
extensions. Everything below is what `src/io/cantera_yaml.jl` actually parses.

## Document structure

```yaml
units: {length: cm, time: s, quantity: mol, activation-energy: cal/mol}

phases:
- name: gri30
  thermo: ideal-gas
  # species: list of species names (order fixes the 1-based SpeciesID)
  # elements: optional — the KPP/MCM converter omits it

species:
- name: CH4
  composition: {C: 1, H: 4}
  thermo: …            # NASA7 two-range polynomial

reactions:
- equation: CH4 + O2 <=> CH3 + HO2
  rate-constant: [A, b, Ea]
  # type: elementary (default) | three-body | falloff | pressure-dependent-Arrhenius
  # duplicate: true  (bookkept on ReactionMeta)
  # three-body/falloff rows: efficiencies: {SP: α, …} and default-efficiency: α
  #   (every species the map does not list uses default-efficiency, 1.0 unless set)
```

- `units` drives unit conversion of A-factors and activation energies at parse time
  (`convert_afactor`, `ea_to_J_per_mol`).
- Species `composition` feeds `molecular_weight`; `elements` in the phase block is
  optional.
- Thermo: NASA7 two-coefficient-range polynomials. The MCM dialect's
  `constant-cp` thermo and `activation-energy: K` / `quantity: molec` units are
  handled.

## Built-in reaction types

| YAML `type` | ChemMechSim type |
|---|---|
| `elementary` (default) | `ElementaryArrhenius` |
| `three-body` | `ThirdBodyArrhenius` |
| `falloff` (Lindemann / Troe / SRI sub-forms) | `LindemannFalloff` / `TroeFalloff` / `SRIFalloff` |
| `pressure-dependent-Arrhenius` | `PlogRate` |

`ChebyshevRate` exists as a data-layer struct but is **not parsed** today.

`three-body` and `falloff` rows accept the Cantera `efficiencies` map (species →
third-body enhancement factor α); `default-efficiency` (default `1.0`) sets α for
every species the map does not list.

Reversible arrows (`<=>`) lower via the reaction's reverse policy (thermodynamic
`K_c` from NASA7 by default); explicit reverse rate-constants are read where
Cantera's format carries them.

## Negative product coefficients (ensemble bookkeeping)

The product side may carry **negative stoichiometric coefficients** — the
KPP/MCM-dialect form used for ensemble/bookkeeping species (e.g. NCAR MM's X*OO
radicals, consumed by the reaction that creates the observable products):

```yaml
- equation: A + 2.0 B => 1.0 A - 2.0 XAOO     # XAOO consumed as an ensemble member
  rate-constant: {A: 1.0e-12, b: 0.0, Ea: 0.0}
```

A whitespace-bounded `-` separates product terms exactly like `+` does, and an
explicit per-term sign (`B + -2.0 XAOO`) is accepted too; species names contain
no spaces, so this can never split a hyphenated name like `C4H5-2`. Semantics:
the rate law is mass-action over the **reactants only**, and a negative product
coefficient contributes a negative stoichiometric term to that species' ODE
(`d[XAOO]/dt = -2·rate` above), mirroring how the NCAR MM gear integrator
accumulates `ct += sc·rt`. Two restrictions, both loud errors rather than
silent mis-lowering: reactant-side coefficients must stay positive (the rate
law and the A-factor unit conversion both consume Σν_reactants), and reactions
with negative product stoichiometry must be **irreversible** (`=>`) — a reverse
rate is mass-action over the forward products, and a negative exponent would
divide by a concentration.

## Unsupported rate types — extend, don't fork

Reactions whose `type` has no parser are **skipped with one aggregated warning**
naming the counts — a mechanism with unhandled types announces itself instead of
silently loading a chemically wrong subset. Handle them without touching the
package: the `rate_type_handlers` keyword maps type strings to parser functions,

```julia
mech = load_mechanism("mcm.yaml";
    rate_type_handlers = Dict("Arrhenius-Photo" => my_parser))
```

with handler signature `(rxn_dict, reactants, name_to_id, ctx) -> AbstractKinetics`
(entries override built-ins; use `convert_afactor` / `ea_to_J_per_mol` on `ctx`).
The worked end-to-end case — two MCM rate types, `zenith-angle-photolysis` and
`sigmoid-branching`, parsed example-side — is
`examples/atmospheric/tools/mcm_rate_types.jl`, driven by
`examples/atmospheric/mcm_box.jl`.

## Fixtures in this repository

`examples/mechanism/` carries published mechanisms used by the validation and
performance scripts: `gri30.yaml` (GRI-30), `h2o2.yaml` (H2-O2), `FFCM2.yaml`
(FFCM-2), `AramcoMech3.0.{yaml,MECH,THERM,TRAN}` (Aramco 3.0, Cantera YAML plus the
original CHEMKIN files).
