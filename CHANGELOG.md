# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-10-10

### Added

- Adjoint (reverse-mode) gradients on both `build_problem` paths, verified against
  analytic gold: the reaction-sharded `jac!` accepts the dense `AbstractMatrix` buffer
  SciMLSensitivity's reverse pass hands it (dispatch-shadowed; the sparse hot path is
  byte-identical).
- `flat_params(prob)` / `flat_to_mtk(prob, v)` — the verified parameter-gradient
  channel (`remake(p=Vector)` is broken on this ModelingToolkit version; a raw Vector
  fed to the generated RHS is silently wrong — the helpers route around both, and their
  docstrings + README "Adjoint gradients" record the pinned recipe).
- `state_index(sys, name)` — symbol-indexed objectives; the lowered state order is
  ModelingToolkit's, not species-id order.
- Golden adjoint test suite (`test_adjoint.jl`, `test_adjoint_sensitivity.jl`):
  u₀ and parameter gradients vs independent ForwardDiff gold + pinned literals, on both
  paths. SciMLSensitivity/Zygote et al. are test-target extras only; `[deps]` is
  unchanged.

## [1.0.0] - 2026-10-05

Initial release.

### Added

- Mechanism data layer on a unit-aware quantities layer (`ChemUnits`, built on
  DynamicQuantities): `SpeciesData` with NASA7 two-range thermo, `ReactionData`
  with stoichiometry and metadata.
- `load_mechanism(path)` parsing the Cantera-YAML subset emitted by `ck2yaml`,
  plus KPP/MCM converter dialect extensions (constant-cp thermo,
  `activation-energy: K`, `quantity: molec`); A-factors and activation energies
  are unit-converted at parse time.
- Built-in rate laws: elementary Arrhenius (with NASA7-based
  equilibrium-constant reverse rates), three-body, falloff
  (Lindemann / Troe / SRI), and pressure-dependent Arrhenius (PLOG).
- Extensible rate-law protocol: custom rate types plug in via
  `rate_type_handlers`; demonstrated on MCM-style sigmoid expressions in the
  examples.
- Four batch-reactor modes behind `BatchReactor(mech; mode=...)`: `:kinetic`
  (pure isothermal kinetics), `:fixedT` (isothermal, ideal-gas pressure as an
  observed output), `:adiabatic_constV` (T as a state, U conserved),
  `:adiabatic_constP` (moles as states, H conserved).
- Configurable lowering via `MechanismConfig` (energy / constraint / EOS /
  reverse-rate / state-basis switches) and selectable Jacobian strategies
  (`:auto`, `:reaction_sharded`, `:mtk`, `:none`).
- Solve API (`build_problem`, `simulate`) on ModelingToolkit/Catalyst lowering
  and OrdinaryDiffEq solvers, with solver and linear-solver guidance for stiff
  chemistry (FBDF; KLU / UMFPACK / dense LU / MUMPS trade-offs documented).
- Large-mechanism support: shared third-body-efficiency variables, PLOG
  lowering, and reaction-sharded Jacobians — parsed and solved up to Aramco 3.0
  (581 species) and FFCM-2.
- Validation harness against Cantera (`examples/validation/`): a per-mechanism
  ignition-delay workflow and combined-species trajectories (GRI-30 Δt_ign
  0.065%, FFCM-2 0.168%, Aramco 3.0 0.333%); PLOG rate constants validated
  against Cantera in the test suite.
- Minimal dependency footprint: the package depends only on what `src/` uses;
  example scripts (plotting, benchmarks, validation) run in their own `examples/`
  environment.
- Documentation site with automated deployment: getting started, ignition and
  atmospheric box tutorials (diurnal photolysis), mechanism format, solver
  guide, validation, performance, and full API reference.

[unreleased]: https://github.com/VANvonZHANG/ChemMechSim.jl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/VANvonZHANG/ChemMechSim.jl/releases/tag/v1.0.0
