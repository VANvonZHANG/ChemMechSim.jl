# Task 8 Report: H2-O2 acceptance + handwritten-RHS comparison + Jacobian (§3.4 #5/#6)

## Status: DONE

## Summary
7-species H2-O2 subset (3 elementary/Catalyst + 2 third-body/direct + 1 Troe/direct) lowered through the mixed path into ONE ODESystem, verified against an independent handwritten RHS at T=1000 K, plus end-to-end integration solve and symbolic Jacobian generation.

## Files modified
- `test/test_h2o2.jl` — new file: `_h2o2_mech()` (7 species, 6 reactions), `_h2o2_rhs!` (independent handwritten reference), 3 testsets (§3.4 #6 pointwise match, #1/#2/#3 lower+solve+trajectory, #5 Jacobian).
- `test/runtests.jl` — added `include("test_h2o2.jl")` after `include("test_complex_kinetics.jl")`.
- `examples/h2o2_subset.jl` — new demo: 2-reaction subset (1 elementary/Catalyst + 1 Troe/direct) through BatchReactor, prints equations + 4 concentrations at t=1e-5.

## Bug fixes in the provided spec
1. **`_h2o2_rhs!` dc[6] (OH) missing -r5 term:** The brief's handwritten RHS had `dc[6] = 2r1 - r2 + r3`, but R5 is `H+OH+M->H2O+M` which consumes OH (stoich -1). Corrected to `dc[6] = 2r1 - r2 + r3 - r5`. Without this fix, the lowered system (correctly including -r5) mismatched by exactly r5 = 600 at the test point (rel err ~2e-4, above rtol 1e-6).
2. **Second testset missing `idx = _state_index(sys)`:** The brief's second testset used `idx[name]` on the trajectory-comparison loop but never defined `idx` within its scope (Julia `@testset` blocks do not inherit locals from sibling testsets). Added the definition.

## Verification
- **§3.4 #6 (pointwise RHS match):** 7/7 pass. At c=[2,1.5,0.5,0.3,0.1,0.4,0.2], T=1000 K: max |du-dc| mismatch = 4.66e-10 (well within rtol 1e-6).
- **§3.4 #1/#2/#3 (lower + solve + trajectory):** 16/16 pass. System has 7 unknowns; `simulate` to t=1e-6 returns all-finite; trajectories at t=1e-7 and t=1e-6 match handwritten RHS within rtol 1e-6.
- **§3.4 #5 (Jacobian):** 1/1 pass. `ModelingToolkit.calculate_jacobian(sys)` returns a 7x7 symbolic matrix.
- **Full suite:** all tests pass, 0 Julia warnings/deprecations.
- **Demo:** `julia --project=. examples/h2o2_subset.jl` prints reactor, 7 equations (including full Troe symbolic expression), and 4 finite concentrations (H2=0.503266, O2=0.002518, HO2=0.000748, OH=2.993467 at t=1e-5). No stack trace or NaN.

## Notes
- TroeFalloff constructed `TroeFalloff(low_rate, high_rate, eff, troe)` — low_rate FIRST (k0=1e9), matching the struct field order and the lowering's `kin.high_rate`->kinf / `kin.low_rate`->k0 mapping.
- The handwritten RHS and the mechanism agree exactly after the dc[6] fix — the lowered system was correct; the spec's reference function had the stoichiometry omission.

- Commit: `773504f` — `test(phase25b): H2-O2 mixed-lowering acceptance vs handwritten RHS + Jacobian (§3.4 #5/#6)`

## Fix I1/I2/M5 (whole-branch review)

**I1 — Guard ThermoReverse against Δν≠0 (silent-wrong-K_c trap):** `_reverse_rate(::ThermoReverse, …)` in `src/lowering.jl` now computes `dnu = sum(values(rx.products)) - sum(values(rx.reactants))` and errors if `dnu != 0`, with a message explaining the missing `(P°/RT)^Δν` concentration-basis factor is deferred. This mirrors the existing helpful-error pattern in `_net_rate`. The existing Δν=0 test (`A<->B`) still passes; a new test (`A+B<->C`, Δν=-1) confirms the guard fires via `@test_throws ErrorException`.

**I2 — Robust species-by-id lookup:** Added `species_by_id(mech, sid)` to `src/data/mechanism.jl` as a pure-data helper using `findfirst(sp -> sp.id == sid, mech.species)`. Both `src/lowering.jl` (`_species_by_id`, used by `_thermo_of`) and `src/validation.jl` (`_element_totals`) now call this shared helper instead of the fragile `mech.species[sid]` index assumption. Non-contiguous/non-1-based species ids can no longer silently look up the wrong species.

**M5 — Float-robust element-count comparison:** `_check_element_conservation` in `src/validation.jl` now uses `_dicts_approx_equal(a, b)` (same keys + all values `isapprox(…; atol=1e-9)`) instead of exact `Dict{String,Float64}` equality, so float roundoff from non-integer stoichiometry cannot cause false element-imbalance errors.

**Files modified:** `src/lowering.jl`, `src/validation.jl`, `src/data/mechanism.jl`, `test/test_complex_kinetics.jl`.

**Verification:** Full suite `Pkg.test()` — all tests pass, 0 warnings. New testset "ThermoReverse: Δν≠0 is rejected" passes (1/1).

- Commit: `ce34d34` — `fix(phase25b): guard ThermoReverse Δν≠0 + robust species-id lookup + float-robust validation (review I1/I2/M5)`
