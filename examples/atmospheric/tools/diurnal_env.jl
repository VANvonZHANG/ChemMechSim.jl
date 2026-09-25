# The diurnal photolysis environment, ported VERBATIM from the converter repo
# (kpp-cantera-converter_geoschem). Pure functions only — this file is `include`d by
# the diurnal mode of mcm_box.jl, by throwaway probes, and by test/test_atmospheric_diurnal.jl,
# not depend on ChemMechSim or run anything at include time.
#
# Upstream roles: the converter's Cantera driver keeps a module-level ENV_STATE['zenith'] that
# its simulator updates once per 60-s step (cantera_sim/simulator.py run()); the custom rate
# class ZenithAnglePhotolysisRate reads it (custom_rates/generic.py). This file is the same
# physics with the file I/O removed.

"Solar zenith angle [rad] at simulation time t [s] — the converter's MCMDiurnalEnvironment clock
 (cantera_sim/environment.py:39): a triangular day, χ = 0 (overhead sun) at 12:00, χ = 180° at
 00:00, CLAMPED at 89.5° (max_sza_deg default). The clamp means midnight's cos χ is cos(89.5°) ≈
 8.7e-3, not 0 — reactions with n ≈ 0 keep a small residual night J. That is upstream's
 semantics, kept deliberately; do not 'fix' it here."
zenith_rad(t) = min(deg2rad(89.5), abs(2π * mod(t, 86400.0) / 86400.0 - π))

"cos(χ) floored at 0 — the rate class's own clamp (custom_rates/generic.py:82)."
cos_zenith(t) = max(0.0, cos(zenith_rad(t)))

"J = l·cz^m·exp(−n/cz) [s⁻¹ for first-order photolysis], 0 for cz ≤ 1e-10 — generic.py:102-104,
 never NaN/Inf. The frozen mode evaluates this law once at cz = 1 (the ZenithPhotolysis\nA-factor default, tools/mcm_rate_types.jl); the diurnal mode drives it per 60-s tick."
photolysis_J(l, m, n, cz) = cz <= 1e-10 ? 0.0 : l * cz^m * exp(-n / cz)
