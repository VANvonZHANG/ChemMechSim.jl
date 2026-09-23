using Test

# The diurnal environment is an EXAMPLE tool — `include`d like the preprocessor tests, for the
# same reason (see test_atmospheric_preprocessor.jl): pure functions, no mechanism needed.
include(joinpath(@__DIR__, "..", "examples", "atmospheric", "tools", "diurnal_env.jl"))

@testset "triangular zenith clock (MCMDiurnalEnvironment port)" begin
    @test zenith_rad(43200.0) ≈ 0.0 atol = 1e-12        # 12:00 — overhead sun (2π·0.5 − π ≈ 4e-16)
    @test zenith_rad(32400.0) ≈ π / 4                   # 10:00 — unclamped, mid-morning
    # 06:00/18:00 sit at exactly 90°, already past the 89.5° clamp — and so does 06:01 (the
    # clamp lifts only at ~06:01:54). ALL night shares the clamped value, so cos(χ) is
    # +0.0087 all night: never negative, and never below the rate class's 1e-10 cutoff.
    @test zenith_rad(21600.0) == deg2rad(89.5)          # 06:00
    @test zenith_rad(64800.0) == deg2rad(89.5)          # 18:00
    @test zenith_rad(0.0) == deg2rad(89.5)              # midnight: 180° clamped
    @test zenith_rad(3600.0) == deg2rad(89.5)           # 01:00
    @test zenith_rad(86399.0) == deg2rad(89.5)          # 23:59:59
    # cz rises after the clamp lifts (06:01:54) and falls toward dusk
    @test cos_zenith(22000.0) > cos_zenith(21600.0)
    @test cos_zenith(64000.0) > cos_zenith(64800.0)
    @test cos_zenith(43200.0) == 1.0                    # noon
    @test cos_zenith(32400.0) ≈ 0.7071067811865476      # 10:00 = cos(π/4)
    @test cos_zenith(0.0) ≈ 0.008726535498373897        # cos(89.5°) — the night floor
    # the clock repeats daily, and is symmetric about noon (≈: the two float paths differ in
    # their intermediate roundings)
    @test cos_zenith(12345.0) == cos_zenith(12345.0 + 86400.0)
    @test cos_zenith(43200.0 - 7777.0) ≈ cos_zenith(43200.0 + 7777.0)
end

@testset "photolysis rate law J = l·cz^m·exp(−n/cz)" begin
    # l/m/n are the real MCM J_NO2 entry (converter YAML reaction #39). Expected values pinned
    # from an independent evaluation — NOT recomputed from the same expression as the function.
    l, m, n = 1.1650e-02, 2.4400e-01, 2.6700e-01
    @test photolysis_J(l, m, n, 1.0) ≈ 0.008920091282571607             rtol = 1e-12
    @test photolysis_J(l, m, n, 0.7071067811865476) ≈ 0.007338593498667751 rtol = 1e-12
    # night cutoff: exactly 0 (not NaN/Inf) at and below cz = 1e-10
    @test photolysis_J(l, m, n, 1e-10) == 0.0
    @test photolysis_J(l, m, n, 0.0) == 0.0
    @test isfinite(photolysis_J(l, m, n, 1e-3))
    # the night floor cz = cos(89.5°) kills an n>0 rate exponentially (exp(−30.6))...
    @test photolysis_J(l, m, n, 0.008726535498373897) < 1e-12 * photolysis_J(l, m, n, 1.0)
    # ...but an n=0 reaction KEEPS a residual night J (upstream semantics, kept deliberately —
    # see the diurnal_env.jl note; do not "fix")
    @test photolysis_J(1.0, 1.0, 0.0, 0.008726535498373897) ≈ 0.008726535498373897
end
