using Test

# The preprocessor is an EXAMPLE tool, not part of the package — so it is `include`d here
# rather than reached through `ChemMechSim.`. Its `if abspath(PROGRAM_FILE) == @__FILE__`
# guard keeps the script body (file I/O over the 24 MB source mechanism) from running on
# include, so these tests exercise the two transforms as pure functions and need neither
# the source file nor a CLI invocation.
include(joinpath(@__DIR__, "..", "examples", "atmospheric", "tools", "flatten_photolysis.jl"))

@testset "photolysis flattening: J = l·cos(χ)^m·exp(−n/cos χ) at the chosen zenith" begin
    # l=6.073e-5, m=1.743, n=0.474 is the real MCM entry #36 (O3 => O1D).
    rx = Dict{Any,Any}("equation" => "O3 => O1D", "order" => 1,
                       "type" => "zenith-angle-photolysis",
                       "l" => 6.0730e-05, "m" => 1.7430e+00, "n" => 4.7400e-01)

    out = _flatten_photolysis(rx, 0.0)                      # χ = 0 → cos χ = 1
    @test !haskey(out, "type")                              # absent type == elementary
    @test out["rate-constant"]["type"] == "arrhenius"
    @test out["rate-constant"]["A"] ≈ 6.0730e-05 * exp(-0.474)  rtol=1e-12
    @test out["rate-constant"]["b"] == 0.0
    @test out["rate-constant"]["Ea"] == 0.0
    @test out["equation"] == "O3 => O1D"                    # equation preserved
    @test !haskey(out, "l") && !haskey(out, "m") && !haskey(out, "n")

    # χ = 90° → cos χ = 0 ≤ 1e-10 → J = 0, and crucially NOT NaN/Inf from exp(-n/0).
    out_edge = _flatten_photolysis(rx, pi / 2)
    @test out_edge["rate-constant"]["A"] == 0.0
    @test isfinite(out_edge["rate-constant"]["A"])

    # χ = 180° → cos χ = -1 (also below the cutoff).
    @test _flatten_photolysis(rx, pi)["rate-constant"]["A"] == 0.0

    # A nonzero zenith must actually change J — guards against χ being ignored.
    out_45 = _flatten_photolysis(rx, pi / 4)
    @test out_45["rate-constant"]["A"] != out["rate-constant"]["A"]
end

@testset "sigmoid pairing: k_total split into σ and (1−σ) branches" begin
    # The real MCM CH3O2+HO2 pair. Ground truth comes from the source file's own
    # '# Original Rate Expression' comments:
    #   CH3OOH = 3.8e-13·exp(780/T)·(1 − σ)
    #   HCHO   = 3.8e-13·exp(780/T)·σ
    # Negative A marks the (1−σ) branch, positive A the σ branch.
    T0 = 298.0
    a, b, c, d = 3.8e-13, 780.0, 498.0, -1160.0
    σ = 1.0 / (1.0 + c * exp(d / T0))
    ktot = a * exp(b / T0)

    neg = Dict{Any,Any}("equation" => "CH3O2 + HO2 => CH3OOH", "order" => 2,
                        "type" => "sigmoid-branching", "A" => -a, "B" => b,
                        "C" => c, "D" => d, "is_complement" => false)
    pos = Dict{Any,Any}("equation" => "CH3O2 + HO2 => HCHO", "order" => 2,
                        "type" => "sigmoid-branching", "A" => a, "B" => b,
                        "C" => c, "D" => d, "is_complement" => false)

    g = _flatten_sigmoid_pair([neg, pos], T0)
    @test length(g) == 2
    by_eq = Dict(r["equation"] => r for r in g)
    @test by_eq["CH3O2 + HO2 => CH3OOH"]["rate-constant"]["A"] ≈ ktot * (1 - σ)  rtol=1e-12
    @test by_eq["CH3O2 + HO2 => HCHO"]["rate-constant"]["A"] ≈ ktot * σ          rtol=1e-12
    # The two channels must reconstitute the total rate — this is the check that catches a
    # mis-assigned branch (the converter's own evaluator would give a NEGATIVE rate here).
    @test sum(r["rate-constant"]["A"] for r in g) ≈ ktot  rtol=1e-12
    @test all(r -> r["rate-constant"]["b"] == 0.0 && r["rate-constant"]["Ea"] == 0.0, g)

    # Malformed groups must ERROR rather than silently mis-assign a channel.
    @test_throws ErrorException _flatten_sigmoid_pair([neg, neg], T0)      # same sign
    bad_c = copy(pos); bad_c["C"] = 999.0
    @test_throws ErrorException _flatten_sigmoid_pair([neg, bad_c], T0)    # C mismatch
    bad_d = copy(pos); bad_d["D"] = -42.0
    @test_throws ErrorException _flatten_sigmoid_pair([neg, bad_d], T0)    # D mismatch
    bad_b = copy(pos); bad_b["B"] = 1.0
    @test_throws ErrorException _flatten_sigmoid_pair([neg, bad_b], T0)    # B mismatch
    @test_throws ErrorException _flatten_sigmoid_pair([neg], T0)           # not a pair
end
