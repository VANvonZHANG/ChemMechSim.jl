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

@testset "sigmoid entries: per-entry evaluator preserving the sign of A" begin
    # REGRESSION (found in review): the converter emits MCM's CH3O2+HO2 as THREE entries, not
    # two. MCM's own .eqn is the ground truth:
    #   <53>   CH3O2 + HO2 = CH3OOH : 3.8E-13*EXP(780./TEMP)*(1.-1./(1.+498.*EXP(-1160./TEMP)))
    #   <3778> CH3O2 + HO2 = HCHO  : 3.8E-13*EXP(780./TEMP)*(1./(1.+498.*EXP(-1160./TEMP)))
    # which the converter splits into a plain k_total term plus a SIGNED sigmoid correction:
    #   plain arrhenius  A=+3.8e-13, Ea=-780        -> k_total
    #   sigmoid          A=-3.8e-13, is_complement=F -> -k_total·σ
    #   sigmoid          A=+3.8e-13, is_complement=F -> +k_total·σ
    # ChemMechSim sums duplicate equations natively, so the correct transform is PER-ENTRY with
    # A's sign preserved. An earlier version instead paired the two sigmoids and used |A| as
    # k_total, which double-counted the reaction (2.0x the total sink, 2.1x CH3OOH). The old
    # test could not see it because its fixture omitted the plain sibling — it encoded the same
    # misreading as the implementation.
    T0 = 298.0
    a, b, c, d = 3.8e-13, 780.0, 498.0, -1160.0
    σ = 1.0 / (1.0 + c * exp(d / T0))
    ktot = a * exp(b / T0)

    plain = Dict{Any,Any}("equation" => "CH3O2 + HO2 => CH3OOH", "order" => 2,
                          "duplicate" => true,
                          "rate-constant" => Dict{Any,Any}("type" => "arrhenius",
                                                           "A" => a, "b" => 0.0, "Ea" => -b))
    sig_neg = Dict{Any,Any}("equation" => "CH3O2 + HO2 => CH3OOH", "order" => 2,
                            "duplicate" => true, "type" => "sigmoid-branching",
                            "A" => -a, "B" => b, "C" => c, "D" => d, "is_complement" => false)
    sig_pos = Dict{Any,Any}("equation" => "CH3O2 + HO2 => HCHO", "order" => 2,
                            "type" => "sigmoid-branching",
                            "A" => a, "B" => b, "C" => c, "D" => d, "is_complement" => false)

    out = _flatten_mechanism!(Dict{Any,Any}("reactions" => [plain, sig_neg, sig_pos],
                                            "phases" => [Dict{Any,Any}("species" => ["M"])],
                                            "species" => [Dict{Any,Any}("name" => "M")]),
                             0.0, T0).reactions

    # Evaluate each entry's rate AT T0 before summing — the plain sibling is an Arrhenius with
    # Ea = -780 K, so its rate is A·exp(+780/T0) = 5.206e-12, not its raw A = 3.8e-13. (Ea is in
    # kelvin here, per the file's `activation-energy: K` header, so the law is A·T^b·exp(−Ea/T).)
    rate_T0(r) = (rc = r["rate-constant"];
                  Float64(rc["A"]) * T0^Float64(rc["b"]) * exp(-Float64(rc["Ea"]) / T0))

    # Same-equation entries must SUM to MCM's own rate, and the two channels to k_total.
    ch3ooh = sum(rate_T0(r) for r in out if r["equation"] == "CH3O2 + HO2 => CH3OOH")
    hcho = sum(rate_T0(r) for r in out if r["equation"] == "CH3O2 + HO2 => HCHO")
    @test ch3ooh ≈ ktot * (1 - σ)  rtol=1e-12
    @test hcho ≈ ktot * σ          rtol=1e-12
    @test ch3ooh + hcho ≈ ktot     rtol=1e-12       # total sink == MCM's total
    @test ch3ooh ≈ 4.739565564253275e-12  rtol=1e-9 # and not 2x it (the shipped regression)
end

@testset "is_complement selects the other branch" begin
    T0 = 298.0
    a, b, c, d = 2.5e-12, 500.0, 300.0, -900.0
    σ = 1.0 / (1.0 + c * exp(d / T0))
    k = a * exp(b / T0)
    mk(comp) = Dict{Any,Any}("equation" => "X => Y", "order" => 1,
                             "type" => "sigmoid-branching", "A" => a, "B" => b,
                             "C" => c, "D" => d, "is_complement" => comp)
    @test Float64(_flatten_sigmoid(mk(false), T0)["rate-constant"]["A"]) ≈ k * σ     rtol=1e-12
    @test Float64(_flatten_sigmoid(mk(true),  T0)["rate-constant"]["A"]) ≈ k * (1 - σ) rtol=1e-12
end
