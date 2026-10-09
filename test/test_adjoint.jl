using Test
using ChemMechSim
using ChemMechSim: SpeciesData, ReactionData, ElementaryArrhenius, Irreversible
using ModelingToolkit
using ModelingToolkit: getname, parameters
using OrdinaryDiffEq

# ---- the golden toy used by every adjoint test: A --k1--> B --k2--> A ----
# Linear, so the exact solution and Jacobian are closed forms. STATE ORDER IS MTK's:
# unknowns(sys) = [B, A] (verified 2026-10-09) — never assume species-id order.
function ab_toy()
    spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B")
    rx1 = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(0.8, 0.0, 0.0), reverse_policy=Irreversible())
    rx2 = ReactionData(reactants=Dict(2 => 1.0), products=Dict(1 => 1.0),
                       kinetics=ElementaryArrhenius(0.3, 0.0, 0.0), reverse_policy=Irreversible())
    return Mechanism(species=[spA, spB], reactions=[rx1, rx2])
end

@testset "jac! dense branch (the adjoint call shape)" begin
    mech = ab_toy()
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:adjtest)
    prob = build_problem(r, Dict("A" => 1.0, "B" => 0.0), (0.0, 3.0); jac=true)

    u = [0.7, 0.3]                      # state order [B, A]
    p = prob.p
    # sparse path (forward-solver shape) — the reference:
    Js = copy(prob.f.jac_prototype)
    prob.f.jac(Js, u, p, 0.0)
    # dense path — THIS EXACT CALL was the 2026-10-08 probe's MethodError (SciMLSensitivity's
    # reverse pass allocates its own dense Matrix and hands it to f.jac):
    Jd = Matrix{Float64}(undef, 2, 2)
    prob.f.jac(Jd, u, p, 0.0)

    # 1. dense == sparse (bitwise by construction: fill-then-copyto!)
    @test Jd == Matrix(Js)
    # 2. == analytic Jacobian in [B,A] rows/cols:
    #    dB/dt = k1*A - k2*B;  dA/dt = -k1*A + k2*B;  k1=0.8, k2=0.3
    @test Jd ≈ [-0.3  0.8;
                0.3 -0.8] atol = 1e-12
    # 3. p as a plain Vector goes through _parameter_vector in the SAME values
    #    (MTKParameters' first field is [k_2_A, k_1_A] = [0.3, 0.8]):
    Jd2 = Matrix{Float64}(undef, 2, 2)
    prob.f.jac(Jd2, u, [0.3, 0.8], 0.0)
    @test Jd2 == Jd
end

@testset "flat parameter helpers (the verified gradient channel)" begin
    mech = ab_toy()
    r = BatchReactor(mech; mode=:kinetic, checks=false, name=:adjtest)
    sys = extract_system(r)

    for (name, prob) in (("fd(MTK)", build_problem(r, Dict("A"=>1.0,"B"=>0.0), (0.0,3.0); jac=false)),
                         ("sharded", build_problem(r, Dict("A"=>1.0,"B"=>0.0), (0.0,3.0); jac=true)))
        # 1. flat_params: plain Vector, parameters(sys) order — [k_2_A, k_1_A] = [0.3, 0.8]
        v0 = ChemMechSim.flat_params(prob)
        @test v0 isa Vector{Float64}
        @test v0 == [0.3, 0.8]
        @test length(v0) == length(parameters(sys))

        # 2. flat_to_mtk: same-type p; the generated rhs consumes it with CORRECT values.
        #    (A raw Vector fed to the sharded rhs is SILENTLY WRONG — probe 2026-10-09,
        #    du = ∓0.12 vs correct ±0.03 at u=[B,A]=[0.7,0.3]. This pins the fix.)
        pmtk = ChemMechSim.flat_to_mtk(prob, v0)
        @test pmtk isa typeof(prob.p)
        du = zeros(2)
        prob.f(du, [0.7, 0.3], pmtk, 0.0)
        @test du ≈ [0.03, -0.03] atol = 1e-12     # [dB, dA] = [k1·A−k2·B, −k1·A+k2·B]

        # 3. length guard
        @test_throws DimensionMismatch ChemMechSim.flat_to_mtk(prob, [1.0])
    end

    # 4. end-to-end remake channel on the SHARDED path vs the closed form
    #    A(t) = ss + (A0−ss)e^{−(k1+k2)t}, ss = k2/(k1+k2) (N = A0+B0 = 1)
    prob = build_problem(BatchReactor(ab_toy(); mode=:kinetic, checks=false, name=:adjtest),
                         Dict("A"=>1.0,"B"=>0.0), (0.0,3.0); jac=true)
    goldA(t; k1=0.8, k2=0.3, A0=1.0, N=1.0) = (s = k1 + k2; ss = k2/s*N; ss + (A0-ss)*exp(-s*t))
    sol = solve(remake(prob, p = ChemMechSim.flat_to_mtk(prob, [0.5, 0.2])),   # k2=0.5, k1=0.2
                FBDF(autodiff = false); reltol=1e-10, abstol=1e-12, saveat=[3.0])
    @test sol.u[end] ≈ [1 - goldA(3.0; k1=0.2, k2=0.5), goldA(3.0; k1=0.2, k2=0.5)] rtol = 1e-8
end
