using Test
using ChemMechSim
using ModelingToolkit
using ModelingToolkit: unknowns, getname
using OrdinaryDiffEq

@testset "third-body: [M]_eff rate (all-species convention) and trajectory" begin
    # H + O2 + M -> HO2 + M ; rate = k*[H]*[O2]*[M]_eff, [M]_eff = Σ all species (α=1 default)
    H = SpeciesData(id=1, name="H");  O2  = SpeciesData(id=2, name="O2")
    HO2 = SpeciesData(id=3, name="HO2"); M = SpeciesData(id=4, name="M")
    rxn = ReactionData(reactants=Dict(1 => 1.0, 2 => 1.0), products=Dict(3 => 1.0),
                       kinetics=ThirdBodyArrhenius(ElementaryArrhenius(2.0, 0.0, 0.0),
                                                   Dict(4 => 1.0)))
    mech = Mechanism(species=[H, O2, HO2, M], reactions=[rxn])
    sys = lower_to_mtk(mech)
    @test !any(p -> String(ModelingToolkit.getname(p)) == "T", ModelingToolkit.parameters(sys))  # base is constant
    # at H=1,O2=2,M=3,HO2=0: [M]_eff = 1+2+0+3 = 6 -> rate = 2*1*2*6 = 24 -> dH=-24, dHO2=+24, dM=0
    idx = _state_index(sys); u = zeros(4)
    u[idx["H"]] = 1.0; u[idx["O2"]] = 2.0; u[idx["M"]] = 3.0
    du = zeros(4); ODEFunction(sys)(du, u, _pvals(sys), 0.0)
    @test du[idx["H"]]   ≈ -24.0
    @test du[idx["HO2"]] ≈  24.0
    @test du[idx["M"]]   ≈ 0.0
    # efficiency weighting alpha_M=2: [M]_eff = 1+2+0+2*3 = 9 -> rate = 2*1*2*9 = 36
    rxn2 = ReactionData(reactants=Dict(1 => 1.0, 2 => 1.0), products=Dict(3 => 1.0),
                        kinetics=ThirdBodyArrhenius(ElementaryArrhenius(2.0, 0.0, 0.0), Dict(4 => 2.0)))
    sys2 = lower_to_mtk(Mechanism(species=[H, O2, HO2, M], reactions=[rxn2]))
    i2 = _state_index(sys2); u2 = zeros(4)
    u2[i2["H"]] = 1.0; u2[i2["O2"]] = 2.0; u2[i2["M"]] = 3.0
    du2 = zeros(4); ODEFunction(sys2)(du2, u2, _pvals(sys2), 0.0)
    @test du2[i2["H"]] ≈ -36.0
end

@testset "Troe falloff: lowers (T-dependent), solves, k matches the formula" begin
    H = SpeciesData(id=1, name="H");  O2  = SpeciesData(id=2, name="O2")
    HO2 = SpeciesData(id=3, name="HO2"); M = SpeciesData(id=4, name="M")
    Ah, bh, Eah = 1.0e15, -1.0, 0.0      # high-pressure limit (kinf)
    Al, bl, Eal = 1.0e20, -1.4, 0.0      # low-pressure limit  (k0)
    α, T1, T2, T3 = 0.5, 1.0e-30, 1.0e30, 1.0e30
    rxn = ReactionData(reactants=Dict(1 => 1.0, 2 => 1.0), products=Dict(3 => 1.0),
        kinetics=TroeFalloff(ElementaryArrhenius(Al, bl, Eal),   # low_rate  (k0)
                             ElementaryArrhenius(Ah, bh, Eah),   # high_rate (kinf)
                             Dict(4 => 1.0), TroeParams(α, T1, T2, T3)))
    mech = Mechanism(species=[H, O2, HO2, M], reactions=[rxn])
    sys = lower_to_mtk(mech)
    @test any(p -> String(ModelingToolkit.getname(p)) == "T", ModelingToolkit.parameters(sys))  # Troe is T-dependent
    phase = ChemPhaseSystem(mech)
    sol = simulate(phase, (0.0, 1.0e-6);
                   u0=Dict("H" => 1.0, "O2" => 1.0, "HO2" => 0.0, "M" => 1.0e3),
                   params=[_param(sys, "T") => 1000.0], reltol=1e-9, abstol=1e-12,
                   dt=1.0e-7)
    @test all(isfinite, sol.u[end])

    # k from the lowered RHS matches the Troe formula at one point (§3.4 #2)
    idx = _state_index(sys)
    Tv, Hv, O2v, Mv = 1000.0, 2.0, 3.0, 5.0
    u = zeros(4); u[idx["H"]] = Hv; u[idx["O2"]] = O2v; u[idx["M"]] = Mv
    du = zeros(4); ODEFunction(sys)(du, u, _pvals(sys) |> x -> replace_T(x, sys, Tv), 0.0)
    k_cs = -du[idx["H"]] / (Hv * O2v)
    # hand-computed Troe k (see _direct_rate(::TroeFalloff)); [M]_eff = Hv+O2v+0+Mv
    kinf = Ah * Tv^bh * exp(-Eah / (8.314 * Tv)); k0 = Al * Tv^bl * exp(-Eal / (8.314 * Tv))
    Meff = Hv + O2v + Mv; Pr = k0 * Meff / kinf
    Fc = (1 - α) * exp(-Tv / T3) + α * exp(-Tv / T1) + exp(-Tv / T2)
    lFc, lPr = log10(Fc), log10(Pr)
    cT = -0.4 - 0.67 * lFc; NT = 0.75 - 1.27 * lFc
    f1 = lPr + cT; f2 = NT - 0.14 * f1
    F = 10^(lFc / (1 + (f1 / f2)^2))
    @test k_cs ≈ kinf * (Pr / (1 + Pr)) * F  rtol = 1e-6
end
