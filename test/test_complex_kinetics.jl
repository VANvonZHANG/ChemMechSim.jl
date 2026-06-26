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
