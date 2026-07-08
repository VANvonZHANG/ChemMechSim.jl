using Test
using ChemMechSim
using ChemMechSim: SpeciesData, SpeciesID, ReactionData, Mechanism, ElementaryArrhenius,
                   LindemannFalloff, load_mechanism, validate, BatchReactor, ChemPhaseSystem,
                   extract_system, simulate, lower_to_mtk, generate_jacobian, convenience_config,
                   u_molar
using ModelingToolkit: unknowns, getname, equations
using OrdinaryDiffEq: ODEFunction, FBDF, ReturnCode
using SparseArrays: nnz

# —— Phase 5b Task 1: characterize the two probe-unblocker fixes ———————————————

@testset "Phase 5b: CH2(S) labelled-state name parses (parens allowed)" begin
    # Regression: species names with parens (GRI30's singlet methylene CH2(S))
    # must tokenize. Before the fix, _parse_terms rejected them (regex [A-Za-z0-9]*).
    r = ChemMechSim._parse_equation("O + CH2(S) <=> H2 + CO")
    @test r.reactants == Dict("O" => 1.0, "CH2(S)" => 1.0)
    @test r.products  == Dict("H2" => 1.0, "CO" => 1.0)
    @test r.reversible
    @test !r.third_body
    # coefficient form + irreversible arrow
    r2 = ChemMechSim._parse_equation("2 CH2(S) => CH2")
    @test r2.reactants == Dict("CH2(S)" => 2.0)
    @test r2.products  == Dict("CH2" => 1.0)
    @test !r2.reversible
end

@testset "Phase 5b: LindemannFalloff lowers + matches analytic rate (F≡1)" begin
    # Regression: before the fix, LindemannFalloff had no _direct_kf method and crashed
    # at lowering ("_direct_kf: not implemented for LindemannFalloff"). GRI30 has 3 such
    # reactions (#12 O+CO(+M)<=>CO2(+M), #185 N2O(+M)<=>N2+O(+M), #237 H+HCN(+M)<=>H2CN(+M)).
    # A + B (+M) <=> C (+M), k0=1e6 (low), kinf=1e3 (high); every species third-body α=1.
    sp = [SpeciesData(id=SpeciesID(1), name="A"),
          SpeciesData(id=SpeciesID(2), name="B"),
          SpeciesData(id=SpeciesID(3), name="C")]
    eff = Dict(SpeciesID(1) => 1.0, SpeciesID(2) => 1.0, SpeciesID(3) => 1.0)
    kin = LindemannFalloff(ElementaryArrhenius(1.0e6, 0.0, 0.0),   # k0 (low)
                           ElementaryArrhenius(1.0e3, 0.0, 0.0),   # kinf (high)
                           eff)
    rxn = ReactionData(reactants=Dict(SpeciesID(1) => 1.0, SpeciesID(2) => 1.0),
                       products =Dict(SpeciesID(3) => 1.0), kinetics=kin)
    mech = Mechanism(species=sp, reactions=[rxn])
    phase = ChemPhaseSystem(mech); sys = extract_system(phase)
    @test length(unknowns(sys)) == 3                       # lowered without crashing
    idx = _state_index(sys)
    # Hand-written Lindemann RHS (F≡1): rate = kinf·(Pr/(1+Pr))·[A]·[B], Pr = k0·[M]/kinf
    function lind_rhs!(dc, c)
        A, B, C = c; Meff = A + B + C
        k0, kinf = 1.0e6, 1.0e3
        Pr = k0 * Meff / kinf
        rate = kinf * (Pr / (1 + Pr)) * A * B
        dc[1] = -rate; dc[2] = -rate; dc[3] = rate
        return dc
    end
    c = [0.7, 0.5, 0.3]                                    # id-ordered [A,B,C]
    u = zeros(3); for (i, n) in enumerate(("A", "B", "C")); u[idx[n]] = c[i]; end
    du = zeros(3); ODEFunction(sys)(du, u, _pvals(sys), 0.0)
    dc = lind_rhs!(zeros(3), c)
    for (i, n) in enumerate(("A", "B", "C"))
        @test du[idx[n]] ≈ dc[i]  rtol = 1e-7             # CS-order du vs id-order dc, by name
    end
end
