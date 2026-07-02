using Test
using ChemMechSim
using ModelingToolkit
using ModelingToolkit: unknowns, parameters, getname, getdefault, observed
using OrdinaryDiffEq: ODEFunction, Rodas5P, ReturnCode

const R4A = 8.314

"Exothermic A -> B with NASA7 thermo (constant cp = 2.5 R); B is HEAT J/mol lower in enthalpy."
function _exothermic_ab_mech(; A=1.0, b=0.0, Ea=0.0, HEAT=10000.0)
    a1 = 2.5; a6A = -a1 * 298.15; a6B = a6A - HEAT / R4A
    nA = NASA7((a1,0,0,0,0,a6A,0.0),(a1,0,0,0,0,a6A,0.0), 200.0, 1000.0, 3500.0)
    nB = NASA7((a1,0,0,0,0,a6B,0.0),(a1,0,0,0,0,a6B,0.0), 200.0, 1000.0, 3500.0)
    spA = SpeciesData(id=1, name="A", thermo=nA); spB = SpeciesData(id=2, name="B", thermo=nB)
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(A, b, Ea))
    Mechanism(species=[spA, spB], reactions=[rxn])
end

@testset ":adiabatic_constV: dT/dt analytic + T rises + U conserved + P observed" begin
    HEAT = 10000.0
    mech = _exothermic_ab_mech(; A=1.0, HEAT=HEAT)         # k = 1 s⁻¹ (constant)
    phase = ChemPhaseSystem(mech; config=convenience_config(:adiabatic_constV))
    sys = extract_system(phase); idx = _state_index(sys)
    # analytic dT/dt at t=0: r = k·A = 1·1 = 1; Σc·cv = (A+B)·(cp/R−1)·R = 1·1.5·R
    dT_analytic = 1.0 * HEAT / (1.0 * 1.5 * R4A)
    f = ODEFunction(sys); du = zeros(length(idx))
    u0 = zeros(length(idx)); u0[idx["A"]] = 1.0; u0[idx["B"]] = 0.0; u0[idx["T"]] = 300.0
    f(du, u0, [getdefault(p) for p in parameters(sys)], 0.0)
    @test du[idx["T"]] ≈ dT_analytic  rtol=1e-9
    # solve: T rises, A depletes, B grows
    sol = simulate(phase, (0.0, 5.0);
                   u0=Dict("A"=>1.0, "B"=>0.0, "T"=>300.0),
                   solver=Rodas5P(), reltol=1e-8, abstol=1e-10)
    @test sol.retcode == ReturnCode.Success
    Av = Float64(sol(5.0; idxs=_var(sys,"A"))); Bv = Float64(sol(5.0; idxs=_var(sys,"B")))
    Tv = Float64(sol(5.0; idxs=_var(sys,"T")))
    @test Tv > 300.0; @test Av < 0.1; @test Bv > 0.9
    # energy conservation: U/V = A·ū_A(T) + B·ū_B(T) ≈ const (adiabatic, const V)
    nA = mech.species[1].thermo; nB = mech.species[2].thermo
    U(T2, a, b2) = a * u_molar(nA, T2) + b2 * u_molar(nB, T2)
    @test U(Tv, Av, Bv) ≈ U(300.0, 1.0, 0.0)  rtol=1e-6
    # observed P rises with T at const V (Σc ≈ const)
    Pvar = [o.lhs for o in observed(sys) if getname(o.lhs)==:P][1]
    P0 = Float64(sol(0.0; idxs=Pvar)); Pe = Float64(sol(5.0; idxs=Pvar))
    @test Pe > P0
end

@testset ":adiabatic_constV: T-dependent Arrhenius rate with T as state solves" begin
    mech = _exothermic_ab_mech(; A=1.0e3, Ea=R4A * 3000.0)  # θ = Ea/R = 3000 K
    phase = ChemPhaseSystem(mech; config=convenience_config(:adiabatic_constV))
    sys = extract_system(phase)
    sol = simulate(phase, (0.0, 5.0);
                   u0=Dict("A"=>1.0, "B"=>0.0, "T"=>300.0),
                   solver=Rodas5P(), reltol=1e-8, abstol=1e-10)
    @test sol.retcode == ReturnCode.Success
    @test Float64(sol(5.0; idxs=_var(sys,"T"))) > 300.0     # T rises despite T-dependent k
end

@testset ":adiabatic_constV: Jacobian carries T-coupling" begin
    mech = _exothermic_ab_mech(; A=1.0e3, Ea=R4A * 3000.0)  # T-dependent rate → coupled
    sys = extract_system(ChemPhaseSystem(mech; config=convenience_config(:adiabatic_constV)))
    jac = ModelingToolkit.calculate_jacobian(sys)
    @test size(jac) == (3, 3)
    idx = _state_index(sys)
    @test !isequal(0.0, jac[idx["T"], idx["T"]])           # ∂(dT/dt)/∂T  ≠ 0 (cp/cv + Arrhenius)
    @test !isequal(0.0, jac[idx["A"], idx["T"]])           # ∂(dc_A/dt)/∂T ≠ 0 (rate T-coupling)
end

@testset "validate: :adiabatic requires NASA7 thermo on all species (§5.3.4)" begin
    # species WITHOUT thermo → validate reports an error under :adiabatic_constV
    x = SpeciesData(id=1, name="X"); y = SpeciesData(id=2, name="Y")
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech_noThermo = Mechanism(species=[x, y], reactions=[rxn])
    rep = validate(mech_noThermo; config=convenience_config(:adiabatic_constV))
    @test !isempty(rep.errors)
    @test any(occursin("adiabatic requires NASA7", e) for e in rep.errors)
    # with thermo → no error from this check
    mech_ok = _exothermic_ab_mech()
    rep2 = validate(mech_ok; config=convenience_config(:adiabatic_constV))
    @test !any(occursin("adiabatic requires NASA7", e) for e in rep2.errors)
    # isothermal config → thermo-not-required (no error even without thermo)
    rep3 = validate(mech_noThermo; config=convenience_config(:fixedT))
    @test !any(occursin("adiabatic requires NASA7", e) for e in rep3.errors)
end
