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

@testset ":adiabatic_constV: dT/dt analytic + T rises + U conserved + P a differential state" begin
    HEAT = 10000.0
    mech = _exothermic_ab_mech(; A=1.0, HEAT=HEAT)         # k = 1 s⁻¹ (constant)
    phase = ChemPhaseSystem(mech; config=convenience_config(:adiabatic_constV))
    sys = extract_system(phase); idx = _state_index(sys)
    # P is now a differential STATE under const-V (Task 4), not observed
    @test "P" in keys(idx)
    @test !("P" in [String(getname(o.lhs)) for o in observed(sys)])
    # analytic dT/dt at t=0: r = k·A = 1·1 = 1; Σc·cv = (A+B)·(cp/R−1)·R = 1·1.5·R
    dT_analytic = 1.0 * HEAT / (1.0 * 1.5 * R4A)
    f = ODEFunction(sys); du = zeros(length(idx))
    u0 = zeros(length(idx)); u0[idx["A"]] = 1.0; u0[idx["B"]] = 0.0; u0[idx["T"]] = 300.0
    f(du, u0, [getdefault(p) for p in parameters(sys)], 0.0)
    @test du[idx["T"]] ≈ dT_analytic  rtol=1e-9
    # solve: T rises, A depletes, B grows (build_problem auto-fills P0 from EOS)
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
    # P is a STATE now: access via _var(sys, "P"). P rises with T at const V (Σc ≈ const).
    Pvar = _var(sys, "P")
    P0 = Float64(sol(0.0; idxs=Pvar)); Pe = Float64(sol(5.0; idxs=Pvar))
    @test Pe > P0
    # P0 matches the EOS initial pressure (Σc)·R·T0 = 1·8.314·300 (build_problem auto-fill)
    @test P0 ≈ 1.0 * R4A * 300.0  rtol=1e-6
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

@testset ":adiabatic_constV: Jacobian carries T- and P-coupling" begin
    mech = _exothermic_ab_mech(; A=1.0e3, Ea=R4A * 3000.0)  # T-dependent rate → coupled
    sys = extract_system(ChemPhaseSystem(mech; config=convenience_config(:adiabatic_constV)))
    jac = ModelingToolkit.calculate_jacobian(sys)
    @test size(jac) == (4, 4)                              # A, B, T, P (P is a differential state, Task 4)
    idx = _state_index(sys)
    @test !isequal(0.0, jac[idx["T"], idx["T"]])           # ∂(dT/dt)/∂T  ≠ 0 (cp/cv + Arrhenius)
    @test !isequal(0.0, jac[idx["A"], idx["T"]])           # ∂(dc_A/dt)/∂T ≠ 0 (rate T-coupling)
    @test !isequal(0.0, jac[idx["P"], idx["P"]])           # ∂(dP/dt)/∂P  ≠ 0 (const-V P-ODE couples P)
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

@testset ":adiabatic_constV via BatchReactor (example smoke)" begin
    a1 = 2.5; a6A = -a1 * 298.15; a6B = a6A - 10000.0 / R4A
    nA = NASA7((a1,0,0,0,0,a6A,0.0),(a1,0,0,0,0,a6A,0.0), 200.0, 1000.0, 3500.0)
    nB = NASA7((a1,0,0,0,0,a6B,0.0),(a1,0,0,0,0,a6B,0.0), 200.0, 1000.0, 3500.0)
    spA = SpeciesData(id=1, name="A", thermo=nA); spB = SpeciesData(id=2, name="B", thermo=nB)
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                       kinetics=ElementaryArrhenius(0.5, 0.0, 0.0))
    mech = Mechanism(species=[spA, spB], reactions=[rxn])
    reactor = BatchReactor(mech; mode=:adiabatic_constV)        # Layer-1 API the example uses
    sys = extract_system(reactor)
    sol = simulate(reactor, (0.0, 10.0);
                   u0=Dict("A"=>1.0, "B"=>0.0, "T"=>300.0),
                   solver=Rodas5P(), reltol=1e-8, abstol=1e-10)
    @test sol.retcode == ReturnCode.Success
    @test Float64(sol(10.0; idxs=_var(sys,"T"))) > 300.0
end

const R4B = 8.314

"Exothermic A → B with NASA7 thermo (constant cp = 2.5 R); B is HEAT J/mol lower in enthalpy. (const-P test)"
function _exothermic_ab_constP_mech(; A=1.0, HEAT=10000.0)
    a1 = 2.5; a6A = -a1 * 298.15; a6B = a6A - HEAT / R4B
    nA = NASA7((a1,0,0,0,0,a6A,0.0),(a1,0,0,0,0,a6A,0.0), 200.0, 1000.0, 3500.0)
    nB = NASA7((a1,0,0,0,0,a6B,0.0),(a1,0,0,0,0,a6B,0.0), 200.0, 1000.0, 3500.0)
    spA = SpeciesData(id=1, name="A", thermo=nA); spB = SpeciesData(id=2, name="B", thermo=nB)
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0), kinetics=ElementaryArrhenius(A, 0.0, 0.0))
    Mechanism(species=[spA, spB], reactions=[rxn])
end

@testset ":adiabatic_constP: dT/dt analytic + T rises + enthalpy H conserved" begin
    HEAT = 10000.0
    mech = _exothermic_ab_constP_mech(; A=1.0, HEAT=HEAT)            # k = 1 s⁻¹ (constant)
    phase = ChemPhaseSystem(mech; config=convenience_config(:adiabatic_constP))
    sys = extract_system(phase); idx = _state_index(sys)
    @test sort(collect(keys(idx))) == sort(["n_A", "n_B", "T"])      # 3 states (T is a state under :adiabatic)
    # analytic dT/dt at t=0: dn_A/dt = V·(-k·c_A) = -k·n_A; V·r = k·n_A0 = 1.
    # dT/dt = V·r·HEAT / (Σn·cp); cp = 2.5·R; Σn·cp = 1·2.5·R  ⟹  dT/dt = HEAT/(2.5·R)
    dT_analytic = HEAT / (2.5 * R4B)
    f = ODEFunction(sys); du = zeros(length(idx))
    u0 = zeros(length(idx)); u0[idx["n_A"]] = 1.0; u0[idx["n_B"]] = 0.0; u0[idx["T"]] = 300.0
    f(du, u0, _pvals(sys), 0.0)
    @test du[idx["T"]] ≈ dT_analytic  rtol=1e-9
    # solve: T rises; n_A depletes; enthalpy H = Σ nᵢ·h̄ᵢ(T) conserved (const-P adiabatic invariant)
    sol = simulate(phase, (0.0, 5.0); u0=Dict("n_A"=>1.0,"n_B"=>0.0,"T"=>300.0),
                   solver=Rodas5P(), reltol=1e-8, abstol=1e-10)
    @test sol.retcode == ReturnCode.Success
    nAv = Float64(sol(5.0; idxs=_var(sys,"n_A"))); nBv = Float64(sol(5.0; idxs=_var(sys,"n_B")))
    Tv  = Float64(sol(5.0; idxs=_var(sys,"T")))
    @test Tv > 300.0 && nAv < 0.1 && nBv > 0.9
    nA_th = mech.species[1].thermo; nB_th = mech.species[2].thermo
    H(T2,a,b2) = a * h_molar(nA_th, T2) + b2 * h_molar(nB_th, T2)
    H0 = H(300.0, 1.0, 0.0)
    scale = max(abs(H0), abs(nBv * h_molar(nB_th, Tv)), 1.0)         # H terms are O(1e4–1e5); check RELATIVE drift
    @test abs(H(Tv, nAv, nBv) - H0) / scale < 1e-6
end

@testset ":adiabatic_constP: Jacobian carries T- and n-coupling" begin
    mech = _exothermic_ab_constP_mech(; A=1.0e3, HEAT=R4B * 3000.0)   # T-dependent Arrhenius (θ = 3000 K)
    sys = extract_system(ChemPhaseSystem(mech; config=convenience_config(:adiabatic_constP)))
    jac = ModelingToolkit.calculate_jacobian(sys)
    @test size(jac) == (3, 3)
    idx = _state_index(sys)
    @test !isequal(0.0, jac[idx["T"],   idx["T"]])    # ∂(Ṫ/∂T)   ≠ 0 (cp/h + Arrhenius)
    @test !isequal(0.0, jac[idx["n_A"], idx["T"]])    # ∂(ṅ_A/∂T) ≠ 0 (rate T-coupling via c=n/V)
    @test !isequal(0.0, jac[idx["T"],   idx["n_A"]])  # ∂(Ṫ/∂n_A) ≠ 0 (cp_sum + Δh̄ source depend on n_A)
    # generate_jacobian (code-export path) builds for const-P
    @test !isempty(string(ChemMechSim.generate_jacobian(sys)))
end

@testset ":adiabatic_constP via BatchReactor — end-to-end (ignition-like T rise)" begin
    # H2 + OH ↔ H + H2O style: exothermic, T-dependent, reversible via ExplicitReverse.
    a1 = 3.5; a6H2 = -a1*298.15; a6OH = -a1*500.0
    a6H  = -a1*298.15 - 5000.0/R4B;  a6H2O = -a1*298.15 - 12000.0/R4B   # H2O lowest (exothermic forward)
    nH2  = NASA7((a1,0,0,0,0,a6H2,0.0),(a1,0,0,0,0,a6H2,0.0), 200.0,1000.0,3500.0)
    nOH  = NASA7((a1,0,0,0,0,a6OH,0.0),(a1,0,0,0,0,a6OH,0.0), 200.0,1000.0,3500.0)
    nH   = NASA7((a1,0,0,0,0,a6H,0.0), (a1,0,0,0,0,a6H,0.0),  200.0,1000.0,3500.0)
    nH2O = NASA7((a1,0,0,0,0,a6H2O,0.0),(a1,0,0,0,0,a6H2O,0.0),200.0,1000.0,3500.0)
    sp = [SpeciesData(id=1,name="H2",thermo=nH2), SpeciesData(id=2,name="OH",thermo=nOH),
          SpeciesData(id=3,name="H",thermo=nH),  SpeciesData(id=4,name="H2O",thermo=nH2O)]
    fwd = ElementaryArrhenius(1.0e2, 0.0, R4B*3000.0)                  # θ = Ea/R = 3000 K (T-dependent)
    rev = ElementaryArrhenius(5.0e1, 0.0, R4B*3000.0)
    rxn = ReactionData(reactants=Dict(1=>1.0,2=>1.0), products=Dict(3=>1.0,4=>1.0),
                       kinetics=fwd, reverse_policy=ExplicitReverse(rev))
    mech = Mechanism(species=sp, reactions=[rxn])
    reactor = BatchReactor(mech; mode=:adiabatic_constP)               # Layer-1 API
    sys = extract_system(reactor); idx = _state_index(sys)
    @test Set(keys(idx)) == Set(["n_H2","n_OH","n_H","n_H2O","T"])     # 4 moles + T
    sol = simulate(reactor, (0.0, 5.0);
                   u0=Dict("n_H2"=>1.0,"n_OH"=>1.0,"n_H"=>0.0,"n_H2O"=>0.0,"T"=>1500.0),
                   solver=Rodas5P(), reltol=1e-8, abstol=1e-10)
    @test sol.retcode == ReturnCode.Success
    @test Float64(sol(5.0; idxs=_var(sys,"T"))) > 1500.0               # exothermic → T rises at const P
    @test Float64(sol(5.0; idxs=_var(sys,"n_H2O"))) > 0.0              # H2O produced
    # V grows with T at const P (Σn ≈ const for this mole-neutral reaction, so V ∝ T)
    Vobs = [o.lhs for o in observed(sys) if getname(o.lhs)==:V][1]
    @test Float64(sol(5.0; idxs=Vobs)) > Float64(sol(0.0; idxs=Vobs))
end

using ChemMechSim: _sum_species_rhs, PlogPoint, PlogRate
using OrdinaryDiffEq
using OrdinaryDiffEq: FBDF
using SparseArrays: nnz, nonzeros

"Small inline PLOG mechanism for the const-V P-differential integration test:
 A → B with one PLOG reaction. Both species carry NASA7 thermo (required for :adiabatic)."
function _plog_ab_mech()
    a1 = 2.5; a6A = -a1 * 298.15; a6B = a6A - 5000.0 / R4A   # B is 5 kJ/mol lower (mildly exothermic)
    nA = NASA7((a1,0,0,0,0,a6A,0.0),(a1,0,0,0,0,a6A,0.0), 200.0, 1000.0, 3500.0)
    nB = NASA7((a1,0,0,0,0,a6B,0.0),(a1,0,0,0,0,a6B,0.0), 200.0, 1000.0, 3500.0)
    spA = SpeciesData(id=1, name="A", thermo=nA); spB = SpeciesData(id=2, name="B", thermo=nB)
    kin = PlogRate([PlogPoint(1e4, 1e3, 0.0, 0.0), PlogPoint(1e6, 1e2, 0.0, 0.0)])
    rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0), kinetics=kin)
    Mechanism(species=[spA, spB], reactions=[rxn])
end

@testset "const-V: P differential state + opaque-PLOG + jac tractable + solve" begin
    # Task 4 integration: const-V adiabatic + opaque-PLOG ⇒ P is a DIFFERENTIAL state,
    # the sparse Jacobian is finite (does not explode), and FBDF solves to Success.
    mech = _plog_ab_mech()
    phase = ChemMechSim.ChemPhaseSystem(mech; config=convenience_config(:adiabatic_constV))
    sys = ChemMechSim.extract_system(phase)
    names = Set(String(ModelingToolkit.getname(u)) for u in unknowns(sys))
    @test "P" in names                                        # P is now a STATE (not observed)
    @test !("P" in Set(String(ModelingToolkit.getname(o.lhs)) for o in observed(sys)))  # not observed
    # sparse Jacobian must be finite (was exploding symbolically before the opaque-PLOG node)
    jac = ModelingToolkit.calculate_jacobian(sys; sparse=true)
    @test nnz(jac) > 0                                        # finite nnz ⇒ tractable
    @test all(isfinite, nonzeros(jac))                       # no Inf/NaN
    # solve a tiny tspan with FBDF; P0 auto-filled by build_problem (no P in u0)
    sol = simulate(phase, (0.0, 0.1);
                   u0=Dict("A"=>1.0, "B"=>0.0, "T"=>1000.0),
                   solver=FBDF(), reltol=1e-6, abstol=1e-9)
    @test sol.retcode == OrdinaryDiffEq.ReturnCode.Success
end

@testset ":fixedT const-V: P is a differential state" begin
    mech = load_mechanism(joinpath(@__DIR__, "data", "gri30.yaml"))
    phase = ChemMechSim.ChemPhaseSystem(mech; config=convenience_config(:fixedT))
    sys = ChemMechSim.extract_system(phase)
    unk_names = Set(String(ModelingToolkit.getname(u)) for u in unknowns(sys))
    @test "P" in unk_names                       # P is now a STATE under :fixedT const-V (was observed)
    @test !("P" in [String(ModelingToolkit.getname(o.lhs)) for o in observed(sys)])  # not observed
    # The T parameter exists (isothermal: T is a parameter, not a state) and P does not
    # depend on a missing T-state — _p_ode_constV with is_adiabatic=false emits D(P)~R·T·Σ物种RHS
    # (energy_rhs=0 since there is no energy ODE under isothermal).
    @test !("T" in unk_names)                     # T stays a parameter under :fixedT
    Tparam = _param(sys, "T")
    @test ModelingToolkit.getdefault(Tparam) ≈ 300.0    # default 300 K from rate_param(:T, 300.0, u"K")
end

@testset ":fixedT const-V: P-ODE rhs reduces to R·T·Σ物种RHS (energy_rhs=0)" begin
    # Use a simple 2-species mechanism so we can compute Σ物种RHS analytically.
    # A -> B (Δn_total = 0): even at nonzero rate, Σ dc/dt = 0, so dP/dt = R·T·0 = 0
    # at t=0 — P is a state but does not move (mole-neutral reaction). Verify the
    # state arrangement + that D(P) is well-defined (not dangling).
    sp = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")]
    rx = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                      kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=sp, reactions=[rx])
    phase = ChemMechSim.ChemPhaseSystem(mech; config=convenience_config(:fixedT))
    sys = ChemMechSim.extract_system(phase); idx = _state_index(sys)
    @test sort(collect(keys(idx))) == sort(["A", "B", "P"])     # P joined A,B as a state
    f = ODEFunction(sys); du = zeros(length(idx))
    u0 = zeros(length(idx)); u0[idx["A"]] = 1.0; u0[idx["B"]] = 0.0; u0[idx["P"]] = 1.0 * R4A * 300.0
    f(du, u0, _pvals(sys), 0.0)
    @test du[idx["P"]] ≈ 0.0                                     # mole-neutral reaction ⇒ ΣRHS=0 ⇒ dP/dt=0
    @test du[idx["A"]] ≈ -1.0                                    # dc_A/dt = -k·A = -1 at t=0
    @test du[idx["B"]] ≈ +1.0                                    # dc_B/dt = +k·A = +1
end

@testset ":fixedT const-V: build_problem P0 from T-param default" begin
    # Caller supplies species u0 but NEITHER T (it's a param) NOR P (auto-filled).
    # build_problem must use the T param's default (300 K) for the P0 auto-fill.
    sp = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")]
    rx = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                      kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=sp, reactions=[rx])
    phase = ChemMechSim.ChemPhaseSystem(mech; config=convenience_config(:fixedT))
    sys = ChemMechSim.extract_system(phase); idx = _state_index(sys)
    prob = build_problem(phase, Dict("A"=>1.0, "B"=>0.0), (0.0, 1.0))
    P0 = prob.u0[idx["P"]]
    @test P0 ≈ 1.0 * R4A * 300.0 rtol=1e-9                       # csum=1, T-param default=300
end

@testset "P-ODE helpers (_sum_species_rhs)" begin
    # A + B -> C (Δn_total = 1-2 = -1 per event); rate r ⇒ Σ dc/dt = -r
    sp = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B"), SpeciesData(id=3, name="C")]
    rx = ReactionData(reactants=Dict(1=>1.0, 2=>1.0), products=Dict(3=>1.0),
                      kinetics=ElementaryArrhenius(1e9, 0.0, 0.0))
    mech = Mechanism(species=sp, reactions=[rx])
    @test _sum_species_rhs(mech, [2.5]) ≈ -2.5                # Δn_total(A+B->C) = 1-2 = -1; ×rate 2.5
    # 2A -> 2A (Δn_total = 0) ⇒ Σ dc/dt = 0
    rx2 = ReactionData(reactants=Dict(1=>2.0), products=Dict(1=>2.0),
                       kinetics=ElementaryArrhenius(1e9, 0.0, 0.0))
    mech2 = Mechanism(species=sp, reactions=[rx2])
    @test _sum_species_rhs(mech2, [2.5]) == 0.0
end
