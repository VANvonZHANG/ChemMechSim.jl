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

# —— Phase 5b Task 2: GRI30 scale characterization ————————————————————————————

const _GRI30_YAML = joinpath(@__DIR__, "data", "gri30.yaml")

@testset "Phase 5b: GRI30 load + validate + structure" begin
    mech = load_mechanism(_GRI30_YAML)
    @test length(mech.species) == 53
    @test length(mech.reactions) == 325
    counts = Dict{DataType,Int}()
    for r in mech.reactions
        counts[typeof(r.kinetics)] = get(counts, typeof(r.kinetics), 0) + 1
    end
    @test counts[ElementaryArrhenius]  == 284
    @test counts[ThirdBodyArrhenius]   == 12
    @test counts[TroeFalloff]          == 26
    @test counts[LindemannFalloff]     == 3
    @test count(r -> r.meta.duplicate, mech.reactions) == 6
    # validate under the adiabatic-constV config the ignition test uses (Task 3);
    # T_range trimmed to [300,3000] to avoid the over-conservative low-end NASA warnings.
    rep = validate(mech; config=convenience_config(:adiabatic_constV), T_range=(300.0, 3000.0))
    @test isempty(rep.errors)
    @info "GRI30 validate" n_errors=length(rep.errors) n_warnings=length(rep.warnings)
end

@testset "Phase 5b: GRI30 lowers + compiles + dense Jacobian builds" begin
    mech = load_mechanism(_GRI30_YAML)
    sys = lower_to_mtk(mech; config=convenience_config(:adiabatic_constV))
    @test length(unknowns(sys)) == 54                 # 53 concentrations + T
    @test length(equations(sys)) == 54
    # Dense Jacobian = the production path (sparse codegen is pathological — Task 4).
    jac = ModelingToolkit.calculate_jacobian(sys; sparse=false)
    @test size(jac) == (54, 54)
    @test !isempty(string(generate_jacobian(sys; sparse=false)))
end

# —— Phase 5b Task 3: GRI30 CH4-air ignition end-to-end ———————————————————————
# Stoichiometric CH4-air (CH4 + 2(O2 + 3.76 N2)) at T0=1500 K, P=1 atm, const-V adiabatic.
# Probe (2026-07-07): FBDF solves in 1.34 s, T_end=2902.6 K, t_ignition=1.72 ms (3-solver agree).

const _R5B    = 8.314
const _P_STD  = 101325.0
const _T0_5B  = 1500.0
const _X_CH4_5B, _X_O2_5B, _X_N2_5B = 1.0/10.52, 2.0/10.52, 7.52/10.52

@testset "Phase 5b: GRI30 CH4-air const-V ignition (FBDF, U conserved)" begin
    mech = load_mechanism(_GRI30_YAML)
    reactor = BatchReactor(mech; mode=:adiabatic_constV)
    sys = extract_system(reactor)
    c_tot = _P_STD / (_R5B * _T0_5B)                          # ≈ 8.122 mol/m³
    X0 = Dict("CH4" => _X_CH4_5B, "O2" => _X_O2_5B, "N2" => _X_N2_5B)
    u0 = Dict(sp.name => get(X0, sp.name, 0.0) * c_tot for sp in mech.species)
    u0["T"] = _T0_5B
    sol = simulate(reactor, (0.0, 5.0e-3); u0=u0, solver=FBDF(), reltol=1e-8, abstol=1e-12)
    @test sol.retcode == ReturnCode.Success
    # ignition: T rises well above T0 (probe T_end 2902.6 K; give margin for solver variance)
    T_end = Float64(sol(5.0e-3; idxs=_var(sys, "T")))
    @test T_end > 2500.0
    # ignition delay (argmax |dT/dt|) in a sane window (probe 1.72 ms)
    ts = range(0.0, 5.0e-3; length=2001)
    Ts = [Float64(sol(t; idxs=_var(sys, "T"))) for t in ts]
    dTdt = diff(Ts) ./ diff(ts)
    t_ign = ts[argmax(abs.(dTdt)) + 1]
    @test 0.5e-3 < t_ign < 5.0e-3
    # adiabatic const-V invariant: U = Σ cᵢ·ūᵢ(T) conserved (relative drift).
    # h2o2 (Phase 5a) hit 1e-6; GRI30 is stiffer/larger → 1e-4 headroom.
    function U_at(t)
        Tv = Float64(sol(t; idxs=_var(sys, "T")))
        u = 0.0
        for sp in mech.species
            u += Float64(sol(t; idxs=_var(sys, sp.name))) * u_molar(sp.thermo, Tv)
        end
        return u
    end
    U0 = U_at(0.0); U1 = U_at(5.0e-3)
    @test abs(U1 - U0) / abs(U0) < 1.0e-4
    @info "GRI30 CH4-air ignition" T_end=T_end t_ignition_ms=round(t_ign*1e3, digits=3) U_drift_rel=abs(U1-U0)/abs(U0)
end

# —— Phase 5b Task 4: Jacobian density + sparse-codegen known issue —————————————

@testset "Phase 5b: GRI30 Jacobian is dense (sparse codegen pathological — known issue)" begin
    mech = load_mechanism(_GRI30_YAML)
    sys = lower_to_mtk(mech; config=convenience_config(:adiabatic_constV))
    jac_sp = ModelingToolkit.calculate_jacobian(sys; sparse=true)   # cheap: ~10 s, small object
    n = length(unknowns(sys))
    density = nnz(jac_sp) / n^2
    @test density > 0.5                                             # probe: 87.8% — sparse not worthwhile
    @info "GRI30 Jacobian density" n=n nnz=nnz(jac_sp) density_pct=round(density*100, digits=1)
    # KNOWN ISSUE (do NOT invoke in tests): generate_jacobian(sys; sparse=true) produces a
    # ~4.2 GB string (~57 s) at GRI30 scale — ModelingToolkit.build_function does not CSE the
    # NASA7 ifelse(T<=Tmid, lo, hi) branches across the 54×54 Jacobian entries. Use the DENSE
    # path (calculate_jacobian(sparse=false) / generate_jacobian(sparse=false)), characterized
    # in Task 2. Investigation tracked in docs/superpowers/plans/2026-07-07-phase5b-gri30-scaling.md
    # and .superpowers/sdd/progress.md. Remove this comment if a future MTK version fixes it.
end
