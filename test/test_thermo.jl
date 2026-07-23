using Test
using ChemMechSim

@testset "data/thermo" begin
    # ThermoModel is abstract (concrete NASA7/NASA9 deferred to a later phase)
    @test ThermoModel isa Type
    @test !isconcretetype(ThermoModel)

    # ThermoDatabase default-constructs empty, typed Dict{String,ThermoModel}
    db = ThermoDatabase()
    @test isempty(db.entries)
    @test db.entries isa Dict{String,ThermoModel}
end

const R_THERMO = 8.314

@testset "NASA7: constant-cp species (closed-form check)" begin
    a1 = 2.5; a6 = -a1 * 298.15; a7 = -a1 * log(298.15)
    coeffs = (a1, 0.0, 0.0, 0.0, 0.0, a6, a7)
    m = NASA7(coeffs, coeffs, 200.0, 1000.0, 3500.0)
    @test cp_molar(m, 500.0)  ≈ 2.5 * R_THERMO
    @test cp_molar(m, 1500.0) ≈ 2.5 * R_THERMO
    @test h_molar(m, 298.15)  ≈ 0.0  atol = 1e-6
    @test s_molar(m, 298.15)  ≈ 0.0  atol = 1e-6
    @test h_molar(m, 1000.0) ≈ R_THERMO * (2.5 * 1000.0 + a6)
    @test s_molar(m, 1000.0) ≈ R_THERMO * (2.5 * log(1000.0) + a7)
end

@testset "NASA7: thermodynamic consistency (cp=dh/dT, g=h-Ts)" begin
    a1 = 3.5
    coeffs = (a1, 1.0e-3, -5.0e-7, 0.0, 0.0, -a1 * 298.15, -a1 * log(298.15))
    m = NASA7(coeffs, coeffs, 200.0, 1000.0, 3500.0)
    for T in (400.0, 800.0, 1200.0, 2000.0)
        dh = h_molar(m, T + 1e-3) - h_molar(m, T - 1e-3)
        @test (dh / 2e-3) ≈ cp_molar(m, T)  atol = 1e-2
    end
    for T in (500.0, 1000.0, 1500.0)
        @test g_molar(m, T) ≈ h_molar(m, T) - T * s_molar(m, T)
    end
end

@testset "NASA7: coefficient range switches at Tmid" begin
    m = NASA7((2.0, 0,0,0,0, 0, 0), (4.0, 0,0,0,0, 0, 0), 200.0, 1000.0, 3500.0)
    @test cp_molar(m, 999.0)  ≈ 2.0 * R_THERMO
    @test cp_molar(m, 1001.0) ≈ 4.0 * R_THERMO
end

@testset "K_c(T): isomerization A<->B with entropy-only offset" begin
    a1 = 2.5
    base = (a1, 0.0, 0.0, 0.0, 0.0, -a1 * 298.15, -a1 * log(298.15))
    δ = log(2.0)
    aB = (a1, 0.0, 0.0, 0.0, 0.0, -a1 * 298.15, -a1 * log(298.15) + δ)
    nA = NASA7(base, base, 200.0, 1000.0, 3500.0)
    nB = NASA7(aB,   aB,   200.0, 1000.0, 3500.0)
    spA = SpeciesData(id=1, name="A", thermo=nA)
    spB = SpeciesData(id=2, name="B", thermo=nB)
    rxn = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(1.0, 0.0, 0.0))
    mech = Mechanism(species=[spA, spB], reactions=[rxn])
    import ChemMechSim: _equilibrium_constant, ThermoCtx
    for T in (400.0, 800.0, 1200.0)
        tcx = ThermoCtx(T, nothing, nothing, Dict{Int,Any}())
        @test _equilibrium_constant(mech, rxn, T, tcx) ≈ 2.0  rtol = 1e-9   # K_c = exp(δ) = 2
    end
end

@testset "NASA7: internal energy + cv (u = h − RT, cv = cp − R)" begin
    a1 = 3.5
    coeffs = (a1, 1.0e-3, -5.0e-7, 0.0, 0.0, -a1 * 298.15, -a1 * log(298.15))
    m = NASA7(coeffs, coeffs, 200.0, 1000.0, 3500.0)
    for T in (400.0, 800.0, 1500.0)
        @test u_molar(m, T)    ≈ h_molar(m, T) - R_THERMO * T
        @test cv_molar(m, T)   ≈ cp_molar(m, T) - R_THERMO
        @test u_over_RT(m, T)  ≈ h_over_RT(m, T) - 1
        @test cv_over_R(m, T)  ≈ cp_over_R(m, T) - 1
        @test u_molar(m, T)    ≈ u_over_RT(m, T) * R_THERMO * T     # consistency of the pair
    end
end

@testset "K_c numeric + analytic derivative (data layer)" begin
    using ChemMechSim: KcData, equilibrium_constant, equilibrium_constant_dT
    import ChemMechSim: _equilibrium_constant, ThermoCtx  # lowering-side cross-check (private but accessible)
    using ChemMechSim: load_mechanism

    # Use a real reversible reaction from GRI30 (has NASA7 + ThermoReverse).
    mech = load_mechanism(joinpath(@__DIR__, "data", "gri30.yaml"))
    # find a ThermoReverse reaction with Δν != 0 (e.g. A <=> B + C)
    rxidx = findfirst(r -> r.reverse_policy isa ChemMechSim.ThermoReverse &&
                           (sum(values(r.products)) - sum(values(r.reactants))) != 0,
                      mech.reactions)
    @test rxidx !== nothing
    rx = mech.reactions[rxidx]
    kcd = KcData(rx, mech)
    # Cross-check: equilibrium_constant(KcData, T) must match lowering's _equilibrium_constant numerically.
    for T in (900.0, 1500.0, 2500.0)
        tcx = ThermoCtx(T, R_GAS, P_STD, Dict{Int,Any}())
        @test equilibrium_constant(kcd, T) ≈ _equilibrium_constant(mech, rx, T, tcx)  rtol = 1e-12
    end
    # 1) analytic dT matches central finite difference
    for T in (900.0, 1500.0, 2500.0)
        h = 1e-3
        fdT = (equilibrium_constant(kcd, T+h) - equilibrium_constant(kcd, T-h)) / 2h
        @test abs(fdT - equilibrium_constant_dT(kcd, T)) / (abs(fdT) + 1e-30) < 1e-6
    end
    # 2) Δν=0 case: pick a thermo-reverse reaction with Δν == 0 (A+B <=> C+D style) if present.
    #    The (P°/RT)^0 factor must be 1, so K_c reduces to exp(-Δg°/RT).
    rxidx0 = findfirst(r -> r.reverse_policy isa ChemMechSim.ThermoReverse &&
                            (sum(values(r.products)) - sum(values(r.reactants))) == 0,
                       mech.reactions)
    if rxidx0 !== nothing
        rx0 = mech.reactions[rxidx0]
        kcd0 = KcData(rx0, mech)
        for T in (900.0, 1500.0, 2500.0)
            h = 1e-3
            fdT0 = (equilibrium_constant(kcd0, T+h) - equilibrium_constant(kcd0, T-h)) / 2h
            @test abs(fdT0 - equilibrium_constant_dT(kcd0, T)) / (abs(fdT0) + 1e-30) < 1e-6
        end
    end
end

@testset "ThermoReverse reverse-rate is an opaque keq call node" begin
    using ChemMechSim: keq, keq_dT  # registered symbolic functions (Task 2 adds them)
    import ModelingToolkit: equations

    mech = load_mechanism(joinpath(@__DIR__, "data", "gri30.yaml"))
    rxidx = findfirst(r -> r.reverse_policy isa ChemMechSim.ThermoReverse, mech.reactions)
    @test rxidx !== nothing
    # Use :fixedT (T as a parameter) so the ONLY place Tmid_sp could appear is K_c.
    # Under :adiabatic_constV the energy equation also uses NASA7 cp/h polynomials which
    # legitimately carry Tmid_sp coefficients (for cp/h, NOT K_c).
    phase = ChemMechSim.ChemPhaseSystem(mech; config=convenience_config(:fixedT))
    sys = ChemMechSim.extract_system(phase)
    # Find a species equation whose RHS involves the reverse rate (a product species of rx).
    # Its RHS should reference keq(...) (opaque), NOT an inlined NASA7 g/RT polynomial tree.
    eqs = equations(sys)
    rhs_with_keq = findfirst(eq -> occursin("keq", string(eq.rhs)), eqs)
    @test rhs_with_keq !== nothing                  # some RHS references keq
    # And no RHS contains the inlined _g_over_RT polynomial marker (e.g. "Tmid_sp" coeff names)
    @test !any(eq -> occursin("Tmid_sp", string(eq.rhs)), eqs)   # K_c no longer inlined
end

@testset "ThermoReverse + keq call node passes check_units under :adiabatic_constV" begin
    # Regression guard: the brief flagged conc^Δν unit-construction as the risk area. MTK's
    # get_unit probes the registered keq() with id=Quantity(1.0) and assigns that unit to every
    # call node, so the opaque function's unit must be INVARIANT across reactions. The split
    # (keq = dimensionless NASA7 part + inlined (P°/RT)^Δν) achieves this. This test confirms
    # GRI30 (with its mix of dnu=0 and dnu≠0 ThermoReverse reactions) lowers without a unit error.
    import ModelingToolkit: equations
    mech = load_mechanism(joinpath(@__DIR__, "data", "gri30.yaml"))
    phase = ChemMechSim.ChemPhaseSystem(mech; config=convenience_config(:adiabatic_constV))
    sys = ChemMechSim.extract_system(phase)
    @test length(equations(sys)) > 0                # lowering succeeded (check_units passed)
    # And the keq call node still appears (not accidentally optimized away)
    @test any(eq -> occursin("keq", string(eq.rhs)), equations(sys))
end

@testset "keq(T,id) numeric value matches equilibrium_constant (behavior-preserving)" begin
    # Verify the opaque call node evaluates to the correct K_c VALUE (not just that a node exists).
    # The split design: keq(T,id) returns exp(-Δg°/RT); _reverse_rate multiplies by (P°/RT)^Δν inline.
    # So keq(T,id) · (P°/(R·T))^Δν must equal equilibrium_constant(KcData, T) (the data-layer truth).
    using ChemMechSim: KcData, equilibrium_constant, keq, keq_dT
    import ChemMechSim: KEQ_TABLE, KEQ_NEXT_ID

    mech = load_mechanism(joinpath(@__DIR__, "data", "gri30.yaml"))
    # Pick the first ThermoReverse reaction with dnu != 0 (covers both factors of the split)
    rxidx = findfirst(r -> r.reverse_policy isa ChemMechSim.ThermoReverse &&
                           (sum(values(r.products)) - sum(values(r.reactants))) != 0,
                      mech.reactions)
    rx = mech.reactions[rxidx]
    # Lower a minimal system so _reverse_rate populates KEQ_TABLE
    phase = ChemMechSim.ChemPhaseSystem(mech; config=convenience_config(:fixedT))
    sys = ChemMechSim.extract_system(phase)
    # For each registered id, verify keq(T, id) is behavior-preserving with the data layer.
    ids = sort(collect(keys(KEQ_TABLE)))
    @test !isempty(ids)
    id_real = Float64(first(ids))
    for T in (900.0, 1500.0, 2500.0)
        @test keq(T, id_real) ≈ keq(T, first(ids))  rtol = 1e-12
        @test keq_dT(T, id_real) ≈ keq_dT(T, first(ids))  rtol = 1e-12
    end
    R_GAS_v = ChemMechSim.R_GAS
    P_STD_v = ChemMechSim.P_STD
    for id in ids
        kcd = KEQ_TABLE[id]
        for T in (900.0, 1500.0, 2500.0)
            # The opaque call node value (Float64 method):
            keq_node = keq(T, id)
            # The full K_c from the data layer (truth):
            Kc_full = equilibrium_constant(kcd, T)
            # The inlined factor (P°/(R·T))^Δν — what _reverse_rate multiplies in:
            factor = iszero(kcd.dnu) ? 1.0 : (P_STD_v / (R_GAS_v * T))^kcd.dnu
            @test keq_node * factor ≈ Kc_full  rtol = 1e-12
        end
    end
end
