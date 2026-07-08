using Test
using ChemMechSim
using ModelingToolkit
using ModelingToolkit: unknowns, getname, parameters
using OrdinaryDiffEq

"7-species H2-O2 subset: 3 elementary (catalyst) + 2 third-body (direct) + 1 Troe (direct)."
function _h2o2_mech()
    H2 = SpeciesData(id=1, name="H2");  O2  = SpeciesData(id=2, name="O2")
    H2O= SpeciesData(id=3, name="H2O"); H   = SpeciesData(id=4, name="H")
    O  = SpeciesData(id=5, name="O");   OH  = SpeciesData(id=6, name="OH")
    HO2= SpeciesData(id=7, name="HO2")
    allM = Dict(sid => 1.0 for sid in 1:7)     # every species is a third body, α=1
    rxns = [
        ReactionData(reactants=Dict(1=>1.0, 2=>1.0), products=Dict(6=>2.0),               # R1 O2+H2->2OH
                     kinetics=ElementaryArrhenius(1.0e6, 0.0, 0.0)),
        ReactionData(reactants=Dict(6=>1.0, 1=>1.0), products=Dict(3=>1.0, 4=>1.0),       # R2 OH+H2->H2O+H
                     kinetics=ElementaryArrhenius(5.0e6, 0.0, 0.0)),
        ReactionData(reactants=Dict(4=>1.0, 2=>1.0), products=Dict(6=>1.0, 5=>1.0),       # R3 H+O2->OH+O
                     kinetics=ElementaryArrhenius(2.0e6, 0.0, 0.0)),
        ReactionData(reactants=Dict(4=>1.0, 2=>1.0), products=Dict(7=>1.0),               # R4 H+O2+M->HO2+M (3rd-body)
                     kinetics=ThirdBodyArrhenius(ElementaryArrhenius(1.0e3, 0.0, 0.0), allM)),
        ReactionData(reactants=Dict(4=>1.0, 6=>1.0), products=Dict(3=>1.0),               # R5 H+OH+M->H2O+M (3rd-body)
                     kinetics=ThirdBodyArrhenius(ElementaryArrhenius(1.0e3, 0.0, 0.0), allM)),
        ReactionData(reactants=Dict(4=>1.0, 2=>1.0), products=Dict(7=>1.0),               # R6 H+O2(+M)->HO2(+M) (Troe)
                     kinetics=TroeFalloff(ElementaryArrhenius(1.0e9, 0.0, 0.0),           # LOW_rate = k0 = 1e9
                                          ElementaryArrhenius(1.0e6, 0.0, 0.0),           # HIGH_rate = kinf = 1e6
                                          allM, TroeParams(0.5, 1.0e-30, 1.0e30, 1.0e30))),
    ]
    Mechanism(species=[H2, O2, H2O, H, O, OH, HO2], reactions=rxns)
end

"Hand-written RHS mirroring the mechanism exactly (independent reference, §3.4 #6).
 c is id-ordered: [H2,O2,H2O,H,O,OH,HO2]. [M]_eff = Σ all species (α=1)."
function _h2o2_rhs!(dc, c, T)
    H2,O2,H2O,H,O,OH,HO2 = c
    Meff = sum(c)
    kinf, k0 = 1.0e6, 1.0e9                      # kinf=high=1e6, k0=low=1e9
    Pr = k0 * Meff / kinf
    Fcent = (1-0.5)*exp(-T/1.0e30) + 0.5*exp(-T/1.0e-30) + exp(-1.0e30/T)
    lFc, lPr = log10(Fcent), log10(Pr)
    cT = -0.4 - 0.67*lFc; NT = 0.75 - 1.27*lFc
    f1 = lPr + cT; f2 = NT - 0.14*f1
    F = 10^(lFc / (1 + (f1/f2)^2))
    k_troe = kinf * (Pr/(1+Pr)) * F
    r1 = 1.0e6 * H2 * O2;  r2 = 5.0e6 * OH * H2;  r3 = 2.0e6 * H * O2
    r4 = 1.0e3 * H * O2 * Meff;  r5 = 1.0e3 * H * OH * Meff;  r6 = k_troe * H * O2
    dc[1] = -r1 - r2;  dc[2] = -r1 - r3 - r4 - r6;  dc[3] =  r2 + r5
    dc[4] =  r2 - r3 - r4 - r5 - r6;  dc[5] =  r3;  dc[6] = 2r1 - r2 + r3 - r5;  dc[7] =  r4 + r6
    return dc
end

const _H2O2_NAMES = ["H2","O2","H2O","H","O","OH","HO2"]   # id order (id = position)

@testset "§3.4 #6: H2-O2 mixed lowering matches handwritten RHS" begin
    mech = _h2o2_mech(); phase = ChemPhaseSystem(mech); sys = extract_system(phase)
    idx = _state_index(sys); Tv = 1000.0
    c = [2.0, 1.5, 0.5, 0.3, 0.1, 0.4, 0.2]               # id-ordered concentrations
    u = zeros(7); for (i, name) in enumerate(_H2O2_NAMES); u[idx[name]] = c[i]; end
    du = zeros(7); ODEFunction(sys)(du, u, replace_T(_pvals(sys), sys, Tv), 0.0)
    dc = _h2o2_rhs!(zeros(7), c, Tv)
    for (i, name) in enumerate(_H2O2_NAMES)
        @test du[idx[name]] ≈ dc[i]  rtol = 1e-6           # CS-order du vs id-order dc, by name
    end
end

@testset "§3.4 #1/#2/#3: H2-O2 lowers (3rd-body + Troe + elementary) and solves" begin
    mech = _h2o2_mech(); phase = ChemPhaseSystem(mech); sys = extract_system(phase)
    idx = _state_index(sys)
    @test length(unknowns(sys)) == 7
    Tp = _param(sys, "T")
    sol = simulate(phase, (0.0, 1.0e-6);
                   u0=Dict("H2"=>2.0,"O2"=>1.5,"H2O"=>0.0,"H"=>1e-3,"O"=>0.0,"OH"=>0.0,"HO2"=>0.0),
                   params=[Tp => 1000.0], reltol=1e-9, abstol=1e-12)
    @test all(isfinite, sol.u[end])
    # trajectories agree with handwritten RHS over the integration
    f = ODEFunction(sys)
    for t in (1.0e-7, 1.0e-6)
        c_id = [sol(t; idxs=_var(sys, n)) for n in _H2O2_NAMES]
        u_cs = zeros(7); for (i, name) in enumerate(_H2O2_NAMES); u_cs[idx[name]] = c_id[i]; end
        dc_cs = zeros(7); f(dc_cs, u_cs, replace_T(_pvals(sys), sys, 1000.0), t)
        dc_hw = _h2o2_rhs!(zeros(7), c_id, 1000.0)
        for (i, name) in enumerate(_H2O2_NAMES)
            @test dc_cs[idx[name]] ≈ dc_hw[i]  rtol = 1e-6
        end
    end
end

@testset "§3.4 #5: Jacobian of the mixed system is generated" begin
    sys = extract_system(ChemPhaseSystem(_h2o2_mech()))
    jac = ModelingToolkit.calculate_jacobian(sys)
    @test size(jac) == (7, 7)                              # builds => symbolic Jacobian feasible
end

# —— Phase 5a T5: end-to-end H2-O2 ignition (const-V + const-P) ——————————————————
# Spec: load_mechanism(h2o2.yaml) → BatchReactor(:adiabatic_constV/:adiabatic_constP) → simulate
# Asserts ignition (T_max > T_init + 400 K) and conservation (U or H rel drift < 1e-6).
# IC: H2:O2:N2 = 2:1:4 mole fraction, T=1000 K, P=1 atm.

using ChemMechSim: load_mechanism, validate, BatchReactor, u_molar, h_molar
using OrdinaryDiffEq: Rodas5P, ReturnCode

const _H2O2_YAML_5A = joinpath(@__DIR__, "data", "h2o2.yaml")

# IC: stoichiometric H2-O2 diluted with N2. mole fractions H2:O2:N2 = 2:1:4.
# T_init = 1000 K, P_init = 1 atm. Solver: Rodas5P, reltol=1e-8, abstol=1e-12.
const _R5A    = 8.314
const _P_STD  = 101325.0
const _T_INIT = 1000.0
const _Y_H2_5A, _Y_O2_5A, _Y_N2_5A = 2/7, 1/7, 4/7

@testset "Phase 5a: validate(load_mechanism(h2o2.yaml)) passes" begin
    mech = load_mechanism(_H2O2_YAML_5A)
    rep = validate(mech)
    @test isempty(rep.errors)
end

@testset "Phase 5a: H2-O2 const-V ignition — U conserved, ignition occurs" begin
    mech = load_mechanism(_H2O2_YAML_5A)
    reactor = BatchReactor(mech; mode=:adiabatic_constV)
    sys = extract_system(reactor)
    c_tot = _P_STD / (_R5A * _T_INIT)
    u0 = Dict("H2"=>_Y_H2_5A*c_tot, "O2"=>_Y_O2_5A*c_tot, "N2"=>_Y_N2_5A*c_tot, "T"=>_T_INIT,
              "H"=>0.0, "O"=>0.0, "OH"=>0.0, "H2O"=>0.0, "HO2"=>0.0, "H2O2"=>0.0, "AR"=>0.0)
    sol = simulate(reactor, (0.0, 1.0); u0=u0, solver=Rodas5P(), reltol=1e-8, abstol=1e-12)
    @test sol.retcode == ReturnCode.Success
    # ignition: T_max > T_init + 400 K
    T_end = Float64(sol(1.0; idxs=_var(sys,"T")))
    T_max = maximum(Float64(sol(t; idxs=_var(sys,"T"))) for t in 0.0:0.01:1.0)
    @test T_max > _T_INIT + 400.0
    # U conservation (relative drift): U = Σ cᵢ·ūᵢ(T)
    function U_at(t)
        Tv = Float64(sol(t; idxs=_var(sys,"T")))
        u = 0.0
        for sp in mech.species
            u += Float64(sol(t; idxs=_var(sys, sp.name))) * u_molar(sp.thermo, Tv)
        end
        return u
    end
    U0 = U_at(0.0); U1 = U_at(1.0)
    @test abs(U1 - U0) / abs(U0) < 1e-6
    @info "const-V ignition" T_max=T_max T_end=T_end U_drift_rel=abs(U1-U0)/abs(U0)
end

@testset "Phase 5a: H2-O2 const-P ignition — H conserved, ignition occurs" begin
    mech = load_mechanism(_H2O2_YAML_5A)
    reactor = BatchReactor(mech; mode=:adiabatic_constP)
    sys = extract_system(reactor)
    n_tot = 1.0
    u0 = Dict("n_H2"=>_Y_H2_5A*n_tot, "n_O2"=>_Y_O2_5A*n_tot, "n_N2"=>_Y_N2_5A*n_tot, "T"=>_T_INIT,
              "n_H"=>0.0, "n_O"=>0.0, "n_OH"=>0.0, "n_H2O"=>0.0, "n_HO2"=>0.0, "n_H2O2"=>0.0, "n_AR"=>0.0)
    sol = simulate(reactor, (0.0, 1.0); u0=u0, solver=Rodas5P(), reltol=1e-8, abstol=1e-12)
    @test sol.retcode == ReturnCode.Success
    # ignition: T_max > T_init + 400 K
    T_end = Float64(sol(1.0; idxs=_var(sys,"T")))
    T_max = maximum(Float64(sol(t; idxs=_var(sys,"T"))) for t in 0.0:0.01:1.0)
    @test T_max > _T_INIT + 400.0
    # H conservation (relative drift): H = Σ nᵢ·h̄ᵢ(T)
    function H_at(t)
        Tv = Float64(sol(t; idxs=_var(sys,"T")))
        h = 0.0
        for sp in mech.species
            h += Float64(sol(t; idxs=_var(sys, "n_"*sp.name))) * h_molar(sp.thermo, Tv)
        end
        return h
    end
    H0 = H_at(0.0); H1 = H_at(1.0)
    @test abs(H1 - H0) / abs(H0) < 1e-6
    @info "const-P ignition" T_max=T_max T_end=T_end H_drift_rel=abs(H1-H0)/abs(H0)
end
