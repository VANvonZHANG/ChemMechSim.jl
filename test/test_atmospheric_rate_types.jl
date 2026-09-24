using Test
using ChemMechSim

# The MCM rate types are EXAMPLE-side — parsed through load_mechanism's rate_type_handlers
# registry, no src involvement. Like the old preprocessor tests, this includes the example
# file directly: it defines types + pure functions, no mechanism needed.
include(joinpath(@__DIR__, "..", "examples", "atmospheric", "tools", "mcm_rate_types.jl"))

@testset "ZenithPhotolysis: the J parameter, T-free" begin
    # l=6.073e-5, m=1.743, n=0.474 is the real MCM entry #36 (O3 => O1D). The afactor
    # default must be J at overhead sun: l·exp(−n) — the value the frozen box solves with
    # and the diurnal driver's guard re-derives.
    kin = ZenithPhotolysis(6.0730e-05, 1.7430e+00, 4.7400e-01, 6.0730e-05 * exp(-0.474))
    @test rate_constant(kin, 250.0) ≈ 6.0730e-05 * exp(-0.474)   rtol = 1e-12
    @test rate_constant(kin, 350.0) == rate_constant(kin, 250.0) # T-free by construction
    @test needs_T(kin) == false
    spec = paramspec(kin)
    @test length(spec) == 1 && spec[1][1] == :A && spec[1][2] isa AFactor
end

@testset "SigmoidBranching: exact closed form, sign preserved" begin
    # REGRESSION (ported from the preprocessor tests): MCM's CH3O2+HO2 arrives as THREE
    # entries — a plain Arrhenius carrying k_total plus SIGNED sigmoid corrections — and
    # the channels must sum to MCM's own rate. Pinned from MCM's <53>/<3778> at 298 K.
    T0 = 298.0
    a, b, c, d = 3.8e-13, 780.0, 498.0, -1160.0
    σ = 1.0 / (1.0 + c * exp(d / T0))
    ktot = a * exp(b / T0)

    sig_neg = SigmoidBranching(-a, b, c, d, false)   # A < 0: the deliberate signed correction
    sig_pos = SigmoidBranching(+a, b, c, d, false)
    @test rate_constant(sig_neg, T0) ≈ -ktot * σ     rtol = 1e-12
    @test rate_constant(sig_pos, T0) ≈ +ktot * σ     rtol = 1e-12
    @test needs_T(sig_pos) == true
    # exactness at a second temperature (the old flattener could only do 298 K)
    @test rate_constant(sig_pos, 310.0) ≈ a * exp(b / 310.0) / (1 + c * exp(d / 310.0)) rtol = 1e-12
    # complement branch selects (1 − σ)
    @test rate_constant(SigmoidBranching(a, b, c, d, true), T0) ≈ ktot * (1 - σ) rtol = 1e-12
end

@testset "handler wiring through load_mechanism" begin
    # Minimal KPP/MCM-dialect file (real entries): the registry consults the handlers and
    # the unit conversion (order-2 A under quantity: molec) is applied by the handler.
    yaml = """
    units: {length: cm, time: s, quantity: molec, activation-energy: K}
    phases:
      - name: gas
        thermo: ideal-gas
        species: [NO2, NO, O, CH3O2, HO2, CH3OOH, HCHO]
        reactions: all
    species:
      - {name: NO2,    composition: {},      thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
      - {name: NO,     composition: {},      thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
      - {name: O,      composition: {},      thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
      - {name: CH3O2,  composition: {},      thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
      - {name: HO2,    composition: {Ar: 1}, thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
      - {name: CH3OOH, composition: {Ar: 1}, thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
      - {name: HCHO,   composition: {Ar: 1}, thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
    reactions:
      # real MCM entry #39 (J_NO2): l=1.165e-2, m=0.244, n=0.267
      - {equation: "NO2 => NO + O", type: zenith-angle-photolysis,
         l: 1.165e-02, m: 2.44e-01, n: 2.67e-01}
      # real MCM <53>/<3778> siblings (order 2, quantity molec): plain + sigmoid
      - {equation: "CH3O2 + HO2 => CH3OOH", duplicate: true,
         rate-constant: {A: 3.8e-13, b: 0.0, Ea: -780.0}}
      - {equation: "CH3O2 + HO2 => HCHO", type: sigmoid-branching,
         A: 3.8e-13, B: 780.0, C: 498.0, D: -1160.0}
    """
    path = tempname() * ".yaml"
    write(path, yaml)
    mech = load_mechanism(path; rate_type_handlers = mcm_rate_handlers())

    @test length(mech.reactions) == 3
    jno2 = mech.reactions[1].kinetics
    @test jno2 isa ZenithPhotolysis
    @test (jno2.l, jno2.m, jno2.n) == (1.165e-02, 2.44e-01, 2.67e-01)
    @test jno2.A ≈ 0.008920091282571607 rtol = 1e-12   # pinned: J(l,m,n,cz=1), independently evaluated
    sig = mech.reactions[3].kinetics
    @test sig isa SigmoidBranching
    # order-2 A under quantity:molec → converted to canonical units by the handler:
    # cm³/molec/s → m³/mol/s = ×(1e-2)³·N_A  (pinned by test_yaml_parser.jl's _a_factor tests)
    @test sig.A ≈ 3.8e-13 * 1e-6 * 6.02214076e23       rtol = 1e-9
    # the two channels SUM to MCM's own total (in canonical units) — the shipped regression.
    # The plain sibling is a BUILT-IN ElementaryArrhenius: built-ins declare explicit
    # symbolic_kf instead of paramspec, so their numeric form is evaluated directly
    # (A·T^b·exp(−Ea/RT), parsed Ea in J/mol — the same convention tools/budget.jl uses).
    σ = 1.0 / (1.0 + 498.0 * exp(-1160.0 / 298.0))
    ktot = 3.8e-13 * 1e-6 * 6.02214076e23 * exp(780.0 / 298.0)
    r2 = mech.reactions[2].kinetics
    k_plain = r2.A * 298.0^r2.b * exp(-r2.Ea / (8.314 * 298.0))
    k_sig   = rate_constant(sig, 298.0)
    @test k_sig   ≈ ktot * σ   rtol = 1e-9              # and NOT 2× (the shipped double-count bug)
    # the plain sibling carries MCM's FULL total (ktot) through the parser's own order-2
    # molec conversion — the real mechanism's negative sibling then subtracts ktot·σ from
    # the same equation; per-entry sign semantics are proven in testset 2 above
    @test k_plain ≈ ktot               rtol = 1e-9
end

@testset "handler order guard" begin
    yaml = """
    phases:
      - name: gas
        thermo: ideal-gas
        species: [A, B]
        reactions: all
    species:
      - {name: A, composition: {Ar: 1}, thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
      - {name: B, composition: {Ar: 1}, thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
    reactions:
      - {equation: "A + B => A", type: zenith-angle-photolysis, l: 1.0, m: 1.0, n: 0.5}
    """
    path = tempname() * ".yaml"
    write(path, yaml)
    @test_throws ErrorException load_mechanism(path; rate_type_handlers = mcm_rate_handlers())
end
