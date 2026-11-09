using Test
using ModelingToolkit
using ChemMechSim
using OrdinaryDiffEq: ReturnCode

@testset "Phase 5a T1: molecular_weight from composition" begin
    @test molecular_weight(Dict("H"=>2)) ≈ 0.002016  atol=1e-6          # H2
    @test molecular_weight(Dict("H"=>2,"O"=>1)) ≈ 0.018015  atol=1e-5     # H2O
    @test molecular_weight(Dict("O"=>2)) ≈ 0.031998  atol=1e-5            # O2
    @test molecular_weight(Dict("Ar"=>1)) ≈ 0.039948  atol=1e-5           # AR
    @test molecular_weight(Dict("N"=>2)) ≈ 0.028014  atol=1e-5            # N2
    @test molecular_weight(Dict("H"=>2,"O"=>2)) ≈ 0.034014  atol=1e-5     # H2O2
    # unknown element → 明确报错（spec §5.3.4 渐进式数据需求）
    @test_throws ErrorException molecular_weight(Dict("Xx"=>1))
end

@testset "Phase 5a T2: _parse_equation direct (all syntax forms)" begin
    # falloff (+M)
    r = ChemMechSim._parse_equation("2 OH (+M) <=> H2O2 (+M)")
    @test r.reactants == Dict("OH"=>2.0)
    @test r.products == Dict("H2O2"=>1.0)
    @test r.reversible && r.third_body
    # three-body + M
    r = ChemMechSim._parse_equation("H + O2 + M <=> HO2 + M")
    @test r.reactants == Dict("H"=>1.0, "O2"=>1.0)
    @test r.products == Dict("HO2"=>1.0)
    @test r.third_body
    # irreversible =>
    r = ChemMechSim._parse_equation("H + O2 => HO2")
    @test !r.reversible
    @test !r.third_body
    # float stoich + missing coef defaults to 1
    r = ChemMechSim._parse_equation("0.5 O2 + H2 <=> H2O")
    @test r.reactants["O2"] == 0.5
    @test r.reactants["H2"] == 1.0
    @test r.products["H2O"] == 1.0
    # bare = arrow (equivalent to <=>)
    r = ChemMechSim._parse_equation("A = B")
    @test r.reversible
end

@testset "Phase 5a T3: _parse_units + _a_factor + _parse_thermo + _parse_species" begin
    # _parse_units: cm→0.01, cal/mol→4.184
    ctx = ChemMechSim._parse_units(Dict("length"=>"cm","quantity"=>"mol","activation-energy"=>"cal/mol"))
    @test ctx.length_m == 0.01
    @test ctx.ea_J_per_mol == 4.184
    # _a_factor: A_canon = A_cantera × (1/length_m)^(3·(1−order))
    @test ChemMechSim._a_factor(ctx, 1) ≈ 1.0       # unimolecular (Σν=1)
    @test ChemMechSim._a_factor(ctx, 2) ≈ 1e-6      # bimolecular (Σν=2)
    @test ChemMechSim._a_factor(ctx, 3) ≈ 1e-12     # trimolecular (Σν=3)
    # default SI when no units block
    ctx_si = ChemMechSim._parse_units(nothing)
    @test ctx_si.length_m == 1.0 && ctx_si.ea_J_per_mol == 1.0
    # _parse_thermo: NASA7 from Cantera dict
    thermo_dict = Dict("model"=>"NASA7",
        "temperature-ranges"=>[200.0,1000.0,3500.0],
        "data"=>[[2.34433112,7.98052075e-03,-1.9478151e-05,2.01572094e-08,-7.37611761e-12,-917.935173,0.683010238],
                 [3.3372792,-4.94024731e-05,4.99456778e-07,-1.79566394e-10,2.00255376e-14,-950.158922,-3.20502331]])
    th = ChemMechSim._parse_thermo(thermo_dict)
    @test th isa NASA7
    @test th.Tlow == 200.0 && th.Tmid == 1000.0 && th.Thigh == 3500.0
    @test th.low_coeffs[1] == 2.34433112
    @test th.high_coeffs[6] == -950.158922
    # _parse_species: MW from composition, name_to_id filters phase species
    name_to_id = Dict("H2"=>SpeciesID(1), "AR"=>SpeciesID(2))
    species_list = [
        Dict("name"=>"H2","composition"=>Dict("H"=>2),"thermo"=>thermo_dict),
        Dict("name"=>"AR","composition"=>Dict("Ar"=>1)),
        Dict("name"=>"XYZ","composition"=>Dict("X"=>1))   # not in name_to_id → skipped
    ]
    sp, db = ChemMechSim._parse_species(species_list, name_to_id)
    @test length(sp) == 2                                 # XYZ skipped
    h2 = first(s for s in sp if s.name=="H2")
    @test h2.molecular_weight ≈ 0.002016  atol=1e-6
    @test h2.elements == Dict("H"=>2)
    @test h2.thermo isa NASA7
    @test haskey(db.entries, "H2")
    ar = first(s for s in sp if s.name=="AR")
    @test ar.molecular_weight ≈ 0.039948  atol=1e-5
    @test ar.thermo === nothing                           # AR dict had no thermo key
end

# —— Phase 5a T4: load_mechanism + per-type characterization (h2o2.yaml) ——

using ChemMechSim: ElementaryArrhenius, ThirdBodyArrhenius, TroeFalloff, TroeParams,
                   ThermoReverse, Irreversible, load_mechanism, PlogRate

const _H2O2_YAML = joinpath(@__DIR__, "data", "h2o2.yaml")

@testset "Phase 5a T4: load_mechanism(h2o2.yaml) structure" begin
    mech = load_mechanism(_H2O2_YAML)
    @test length(mech.species) == 10
    @test length(mech.reactions) == 29
    @test Set(mech.elements) == Set(["O","H","Ar","N"])
    @test Set(sp.name for sp in mech.species) == Set(["H2","H","O","O2","OH","H2O","HO2","H2O2","AR","N2"])
end

@testset "Phase 5a T4: species MW + NASA7 parsed correctly" begin
    mech = load_mechanism(_H2O2_YAML)
    h2 = first(sp for sp in mech.species if sp.name == "H2")
    @test h2.molecular_weight ≈ 0.002016  atol=1e-6
    @test h2.elements == Dict("H"=>2)
    @test h2.thermo isa NASA7
    @test h2.thermo.Tlow ≈ 200.0  atol=1e-6
    @test h2.thermo.Tmid ≈ 1000.0
    @test h2.thermo.Thigh ≈ 3500.0
    @test h2.thermo.low_coeffs[1] ≈ 2.34433112  atol=1e-9
    @test h2.thermo.high_coeffs[6] ≈ -950.158922  atol=1e-6
end

@testset "Phase 5a T4: elementary reaction units conversion (#3 O+H2<=>H+OH)" begin
    mech = load_mechanism(_H2O2_YAML)
    # Reaction 3 in h2o2.yaml (1-based among the 29)
    r3 = mech.reactions[3]
    @test r3.kinetics isa ElementaryArrhenius
    # Σν=2 (bimolecular) → A_cantera × (1/0.01)^(3·(1−2)) = A × 100^(−3) = A × 1e-6
    @test r3.kinetics.A ≈ 3.87e4 * 1e-6   rtol=1e-9
    @test r3.kinetics.b ≈ 2.7             rtol=1e-9
    # Ea: 6260 cal/mol × 4.184 = 26191.84 J/mol
    @test r3.kinetics.Ea ≈ 6260.0 * 4.184  rtol=1e-9
    @test r3.reverse_policy isa ThermoReverse             # <=> → ThermoReverse
    @test !r3.meta.duplicate
end

@testset "Phase 5a T4: three-body reaction + efficiencies (#1 2O+M<=>O2+M)" begin
    mech = load_mechanism(_H2O2_YAML)
    r1 = mech.reactions[1]
    @test r1.kinetics isa ThirdBodyArrhenius
    # Σν=3 (2 O + M) → A × (100)^(3·(1−3)) = A × 100^(−6) = A × 1e-12
    @test r1.kinetics.base.A ≈ 1.2e17 * 1e-12  rtol=1e-9
    @test r1.kinetics.base.Ea ≈ 0.0
    eff = r1.kinetics.efficiencies
    ar_id = first(sp.id for sp in mech.species if sp.name == "AR")
    @test eff[ar_id] ≈ 0.83
    h2o_id = first(sp.id for sp in mech.species if sp.name == "H2O")
    @test eff[h2o_id] ≈ 15.4
end

@testset "Phase 5a T4: falloff-Troe field alignment (#22 2OH(+M)<=>H2O2(+M))" begin
    mech = load_mechanism(_H2O2_YAML)
    r22 = mech.reactions[22]
    @test r22.kinetics isa TroeFalloff
    # Cantera Troe {A:0.7346, T3:94, T1:1756, T2:5182} → TroeParams(α=0.7346, T1=1756, T2=5182, T3=94)
    # FIELD-ALIGNED, no reorder (spec T1; lowering.jl:141 formula confirmed)
    tp = r22.kinetics.troe
    @test tp.α ≈ 0.7346   rtol=1e-9
    @test tp.T1 ≈ 1756.0  rtol=1e-9
    @test tp.T2 ≈ 5182.0  rtol=1e-9
    @test tp.T3 ≈ 94.0    rtol=1e-9
    # high-P Σν=2 (2 OH) → ×1e-6;  low-P Σν=3 (2 OH + M) → ×1e-12
    @test r22.kinetics.high_rate.A ≈ 7.4e13 * 1e-6   rtol=1e-9
    @test r22.kinetics.low_rate.A  ≈ 2.3e18 * 1e-12  rtol=1e-9
    @test r22.kinetics.low_rate.Ea ≈ -1700.0 * 4.184  rtol=1e-9
end

@testset "Phase 5a T4: duplicate flag (#24 OH+HO2<=>O2+H2O)" begin
    mech = load_mechanism(_H2O2_YAML)
    r24 = mech.reactions[24]
    @test r24.meta.duplicate == true
    @test mech.reactions[3].meta.duplicate == false       # non-duplicate contrast
    # count duplicate reactions (24-29 are duplicates in h2o2.yaml)
    n_dup = count(r -> r.meta.duplicate, mech.reactions)
    @test n_dup == 6
end

@testset "Phase 6 T4: PLOG YAML parser" begin
    mech = load_mechanism(joinpath(@__DIR__, "data", "plog_minimal.yaml"))
    @test length(mech.reactions) == 1
    kin = mech.reactions[1].kinetics
    @test kin isa PlogRate
    @test length(kin.points) == 3
    # P converted atm → Pa, sorted ascending
    @test issorted([p.P for p in kin.points])
    @test kin.points[1].P ≈ 0.1 * 101325.0
    @test kin.points[3].P ≈ 10.0 * 101325.0
    # A converted cm→m (order 1 → factor (1/0.01)^(3·0) = 1, so A unchanged here); Ea cal→J
    @test kin.points[1].A ≈ 1.2e15
    @test kin.points[1].Ea ≈ 0.0
    # gri30 (no PLOG) still loads unchanged
    gri = load_mechanism(joinpath(@__DIR__, "..", "examples", "mechanism", "gri30.yaml"))
    @test length(gri.reactions) > 0
    @test !any(r -> r.kinetics isa PlogRate, gri.reactions)
end

# —— KPP/MCM dialect (kpp-cantera-converter output): 4 parser gaps ——

const _MCM_LIKE_YAML = joinpath(@__DIR__, "data", "mcm_like_minimal.yaml")
const _NA = 6.02214076e23

@testset "KPP/MCM dialect: activation-energy K + quantity molec" begin
    ctx = ChemMechSim._parse_units(Dict("length"=>"cm", "quantity"=>"molec",
                                       "activation-energy"=>"K"))
    @test ctx.length_m == 0.01
    @test ctx.ea_J_per_mol == 8.314            # Ea given in K → Ea_SI = Ea_K × R
    @test ctx.amount_per_mol == _NA            # molecules per mole
    # A_canon = A_decl × (1/length_m)^(3(1−order)) × amount_per_mol^(order−1)
    @test ChemMechSim._a_factor(ctx, 1) ≈ 1.0
    @test ChemMechSim._a_factor(ctx, 2) ≈ 1e-6 * _NA
    @test ChemMechSim._a_factor(ctx, 3) ≈ 1e-12 * _NA^2
    # quantity: mol must be bit-for-bit today's behaviour (GRI30/FFCM2/Aramco regression)
    ctx_mol = ChemMechSim._parse_units(Dict("length"=>"cm", "quantity"=>"mol"))
    @test ctx_mol.amount_per_mol == 1.0
    @test ChemMechSim._a_factor(ctx_mol, 1) ≈ 1.0
    @test ChemMechSim._a_factor(ctx_mol, 2) ≈ 1e-6
    @test ChemMechSim._a_factor(ctx_mol, 3) ≈ 1e-12
    # quantity omitted → assumed mol (existing call sites pass only length/Ea)
    @test ChemMechSim._parse_units(Dict("length"=>"cm")).amount_per_mol == 1.0
    @test ChemMechSim._parse_units(nothing).amount_per_mol == 1.0
    # unknown quantity → loud error, never a silent wrong A
    @test_throws ErrorException ChemMechSim._parse_units(Dict("quantity"=>"furlong"))
end

@testset "constant-cp thermo parses to nothing (no usable thermo data)" begin
    th = ChemMechSim._parse_thermo(Dict("model"=>"constant-cp", "T0"=>298.15,
                                        "h0"=>0, "s0"=>0, "cp0"=>0))
    @test th === nothing
end

@testset "load_mechanism: KPP/MCM dialect fixture end-to-end" begin
    mech = load_mechanism(_MCM_LIKE_YAML)
    @test length(mech.species) == 8
    @test mech.elements == String[]                    # phase declares no `elements`
    @test all(sp -> sp.thermo === nothing, mech.species)       # constant-cp → nothing
    @test all(sp -> sp.molecular_weight == 0.0, mech.species)  # composition: {}
    @test length(mech.reactions) == 3

    # three-body, Σν = 3 (O + O2, +1 for [M]); M stripped from the equation by the parser
    r1 = mech.reactions[1]
    @test r1.kinetics isa ThirdBodyArrhenius
    @test r1.kinetics.base.A ≈ 1.5442e-27 * 1e-12 * _NA^2  rtol=1e-9
    @test r1.kinetics.base.b ≈ -2.6  rtol=1e-9
    @test r1.kinetics.base.Ea == 0.0

    # elementary bimolecular, Σν = 2
    r2 = mech.reactions[2]
    @test r2.kinetics isa ElementaryArrhenius
    @test r2.kinetics.A ≈ 1.4e-12 * 1e-6 * _NA  rtol=1e-9

    # unimolecular: A unchanged (factor 1), Ea K → J/mol
    r3 = mech.reactions[3]
    @test r3.kinetics.A ≈ 1.0e-5  rtol=1e-9
    @test r3.kinetics.Ea ≈ 100.0 * 8.314  rtol=1e-9
    @test r3.reverse_policy isa Irreversible            # `=>` in the file
end

@testset "brusselator.yaml: empty-side source/sink terms (abstract species)" begin
    mech = load_mechanism(joinpath(@__DIR__, "data", "brusselator.yaml"))
    @test length(mech.species) == 2
    @test sort([s.name for s in mech.species]) == ["X", "Y"]
    @test length(mech.reactions) == 4
    # source (=> X) and sink (X =>): exactly one empty reactant / product side
    @test count(r -> isempty(r.reactants), mech.reactions) == 1
    @test count(r -> isempty(r.products),  mech.reactions) == 1
    # all constant-rate, irreversible; SI units leave A unchanged
    for r in mech.reactions
        @test r.kinetics isa ElementaryArrhenius
        @test r.kinetics.b == 0.0 && r.kinetics.Ea == 0.0
        @test r.reverse_policy isa Irreversible
    end
    r3 = mech.reactions[3]                       # X => Y with A = 3.0
    @test r3.kinetics.A == 3.0
end

# —— rate-type registry (2026-09-24 direct-load spec §1) ——————————————————————————
# load_mechanism dispatches type strings through DEFAULT_RATE_PARSERS; the
# rate_type_handlers kwarg merges user entries OVER the defaults (override allowed).
# Unknown types are skipped with ONE aggregated warning that names the kwarg.

struct _RegToy <: AbstractKinetics
    k0::Float64
end
ChemMechSim.paramspec(kin::_RegToy) = (afactor(:k0, "", 0.0),)   # → parameter k_{j}_A
ChemMechSim.body(kin::_RegToy)      = (k0, T) -> k0
ChemMechSim.needs_T(kin::_RegToy)   = false

_regtoy_yaml(reactions_block) = """
phases:
  - name: gas
    thermo: ideal-gas
    species: [A, B]
    reactions: all
species:
  - name: A
    composition: {Ar: 1}
    thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}
  - name: B
    composition: {Ar: 1}
    thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}
reactions:
$reactions_block
"""

@testset "rate-type registry" begin
    # 1. custom type via kwarg: parsed (not skipped), correct instance, numeric rate works
    path = tempname() * ".yaml"
    write(path, _regtoy_yaml("  - {equation: \"A => B\", type: reg-toy, k0: 0.5}\n"))
    mech = load_mechanism(path; rate_type_handlers = Dict{String,Function}(
        "reg-toy" => (rxn, reactants, name_to_id, ctx) -> _RegToy(Float64(rxn["k0"]))))
    @test length(mech.reactions) == 1
    @test mech.reactions[1].kinetics isa _RegToy
    @test rate_constant(mech.reactions[1].kinetics, 300.0) == 0.5

    # 2. a user entry can OVERRIDE a built-in type ("elementary"); without it, the built-in runs
    path_el = tempname() * ".yaml"
    write(path_el, _regtoy_yaml(
        "  - {equation: \"A => B\", rate-constant: {A: 1.0e-12, b: 0.0, Ea: 0.0}}\n"))
    mech2 = load_mechanism(path_el; rate_type_handlers = Dict{String,Function}(
        "elementary" => (rxn, reactants, name_to_id, ctx) -> _RegToy(42.0)))
    @test mech2.reactions[1].kinetics isa _RegToy
    mech2b = load_mechanism(path_el)
    @test mech2b.reactions[1].kinetics isa ElementaryArrhenius

    # 3. unknown types are skipped with ONE aggregated warning naming the kwarg
    path2 = tempname() * ".yaml"
    write(path2, _regtoy_yaml(
        "  - {equation: \"A => B\", type: custom-x, k0: 1.0}\n" *
        "  - {equation: \"B => A\", type: custom-y, k0: 1.0}\n" *
        "  - {equation: \"A => B\", type: custom-x, k0: 1.0}\n"))
    logs, mech3 = Test.collect_test_logs() do
        load_mechanism(path2)
    end
    @test length(mech3.reactions) == 0                       # all three dropped
    warns = [l for l in logs if l.level == Base.CoreLogging.Warn]
    @test length(warns) == 1                                 # ONE warning, not three
    @test occursin("custom-x ×2", warns[1].message)
    @test occursin("custom-y ×1", warns[1].message)
    @test occursin("rate_type_handlers", warns[1].message)

    # 4. the four built-ins are registered
    @test sort(collect(keys(ChemMechSim.DEFAULT_RATE_PARSERS))) ==
          ["elementary", "falloff", "pressure-dependent-Arrhenius", "three-body"]

    # 5. handler-facing unit helpers match the parse-time conversions
    ctx = ChemMechSim._parse_units(Dict("length" => "cm", "quantity" => "molec",
                                        "activation-energy" => "K"))
    @test convert_afactor(3.8e-13, ctx, 2) ≈ 3.8e-13 * 1e-6 * 6.02214076e23   # cm³/molec/s → m³/mol/s
    @test convert_afactor(0.0089, ctx, 1) ≈ 0.0089                            # order 1: s⁻¹ unchanged
    @test ea_to_J_per_mol(600.0, ctx) == 600.0 * 8.314                        # K → J/mol
end

# —— default-efficiency (three-body / falloff) ———————————————————————————————————
# Cantera semantics: unlisted third bodies take the row's `default-efficiency` (converter
# output uses 0.0 on catalyst-preservation rows). Key absent → 1.0 → the historical
# sparse efficiencies dict, bit-identically (zero-drift guard below).

_p1_yaml(reactions_block) = """
phases:
  - name: gas
    thermo: ideal-gas
    species: [A, B, C]
    reactions: all
species:
  - name: A
    composition: {Ar: 1}
    thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}
  - name: B
    composition: {Ar: 1}
    thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}
  - name: C
    composition: {Ar: 1}
    thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}
reactions:
$reactions_block
"""

@testset "P1: default-efficiency" begin
    sid(mech, name) = first(sp.id for sp in mech.species if sp.name == name)

    # a) three-body, default-efficiency: 0.0 → MATERIALIZED: every species present,
    #    listed → its α, unlisted → 0.0
    path_a = tempname() * ".yaml"
    write(path_a, _p1_yaml(
        "  - {equation: \"A + M => B + M\", type: three-body, " *
        "rate-constant: {A: 1.0e-30, b: -1.0, Ea: 0.0}, " *
        "efficiencies: {C: 1.0}, default-efficiency: 0.0}\n"))
    mech_a = load_mechanism(path_a)
    @test mech_a.reactions[1].kinetics isa ThirdBodyArrhenius
    eff_a = mech_a.reactions[1].kinetics.efficiencies
    @test length(eff_a) == 3                       # all three species materialized
    @test eff_a[sid(mech_a, "C")] == 1.0           # listed → its α
    @test eff_a[sid(mech_a, "A")] == 0.0           # unlisted → default 0.0
    @test eff_a[sid(mech_a, "B")] == 0.0

    # b) falloff, default-efficiency: 2.5 → unlisted species α == 2.5
    path_b = tempname() * ".yaml"
    write(path_b, _p1_yaml(
        "  - {equation: \"2 A (+M) => B (+M)\", type: falloff, " *
        "high-P-rate-constant: {A: 1.0e-10, b: 0.0, Ea: 0.0}, " *
        "low-P-rate-constant: {A: 1.0e-30, b: -1.0, Ea: 0.0}, " *
        "Troe: {A: 0.6, T3: 90.0, T1: 1500.0}, " *
        "efficiencies: {C: 1.0}, default-efficiency: 2.5}\n"))
    mech_b = load_mechanism(path_b)
    @test mech_b.reactions[1].kinetics isa TroeFalloff
    eff_b = mech_b.reactions[1].kinetics.efficiencies
    @test eff_b[sid(mech_b, "C")] == 1.0
    @test eff_b[sid(mech_b, "A")] == 2.5           # unlisted → default 2.5
    @test eff_b[sid(mech_b, "B")] == 2.5

    # c) key absent → sparse dict exactly as before: only the listed species present
    path_c = tempname() * ".yaml"
    write(path_c, _p1_yaml(
        "  - {equation: \"A + M => B + M\", type: three-body, " *
        "rate-constant: {A: 1.0e-30, b: -1.0, Ea: 0.0}, " *
        "efficiencies: {C: 1.0}}\n"))
    mech_c = load_mechanism(path_c)
    eff_c = mech_c.reactions[1].kinetics.efficiencies
    @test collect(keys(eff_c)) == [sid(mech_c, "C")]
    @test eff_c[sid(mech_c, "C")] == 1.0

    # d) default-efficiency: 1.0 → sparse (early return), NOT materialized
    path_d = tempname() * ".yaml"
    write(path_d, _p1_yaml(
        "  - {equation: \"A + M => B + M\", type: three-body, " *
        "rate-constant: {A: 1.0e-30, b: -1.0, Ea: 0.0}, " *
        "efficiencies: {C: 2.0}, default-efficiency: 1.0}\n"))
    mech_d = load_mechanism(path_d)
    eff_d = mech_d.reactions[1].kinetics.efficiencies
    @test collect(keys(eff_d)) == [sid(mech_d, "C")]
    @test eff_d[sid(mech_d, "C")] == 2.0

    # e) efficiencies referencing an unknown species → loud error
    path_e = tempname() * ".yaml"
    write(path_e, _p1_yaml(
        "  - {equation: \"A + M => B + M\", type: three-body, " *
        "rate-constant: {A: 1.0e-30, b: -1.0, Ea: 0.0}, " *
        "efficiencies: {Z: 1.0}, default-efficiency: 0.5}\n"))
    @test_throws ErrorException load_mechanism(path_e)

    # f) NO efficiencies key + default ≠ 1.0 → every species at the default.
    #    Guards the first early-return: regressing it to `eff_dict === nothing &&
    #    return out` would silently hand every species α=1.0 (a chemically wrong
    #    rate with no error) and no other case would notice.
    path_f = tempname() * ".yaml"
    write(path_f, _p1_yaml(
        "  - {equation: \"A + M => B + M\", type: three-body, " *
        "rate-constant: {A: 1.0e-30, b: -1.0, Ea: 0.0}, " *
        "default-efficiency: 0.0}\n"))
    mech_f = load_mechanism(path_f)
    eff_f = mech_f.reactions[1].kinetics.efficiencies
    @test length(eff_f) == 3                       # nothing listed → all materialized
    @test all(==(0.0), values(eff_f))
end

@testset "P2: underscore species names" begin
    # WRF mechanisms use names like O1D_CB4; underscore joins the allowed charset
    @test ChemMechSim._parse_terms("2 O1D_CB4 + N2") == Dict("O1D_CB4"=>2.0, "N2"=>1.0)
    # regression guards for the already-widened class: each parses to itself with coeff 1.0
    @test ChemMechSim._parse_terms("CH*") == Dict("CH*"=>1.0)                 # asterisk (Aramco/FFCM2)
    @test ChemMechSim._parse_terms("C2H4O1-2") == Dict("C2H4O1-2"=>1.0)       # hyphenated isomer label
    @test ChemMechSim._parse_terms("CH2(S)") == Dict("CH2(S)"=>1.0)           # GRI30 singlet methylene
    # end-to-end through the equation parser (bare = arrow, N2 on both sides)
    r = ChemMechSim._parse_equation("O1D_CB4 + N2 = O + O2 + N2")
    @test r.reactants == Dict("O1D_CB4"=>1.0, "N2"=>1.0)
    @test r.products == Dict("O"=>1.0, "O2"=>1.0, "N2"=>1.0)
end

# —— kinetic-order override (P3, 2026-11-09) ————————————————————————————————————
# Row-level apparent-order override, MAPPING form only: the effective total order is
# ALWAYS the per-species sum (override where given, stoichiometric ν otherwise) — there
# is deliberately NO scalar form (P3 grilling decision), so N and the mass-action powers
# cannot disagree by construction. Rate-law only: powers, A-factor conversion and the
# k-param unit derivation all consume the same per-species powers; the ODE net
# stoichiometry (and Δν for K_c / energy) stays stoichiometric — the zero-order test
# below asserts both halves (rate law linear, ledger still consuming A).

_p3_yaml(reactions_block) = """
units: {length: cm, quantity: molec, activation-energy: K}
phases:
  - name: gas
    thermo: ideal-gas
    species: [A, B, C]
    reactions: all
species:
  - name: A
    composition: {Ar: 1}
    thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}
  - name: B
    composition: {Ar: 1}
    thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}
  - name: C
    composition: {Ar: 1}
    thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}
reactions:
$reactions_block
"""

@testset "P3: kinetic-order mapping — orders, effective order, A-factor" begin
    sid(mech, name) = first(sp.id for sp in mech.species if sp.name == name)

    # partial override: the 0.67 LEDGER coefficient decouples from the 1.0 POWER.
    # Effective N = 1.0 (A) + 1.0 (B, unlisted → ν) = 2.0 — never stated directly.
    path = tempname() * ".yaml"
    write(path, _p3_yaml(
        "  - {equation: \"0.67 A + B => 0.67 A + C\", " *
        "rate-constant: {A: 1.0e-12, b: 0.0, Ea: 0.0}, kinetic-order: {A: 1.0}}\n"))
    mech = load_mechanism(path)
    rx = mech.reactions[1]
    @test rx.meta.orders == Dict(sid(mech, "A") => 1.0)
    @test ChemMechSim.rate_powers(rx) ==
        Dict(sid(mech, "A") => 1.0, sid(mech, "B") => 1.0)
    @test ChemMechSim.effective_order(rx) == 2.0
    @test rx.reactants[sid(mech, "A")] == 0.67                  # ledger untouched
    # A-factor conversion consumed N=2 (cm³ molec → m³ mol): 1e-6 · N_A
    @test rx.kinetics.A ≈ 1.0e-12 * 1e-6 * _NA  rtol=1e-9

    # peroxy shape: monomolecular equation, sole reactant carries the full order 2
    path2 = tempname() * ".yaml"
    write(path2, _p3_yaml(
        "  - {equation: \"A => B + C\", rate-constant: {A: 1.0e-12, b: 0.0, Ea: 0.0}, " *
        "kinetic-order: {A: 2.0}}\n"))
    mech2 = load_mechanism(path2)
    @test ChemMechSim.effective_order(mech2.reactions[1]) == 2.0
    @test mech2.reactions[1].kinetics.A ≈ 1.0e-12 * 1e-6 * _NA  rtol=1e-9
    # control: identical row WITHOUT the key → N=1 → A unchanged
    path3 = tempname() * ".yaml"
    write(path3, _p3_yaml(
        "  - {equation: \"A => B + C\", rate-constant: {A: 1.0e-12, b: 0.0, Ea: 0.0}}\n"))
    mech3 = load_mechanism(path3)
    @test isempty(mech3.reactions[1].meta.orders)
    @test ChemMechSim.effective_order(mech3.reactions[1]) == 1.0
    @test mech3.reactions[1].kinetics.A ≈ 1.0e-12  rtol=1e-12

    # empty mapping {} is treated as no key
    path4 = tempname() * ".yaml"
    write(path4, _p3_yaml(
        "  - {equation: \"0.67 A + B => 0.67 A + C\", " *
        "rate-constant: {A: 1.0e-12, b: 0.0, Ea: 0.0}, kinetic-order: {}}\n"))
    mech4 = load_mechanism(path4)
    @test isempty(mech4.reactions[1].meta.orders)
    @test ChemMechSim.effective_order(mech4.reactions[1]) == 0.67 + 1.0

    # ctx-ridden 2-arg convert_afactor: same as 3-arg with ctx.effective_order
    ctx = ChemMechSim._UnitCtx(0.01, 8.314, 1234.0, 2.5)
    @test convert_afactor(7.0e-11, ctx) == convert_afactor(7.0e-11, ctx, 2.5)
end

@testset "P3: zero order — species leaves the rate law, stays in the ledger" begin
    # SI units (no units block) → A unchanged; k = 2.0 constant; rate = 2·[A]^0 ≡ 2.
    # Zero-order kinetics: A decays LINEARLY (3 - 2t), not exponentially; B grows 2t.
    path = tempname() * ".yaml"
    write(path, _p1_yaml(
        "  - {equation: \"A => B\", rate-constant: {A: 2.0, b: 0.0, Ea: 0.0}, " *
        "kinetic-order: {A: 0.0}}\n"))
    mech = load_mechanism(path)
    r = BatchReactor(mech)
    sol = simulate(r, (0.0, 1.2); u0 = Dict("A" => 3.0, "B" => 0.0, "C" => 0.0),
                   reltol = 1e-10, abstol = 1e-10)
    iA = findfirst(s -> String(ModelingToolkit.getname(s)) == "A", unknowns(extract_system(r)))
    iB = findfirst(s -> String(ModelingToolkit.getname(s)) == "B", unknowns(extract_system(r)))
    @test sol.u[end][iA] ≈ 3.0 - 2.0 * 1.2  atol = 1e-6     # linear, not exp
    @test sol.u[end][iB] ≈ 2.0 * 1.2        atol = 1e-6
end

@testset "P3: validation — loud errors" begin
    loadk(block) = (p = tempname() * ".yaml"; write(p, _p3_yaml(block));
                    load_mechanism(p))
    # scalar form rejected: N is always the per-species sum, never stated directly
    err = try loadk("  - {equation: \"A => B\", rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}, " *
                    "kinetic-order: 2}\n"); nothing
        catch e; e end
    @test err !== nothing && occursin("never stated directly", sprint(showerror, err))
    # key must be a reactant (C is a product here)
    @test_throws ErrorException loadk(
        "  - {equation: \"A + B => C\", rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}, " *
        "kinetic-order: {C: 1.0}}\n")
    # unknown species key
    @test_throws ErrorException loadk(
        "  - {equation: \"A => B\", rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}, " *
        "kinetic-order: {ZZ: 1.0}}\n")
    # negative value (a negative power would divide by a concentration)
    @test_throws ErrorException loadk(
        "  - {equation: \"A + B => C\", rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}, " *
        "kinetic-order: {A: -0.5}}\n")
    # enhanced rows: [M]_eff is not a per-species power
    @test_throws ErrorException loadk(
        "  - {equation: \"A + B + M => C + M\", type: three-body, " *
        "rate-constant: {A: 1.0e-30, b: -1.0, Ea: 0.0}, kinetic-order: {A: 1.0}}\n")
    @test_throws ErrorException loadk(
        "  - {equation: \"A + B (+M) => C (+M)\", type: falloff, " *
        "high-P-rate-constant: {A: 1.0e-10, b: 0.0, Ea: 0.0}, " *
        "low-P-rate-constant: {A: 1.0e-30, b: -1.0, Ea: 0.0}, kinetic-order: {A: 1.0}}\n")
end

@testset "P3: reversible rows reject kinetic-order (thermodynamic reverse needs mass-action)" begin
    # parse-time: <=> + kinetic-order → loud error. With N_eff ≠ Σν one kf would need
    # two different units (conc^(1-N_eff)·s⁻¹ forward vs conc^(1-Σν)·s⁻¹ reverse) — the
    # symbolic unit check rejects exactly this. P3 grilling amendment, 2026-11-09.
    path = tempname() * ".yaml"
    write(path, _p3_yaml(
        "  - {equation: \"A + B <=> C\", rate-constant: {A: 1.0e-12, b: 0.0, Ea: 0.0}, " *
        "kinetic-order: {A: 0.5}}\n"))
    @test_throws ErrorException load_mechanism(path)

    # programmatic bypass guard: hand-built ThermoReverse + orders fails loudly at
    # lowering with the reason, not as a cryptic unit-check failure
    gri = load_mechanism(joinpath(@__DIR__, "..", "examples", "mechanism", "gri30.yaml"))
    j = findfirst(rx -> rx.reverse_policy isa ThermoReverse &&
                        rx.kinetics isa ElementaryArrhenius, gri.reactions)
    rx = gri.reactions[j]
    rxo = ReactionData(reactants = rx.reactants, products = rx.products,
                       kinetics = rx.kinetics, reverse_policy = rx.reverse_policy,
                       meta = ReactionMeta(orders = Dict(first(keys(rx.reactants)) => 0.5)))
    mecho = Mechanism(species = gri.species,
                      reactions = [i == j ? rxo : rx for (i, rx) in enumerate(gri.reactions)],
                      thermo = gri.thermo, elements = gri.elements)
    @test_throws ErrorException BatchReactor(mecho)
end

@testset "P3: fixtures carry no kinetic-order key (zero-drift nail)" begin
    for f in ("gri30.yaml", "h2o2.yaml", "FFCM2.yaml", "AramcoMech3.0.yaml")
        txt = read(joinpath(@__DIR__, "..", "examples", "mechanism", f), String)
        @test !occursin("kinetic-order", txt)
    end
end

# —— negative product stoichiometry (NCAR MM X*OO ensemble-bookkeeping dialect) ———————
# Semantics guaranteed downstream: rate law = mass-action over REACTANTS ONLY; a negative
# product coefficient contributes a negative stoichiometric term to that species' ODE
# (d[C]/dt = -1·rate for `A => B - 1.0 C`) — exactly NCAR MM gear DIFFUN's ct += sc·rt.

@testset "negative product coefficients: parse-level" begin
    # canonical dialect form: whitespace-bounded minus separates product terms like + does
    r = ChemMechSim._parse_equation("A => B - 1.0 C")
    @test r.reactants == Dict("A"=>1.0)
    @test r.products == Dict("B"=>1.0, "C"=>-1.0)
    @test !r.reversible && !r.third_body
    # the de-risk-fixture form (NCAR MM project, julia/derisk_negative.yaml): species on
    # BOTH sides stays in separate dicts (net A = 1 - 1 = 0 happens in the ODE, not here)
    r2 = ChemMechSim._parse_equation("A + 2.0 B => 1.0 A - 2.0 XAOO")
    @test r2.reactants == Dict("A"=>1.0, "B"=>2.0)
    @test r2.products == Dict("A"=>1.0, "XAOO"=>-2.0)
    # explicit per-term sign after the + separator: -X, -1.0 X, - 1.0 X
    @test ChemMechSim._parse_equation("A => B + -X").products == Dict("B"=>1.0, "X"=>-1.0)
    @test ChemMechSim._parse_equation("A => B + -1.0 X").products == Dict("B"=>1.0, "X"=>-1.0)
    @test ChemMechSim._parse_equation("A => B + - 1.0 X").products == Dict("B"=>1.0, "X"=>-1.0)
    @test ChemMechSim._parse_equation("A => B - X").products == Dict("B"=>1.0, "X"=>-1.0)
    @test ChemMechSim._parse_equation("A => B -  1.0  X").products == Dict("B"=>1.0, "X"=>-1.0)
    # repeated terms net with their signs within one side
    @test ChemMechSim._parse_equation("A => 2.0 C - 0.5 C").products == Dict("C"=>1.5)
    @test ChemMechSim._parse_equation("A => C - 2.0 C").products == Dict("C"=>-1.0)
    # hyphenated species names are unaffected: sign binds only at term start, and the
    # ` - ` separator can never occur inside a space-free species name
    r3 = ChemMechSim._parse_equation("C4H5-2 + OH <=> 2.0 C4H5-2")
    @test r3.reactants == Dict("C4H5-2"=>1.0, "OH"=>1.0)
    @test r3.products == Dict("C4H5-2"=>2.0)
    @test ChemMechSim._parse_equation("A => B - 1.0 C4H5-2").products == Dict("B"=>1.0, "C4H5-2"=>-1.0)
    # reactant side must stay nonnegative: loud parse error, never a silent [X]^-ν rate law
    @test_throws ErrorException ChemMechSim._parse_equation("A - 1.0 B => C")
    @test_throws ErrorException ChemMechSim._parse_equation("A + -1.0 B => C")
end

@testset "negative product coefficients: load + lowering (A => B - 1.0 C)" begin
    yaml = """
    phases:
      - name: gas
        thermo: ideal-gas
        elements: [Ar]
        species: [A, B, C]
        reactions: all
    species:
      - {name: A, composition: {Ar: 1}, thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
      - {name: B, composition: {Ar: 1}, thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
      - {name: C, composition: {Ar: 1}, thermo: {model: constant-cp, h0: 0, s0: 0, cp0: 0}}
    reactions:
      - {equation: 'A => B - 1.0 C', rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}}
    """
    path = tempname() * ".yaml"
    write(path, yaml)
    mech = load_mechanism(path)
    @test length(mech.reactions) == 1
    rx = mech.reactions[1]
    @test rx.reactants == Dict(SpeciesID(1) => 1.0)
    @test rx.products[SpeciesID(2)] == 1.0            # B
    @test rx.products[SpeciesID(3)] == -1.0           # C: negative bookkeeping coefficient
    @test rx.reverse_policy isa Irreversible
    @test rx.kinetics.A == 1.0                        # SI units, order 1: A unchanged

    # lowering: rate = k·[A] only; d[A]/dt = -rate, d[B]/dt = +rate, d[C]/dt = -rate.
    # Analytic solution at k=1: A=e⁻ᵗ, B=1-e⁻ᵗ, C=C₀-(1-e⁻ᵗ).
    phase = ChemMechSim.ChemPhaseSystem(mech; config=MechanismConfig())
    sys = ChemMechSim.extract_system(phase)
    var(n) = unknowns(sys)[findfirst(s -> String(ModelingToolkit.getname(s)) == n, unknowns(sys))]
    A, B, C = var("A"), var("B"), var("C")
    sol = simulate(phase, (0.0, 2.0); u0=Dict("A"=>1.0, "B"=>0.0, "C"=>0.5),
                   reltol=1e-10, abstol=1e-12)
    for t in (0.5, 1.0, 2.0)
        @test sol(t; idxs=A) ≈ exp(-t)          rtol=1e-8
        @test sol(t; idxs=B) ≈ 1 - exp(-t)      rtol=1e-8
        @test sol(t; idxs=C) ≈ 0.5 - (1 - exp(-t))  rtol=1e-8
    end
    # rate ∝ [A] ONLY: perturbing [C](0) must not move the A/B trajectories (a rate law
    # accidentally containing [C] — e.g. a mis-signed mass-action — would diverge here)
    sol2 = simulate(phase, (0.0, 2.0); u0=Dict("A"=>1.0, "B"=>0.0, "C"=>5.0),
                    reltol=1e-10, abstol=1e-12)
    @test sol2(2.0; idxs=A) ≈ sol(2.0; idxs=A)   rtol=1e-8
    @test sol2(2.0; idxs=B) ≈ sol(2.0; idxs=B)   rtol=1e-8
    @test sol2(2.0; idxs=C) ≈ 5.0 - (1 - exp(-2.0))  rtol=1e-8

    # validation semantics with signed stoichiometry (documented decisions):
    rep = validate(mech)
    # (a) element conservation counts signed coefficients: consuming Ar-composed C at -1
    #     makes the reaction genuinely unbalanced → error (ensemble species with empty
    #     composition are skipped with a warning instead, so MCM dialect files stay quiet)
    @test any(e -> occursin("not element-balanced", e), rep.errors)
    # (b) duplicate detection compares exact SIGNED per-side maps (no cross-side netting):
    #     two identical negative-stoich equations still see each other as twins
    yaml_dup = replace(yaml,
        "- {equation: 'A => B - 1.0 C', rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}}" =>
        "- {equation: 'A => B - 1.0 C', rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}}\n" *
        "  - {equation: 'A => B - 1.0 C', rate-constant: {A: 2.0, b: 0.0, Ea: 0.0}}\n" *
        "  - {equation: 'A => B - 1.0 C', rate-constant: {A: 3.0, b: 0.0, Ea: 0.0}}")
    path_dup = tempname() * ".yaml"
    write(path_dup, yaml_dup)
    mech_dup = load_mechanism(path_dup)
    @test length(mech_dup.reactions) == 3
    rep_dup = validate(mech_dup)
    @test count(w -> occursin("same stoichiometry", w), rep_dup.warnings) == 3   # none flagged duplicate
    # ... and WITH the duplicate flag on all three the warning stays silent
    yaml_flagged = replace(yaml_dup,
        "- {equation: 'A => B - 1.0 C', rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}}" =>
        "- {equation: 'A => B - 1.0 C', rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}, duplicate: true}",
        "- {equation: 'A => B - 1.0 C', rate-constant: {A: 2.0, b: 0.0, Ea: 0.0}}" =>
        "- {equation: 'A => B - 1.0 C', rate-constant: {A: 2.0, b: 0.0, Ea: 0.0}, duplicate: true}",
        "- {equation: 'A => B - 1.0 C', rate-constant: {A: 3.0, b: 0.0, Ea: 0.0}}" =>
        "- {equation: 'A => B - 1.0 C', rate-constant: {A: 3.0, b: 0.0, Ea: 0.0}, duplicate: true}")
    path_fl = tempname() * ".yaml"
    write(path_fl, yaml_flagged)
    rep_fl = validate(load_mechanism(path_fl))
    @test count(w -> occursin("same stoichiometry", w), rep_fl.warnings) == 0

    # reversible + negative product coefficient: parses, but lowering must FAIL LOUDLY
    # (reverse mass-action over products would divide by [C]) — never a silent wrong rate
    path_rev = tempname() * ".yaml"
    write(path_rev, replace(yaml, "A => B" => "A <=> B"))
    mech_rev = load_mechanism(path_rev)
    @test mech_rev.reactions[1].reverse_policy isa ThermoReverse
    err = try
        ChemMechSim.ChemPhaseSystem(mech_rev; config=MechanismConfig())
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("negative product", err.msg)
end
