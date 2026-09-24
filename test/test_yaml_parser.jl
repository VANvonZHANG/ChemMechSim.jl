using Test
using ChemMechSim

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
