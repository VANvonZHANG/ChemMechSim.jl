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
                   ThermoReverse, Irreversible, load_mechanism

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
