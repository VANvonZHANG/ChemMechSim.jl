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
