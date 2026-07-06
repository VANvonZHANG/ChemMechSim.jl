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
