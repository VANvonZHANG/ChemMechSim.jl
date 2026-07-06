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
