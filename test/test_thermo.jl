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
