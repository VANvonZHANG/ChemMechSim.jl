using Test
using ChemMechSim

@testset "config" begin
    c = MechanismConfig()
    @test c.energy === :isothermal
    @test c.constraint === :none
    @test c.eos === :off
    @test c.thermo_data === :none
    @test c.reverse_rate === :irreversible
    @test c.state_basis === :concentration

    c2 = MechanismConfig(energy=:adiabatic, constraint=:constant_volume,
                         eos=:ideal_gas, thermo_data=:nasa7,
                         reverse_rate=:thermo_equilibrium,
                         state_basis=:mole_fractions)
    @test c2.energy === :adiabatic
    @test c2.constraint === :constant_volume
    @test c2.state_basis === :mole_fractions
end
