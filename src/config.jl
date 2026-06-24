# MechanismConfig: how a Mechanism is lowered into an ODESystem/DAE (spec §5.3).
# Named MechanismConfig (not the generic "Config") — it travels with a Mechanism.

struct MechanismConfig
    energy::Symbol        # energy layer:     :isothermal | :adiabatic
    constraint::Symbol    # constraint layer: :none | :constant_volume | :constant_pressure
    eos::Symbol           # equation of state: :off | :ideal_gas
    thermo_data::Symbol   # thermo data:      :none | :nasa7 | :nasa9
    reverse_rate::Symbol  # reverse rate:     :irreversible | :explicit | :thermo_equilibrium
    state_basis::Symbol   # state basis:      :concentration | :moles | :mass_fractions | :mole_fractions
end

# Default = the :kinetic zero-point (pure-kinetics bare ODE).
function MechanismConfig(;
        energy::Symbol=:isothermal,
        constraint::Symbol=:none,
        eos::Symbol=:off,
        thermo_data::Symbol=:none,
        reverse_rate::Symbol=:irreversible,
        state_basis::Symbol=:concentration)
    MechanismConfig(energy, constraint, eos, thermo_data, reverse_rate, state_basis)
end
