# MechanismConfig: how a Mechanism is lowered into an ODESystem/DAE (spec §5.3).
# Named MechanismConfig (not the generic "Config") — it travels with a Mechanism.

"""
    MechanismConfig(; energy=:isothermal, constraint=:none, eos=:off,
                    thermo_data=:none, reverse_rate=:irreversible,
                    state_basis=:concentration)

Per-layer configuration that travels with a `Mechanism` into the lowering:

| Layer | Values |
|---|---|
| `energy` | `:isothermal` \\| `:adiabatic` |
| `constraint` | `:none` \\| `:constant_volume` \\| `:constant_pressure` |
| `eos` | `:off` \\| `:ideal_gas` |
| `thermo_data` | `:none` \\| `:nasa7` \\| `:nasa9` |
| `reverse_rate` | `:irreversible` \\| `:explicit` \\| `:thermo_equilibrium` |
| `state_basis` | `:concentration` \\| `:moles` \\| `:mass_fractions` \\| `:mole_fractions` |

The defaults form the `:kinetic` zero-point — a pure-kinetics bare ODE with no
energy, constraint, or EOS layers. Convenience presets: `convenience_config` or
`BatchReactor(; mode=...)`.

# Example
cfg = MechanismConfig(energy=:adiabatic, constraint=:constant_volume,
                      eos=:ideal_gas, thermo_data=:nasa7, reverse_rate=:thermo_equilibrium)
"""
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
