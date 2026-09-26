# Reactor assembly. Phase 1 ships a minimal ChemPhaseSystem: the lowering entry
# point / wrapper around a Mechanism (+ optional Catalyst import). The full
# @mtkmodel reactor family (BatchReactor) arrives in Phase 2 (spec §5.5).

"A ChemPhaseSystem wraps a lowered MTK ODESystem with its source Mechanism and config."
struct ChemPhaseSystem
    sys::Any          # an mtkcompile'd ModelingToolkit.ODESystem
    mech::Mechanism
    config::MechanismConfig
end

"Build a ChemPhaseSystem from a Mechanism (lowers with the given config).
 `checks` is forwarded to `lower_to_mtk` (default true; pass false for large
 mechanisms whose inlined K_c (NASA7) reverse-rate terms trip MTK's unit
 validator — the equations are dimensionally correct, the check just cannot
 fold the long ifelse(T<=Tmid,...) chains). See examples/aramco_ignition.jl."
function ChemPhaseSystem(mech::Mechanism; config::MechanismConfig=MechanismConfig(),
                         checks::Bool=true)
    return ChemPhaseSystem(lower_to_mtk(mech; config=config, checks=checks), mech, config)
end

"Build a ChemPhaseSystem from a Catalyst ReactionSystem (imports, then lowers)."
function ChemPhaseSystem(rn; config::MechanismConfig=MechanismConfig(), checks::Bool=true)
    return ChemPhaseSystem(import_from_catalyst(rn); config=config, checks=checks)
end

# Reactor assembly (spec §5.5). Phase 2 ships a zero-point BatchReactor: a thin
# wrapper around a ChemPhaseSystem that provides the Layer-1 script API and the
# solve/build/extract dispatch surface. On the :kinetic zero-point the reactor
# adds NO constraint equations (spec §5.5: "零点连约束层都不带，就是裸 ODE"), so it
# is a plain struct rather than an @mtkmodel — the @mtkmodel reactor family
# (which composes constraint layers) arrives in Phase 4, once the energy/EOS
# layers it would compose actually exist.

"""A `ChemPhaseSystem` wrapper — the Layer-1 reactor entry point: the
script-level object accepted by `simulate` / `build_problem` / `extract_system`."""
struct BatchReactor
    phase::ChemPhaseSystem
    name::Symbol
end

"""Build a `BatchReactor` from a `Mechanism`.

    BatchReactor(mech; mode=nothing, energy=:isothermal, constraint=:none,
                 eos=:off, thermo_data=:none, reverse_rate=:irreversible,
                 state_basis=:concentration, checks=true, name=:batch)

Keyword args mirror `MechanismConfig` (defaults = the `:kinetic` zero-point);
pass `mode` to select a convenience preset (`:kinetic`, `:fixedT`,
`:adiabatic_constV`, `:adiabatic_constP`) — when given it overrides the
individual layer kwargs. `checks` forwards to `lower_to_mtk` (pass `false` for
very large mechanisms; see the `ChemPhaseSystem` constructor note).

# Example
reactor = BatchReactor(load_mechanism("examples/mechanism/gri30.yaml");
                       mode=:adiabatic_constV)
"""
function BatchReactor(mech::Mechanism;
        mode::Union{Symbol,Nothing}=nothing,
        energy::Symbol=:isothermal,
        constraint::Symbol=:none,
        eos::Symbol=:off,
        thermo_data::Symbol=:none,
        reverse_rate::Symbol=:irreversible,
        state_basis::Symbol=:concentration,
        checks::Bool=true,
        name::Symbol=:batch)
    config = isnothing(mode) ?
        MechanismConfig(energy=energy, constraint=constraint, eos=eos,
                        thermo_data=thermo_data, reverse_rate=reverse_rate,
                        state_basis=state_basis) :
        convenience_config(mode)
    phase = ChemPhaseSystem(mech; config=config, checks=checks)   # lower_to_mtk guards zero-point
    return BatchReactor(phase, name)
end

"Wrap an existing ChemPhaseSystem as a BatchReactor (config is unchanged)."
BatchReactor(phase::ChemPhaseSystem; name::Symbol=:batch) = BatchReactor(phase, name)

"Build a BatchReactor from a Catalyst ReactionSystem (imports, then wraps)."
BatchReactor(rn; kwargs...) = BatchReactor(import_from_catalyst(rn); kwargs...)

"Mechanism-file parsing (YAML/CHEMKIN) is not implemented yet (spec §6 Layer 1)."
BatchReactor(s::AbstractString; kwargs...) =
    error("BatchReactor: mechanism-file parsing (\"$s\") arrives in a later phase; " *
          "pass a Mechanism or a Catalyst ReactionSystem.")

Base.show(io::IO, r::BatchReactor) =
    print(io, "BatchReactor(:$(r.name), energy=$(r.phase.config.energy), " *
              "constraint=$(r.phase.config.constraint))")

# —— Convenience modes (spec §5.3.3) ————————————————————————————————
# Short symbols that expand to a MechanismConfig. Only :kinetic (the zero-point)
# is lowerable in Phase 2; the others document the target API and error helpfully
# (via lower_to_mtk's zero-point guard) until their layers (EOS/energy/NASA) land.

const _CONVENIENCE_MODES = Dict{Symbol,MechanismConfig}(
    :kinetic          => MechanismConfig(),
    :fixedT           => MechanismConfig(energy=:isothermal,  constraint=:constant_volume,
                                         eos=:ideal_gas, thermo_data=:nasa7, reverse_rate=:explicit),
    :adiabatic_constV => MechanismConfig(energy=:adiabatic,   constraint=:constant_volume,
                                         eos=:ideal_gas, thermo_data=:nasa7, reverse_rate=:thermo_equilibrium),
    :adiabatic_constP => MechanismConfig(energy=:adiabatic,   constraint=:constant_pressure,
                                         eos=:ideal_gas, thermo_data=:nasa7, reverse_rate=:thermo_equilibrium),
)

"""Map a convenience-mode symbol to its MechanismConfig.
 Known modes: :kinetic, :fixedT, :adiabatic_constV, :adiabatic_constP.
 When `mode` is passed to BatchReactor it overrides the individual
 energy/constraint/eos/thermo_data/reverse_rate/state_basis kwargs.

# Example
cfg = convenience_config(:adiabatic_constV)   # :kinetic | :fixedT | :adiabatic_constV | :adiabatic_constP
"""
function convenience_config(mode::Symbol)
    haskey(_CONVENIENCE_MODES, mode) ||
        error("convenience_config: unknown mode :$mode; known modes: " *
              "$(sort(collect(keys(_CONVENIENCE_MODES))))")
    return _CONVENIENCE_MODES[mode]
end
