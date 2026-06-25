# Reactor assembly. Phase 1 ships a minimal ChemPhaseSystem: the lowering entry
# point / wrapper around a Mechanism (+ optional Catalyst import). The full
# @mtkmodel reactor family (BatchReactor) arrives in Phase 2 (spec §5.5).

"A ChemPhaseSystem wraps a lowered MTK ODESystem with its source Mechanism and config."
struct ChemPhaseSystem
    sys::Any          # an mtkcompile'd ModelingToolkit.ODESystem
    mech::Mechanism
    config::MechanismConfig
end

"Build a ChemPhaseSystem from a Mechanism (lowers with the given config)."
function ChemPhaseSystem(mech::Mechanism; config::MechanismConfig=MechanismConfig())
    return ChemPhaseSystem(lower_to_mtk(mech; config=config), mech, config)
end

"Build a ChemPhaseSystem from a Catalyst ReactionSystem (imports, then lowers)."
function ChemPhaseSystem(rn; config::MechanismConfig=MechanismConfig())
    return ChemPhaseSystem(import_from_catalyst(rn); config=config)
end

# Reactor assembly (spec §5.5). Phase 2 ships a zero-point BatchReactor: a thin
# wrapper around a ChemPhaseSystem that provides the Layer-1 script API and the
# solve/build/extract dispatch surface. On the :kinetic zero-point the reactor
# adds NO constraint equations (spec §5.5: "零点连约束层都不带，就是裸 ODE"), so it
# is a plain struct rather than an @mtkmodel — the @mtkmodel reactor family
# (which composes constraint layers) arrives in Phase 4, once the energy/EOS
# layers it would compose actually exist (see plan Global Constraints).

"A ChemPhaseSystem wrapper — the Layer-1 reactor entry point (zero-point for now).
 Holds the lowered phase; Phase 4 will extend the reactor concept with
 constraint-layer equations (const T / const V / const P)."
struct BatchReactor
    phase::ChemPhaseSystem
    name::Symbol
end

"Build a BatchReactor from a Mechanism. Keyword args mirror MechanismConfig
 (default = the :kinetic zero-point). Pass `mode` to select a convenience preset
 (spec §5.3.3); when given, `mode` overrides the individual layer kwargs.
 Non-zero-point configs error inside lower_to_mtk until the energy/EOS/thermo
 layers arrive in later phases."
function BatchReactor(mech::Mechanism;
        mode::Union{Symbol,Nothing}=nothing,
        energy::Symbol=:isothermal,
        constraint::Symbol=:none,
        eos::Symbol=:off,
        thermo_data::Symbol=:none,
        reverse_rate::Symbol=:irreversible,
        state_basis::Symbol=:concentration,
        name::Symbol=:batch)
    config = isnothing(mode) ?
        MechanismConfig(energy=energy, constraint=constraint, eos=eos,
                        thermo_data=thermo_data, reverse_rate=reverse_rate,
                        state_basis=state_basis) :
        convenience_config(mode)
    phase = ChemPhaseSystem(mech; config=config)   # lower_to_mtk guards zero-point
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

"Map a convenience-mode symbol to its MechanismConfig (spec §5.3.3).
 Known modes: :kinetic, :fixedT, :adiabatic_constV, :adiabatic_constP.
 When `mode` is passed to BatchReactor it overrides the individual
 energy/constraint/eos/thermo_data/reverse_rate/state_basis kwargs."
function convenience_config(mode::Symbol)
    haskey(_CONVENIENCE_MODES, mode) ||
        error("convenience_config: unknown mode :$mode; known modes: " *
              "$(sort(collect(keys(_CONVENIENCE_MODES))))")
    return _CONVENIENCE_MODES[mode]
end
