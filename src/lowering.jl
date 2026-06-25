# Lowering pipeline: Mechanism + config → MTK ODESystem (spec §5.4).
# Phase 1 (MVP): DIRECT-MTK path only; unit-free lowering; :kinetic zero-point only.
# The Catalyst-backend lowering path (catalyst_lowering) and constraint-layer
# assembly (append_constraint_layers!) remain stubs (Phase 2.5 spike / Phase 4).

"Rate constant for an elementary Arrhenius law. Phase 1 supports only constant
 rates (b = Ea = 0 → the bare A-factor, e.g. Brusselator); the temperature-dependent
 form k(T) = A·T^b·exp(-Ea/RT) needs a temperature model and arrives in Phase 3.
 (Phase 1's zero-point has no T state/parameter, so a T-dependent rate cannot be
 lowered correctly here — guard it rather than silently misuse the time variable.)"
function _arrhenius_k(kin::ElementaryArrhenius)
    iszero(kin.b) && iszero(kin.Ea) ||
        error("_arrhenius_k: temperature-dependent Arrhenius (b=$(kin.b), Ea=$(kin.Ea)) " *
              "is not supported in Phase 1 (no temperature model yet); see the Phase 3 plan.")
    return kin.A
end

"Mass-action product ∏ c[sid]^ν over a stoichiometry map."
function _mass_action(stoich::Dict{SpeciesID,Float64}, cvar)
    ma = 1.0
    for (sid, nu) in stoich
        ma = ma * cvar[sid]^nu
    end
    return ma
end

"Whether a reaction could use the Catalyst mass-action lowering backend.
 Phase 1 always lowers directly, so this returns false; the real predicate and the
 Catalyst lowering path arrive in the Phase 2.5 spike."
catalyst_native(rx::ReactionData, config::MechanismConfig) = false

"Symbolic rate expression for one reaction (dispatches on catalyst_native)."
function lower_reaction(rx::ReactionData, mech::Mechanism, cvar, config::MechanismConfig)
    return catalyst_native(rx, config) ? catalyst_lowering(rx, mech, cvar) :
                                         direct_mtk_lowering(rx, mech, cvar)
end

"Direct-MTK lowering path: build k·∏c^ν symbolically (spec §5.4)."
function direct_mtk_lowering(rx::ReactionData, mech::Mechanism, cvar)
    k = _arrhenius_k(rx.kinetics)
    return k * _mass_action(rx.reactants, cvar)
end

"net rate of change Σⱼ netstoichⱼᵢ·rateⱼ for the species with id `sid`."
function _species_rhs(sid::SpeciesID, mech::Mechanism, rates)
    rhs = 0.0
    for (j, rx) in enumerate(mech.reactions)
        net = get(rx.products, sid, 0.0) - get(rx.reactants, sid, 0.0)
        iszero(net) || (rhs += net * rates[j])
    end
    return rhs
end

"True iff `config` is the :kinetic zero-point (the only config Phase 1 lowers)."
function _is_zero_point(c::MechanismConfig)
    return c.energy === :isothermal && c.constraint === :none && c.eos === :off &&
           c.thermo_data === :none && c.reverse_rate === :irreversible &&
           c.state_basis === :concentration
end

"Lower a Mechanism into a structural_simplify'd MTK ODESystem (zero-point)."
function lower_to_mtk(mech::Mechanism; config::MechanismConfig=MechanismConfig())
    _is_zero_point(config) ||
        error("lower_to_mtk: Phase 1 supports only the :kinetic zero-point config (MechanismConfig()). " *
              "Energy/EOS/thermo/reverse layers arrive in later phases.")
    t = ModelingToolkit.t_nounits
    D = ModelingToolkit.D_nounits
    cvars = [only(@variables ($(Symbol(sp.name)))(t)) for sp in mech.species]
    cvar = Dict(mech.species[i].id => cvars[i] for i in eachindex(mech.species))
    rates = [lower_reaction(rx, mech, cvar, config) for rx in mech.reactions]
    eqs = [D(cvars[i]) ~ _species_rhs(mech.species[i].id, mech, rates)
           for i in eachindex(mech.species)]
    @named raw = System(eqs, t)
    return mtkcompile(raw)
end

# —— Stubs for paths that arrive after Phase 1 ——

"Catalyst mass-action lowering path (spec §5.4). (stub — Phase 2.5 spike)"
catalyst_lowering(rx, mech, cvar) =
    error("catalyst_lowering: not implemented in Phase 1; see the Phase 2.5 spike plan.")

"Append energy/EOS/reactor constraint layers to the equation set. (stub — Phase 4)"
append_constraint_layers!(eqs, mech, config) =
    error("append_constraint_layers!: not implemented in Phase 1; see the Phase 4 plan.")
