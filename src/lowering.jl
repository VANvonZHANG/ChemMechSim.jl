# Lowering pipeline: Mechanism + config → MTK ODESystem (spec §5.4).
# Phase 2.5a: @species variables + T-dependent Arrhenius k(T) + the Catalyst
# mass-action backend (catalyst_lowering via oderatelaw). Constraint-layer
# assembly (append_constraint_layers!) remains a stub (Phase 4).

# Molar gas constant (spec §5.6.2: R = 8.314 J/(mol·K)).
const R_GAS = 8.314

"Arrhenius rate constant k(T) = A·T^b·exp(-Ea/RT). Collapses to the bare
 A-factor when b = Ea = 0 (no temperature model needed). `T` is the symbolic
 temperature parameter, or `nothing` when no reaction is T-dependent; a
 T-dependent law with T === nothing is a bug and is rejected."
function _arrhenius_k(kin::ElementaryArrhenius, T)
    A, b, Ea = kin.A, kin.b, kin.Ea
    (iszero(b) && iszero(Ea)) && return A
    T === nothing &&
        error("_arrhenius_k: T-dependent Arrhenius (b=$b, Ea=$Ea) needs a T parameter, " *
              "but none was introduced. This should not happen — lower_to_mtk creates T " *
              "iff a reaction is T-dependent.")
    return A * T^b * exp(-Ea / (R_GAS * T))
end

"True iff rate law `kin` needs a temperature symbol (T-dependent Arrhenius).
 Other kinetics types arrive in Phase 2.5b and declare their own needs there."
_is_T_dependent(kin::ElementaryArrhenius) = !(iszero(kin.b) && iszero(kin.Ea))
_is_T_dependent(kin::AbstractKinetics) = false

"True iff any reaction in `mech` needs a T parameter."
_needs_T(mech::Mechanism) = any(_is_T_dependent(rx.kinetics) for rx in mech.reactions)

"Mass-action product ∏ c[sid]^ν over a stoichiometry map."
function _mass_action(stoich::Dict{SpeciesID,Float64}, cvar)
    ma = 1.0
    for (sid, nu) in stoich
        ma = ma * cvar[sid]^nu
    end
    return ma
end

"Whether a reaction lowers via the Catalyst mass-action backend (spec §3.3/§5.4).
 True for plain elementary Arrhenius (mass-action); false for rate types Catalyst
 does not represent natively (third-body/falloff/PLOG/Chebyshev arrive in 2.5b)."
catalyst_native(rx::ReactionData, config::MechanismConfig) =
    rx.kinetics isa ElementaryArrhenius

"Symbolic rate expression for one reaction (dispatches on catalyst_native)."
function lower_reaction(rx::ReactionData, mech::Mechanism, cvar, T, config::MechanismConfig)
    return catalyst_native(rx, config) ? catalyst_lowering(rx, mech, cvar, T) :
                                         direct_mtk_lowering(rx, mech, cvar, T)
end

"Direct-MTK lowering path: build k(T)·∏c^ν symbolically (spec §5.4)."
function direct_mtk_lowering(rx::ReactionData, mech::Mechanism, cvar, T)
    k = _arrhenius_k(rx.kinetics, T)
    return k * _mass_action(rx.reactants, cvar)
end

"Catalyst mass-action lowering path (spec §5.4). Builds a Catalyst.Reaction from
 the ReactionData (referencing the shared @species), then reads its symbolic rate
 law via oderatelaw. combinatoric_ratelaw=false gives deterministic k·∏c^ν with NO
 factorial scaling (required for combustion, spec §3.1). Verified 2026-06-26 to
 handle ∅→X, X→∅, species-on-both-sides, Float64 stoich, and symbolic k(T) rates."
function catalyst_lowering(rx::ReactionData, mech::Mechanism, cvar, T)
    kin = rx.kinetics
    kin isa ElementaryArrhenius ||
        error("catalyst_lowering: only ElementaryArrhenius is Catalyst-native so far; " *
              "other rate laws arrive in Phase 2.5b.")
    subs       = [cvar[sid] for sid in keys(rx.reactants)]
    substoich  = collect(values(rx.reactants))
    prods      = [cvar[sid] for sid in keys(rx.products)]
    prodstoich = collect(values(rx.products))
    # 5 positional args: stoichiometry is positional (keywords substoich=/prodstoich= are ignored).
    crate = Catalyst.Reaction(_arrhenius_k(kin, T), subs, prods, substoich, prodstoich)
    return Catalyst.oderatelaw(crate; combinatoric_ratelaw=false)
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

"True iff `config` is the :kinetic zero-point (the only config lowered so far)."
function _is_zero_point(c::MechanismConfig)
    return c.energy === :isothermal && c.constraint === :none && c.eos === :off &&
           c.thermo_data === :none && c.reverse_rate === :irreversible &&
           c.state_basis === :concentration
end

"Lower a Mechanism into a structural_simplify'd MTK ODESystem (zero-point).
 Creates a T parameter (default 300 K) iff any reaction is T-dependent."
function lower_to_mtk(mech::Mechanism; config::MechanismConfig=MechanismConfig())
    _is_zero_point(config) ||
        error("lower_to_mtk: only the :kinetic zero-point config (MechanismConfig()) is supported so far; " *
              "energy/EOS/thermo/reverse layers arrive in later phases.")
    t = ModelingToolkit.t_nounits
    D = ModelingToolkit.D_nounits
    # @species (Catalyst) — not plain @variables — so the lowered unknowns are
    # Catalyst-recognized and can feed Catalyst.Reaction/oderatelaw in the backend
    # lowering path (Phase 2.5a Task 3). @species vars are ordinary MTK unknowns.
    cvars = [only(@species ($(Symbol(sp.name)))(t)) for sp in mech.species]
    cvar = Dict(mech.species[i].id => cvars[i] for i in eachindex(mech.species))
    Tparam = _needs_T(mech) ? only(@parameters ($(Symbol("T"))) = 300.0) : nothing
    rates = [lower_reaction(rx, mech, cvar, Tparam, config) for rx in mech.reactions]
    eqs = [D(cvars[i]) ~ _species_rhs(mech.species[i].id, mech, rates)
           for i in eachindex(mech.species)]
    @named raw = System(eqs, t)
    return mtkcompile(raw)
end

# —— Constraint-layer assembly (stub until Phase 4) ——

"Append energy/EOS/reactor constraint layers to the equation set. (stub — Phase 4)"
append_constraint_layers!(eqs, mech, config) =
    error("append_constraint_layers!: not implemented; see the Phase 4 plan.")
