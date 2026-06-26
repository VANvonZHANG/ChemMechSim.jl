# Lowering pipeline: Mechanism + config → MTK ODESystem (spec §5.4).
# Phase 2.5b: unit-aware lowering (§5.6). @species carry [unit=conc], T [unit=K],
# and each reaction's rate constant k is a unit-bearing parameter (stoichiometrically
# derived unit), so MTK's dimension check fires at System construction.
# The Catalyst mass-action backend (catalyst_lowering via oderatelaw) shares the
# same unit-bearing k. Constraint-layer assembly (append_constraint_layers!) is a
# stub (Phase 4).

using DynamicQuantities     # for u"..." unit literals in rate-param construction

# Molar gas constant (spec §5.6.2: R = 8.314 J/(mol·K)).
# Stays here for Task 1; moves to src/data/types.jl in Task 5.
const R_GAS = 8.314

"Attach a DynamicQuantities unit to a symbolic variable/parameter (the @species/@parameters
 macros reject interpolated names with [unit=...], so units are attached via setmetadata)."
_attach_unit(sym, unit) = ModelingToolkit.setmetadata(sym, ModelingToolkit.VariableUnit, unit)

"Build a rate-constant parameter `name` with `default` value and a derived `unit` (§5.6.5).
 The @parameters macro only accepts a LITERAL default with interpolation, so create with a
 placeholder then setdefault + setmetadata."
function rate_param(name::Symbol, default, unit)
    kp = only(@parameters ($(name)) = 1.0)
    kp = ModelingToolkit.setdefault(kp, default)
    return _attach_unit(kp, unit)
end

"Expected unit of a rate constant for overall reaction order `order` (Σ reactant stoich,
 incl. the third-body/[M] factor where applicable) and Arrhenius exponent `b`:
 [k] = conc^(1-order)·s⁻¹ ; the A-factor absorbs T^b -> [A] = [k] / K^b."
_k_unit(order, b) = ChemUnits.conc^(1 - order) * u"s^-1" / (u"K"^b)

"Symbolic rate constant k(T) for an ElementaryArrhenius law, as a unit-bearing parameter.
 `order` = Σ reactant stoichiometry (for unit derivation). Creates A (and θ, T if needed)."
function _arrhenius_k_param(kin::ElementaryArrhenius, order::Real, nameprefix::AbstractString, T)
    b = kin.b
    A = rate_param(Symbol(nameprefix, "_A"), kin.A, _k_unit(order, b))
    if iszero(b) && iszero(kin.Ea)
        return A                                   # constant rate, no T dependence
    end
    # T-dependent: k = A·T^b·exp(-θ/T), θ = Ea/R (K) so the exponent is dimensionless
    θ = rate_param(Symbol(nameprefix, "_theta"), kin.Ea / R_GAS, u"K")
    return A * T^b * exp(-θ / T)
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

"Symbolic NET rate for one reaction (forward minus reverse). Irreducible elementary
 reactions may go via the Catalyst path; everything else (and all reversible reactions)
 use the direct path. `j` is the reaction index (for naming its rate parameters)."
function lower_reaction(rx::ReactionData, mech::Mechanism, cvar, T, config::MechanismConfig, j::Int)
    rx.reverse_policy isa Irreversible ||
        return _net_rate(rx, mech, cvar, T, j)            # ThermoReverse (Task 6)
    return catalyst_native(rx, config) ? catalyst_lowering(rx, mech, cvar, T, j) :
                                         direct_mtk_lowering(rx, mech, cvar, T, j)
end

"Direct-MTK lowering path: build the symbolic rate by dispatching on kinetics type."
direct_mtk_lowering(rx::ReactionData, mech::Mechanism, cvar, T, j::Int) =
    _direct_rate(rx.kinetics, rx, mech, cvar, T, j)

"Elementary Arrhenius forward rate: k(T)·∏ reactants. k is a unit-bearing rate_param."
function _direct_rate(kin::ElementaryArrhenius, rx, mech, cvar, T, j)
    order = sum(values(rx.reactants))
    k = _arrhenius_k_param(kin, order, "k_$j", T)
    return k * _mass_action(rx.reactants, cvar)
end

# Fallback for kinetics types not yet unit-aware (third-body/Troe/etc. arrive in Tasks 2-4).
_direct_rate(kin::AbstractKinetics, rx, mech, cvar, T, j) =
    error("_direct_rate: unit-aware lowering for $(typeof(kin)) arrives in a later task.")

"Catalyst mass-action lowering path (spec §5.4). Builds a Catalyst.Reaction on the
 shared @species with the SAME unit-bearing k, then reads its rate law via oderatelaw."
function catalyst_lowering(rx::ReactionData, mech::Mechanism, cvar, T, j::Int)
    kin = rx.kinetics
    kin isa ElementaryArrhenius ||
        error("catalyst_lowering: only ElementaryArrhenius is Catalyst-native so far.")
    order = sum(values(rx.reactants))
    k = _arrhenius_k_param(kin, order, "k_$j", T)
    subs       = [cvar[sid] for sid in keys(rx.reactants)]
    substoich  = collect(values(rx.reactants))
    prods      = [cvar[sid] for sid in keys(rx.products)]
    prodstoich = collect(values(rx.products))
    crate = Catalyst.Reaction(k, subs, prods, substoich, prodstoich)
    return Catalyst.oderatelaw(crate; combinatoric_ratelaw=false)
end

# ThermoReverse net rate (stub — Task 6 implements the reverse-rate path).
_net_rate(rx, mech, cvar, T, j) =
    error("_net_rate: reversible (ThermoReverse) lowering arrives in Task 6.")

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

"Lower a Mechanism into a structural_simplify'd MTK ODESystem (unit-aware, zero-point).
 Species get [unit=conc]; T [unit=K] is created iff any reaction is T-dependent or
 ThermoReverse. Each reaction's rate constant is a unit-bearing parameter (default = stored
 value), so MTK's dimension check fires at System construction (§5.6)."
function lower_to_mtk(mech::Mechanism; config::MechanismConfig=MechanismConfig())
    _is_zero_point(config) ||
        error("lower_to_mtk: only the :kinetic zero-point config (MechanismConfig()) is supported so far; " *
              "energy/EOS/thermo layers arrive in later phases.")
    t = ModelingToolkit.t
    D = ModelingToolkit.D
    cvars = [_attach_unit(only(@species ($(Symbol(sp.name)))(t)), ChemUnits.conc)
             for sp in mech.species]
    cvar = Dict(mech.species[i].id => cvars[i] for i in eachindex(mech.species))
    Tparam = _needs_T(mech) ? rate_param(:T, 300.0, u"K") : nothing
    rates = [lower_reaction(rx, mech, cvar, Tparam, config, j)
             for (j, rx) in enumerate(mech.reactions)]
    eqs = [D(cvars[i]) ~ _species_rhs(mech.species[i].id, mech, rates)
           for i in eachindex(mech.species)]
    @named raw = System(eqs, t)          # dimension check fires here (ValidationError on mismatch)
    return mtkcompile(raw)
end

# —— Constraint-layer assembly (stub until Phase 4) ——

"Append energy/EOS/reactor constraint layers to the equation set. (stub — Phase 4)"
append_constraint_layers!(eqs, mech, config) =
    error("append_constraint_layers!: not implemented; see the Phase 4 plan.")
