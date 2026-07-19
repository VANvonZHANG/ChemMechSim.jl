# Lowering core: assembly entry points (lower_to_mtk, const-V/const-P lowering,
# _lower_with_eos, _lowerable config guard, _species_rhs) — spec §5.4.
#
# Task 1 of the lowering-modularization refactor seeds this file with the entire
# former src/lowering.jl body. Tasks 2–6 move cohesive chunks out to sibling files
# (units/state/kinetics/thermo/energy); what remains here at the end is the
# assembly core listed above.

"net rate of change Σⱼ netstoichⱼᵢ·rateⱼ for the species with id `sid`."
function _species_rhs(sid::SpeciesID, mech::Mechanism, rates)
    rhs = 0.0
    for (j, rx) in enumerate(mech.reactions)
        net = get(rx.products, sid, 0.0) - get(rx.reactants, sid, 0.0)
        iszero(net) || (rhs += net * rates[j])
    end
    return rhs
end

"True iff `config` is lowerable: concentration basis; energy ∈ {:isothermal,:adiabatic};
 constraint ∈ {:none,:constant_volume,:constant_pressure}; :adiabatic allows const-V or const-P,
 :isothermal allows any constraint; eos ∈ {:off,:ideal_gas}; :constant_pressure REQUIRES :ideal_gas
 (V is defined by the EOS). Const-P lowering stays a pure ODE (moles basis; V and c observed)."
_lowerable(c::MechanismConfig) =
    c.state_basis === :concentration &&
    c.energy in (:isothermal, :adiabatic) &&
    (c.energy === :adiabatic ? c.constraint in (:constant_volume, :constant_pressure)
                             : c.constraint in (:none, :constant_volume, :constant_pressure)) &&
    (c.constraint === :constant_pressure ? c.eos === :ideal_gas : c.eos in (:off, :ideal_gas))

"Lower a Mechanism into a structural_simplify'd MTK ODESystem (unit-aware, zero-point).
 Species get [unit=conc]; T [unit=K] is created iff any reaction is T-dependent or
 ThermoReverse. Each reaction's rate constant is a unit-bearing parameter (default = stored
 value), so MTK's dimension check fires at System construction (§5.6)."
function lower_to_mtk(mech::Mechanism; config::MechanismConfig=MechanismConfig(), checks::Bool=true)
    _lowerable(config) ||
        error("lower_to_mtk: config not supported. Supported: state_basis=:concentration, " *
              "energy∈{:isothermal,:adiabatic}, eos∈{:off,:ideal_gas}; :constant_pressure " *
              "requires eos=:ideal_gas; :adiabatic requires constraint∈{:constant_volume,:constant_pressure}. " *
              "Got: energy=$(config.energy) constraint=$(config.constraint) eos=$(config.eos) " *
              "basis=$(config.state_basis). Use MechanismConfig() (:kinetic), convenience_config(:fixedT), " *
              "convenience_config(:adiabatic_constV), or convenience_config(:adiabatic_constP).")
    config.constraint === :constant_pressure && return _lower_constP(mech, config, checks)
    t = ModelingToolkit.t
    D = ModelingToolkit.D
    cvars = [_attach_unit(only(@species ($(Symbol(sp.name)))(t)), ChemUnits.conc)
             for sp in mech.species]
    cvar = Dict(mech.species[i].id => cvars[i] for i in eachindex(mech.species))
    is_adiabatic = config.energy === :adiabatic
    # T exists iff any reaction is T-dependent, EOS needs it, OR the energy layer makes it a state.
    # Under :adiabatic T is a STATE (@variables); otherwise a unit-bearing parameter.
    create_T = _needs_T(mech) || config.eos === :ideal_gas || is_adiabatic
    Tparam = if !create_T
        nothing
    elseif is_adiabatic
        _attach_unit(only(@variables T(t)), u"K")     # temperature STATE
    else
        rate_param(:T, 300.0, u"K")                    # temperature parameter (isothermal)
    end
    tcx = make_thermo_ctx(Tparam)
    # P symbol exists iff eos=:ideal_gas (observed P under const-V; Pparam under const-P).
    # Created here (not inside _lower_with_eos) so P-needing rate laws (PLOG) can reach it via ctx.
    Pvar = config.eos === :ideal_gas ? _attach_unit(only(@variables P(t)), ChemUnits.press) : nothing
    # Early validation: pressure-dependent reactions require an EOS-provided P.
    _needs_P(mech) && Pvar === nothing &&
        error("lower_to_mtk: mechanism has pressure-dependent (e.g. PLOG) reactions but the config " *
              "provides no pressure. Use a config with eos=:ideal_gas: convenience_config(:fixedT), " *
              ":adiabatic_constV, or :adiabatic_constP. Got eos=$(config.eos).")
    meff_eqs = Any[]                     # M_eff algebraic eqs (shared across per-reaction RateCtx)
    rates = [lower_reaction(rx, mech, cvar, Tparam, config, j,
                RateCtx(mech, cvar, Tparam, j,
                        sum(values(rx.reactants)), tcx.R, tcx.P_std, tcx.coeff_cache, Pvar, meff_eqs))
             for (j, rx) in enumerate(mech.reactions)]
    eqs = [D(cvars[i]) ~ _species_rhs(mech.species[i].id, mech, rates)
           for i in eachindex(mech.species)]
    # Constraint-layer assembly (energy layer adds the const-V dT/dt under :adiabatic; spec §5.4).
    eqs = append_constraint_layers!(eqs, mech, config, cvar, Tparam, rates; tcx=tcx)
    # P differential state (spec §5.3 iii; Task 4 + Task 3): under :constant_volume with
    # an EOS-provided P, the EOS total derivative D(P) ~ R·(T·Σ物种RHS + (Σc)·能量RHS) is
    # pushed so P becomes a state rather than an observed variable. This applies to BOTH
    # energy regimes: :adiabatic_constV (full coupling — the energy RHS is the dT/dt ODE)
    # and :fixedT / isothermal const-V (energy RHS = 0 since T is a parameter with dT/dt=0,
    # so the ODE collapses to D(P) ~ R·T·Σ物种RHS = d/dt[(Σc)·R·T] at fixed T). const-P is
    # unaffected (early-returned above); constraint=:none+eos keeps P observed/algebraic.
    p_differential = config.constraint === :constant_volume && Pvar !== nothing
    p_differential &&
        push!(eqs, _p_ode_constV(mech, cvar, Tparam, rates, tcx, Pvar, is_adiabatic))
    # M_eff algebraic eqs: MTK tearing eliminates M_eff_j → observed (state+algebraic, §7.1).
    if config.eos === :ideal_gas
        return _lower_with_eos(eqs, t, cvars, Tparam, is_adiabatic, tcx, Pvar, _needs_P(mech),
                               meff_eqs, checks; p_differential=p_differential)
    end
    append!(eqs, meff_eqs)               # non-EOS: auto-discover path handles M_eff_j as states
    @named raw = System(eqs, t)          # auto-discovers states [c₁..cₙ, T] and RHS params
    return mtkcompile(raw; checks=checks)
end

"Build the system with EOS P. Two regimes (spec §5.3 iii; Task 4 + Task 3):
   • const-V (p_differential=true; applies to BOTH :adiabatic_constV AND :fixedT): P is a
     DIFFERENTIAL state — its ODE was already pushed into `eqs` by `lower_to_mtk`. Under
     :adiabatic the ODE is D(P) ~ R·(T·Σ物种RHS + (Σc)·能量RHS); under :fixedT (isothermal)
     it collapses to D(P) ~ R·T·Σ物种RHS (energy RHS = 0). We add Pvar to `states` WITHOUT
     an algebraic/observed P eq (the ODE defines P).
   • otherwise (constraint=:none+eos, isothermal const-V was promoted to p_differential
     above): P stays OBSERVED (P ~ (Σc)·R·T) or algebraic (when `needs_P_flag` — PLOG under
     constraint=:none+eos) exactly as before.
 Under :adiabatic T is a STATE; under :isothermal T is a parameter (retained via the
 observed-param fix). R is retained automatically (appears in the energy / P ODE RHS).
 M_eff_j (third-body) algebraic eqs follow the same state+algebraic pattern: M_eff_j is
 added to states with its algebraic eq in eqs; MTK tearing eliminates M_eff_j → observed."
function _lower_with_eos(eqs, t, cvars, Tparam, is_adiabatic::Bool, tcx, Pvar,
                         needs_P_flag::Bool, meff_eqs=Any[], checks::Bool=true; p_differential::Bool=false)
    Rparam = tcx.R                                   # shared with K_c / energy eq (from tcx)
    states = is_adiabatic ? [cvars; Tparam] : cvars   # T is a state under :adiabatic
    obs = Equation[]
    if p_differential
        # P is a differential state: its ODE was pushed by lower_to_mtk. Just register Pvar
        # as a state — NO algebraic/observed P equation (the ODE defines P).
        push!(states, Pvar)
    else
        P_eq = Pvar ~ sum(cvars) * Rparam * Tparam   # Pvar created by caller (lower_to_mtk)
        if needs_P_flag
            # P-dependent kinetics (PLOG under isothermal const-V): P appears in the rate RHS,
            # so it must be a state with an algebraic eq. MTK tearing eliminates P → observed.
            push!(eqs, P_eq)
            push!(states, Pvar)
        else
            # No P-dependent kinetics: P is purely observed (never referenced in RHS eqs).
            push!(obs, P_eq)
        end
    end
    # M_eff_j algebraic eqs (state+algebraic): add to eqs + states; tearing eliminates → observed.
    for eq in meff_eqs
        push!(eqs, eq)
        push!(states, eq.lhs)
    end
    @named _tmp = System(eqs, t)                   # auto-discover RHS params
    rhsparams = ModelingToolkit.parameters(_tmp)
    rhsnames = Set(ModelingToolkit.getname(p) for p in rhsparams)
    extras = Any[]
    ModelingToolkit.getname(Rparam) in rhsnames || push!(extras, Rparam)
    is_adiabatic || (ModelingToolkit.getname(Tparam) in rhsnames || push!(extras, Tparam))  # T param only when isothermal
    @named raw = System(eqs, t, states, [rhsparams; extras]; observed=obs)
    return mtkcompile(raw; checks=checks)
end

"Lower under :constant_pressure — path A, PURE ODE (probed 2026-07-03 P1/P2).
 Moles `nᵢ` are the states; `V ~ (Σn)RT/P` and concentrations `cᵢ ~ nᵢ/V` are observed (structural_simplify
 flattens them). The rate laws consume `cvar` (concentrations) UNCHANGED — only the outer assembly differs
 from the concentration-state path. :adiabatic adds the enthalpy energy equation via append_constraint_layers!."
function _lower_constP(mech::Mechanism, config::MechanismConfig, checks::Bool=true)
    t = ModelingToolkit.t; D = ModelingToolkit.D
    is_adiabatic = config.energy === :adiabatic
    cvars = [_attach_unit(only(@species ($(Symbol(sp.name)))(t)), ChemUnits.conc) for sp in mech.species]
    cvar  = Dict(mech.species[i].id => cvars[i] for i in eachindex(mech.species))
    nvars = [_attach_unit(only(@species ($(Symbol("n_", sp.name)))(t)), ChemUnits.mol) for sp in mech.species]
    nvar  = Dict(mech.species[i].id => nvars[i] for i in eachindex(mech.species))
    Pparam = rate_param(:P, P_STD, ChemUnits.press)
    Tsym   = is_adiabatic ? _attach_unit(only(@variables T(t)), u"K") : rate_param(:T, 300.0, u"K")
    tcx = make_thermo_ctx(Tsym)
    Rparam = tcx.R
    Vvar   = _attach_unit(only(@variables V(t)), ChemUnits.vol)
    meff_eqs = Any[]                     # M_eff algebraic eqs (shared across per-reaction RateCtx)
    rates  = [lower_reaction(rx, mech, cvar, Tsym, config, j,
                 RateCtx(mech, cvar, Tsym, j,
                         sum(values(rx.reactants)), tcx.R, tcx.P_std, tcx.coeff_cache, Pparam, meff_eqs))
              for (j, rx) in enumerate(mech.reactions)]
    eqs = Equation[D(nvars[i]) ~ Vvar * _species_rhs(mech.species[i].id, mech, rates)
              for i in eachindex(mech.species)]
    push!(eqs, Vvar ~ sum(nvars) * Rparam * Tsym / Pparam)            # EOS → observed V
    for i in eachindex(mech.species)
        push!(eqs, cvars[i] ~ nvars[i] / Vvar)                        # observed concentrations (rate-law input)
    end
    eqs = append_constraint_layers!(eqs, mech, config, cvar, Tsym, rates; tcx=tcx, nvar=nvar, Vvar=Vvar)
    append!(eqs, meff_eqs)               # M_eff algebraic eqs → MTK tearing eliminates to observed
    @named raw = System(eqs, t)
    return mtkcompile(raw; checks=checks)
end
