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
function lower_to_mtk(mech::Mechanism; config::MechanismConfig=MechanismConfig())
    _lowerable(config) ||
        error("lower_to_mtk: config not supported. Supported: state_basis=:concentration, " *
              "energy∈{:isothermal,:adiabatic}, eos∈{:off,:ideal_gas}; :constant_pressure " *
              "requires eos=:ideal_gas; :adiabatic requires constraint∈{:constant_volume,:constant_pressure}. " *
              "Got: energy=$(config.energy) constraint=$(config.constraint) eos=$(config.eos) " *
              "basis=$(config.state_basis). Use MechanismConfig() (:kinetic), convenience_config(:fixedT), " *
              "convenience_config(:adiabatic_constV), or convenience_config(:adiabatic_constP).")
    config.constraint === :constant_pressure && return _lower_constP(mech, config)
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
    rates = [lower_reaction(rx, mech, cvar, Tparam, config, j,
                RateCtx(mech, cvar, Tparam, j,
                        sum(values(rx.reactants)), tcx.R, tcx.P_std, tcx.coeff_cache, Pvar))
             for (j, rx) in enumerate(mech.reactions)]
    eqs = [D(cvars[i]) ~ _species_rhs(mech.species[i].id, mech, rates)
           for i in eachindex(mech.species)]
    # Constraint-layer assembly (energy layer adds the const-V dT/dt under :adiabatic; spec §5.4).
    eqs = append_constraint_layers!(eqs, mech, config, cvar, Tparam, rates; tcx=tcx)
    if config.eos === :ideal_gas
        return _lower_with_eos(eqs, t, cvars, Tparam, is_adiabatic, tcx, Pvar)
    end
    @named raw = System(eqs, t)          # auto-discovers states [c₁..cₙ, T] and RHS params
    return mtkcompile(raw)
end

"Build the system with EOS observed P ~ (Σc)·R·T. Under :adiabatic T is a STATE (included in
 `states`); under :isothermal T is a parameter retained via the observed-param fix (Phase 3).
 R appears in the energy-equation RHS under :adiabatic so it is retained automatically."
function _lower_with_eos(eqs, t, cvars, Tparam, is_adiabatic::Bool, tcx, Pvar)
    Rparam = tcx.R                                   # shared with K_c / energy eq (from tcx)
    obs = [Pvar ~ sum(cvars) * Rparam * Tparam]      # Pvar created by caller (lower_to_mtk)
    states = is_adiabatic ? [cvars; Tparam] : cvars   # T is a state under :adiabatic
    @named _tmp = System(eqs, t)                   # auto-discover RHS params
    rhsparams = ModelingToolkit.parameters(_tmp)
    rhsnames = Set(ModelingToolkit.getname(p) for p in rhsparams)
    extras = Any[]
    ModelingToolkit.getname(Rparam) in rhsnames || push!(extras, Rparam)
    is_adiabatic || (ModelingToolkit.getname(Tparam) in rhsnames || push!(extras, Tparam))  # T param only when isothermal
    @named raw = System(eqs, t, states, [rhsparams; extras]; observed=obs)
    return mtkcompile(raw)
end

"Lower under :constant_pressure — path A, PURE ODE (probed 2026-07-03 P1/P2).
 Moles `nᵢ` are the states; `V ~ (Σn)RT/P` and concentrations `cᵢ ~ nᵢ/V` are observed (structural_simplify
 flattens them). The rate laws consume `cvar` (concentrations) UNCHANGED — only the outer assembly differs
 from the concentration-state path. :adiabatic adds the enthalpy energy equation via append_constraint_layers!."
function _lower_constP(mech::Mechanism, config::MechanismConfig)
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
    rates  = [lower_reaction(rx, mech, cvar, Tsym, config, j,
                 RateCtx(mech, cvar, Tsym, j,
                         sum(values(rx.reactants)), tcx.R, tcx.P_std, tcx.coeff_cache, Pparam))
              for (j, rx) in enumerate(mech.reactions)]
    eqs = Equation[D(nvars[i]) ~ Vvar * _species_rhs(mech.species[i].id, mech, rates)
              for i in eachindex(mech.species)]
    push!(eqs, Vvar ~ sum(nvars) * Rparam * Tsym / Pparam)            # EOS → observed V
    for i in eachindex(mech.species)
        push!(eqs, cvars[i] ~ nvars[i] / Vvar)                        # observed concentrations (rate-law input)
    end
    eqs = append_constraint_layers!(eqs, mech, config, cvar, Tsym, rates; tcx=tcx, nvar=nvar, Vvar=Vvar)
    @named raw = System(eqs, t)
    return mtkcompile(raw)
end
