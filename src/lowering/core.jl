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
 (V is defined by the EOS). Phase 4b adds const-P (path A, pure ODE)."
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
    _PSTD_PARAM[] = nothing        # reset lowering-scoped thermo-constant singletons (Task 4)
    _RGAS_PARAM[] = nothing
    _COEFF_CACHE[] = Dict{Int,Any}()   # per-species NASA7 coeff cache (Phase 4a)
    _lowerable(config) ||
        error("lower_to_mtk: config not supported in Phase 4a. Supported: energy∈{:isothermal,:adiabatic}, " *
              "state_basis=:concentration, eos∈{:off,:ideal_gas}; constraint must be :constant_volume for " *
              ":adiabatic; non-concentration bases arrive in Phase 4b+. Got: " *
              "energy=$(config.energy) constraint=$(config.constraint) eos=$(config.eos) " *
              "basis=$(config.state_basis). Use MechanismConfig() (:kinetic), convenience_config(:fixedT), " *
              "or convenience_config(:adiabatic_constV).")
    config.constraint === :constant_pressure && return _lower_constP(mech, config)
    t = ModelingToolkit.t
    D = ModelingToolkit.D
    cvars = [_attach_unit(only(@species ($(Symbol(sp.name)))(t)), ChemUnits.conc)
             for sp in mech.species]
    cvar = Dict(mech.species[i].id => cvars[i] for i in eachindex(mech.species))
    is_adiabatic = config.energy === :adiabatic
    # T exists iff any reaction is T-dependent, EOS needs it, OR the energy layer makes it a state.
    # Under :adiabatic T is a STATE (@variables); otherwise a unit-bearing parameter.
    needs_T = _needs_T(mech) || config.eos === :ideal_gas || is_adiabatic
    Tparam = if !needs_T
        nothing
    elseif is_adiabatic
        _attach_unit(only(@variables T(t)), u"K")     # temperature STATE
    else
        rate_param(:T, 300.0, u"K")                    # temperature parameter (isothermal)
    end
    rates = [lower_reaction(rx, mech, cvar, Tparam, config, j)
             for (j, rx) in enumerate(mech.reactions)]
    eqs = [D(cvars[i]) ~ _species_rhs(mech.species[i].id, mech, rates)
           for i in eachindex(mech.species)]
    # Constraint-layer assembly (energy layer adds the const-V dT/dt under :adiabatic; spec §5.4).
    eqs = append_constraint_layers!(eqs, mech, config, cvar, Tparam, rates)
    if config.eos === :ideal_gas
        return _lower_with_eos(eqs, t, cvars, Tparam, is_adiabatic)
    end
    @named raw = System(eqs, t)          # auto-discovers states [c₁..cₙ, T] and RHS params
    return mtkcompile(raw)
end

"Build the system with EOS observed P ~ (Σc)·R·T. Under :adiabatic T is a STATE (included in
 `states`); under :isothermal T is a parameter retained via the observed-param fix (Phase 3).
 R appears in the energy-equation RHS under :adiabatic so it is retained automatically."
function _lower_with_eos(eqs, t, cvars, Tparam, is_adiabatic::Bool)
    Rparam = _r_param()                            # shared with K_c / energy eq (memoized singleton)
    Pvar = _attach_unit(only(@variables P(t)), ChemUnits.press)
    obs = [Pvar ~ sum(cvars) * Rparam * Tparam]
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
    Rparam = _r_param()
    Vvar   = _attach_unit(only(@variables V(t)), ChemUnits.vol)
    rates  = [lower_reaction(rx, mech, cvar, Tsym, config, j) for (j, rx) in enumerate(mech.reactions)]
    eqs = Equation[D(nvars[i]) ~ Vvar * _species_rhs(mech.species[i].id, mech, rates)
              for i in eachindex(mech.species)]
    push!(eqs, Vvar ~ sum(nvars) * Rparam * Tsym / Pparam)            # EOS → observed V
    for i in eachindex(mech.species)
        push!(eqs, cvars[i] ~ nvars[i] / Vvar)                        # observed concentrations (rate-law input)
    end
    eqs = append_constraint_layers!(eqs, mech, config, cvar, Tsym, rates; nvar=nvar, Vvar=Vvar)
    @named raw = System(eqs, t)
    return mtkcompile(raw)
end

# —— Constraint-layer assembly (Phase 4a: energy layer; EOS-as-DAE + const-P arrive in 4b) ——

"Append energy/reactor constraint layers to the equation set (spec §5.4). Phase 4a: the energy
 layer (:adiabatic) adds the const-V energy ODE for T. `cvar`/`T`/`rates` are the shared species
 vars, the temperature symbol, and the per-reaction symbolic net rates from lower_to_mtk."
function append_constraint_layers!(eqs, mech, config, cvar, T, rates; nvar=nothing, Vvar=nothing)
    config.energy === :adiabatic || return eqs
    if config.constraint === :constant_pressure
        push!(eqs, _energy_ode_constP(mech, nvar, Vvar, T, rates))    # Task 2
    else
        push!(eqs, _energy_ode_constV(mech, cvar, T, rates))
    end
    return eqs
end

"Constant-pressure adiabatic energy equation (spec §5.3/§11 Phase 4; probed 2026-07-03 P2):
 dT/dt = -V·Σⱼ rⱼ·Δh̄ⱼ / Σᵢ nᵢ·cpᵢ, with Δh̄ⱼ = Σ_prod ν·h̄ − Σ_react ν·h̄ and cpᵢ = (cp/R)·R (ideal gas).
 All species must carry NASA7 thermo (spec §5.3.4 — clear error otherwise). H = Σnᵢh̄ᵢ(T) is conserved."
function _energy_ode_constP(mech::Mechanism, nvar, Vvar, T, rates)
    D = ModelingToolkit.D
    R = _r_param()
    for sp in mech.species
        sp.thermo isa NASA7 ||
            error("_energy_ode_constP: species $(sp.name) (id $(sp.id)) has no NASA7 thermo; " *
                  ":adiabatic requires NASA7 thermo on all species (spec §5.3.4). " *
                  "Use energy=:isothermal or provide NASA7 thermo.")
    end
    cp_sum = sum(nvar[sp.id] * _cp_over_R(sp.thermo, T, sp.id) * R for sp in mech.species)   # [J/K]
    src = 0.0
    for (j, rx) in enumerate(mech.reactions)
        delta_h = 0.0
        for (sid, nu) in rx.products
            delta_h += nu * _h_over_RT(_species_by_id(mech, sid).thermo, T, sid) * R * T
        end
        for (sid, nu) in rx.reactants
            delta_h -= nu * _h_over_RT(_species_by_id(mech, sid).thermo, T, sid) * R * T
        end
        src += rates[j] * (-delta_h)                                  # Σⱼ rⱼ·(-Δh̄ⱼ)  [J/(m³·s)]
    end
    return D(T) ~ Vvar * src / cp_sum
end

"Constant-volume adiabatic energy equation (spec §5.3, §11 Phase 4; verified 2026-07-02):
 dT/dt = -Σⱼ rⱼ·Δūⱼ / Σᵢ cᵢ·cvᵢ, with ūᵢ=(h/RT-1)·R·T and cvᵢ=(cp/R-1)·R (ideal gas).
 All species must carry NASA7 thermo (spec §5.3.4 — clear error otherwise)."
function _energy_ode_constV(mech::Mechanism, cvar, T, rates)
    D = ModelingToolkit.D
    R = _r_param()
    for sp in mech.species
        sp.thermo isa NASA7 ||
            error("_energy_ode_constV: species $(sp.name) (id $(sp.id)) has no NASA7 thermo; " *
                  ":adiabatic requires NASA7 thermo on all species (spec §5.3.4). " *
                  "Use energy=:isothermal or provide NASA7 thermo.")
    end
    # Σᵢ cᵢ·cvᵢ  [J/(m³·K)]
    cv_sum = sum(cvar[sp.id] * (_cp_over_R(sp.thermo, T, sp.id) - 1) * R for sp in mech.species)
    # -Σⱼ rⱼ·Δūⱼ  [J/(m³·s)],  Δūⱼ = Σ_products ν·ū − Σ_reactants ν·ū
    src = 0.0
    for (j, rx) in enumerate(mech.reactions)
        delta_u = 0.0
        for (sid, nu) in rx.products
            th = _species_by_id(mech, sid).thermo
            delta_u += nu * (_h_over_RT(th, T, sid) - 1) * R * T
        end
        for (sid, nu) in rx.reactants
            th = _species_by_id(mech, sid).thermo
            delta_u -= nu * (_h_over_RT(th, T, sid) - 1) * R * T
        end
        src += rates[j] * (-delta_u)
    end
    return D(T) ~ src / cv_sum
end
