# Constraint-layer assembly (spec §5.4; Phase 4a const-V, Phase 4b const-P):
# append_constraint_layers! adds the adiabatic energy equation for T to the equation set.
#   - _energy_ode_constV : dT/dt = -Σⱼ rⱼ·Δūⱼ / Σᵢ cᵢ·cvᵢ  (ū = h̄ - RT, cv = cp - R; U conserved)
#   - _energy_ode_constP : dT/dt = -V·Σⱼ rⱼ·Δh̄ⱼ / Σᵢ nᵢ·cpᵢ   (H conserved)
# Both require NASA7 thermo on every species (spec §5.3.4 — clear error otherwise).

"Append energy/reactor constraint layers to the equation set (spec §5.4). Phase 4a: the energy
 layer (:adiabatic) adds the const-V energy ODE for T. `cvar`/`T`/`rates` are the shared species
 vars, the temperature symbol, and the per-reaction symbolic net rates from lower_to_mtk."
function append_constraint_layers!(eqs, mech, config, cvar, T, rates; tcx, nvar=nothing, Vvar=nothing)
    config.energy === :adiabatic || return eqs
    if config.constraint === :constant_pressure
        push!(eqs, _energy_ode_constP(mech, nvar, Vvar, T, rates, tcx))    # Task 2
    else
        push!(eqs, _energy_ode_constV(mech, cvar, T, rates, tcx))
    end
    return eqs
end

"Constant-pressure adiabatic energy equation (spec §5.3/§11 Phase 4; probed 2026-07-03 P2):
 dT/dt = -V·Σⱼ rⱼ·Δh̄ⱼ / Σᵢ nᵢ·cpᵢ, with Δh̄ⱼ = Σ_prod ν·h̄ − Σ_react ν·h̄ and cpᵢ = (cp/R)·R (ideal gas).
 All species must carry NASA7 thermo (spec §5.3.4 — clear error otherwise). H = Σnᵢh̄ᵢ(T) is conserved."
function _energy_ode_constP(mech::Mechanism, nvar, Vvar, T, rates, tcx)
    D = ModelingToolkit.D
    R = tcx.R
    for sp in mech.species
        sp.thermo isa NASA7 ||
            error("_energy_ode_constP: species $(sp.name) (id $(sp.id)) has no NASA7 thermo; " *
                  ":adiabatic requires NASA7 thermo on all species (spec §5.3.4). " *
                  "Use energy=:isothermal or provide NASA7 thermo.")
    end
    cp_sum = sum(nvar[sp.id] * _cp_over_R(sp.thermo, T, sp.id, tcx) * R for sp in mech.species)   # [J/K]
    src = 0.0
    for (j, rx) in enumerate(mech.reactions)
        delta_h = 0.0
        for (sid, nu) in rx.products
            delta_h += nu * _h_over_RT(_species_by_id(mech, sid).thermo, T, sid, tcx) * R * T
        end
        for (sid, nu) in rx.reactants
            delta_h -= nu * _h_over_RT(_species_by_id(mech, sid).thermo, T, sid, tcx) * R * T
        end
        src += rates[j] * (-delta_h)                                  # Σⱼ rⱼ·(-Δh̄ⱼ)  [J/(m³·s)]
    end
    return D(T) ~ Vvar * src / cp_sum
end

"Per-reaction Δūⱼ = Σ_prod ν·ū − Σ_react ν·ū with ū = (h/RT − 1)·R·T (ideal-gas internal energy).
 Factored out of _energy_rhs_constV so the reaction-sharded adiabatic Jacobian can form each
 reaction's energy contribution (rateⱼ·Δūⱼ) without duplicating the NASA7 h/RT sum."
function _reaction_delta_u(mech::Mechanism, rx::ReactionData, T, tcx)
    R = tcx.R
    du = 0.0
    for (sid, nu) in rx.products
        du += nu * (_h_over_RT(_thermo_of(mech, sid), T, sid, tcx) - 1) * R * T
    end
    for (sid, nu) in rx.reactants
        du -= nu * (_h_over_RT(_thermo_of(mech, sid), T, sid, tcx) - 1) * R * T
    end
    return du
end

"Σᵢ cᵢ·cvᵢ with cvᵢ = (cp/R − 1)·R (ideal-gas const-V heat-capacity density) [J/(m³·K)].
 Factored out of _energy_rhs_constV; reused by the reaction-sharded adiabatic Jacobian."
_cv_sum_constV(mech::Mechanism, cvar, T, tcx) =
    sum(cvar[sp.id] * (_cp_over_R(sp.thermo, T, sp.id, tcx) - 1) * tcx.R for sp in mech.species)

"Const-V adiabatic energy RHS dT/dt (the expression, not the equation). Refactored out so the
 P-ODE (Task 4) can reuse it without duplicating the Σⱼ rⱼ·Δūⱼ / Σᵢ cᵢ·cvᵢ expression."
function _energy_rhs_constV(mech::Mechanism, cvar, T, rates, tcx)
    for sp in mech.species
        sp.thermo isa NASA7 ||
            error("_energy_rhs_constV: species $(sp.name) (id $(sp.id)) has no NASA7 thermo; " *
                  ":adiabatic requires NASA7 thermo on all species (spec §5.3.4). " *
                  "Use energy=:isothermal or provide NASA7 thermo.")
    end
    cv_sum = _cv_sum_constV(mech, cvar, T, tcx)
    src = sum(rates[j] * (-_reaction_delta_u(mech, rx, T, tcx))
              for (j, rx) in enumerate(mech.reactions))
    return src / cv_sum
end

"Constant-volume adiabatic energy equation D(T) ~ dT/dt (spec §5.3, §11 Phase 4; verified
 2026-07-02). Delegates the RHS to `_energy_rhs_constV` so the P-ODE (Task 4) can reuse it.
 dT/dt = -Σⱼ rⱼ·Δūⱼ / Σᵢ cᵢ·cvᵢ, with ūᵢ=(h/RT-1)·R·T and cvᵢ=(cp/R-1)·R (ideal gas)."
function _energy_ode_constV(mech::Mechanism, cvar, T, rates, tcx)
    D = ModelingToolkit.D
    return D(T) ~ _energy_rhs_constV(mech, cvar, T, rates, tcx)
end

"Σᵢ dcᵢ/dt = Σⱼ (Σ_prod ν − Σ_react ν)·rateⱼ — total concentration RHS (for the const-V P-ODE,
 spec §5.3 iii). Pure: depends only on stoichiometry and the per-reaction rates."
_sum_species_rhs(mech::Mechanism, rates) =
    sum((sum(values(rx.products)) - sum(values(rx.reactants))) * rates[j]
        for (j, rx) in enumerate(mech.reactions))

"P-ODE for const-V (spec §5.3 iii): D(P) ~ R·(T·Σ物种RHS + (Σc)·能量RHS) = d/dt[(Σc)RT].
 `is_adiabatic=false` (isothermal) ⇒ energy RHS term = 0 (T is a parameter, dT/dt=0).
 Task 4 consumes this to add P as a differential state under :adiabatic_constV."
function _p_ode_constV(mech::Mechanism, cvar, Tparam, rates, tcx, Pvar, is_adiabatic::Bool)
    D = ModelingToolkit.D
    sum_rhs = _sum_species_rhs(mech, rates)
    csum = sum(values(cvar))
    energy_rhs = is_adiabatic ? _energy_rhs_constV(mech, cvar, Tparam, rates, tcx) : 0.0
    return D(Pvar) ~ tcx.R * (Tparam * sum_rhs + csum * energy_rhs)
end
