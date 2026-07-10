# Constraint-layer assembly (spec §5.4; Phase 4a const-V, Phase 4b const-P):
# append_constraint_layers! adds the adiabatic energy equation for T to the equation set.
#   - _energy_ode_constV : dT/dt = -Σⱼ rⱼ·Δūⱼ / Σᵢ cᵢ·cvᵢ  (ū = h̄ - RT, cv = cp - R; U conserved)
#   - _energy_ode_constP : dT/dt = -V·Σⱼ rⱼ·Δh̄ⱼ / Σᵢ nᵢ·cpᵢ   (H conserved)
# Both require NASA7 thermo on every species (spec §5.3.4 — clear error otherwise).

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
