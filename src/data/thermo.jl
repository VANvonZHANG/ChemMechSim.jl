# Thermodynamic model types. Pure Julia (no MTK/unit dependency).
#
# NASA7 polynomial model: cp/R, h/RT, s/R as 7-coefficient polynomials in T.
# Task 6 (K_c/ThermoReverse) consumes g_over_RT = h_over_RT - s_over_R.

"Abstract parent of thermodynamic models (e.g. NASA7, NASA9)."
abstract type ThermoModel end

"NASA7 7-coefficient polynomial thermo model (spec §5.5).
 Two coefficient sets span [Tlow,Tmid] (low) and [Tmid,Thigh] (high); each set is
 (a1,a2,a3,a4,a5,a6,a7) where:
   cp/R = a1 + a2·T + a3·T² + a4·T³ + a5·T⁴
   h/RT = a1 + a2·T/2 + a3·T²/3 + a4·T³/4 + a5·T⁴/5 + a6/T
   s/R  = a1·ln(T) + a2·T + a3·T²/2 + a4·T³/3 + a5·T⁴/4 + a7"
struct NASA7 <: ThermoModel
    low_coeffs::NTuple{7,Float64}
    high_coeffs::NTuple{7,Float64}
    Tlow::Float64
    Tmid::Float64
    Thigh::Float64
end

"Select the active coefficient tuple for temperature T (low range ≤ Tmid, high otherwise)."
function _nasa7_coeffs(m::NASA7, T::Real)
    T <= m.Tmid ? m.low_coeffs : m.high_coeffs
end

# —— NASA7 polynomial bodies (generic; work for Real and symbolic Num). Single truth source. ——
"Dimensionless cp/R = a1 + a2·T + a3·T² + a4·T³ + a5·T⁴. Generic over T and the coeff tuple."
_nasa7_cp((a1, a2, a3, a4, a5, _, _), T) = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4

"Dimensionless h/RT = a1 + a2·T/2 + a3·T²/3 + a4·T³/4 + a5·T⁴/5 + a6/T. Generic over T."
_nasa7_h((a1, a2, a3, a4, a5, a6, _), T) = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T

"Dimensionless s/R = a1·ln(T) + a2·T + a3·T²/2 + a4·T³/3 + a5·T⁴/4 + a7. Generic over T."
function _nasa7_s((a1, a2, a3, a4, a5, _, a7), T)
    return a1*log(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
end

# —— plain-Real entries (public API unchanged): select coeffs, call the generic body ——
"Dimensionless cp/R (selects the active coeff range, then calls the generic body)."
cp_over_R(m::NASA7, T::Real) = _nasa7_cp(_nasa7_coeffs(m, T), T)

"Dimensionless h/RT."
h_over_RT(m::NASA7, T::Real) = _nasa7_h(_nasa7_coeffs(m, T), T)

"Dimensionless s/R."
s_over_R(m::NASA7, T::Real)  = _nasa7_s(_nasa7_coeffs(m, T), T)

"Dimensionless g/RT = h/RT - s/R (Task 6 uses this for K_c)."
g_over_RT(m::NASA7, T::Real) = h_over_RT(m, T) - s_over_R(m, T)

# —— Equilibrium constant K_c(T) + analytic ∂K_c/∂T (data layer; MTK-free) ——
# Task 1 of the K_c-opaque plan: numeric primitives that Task 2's registered `keq`
# opaque call node will invoke at runtime (Float64). Mirrors the PLOG opaque pattern:
# the data layer owns the formula; the lowering layer registers the opaque wrapper.
# Behavior reference: src/lowering/thermo.jl `_equilibrium_constant` (same R_GAS, P_STD).

"Per-reaction precomputed thermo for the equilibrium constant K_c: (NASA7, ν) pairs for
 products and reactants, plus Δν = Σν_prod − Σν_react. Built once per lowering; the opaque
 `keq` call node reads it. MTK-free."
struct KcData
    prod::Vector{Tuple{NASA7,Float64}}
    react::Vector{Tuple{NASA7,Float64}}
    dnu::Float64
end
"Build KcData from a reaction + mechanism (looks up each species' NASA7 thermo).
 Types unannotated because thermo.jl is included before reaction.jl/mechanism.jl in
 src/ChemMechSim.jl; the names ReactionData/Mechanism/species_by_id resolve at call time
 via Julia's late binding. `rx.products`/`rx.reactants` are Dict{SpeciesID,Float64};
 `species_by_id(mech, sid).thermo` must return NASA7 (ThermoReverse requirement)."
function KcData(rx, mech)
    prod  = Tuple{NASA7,Float64}[(species_by_id(mech, sid).thermo::NASA7, nu) for (sid,nu) in rx.products]
    react = Tuple{NASA7,Float64}[(species_by_id(mech, sid).thermo::NASA7, nu) for (sid,nu) in rx.reactants]
    dnu = sum(values(rx.products)) - sum(values(rx.reactants))
    KcData(prod, react, dnu)
end

"Equilibrium constant K_c(T) = exp(-Δg°/RT)·(P°/(R·T))^Δν (spec §3.4 #4). MTK-free, generic
 over Real. Numerically identical to lowering's _equilibrium_constant (R_GAS=8.314, P_STD=101325)."
function equilibrium_constant(kcd::KcData, T; R::Real=R_GAS, P_STD::Real=P_STD)
    g = 0.0
    for (th, nu) in kcd.prod;  g += nu * g_over_RT(th, T); end
    for (th, nu) in kcd.react; g -= nu * g_over_RT(th, T); end
    base = exp(-g)
    iszero(kcd.dnu) && return base
    return base * (P_STD / (R * T)) ^ kcd.dnu
end

"∂K_c/∂T = K_c·(Δh°/(R·T²) − Δν/T) (van't Hoff + the (P°/RT)^Δν chain rule).
 Δh°/(RT) = Σ ν·h_over_RT. MTK-free, analytic."
function equilibrium_constant_dT(kcd::KcData, T; R::Real=R_GAS, P_STD::Real=P_STD)
    Kc = equilibrium_constant(kcd, T; R=R, P_STD=P_STD)
    h_RT = 0.0
    for (th, nu) in kcd.prod;  h_RT += nu * h_over_RT(th, T); end
    for (th, nu) in kcd.react; h_RT -= nu * h_over_RT(th, T); end
    # ∂(ln K_c)/∂T = Δh°/(R·T²) − Δν/T = (h_RT − Δν)/T  where h_RT = Δh°/(RT)
    return Kc * ((h_RT - kcd.dnu) / T)
end

"Molar cp = (cp/R)·R  [J/(mol·K)]."
cp_molar(m::NASA7, T::Real) = cp_over_R(m, T) * R_GAS

"Molar h = (h/RT)·R·T  [J/mol]."
h_molar(m::NASA7, T::Real) = h_over_RT(m, T) * R_GAS * T

"Molar s = (s/R)·R  [J/(mol·K)]."
s_molar(m::NASA7, T::Real) = s_over_R(m, T) * R_GAS

"Molar g = (g/RT)·R·T  [J/mol] (equivalently h - T·s)."
g_molar(m::NASA7, T::Real) = g_over_RT(m, T) * R_GAS * T

"Molar internal energy ū = h̄ − R·T  [J/mol] (ideal gas; const-V energy equation, Phase 4a)."
u_molar(m::NASA7, T::Real) = h_molar(m, T) - R_GAS * T

"Molar constant-volume heat capacity cv = cp − R  [J/(mol·K)] (ideal gas)."
cv_molar(m::NASA7, T::Real) = cp_molar(m, T) - R_GAS

"Dimensionless ū/(RT) = h/RT − 1."
u_over_RT(m::NASA7, T::Real) = h_over_RT(m, T) - 1

"Dimensionless cv/R = cp/R − 1."
cv_over_R(m::NASA7, T::Real) = cp_over_R(m, T) - 1

"A species-keyed collection of ThermoModel entries (the NASA-coefficient store)."
struct ThermoDatabase
    entries::Dict{String,ThermoModel}
end

"Construct an empty ThermoDatabase."
ThermoDatabase() = ThermoDatabase(Dict{String,ThermoModel}())
