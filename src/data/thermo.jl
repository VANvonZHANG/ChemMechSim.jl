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

"Dimensionless cp/R = a1 + a2·T + a3·T² + a4·T³ + a5·T⁴."
function cp_over_R(m::NASA7, T::Real)
    a1, a2, a3, a4, a5 = _nasa7_coeffs(m, T)
    return a1 + a2 * T + a3 * T^2 + a4 * T^3 + a5 * T^4
end

"Dimensionless h/RT = a1 + a2·T/2 + a3·T²/3 + a4·T³/4 + a5·T⁴/5 + a6/T."
function h_over_RT(m::NASA7, T::Real)
    a1, a2, a3, a4, a5, a6 = _nasa7_coeffs(m, T)
    return a1 + a2 * T / 2 + a3 * T^2 / 3 + a4 * T^3 / 4 + a5 * T^4 / 5 + a6 / T
end

"Dimensionless s/R = a1·ln(T) + a2·T + a3·T²/2 + a4·T³/3 + a5·T⁴/4 + a7."
function s_over_R(m::NASA7, T::Real)
    a1, a2, a3, a4, a5, _, a7 = _nasa7_coeffs(m, T)
    return a1 * log(T) + a2 * T + a3 * T^2 / 2 + a4 * T^3 / 3 + a5 * T^4 / 4 + a7
end

"Dimensionless g/RT = h/RT - s/R (Task 6 uses this for K_c)."
g_over_RT(m::NASA7, T::Real) = h_over_RT(m, T) - s_over_R(m, T)

"Molar cp = (cp/R)·R  [J/(mol·K)]."
cp_molar(m::NASA7, T::Real) = cp_over_R(m, T) * R_GAS

"Molar h = (h/RT)·R·T  [J/mol]."
h_molar(m::NASA7, T::Real) = h_over_RT(m, T) * R_GAS * T

"Molar s = (s/R)·R  [J/(mol·K)]."
s_molar(m::NASA7, T::Real) = s_over_R(m, T) * R_GAS

"Molar g = (g/RT)·R·T  [J/mol] (equivalently h - T·s)."
g_molar(m::NASA7, T::Real) = g_over_RT(m, T) * R_GAS * T

"A species-keyed collection of ThermoModel entries (the NASA-coefficient store)."
struct ThermoDatabase
    entries::Dict{String,ThermoModel}
end

"Construct an empty ThermoDatabase."
ThermoDatabase() = ThermoDatabase(Dict{String,ThermoModel}())
