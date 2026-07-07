# Standard atomic masses (kg/mol, §5.6 strict-SI quantity=mol).
# Covers elements common in gas-phase combustion mechanisms; extend as needed.
const ATOMIC_MASSES = Dict{String,Float64}(
    "H"  => 0.001008,
    "O"  => 0.015999,
    "C"  => 0.012011,
    "N"  => 0.014007,
    "Ar" => 0.039948,
    "He" => 0.004003,
    "S"  => 0.03206,
    "Cl" => 0.03545,
    "Na" => 0.022990,
    "Mg" => 0.024305,
    "Si" => 0.028085,
    "K"  => 0.039098,
    "Ca" => 0.040078,
)

"Molar mass [kg/mol] from elemental composition (§5.6 canonical SI, quantity=mol).
 Throws on unknown element so missing data is caught at parse time (§5.3.4)."
function molecular_weight(composition::Dict{String,Int})::Float64
    mw = 0.0
    for (elem, count) in composition
        haskey(ATOMIC_MASSES, elem) ||
            error("molecular_weight: unknown element \"$elem\"; extend ATOMIC_MASSES")
        mw += ATOMIC_MASSES[elem] * count
    end
    return mw
end
