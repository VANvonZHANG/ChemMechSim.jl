# Canonical units for ChemMechSim — strict SI base, amount-of-substance in mol.
# DynamicQuantities is MTK's native unit backend (Unitful is deprecated by SciML).
module ChemUnits
    using DynamicQuantities

    const conc      = u"mol/m^3"   # concentration (state basis :concentration)
    const temp      = u"K"         # temperature
    const press     = u"Pa"        # pressure
    const vol       = u"m^3"       # volume
    const molmass   = u"kg/mol"    # molar mass (H2O = 0.018015 kg/mol)
    const molenergy = u"J/mol"     # molar energy (Ea, h, g)

    """
        canonical(q, ref)

    Normalize a quantity (or bare number) to a canonical bare number (SI base).
    Regular `u"..."` quantities are stored eagerly in SI base units, so the value
    is obtained directly with `ustrip`; `ref` is used only to validate that `q`
    carries the expected physical dimension (a mismatch throws). A bare `Real` is
    assumed already canonical and returned unchanged.

    Note: `uconvert(ref, q)` (as in spec §5.6.4) does NOT work with regular
    `u"..."` units — DynamicQuantities only allows `uconvert` for symbolic
    (`us"..."`) dimensions. Plain `ustrip` is the correct canonicalizer here.
    """
    function canonical(q::AbstractQuantity, ref)
        dimension(q) == dimension(ref) ||
            error("canonical: dimension mismatch ($(dimension(q)) ≠ $(dimension(ref)))")
        return ustrip(q)
    end
    canonical(x::Real, ref) = x

    export canonical, conc, temp, press, vol, molmass, molenergy
end # module
