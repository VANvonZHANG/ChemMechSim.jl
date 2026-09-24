# The two MCM/KPP rate types ChemMechSim's parser does not know — defined HERE, example
# side, and handed to load_mechanism through its rate_type_handlers registry:
#
#   include("tools/mcm_rate_types.jl")        # from the driver / budget / bench / tests
#   load_mechanism(SRC; rate_type_handlers = mcm_rate_handlers())
#
# Both ride the generic paramspec/body/needs_T protocol (precedent: MyArrhenius in
# test/test_custom_ratelaw.jl): paramspec declares fields→parameters, `body` is the
# formula's single definition, and the framework's generic rate_constant (numeric) and
# symbolic_kf (lowering) handle the rest. NO src involvement.

# `import` (not `using`) for the three traits — defining methods on another module's
# function requires explicit import.
import ChemMechSim: paramspec, body, needs_T
using ChemMechSim: AbstractKinetics, afactor, kvalue, plain, convert_afactor

"""
Zenith-angle photolysis: J = l·cos(χ)^m·exp(−n/cos(χ)).

Lowered, this is an elementary rate whose value is the single parameter `k_{j}_A`
(afactor, order 1 → s⁻¹), defaulting to J at overhead sun (cz = 1 → l·exp(−n)). The
frozen box solves with that default (perpetual noon); the diurnal box drives the same
parameters from the zenith clock (`mcm_box.jl`, callback on the 60-s grid). The raw
l/m/n ride on the struct so the driver never needs a sidecar file.
"""
struct ZenithPhotolysis <: AbstractKinetics
    l::Float64
    m::Float64
    n::Float64
    A::Float64                # = l·exp(−n), materialized by the handler (J at cz = 1)
end

paramspec(kin::ZenithPhotolysis) = (afactor(:A, "", 0.0),)   # → parameter k_{j}_A
body(kin::ZenithPhotolysis)      = (A, T) -> A
needs_T(kin::ZenithPhotolysis)   = false

"""
Sigmoid branching (MCM pressure-dependent branching correction):

    k = A·exp(B/T)·σ(T),  σ(T) = 1/(1 + C·exp(D/T))      normally
    k = A·exp(B/T)·(1 − σ(T))                            when is_complement

EXACT for all T (B/D are kelvin, C dimensionless; materialized as k_{j}_troeB /
k_{j}_troeD — the KValue role's Troe-flavoured param name, harmless: the driver maps only
k_{j}_A). A's SIGN is preserved per entry: the converter splits MCM's summed rate
expressions into one entry per term, and duplicate equations are summed natively
downstream — a negative A is a deliberate correction, not an error. (Regression history:
an earlier preprocessor PAIRED the two CH3O2+HO2 sigmoids and ran the reaction at 2×
MCM's rate; the pinned test in test_atmospheric_rate_types.jl guards against it.)
"""
struct SigmoidBranching <: AbstractKinetics
    A::Float64
    B::Float64
    C::Float64
    D::Float64
    is_complement::Bool
end

paramspec(kin::SigmoidBranching) = (afactor(:A, "", 0.0), kvalue(:B, "B"), plain(:C),
                                    kvalue(:D, "D"), plain(:is_complement))
function _sigmoid_body(A, B, C, D, comp, T)
    σ = 1 / (1 + C * exp(D / T))
    return comp ? A * exp(B / T) * (1 - σ) : A * exp(B / T) * σ
end
body(kin::SigmoidBranching)      = _sigmoid_body
needs_T(kin::SigmoidBranching)   = true

"""
The handlers dict passed to `load_mechanism(; rate_type_handlers = …)`.

Registry contract: (rxn_dict, reactants, name_to_id, ctx) -> AbstractKinetics.
- zenith: order must be 1 (J in s⁻¹); a first-order A is s⁻¹ on any amount basis, so no
  convert_afactor is needed. A = J(cz = 1) = l·exp(−n).
- sigmoid: order-2 A factors carry the `quantity: molec` basis → convert_afactor.
"""
function mcm_rate_handlers()
    Dict{String,Function}(
        "zenith-angle-photolysis" => function (rxn, reactants, name_to_id, ctx)
            order = sum(values(reactants))
            order == 1 || error("mcm_rate_types: \"$(rxn["equation"])\" has order $order " *
                                "(expected 1) — the J parameter must be s⁻¹")
            l = Float64(rxn["l"]); m = Float64(rxn["m"]); n = Float64(rxn["n"])
            return ZenithPhotolysis(l, m, n, l * exp(-n))
        end,
        "sigmoid-branching" => function (rxn, reactants, name_to_id, ctx)
            order = sum(values(reactants))
            return SigmoidBranching(convert_afactor(Float64(rxn["A"]), ctx, order),
                                    Float64(rxn["B"]), Float64(rxn["C"]),
                                    Float64(rxn["D"]),
                                    Bool(get(rxn, "is_complement", false)))
        end)
end
