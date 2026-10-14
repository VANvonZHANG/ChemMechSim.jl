# Shared test fixtures for the ChemMechSim test suite. Included once by runtests.jl
# BEFORE any test file, so fixtures like _brusselator_mech / _var live in one place
# and do not trigger Julia method-redefinition warnings from duplicate per-file
# definitions.
using ChemMechSim
using ModelingToolkit: unknowns, getname

"Brusselator mechanism (programmatic path): ∅→X, 2X+Y→3X, X→Y, X→∅."
function _brusselator_mech()
    X = SpeciesData(id=1, name="X"); Y = SpeciesData(id=2, name="Y")
    rxns = [
        ReactionData(reactants=Dict{Int,Float64}(),  products=Dict(1=>1.0), kinetics=ElementaryArrhenius(1.0,0,0)),
        ReactionData(reactants=Dict(1=>2.0, 2=>1.0), products=Dict(1=>3.0), kinetics=ElementaryArrhenius(1.0,0,0)),
        ReactionData(reactants=Dict(1=>1.0),         products=Dict(2=>1.0), kinetics=ElementaryArrhenius(3.0,0,0)),
        ReactionData(reactants=Dict(1=>1.0),         products=Dict{Int,Float64}(), kinetics=ElementaryArrhenius(1.0,0,0)),
    ]
    Mechanism(species=[X, Y], reactions=rxns)
end

"Look up a named unknown on a simplified system (order may be reordered)."
_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]

"Index of each named state in the simplified system (order may be reordered)."
_state_index(sys) = Dict(String(getname(s)) => i for (i, s) in enumerate(unknowns(sys)))

"Look up a named parameter on a simplified system (order may be reordered)."
_param(sys, name) =
    parameters(sys)[findfirst(p -> String(getname(p)) == name, parameters(sys))]

"Default parameter values of a simplified system, in parameters(sys) order (for ODEFunction
 out-of-place calls under units, where `nothing` does not substitute defaults)."
_pvals(sys) = [ModelingToolkit.getdefault(p) for p in ModelingToolkit.parameters(sys)]

"Replace the T entry in a default param vector with Tv (for ODEFunction pointwise checks)."
function replace_T(pvals, sys, Tv)
    ps = ModelingToolkit.parameters(sys)
    [String(ModelingToolkit.getname(p)) == "T" ? Tv : v for (p, v) in zip(ps, pvals)]
end

# ---- the golden toy used by every adjoint test: A --k1--> B --k2--> A ----
# Linear, so the exact solution and Jacobian are closed forms. STATE ORDER IS MTK's:
# unknowns(sys) = [B, A] (verified 2026-10-09) — never assume species-id order.
function ab_toy()
    spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B")
    rx1 = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=ElementaryArrhenius(0.8, 0.0, 0.0), reverse_policy=Irreversible())
    rx2 = ReactionData(reactants=Dict(2 => 1.0), products=Dict(1 => 1.0),
                       kinetics=ElementaryArrhenius(0.3, 0.0, 0.0), reverse_policy=Irreversible())
    return Mechanism(species=[spA, spB], reactions=[rx1, rx2])
end
