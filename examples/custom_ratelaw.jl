# Custom rate law example (Plan A T7): a user-defined scaled-Arrhenius law.
# Run: julia --project=. examples/custom_ratelaw.jl
#
# The user defines struct + formula + paramspec/needs_T ONLY — no ChemMechSim/lowering
# edits — and the law lowers into a simulatable MTK system. This demonstrates the L2
# generic materializer (Task 6): any kinetics type declaring paramspec + body is lowered
# automatically by the fallback symbolic_kf(::AbstractKinetics, ctx).
using ChemMechSim
using ChemMechSim: afactor, ktemp, plain, paramspec, body, needs_T
using ModelingToolkit: getname, parameters

# 1. Define the rate-law type and its formula.
struct MyArrhenius <: AbstractKinetics
    A::Float64; b::Float64; Ea::Float64; f::Float64        # k(T) = f·A·T^b·exp(-Ea/RT)
end
_my_arrhenius_body(A, b, θ, f, T) = f * A * T^b * exp(-θ / T)

# 2. Declare parameter roles + body + T-dependence (the whole "lowering" contract).
#    These extend ChemMechSim's generic functions, so qualify them with the module prefix.
ChemMechSim.paramspec(kin::MyArrhenius) = (afactor(:A, "", kin.b), plain(:b), ktemp(:Ea, ""), plain(:f))
ChemMechSim.body(kin::MyArrhenius) = _my_arrhenius_body
ChemMechSim.needs_T(kin::MyArrhenius) = true

# 3. Use it like any built-in law.
#    Use the zero-point config (MechanismConfig() default): pure-kinetics bare ODE.
#    T is auto-created as a parameter (needs_T=true) with default 300 K; override to 1000 K.
mech = Mechanism(;
    species = [SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
    reactions = [ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                              kinetics=MyArrhenius(1.0, 0.5, 5000.0, 2.0),
                              reverse_policy=Irreversible())])
phase = ChemMechSim.ChemPhaseSystem(mech)
sys = ChemMechSim.extract_system(phase)
Tparam = parameters(sys)[findfirst(p -> String(getname(p)) == "T", parameters(sys))]
sol = simulate(phase, (0.0, 1.0); u0=Dict("A"=>1.0, "B"=>0.0), params=[Tparam => 1000.0],
               reltol=1e-9, abstol=1e-12)
println("Custom law MyArrhenius: k(T) = f·A·T^b·exp(-Ea/RT)")
println("  A(0)   = ", sol(0.0; idxs=sys.A))
println("  A(1.0) = ", sol(1.0; idxs=sys.A), "  (decays)")
println("  B(1.0) = ", sol(1.0; idxs=sys.B), "  (grows)")
