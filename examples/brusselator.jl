# Brusselator limit-cycle demo for ChemMechSim (Phase 1 MVP).
# Run:  julia --project=. examples/brusselator.jl
#
# Produces a phase-portrait PNG if Plots is loadable in this environment;
# otherwise prints the limit-cycle statistics (the cycle is demonstrated
# numerically either way). Plots is intentionally NOT a runtime dependency of
# ChemMechSim (the offline registry here omits it), so it is loaded optionally.
using ChemMechSim
using ModelingToolkit: equations, unknowns, getname

X = SpeciesData(id=1, name="X"); Y = SpeciesData(id=2, name="Y")
mech = Mechanism(species=[X, Y], reactions=[
    ReactionData(reactants=Dict{Int,Float64}(),   products=Dict(1=>1.0), kinetics=ElementaryArrhenius(1.0,0,0)),
    ReactionData(reactants=Dict(1=>2.0, 2=>1.0),  products=Dict(1=>3.0), kinetics=ElementaryArrhenius(1.0,0,0)),
    ReactionData(reactants=Dict(1=>1.0),          products=Dict(2=>1.0), kinetics=ElementaryArrhenius(3.0,0,0)),
    ReactionData(reactants=Dict(1=>1.0),          products=Dict{Int,Float64}(), kinetics=ElementaryArrhenius(1.0,0,0)),
])

phase = ChemPhaseSystem(mech)
println("Brusselator equations:")
for eq in equations(extract_system(phase)); println("  ", eq); end

sol = simulate(phase, (0.0, 40.0); u0=Dict("X"=>1.0, "Y"=>0.5), reltol=1e-9, abstol=1e-9)
unn = unknowns(extract_system(phase))
xv = unn[findfirst(s -> String(getname(s)) == "X", unn)]
xs = sol[xv]
peaks = Float64[]
for i in 2:length(xs)-1
    if xs[i] > xs[i-1] && xs[i] >= xs[i+1]; push!(peaks, sol.t[i]); end
end
periods = diff(peaks)
println("Limit cycle: X ∈ [", round(minimum(xs), digits=3), ", ", round(maximum(xs), digits=3), "]",
        "  period ≈ ", isempty(periods) ? "?" : round(sum(periods)/length(periods), digits=3))

has_plots = try; @eval using Plots; true; catch _; false; end
if has_plots
    yv = unn[findfirst(s -> String(getname(s)) == "Y", unn)]
    p1 = plot(sol; idxs=[xv, yv], title="Brusselator (A=1, B=3)", xlabel="t")
    p2 = plot(sol; idxs=(xv, yv), title="limit cycle", xlabel="X", ylabel="Y")
    plot(p1, p2; layout=(1, 2), size=(1000, 400))
    savefig("examples/brusselator.png")
    println("saved examples/brusselator.png")
else
    println("Plots not loadable in this env; skipping PNG (limit cycle shown numerically above).")
end
