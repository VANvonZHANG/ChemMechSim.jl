# Brusselator demo for ChemMechSim (Phase 1 MVP): one system, three input
# routes -> one Mechanism -> one solution, closed by a CairoMakie portrait of
# the A=1/B=3 limit cycle.
# Run:  julia --project=. examples/demos/brusselator.jl
using ChemMechSim
using Catalyst: @reaction_network
using ModelingToolkit: equations, unknowns, getname
using CairoMakie

_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]

# -- Route 1: programmatic construction (plain Julia types, no DSL / no file) --
mech_programmatic = Mechanism(
    species=[SpeciesData(id=1, name="X"), SpeciesData(id=2, name="Y")],
    reactions=[
        ReactionData(reactants=Dict{Int,Float64}(),  products=Dict(1=>1.0),        kinetics=ElementaryArrhenius(1.0,0.0,0.0)),  # none -> X
        ReactionData(reactants=Dict(1=>2.0, 2=>1.0), products=Dict(1=>3.0),        kinetics=ElementaryArrhenius(1.0,0.0,0.0)),  # 2X + Y -> 3X
        ReactionData(reactants=Dict(1=>1.0),         products=Dict(2=>1.0),        kinetics=ElementaryArrhenius(3.0,0.0,0.0)),  # X -> Y
        ReactionData(reactants=Dict(1=>1.0),         products=Dict{Int,Float64}(), kinetics=ElementaryArrhenius(1.0,0.0,0.0)),  # X -> none
    ])

# -- Route 2: Catalyst ReactionSystem import (mass-action numeric subset) --
rn = @reaction_network begin
    1.0, ∅ → X
    1.0, 2*X + Y → 3*X
    3.0, X → Y
    1.0, X → ∅
end
mech_catalyst = import_from_catalyst(rn)

# -- Route 3: Cantera-YAML mechanism file (subset notes in the fixture header) --
mech_yaml = load_mechanism(joinpath(@__DIR__, "..", "mechanism", "brusselator.yaml"))

# -- All routes converge on the same intermediate representation --
routes = [("programmatic", mech_programmatic), ("catalyst import", mech_catalyst), ("yaml file", mech_yaml)]
println("Route              -> Mechanism (intermediate representation)")
for (tag, m) in routes
    println(rpad(tag, 18), "-> ", length(m.species), " species, ", length(m.reactions), " reactions")
end

println("\nSpecies balance from route 3 (YAML):")
for eq in equations(extract_system(ChemPhaseSystem(mech_yaml))); println("  ", eq); end

# -- Same representation => same physics: identical limit cycle from all routes --
println("\nSame IR, same solve -- X(t) at t = 40 from each route:")
X40 = Float64[]
sol = nothing
for (tag, m) in routes
    global sol
    phase = ChemPhaseSystem(m)
    sys   = extract_system(phase)
    sol   = simulate(phase, (0.0, 40.0); u0=Dict("X"=>1.0, "Y"=>0.5), reltol=1e-9, abstol=1e-9)
    x40   = Float64(sol(40.0; idxs=_var(sys, "X")))
    push!(X40, x40)
    println("  ", rpad(tag, 18), "X(40) = ", round(x40, digits=6))
end
spread = maximum(X40) - minimum(X40)
println(spread < 1e-6 ? "PASS: three routes agree (same Mechanism => same limit cycle)" :
                        "FAIL: routes diverge -- X(40) spread = $spread")

# -- Limit-cycle statistics + portrait (last route's sol; all routes agree) --
sys = extract_system(ChemPhaseSystem(mech_yaml))
xv, yv = _var(sys, "X"), _var(sys, "Y")
ts = range(0.0, 40.0; length=1000)
xs = [Float64(sol(t; idxs=xv)) for t in ts]
ys = [Float64(sol(t; idxs=yv)) for t in ts]

peaks = [ts[i] for i in 2:length(ts)-1 if xs[i] > xs[i-1] && xs[i] >= xs[i+1]]
periods = diff(peaks)
println("\nLimit cycle: X ∈ [", round(minimum(xs), digits=3), ", ", round(maximum(xs), digits=3), "]",
        "  Y ∈ [", round(minimum(ys), digits=3), ", ", round(maximum(ys), digits=3), "]",
        "  period ≈ ", isempty(periods) ? "?" : round(sum(periods)/length(periods), digits=3))

# -- Portrait: the fixed point repels, the limit cycle attracts --
# (A, B/A) = (1, 3) is a spiral source: the Jacobian there has eigenvalues
# 0.5 ± 0.866i, so perturbations grow like exp(t/2) while rotating. Starting
# exactly ON the equilibrium would stay put forever; nudging X by 2% makes the
# tiny wobble grow exponentially and settle onto the cycle. Trajectories
# started anywhere else collapse onto the same closed orbit from both sides.
phase = ChemPhaseSystem(mech_yaml)

sol_main = simulate(phase, (0.0, 50.0); u0=Dict("X"=>1.02, "Y"=>3.0), reltol=1e-9, abstol=1e-9)
tm = range(0.0, 50.0; length=2000)
xm = [Float64(sol_main(t; idxs=xv)) for t in tm]
ym = [Float64(sol_main(t; idxs=yv)) for t in tm]

tg = range(0.0, 40.0; length=800)
# 200 gray trajectories: concentric rings around the fixed point, densely
# packed near it and thinning outward, so the flow is seen converging onto one
# closed orbit from everywhere in the phase plane
gray_u0 = vcat(
    [(1.0 + 0.50cos(θ), 3.0 + 0.50sin(θ)) for θ in range(0, 2π; length=56)[1:end-1]],  # 55
    [(1.0 + 0.90cos(θ), 3.0 + 0.90sin(θ)) for θ in range(0, 2π; length=46)[1:end-1]],  # 45
    [(1.0 + 1.30cos(θ), 3.0 + 1.30sin(θ)) for θ in range(0, 2π; length=41)[1:end-1]],  # 40
    [(1.0 + 1.70cos(θ), 3.0 + 1.70sin(θ)) for θ in range(0, 2π; length=36)[1:end-1]],  # 35
    [(1.0 + 2.05cos(θ), 3.0 + 2.05sin(θ)) for θ in range(0, 2π; length=26)[1:end-1]],  # 25
)
gray_trajs = Tuple{Vector{Float64},Vector{Float64}}[]
for (x0, y0) in gray_u0
    s = simulate(phase, (0.0, 40.0); u0=Dict("X"=>x0, "Y"=>y0), reltol=1e-8, abstol=1e-8)
    push!(gray_trajs, ([Float64(s(t; idxs=xv)) for t in tg], [Float64(s(t; idxs=yv)) for t in tg]))
end

# publication sizing: two-column figure (~190 mm print width). Font sizes are
# chosen so that after scaling 1050 px -> 190 mm the printed sizes are ~8 pt
# axis labels / ~7 pt ticks and annotations.
pub_theme = Theme(fontsize=14,
                  Axis=(xlabelsize=16, ylabelsize=16,
                        xticklabelsize=14, yticklabelsize=14,
                        titlesize=20, titlealign=:left))
fig = Figure(size=(1050, 430), backgroundcolor=:white, theme=pub_theme)

ax_t = Axis(fig[1, 1]; xlabel="t", ylabel="concentration",
            title="(a)", titlealign=:left)
lines!(ax_t, tm, xm; color=:midnightblue, linewidth=2.5, label="X(t)")
lines!(ax_t, tm, ym; color=:cornflowerblue, linewidth=2, linestyle=:dash, label="Y(t)")
axislegend(ax_t; position=:rt)

# limits chosen so the cycle (and the fixed point it encircles) sits near the
# middle of the axes instead of being pushed aside by the outer starts
ax_p = Axis(fig[1, 2]; xlabel="X", ylabel="Y", limits=((-1.1, 4.1), (0.0, 5.6)),
            title="(b)", titlealign=:left)
for (gx, gy) in gray_trajs
    lines!(ax_p, gx, gy; color=(:black, 0.20), linewidth=0.9)
end
lines!(ax_p, xm, ym; color=:midnightblue, linewidth=3)
scatter!(ax_p, [1.02], [3.0]; color=:midnightblue, markersize=8)  # the nudge
scatter!(ax_p, [1.0], [3.0]; color=:black, marker=:xcross, markersize=18)
poly!(ax_p, Rect2f(1.12, 3.02, 2.05, 0.62); color=(:white, 0.9), strokewidth=0)
text!(ax_p, "unstable fixed point"; position=Point2f(1.19, 3.32),
      fontsize=14, align=(:left, :bottom), color=:black)
text!(ax_p, "(X*, Y*) = (A, B/A) = (1, 3)"; position=Point2f(1.19, 3.10),
      fontsize=14, align=(:left, :bottom), color=:black)

png_out = joinpath(@__DIR__, "brusselator.png")
save(png_out, fig)
println("saved ", png_out)
