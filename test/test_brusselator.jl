using Test
using ChemMechSim
using Catalyst
using ModelingToolkit: equations, unknowns, getname

# ---- programmatic Brusselator mechanism (A=1, B=3) ----
function _brusselator_programmatic()
    X = SpeciesData(id=1, name="X"); Y = SpeciesData(id=2, name="Y")
    rxns = [
        ReactionData(reactants=Dict{Int,Float64}(),   products=Dict(1=>1.0), kinetics=ElementaryArrhenius(1.0,0,0)),
        ReactionData(reactants=Dict(1=>2.0, 2=>1.0),  products=Dict(1=>3.0), kinetics=ElementaryArrhenius(1.0,0,0)),
        ReactionData(reactants=Dict(1=>1.0),          products=Dict(2=>1.0), kinetics=ElementaryArrhenius(3.0,0,0)),
        ReactionData(reactants=Dict(1=>1.0),          products=Dict{Int,Float64}(), kinetics=ElementaryArrhenius(1.0,0,0)),
    ]
    Mechanism(species=[X, Y], reactions=rxns)
end

# ---- Catalyst Brusselator (numeric rates → lossless import) ----
function _brusselator_catalyst()
    rn = @reaction_network begin
        1.0, ∅ → X
        1.0, 2*X + Y → 3*X
        3.0, X → Y
        1.0, X → ∅
    end
    ChemPhaseSystem(rn)
end

@testset "MVP §10.3 #3: dual-path equations are identical" begin
    sysP = extract_system(ChemPhaseSystem(_brusselator_programmatic()))
    sysC = extract_system(_brusselator_catalyst())
    @test equations(sysP) == equations(sysC)
end

@testset "MVP §10.3 #4: extract_system is viewable" begin
    sys = extract_system(ChemPhaseSystem(_brusselator_programmatic()))
    @test length(equations(sys)) == 2
    @test length(unknowns(sys)) == 2
    @test sort([String(getname(s)) for s in unknowns(sys)]) == ["X", "Y"]
end

@testset "MVP §10.3 #5/#6: limit cycle, bounded + stable period" begin
    phase = ChemPhaseSystem(_brusselator_programmatic())
    sol = simulate(phase, (0.0, 40.0);
                   u0=Dict("X" => 1.0, "Y" => 0.5),
                   reltol=1e-9, abstol=1e-9, saveat=0.05)
    unn = unknowns(phase.sys)
    Xv = unn[findfirst(s -> String(getname(s)) == "X", unn)]
    xs = sol[Xv]

    @test all(isfinite, xs)                         # bounded / non-divergent
    @test minimum(xs) > 0.0 && maximum(xs) < 10.0   # sane amplitude for A=1, B=3

    # successive maxima of X give a stable period (within 5%)
    peaks = Float64[]
    for i in 2:length(xs)-1
        if xs[i] > xs[i-1] && xs[i] >= xs[i+1]
            push!(peaks, sol.t[i])
        end
    end
    periods = diff(peaks)
    @test length(periods) >= 2
    mean_period = sum(periods) / length(periods)
    @test (maximum(periods) - minimum(periods)) / mean_period < 0.05
end

@testset "MVP §10.3 #3 (numeric): dual-path trajectories agree" begin
    phP = ChemPhaseSystem(_brusselator_programmatic())
    phC = _brusselator_catalyst()
    common = (0.0, 10.0)
    sP = simulate(phP, common; u0=Dict("X"=>1.0, "Y"=>0.5), reltol=1e-11, abstol=1e-11, saveat=0.1)
    sC = simulate(phC, common; u0=Dict("X"=>1.0, "Y"=>0.5), reltol=1e-11, abstol=1e-11, saveat=0.1)
    @test sP.t ≈ sC.t
    maxdiff = maximum(maximum(abs.(a .- b)) for (a, b) in zip(sP.u, sC.u))
    @test maxdiff < 1e-6
end
