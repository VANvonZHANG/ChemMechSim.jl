using Test
using ChemMechSim
using ModelingToolkit: equations, ODEProblem, unknowns, getname

phase() = ChemPhaseSystem(
    Mechanism(species=[SpeciesData(id=1,name="A"), SpeciesData(id=2,name="B")],
        reactions=[ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                                kinetics=ElementaryArrhenius(2.0,0,0))]))

@testset "extract_system" begin
    ph = phase()
    @test extract_system(ph) === ph.sys
    @test length(equations(extract_system(ph))) == 2
end

@testset "build_problem" begin
    ph = phase()
    prob = build_problem(ph, Dict("A"=>3.0, "B"=>0.0), (0.0, 1.0))
    @test prob isa ODEProblem
end

@testset "simulate: A->B decay matches analytic 3*exp(-2t)" begin
    ph = phase()
    sol = simulate(ph, (0.0, 1.0); u0=Dict("A"=>3.0, "B"=>0.0), reltol=1e-10, abstol=1e-10)
    a_idx = findfirst(s -> String(getname(s)) == "A", unknowns(ph.sys))
    @test sol.u[end][a_idx] ≈ 3*exp(-2*1.0) atol=1e-4
    @test all(isfinite, sol.u[end])
end

@testset "generate_function" begin
    ph = phase()
    code = generate_function(extract_system(ph))
    @test code isa Expr
    s = string(code)
    @test length(s) > 0
    @test occursin("function", s)   # standalone RHS function (build_function out-of-place form)
end
