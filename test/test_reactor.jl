using Test
using ChemMechSim
using Catalyst
using ModelingToolkit: equations, ODEProblem, unknowns, getname

_decay_mech() = Mechanism(
    species=[SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
    reactions=[ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                            kinetics=ElementaryArrhenius(2.0, 0.0, 0.0))])

@testset "BatchReactor: construct from Mechanism (default zero-point)" begin
    r = BatchReactor(_decay_mech())
    @test r isa BatchReactor
    @test r.phase isa ChemPhaseSystem
    @test r.name === :batch
    @test extract_system(r) === r.phase.sys
    @test length(equations(extract_system(r))) == 2
    @test occursin("BatchReactor", string(r))            # show method
end

@testset "BatchReactor: construct from ChemPhaseSystem" begin
    phase = ChemPhaseSystem(_decay_mech())
    r = BatchReactor(phase; name=:decay)
    @test r.phase === phase
    @test r.name === :decay
end

@testset "BatchReactor: construct from Catalyst ReactionSystem" begin
    rn = @reaction_network begin
        2.0, A → B
    end
    r = BatchReactor(rn)
    @test r isa BatchReactor
    @test length(unknowns(extract_system(r))) == 2
end

@testset "BatchReactor: filename string rejected with guidance" begin
    @test_throws ErrorException BatchReactor("gri30.yaml")
end

@testset "BatchReactor: non-zero-point config errors" begin
    # EOS needs MW + units-lowering; not in Phase 2.
    @test_throws ErrorException BatchReactor(_decay_mech(); eos=:ideal_gas)
end

@testset "BatchReactor: simulate A->B decay matches analytic 3*exp(-2t)" begin
    r = BatchReactor(_decay_mech())
    sol = simulate(r, (0.0, 1.0); u0=Dict("A" => 3.0, "B" => 0.0), reltol=1e-10, abstol=1e-10)
    @test sol.u[end][findfirst(s -> String(getname(s)) == "A", unknowns(extract_system(r)))] ≈ 3 * exp(-2 * 1.0) atol=1e-4
    @test all(isfinite, sol.u[end])
end

@testset "BatchReactor: build_problem returns ODEProblem" begin
    r = BatchReactor(_decay_mech())
    prob = build_problem(r, Dict("A" => 1.0, "B" => 0.0), (0.0, 0.5))
    @test prob isa ODEProblem
end

@testset "BatchReactor: generate_function dispatches" begin
    r = BatchReactor(_decay_mech())
    code = generate_function(r)
    @test code isa Expr
    @test occursin("function", string(code))
end

@testset "convenience_config: maps all four modes" begin
    k = convenience_config(:kinetic)
    @test k.energy === :isothermal && k.constraint === :none && k.eos === :off &&
          k.thermo_data === :none && k.reverse_rate === :irreversible
    f = convenience_config(:fixedT)
    @test f.energy === :isothermal && f.constraint === :constant_volume &&
          f.eos === :ideal_gas && f.thermo_data === :nasa7 && f.reverse_rate === :explicit
    av = convenience_config(:adiabatic_constV)
    @test av.energy === :adiabatic && av.constraint === :constant_volume &&
          av.eos === :ideal_gas && av.thermo_data === :nasa7 && av.reverse_rate === :thermo_equilibrium
    ap = convenience_config(:adiabatic_constP)
    @test ap.energy === :adiabatic && ap.constraint === :constant_pressure &&
          ap.eos === :ideal_gas && ap.thermo_data === :nasa7 && ap.reverse_rate === :thermo_equilibrium
end

@testset "convenience_config: unknown mode errors" begin
    @test_throws ErrorException convenience_config(:bogus)
end

@testset "BatchReactor: mode=:kinetic runs (zero-point)" begin
    r = BatchReactor(_decay_mech(); mode=:kinetic)
    @test r.phase.config.constraint === :none          # zero-point
    @test r.phase.config.eos === :off
    sol = simulate(r, (0.0, 1.0); u0=Dict("A" => 3.0, "B" => 0.0), reltol=1e-10, abstol=1e-10)
    @test all(isfinite, sol.u[end])
end

@testset "BatchReactor: non-zero-point mode errors helpfully" begin
    # :fixedT needs EOS + NASA → not lowerable in Phase 2.
    @test_throws ErrorException BatchReactor(_decay_mech(); mode=:fixedT)
    @test_throws ErrorException BatchReactor(_decay_mech(); mode=:adiabatic_constV)
end

@testset "BatchReactor: mode overrides individual kwargs" begin
    r = BatchReactor(_decay_mech(); mode=:kinetic, constraint=:constant_pressure)
    @test r.phase.config.constraint === :none          # mode wins → zero-point
end

@testset "Phase 2 acceptance: Brusselator in BatchReactor (limit cycle)" begin
    r = BatchReactor(_brusselator_mech())
    sol = simulate(r, (0.0, 40.0); u0=Dict("X"=>1.0, "Y"=>0.5), reltol=1e-9, abstol=1e-9)
    xs = sol[_var(extract_system(r), "X")]
    @test minimum(xs) > 0 && maximum(xs) < 10                    # bounded limit cycle
    peaks = Float64[]
    for i in 2:length(xs)-1
        if xs[i] > xs[i-1] && xs[i] >= xs[i+1]; push!(peaks, sol.t[i]); end
    end
    periods = diff(peaks)
    @test length(periods) >= 2
    @test maximum(periods) - minimum(periods) < 0.1              # stable period
end

@testset "Phase 2 acceptance: dual-path (Mechanism vs Catalyst) agree via reactor" begin
    r_mech = BatchReactor(_brusselator_mech())
    rn = @reaction_network begin
        1.0, ∅ → X
        1.0, 2*X + Y → 3*X
        3.0, X → Y
        1.0, X → ∅
    end
    r_rn = BatchReactor(rn)
    u0 = Dict("X"=>1.0, "Y"=>0.5)
    sol_m = simulate(r_mech, (0.0, 10.0); u0=u0, reltol=1e-9, abstol=1e-9)
    sol_r = simulate(r_rn,  (0.0, 10.0); u0=u0, reltol=1e-9, abstol=1e-9)
    for sn in ("X", "Y")
        vm = _var(extract_system(r_mech), sn)
        vr = _var(extract_system(r_rn), sn)
        @test maximum(abs.(sol_m[vm] .- sol_r[vr])) < 1e-6      # same equations → same path
    end
end
