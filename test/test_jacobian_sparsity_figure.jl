using Test

@testset "jacobian sparsity figure example supports all data sources" begin
    script = joinpath(@__DIR__, "..", "examples", "perf", "jacobian_sparsity_figure.jl")
    include(script)

    cfg = JacobianSparsityFigure.parse_args([
        "--mechs", "h2o2",
        "--sources", "sharded,mtk,bench",
        "--no-plot",
    ])
    @test cfg[:mechs] == ["h2o2"]
    @test cfg[:sources] == ["sharded", "mtk", "bench"]

    tmp = mktempdir()
    bench_csv = joinpath(tmp, "bench_matrix.csv")
    open(bench_csv, "w") do io
        println(io, "mech,n_states,nnz_jac,density_pct,linsolve,run_idx,wall_s,alloc_bytes,steps,retcode,T_end")
        println(io, "h2o2,9,17,21.0,klu,1,0.1,0,1,Success,1000.0")
    end

    out_dir = joinpath(tmp, "out")
    JacobianSparsityFigure.main([
        "--mechs", "h2o2",
        "--sources", "bench",
        "--no-plot",
        "--out-dir", out_dir,
        "--bench-csv", bench_csv,
    ])

    summary = read(joinpath(out_dir, "jacobian_sparsity_summary.csv"), String)
    @test occursin("bench,h2o2", summary)
    @test occursin("summary only", summary)

    points = read(joinpath(out_dir, "jacobian_sparsity_points.csv"), String)
    @test !occursin("bench,h2o2", points)
    @test !isfile(joinpath(out_dir, "fig04_jacobian_sparsity.png"))
end
