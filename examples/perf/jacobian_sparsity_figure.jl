#!/usr/bin/env julia
# Fig. 7 Jacobian nonzero-pattern workflow.
#
# Run from the package root:
#   julia --project=. examples/perf/jacobian_sparsity_figure.jl
#   julia --project=. examples/perf/jacobian_sparsity_figure.jl --mechs gri30 --sources sharded,mtk
#   julia --project=. examples/perf/jacobian_sparsity_figure.jl --sources sharded,bench
#
# Outputs under examples/perf/output/ by default:
#   fig07_jacobian_sparsity.{svg,pdf,png}
#   jacobian_sparsity_summary.csv
#   jacobian_sparsity_points.csv

module JacobianSparsityFigure

using ChemMechSim
using ModelingToolkit
using SparseArrays
using Printf

const HERE = @__DIR__
const DEFAULT_OUT_DIR = joinpath(HERE, "output")
const DEFAULT_BENCH_CSV = joinpath(DEFAULT_OUT_DIR, "bench_matrix.csv")

struct MechSpec
    name::String
    yaml::String
end

const MECHS = Dict(
    "gri30" => MechSpec("gri30", joinpath(HERE, "..", "mechanism", "gri30.yaml")),
    "aramco" => MechSpec("aramco", joinpath(HERE, "..", "mechanism", "AramcoMech3.0.yaml")),
    "ffcm2" => MechSpec("ffcm2", joinpath(HERE, "..", "mechanism", "FFCM2.yaml")),
    "h2o2" => MechSpec("h2o2", joinpath(HERE, "..", "mechanism", "h2o2.yaml")),
)

struct PatternData
    source::String
    mech::String
    n_states::Int
    n_nonzeros::Int
    density_pct::Float64
    rows::Vector{Int}
    cols::Vector{Int}
end

struct SummaryRow
    source::String
    mech::String
    n_species::Int
    n_reactions::Int
    n_states::Int
    n_nonzeros::Int
    density_pct::Float64
    build_s::Float64
    note::String
end

function _usage()
    return """
    Usage:
      julia --project=. examples/perf/jacobian_sparsity_figure.jl [options]

    Options:
      --mechs LIST          Comma list: gri30,aramco,ffcm2,h2o2. Default: gri30,aramco
      --sources LIST        Comma list: sharded,mtk,bench. Default: sharded
      --out-dir DIR         Output directory. Default: examples/perf/output
      --bench-csv FILE      bench_matrix.csv path for source=bench
      --no-plot             Write CSVs only
      --allow-large-mtk     Permit source=mtk on aramco/ffcm2
      --help                Show this message
    """
end

function _split_list(s::String)
    items = String[strip(lowercase(String(x))) for x in split(s, ",")]
    return filter(!isempty, items)
end

function parse_args(args::Vector{String})
    cfg = Dict{Symbol,Any}(
        :mechs => ["gri30", "aramco"],
        :sources => ["sharded"],
        :out_dir => DEFAULT_OUT_DIR,
        :bench_csv => DEFAULT_BENCH_CSV,
        :plot => true,
        :allow_large_mtk => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--help" || a == "-h"
            println(_usage())
            return nothing
        elseif a == "--no-plot"
            cfg[:plot] = false
        elseif a == "--allow-large-mtk"
            cfg[:allow_large_mtk] = true
        elseif a == "--mechs"
            i += 1; i <= length(args) || error("missing value for --mechs")
            cfg[:mechs] = _split_list(args[i])
        elseif a == "--sources" || a == "--source"
            i += 1; i <= length(args) || error("missing value for $a")
            cfg[:sources] = _split_list(args[i])
        elseif a == "--out-dir"
            i += 1; i <= length(args) || error("missing value for --out-dir")
            cfg[:out_dir] = args[i]
        elseif a == "--bench-csv"
            i += 1; i <= length(args) || error("missing value for --bench-csv")
            cfg[:bench_csv] = args[i]
        else
            error("unknown argument: $a")
        end
        i += 1
    end

    for mech in cfg[:mechs]
        haskey(MECHS, mech) || error("unknown mechanism '$mech'; available: $(join(sort(collect(keys(MECHS))), ","))")
    end
    for source in cfg[:sources]
        source in ("sharded", "mtk", "bench") ||
            error("unknown source '$source'; available: sharded,mtk,bench")
    end
    return cfg
end

function _sparse_points(J::SparseMatrixCSC)
    rows = Int[]
    cols = Int[]
    sizehint!(rows, nnz(J))
    sizehint!(cols, nnz(J))
    for col in 1:size(J, 2)
        for p in J.colptr[col]:(J.colptr[col + 1] - 1)
            push!(rows, J.rowval[p])
            push!(cols, col)
        end
    end
    return rows, cols
end

function _summary(source, spec::MechSpec, mech, n_states, n_nonzeros, build_s, note)
    density_pct = n_states == 0 ? NaN : 100 * n_nonzeros / n_states^2
    return SummaryRow(source, spec.name, length(mech.species), length(mech.reactions),
                      n_states, n_nonzeros, density_pct, build_s, note)
end

function _pattern(source, spec::MechSpec, J::SparseMatrixCSC)
    rows, cols = _sparse_points(J)
    n = size(J, 1)
    nz = nnz(J)
    return PatternData(source, spec.name, n, nz, 100 * nz / n^2, rows, cols)
end

function _system_for(mech)
    config = convenience_config(:adiabatic_constV)
    phase = ChemPhaseSystem(mech; config=config, checks=false)
    return config, extract_system(phase)
end

function _from_sharded(spec::MechSpec)
    mech = load_mechanism(spec.yaml)
    local J
    elapsed = @elapsed begin
        config, sys = _system_for(mech)
        _, J, _stats = ChemMechSim.build_reaction_sharded_jac(
            mech; config=config, checks=false, sys=sys, return_stats=true)
    end
    return _pattern("sharded", spec, J),
           _summary("sharded", spec, mech, size(J, 1), nnz(J), elapsed,
                    "reaction-sharded sparsity template; no full symbolic Jacobian expansion")
end

function _from_mtk(spec::MechSpec; allow_large::Bool=false)
    mech = load_mechanism(spec.yaml)
    if spec.name in ("aramco", "ffcm2") && !allow_large
        return nothing,
               _summary("mtk", spec, mech, 0, 0, NaN,
                        "skipped; pass --allow-large-mtk to calculate sparse MTK Jacobian")
    end
    local J
    elapsed = @elapsed begin
        _config, sys = _system_for(mech)
        J = ModelingToolkit.calculate_jacobian(sys; sparse=true)
    end
    J_sparse = J isa SparseMatrixCSC ? J : sparse(J)
    return _pattern("mtk", spec, J_sparse),
           _summary("mtk", spec, mech, size(J_sparse, 1), nnz(J_sparse), elapsed,
                    "ModelingToolkit.calculate_jacobian(sys; sparse=true)")
end

function _csv_value(row::Dict{String,String}, key::String, default="")
    return get(row, key, default)
end

function _read_simple_csv(path::String)
    isfile(path) || error("bench CSV not found: $path")
    lines = readlines(path)
    isempty(lines) && return Dict{String,String}[]
    header = split(lines[1], ",")
    rows = Dict{String,String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ",")
        row = Dict{String,String}()
        for (i, h) in pairs(header)
            row[h] = i <= length(vals) ? vals[i] : ""
        end
        push!(rows, row)
    end
    return rows
end

function _bench_rows(path::String, selected_mechs::Vector{String})
    rows = _read_simple_csv(path)
    selected = Set(selected_mechs)
    seen = Set{String}()
    out = SummaryRow[]
    for row in rows
        mech_name = lowercase(_csv_value(row, "mech"))
        mech_name in selected || continue
        mech_name in seen && continue
        push!(seen, mech_name)
        n_states = parse(Int, _csv_value(row, "n_states", "0"))
        n_nonzeros = parse(Int, _csv_value(row, "nnz_jac", "0"))
        density = parse(Float64, _csv_value(row, "density_pct", "NaN"))
        push!(out, SummaryRow("bench", mech_name, 0, 0, n_states, n_nonzeros,
                              density, NaN,
                              "bench_matrix summary only; no row-col nonzero coordinates"))
    end
    return out
end

function _fmt(x::Float64; digits::Int=3)
    return isfinite(x) ? @sprintf("%.*f", digits, x) : ""
end

function _write_summary(path::String, rows::Vector{SummaryRow})
    open(path, "w") do io
        println(io, "source,mech,n_species,n_reactions,n_states,nnz,density_pct,build_s,note")
        for r in rows
            println(io, join(Any[
                r.source, r.mech, r.n_species, r.n_reactions, r.n_states, r.n_nonzeros,
                _fmt(r.density_pct), _fmt(r.build_s), r.note,
            ], ","))
        end
    end
end

function _write_points(path::String, patterns::Vector{PatternData})
    open(path, "w") do io
        println(io, "source,mech,row,col")
        for p in patterns
            for i in eachindex(p.rows)
                println(io, "$(p.source),$(p.mech),$(p.rows[i]),$(p.cols[i])")
            end
        end
    end
end

function _plot_patterns(patterns::Vector{PatternData}, out_dir::String)
    isempty(patterns) && return
    @eval using CairoMakie
    return Base.invokelatest(_plot_patterns_loaded, patterns, out_dir)
end

function _plot_patterns_loaded(patterns::Vector{PatternData}, out_dir::String)
    fig = CairoMakie.Figure(size=(360 * length(patterns), 330), backgroundcolor=:white)
    for (i, p) in enumerate(patterns)
        ax = CairoMakie.Axis(
            fig[1, i];
            title="$(uppercase(p.mech)): $(p.n_states) states, nnz=$(p.n_nonzeros), density=$(_fmt(p.density_pct; digits=1))%",
            xlabel="state index",
            ylabel=i == 1 ? "state index" : "",
            aspect=CairoMakie.DataAspect(),
        )
        marker_size = p.n_states <= 100 ? 2.4 : 0.55
        CairoMakie.scatter!(ax, p.cols, p.rows; marker=:rect, markersize=marker_size,
                            color="#0B1F3A")
        CairoMakie.xlims!(ax, 0.5, p.n_states + 0.5)
        CairoMakie.ylims!(ax, p.n_states + 0.5, 0.5)
    end

    for ext in ("svg", "pdf", "png")
        path = joinpath(out_dir, "fig07_jacobian_sparsity.$ext")
        CairoMakie.save(path, fig; dpi=300)
        println("wrote $path")
    end
    return nothing
end

function main(args::Vector{String}=ARGS)
    cfg = parse_args(args)
    cfg === nothing && return

    out_dir = cfg[:out_dir]
    mkpath(out_dir)
    patterns = PatternData[]
    summaries = SummaryRow[]

    for source in cfg[:sources]
        if source == "bench"
            append!(summaries, _bench_rows(cfg[:bench_csv], cfg[:mechs]))
            continue
        end
        for mech_name in cfg[:mechs]
            spec = MECHS[mech_name]
            if source == "sharded"
                pattern, summary = _from_sharded(spec)
            elseif source == "mtk"
                pattern, summary = _from_mtk(spec; allow_large=cfg[:allow_large_mtk])
            else
                error("unhandled source: $source")
            end
            pattern === nothing || push!(patterns, pattern)
            push!(summaries, summary)
        end
    end

    summary_path = joinpath(out_dir, "jacobian_sparsity_summary.csv")
    points_path = joinpath(out_dir, "jacobian_sparsity_points.csv")
    _write_summary(summary_path, summaries)
    _write_points(points_path, patterns)
    println("wrote $summary_path")
    println("wrote $points_path")

    cfg[:plot] && _plot_patterns(patterns, out_dir)
    return (patterns=patterns, summaries=summaries)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    JacobianSparsityFigure.main(ARGS)
end
