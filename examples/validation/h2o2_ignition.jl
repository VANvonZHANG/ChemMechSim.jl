# Phase 5a example: H2-O2 ignition — ChemMechSim vs Cantera (if ref CSV present) + CairoMakie plot.
# Run: julia --project=. examples/validation/h2o2_ignition.jl
# Requires examples/validation/h2o2_ref_{constV,constP}.csv for Cantera compare (run the .py first).
using ChemMechSim
using OrdinaryDiffEq: Rodas5P
using ModelingToolkit: unknowns, getname
using DelimitedFiles
using CairoMakie

const YAML_PATH = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml")
const YAML_FALLBACK = joinpath(@__DIR__, "..", "..", "test", "data", "h2o2.yaml")
const yaml = isfile(YAML_PATH) ? YAML_PATH : YAML_FALLBACK
const REF_DIR = joinpath(@__DIR__, "output")
mkpath(REF_DIR)
const PNG_OUT = joinpath(REF_DIR, "h2o2_ignition.png")

_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]

const KEY_SPECIES = ["H2", "O2", "H2O", "OH"]
const SPECIES_COLORS = Dict("H2"=>:steelblue, "O2"=>:crimson, "H2O"=>:forestgreen, "OH"=>:darkorange)

"Build u0 for H2:O2:N2=2:1:4 at T,P. prefix=\"\" (const-V, concentration) or \"n_\" (const-P, moles)."
function _u0(mech, prefix, T0=1000.0, P0=101325.0, V0=1.0)
    R = 8.314; c_tot = P0 / (R * T0)
    u0 = Dict{String,Float64}()
    for sp in mech.species
        X = sp.name == "H2" ? 2/7 : sp.name == "O2" ? 1/7 : sp.name == "N2" ? 4/7 : 0.0
        u0[prefix * sp.name] = X * c_tot * V0
    end
    u0["T"] = T0
    return u0
end

"Time of maximum |dT/dt| (ignition delay)."
function _t_ignition(sol, sys, n=2001)
    ts = range(0, stop=sol.t[end], length=n)
    Ts = [Float64(sol(t; idxs=_var(sys, "T"))) for t in ts]
    dTdt = diff(Ts) ./ diff(ts)
    return ts[argmax(abs.(dTdt)) + 1]
end

mech = load_mechanism(yaml)
println("Loaded mechanism: $(length(mech.species)) species, $(length(mech.reactions)) reactions")

fig = Figure(size=(1000, 700))
Label(fig[1, 1:2], "H2-O2 ignition (h2o2.yaml, H2:O2:N2=2:1:4, T₀=1000 K, P=1 atm)";
      fontsize=14, font=:bold, tellwidth=false)

for (col, (mode, prefix, ref_csv_name)) in enumerate([
        (:adiabatic_constV, "",  "h2o2_ref_constV.csv"),
        (:adiabatic_constP, "n_", "h2o2_ref_constP.csv")])
    reactor = BatchReactor(mech; mode=mode)
    sys = extract_system(reactor)
    sol = simulate(reactor, (0.0, 1e-3); u0=_u0(mech, prefix),
                   solver=Rodas5P(), reltol=1e-8, abstol=1e-12)
    t_ign = _t_ignition(sol, sys)
    T_max = Float64(sol(1e-3; idxs=_var(sys, "T")))
    println("\n[$mode] retcode=$(sol.retcode), t_ignition=$(round(t_ign*1e6, digits=2)) μs, T_max=$(round(T_max, digits=1)) K")

    ref_csv = joinpath(REF_DIR, ref_csv_name)
    has_ref = isfile(ref_csv)
    ct_t, ct_T = Float64[], Float64[]
    if has_ref
        data, _ = readdlm(ref_csv, ',', header=true)
        ct_t = vec(data[:, 1]); ct_T = vec(data[:, 2])
        t_ign_ct = ct_t[argmax(abs.(diff(ct_T) ./ diff(ct_t))) + 1]
        rel = abs(t_ign - t_ign_ct) / t_ign_ct
        tol = mode == :adiabatic_constV ? 0.05 : 0.08
        println("  Cantera t_ignition=$(round(t_ign_ct*1e6, digits=2)) μs, rel diff=$(round(rel*100, digits=2))% (tol $(tol*100)%)")
        println("  ", rel < tol ? "PASS" : "FAIL")
    else
        println("  (no Cantera ref at $ref_csv — run examples/validation/h2o2_ignition.py first)")
    end

    # trajectories for plotting
    ts = range(0, 1e-3; length=200)
    Ts_julia = [Float64(sol(t; idxs=_var(sys, "T"))) for t in ts]
    # all-species totals (for mole-fraction normalization: X_i = val_i / Σ val_j)
    totals = [sum(Float64(sol(t; idxs=_var(sys, prefix * sp.name))) for sp in mech.species) for t in ts]
    Xs = Dict(name => [Float64(sol(t; idxs=_var(sys, prefix * name))) for t in ts] ./ totals
              for name in KEY_SPECIES)

    # Panel row 2: T(t) vs Cantera
    ax_T = Axis(fig[2, col]; xlabel="time (μs)", ylabel="T (K)",
                title=string(mode) * (has_ref ? " — T(t) vs Cantera" : " — T(t)"))
    lines!(ax_T, ts .* 1e6, Ts_julia; label="ChemMechSim", linewidth=2)
    if has_ref
        lines!(ax_T, ct_t .* 1e6, ct_T; linestyle=:dash, label="Cantera", linewidth=2)
    end
    axislegend(ax_T; position=:lt)

    # Panel row 3: key-species mole fractions
    ax_X = Axis(fig[3, col]; xlabel="time (μs)", ylabel="mole fraction",
                title=string(mode) * " — key species")
    for name in KEY_SPECIES
        lines!(ax_X, ts .* 1e6, Xs[name]; label=name, color=SPECIES_COLORS[name], linewidth=1.5)
    end
    axislegend(ax_X; position=:rt)
end

save(PNG_OUT, fig)
println("\nSaved plot: $PNG_OUT")
