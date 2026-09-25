# Atmospheric box model: MCM alkanes/alkenes, isothermal at 298 K, fixed pressure.
#
#   julia --project=. examples/atmospheric/mcm_box.jl <frozen|diurnal> [span_days]
#
# One driver, two photolysis modes (span_days defaults to 8 for both):
#   frozen  — photolysis parameters at their defaults = J at overhead sun (perpetual day)
#   diurnal — the same parameters driven by the zenith clock (Cantera-method port):
#             J = l·cz^m·exp(−n/cz), piecewise-constant per 60-s step, solver NOT restarted
#
# Reads the 24 MB SOURCE mechanism directly — no preprocessor, no derived mechanism, no
# sidecar. The two MCM rate types the parser does not know are handled example-side via
# load_mechanism's rate_type_handlers registry (tools/mcm_rate_types.jl).
#
# Mode is :kinetic (the MechanismConfig() zero point) — the only correct choice for an
# atmospheric box model:
#   * T is given by the scenario (298 K), never solved from an energy balance;
#   * P is given (102858 Pa) and the state basis is concentration, so no EOS is needed;
#   * MCM emits only forward rates (`=>`) and its thermo data is uniformly zero, so a
#     reverse rate could not be formed anyway.
# The convenience modes (:fixedT, :adiabatic_constV, :adiabatic_constP) would each inject
# an energy equation and/or an EOS that this problem does not have.
#
# Scenario values are copied verbatim from the converter's mcm_alkanes_alkenes_repro_config
# so this run stays comparable to its Cantera reference.

using ChemMechSim
using ModelingToolkit
using ModelingToolkit: getname, parameters, setp
using OrdinaryDiffEq
using LinearSolve
using Printf

include(joinpath(@__DIR__, "tools", "mcm_rate_types.jl"))
include(joinpath(@__DIR__, "tools", "diurnal_env.jl"))   # zenith clock + J law (pure functions)

# ———————————————————— script body (skipped when included by anything else) ————————
if abspath(PROGRAM_FILE) == @__FILE__
    usage = "usage: julia --project=. examples/atmospheric/mcm_box.jl <frozen|diurnal> [span_days]"
    isempty(ARGS) && error(usage)
    const MODE = ARGS[1]
    MODE in ("frozen", "diurnal") || error(usage)
    const DAYS = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 8.0
    const T_END = DAYS * 86400.0

    const SRC     = joinpath(@__DIR__, "mcm_alkanes_alkenes_converted.yaml")
    const T0      = 298.0          # K
    const P0      = 102858.0       # Pa
    const R_GAS   = 8.314          # J/(mol·K)
    const DT_STEP = 60.0           # s — the converter's simulator step (diurnal mode)

    # Mole fractions and monitor list, from the reference config.
    const X_INIT = Dict("N2" => 0.78, "O2" => 0.21, "H2O" => 0.01,
                        "O3" => 3.0e-8, "NO2" => 1.0e-10, "CH4" => 1.8e-6)
    const MONITOR = ["O3", "NO", "NO2", "NO3", "OH", "HO2", "CH4"]

    isfile(SRC) || error("mcm_box: source mechanism not found at\n  $SRC\n" *
                         "It is not committed (24 MB). Copy it from the converter:\n" *
                         "  cp <kpp-cantera-converter>/examples/mcm/mcm_alkanes_alkenes_converted.yaml \\\n" *
                         "     examples/atmospheric/\nSee examples/atmospheric/README.md.")

    # --- load + shape invariants ------------------------------------------------------------
    # Type-based counts (the source is hand-copied and 24 MB, so a wrong or stale copy is
    # the likeliest fresh-clone failure — and it would otherwise fail SILENTLY: the box
    # would still run and the OH>0 check below would still pass). 1843 = 1842 + the
    # phantom `M` (the parser strips `M` from equations; the species list keeps it).
    println("loading ", basename(SRC), " (direct, rate_type_handlers) ...")
    t_parse = @elapsed mech = load_mechanism(SRC; rate_type_handlers = mcm_rate_handlers())
    @printf("  %d species, %d reactions  (%.1f s)\n",
            length(mech.species), length(mech.reactions), t_parse)
    length(mech.species)   == 1843 || error("mcm_box: expected 1843 species, got ",
                                            length(mech.species), " — wrong or stale source file?")
    length(mech.reactions) == 5600 || error("mcm_box: expected 5600 reactions, got ",
                                            length(mech.reactions))
    count(r -> r.kinetics isa ZenithPhotolysis, mech.reactions) == 1041 ||
        error("mcm_box: expected 1041 ZenithPhotolysis reactions")
    count(r -> r.kinetics isa SigmoidBranching, mech.reactions) == 2 ||
        error("mcm_box: expected 2 SigmoidBranching reactions")

    # --- lower -------------------------------------------------------------------------------
    # checks=false is REQUIRED here, not an optimisation: with checks=true, MTK's unit
    # validator cannot fold this mechanism's equations and lowering did not finish in
    # 16 minutes. The equations are dimensionally correct; the check cannot prove it.
    t_low = @elapsed r = BatchReactor(mech; mode = :kinetic, checks = false,
                                      name = Symbol("mcm_box_", MODE))
    @printf("  lowered in %.1f s  (peak %.2f GiB)\n", t_low, Sys.maxrss() / 2^30)
    sys = extract_system(r)
    ps = parameters(sys)
    state_index = Dict(String(getname(u)) => i
                       for (i, u) in enumerate(ModelingToolkit.unknowns(sys)))
    length(state_index) == length(mech.species) ||
        error("mcm_box: the state has $(length(state_index)) unknowns but the mechanism has " *
              "$(length(mech.species)) species — exports would be incomplete")

    # --- initial conditions --------------------------------------------------------------------
    # X_INIT is mole fractions; the state basis is concentration [mol/m^3], so
    # c = X·P/(R·T). EVERY species is listed explicitly — the 1836 not in X_INIT get
    # exactly 0.0. Passing a partial u0 is NOT safe here: observed doing so, the unlisted
    # species came back with arbitrary non-zero values and the integration blew up with
    # NaN on the first solve.
    const C_AIR = P0 / (R_GAS * T0)
    @printf("T = %.1f K, P = %.0f Pa, c_air = %.3f mol/m^3\n", T0, P0, C_AIR)
    u0 = Dict(String(sp.name) => get(X_INIT, String(sp.name), 0.0) * C_AIR
              for sp in mech.species)
    Tparam = ps[findfirst(p -> String(getname(p)) == "T", ps)]

    # --- photolysis parameter mapping + ORDER/VALUE GUARD (both modes) ------------------------
    # Resolve each k_{j}_A BY NAME (the sharded-Jacobian lesson: never assume parameter
    # ordering — commit 8277716 fixed a 20-50-orders-of-magnitude bug from exactly that).
    # j is the 1-based enumerate index over mech.reactions, which is the lowering's k_{j}
    # index by construction. The value guard closes the loop end-to-end: the default IS
    # J(cz=1) = l·exp(−n) (set by the handler), so if names, ordering or values drift,
    # this errors BEFORE the solve rather than producing a plausible-but-wrong run.
    pindex = Dict(String(getname(p)) => i for (i, p) in enumerate(ps))
    photos = [(j = j, kin = rx.kinetics) for (j, rx) in enumerate(mech.reactions)
              if rx.kinetics isa ZenithPhotolysis]
    photo_syms = Any[]
    for ph in photos
        name = "k_$(ph.j)_A"
        haskey(pindex, name) || error("mcm_box: no parameter $name — the parameter/reaction ",
                                      "index contract broke for reaction ", ph.j)
        sym = ps[pindex[name]]
        isapprox(ModelingToolkit.getdefault(sym), ph.kin.A; rtol = 1e-12) ||
            error("mcm_box: $name defaults to ", ModelingToolkit.getdefault(sym),
                  " but reaction $(ph.j)'s J(cz=1) is ", ph.kin.A,
                  " — parameter mapping is wrong; not solving")
        push!(photo_syms, sym)
    end
    length(unique(photo_syms)) == length(photo_syms) ||
        error("mcm_box: duplicate k_{j}_A mapping across ", length(photo_syms),
              " photolysis reactions")
    @printf("  all %d photolysis parameters verified (default = J at cz = 1)\n", length(photos))

    # Setters built from the BARE SYSTEM (SymbolicIndexingInterface.setp): the sharded-jac
    # path wraps the problem in a hand-built ODEProblem that carries no symbolic index,
    # but a ParameterIndex from `sys` applies to any target whose parameter buffer shares
    # the layout — the problem AND its integrators alike.
    KSETTERS = [setp(sys, sym) for sym in photo_syms]

    # J_NO2 = the NO2 photolysis row (the reference J for exports and cross-checks).
    id_no2 = only(sp.id for sp in mech.species if sp.name == "NO2")
    j_jno2 = only(ph.j for ph in photos if haskey(mech.reactions[ph.j].reactants, id_no2))
    jno2 = mech.reactions[j_jno2].kinetics

    # --- build + solve ---------------------------------------------------------------------------
    # Jacobian strategy is MODE-DEPENDENT, a measured same-session crossover (2026-09-24):
    # at the frozen mode's reltol 1e-6 the reaction-sharded analytic Jacobian wins big
    # (8-day solve 819 s FD -> 61 s analytic, 13.5x), but at the diurnal mode's reltol
    # 1e-4 with its 11520 forced 60-s ticks the analytic path LOSES ~5x on the same box
    # (solve 63 s FD vs 315 s analytic; the loose tolerance needs few Newton iterations,
    # so the cheap-to-form FD Jacobian amortizes better). Same-session ratios only —
    # never compare absolute seconds across sessions.
    # Optional third CLI arg ("jac=true" / "jac=false") overrides the default — a probe
    # hook for paired same-load A/B measurements (seconds on this shared box move 3-6x
    # BETWEEN runs; only back-to-back pairs are comparable).
    const USE_JAC = if length(ARGS) >= 3
        arg = ARGS[3]
        arg in ("jac=true", "jac=false") || error(usage * "\n  optional 3rd arg: jac=true|jac=false")
        arg == "jac=true"
    else
        MODE == "frozen"
    end
    @printf("building problem (jac=%s) ...\n", USE_JAC)
    t_build = @elapsed prob = build_problem(r, u0, (0.0, T_END);
                                            params = [Tparam => T0], jac = USE_JAC)
    @printf("  built in %.1f s\n", t_build)

    local sol, t_solve
    if MODE == "diurnal"
        # Initial J's at t = 0 (MIDNIGHT: cz = cos(89.5°)). Pre-solve setter writes on the
        # problem DO reach the solve (mini-verified) — and the mod-grid callback below
        # does NOT fire at t = 0 (discrete callbacks fire at tstops; only a literally-true
        # condition is evaluated during initialization), so without this block the first
        # 60 s would run at the NOON defaults.
        cz0 = cos_zenith(0.0)
        for i in eachindex(photos)
            KSETTERS[i](prob, photolysis_J(photos[i].kin.l, photos[i].kin.m,
                                           photos[i].kin.n, cz0))
        end

        # Condition `iszero(mod(t, DT_STEP))`: discrete callbacks with a literally-TRUE
        # condition fire after EVERY step AND during initialization (mini-verified: 143
        # fires on a 2-s toy problem); the mod test restricts firing to exactly the 60-s
        # grid. save_positions=(false,false): a parameter change is not a state event.
        const N_PHOTO = length(photos)
        apply_photolysis!(integ) = begin
            cz = cos_zenith(integ.t)
            @inbounds for i in 1:N_PHOTO
                kin = photos[i].kin
                KSETTERS[i](integ, photolysis_J(kin.l, kin.m, kin.n, cz))
            end
            return nothing
        end
        cb = DiscreteCallback((u, t, integ) -> iszero(mod(t, DT_STEP)), apply_photolysis!;
                              save_positions = (false, false))
        tstops = DT_STEP:DT_STEP:T_END

        # TOLERANCE POLICY — the trade-off actually taken, not the ideal one. Night-time
        # trace species sit BELOW the flat abstol (OH's night trough ~7e-17 mol/m^3,
        # NO3's peak ~7e-18, both vs abstol 1e-12): the solver may return anything up to
        # ~1e-12 for them, so their night values (and all of NO3) are NOT resolved and
        # fig3 floors them at the tolerance line as bounds. Resolving them needs a
        # per-state abstol ~1e-20..1e-22 — measured: 1e-22 ran >94 min of solve without
        # finishing and was abandoned. reltol is loosened to 1e-4 (the frozen mode uses
        # 1e-6); the majors still match a 1e-6 run to 4-5 significant digits — but do NOT
        # quote fine percentages off this run without re-running tighter.
        @printf("solving %.1f days, diurnal (dt_step = %.0f s, reltol 1e-4, abstol 1e-12) ...\n",
                DAYS, DT_STEP)
        t_solve = @elapsed sol = solve(prob, FBDF(autodiff = false);
                                       reltol = 1e-4, abstol = 1e-12, saveat = 1200.0,
                                       callback = cb, tstops = tstops)
    else
        # Dense LU for the Newton matrix: this Jacobian is 76% dense (2.58M nnz of 3.4M
        # possible), and FBDF's DEFAULT sparse linsolve path pays a per-linear-solve
        # dropzeros COPY of W — measured 69 GiB/day of pure allocation. Dense LU measured
        # 3.81 s/day warm at 2.70 GiB (26x less); Sparspak allocates least (0.92 GiB) but
        # sparse fill-in at this density loses to BLAS 9x (37 s/day). Paired probe 2026-09-25.
        @printf("solving %.1f days, frozen (perpetual noon, reltol 1e-6, abstol 1e-12, dense LU) ...\n",
                DAYS)
        t_solve = @elapsed sol = solve(prob, FBDF(autodiff = false,
                                                  linsolve = LinearSolve.LUFactorization());
                                       reltol = 1e-6, abstol = 1e-12, saveat = 1200.0)
    end
    @printf("  solved in %.1f s, retcode = %s\n", t_solve, sol.retcode)

    # --- exports: output/<mode>/, uniform schema so the figures read one way ------------------
    OUTDIR = joinpath(@__DIR__, "output", MODE)
    mkpath(OUTDIR)
    OUT_CSV = joinpath(OUTDIR, "series.csv")
    open(OUT_CSV, "w") do io
        println(io, join(vcat("time_s", "cz", "J_NO2", MONITOR), ","))
        for (k, t) in enumerate(sol.t)
            # 1200 s is a whole multiple of the 60-s tick, so the formula evaluated AT a
            # save time IS the applied piecewise value. Frozen: cz = 1 — the perpetual
            # noon the parameters default to — and J_NO2 = that default.
            cz = MODE == "diurnal" ? cos_zenith(t) : 1.0
            row = Any[t, cz, photolysis_J(jno2.l, jno2.m, jno2.n, cz)]
            for name in MONITOR
                push!(row, sol.u[k][state_index[name]])
            end
            println(io, join(row, ","))
        end
    end
    println("wrote ", OUT_CSV, "  (", length(sol.t), " rows)")

    OUT_STATE = joinpath(OUTDIR, "final_state.csv")
    open(OUT_STATE, "w") do io
        println(io, "species,concentration_mol_m3")
        for sp in mech.species
            println(io, String(sp.name), ",", sol.u[end][state_index[String(sp.name)]])
        end
    end
    println("wrote ", OUT_STATE, "  (", length(mech.species), " species)")

    open(joinpath(OUTDIR, "run_meta.txt"), "w") do io
        println(io, "mode=", MODE)
        MODE == "diurnal" && println(io, "dt_step_s=", DT_STEP)
        println(io, "reltol=", MODE == "diurnal" ? "1e-4" : "1e-6")
        println(io, "abstol=1e-12")     # trace species below this are UNRESOLVED (diurnal)
        println(io, "span_days=", DAYS)
        println(io, "t_lower_s=", round(t_low, digits = 1))
        # build+solve COMBINED (diurnal's solve includes the callback ticks) — NOT
        # comparable to bench_jac.csv's split measurements.
        println(io, "t_simulate_s=", round(t_build + t_solve, digits = 1))
        println(io, "jac=", USE_JAC)
        # Process-LIFETIME peak (Sys.maxrss high-water mark), dominated by lowering+codegen.
        println(io, "peak_rss_gib=", round(Sys.maxrss() / 2^30, digits = 2))
    end

    # --- report + chemistry checks ------------------------------------------------------------
    # Read the trajectory straight out of sol.u. The DE solution's `sol[i, j]` indexes
    # (component, timestep) — the opposite order from the intuitive reading — so reading
    # sol.u avoids the trap.
    series(name) = [u[state_index[name]] for u in sol.u]
    println("\nspecies        initial [mol/m^3]      final [mol/m^3]")
    for name in MONITOR
        v = series(name)
        @printf("  %-8s %18.6e %18.6e\n", name, v[1], v[end])
    end

    # The 1041 photolysis reactions are the radical source. Without them the box would
    # sit at its initial zeros forever — assert the chemistry actually ran.
    oh = series("OH")
    maximum(oh) > 0.0 ||
        error("mcm_box ($MODE): OH stayed at zero. Frozen: the photolysis defaults are " *
              "noon values — check the shape guards above. Diurnal: the callback is not " *
              "wired (check apply_photolysis! and the parameter guard).")
    @printf("\nOH peak = %.3e mol/m^3  (radical source is active)\n", maximum(oh))

    if MODE == "diurnal"
        # Soft report: NO3 at this tolerance is an upper bound, and this scenario is
        # NOx-starved (a single 0.1-ppb NO2 pulse, HNO3 terminal) — no night accumulation
        # is expected. Night test is GEOMETRIC, never a cz threshold: the clock's 89.5°
        # clamp pins cz at 0.0087 all night, so any threshold lies.
        no3 = series("NO3")
        no3_at = sol.t[argmax(no3)]
        tod = no3_at % 86400.0
        @printf("NO3 peak = %.3e mol/m^3 (%.2f molec/cm^3) at t = %.2f d [%s]\n",
                maximum(no3), maximum(no3) * 6.02214076e23 / 1e6, no3_at / 86400,
                (tod <= 21600.0 || tod >= 64800.0) ? "night ✓" : "DAY — unexpected")
    end
end
