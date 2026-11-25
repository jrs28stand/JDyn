# Figures: the data behind each panel against what Dynare 7.2's drawing code draws. Shock decompositions are compared with the fill()/plot() calls of
# graph_decomp.m recorded under Octave with stub figure functions (pydyn's sd_nk_case1_drawn.mat, made by r_shock_decomposition.m); IRFs, OccBin figures,
# estimation, forecast and rplot figures against the arrays and the layout / naming rules of the Dynare sources.
using Test, JDyn, MAT, LinearAlgebra, Random, Logging
isdefined(@__MODULE__, :REF) || include(joinpath(@__DIR__, "common.jl"))
using JDyn: Estimation, Figure, Panel, Curve

quiet(f) = with_logger(f, NullLogger())

@testset "figure model" begin
    f = Figure("fig1", "A title", 1, 2, [Panel("a", [Curve(1:3, [1.0, 2, 3]; label="x"), Curve(1:3, [0.5, 1, 1.5]; kind=:dashed)]),
                                         Panel("b", [Curve(1:3, [-1.0, 1, 0]; kind=:bar, label="u"), Curve(1:3, [1.0, 1, 1]; kind=:bar, label="v"),
                                                     Curve(1:3, [0.0, 0, 0]; kind=:band, y2=[1.0, 1, 1])])])
    svg = JDyn.svg_figure(f)
    @test startswith(svg, "<svg") && occursin("A title", svg) && occursin("<polyline", svg) && occursin("<rect", svg) && occursin("<polygon", svg)
    d = mktempdir()
    @test save_figures(f, d) == [joinpath(d, "fig1.svg")] && isfile(joinpath(d, "fig1.svg"))
    @test isempty(save_figures(f, d; nograph=true))
    @test isempty(save_figures(f, d; format="none"))
    @test_throws ArgumentError save_figures(f, d; format=["svg", "pdf"])        # the built-in writer: SVG only
    @test_throws ArgumentError save_figures(f, d; format="png", backend=:plots) # Plots not loaded
    t = figure_table(f)
    @test length(t) == 15 && t[1].figure == "fig1" && t[1].panel == "a"
    @test [p.title for p in arrange_figures("s", "T", [Panel(string(i), Curve[]) for i in 1:11])[1].panels] == string.(1:9)
    figs = arrange_figures("s", "T", [Panel(string(i), Curve[]) for i in 1:11])
    @test [f.name for f in figs] == ["s1", "s2"] && figs[2].title == "T figure 2" && (figs[2].nrows, figs[2].ncols) == (2, 1)
end

# ------------------------------------------------------------------ shock decomposition: Dynare's drawn bars and lines
const _PD = mktempdir()
for f in ("sd_nk.mod", "est_nk_data.m")
    cp(joinpath(REF, f), joinpath(_PD, f))
end
@testset "shock decomposition figures match graph_decomp.m" begin
    m = quiet(() -> JDyn.Model(JDyn.run_preprocessor(joinpath(_PD, "sd_nk.mod"); extra_args=["-DCASE=1"])))
    est = Estimation(m)
    r = matread(joinpath(REF, "sd_nk_case1.mat"))
    res = quiet(() -> JDyn.smooth(m; estimation=est, theta=vec(r["theta_mode"]), parameter_set="posterior_mode"))
    dec = shock_decomposition(m, res)
    icd = initial_condition_decomposition(m, res)
    G = matread(joinpath(REF, "sd_nk_case1_drawn.mat"))
    function dynare_bars(g, k)
        b = g["f$(k)_p0_bars"]
        ncomp = Int(maximum(b[:, 1]))
        T = size(b, 1) ÷ ncomp
        B = permutedims(reshape(permutedims(b), 5, ncomp, T), (2, 3, 1))     # component, period, field
        B
    end
    worst = Ref(0.0)
    function check(which, d::DecompositionPlotData, labels; kw...)
        g = G["drawn_" * which]
        figs = decomposition_figures(d, m; kw...)
        @test length(figs) == Int(g["nfig"])
        names = split(g["names"], "|")
        for (k, fig) in enumerate(figs)
            p = fig.panels[1]
            bars = filter(c -> c.kind == :bar, p.curves)
            line = only(filter(c -> c.kind == :line, p.curves))
            B = dynare_bars(g, k)
            @test [c.label for c in bars] == labels
            dyn = split(g["f$(k)_p0_labels"], "|")[2:end]
            # for the initial-condition decomposition the remainder is the effect of the shocks, which Dynare's figure calls "Initial values"
            @test dyn == reverse(which == "icd" ? vcat(labels[1:end-1], "Initial values") : labels)
            H = reduce(vcat, (c.y' for c in bars))
            x = bars[1].x
            @test x ≈ vec((B[1, :, 2] .+ B[1, :, 3]) ./ 2)
            bottom, top = JDyn.stack_bars(H)
            e = max(maximum(abs, H .- (B[:, :, 5] .- B[:, :, 4])), maximum(abs, bottom .- B[:, :, 4]))
            worst[] = max(worst[], e)
            @test e < 1e-12
            @test line.y ≈ vec(g["f$(k)_p0_line"]) atol = 1e-12
            @test line.x == vec(g["f$(k)_p0_line_x"])
            @test startswith(fig.title, split(strip(names[k]), ":")[1])
            @test endswith(fig.title, ": " * p.title * " (Posterior Mode).")
            @test [l[1] for l in fig.legend] == labels
        end
        figs
    end
    crop = (plot_init_date=11, plot_end_date=40)
    dd(; kw...) = decomposition_plot_data(dec, ["pi", "robs"]; model=m, crop..., kw...)
    figs = check("grp", dd(use_shock_groups="grp"), ["supply", "demand and policy", "Initial values"])
    @test figs[1].name == "sd_nk_shock_decomposition_pi_group_grp_posterior_mode"
    check("plain", dd(), ["ea", "eu", "er", "Initial values"])
    check("partial", dd(use_shock_groups="partial"), ["policy", "Others + Initial Values"])
    check("ss", dd(steadystate=true, flip=true), ["ea", "eu", "er", "Initial values"])
    check("full", decomposition_plot_data(dec, ["pi", "robs"]; model=m), ["ea", "eu", "er", "Initial values"])
    check("icd", decomposition_plot_data(icd, ["robs"]; model=m, crop...), vcat(m.endo_names, ["Shocks"]))
    @info "shock decomposition figures: worst difference to Dynare's drawn bars $(worst[])"
    # files: one SVG per variable in <fname>/graphs; nograph writes nothing
    d = joinpath(_PD, "out")
    figs, files = plot_shock_decomposition(m, dec, ["pi", "robs"]; outdir=d, crop...)
    @test length(files) == 2 && all(isfile, files)
    @test isempty(plot_shock_decomposition(m, dec, ["pi"]; outdir=d, nograph=true, crop...)[2])
end

# ------------------------------------------------------------------ IRFs at orders 1-3, saved as Dynare names them
const EX1 = "/root/jl/src/Dynare.jl/test/models/example1/example1.mod"
@testset "IRF figures at orders 1, 2, 3" begin
    d = mktempdir()
    m = build(replace(read(EX1, String), "var e, u = 0.009*0.009;" => "var e, u = 0;"); name="example1", output="third")
    figs1 = to_figure(irf_figures(m))
    @test [f.name for f in figs1] == ["example1_IRF_e", "example1_IRF_u"]
    @test all(f -> f.title in ("Orthogonalized shock to e", "Orthogonalized shock to u"), figs1)
    @test [p.title for p in figs1[1].panels] == ["y", "c", "k", "a", "h", "b"]
    files = plot_irfs(m; outdir=joinpath(d, "out"), format="svg").files
    @test basename.(files) == ["example1_IRF_e.svg", "example1_IRF_u.svg"] && all(isfile, files)
    @test isempty(plot_irfs(m; outdir=joinpath(d, "o2"), nograph=true).files) && !isdir(joinpath(d, "o2"))
    @test isempty(plot_irfs(m; outdir=joinpath(d, "o3"), format="none").files)
    for ord in (2, 3)
        kw = (order=ord, pruning=true, replic=20, drop=50, periods=12)
        r = irf_figures(m; kw..., rng=Random.MersenneTwister(1))
        @test r.order == ord && all(f -> size(f.series, 1) == 12, r.figures)
        dr = JDyn.k_order(m; order=ord, pruning=true)
        L = JDyn._lower_cholesky(m.Sigma_e)
        y = JDyn.irf(dr, L[:, 1]; Sigma_e=m.Sigma_e, periods=12, order=ord, pruning=true, drop=50, replic=20, rng=Random.MersenneTwister(1))
        f = r.figures[1]
        @test f.series ≈ y[:, [m.ie[v] for v in f.variables]]
        @test to_figure(f).panels[2].curves[1].y == f.series[:, 2]
    end
end

# ------------------------------------------------------------------ OccBin
@testset "OccBin IRF and graph figures" begin
    m = load_copy(joinpath(REF, "occ_borrcon.mod"))
    ob = Occbin(m)
    ir = irf_occbin(ob; periods=30, shocksize=0.3, exo_names=["u"], shocksigns=("pos", "neg"), check_ahead_periods=12, maxit=25)
    figs = occbin_irf_figures(ir, m; variables=ir.variables[1:4])
    @test [f.name for f in figs] == ["$(m.name)_irf_occbin_u_pos_1", "$(m.name)_irf_occbin_u_neg_1"]
    @test figs[1].title == "OccBin IRFs to positive u shock " && (figs[1].nrows, figs[1].ncols) == (3, 3)
    p = figs[1].panels[1]
    r = ir[("u", "pos")]
    z(x) = replace(v -> abs(v) < 1e-10 ? 0.0 : v, x)
    @test p.title == ir.variables[1] && p.curves[1].y == z(r.linear[:, 1]) && p.curves[2].y == z(r.piecewise[:, 1])
    @test p.curves[1].kind == :line && p.curves[2].kind == :dashed
    res = simulate(ob, Matrix{Float64}(refmat("occ_borrcon.mat")["S"]); periods=80)
    g = occbin_graph_figures(res, m; variables=res.endo_names[1:3])
    @test length(g) == 1 && g[1].name == "$(m.name)_occbin_1" && g[1].title == "OccBin simulated paths, figure 1"
    @test [q.title for q in g[1].panels] == res.endo_names[1:3]
    @test g[1].panels[2].curves[1].y == res.piecewise[:, 2] || g[1].panels[2].curves[1].y == res.piecewise[:, 2] .- res.ys[2]
end

# ------------------------------------------------------------------ estimation
const _EST = mktempdir()
for f in readdir(joinpath(@__DIR__, "ref_oo_estimation"); join=true)
    cp(f, joinpath(_EST, basename(f)))
end
@testset "estimation figures" begin
    m = quiet(() -> JDyn.Model(JDyn.run_preprocessor(joinpath(_EST, "est_small.mod"); extra_args=["-DVARIANT=1"])))
    est = Estimation(m)
    names = [dynare_parameter_name(est, k) for k in 1:est.n]
    @test all(n -> startswith(n, "SE_") || startswith(n, "CC_") || !occursin(" ", n), names)
    # priors: Dynare's 200-point grids between the 0.1% and 99.9% quantiles; densities integrate to ~1 over them
    for k in 1:est.n
        x, f = prior_density(est.prior, k)
        @test length(x) == 200 && issorted(x)
        ok = .!isnan.(f)
        est.prior.shape[k] == 5 || @test 0.9 < sum(0.5 .* (f[ok][1:end-1] .+ f[ok][2:end]) .* diff(x[ok])) < 1.01
    end
    pf = prior_figures(est)
    @test pf[1].name == "est_small_Priors1" && pf[1].title == "Priors" && length(pf[1].panels) == min(9, est.n)
    @test [p.title for p in pf[1].panels] == names[1:min(9, est.n)]
    # mode check around the starting values
    mf = mode_check_figures(est, est.x0; number_of_points=6)
    @test mf[1].name == "est_small_CheckPlots1" && mf[1].title == "Mode check plots"
    cd_ = mode_check_data(est, est.x0; number_of_points=6)
    k = 1
    z = cd_[k].z
    @test issorted(z) && any(≈(est.x0[k]), z)
    i0 = findfirst(≈(est.x0[k]), z)
    @test cd_[k].logpost[i0] ≈ -JDyn._dsge_likelihood(est, est.x0)[1]
    @test cd_[k].loglik[i0] ≈ cd_[k].logpost[i0]      # y2 = y1 + lnprior(z) - lnprior(x): equal at the point itself
end

@testset "posterior, trace, convergence, bayesian IRF, series, smoother figures" begin
    m = quiet(() -> JDyn.Model(JDyn.run_preprocessor(joinpath(_EST, "est_small.mod"); extra_args=["-DVARIANT=1"])))
    est = Estimation(m)
    n = est.n
    rng = Random.MersenneTwister(3)
    T = 2500
    draws = cumsum(0.01 .* randn(rng, 2, T, n); dims=2) .+ reshape(est.x0, 1, 1, :)
    lp = cumsum(0.01 .* randn(rng, 2, T); dims=2)
    mc = JDyn.MCMCResult(draws, lp, [0.3, 0.3], est.names, repeat(est.x0', 2), 0.5, trues(2, T))
    pp = prior_posterior_figures(est, mc; mode=est.x0)
    @test pp[1].name == "est_small_PriorsAndPosteriors1" && (pp[1].nrows, pp[1].ncols) == (3, 3)
    @test length(pp[1].panels[1].curves) == 3 && pp[1].panels[1].curves[2].x |> length == 512
    tf = trace_figures(est, mc; chains=[1], ma=10)
    @test tf[1].name == "TracePlot_Posterior_blck_1" && tf[1].title == "Trace plot for the posterior density (block number 1)."
    @test length(tf) == n + 1 && tf[2].name == "TracePlot_$(names_ = dynare_parameter_name(est, 1))_blck_1"
    c = tf[2].panels[1].curves
    @test c[1].y == draws[1, :, 1] && c[2].y[11] ≈ sum(draws[1, 1:21, 1]) / 21 && isnan(c[2].y[10])
    tf2 = trace_figures(est, mc)
    @test tf2[2].name == "TracePlot_$(dynare_parameter_name(est, 1))_blck_1_2" && length(tf2[2].panels[1].curves) == 2
    bg = brooks_gelman_figures(est, mc; plotrows=3)
    @test !isempty(bg) && bg[end].name == "est_small_mdiag" && length(bg[end].panels) == 3 && (bg[end].nrows, bg[end].ncols) == (3, 1)
    @test bg[1].name == "est_small_udiag1" && bg[1].panels[1].title == dynare_parameter_name(est, 1) * " (Interval)"
    # no diagnostics with <= 2000 draws
    short = JDyn.MCMCResult(draws[:, 1:500, :], lp[:, 1:500], [0.3, 0.3], est.names, repeat(est.x0', 2), 0.5, trues(2, 500))
    @test isempty(brooks_gelman_figures(est, short))
    # Bayesian IRFs from replayed parameter draws
    ir = JDyn.posterior_irf(est; params=repeat(est.x0', 5), periods=10)
    bi = bayesian_irf_figures(m, ir)
    @test !isempty(bi) && occursin("_Bayesian_IRF_", bi[1].name) && bi[1].title == "Orthogonalized shock to " * ir.shocks[1]
    @test all(p -> any(c -> c.label == "mean", p.curves), bi[1].panels)
    # the posterior series: deciles and mean
    x = randn(rng, 40, 12, 3)
    sd = JDyn.series_distribution(["a", "b", "c"], x, 0.9; density=false)
    sf = posterior_series_figures(m, sd, "SmoothedVariables")
    @test length(sf) == 1 && sf[1].name == "est_small_SmoothedVariables_c" && length(sf[1].panels[1].curves) == 10 && sf[1].panels[1].curves[3].y == sd.deciles[3, :, 1] && all(c -> length(c.y) == 12, sf[1].panels[1].curves)
    # smoother figures
    res = quiet(() -> JDyn.smooth(m; estimation=est, theta=est.x0))
    sm = smoother_figures(m, res, est.rawdata, est.varobs)
    @test sm[1].name == "est_small_SmoothedShocks1" && sm[1].panels[1].curves[2].y == res.SmoothedShocks[m.exo_names[1]]
    h = filter(f -> startswith(f.name, "est_small_HistoricalAndSmoothedVariables"), sm)[1]
    @test h.panels[1].curves[2].kind == :dashed && h.panels[1].curves[2].y == est.rawdata[:, 1]
end

# ------------------------------------------------------------------ forecasts and rplot
@testset "forecast, conditional forecast and rplot figures" begin
    mk(v) = JDyn.ForecastResult(["a", "b"], Dict("a" => v, "b" => 2v), Dict("a" => v .- 1, "b" => 2v .- 1), Dict("a" => v .+ 1, "b" => 2v .+ 1),
                                Dict{String,Vector{Float64}}(), Dict{String,Vector{Float64}}(), length(v), 0.9, 1)
    m = (name="mm",)
    ff = forecast_figures(m, mk(collect(1.0:6)))
    @test length(ff) == 1 && ff[1].name == "forcst1" && ff[1].title == "Forecasts (I)" && (ff[1].nrows, ff[1].ncols) == (3, 4)
    @test [length(p.curves) for p in ff[1].panels] == [3, 3] && ff[1].panels[2].curves[1].y == 2 .* collect(1.0:6)
    r = JDyn.ForecastResult(["v$i" for i in 1:14], Dict("v$i" => ones(3) for i in 1:14), Dict("v$i" => zeros(3) for i in 1:14), Dict("v$i" => 2ones(3) for i in 1:14),
                            Dict{String,Vector{Float64}}(), Dict{String,Vector{Float64}}(), 3, 0.9, 1)
    ff = forecast_figures(m, r)
    @test [f.name for f in ff] == ["forcst1", "forcst2"] && ff[2].title == "Forecasts (2)" && length(ff[1].panels) == 12 && length(ff[2].panels) == 2
    cf = JDyn.ConditionalForecastResult(["k"], Dict("k" => [1.0, 2, 3, 4]), Dict("k" => [0.0 1 2 3; 2 3 4 5]), Dict("k" => [1.0, 1.5, 2, 2.5]),
                                        Dict("k" => [0.5 0.5 1 1; 1.5 2 3 4]), ["e"], Dict{String,Vector{Float64}}(), Dict{String,Matrix{Float64}}(), [1], ["k"],
                                        "my title", "mm", 3, 10, 0.9, 0.0)
    cff = conditional_forecast_figures(m, cf)
    @test cff[1].name == "Conditional_forecast_my_title_k" && cff[1].title == "Conditional forecast (my title): k."
    cv = cff[1].panels[1].curves
    @test cv[1].kind == :band && cv[1].y == [0.0, 1, 2, 3] && cv[1].y2 == [2.0, 3, 4, 5] && cv[2].y == [1.0, 2, 3, 4] && cv[3].y == [1.0, 1.5, 2, 2.5]
    @test length(conditional_forecast_figures(m, cf; periods=1)[1].panels[1].curves[2].y) == 2
    # rplot: types 0, 1, 2
    mm = (name="mm", ie=Dict("a" => 1, "b" => 2), ix=Dict("e" => 1))
    E = [1.0 2 3 4 5; 5 4 3 2 1]
    X = reshape([0.0, 0.1, 0.2, 0.3, 0.4], :, 1)
    r0 = rplot_figures(mm, E, X, ["a", "b"])
    @test length(r0) == 1 && r0[1].name == "SimulatedTrajectory_a" && r0[1].panels[1].title == "Plot of a b" && r0[1].panels[1].curves[1].x == [0.0, 1, 2, 3, 4]
    @test length(rplot_figures(mm, E, X, ["a", "b", "e"]; rplottype=1)) == 3
    r2 = rplot_figures(mm, E, X, ["a", "b"]; rplottype=2, steady_state=[10.0, 20.0])
    @test (r2[1].nrows, r2[1].ncols) == (1, 2) && r2[1].panels[2].curves[2].y == fill(20.0, 5) && r2[1].panels[2].curves[2].kind == :dotted
    @test rplot_figures(mm, E, X, ["a"]; smpl=[1, 3])[1].panels[1].curves[1].y == [2.0, 3, 4]
    @test_throws ArgumentError rplot_figures(mm, E, X, ["zzz"])
end

# ------------------------------------------------------------------ statements write figures (dynare(file; plots=true))
@testset "dynare(...; plots=true) writes <fname>/graphs" begin
    d = mktempdir()
    txt = replace(read(EX1, String), "var e, u = 0.009*0.009;" => "var e, u = 0;")
    write(joinpath(d, "example1.mod"), txt)
    ws = quiet(() -> dynare(joinpath(d, "example1.mod"); plots=true))
    files = ws.state[:figure_files]
    @test basename.(files) == ["example1_IRF_e.svg", "example1_IRF_u.svg"] && all(isfile, files)
    @test dirname(files[1]) == joinpath(d, "example1", "graphs")
    d2 = mktempdir()                                    # plots off by default; nograph respected
    write(joinpath(d2, "example1.mod"), txt)
    quiet(() -> dynare(joinpath(d2, "example1.mod")))
    @test !isdir(joinpath(d2, "example1", "graphs"))
    d3 = mktempdir()
    write(joinpath(d3, "example1.mod"), replace(txt, r"stoch_simul\(" => "stoch_simul(nograph, "))
    quiet(() -> dynare(joinpath(d3, "example1.mod"); plots=true))
    @test !isdir(joinpath(d3, "example1", "graphs"))
end

# ------------------------------------------------------------------ PNG / PDF / EPS through the Plots.jl extension
# Plots is a weak dependency: it is loaded from a separate environment (JDYN_PLOTS_ENV, default test/plotsenv; instantiate it once with `julia --project=test/plotsenv -e "using Pkg; Pkg.instantiate()"`, any environment with Plots installed) stacked on LOAD_PATH.
const _PLOTS_ENV = get(ENV, "JDYN_PLOTS_ENV", joinpath(@__DIR__, "plotsenv"))
@testset "Plots.jl extension: png, pdf, eps (all curve kinds, grids, legends)" begin
    if !isfile(joinpath(_PLOTS_ENV, "Project.toml"))
        @test_skip false
    else
        ENV["GKSwstype"] = "100"
        _PLOTS_ENV in LOAD_PATH || push!(LOAD_PATH, _PLOTS_ENV)
        @eval Main using Plots
        f = Figure("kinds", "All kinds", 1, 2,
                   [Panel("a", [Curve(1:4, [1.0, 2, 3, 2]; label="x"), Curve(1:4, [0.5, 1, 1.5, 1]; kind=:dashed, label="d"), Curve(1:4, [0.2, 0.3, 0.4, 0.5]; kind=:dotted),
                                Curve(1:4, [0.0, 0, 0, 0]; kind=:band, y2=[1.0, 1, 1, 1], label="band"), Curve([2.0]; label="vl")]),
                    Panel("b", [Curve(1:4, [-1.0, 1, 2, 0]; kind=:bar, label="u"), Curve(1:4, [1.0, 1, 1, 1]; kind=:bar, label="v"),
                                Curve(1:4, [0.2, 0.3, 0.4, 0.5]; kind=:scatter, label="sc")]; zeroline=true)])
        grid = Figure("grid34", "3x4", 3, 4, [Panel("p$i", [Curve(1:5, cumsum(randn(MersenneTwister(i), 5)); label="s")]) for i in 1:12])
        d = mktempdir()
        files = save_figures([f, grid], d; format=["png", "pdf", "eps", "svg"], backend=:plots)
        @test length(files) == 8 && all(isfile, files) && all(x -> filesize(x) > 1000, files)
        @test read(joinpath(d, "kinds.png"), 4) == UInt8[0x89, 0x50, 0x4e, 0x47]
        @test String(read(joinpath(d, "kinds.pdf"), 4)) == "%PDF"
        @test startswith(read(joinpath(d, "kinds.eps"), String), "%!PS")
        # real figures: IRFs and a shock decomposition
        m = build(replace(read(EX1, String), "var e, u = 0.009*0.009;" => "var e, u = 0;"); name="example1")
        r = plot_irfs(m; outdir=joinpath(d, "irf"), format=["png", "pdf"], backend=:plots)
        @test length(r.files) == 4 && all(isfile, r.files)
        @test isempty(plot_irfs(m; outdir=joinpath(d, "irf2"), format="png", backend=:plots, nograph=true).files)
    end
end
