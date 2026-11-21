include("common.jl")

const EX1 = "/root/jl/src/Dynare.jl/test/models/example1/example1.mod"
# example1 as shipped has perfectly correlated shocks (covariance = variance); ex1() makes them independent
const EX1TXT = read(EX1, String)
ex1(txt=EX1TXT) = build(replace(txt, "var e, u = 0.009*0.009;" => "var e, u = 0;"); name="example1")
const AG = joinpath(SS, "aguiar_gopinath_2007_trend.mod")

@testset "pltorg" begin
    # Dynare's rule: (nbplt, nr, nc, lr, lc, nstar)
    @test pltorg(0)[1] == 0
    for (n, rc) in zip(1:9, [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 2), (3, 3), (3, 3), (3, 3)])
        @test pltorg(n) == (1, rc..., 0, 0, 9)
    end
    @test pltorg(10) == (2, 3, 3, 1, 1, 9)
    @test pltorg(14) == (2, 3, 3, 3, 2, 9)
    @test pltorg(18) == (2, 3, 3, 3, 3, 9)
    @test pltorg(19) == (3, 3, 3, 1, 1, 9)
end

@testset "example1 as shipped: perfectly correlated shocks" begin
    # Dynare's Cholesky factor then has a ~1e-7 diagonal entry for u, below its 5e-7 cut: no figure for u
    m = load_copy(EX1)
    res = irf_figures(m)
    @test [f.shock for f in res.figures] == ["e"] && res.skipped_shocks == ["u"]
end

@testset "example1: content" begin
    m = ex1()
    sol = first_order(m)
    res = irf_figures(m, sol)
    @test res.periods == 40 && res.order == 1 && res.threshold == 1e-10
    @test [f.shock for f in res.figures] == ["e", "u"]
    @test all(f -> (f.nrows, f.ncols) == (3, 2) && f.nfigs == 1, res.figures)
    @test res.figures[1].variables == ["y", "c", "k", "a", "h", "b"]   # no var list: all, in declaration order
    @test res.figures[1].title == "Orthogonalized shock to e"
    @test res.figures[1].name == "example1_IRF_e"
    # a one-standard-deviation shock: Sigma_e = diag(0.009^2) plus covariance 0.009^2 between e and u
    L = JDyn._lower_cholesky(m.Sigma_e)
    @test L * L' ≈ m.Sigma_e atol=1e-12
    for (i, f) in enumerate(res.figures)
        @test f.series ≈ irf(sol, L[:, i]; periods=40)[:, [m.ie[v] for v in f.variables]]
    end
    @test size(res.figures[1].series) == (40, 6)
    @test res.irfs["a_e"] == res.figures[1].series[:, 4]
    # the impact response of the shocked variable at t = 1
    @test res.figures[1].series[1, 4] ≈ L[1, 1]
end

@testset "example1: selection rules" begin
    m = ex1()
    sol = first_order(m)
    # threshold: with a huge threshold nothing is drawn but everything is still computed
    r = irf_figures(m, sol; threshold=1e3)
    @test isempty(r.figures) && length(r.skipped) == 12 && length(r.irfs) == 12
    # a threshold between the responses of different variables drops only the small ones
    big = maximum(abs, irf_figures(m, sol).figures[1].series[:, 4])
    r = irf_figures(m, sol; threshold=0.5 * big, shocks=["e"])
    @test r.figures[1].variables ⊆ ["y", "c", "k", "a", "h", "b"]
    @test all(maximum(abs, r.figures[1].series[:, j]) >= 0.5 * big for j in 1:size(r.figures[1].series, 2))
    @test all(p -> p[2] == "e", r.skipped) && !isempty(r.skipped)
    # variable list, shocks, periods
    r = irf_figures(m, sol; var_list=["c", "y"], shocks=["u"], periods=12)
    @test length(r.figures) == 1 && r.figures[1].shock == "u" && r.figures[1].variables == ["c", "y"]
    @test size(r.figures[1].series, 1) == 12 && (r.figures[1].nrows, r.figures[1].ncols) == (2, 1)
    @test_throws ArgumentError irf_figures(m, sol; shocks=["nope"])
    # a shock with zero standard deviation is skipped as in Dynare
    set_shock_stderr!(m, "u", 0.0)
    r = irf_figures(m, first_order(m))
    @test [f.shock for f in r.figures] == ["e"] && r.skipped_shocks == ["u"]
    # relative_irf: percent of a one-standard-deviation shock
    m = ex1()
    r = irf_figures(m, first_order(m); relative=true, shocks=["e"])
    @test r.figures[1].title == "Relative response to orthogonalized shock to e"
    @test r.figures[1].series[1, 4] ≈ 100.0
end

@testset "tidy export" begin
    m = ex1()
    res = irf_figures(m; var_list=["y", "c"], periods=5)
    tab = irf_table(res)
    @test length(tab) == 2 * 2 * 5
    @test keys(first(tab)) == (:shock, :variable, :period, :value, :order)
    @test first(tab).shock == "e" && first(tab).variable == "y" && first(tab).period == 1 && first(tab).order == 1
    @test [r.value for r in tab if r.shock == "e" && r.variable == "c"] == res.figures[1].series[:, 2]
    f = tempname() * ".csv"
    write_csv(f, tab)
    lines = readlines(f)
    @test lines[1] == "shock,variable,period,value,order" && length(lines) == 1 + length(tab)
    back = [split(l, ",") for l in lines[2:end]]
    @test parse.(Float64, getindex.(back, 4)) == [r.value for r in tab]       # exact round trip
    @test getindex.(back, 3)[1:5] == string.(1:5)
    @test length(irf_table(irf_figures(m; periods=5, threshold=1e3); all=true)) == 2 * 6 * 5
    @test isempty(irf_table(irf_figures(m; periods=5, threshold=1e3)))
end

@testset "files, nograph, SVG" begin
    m = ex1()
    d = mktempdir()
    res = plot_irfs(m; outdir=d, periods=20)
    @test sort(basename.(res.files)) == ["example1_IRF_e.svg", "example1_IRF_u.svg"]
    svg = read(res.files[1], String)
    @test startswith(svg, "<svg") && endswith(strip(svg), "</svg>")
    @test count("<polyline", svg) == 6 && occursin("Orthogonalized shock to e", svg)
    @test occursin(">y</text>", svg) && occursin(">h</text>", svg)
    @test_throws ArgumentError plot_irfs(m; outdir=d, format="png")          # needs Plots
    @test_throws ArgumentError plot_irfs(m; outdir=d, backend=:plots)        # extension not loaded here
    # default folder is <model folder>/<name>/graphs, as Dynare's <dname>/graphs
    res = plot_irfs(m; periods=5, shocks=["e"])
    @test dirname(res.files[1]) == joinpath(m.mod_dir, "example1", "graphs")
    # nograph from the .mod file: content returned, no files
    txt = replace(EX1TXT, "stoch_simul(dr=cycle_reduction, order=1);" => "stoch_simul(dr=cycle_reduction, order=1, nograph, irf=15, irf_plot_threshold=0.005, irf_shocks=(e));")
    m2 = ex1(txt)
    @test get(m2.options["stoch_simul"], "nograph", false) in (true, 1)
    d2 = mktempdir()
    r2 = plot_irfs(m2; outdir=d2)
    @test isempty(r2.files) && isempty(readdir(d2)) && r2.periods == 15 && r2.threshold == 0.005
    @test [f.shock for f in r2.figures] == ["e"]                           # irf_shocks = (e)
    r3 = plot_irfs(m2; outdir=d2, nograph=false)
    @test length(r3.files) == length(r3.figures) == 1 && !isempty(readdir(d2))
end

@testset "aguiar_gopinath_2007: var list, several figures" begin
    isfile(AG) || return
    m = load_copy(AG)
    sol = first_order(m)
    res = irf_figures(m, sol)                         # stoch_simul(order=1) tb_y c_y i_y
    @test all(f -> f.variables ⊆ ["tb_y", "c_y", "i_y"] && f.nrows * f.ncols >= length(f.variables), res.figures)
    @test sort(unique(f.shock for f in res.figures)) ⊆ ["eps_g", "eps_z"]
    names = JDyn.orig_endo_names(m)                  # the default variable list without a list in stoch_simul
    n = length(names)
    @test n > 9
    nbplt, nr, nc, lr, lc, nstar = pltorg(n)
    alln = irf_figures(m, sol; var_list=names, threshold=0.0)
    for s in ("eps_z", "eps_g")
        fs = filter(f -> f.shock == s, alln.figures)
        @test length(fs) == nbplt && nbplt >= 2
        @test [length(f.variables) for f in fs] == [fill(nstar, nbplt - 1); n - (nbplt - 1) * nstar]
        @test [(f.nrows, f.ncols) for f in fs] == [fill((nr, nc), nbplt - 1); (lr, lc)]
        @test [f.name for f in fs] == ["aguiar_gopinath_2007_trend_IRF_$s$k" for k in 1:nbplt]
        @test fs[1].title == "Orthogonalized shock to $s figure 1"
        @test reduce(vcat, [f.variables for f in fs]) == names                  # order kept
    end
end

# the Plots.jl extension is exercised only when Plots is installed in the test environment
if Base.find_package("Plots") !== nothing
    @eval using Plots
    @testset "Plots extension" begin
        m = ex1()
        d = mktempdir()
        res = withenv("GKSwstype" => "100") do
            plot_irfs(m; outdir=d, backend=:plots, format="png", shocks=["e"], periods=10)
        end
        @test isfile(res.files[1]) && filesize(res.files[1]) > 1000
    end
end
