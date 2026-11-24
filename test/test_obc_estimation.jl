# obc_method / obc_horizon / obc_selection in estimation, the news solvers of the inversion filter (src/obc_filter.jl; port of pydyn's
# tests/test_obc_estimation.py and tests/test_obc_selection.py; docs/OBC_ESTIMATION.md). References: test/ref_occbin (Dynare 7.2 under Octave: the NKM
# example's stored likelihoods), Dynare's tests/estimation/slice/bf.mod (a constraint that switches two equations; data occest_bf_data.mat =
# Dynare's datafile.mat) and JDyn's own OccBin solver.
using Test; include("common.jl")
using Random

const OREF = joinpath(@__DIR__, "ref_occbin")
const EDIR = joinpath(@__DIR__, "obc", "est")
oref(name) = matread(joinpath(OREF, name * ".mat"))
mxd(a, b) = maximum(abs.(a .- b))
const NKM = load_copy(joinpath(OREF, "occest_nkm.mod"))
const NKMS = load_copy(joinpath(OREF, "occest_nkms.mod"))
const IVF = Dict("likelihood_inversion_filter" => true)

odata(r) = (d = r["data"]; d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d))
opoints(r, n) = (X = r["X"]; X isa AbstractVector ? reshape(X, n, :) : (size(X, 1) == n ? X : Matrix(X')))
function mkest(m, data; occbin=IVF, kw...)
    with_logger(NullLogger()) do
        JDyn.Estimation(m; data=data, nobs=size(data, 1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=occbin, kw...)
    end
end
"The model with extra lines in front of its text (native `options_.occbin.obc_*` lines)."
function with_lines(file, lines; name="variant.mod")
    d = mktempdir()
    for f in readdir(dirname(file))
        p = joinpath(dirname(file), f)
        isfile(p) && cp(p, joinpath(d, f); force=true)
    end
    write(joinpath(d, name), lines * read(file, String))
    with_logger(NullLogger()) do; JDyn.load(joinpath(d, name)); end
end

@testset "obc_* options of the estimation" begin
    r = oref("occest_nkms_lik")
    data = odata(r)
    X = opoints(r, 3)
    e = mkest(NKMS, data)
    s = JDyn.obc_settings(e)
    @test (s.method, s.horizon, s.selection, s.milp_time_limit) == (:occbin, 40, :occbin, 0.0)
    JDyn.set_obc!(e; method="news")
    @test JDyn.obc_settings(e).method == :news
    for bad in ("holden", "", 3)
        @test_throws ArgumentError JDyn.set_obc!(e; method=bad)
        @test_throws ArgumentError mkest(NKMS, data; obc_method=bad)
    end
    @test_throws ArgumentError JDyn.set_obc!(e; selection="median")
    for bad in (0, -3, 2.5, "x")
        @test_throws ArgumentError JDyn.set_obc!(e; horizon=bad)
    end
    e2 = mkest(NKMS, data; obc_method="news", obc_horizon=25)
    @test (JDyn.obc_settings(e2).method, JDyn.obc_settings(e2).horizon) == (:news, 25)

    # minnorm in the estimation is news with min_wedge (pydyn's alias): constructor, assignment, native mod line, live override
    em = mkest(NKMS, data; obc_method="minnorm")
    @test (JDyn.obc_settings(em).method, JDyn.obc_settings(em).selection) == (:news, :min_wedge)
    em2 = mkest(NKMS, data); JDyn.set_obc!(em2; method="minnorm")
    @test (JDyn.obc_settings(em2).method, JDyn.obc_settings(em2).selection) == (:news, :min_wedge)
    ref = mkest(NKMS, data; obc_method="news", obc_selection="min_wedge")
    @test JDyn.log_posterior(em, X[:, 1]) == JDyn.log_posterior(ref, X[:, 1])
    @test_throws ArgumentError mkest(NKMS, data; obc_method="minnorm", obc_selection="longest")
    mn = with_lines(joinpath(OREF, "occest_nkms.mod"), "options_.occbin.obc_method = 'minnorm';\n")
    e3 = mkest(mn, data)
    @test (JDyn.obc_settings(e3).method, JDyn.obc_settings(e3).selection) == (:news, :min_wedge)
    @test JDyn.log_posterior(e3, X[:, 1]) == JDyn.log_posterior(ref, X[:, 1])
    @test_throws EstimationError mkest(mn, data; occbin=nothing)                 # the piecewise Kalman filter has no news version

    # native lines, keyword wins, live occbin= override
    nat = with_lines(joinpath(OREF, "occest_nkms.mod"), "options_.occbin.obc_method = 'news';\noptions_.occbin.obc_horizon = 30;\noptions_.occbin.obc_selection = 'shortest';\n")
    e4 = mkest(nat, data)
    @test (JDyn.obc_settings(e4).method, JDyn.obc_settings(e4).horizon, JDyn.obc_settings(e4).selection) == (:news, 30, :shortest)
    e5 = mkest(nat, data; obc_method="occbin", obc_horizon=20, obc_selection="occbin")
    @test (JDyn.obc_settings(e5).method, JDyn.obc_settings(e5).horizon) == (:occbin, 20)
    e6 = mkest(NKMS, data; occbin=Dict("likelihood_inversion_filter" => true, "obc_method" => "news", "obc_horizon" => 33))
    @test (JDyn.obc_settings(e6).method, JDyn.obc_settings(e6).horizon) == (:news, 33)
    @test_throws ArgumentError with_lines(joinpath(OREF, "occest_nkms.mod"), "options_.occbin.obc_method = 'holden';\n") |> m -> mkest(m, data)
    @test_throws EstimationError mkest(NKMS, data; occbin=nothing, obc_method="news")

    # the piecewise Kalman filter refuses news everywhere
    ep = mkest(NKMS, data; occbin=nothing)
    @test_throws EstimationError JDyn.set_obc!(ep; method="news")
    JDyn.set_obc!(ep; method="occbin")                                           # still allowed
    ei = mkest(NKMS, data; obc_method="news")
    ei.options["occbin"]["likelihood.inversion_filter"] = false                   # defeat the set-up check: the filter itself refuses
    @test_throws EstimationError JDyn.occbin_filter(ei, X[:, 1])
    # selection needs news
    es = mkest(NKMS, data; obc_selection="longest")
    @test_throws EstimationError JDyn.log_posterior(es, X[:, 1])

    # defaults unchanged: bit for bit equal to an explicit occbin, and Dynare's value
    a, b = mkest(NKMS, data), mkest(NKMS, data; obc_method="occbin")
    for k in 1:3
        @test JDyn.log_posterior(a, X[:, k]) === JDyn.log_posterior(b, X[:, k])
        @test isapprox(JDyn.log_posterior(a, X[:, k]), -vec(r["ivf_fval"])[k]; rtol=1e-12, atol=1e-8)
    end
    @test JDyn.obc_settings(a).last_solver === nothing
    # a model without constraints refuses the options
    let m = load_copy(joinpath(@__DIR__, "optim", "est_opt_nk.mod"))
        @test_throws EstimationError JDyn.Estimation(m; mode_compute=0, mh_replic=0, allow_unsupported=true, obc_method="news")
    end
end

@testset "inversion filter with the news solver: NKM example" begin
    for (m, case, ks, tol) in ((NKMS, "nkms", 1:3, 1e-9), (NKM, "nkm", 1:4, 1e-9))
        r = oref("occest_$(case)_lik")
        data = odata(r)
        e0, e1 = mkest(m, data), mkest(m, data; obc_method="news")
        X = opoints(r, e0.n)
        for k in ks
            a, b = JDyn.log_posterior(e0, X[:, k]), JDyn.log_posterior(e1, X[:, k])
            @test JDyn.obc_settings(e1).last_solver isa JDyn.LCPFilterSolver
            @test abs(a - b) < tol
            @test isapprox(b, -vec(r["ivf_fval"])[k]; rtol=1e-11, atol=1e-8)
            ra, rb = JDyn.occbin_filter(e0, X[:, k]), JDyn.occbin_filter(e1, X[:, k])
            @test ra.regime_history == rb.regime_history
            @test mxd(ra.filtered_errs, rb.filtered_errs) < 1e-9
            @test mxd(ra.stateval, rb.stateval) < 1e-9
            @test count(reg -> reg[1][1][1] == 1, rb.regime_history) > 5
            println("  news vs OccBin, $case, vector $k: |dlogpost| = ", abs(a - b))
        end
    end
    # fail fast: the 4th stored NKM vector has one trial shock that only an integer programme could decide
    r = oref("occest_nkm_lik")
    e = mkest(NKM, odata(r); obc_method="news")
    X = opoints(r, e.n)
    t = @elapsed lp = JDyn.log_posterior(e, X[:, 4])
    @test isapprox(lp, -vec(r["ivf_fval"])[4]; rtol=1e-11, atol=1e-8)
    st = JDyn.obc_settings(e).last_solver.stats
    @test (st.fallbacks, st.proved_infeasible, st.skipped, st.timeouts, st.rescued) == (2, 1, 1, 0, 0) && st.errors == Dict(313 => 2)
    @test t < 5.0
    # the integer programme can be switched on (time limited); the likelihood does not change
    e2 = mkest(NKM, odata(r); obc_method="news", obc_milp_time_limit=2.0)
    @test abs(JDyn.log_posterior(e2, X[:, 4]) - lp) < 1e-9
end

@testset "regime filter solver: bf.mod and bf_ppf.mod (two equations switch)" begin
    dd = matread(joinpath(EDIR, "occest_bf_data.mat"))
    data = hcat(Float64.(vec(dd["q"])), Float64.(vec(dd["r"])))
    for file in ("bf.mod", "bf_ppf.mod")
        m = load_copy(joinpath(EDIR, file))
        e0, e1 = mkest(m, data), mkest(m, data; obc_method="news")
        rng = MersenneTwister(1)
        x0 = copy(e0.x0)
        nfail, nbind = 0, 0
        for k in 0:7
            th = k == 0 ? x0 : x0 .* exp.(0.05 .* randn(rng, 5))
            a, b = JDyn.log_posterior(e0, th), JDyn.log_posterior(e1, th)
            @test JDyn.obc_settings(e1).last_solver isa JDyn.RegimeFilterSolver
            if !isfinite(a)
                nfail += 1
                @test !isfinite(b) && e0.last_info_code == e1.last_info_code
                continue
            end
            @test abs(a - b) < 1e-9
            ra, rb = JDyn.occbin_filter(e0, th), JDyn.occbin_filter(e1, th)
            @test ra.regime_history == rb.regime_history
            @test mxd(ra.filtered_errs, rb.filtered_errs) < 1e-12
            @test mxd(ra.stateval, rb.stateval) < 1e-12
            nbind += count(reg -> reg[1][1][1] == 1, rb.regime_history)
        end
        @test nbind >= 5 && nfail <= 2
        # the first period's rule (T, R, C) of the regime solver equals OccBin's, in binding periods
        ob = JDyn.occbin_at(e1, x0)
        sv = JDyn.news_filter_solver(ob)
        sim = JDyn._occ_sim(JDyn.OCCBIN_FILTER_DEFAULTS, "likelihood")
        rng = MersenneTwister(2)
        sd = sqrt.(diag(Matrix(m.Sigma_e)))
        nb = 0
        for _ in 1:40
            init = 0.05 .* randn(rng, ob.n)
            shock = randn(rng, ob.n_exo) .* sd .* 2
            f1, r1 = JDyn.filter_solve(sv, reshape(shock, 1, :), init, nothing)
            f2, r2 = JDyn._occ_solve(ob, reshape(shock, 1, :), init, sim)
            @test f1 == f2 == 0
            @test r1.regime_history[1] == r2.regime_history[1]
            nb += r1.regime_history[1][1][1][1]
            @test all(mxd(u, v) < 1e-11 for (u, v) in zip(r1.ss[1], r2.ss[1]))
            @test mxd(r1.piecewise, r2.piecewise[1:1, :]) < 1e-11
        end
        @test nb >= 5
    end
    # equilibrium selection is not implemented for the regime formulation: refused, not ignored
    m = load_copy(joinpath(EDIR, "bf.mod"))
    @test_throws ArgumentError JDyn.news_filter_solver(JDyn.Occbin(m); selection=:longest)
end

@testset "obc_selection" begin
    r = oref("occest_nkms_lik")
    data, X = odata(r), opoints(r, 3)
    base = JDyn.log_posterior(mkest(NKMS, data; obc_method="news"), X[:, 1])
    for sel in ("shortest", "min_wedge")
        e = mkest(NKMS, data; obc_method="news", obc_selection=sel)
        @test abs(JDyn.log_posterior(e, X[:, 1]) - base) < 1e-9
        @test JDyn.obc_settings(e).last_solver.selection == Symbol(sel)
    end
    # 'longest' picks the 28-period deflationary-trap solution where there is one (pydyn: same): the trial points hop between branches, the root finder fails
    el = mkest(NKMS, data; obc_method="news", obc_selection="longest")
    @test JDyn.log_posterior(el, X[:, 1]) != base
    # a problem with several single-spell solutions: M = [1 2; 2 1]-like structure through the helpers
    M = [1.0 -0.5 -0.2 0.0; 0.0 1.0 -0.5 -0.2; 0.0 0.0 1.0 -0.5; 0.0 0.0 0.0 1.0]
    q = [-1.0, 0.2, 0.3, 0.4]
    c = JDyn._single_spells(M, q, 4)
    @test !isempty(c) && all(x -> x[1] >= 0 && x[2] >= 1, c)
    for x in c
        g = q .+ M * x[3]
        @test minimum(g) >= -1e-8 && all(>=(0), x[3])
    end
    @test JDyn._pick_spell(c, :shortest)[2] == minimum(x[2] for x in c)
    @test JDyn._pick_spell(c, :longest)[2] == maximum(x[2] for x in c)
    @test sum(abs2, JDyn._pick_spell(c, :min_wedge)[3]) == minimum(sum(abs2, x[3]) for x in c)
    # in simulation
    ob = Occbin(load_copy(joinpath(@__DIR__, "obc", "obc_zlb.mod")))
    for sel in (:shortest, :longest, :min_wedge)
        rs = simulate(ob; obc_method=:news, obc_selection=sel)
        r0 = simulate(ob; obc_method=:news)
        @test isapprox(rs.piecewise, r0.piecewise; atol=1e-8)       # one solution at this shock: every rule returns it
    end
    @test_throws ArgumentError simulate(ob; obc_selection=:longest)         # needs news
end

@testset "contract rule 7: the orientation test" begin
    zlb = read(joinpath(@__DIR__, "obc", "obc_zlb.mod"), String)
    for (bind, relax, cls) in (("rn < rbar", "rn >= rbar", :complementarity), ("rn < rbar", nothing, :complementarity),
                               ("rn < rbar", "rn < rbar", :inconsistent), ("rn < rbar", "rn <= rbar", :inconsistent),
                               ("rn < rbar", "rbar > rn", :inconsistent))
        txt = replace(zlb, "bind rn < rbar; relax rn >= rbar;" => "bind $bind;" * (relax === nothing ? "" : " relax $relax;"))
        m = build(txt; name="orient")
        c = obc_constraints(m)[1]
        @test c.class == cls
        cls == :inconsistent && @test occursin("orientation", c.why)
    end
end

@testset "time per likelihood evaluation: OccBin vs news (NKM)" begin
    r = oref("occest_nkm_lik")
    data = odata(r)
    e0, e1 = mkest(NKM, data), mkest(NKM, data; obc_method="news")
    X = opoints(r, e0.n)
    th = X[:, 2]
    JDyn.log_posterior(e0, th); JDyn.log_posterior(e1, th)
    t0 = minimum(@elapsed(JDyn.log_posterior(e0, th)) for _ in 1:5)
    t1 = minimum(@elapsed(JDyn.log_posterior(e1, th)) for _ in 1:5)
    println("  NKM likelihood evaluation: OccBin ", round(1000t0, digits=1), " ms, news ", round(1000t1, digits=1), " ms (", round(t0 / t1, digits=1), "x)")
    @test t1 < t0
end
