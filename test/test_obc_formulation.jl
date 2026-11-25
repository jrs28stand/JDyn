# obc_formulation (auto / lcp / regime) of obc_method = news, simulation and estimation (src/obc.jl `_resolve_formulation`, src/obc_filter.jl `news_filter_solver`).
# Decision: constraints of the APPROXIMATE complementarity class (the two versions of the switched equation differ by a term the condition lacks; relative
# discrepancy above 1e-6, accepted up to 5%) use the regime formulation by default; exact-class models keep the LCP. Evidence (pydyn, 13 models, ~5,900
# log-posterior pairs, docs/OBC_ESTIMATION.md "Approx-class constraints"): the LCP differs from OccBin in 3-100% of the evaluations, the regime formulation in none.
# Here: the NKM examples, pydyn's 81 knife-edge vectors (test/obc/est/nkm_knife_edge.csv), a two-constraint model of pydyn's study (zlb2m).
using Test; include("common.jl")
using Random

const OREF = joinpath(@__DIR__, "ref_occbin")
const EDIR = joinpath(@__DIR__, "obc", "est")
const OBCD = joinpath(@__DIR__, "obc")
oref(name) = matread(joinpath(OREF, name * ".mat"))
odata(r) = (d = r["data"]; d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d))
opoints(r, n) = (X = r["X"]; X isa AbstractVector ? reshape(X, n, :) : (size(X, 1) == n ? X : Matrix(X')))
const IVF = Dict("likelihood_inversion_filter" => true)
mkest(m, data; occbin=IVF, kw...) = with_logger(NullLogger()) do
    JDyn.Estimation(m; data=data, nobs=size(data, 1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=occbin, kw...)
end
function with_lines(file, lines; name="variant.mod")
    d = mktempdir()
    for f in readdir(dirname(file))
        p = joinpath(dirname(file), f); isfile(p) && cp(p, joinpath(d, f); force=true)
    end
    write(joinpath(d, name), lines * read(file, String))
    with_logger(NullLogger()) do; JDyn.load(joinpath(d, name)); end
end
quiet(f) = with_logger(f, NullLogger())

const NKM = load_copy(joinpath(OREF, "occest_nkm.mod"))
const NKMS = load_copy(joinpath(OREF, "occest_nkms.mod"))
const ZLB = load_copy(joinpath(OBCD, "obc_zlb.mod"))

@testset "obc_formulation: values, routes and refusals" begin
    @test JDyn._obc_formulation("auto") == :auto && JDyn._obc_formulation(:LCP) == :lcp && JDyn._obc_formulation(["regime"]) == :regime
    @test_throws ArgumentError JDyn._obc_formulation("holden")
    err = try JDyn._obc_formulation("hybrid"); nothing catch e; e end
    @test err isa ArgumentError && occursin("hybrid", err.msg) && occursin("not implemented", err.msg)
    # simulation
    ob = JDyn.Occbin(ZLB)
    @test_throws ArgumentError simulate(ob, zeros(1, ob.n_exo); obc_method=:minnorm, obc_formulation="regime")     # needs news
    @test_throws ArgumentError simulate(ob, zeros(1, ob.n_exo); obc_method=:news, obc_formulation="bogus")
    @test_throws ArgumentError simulate(ob, zeros(1, ob.n_exo); obc_method=:news, obc_formulation="hybrid")
    # the settings of an estimation
    r = oref("occest_nkms_lik"); data = odata(r)
    f(e) = JDyn.obc_settings(e).formulation
    @test f(mkest(NKMS, data)) == :auto
    @test f(mkest(NKMS, data; obc_method="news", obc_formulation="regime")) == :regime
    @test f(mkest(NKMS, data; occbin=Dict("likelihood_inversion_filter" => true, "obc_method" => "news", "obc_formulation" => "lcp"))) == :lcp
    nat = with_lines(joinpath(OREF, "occest_nkms.mod"), "options_.occbin.obc_method = 'news';\noptions_.occbin.obc_formulation = 'lcp';\n")
    @test f(mkest(nat, data)) == :lcp
    @test f(mkest(nat, data; obc_formulation="regime")) == :regime                 # the keyword wins
    e = mkest(NKMS, data; obc_method="news")
    JDyn.set_obc!(e; formulation="regime"); @test f(e) == :regime
    for bad in ("hybrid", "x", 3)
        @test_throws ArgumentError JDyn.set_obc!(e; formulation=bad)
        @test_throws ArgumentError mkest(NKMS, data; obc_formulation=bad)
    end
    @test_throws ArgumentError with_lines(joinpath(OREF, "occest_nkms.mod"), "options_.occbin.obc_formulation = 'hybrid';\n") |> m -> mkest(m, data)
    @test_throws EstimationError mkest(NKMS, data; occbin=nothing, obc_method="news", obc_formulation="regime")
    # lcp needs every constraint to be one complementary pair: bf switches two equations
    dd = matread(joinpath(EDIR, "occest_bf_data.mat")); bdata = hcat(Float64.(vec(dd["q"])), Float64.(vec(dd["r"])))
    bf = load_copy(joinpath(EDIR, "bf.mod"))
    @test_throws ArgumentError mkest(bf, bdata; obc_method="news", obc_formulation="lcp")
    eb = mkest(bf, bdata; obc_method="news"); @test_throws ArgumentError JDyn.set_obc!(eb; formulation="lcp")
    @test_throws ArgumentError simulate(JDyn.Occbin(bf), zeros(1, 1); obc_method=:news, obc_formulation="lcp")
end

@testset "obc_formulation = auto: what it resolves to" begin
    obn, obz = JDyn.Occbin(NKM), JDyn.Occbin(ZLB)
    cons = JDyn.obc_constraints(obn)
    @test cons[1].class == :complementarity && 1e-6 < cons[1].approx < 0.05          # NKM: approximate class, 0.57%
    @test JDyn.obc_constraints(obz)[1].approx == 0                                   # exact
    @test JDyn._resolve_formulation(obz, :auto) == (:lcp, "")
    f, note = JDyn._resolve_formulation(obn, :auto)
    @test f == :regime && occursin("'zlb'", note) && occursin("0.0057", note) && occursin("regime formulation", note)
    @test JDyn._resolve_formulation(obn, :lcp) == (:lcp, "") && JDyn._resolve_formulation(obn, :regime) == (:regime, "")
    f2, note2 = JDyn._resolve_formulation(obn, :auto; selection=:shortest)           # selection exists for the LCP: kept, with a note
    @test f2 == :lcp && occursin("obc_selection = shortest", note2)
    # the solver and its record
    sv = JDyn.news_filter_solver(obn)
    @test sv isa JDyn.RegimeFilterSolver && sv.stats.formulation == :regime && occursin("0.0057", sv.stats.note)
    sv = JDyn.news_filter_solver(obz)
    @test sv isa JDyn.LCPFilterSolver && sv.stats.formulation == :lcp && sv.stats.note == ""
    @test JDyn.news_filter_solver(obn; formulation=:lcp) isa JDyn.LCPFilterSolver
    @test JDyn.news_filter_solver(obz; formulation=:regime) isa JDyn.RegimeFilterSolver
    # a one-time informational message naming the constraint and its discrepancy
    m = load_copy(joinpath(OREF, "occest_nkm.mod")); o2 = JDyn.Occbin(m)
    @test_logs (:info, r"approximate complementarity constraint 'zlb' \(relative discrepancy 0\.0057\)") JDyn.news_filter_solver(o2)
    @test_logs JDyn.news_filter_solver(o2)                                           # not repeated
end

@testset "simulation: auto on an approximate-class model follows OccBin (regime), exact class keeps the LCP bit for bit" begin
    # exact: auto === lcp
    ob = JDyn.Occbin(ZLB)
    sh = 0.02 .* randn(MersenneTwister(4), 20, ob.n_exo)
    a = simulate(ob, sh; periods=60, obc_method=:news)
    b = simulate(ob, sh; periods=60, obc_method=:news, obc_formulation="lcp")
    @test a.piecewise == b.piecewise && a.news.formulation == :lcp && a.news.note == ""
    # NKM (approximate): regime, close to OccBin; the note is recorded in the result
    obn = JDyn.Occbin(NKM)
    worst_auto, worst_lcp, ndiff = 0.0, 0.0, 0
    for sd in 1:6
        shn = (sd <= 3 ? 1.0 : 3.0) .* randn(MersenneTwister(sd), 25, obn.n_exo) .* 0.01
        o = try simulate(obn, shn; periods=80, check_ahead_periods=60) catch; continue end
        quiet() do
            n1 = simulate(obn, shn; periods=80, check_ahead_periods=60, obc_method=:news)
            @test n1.news.formulation == :regime && occursin("0.0057", n1.news.note)
            worst_auto = max(worst_auto, maximum(abs.(n1.piecewise .- o.piecewise)))
            n2 = simulate(obn, shn; periods=80, check_ahead_periods=60, obc_method=:news, obc_formulation=:lcp)
            @test n2.news.formulation == :lcp
            d = maximum(abs.(n2.piecewise .- o.piecewise)); worst_lcp = max(worst_lcp, d); ndiff += d > 1e-8
        end
    end
    println("  NKM simulation, 6 shock sequences: max |auto (regime) - OccBin| = ", worst_auto, "; max |lcp - OccBin| = ", worst_lcp, " (", ndiff, " sequences differ)")
    @test worst_auto < 1e-6
    # higher order and cubature: the same routing as order 1 (src/obc_regime_ho.jl; the approximate class is the regime formulation by default, lcp is the explicit opt-out)
    shs = 0.01 .* randn(MersenneTwister(9), 6, obn.n_exo)
    ra = quiet() do; simulate(obn, shs; periods=20, obc_method=:news, obc_order=2, obc_check_ahead=40, obc_horizon=40); end
    @test ra.news.formulation == :regime && occursin("0.0057", ra.news.note)
    rl = quiet() do; simulate(obn, shs; periods=20, obc_method=:news, obc_order=2, obc_check_ahead=40, obc_horizon=40, obc_formulation=:lcp); end
    @test rl.news.formulation == :lcp
    rr = quiet() do; simulate(obn, shs; periods=20, obc_method=:news, obc_order=2, obc_check_ahead=40, obc_horizon=40, obc_formulation="regime"); end
    @test rr.piecewise == ra.piecewise
    @test_logs match_mode=:any simulate(obn, shs; periods=20, obc_method=:news, obc_order=2, obc_check_ahead=40, obc_horizon=40, obc_formulation=:lcp)      # no warning for the opt-out
end

@testset "estimation: auto = regime for the NKM examples, equal to OccBin at the stored vectors" begin
    for (m, case, ks) in ((NKMS, "nkms", 1:3), (NKM, "nkm", 1:4))
        r = oref("occest_$(case)_lik"); data = odata(r)
        eo, ea = mkest(m, data), mkest(m, data; obc_method="news")
        X = opoints(r, eo.n)
        for k in ks
            a, o = JDyn.log_posterior(ea, X[:, k]), JDyn.log_posterior(eo, X[:, k])
            sv = JDyn.obc_settings(ea).last_solver
            @test sv isa JDyn.RegimeFilterSolver && sv.stats.formulation == :regime && !isempty(sv.stats.note)
            @test isapprox(a, o; rtol=1e-12, atol=1e-8)
            @test isapprox(a, -vec(r["ivf_fval"])[k]; rtol=1e-11, atol=1e-8)
            ra, ro = JDyn.occbin_filter(ea, X[:, k]; rng=Xoshiro(1)), JDyn.occbin_filter(eo, X[:, k]; rng=Xoshiro(1))
            @test ra.regime_history == ro.regime_history
        end
    end
end

@testset "estimation: pydyn's 81 knife-edge vectors (LCP differs from OccBin, regime does not)" begin
    rows = [split(l, ",") for l in readlines(joinpath(EDIR, "nkm_knife_edge.csv")) if !startswith(l, "#")]
    @test length(rows) == 81
    ests = Dict{String,Any}()
    for c in ("nkm", "nkms")
        r = oref("occest_$(c)_lik"); m = c == "nkm" ? NKM : NKMS; d = odata(r)
        ests[c] = (mkest(m, d), mkest(m, d; obc_method="news", obc_formulation="lcp"), mkest(m, d; obc_method="news"))
    end
    nlcp, nreg, nfail, maxlcp, maxreg, maxpy = 0, 0, 0, 0.0, 0.0, 0.0
    for rw in rows
        c = rw[1]; th = parse.(Float64, rw[5:end]); pyo = parse(Float64, rw[2])
        eo, el, ea = ests[c]
        o, l, a = JDyn.log_posterior(eo, th), JDyn.log_posterior(el, th), JDyn.log_posterior(ea, th)
        if !isfinite(o); nfail += 1; @test !isfinite(a); continue; end
        isfinite(l) && abs(l - o) > 1e-8 * abs(o) && (nlcp += 1; maxlcp = max(maxlcp, abs(l - o)))
        abs(a - o) > 1e-8 * abs(o) && (nreg += 1; maxreg = max(maxreg, abs(a - o)))
        isfinite(pyo) && (maxpy = max(maxpy, abs(o - pyo)))
    end
    println("  81 knife-edge vectors: OccBin fails at ", nfail, "; LCP differs from OccBin (> 1e-8 relative) at ", nlcp, " (max ", maxlcp, "); auto/regime at ", nreg, " (max ", maxreg, "); JDyn OccBin vs pydyn OccBin max ", maxpy)
    @test nreg == 0
    @test nlcp >= 70                                  # all 81 in the first run
    @test maxpy < 1e-6
end

# the two-constraint model of pydyn's approx-class study (soak/approx/gen.py zlbm(d, two=True)): zlb2m_0 exact, zlb2m_1e_2 with discrepancy 0.0100 (zlb constraint)
function zlb2m_est(name; T=100)
    m = load_copy(joinpath(EDIR, name * ".mod"))
    ob = JDyn.Occbin(m)
    obs = [m.ie[v] for v in ("pi", "r")]
    for sd in 7:60
        try
            res = simulate(ob, randn(MersenneTwister(sd), T, m.n_exo); periods=T, check_ahead_periods=60)
            return m, mkest(m, res.piecewise[:, obs]), res
        catch err
            err isa JDyn.OccbinError || rethrow()
        end
    end
    error("no data seed for $name")
end

@testset "two constraints: exact class bit for bit LCP, approximate class regime = OccBin" begin
    m0, e0, res0 = zlb2m_est("zlb2m_0")
    ea0, el0 = mkest(m0, e0.data; obc_method="news"), mkest(m0, e0.data; obc_method="news", obc_formulation="lcp")
    ex0 = e0.x0
    for k in 0:3
        th = k == 0 ? copy(ex0) : ex0 .* exp.(0.03 .* randn(MersenneTwister(k), length(ex0)))
        a, l, o = JDyn.log_posterior(ea0, th), JDyn.log_posterior(el0, th), JDyn.log_posterior(e0, th)
        @test JDyn.obc_settings(ea0).last_solver isa JDyn.LCPFilterSolver
        @test a === l
        isfinite(o) && @test isapprox(a, o; rtol=1e-10, atol=1e-8)
    end
    m1, e1, res1 = zlb2m_est("zlb2m_1e_2")
    cl = JDyn.obc_constraints(JDyn.Occbin(m1))
    @test [c.class for c in cl] == [:complementarity, :complementarity] && cl[1].approx > 1e-6 && cl[2].approx == 0
    ea1, el1 = mkest(m1, e1.data; obc_method="news"), mkest(m1, e1.data; obc_method="news", obc_formulation="lcp")
    ex1 = e1.x0
    nd_lcp, nd_reg, nfin = 0, 0, 0
    for k in 0:11
        th = k == 0 ? copy(ex1) : ex1 .* exp.(0.02 .* randn(MersenneTwister(100 + k), length(ex1)))
        o = JDyn.log_posterior(e1, th)
        isfinite(o) || continue
        nfin += 1
        a, l = JDyn.log_posterior(ea1, th), JDyn.log_posterior(el1, th)
        @test JDyn.obc_settings(ea1).last_solver isa JDyn.RegimeFilterSolver
        isfinite(l) && abs(l - o) > 1e-8 * abs(o) && (nd_lcp += 1)
        abs(a - o) > 1e-8 * abs(o) && (nd_reg += 1)
    end
    println("  zlb2m (two constraints, d = 0.0100): ", nfin, " vectors where OccBin is finite; LCP differs at ", nd_lcp, ", auto (regime) at ", nd_reg)
    @test nd_reg == 0 && nfin >= 3
end

@testset "mode search and chains under the default (regime) follow OccBin's, draw for draw (small NKM)" begin
    r, md = oref("occest_nkms_lik"), oref("occest_nkms_mode")
    data = odata(r)
    eo = mkest(NKMS, data; mode_compute=4); ea = mkest(NKMS, data; mode_compute=4, obc_method="news")
    to = @elapsed mo = quiet() do; JDyn.posterior_mode(eo; compute_hessian=false); end
    ta = @elapsed ma = quiet() do; JDyn.posterior_mode(ea; compute_hessian=false); end
    @test mo.success && ma.success && ma.nfev == mo.nfev && abs(ma.log_posterior - mo.log_posterior) < 1e-8
    sd = sqrt.(diag(inv(md["hh"])))
    @test all(abs.(mo.x .- ma.x) .< 1e-4 .* sd)                       # the LCP's mode was 3e-3 sd away (5.6e-6), the regime formulation's is 1.8e-7
    x, cov = vec(md["xparam1"]), inv(md["hh"])
    tco = @elapsed mco = quiet() do; JDyn.sample_posterior(eo; replic=150, nblocks=2, mode=x, cov=cov, jscale=1.3, seed=20261124); end
    tca = @elapsed mca = quiet() do; JDyn.sample_posterior(ea; replic=150, nblocks=2, mode=x, cov=cov, jscale=1.3, seed=20261124); end
    @test mco.draws == mca.draws && mco.accepted == mca.accepted && all(0.1 .< mca.acceptance_rate .< 0.7)
    println("  mode search: OccBin ", round(to, digits=1), " s (", mo.nfev, " evaluations), auto ", round(ta, digits=1), " s (", ma.nfev, "); chains 2 x 150: OccBin ", round(tco, digits=1), " s, auto ", round(tca, digits=1), " s, draws identical, acceptance ", mca.acceptance_rate)
    @test JDyn.obc_settings(ea).last_solver isa JDyn.RegimeFilterSolver
end
