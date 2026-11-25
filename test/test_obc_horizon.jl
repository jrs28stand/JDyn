# Why a news inversion-filter evaluation failed (`obc_failure(est)`, causes in the solver stats) and the opt-in `obc_horizon_max` (retry with the horizon
# doubled). Port of the semantics of pydyn's tests/test_obc_horizon_cause.py; found by pydyn's soak test: at extreme NKM vectors the news filter at horizon 40 returned
# the generic code 304 where OccBin has a finite value and horizon 80 reproduces it (soak/repro/finite_vs_inf_news_304.py). docs/OBC_HO_PROGRESS.md, "Horizon too short".
using Test; include("common.jl")
using Random

const OREF = joinpath(@__DIR__, "ref_occbin")
const OBCD = joinpath(@__DIR__, "obc")
oref(name) = matread(joinpath(OREF, name * ".mat"))
odata(r) = (d = r["data"]; d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d))
opoints(r, n) = (X = r["X"]; X isa AbstractVector ? reshape(X, n, :) : (size(X, 1) == n ? X : Matrix(X')))
const IVF = Dict("likelihood_inversion_filter" => true)
# NKM / NKMS: the LCP formulation explicitly (the default, auto, is the regime formulation for their approximate-class constraint; see test_obc_formulation.jl)
mkest(m, data; occbin=IVF, kw...) = with_logger(NullLogger()) do
    kw = haskey(kw, :obc_method) && !haskey(kw, :obc_formulation) && m !== nothing && m.name in ("occest_nkm", "occest_nkms") ? (; kw..., obc_formulation="lcp") : kw
    JDyn.Estimation(m; data=data, nobs=size(data, 1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=occbin, kw...)
end
function load_dir2(path)
    d = mktempdir()
    for f in readdir(dirname(path))
        p = joinpath(dirname(path), f); isfile(p) && cp(p, joinpath(d, f))
    end
    with_logger(NullLogger()) do; JDyn.load(joinpath(d, basename(path))); end
end
function with_lines(file, lines; name="variant.mod")
    d = mktempdir()
    for f in readdir(dirname(file))
        p = joinpath(dirname(file), f); isfile(p) && cp(p, joinpath(d, f); force=true)
    end
    write(joinpath(d, name), lines * read(file, String))
    with_logger(NullLogger()) do; JDyn.load(joinpath(d, name)); end
end

same_period(a, b) = a.piecewise == b.piecewise && a.ss == b.ss && a.regime_history == b.regime_history

const NKM = load_copy(joinpath(OREF, "occest_nkm.mod"))
const NKMS = load_copy(joinpath(OREF, "occest_nkms.mod"))
# pydyn's soak vector A#75: OccBin's inversion filter gives -210551.277947312; news at horizon 40 fails (code 304), horizon 80 gives the same value
const TH = [114.66637644899114, 1.0892489266590764, 0.19766855658573093, 0.6874612608464392, 0.6985799052891387, 0.46604253593794787,
            0.005397842157745668, 0.0016956255944929521, 0.0021002683787342213]

@testset "horizon too short: the cause is carried, the code is not changed" begin
    r = oref("occest_nkm_lik"); data = odata(r)
    eo, en = mkest(NKM, data), mkest(NKM, data; obc_method="news")
    lpo = JDyn.log_posterior(eo, TH)
    @test isfinite(lpo) && eo.last_info_code == 0 && JDyn.obc_failure(eo) === nothing
    @test lpo ≈ -210551.277947312 rtol=1e-10                            # pydyn's OccBin value
    lp = JDyn.log_posterior(en, TH)
    @test lp == -Inf && en.last_info_code == 304                        # Dynare's code: unchanged
    f = JDyn.obc_failure(en)
    @test f isa JDyn.ObcFailure && f.primary in (:horizon_too_short, :undecided) && f.horizon == 40 && f.horizon_max == 0
    @test 1 <= f.period <= size(data, 1)
    @test occursin("news horizon 40 too short at period $(f.period)", f.advice) || f.primary == :undecided
    @test occursin("obc_horizon", f.advice) && occursin("period $(f.period)", en.last_info) && occursin(string(f.primary), en.last_info) && occursin("info 304", en.last_info)
    st = JDyn.obc_settings(en).last_solver.stats
    @test sum(values(st.causes)) == st.errors[313] && st.retries == 0 && st.retried_ok == 0
    @test sum(values(f.counts)) > 0 && f.counts == Dict(k => v for (k, v) in st.period_causes if v > 0)
    println("  extreme NKM vector, news horizon 40: ", f, "; causes ", st.causes)
    # OccBin's own failure carries no news information, and a success clears it
    @test JDyn.obc_failure(eo) === nothing
    X = opoints(r, en.n)
    @test isfinite(JDyn.log_posterior(en, X[:, 1])) && JDyn.obc_failure(en) === nothing
    # occbin_filter returns it too
    ivf = JDyn.occbin_filter(en, TH; rng=Xoshiro(1))
    @test ivf.error_code == 304 && ivf.failure isa JDyn.ObcFailure && ivf.failure.period == ivf.error_period
    @test JDyn.occbin_filter(eo, TH; rng=Xoshiro(1)).failure === nothing
end

@testset "obc_horizon_max: the extreme vector gives OccBin's value" begin
    r = oref("occest_nkm_lik"); data = odata(r)
    eo = mkest(NKM, data)
    lpo = JDyn.log_posterior(eo, TH)
    e = mkest(NKM, data; obc_method="news", obc_horizon_max=80)
    @test JDyn.obc_settings(e).horizon_max == 80
    t = @elapsed lp = JDyn.log_posterior(e, TH)
    st = JDyn.obc_settings(e).last_solver.stats
    @test e.last_info_code == 0 && JDyn.obc_failure(e) === nothing
    @test abs(lp - lpo) < 1e-6 * 1e-5 * abs(lpo) + 1e-8                 # |lp| = 2e5: a few ulp
    @test st.retries > 0 && st.retried_ok == st.retries && sum(values(st.causes)) == 0 && st.retry_seconds > 0
    @test collect(keys(JDyn.obc_settings(e).last_solver.retry.big)) == [80]
    println("  obc_horizon_max = 80: lp = ", lp, ", OccBin ", lpo, ", |diff| = ", abs(lp - lpo), "; ", st.retries, " retries, all solved, ", round(st.retry_seconds, digits=2), " s in retries, ", round(t, digits=2), " s in all")
    # a maximum between: 60 is not enough (40 -> 60), the failure then says so
    e60 = mkest(NKM, data; obc_method="news", obc_horizon_max=60)
    lp60 = JDyn.log_posterior(e60, TH)
    if lp60 == -Inf
        @test e60.last_info_code == 304 && occursin("obc_horizon_max = 60 was tried", JDyn.obc_failure(e60).advice)
    end
    # where the default succeeds the value is bit for bit the same, whatever the maximum
    ed, ex = mkest(NKM, data; obc_method="news"), mkest(NKM, data; obc_method="news", obc_horizon_max=160)
    X = opoints(r, ed.n)
    for k in 1:4
        @test JDyn.log_posterior(ed, X[:, k]) === JDyn.log_posterior(ex, X[:, k])
    end
end

@testset "obc_horizon_max: routes and refusals" begin
    r = oref("occest_nkms_lik"); data = odata(r)
    h(e) = JDyn.obc_settings(e).horizon_max
    @test h(mkest(NKMS, data; obc_method="news")) == 0
    @test h(mkest(NKMS, data; obc_method="news", obc_horizon_max=100)) == 100
    @test h(mkest(NKMS, data; occbin=Dict("likelihood_inversion_filter" => true, "obc_method" => "news", "obc_horizon_max" => 90))) == 90
    nat = with_lines(joinpath(OREF, "occest_nkms.mod"), "options_.occbin.obc_method = 'news';\noptions_.occbin.obc_horizon_max = 70;\n")
    @test h(mkest(nat, data)) == 70
    @test h(mkest(nat, data; obc_horizon_max=120)) == 120                 # the keyword wins
    @test h(mkest(NKMS, data; obc_horizon_max=0)) == 0
    e = mkest(NKMS, data; obc_method="news")
    JDyn.set_obc!(e; horizon_max=80); @test h(e) == 80
    JDyn.set_obc!(e; horizon_max=0); @test h(e) == 0
    for bad in (-1, 2.5, "x", [1, 2], NaN)
        @test_throws ArgumentError JDyn.set_obc!(e; horizon_max=bad)
        @test_throws ArgumentError mkest(NKMS, data; obc_horizon_max=bad)
    end
    @test_throws ArgumentError mkest(NKMS, data; obc_horizon_max=30)              # below the default horizon 40
    @test_throws ArgumentError mkest(NKMS, data; obc_horizon=50, obc_horizon_max=45)
    @test_throws ArgumentError JDyn.set_obc!(e; horizon_max=39)
    JDyn.set_obc!(e; horizon_max=60)
    @test_throws ArgumentError JDyn.set_obc!(e; horizon=61)                       # raising the horizon above the maximum
    @test h(e) == 60 && JDyn.obc_settings(e).horizon == 40                        # a refused change leaves the settings alone
    @test_throws ArgumentError with_lines(joinpath(OREF, "occest_nkms.mod"), "options_.occbin.obc_horizon_max = -4;\n") |> m -> mkest(m, data)
    @test_throws ArgumentError with_lines(joinpath(OREF, "occest_nkms.mod"), "options_.occbin.obc_horizon_max = 10;\n") |> m -> mkest(m, data)
    @test_throws ArgumentError JDyn.LCPFilterSolver(JDyn.Occbin(NKMS); horizon=40, horizon_max=20)
    let m = load_copy(joinpath(@__DIR__, "optim", "est_opt_nk.mod"))               # a model without constraints refuses the option
        @test_throws EstimationError JDyn.Estimation(m; mode_compute=0, mh_replic=0, allow_unsupported=true, obc_horizon_max=80)
    end
    # the option is an option of the news filter: with obc_method = occbin it changes nothing
    a, b = mkest(NKMS, data), mkest(NKMS, data; obc_horizon_max=80)
    X = opoints(r, a.n)
    @test JDyn.log_posterior(a, X[:, 1]) === JDyn.log_posterior(b, X[:, 1])
end

@testset "causes in the solvers: LCP and regime formulations on borrcon (a model with a long binding spell)" begin
    ob = JDyn.Occbin(load_dir2(joinpath(DYNARE_SRC, "tests", "occbin", "model_borrcon", "borrcon.mod")))
    function scan(mk; n=400)
        sv = mk()
        rng = MersenneTwister(1)
        for _ in 1:n
            sh = randn(rng, 1, ob.n_exo) .* rand(rng, [0.05, 0.2, 0.6, 1.5])
            init = randn(rng, ob.n) .* rand(rng, [0.0, 0.05, 0.2])
            JDyn.filter_solve(sv, sh, init, nothing)
        end
        sv
    end
    for (name, S) in (("LCP", JDyn.LCPFilterSolver), ("regime", JDyn.RegimeFilterSolver))
        a = scan(() -> S(ob; horizon=20))
        nf = sum(values(a.stats.errors))
        @test sum(values(a.stats.causes)) == nf && nf > 20
        @test all(c -> c in JDyn.OBC_FAILURE_CAUSES, keys(a.stats.causes))
        @test get(a.stats.causes, :horizon_too_short, 0) > 0
        b = scan(() -> S(ob; horizon=20, horizon_max=80))
        nb = sum(values(b.stats.errors))
        @test b.stats.retried_ok > 0 && nb < nf && sum(values(b.stats.causes)) == nb
        @test b.stats.retries >= b.stats.retried_ok && haskey(b.retry.big, 40)
        @test all(h -> h <= 80, keys(b.retry.big)) && b.stats.retry_seconds > 0
        println("  $name solver, horizon 20, 400 trial points: causes ", a.stats.causes, "; with obc_horizon_max = 80: ", b.stats.causes, ", ", b.stats.retried_ok, " of ", b.stats.retries, " retries solved")
        # a retried point has the larger solver's solution: the same as asking that solver
        sh, init = reshape([1.0], 1, 1), zeros(ob.n)
        for _ in 1:1
            rng = MersenneTwister(2)
            for _ in 1:200
                sh = randn(rng, 1, ob.n_exo) .* 1.5; init = zeros(ob.n)
                fa, ra = JDyn.filter_solve(S(ob; horizon=20), sh, init, nothing)
                fb, rb = JDyn.filter_solve(S(ob; horizon=20, horizon_max=80), sh, init, nothing)
                fa == 0 && (@test fb == 0 && same_period(ra, rb); continue)
                if fb == 0
                    fc, rc = JDyn.filter_solve(S(ob; horizon=40), sh, init, nothing)
                    fc == 0 && @test same_period(rb, rc)
                end
            end
        end
    end
    sv = JDyn.LCPFilterSolver(ob; horizon=20)
    JDyn.begin_period!(sv, 7)
    f = JDyn.failure_info(sv)
    @test f.primary == :root_search && f.period == 7 && isempty(f.counts)
end

@testset "the regime formulation (default for the NKM's approximate-class constraint) needs more horizon than the LCP at the extreme vector" begin
    r = oref("occest_nkm_lik"); data = odata(r)
    eo = mkest(NKM, data); lpo = JDyn.log_posterior(eo, TH)
    # auto = regime here; pydyn: horizon 80 is not enough for the regime formulation (horizon_too_short at period 92), 160 is
    e80 = with_logger(NullLogger()) do
        JDyn.Estimation(NKM; data=data, nobs=size(data, 1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=IVF, obc_method="news", obc_horizon_max=80)
    end
    lp80 = JDyn.log_posterior(e80, TH)
    @test JDyn.obc_settings(e80).last_solver isa JDyn.RegimeFilterSolver
    f = JDyn.obc_failure(e80)
    println("  regime, obc_horizon_max = 80: lp = ", lp80, ", failure: ", f)
    e160 = with_logger(NullLogger()) do
        JDyn.Estimation(NKM; data=data, nobs=size(data, 1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=IVF, obc_method="news", obc_horizon_max=160)
    end
    lp160 = JDyn.log_posterior(e160, TH)
    st = JDyn.obc_settings(e160).last_solver.stats
    println("  regime, obc_horizon_max = 160: lp = ", lp160, ", OccBin ", lpo, ", |diff| = ", abs(lp160 - lpo), "; retries ", st.retries, ", solved ", st.retried_ok, ", ", round(st.retry_seconds, digits=2), " s")
    @test e160.last_info_code == 0 && abs(lp160 - lpo) < 1e-5 * 1e-4 * abs(lpo) + 1e-8
    if lp80 == -Inf
        @test e80.last_info_code == 304 && f !== nothing && f.primary == :horizon_too_short && f.horizon_max == 80
    end
end
