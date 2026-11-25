# Parity batch 2, OccBin: forecasts (occbin.forecast, Gaussian and Sobol bands), qmc_sequence, findmin and the smoother's first
# period against Dynare 7.2 under Octave 8.4. References in test/ref_parity2/ (pydyn's reference store; generators
# r_occ_r_forecast.m, r_occ_r_findmin.m, r_occ_r_smoother_fpou.m there).
using Test; include("common.jl")
using Random

const P2REF = joinpath(@__DIR__, "ref_parity2")
p2ref(name) = matread(joinpath(P2REF, name * ".mat"))
const OOREF = joinpath(@__DIR__, "ref_occbin")
mxd(a, b) = maximum(abs.(a .- b))

@testset "qmc_sequence (Dynare's mex file)" begin
    f = p2ref("occ_forecast_nkm")
    @test mxd(JDyn.qmc_sequence(24, 1, 1, 127)[1], f["q1"]) < 1e-15
    @test mxd(JDyn.qmc_sequence(7, 1, 0, 20)[1], f["q2"]) < 1e-16
    @test mxd(JDyn.qmc_sequence(1111, 1, 0, 3)[1], f["q3"]) < 1e-16
    a, s = JDyn.qmc_sequence(300, 5, 1, 6)
    @test mxd(a, f["q4"]) < 1e-15 && s == Int(f["q4_seed"])
    a5 = JDyn.qmc_sequence(5, 1, 2, 4, 2.5)[1]
    q5 = f["q5"]
    @test all(isnan.(a5) .== isnan.(q5)) && mxd(a5[.!isnan.(a5)], q5[.!isnan.(q5)]) < 2e-16
    @test mxd(JDyn.qmc_sequence(3, 1, 1, 5, f["q6_L"])[1], f["q6"]) < 1e-15
end

# the NKM model at the reference parameters
const OMOD = load_copy(joinpath(OOREF, "occest_nkm.mod"))
function nkm_est()
    r = matread(joinpath(OOREF, "occest_nkm_lik.mat"))
    d = r["data"]; d = d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d)
    with_logger(NullLogger()) do
        JDyn.Estimation(OMOD; data=d, nobs=size(d, 1), first_obs=1, mode_compute=0, mh_replic=0)
    end
end

fstat(f, pre, fld, st, i) = (r = f["$(pre)_$(fld)_$(st)"]; reshape(r[:, :, i], size(r, 1), :))
function cmp_forecast(f, pre, fc, names, stats, atol; skip=())
    worst = 0.0
    for fld in ("linear", "piecewise"), st in stats, (i, nm) in enumerate(names)
        (st, nm) in skip && continue
        a = getfield(fc, Symbol(fld))[st][nm]
        worst = max(worst, mxd(a, fstat(f, pre, fld, st, i)))
    end
    @test worst < atol
    worst
end

@testset "OccBin forecasts: NKM against Dynare" begin
    f = p2ref("occ_forecast_nkm")
    est = nkm_est()
    x0 = vec(f["x0"])
    names = OMOD.endo_names
    Q = JDyn.covariances(est, x0)[1]
    @test mxd(Q, f["Sigma_e"]) == 0
    h0 = vec(f["histval"])
    fc = JDyn.occbin_forecast(est, x0, 8; endo_histval=h0)
    @test fc.ok && fc.error_flag == 0
    println("forecast A (zero shocks): ", cmp_forecast(f, "A", fc, names, ["Mean"], 1e-12))
    @test mxd(fc.yf, f["A_yf"]) < 1e-12
    fc = JDyn.occbin_forecast(est, x0, 8; endo_histval=h0, shocks=f["D_shocks"])
    println("forecast D (SHOCKS0 matrix): ", cmp_forecast(f, "D", fc, names, ["Mean"], 1e-12))
    fc = JDyn.occbin_forecast(est, x0, 8; endo_histval=h0, shocks=[("epsi", [-1.5, -1, -0.5, 0, 0, 0, 0, 0])])
    println("forecast E (cell SHOCKS0): ", cmp_forecast(f, "E", fc, names, ["Mean"], 1e-12))
    STATS = JDyn.OCCBIN_FORECAST_STATISTICS
    # Gaussian bands from Octave's generator, draw for draw
    rng = JDyn.OctaveRNG(); Random.seed!(rng, 7)
    fc = JDyn.occbin_forecast(est, x0, 8; endo_histval=h0, replic=60, rng=rng)
    println("forecast C (60 Gaussian draws): ", cmp_forecast(f, "C", fc, names, STATS, 1e-11))
    @test mxd(fc.yf, f["C_yf"]) < 1e-11
    @test mxd(randn(rng, 3), vec(f["C_state_after"])) < 1e-14        # the same number of draws consumed
    rng = JDyn.OctaveRNG(); Random.seed!(rng, 3)
    fc = JDyn.occbin_forecast(est, x0, 8; endo_histval=h0, shocks=f["D_shocks"], replic=40, rng=rng)
    println("forecast D2 (SHOCKS0 and 40 draws): ", cmp_forecast(f, "D2", fc, names, STATS, 1e-11))
    # Sobol bands (tests/occbin/filter/NKM.mod: qmc, 127 replications); mp = sigi * epsi has exact ties between candidate HPD intervals
    fc = JDyn.occbin_forecast(est, x0, 8; endo_histval=h0, replic=127, qmc=true)
    @test fc.replic == 127
    println("forecast B (127 Sobol draws): ", cmp_forecast(f, "B", fc, names, STATS, 1e-11; skip=Set([("HPDinf", "mp"), ("HPDsup", "mp")])))
    @test mxd(fc.yf, f["B_yf"]) < 1e-11
end

@testset "OccBin forecasts through dynare() (shared Octave stream)" begin
    d = mktempdir()
    cp(joinpath(P2REF, "occ_forecast_fc1.mod"), joinpath(d, "occ_forecast_fc1.mod"))
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "occ_forecast_fc1.mod"); random=:octave)
    end
    o = p2ref("occ_forecast_fc1")
    names = ws.model.endo_names
    for (jk, rk) in (("forecast", "forecast"), ("forecast1", "forecast1"))
        fcv = ws.vars[jk]; rf = o[rk]
        for fld in ("linear", "piecewise"), st in keys(rf[fld]), nm in names
            @test mxd(fcv[fld][st][nm], reshape(rf[fld][st][nm], size(fcv[fld][st][nm]))) < 1e-13
        end
    end
    for (k, rk) in (("yf", "yf"), ("yf1", "yf1"))
        @test mxd(ws.vars[k], o[rk]) < 1e-13
    end
    # the surprise shocks cover fewer shocks than the model has: Dynare's forecast.m (replic = 0) stops with a size mismatch
    txt = replace(read(joinpath(P2REF, "occ_forecast_fc1.mod"), String), "values 0.01;" => "values 0;")
    write(joinpath(d, "e.mod"), txt)
    err = try; with_logger(NullLogger()) do; JDyn.dynare(joinpath(d, "e.mod")); end; nothing; catch e; e; end
    @test err !== nothing && occursin("exo_pos", sprint(showerror, err))
end

# ---- the loss-function regime guess (occbin.findmin) and Octave's fminsearch
function fm_est(fm, ov)
    d = fm["data"]
    with_logger(NullLogger()) do
        JDyn.Estimation(OMOD; data=Matrix{Float64}(d), nobs=size(d, 1), first_obs=1, mode_compute=0, mh_replic=0, occbin=ov)
    end
end

@testset "fminsearch_octave" begin
    x = JDyn.fminsearch_octave(z -> (1 - z[1])^2 + 100 * (z[2] - z[1]^2)^2, [-1.2, 1.0])[1]
    @test isapprox(x, [1, 1]; atol=5e-3)
end

@testset "OccBin loss-function regime guess (occbin.findmin)" begin
    for (variant, extra) in (("early", Dict("likelihood.number_of_initial_periods_with_extra_regime_guess" => 20)),
                             ("fail", Dict("likelihood.max_number_of_iterations" => 1)))
        fm = p2ref("occ_findmin_$variant")
        ov = merge(Dict("likelihood.loss_function_regime_guess" => true, "likelihood.brute_force_regime_guess" => false,
                        "likelihood.brute_force_extra_regime_guess" => false), extra)
        est = fm_est(fm, ov)
        X = fm["X"]; X = size(X, 1) == est.n ? X : Matrix(X')
        JDyn.FINDMIN_CALLS[] = 0
        lp = log_posterior(est, X[:, 1])
        println("findmin $variant: |log posterior - Dynare| = ", abs(lp + fm["fval"][1]), ", calls ", JDyn.FINDMIN_CALLS[], " (Dynare ", Int(fm["nfind"][1]), ")")
        @test isapprox(lp, -fm["fval"][1]; rtol=1e-12, atol=1e-8)
        @test JDyn.FINDMIN_CALLS[] == Int(fm["nfind"][1])
        res = JDyn.occbin_filter(est, X[:, 1])
        @test mxd(2 .* res.lik, vec(fm["lik_1"])) < 1e-8
        @test mxd(res.a0, fm["a0_1"]) < 1e-10
    end
    # the same limit without the loss-function guess: Dynare's likelihood is Inf (info 320)
    fm = p2ref("occ_findmin_failbase")
    est = fm_est(fm, Dict("likelihood.max_number_of_iterations" => 1, "likelihood.brute_force_regime_guess" => false,
                          "likelihood.brute_force_extra_regime_guess" => false))
    X = fm["X"]; X = size(X, 1) == est.n ? X : Matrix(X')
    @test log_posterior(est, X[:, 1]) == -Inf && fm["info"][1] == 320
end

# ---- smoother_first_period_occbin_update
dregimes2(c) = [JDyn.OccRegime([(Int.(vec(collect(c[2j-1, t]))), Int.(vec(collect(c[2j, t])))) for j in 1:size(c, 1)÷2]) for t in 1:size(c, 2)]
const OMODS = Dict("nkm" => OMOD, "rbc" => load_copy(joinpath(OOREF, "occest_rbc.mod")))
function sm_est(case, r, f)
    m = OMODS[case]
    d = r["data"]; d = d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d)
    est = with_logger(NullLogger()) do
        JDyn.Estimation(m; data=d, nobs=size(d, 1), first_obs=1, mode_compute=0, mh_replic=0, occbin=Dict("smoother.first_period_occbin_update" => f))
    end
    est, d
end
@testset "OccBin smoother: smoother_first_period_occbin_update" begin
    for (name, case, fpous) in (("occ_smoother_fpou_nkm", "nkm", (3, 40)), ("occ_smoother_fpou_nkm2", "nkm", (55, 100)),
                                ("occ_smoother_fpou_rbc", "rbc", (60,)), ("occ_smoother_fpou_rbc2", "rbc", (10, 120)))
        s = p2ref(name)
        r = matread(joinpath(OOREF, "occest_$(case)_lik.mat"))
        for f in fpous
            est, d = sm_est(case, r, f)
            res = smooth(OMODS[case], d; estimation=est, theta=vec(s["x0"]))
            sm = res.occbin.smoother
            pre = "f$f"
            @test sm.error_flag == Int(s[pre * "_error_flag"]) == 0
            ys = sm.ys'
            w = max(mxd(sm.smoothed .+ ys, s[pre*"_SmoothedVariables"]), mxd(sm.shocks, s[pre*"_SmoothedShocks"]),
                    mxd(sm.updated .+ ys, s[pre*"_UpdatedVariables"]), mxd(sm.filtered .+ ys, s[pre*"_FilteredVariables"]))
            println("smoother first_period_occbin_update=$f ($case): max difference ", w)
            @test w < 1e-11
            @test sm.regime_history == dregimes2(s[pre*"_regimes"])
            @test sm.realtime_regime_history == dregimes2(s[pre*"_rt_regimes"])
        end
    end
end
