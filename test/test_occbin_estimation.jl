# OccBin likelihoods (inversion filter, piecewise Kalman filter), smoother, shock decomposition and estimation against
# Dynare 7.2 (port of pydyn's tests/test_occbin_estimation.py). References in test/ref_occbin/, written by
# r_occbin_estimation.m (Dynare 7.2 under Octave 8.4):
#   occest_nkm   Dynare's tests/occbin/filter NKM model (ZLB, 24 variables, 3 shocks, 3 observables, 9 estimated
#                parameters) with its data; PKF and IVF likelihoods at 4 parameter vectors around the stored mode, the
#                PKF's per-period likelihood, filtered states and regimes, evaluate_smoother at the mode;
#   occest_rbc   tests/occbin/model_irrcap_twoconstraints (two constraints, one shock observed through consumption);
#                the smoother needs two iterations and Dynare's conditional-smoother attempt;
#   occest_nkms  the NKM model with small_model = 1 and three estimated parameters; posterior mode by mode_compute = 4;
#   occest_nkmopt the NKM likelihoods with filter_use_relaxation, likelihood_max_kalman_iterations = 3,
#                likelihood_periodic_solution and likelihood_check_ahead_periods = 100.
using Test; include("common.jl")

const OREF = joinpath(@__DIR__, "ref_occbin")
const SLOW = get(ENV, "JDYN_SLOW_TESTS", "0") == "1"
oref(name) = matread(joinpath(OREF, name * ".mat"))
const OMODELS = Dict(n => load_copy(joinpath(OREF, n * ".mod")) for n in ("occest_nkm", "occest_rbc", "occest_nkms"))

"Dynare's regime history (cell: 2 rows per constraint x periods) as JDyn regimes."
dregimes(c) = [JDyn.OccRegime([(Int.(vec(collect(c[2j-1, t]))), Int.(vec(collect(c[2j, t])))) for j in 1:size(c, 1)÷2]) for t in 1:size(c, 2)]
odata(r) = (d = r["data"]; d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d))
function oest(case, r; kw...)
    d = odata(r)
    with_logger(NullLogger()) do
        JDyn.Estimation(OMODELS[case]; data=d, nobs=size(d, 1), first_obs=1, mode_compute=0, mh_replic=0, kw...)
    end
end
opoints(r, n) = (X = r["X"]; X isa AbstractVector ? reshape(X, n, :) : (size(X, 1) == n ? X : Matrix(X')))
mxd(a, b) = maximum(abs.(a .- b))

const CASES = [("nkm", "occest_nkm", 1:4), ("rbc", "occest_rbc", 1:3), ("nkms", "occest_nkms", 1:3)]

@testset "OccBin inversion filter likelihood" begin
    for (case, mod, ks) in CASES
        r = oref("occest_$(case)_lik")
        est = oest(mod, r; occbin=Dict("likelihood_inversion_filter" => true))
        X = opoints(r, est.n)
        worst = 0.0
        for k in ks
            lp = log_posterior(est, X[:, k])
            @test est.last_filter == "inversion filter"
            fv = vec(r["ivf_fval"])[k]
            worst = max(worst, abs(lp + fv))
            @test isapprox(lp, -fv; rtol=1e-12, atol=1e-8)
            res = JDyn.occbin_filter(est, X[:, k])
            if haskey(r, "ivf_atT_$k")               # states (decision-rule order) and shocks
                ov = Int.(vec(r["order_var"]))
                @test mxd(res.stateval[:, ov]', r["ivf_atT_$k"]) < 1e-10
                innov = r["ivf_innov_$k"]
                @test mxd(res.filtered_errs, reshape(Matrix(innov'), size(res.filtered_errs))) < 1e-9
            end
            @test res.regime_history == dregimes(r["ivf_regimes_$k"])[1:length(res.regime_history)]
        end
        println("IVF $case: max |log posterior - Dynare| = ", worst)
    end
    # each evaluation draws from its own generator, so the result does not depend on the shared stream's position
    let r = oref("occest_nkm_lik"), est = oest("occest_nkm", r; occbin=Dict("likelihood_inversion_filter" => true))
        x = opoints(r, est.n)[:, 1]
        a = JDyn.occbin_filter(est, x)
        rand(JDyn.dynare_rng(est.model), 1000)
        b = JDyn.occbin_filter(est, x)
        @test a.error_code == b.error_code && a.LIK == b.LIK && a.filtered_errs == b.filtered_errs
        @test JDyn._ivf_rng(est.model) !== JDyn.dynare_rng(est.model)
    end
    # the NKM data reach the zero lower bound: the inversion filter goes through binding periods
    rh = dregimes(oref("occest_nkm_lik")["ivf_regimes_1"])
    @test count(reg -> reg[1][1][1] == 1, rh) > 20
end

@testset "blank estimated_params bounds fall back to the prior's (MATLAB max/min skip NaN)" begin
    # occest_nkms writes `phipi,,,,NORMAL_PDF,2,0.25;`: explicitly empty bounds reach JDyn as NaN
    est = oest("occest_nkms", oref("occest_nkms_lik"))
    @test !any(isnan, est.lb) && !any(isnan, est.ub)
    plb, pub = JDyn.prior_bounds(est.prior, est.options["prior_trunc"])
    @test est.lb == plb && est.ub == pub
    @test JDyn._matlab_max(NaN, 1.0) == 1.0 && JDyn._matlab_min(2.0, NaN) == 2.0 && isnan(JDyn._matlab_max(NaN, NaN))
end

@testset "OccBin piecewise Kalman filter likelihood" begin
    # periods where two self-consistent regimes have likelihoods within 2e-15 of each other (the second constraint expected
    # to bind until period 113 or 116, 91 or 95, ...): both programs combine the two densities (kalman_update_engine.m), so the
    # likelihood is the same; which regime is reported is decided by the strict `<` between the two values, i.e. by rounding
    ties = Dict(("rbc", 1) => [50], ("rbc", 2) => [53, 54])
    for (case, mod, ks) in CASES
        r = oref("occest_$(case)_lik")
        est = oest(mod, r)
        X = opoints(r, est.n)
        worst = 0.0
        for k in ks
            lp = log_posterior(est, X[:, k])
            @test est.last_filter == "piecewise Kalman filter"
            fv = vec(r["pkf_fval"])[k]
            worst = max(worst, abs(lp + fv))
            @test isapprox(lp, -fv; rtol=1e-12, atol=1e-8)
            res = JDyn.occbin_filter(est, X[:, k])
            @test mxd(2 .* res.lik, vec(r["pkf_lik_$k"])) < 1e-8          # Dynare's lik before the final halving
            @test mxd(res.a0, r["pkf_a0_$k"]) < 1e-10
            dyn = dregimes(r["pkf_regimes_$k"])
            T = length(res.regimes)
            diff = [t for t in 1:T if res.regimes[t] != dyn[t]]
            @test diff == get(ties, (case, k), Int[])
            for t in diff      # the tie concerns the start of the last regime of the second constraint only
                @test res.regimes[t][1] == dyn[t][1] && res.regimes[t][2][1] == dyn[t][2][1]
            end
        end
        println("PKF $case: max |log posterior - Dynare| = ", worst)
    end
end

@testset "OccBin filter options against Dynare" begin
    # filter_use_relaxation, likelihood_max_kalman_iterations=3, likelihood_periodic_solution, likelihood_check_ahead_periods=100
    r = oref("occest_nkmopt_lik")
    opts = Dict("filter_use_relaxation" => true, "likelihood_max_kalman_iterations" => 3, "likelihood_periodic_solution" => true,
                "likelihood_check_ahead_periods" => 100)
    for ivf in (false, true)
        est = oest("occest_nkm", r; occbin=merge(opts, Dict("likelihood_inversion_filter" => ivf)))
        oc = JDyn.occbin_options(est)
        @test oc["filter.use_relaxation"] && oc["likelihood.check_ahead_periods"] == 100 && oc["likelihood.max_number_of_iterations"] == 3
        X = opoints(r, est.n)
        for k in 1:size(X, 2)
            @test isapprox(log_posterior(est, X[:, k]), -vec(r[ivf ? "ivf_fval" : "pkf_fval"])[k]; rtol=1e-12, atol=1e-8)
        end
    end
end

function check_smoother(res, s, prefix, atol)
    sm = res.occbin.smoother
    ys = sm.ys
    @test mxd(sm.smoothed .+ ys', s["$(prefix)_SmoothedVariables"]) < atol
    @test mxd(sm.shocks, s["$(prefix)_SmoothedShocks"]) < atol
    # the SmootherResult (what goes to oo_) carries the same numbers
    nm = res.endo_names
    @test mxd(hcat([res.SmoothedVariables[v] for v in nm]...), s["$(prefix)_SmoothedVariables"]) < atol
    @test mxd(hcat([res.SmoothedShocks[v] for v in res.exo_names]...), s["$(prefix)_SmoothedShocks"]) < atol
end

@testset "OccBin smoother and shock decomposition: NKM" begin
    r, s = oref("occest_nkm_lik"), oref("occest_nkm_smoother")
    m = OMODELS["occest_nkm"]
    est = oest("occest_nkm", r)
    x0 = vec(s["x0"])
    res = smooth(m, odata(r); estimation=est, theta=x0)
    sm = res.occbin.smoother
    @test sm.error_flag == s["pkf_error_flag"] == 0 && sm.converged && sm.iterations == 1
    check_smoother(res, s, "pkf", 1e-10)
    @test mxd(sm.updated .+ sm.ys', s["pkf_UpdatedVariables"]) < 1e-10
    @test mxd(sm.filtered .+ sm.ys', s["pkf_FilteredVariables"]) < 1e-10
    @test mxd(hcat([res.FilteredVariables[v] for v in m.endo_names]...), s["pkf_FilteredVariables"]) < 1e-10
    @test sm.regime_history == dregimes(s["pkf_regimes"])
    @test sm.realtime_regime_history == dregimes(s["pkf_rt_regimes"])
    ov = Int.(vec(r["order_var"]))
    @test mxd(sm.linear_smoothed[:, ov]', s["pkf_linear_alphahat"]) < 1e-10
    @test mxd(sm.linear_shocks', s["pkf_linear_etahat"]) < 1e-10
    println("OccBin smoother NKM: smoothed ", mxd(sm.smoothed .+ sm.ys', s["pkf_SmoothedVariables"]), ", shocks ",
            mxd(sm.shocks, s["pkf_SmoothedShocks"]))
    # inversion filter as smoother (smoother_inversion_filter)
    iv = smooth(m, odata(r); estimation=est, theta=x0, occbin=Dict("smoother_inversion_filter" => true))
    @test iv.occbin.smoother.inversion_filter
    check_smoother(iv, s, "ivf", 1e-10)
    @test all(isnan, iv.UpdatedVariables["yg"])
    # shock decomposition (occbin.shock_decomposition, first period as TINIT)
    d = occbin_shock_decomposition(m, res)
    rows = Int.(vec(s["sd_rows"]))
    n = m.n_endo
    worst = 0.0
    for (key, dk) in (("sd_decomp", "decomp"), ("sd_wdecomp", "wdecomp"), ("sd_linear", "linear"), ("sd_init", "init_decomp"),
                      ("sd_linear_init", "linear_init"))
        z = d[dk].z[rows, :, :]
        occursin("init", key) && (z = z[:, 1:n, :])        # Dynare keeps only the n variable columns
        worst = max(worst, mxd(z, reshape(s[key], size(z))))
        @test mxd(z, reshape(s[key], size(z))) < 1e-12
    end
    println("OccBin shock decomposition NKM: ", worst)
    z = d["decomp"].z
    @test mxd(dropdims(sum(z[:, 1:end-1, :]; dims=2); dims=2), z[:, end, :]) < 1e-12     # adds up
    @test mxd(z[:, end, :]', sm.smoothed) < 1e-10
    names = m.endo_names[rows]
    sq = squeeze_occbin_shock_decomposition(d, names)
    @test mxd(sq["decomp"].z, reshape(s["sd_decomp"], size(sq["decomp"].z))) < 1e-12
    @test sq["wdecomp"].endo_names == names
end

@testset "OccBin smoother: two constraints" begin
    r, s = oref("occest_rbc_lik"), oref("occest_rbc_smoother")
    m = OMODELS["occest_rbc"]
    est = oest("occest_rbc", r)
    x0 = vec(s["x0"])
    res = smooth(m, odata(r); estimation=est, theta=x0)
    sm = res.occbin.smoother
    # Dynare: converged after iterating, to a regime different from the real-time one (warning 320)
    @test sm.error_flag == 0 && sm.warning_flag == 320 && sm.iterations == 2
    check_smoother(res, s, "pkf", 1e-11)
    @test mxd(sm.updated .+ sm.ys', reshape(s["pkf_UpdatedVariables"], size(sm.updated))) < 1e-11
    @test mxd(sm.filtered .+ sm.ys', reshape(s["pkf_FilteredVariables"], size(sm.filtered))) < 1e-11
    # regime histories: the same rounding tie as the likelihood at period 50 (start of the last regime of constraint 2)
    for (mine, dyn) in ((sm.regime_history, dregimes(s["pkf_regimes"])), (sm.realtime_regime_history, dregimes(s["pkf_rt_regimes"])))
        @test length(mine) == length(dyn)
        diff = [t for t in eachindex(mine) if mine[t] != dyn[t]]
        @test diff == [50]
        @test mine[50][1] == dyn[50][1] && mine[50][2][1] == dyn[50][2][1]
    end
    @test sm.regime_history != sm.realtime_regime_history
    iv = smooth(m, odata(r); estimation=est, theta=x0, occbin=Dict("smoother_inversion_filter" => true))
    check_smoother(iv, s, "ivf", 1e-12)
    d = occbin_shock_decomposition(m, res)
    @test mxd(d["decomp"].z, reshape(s["sd_decomp"], size(d["decomp"].z))) < 1e-11
    @test mxd(d["wdecomp"].z, reshape(s["sd_wdecomp"], size(d["wdecomp"].z))) < 1e-11
    @test mxd(d["linear"].z, reshape(s["sd_linear"], size(d["linear"].z))) < 1e-12
    # regime table (occbin_write_regimes) and indicator
    tab = regime_table(sm.regime_history; periods=[1, 2, 3])
    @test Set(keys(tab[1])) == Set(["time", "regime_sequence_1", "starting_period_of_regime_1", "regime_sequence_2", "starting_period_of_regime_2"])
    @test tab[1]["regime_sequence_1"] == join(string.(sm.regime_history[1][1][1]), "  ")
    @test size(JDyn.regime_indicator(sm.regime_history)) == (length(sm.regime_history), 2)
end

@testset "OccBin estimation: Hessian, Laplace, sampler" begin
    # At Dynare's posterior mode (mode_compute = 4 from the prior mean, small NKM model, PKF): the objective, Dynare's Hessian
    # (rtol 2e-3: finite differences on a piecewise-linear filter) and the Laplace approximation (Dynare printed 1515.166990)
    r, md = oref("occest_nkms_lik"), oref("occest_nkms_mode")
    est = oest("occest_nkms", r)
    x = vec(md["xparam1"])
    @test isapprox(log_posterior(est, x), -md["fval"]; rtol=1e-12)
    H = JDyn.hessian(est, x)
    @test isapprox(H, md["hh"]; rtol=2e-3)
    lap = JDyn.laplace(est, x, JDyn.inverse_hessian(H))
    println("OccBin Laplace: ", lap, " (Dynare 1515.166990), Hessian rel. diff ", maximum(abs.(H .- md["hh"]) ./ abs.(md["hh"])))
    @test isapprox(lap, 1515.166990; atol=2e-3)
    # a short random-walk Metropolis-Hastings chain on the OccBin posterior runs end to end
    replic = SLOW ? 200 : 8
    mc = with_logger(NullLogger()) do
        JDyn.sample_posterior(est; replic=replic, nblocks=1, cov=inv(md["hh"]), jscale=0.5, x0=reshape(x, 1, :), seed=1)
    end
    @test all(isfinite, mc.draws) && size(mc.draws)[end] == 3
end

if SLOW
    @testset "OccBin posterior mode (mode_compute = 4) against Dynare" begin
        # csminwel from the prior mean, as Dynare's run (pydyn: modes within 3e-4 posterior std, log posterior within 4.5e-8)
        r, md = oref("occest_nkms_lik"), oref("occest_nkms_mode")
        est = oest("occest_nkms", r; mode_compute=4)
        t = @elapsed mode = with_logger(NullLogger()) do; posterior_mode(est); end
        sd = sqrt.(diag(inv(md["hh"])))
        println("OccBin posterior mode: ", round(t; digits=1), " s; |x - Dynare|/sd = ", maximum(abs.(mode.x .- vec(md["xparam1"])) ./ sd),
                ", log posterior diff ", mode.log_posterior + md["fval"])
        @test all(abs.(mode.x .- vec(md["xparam1"])) .< 1e-3 .* sd)
        @test isapprox(mode.log_posterior, -md["fval"]; atol=1e-6)
    end
end

@testset "OccBin options" begin
    m = OMODELS["occest_rbc"]
    o = occbin_filter_options(m)
    # occest_rbc.mod: occbin_setup(smoother_periods=200, smoother_maxit=200, smoother_curb_retrench, ...)
    @test o["smoother.periods"] == 200 && o["smoother.maxit"] == 200 && o["smoother.curb_retrench"]
    @test o["likelihood.periods"] == 3 && o["likelihood.maxit"] == 30          # Dynare's defaults
    @test !o["likelihood.inversion_filter"]
    o2 = occbin_filter_options(m, Dict("likelihood_maxit" => 7, "likelihood_inversion_filter" => true, "likelihood_max_kalman_iterations" => 4))
    @test o2["likelihood.maxit"] == 7 && o2["likelihood.inversion_filter"] && o2["likelihood.max_number_of_iterations"] == 4
    # options that Dynare's occbin.setup drops are reported once; a later occbin_setup does not reset earlier ones
    src = read(joinpath(OREF, "occest_rbc.mod"), String) *
          "\noccbin_setup(smoother_max_number_of_iterations=3, filter_particle_number_of_particles=10, likelihood_inversion_filter);\n"
    m2 = build(src; name="occest_rbc2")
    o3 = @test_logs (:warn, r"smoother.max_number_of_iterations") match_mode=:any occbin_filter_options(m2)
    @test o3["likelihood.inversion_filter"] && o3["smoother.max_number_of_iterations"] == 10 && o3["smoother.periods"] == 200
    # options Dynare acts on but JDyn does not implement raise
    @test_throws EstimationError occbin_filter_options(m, Dict("likelihood_loss_function_regime_guess" => true))
    @test_throws EstimationError occbin_filter_options(m, Dict("filter.particle.status" => true))
    @test_throws ArgumentError occbin_filter_options(m, Dict("nonsense" => 1))
    # IVF: exactly as many shocks as observables, no measurement errors
    r = oref("occest_nkm_lik")
    @test_throws EstimationError oest("occest_nkm", r; occbin=Dict("likelihood_inversion_filter" => true,
                                                                   "likelihood.IVF_shock_observable_mapping" => ["epsg", "epsi"]))
    # Dynare's initial_estimation_checks
    @test_throws EstimationError oest("occest_nkm", r; prefilter=1)
    @test_throws EstimationError oest("occest_nkm", r; kalman_algo=2)
    @test_throws EstimationError oest("occest_nkm", r; diffuse_filter=true)
    # smoother options needing regime-dependent variances are refused
    s = oref("occest_nkm_smoother")
    est = oest("occest_nkm", r)
    @test_throws SmootherError smooth(OMODELS["occest_nkm"], odata(r); estimation=est, theta=vec(s["x0"]), smoothed_state_uncertainty=true)
end

@testset "regime table" begin
    rh = [JDyn.OccRegime([([0], [1])]), JDyn.OccRegime([([1, 0], [1, 3])]), JDyn.OccRegime([([0, 1, 0], [1, 2, 4])])]
    p = tempname() * ".csv"
    rows = write_regime_table(p, rh)
    @test rows[2]["time"] == 2 && rows[2]["regime_sequence"] == "1  0" && rows[2]["starting_period_of_regime"] == "1  3"
    text = readlines(p)
    @test text[1] == "time,regime_sequence,starting_period_of_regime" && length(text) == 4
    # Dynare labels rows with `periods` but fills them from the first regimes
    @test regime_table(rh; periods=[3])[1]["regime_sequence"] == "0  1  0"
    @test regime_table(rh; periods=[3], dynare_indexing=true)[1]["regime_sequence"] == "0"
end

@testset "dynare(): OccBin estimation(smoother) fills oo_ as Dynare" begin
    s, r = oref("occest_nkm_smoother"), oref("occest_nkm_lik")
    names = split(r["names"], ';')
    x0 = vec(s["x0"])
    src = read(joinpath(OREF, "occest_nkm.mod"), String)
    i = findfirst("occbin_setup(likelihood_piecewise_kalman_filter);", src)
    init = "estimated_params_init;\n" * join(["$(nm), $(@sprintf("%.17g", x));" for (nm, x) in zip(names, x0)], "\n") * "\nend;\n"
    src = src[1:first(i)-1] * init * "options_.occbin.likelihood.maxit = 31;\n" *
          "estimation(datafile=dataobsfile, mode_compute=0, nobs=120, first_obs=1, mh_replic=0, plot_priors=0, nograph, smoother) yg inom pi;\n"
    d = mktempdir()
    write(joinpath(d, "occ_dyn.mod"), src)
    cp(joinpath(OREF, "occest_nkm_data.csv"), joinpath(d, "dataobsfile.csv"))
    ws = with_logger(NullLogger()) do; dynare(joinpath(d, "occ_dyn.mod")); end
    oo = ws.oo_
    @test oo.Smoother.occbin == true
    @test JDyn._occbin_overrides(ws)["likelihood.maxit"] == 31
    m = ws.model
    SV = hcat([vec(oo.SmoothedVariables[v]) for v in m.endo_names]...)
    @test mxd(SV, s["pkf_SmoothedVariables"]) < 1e-10
    SS = hcat([vec(oo.SmoothedShocks[v]) for v in m.exo_names]...)
    @test mxd(SS, s["pkf_SmoothedShocks"]) < 1e-10
    UV = hcat([vec(oo.UpdatedVariables[v]) for v in m.endo_names]...)
    @test mxd(UV, s["pkf_UpdatedVariables"]) < 1e-10
    occ = oo.occbin.smoother
    @test occ.error_flag == 0 && occ.warning_flag == 0
    rh = dregimes(s["pkf_regimes"])
    @test occ.regime_history.shape == (1, length(rh))
    @test all(vec(occ.regime_history.items[t].regime) == rh[t][1][1] && vec(occ.regime_history.items[t].regimestart) == rh[t][1][2] for t in eachindex(rh))
    @test mxd(oo.occbin.linear_smoother.alphahat, s["pkf_linear_alphahat"]) < 1e-10
    @test mxd(oo.occbin.linear_smoother.etahat, s["pkf_linear_etahat"]) < 1e-10
    @test mxd(occ.alphahat, Matrix((s["pkf_SmoothedVariables"] .- vec(occ.SteadyState)')[:, Int.(vec(r["order_var"]))]')) < 1e-10
    # re-run on the workspace with the inversion filter as smoother
    estimation!(ws, ["yg"]; mode_compute=0, smoother=true)
    @test mxd(hcat([vec(ws.oo_.SmoothedVariables[v]) for v in m.endo_names]...), s["pkf_SmoothedVariables"]) < 1e-10
end

@testset "dynare(): occbin_solver regime history and occbin_write_regimes" begin
    src = read(joinpath(OREF, "occest_rbc.mod"), String) * "\noccbin_write_regimes(periods=[1 2 50], filename='rbc_regimes');\n"
    d = mktempdir()
    write(joinpath(d, "occ_wr.mod"), src)
    ws = with_logger(NullLogger()) do; dynare(joinpath(d, "occ_wr.mod")); end
    m = ws.model
    ob = Occbin(m)
    res = with_logger(NullLogger()) do; simulate(ob); end
    rh = ws.oo_.occbin.simul.regime_history
    @test rh.shape == (1, length(res.regime_history))
    @test JDyn._regimes_from_cell(rh) == res.regime_history
    @test haskey(rh.items[1], "regime2") && !haskey(rh.items[1], "regime")
    lines = readlines(joinpath(d, "Output", "rbc_regimes.csv"))
    @test lines[1] == "time,regime_sequence_1,starting_period_of_regime_1,regime_sequence_2,starting_period_of_regime_2"
    @test length(lines) == 4 && startswith(lines[4], "50,")
    reg50 = res.regime_history[50]
    @test lines[4] == "50," * join([join(string.(reg50[1][1]), "  "), join(string.(reg50[1][2]), "  "), join(string.(reg50[2][1]), "  "),
                                    join(string.(reg50[2][2]), "  ")], ",")
    # Dynare 7.2 fills the rows with the first regimes (row index instead of period)
    ws.options_.replicate_dynare_bugs = true
    occbin_write_regimes!(ws; type="simul")
    lines2 = readlines(joinpath(d, "Output", "rbc_regimes.csv"))
    @test startswith(lines2[4], "50,") && lines2[4] != lines[4] || res.regime_history[3] == reg50
    @test_throws ModelError occbin_write_regimes!(ws; type="smoother")       # no smoother run: Dynare errors too
    # loglinear with occbin_solver: refused (Dynare 7.2 mixes levels and logs there)
    ws.options_.loglinear = true
    @test_throws ModelError occbin_solver!(ws)
    ws.options_.loglinear = false
end
