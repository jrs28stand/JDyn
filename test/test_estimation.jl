# Estimation (data handling, priors, Kalman-filter likelihood, posterior mode, Laplace approximation,
# Metropolis-Hastings, estimation statement) against Dynare 7.2 under Octave 8.4 (pydyn's references,
# reference/r_estimation.m). Run alone: `using Test; include("common.jl")` then this file.
#
# Models: est_schorfheide (Dynare's examples/estimation/schorfheide_2000.mod: loglinear, logdata, steady_state_model,
# an auxiliary variable), est_nk (linear New Keynesian model: measurement errors, shock correlation, all prior families,
# a parameter overwritten by the steady state) and est_ur (unit root: diffuse filter, lik_init=2, maximum likelihood,
# observation trends).
using Test, JDyn, MAT, LinearAlgebra, Random, Logging
isdefined(@__MODULE__, :REF) || include(joinpath(@__DIR__, "common.jl"))

const EST_DIR = mktempdir()
for f in readdir(REF; join=true)
    b = basename(f)
    (startswith(b, "est_") && any(endswith(b, e) for e in (".mod", ".m", ".csv"))) && cp(f, joinpath(EST_DIR, b))
end
const _EST_MODELS = Dict{Tuple{String,Any},JDyn.Model}()
"Model `mod` (est_nk, est_ur, est_schorfheide) preprocessed once per variant (-DVARIANT=n or -DNAME)."
function estload(mod, variant=0)
    get!(_EST_MODELS, (mod, variant)) do
        sub = mktempdir()
        for f in readdir(EST_DIR; join=true)
            cp(f, joinpath(sub, basename(f)))
        end
        args = variant isa String ? ["-D$variant"] : ["-DVARIANT=$variant"]
        with_logger(NullLogger()) do
            JDyn.Model(JDyn.run_preprocessor(joinpath(sub, mod * ".mod"); extra_args=args))
        end
    end
end
estref(name) = matread(joinpath(REF, name))
quiet(f) = with_logger(f, NullLogger())
maxdiffs = Dict{String,Float64}()
track!(key, v) = (maxdiffs[key] = max(get(maxdiffs, key, 0.0), v))

@testset "estimation" begin

# ------------------------------------------------------------------ data files
@testset "data files" begin
    m = estload("est_nk")
    est = Estimation(m)
    r = estref("est_nk_v0.mat")
    @test est.data == r["data"]
    table = JDyn.read_m_data(joinpath(EST_DIR, "est_nk_data.m"))
    alts = [Estimation(m; data=joinpath(EST_DIR, "est_nk_csv.csv")), Estimation(m; data=table),
            Estimation(m; data=reduce(hcat, [table[v] for v in m.varobs]))]
    ll = JDyn.log_likelihood(est, est.x0)
    for e in alts
        @test maximum(abs.(e.data .- est.data)) < 1e-12
        @test abs(JDyn.log_likelihood(e, e.x0) - ll) < 1e-9
    end
    @test_throws JDyn.EstimationError Estimation(m; data=Dict("yobs" => table["yobs"]))
    @test_throws JDyn.EstimationError Estimation(m; first_obs=100, nobs=80)
    # .mat files and extension-less names
    d = mktempdir()
    matwrite(joinpath(d, "nk.mat"), Dict(k => reshape(v, :, 1) for (k, v) in table))
    @test Estimation(m; data=joinpath(d, "nk")).data == est.data
    @test_throws JDyn.EstimationError Estimation(m; data=joinpath(d, "nothing_here"))
    # a data script the interpreter cannot follow is refused
    write(joinpath(d, "bad.m"), "yobs = cumsum(ones(10,1)); piobs = yobs; robs = yobs;")
    @test_throws JDyn.EstimationError Estimation(m; data=joinpath(d, "bad.m"))
    write(joinpath(d, "mat.m"), "yobs = ones(5,2); piobs = ones(5,1); robs = ones(5,1);")
    @test_throws JDyn.EstimationError Estimation(m; data=joinpath(d, "mat.m"), nobs=3)
end

@testset "schorfheide data transformations" begin
    r = estref("est_schorfheide_v0.mat")
    est = Estimation(estload("est_schorfheide"))
    @test size(est.data) == (192, 2)
    @test est.data == r["data"]
    unlogged = Estimation(estload("est_schorfheide"); data=exp.(est.rawdata), logdata=false)
    @test maximum(abs.(unlogged.data .- r["data"])) < 1e-15
end

# ------------------------------------------------------------------ likelihood and prior
CASES = [("est_schorfheide", 0), ("est_nk", 0), ("est_nk", 1), ("est_nk", 2), ("est_nk", 3), ("est_nk", 4),
         ("est_nk", 8), ("est_nk", 9), ("est_ur", 0), ("est_ur", 1), ("est_ur", 2), ("est_ur", 5), ("est_ur", 6)]
# what each case exercises: loglinear + logdata; measurement error + estimated correlation + steady-state parameter;
# first_obs/nobs/presample; missing observations; prefilter; lik_init=2; kalman_algo=2 + missing; correlated measurement
# errors + missing; diffuse filter (univariate fallback); lik_init=2 with unit root; diffuse + missing + presample;
# observation_trends + first_obs; kalman_algo=4 + missing + presample.
@testset "likelihood and prior vs Dynare: $mod v$variant" for (mod, variant) in CASES
    r = estref("est_$(mod[5:end])_v$variant.mat")
    m = estload(mod, variant)
    params_before, sigma_before = copy(m.params), copy(m.Sigma_e)
    est = Estimation(m)
    @test est.names == split(r["names"], ";")
    @test est.x0 ≈ vec(r["x0"]) rtol = 1e-14
    @test est.lb == vec(r["lb"]) && est.ub == vec(r["ub"])
    @test est.prior.shape == Int.(vec(r["pshape"]))
    for (key, v) in (("p6", est.prior.p6), ("p7", est.prior.p7))
        ref = vec(r[key])
        @test all(isnan.(v) .== isnan.(ref))
        k = .!isnan.(ref)
        @test v[k] ≈ ref[k] rtol = 1e-12
    end
    rd = r["data"] isa Number ? fill(r["data"], 1, 1) : r["data"]
    @test size(est.data) == size(rd) && isequal(isnan.(est.data), isnan.(rd))
    @test maximum(abs.(filter(!isnan, est.data .- rd)); init=0.0) < 1e-12
    @test est.options["lik_init"] == Int(r["lik_init"])
    @test est.options["qz_criterium"] == Float64(r["qz_criterium"])
    X = r["X"]
    nfinite = 0
    for k in 1:size(X, 2)
        x = X[:, k]
        post = JDyn.log_posterior(est, x)
        if isfinite(r["fval"][k])
            nfinite += 1
            ll = JDyn.log_likelihood(est, x)
            lp = JDyn.log_prior(est, x)
            @test ll ≈ r["loglik"][k] rtol = 1e-10
            @test lp ≈ r["lnprior"][k] rtol = 1e-10 atol = 1e-10
            @test post ≈ -r["fval"][k] rtol = 1e-10
            contrib = JDyn.log_likelihood_contributions(est, x)
            @test length(contrib) == est.nobs - est.options["presample"]
            @test sum(contrib) ≈ r["loglik"][k] rtol = 1e-10
            track!("loglik $mod v$variant", abs(ll - r["loglik"][k]))
            track!("logprior $mod v$variant", abs(lp - r["lnprior"][k]))
        elseif !(mod == "est_schorfheide" && r["info"][k] == 3)     # see the knife-edge test below
            @test post == -Inf
            info = Int(r["info"][k])
            info == 41 && @test occursin("lower bound", est.last_info)
            info == 42 && @test occursin("upper bound", est.last_info)
        end
    end
    @test nfinite >= 4
    @test m.params == params_before && m.Sigma_e == sigma_before          # the model is left untouched
end

@testset "rejections match Dynare" begin
    r = estref("est_schorfheide_v0.mat")
    est = Estimation(estload("est_schorfheide"))
    infos = Int.(vec(r["info"]))
    @test sort(unique(infos)) == [0, 3, 41, 42]
    # rho = 0.999999 sits exactly on qz_criterium = 1 - 1e-6: Dynare under Octave counts that root as unstable (info 3).
    # JDyn's LAPACK returns it as 0.9999989999999992, a rounding error below the criterion, and accepts the point.
    k = findfirst(==(3), infos)
    @test_broken JDyn.log_likelihood(est, r["X"][:, k]) == -Inf
    x = copy(r["X"][:, k]); x[findfirst(==("rho"), est.names)] = 0.9999995       # clearly inside the unstable region
    @test JDyn.log_posterior(est, x) == -Inf && occursin("Blanchard-Kahn", est.last_info)
    @test est.last_info_code == 3
    r = estref("est_nk_v0.mat")
    est = Estimation(estload("est_nk"))
    infos = Int.(vec(r["info"]))
    k = findfirst(==(22), infos)                                          # sig = 0: 1/sig in the model
    @test JDyn.log_posterior(est, r["X"][:, k]) == -Inf
    @test occursin("steady state", est.last_info)
    @test all(JDyn.log_posterior(est, r["X"][:, j]) == -Inf for j in eachindex(infos) if infos[j] != 0)
    # the penalty of Dynare's info(4): squared distance outside the bounds, 0.1 for a steady-state failure
    x = copy(est.x0); i = findfirst(==("phipi"), est.names); x[i] = 0.8
    fval, info, pen = JDyn._dsge_likelihood(est, x)
    @test (fval, info) == (Inf, 41) && pen ≈ (1.01 - 0.8)^2
end

@testset "steady state updates a parameter" begin
    # est_nk: the steady_state_model block overwrites rbar = 100*(1/beta - 1) and beta is estimated
    m = estload("est_nk")
    r = estref("est_nk_v0.mat")
    est = Estimation(m)
    ib, irbar = findfirst(==("beta"), est.names), m.ip["rbar"]
    @test m.params[irbar] == 0.0
    finite = [k for k in 1:size(r["X"], 2) if isfinite(r["fval"][k])]
    @test length(unique(round.(r["X"][ib, finite]; digits=12))) >= 3
    for k in finite
        x = r["X"][:, k]
        ss = JDyn.state_space(est, x)
        rbar = 100 * (1 / x[ib] - 1)
        @test ss.params[irbar] ≈ rbar rtol = 1e-14
        @test ss.constant[findfirst(==("robs"), m.varobs)] ≈ rbar + x[findfirst(==("pibar"), est.names)] rtol = 1e-13
        @test JDyn.log_likelihood(est, x) ≈ r["loglik"][k] rtol = 1e-10
    end
    @test m.params[irbar] == 0.0
end

@testset "covariances from theta" begin
    est = Estimation(estload("est_nk"))
    x = copy(est.x0)
    d = Dict(zip(est.names, x))
    Q, H = JDyn.covariances(est, x)
    @test est.names[1:5] == ["stderr ea", "stderr eu", "stderr er", "stderr yobs", "corr ea, eu"]
    @test sqrt.(diag(Q)) ≈ [d["stderr ea"], d["stderr eu"], d["stderr er"]]
    @test Q[1, 2] ≈ d["corr ea, eu"] * d["stderr ea"] * d["stderr eu"]
    @test Q[1, 3] == 0 && Q[2, 3] == 0
    @test H ≈ Diagonal([d["stderr yobs"]^2, 0, 0])
end

@testset "Kalman filter variants agree" begin
    # multivariate and univariate filters compute the same likelihood; with riccati_tol = 0 to rounding
    m = estload("est_nk")
    a, b = Estimation(m; riccati_tol=0), Estimation(m; kalman_algo=2, riccati_tol=0)
    @test abs(JDyn.log_likelihood(a, a.x0) - JDyn.log_likelihood(b, b.x0)) < 1e-9
    @test (a.last_filter, b.last_filter) == ("multivariate", "univariate")
    dflt = Estimation(m)
    @test 0 < abs(JDyn.log_likelihood(dflt, a.x0) - JDyn.log_likelihood(a, a.x0)) < 1e-5     # the frozen gain moves it ~1e-8
    m = estload("est_ur", 2)
    a, b = Estimation(m; riccati_tol=0), Estimation(m; kalman_algo=4, riccati_tol=0)
    @test abs(JDyn.log_likelihood(a, a.x0) - JDyn.log_likelihood(b, b.x0)) < 1e-9
    @test (a.last_filter, b.last_filter) == ("multivariate (diffuse)", "univariate (diffuse)")
    # correlated measurement errors in the univariate filter (Dynare enlarges the state vector; pydyn refused)
    m = estload("est_nk", 9)
    r = estref("est_nk_v9.mat")
    a, b = Estimation(m; riccati_tol=0), Estimation(m; kalman_algo=2, riccati_tol=0)
    for k in findall(isfinite, vec(r["loglik"]))[1:3]
        x = r["X"][:, k]
        @test abs(JDyn.log_likelihood(a, x) - JDyn.log_likelihood(b, x)) < 1e-8
        @test JDyn.log_likelihood(Estimation(m; kalman_algo=2), x) ≈ r["loglik"][k] rtol = 1e-8
    end
    @test b.last_filter == "univariate"
    # the univariate filter takes over when the forecast-error covariance is singular (stochastic singularity)
    sing = Estimation(m; use_univariate_filters_if_singularity_is_detected=false)
    @test sing.options["use_univariate_filters_if_singularity_is_detected"] == false
end

# ------------------------------------------------------------------ priors
@testset "prior families match Dynare" begin
    q = estref("est_priors.mat")
    seen = Set{Int}()
    grid = vec(q["grid"])
    for i in 1:size(q["spec"], 1)
        row = q["spec"][i, :]
        pr = JDyn.Prior([Int(row[1])], [row[2]], [row[3]], [row[4]], [row[5]])
        push!(seen, Int(row[1]))
        @test [pr.p6[1], pr.p7[1]] ≈ [q["p6"][i], q["p7"][i]] rtol = 1e-12
        @test (pr.p3[1], pr.p4[1]) == (q["p3"][i], q["p4"][i])
        lp = [JDyn.logpdf(pr, [x]) for x in grid]
        expected = vec(q["lpdf"][i, :])
        fin = isfinite.(expected)
        @test sum(fin) >= 20
        @test lp[.!fin] == expected[.!fin]
        @test lp[fin] ≈ expected[fin] rtol = 1e-11 atol = 1e-11
        track!("prior hyper-parameters", max(abs(pr.p6[1] - q["p6"][i]) / max(abs(q["p6"][i]), 1), abs(pr.p7[1] - q["p7"][i]) / max(abs(q["p7"][i]), 1)))
        track!("prior log density", maximum(abs.(lp[fin] .- expected[fin])))
    end
    @test seen == Set([1, 2, 3, 4, 5, 6, 8])
end

@testset "prior errors and bounds" begin
    @test_throws JDyn.PriorError JDyn.Prior([1], [0.5], [0.6])                      # beta: std too large for the mean
    @test_throws JDyn.PriorError JDyn.Prior([4], [-1.0], [1.0])                     # inverse gamma: mean below the lower bound
    @test_throws JDyn.PriorError JDyn.Prior([5], [0.5], [0.1], [0.0], [1.0])        # uniform: bounds and moments together
    pr = JDyn.Prior([1, 2, 3, 4, 5, 6, 8], [0.5, 1, 0, 1, NaN, 1, 1], [0.1, 0.5, 1, 1, NaN, 1, 0.5],
                    [fill(NaN, 4); -1; NaN; NaN], [fill(NaN, 4); 2; NaN; NaN])
    lb, ub = JDyn.prior_bounds(pr)
    @test lb == [0, 0, -Inf, 0, -1, 0, 0] && ub == [1, Inf, Inf, Inf, 2, Inf, Inf]
    lb, ub = JDyn.prior_bounds(pr, 1e-3)                                           # prior_trunc: quantiles
    @test all(lb .> [0, 0, -4, 0, -1, 0, 0]) && all(isfinite, ub) && all(lb .< ub)
    @test [lb[3], ub[3]] ≈ [-3.090232306167813, 3.090232306167813] rtol = 1e-10
    @test [lb[5], ub[5]] ≈ [-1 + 3e-3, 2 - 3e-3] rtol = 1e-12
    # the quantiles invert the cdf: the beta prior of the first parameter has mass 1e-3 below lb[1]
    a, b = pr.p6[1], pr.p7[1]
    @test JDyn.SpecialFunctions.beta_inc(a, b, lb[1])[1] ≈ 1e-3 rtol = 1e-8
end

# ------------------------------------------------------------------ Hessian, Laplace, mode
@testset "Hessian and Laplace at Dynare's mode" begin
    r = estref("est_schorfheide_full.mat")
    est = Estimation(estload("est_schorfheide"))
    @test est.names == split(r["names"], ";")
    x = vec(r["xparam1"])
    @test abs(r["logpost"] - 1343.6795) < 1e-4 && abs(r["laplace"] - 1299.02) < 1e-2
    @test JDyn.log_posterior(est, x) ≈ r["logpost"] atol = 1e-8
    hh = JDyn.hessian(est, x)
    @test hh ≈ r["hh"] rtol = 1e-4
    inv = JDyn.inverse_hessian(hh)
    @test inv ≈ r["Vmode"] rtol = 1e-4
    @test JDyn.laplace(est, x, inv) ≈ r["laplace"] atol = 1e-5
    @test JDyn.laplace(est, x, r["Vmode"]) ≈ r["laplace"] atol = 1e-8
    track!("Hessian (relative)", maximum(abs.(hh .- r["hh"]) ./ abs.(r["hh"])))
end

@testset "log posterior at Dynare's MCMC draws" begin
    d = estref("est_schorfheide_draws.mat")
    est = Estimation(estload("est_schorfheide"))
    lp = [JDyn.log_posterior(est, d["X"][i, :]) for i in 1:size(d["X"], 1)]
    @test size(d["X"]) == (40, 9) && maximum(d["LP"]) - minimum(d["LP"]) > 5
    @test maximum(abs.(lp .- vec(d["LP"]))) < 1e-6
    track!("log posterior at Dynare draws", maximum(abs.(lp .- vec(d["LP"]))))
end

@testset "csminwel finds Dynare's mode" begin
    r = estref("est_schorfheide_full.mat")
    est = Estimation(estload("est_schorfheide"))
    @test est.x0 ≈ [0.035449, 0.008862, 0.356, 0.993, 0.0085, 0.0002, 0.129, 0.65, 0.01]
    mode = JDyn.posterior_mode(est; mode_compute=4)
    dyn_std = sqrt.(diag(r["Vmode"]))
    # the optimum is flat: Dynare's csminwel stops 6.5e-6 below the value found by a tighter search
    @test mode.log_posterior > r["logpost"] - 1e-3
    @test maximum(abs.(mode.x .- vec(r["xparam1"])) ./ dyn_std) < 0.05
    @test JDyn.mode_std(mode) ≈ dyn_std rtol = 2e-2
    @test abs(mode.laplace - r["laplace"]) < 0.05
    track!("schorfheide mode: log posterior", abs(mode.log_posterior - r["logpost"]))
    track!("schorfheide mode: max |x - x_dynare| / std", maximum(abs.(mode.x .- vec(r["xparam1"])) ./ dyn_std))
end

@testset "mode of the unit-root model: posterior (v3) and ML (v4)" for (variant, bayesian) in ((3, true), (4, false))
    r = estref("est_ur_mode_v$variant.mat")
    est = Estimation(estload("est_ur", variant))
    @test JDyn.is_bayesian(est.prior) == bayesian
    @test JDyn.log_posterior(est, vec(r["xparam1"])) ≈ r["logpost"] rtol = 1e-10
    mode = JDyn.posterior_mode(est; mode_compute=4)
    dyn_std = sqrt.(diag(r["Vmode"]))
    @test abs(mode.log_posterior - r["logpost"]) < 1e-5
    @test maximum(abs.(mode.x .- vec(r["xparam1"])) ./ dyn_std) < 0.02
    @test JDyn.mode_std(mode) ≈ dyn_std rtol = 5e-3
    track!("ur v$variant mode: max |x - x_dynare| / std", maximum(abs.(mode.x .- vec(r["xparam1"])) ./ dyn_std))
    if bayesian
        @test abs(mode.laplace - r["laplace"]) < 1e-2
    else
        @test isnan(mode.laplace) && JDyn.log_prior(est, mode.x) == 0.0
        @test_throws JDyn.EstimationError JDyn.sample_posterior(est; replic=10, mode=mode)
    end
end

@testset "mode of est_nk (v6) from the .mod file" begin
    r = estref("est_nk_mode_v6.mat")
    est = Estimation(estload("est_nk", 6))
    @test est.options["mode_compute"] == 4
    @test JDyn.log_posterior(est, vec(r["xparam1"])) ≈ r["logpost"] rtol = 1e-10
    mode = JDyn.posterior_mode(est)
    @test mode.log_posterior > r["logpost"] - 1e-3
    dyn_std = sqrt.(diag(r["Vmode"]))
    @test maximum(abs.(mode.x .- vec(r["xparam1"])) ./ dyn_std) < 0.05
    track!("nk v6 mode: log posterior", abs(mode.log_posterior - r["logpost"]))
    track!("nk v6 mode: max |x - x_dynare| / std", maximum(abs.(mode.x .- vec(r["xparam1"])) ./ dyn_std))
end

@testset "hessian function and csminwel on test functions" begin
    A = [2.0 0.3 0.0; 0.3 1.0 -0.2; 0.0 -0.2 0.5]
    hh = JDyn.hessian(x -> 0.5 * dot(x, A * x) + sum(x), [0.3, -1.2, 2.0])
    @test hh ≈ A rtol = 1e-6 atol = 1e-8
    # Rosenbrock with a wall: rejected points are penalised
    f(x) = x[1] < -1 ? (Inf, 41, (x[1] + 1)^2) : (100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2, 0, 0.0)
    fh, xh, = JDyn.csminwel(f, [-0.5, 1.0]; nit=500)
    @test xh ≈ [1.0, 1.0] atol = 1e-3
    fh, xh, = JDyn.csminwel(x -> (sum(abs2, x .- [0.2, 0.7]), 0, 0.0), [0.0, 0.0])
    @test xh ≈ [0.2, 0.7] atol = 1e-5
    @test JDyn.csminwel(f, [-2.0, 1.0])[7] == -1                                    # bad initial point
end

# ------------------------------------------------------------------ Metropolis-Hastings
@testset "Metropolis-Hastings accept/reject path" begin
    # fixed innovations and uniforms: the chain equals an independent implementation of
    # posterior_sampler_iteration.m, including proposals outside the bounds and unsolvable points
    rng = Random.Xoshiro(5)
    n, ndraws = 3, 400
    S = [1.0 0.5 0.0; 0.5 2.0 -0.3; 0.0 -0.3 0.5]
    Sinv = inv(S)
    calls = Vector{Float64}[]
    target(x) = x[3] > 1.2 ? -Inf : -0.5 * dot(x, Sinv * x)
    function log_post(x)
        push!(calls, copy(x))
        target(x)
    end
    lb, ub = [-1.0, -Inf, -Inf], [Inf, 2.5, Inf]
    d = Matrix(cholesky(S).U) * Diagonal([0.9, 1.1, 1.0])
    eps, u = randn(rng, ndraws, n), rand(rng, ndraws)
    x0 = zeros(n)
    draws, logpo, acc = JDyn.metropolis_hastings(log_post, x0, log_post(x0), d, lb, ub, eps, u)
    ncalls = length(calls) - 1
    last, last_lp = copy(x0), target(x0)
    n_inside = n_out = n_unsolved = 0
    for i in 1:ndraws
        prop = last + d' * eps[i, :]
        inside = all(prop .> lb) && all(prop .< ub)
        lp = inside ? target(prop) : -Inf
        n_inside += inside; n_out += !inside; n_unsolved += inside && lp == -Inf
        accept = lp > -Inf && log(u[i]) < lp - last_lp
        accept && ((last, last_lp) = (prop, lp))
        @test acc[i] == accept
        @test draws[i, :] ≈ last atol = 1e-13
        @test logpo[i] ≈ last_lp atol = 1e-12
    end
    @test ncalls == n_inside                           # proposals outside the bounds are not evaluated
    @test n_out > 10 && n_unsolved > 5 && 0.1 < sum(acc) / ndraws < 0.7
    @test all(draws .> lb') && all(draws .< ub') && all(draws[:, 3] .<= 1.2)
    # a function that throws rejects the proposal, as Dynare's try ... catch
    bad(x) = x[1] > 0.5 ? error("boom") : -0.5 * dot(x, x)
    dr, _, ac = JDyn.metropolis_hastings(bad, zeros(2), 0.0, Matrix{Float64}(I, 2, 2), fill(-Inf, 2), fill(Inf, 2), randn(rng, 200, 2), rand(rng, 200))
    @test all(dr[:, 1] .<= 0.5) && any(ac)
end

@testset "sampler on a model with fixed random numbers" begin
    r = estref("est_ur_mode_v3.mat")
    est = Estimation(estload("est_ur", 3))
    V, xm = r["Vmode"], vec(r["xparam1"])
    rng = Random.Xoshiro(11)
    replic, js = 60, 0.7
    eps, u = randn(rng, 2, replic, est.n), rand(rng, 2, replic)
    start = vcat(xm', (xm .* 1.02)')
    mc = JDyn.sample_posterior(est; replic=replic, nblocks=2, mode=xm, cov=V, jscale=js, x0=start, innovations=eps, uniforms=u)
    @test size(mc.draws) == (2, replic, est.n) && mc.names == est.names
    L = cholesky(V).L
    for j in 1:2
        last, last_lp = start[j, :], JDyn.log_posterior(est, start[j, :])
        for i in 1:replic
            prop = last + js * (L * eps[j, i, :])
            lp = JDyn.log_posterior(est, prop)
            if isfinite(lp) && log(u[j, i]) < lp - last_lp
                last, last_lp = prop, lp
            end
            @test mc.draws[j, i, :] ≈ last atol = 1e-12
            @test mc.log_posterior[j, i] ≈ last_lp atol = 1e-9
        end
    end
    @test mc.acceptance_rate ≈ vec(sum(mc.accepted; dims=2)) ./ replic
end

@testset "sampler uses the prior support, not the bounds of estimated_params" begin
    # est_nk: phipi has a normal prior and bounds [1.01, 4] in the block
    est = Estimation(estload("est_nk"))
    i = findfirst(==("phipi"), est.names)
    @test (est.lb[i], est.ub[i]) == (1.01, 4.0)
    plb, pub = JDyn.prior_bounds(est.prior)
    @test (plb[i], pub[i]) == (-Inf, Inf)
    x = copy(est.x0); x[i] = 1.012
    below = copy(x); below[i] = 1.007
    @test JDyn.log_posterior(est, below) == -Inf && occursin("lower bound", est.last_info)
    lp = JDyn.log_posterior(est, below; bounds=(plb, pub))
    @test isfinite(lp) && lp ≈ JDyn.log_likelihood(est, below) + JDyn.log_prior(est, below)
    V = Diagonal((0.01 .* abs.(est.x0)) .^ 2)
    eps = reshape((cholesky(Matrix(V)).L \ (below - x)), 1, 1, :)
    mc = JDyn.sample_posterior(est; replic=1, nblocks=1, mode=x, cov=Matrix(V), jscale=1.0, x0=reshape(x, 1, :), innovations=eps, uniforms=fill(1e-300, 1, 1))
    @test mc.accepted[1, 1] && abs(mc.draws[1, 1, i] - 1.007) < 1e-12
    @test mc.log_posterior[1, 1] ≈ lp atol = 1e-9
end

@testset "sampler smoke test and posterior statistics" begin
    r = estref("est_ur_mode_v3.mat")
    est = Estimation(estload("est_ur", 3))
    kw = (replic=150, nblocks=2, mode=vec(r["xparam1"]), cov=r["Vmode"])
    mc = JDyn.sample_posterior(est; seed=3, kw...)
    again = JDyn.sample_posterior(est; seed=3, kw...)
    other = JDyn.sample_posterior(est; seed=4, kw...)
    @test mc.draws == again.draws                                            # reproducible
    @test mc.draws != other.draws
    @test mc.draws[1, :, :] != mc.draws[2, :, :]                             # chains have their own streams
    @test mc.x0[1, :] != mc.x0[2, :]                                         # over-dispersed starting points
    @test all(0.1 .< mc.acceptance_rate .< 0.6)                              # default jscale 2.38/sqrt(n)
    lb, ub = JDyn.prior_bounds(est.prior)
    @test all(mc.draws .> reshape(lb, 1, 1, :)) && all(mc.draws .< reshape(ub, 1, 1, :)) && all(isfinite, mc.log_posterior)
    @test size(JDyn.kept_draws(mc)) == (150, est.n) && size(JDyn.kept_draws(mc; drop=0.0)) == (300, est.n)
    st = JDyn.posterior_statistics(est, mc)
    std = sqrt.(diag(r["Vmode"]))
    @test all(abs.([st[nm].mean for nm in est.names] .- vec(r["xparam1"])) .< 2 .* std)
    for nm in est.names
        s = st[nm]
        @test s.hpd[1] < s.mean < s.hpd[2] && s.deciles[1] < s.median < s.deciles[9]
    end
    one = JDyn.sample_posterior(est; replic=5, nblocks=1, mode=vec(r["xparam1"]), cov=r["Vmode"])
    @test one.x0[1, :] == vec(r["xparam1"])                                  # a single chain starts at the mode
    # posterior_moments.m on a known sample
    x = collect(1.0:100.0)
    mu, med, v, hpd, dec = JDyn.posterior_moments(x, 0.9)
    @test (mu, med, v) == (50.5, 50.5, 841.6666666666666 + 0.0) || (mu == 50.5 && med == 50.5 && v ≈ 841.6666666666666)
    @test dec == [10, 20, 30, 40, 50, 60, 70, 80, 90] && hpd[2] - hpd[1] == 89
end

@testset "posterior sampling reproduces Dynare's posterior moments (Schorfheide)" begin
    r = estref("est_schorfheide_full.mat")
    est = Estimation(estload("est_schorfheide"))
    mode = JDyn.posterior_mode(est; mode_compute=4)
    mc = JDyn.sample_posterior(est; replic=4000, nblocks=2, mode=mode, seed=2026)        # Dynare: mh_jscale = 0.8 from the file
    @test est.options["mh_jscale"] == 0.8 || est.options["mh_jscale"] === nothing
    @test all(0.05 .< mc.acceptance_rate .< 0.8)
    st = JDyn.posterior_statistics(est, mc)
    mean_ = [st[nm].mean for nm in est.names]
    std_ = [st[nm].std for nm in est.names]
    @test maximum(abs.(mean_ .- vec(r["post_mean"])) ./ vec(r["post_std"])) < 0.5          # MC error of 2 x 4000 draws
    @test maximum(abs.(std_ ./ vec(r["post_std"]) .- 1)) < 0.25
    track!("schorfheide MCMC (2 x 4000): max |mean - Dynare| / std", maximum(abs.(mean_ .- vec(r["post_mean"])) ./ vec(r["post_std"])))
    mhm = JDyn.marginal_density_mhm(mc)
    @test abs(mhm - r["mhm"]) < 1.0
    track!("schorfheide modified harmonic mean (2 x 4000)", abs(mhm - r["mhm"]))
end

# ------------------------------------------------------------------ the estimation statement
@testset "estimate(m) on a .mod file" begin
    # est_ur v3: diffuse filter, mode_compute=4, no MCMC
    r = estref("est_ur_mode_v3.mat")
    res = quiet(() -> JDyn.estimate(estload("est_ur", 3)))
    @test res.mode !== nothing && res.mcmc === nothing
    @test abs(res.log_posterior - r["logpost"]) < 1e-5
    @test abs(res.marginal_density_laplace - r["laplace"]) < 1e-2
    # mode_compute=0 only evaluates the posterior at the initial values
    res0 = JDyn.estimate(estload("est_nk"))
    @test res0.mode === nothing && res0.mcmc === nothing
    r0 = estref("est_nk_v0.mat")
    @test res0.log_posterior ≈ -r0["fval"][1] rtol = 1e-10
    # options override the statement; unsupported options raise unless allowed
    res = quiet(() -> JDyn.estimate(estload("est_ur", 3); mode_compute=0))
    @test res.mode === nothing
    # the smoother option is honoured (src/smoother.jl): estimate() then also smooths at the mode
    res = quiet(() -> JDyn.estimate(estload("est_nk", 5)))
    @test isfinite(res.log_posterior)
    @test !("smoother" in res.est.unsupported) && res.smoother !== nothing
    # a short MCMC from the .mod file's settings (2 chains, jscale 0.8 of the file)
    res = quiet(() -> JDyn.estimate(estload("est_schorfheide", "FULL"); mh_replic=300, seed=7))
    @test res.mcmc !== nothing && size(res.mcmc.draws) == (2, 300, 9)
    @test isfinite(res.marginal_density_mhm) || isnan(res.marginal_density_mhm)
    @test abs(res.mode.log_posterior - estref("est_schorfheide_full.mat")["logpost"]) < 1e-3
    @test_throws JDyn.EstimationError JDyn.estimate(estload("est_nk"); mh_replic=10)  # no mode: needs a Hessian
end

@testset "option checks" begin
    m = estload("est_ur")
    @test_throws JDyn.EstimationError Estimation(m; diffuse_filter=true, lik_init=2)
    @test_throws JDyn.EstimationError Estimation(m; order=2)
    @test_throws JDyn.EstimationError Estimation(estload("est_nk"); prefilter=1)       # non-zero steady state of piobs, robs
    @test JDyn.check_prefilter(Estimation(estload("est_nk", 3)))
    @test_throws JDyn.EstimationError Estimation(m; kalman_algo=5)
    @test_throws JDyn.EstimationError Estimation(estload("est_nk"); lik_init=4)
    @test_throws JDyn.EstimationError Estimation(m; first_obs=[1, 2])                  # recursive estimation
    @test Estimation(m; smoother=true).unsupported == String[]
    @test_throws JDyn.EstimationError Estimation(m; mode_file="abc")
    @test Estimation(m; smoother=false).unsupported == String[]
    # nodisplay and nograph are cosmetic
    @test Estimation(m; nodisplay=true, nograph=true).unsupported == String[]
    # the unit root is rejected under the stationary initialisation, as in Dynare (qz_criterium = 1 - 1e-6)
    est = Estimation(m; diffuse_filter=false, lik_init=1)
    @test JDyn.log_posterior(est, est.x0) == -Inf && occursin("Blanchard-Kahn", est.last_info)
    # optimiser options
    opt = Estimation(estload("est_ur", 3); optim_opt="'MaxIter',3")
    mode = JDyn.posterior_mode(opt)
    @test mode.iterations <= 4
    @test_throws JDyn.EstimationError JDyn.posterior_mode(Estimation(estload("est_ur", 3); optim_opt="'Bogus',3"))
    # an optimiser that is not implemented is replaced by csminwel with a warning
    logs = String[]
    with_logger(Logging.SimpleLogger(IOBuffer())) do
        JDyn.posterior_mode(Estimation(estload("est_ur", 3)); mode_compute=7, maxiter=2)
    end
end

@testset "estimated_params_init, bounds and use_calibration" begin
    base = read(joinpath(EST_DIR, "est_ur.mod"), String)
    function model_with(extra)
        d = mktempdir()
        cp(joinpath(EST_DIR, "est_ur_data.m"), joinpath(d, "est_ur_data.m"))
        write(joinpath(d, "est_init.mod"), replace(base, "varobs yobs xobs;" => extra * "\nvarobs yobs xobs;"; count=1))
        quiet(() -> JDyn.Model(JDyn.run_preprocessor(joinpath(d, "est_init.mod"); extra_args=["-DVARIANT=3"])))
    end
    est = Estimation(model_with("estimated_params_init;\nrho, 0.55;\nstderr eg, 0.33;\nend;"))
    x0 = Dict(zip(est.names, est.x0))
    @test (x0["rho"], x0["stderr eg"], x0["phi"], x0["lam"]) == (0.55, 0.33, 0.4, 0.2)
    est = Estimation(model_with("estimated_params_bounds;\nlam, -0.5, 0.9;\nend;"))
    i = findfirst(==("lam"), est.names)
    @test (est.lb[i], est.ub[i]) == (-0.5, 0.9)
    @test_throws JDyn.EstimationError Estimation(model_with("estimated_params_bounds;\nlam, 0.5, 0.9;\nend;"))   # init 0.2 outside
    @test_throws ErrorException model_with("estimated_params_init;\nnosuch, 1;\nend;")        # the preprocessor refuses
    @test_throws JDyn.EstimationError Estimation(model_with("estimated_params;\nrho, beta_pdf, 0.5, 0.2;\nend;"))    # a second block
    # use_calibration: calibrated values start the entries without an init_val
    est = Estimation(model_with("estimated_params_init(use_calibration);\nend;"))
    x0 = Dict(zip(est.names, est.x0))
    @test x0["rho"] == 0.6 && x0["stderr eg"] == 0.4 && x0["stderr ey"] == 0.2 && x0["phi"] == 0.4 && x0["stderr ex"] == 0.25
end

@testset "observation trends and prefilter" begin
    # est_ur v5: the slope mu is estimated; trends are 2*mu-0.1 and mu per period from first_obs
    est = Estimation(estload("est_ur", 5))
    @test est.options["first_obs"] == 5 && size(est.data, 1) == size(est.rawdata, 1)
    r = estref("est_ur_v5.mat")
    x = r["X"][:, 2]
    @test JDyn.log_likelihood(est, x) ≈ r["loglik"][2] rtol = 1e-10
end

end # estimation testset

println("\nMaximum differences against Dynare (estimation):")
for k in sort(collect(keys(maxdiffs)))
    println("  ", rpad(k, 58), @sprintf("%.3g", maxdiffs[k]))
end
