# Second-order estimation with Dynare's nonlinear filters, against Dynare 7.2 under Octave 8.4.
#
# References (test/ref_nonlinear; generators r_nonlinear_filters.m, r_octave_rng.m and gen_e2e.m next to them) for the RBC
# model of Dynare's tests/particle (nl_rbc.mod: 50 observations of y, l, i with estimated measurement errors):
#   * nl_rbc_lik.mat    non_linear_dsge_likelihood (minus the log posterior) for 35 filter configurations (sis, apf, cpf, gf,
#                       nlkf and their options) at three parameter vectors, the per-period log likelihoods and Dynare's
#                       reduced form at the first vector;
#   * nl_rbc_seeds.mat  sis and apf (1000 particles) with the filter seed set to 1..30;
#   * nl_rbc_smooth.mat inputs and outputs of multivariate_smooth_resampling in sis;
#   * nl_rbc_simplex.mat Dynare's simplex (mode_compute=8) from the initial values, stopped after 300 calls;
#   * octave_rng.mat    Octave's rand/randn numbers for four seeds;
#   * nl_e2e_v*_{oo,M,opt}.mat  whole `estimation` runs (oo_, M_, options_ fields).
#
# Random numbers: JDyn's OctaveRNG reproduces Octave's generators bit for bit, so the particle filters are compared exactly (same
# draws), not statistically. Run on Dynare's own reduced form the filters agree with Dynare to about 1e-10 per period. Through
# JDyn's own second-order solution, whose coefficients differ from Dynare's by about 1e-12 (rounding in the model derivatives),
# the likelihoods agree to about 1e-9 relative to the sum of the absolute per-period log likelihoods: the measurement errors of
# this model are tiny (standard deviation 0.001), which magnifies every difference in the predicted observables by about 1e6.
#
# Not compared exactly, with the cause:
#   * resampling_method=smooth: Dynare's eig(P'*(P/N)) takes the symmetric or the general eigensolver according to whether the
#     product is exactly symmetric, which depends on BLAS rounding; the function is checked step by step with Dynare's choice;
#   * gf with distribution_approximation=montecarlo and resampling: the particles collapse (all weight on one particle), and
#     whether the Cholesky factorisation of their (numerically zero) covariance succeeds is decided by rounding; compared only
#     where neither stops.
include("common.jl")
include("oo_compare.jl")
using Random, Logging

const NLREF = joinpath(@__DIR__, "ref_nonlinear")
const J = JDyn
nlref(name) = matread(joinpath(NLREF, name))
quiet(f) = with_logger(f, NullLogger())

# name -> filter_algorithm, number_of_particles, keyword options (as in r_nonlinear_filters.m)
const NL_CASES = Dict(
    "gf_uu" => ("gf", 1000, ()), "gf_cu" => ("gf", 1000, (proposal_approximation="cubature",)),
    "gf_uc" => ("gf", 1000, (distribution_approximation="cubature",)),
    "gf_cc" => ("gf", 1000, (proposal_approximation="cubature", distribution_approximation="cubature")),
    "gf_uu_abk" => ("gf", 1000, (unscented_alpha=0.8, unscented_beta=1.0, unscented_kappa=2.0)),
    "nlkf_u" => ("nlkf", 1000, ()), "nlkf_c" => ("nlkf", 1000, (proposal_approximation="cubature",)),
    "nlkf_u_abk" => ("nlkf", 1000, (unscented_alpha=0.8, unscented_beta=1.0, unscented_kappa=2.0)),
    "gf_init3" => ("gf", 1000, (nonlinear_filter_initialization=3, initial_state_prior_std=0.5)),
    "nlkf_m" => ("nlkf", 500, (proposal_approximation="montecarlo",)),
    "gf_mu" => ("gf", 500, (proposal_approximation="montecarlo",)),
    "gf_um" => ("gf", 500, (distribution_approximation="montecarlo",)),
    "gf_mm" => ("gf", 500, (proposal_approximation="montecarlo", distribution_approximation="montecarlo")),
    "gf_um_generic" => ("gf", 500, (distribution_approximation="montecarlo", resampling="generic")),
    "gf_um_none" => ("gf", 500, (distribution_approximation="montecarlo", resampling="none")),
    "sis" => ("sis", 1000, ()), "sis_strat" => ("sis", 1000, (resampling_method="stratified",)),
    "sis_smooth" => ("sis", 1000, (resampling_method="smooth",)),
    "sis_generic" => ("sis", 1000, (resampling="generic",)),
    "sis_generic_t" => ("sis", 1000, (resampling="generic", resampling_threshold=19.0)),
    "sis_generic_low" => ("sis", 1000, (resampling="generic", resampling_threshold=0.001)),
    "sis_none" => ("sis", 1000, (resampling="none",)), "sis_pruning" => ("sis", 1000, (particle_pruning=true,)),
    "sis_init3" => ("sis", 1000, (nonlinear_filter_initialization=3,)),
    "apf" => ("apf", 1000, ()), "apf_none" => ("apf", 1000, (resampling="none",)),
    "apf_strat" => ("apf", 1000, (resampling_method="stratified",)),
    "apf_generic" => ("apf", 1000, (resampling="generic", resampling_threshold=19.0)),
    "apf_generic_low" => ("apf", 1000, (resampling="generic", resampling_threshold=0.001)),
    "apf_pruning" => ("apf", 1000, (particle_pruning=true,)),
    "cpf_u" => ("cpf", 100, ()), "cpf_c" => ("cpf", 100, (proposal_approximation="cubature",)),
    "cpf_m" => ("cpf", 100, (proposal_approximation="montecarlo",)),
    "cpf_u_mjp" => ("cpf", 100, (cpf_weights="murrayjonesparslow",)),
    "cpf_u_none" => ("cpf", 100, (resampling="none",)))
const NL_SMOOTH = Set(["sis_smooth"])
const NL_GF_MC = Set(["gf_um", "gf_mm", "gf_um_generic"])

const NL_DIR = mktempdir()
for f in ("nl_rbc.mod", "nl_rbc_data.m")
    cp(joinpath(NLREF, f), joinpath(NL_DIR, f))
end
const NL_MODEL = quiet(() -> J.Model(J.run_preprocessor(joinpath(NL_DIR, "nl_rbc.mod"))))
const NL_LIK = nlref("nl_rbc_lik.mat")

nl_est(name; kw...) = (alg, n, o = NL_CASES[name]; quiet(() -> Estimation(NL_MODEL; filter_algorithm=alg, number_of_particles=n, particle_rng=:octave, o..., kw...)))

"Dynare's reduced form (the `rf` struct of the reference) as a JDyn NLReducedForm."
function dynare_reduced_form(R)
    a2(v) = v isa Number ? fill(Float64(v), 1, 1) : ndims(v) == 1 ? reshape(Float64.(v), :, 1) : Matrix{Float64}(v)
    vec_(v) = Float64.(vec(v))
    J.NLReducedForm(a2(R["ghx"]), a2(R["ghu"]), a2(R["ghxx"]), a2(R["ghuu"]), a2(R["ghxu"]), vec_(R["ghs2"]), vec_(R["steadystate"]),
                    vec_(R["constant"]), vec_(R["state_variables_steady_state"]), Int.(vec(R["mf0"])), Int.(vec(R["mf1"])), a2(R["Q"]),
                    a2(R["H"]), vec_(R["StateVectorMean"]), a2(R["StateVectorVariance"]))
end

const nl_worst = Dict{String,Float64}()
nl_track!(k, v) = (nl_worst[k] = max(get(nl_worst, k, 0.0), v))

@testset "nonlinear filters (order 2)" begin

@testset "Octave's random number generators, bit for bit" begin
    r = nlref("octave_rng.mat")
    idx = Int.(vec(r["idx"]))
    for (k, seed) in enumerate(vec(r["seeds"]))
        g = J.OctaveRNG(seed)
        u = Random.rand!(g, Vector{Float64}(undef, 200000))
        x = Random.randn!(g, Vector{Float64}(undef, 200000))
        @test u[1:50] == vec(r["u_head_$k"]) && x[1:50] == vec(r["x_head_$k"])
        @test u[idx] == vec(r["u_$k"]) && x[idx] == vec(r["x_$k"])
        @test sum(u) ≈ r["u_sum_$k"] rtol = 1e-14
        @test sum(x) ≈ r["x_sum_$k"] rtol = 1e-12 atol = 1e-9
        @test maximum(abs.(x)) == r["x_max_$k"]            # a draw from the ziggurat's tail
    end
    g = J.OctaveRNG(0)
    a, b, c, d = rand(g, 2, 3), randn(g, 3, 2), rand(g, 1, 4), randn(g, 5, 1)
    @test a == r["a"] && b == r["b"] && c == r["c"] && d == r["d"]
    # the generator state, as rand('state') returns it, and a copy continuing the stream
    g1 = J.OctaveRNG(5)
    rand(g1, 1000); randn(g1, 1000)
    g2 = copy(g1)
    @test rand(g1, 7) == rand(g2, 7) && randn(g1, 7) == randn(g2, 7)
    s = J.octave_state(J.OctaveRNG(0).u)
    @test length(s) == 625 && s[end] == 1.0 && s[1] == Float64(0x80000000) || s[1] > 0     # fresh: left = 1
    @test J.octave_seed!(J.OctaveRNG(3), 3).u.mt == J.OctaveRNG(3).u.mt
    # Octave's randperm
    @test sort(Random.randperm(J.OctaveRNG(1), 10)) == 1:10
end

@testset "streams" begin
    rp = J.ReplayStream(normals=0:5, uniforms=[0.5])
    @test J.stream_randn(rp, 2, 3) == reshape(0.0:5.0, 2, 3)
    @test J.stream_rand1(rp) == 0.5
    @test_throws ErrorException J.stream_rand1(rp)
    a, b = J.stream_randn(J.JuliaStream(3), 4, 5), J.stream_randn(J.JuliaStream(3), 4, 5)
    @test a == b && size(a) == (4, 5)
    m = NL_MODEL
    J.set_rng_kind!(m, :octave); @test J.rng_kind(m) == :octave
    J.set_rng_kind!(m, :julia); @test J.rng_kind(m) == :julia
    @test_throws ArgumentError J.set_rng_kind!(m, :mersenne)
end

@testset "building blocks" begin
    # second-order transition = Dynare's local_state_space_iteration_2 loop (Kronecker products)
    rng = MersenneTwister(0)
    m, n, q, N = 4, 3, 2, 5
    rf = J.NLReducedForm(randn(rng, m, n), randn(rng, m, q), randn(rng, m, n * n), randn(rng, m, q * q), randn(rng, m, n * q), zeros(m),
                         randn(rng, m), randn(rng, m), randn(rng, n), [1, 2, 3], [4], Matrix(1.0I, q, q), Matrix(1.0I, 1, 1), zeros(n), Matrix(1.0I, n, n))
    yh, ep, yp = randn(rng, n, N), randn(rng, q, N), randn(rng, n, N)
    tr = J.Transition(rf, N)
    y = copy(J.iterate_law_of_motion!(tr, yh .+ rf.state_variables_steady_state, ep))
    y_pruned = copy(J.iterate_law_of_motion!(tr, yh .+ rf.state_variables_steady_state, ep, yp .+ rf.state_variables_steady_state))
    ypr = copy(tr.pr)
    for j in 1:N
        e = rf.constant + rf.ghx * yh[:, j] + rf.ghu * ep[:, j] + 0.5 * rf.ghxx * kron(yh[:, j], yh[:, j]) +
            0.5 * rf.ghuu * kron(ep[:, j], ep[:, j]) + rf.ghxu * kron(yh[:, j], ep[:, j])
        @test y[:, j] ≈ e atol = 1e-12
        e2 = rf.constant + rf.ghx * yh[:, j] + rf.ghu * ep[:, j] + 0.5 * rf.ghxx * kron(yp[:, j], yp[:, j]) +
             0.5 * rf.ghuu * kron(ep[:, j], ep[:, j]) + rf.ghxu * kron(yp[:, j], ep[:, j])
        @test y_pruned[:, j] ≈ e2 atol = 1e-12
        @test ypr[:, j] ≈ rf.ghx * yp[:, j] + rf.ghu * ep[:, j] + rf.steadystate atol = 1e-12
    end
    # one state vector for all shock columns
    ys1 = copy(J.iterate_law_of_motion!(tr, reshape(yh[:, 1] .+ rf.state_variables_steady_state, :, 1), ep))
    @test ys1[:, 3] ≈ rf.constant + rf.ghx * yh[:, 1] + rf.ghu * ep[:, 3] + 0.5 * rf.ghxx * kron(yh[:, 1], yh[:, 1]) +
                      0.5 * rf.ghuu * kron(ep[:, 3], ep[:, 3]) + rf.ghxu * kron(yh[:, 1], ep[:, 3]) atol = 1e-12
    # resampling, with Dynare's slips
    w = [0.1, 0.2, 0.3, 0.4]
    idx, c = zeros(Int, 4), zeros(4)
    @test J.kitagawa_indices!(idx, c, w, 0.5) == [2, 3, 4, 4]
    @test J.stratified_indices!(idx, c, w, [0.9, 0.5, 0.9, 0.9]) == [4, 3, 3, 3]
    po = J.ParticleOptions(resampling="generic", resampling_threshold=0.5)
    @test !J.must_resample(po, fill(0.01, 100), 50)              # neff 100 >= 0.5 * 50 periods
    @test J.must_resample(po, [0.99; fill(0.01 / 99, 99)], 50)
    @test J.must_resample(J.ParticleOptions(), [0.5, 0.5], 1)
    @test !J.must_resample(J.ParticleOptions(resampling="none"), [1.0], 1)
    @test_throws J.NonlinearFilterError J.kitagawa_indices!(zeros(Int, 2), zeros(2), [0.1, 0.1], 0.99)
    # sigma points
    nodes, wm, wc = J.unscented_sigma_points(2, 1.0, 2.0, 1.0)
    @test size(nodes) == (5, 2) && sum(wm) ≈ 1 && wc[1] ≈ wm[1] + 2
    nodes, wt = J.spherical_radial_sigma_points(3)
    @test size(nodes) == (6, 3) && wt * 6 ≈ 1 && nodes[1, 1] ≈ sqrt(3)
    # reduced_rank_cholesky: full rank = chol, singular = reduced factor
    X = randn(rng, 4, 6); X = X * X'
    @test J.reduced_rank_cholesky(X) ≈ cholesky(Symmetric(X)).U
    v = randn(rng, 4, 2); S = v * v'
    T = J.reduced_rank_cholesky(S)
    @test size(T, 1) == 2 && T' * T ≈ S atol = 1e-12
    @test_throws J.NonlinearFilterError J.reduced_rank_cholesky([1.0 0; 0 -1])
    # probability2 against the normal density
    Sg = [1 0.5 0; 0.5 1 0; 0 0 0.05]; L = Matrix(cholesky(Sg).L); xx = [2.3, 0.34, 0.87]
    @test J.probability2(zeros(3), L, reshape(xx, 3, 1))[1] ≈ exp(-0.5 * xx' * inv(Sg) * xx) / sqrt((2pi)^3 * det(Sg)) + 1e-99 rtol = 1e-12
end

@testset "reduced form of non_linear_dsge_likelihood" begin
    est = nl_est("gf_uu")
    @test est.names == String.(split(NL_LIK["names"], ";"))
    @test est.data == NL_LIK["data"]
    th = NL_LIK["X"][:, 1]
    Q, H = J.covariances(est, th)
    rf = J._nl_reduced_form(est, th, Q, H)
    R = dynare_reduced_form(NL_LIK["rf"])
    @test rf.mf0 == R.mf0 && rf.mf1 == R.mf1
    for f in (:ghx, :ghu, :ghxx, :ghuu, :ghxu, :ghs2, :constant, :steadystate, :StateVectorMean, :StateVectorVariance, :Q, :H)
        a, b = getfield(rf, f), getfield(R, f)
        d = maximum(abs.(a .- reshape(b, size(a)))) / max(1.0, maximum(abs.(b)))
        nl_track!("reduced form", d)
        @test d < 1e-9
    end
end

@testset "filters on Dynare's reduced form (per-period log likelihoods)" begin
    R = dynare_reduced_form(NL_LIK["rf"])
    Y = Matrix{Float64}(NL_LIK["data"]')
    for name in sort(collect(keys(NL_CASES)))
        (name in NL_SMOOTH || endswith(name, "init3") || name in ("gf_mm", "gf_um_generic")) && continue
        alg, N, kw = NL_CASES[name]
        po = J.ParticleOptions(; filter_algorithm=alg, number_of_particles=N,
                               (k => v for (k, v) in pairs(kw) if k != :nonlinear_filter_initialization && k != :particle_pruning && k != :initial_state_prior_std)...,
                               pruning=get(kw, :particle_pruning, false))
        L = vec(NL_LIK["$(name)_lik"][:, 1])
        f = J.NONLINEAR_FILTERS[alg]
        kws = alg == "cpf" ? (; model_pruning=false) : (;)
        if name == "gf_um"                       # Dynare stops: the particles collapse
            @test NL_LIK["$(name)_info"][1] == -1
            @test_throws J.NonlinearFilterError f(R, Y, 1, po, J.OctaveStream(0))
            continue
        end
        LIK, lik, cplx = f(R, Y, 1, po, J.OctaveStream(0); kws...)
        fin = isfinite.(L)
        @test fin == isfinite.(lik)
        d = maximum(abs.(lik[fin] .- L[fin]))
        nl_track!("filter per period", d)
        @test d < 1e-9
        NL_LIK["$(name)_info"][1] == 0 && @test LIK ≈ -sum(L) atol = 1e-8
    end
end

@testset "likelihood at three parameter vectors (JDyn's own solution)" begin
    X = NL_LIK["X"]
    for name in sort(collect(keys(NL_CASES)))
        name in NL_SMOOTH && continue
        est = nl_est(name)
        fv, info = vec(NL_LIK["$(name)_fval"]), vec(NL_LIK["$(name)_info"])
        L = NL_LIK["$(name)_lik"]
        compared = 0
        for k in 1:size(X, 2)
            x = X[:, k]
            local r
            try
                r = J._nonlinear_dsge_likelihood(est, x)
            catch e
                e isa J.NonlinearFilterError || rethrow()
                @test name in NL_GF_MC                 # collapsed particles
                continue
            end
            if name in NL_GF_MC && (r[2] != 0 || info[k] != 0)
                continue
            end
            @test r[2] == info[k]
            if r[2] == 0
                scale = sum(abs, filter(!isnan, L[:, k]))
                d = abs(r[1] - fv[k]) / scale
                nl_track!("likelihood (relative to sum |lik|)", d)
                @test d < 1e-9
                name in NL_GF_MC || @test -log_posterior(est, x) ≈ fv[k] atol = 1e-9 * scale
                compared += 1
            end
        end
        @test compared >= (name in NL_GF_MC ? 1 : (all(info .!= 0) ? 0 : 3))
    end
end

@testset "filter seeds 1..30 (sis, apf)" begin
    s = nlref("nl_rbc_seeds.mat")
    x0 = vec(s["x0"])
    for alg in ("sis", "apf")
        fvals = vec(s["$(alg)_fval"])
        for (seed, fval) in zip(vec(s["seeds"]), fvals)
            est = quiet(() -> Estimation(NL_MODEL; filter_algorithm=alg, number_of_particles=1000, particle_rng=:octave, particle_seed=Int(seed)))
            got = -log_posterior(est, x0)
            @test got ≈ fval atol = 1e-8 * max(1.0, abs(fval)) + 2e-6
        end
    end
end

@testset "smooth resampling, step by step" begin
    r = nlref("nl_rbc_smooth.mat")
    u = vec(rand(J.OctaveRNG(0), 12, 1))
    for k in 1:6
        P, w, out = Matrix{Float64}(r["P_$k"]), vec(r["w_$k"]), Matrix{Float64}(r["out_$k"])
        got = J.multivariate_smooth_resampling(P, w, J.ReplayStream(uniforms=u[2k-1:2k]); symmetric=r["sym_$k"] != 0)
        d = maximum(abs.(got .- out)) / maximum(abs.(out))
        nl_track!("smooth resampling", d)
        @test d < 1e-9
    end
end

@testset "options: statement and keywords" begin
    d = mktempdir()
    cp(joinpath(NLREF, "nl_rbc_data.m"), joinpath(d, "nl_rbc_data.m"))
    txt = replace(read(joinpath(NLREF, "nl_rbc.mod"), String),
        "estimation(order=2, datafile=nl_rbc_data, nobs=50, number_of_particles=1000," =>
        "estimation(order=2, datafile=nl_rbc_data, nobs=50, number_of_particles=1000, filter_algorithm=apf, resampling_method=stratified, particle_filter_options=('unscented_alpha', 0.5),")
    write(joinpath(d, "nl_rbc.mod"), txt)
    m = quiet(() -> J.Model(J.run_preprocessor(joinpath(d, "nl_rbc.mod"))))
    est = Estimation(m; particle_rng=:octave)
    po = est.options["particle"]
    @test (po.filter_algorithm, po.number_of_particles, po.resampling_method, po.unscented_alpha) == ("apf", 1000, "stratified", 0.5)
    x = NL_LIK["X"][:, 1]
    scale = sum(abs, filter(!isnan, NL_LIK["apf_strat_lik"][:, 1]))
    @test -log_posterior(est, x) ≈ NL_LIK["apf_strat_fval"][1] atol = 1e-9 * scale
    @test Estimation(m; resampling_method="kitagawa").options["particle"].resampling_method == "kitagawa"    # a keyword wins
    po, = (J.particle_options(Dict{String,Any}("particle.filter_algorithm" => "sis", "particle.resampling.status.generic" => true,
        "particle.resampling.status.systematic" => false, "particle.resampling.threshold" => 0.3, "particle.initialization" => 3,
        "particle.particle_filter_options" => "''pruning'',true,''initial_state_prior_std'',0.2")),)
    @test (po.resampling, po.resampling_threshold, po.initialization, po.pruning, po.initial_state_prior_std) == ("generic", 0.3, 3, true, 0.2)
end

@testset "options that are refused" begin
    refused = [
        (; order=3) => ErrorException, (; loglinear=true) => J.EstimationError, (; filter_algorithm="gmf") => J.Unsupported,
        (; nonlinear_filter_initialization=2) => J.Unsupported, (; resampling_method="residual") => J.Unsupported,
        (; filter_algorithm="cpf", pruning=true) => J.Unsupported, (; filter_algorithm="gf", particle_pruning=true) => J.Unsupported,
        (; filter_algorithm="apf", resampling_method="smooth") => J.Unsupported,
        (; filter_algorithm="cpf", proposal_approximation="cubature", cpf_weights="murrayjonesparslow") => J.Unsupported,
        (; prefilter=1) => J.EstimationError, (; filter_algorithm="xyz") => J.EstimationError,
        (; particle_filter_options="'pruning', 1") => J.EstimationError]
    for (kw, ex) in refused
        @test_throws Exception quiet(() -> Estimation(NL_MODEL; kw...))
    end
    @test_throws J.EstimationError quiet(() -> Estimation(NL_MODEL; filter_algorithm="gf", smoother=true) |> J._check_nonlinear_options!)
    est = nl_est("gf_uu")
    @test_throws J.EstimationError J._check_nonlinear_options!(est; smoother=true)
    @test_throws J.EstimationError J._check_nonlinear_options!(est; filtered=true)
    @test_throws J.EstimationError J._check_nonlinear_options!(est; forecast=4)
    # order 1 keeps its own error for particle options
    @test_throws J.EstimationError quiet(() -> Estimation(NL_MODEL; order=1, filter_algorithm="sis"))
end

@testset "mode finding and sampling with a particle filter" begin
    x0 = NL_LIK["X"][:, 1]
    est = nl_est("gf_uu")
    # deterministic in the parameters (common random numbers), up to the last bits of the k-order solution
    @test log_posterior(est, x0) ≈ log_posterior(est, x0) rtol = 1e-12
    sis = nl_est("sis")
    @test log_posterior(sis, x0) ≈ log_posterior(sis, x0) rtol = 1e-11
    @test log_posterior(sis, x0) ≈ -NL_LIK["sis_fval"][1] rtol = 1e-8
    md = quiet(() -> posterior_mode(est; x0=x0, mode_compute=7, compute_hessian=false))
    @test md.log_posterior > log_posterior(est, x0) + 1.0
    cov = Matrix(Diagonal((0.001 .* abs.(md.x) .+ 1e-6) .^ 2))
    mc = quiet(() -> sample_posterior(est; replic=30, nblocks=1, mode=md.x, cov=cov, seed=1))
    @test size(mc.draws) == (1, 30, est.n) && all(isfinite, mc.log_posterior)
    # a failed solution is a rejected point, not an error
    bad = copy(x0); bad[findfirst(==("bet"), est.names)] = 1.2
    @test log_posterior(est, bad) == -Inf
    # the filter itself, with the reduced form
    LIK, lik, rf = J.nonlinear_filter_likelihood(est, x0)
    @test LIK ≈ -sum(lik) && size(rf.ghx) == (5, 2)
end

@testset "Dynare's simplex (mode_compute=8) with a particle filter" begin
    r = nlref("nl_rbc_simplex.mat")
    for (case, alg, n) in (("gf_300", "gf", 1000), ("sis_300", "sis", 200))
        est = quiet(() -> Estimation(NL_MODEL; filter_algorithm=alg, number_of_particles=n, particle_rng=:octave, optim_opt="'MaxFunEvals',300", mode_compute=8))
        md = quiet(() -> posterior_mode(est; mode_compute=8, compute_hessian=false))
        @test md.nfev == r["$(case)_funcCount"]
        @test -md.log_posterior ≈ r["$(case)_fval"] atol = 1e-8
        @test maximum(abs.(md.x ./ vec(r["$(case)_x"]) .- 1)) < 1e-8
        nl_track!("simplex mode (relative)", maximum(abs.(md.x ./ vec(r["$(case)_x"]) .- 1)))
    end
end

# ---------------------------------------------------------------------------------------------- whole runs
@testset "dynare(): estimation(order=2, ...) and oo_" begin
    for v in 1:4
        isfile(joinpath(NLREF, "e2e", "nl_e2e_v$(v)_oo.mat")) || continue
        d = mktempdir()
        cp(joinpath(NLREF, "nl_rbc_data.m"), joinpath(d, "nl_rbc_data.m"))
        cp(joinpath(NLREF, "e2e", "nl_e2e_v$v.mod"), joinpath(d, "nl_e2e_v$v.mod"))
        ref = matread(joinpath(NLREF, "e2e", "nl_e2e_v$(v)_oo.mat"))
        ws = quiet(() -> J.dynare(joinpath(d, "nl_e2e_v$v.mod"); particle_rng=:octave))     # Octave's numbers in the filters
        probs, worst, at = oo_compare(ws.oo_, ref; skip=Set(["time", "runtime", "initval_series", "initial_", "endo_simul", "exo_simul", "exo_det_simul", "gui", "dr.eigval", "dr.Gy", "dr.gx"]))
        @info "nl_e2e_v$v: worst oo_ difference $worst at $at" probs
        @test worst < 1e-5
    end
end

end # nonlinear filters

println("\nnonlinear estimation, worst differences against Dynare:")
for k in sort(collect(keys(nl_worst)))
    println("  ", rpad(k, 40), nl_worst[k])
end
