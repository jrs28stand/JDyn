# Kalman smoother against Dynare 7.2 under Octave 8.4 (calib_smoother; the smoother estimation runs). References are pydyn's:
# smo_*_case*.mat made by reference/r_smoother.m from smo_*.mod (one case per value of the macro variable CASE) and the simulated
# data smo_*_data.m; est_*_smoother*.mat (smoother at the initial values of estimated models); sw_ref_* (Smets-Wouters scale).
#
# Models: smo_fs (cash-in-advance model, 2 observables, loglinear), smo_me (linear NK model, 3 observables, measurement error on
# two, missing observations, correlated shocks), smo_ur (two unit roots, missing observations, diffuse filter).
# Run alone: `using Test; include("common.jl")` then this file.
using Test, JDyn, MAT, LinearAlgebra, Random, Logging, Printf
isdefined(@__MODULE__, :REF) || include(joinpath(@__DIR__, "common.jl"))

const SMO_DIR = mktempdir()
for f in readdir(REF; join=true)
    b = basename(f)
    (startswith(b, "smo_") && (endswith(b, ".mod") || endswith(b, "data.m"))) && cp(f, joinpath(SMO_DIR, b))
end
const _SMO_MODELS = Dict{Tuple{String,Int},JDyn.Model}()
function smoload(name, case)
    get!(_SMO_MODELS, (name, case)) do
        with_logger(NullLogger()) do
            JDyn.Model(JDyn.run_preprocessor(joinpath(SMO_DIR, name * ".mod"); extra_args=["-DCASE=$case"]))
        end
    end
end
const _SMO_RUNS = Dict{Tuple{String,Int,Bool},JDyn.SmootherResult}()
smorun(name, case, compat) = get!(_SMO_RUNS, (name, case, compat)) do
    with_logger(NullLogger()) do
        JDyn.smooth(smoload(name, case); replicate_dynare_bugs=compat)
    end
end
smoref(name, case) = matread(joinpath(REF, "$(name)_case$(case).mat"))
tab(d, names) = reduce(hcat, [d[v] for v in names])
asvec(x) = x isa Number ? [x] : vec(x)
idxs(r, key="idx") = Int.(asvec(r[key]))
maxdiffs = Dict{String,Float64}()
track!(key, v) = (maxdiffs[key] = max(get(maxdiffs, key, 0.0), v))
# maximum absolute difference; NaN must sit in the same places (missing observations)
function md(a, b)
    b = reshape(b, size(a))
    @assert isnan.(a) == isnan.(b) "NaN pattern differs"
    ok = .!isnan.(b)
    any(ok) ? maximum(abs.(a[ok] .- b[ok])) : 0.0
end

"Differences between a result and every field Dynare stored for this case."
function smocompare(res, r, m; skip=())
    out = Dict{String,Float64}()
    @test split(r["endo_names"], ",") == m.endo_names
    @test split(r["varobs"], ",") == res.varobs
    out["smoothed"] = md(tab(res.SmoothedVariables, m.endo_names), r["smoothed"])
    out["updated"] = md(tab(res.UpdatedVariables, m.endo_names), r["updated"])
    out["shocks"] = md(tab(res.SmoothedShocks, m.exo_names), r["shocks"])
    out["constant"] = md([res.Constant[v][1] for v in m.endo_names], r["constant"])
    out["init_state"] = md([res.Init_State[v] for v in m.endo_names], r["init_state"])
    out["init_state_vector"] = md(res.Init_State_Vector, r["init_state_vector"])
    @test Bool(r["loglinear"]) == res.loglinear
    haskey(r, "filtered") && (out["filtered"] = md(tab(res.FilteredVariables, m.endo_names), r["filtered"]))
    if haskey(r, "meas_err")
        out["measurement_errors"] = md(tab(res.SmoothedMeasurementErrors, res.varobs), r["meas_err"])
    else
        @test isempty(res.SmoothedMeasurementErrors)
    end
    if haskey(r, "kstep")
        @test Int.(asvec(r["filter_step_ahead"])) == res.filter_step_ahead
        out["kstep"] = md(res.FilteredVariablesKStepAhead, r["kstep"])
    end
    haskey(r, "kstep_var") && (out["kstep_var"] = md(res.FilteredVariablesKStepAheadVariances[:, :, :, idxs(r, "kidx")], r["kstep_var"]))
    haskey(r, "variance") && (out["variance"] = md(res.Variance[:, :, vcat(idxs(r), JDyn.nobs(res) + 1)], r["variance"]))
    if haskey(r, "state_uncertainty") && !("V" in skip)
        out["state_uncertainty"] = md(res.State_uncertainty[:, :, idxs(r)], r["state_uncertainty"])
        out["init_state_uncertainty"] = md(res.Init_State_uncertainty, r["init_state_uncertainty"])
    end
    out
end
worst(d) = maximum(values(d))

const SMO_CASES = [("smo_fs", 1), ("smo_fs", 2), ("smo_fs", 3), ("smo_fs", 4), ("smo_me", 1), ("smo_me", 2), ("smo_me", 3),
                   ("smo_ur", 1), ("smo_ur", 2), ("smo_ur", 3), ("smo_ur", 4), ("smo_ur", 5), ("smo_ur", 6), ("smo_ur", 7)]
# filter Dynare ends up using (1/2 stationary multi/univariate, 3/4 diffuse) and number of diffuse periods
const SMO_EXPECTED = Dict(("smo_fs", 1) => (1, 0), ("smo_fs", 2) => (2, 0), ("smo_fs", 3) => (1, 0), ("smo_fs", 4) => (1, 0),
    ("smo_me", 1) => (1, 0), ("smo_me", 2) => (2, 0), ("smo_me", 3) => (1, 0), ("smo_ur", 1) => (3, 2), ("smo_ur", 2) => (4, 2),
    ("smo_ur", 3) => (4, 2), ("smo_ur", 4) => (3, 1), ("smo_ur", 5) => (4, 2), ("smo_ur", 6) => (3, 2), ("smo_ur", 7) => (4, 2))
# cases whose Dynare output carries one of its defects (multivariate diffuse smoother with a diffuse period without data or F_inf = 0;
# univariate diffuse state uncertainty in the same situation)
const SMO_DEFECT = Set([("smo_ur", 1), ("smo_ur", 2), ("smo_ur", 6), ("smo_ur", 7)])

@testset "smoother" begin

@testset "against Dynare, replicating its defects: $name case $case" for (name, case) in SMO_CASES
    res = smorun(name, case, true)
    @test (res.kalman_algo, res.diffuse_periods) == SMO_EXPECTED[name, case]
    d = smocompare(res, smoref(name, case), smoload(name, case))
    track!("smoother, replicate_dynare_bugs: $name", worst(d))
    @test worst(d) < 1e-12
end

@testset "default corrected output equals Dynare where it has no defect: $name case $case" for (name, case) in filter(c -> !(c in SMO_DEFECT), SMO_CASES)
    d = smocompare(smorun(name, case, false), smoref(name, case), smoload(name, case))
    @test worst(d) < 1e-12
end

@testset "steady state and decision rules are those of the references" begin
    for (name, case) in (("smo_fs", 2), ("smo_me", 3), ("smo_ur", 1))
        r, m = smoref(name, case), smoload(name, case)
        sol = with_logger(NullLogger()) do
            JDyn.first_order(m; loglinear=false)
        end
        @test maximum(abs.(sol.ys .- vec(r["ys"]))) < 1e-12
        res = smorun(name, case, true)
        ov = Int.(vec(r["order_var"]))
        # the log-linear rules of smo_fs are in logs: compare ghu only to the extent the references store levels (not loglinear cases)
        if !res.loglinear
            @test maximum(abs.(res.H[ov, :] .- r["ghu"])) < 1e-11
            @test maximum(abs.(res.G[ov, sol.state_var] .- r["ghx"])) < 1e-11
        end
    end
end

# ------------------------------------------------------------------ Dynare's diffuse-smoother defects
@testset "multivariate diffuse smoother, corrected, equals Dynare's univariate" begin
    # Period 1 has no observation (smo_ur case 1) or F_inf = 0 (case 6): Dynare's kalman_algo=3 output differs from its kalman_algo=4
    # output (cases 5 and 7); the default follows the univariate (correct) numbers while still running the multivariate recursions.
    for (mv, uv, skip) in ((1, 5, ()), (6, 7, ("V",)))
        res = smorun("smo_ur", mv, false)
        @test res.kalman_algo == 3
        d = smocompare(res, smoref("smo_ur", uv), smoload("smo_ur", mv); skip=skip)
        track!("multivariate diffuse smoother (corrected) against Dynare's univariate", worst(d))
        @test worst(d) < 1e-12
        r_mv, r_uv = smoref("smo_ur", mv), smoref("smo_ur", uv)
        @test maximum(abs.(r_mv["smoothed"] .- r_uv["smoothed"])) > 0.1        # size of the defect in Dynare's own output
        @test maximum(abs.(r_mv["shocks"] .- r_uv["shocks"])) > 0.05
    end
    # case 6/7: Dynare's univariate state uncertainty is itself off in the first period
    res, r = smorun("smo_ur", 6, false), smoref("smo_ur", 7)
    ix = idxs(r)
    @test md(res.State_uncertainty[:, :, ix[2:end]], r["state_uncertainty"][:, :, 2:end]) < 1e-12
    @test maximum(abs.(res.State_uncertainty[:, :, ix[1]] .- r["state_uncertainty"][:, :, 1])) > 0.1
end

"Smoothed mean and covariance from the joint normal distribution of the states and the data, the diffuse part of the initial variance a large number."
function brute_force(res, m; kappa=1e6)
    G, H, Q, mf = res.G, res.H, m.Sigma_e, res.obs_index
    n, N = size(G, 1), JDyn.nobs(res)
    Pstar, Pinf = JDyn.compute_Pinf_Pstar(mf, G, H, Q)
    cov = [Pstar + kappa * Pinf]
    for _ in 1:N
        push!(cov, G * cov[end] * G' + H * Q * H')
    end
    C = zeros((N + 1) * n, (N + 1) * n)
    for i in 0:N
        blk = cov[i+1]
        for j in i:N
            C[i*n+1:(i+1)*n, j*n+1:(j+1)*n] = blk
            C[j*n+1:(j+1)*n, i*n+1:(i+1)*n] = blk'
            blk = blk * G'
        end
    end
    seen = [(t, i) for t in 1:N for i in eachindex(mf) if !isnan(res.data[t, i])]
    sel = [t * n + mf[i] for (t, i) in seen]
    y = [res.data[t, i] for (t, i) in seen]
    Cyy = C[sel, sel] + Diagonal([res.me_variance[i] for (_, i) in seen])
    gain = (Cyy \ C[sel, :])'
    V = C - gain * C[sel, :]
    mean_ = reshape(gain * y, n, N + 1)'
    mean_, cat([V[k*n+1:(k+1)*n, k*n+1:(k+1)*n] for k in 0:N]...; dims=3)
end

@testset "diffuse smoother against brute force: case $case algo $algo" for (case, algo) in [(1, 3), (2, 4), (6, 3), (7, 4), (4, 3), (3, 4)]
    m = smoload("smo_ur", case)
    res = with_logger(NullLogger()) do
        JDyn.smooth(m; nobs=6, kalman_algo=algo, smoothed_state_uncertainty=true)
    end
    @test res.kalman_algo == algo
    mean_, V = brute_force(res, m)
    tol = 2e-5                                      # error of the large-kappa approximation
    @test maximum(abs.(res.smoothed .- mean_[2:end, :])) < tol
    @test maximum(abs.(res.init_state .- mean_[1, :])) < tol
    @test maximum(abs.(res.State_uncertainty .- V[:, :, 2:end])) < tol
    @test maximum(abs.(res.Init_State_uncertainty .- V[:, :, 1])) < tol
    track!("diffuse smoother against brute force (kappa = 1e6)", max(maximum(abs.(res.smoothed .- mean_[2:end, :])), maximum(abs.(res.State_uncertainty .- V[:, :, 2:end]))))
    if ("smo_ur", case) in SMO_DEFECT
        bug = with_logger(NullLogger()) do
            JDyn.smooth(m; replicate_dynare_bugs=true, nobs=6, kalman_algo=algo, smoothed_state_uncertainty=true)
        end
        err = max(maximum(abs.(bug.State_uncertainty .- V[:, :, 2:end])), maximum(abs.(bug.Init_State_uncertainty .- V[:, :, 1])))
        @test err > 0.1                               # Dynare's numbers are wrong here
    end
end

# ------------------------------------------------------------------ consistency checks
@testset "identities: $name case $case" for (name, case) in SMO_CASES
    res = smorun(name, case, false)
    chk = JDyn.check_smoother(res)
    @test Set(keys(chk)) == Set(["transition", "simulation", "measurement"])
    @test maximum(values(chk)) < 1e-10
    if !res.loglinear                      # sol.G, sol.H are the rules used
        sol = with_logger(NullLogger()) do
            JDyn.first_order(smoload(name, case); loglinear=false)
        end
        @test maximum(values(JDyn.check_smoother(res; sol=sol))) < 1e-10
    end
end

@testset "check_smoother detects Dynare's defect: case $case" for case in (1, 6)
    chk = JDyn.check_smoother(smorun("smo_ur", case, true))
    @test chk["transition"] > 0.1 && chk["measurement"] < 1e-10
end

# ------------------------------------------------------------------ inputs and options
@testset "data from Julia, csv and file names" begin
    m, base = smoload("smo_me", 1), smorun("smo_me", 1, false)
    series = with_logger(NullLogger()) do
        JDyn.read_m_data(joinpath(SMO_DIR, "smo_me_data.m"))
    end
    arr = reduce(hcat, [series[v] for v in m.varobs])
    @test count(isnan, arr) == 14 && all(isnan, arr[18, :])
    csv = joinpath(mktempdir(), "d.csv")
    names = reverse(collect(keys(series)))                 # column order must not matter
    write(csv, join([join(names, ",")]) * "\n" * join([join([isnan(series[v][t]) ? "" : repr(Float64(series[v][t])) for v in names], ",") for t in eachindex(arr[:, 1])], "\n") * "\n")
    for data in (arr, series, csv, joinpath(SMO_DIR, "smo_me_data"))
        res = with_logger(NullLogger()) do
            JDyn.smooth(m, data)
        end
        @test res.smoothed == base.smoothed && res.shocks == base.shocks
    end
    @test_throws JDyn.SmootherError JDyn.smooth(m, arr[:, 1:2])
    @test_throws JDyn.SmootherError JDyn.smooth(m, Dict("yobs" => arr[:, 1]))
end

@testset "options from Julia equal options from the file" begin
    m = smoload("smo_fs", 4)                               # its own statement: levels, nobs=100
    old = copy(m.params)
    try
        JDyn.set_params!(m; rho=0.5, alp=0.36)
        res = with_logger(NullLogger()) do
            JDyn.smooth(m; datafile="smo_fs_data", loglinear=true, first_obs=11, nobs=80, kalman_algo=2, filtered_vars=true,
                        smoothed_state_uncertainty=true)
        end
        @test worst(smocompare(res, smoref("smo_fs", 2), m)) < 1e-12
    finally
        m.params .= old
        empty!(m.cache)
    end
    # logdata: case 3 sets options_.logdata in the .mod file; here the keyword does it
    res = with_logger(NullLogger()) do
        JDyn.smooth(m; datafile="smo_fs_logdata", loglinear=true, logdata=true, nobs=60, presample=4, filtered_vars=false)
    end
    r = smoref("smo_fs", 3)
    @test maximum(abs.(res.smoothed .+ log.(res.SteadyState)' .- r["smoothed"])) < 1e-12
    @test maximum(abs.(res.shocks .- r["shocks"])) < 1e-12
    @test res.FilteredVariablesKStepAhead === nothing && res.Variance === nothing && res.State_uncertainty === nothing
end

@testset "Dynare's conventions" begin
    # loglinear: variables come back in logs; Init_State_Vector adds the level steady state
    res, m = smorun("smo_fs", 1, false), smoload("smo_fs", 1)
    ys = res.SteadyState
    k = findfirst(==("c"), m.endo_names)
    @test res.SmoothedVariables["c"] ≈ res.smoothed[:, k] .+ log(ys[k]) atol = 1e-14
    @test res.Init_State_Vector ≈ res.init_state .+ ys atol = 1e-14
    # FilteredVariables[t] is the forecast for t+1 made at t
    @test maximum(abs.(tab(res.FilteredVariables, m.endo_names)[1:end-1, :] .- log.(ys)' .- (res.updated*res.G')[1:end-1, :])) < 1e-12
    @test size(res.FilteredVariablesKStepAhead) == (3, m.n_endo, JDyn.nobs(res) + 4)
    @test res.filter_step_ahead == [1, 2, 4]
    # prefilter: observed variables come back around the sample mean of the data
    res, m = smorun("smo_me", 3, false), smoload("smo_me", 3)
    raw = with_logger(NullLogger()) do
        JDyn.load_data("smo_me_data", m.varobs; base_dir=m.mod_dir)
    end[1:90, :]
    @test res.obs_constant ≈ [sum(filter(!isnan, raw[:, j])) / count(!isnan, raw[:, j]) for j in 1:3] atol = 1e-12
    seen = .!isnan.(raw[:, 3])
    @test maximum(abs.(res.SmoothedVariables["iobs"][seen] .- raw[seen, 3])) < 1e-10      # no measurement error on iobs
    @test all(res.Constant["iobs"] .== res.obs_constant[3]) && res.SteadyState[findfirst(==("iobs"), m.endo_names)] == 0
    # missing observations: the measurement error is NaN, the smoothed variable is not
    @test isnan(res.SmoothedMeasurementErrors["yobs"][3]) && isfinite(res.SmoothedVariables["yobs"][3])
    @test Set(keys(res.SmoothedMeasurementErrors)) == Set(m.varobs)              # all observables, as in Dynare
end

@testset "errors and refusals" begin
    m = smoload("smo_ur", 1)
    @test_throws JDyn.SmootherError JDyn.smooth(m; diffuse_filter=false)                    # unit roots need the diffuse filter
    @test_throws JDyn.SmootherError JDyn.smooth(m; no_such_option=1)
    @test_throws JDyn.SmootherError JDyn.smooth(m; lik_init=4, diffuse_filter=false)
    @test_throws JDyn.SmootherError JDyn.smooth(m; first_obs=80, nobs=20)
    @test_throws JDyn.SmootherError JDyn.smooth(m; lik_init=2, diffuse_filter=true)         # contradictory
    @test_throws JDyn.SmootherError JDyn.smooth(smoload("smo_fs", 4); prefilter=1)           # non-zero steady state of the observables
    @test_throws JDyn.SmootherError JDyn.smooth(m; parameter_set="posterior_mode")          # needs an Estimation
    @test_throws JDyn.SmootherError JDyn.smooth(m; theta=[1.0])
end


# ------------------------------------------------------------------ estimated models: smoother at parameter vectors
SD_DIR = mktempdir()
for f in ("sd_nk.mod", "sd_ur.mod", "est_nk_data.m", "est_ur_data.m", "est_ur_miss.m", "est_nk_miss.m")
    isfile(joinpath(REF, f)) && cp(joinpath(REF, f), joinpath(SD_DIR, f))
end
_SD_MODELS = Dict{Tuple{String,Int},JDyn.Model}()
function sdload(name, case)
    get!(_SD_MODELS, (name, case)) do
        with_logger(NullLogger()) do
            JDyn.Model(JDyn.run_preprocessor(joinpath(SD_DIR, name * ".mod"); extra_args=["-DCASE=$case"]))
        end
    end
end
quiet(f) = with_logger(f, NullLogger())
"Every smoother field Dynare stored, in the layout of the sd_* references (a struct read as a Dict)."
function sdcompare(res, r, m)
    out = Dict{String,Float64}()
    @test split(r["endo_names"], ",") == m.endo_names && split(r["varobs"], ",") == res.varobs
    out["smoothed"] = md(tab(res.SmoothedVariables, m.endo_names), r["smoothed"])
    out["shocks"] = md(tab(res.SmoothedShocks, m.exo_names), r["shocks"])
    out["constant"] = md([res.Constant[v][1] for v in m.endo_names], r["constant"])
    out["init_state"] = md([res.Init_State[v] for v in m.endo_names], r["init_state"])
    out["init_state_vector"] = md(res.Init_State_Vector, r["init_state_vector"])
    out["trend_coeffs"] = md(res.TrendCoeffs, r["trend_coeffs"])
    out["steady_state"] = md(res.SteadyState, r["ys"])
    for (key, mine) in (("updated", res.UpdatedVariables), ("filtered", res.FilteredVariables), ("trend", res.Trend))
        haskey(r, key) && (out[key] = md(tab(mine, m.endo_names), r[key]))
    end
    haskey(r, "meas_err") && (out["measurement_errors"] = md(tab(res.SmoothedMeasurementErrors, res.varobs), r["meas_err"]))
    if haskey(r, "kstep")
        @test Int.(asvec(r["filter_step_ahead"])) == res.filter_step_ahead
        out["kstep"] = md(res.FilteredVariablesKStepAhead, r["kstep"])
    end
    out
end

@testset "smoother at Dynare's parameter vectors: $pset" for (pset, vecname) in
        (("posterior_mode", "theta_mode"), ("posterior_mean", "theta_mean"), ("posterior_median", "theta_median"), ("prior_mean", "theta_prior_mean"))
    # Bayesian estimation in Dynare (mode_compute=4, 1200 draws): its vectors through theta=. Covers shock and measurement-error
    # standard deviations, a shock correlation, structural parameters and a parameter recomputed in the steady state (rbar from beta).
    m = sdload("sd_nk", 1)
    est = Estimation(m)
    r = matread(joinpath(REF, "sd_nk_case1.mat"))
    @test est.names == split(r["names"], ";")
    before = (copy(m.params), copy(m.Sigma_e), copy(m.measurement_errors))
    res = quiet(() -> JDyn.smooth(m; estimation=est, theta=vec(r[vecname]), filtered_vars=true))
    d = sdcompare(res, r[pset], m)
    track!("smoother at estimated parameters (sd_nk)", worst(d))
    @test worst(d) < 1e-12
    @test res.parameter_set == "theta" && res.theta == vec(r[vecname])
    @test Set(keys(res.SmoothedMeasurementErrors)) == Set(m.varobs)
    # the caller's model is untouched
    @test m.params == before[1] && m.Sigma_e == before[2] && m.measurement_errors == before[3]
end

@testset "parameter_set names go through the same path as theta" begin
    m = sdload("sd_nk", 1); est = Estimation(m); r = matread(joinpath(REF, "sd_nk_case1.mat"))
    base = quiet(() -> JDyn.smooth(m; estimation=est, theta=vec(r["theta_mode"])))
    mode = JDyn.ModeResult(vec(r["theta_mode"]), est.names, 0.0)
    for kw in (Dict(:mode => mode), Dict(:mode => vec(r["theta_mode"])))
        res = quiet(() -> JDyn.smooth(m; estimation=est, parameter_set="posterior_mode", kw...))
        @test res.parameter_set == "posterior_mode" && res.smoothed == base.smoothed && res.shocks == base.shocks
    end
    @test_throws JDyn.SmootherError JDyn.smooth(m; estimation=est, parameter_set="posterior_mode")                    # no mode given
    @test_throws JDyn.SmootherError JDyn.smooth(m; estimation=est, parameter_set="mle_mode", mode=vec(r["theta_mode"]))   # Bayesian estimation
    @test_throws JDyn.SmootherError JDyn.smooth(m; theta=vec(r["theta_mode"]))                                           # no Estimation
    @test_throws JDyn.SmootherError JDyn.smooth(m; estimation=est, theta=vec(r["theta_mode"])[1:end-1])
    @test_throws JDyn.SmootherError JDyn.smooth(m; estimation=est, parameter_set="posterior")
    # posterior mean / median from draws: the statistics of the draws kept after mh_drop
    rng = MersenneTwister(3)
    draws = reshape(vec(r["theta_mode"]), 1, 1, :) .+ 1e-3 .* randn(rng, 2, 40, length(est.names))
    mc = JDyn.MCMCResult(draws, zeros(2, 40), ones(2), est.names, draws[:, 1, :], 0.5, trues(2, 40))
    kept = reshape(permutedims(draws[:, 21:end, :], (2, 1, 3)), :, length(est.names))
    for (pset, v) in (("posterior_mean", vec(sum(kept; dims=1)) ./ size(kept, 1)), ("posterior_median", [JDyn._median_(kept[:, j]) for j in axes(kept, 2)]))
        @test JDyn.parameter_vector(est, pset; mcmc=mc) == v
        a = quiet(() -> JDyn.smooth(m; estimation=est, parameter_set=pset, mcmc=mc))
        b = quiet(() -> JDyn.smooth(m; estimation=est, theta=v))
        @test a.smoothed == b.smoothed
    end
    @test maximum(abs.(JDyn.parameter_vector(est, "prior_mean") .- vec(r["theta_prior_mean"]))) < 1e-14
end

@testset "maximum likelihood mode" begin
    m = sdload("sd_nk", 2); est = Estimation(m); r = matread(joinpath(REF, "sd_nk_case2.mat"))
    @test est.names == split(r["names"], ";")
    res = quiet(() -> JDyn.smooth(m; estimation=est, theta=vec(r["theta_mode"]), filtered_vars=true))
    d = sdcompare(res, r["mle_mode"], m)
    track!("smoother at estimated parameters (sd_nk)", worst(d))
    @test worst(d) < 1e-12
    @test_throws JDyn.SmootherError JDyn.smooth(m; estimation=est, parameter_set="prior_mean")        # no priors
    mode = quiet(() -> posterior_mode(est; compute_hessian=false))
    @test maximum(abs.(mode.x .- vec(r["theta_mode"]))) < 1e-4                                       # optimisers differ
    own = quiet(() -> JDyn.smooth(m; estimation=est, parameter_set="mle_mode", mode=mode))
    @test own.parameter_set == "mle_mode"
end

@testset "prior mode" begin
    # bayestopt_.p5 for inverse gamma (both types), gamma, beta (also generalised), normal, Weibull and uniform priors;
    # Dynare's smoother fails at the prior mode when a prior is uniform
    m = sdload("sd_nk", 1); est = Estimation(m); r = matread(joinpath(REF, "sd_nk_case1.mat"))
    pm = JDyn.prior_mode(est.prior)
    ref_ = vec(r["theta_prior_mode"])
    @test isnan.(pm) == isnan.(ref_) && count(isnan, pm) == 2
    @test maximum(abs.(pm[.!isnan.(pm)] .- ref_[.!isnan.(ref_)])) < 1e-12
    @test occursin("NaN", r["error_prior_mode"])
    @test_throws JDyn.SmootherError JDyn.smooth(m; estimation=est, parameter_set="prior_mode")
    e4 = Estimation(sdload("sd_ur", 4)); r4 = matread(joinpath(REF, "sd_ur_case4.mat"))
    @test maximum(abs.(JDyn.parameter_vector(e4, "prior_mode") .- vec(r4["theta_prior_mode"]))) < 1e-12
end

@testset "observation trends, prefilter, lik_init = 2" begin
    for (case, algo) in ((1, (4, 1)), (2, (3, 2)))
        m, r = sdload("sd_ur", case), matread(joinpath(REF, "sd_ur_case$case.mat"))
        res = quiet(() -> JDyn.smooth(m; replicate_dynare_bugs=true))
        @test (res.kalman_algo, res.diffuse_periods) == algo
        d = sdcompare(res, r, m)
        track!("smoother with observation_trends (sd_ur)", worst(d))
        @test worst(d) < 1e-12
        i = findfirst(==("yobs"), m.endo_names)
        @test res.TrendCoeffs[i] == 0.05 && all(res.Trend["g"] .== 0)
        want = 0.05 .* collect(res.first_obs:res.first_obs+JDyn.nobs(res)-1)
        @test maximum(abs.(res.Trend["yobs"] .- (case == 2 ? want .- sum(want) / length(want) : want))) < 1e-14
        seen = .!isnan.(res.data[:, 1])      # the trend belongs to the observed variables only
        @test maximum(abs.((res.SmoothedVariables["yobs"].-res.Trend["yobs"].-res.Constant["yobs"])[seen] .- res.data[seen, 1])) < 1e-10
        @test res.SmoothedVariables["y"] == res.smoothed[:, findfirst(==("y"), m.endo_names)]
    end
    # case 2 (prefilter, missing data) with the default (corrected) multivariate diffuse smoother differs: Dynare's defect
    m, r = sdload("sd_ur", 2), matread(joinpath(REF, "sd_ur_case2.mat"))
    res = quiet(() -> JDyn.smooth(m))
    @test maximum(abs.(tab(res.SmoothedVariables, m.endo_names) .- r["smoothed"])) > 0.05
    @test maximum(values(JDyn.check_smoother(res))) < 1e-10
    # estimation(mode_compute=0, smoother): the smoother at the initial values; the trend slope is an estimated parameter
    for (case, algo) in ((3, 1), (4, 4))
        m, r = sdload("sd_ur", case), matread(joinpath(REF, "sd_ur_case$case.mat"))
        est = Estimation(m)
        @test maximum(abs.(est.x0 .- vec(r["theta0"]))) == 0
        res = quiet(() -> JDyn.smooth(m; estimation=est, theta=vec(r["theta0"]), replicate_dynare_bugs=true, filtered_vars=(case == 3)))
        @test res.kalman_algo == algo
        d = sdcompare(res, r["at_theta0"], m)
        track!("smoother with observation_trends (sd_ur)", worst(d))
        @test worst(d) < 1e-12
        @test res.TrendCoeffs[findfirst(==("yobs"), m.endo_names)] == 0.08
    end
    # lik_init=2 conventions: period 1 starts from the large variance directly (Dynare), so the transition identity holds from period 2
    m = sdload("sd_ur", 3); est = Estimation(m)
    res = quiet(() -> JDyn.smooth(m; estimation=est, theta=est.x0))
    @test res.options["lik_init"] == 2 && res.diffuse_periods == 0
    @test maximum(abs.(res.smoothed[1:end-1, :] * res.G' .+ res.shocks[2:end, :] * res.H' .- res.smoothed[2:end, :])) < 1e-10
    @test JDyn.check_smoother(res)["measurement"] < 1e-10
    res5 = quiet(() -> JDyn.smooth(m; estimation=est, theta=est.x0, Harvey_scale_factor=50.0))
    @test maximum(abs.(res5.smoothed .- res.smoothed)) > 1e-6
    @test_throws JDyn.SmootherError JDyn.smooth(m; estimation=est, theta=est.x0, diffuse_filter=true)    # contradictory with lik_init=2
end

@testset "filter_initial_state: case $case" for (case, algo) in ((4, 1), (5, 2))
    m, r = sdload("sd_nk", case), matread(joinpath(REF, "sd_nk_case$case.mat"))
    res = quiet(() -> JDyn.smooth(m; replicate_dynare_bugs=true))
    @test res.kalman_algo == algo
    d = sdcompare(res, r, m)
    track!("smoother with filter_initial_state (sd_nk)", worst(d))
    @test worst(d) < 1e-12
    # Dynare centres the period-0 state on T*a: the transition identity fails in period 1 only
    @test JDyn.check_smoother(res)["transition"] > 0.1
    good = quiet(() -> JDyn.smooth(m))
    @test good.smoothed == res.smoothed && good.shocks == res.shocks
    @test maximum(values(JDyn.check_smoother(good))) < 1e-12
    @test maximum(abs.(good.init_state .- res.init_state)) > 0.1
    # the block matters: without it the smoothed path differs
    saved = m.statements
    m.statements = [st for st in m.statements if get(st, "statementName", "") != "filter_initial_state"]
    plain = try
        quiet(() -> JDyn.smooth(m))
    finally
        m.statements = saved
    end
    @test maximum(abs.(plain.smoothed .- good.smoothed)) > 1e-3
end

@testset "correlated measurement errors: the univariate smoother (augmented state) equals the multivariate one" begin
    # est_nk variant 9: correlated measurement errors on yobs and piobs, missing observations
    sub = mktempdir()
    for f in readdir(REF; join=true)
        b = basename(f)
        (startswith(b, "est_nk") && (endswith(b, ".mod") || endswith(b, ".m") || endswith(b, ".csv"))) && cp(f, joinpath(sub, b))
    end
    m = quiet(() -> JDyn.Model(JDyn.run_preprocessor(joinpath(sub, "est_nk.mod"); extra_args=["-DVARIANT=9"])))
    est = Estimation(m)
    kw = (smoothed_state_uncertainty=true, filter_covariance=true, filtered_vars=true)
    a = quiet(() -> JDyn.smooth(m; estimation=est, theta=est.x0, kw...))
    @test a.kalman_algo == 1 && maximum(abs.(a.me_cov .- Diagonal(diag(a.me_cov)))) > 1e-4
    for (label, extra) in (("kalman_algo=2", (kalman_algo=2,)), ("diffuse kalman_algo=4", (kalman_algo=4, diffuse_filter=true)))
        b = quiet(() -> JDyn.smooth(m; estimation=est, theta=est.x0, kw..., extra...))
        @test b.kalman_algo in (2, 4)
        e = max(md(a.smoothed, b.smoothed), md(a.shocks, b.shocks), md(a.updated, b.updated), md(a.init_state, b.init_state),
                maximum(abs.(a.State_uncertainty .- b.State_uncertainty)), maximum(abs.(a.Variance .- b.Variance)),
                md(a.FilteredVariables["y"], b.FilteredVariables["y"]))
        seen = .!isnan.(a.measurement_errors)
        e = max(e, maximum(abs.(a.measurement_errors[seen] .- b.measurement_errors[seen])))
        track!("smoother, correlated measurement errors: univariate against multivariate", e)
        @test e < 1e-12
        @test all(isfinite, b.measurement_errors)            # the augmented-state smoother gives the error also where data are missing, as Dynare
    end
end

@testset "estimation(smoother): $mod $variant" for (mod, variant, name) in
        (("est_nk", 5, "est_nk_smoother.mat"), ("est_nk", 7, "est_nk_smoother_miss.mat"), ("est_schorfheide", "SMOOTHER", "est_schorfheide_smoother.mat"))
    # the smoother at the initial values of estimated models: complete data, missing observations, loglinear model (with logdata)
    sub = mktempdir()
    for f in readdir(REF; join=true)
        b = basename(f)
        (startswith(b, "est_") && (endswith(b, ".mod") || endswith(b, ".m") || endswith(b, ".csv"))) && cp(f, joinpath(sub, b))
    end
    args = variant isa String ? ["-D$variant"] : ["-DVARIANT=$variant"]
    m = quiet(() -> JDyn.Model(JDyn.run_preprocessor(joinpath(sub, mod * ".mod"); extra_args=args)))
    s = matread(joinpath(REF, name))
    est = Estimation(m)
    res = quiet(() -> JDyn.smooth(m; estimation=est, theta=est.x0))
    for (i, v) in enumerate(split(s["endo_names"], ";"))
        track!("smoother at estimation start (est_*)", md(res.SmoothedVariables[v], s["smoothed"][:, i]))
        track!("smoother at estimation start (est_*)", md(res.UpdatedVariables[v], s["updated"][:, i]))
        @test md(res.SmoothedVariables[v], s["smoothed"][:, i]) < 1e-12
    end
    for (i, v) in enumerate(split(s["exo_names"], ";"))
        track!("smoother at estimation start (est_*)", md(res.SmoothedShocks[v], s["shocks"][:, i]))
        @test md(res.SmoothedShocks[v], s["shocks"][:, i]) < 1e-12
        @test maximum(abs.(s["shocks"][:, i])) > 1e-3
    end
    if haskey(s, "me_yobs")
        @test md(res.SmoothedMeasurementErrors["yobs"], vec(s["me_yobs"])) < 1e-12
        @test count(isnan, vec(s["me_yobs"])) == count(isnan, est.data[:, 1])           # NaN where yobs is missing
    end
    # estimate() runs the smoother the statement asks for, at the initial values here (mode_compute=0)
    out = quiet(() -> estimate(m))
    @test out.smoother isa JDyn.SmootherResult && out.smoother.smoothed == res.smoothed
end

@testset "Smets-Wouters scale: 40 variables, near-unit roots, gaps at the start" begin
    # sw_big.mod (Dynare's SW_2007 with the estimation replaced by calib_smoother), data with two observables missing for the first 32
    # periods. Cases: 1 baseline (largest root 0.9977); 2 all persistences 0.9999; 3 three persistences 0.99999; 4 two at 0.9999995,
    # which Dynare treats as unit roots (cut-off 1 - 1e-6) and smooths with the diffuse filter (the stationary run is not in the references).
    sw = mktempdir()
    for f in ("sw_big.mod", "sw_data.m")
        cp(joinpath(REF, f), joinpath(sw, f))
    end
    swload(case, algo=0) = quiet(() -> JDyn.Model(JDyn.run_preprocessor(joinpath(sw, "sw_big.mod"); extra_args=["-DCASE=$case", "-DALGO=$algo"])))
    relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
    for (case, tol) in ((1, 1e-9), (2, 1e-6), (3, 1e-6))
        m, r = swload(case), matread(joinpath(REF, "sw_ref_case$case.mat"))
        @test m.n_endo == 40 && m.n_exo == 7
        res = quiet(() -> JDyn.smooth(m))
        t = minimum(@elapsed(quiet(() -> JDyn.smooth(m))) for _ in 1:3)
        smn, shn = split(r["sm_names"], ","), split(r["sh_names"], ",")
        e1 = relerr(reduce(vcat, [res.SmoothedVariables[v]' for v in smn]), r["smoothed"])
        e2 = relerr(reduce(vcat, [res.SmoothedShocks[v]' for v in shn]), r["shocks"])
        track!("smoother, Smets-Wouters scale, stationary (relative to the largest value)", max(e1, e2))
        @test e1 < tol && e2 < tol
        @test res.kalman_algo == 1 && res.diffuse_periods == 0
        @test maximum(values(JDyn.check_smoother(res))) < 1e-9 * maximum(abs.(res.smoothed))     # values of order 100, near-unit roots
        println("  SW case $case (stationary, $(JDyn.nobs(res)) periods, filtered_vars, filter_step_ahead=[1 4]): ", @sprintf("%.3f s", t),
                " for first_order + smoother; relative differences ", @sprintf("%.2g / %.2g", e1, e2))
    end
    for (case, atol) in ((2, 1e-5), (3, 1e-5), (4, 1e-8))
        m = swload(case, 3)
        res = quiet(() -> JDyn.smooth(m; diffuse_filter=true, kalman_algo=3))
        for algo in (3, 4)
            r = matread(joinpath(REF, "sw_ref_diffuse_case$(case)_algo$algo.mat"))
            sm = reduce(vcat, [res.SmoothedVariables[v]' for v in split(r["sm_names"], ",")])
            e = maximum(abs.(sm .- r["smoothed"]))
            track!("smoother, Smets-Wouters scale, diffuse (absolute; values of order 100)", e)
            @test e < atol
        end
        @test res.kalman_algo == (case == 4 ? 4 : 3) && res.diffuse_periods == (case == 4 ? 2 : 0)
    end
end

end # smoother testset

println("\nMaximum differences against Dynare (smoother):")
for k in sort(collect(keys(maxdiffs)))
    println("  ", rpad(k, 66), @sprintf("%.3g", maxdiffs[k]))
end
