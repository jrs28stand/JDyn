# Likelihood-based estimation of first-order DSGE models (port of pydyn/estimation.py, after Dynare 7.2's
# matlab/estimation/{dsge_likelihood, dynare_estimation_init, dynare_estimation_1, set_prior, posterior_sampler*,
# posterior_moments, marginal_density, check_bounds_and_definiteness_estimation}.m, matlab/hessian.m and
# matlab/optimization/{csminwel1, csminit1, bfgsi1, numgrad2, numgrad3, penalty_objective_function}.m).
#
#     m = JDyn.load("model.mod")
#     est = Estimation(m)                      # options and data file taken from the .mod file
#     est.names, est.x0                        # ordering of Dynare's bayestopt_.name / xparam1
#     log_likelihood(est, θ); log_prior(est, θ); log_posterior(est, θ)
#     mode = posterior_mode(est)               # csminwel (mode_compute=4), Hessian, Laplace approximation
#     mc = sample_posterior(est; mode)         # random-walk Metropolis-Hastings
#     res = estimate(m)                        # what the .mod file's estimation statement asks for
#
# The parameter vector is ordered as in Dynare: standard deviations of structural shocks, standard deviations of
# measurement errors, correlations of structural shocks, correlations of measurement errors, structural parameters;
# within each group in the order of the `estimated_params` block. The model object is never modified by an
# evaluation: every call works on copies of the parameter vector and the covariance matrices, recomputes the steady
# state and uses the parameters returned by the steady state (a `steady_state_model` block may update parameters).

struct EstimationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::EstimationError) = print(io, "EstimationError: ", e.msg)

# ====================================================================== data files
_as_matrix(v::Real) = fill(Float64(v), 1, 1)
_as_matrix(v::AbstractMatrix{<:Real}) = Matrix{Float64}(v)
_as_matrix(v::AbstractVector{<:Real}) = reshape(Float64.(v), :, 1)
_as_matrix(v) = nothing

"""
    read_m_data(path) -> Dict{String,Vector{Float64}}

Run a MATLAB `.m` data file (a script that defines one vector per observed variable) with JDyn's MATLAB interpreter,
as Dynare's `load_m_file_data_legacy.m` does with `eval`, and return the vectors it defines.
"""
function read_m_data(path::AbstractString)
    text = read(path, String)
    out = Dict{String,Vector{Float64}}()
    try
        body, funcs = mparse(text)
        I = Interp(dirname(abspath(path)), nothing)
        sc = Scope(; vars=Dict{String,Any}())
        push!(I.scope_stack, sc)
        run_script(I, Unit(String(path), body, funcs), sc)
        for (k, v) in sc.vars
            M = _as_matrix(v)
            (M !== nothing && min(size(M)...) == 1) && (out[k] = vec(M))
        end
    catch e
        e isa Unsupported && throw(EstimationError("data file $path cannot be interpreted: $(e.msg)"))
        e isa MatlabError && throw(EstimationError("data file $path: MATLAB error: $(e.msg)"))
        e isa ParseError && throw(EstimationError("data file $path cannot be parsed: $(e.msg)"))
        rethrow()
    end
    out
end

function _read_csv(path)
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && throw(EstimationError("empty data file $path"))
    header = strip.(split(lines[1], ','))
    rows = [strip.(split(l, ',')) for l in lines[2:end]]
    num(s) = (s == "" || lowercase(s) in ("nan", "na")) ? NaN : parse(Float64, s)
    out = Dict{String,Vector{Float64}}()
    for (j, name) in enumerate(header)
        col = try
            Float64[j <= length(r) ? num(r[j]) : NaN for r in rows]
        catch
            continue                              # a column of dates or labels
        end
        out[String(name)] = col
    end
    out
end

function _read_mat(path)
    d = MAT.matread(path)
    out = Dict{String,Vector{Float64}}()
    for (k, v) in d
        M = _as_matrix(v)
        (M !== nothing && min(size(M)...) == 1) && (out[String(k)] = vec(M))
    end
    out
end

"""
    load_data(source, varobs; base_dir=nothing) -> Matrix (periods x length(varobs))

`source` may be a Dict `name => series`, a matrix whose columns follow `varobs`, or the name of a `.m`, `.mat` or
`.csv` file (extension optional, as in Dynare; relative names are looked up in `base_dir`).
"""
function load_data(source, varobs::AbstractVector{<:AbstractString}; base_dir=nothing)
    local table
    if source isa AbstractDict
        table = Dict{String,Vector{Float64}}(String(k) => Float64.(vec(v)) for (k, v) in source)
    elseif source isa AbstractString
        p = String(source)
        (!isabspath(p) && base_dir !== nothing) && (p = joinpath(base_dir, p))
        if !(lowercase(splitext(p)[2]) in (".m", ".mat", ".csv"))
            found = [p * e for e in (".m", ".mat", ".csv") if isfile(p * e)]
            length(found) == 1 || throw(EstimationError("data file $p: $(isempty(found) ? "no" : "more than one") file with extension .m, .mat or .csv"))
            p = found[1]
        end
        isfile(p) || throw(EstimationError("data file $p not found"))
        ext = lowercase(splitext(p)[2])
        table = ext == ".m" ? read_m_data(p) : ext == ".mat" ? _read_mat(p) : _read_csv(p)
    elseif source isa AbstractMatrix
        size(source, 2) == length(varobs) || throw(EstimationError("data array must have one column per observed variable ($(length(varobs)))"))
        return Matrix{Float64}(source)
    else
        throw(EstimationError("unsupported data source $(typeof(source))"))
    end
    miss = [v for v in varobs if !haskey(table, v)]
    isempty(miss) || throw(EstimationError("observed variables not found in the data: $miss"))
    cols = [table[v] for v in varobs]
    length(unique(length.(cols))) == 1 || throw(EstimationError("observed variables have different numbers of observations"))
    reduce(hcat, cols)
end

# ====================================================================== numerical Hessian
"""
    hessian(f, x; gstep=(1e-2, 1.0))

Numerical Hessian of a scalar function, the algorithm of Dynare's `hessian.m` (Abramowitz and Stegun 25.3.23 and
25.3.27) with the same step sizes.
"""
function hessian(func, x::AbstractVector, gstep=(1e-2, 1.0))
    x = Float64.(x)
    n = length(x)
    h1 = max.(abs.(x), sqrt(gstep[1])) .* eps()^(1 / 6) .* gstep[2]
    h_1 = copy(h1)
    h1 = (x .+ h1) .- x
    h_1 = x .- (x .- h_1)
    f0 = func(x)
    f1, f_1 = zeros(n), zeros(n)
    for i in 1:n
        xh = copy(x)
        xh[i] = x[i] + h1[i]
        f1[i] = func(xh)
        xh[i] = x[i] - h_1[i]
        f_1[i] = func(xh)
    end
    temp = f1 .+ f_1 .- f0
    hh = zeros(n, n)
    for i in 1:n
        hh[i, i] = (f1[i] + f_1[i] - 2 * f0) / (h1[i] * h_1[i])
        for j in i+1:n
            xp, xm = copy(x), copy(x)
            xp[i] = x[i] + h1[i]; xp[j] = x[j] + h_1[j]
            xm[i] = x[i] - h1[i]; xm[j] = x[j] - h_1[j]
            hh[i, j] = hh[j, i] = -(-func(xp) - func(xm) + temp[i] + temp[j]) / (2 * h1[i] * h_1[j])
        end
    end
    hh
end

# ====================================================================== csminwel (mode_compute=4)
"""
    penalty_objective(fcn, x, base_penalty) -> (f, ok)

`penalty_objective_function.m`: `fcn(x)` returns `(fval, info, penalty)` like `dsge_likelihood`; a rejected point is
valued `base_penalty + penalty`.
"""
function penalty_objective(fcn, x, base_penalty)
    fval, info, pen = fcn(x)
    info != 0 ? (base_penalty + pen, false) : (fval, true)
end

function _numgrad(method, fcn, f0, x0, penalty, epsilon)
    n = length(x0)
    g = zeros(n)
    badg = false
    x = copy(x0)
    for i in 1:n
        xi = x[i]
        if method == 2
            x[i] = xi + epsilon
            fh = penalty_objective(fcn, x, penalty)[1]
            g0 = (fh - f0) / epsilon
        else
            x[i] = xi + epsilon
            f1 = penalty_objective(fcn, x, penalty)[1]
            x[i] = xi - epsilon
            f2 = penalty_objective(fcn, x, penalty)[1]
            g0 = (f1 - f2) / (2 * epsilon)
        end
        if abs(g0) < 1e15
            g[i] = g0
        else
            g[i] = 0.0
            badg = true
        end
        x[i] = xi
    end
    g, badg, (method == 2 ? n : 2n)
end

function _bfgsi(H0, dg, dx)
    Hdg = H0 * dg
    dgdx = dot(dg, dx)
    abs(dgdx) > 1e-12 || return H0
    H0 + (1 + dot(dg, Hdg) / dgdx) * (dx * dx') / dgdx - (dx * Hdg' + Hdg * dx') / dgdx
end

function _csminit(fcn, x0, penalty, f0, g0, badg, H0)
    ANGLE, THETA, FCHANGE, MINLAMB, MINDFAC = 0.005, 0.3, 1000.0, 1e-9, 0.01
    fcount = 0
    lambda = 1.0
    xhat = copy(x0)
    fhat = f0
    g = g0
    gnorm = norm(g)
    retcode = 0
    if gnorm < 1e-12 && !badg
        return fhat, xhat, fcount, 1
    end
    dx = -H0 * g
    dxnorm = norm(dx)
    dxnorm > 1e12 && (dx = dx * FCHANGE / dxnorm)
    dfhat = dot(dx, g0)
    if !badg
        a = -dfhat / (gnorm * dxnorm)
        if a < ANGLE
            dx = dx - (ANGLE * dxnorm / gnorm + dfhat / (gnorm * gnorm)) * g
            dfhat = dot(dx, g)
            dxnorm = norm(dx)
        end
    end
    done = false
    factor = 3.0
    shrink = true
    lambdaMax = Inf
    lambdaPeak = 0.0
    fPeak = f0
    while !done
        dxtest = x0 + dx * lambda
        f = penalty_objective(fcn, dxtest, penalty)[1]
        if f < fhat
            fhat = f
            xhat = dxtest
        end
        fcount += 1
        shrinkSignal = (!badg && (f0 - f < max(-THETA * dfhat * lambda, 0.0))) || (badg && (f0 - f) < 0)
        growSignal = !badg && (lambda > 0 && (f0 - f > -(1 - THETA) * dfhat * lambda))
        if shrinkSignal && (lambda > lambdaPeak || lambda < 0)
            if lambda > 0 && (!shrink || lambda / factor <= lambdaPeak)
                shrink = true
                factor = factor^0.6
                while lambda / factor <= lambdaPeak
                    factor = factor^0.6
                end
                if abs(factor - 1) < MINDFAC
                    retcode = abs(lambda) < 4 ? 2 : 7
                    done = true
                end
            end
            (lambda < lambdaMax && lambda > lambdaPeak) && (lambdaMax = lambda)
            lambda = lambda / factor
            if abs(lambda) < MINLAMB
                if lambda > 0 && f0 <= fhat
                    lambda = -lambda * factor^6          # try going against the gradient, which may be inaccurate
                else
                    retcode = lambda < 0 ? 6 : 3
                    done = true
                end
            end
        elseif (growSignal && lambda > 0) || (shrinkSignal && (lambda <= lambdaPeak && lambda > 0))
            if shrink
                shrink = false
                factor = factor^0.6
                if abs(factor - 1) < MINDFAC
                    retcode = abs(lambda) < 4 ? 4 : 7
                    done = true
                end
            end
            if f < fPeak && lambda > 0
                fPeak = f
                lambdaPeak = lambda
                lambdaMax <= lambdaPeak && (lambdaMax = lambdaPeak * factor * factor)
            end
            lambda = lambda * factor
            if abs(lambda) > 1e20
                retcode = 5
                done = true
            end
        else
            done = true
            retcode = factor < 1.2 ? 7 : 0
        end
    end
    fhat, xhat, fcount, retcode
end

const _CSMINWEL_MESSAGES = Dict(0 => "normal step", 1 => "zero gradient", 6 => "smallest step still improving too slow, reversed gradient",
    5 => "largest step still improving too fast", 4 => "back and forth on step length never finished",
    2 => "back and forth on step length never finished", 3 => "smallest step still improving too slow",
    7 => "warning: possible inaccuracy in H matrix", 8 => "warning: iteration count termination", -1 => "Bad initial parameter.",
    101 => "stuck")

"""
    csminwel(fcn, x0; H0, crit, nit, method, epsilon, rng) -> (fh, xh, gh, H, itct, fcount, retcode, message)

Sims' csminwel as Dynare's `csminwel1.m` (mode_compute=4) with numerical gradients (`gradient_method` 2 or 3,
`gradient_epsilon`). `fcn(x)` returns `(fval, info, penalty)`.
"""
function csminwel(fcn, x0::AbstractVector; H0=nothing, crit=1e-7, nit=1000, method=2, epsilon=1e-6, rng=Random.default_rng())
    nx = length(x0)
    H0 = H0 === nothing ? 1e-4 * Matrix{Float64}(I, nx, nx) : H0
    penalty = 1e8
    fcount = 0
    f0, ok = penalty_objective(fcn, x0, penalty)
    fcount += 1
    ok || return nothing, nothing, nothing, nothing, 0, fcount, -1, _CSMINWEL_MESSAGES[-1]
    g, badg, fcg = _numgrad(method, fcn, f0, x0, penalty, epsilon)
    fcount += fcg
    x = Float64.(x0); f = f0; H = H0
    done = false
    itct = 0
    fh = f; xh = x; gh = nothing; retcodeh = 0
    while !done
        penalty = f
        g1 = g2 = g3 = nothing
        itct += 1
        f1, x1, fc, retcode1 = _csminit(fcn, x, penalty, f, g, badg, H)
        fcount += fc
        local f2, f3, x2, x3, badg1, badg2, badg3, retcode2, retcode3
        badg1 = true
        if retcode1 != 1
            if retcode1 == 2 || retcode1 == 4
                wall1 = true; badg1 = true
            else
                g1, badg1, fcg = _numgrad(method, fcn, f1, x1, penalty, epsilon)
                fcount += fcg
                wall1 = badg1
            end
            if wall1
                Hcliff = H + Diagonal(diag(H) .* rand(rng, nx))
                f2, x2, fc, retcode2 = _csminit(fcn, x, penalty, f, g, badg, Hcliff)
                fcount += fc
                if f2 < f
                    if retcode2 == 2 || retcode2 == 4
                        wall2 = true; badg2 = true
                    else
                        g2, badg2, fcg = _numgrad(method, fcn, f2, x2, penalty, epsilon)
                        fcount += fcg
                        wall2 = badg2
                    end
                    if wall2
                        if norm(x2 - x1) < 1e-13
                            f3 = f; x3 = x; badg3 = true; retcode3 = 101
                        else
                            gcliff = ((f2 - f1) / (norm(x2 - x1))^2) * (x2 - x1)
                            f3, x3, fc, retcode3 = _csminit(fcn, x, penalty, f, gcliff, false, Matrix{Float64}(I, nx, nx))
                            fcount += fc
                            if retcode3 == 2 || retcode3 == 4
                                badg3 = true
                            else
                                g3, badg3, fcg = _numgrad(method, fcn, f3, x3, penalty, epsilon)
                                fcount += fcg
                            end
                        end
                    else
                        f3 = f; x3 = x; badg3 = true; retcode3 = 101
                    end
                else
                    f3 = f; x3 = x; badg3 = true; retcode3 = 101
                end
            else
                f2 = f; f3 = f; x2 = x; x3 = x; badg2 = true; badg3 = true; retcode2 = 101; retcode3 = 101
            end
        else
            f2 = f; f3 = f; f1 = f; retcode2 = retcode1; retcode3 = retcode1
            x1 = x; x2 = x; x3 = x; badg2 = true; badg3 = true
        end
        # how to pick gh and xh
        badgh = true
        if f3 < f - crit && !badg3 && f3 < f2 && f3 < f1
            fh, xh, gh, badgh, retcodeh = f3, x3, g3, badg3, retcode3
        elseif f2 < f - crit && !badg2 && f2 < f1
            fh, xh, gh, badgh, retcodeh = f2, x2, g2, badg2, retcode2
        elseif f1 < f - crit && !badg1
            fh, xh, gh, badgh, retcodeh = f1, x1, g1, badg1, retcode1
        else
            ih = argmin([f1, f2, f3])
            fh = [f1, f2, f3][ih]
            xh = (x1, x2, x3)[ih]
            retcodeh = (retcode1, retcode2, retcode3)[ih]
            badgh = true
            if gh === nothing                   # Dynare recomputes the gradient only when `gh` is empty (first iteration)
                gh, badgh, fcg = _numgrad(method, fcn, fh, xh, penalty, epsilon)
                fcount += fcg
            end
        end
        stuck = abs(fh - f) < crit
        if !badg && !badgh && !stuck
            H = _bfgsi(H, gh - g, xh - x)
        end
        if itct > nit
            done = true
            retcodeh = 8
        elseif stuck
            done = true
        end
        f = fh; x = xh; g = gh; badg = badgh
    end
    fh, xh, gh, H, itct, fcount, retcodeh, get(_CSMINWEL_MESSAGES, retcodeh, "unaccounted case")
end

# ====================================================================== results
"Posterior mode (or maximum-likelihood estimate)."
mutable struct ModeResult
    x::Vector{Float64}                 # mode, ordered as Estimation.names
    names::Vector{String}
    log_posterior::Float64             # log posterior kernel (log likelihood for ML) at the mode
    hessian::Union{Nothing,Matrix{Float64}}      # Hessian of minus the log posterior
    inv_hessian::Union{Nothing,Matrix{Float64}}  # posterior covariance approximation
    laplace::Float64                   # Laplace approximation of the log marginal data density
    success::Bool
    message::String
    nfev::Int
    iterations::Int
end
ModeResult(x, names, lp; kw...) = ModeResult(x, names, lp, nothing, nothing, NaN, true, "", 0, 0)

"Standard deviations from the inverse Hessian (NaN if it could not be computed)."
mode_std(m::ModeResult) = m.inv_hessian === nothing ? fill(NaN, length(m.x)) : sqrt.(diag(m.inv_hessian))

"Output of the Metropolis-Hastings sampler: draws are (chains, replic, parameters)."
struct MCMCResult
    draws::Array{Float64,3}
    log_posterior::Matrix{Float64}
    acceptance_rate::Vector{Float64}
    names::Vector{String}
    x0::Matrix{Float64}                # initial point of each chain
    drop::Float64
    accepted::Matrix{Bool}
end

"Draws after discarding the first `floor(drop*replic)` of every chain (Dynare's `CutSample`), pooled: (draws, parameters)."
function kept_draws(mc::MCMCResult; drop=mc.drop)
    first = floor(Int, drop * size(mc.draws, 2))
    d = mc.draws[:, first+1:end, :]
    reshape(permutedims(d, (2, 1, 3)), :, size(d, 3))
end

"""
    posterior_moments(x, mh_conf_sig) -> (mean, median, var, hpd, deciles)

Dynare's `posterior_moments.m`: the HPD interval is the shortest interval holding `mh_conf_sig` of the sorted draws,
deciles are order statistics at `round(k/10 * N)`.
"""
function posterior_moments(x::AbstractVector, mh_conf_sig=0.9)
    xx = sort(Float64.(x))
    n = length(xx)
    post_mean, post_var = mean_(xx), var_(xx)
    post_median = iseven(n) ? (xx[n÷2] + xx[n÷2+1]) / 2 : xx[(n+1)÷2]
    hpd_draws = round_half_up((1 - mh_conf_sig) * n)
    hpd = fill(NaN, 2)
    if hpd_draws > 2
        # MATLAB: jj = n - hpd_draws, kk(ii) = xx(jj+ii-1) - xx(ii)
        kk = [xx[(n - hpd_draws) + ii - 1] - xx[ii] for ii in 1:hpd_draws]
        kmin, idx = findmin(kk)
        hpd = [xx[idx], xx[idx] + kmin]
    end
    dec = n > 9 ? [xx[max(1, round_half_up(k / 10 * n))] for k in 1:9] : fill(NaN, 9)
    post_mean, post_median, post_var, hpd, dec
end
mean_(x) = sum(x) / length(x)
var_(x) = (m = mean_(x); sum(abs2, x .- m) / (length(x) - 1))
round_half_up(x) = Int(floor(x + 0.5))

# ====================================================================== the estimation object
const _COSMETIC_EST = Set(["nograph", "nodisplay", "graph_format", "TeX", "tex", "nodiagnostic", "console_mode", "silent_optimizer",
    "plot_priors", "debug", "noprint", "no_graph.posterior", "graph_save_formats", "dirname", "verbosity", "mode_check.nolik",
    "mode_check.neighbourhood_size", "mode_check.symmetric_plots", "xls_sheet", "xls_range", "sub_draws", "nodecomposition"])
# options the code acts on
const _HONOURED_EST = Set(["datafile", "first_obs", "nobs", "loglinear", "logdata", "prefilter", "presample", "lik_init",
    "diffuse_filter", "kalman_algo", "kalman_tol", "diffuse_kalman_tol", "riccati_tol", "prior_trunc", "mode_compute", "mh_replic",
    "mh_nblck", "mh_jscale", "mh_drop", "mh_init_scale_factor", "order", "use_univariate_filters_if_singularity_is_detected",
    "rescale_prediction_error_covariance", "cova_compute", "mh_conf_sig", "optim_opt", "gstep", "Harvey_scale_factor",
    "lyapunov_complex_threshold", "mode_check.status", "nodiagnostic",
    "smoother", "filtered_vars", "filter_step_ahead", "filter_covariance", "smoothed_state_uncertainty", "forecast", "forecasts.conf_sig"])
const _EST_DEFAULTS = Dict{String,Any}("datafile" => nothing, "first_obs" => 1, "nobs" => nothing, "loglinear" => false,
    "logdata" => false, "prefilter" => 0, "presample" => 0, "lik_init" => 1, "diffuse_filter" => false, "kalman_algo" => 0,
    "kalman_tol" => 1e-10, "diffuse_kalman_tol" => 1e-6, "riccati_tol" => 1e-6, "prior_trunc" => 0.0, "mode_compute" => 5,
    "mh_replic" => 20000, "mh_nblck" => 2, "mh_jscale" => nothing, "mh_drop" => 0.5, "mh_init_scale_factor" => 2.0, "order" => 1,
    "use_univariate_filters_if_singularity_is_detected" => true, "rescale_prediction_error_covariance" => false,
    "cova_compute" => 1, "mh_conf_sig" => 0.9, "optim_opt" => "", "gstep" => (1e-2, 1.0), "Harvey_scale_factor" => 10.0,
    "lyapunov_complex_threshold" => 1e-15, "qz_criterium" => nothing)

_est_flag(v) = !(v === false || v === 0 || v === 0.0 || v === nothing || v == "")

mutable struct Estimation
    model::Model
    options::Dict{String,Any}
    varobs::Vector{String}
    obs_decl::Vector{Int}
    names::Vector{String}
    x0::Vector{Float64}
    lb::Vector{Float64}                # prior support intersected with the bounds of estimated_params
    ub::Vector{Float64}
    prior::Prior
    jscale_individual::Vector{Float64}
    n::Int
    groups::NamedTuple
    Sigma_e0::Matrix{Float64}
    corr0::Matrix{Float64}
    H0::Matrix{Float64}
    Q_check::Vector{Int}
    H_check::Vector{Int}
    restrict::Vector{Int}              # declaration indices of the filter states
    mf::Vector{Int}                    # position of the observed variables in the filter state vector
    restrict_columns::Vector{Int}
    state_var::Vector{Int}
    loggable::BitVector
    trend_expr::Union{Nothing,Vector{Any}}
    rawdata::Matrix{Float64}
    data::Matrix{Float64}              # periods x observed variables
    nobs::Int
    data_index::Union{Nothing,Vector{Vector{Int}}}
    no_more_missing::Int
    unsupported::Vector{String}
    last_info::String
    last_filter::String
    last_info_code::Int
end

"""
    Estimation(m::Model; data=nothing, allow_unsupported=false, options...)

Estimation problem for the first-order approximation of `m`, which needs a `varobs` statement and an
`estimated_params` block. For a model with `occbin_constraints` the likelihood is OccBin's (piecewise Kalman filter, or the
inversion filter; `occbin=Dict(...)` overrides the options of `occbin_setup`, see src/occbin_estimation.jl). Options are those of the `estimation` statement of the .mod file (the last one), overridden
by keyword arguments with Dynare's names (`first_obs`, `nobs`, `loglinear`, `logdata`, `prefilter`, `presample`,
`lik_init`, `diffuse_filter`, `kalman_algo`, `kalman_tol`, `riccati_tol`, `prior_trunc`, `mode_compute`, `mh_replic`,
`mh_nblck`, `mh_jscale`, `mh_drop`, ...). `data` is a Dict, a matrix (columns in `varobs` order) or a file name,
default the `datafile` option. Options the code does not act on raise an `EstimationError` (listing them) unless
`allow_unsupported=true`, which turns the error into one warning.
"""
function Estimation(m::Model; data=nothing, allow_unsupported::Bool=false, statement=nothing, occbin=nothing, options...)
    raw = Dict{String,Any}()
    ests = [st for st in m.statements if get(st, "statementName", "") == "estimation"]
    if statement === nothing
        length(ests) > 1 && throw(EstimationError("the .mod file has $(length(ests)) estimation statements; pass statement=k to choose one"))
    end
    if !isempty(ests) && statement != 0          # statement=0: the options come from the keywords alone (dynare()'s options_)
        st = ests[statement === nothing ? end : statement]
        merge!(raw, get_or(st, "options", Dict{String,Any}()))
    end
    for (k, v) in options
        raw[string(k) == "mh_nblocks" ? "mh_nblck" : string(k)] = v
    end
    opts = copy(_EST_DEFAULTS)
    unsupported = String[]
    for (k, v) in raw
        k == "mh_nblocks" && (k = "mh_nblck")
        k == "optim" && (k = "optim_opt")
        if v isa AbstractVector && length(v) == 1 && k != "gstep"
            v = v[1]
        end
        if (k == "nobs" || k == "first_obs") && v isa AbstractVector
            throw(EstimationError("recursive estimation ($k given as a vector) is not supported"))
        end
        if k in _HONOURED_EST
            opts[k] = v
        elseif k in _COSMETIC_EST
        elseif !_est_flag(v)
        else
            push!(unsupported, k)
        end
    end
    sort!(unsupported)
    if !isempty(unsupported)
        msg = "Estimation does not act on these estimation options: " * join(unsupported, ", ") *
              " (forecasts, posterior IRFs and moments, higher orders, other samplers and optimisers are not implemented)"
        allow_unsupported ? warn_once(m, msg) : throw(EstimationError(msg))
    end
    haskey(raw, "mode_check.status") && _est_flag(raw["mode_check.status"]) && warn_once(m, "estimation: mode_check draws plots only; no plots are made")
    Int(opts["order"]) == 1 || throw(EstimationError("estimation is implemented for order=1 only"))
    opts["loglinear"] = _est_flag(opts["loglinear"])
    opts["logdata"] = _est_flag(opts["logdata"])
    opts["rescale_prediction_error_covariance"] = _est_flag(opts["rescale_prediction_error_covariance"])
    opts["use_univariate_filters_if_singularity_is_detected"] = _est_flag(opts["use_univariate_filters_if_singularity_is_detected"])
    opts["lik_init"] = Int(opts["lik_init"])
    opts["kalman_algo"] = Int(opts["kalman_algo"])
    opts["mode_compute"] = Int(opts["mode_compute"])
    opts["presample"] = Int(opts["presample"])
    opts["mh_replic"] = Int(opts["mh_replic"])
    opts["mh_nblck"] = Int(opts["mh_nblck"])
    opts["prefilter"] = Int(opts["prefilter"])
    for k in ("mh_drop", "mh_conf_sig", "prior_trunc", "mh_init_scale_factor", "kalman_tol", "diffuse_kalman_tol", "riccati_tol", "Harvey_scale_factor")
        opts[k] = Float64(opts[k])
    end
    if _est_flag(opts["diffuse_filter"]) || opts["kalman_algo"] in (3, 4)
        opts["lik_init"] == 2 && throw(EstimationError("options diffuse_filter, lik_init and/or kalman_algo have contradictory settings"))
        opts["lik_init"] = 3
    end
    opts["kalman_algo"] in (0, 1, 2, 3, 4) || throw(EstimationError("kalman_algo=$(opts["kalman_algo"]) is not implemented"))
    (opts["kalman_algo"] in (1, 2) && opts["lik_init"] == 3) && throw(EstimationError("the diffuse filter needs kalman_algo 0, 3 or 4"))
    opts["lik_init"] in (1, 2, 3) || throw(EstimationError("lik_init=$(opts["lik_init"]) is not implemented"))
    # select_qz_criterium_value.m (qz_criterium is a global option, not an estimation option)
    qz = solver_option(m, "qz_criterium", nothing)
    qz = qz === nothing ? nothing : Float64(qz)
    if qz === nothing
        qz = opts["lik_init"] == 1 ? 1 - 1e-6 : 1 + 1e-6
    elseif opts["lik_init"] == 1
        qz > 1 - eps() && throw(EstimationError("option qz_criterium is too large for estimating a stationary model. If your model contains unit roots, use option diffuse_filter"))
    elseif qz <= 1
        qz = 1 + 1e-6
    end
    opts["qz_criterium"] = qz
    opts["Harvey_scale_factor"] = Float64(solver_option(m, "Harvey_scale_factor", opts["Harvey_scale_factor"]))
    opts["lyapunov_complex_threshold"] = Float64(solver_option(m, "lyapunov_complex_threshold", opts["lyapunov_complex_threshold"]))
    if opts["gstep"] isa AbstractVector
        opts["gstep"] = Tuple(Float64.(opts["gstep"]))
    end
    varobs = copy(m.varobs)
    isempty(varobs) && throw(EstimationError("the model has no varobs statement"))
    length(unique(varobs)) == length(varobs) || throw(EstimationError("a variable is declared twice in varobs"))
    for v in varobs
        haskey(m.ie, v) || throw(EstimationError("observed variable $v is not an endogenous variable"))
    end
    obs_decl = Int[m.ie[v] for v in varobs]
    est = Estimation(m, opts, varobs, obs_decl, String[], Float64[], Float64[], Float64[], Prior(Int[], Float64[], Float64[]),
                     Float64[], 0, (vx=Any[], vn=Any[], cx=Any[], cn=Any[], p=Any[]), zeros(0, 0), zeros(0, 0), zeros(0, 0),
                     Int[], Int[], Int[], Int[], Int[], Int[], trues(0), nothing, zeros(0, 0), zeros(0, 0), 0, nothing, 1,
                     unsupported, "", "", 0)
    _setup_parameters!(est)
    _setup_state_space!(est)
    _setup_trends!(est)
    _setup_data!(est, data)
    _setup_occbin!(est, occbin)                  # OccBin models: piecewise Kalman or inversion filter (src/occbin_estimation.jl)
    if opts["prefilter"] != 0
        centred = try
            check_prefilter(est)
        catch e
            (e isa SteadyStateError || e isa BlanchardKahnError || e isa LoglinearError) || rethrow()
            true
        end
        centred || throw(EstimationError("the option \"prefilter\" is inconsistent with the non-zero mean measurement equations in the model"))
    end
    est
end

_num(est::Estimation, s) = s isa Real ? Float64(s) : Float64(_value(est.model, s))

# ------------------------------------------------------------------ parameters and priors
function _setup_parameters!(est::Estimation)
    m = est.model
    blocks = [st for st in m.statements if get(st, "statementName", "") == "estimated_params"]
    for st in m.statements
        k = get(st, "statementName", "")
        k in ("estimated_params_remove", "prior", "std_prior", "corr_prior") && throw(EstimationError("$k blocks are not supported"))
    end
    isempty(blocks) && throw(EstimationError("the model has no estimated_params block"))
    # several blocks: the JSON does not say whether a later block has the 'overwrite' option
    length(blocks) > 1 && throw(EstimationError("several estimated_params blocks are not supported"))
    entries = collect(get_or(blocks[1], "params", []))
    isempty(entries) && throw(EstimationError("the estimated_params block is empty"))
    vx, vn, cx, cn, p = Any[], Any[], Any[], Any[], Any[]
    for e in entries
        if haskey(e, "param")
            haskey(m.ip, e["param"]) || throw(EstimationError("estimated_params: unknown parameter $(e["param"])"))
            push!(p, (e, m.ip[e["param"]], e["param"]))
        elseif haskey(e, "var")
            v = e["var"]
            if haskey(m.ix, v)
                push!(vx, (e, m.ix[v], "stderr $v"))
            elseif v in est.varobs
                push!(vn, (e, findfirst(==(v), est.varobs), "stderr $v"))
            else
                throw(EstimationError("the variable $v has to be declared as observable since you assume a measurement error on it"))
            end
        elseif haskey(e, "var1")
            a, b = e["var1"], e["var2"]
            if haskey(m.ix, a) && haskey(m.ix, b)
                push!(cx, (e, (m.ix[a], m.ix[b]), "corr $a, $b"))
            elseif a in est.varobs && b in est.varobs
                push!(cn, (e, (findfirst(==(a), est.varobs), findfirst(==(b), est.varobs)), "corr $a, $b"))
            else
                throw(EstimationError("corr $a, $b: both must be shocks or both observed variables"))
            end
        else
            throw(EstimationError("estimated_params entry not supported: $e"))
        end
    end
    est.groups = (vx=vx, vn=vn, cx=cx, cn=cn, p=p)
    ordered = vcat(vx, vn, cx, cn, p)
    est.names = String[o[3] for o in ordered]
    get_(key) = Float64[_num(est, o[1][key]) for o in ordered]
    init, lb, ub = get_("init_val"), get_("lower_bound"), get_("upper_bound")
    n = length(ordered)
    ncorr0 = length(vx) + length(vn)
    ncorr1 = ncorr0 + length(cx) + length(cn)
    ub[ncorr0+1:ncorr1] = clamp.(ub[ncorr0+1:ncorr1], -1, 1)
    lb[ncorr0+1:ncorr1] = clamp.(lb[ncorr0+1:ncorr1], -1, 1)
    shape = Int[Int(o[1]["prior_distribution"]) for o in ordered]
    (any(==(0), shape) && any(!=(0), shape)) && throw(EstimationError("estimation must be either fully ML or fully Bayesian"))
    est.prior = try
        Prior(shape, get_("mean"), get_("std"), get_("p3"), get_("p4"); names=est.names)
    catch e
        e isa PriorError && throw(EstimationError(e.msg))
        rethrow()
    end
    est.jscale_individual = get_("jscale")
    est.n = n
    # estimated_params_init / estimated_params_bounds blocks, in the order of the file; use_calibration
    use_calibration = false
    for st in m.statements
        k = get(st, "statementName", "")
        if k == "estimated_params_init"
            if get(st, "use_calibration_initialization", 0) in (1, true) || get(st, "use_calibration", 0) in (1, true)
                use_calibration = true
            end
            for e in get_or(st, "params", [])
                nm = _entry_name(e)
                i = findfirst(==(nm), est.names)
                i === nothing && throw(EstimationError("estimated_params_init: $nm is not estimated"))
                init[i] = _num(est, e["init_val"])
            end
        elseif k == "estimated_params_bounds"
            for e in get_or(st, "params", [])
                nm = _entry_name(e)
                i = findfirst(==(nm), est.names)
                i === nothing && throw(EstimationError("estimated_params_bounds: $nm is not estimated"))
                lb[i] = _num(est, e["lower_bound"])
                ub[i] = _num(est, e["upper_bound"])
                (ncorr0 < i <= ncorr1) && (lb[i] = max(lb[i], -1); ub[i] = min(ub[i], 1))
            end
        end
    end
    # calibrated covariance structure
    S = Matrix{Float64}(m.Sigma_e)
    sd = sqrt.(diag(S))
    C = S ./ (sd * sd')
    C[.!isfinite.(C)] .= 0.0
    for i in axes(C, 1)
        C[i, i] = 1.0
    end
    est.Sigma_e0, est.corr0 = S, C
    pp = length(est.varobs)
    H = zeros(pp, pp)
    for (name, var) in m.measurement_errors
        name in est.varobs || throw(EstimationError("measurement error on $name, which is not an observed variable"))
        i = findfirst(==(name), est.varobs)
        H[i, i] = var
    end
    est.H0 = H
    any(s -> occursin("measurement-error covariance", s), m.unhandled_statements) &&
        warn_once(m, "calibrated measurement-error covariances are ignored by estimation")
    nz = .!vec(all(S .== 0; dims=1))
    for (_, k, _) in vx
        nz[k] = true
    end
    est.Q_check = findall(nz)
    nz = .!vec(all(H .== 0; dims=1))
    for (_, k, _) in vn
        nz[k] = true
    end
    est.H_check = findall(nz)
    if use_calibration
        calib = _calibrated_values(est)
        all(==(0), shape) && any(i -> isnan(init[i]) && isnan(calib[i]), 1:n) &&
            throw(EstimationError("When using the use_calibration option with ML, the parameters must be properly initialized."))
        for i in 1:n
            if isnan(init[i]) && !isnan(calib[i])
                (calib[i] == 0 && shape[i] == 4) || (init[i] = calib[i])
            end
        end
    end
    k = isnan.(init)
    any(k) && all(==(0), shape) && throw(EstimationError("ML estimation requires all estimated parameters to be initialized, either in an estimated_params or estimated_params_init-block"))
    init[k] = est.prior.p1[k]
    est.x0 = init
    if is_bayesian(est.prior)
        plb, pub = prior_bounds(est.prior, est.options["prior_trunc"])
        est.lb, est.ub = max.(plb, lb), min.(pub, ub)
    else
        est.lb, est.ub = lb, ub
    end
    bad = findall((init .< est.lb) .| (init .> est.ub))
    isempty(bad) || throw(EstimationError("initial value(s) of $(join(est.names[bad], ", ")) outside the parameter bounds [$(join(["$(est.lb[i]), $(est.ub[i])" for i in bad], "; "))]"))
    est
end

function _entry_name(e)
    haskey(e, "param") ? e["param"] : haskey(e, "var") ? "stderr $(e["var"])" : "corr $(e["var1"]), $(e["var2"])"
end

"Calibrated values of the estimated parameters (`get_all_parameters.m`)."
function _calibrated_values(est::Estimation)
    m = est.model
    out = fill(NaN, est.n)
    o = 0
    for (_, k, _) in est.groups.vx
        out[o+1] = sqrt(est.Sigma_e0[k, k]); o += 1
    end
    for (_, k, _) in est.groups.vn
        out[o+1] = sqrt(est.H0[k, k]); o += 1
    end
    for (_, (k1, k2), _) in est.groups.cx
        out[o+1] = est.corr0[k1, k2]; o += 1
    end
    for _ in est.groups.cn
        o += 1                                          # correlations of measurement errors are not calibrated here
    end
    for (_, k, _) in est.groups.p
        out[o+1] = m.params[k]; o += 1
    end
    out
end

"Structural-shock and measurement-error covariance matrices implied by `theta` (`set_all_parameters.m`)."
function covariances(est::Estimation, theta::AbstractVector)
    g = est.groups
    Q, C = copy(est.Sigma_e0), copy(est.corr0)
    H, CH = copy(est.H0), Matrix{Float64}(I, length(est.varobs), length(est.varobs))
    o = 0
    for (i, (_, k, _)) in enumerate(g.vx)
        Q[k, k] = theta[o+i]^2
    end
    o += length(g.vx)
    for (i, (_, k, _)) in enumerate(g.vn)
        H[k, k] = theta[o+i]^2
    end
    o += length(g.vn)
    for (i, (_, (k1, k2), _)) in enumerate(g.cx)
        C[k1, k2] = C[k2, k1] = theta[o+i]
    end
    o += length(g.cx)
    for (i, (_, (k1, k2), _)) in enumerate(g.cn)
        CH[k1, k2] = CH[k2, k1] = theta[o+i]
    end
    if !isempty(g.vx) || !isempty(g.cx)
        sd = sqrt.(diag(Q))
        Q = C .* (sd * sd')
    end
    if !isempty(g.vn) || !isempty(g.cn)
        sd = sqrt.(diag(H))
        H = CH .* (sd * sd')
    end
    Q, H
end

"Copy of the model's parameter vector with the estimated structural parameters set."
function model_params(est::Estimation, theta::AbstractVector)
    p = copy(est.model.params)
    o = est.n - length(est.groups.p)
    for (i, (_, k, _)) in enumerate(est.groups.p)
        p[k] = theta[o+i]
    end
    p
end

# ------------------------------------------------------------------ state space
function _setup_state_space!(est::Estimation)
    m = est.model
    order_var, state_var, counts = dr_ordering(m.lead_lag_incidence)
    inv = invperm(order_var)
    obs_dr = inv[est.obs_decl]
    state_dr = collect(counts.nstatic+1:counts.nstatic+counts.npred+counts.nboth)
    k2 = sort(union(obs_dr, state_dr))
    est.restrict = order_var[k2]
    est.mf = Int[findfirst(==(o), k2) for o in obs_dr]
    est.restrict_columns = Int[findfirst(==(s), k2) for s in state_dr]
    est.state_var = state_var
    est.loggable = log_mask(m)
    est
end

"""
    solve(est, theta) -> (ys, G, H, params)

Steady state and first-order solution at `theta`: `y_t - ys = G (y_{t-1} - ys) + H u_t` in declaration order (in logs
under `loglinear`) and the parameter vector actually used. Throws `SteadyStateError` or `BlanchardKahnError`.
"""
function solve(est::Estimation, theta::AbstractVector)
    m = est.model
    o = est.options
    sol = first_order(m; params=model_params(est, theta), qz_criterium=o["qz_criterium"], loglinear=o["loglinear"])
    all(isfinite, sol.params) || throw(SteadyStateError("parameters are NaN or Inf after the steady-state computation"))
    (all(isfinite, sol.G) && all(isfinite, sol.H)) || throw(BlanchardKahnError("the decision rule contains NaN or Inf"))
    sol.ys, sol.G, sol.H, sol.params
end

"Matrices of the filter at `theta`: `T`, `R`, `Q`, `H`, observation index `z`, measurement-equation `constant`, steady state and parameters."
function state_space(est::Estimation, theta::AbstractVector)
    Q, Hm = covariances(est, theta)
    ys, G, Hs, params = solve(est, theta)
    r = est.restrict
    (T=G[r, r], R=Hs[r, :], Q=Q, H=Hm, z=est.mf, constant=ys[est.obs_decl], ys=ys, params=params, G=G, Hfull=Hs,
     state_names=m_names(est, r))
end
m_names(est, r) = est.model.endo_names[r]

# ------------------------------------------------------------------ data
function _setup_trends!(est::Estimation)
    est.trend_expr = nothing
    for st in est.model.statements
        get(st, "statementName", "") == "observation_trends" || continue
        tr = get_or(st, "trends", Dict{String,Any}())
        unknown = [v for v in keys(tr) if !(v in est.varobs)]
        isempty(unknown) || throw(EstimationError("observation_trends on non-observed variables: $unknown"))
        est.trend_expr = Any[get(tr, v, nothing) for v in est.varobs]
    end
end

function _trend(est::Estimation, params, nobs)
    est.trend_expr === nothing && return nothing
    m = est.model
    locals = Dict{String,Float64}(zip(m.param_names, params))
    coeff = Float64[e === nothing ? 0.0 : Float64(_value(m, e; locals=locals)) for e in est.trend_expr]
    first = Int(est.options["first_obs"])
    add = coeff * collect(Float64, first:first+nobs-1)'
    est.options["prefilter"] != 0 && (add = add .- sum(add; dims=2) ./ nobs)
    add
end

function _setup_data!(est::Estimation, data)
    o = est.options
    if data === nothing
        o["datafile"] === nothing && throw(EstimationError("no data: pass data=... or set the datafile option"))
        data = o["datafile"]
    end
    raw = load_data(data, est.varobs; base_dir=est.model.mod_dir)
    if o["loglinear"] && !o["logdata"]
        any(x -> !isnan(x) && x <= 0, raw) && throw(EstimationError("loglinear without logdata: the data contain non-positive values, whose logarithm is complex"))
        raw = log.(raw)
    end
    first = Int(o["first_obs"])
    nobs = o["nobs"] === nothing ? size(raw, 1) - first + 1 : Int(o["nobs"])
    (first < 1 || nobs < 1 || first - 1 + nobs > size(raw, 1)) &&
        throw(EstimationError("first_obs=$first and nobs=$nobs are not consistent with the $(size(raw, 1)) observations in the data"))
    raw = raw[first:first+nobs-1, :]
    est.rawdata = copy(raw)
    if o["prefilter"] != 0
        for j in axes(raw, 2)
            c = filter(!isnan, raw[:, j])
            raw[:, j] .-= isempty(c) ? 0.0 : sum(c) / length(c)
        end
    end
    est.data = raw
    est.nobs = nobs
    nan = isnan.(raw)
    if any(nan)
        est.data_index = [findall(.!nan[t, :]) for t in 1:nobs]
        rows = findall(vec(any(nan; dims=2)))
        est.no_more_missing = rows[end] + 1
    else
        est.data_index, est.no_more_missing = nothing, 1
    end
    o["presample"] >= nobs && throw(EstimationError("presample must be smaller than the number of observations"))
    est
end

"True if the observed variables have a zero steady state (zero log steady state under `loglinear`); Dynare refuses `prefilter` otherwise."
function check_prefilter(est::Estimation, theta=est.x0)
    ss = state_space(est, theta)
    all(c -> abs(c) < 1e-9, ss.constant)
end

# ------------------------------------------------------------------ likelihood
struct _Rejected <: Exception
    info::Int
    penalty::Float64
    msg::String
end

_ispd(A) = isempty(A) || (all(isfinite, A) && isposdef(Hermitian(A)))
function _ispd_penalty(A)
    _ispd(A) && return 0.0
    all(isfinite, A) || return 1.0
    ev = eigvals(Symmetric((A + A') / 2))
    neg = ev[ev .< 0]
    isempty(neg) ? 0.0 : sum(-neg)
end

"Augmented system for the univariate filter with correlated measurement errors (the errors become states, as in Dynare)."
function _expand_for_univariate(T, R, Q, P, Pinf, Z1, Hm)
    pp = size(Hm, 1)
    mm = size(T, 1)
    Z = hcat(Z1, Matrix{Float64}(I, pp, pp))
    Te = [T zeros(mm, pp); zeros(pp, mm + pp)]
    Qe = [Q zeros(size(Q, 1), pp); zeros(pp, size(Q, 1)) Hm]
    Re = [R zeros(size(R, 1), pp); zeros(pp, size(R, 2)) Matrix{Float64}(I, pp, pp)]
    Pe = P === nothing ? nothing : [P zeros(mm, pp); zeros(pp, mm) Hm]
    Pie = Pinf === nothing ? nothing : [Pinf zeros(mm, pp); zeros(pp, mm + pp)]
    Z, Te, Qe, Re, Pe, Pie
end

function _measurement_matrix(est::Estimation, mm)
    Z = zeros(length(est.mf), mm)
    for (i, k) in enumerate(est.mf)
        Z[i, k] = 1.0
    end
    Z
end

"""
    _dsge_likelihood(est, theta; bounds=(est.lb, est.ub), with_lik=false) -> (fval, info, penalty, loglik, logprior, lik)

Dynare's `dsge_likelihood`: `fval` is minus the log posterior kernel (minus the log likelihood for ML) and Inf when the
point is rejected, `info` Dynare's code (0 = fine; 41/42 bounds, 43/44 covariance not positive definite, 3/4/5 Blanchard-Kahn,
20-22 steady state, 45/46/50 likelihood, 40/47 prior), `penalty` Dynare's `info(4)`.
"""
function _dsge_likelihood(est::Estimation, theta::AbstractVector; bounds=(est.lb, est.ub), with_lik=false)
    o = est.options
    theta = Float64.(theta)
    est.last_info = ""
    est.last_info_code = 0
    reject(info, pen, msg) = (est.last_info = msg; est.last_info_code = info; (Inf, info, pen, NaN, NaN, Float64[]))
    lb, ub = bounds
    if any(theta .< lb)
        k = findall(theta .< lb)
        return reject(41, sum(abs2, lb[k] - theta[k]), "below the lower bound (info 41)")
    end
    if any(theta .> ub)
        k = findall(theta .> ub)
        return reject(42, sum(abs2, theta[k] - ub[k]), "above the upper bound (info 42)")
    end
    Q, Hm = covariances(est, theta)
    if !isempty(est.groups.cx) && !_ispd(Q[est.Q_check, est.Q_check])
        return reject(43, _ispd_penalty(Q[est.Q_check, est.Q_check]), "covariance matrix of the structural shocks is not positive definite (info 43)")
    end
    if !isempty(est.groups.cn) && !_ispd(Hm[est.H_check, est.H_check])
        return reject(44, _ispd_penalty(Hm[est.H_check, est.H_check]), "covariance matrix of the measurement errors is not positive definite (info 44)")
    end
    get(o, "occbin", nothing) === nothing || return _occbin_likelihood(est, theta, with_lik)    # src/occbin_estimation.jl
    local ys, G, Hs, params
    try
        ys, G, Hs, params = solve(est, theta)
    catch e
        if e isa SteadyStateError
            return reject(22, 0.1, "steady state: $(e.msg)")
        elseif e isa BlanchardKahnError
            pen = isfinite(e.penalty) ? e.penalty : 0.1
            return reject(3, pen, "solution: $(e.msg)")
        elseif e isa LoglinearError
            return reject(22, 0.1, "steady state: $(e.msg)")
        elseif e isa DomainError || e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException
            return reject(22, 0.1, "steady state or solution: $(sprint(showerror, e))")
        end
        rethrow()
    end
    r = est.restrict
    T, R, z = G[r, r], Hs[r, :], est.mf
    (all(isfinite, T) && all(isfinite, R)) || return reject(3, 0.1, "the decision rule contains NaN or Inf")
    mm = size(T, 1)
    pp = length(z)
    Y = permutedims(est.data) .- ys[est.obs_decl]
    trend = _trend(est, params, est.nobs)
    trend === nothing || (Y = Y .- trend)
    QQ = R * Q * R'
    a = zeros(mm)
    pre = o["presample"]
    kt, rt = o["kalman_tol"], o["riccati_tol"]
    di_all = est.data_index
    algo = o["kalman_algo"]
    dLIK = 0.0
    dp = 0
    dlik = Float64[]
    P = zeros(mm, mm)
    Pinf = nothing
    expanded = false
    Zu, Tu, Qu, Ru, QQu, Hdiag = nothing, T, Q, R, QQ, nothing
    function prepare_univariate!(P_, Pinf_)
        # diagonal measurement-error covariance: Z is a selector; otherwise the errors are added to the state vector
        if all(abs.(Hm - Diagonal(diag(Hm))) .< 1e-14)
            Zu, Tu, Qu, Ru, QQu, Hdiag = _measurement_matrix(est, mm), T, Q, R, QQ, diag(Hm)
            return P_, Pinf_
        end
        Z, Te, Qe, Re, Pe, Pie = _expand_for_univariate(T, R, Q, P_, Pinf_, _measurement_matrix(est, mm), Hm)
        expanded = true
        Zu, Tu, Qu, Ru, QQu, Hdiag = Z, Te, Qe, Re, Re * Qe * Re', zeros(pp)
        Pe, Pie
    end
    if o["lik_init"] == 1
        algo = algo == 2 ? 2 : 1
        P = lyapunov_symm(T, QQ; qz_criterium=o["qz_criterium"], complex_threshold=o["lyapunov_complex_threshold"])[1]
    elseif o["lik_init"] == 2
        algo = algo == 2 ? 2 : 1
        P = o["Harvey_scale_factor"] * Matrix{Float64}(I, mm, mm)
        o["Harvey_scale_factor"] == 0 && (P = T * P * T' + QQ)
    else
        algo = algo == 0 ? 3 : algo
        Pstar, Pinf = compute_Pinf_Pstar(z, T, R, Q, o["qz_criterium"], est.restrict_columns)
        singular = false
        P = Pstar
        if algo == 3
            dLIK, dlik, a1, P1 = kalman_filter_d(Y, a, Pinf, Pstar, T, QQ, Hm, z; kalman_tol=kt,
                                                 diffuse_kalman_tol=o["diffuse_kalman_tol"], presample=pre, data_index=di_all)
            if isinf(dLIK)
                singular = true               # as Dynare: redo the diffuse stage observation by observation
            else
                a, P = a1, P1
            end
        end
        if singular || algo == 4
            P0, Pinf0 = prepare_univariate!(Pstar, Pinf)
            a = zeros(size(P0, 1))
            dLIK, dlik, a, P = univariate_kalman_filter_d(Y, a, Pinf0, P0, Tu, QQu, Hdiag, Zu; kalman_tol=kt,
                                                          diffuse_kalman_tol=o["diffuse_kalman_tol"], presample=pre, data_index=di_all)
        end
        isnan(dLIK) && return reject(45, 0.1, "the diffuse filter did not leave the diffuse stage, or a variance is negative (info 45)")
        dp = length(dlik)
    end
    all(isfinite, P) || return reject(50, 0.1, "initial state covariance is not finite")
    di = di_all === nothing ? nothing : di_all[dp+1:end]
    nmm = max(est.no_more_missing - dp, 1)
    Ys = Y[:, dp+1:end]
    LIK = 0.0
    lik = Float64[]
    if algo == 1 || algo == 3
        LIK, lik = kalman_filter(Ys, a, P, T, QQ, Hm, z; kalman_tol=kt, riccati_tol=rt, presample=pre, diffuse_periods=dp,
                                 data_index=di, no_more_missing=nmm, rescale=o["rescale_prediction_error_covariance"])
        if isinf(LIK)
            o["use_univariate_filters_if_singularity_is_detected"] || return reject(50, 0.1, "the forecast-error covariance is singular (info 50)")
            algo = algo == 1 ? 2 : 4
        end
    end
    if algo == 2 || algo == 4
        if Zu === nothing
            P, _ = prepare_univariate!(P, nothing)
            expanded && length(a) == mm && (a = vcat(a, zeros(pp)))
        end
        LIK, lik = univariate_kalman_filter(Ys, a, P, Tu, QQu, Hdiag, Zu; kalman_tol=kt, riccati_tol=rt, presample=pre,
                                            diffuse_periods=dp, data_index=di, no_more_missing=nmm)
    end
    est.last_filter = Dict(1 => "multivariate", 2 => "univariate", 3 => "multivariate (diffuse)", 4 => "univariate (diffuse)")[algo]
    isnan(LIK) && return reject(45, 0.1, "the likelihood is NaN (info 45)")
    isinf(LIK) && return reject(50, 0.1, "the forecast-error covariance is singular (info 50)")
    LIK += dLIK
    lnprior = logpdf(est.prior, theta)
    if isinf(lnprior) || isnan(lnprior)
        info = isnan(lnprior) ? 47 : 40
        est.last_info = "the prior density is zero, infinite or NaN (info $info)"
        est.last_info_code = info
        return Inf, info, 0.1, -LIK, lnprior, Float64[]
    end
    contrib = with_lik ? vcat(dlik, lik)[pre+1:end] : Float64[]
    LIK - lnprior, 0, 0.0, -LIK, lnprior, contrib
end

"""
    log_likelihood(est, theta) -> Float64

Log likelihood of the data at `theta` (bounds are not checked); `-Inf` if the steady state or the solution fails or
the filter breaks down (the reason is kept in `est.last_info`).
"""
function log_likelihood(est::Estimation, theta::AbstractVector)
    _, info, _, ll = _dsge_likelihood(est, theta; bounds=(fill(-Inf, est.n), fill(Inf, est.n)))
    (info == 0 || info == 40 || info == 47) ? ll : -Inf
end

"Log prior density (0 for maximum likelihood)."
log_prior(est::Estimation, theta::AbstractVector) = logpdf(est.prior, theta)

"""
    log_posterior(est, theta; bounds=nothing) -> Float64

Log posterior kernel, minus the value of Dynare's `dsge_likelihood`: `-Inf` outside the bounds, when the model cannot be
solved, or when the prior density is zero. `bounds=(lb, ub)` replaces the bounds of the mode search (the sampler passes
the prior support alone, as Dynare's Metropolis-Hastings does).
"""
function log_posterior(est::Estimation, theta::AbstractVector; bounds=nothing)
    fval, info, _ = _dsge_likelihood(est, theta; bounds=bounds === nothing ? (est.lb, est.ub) : bounds)
    info == 0 ? -fval : -Inf
end

"Log density of each observation after the presample."
function log_likelihood_contributions(est::Estimation, theta::AbstractVector)
    _, info, _, _, _, lik = _dsge_likelihood(est, theta; with_lik=true)
    info == 0 || throw(EstimationError("the likelihood cannot be evaluated: $(est.last_info)"))
    -lik
end

# ------------------------------------------------------------------ mode, Hessian, Laplace
"Hessian of minus the log posterior at `theta` (Dynare's `hessian.m`)."
hessian(est::Estimation, theta::AbstractVector; gstep=est.options["gstep"]) =
    hessian(x -> -log_posterior(est, x), theta, gstep)

"Inverse computed after scaling to unit diagonal, as in `dynare_estimation_1.m`."
function inverse_hessian(hh::AbstractMatrix)
    hsd = sqrt.(diag(hh))
    s = hsd * hsd'
    inv(hh ./ s) ./ s
end

"Laplace approximation of the log marginal data density."
function laplace(est::Estimation, theta::AbstractVector, invhess::AbstractMatrix)
    stdh = sqrt.(diag(invhess))
    (any(!isfinite, stdh) || !_ispd(invhess)) && return NaN
    logdet_ = log(det(invhess ./ (stdh * stdh'))) + 2 * sum(log, stdh)
    0.5 * est.n * log(2 * pi) + 0.5 * logdet_ + log_posterior(est, theta)
end

# optim options of each mode_compute, as dynare_minimize_objective.m reads them ("verbosity"/"SaveFiles": display only)
const _OPTIM_KEYS = Dict(0 => String[], 4 => ["MaxIter", "TolFun", "NumgradEpsilon", "NumgradAlgorithm", "verbosity", "SaveFiles"],
    5 => ["MaxIter", "Hessian", "robust", "TolFun", "TolGstep", "TolGstepRel", "verbosity", "SaveFiles"],
    7 => ["MaxIter", "MaxFunEvals", "TolFun", "TolX", "Display"],
    8 => ["MaxIter", "TolFun", "TolX", "MaxFunEvals", "MaxFunEvalFactor", "InitialSimplexSize", "verbosity"],
    9 => ["MaxIter", "TolFun", "TolX", "MaxFunEvals", "verbosity", "SaveFiles"])

function _parse_optim_opt(s)
    out = Dict{String,Any}()
    s = strip(String(s))
    isempty(s) && return out
    parts = [strip(p) for p in split(s, ",")]
    length(parts) % 2 == 0 || throw(EstimationError("optim: cannot parse $(repr(s))"))
    for i in 1:2:length(parts)
        key = strip(parts[i], ['\'', '"'])
        val = strip(parts[i+1], ['\'', '"'])
        out[key] = something(tryparse(Float64, val), val)
    end
    out
end

"""
    posterior_mode(est; x0=est.x0, compute_hessian=true, rng) -> ModeResult

Maximise the log posterior (the log likelihood for ML) from `x0` (default the initial values of the `estimated_params`
block, else the prior means) with Dynare's optimiser `mode_compute`, then the Hessian by `hessian.m` and its inverse and the
Laplace approximation as `dynare_estimation_1.m`. `mode_compute=0` only evaluates the posterior at `x0`.

- 4: `csminwel` (numerical gradient, BFGS, the penalty objective of `penalty_objective_function.m`, `csminwel.tolerance.f`
  1e-7, `gradient_epsilon` 1e-6, initial inverse Hessian 1e-4 I);
- 5 (Dynare's default): `newrat` (src/optimizers.jl), Dynare's iterates;
- 7: `fminsearch` (MATLAB's Nelder-Mead; Dynare's options MaxIter 6000, TolFun 1e-8, TolX 1e-6);
- 8: Dynare's `simplex_optimization_routine`;
- 9: CMA-ES (random: `rng`).
Other values (MATLAB-toolbox optimisers 1, 3, 12, 13; Dynare's 2, 6, 10, 101, 102) raise an `EstimationError`.
`maxiter`, `tol` and `epsilon` override the optimiser's `MaxIter`, `TolFun` and gradient step.
"""
function posterior_mode(est::Estimation; x0=est.x0, compute_hessian=nothing, mode_compute=est.options["mode_compute"],
                        rng=Random.default_rng(), maxiter=nothing, tol=nothing, epsilon=nothing, gradient_method=2)
    x = Float64.(x0)
    ok_ = _dsge_likelihood(est, x)
    ok_[2] == 0 || throw(EstimationError("the log posterior is not finite at the initial values: $(est.last_info)"))
    compute_hessian = compute_hessian === nothing ? est.options["cova_compute"] != 0 : compute_hessian
    opt = _parse_optim_opt(est.options["optim_opt"])
    allowed = get(_OPTIM_KEYS, mode_compute, nothing)
    allowed === nothing && throw(EstimationError("mode_compute=$mode_compute is not implemented (0, 4 csminwel, 5 newrat, 7 fminsearch, 8 simplex, 9 cmaes are)"))
    for k in keys(opt)
        k in allowed || throw(EstimationError("optim option $k is not supported with mode_compute=$mode_compute (supported: $(join(allowed, ", ")))"))
    end
    nfev = 0; iters = 0; msg = "mode_compute=0: no optimisation"; success = true
    if mode_compute == 4
        fcn = th -> (r = _dsge_likelihood(est, th); (r[1], r[2], r[3]))
        fh, xh, _, _, iters, nfev, retcode, msg = csminwel(fcn, x; crit=Float64(something(tol, get(opt, "TolFun", 1e-7))),
            nit=Int(something(maxiter, get(opt, "MaxIter", 1000))), method=Int(get(opt, "NumgradAlgorithm", gradient_method)),
            epsilon=Float64(something(epsilon, get(opt, "NumgradEpsilon", 1e-6))), rng=rng)
        xh === nothing && throw(EstimationError("csminwel: bad initial parameter"))
        x = xh
        success = retcode in (0, 1, 3, 6, 7)
    elseif mode_compute == 5
        Int(get(opt, "Hessian", 1)) == 1 || throw(EstimationError("newrat: only optim 'Hessian',1 (Dynare's default) is implemented"))
        # dsge_likelihood's fourth output: [-log prior; minus the log-likelihood contribution of each period]
        fcn = th -> (r = _dsge_likelihood(est, th; with_lik=true); (r[1], r[2], r[3], r[2] == 0 ? vcat(-r[5], r[6]) : Float64[]))
        xh, _, _, _, _, iters, nfev, exitflag, msg = newrat(fcn, x, est.lb, est.ub; ftol=Float64(something(tol, get(opt, "TolFun", 1e-5))),
            nit=Int(something(maxiter, get(opt, "MaxIter", 1000))), htol=Float64(get(opt, "TolGstep", NaN)),
            htol_rel=Float64(get(opt, "TolGstepRel", NaN)), robust=get(opt, "robust", 0) in (1, 1.0, true, "true"),
            gradient_epsilon=Float64(something(epsilon, 1e-6)))
        exitflag == -2 && throw(EstimationError("newrat: bad initial parameter"))
        x = xh
        success = exitflag in (1, -1)
    elseif mode_compute == 7
        fcn = th -> _dsge_likelihood(est, th)[1]
        xh, _, exitflag, iters, nfev = fminsearch(fcn, x; maxiter=Int(something(maxiter, get(opt, "MaxIter", 6000))),
            maxfunevals=Int(get(opt, "MaxFunEvals", 1_000_000)), tolx=Float64(get(opt, "TolX", 1e-6)), tolf=Float64(something(tol, get(opt, "TolFun", 1e-8))))
        x = xh
        success = exitflag == 1
        msg = success ? "fminsearch: converged" : "fminsearch: maximum number of iterations or function evaluations reached"
    elseif mode_compute == 8
        fcn = th -> (r = _dsge_likelihood(est, th); (r[1], r[2] == 0))
        mf = get(opt, "MaxFunEvals", nothing)
        xh, _, exitflag, iters, nfev, msg = simplex_optimization(fcn, x; maxiter=Int(something(maxiter, get(opt, "MaxIter", 10000))),
            tolx=Float64(get(opt, "TolX", 1e-4)), tolf=Float64(something(tol, get(opt, "TolFun", 1e-4))),
            maxfcall=mf === nothing ? nothing : Int(mf), maxfcallfactor=Float64(get(opt, "MaxFunEvalFactor", 500)),
            delta_factor=Float64(get(opt, "InitialSimplexSize", 0.05)))
        x = xh
        success = exitflag == 1
    elseif mode_compute == 9
        fcn = th -> _dsge_likelihood(est, th)[1]
        xh, _, iters, nfev, msg, success = cmaes(fcn, x, est.lb, est.ub; rng=rng, maxiter=maxiter === nothing ? get(opt, "MaxIter", nothing) : maxiter,
            tolfun=Float64(something(tol, get(opt, "TolFun", 1e-7))), tolx=Float64(get(opt, "TolX", 1e-7)),
            maxfunevals=get(opt, "MaxFunEvals", nothing))
        x = xh
    end
    out = ModeResult(copy(x), est.names, log_posterior(est, x), nothing, nothing, NaN, success, msg, nfev, iters)
    mode_compute != 0 && compute_hessian && add_hessian!(est, out)
    out
end

"Hessian, its inverse and the Laplace approximation at `mode.x`."
function add_hessian!(est::Estimation, mode::ModeResult)
    hh = hessian(est, mode.x)
    mode.hessian = hh
    if all(isfinite, hh) && all(>(0), diag(hh))
        mode.inv_hessian = inverse_hessian(hh)
        is_bayesian(est.prior) && (mode.laplace = laplace(est, mode.x, mode.inv_hessian))
    else
        mode.inv_hessian = nothing
    end
    mode
end

# ------------------------------------------------------------------ Metropolis-Hastings
"""
    metropolis_hastings(log_post, x0, logpost0, chol_upper, lb, ub, innovations, uniforms) -> (draws, logpost, accepted)

One random-walk Metropolis-Hastings chain driven by given random numbers (`posterior_sampler_iteration.m`): the proposal is
`last + innovations[i,:]' * chol_upper` (the upper Cholesky factor of the scaled proposal covariance); it is evaluated only if it lies strictly
inside the bounds and accepted if its log posterior is finite and `log(u) < log_post(proposal) - log_post(last)`. A failure
of `log_post` rejects the proposal, as Dynare's `try ... catch`. `uniforms` may be a function `() -> u` (drawn only when
needed, as Dynare calls `rand` only for finite proposals) or a vector.
"""
function metropolis_hastings(log_post, x0, logpost0, chol_upper, lb, ub, innovations::AbstractMatrix, uniforms)
    n_draws, n = size(innovations)
    draws = zeros(n_draws, n)
    logpo = zeros(n_draws)
    acc = falses(n_draws)
    last = Float64.(x0)
    last_lp = Float64(logpost0)
    for i in 1:n_draws
        par = last + vec(transpose(innovations[i, :]) * chol_upper)
        lp = -Inf
        if all(par .> lb) && all(par .< ub)
            lp = try
                Float64(log_post(par))
            catch
                -Inf
            end
        end
        if lp > -Inf
            u = uniforms isa AbstractVector ? uniforms[i] : uniforms()
            if log(u) < lp - last_lp
                last, last_lp = par, lp
                acc[i] = true
            end
        end
        draws[i, :] = last
        logpo[i] = last_lp
    end
    draws, logpo, acc
end

"Scale of the jumping distribution for every parameter (`bayestopt_.jscale`): the individual `jscale` of the block, else `mh_jscale` (default 2.38/sqrt(n))."
function jscale(est::Estimation, mh_jscale=nothing)
    base = mh_jscale !== nothing ? mh_jscale : est.options["mh_jscale"]
    base = base === nothing ? 2.38 / sqrt(est.n) : Float64(base)
    js = copy(est.jscale_individual)
    js[isnan.(js)] .= base
    js, base
end

"""
    sample_posterior(est; replic, nblocks, mode, cov, jscale, seed, x0, drop, init_scale_factor, innovations, uniforms, rng) -> MCMCResult

Random-walk Metropolis-Hastings (Dynare's default posterior sampler). The proposal is `last + randn(1,n) * chol(V) * diag(jscale)`
with `V` the inverse Hessian at the mode (`cov`, or `mode.inv_hessian`). The chains live in the support of the priors
(`prior_trunc` applies; the bounds of `estimated_params` restrict the mode search only: `set_mcmc_prior_bounds.m`). With several
chains each starts from `mode + randn(1,n) * chol(V) * (mh_init_scale_factor * mh_jscale)` inside the bounds with a finite
posterior (`posterior_sampler_initialization.m`, with the 10 percent reduction of the factor after 100 failed trials); a
single chain starts at the mode. `x0` (chains x parameters) overrides the starting points. `innovations` (chains, replic,
n) and `uniforms` (chains, replic) replace the random numbers (tests, replication). `seed` seeds one Xoshiro stream per chain.
The draws cannot reproduce Dynare's run draw for draw: Octave's and MATLAB's generators are not available.
"""
function sample_posterior(est::Estimation; replic=nothing, nblocks=nothing, mode=nothing, cov=nothing, jscale=nothing,
                          seed=0, x0=nothing, drop=nothing, init_scale_factor=nothing, innovations=nothing, uniforms=nothing)
    o = est.options
    is_bayesian(est.prior) || throw(EstimationError("the sampler needs priors"))
    replic = Int(replic === nothing ? o["mh_replic"] : replic)
    nblocks = Int(nblocks === nothing ? o["mh_nblck"] : nblocks)
    drop = drop === nothing ? o["mh_drop"] : Float64(drop)
    xmode = mode isa ModeResult ? mode.x : mode === nothing ? est.x0 : Float64.(mode)
    if cov === nothing
        (mode isa ModeResult && mode.inv_hessian !== nothing) || throw(EstimationError("pass cov=... or a ModeResult with an inverse Hessian"))
        cov = mode.inv_hessian
    end
    js, base = JDyn.jscale(est, jscale)
    d = Matrix(cholesky(Symmetric(Matrix{Float64}(cov))).U)       # upper factor, d' d = cov
    chol_prop = d * Diagonal(js)
    lb, ub = prior_bounds(est.prior, o["prior_trunc"])
    (any(xmode .< lb) || any(xmode .> ub)) && throw(EstimationError("the mode is outside the prior bounds"))
    log_post(x) = log_posterior(est, x; bounds=(lb, ub))
    rngs = [Random.Xoshiro(seed * 7919 + j) for j in 1:nblocks]
    scale0 = (init_scale_factor === nothing ? o["mh_init_scale_factor"] : init_scale_factor) * base
    starts = zeros(nblocks, est.n)
    lp0 = zeros(nblocks)
    for j in 1:nblocks
        if x0 !== nothing
            starts[j, :] = reshape(Float64.(x0), nblocks, est.n)[j, :]
            lp0[j] = log_post(starts[j, :])
            isfinite(lp0[j]) || throw(EstimationError("chain $j: the log posterior is not finite at the starting point"))
        elseif nblocks == 1
            starts[j, :] = xmode
            lp0[j] = log_post(xmode)
        else
            factor = scale0
            valid = false
            init_iter, trial = 0, 1
            while !valid && trial <= 10
                cand = xmode + vec(randn(rngs[j], 1, est.n) * (d * factor))
                if all(cand .>= lb) && all(cand .<= ub)
                    lp = log_post(cand)
                    if isfinite(lp)
                        starts[j, :] = cand; lp0[j] = lp; valid = true
                    end
                end
                init_iter += 1
                if !valid && init_iter > 100
                    factor *= 0.9               # Dynare (nointeractive): reduce the scale factor by 10 percent
                    trial += 1
                end
            end
            valid || throw(EstimationError("chain $j: no valid starting point found"))
        end
    end
    draws = zeros(nblocks, replic, est.n)
    logpo = zeros(nblocks, replic)
    acc = falses(nblocks, replic)
    for j in 1:nblocks
        eps = innovations === nothing ? randn(rngs[j], replic, est.n) : Float64.(innovations[j, :, :])
        u = uniforms === nothing ? (() -> rand(rngs[j])) : Float64.(uniforms[j, :])
        dr, lp, ac = metropolis_hastings(log_post, starts[j, :], lp0[j], chol_prop, lb, ub, eps, u)
        draws[j, :, :] = dr; logpo[j, :] = lp; acc[j, :] = ac
    end
    MCMCResult(draws, logpo, vec(sum(acc; dims=2)) ./ replic, est.names, starts, drop, acc)
end

"""
    marginal_density_mhm(mc; tolerance=0.01) -> Float64

Geweke's modified harmonic mean estimator of the log marginal data density, `marginal_density.m`: the kept draws' mean
and covariance, the posterior kernel at the highest draw, nine coverage probabilities, a widening of the weighting
distribution while the nine estimates disagree.
"""
function marginal_density_mhm(mc::MCMCResult; tolerance=0.01)
    nblck, replic, npar = size(mc.draws)
    first = floor(Int, mc.drop * replic)
    x = reshape(permutedims(mc.draws[:, first+1:end, :], (2, 1, 3)), :, npar)
    lp = vec(permutedims(mc.log_posterior[:, first+1:end]))
    N = size(x, 1)
    lpost_mode = maximum(mc.log_posterior)
    MU = vec(sum(x; dims=1)) ./ N
    xc = x .- MU'
    SIGMA = (xc' * xc) / N
    invS = inv(SIGMA)
    logdet = try
        2 * sum(log, diag(cholesky(Symmetric(SIGMA)).U))
    catch
        return NaN
    end
    dev0 = vec(sum((xc * invS) .* xc; dims=2))
    increase = 1.0
    marginal = zeros(9)
    while true
        for (k, p) in enumerate(0.1:0.1:0.9)
            critval = 2 * SpecialFunctions.gamma_inc_inv(npar / 2, p, 1 - p)
            dev = dev0 ./ increase
            sel = dev .<= critval
            lft = -log(p) .- (npar * log(2pi) .+ (npar * log(increase) + logdet) .+ dev[sel]) ./ 2
            terms = lft .- lp[sel] .+ lpost_mode
            tmp = sum(exp.(terms))
            norm_proba = 0.0
            if isinf(tmp)
                norm_proba = maximum(terms)
                tmp = sum(exp.(terms .- norm_proba))
            end
            marginal[k] = lpost_mode - log(tmp / N) - norm_proba
        end
        if abs((marginal[9] - marginal[1]) / marginal[9]) > tolerance || any(isinf, marginal)
            increase *= 1.2
            increase > 20 && break
        else
            break
        end
    end
    sum(marginal) / 9
end

# ====================================================================== what the .mod file asks for
"Result of `estimate`."
struct EstimationResult
    est::Estimation
    x0::Vector{Float64}
    mode::Union{Nothing,ModeResult}
    mcmc::Union{Nothing,MCMCResult}
    log_posterior::Float64             # at the mode (or at the initial values when no mode search was requested)
    marginal_density_laplace::Float64
    marginal_density_mhm::Float64
    posterior::Union{Nothing,Dict{String,Any}}   # posterior statistics from the draws
    smoother::Any                      # SmootherResult at the mode (smoother, filtered_vars, filter_step_ahead options), else nothing
    forecast::Any                      # ForecastResult of the `forecast` option (oo_.forecast; src/forecast.jl), else nothing
end
EstimationResult(est, x0, mode, mcmc, lp, lap, mhm, post, sm) = EstimationResult(est, x0, mode, mcmc, lp, lap, mhm, post, sm, nothing)

"""
    posterior_statistics(est, mc; mh_conf_sig=est.options["mh_conf_sig"]) -> Dict

Mean, median, variance, std, HPD interval and deciles of every parameter from the kept draws (`posterior_moments.m`).
"""
function posterior_statistics(est::Estimation, mc::MCMCResult; mh_conf_sig=est.options["mh_conf_sig"])
    X = kept_draws(mc)
    out = Dict{String,Any}()
    for (j, nm) in enumerate(est.names)
        mu, med, v, hpd, dec = posterior_moments(X[:, j], mh_conf_sig)
        out[nm] = (mean=mu, median=med, var=v, std=sqrt(v), hpd=hpd, deciles=dec)
    end
    out
end

"""
    estimate(m::Model; statement=nothing, allow_unsupported=false, seed=<model seed>, rng=<model stream>, data=nothing, options...) -> EstimationResult

Run what the .mod file's `estimation` statement asks for within the implemented scope (first order): the posterior mode with
csminwel unless `mode_compute=0`, the Hessian, the Laplace approximation, and a random-walk Metropolis-Hastings run when
`mh_replic > 0` (with posterior statistics and the modified harmonic mean). Keyword options override the statement's
options; the model is not modified. Options for outputs not implemented (forecasts, posterior IRFs, ...) raise an
`EstimationError` unless `allow_unsupported=true`. With no `estimation` statement in the file, pass options as keywords.
The `smoother` (and `filtered_vars`, `filter_step_ahead`) options run `smooth` at the mode (at the initial values with
`mode_compute=0`) and return it in the `smoother` field; with `mh_replic > 0` they are refused.
"""
function estimate(m::Model; seed=nothing, rng=nothing, kw...)
    seed === nothing && (seed = _streams(m).seed)          # the model's seed (set_dynare_seed; default 0) seeds the chains
    rng === nothing && (rng = dynare_rng(m))
    est = Estimation(m; kw...)
    o = est.options
    initial_estimation_checks(est)
    mode = nothing
    lp = log_posterior(est, est.x0)
    if o["mode_compute"] != 0
        mode = posterior_mode(est; rng=rng)
        lp = mode.log_posterior
    end
    mc = nothing
    post = nothing
    mhm = NaN
    if o["mh_replic"] > 0 && is_bayesian(est.prior)
        (mode !== nothing && mode.inv_hessian !== nothing) ||
            throw(EstimationError("mh_replic > 0 needs a posterior mode and its Hessian (mode_compute != 0 and cova_compute != 0)"))
        mc = sample_posterior(est; mode=mode, seed=seed)
        post = posterior_statistics(est, mc)
        mhm = marginal_density_mhm(mc)
    end
    lap = mode === nothing ? NaN : mode.laplace
    sm = nothing
    if _est_flag(get(o, "smoother", false)) || _est_flag(get(o, "filtered_vars", false)) || !isempty(something(get(o, "filter_step_ahead", nothing), []))
        # Dynare's frequentist smoother at the mode (xparam1); the draws' smoothed variables are not implemented
        (mc !== nothing) && throw(EstimationError("smoother with mh_replic > 0 (smoothed variables at the posterior draws) is not implemented; run smooth(m; estimation=est, parameter_set=\"posterior_mean\", mcmc=...) for one parameter vector"))
        theta = mode === nothing ? est.x0 : mode.x
        sm = smooth(m; estimation=est, theta=theta)
    end
    fc = nothing
    nf = Int(something(get(o, "forecast", 0), 0))
    if nf > 0
        # dynare_estimation_1.m: the classical forecast from the smoother at the mode; with mode_compute=0 only when the
        # smoother option is set; over the posterior draws (mh_replic > 0) not implemented
        mc !== nothing && throw(EstimationError("forecast with mh_replic > 0 (forecasts over the posterior draws) is not implemented"))
        if mode !== nothing || sm !== nothing
            sm === nothing && (sm = smooth(m; estimation=est, theta=mode.x))
            fc = forecast(sm, m; periods=nf, conf_sig=Float64(get(o, "forecasts.conf_sig", 0.9)), var_list=_estimation_var_list(m, kw))
        end
    end
    EstimationResult(est, est.x0, mode, mc, lp, lap, mhm, post, sm, fc)
end

"Variables listed after the estimation statement (Dynare's var_list_)."
function _estimation_var_list(m::Model, kw)
    ests = [st for st in m.statements if get(st, "statementName", "") == "estimation"]
    isempty(ests) && return String[]
    k = get(kw, :statement, nothing)
    String.(get_or(ests[k === nothing ? end : k], "symbol_list", String[]))
end

"""
    initial_estimation_checks(est)

The stochastic-singularity check of Dynare's `initial_estimation_checks.m`: evaluate the likelihood at the initial values
with the automatic switch to the univariate filter turned off; a singular forecast-error variance is an error.
"""
function initial_estimation_checks(est::Estimation)
    o = est.options
    old = o["use_univariate_filters_if_singularity_is_detected"]
    o["use_univariate_filters_if_singularity_is_detected"] = false
    info = try
        _dsge_likelihood(est, est.x0; bounds=(fill(-Inf, est.n), fill(Inf, est.n)))[2]
    finally
        o["use_univariate_filters_if_singularity_is_detected"] = old
    end
    info == 50 && throw(EstimationError("initial_estimation_checks: The forecast error variance in the multivariate Kalman filter became singular. " *
        "This is often a sign of stochastic singularity, but can also sometimes happen by chance for a particular combination of parameters and data realizations. " *
        "If you think the latter is the case, you should try with different initial values for the estimated parameters."))
    nothing
end

"Set the estimated parameters and standard deviations of `m` to `theta` (Dynare leaves the model at the estimates)."
function apply_estimates!(m::Model, est::Estimation, theta::AbstractVector)
    p = model_params(est, theta)
    Q, H = covariances(est, theta)
    m.params .= p
    m.Sigma_e .= Q
    for (nm, _) in collect(m.measurement_errors)
        k = findfirst(==(nm), est.varobs)
        m.measurement_errors[nm] = H[k, k]
    end
    for (_, k, _) in est.groups.vn
        m.measurement_errors[est.varobs[k]] = H[k, k]
    end
    empty!(m.cache)
    m
end
