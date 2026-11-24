# DSGE-VAR (Del Negro and Schorfheide 2004: Dynare's `dsge_var` and `dsge_varlag`, matlab/estimation/dsge_var_likelihood.m,
# var_sample_moments.m, check_dsge_var_model.m) and the endogenous prior of Christiano, Trabandt and Walentin (2011)
# (matlab/estimation/endogenous_prior.m, used by dsge_likelihood.m). Hooks in `_dsge_likelihood` (src/estimation.jl) call
# `_dvar_precheck`, `_dvar_likelihood` and `endogenous_log_prior`.
#
# A model is a DSGE-VAR when the preprocessor added the parameter `dsge_prior_weight` (dsge_var=VALUE: calibrated, `dsge_var`
# alone: to be estimated with an `estimated_params` line, `dsge_var=Inf`: the DSGE model alone). The VAR has `dsge_varlag` lags
# and a constant. As in Dynare the sample size in the formulas is `nobs + dsge_varlag` (`dataset_.nobs` after makedataset added
# the initial conditions), although the VAR is fitted on `nobs` observations.



"True if the model is estimated as a DSGE-VAR (Dynare: the parameter `dsge_prior_weight` exists)."
dsge_var_active(est::Estimation) = haskey(est.model.ip, "dsge_prior_weight")

"""
    dsge_var_check(est)

`check_dsge_var_model.m` and the refusals of `dynare_estimation_init.m` / `dynare_estimation_1.m`.
"""
function dsge_var_check(est::Estimation)
    o = est.options
    (!isempty(est.groups.vn) || !isempty(est.groups.cn) || any(!iszero, est.H0)) &&
        throw(EstimationError("Estimation::DsgeVarLikelihood: Measurement errors are not allowed!"))
    S = est.Sigma_e0
    (!isempty(est.groups.cx) || (size(S, 1) > 1 && any(!iszero, tril(S, -1)))) &&
        throw(EstimationError("Estimation::DsgeVarLikelihood: Structural innovations cannot be correlated using Dynare's interface! Introduce the correlations in the model block instead."))
    est.trend_expr === nothing || throw(EstimationError("Estimation::DsgeVarLikelihood: Linear trend is not yet implemented!"))
    o["presample"] != 0 && throw(EstimationError("DSGE-VAR does not support the presample option."))
    est.data_index === nothing || throw(EstimationError("Estimation::DsgeVarLikelihood: I cannot estimate a DSGE-VAR model with missing observations!"))
    o["prefilter"] != 0 && throw(EstimationError("DSGE-VAR with prefilter is not supported"))
    o["lik_init"] == 1 || throw(EstimationError("DSGE-VAR needs a stationary model (lik_init=1)"))
    haskey(o, "nl") && throw(EstimationError("DSGE-VAR with order=2 is not supported"))
    nothing
end

"`var_sample_moments.m` with a constant: `(mYY, mYX, mXX)` from `data` (T_all x n) whose first `nlag` rows are initial conditions."
function dvar_sample_moments(data::AbstractMatrix, nlag::Int)
    Y = data[nlag+1:end, :]
    N, n = size(Y)
    X = ones(N, n * nlag + 1)
    for lag in 1:nlag
        X[:, (lag-1)*n+1:lag*n] = data[nlag-lag+1:nlag-lag+N, :]
    end
    Y' * Y, Y' * X, X' * X
end

"The moments of the sample (with `dsge_varlag` initial conditions before `first_obs`), computed once."
function _dvar_moments(est::Estimation)
    o = est.options
    haskey(o, "_dvar_moments") && return o["_dvar_moments"]
    dsge_var_check(est)
    nlag = Int(o["dsge_varlag"])
    first = Int(o["first_obs"])
    first - nlag < 1 && throw(EstimationError("DSGE-VAR: first_obs ($first) - $nlag cannot be less than the first observation in the dataset; increase first_obs"))
    all_ = o["_alldata"]::Matrix{Float64}
    o["_dvar_moments"] = dvar_sample_moments(all_[first-nlag:first-1+est.nobs, :], nlag)
end

_ispd_dv(A) = (all(isfinite, A) && isposdef(Hermitian((A + A') / 2)))

"`ispd.m`: (positive definite, penalty = sum of the absolute negative eigenvalues)."
function _ispd_pen(A)
    ok = try
        cholesky(Hermitian(A, :U)); true
    catch
        false
    end
    ok && return true, 0.0
    all(isfinite, A) || return false, 1.0
    ev = eigvals(Symmetric((A + A') / 2))
    false, sum(abs, ev[ev .< 0])
end

"The weight of the DSGE prior at `theta`."
dvar_weight(est::Estimation, theta) = model_params(est, theta)[est.model.ip["dsge_prior_weight"]]

"The check of `dsge_var_likelihood.m` that precedes the solution of the model: a proper DSGE prior (info 51)."
function _dvar_precheck(est::Estimation, theta)
    _dvar_moments(est)
    o = est.options
    n = length(est.varobs); k = n * Int(o["dsge_varlag"]) + 1
    TT = est.nobs + Int(o["dsge_varlag"])
    lam = dvar_weight(est, theta)
    if lam < (k + n) / TT
        est.last_info = "dsge_prior_weight is below (number of VAR parameters + observables) / T (info 51)"
        est.last_info_code = 51
        return (Inf, 51, abs(TT * lam - (k + n)), NaN, NaN, Float64[])
    end
    nothing
end

"""
    dsge_var_terms(est, theta, ys, T, R, Q, mf) -> (; lik, info, penalty, PHI_tilde, SIGMA_u_tilde, iXX, prior)

Minus the DSGE-VAR log likelihood (no prior) from the solution of the model, and the objects `dsge_var_likelihood.m` returns at
the posterior mode (`PHI_tilde`, `SIGMA_u_tilde`, `iXX`, `prior`). `info` is Dynare's code when the likelihood is undefined
(52, 53 a covariance matrix that is not positive definite, 50, 45, 46).
"""
function dsge_var_terms(est::Estimation, theta, ys, T, R, Q, mf)
    o = est.options
    mYY, mYX, mXX = _dvar_moments(est)
    nlag = Int(o["dsge_varlag"])
    n = length(mf); k = n * nlag + 1
    TT = est.nobs + nlag
    lam = dvar_weight(est, theta)
    c = ys[est.obs_decl]
    o["loglinear"] && (c = log.(c))
    cc = c * c'
    tmp0 = lyapunov_symm(T, R * Q * R'; qz_criterium=o["qz_criterium"], complex_threshold=o["lyapunov_complex_threshold"])[1]
    acov = [tmp0[mf, mf] + cc]
    for _ in 1:nlag
        tmp0 = T * tmp0
        push!(acov, tmp0[mf, mf] + cc)
    end
    GYX = zeros(n, k)
    for i in 1:nlag
        GYX[:, (i-1)*n+1:i*n] = acov[i+1]
    end
    GYX[:, end] = c
    GXX = kron(Matrix{Float64}(I, nlag, nlag), acov[1])
    for i in 1:nlag-1
        GXX += kron(diagm(i => ones(nlag - i)), acov[i+1]) + kron(diagm(-i => ones(nlag - i)), acov[i+1]')
    end
    GXX = [GXX kron(ones(nlag, 1), reshape(c, :, 1)); kron(ones(1, nlag), reshape(c, 1, :)) 1.0]
    GYY = acov[1]
    iGXX = inv(GXX)
    PHI_star = iGXX * GYX'
    SIGMA_u_star = GYY - GYX * PHI_star
    ok, pen = _ispd_pen(SIGMA_u_star)
    fail(code, p) = (; lik=Inf, info=code, penalty=p, PHI_tilde=nothing, SIGMA_u_tilde=nothing, iXX=nothing, prior=nothing)
    ok || return fail(53, pen)
    local PHI_tilde, SIGMA_u_tilde, iXX, lik
    if isfinite(lam)
        t0 = lam * TT * acov[1] + mYY
        t1 = lam * TT * GYX + mYX
        t2 = inv(lam * TT * GXX + mXX)
        S = t0 - t1 * t2 * t1'
        ok, pen = _ispd_pen(S)
        ok || return fail(52, pen)
        SIGMA_u_tilde = S / (TT * (1 + lam))
        PHI_tilde = t2 * t1'
        iXX = t2
        j = 1:n
        prodlng1 = sum(SpecialFunctions.loggamma.(0.5 .* ((1 + lam) * TT - k + 1 .- j)))
        prodlng2 = sum(SpecialFunctions.loggamma.(0.5 .* (lam * TT - k + 1 .- j)))
        ld(A) = log(det(A))
        lik = 0.5 * n * ld(lam * TT * GXX + mXX) +
              0.5 * ((lam + 1) * TT - k) * ld((lam + 1) * TT * SIGMA_u_tilde) -
              0.5 * n * ld(lam * TT * GXX) -
              0.5 * (lam * TT - k) * ld(lam * TT * SIGMA_u_star) +
              0.5 * n * TT * log(2 * pi) -
              0.5 * log(2) * n * ((lam + 1) * TT - k) +
              0.5 * log(2) * n * (lam * TT - k) - prodlng1 + prodlng2
    else
        lik = TT * (log(det(SIGMA_u_star)) + n * log(2 * pi) +
                    tr(inv(SIGMA_u_star) * (mYY - (mYX * PHI_star)' - mYX * PHI_star + PHI_star' * mXX * PHI_star) / TT))
        lik = 0.5 * lik
        SIGMA_u_tilde = SIGMA_u_star; PHI_tilde = PHI_star; iXX = iGXX
    end
    isinf(lik) && return fail(50, 0.1)
    isnan(lik) && return fail(45, 0.1)
    ass = isfinite(lam) ? Float64(trunc(lam * TT)) : Inf                  # fix(dsge_prior_weight * nobs)
    prior = (SIGMA_u_star=SIGMA_u_star, PHI_star=PHI_star, ArtificialSampleSize=ass, DF=ass - k - n, iGXX_star=iGXX)
    (; lik=Float64(lik), info=0, penalty=0.0, PHI_tilde=PHI_tilde, SIGMA_u_tilde=SIGMA_u_tilde, iXX=iXX, prior=prior)
end

"The DSGE-VAR likelihood inside `_dsge_likelihood` (after the model is solved): the same return tuple."
function _dvar_likelihood(est::Estimation, theta, ys, T, R, Q, z)
    t = dsge_var_terms(est, theta, ys, T, R, Q, z)
    if t.info != 0
        est.last_info = "DSGE-VAR likelihood not defined (info $(t.info))"; est.last_info_code = t.info
        return (Inf, t.info, t.penalty, NaN, NaN, Float64[])
    end
    est.last_filter = "DSGE-VAR"
    lnprior = logpdf(est.prior, theta)
    if isinf(lnprior) || isnan(lnprior)
        info = isnan(lnprior) ? 47 : 40
        est.last_info = "the prior density is zero, infinite or NaN (info $info)"; est.last_info_code = info
        return (Inf, info, 0.1, -t.lik, lnprior, Float64[])
    end
    (t.lik - lnprior, 0, 0.0, -t.lik, lnprior, Float64[])
end

"""
    dsge_var_posterior(est, theta) -> (; PHI_tilde, SIGMA_u_tilde, iXX, prior)

The DSGE-VAR objects of `dsge_var_likelihood.m` at `theta` (Dynare's `oo_.dsge_var.posterior_mode`).
"""
function dsge_var_posterior(est::Estimation, theta::AbstractVector)
    dsge_var_active(est) || throw(EstimationError("the model is not a DSGE-VAR (no dsge_prior_weight parameter)"))
    ys, G, Hs, _ = solve(est, theta)
    Q, _ = covariances(est, theta)
    r = est.restrict
    t = dsge_var_terms(est, theta, ys, G[r, r], Hs[r, :], Q, est.mf)
    t.info == 0 || throw(EstimationError("the DSGE-VAR likelihood is not defined at theta (info $(t.info))"))
    (PHI_tilde=t.PHI_tilde, SIGMA_u_tilde=t.SIGMA_u_tilde, iXX=t.iXX, prior=t.prior)
end

# ---------------------------------------------------------------------------------------------------- endogenous prior
"""
    endogenous_log_prior(Y, Pstar, mf, H) -> Float64

`endogenous_prior.m`: log density of the sample variances of the observed variables (`Y`, n x T, after the last missing
observation) given the variances the model implies (`Pstar`, the state covariance; `mf` the observed positions; `H` the
measurement-error covariance), with a Newey-West (two lags, Bartlett weights) estimate of the covariance of the estimates.
"""
function endogenous_log_prior(Y::AbstractMatrix, Pstar::AbstractMatrix, mf, H::AbstractMatrix)
    Yt = permutedims(Y)
    Tsamp, n = size(Yt)
    Yd = Yt .- sum(Yt; dims=1) ./ Tsamp
    Fhat = vec(sum(Yd .* Yd; dims=1)) ./ Tsamp
    h = permutedims(Yd .* Yd .- Fhat')
    C0 = zeros(n, n); C1 = zeros(n, n); C2 = zeros(n, n)
    for t in 1:Tsamp
        C0 += 1 / Tsamp * h[:, t] * h[:, t]'
    end
    for t in 2:Tsamp
        C1 += 1 / (Tsamp - 1) * h[:, t] * h[:, t-1]'
    end
    for t in 3:Tsamp
        C2 += 1 / (Tsamp - 2) * h[:, t] * h[:, t-2]'
    end
    Shat = C0 + (1 - 1 / 3) * (C1 + C1') + (1 - 2 / 3) * (C2 + C2')
    Ftheta = diag(Pstar[mf, mf] + H)
    d = Fhat - Ftheta
    0.5 * n * log(Tsamp / (2 * pi)) - 0.5 * log(det(Shat)) - 0.5 * Tsamp * (d' * (Shat \ d))
end
