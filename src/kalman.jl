# Kalman filters for the likelihood of a linear Gaussian state space (port of pydyn/kalman.py, which follows
# Dynare 7.2's matlab/kalman/likelihood/{kalman_filter, missing_observations_kalman_filter, kalman_filter_d,
# missing_observations_kalman_filter_d, univariate_kalman_filter, univariate_kalman_filter_d}.m,
# matlab/matrix_solver/lyapunov_symm.m and matlab/estimation/compute_Pinf_Pstar.m).
#
# State space (deviations from the steady state):
#     a_{t+1} = T a_t + R eps_{t+1},   eps ~ N(0, Q)
#     y_t     = a_t[z] + eta_t,        eta ~ N(0, H)
# `z` is an index vector (the observed variables are elements of the state vector; 1-based). All filters return
# MINUS the log likelihood, like Dynare, and Inf when the forecast-error covariance is singular.
# `data_index[t]` lists the observed series of period t (missing-observation filters); `no_more_missing` is the
# first period (1-based) after the last one with a missing observation.

const LOG2PI = log(2 * pi)

_norm1(A) = isempty(A) ? 0.0 : maximum(sum(abs, A; dims=1))

"Inverse and reciprocal 1-norm condition number (what MATLAB's `rcond` estimates); `(nothing, 0.0)` if singular or not finite."
function _inv_rcond(F::AbstractMatrix)
    all(isfinite, F) || return nothing, 0.0
    if size(F, 1) == 1
        F[1, 1] == 0 && return nothing, 0.0
        return reshape([1.0 / F[1, 1]], 1, 1), 1.0
    end
    iF = try
        inv(F)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return nothing, 0.0
    end
    all(isfinite, iF) || return nothing, 0.0
    iF, 1.0 / (_norm1(F) * _norm1(iF))
end
_rcond(F) = _inv_rcond(F)[2]

"Real Schur form with the eigenvalues of modulus above `threshold` ordered first: (U, T, k)."
function _schur_unit_first(a::AbstractMatrix, threshold)
    F = schur(Matrix{Float64}(a))
    sel = abs.(F.values) .> threshold
    ordschur!(F, sel)
    F.Z, F.T, count(sel)
end

"""
Solve `X = S X S' + B` on the trailing block (rows and columns `nk+1:end`) of the quasi-triangular `ST` by
backward substitution (lyapunov_symm.m, compute_Pinf_Pstar.m). The result is zero outside the block.
"""
function _lyap_backsub(ST::AbstractMatrix, B::AbstractMatrix, nk::Int; exact_zero=false, complex_threshold=1e-15)
    n = size(ST, 1)
    X = zeros(n, n)
    i = n
    while i >= nk + 2
        sub = ST[i, i-1]
        real_block = exact_zero ? sub == 0 : abs(sub) < complex_threshold
        r = nk+1:i
        m = length(r)
        if real_block
            if i == n
                c = zeros(m)
            else
                c = ST[r, :] * (X[:, i+1:n] * ST[i, i+1:n]) + ST[i, i] * (ST[r, i+1:n] * X[i+1:n, i])
            end
            q = Matrix{Float64}(I, m, m) - ST[r, r] * ST[i, i]
            X[r, i] = q \ (B[r, i] + c)
            X[i, nk+1:i-1] = X[nk+1:i-1, i]
            i -= 1
        else
            if i == n
                c = zeros(m); c1 = zeros(m)
            else
                c = ST[r, :] * (X[:, i+1:n] * ST[i, i+1:n]) + ST[i, i] * (ST[r, i+1:n] * X[i+1:n, i]) +
                    ST[i, i-1] * (ST[r, i+1:n] * X[i+1:n, i-1])
                c1 = ST[r, :] * (X[:, i+1:n] * ST[i-1, i+1:n]) + ST[i-1, i-1] * (ST[r, i+1:n] * X[i+1:n, i-1]) +
                     ST[i-1, i] * (ST[r, i+1:n] * X[i+1:n, i])
            end
            Id = Matrix{Float64}(I, m, m)
            STr = ST[r, r]
            q = [Id-STr*ST[i, i] -STr*ST[i, i-1]; -STr*ST[i-1, i] Id-STr*ST[i-1, i-1]]
            z = q \ vcat(B[r, i] + c, B[r, i-1] + c1)
            X[r, i] = z[1:m]
            X[r, i-1] = z[m+1:end]
            X[i, nk+1:i-1] = X[nk+1:i-1, i]
            X[i-1, nk+1:i-2] = X[nk+1:i-2, i-1]
            i -= 2
        end
    end
    if i == nk + 1
        j = nk + 1
        c = dot(ST[j, :], X[:, j+1:n] * ST[j, j+1:n]) + ST[j, j] * dot(ST[j, j+1:n], X[j+1:n, j])
        X[j, j] = (B[j, j] + c) / (1 - ST[j, j] * ST[j, j])
    end
    X
end

"""
    compute_Pinf_Pstar(mf, T, R, Q, qz_criterium, restrict_columns=nothing) -> (Pstar, Pinf)

Initial covariance matrices of the diffuse filter (`compute_Pinf_Pstar.m`). `mf` indexes the observed variables in
the state vector, `restrict_columns` the states that enter the transition matrix with a lag.
"""
function compute_Pinf_Pstar(mf, T::AbstractMatrix, R::AbstractMatrix, Q::AbstractMatrix, qz_criterium=1 + 1e-6,
                            restrict_columns=nothing)
    T = Matrix{Float64}(T)
    npf = size(T, 1)
    any(!iszero, Q) || return zeros(size(T)), zeros(size(T))
    indx = restrict_columns !== nothing ? collect(Int, restrict_columns) :
           findall(vec(maximum(abs, T; dims=1)) .>= 1e-10)
    indx0 = setdiff(1:npf, indx)
    np0 = length(indx0)
    T0 = T[indx0, indx]
    R0 = R[indx0, :]
    Tr = T[indx, indx]
    Rr = R[indx, :]
    n = size(Tr, 1)
    QT, ST, nk = _schur_unit_first(Tr, 2 - qz_criterium)
    R1 = QT' * Rr
    B = R1 * Q * R1'
    Pstar = _lyap_backsub(ST, B, nk; exact_zero=true)
    order = vcat(indx0, indx)
    obs = [o in mf for o in order]
    if np0 > 0
        Pstar = [zeros(np0, np0) zeros(np0, n); zeros(n, np0) Pstar]
        ST = hcat(zeros(np0 + n, np0), vcat(T0 * QT, ST))
        R1 = vcat(R0, R1)
        STinf = ST[np0+1:np0+nk, np0+1:np0+nk]
        iSTinf = nk > 0 ? inv(STinf) : zeros(0, 0)
        ST0 = copy(ST)
        ST0[:, 1:np0+nk] .= 0
        ST00 = ST[1:np0, np0+1:np0+nk]
        ST0[1:np0, np0+nk+1:end] = ST[1:np0, np0+nk+1:end] - ST00 * (iSTinf * ST[np0+1:np0+nk, np0+nk+1:end])
        R10 = copy(R1)
        R10[1:np0, :] = R1[1:np0, :] - ST00 * (iSTinf * R1[np0+1:np0+nk, :])
        ST0[np0+1:np0+nk, :] .= 0
        R10[np0+1:np0+nk, :] .= 0
        Pstar = ST0 * Pstar * ST0' + R10 * Q * R10'
    end
    Pinf = zeros(n, n)
    for k in 1:nk
        Pinf[k, k] = 1.0
    end
    if np0 > 0
        STtriu = STinf - Matrix{Float64}(I, nk, nk)
        STinf0 = ST00 * (Matrix{Float64}(I, nk, nk) - iSTinf * STtriu)
        Pinf = [zeros(np0, np0) zeros(np0, n); zeros(n, np0) Pinf]
        QT = [Matrix{Float64}(I, np0, np0) zeros(np0, n); zeros(n, np0) QT]
        QTinf = copy(QT)
        QTinf[1:np0, np0+1:np0+nk] = STinf0
        tmp = similar(QTinf)
        tmp[order, :] = QTinf
        QTinf = tmp
        STinf1 = hcat(zeros(np0 + n, np0), vcat(STinf0, Matrix{Float64}(I, nk, nk), zeros(n - nk, nk)), zeros(np0 + n, n - nk))
        for k in 1:nk
            # sic: Dynare indexes the rows with a logical mask over [indx0; indx]
            if norm(QTinf[obs, :] * ST[order, k+np0]) < 1e-8
                Pinf[k+np0, k+np0] = 0
            end
        end
        Pinf = STinf1 * Pinf * STinf1'
        tmp = similar(QT)
        tmp[order, :] = QT
        QT = tmp
    else
        for k in 1:nk
            if norm(QT[collect(Int, mf), :] * ST[:, k]) < 1e-8
                Pinf[k, k] = 0
            end
        end
    end
    QT * Pstar * QT', QT * Pinf * QT'
end

"""
Conditioning tests of kalman_filter.m: `(log det F, inverse of F)`, or `nothing` when F is treated as singular.
`rescale` forces the computations on the correlation matrix.
"""
function _forecast_error_inverse(F::AbstractMatrix, kalman_tol, rescale)
    n = size(F, 1)
    if !rescale
        if n == 1
            f = F[1, 1]
            (f != 0 && isfinite(f)) && return log(f), reshape([1.0 / f], 1, 1)
        elseif n == 2
            a, b, c, d = F[1, 1], F[1, 2], F[2, 1], F[2, 2]
            det2 = a * d - b * c
            if det2 != 0 && isfinite(det2)
                n1 = max(abs(a) + abs(c), abs(b) + abs(d))
                det2 * det2 / (n1 * n1) >= kalman_tol * abs(det2) && return log(det2), [d -b; -c a] ./ det2   # rcond = |det| / ||F||_1^2
            end
        else
            iF, rc = _inv_rcond(F)
            rc >= kalman_tol && return log(det(F)), iF
        end
    end
    d = diag(F)
    (any(<(kalman_tol), d) || !all(isfinite, d)) && return nothing
    sig = sqrt.(d)
    ss = sig * sig'
    iFs, rc = _inv_rcond(F ./ ss)
    rc < kalman_tol && return nothing
    log(det(F ./ ss)) + 2 * sum(log, sig), iFs ./ ss
end

"""
    kalman_filter(Y, a, P, T, QQ, H, z; kalman_tol, riccati_tol, presample, rescale, diffuse_periods, data_index, no_more_missing) -> (LIK, lik)

Standard Kalman filter with a switch to the steady-state gain (`kalman_filter.m`, `missing_observations_kalman_filter.m`).
`Y` is (observed variables, periods) with the constant already removed; `a`, `P` the predicted state and covariance of
the first period; `QQ = R Q R'`; `H` the measurement-error covariance matrix. Returns minus the log likelihood summed
after the presample (Inf if singular) and the per-period values (halved, as Dynare).
"""
function kalman_filter(Y::AbstractMatrix, a::AbstractVector, P::AbstractMatrix, T::AbstractMatrix, QQ::AbstractMatrix,
                       H::AbstractMatrix, z::AbstractVector{<:Integer}; kalman_tol=1e-10, riccati_tol=1e-6, presample=0,
                       rescale=false, diffuse_periods=0, data_index=nothing, no_more_missing=1)
    pp, smpl = size(Y)
    Tt = T'
    lik = zeros(smpl)
    notsteady = true
    oldK = nothing
    t = 1
    missing_obs = data_index !== nothing
    a = Vector{Float64}(a)
    P = Matrix{Float64}(P)
    missing_obs && (P = tril(P) + tril(P, -1)')
    K = zeros(0, 0); iF = zeros(0, 0); log_dF = 0.0
    Hz = H
    while notsteady && t <= smpl
        if missing_obs
            d = data_index[t]
            if isempty(d)
                a = T * a
                P = T * P * Tt + QQ
                t += 1
                continue
            end
            zt = z[d]
            v = Y[d, t] - a[zt]
            F = P[zt, zt] + H[d, d]
        else
            zt = z
            v = Y[:, t] - a[zt]
            F = P[zt, zt] + H
        end
        res = _forecast_error_inverse(F, kalman_tol, rescale)
        res === nothing && return Inf, lik
        log_dF, iF = res
        lik[t] = log_dF + dot(v, iF * v) + length(zt) * LOG2PI
        K = P[:, zt] * iF
        P = T * (P - K * P[zt, :]) * Tt + QQ
        a = T * (a + K * v)
        if !missing_obs || t >= no_more_missing
            Kv = vec(K)
            notsteady = oldK === nothing || length(oldK) != length(Kv) || maximum(abs.(Kv .- oldK)) > riccati_tol
            oldK = Kv
        end
        t += 1
    end
    t1 = t - 1                                          # periods processed with the full recursion
    for s in 1:t1
        lik[s] *= 0.5
    end
    if t1 < smpl                                       # steady-state filter for the remaining periods
        for s in t:smpl
            v = Y[:, s] - a[z]
            lik[s] = dot(v, iF * v)
            a = T * (a + K * v)
        end
        for s in t:smpl
            lik[s] = 0.5 * (lik[s] + log_dF + pp * LOG2PI)
        end
    end
    start = presample > diffuse_periods ? presample - diffuse_periods : 0
    sum(@view lik[start+1:end]), lik
end

"""
    kalman_filter_d(Y, a, Pinf, Pstar, T, QQ, H, z; kalman_tol, diffuse_kalman_tol, presample, data_index) -> (dLIK, dlik, a, Pstar)

Diffuse stage of the filter (`kalman_filter_d.m`, `missing_observations_kalman_filter_d.m`): minus the log likelihood of
the diffuse periods (after the presample), the per-period values, and the predicted state and covariance of the first
non-diffuse period. `dLIK` is Inf if a singularity stops the filter and NaN if the diffuse stage never ends.
"""
function kalman_filter_d(Y::AbstractMatrix, a::AbstractVector, Pinf::AbstractMatrix, Pstar::AbstractMatrix,
                         T::AbstractMatrix, QQ::AbstractMatrix, H::AbstractMatrix, z::AbstractVector{<:Integer};
                         kalman_tol=1e-10, diffuse_kalman_tol=1e-6, presample=0, data_index=nothing)
    pp, smpl = size(Y)
    Tt = T'
    dlik = zeros(smpl)
    missing_obs = data_index !== nothing
    a = Vector{Float64}(a); Pinf = Matrix{Float64}(Pinf); Pstar = Matrix{Float64}(Pstar)
    t = 1
    s = 0
    diffuse_rank(M) = isempty(M) ? 0 : count(>(diffuse_kalman_tol), svdvals(M))
    while t <= smpl && diffuse_rank(missing_obs ? Pinf : Pinf[z, z]) > 0
        s = t
        if missing_obs
            d = data_index[t]
            if isempty(d)
                a = T * a
                Pstar = T * Pstar * Tt + QQ
                Pinf = T * Pinf * Tt
                t += 1
                continue
            end
            zt, Ht = z[d], H[d, d]
            v = Y[d, t] - a[zt]
        else
            zt, Ht = z, H
            v = Y[:, t] - a[zt]
        end
        const_ = length(zt) * LOG2PI
        Finf = Pinf[zt, zt]
        if _rcond(Finf) < diffuse_kalman_tol
            all(x -> abs(x) < diffuse_kalman_tol, Finf) || return Inf, dlik[1:s], a, Pstar
            Fstar = Pstar[zt, zt] + Ht
            _rcond(Fstar) < kalman_tol && return Inf, dlik[1:s], a, Pstar
            iFstar = inv(Fstar)
            Kstar = Pstar[:, zt] * iFstar
            dlik[t] = log(det(Fstar)) + dot(v, iFstar * v) + const_
            Pinf = T * Pinf * Tt
            Pstar = T * (Pstar - Pstar[:, zt] * Kstar') * Tt + QQ
            a = T * (a + Kstar * v)
        else
            dlik[t] = log(det(Finf)) + const_
            iFinf = inv(Finf)
            Kinf = Pinf[:, zt] * iFinf
            Fstar = Pstar[zt, zt] + Ht
            Kstar = (Pstar[:, zt] - Kinf * Fstar) * iFinf
            Pstar = T * (Pstar - Pstar[:, zt] * Kinf' - Pinf[:, zt] * Kstar') * Tt + QQ
            Pinf = T * (Pinf - Pinf[:, zt] * Kinf') * Tt
            a = T * (a + Kinf * v)
        end
        t += 1
    end
    t > smpl && return NaN, dlik, a, Pstar
    dlik = 0.5 .* dlik[1:s]
    sum(@view dlik[presample+1:end]), dlik, a, Pstar
end

"""
    univariate_kalman_filter(Y, a, P, T, QQ, Hdiag, Z; kalman_tol, riccati_tol, presample, diffuse_periods, data_index, no_more_missing) -> (LIK, lik)

Univariate (sequential) Kalman filter (`univariate_kalman_filter.m`, `univariate_kalman_filter_ss.m`). `Z` is the
(observed, states) measurement matrix and `Hdiag` the vector of measurement-error variances (uncorrelated errors; correlated
errors are handled by the caller through an enlarged state vector, as Dynare). Observations whose forecast-error variance is
below `kalman_tol` are skipped. Returns minus the log likelihood (NaN if a variance is negative) and per-period values.
"""
function univariate_kalman_filter(Y::AbstractMatrix, a::AbstractVector, P::AbstractMatrix, T::AbstractMatrix,
                                  QQ::AbstractMatrix, Hdiag::AbstractVector, Z::AbstractMatrix; kalman_tol=1e-10,
                                  riccati_tol=1e-6, presample=0, diffuse_periods=0, data_index=nothing, no_more_missing=1)
    pp, smpl = size(Y)
    mm = size(T, 1)
    Tt = T'
    lik = zeros(smpl, pp)
    K = fill(NaN, mm, pp)
    oldK = nothing
    notsteady = true
    a = Vector{Float64}(a); P = Matrix{Float64}(P)
    t = 1
    while notsteady && t <= smpl
        d = data_index === nothing ? (1:pp) : data_index[t]
        P = (P + P') / 2
        for di in d
            Zi = @view Z[di, :]
            err = Y[di, t] - dot(Zi, a)
            PZ = P * Zi
            Fi = dot(Zi, PZ) + Hdiag[di]
            if Fi > kalman_tol
                Ki = PZ / Fi
                t >= no_more_missing && (K[:, di] = Ki)
                lik[t, di] = log(Fi) + err * err / Fi + LOG2PI
                a = a + Ki * err
                P = P - PZ * Ki'
                P = (P + P') / 2
            elseif Fi < 0
                return NaN, vec(sum(lik; dims=2))
            end
        end
        a = T * a
        P = T * P * Tt + QQ
        if t >= no_more_missing
            Kv = vec(K)
            if oldK === nothing
                notsteady = any(!isnan, Kv)
            else
                mx = -Inf
                for k in eachindex(Kv)
                    dk = abs(Kv[k] - oldK[k])
                    isnan(dk) || (mx = max(mx, dk))
                end
                notsteady = mx > riccati_tol
            end
            oldK = copy(Kv)
        end
        t += 1
    end
    if t <= smpl                                       # the covariance matrix has converged
        for s in t:smpl
            PP = P
            for i in 1:pp
                Zi = @view Z[i, :]
                err = Y[i, s] - dot(Zi, a)
                PPZ = PP * Zi
                Fi = dot(Zi, PPZ) + Hdiag[i]
                if Fi > kalman_tol
                    Ki = PPZ / Fi
                    a = a + Ki * err
                    PP = PP - PPZ * Ki'
                    lik[s, i] = log(Fi) + err * err / Fi + LOG2PI
                elseif Fi < 0
                    return NaN, vec(sum(lik; dims=2))
                end
            end
            a = T * a
        end
    end
    lik .*= 0.5
    start = presample > diffuse_periods ? presample - diffuse_periods : 0
    sum(@view lik[start+1:end, :]), vec(sum(lik; dims=2))
end

"""
    univariate_kalman_filter_d(Y, a, Pinf, Pstar, T, QQ, Hdiag, Z; kalman_tol, diffuse_kalman_tol, presample, data_index)

Diffuse stage of the univariate filter (`univariate_kalman_filter_d.m`); returns `(dLIK, dlik, a, Pstar)` like `kalman_filter_d`.
"""
function univariate_kalman_filter_d(Y::AbstractMatrix, a::AbstractVector, Pinf::AbstractMatrix, Pstar::AbstractMatrix,
                                    T::AbstractMatrix, QQ::AbstractMatrix, Hdiag::AbstractVector, Z::AbstractMatrix;
                                    kalman_tol=1e-10, diffuse_kalman_tol=1e-6, presample=0, data_index=nothing)
    pp, smpl = size(Y)
    Tt = T'
    dlik = zeros(smpl)
    a = Vector{Float64}(a); Pinf = Matrix{Float64}(Pinf); Pstar = Matrix{Float64}(Pstar)
    rank_(M) = isempty(M) ? 0 : count(>(diffuse_kalman_tol), svdvals(M))
    new_rank = rank_(Pinf)
    t = 1
    s = 0
    while new_rank > 0 && t <= smpl
        s = t
        d = data_index === nothing ? (1:pp) : data_index[t]
        for di in d
            Zi = Z[di, :]
            err = Y[di, t] - dot(Zi, a)
            Fstar = dot(Zi, Pstar * Zi) + Hdiag[di]
            Finf = dot(Zi, Pinf * Zi)
            Kstar = Pstar * Zi
            if Finf > diffuse_kalman_tol && new_rank > 0
                Kinf = Pinf * Zi
                Kinf_Finf = Kinf / Finf
                a = a + Kinf_Finf * err
                Pstar = Pstar + (Kinf * Kinf_Finf') * (Fstar / Finf) - Kstar * Kinf_Finf' - Kinf_Finf * Kstar'
                Pinf = Pinf - Kinf * Kinf_Finf'
                dlik[t] += log(Finf) + LOG2PI
            elseif Fstar > kalman_tol
                dlik[t] += log(Fstar) + err * err / Fstar + LOG2PI
                a = a + Kstar * (err / Fstar)
                Pstar = Pstar - Kstar * Kstar' / Fstar
            elseif Fstar < 0 || Finf < 0
                return NaN, dlik[1:s], a, Pstar
            end
        end
        a = T * a
        Pstar = T * Pstar * Tt + QQ
        Pinf = T * Pinf * Tt
        new_rank > 0 && (new_rank = rank_(Z * Pinf * Z'))
        t += 1
    end
    t > smpl && return NaN, dlik, a, Pstar
    dlik = 0.5 .* dlik[1:s]
    sum(@view dlik[presample+1:end]), dlik, a, Pstar
end
