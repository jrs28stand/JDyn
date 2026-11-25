# Identification analysis at order 1: a port of Dynare 7.2's matlab/+identification (analysis.m, checks.m, get_jacobians.m,
# get_minimal_state_representation.m, ...) following pydyn's identification.py.  Orders 2 and 3 are not supported.

const _ID_EPS = eps(Float64)

"`dyn_vech`: the upper triangle of a square matrix, column after column (rows 1..col of column col)."
_id_vech(M::AbstractMatrix) = isempty(M) ? Float64[] : reduce(vcat, [M[1:c, c] for c in 1:size(M, 1)])

"Octave's `svd` (LAPACK gesvd)."
_id_svd(M; full=false) = svd(Matrix{Float64}(M); full=full, alg=LinearAlgebra.QRIteration())

"Octave's `rank(X, tol)`: singular values above `tol` (default `max(size) eps(max sv)`)."
function _id_rank(X::AbstractMatrix, tol=nothing)
    isempty(X) && return 0
    s = _id_svd(X).S
    t = tol === nothing ? maximum(size(X)) * eps(maximum(s)) : tol
    count(>(t), s)
end

function _id_cond(X::AbstractMatrix)
    isempty(X) && return 0.0
    s = _id_svd(X).S
    s[end] > 0 ? s[1] / s[end] : Inf
end

"`identification.vnorm`: 2-norm of the columns of a matrix with more than one row."
_id_vnorm(A::AbstractMatrix) = vec(sqrt.(sum(abs2, A; dims=1)))

"Multiple cosine of the first column of `H` with the span of the others (`cosn.m`; dgelsd with rcond = eps/2 as Octave's `\\`)."
function _id_cosn(H::AbstractMatrix)
    y = H[:, 1]
    X = H[:, 2:end]
    if size(X, 2) == 0
        b = zeros(0)
    else
        sol, _ = LinearAlgebra.LAPACK.gelsd!(Matrix{Float64}(X), reshape(Vector{Float64}(y), :, 1), _ID_EPS / 2)
        b = vec(sol[1:size(X, 2), 1])
        (any(isnan, b) || any(isinf, b)) && (b = zeros(length(b)))
    end
    yhat = size(X, 2) > 0 ? X * b : zeros(length(y))
    _id_rank(reshape(yhat, :, 1)) > 0 ? abs(sum(y .* yhat) / sqrt(sum(abs2, y) * sum(abs2, yhat))) : 0.0
end

"Octave's `A\\B` for a square `A`: LU, also where singular to machine precision; minimum-norm least squares only for an exactly zero pivot."
function _id_mldivide(A::AbstractMatrix, B::AbstractVecOrMat)
    isempty(A) && return zeros(size(A, 2), size(B)...)[1:size(A, 2), ntuple(_ -> :, ndims(B) - 1)...]
    F = lu(Matrix{Float64}(A); check=false)
    if any(iszero, diag(F.factors)) || !all(isfinite, F.factors)
        Bm = B isa AbstractVector ? reshape(Vector{Float64}(B), :, 1) : Matrix{Float64}(B)
        sol, _ = LinearAlgebra.LAPACK.gelsd!(Matrix{Float64}(A), copy(Bm), _ID_EPS / 2)
        sol = sol[1:size(A, 2), :]
        return B isa AbstractVector ? vec(sol) : sol
    end
    F \ B
end

"Octave's `inv`: a singular matrix gives Inf."
function _id_inv(A::AbstractMatrix)
    try
        inv(Matrix{Float64}(A))
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        fill(Inf, size(A))
    end
end

_id_empty_checks() = Dict{Symbol,Any}(:cond => 0.0, :rank => 0.0, :ind0 => zeros(1, 0), :indno => zeros(1, 0), :ino => 0.0,
    :Mco => zeros(0, 1), :Pco => zeros(0, 0), :jweak => zeros(1, 0), :jweak_pair => zeros(1, 0))

"""
`identification.checks`: Dict with cond, rank, ind0 (vector), indno (matrix, one row per null direction), ino, Mco (column),
Pco, jweak, jweak_pair (Dynare's names).
"""
function _id_checks(X0::AbstractMatrix, test_flag::Int, tol_rank, tol_sv, param_nbr=nothing)
    X = Matrix{Float64}(X0)
    isempty(X) && return _id_empty_checks()
    (tol_rank === nothing || tol_rank isa AbstractString) && (tol_rank = maximum(size(X)) * eps(opnorm(X, 2)))
    tol_sv === nothing && (tol_sv = 1e-3)
    param_nbr === nothing && (param_nbr = size(X, 2))
    indno = Vector{Vector{Float64}}([zeros(param_nbr)])
    rankrequired = size(X, 2)
    if test_flag == 2
        Xpar, Xrest = X[:, 1:param_nbr], X[:, param_nbr+1:end]
    else
        Xpar, Xrest = X, zeros(size(X, 1), 0)
    end
    nrest = size(Xrest, 2)
    ind1 = size(Xpar, 1) > 1 ? findall(>=(_ID_EPS), _id_vnorm(Xpar)) : findall(>=(_ID_EPS), vec(abs.(Xpar)))
    nonzero(ind) = test_flag == 3 ? Xpar[ind, ind] : Xpar[:, ind]
    svdV(M) = _id_svd(M; full=size(M, 1) < size(M, 2)).V
    Xn = nonzero(ind1)
    M = nrest == 0 ? Xn : hcat(Xn, Xrest)
    ee1 = svdV(M)
    condX = _id_cond(M)
    rankX = _id_rank(X, tol_rank)
    icheck = false
    if param_nbr > 0 && rankX < rankrequired
        ind11 = ones(length(ind1))
        for j in rankX+1:size(M, 2)
            if count(>(tol_sv), abs.(ee1[:, j])) == 1
                icheck = true
                temp = Float64.(abs.(ee1[:, j]) .<= tol_sv)
                ind11 .*= test_flag == 2 ? temp[1:length(temp)-nrest] : temp
            end
        end
        ind1 = ind1[findall(!iszero, ind11)]
    end
    if icheck
        Xn = nonzero(ind1)
        M = nrest == 0 ? Xn : hcat(Xn, Xrest)
        ee1 = svdV(M)
        condX = _id_cond(M)
        rankX = _id_rank(X, tol_rank)
    end
    ind0 = zeros(param_nbr)
    ind0[ind1] .= 1
    if test_flag == 0 || test_flag == 3
        Mco = test_flag == 3 ? fill(NaN, param_nbr) : zeros(isempty(ind1) ? 0 : maximum(ind1))
        Xs = X[ind1, ind1]
        deltaX = sqrt.(diag(Xs))
        tildaX = Xs ./ (deltaX * deltaX')
        Mco[ind1] = 1 .- 1 ./ diag(_id_inv(tildaX))
        Pco = fill(NaN, param_nbr, param_nbr)
        Pco[ind1, ind1] = _id_inv(Xs)
        sd = sqrt.(diag(Pco))
        Pco = abs.(Pco ./ (sd * sd'))
    else
        Mco = fill(NaN, param_nbr)
        nc = size(Xn, 2)
        for ii in 1:nc
            others = [k for k in 1:nc if k != ii]
            Mco[ind1[ii]] = _id_cosn(hcat(Xn[:, [ii]], Xn[:, others], Xrest))
        end
    end
    ixno = 0
    nn = filter(!isnan, 1 .- Mco)
    low = isempty(nn) ? Inf : minimum(nn)
    if param_nbr > 0 && (rankX < rankrequired || low < tol_rank)
        if length(ind1) < param_nbr
            ixno += 1
            sind = Set(ind1)
            indno = [Float64[k in sind ? 0.0 : 1.0 for k in 1:param_nbr]]
        end
        for j in rankX+1:size(M, 2)
            ixno += 1
            row = zeros(param_nbr)
            temp = Float64.(abs.(ee1[:, j]) .> tol_sv)
            test_flag == 2 && (temp = temp[1:length(temp)-nrest])
            row[ind1] = temp
            while length(indno) < ixno
                push!(indno, zeros(param_nbr))
            end
            indno[ixno] = row
        end
    end
    indnoM = permutedims(reduce(hcat, indno))
    jweak = zeros(param_nbr)
    jweak_pair = zeros(param_nbr, param_nbr)
    if test_flag != 0
        Pco = fill(NaN, param_nbr, param_nbr)
        nc = size(Xn, 2)
        for ii in 1:nc
            Pco[ind1[ii], ind1[ii]] = 1
            for jj in ii+1:nc
                v = _id_cosn(hcat(Xn[:, [ii]], Xn[:, [jj]], Xrest))
                Pco[ind1[ii], ind1[jj]] = v
                Pco[ind1[jj], ind1[ii]] = v
            end
        end
        for j in 1:param_nbr
            if Mco[j] > 1 - tol_rank
                jweak[j] = 1
                for jx in j+1:param_nbr
                    if Pco[j, jx] > 1 - tol_rank
                        jweak_pair[j, jx] = 1
                        jweak_pair[jx, j] = 1
                    end
                end
            end
        end
    end
    Dict{Symbol,Any}(:cond => condX, :rank => Float64(rankX), :ind0 => ind0, :indno => indnoM, :ino => Float64(ixno),
        :Mco => reshape(Mco, :, 1), :Pco => Pco, :jweak => jweak, :jweak_pair => _id_vech(jweak_pair))
end

# ------------------------------------------------------------------------------------------------ minimal system
function _id_check_minimality(a, b, c)
    n = size(a, 1)
    CC, OO = b, c
    if n >= 2
        for k in 1:n-1
            ak = a^k
            CC = hcat(CC, ak * b)
            OO = vcat(OO, c * ak)
        end
    end
    _id_rank(CC) == n && _id_rank(OO) == n
end

function _id_controllability_staircase(a, b, c, tol=nothing)
    ra = size(a, 1)
    ptjn1 = Matrix{Float64}(I, ra, ra)
    ajn1, bjn1 = a, b
    deltajn1 = 0
    k = zeros(ra)
    tol === nothing && (tol = ra * opnorm(a, 1) * _ID_EPS)
    for jj in 1:ra
        uj = _id_svd(bjn1; full=true).U
        rsj = size(bjn1, 1)
        uj = uj[:, end:-1:1]
        bb = uj' * bjn1
        roj = _id_rank(bb, tol)
        sigmaj = size(bb, 1) - roj
        k[jj] = roj
        (roj == 0 || sigmaj == 0) && break
        abxy = uj' * ajn1 * uj
        aj = abxy[1:sigmaj, 1:sigmaj]
        bj = abxy[1:sigmaj, sigmaj+1:sigmaj+roj]
        ajn1, bjn1 = aj, bj
        ruj, cuj = size(uj)
        ptjn1 = ptjn1 * [uj zeros(ruj, deltajn1); zeros(deltajn1, cuj) Matrix{Float64}(I, deltajn1, deltajn1)]
        deltajn1 += roj
    end
    t = ptjn1'
    t * a * t', t * b, c * t', t, k
end

function _id_observability_staircase(a, b, c, tol)
    aa, bb, cc, t, k = _id_controllability_staircase(copy(a'), copy(c'), copy(b'), tol)
    copy(aa'), copy(cc'), copy(bb'), t, k
end

"`identification.get_minimal_state_representation`: `(ok, minns, minSYS)` (`minSYS` nothing if the reduction is not minimal)."
function _id_minimal_state(SYS::Dict, derivs_flag::Bool=false, realsmall=1e-7)
    nspred, exo_nbr = size(SYS[:B])
    varobs_nbr = size(SYS[:C], 1)
    _id_check_minimality(SYS[:A], SYS[:B], SYS[:C]) && return true, nspred, SYS
    a, b, c = SYS[:A], SYS[:B], SYS[:C]
    ns = size(b, 1)
    am, bm, cm, U, k = _id_controllability_staircase(a, b, c, realsmall)
    nu = ns - Int(sum(k))
    am, bm, cm = am[nu+1:ns, nu+1:ns], bm[nu+1:ns, :], cm[:, nu+1:ns]
    ns -= nu
    if ns > 0
        am, bm, cm, _, k = _id_observability_staircase(am, bm, cm, realsmall)
        nu = ns - Int(sum(k))
        am, bm, cm = am[nu+1:ns, nu+1:ns], bm[nu+1:ns, :], cm[:, nu+1:ns]
    end
    minns = size(am, 1)
    _id_check_minimality(am, bm, cm) || return false, nothing, nothing
    out = Dict{Symbol,Any}(:A => am, :B => bm, :C => cm, :D => SYS[:D])
    if derivs_flag
        tot = size(SYS[:dA], 3)
        dA = zeros(minns, minns, tot); dB = zeros(minns, exo_nbr, tot); dC = zeros(varobs_nbr, minns, tot)
        for jp in 1:tot
            dA[:, :, jp] = (U * SYS[:dA][:, :, jp] * U')[1:minns, 1:minns]
            dB[:, :, jp] = (U*SYS[:dB][:, :, jp])[1:minns, :]
            dC[:, :, jp] = (SYS[:dC][:, :, jp]*U')[:, 1:minns]
        end
        out[:dA], out[:dB], out[:dC], out[:dD] = dA, dB, dC, SYS[:dD]
    end
    true, minns, out
end

# ------------------------------------------------------------------------------------------------ pruned state space (order 1)
"`pruned_SS.commutation(n, m)`."
_id_commutation(n, m) = reshape(kron(vec(Matrix{Float64}(I, n, n)), Matrix{Float64}(I, m, m)), n * m, n * m)

"`pruned_SS.duplication(p)`: duplication matrix and its Moore-Penrose inverse."
function _id_duplication(p)
    a = Float64.(tril(ones(p, p)))
    ii = findall(!iszero, vec(a))
    flat = vec(a)
    flat[ii] = 1:length(ii)
    a = reshape(flat, p, p)
    a = a + tril(a, -1)'
    j = Int.(vec(a))
    Dp = zeros(p * p, p * (p + 1) ÷ 2)
    for (r, c) in enumerate(j)
        Dp[r, c] = 1
    end
    Dp, (Dp' * Dp) \ Dp'
end

"""
First-order part of `pruned_SS.pruned_state_space_system`: A, B, C, D of the predetermined variables and observables, the
(auto)covariances of the observables and, with `derivs` (dghx, dghu, dSigma_e, dYss and `nsc`), their Jacobians in all
parameters (stderr, corr, model).  `indy` are 1-based positions in decision-rule order.
"""
function _id_pruned_first_order(ghx, ghu, Sigma, Yss, indy, nlags, useautocorr, nstatic, nspred, derivs=nothing; qz_unit=1 - 1e-6)
    indx = nstatic .+ (1:nspred)
    A, B = ghx[indx, :], ghu[indx, :]
    C, D = ghx[indy, :], ghu[indy, :]
    ev = isempty(A) ? Float64[] : abs.(eigvals(A))
    !isempty(ev) && maximum(ev) >= qz_unit && throw(ArgumentError("identification: unit roots (diffuse filter) are not supported"))
    Varinov = Sigma
    lyap(a, q) = isempty(a) ? zeros(0, 0) : lyapunov_symm(a, q)[1]
    Var_z = lyap(A, B * Varinov * B')
    ny = length(indy)
    out = Dict{Symbol,Any}(:indx => collect(indx), :indy => indy, :A => A, :B => B, :C => C, :D => D, :Varinov => Varinov)
    Var_y = C * Var_z * C' + D * Varinov * D'
    Var_y[abs.(Var_y).<1e-12] .= 0
    sdy = sqrt.(diag(Var_y))
    sy = sdy * sdy'
    tmp = A * Var_z * C' + B * Varinov * D'
    Var_yi = zeros(ny, ny, nlags)
    Ai = Matrix{Float64}(I, size(A, 1), size(A, 1))
    for i in 1:nlags
        Var_yi[:, :, i] = C * Ai * tmp
        Ai = Ai * A
    end
    out[:Var_y], out[:Var_yi], out[:E_y] = Var_y, Var_yi, Yss[indy]
    useautocorr && (out[:Corr_yi] = Var_yi ./ sy)
    derivs === nothing && return out
    dA, dB = derivs[:dghx][indx, :, :], derivs[:dghu][indx, :, :]
    dC, dD = derivs[:dghx][indy, :, :], derivs[:dghu][indy, :, :]
    dS = derivs[:dSigma_e]
    nsc = derivs[:nsc]
    tot = size(dA, 3)
    dVar_z = zeros(size(A, 1), size(A, 1), tot)
    for j in 1:tot
        if j <= nsc
            Q = B * dS[:, :, j] * B'
        else
            dOm = dB[:, :, j] * Varinov * B' + B * Varinov * dB[:, :, j]'
            Q = dA[:, :, j] * Var_z * A' + A * Var_z * dA[:, :, j]' + dOm
        end
        dVar_z[:, :, j] = lyap(A, Q)
    end
    dVar_y = zeros(ny, ny, tot)
    for j in 1:tot
        t = dC[:, :, j] * Var_z * C' + C * dVar_z[:, :, j] * C' + C * Var_z * dC[:, :, j]' +
            dD[:, :, j] * Varinov * D' + D * dS[:, :, j] * D' + D * Varinov * dD[:, :, j]'
        t[abs.(t).<1e-12] .= 0
        dVar_y[:, :, j] = t
    end
    dVar_yi = zeros(ny, ny, nlags, tot)
    dCorr_yi = useautocorr ? zeros(ny, ny, nlags, tot) : nothing
    for j in 1:tot
        Ai = Matrix{Float64}(I, size(A, 1), size(A, 1))
        dAi = zeros(size(A))
        dtmp = dA[:, :, j] * Var_z * C' + A * dVar_z[:, :, j] * C' + A * Var_z * dC[:, :, j]' +
               dB[:, :, j] * Varinov * D' + B * dS[:, :, j] * D' + B * Varinov * dD[:, :, j]'
        for i in 1:nlags
            dVar_yi[:, :, i, j] = dC[:, :, j] * Ai * tmp + C * dAi * tmp + C * Ai * dtmp
            if useautocorr
                dv = 0.5 ./ sdy .* diag(dVar_y[:, :, j])
                dsy = dv * sdy' + sdy * dv'
                dCorr_yi[:, :, i, j] = (dVar_yi[:, :, i, j] .* sy - dsy .* Var_yi[:, :, i]) ./ (sy .* sy)
            end
            dAi = dAi * A + Ai * dA[:, :, j]
            Ai = Ai * A
        end
    end
    dE_y = zeros(ny, tot)
    dE_y[:, nsc+1:end] = derivs[:dYss][indy, :]
    merge!(out, Dict{Symbol,Any}(:dA => dA, :dB => dB, :dC => dC, :dD => dD, :dVarinov => dS, :dVar_y => dVar_y,
        :dVar_yi => dVar_yi, :dE_y => dE_y))
    useautocorr && (out[:dCorr_yi] = dCorr_yi)
    out
end

# ------------------------------------------------------------------------------------------------ asymptotic Hessian
struct _IdNegativeVariance <: Exception end
struct _IdSingularF <: Exception end

_id_outer(a, b) = a * b'

"""
The `asymptotic` Hessian of `univariate_kalman_filter.m` (no measurement error): the filter `dsge_likelihood.m` switches to
where the prediction error covariance is singular.  `Y` [pp x T], `Z` 1-based positions of the observables.
"""
function _id_univariate_asymptotic_hessian(Y, T, R, Q, Z, DT, DYss, DOm, Pstar; kalman_tol=1e-10, riccati_tol=1e-6, presample=0)
    mm, k = size(T, 1), size(DT, 3)
    pp = length(Z)
    QQ = R * Q * R'
    nobs = size(Y, 2)
    a = zeros(mm)
    P = copy(Pstar)
    DP = zeros(mm, mm, k)
    for i in 1:k
        DP[:, :, i] = lyapunov_symm(T, DT[:, :, i] * Pstar * T' + T * Pstar * DT[:, :, i]' + DOm[:, :, i])[1]
    end
    Da = zeros(mm, k)
    Hess = zeros(k, k)
    DDK = zeros(mm, pp, k)
    DDF = zeros(pp, k)
    K = fill(NaN, mm, pp)
    oldK = fill(Inf, mm * pp)

    function dstate(a, P, Da, DP, notsteady)
        Da1 = zeros(size(Da))
        DP1 = DP
        notsteady && (DP1 = zeros(size(DP)))
        for j in 1:k
            Da1[:, j] = T * Da[:, j] + DT[:, :, j] * a
            notsteady && (DP1[:, :, j] = T * DP[:, :, j] * T' + DT[:, :, j] * P * T' + T * P * DT[:, :, j]')
        end
        notsteady && (DP1 = DP1 + DOm)
        Da1, DP1
    end

    notsteady, t = true, 0
    while notsteady && t < nobs
        P = (P + P') / 2
        for i in 1:pp
            z = Z[i]
            pe = Y[i, t+1] - a[z]
            PZ = P[:, z]
            Fi = PZ[z]
            if Fi > kalman_tol
                Ki = PZ / Fi
                K[:, i] = Ki
                Dv = -Da[z, :] - DYss[z, :]
                DF = DP[z, z, :]
                DK = DP[:, z, :] / Fi - _id_outer(PZ, DF) / Fi^2
                DDK[:, i, :] = DK
                DDF[i, :] = DF
                Hesst = _id_outer(DF, DF) / Fi^2 + 2 * _id_outer(Dv, Dv) / Fi
                Da = Da + DK * pe + _id_outer(Ki, Dv)
                DP1 = zeros(size(DP))
                for j in 1:k
                    DP1[:, :, j] = DP[:, :, j] - _id_outer(DP[:, z, j], Ki) - _id_outer(PZ, DK[:, j])
                end
                DP = DP1
                t + 1 > presample && (Hess = Hess + Hesst)
                a = a + Ki * pe
                P = P - _id_outer(PZ, Ki)
                P = (P + P') / 2
            elseif Fi < 0
                throw(_IdNegativeVariance())
            end
        end
        Da, DP = dstate(a, P, Da, DP, true)
        a = T * a
        P = T * P * T' + QQ
        Kv = vec(K)
        diff = abs.(Kv .- oldK)
        nn = filter(!isnan, diff)
        notsteady = !isempty(nn) && maximum(nn) > riccati_tol
        oldK = copy(Kv)
        t += 1
    end
    Hess = -Hess / 2
    if t < nobs
        H2 = zeros(k, k)
        for tt in t:nobs-1
            PP = copy(P)
            for i in 1:pp
                z = Z[i]
                pe = Y[i, tt+1] - a[z]
                PPZ = PP[:, z]
                Fi = PPZ[z]
                if Fi > kalman_tol
                    Ki = PPZ / Fi
                    a = a + Ki * pe
                    PP = PP - _id_outer(PPZ, Ki)
                    Dv = -Da[z, :] - DYss[z, :]
                    DK, DF = DDK[:, i, :], DDF[i, :]
                    H2 = H2 + _id_outer(DF, DF) / Fi^2 + 2 * _id_outer(Dv, Dv) / Fi
                    Da = Da + DK * pe + _id_outer(Ki, Dv)
                end
            end
            Da, _ = dstate(a, P, Da, DP, false)
            a = T * a
        end
        Hess = Hess + (-H2 / 2)
    end
    Hess
end

"Octave's `rcond` (reciprocal condition number in the 1-norm)."
function _id_rcond(F)
    try
        1.0 / (opnorm(F, 1) * opnorm(inv(Matrix{Float64}(F)), 1))
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        0.0
    end
end

"""
The `asymptotic` Hessian of `kalman_filter.m` as `dsge_likelihood.m` returns it (the negative of the information matrix).
`Y` [pp x T], `Z` 1-based positions of the observables in the restricted state vector, `DT`/`DOm` [mm x mm x k], `DYss` [mm x k].
"""
function _id_asymptotic_hessian(Y, T, R, Q, Z, DT, DYss, DOm, Pstar; kalman_tol=1e-10, riccati_tol=1e-6, presample=0)
    mm, k = size(T, 1), size(DT, 3)
    pp = length(Z)
    QQ = R * Q * R'
    nobs = size(Y, 2)
    a = zeros(mm)
    P = copy(Pstar)
    DP = zeros(mm, mm, k)
    for i in 1:k
        DP[:, :, i] = lyapunov_symm(T, DT[:, :, i] * Pstar * T' + T * Pstar * DT[:, :, i]' + DOm[:, :, i])[1]
    end
    Da = zeros(mm, k)
    Hess = zeros(k, k)
    DK = DF = nothing
    notsteady, oldK, t = true, [Inf], 0
    K = iF = nothing
    function step!(Da, v, tmp, K, iF, DK, DF)
        Dv = -Da[Z, :] - DYss[Z, :]
        Dtmp = zeros(size(Da))
        for ii in 1:k
            Dtmp[:, ii] = Da[:, ii] + DK[:, :, ii] * v + K * Dv[:, ii]
            Da[:, ii] = DT[:, :, ii] * tmp + T * Dtmp[:, ii]
        end
        vecDF = reshape(DF, pp * pp, k)
        2 * Dv' * iF * Dv + vecDF' * kron(iF, iF) * vecDF
    end
    while notsteady && t < nobs
        v = Y[:, t+1] - a[Z]
        F = P[Z, Z]
        rescale = false
        sig = nothing
        if _id_rcond(F) < kalman_tol
            sig = sqrt.(diag(F))
            (any(<(kalman_tol), diag(F)) || _id_rcond(F ./ (sig * sig')) < kalman_tol) && throw(_IdSingularF())
            rescale = true
        end
        iF = rescale ? inv(F ./ (sig * sig')) ./ (sig * sig') : inv(F)
        K = P[:, Z] * iF
        Ptmp = T * (P - K * P[Z, :]) * T' + QQ
        tmp = a + K * v
        tmpP = P - K * P[Z, :]
        DF = zeros(pp, pp, k)
        DK = zeros(mm, pp, k)
        DP1 = zeros(size(DP))
        for ii in 1:k
            DF[:, :, ii] = DP[Z, Z, ii]
            DiF = -iF * DF[:, :, ii] * iF
            DK[:, :, ii] = DP[:, Z, ii] * iF + P[:, Z] * DiF
            Dtmp = DP[:, :, ii] - DK[:, :, ii] * P[Z, :] - K * DP[Z, :, ii]
            DP1[:, :, ii] = DT[:, :, ii] * tmpP * T' + T * Dtmp * T' + T * tmpP * DT[:, :, ii]' + DOm[:, :, ii]
        end
        DP = DP1
        Hesst = step!(Da, v, tmp, K, iF, DK, DF)
        t + 1 > presample && (Hess = Hess + Hesst)
        a = T * tmp
        P = Ptmp
        notsteady = maximum(abs.(vec(K) .- oldK)) > riccati_tol
        oldK = vec(K)
        t += 1
    end
    Hess = -Hess / 2
    if t < nobs
        H2 = zeros(k, k)
        for tt in t:nobs-1
            v = Y[:, tt+1] - a[Z]
            tmp = a + K * v
            H2 = H2 + step!(Da, v, tmp, K, iF, DK, DF)
            a = T * tmp
        end
        Hess = Hess + (-H2 / 2)
    end
    Hess
end
