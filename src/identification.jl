# Identification analysis at order 1: a port of Dynare 7.2's matlab/+identification (analysis.m, checks.m, get_jacobians.m,
# get_minimal_state_representation.m, ...) following pydyn's identification.py.  Orders 2 and 3 are not supported.

const _ID_EPS = eps(Float64)

"Sample standard deviation (n-1) along `dims` (Statistics is not a dependency)."
function _id_std(A::AbstractMatrix; dims::Int)
    n = size(A, dims)
    mu = sum(A; dims=dims) ./ n
    _id_sqrt.(sum(abs2, A .- mu; dims=dims) ./ (n - 1))
end

"`cov(X; dims=2)`: covariance of the rows of `X` over its columns (n-1)."
function _id_cov_rows(X::AbstractMatrix)
    n = size(X, 2)
    Xc = X .- sum(X; dims=2) ./ n
    Xc * Xc' / (n - 1)
end

"sqrt of a real number; NaN for a negative one (numpy semantics, which the references reproduce)."
_id_sqrt(x::Real) = x < 0 ? NaN : sqrt(x)

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
_id_vnorm(A::AbstractMatrix) = vec(_id_sqrt.(sum(abs2, A; dims=1)))

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
        deltaX = _id_sqrt.(diag(Xs))
        tildaX = Xs ./ (deltaX * deltaX')
        Mco[ind1] = 1 .- 1 ./ diag(_id_inv(tildaX))
        Pco = fill(NaN, param_nbr, param_nbr)
        Pco[ind1, ind1] = _id_inv(Xs)
        sd = _id_sqrt.(diag(Pco))
        Pco = abs.(Pco ./ (sd * sd'))
    else
        Mco = fill(NaN, param_nbr)
        nc = size(Xn, 2)
        for ii in 1:nc
            others = [k for k in 1:nc if k != ii]
            Mco[ind1[ii]] = _id_cosn(nrest == 0 ? hcat(Xn[:, [ii]], Xn[:, others]) : hcat(Xn[:, [ii]], Xn[:, others], Xrest))
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
                v = _id_cosn(nrest == 0 ? hcat(Xn[:, [ii]], Xn[:, [jj]]) : hcat(Xn[:, [ii]], Xn[:, [jj]], Xrest))
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
    sdy = _id_sqrt.(diag(Var_y))
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
            sig = _id_sqrt.(diag(F))
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

# ------------------------------------------------------------------------------------------------ forward-mode duals
"An analysis that Dynare's identification cannot carry out in this port (refused, not guessed)."
struct IdentificationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::IdentificationError) = print(io, "IdentificationError: ", e.msg)

"The model does not solve at the point tested (Dynare prints a message and returns)."
struct IdentificationStop <: Exception
    msg::String
end

"""
A dual number (value, one directional derivative).  The preprocessor's Julia functions are generic in `Real`, so one
evaluation on duals gives the exact derivative in one direction (a parameter, or a parameter and the steady state moving
along its own derivative).  Functions without a derivative here (`sign`, `max`, `min`, `round`, `floor`, `ceil`, `mod`) throw.
"""
struct _IdDual <: Real
    v::Float64
    d::Float64
end
Base.promote_rule(::Type{_IdDual}, ::Type{T}) where {T<:Real} = _IdDual
Base.convert(::Type{_IdDual}, x::_IdDual) = x
Base.convert(::Type{_IdDual}, x::Real) = _IdDual(Float64(x), 0.0)
_IdDual(x::Real) = _IdDual(Float64(x), 0.0)
Base.zero(::Type{_IdDual}) = _IdDual(0.0, 0.0)
Base.one(::Type{_IdDual}) = _IdDual(1.0, 0.0)
Base.zero(::_IdDual) = _IdDual(0.0, 0.0)
Base.one(::_IdDual) = _IdDual(1.0, 0.0)
Base.float(x::_IdDual) = x
Base.real(x::_IdDual) = x
Base.:+(a::_IdDual, b::_IdDual) = _IdDual(a.v + b.v, a.d + b.d)
Base.:-(a::_IdDual, b::_IdDual) = _IdDual(a.v - b.v, a.d - b.d)
Base.:-(a::_IdDual) = _IdDual(-a.v, -a.d)
Base.:+(a::_IdDual) = a
Base.:*(a::_IdDual, b::_IdDual) = _IdDual(a.v * b.v, a.d * b.v + a.v * b.d)
Base.:/(a::_IdDual, b::_IdDual) = _IdDual(a.v / b.v, (a.d - (a.v / b.v) * b.d) / b.v)
Base.inv(a::_IdDual) = _IdDual(1 / a.v, -a.d / (a.v * a.v))
Base.:(==)(a::_IdDual, b::_IdDual) = a.v == b.v
Base.:<(a::_IdDual, b::_IdDual) = a.v < b.v
Base.:<=(a::_IdDual, b::_IdDual) = a.v <= b.v
Base.isless(a::_IdDual, b::_IdDual) = a.v < b.v
Base.isequal(a::_IdDual, b::_IdDual) = a.v == b.v
Base.hash(a::_IdDual, h::UInt) = hash(a.v, h)
Base.isnan(a::_IdDual) = isnan(a.v) || isnan(a.d)
Base.isinf(a::_IdDual) = isinf(a.v)
Base.isfinite(a::_IdDual) = isfinite(a.v)
Base.abs(a::_IdDual) = _IdDual(abs(a.v), a.v == 0 ? 0.0 : sign(a.v) * a.d)
Base.abs2(a::_IdDual) = a * a
Base.exp(a::_IdDual) = (e = exp(a.v); _IdDual(e, e * a.d))
Base.expm1(a::_IdDual) = _IdDual(expm1(a.v), exp(a.v) * a.d)
Base.log(a::_IdDual) = _IdDual(log(a.v), a.d / a.v)
Base.log1p(a::_IdDual) = _IdDual(log1p(a.v), a.d / (1 + a.v))
Base.log10(a::_IdDual) = _IdDual(log10(a.v), a.d / (a.v * log(10.0)))
Base.log2(a::_IdDual) = _IdDual(log2(a.v), a.d / (a.v * log(2.0)))
Base.sqrt(a::_IdDual) = (s = sqrt(a.v); _IdDual(s, a.d / (2 * s)))
Base.cbrt(a::_IdDual) = (c = cbrt(a.v); _IdDual(c, a.d / (3 * c * c)))
Base.sin(a::_IdDual) = _IdDual(sin(a.v), cos(a.v) * a.d)
Base.cos(a::_IdDual) = _IdDual(cos(a.v), -sin(a.v) * a.d)
Base.tan(a::_IdDual) = (t = tan(a.v); _IdDual(t, (1 + t * t) * a.d))
Base.asin(a::_IdDual) = _IdDual(asin(a.v), a.d / sqrt(1 - a.v^2))
Base.acos(a::_IdDual) = _IdDual(acos(a.v), -a.d / sqrt(1 - a.v^2))
Base.atan(a::_IdDual) = _IdDual(atan(a.v), a.d / (1 + a.v^2))
Base.sinh(a::_IdDual) = _IdDual(sinh(a.v), cosh(a.v) * a.d)
Base.cosh(a::_IdDual) = _IdDual(cosh(a.v), sinh(a.v) * a.d)
Base.tanh(a::_IdDual) = (t = tanh(a.v); _IdDual(t, (1 - t * t) * a.d))
SpecialFunctions.erf(a::_IdDual) = _IdDual(SpecialFunctions.erf(a.v), 2 / sqrt(pi) * exp(-a.v^2) * a.d)
SpecialFunctions.erfc(a::_IdDual) = _IdDual(SpecialFunctions.erfc(a.v), -2 / sqrt(pi) * exp(-a.v^2) * a.d)
function Base.:^(a::_IdDual, b::_IdDual)
    if b.d == 0
        p = b.v
        d = a.d == 0 ? 0.0 : (p == 0 ? 0.0 : p * a.v^(p - 1) * a.d)
        return _IdDual(a.v^p, d)
    end
    val = a.v^b.v
    da = a.d == 0 ? 0.0 : b.v * a.v^(b.v - 1) * a.d
    _IdDual(val, da + val * log(a.v) * b.d)
end
Base.:^(a::_IdDual, p::Integer) = a ^ _IdDual(Float64(p), 0.0)
Base.:^(a::_IdDual, p::Rational) = a ^ _IdDual(Float64(p), 0.0)
for f in (:sign, :round, :floor, :ceil, :trunc)
    @eval Base.$f(::_IdDual) = throw(IdentificationError("the model uses " * $(string(f)) * ", which has no analytic derivative here"))
end
for f in (:mod, :rem, :max, :min)
    @eval Base.$f(::_IdDual, ::_IdDual) = throw(IdentificationError("the model uses " * $(string(f)) * ", which has no analytic derivative here"))
end

const _ID_NONSMOOTH = r"\b(sign|max|min|round|floor|ceil|mod|rem|fix)\s*\("
const _ID_CMP = r"(<=|>=|==|!=|(?<![-=])[<>](?!=))"

"""
Refuse a model whose generated functions use an operator without an analytic derivative (`sign`, `max`, `min`, comparisons,
`round`, `floor`, `ceil`, `mod`) or an external function: its derivatives would silently be wrong.
"""
function _id_check_analytic(m::Model)
    jdir = joinpath(dirname(m.json_dir), "julia")
    bad = String[]
    for f in ("SparseStaticResid!.jl", "SparseStaticResidTT!.jl", "SparseStaticG1!.jl", "SparseStaticG1TT!.jl",
              "SparseDynamicResid!.jl", "SparseDynamicResidTT!.jl", "SparseDynamicG1!.jl", "SparseDynamicG1TT!.jl")
        p = joinpath(jdir, f)
        isfile(p) || continue
        for line in split(read(p, String), '\n')
            s = strip(line)
            (startswith(s, "@assert") || startswith(s, "function") || startswith(s, "#")) && continue
            for mt in eachmatch(_ID_NONSMOOTH, s)
                push!(bad, mt.captures[1])
            end
            occursin(_ID_CMP, s) && push!(bad, "a comparison")
            occursin("_EXT", s) && push!(bad, "an external function")
        end
    end
    isempty(bad) || throw(IdentificationError("analytic parameter derivatives are refused: the model uses " *
        join(sort!(unique(bad)), ", ") * ", which has no analytic derivative here"))
    nothing
end

# ------------------------------------------------------------------------------------------------ generic evaluation
function _id_eval_vec(f::Function, ntemp::Int, nout::Int, args...)
    TT = promote_type(map(eltype, args)...)
    a = [Vector{TT}(x) for x in args]
    T = zeros(TT, max(ntemp, 1))
    out = zeros(TT, nout)
    out, T, a
end

"Static residuals on `Real` vectors (duals)."
function _id_static_resid(m::Model, y, x, p)
    f = m.fns
    TT = promote_type(eltype(y), eltype(x), eltype(p))
    yy, xx, pp = Vector{TT}(y), Vector{TT}(x), Vector{TT}(p)
    T = zeros(TT, max(f.sta_tmp[1], 1))
    r = zeros(TT, _neq(m))
    try
        Base.invokelatest(f.sta_resid_tt, T, yy, xx, pp)
        Base.invokelatest(f.sta_resid, T, r, yy, xx, pp)
    catch e
        e isa DomainError || rethrow()
        throw(IdentificationError("a derivative of the model is undefined at this point (domain error)"))
    end
    r
end

"Dense dynamic Jacobian on `Real` vectors (duals)."
function _id_dynamic_jac(m::Model, yy, x, p, ss)
    f = m.fns
    TT = promote_type(eltype(yy), eltype(x), eltype(p), eltype(ss))
    y1, x1, p1, s1 = Vector{TT}(yy), Vector{TT}(x), Vector{TT}(p), Vector{TT}(ss)
    T = zeros(TT, max(f.dyn_tmp[1] + f.dyn_tmp[2], 1))
    v = zeros(TT, length(f.g1_rows))
    try
        Base.invokelatest(f.dyn_g1_tt, T, y1, x1, p1, s1)
        Base.invokelatest(f.dyn_g1, T, v, y1, x1, p1, s1)
    catch e
        e isa DomainError || rethrow()
        throw(IdentificationError("a derivative of the model is undefined at this point (domain error)"))
    end
    J = zeros(TT, m.n_endo, 3 * m.n_endo + m.n_exo)
    for k in eachindex(v)
        f.g1_cols[k] > size(J, 2) && continue
        J[f.g1_rows[k], f.g1_cols[k]] += v[k]
    end
    J
end

_id_val(x::_IdDual) = x.v
_id_der(x::_IdDual) = x.d
_id_der(M::AbstractArray{_IdDual}) = map(_id_der, M)

# ------------------------------------------------------------------------------------------------ derivatives of the solution
"""
Derivative of the steady state (declaration order) by the implicit function theorem (Ratto and Iskrev 2012, eq. 5) and the
derivative of the static residuals; `[endo x param]`.
"""
function _id_steady_state_derivatives(m::Model, ys, params, qz_criterium)
    n, npar = m.n_endo, length(m.param_names)
    x = exo_steady(m)
    rp = zeros(n, npar)
    ysd = _IdDual.(ys, 0.0)
    xd = _IdDual.(x, 0.0)
    for j in 1:npar
        pc = [_IdDual(params[i], i == j ? 1.0 : 0.0) for i in 1:npar]
        rp[:, j] = _id_der(_id_static_resid(m, ysd, xd, pc)[1:n])
    end
    g1 = static_jac(m, Vector{Float64}(ys), x, Vector{Float64}(params))
    dys = nothing
    c = try
        cond(g1)
    catch
        Inf
    end
    if isfinite(c) && c < 1 / eps()
        F = lu(g1; check=false)
        if issuccess(F)
            # one right-hand side at a time: where a parameter does not enter, the solution stays exactly zero
            dys = -reduce(hcat, [F \ rp[:, j] for j in 1:npar])
        end
    end
    if dys === nothing || !all(isfinite, dys)
        # singular static Jacobian (unit roots): the non-zero eigenvalue part only, as get_perturbation_params_derivs.m does
        F = schur(g1)
        sel = abs.(F.values) .< qz_criterium - 1
        ordschur!(F, sel)
        k = count(sel)
        Ts = F.T[k+1:end, k+1:end]
        Us = F.Z[:, k+1:end]
        dys = -Us * (Ts \ (Us' * rp))
    end
    dys, rp
end

"""
Two-sided finite differences (`fjaco.m`: step `dynatol.x * max(|p|, 1)`, divided by `(p+h)-(p-h)`) of the steady state and the
dynamic Jacobian in the selected model parameters, the steady state (with the steady-state file's parameter changes) recomputed
at each point (`analytic_derivation_mode = -2`).  Returns `dys` [endo x all parameters] and `dJ` [endo x columns x selected].
"""
function _id_numerical_dys_dJ(m::Model, indpmodel, params, x; dynatol_x=1e-5)
    n = m.n_endo
    ncol = 3n + m.n_exo
    dys = zeros(n, length(m.param_names))
    dJ = zeros(n, ncol, length(indpmodel))
    for (jj, jp) in enumerate(indpmodel)
        x0 = params[jp]
        h = dynatol_x * max(abs(x0), 1.0)
        xh1, xh0 = x0 + h, x0 - h
        hh = xh1 - xh0
        vals = Tuple{Vector{Float64},Matrix{Float64}}[]
        for xv in (xh1, xh0)
            pv = copy(params)
            pv[jp] = xv
            ysv, ppv = steady_state(m; params=pv)
            ysv, ppv = Vector{Float64}(ysv), Vector{Float64}(ppv)
            push!(vals, (ysv, dynamic_jac(m, vcat(ysv, ysv, ysv), x, ppv, ysv)))
        end
        dys[:, jp] = (vals[1][1] - vals[2][1]) / hh
        dJ[:, :, jj] = (vals[1][2] - vals[2][2]) / hh
    end
    dys, dJ
end

"""
`analytic_derivation_mode = 1` of get_perturbation_params_derivs.m (Iskrev 2010, appendix A): derivatives of the transition
matrix and of ghu by the implicit function theorem with Kronecker products.  Returns declaration-order `dG` [n x n x nm]
and `dH` [n x nexo x nm].
"""
function _id_kron_solution(G, H, f0, fp, fu, dJ, order_var, nm, nexo, n)
    ov = order_var
    I_n = Matrix{Float64}(I, n, n)
    KA = G[ov, ov]
    GAM0, GAM1 = f0[:, ov], -fp[:, ov]
    cols(rng, sgn) = reduce(hcat, [sgn * vec(dJ[:, rng, j][:, ov]) for j in 1:nm])
    dGAM0 = cols(n+1:2n, 1)
    dGAM1 = cols(2n+1:3n, -1)
    dGAM2 = cols(1:n, -1)
    GAM3 = -fu
    dGAM3 = reduce(hcat, [-vec(dJ[:, 3n+1:end, j]) for j in 1:nm])
    dF = kron(I_n, GAM0) - kron(KA', GAM1) - kron(I_n, GAM1 * KA)
    Fp = kron(KA', I_n) * dGAM0 - kron(KA' * KA', I_n) * dGAM1 - dGAM2
    dKA = -(dF \ Fp)
    MM = GAM0 - GAM1 * KA
    invMM = MM \ I_n
    dghu = -kron((invMM * GAM3)', invMM) * (dGAM0 - kron(KA', I_n) * dGAM1 - kron(I_n, GAM1) * dKA) +
           kron(Matrix{Float64}(I, nexo, nexo), invMM) * dGAM3
    dG = zeros(n, n, nm)
    dH = zeros(n, nexo, nm)
    inv_ov = invperm(ov)
    for j in 1:nm
        dKj = reshape(dKA[:, j], n, n)
        dG[:, :, j] = dKj[inv_ov, inv_ov]
        dH[:, :, j] = reshape(dghu[:, j], n, nexo)[inv_ov, :]
    end
    dG, dH
end

"""
Derivatives of the first-order solution with respect to selected parameters, in Dynare's layout
(`+identification/get_perturbation_params_derivs.m`).  `indpmodel` 1-based positions in `m.param_names`, `indpstderr` the
shocks (declaration order) whose standard deviation is a parameter, `indpcorr` pairs of shocks.  The parameter order of the totals
is stderr, corr, model.  Returns a Dict with Dynare's names (`:dYss`, `:dSigma_e`, `:dghx`, `:dghu`, `:dOm`, `:dg1`, `:ghx`, `:ghu`,
`:order_var`, `:state_var`...).
"""
function _id_params_derivs(m::Model, indpmodel, indpstderr, indpcorr, Sigma, ys, params; numerical::Bool=false, kron::Bool=false,
                           qz_criterium=nothing)
    n_exo_det(m) > 0 && throw(IdentificationError("get_perturbation_params_derivs: not compatible with deterministic exogenous variables, please declare as endogenous."))
    _id_check_analytic(m)
    n, nexo = m.n_endo, m.n_exo
    nm, ns, nc = length(indpmodel), length(indpstderr), length(indpcorr)
    ntot = nm + ns + nc
    sol = first_order(m; ys=ys, params=params, loglinear=false, qz_criterium=qz_criterium)
    G, H = sol.G, sol.H
    order_var, state_var = sol.order_var, sol.state_var
    x = exo_steady(m)
    y3 = vcat(ys, ys, ys)
    J = dynamic_jac(m, y3, x, params, ys)
    f0, fp, fu = J[:, n+1:2n], J[:, 2n+1:3n], J[:, 3n+1:end]
    qzc = qz_criterium === nothing ? JDyn.qz_criterium(m, nothing) : qz_criterium
    dJ = zeros(n, size(J, 2), nm)
    if numerical
        dys, dJ = _id_numerical_dys_dJ(m, indpmodel, params, x)
    else
        dys, _ = _id_steady_state_derivatives(m, ys, params, qzc)
        npar = length(params)
        for (jj, jp) in enumerate(indpmodel)
            d = dys[:, jp]
            pc = [_IdDual(params[i], i == jp ? 1.0 : 0.0) for i in 1:npar]
            yd = [_IdDual(y3[i], d[mod1(i, n)]) for i in 1:3n]
            sd = [_IdDual(ys[i], d[i]) for i in 1:n]
            xd = _IdDual.(x, 0.0)
            dJ[:, :, jj] = _id_der(_id_dynamic_jac(m, yd, xd, pc, sd))
        end
    end
    dG = zeros(n, n, nm)
    dH = zeros(n, nexo, nm)
    st = state_var
    ncs = length(st)
    kron_sol = (kron && n * n <= 6400) ? _id_kron_solution(G, H, f0, fp, fu, dJ, order_var, nm, nexo, n) : nothing
    A = f0 + fp * G
    Gss = G[st, st]
    local Kf
    if ncs > 0
        Kf = lu(Base.kron(Matrix{Float64}(I, ncs, ncs), A) + Base.kron(Gss', fp))
    end
    for jj in 1:nm
        dfm, df0, dfp, dfu = dJ[:, 1:n, jj], dJ[:, n+1:2n, jj], dJ[:, 2n+1:3n, jj], dJ[:, 3n+1:end, jj]
        if ncs > 0
            R = -(dfm + df0 * G + dfp * G * G)[:, st]
            dG[:, st, jj] = reshape(Kf \ vec(R), n, ncs)
        end
        dA = df0 + dfp * G + fp * dG[:, :, jj]
        dH[:, :, jj] = -(A \ (dfu + dA * H))
    end
    kron_sol !== nothing && ((dG, dH) = kron_sol)
    sd = _id_sqrt.(diag(Sigma))
    corr = Sigma ./ (sd * sd')
    corr[.!isfinite.(corr)] .= 0.0
    corr[diagind(corr)] .= 1.0
    dSigma = zeros(nexo, nexo, ntot)
    dCorr = zeros(nexo, nexo, ntot)
    isdiag = !any(!iszero, Sigma - Diagonal(diag(Sigma)))
    for (jp, i) in enumerate(indpstderr)
        dSigma[i, i, jp] = 2 * sqrt(Sigma[i, i])
        if !isdiag
            for kk in 1:nexo
                kk == i && continue
                dSigma[i, kk, jp] = corr[i, kk] * sqrt(Sigma[kk, kk])
                dSigma[kk, i, jp] = dSigma[i, kk, jp]
            end
        end
    end
    for (jp, (i, k)) in enumerate(indpcorr)
        dSigma[i, k, ns+jp] = sqrt(Sigma[i, i]) * sqrt(Sigma[k, k])
        dSigma[k, i, ns+jp] = dSigma[i, k, ns+jp]
        dCorr[i, k, ns+jp] = dCorr[k, i, ns+jp] = 1.0
    end
    ghu = H[order_var, :]
    ghx = G[order_var, st]
    dghx = zeros(n, ncs, nm)
    dghu = zeros(n, nexo, nm)
    dOm = zeros(n, n, ntot)
    for jj in 1:nm
        dghx[:, :, jj] = dG[order_var, st, jj]
        dghu[:, :, jj] = dH[order_var, :, jj]
        dOm[:, :, ns+nc+jj] = dghu[:, :, jj] * Sigma * ghu' + ghu * Sigma * dghu[:, :, jj]'
    end
    for jp in 1:ns+nc
        dOm[:, :, jp] = ghu * dSigma[:, :, jp] * ghu'
    end
    Dict{Symbol,Any}(:indpmodel => indpmodel, :indpstderr => indpstderr, :indpcorr => indpcorr,
        :dYss => dys[order_var, indpmodel], :dSigma_e => dSigma, :dCorrelation_matrix => dCorr,
        :dghx => cat(zeros(n, ncs, ns + nc), dghx; dims=3), :dghu => cat(zeros(n, nexo, ns + nc), dghu; dims=3),
        :dOm => dOm, :dg1 => dJ, :ghx => ghx, :ghu => ghu, :order_var => order_var, :state_var => st)
end

# ------------------------------------------------------------------------------------------------ get_jacobians.m
"""
`identification.get_jacobians` at order 1.  `derivs` is `_id_params_derivs`'s result; `indvobs` the positions of the observables
in decision-rule order.  Returns a Dict.
"""
function _id_get_jacobians(m::Model, ghx, ghu, ys, pp, Sigma, derivs, indvobs, nstatic, nspred, nsc, opts; fd=nothing)
    order_var = derivs[:order_var]
    n, nexo = m.n_endo, m.n_exo
    nlags, useautocorr, grid_nbr = Int(opts["ar"]), opts["useautocorr"] != 0, Int(opts["grid_nbr"])
    mode = Int(opts["analytic_derivation_mode"])
    totparam = size(derivs[:dSigma_e], 3)
    nm = size(derivs[:dYss], 2)
    obs_nbr = length(indvobs)
    Yss = ys[order_var]
    lli = m.lead_lag_incidence
    cols = vcat([(p - 1) * n + i for p in 1:3 for i in 1:n if lli[i, p] > 0], 3n .+ (1:nexo))
    J = dynamic_jac(m, vcat(ys, ys, ys), exo_steady(m), pp, ys)
    g1 = J[:, cols]
    dg1 = derivs[:dg1][:, cols, :]
    r = Dict{Symbol,Any}()
    r[:DYNAMIC] = vcat(Yss, vec(g1))
    r[:dDYNAMIC] = vcat(derivs[:dYss], reshape(dg1, size(dg1, 1) * size(dg1, 2), nm))
    r[:REDUCEDFORM] = vcat(Yss, vec(ghx), _id_vech(ghu * Sigma * ghu'))
    dRF = zeros(n * nspred + n * (n + 1) ÷ 2, totparam)
    for j in 1:totparam
        dRF[:, j] = vcat(vec(derivs[:dghx][:, :, j]), _id_vech(derivs[:dOm][:, :, j]))
    end
    r[:dREDUCEDFORM] = vcat(hcat(zeros(n, nsc), derivs[:dYss]), dRF)
    pr = _id_pruned_first_order(ghx, ghu, Sigma, Yss, indvobs, nlags, useautocorr, nstatic, nspred, merge(derivs, Dict{Symbol,Any}(:nsc => nsc)))
    r[:pruned] = pr
    r[:MEAN], r[:dMEAN] = pr[:E_y], pr[:dE_y]
    if opts["no_identification_moments"] == 0
        E_yyi = useautocorr ? pr[:Corr_yi] : pr[:Var_yi]
        mom = vcat(pr[:E_y], _id_vech(pr[:Var_y]), reduce(vcat, [vec(E_yyi[:, :, i]) for i in 1:nlags]))
        r[:MOMENTS] = mom
        nv = obs_nbr * (obs_nbr + 1) ÷ 2
        d = zeros(obs_nbr + nv + nlags * obs_nbr^2, totparam)
        d[1:obs_nbr, :] = pr[:dE_y]
        if mode == -1
            d[obs_nbr+1:end, :] = fd(1, nlags, useautocorr, grid_nbr)
        else
            dE_yyi = useautocorr ? pr[:dCorr_yi] : pr[:dVar_yi]
            for jp in 1:totparam
                d[obs_nbr+1:obs_nbr+nv, jp] = _id_vech(pr[:dVar_y][:, :, jp])
                for i in 1:nlags
                    a = obs_nbr + nv + (i - 1) * obs_nbr^2
                    d[a+1:a+obs_nbr^2, jp] = vec(dE_yyi[:, :, i, jp])
                end
            end
        end
        r[:dMOMENTS] = d
    else
        r[:MOMENTS], r[:dMOMENTS] = zeros(0), zeros(0, totparam)
    end
    A, B, C, D = pr[:A], pr[:B], pr[:C], pr[:D]
    if opts["no_identification_spectrum"] == 0 && mode == -1
        nf = grid_nbr ÷ 2 + 1
        dOm_all = fd(2, nlags, useautocorr, grid_nbr)
        S = zeros(ComplexF64, totparam, totparam)
        for ig in 1:nf
            dO = dOm_all[(ig-1)*obs_nbr^2+1:ig*obs_nbr^2, :]
            S += (ig == 1 ? 1.0 : 2.0) * (dO' * dO)
        end
        S = real(2π * S / (2nf - 1))
        r[:dSPECTRUM_NO_MEAN] = S
        r[:dSPECTRUM] = S + pr[:dE_y]' * pr[:dE_y]
    elseif opts["no_identification_spectrum"] == 0 && mode == 1 && size(A, 1)^2 <= 6400
        nf = grid_nbr ÷ 2 + 1
        freqs = (0:nf-1) .* (π / (grid_nbr / 2))
        nx = size(A, 1)
        IA = Matrix{Float64}(I, nx, nx)
        Sig, dSig = pr[:Varinov], pr[:dVarinov]
        flat(a) = reshape(a, size(a, 1) * size(a, 2), totparam)
        dA, dB, dC, dD = flat(pr[:dA]), flat(pr[:dB]), flat(pr[:dC]), flat(pr[:dD])
        dVarinov = flat(dSig)
        K_obs_exo = _id_commutation(obs_nbr, size(Sig, 1))
        function deriv_abcd(X1, dX1, X2, dX2, X3, dX3)
            I1 = Matrix{Float64}(I, size(X1, 1), size(X1, 1))
            kron(transpose(X3) * transpose(X2), I1) * dX1 + kron(transpose(X3), X1) * dX2 + kron(Matrix{Float64}(I, size(X3, 2), size(X3, 2)), X1 * X2) * dX3
        end
        S = zeros(ComplexF64, totparam, totparam)
        for (ig, w) in enumerate(freqs)
            z = exp(-1im * w)
            zIA = z * IA - A
            zIAinv = zIA \ IA
            Tf = D + C * zIAinv * B
            dzIA = -dA
            dzIAinv = kron(-(transpose(zIA) \ IA), zIAinv) * dzIA
            dTf = dD + deriv_abcd(C, dC, zIAinv, dzIAinv, B, dB)
            dTf_conjt = K_obs_exo * conj(dTf)
            dO = (1 / (2π)) * deriv_abcd(Tf, dTf, Sig, dVarinov, Tf', dTf_conjt)
            S += (ig == 1 ? 1.0 : 2.0) * (dO' * dO)
        end
        S = real(2π * S / (2nf - 1))
        r[:dSPECTRUM_NO_MEAN] = S
        r[:dSPECTRUM] = S + pr[:dE_y]' * pr[:dE_y]
    elseif opts["no_identification_spectrum"] == 0
        nf = grid_nbr ÷ 2 + 1
        freqs = (0:nf-1) .* (π / (grid_nbr / 2))
        IA = Matrix{Float64}(I, size(A, 1), size(A, 1))
        S = zeros(ComplexF64, totparam, totparam)
        Sig, dSig = pr[:Varinov], pr[:dVarinov]
        for (ig, fr) in enumerate(freqs)
            z = exp(1im * fr)
            inv_ = (z * IA - A) \ IA
            T = D + C * inv_ * B
            Th = T'
            dO = zeros(ComplexF64, obs_nbr^2, totparam)
            for j in 1:totparam
                if j <= nsc
                    t = T * dSig[:, :, j] * Th
                else
                    dinv = -inv_ * (-pr[:dA][:, :, j]) * inv_
                    dT = pr[:dD][:, :, j] + pr[:dC][:, :, j] * inv_ * B + C * dinv * B + C * inv_ * pr[:dB][:, :, j]
                    t = dT * Sig * Th + T * dSig[:, :, j] * Th + T * Sig * dT'
                end
                dO[:, j] = (1 / (2π)) * vec(t)
            end
            S += (ig == 1 ? 1.0 : 2.0) * (dO' * dO)
        end
        S = real(2π * S / (2nf - 1))
        r[:dSPECTRUM_NO_MEAN] = S
        r[:dSPECTRUM] = S + pr[:dE_y]' * pr[:dE_y]
    else
        r[:dSPECTRUM_NO_MEAN] = r[:dSPECTRUM] = zeros(0, 0)
    end
    r[:dMINIMAL] = nothing
    r[:minimal_message] = ""
    if opts["no_identification_minimal"] == 0
        if obs_nbr < nexo
            r[:minimal_message] = "WARNING: Komunjer and Ng (2011) failed:\n       There are more shocks and measurement errors than observables, this is not implemented (yet).\n       Skip identification analysis based on minimal state space system.\n"
        else
            indx = pr[:indx]
            SYS = Dict{Symbol,Any}(:A => ghx[indx, :], :dA => derivs[:dghx][indx, :, :], :B => ghu[indx, :],
                :dB => derivs[:dghu][indx, :, :], :C => ghx[indvobs, :], :dC => derivs[:dghx][indvobs, :, :],
                :D => ghu[indvobs, :], :dD => derivs[:dghu][indvobs, :, :])
            ok, minnx, mS = _id_minimal_state(SYS, true)
            if !ok
                r[:minimal_message] = "WARNING: Komunjer and Ng (2011) failed:\n         Conditions for minimality are not fullfilled:\n         Skip identification analysis based on minimal state space system.\n"
            else
                tot = totparam
                dminA = reshape(mS[:dA], minnx * minnx, tot)
                dminB = reshape(mS[:dB], minnx * nexo, tot)
                dminC = reshape(mS[:dC], obs_nbr * minnx, tot)
                dminD = reshape(mS[:dD], obs_nbr * nexo, tot)
                dvechSig = reshape(derivs[:dSigma_e], nexo * nexo, tot)
                ind = findall(!iszero, vec(Float64.(tril(ones(nexo, nexo)))))
                dvechSig = dvechSig[ind, :]
                Inx, Inu = Matrix{Float64}(I, minnx, minnx), Matrix{Float64}(I, nexo, nexo)
                Enu = _id_duplication(nexo)[2]
                mA, mB, mC, mD = mS[:A], mS[:B], mS[:C], mS[:D]
                nv = nexo * (nexo + 1) ÷ 2
                DL = vcat(dminA, dminB, dminC, dminD, dvechSig)
                DT = vcat(kron(mA', Inx) - kron(Inx, mA), kron(mB', Inx), -kron(Inx, mC),
                          zeros(obs_nbr * nexo, minnx^2), zeros(nv, minnx^2))
                DU = vcat(zeros(minnx^2, nexo^2), kron(Inu, mB), zeros(obs_nbr * minnx, nexo^2), kron(Inu, mD),
                          -2 * Enu * kron(Sigma, Inu))
                dM = hcat(DL, DT, DU)
                r[:dMINIMAL] = vcat(hcat(pr[:dE_y], zeros(obs_nbr, minnx^2 + nexo^2)), dM)
            end
        end
    end
    r
end

# ------------------------------------------------------------------------------------------------ options
const _ID_FLAG_OPTIONS = ("no_identification_strength", "no_identification_reducedform", "no_identification_moments",
                          "no_identification_minimal", "no_identification_spectrum")
const _ID_COSMETIC = Set(["nodisplay", "nograph", "graph_format", "tex", "TeX", "prior_trunc", "gsa_sample_file", "load_ident_files",
    "diffuse_filter", "lik_init", "schur_vec_tol", "kalman_algo", "kalman_tol", "diffuse_kalman_tol",
    "use_univariate_filters_if_singularity_is_detected", "heteroskedastic_filter", "analytic_derivation", "periods", "replic",
    "prior_range"])
const _ID_KNOWN = union(Set(["parameter_set", "ar", "useautocorr", "normalize_jacobians", "tol_rank", "tol_deriv", "tol_sv",
    "checks_via_subsets", "max_dim_subsets_groups", "max_dim_cova_group", "grid_nbr", "advanced", "prior_mc", "order",
    "analytic_derivation_mode"]), Set(_ID_FLAG_OPTIONS), _ID_COSMETIC)

_id_num(x) = x isa Real ? Float64(x) : x isa AbstractString ? parse(Float64, x) : x isa AbstractVector && length(x) == 1 ? _id_num(x[1]) : throw(IdentificationError("an identification option has the value $(repr(x)), a number is expected"))
_id_int(x) = Int(_id_num(x))

"`options_ident` as run.m completes it."
function _id_options(raw::AbstractDict, totparam_nbr::Int, prior_exist::Bool, default_mode::Int=0)
    unknown = sort!(String[k for k in keys(raw) if !(k in _ID_KNOWN)])
    isempty(unknown) || throw(IdentificationError("identification option(s) not supported: " * join(unknown, ", ")))
    o = Dict{String,Any}("gsa_sample_file" => 0, "parameter_set" => "prior_mean", "load_ident_files" => 0, "useautocorr" => 0, "ar" => 1,
        "prior_mc" => 1, "prior_range" => 0, "periods" => 300, "replic" => 100, "advanced" => 0, "normalize_jacobians" => 1,
        "grid_nbr" => 5000, "tol_rank" => "robust", "tol_deriv" => 1e-8, "tol_sv" => 1e-3, "schur_vec_tol" => 1e-11)
    for (k, v) in raw
        o[k] = v
    end
    for k in ("ar", "grid_nbr", "prior_mc", "advanced", "useautocorr", "normalize_jacobians", "load_ident_files", "gsa_sample_file", "prior_range")
        o[k] = o[k] isa Bool ? Int(o[k]) : _id_int(o[k])
    end
    for k in ("tol_deriv", "tol_sv")
        o[k] = _id_num(o[k])
    end
    (o["tol_rank"] isa AbstractString && o["tol_rank"] != "robust") && (o["tol_rank"] = _id_num(o["tol_rank"]))
    o["tol_rank"] isa Real && (o["tol_rank"] = Float64(o["tol_rank"]))
    isodd(o["grid_nbr"]) && (o["grid_nbr"] += 1)
    for f in _ID_FLAG_OPTIONS
        v = get(raw, f, nothing)
        o[f] = (haskey(raw, f) && !(v === false || v == 0)) ? 1 : 0
    end
    if o["no_identification_strength"] == 0 || o["advanced"] == 1
        o["no_identification_reducedform"] = 0
        o["no_identification_moments"] = 0
    end
    (o["gsa_sample_file"] != 0 || o["load_ident_files"] != 0) && throw(IdentificationError("identification: gsa_sample_file and load_ident_files are not supported"))
    get!(o, "lik_init", 1)
    (haskey(raw, "diffuse_filter") && raw["diffuse_filter"] != false && raw["diffuse_filter"] != 0) &&
        throw(IdentificationError("identification: the diffuse filter (unit roots) is not supported"))
    get!(o, "analytic_derivation", 1)
    o["order"] = _id_int(get(o, "order", 1))
    o["analytic_derivation_mode"] = _id_int(get(o, "analytic_derivation_mode", default_mode))
    if !prior_exist
        o["prior_mc"], o["prior_range"] = 1, 0
    end
    totparam_nbr < 2 && (o["advanced"] = 0)
    o["max_dim_cova_group"] = min(_id_int(get(o, "max_dim_cova_group", min(2, totparam_nbr - 1))), totparam_nbr - 1)
    o["checks_via_subsets"] = _id_int(get(o, "checks_via_subsets", 0))
    o["max_dim_subsets_groups"] = _id_int(get(o, "max_dim_subsets_groups", min(4, totparam_nbr - 1)))
    o
end

# ------------------------------------------------------------------------------------------------ the analysis context
mutable struct _IdCtx
    m::Model
    raw::Dict{String,Any}
    opts::Dict{String,Any}
    est::Union{Nothing,Estimation}
    prior_exist::Bool
    indpmodel::Vector{Int}
    indpstderr::Vector{Int}
    indpcorr::Vector{Tuple{Int,Int}}
    names::Vector{String}
    nsc::Int
    totparam_nbr::Int
    modparam_nbr::Int
    stderrparam_nbr::Int
    corrparam_nbr::Int
    order_var::Vector{Int}
    indvobs::Vector{Int}
    nstatic::Int
    nspred::Int
    rng::Any
    pers::Dict{Symbol,Any}
    gen::Any
    not_replayed::Bool
    messages_init::Vector{String}
end

function _id_context(m::Model, raw::AbstractDict; rng=nothing, no_priors::Bool=false)
    isempty(m.varobs) && throw(IdentificationError("identification: the model has no varobs statement (Dynare stops there too)"))
    est = nothing
    if !no_priors && any(st -> get(st, "statementName", "") == "estimated_params", m.statements)
        est = try
            Estimation(m; data=ones(1, length(m.varobs)), statement=0, allow_unsupported=true)
        catch e
            e isa EstimationError && throw(IdentificationError("identification: " * e.msg))
            rethrow()
        end
    end
    prior_exist = est !== nothing
    if prior_exist
        g = est.groups
        (isempty(g.vn) && isempty(g.cn)) || throw(IdentificationError("Identification does not (yet) support measurement errors. Instead, define them explicitly as varexo and provide measurement equations in the model definition."))
        indpmodel = Int[k for (_, k, _) in g.p]
        indpstderr = Int[k for (_, k, _) in g.vx]
        indpcorr = Tuple{Int,Int}[(k[1], k[2]) for (_, k, _) in g.cx]
        names = copy(est.names)
    else
        indpmodel = collect(1:length(m.param_names))
        indpstderr = collect(1:m.n_exo)
        indpcorr = Tuple{Int,Int}[]
        names = vcat(["SE_$e" for e in m.exo_names], m.param_names)
    end
    nsc = length(indpstderr) + length(indpcorr)
    tot = nsc + length(indpmodel)
    order_var, _, cnt = dr_ordering(m.lead_lag_incidence)
    inv = invperm(order_var)
    indvobs = Int[inv[m.ie[v]] for v in m.varobs]
    opts = _id_options(Dict{String,Any}(raw), tot, prior_exist)
    c = _IdCtx(m, Dict{String,Any}(raw), opts, est, prior_exist, indpmodel, indpstderr, indpcorr, names, nsc, tot, length(indpmodel),
               length(indpstderr), length(indpcorr), order_var, indvobs, cnt.nstatic, cnt.npred + cnt.nboth, rng,
               Dict{Symbol,Any}(), nothing, false, String[])
    _id_set_derivation_mode!(c, haskey(raw, "analytic_derivation_mode"))
    (prior_exist && !is_bayesian(est.prior)) && (c.opts["prior_mc"] = 1)
    c.opts["order"] == 1 || throw(IdentificationError("identification: orders 2 and 3 are not ported (order=1 only)"))
    c.opts["analytic_derivation_mode"] in (0, 1, -1, -2) || throw(IdentificationError("identification: analytic_derivation_mode must be 0, 1, -1 or -2"))
    c
end

"dynare_estimation_init.m: where the steady-state file changes parameter values (or leaves NaN) at the estimated parameters varied by 1 %, the mode becomes -2."
function _id_set_derivation_mode!(c::_IdCtx, mode_set::Bool)
    o = c.opts
    (o["analytic_derivation"] == 0 || o["order"] != 1) && return
    m = c.m
    if c.prior_exist
        isempty(c.indpmodel) && return
        pvec = model_params(c.est, c.est.x0)
        pvec[c.indpmodel] .*= 1.01
    else
        pvec = m.params .* 1.01
    end
    pp = try
        steady_state(m; params=pvec)[2]
    catch e
        (e isa SteadyStateError || e isa ErrorException) && return
        rethrow()
    end
    pp = Vector{Float64}(pp)
    if any(!=(0), pp .- pvec)
        if !mode_set
            o["analytic_derivation_mode"] = -2
            push!(c.messages_init, "The steadystate file changed the values of parameters or left some NaN: the derivatives of the Jacobian and steady state are computed numerically (analytic_derivation_mode = -2)")
        end
    end
    nothing
end

function _id_calibration(c::_IdCtx)
    m = c.m
    sd = _id_sqrt.(diag(m.Sigma_e))
    if !c.prior_exist
        return vcat(sd, m.params)
    end
    e = c.est
    vcat(sd[c.indpstderr], [e.corr0[a, b] for (a, b) in c.indpcorr], m.params[c.indpmodel])
end

function _id_point(c::_IdCtx)
    ps = string(c.opts["parameter_set"])
    c.prior_exist || return _id_calibration(c), "Current_params"
    e = c.est
    is_bayesian(e.prior) || return copy(e.x0), "ML_Starting_value"
    ps == "calibration" && return _id_calibration(c), "calibration"
    ps == "prior_mean" && return copy(e.prior.p1), "prior_mean"
    throw(IdentificationError("identification: parameter_set=$ps is not ported (calibration and prior_mean are)"))
end

# ------------------------------------------------------------------------------------------------ solution at a point
function _id_params_sigma(c::_IdCtx, theta)
    m = c.m
    if c.prior_exist
        return model_params(c.est, theta), covariances(c.est, theta)[1]
    end
    Vector{Float64}(theta[m.n_exo+1:end]), copy(m.Sigma_e)
end

"Steady state, first-order solution and parameter derivatives at `theta` (throws `IdentificationStop` where the model does not solve)."
function _id_solve(c::_IdCtx, theta)
    m = c.m
    o = c.opts
    pvec, Sigma = _id_params_sigma(c, theta)
    try
        ys, pp = steady_state(m; params=pvec)
        ys, pp = Vector{Float64}(ys), Vector{Float64}(pp)
        (all(isfinite, ys) && all(isfinite, pp)) || throw(SteadyStateError("steady state or parameters are not finite"))
        mode = o["analytic_derivation_mode"]
        mode == -1 && return _id_solve_fd(c, theta, ys, pp, pvec, Sigma)
        numerical = mode == -2
        if !numerical && any(i -> pp[i] != pvec[i] && !(isnan(pp[i]) && isnan(pvec[i])), eachindex(pp))
            throw(IdentificationError("identification: the steady-state file changes parameter values although analytic_derivation_mode is 0 or 1 (Dynare's analytic derivatives are then wrong); use analytic_derivation_mode=-2"))
        end
        d = _id_params_derivs(m, c.indpmodel, c.indpstderr, c.indpcorr, Sigma, ys, pp; numerical=numerical, kron=mode == 1)
        Dict{Symbol,Any}(:ys => ys, :pp => pp, :Sigma => Sigma, :derivs => d)
    catch e
        (e isa SteadyStateError || e isa BlanchardKahnError) && throw(IdentificationStop(sprint(showerror, e)))
        rethrow()
    end
end

struct _IdFDFailure <: Exception
    msg::String
end

"`compute_decision_rules` at `theta`: steady state, parameters, Sigma_e, ghx, ghu in decision-rule order."
function _id_fd_point(c::_IdCtx, theta)
    m = c.m
    if c.prior_exist
        pvec, Sigma = _id_params_sigma(c, theta)
    else
        n_exo = m.n_exo
        S0 = m.Sigma_e
        sd0 = _id_sqrt.(diag(S0))
        corr = S0 ./ (sd0 * sd0')
        corr[.!isfinite.(corr)] .= 0.0
        corr[diagind(corr)] .= 1.0
        sd = _id_sqrt.(theta[1:n_exo] .^ 2)
        Sigma = Diagonal(sd) * corr * Diagonal(sd)
        pvec = Vector{Float64}(theta[n_exo+1:end])
    end
    sol = try
        s = first_order(m; params=pvec, loglinear=false)
        (all(isfinite, s.ys) && all(isfinite, s.params)) || throw(SteadyStateError("steady state or parameters are not finite"))
        s
    catch e
        (e isa SteadyStateError || e isa BlanchardKahnError) && throw(_IdFDFailure(sprint(showerror, e)))
        rethrow()
    end
    Dict{Symbol,Any}(:ys => sol.ys, :pp => sol.params, :Sigma => Matrix{Float64}(Sigma), :ghx => ghx(sol), :ghu => ghu(sol))
end

"`identification.fjaco`: two-sided differences, step `dynatol.x * max(|x|, 1)`, `dynatol.x = 1e-5`."
function _id_fjaco(c::_IdCtx, f, x)
    x = Vector{Float64}(x)
    h = 1e-5 .* max.(abs.(x), 1.0)
    xh1, xh0 = x .+ h, x .- h
    hh = xh1 .- xh0
    cols = Vector{Any}(undef, length(x))
    for j in eachindex(x)
        vals = Any[]
        for xv in (xh1[j], xh0[j])
            xx = copy(x)
            xx[j] = xv
            try
                push!(vals, f(xx))
            catch e
                e isa _IdFDFailure || rethrow()
                throw(IdentificationError("fjaco.m: numerical two-sided difference method yields errors in solution algorithm (cannot solve the model for $(c.names[j]): $(e.msg))"))
            end
        end
        cols[j] = (vals[1] - vals[2]) / hh[j]
    end
    reduce(hcat, cols)
end

function _id_solve_fd(c::_IdCtx, theta, ys, pp, pvec, Sigma)
    m = c.m
    n, nexo, nspred = m.n_endo, m.n_exo, c.nspred
    ns, nco, nm = c.stderrparam_nbr, c.corrparam_nbr, c.modparam_nbr
    tot = c.totparam_nbr
    x0 = Vector{Float64}(theta)
    x0[c.nsc+1:end] = pp[c.indpmodel]
    base = try
        _id_fd_point(c, x0)
    catch e
        e isa _IdFDFailure && throw(IdentificationStop(e.msg))
        rethrow()
    end
    ghx_, ghu_ = base[:ghx], base[:ghu]
    Sig = base[:Sigma]
    fsol(x) = (r = _id_fd_point(c, x); vcat(vec(r[:Sigma]), vec(r[:ghx]), vec(r[:ghu])))
    dSG = _id_fjaco(c, fsol, x0)
    a = nexo * nexo
    dSigma = reshape(dSG[1:a, :], nexo, nexo, tot)
    dghx = reshape(dSG[a+1:a+n*nspred, :], n, nspred, tot)
    dghu = reshape(dSG[a+n*nspred+1:end, :], n, nexo, tot)
    dOm = zeros(n, n, tot)
    dCorr = zeros(nexo, nexo, tot)
    for jp in 1:ns+nco
        dOm[:, :, jp] = ghu_ * dSigma[:, :, jp] * ghu_'
    end
    for (jp, (i, k)) in enumerate(c.indpcorr)
        dCorr[i, k, ns+jp] = dCorr[k, i, ns+jp] = 1.0
    end
    for jp in 1:nm
        q = ns + nco + jp
        dOm[:, :, q] = dghu[:, :, q] * Sig * ghu_' + ghu_ * Sig * dghu[:, :, q]'
    end
    dys, dJ = _id_numerical_dys_dJ(m, c.indpmodel, pp, exo_steady(m))
    d = Dict{Symbol,Any}(:indpmodel => c.indpmodel, :indpstderr => c.indpstderr, :indpcorr => c.indpcorr,
        :dYss => dys[c.order_var, c.indpmodel], :dSigma_e => dSigma, :dCorrelation_matrix => dCorr, :dghx => dghx, :dghu => dghu,
        :dOm => dOm, :dg1 => dJ, :ghx => ghx_, :ghu => ghu_, :order_var => c.order_var, :state_var => Int[])
    Dict{Symbol,Any}(:ys => base[:ys], :pp => base[:pp], :Sigma => Sig, :derivs => d, :x0 => x0)
end

function _id_fd_for(c::_IdCtx, sol)
    c.opts["analytic_derivation_mode"] == -1 || return nothing
    (flag, nl, uac, gn) -> _id_fd_objective(c, sol[:x0], flag, nl, uac, gn)
end

"`identification.numerical_objective` differentiated by `fjaco`: the Jacobian of vech(Var_y) and the autocovariances (`flag` 1) or of the spectral density on the grid (`flag` 2)."
function _id_fd_objective(c::_IdCtx, x0, flag, nlags, useautocorr, grid_nbr)
    function f(x)
        r = _id_fd_point(c, x)
        Yss = r[:ys][c.order_var]
        pr = _id_pruned_first_order(r[:ghx], r[:ghu], r[:Sigma], Yss, c.indvobs, nlags, useautocorr, c.nstatic, c.nspred)
        if flag == 1
            E = useautocorr ? pr[:Corr_yi] : pr[:Var_yi]
            return vcat(_id_vech(pr[:Var_y]), reduce(vcat, [vec(E[:, :, i]) for i in 1:nlags]))
        end
        freqs = (0:grid_nbr÷2) .* (π / (grid_nbr / 2))
        A, B, C, D = pr[:A], pr[:B], pr[:C], pr[:D]
        IA = Matrix{Float64}(I, size(A, 1), size(A, 1))
        out = ComplexF64[]
        for w in freqs
            T = D + C * ((exp(1im * w) * IA - A) \ B)
            g = (1 / (2π)) * (T * pr[:Varinov] * T')
            append!(out, vec(g))
        end
        out
    end
    _id_fjaco(c, f, x0)
end

# ------------------------------------------------------------------------------------------------ analysis.m
"`find(max(abs(M'),[],1) > tol)`: rows of M with an element above tol."
function _id_rows_above(M::AbstractMatrix, tol)
    isempty(M) && return Int[]
    findall(r -> maximum(abs, view(M, r, :)) > tol, 1:size(M, 1))
end

function _id_norm_rows(M::AbstractMatrix)
    isempty(M) && return zeros(0, 0)
    repeat(maximum(abs.(M); dims=2), 1, size(M, 2))
end

"`[~,S,V] = svd(M,0)` padded as analysis.m does."
function _id_svd_SV(M, totparam_nbr)
    F = _id_svd(M; full=size(M, 1) < size(M, 2))
    V = F.V
    S = size(M, 1) < size(M, 2) ? vcat(F.S, zeros(size(M, 2) - size(M, 1))) : F.S
    if totparam_nbr > 8
        sel = vcat(1:4, length(S)-3:length(S))
        return S[sel], V[:, sel]
    end
    S, V
end

_id_merge!(dst::Dict{String,Any}, src::Dict{Symbol,Any}) = (for (k, v) in src; dst[string(k)] = v; end; dst)
_id_merge!(dst::Dict{String,Any}, src::Dict{String,Any}) = merge!(dst, src)

function _id_analysis(c::_IdCtx, theta; init::Bool=true)
    o = c.opts
    tot, modn = c.totparam_nbr, c.modparam_nbr
    tol_deriv, tol_rank, tol_sv = o["tol_deriv"], o["tol_rank"], o["tol_sv"]
    nlags = o["ar"]
    sol = _id_solve(c, theta)
    d = sol[:derivs]
    ghx_, ghu_ = d[:ghx], d[:ghu]
    fdf = _id_fd_for(c, sol)
    jac(opts) = _id_get_jacobians(c.m, ghx_, ghu_, sol[:ys], sol[:pp], sol[:Sigma], d, c.indvobs, c.nstatic, c.nspred, c.nsc, opts; fd=fdf)
    J = jac(o)
    err = Dict{String,Int}("identification_strength" => 0, "identification_reducedform" => 0, "identification_moments" => 0,
                           "identification_minimal" => 0, "identification_spectrum" => 0)
    messages = isempty(J[:minimal_message]) ? String[] : String[J[:minimal_message]]
    dMINIMAL = J[:dMINIMAL]
    dMINIMAL === nothing && (err["identification_minimal"] = 1)
    MOMENTS, dMOMENTS = J[:MOMENTS], J[:dMOMENTS]
    dSPECTRUM = J[:dSPECTRUM]
    ind_dMOMENTS = ind_dSPECTRUM = ind_dMINIMAL = ind_dREDUCEDFORM = ind_dDYNAMIC = nothing
    if !init
        ind_dMOMENTS, ind_dREDUCEDFORM, ind_dDYNAMIC = (get(c.pers, k, nothing) for k in (:moments, :reducedform, :dynamic))
        (o["no_identification_moments"] == 0 && ind_dMOMENTS === nothing) && throw(IdentificationError("identification: no stored indices of the first analysis"))
    end
    if init
        if o["no_identification_moments"] == 0
            any(isnan, MOMENTS) && throw(IdentificationError("There are NaN's in the theoretical moments."))
            ind_dMOMENTS = _id_rows_above(dMOMENTS, tol_deriv)
        end
        o["no_identification_spectrum"] == 0 && (ind_dSPECTRUM = _id_rows_above(dSPECTRUM, tol_deriv))
        (o["no_identification_minimal"] == 0 && dMINIMAL !== nothing) && (ind_dMINIMAL = _id_rows_above(dMINIMAL, tol_deriv))
        if err["identification_moments"] != 0 && err["identification_minimal"] != 0 && err["identification_spectrum"] != 0
            throw(IdentificationError("identification_analyis: Stationarity condition(s) failed"))
        end
        if o["no_identification_moments"] == 0 && err["identification_moments"] == 0
            opts_local = copy(o)
            while length(ind_dMOMENTS) < tot && nlags < 10
                push!(messages, "The number of moments with non-zero derivative is smaller than the number of parameters\nTry increasing ar = $(nlags + 1)")
                nlags += 1
                opts_local["no_identification_minimal"] = 1
                opts_local["no_identification_spectrum"] = 1
                opts_local["ar"] = nlags
                Jl = jac(opts_local)
                MOMENTS, dMOMENTS = Jl[:MOMENTS], Jl[:dMOMENTS]
                ind_dMOMENTS = _id_rows_above(dMOMENTS, tol_deriv)
            end
            if length(ind_dMOMENTS) < tot
                push!(messages, "WARNING: Order condition for dMOMENTS failed: There are not enough moments and too many parameters.\n         Skip identification analysis based on moments.\n         Skip identification strength analysis.")
                err["identification_moments"] = 1
                err["identification_strength"] = 1
            end
        end
        if o["no_identification_minimal"] == 0 && err["identification_minimal"] == 0
            if length(ind_dMINIMAL) < size(dMINIMAL, 2)
                push!(messages, "WARNING: Order condition for dMINIMAL failed: There are too many parameters or too few observable variables.\n         Skip identification analysis based on minimal state space system.")
                err["identification_minimal"] = 1
            end
        end
        if err["identification_moments"] != 0 && err["identification_minimal"] != 0 && err["identification_spectrum"] != 0
            throw(IdentificationError("identification_analyis: Order condition(s) failed"))
        end
    end
    dREDUCEDFORM, dDYNAMIC = J[:dREDUCEDFORM], J[:dDYNAMIC]
    rf_on = o["no_identification_reducedform"] == 0 && err["identification_reducedform"] == 0
    if init
        rf_on && (ind_dREDUCEDFORM = _id_rows_above(dREDUCEDFORM, tol_deriv))
        ind_dDYNAMIC = _id_rows_above(dDYNAMIC, tol_deriv)
    end
    DYNAMIC = J[:DYNAMIC][ind_dDYNAMIC]
    si_dDYNAMIC = dDYNAMIC[ind_dDYNAMIC, :]
    REDUCEDFORM, si_dREDUCEDFORM = J[:REDUCEDFORM], nothing
    if rf_on
        REDUCEDFORM = REDUCEDFORM[ind_dREDUCEDFORM]
        si_dREDUCEDFORM = dREDUCEDFORM[ind_dREDUCEDFORM, :]
    end
    ide_hess, ide_dynamic, ide_moments, ide_reducedform = (Dict{String,Any}() for _ in 1:4)
    ide_spectrum, ide_minimal = Dict{String,Any}(), Dict{String,Any}()
    mom_on = o["no_identification_moments"] == 0 && err["identification_moments"] == 0
    local si_dMOMENTS
    if mom_on
        MOMENTS = MOMENTS[ind_dMOMENTS]
        si_dMOMENTS = dMOMENTS[ind_dMOMENTS, :]
        if o["no_identification_strength"] == 0 && err["identification_strength"] == 0 && init
            st = _id_strength(c, theta, sol, J, si_dMOMENTS, si_dREDUCEDFORM, si_dDYNAMIC, ind_dMOMENTS, ind_dREDUCEDFORM, ind_dDYNAMIC, nlags)
            merge!(ide_hess, st[:ide_hess])
            ind_dREDUCEDFORM, si_dREDUCEDFORM = st[:ind_dREDUCEDFORM], st[:si_dREDUCEDFORM]
            ind_dDYNAMIC, si_dDYNAMIC = st[:ind_dDYNAMIC], st[:si_dDYNAMIC]
            ide_dynamic["si_dDYNAMICnorm"] = st[:si_dDYNAMICnorm]
            ide_moments["si_dMOMENTSnorm"] = st[:si_dMOMENTSnorm]
            ide_reducedform["si_dREDUCEDFORMnorm"] = st[:si_dREDUCEDFORMnorm]
        end
    end
    init && (c.pers = Dict{Symbol,Any}(:moments => ind_dMOMENTS, :reducedform => ind_dREDUCEDFORM, :dynamic => ind_dDYNAMIC))
    norm = o["normalize_jacobians"] != 0
    norm_dDYNAMIC = norm ? _id_norm_rows(si_dDYNAMIC) : 1
    merge!(ide_dynamic, Dict{String,Any}("ind_dDYNAMIC" => ind_dDYNAMIC, "norm_dDYNAMIC" => norm_dDYNAMIC, "si_dDYNAMIC" => si_dDYNAMIC,
        "dDYNAMIC" => dDYNAMIC, "DYNAMIC" => DYNAMIC))
    if rf_on
        norm_dRF = norm ? _id_norm_rows(si_dREDUCEDFORM) : 1
        merge!(ide_reducedform, Dict{String,Any}("ind_dREDUCEDFORM" => ind_dREDUCEDFORM, "norm_dREDUCEDFORM" => norm_dRF,
            "si_dREDUCEDFORM" => si_dREDUCEDFORM, "dREDUCEDFORM" => dREDUCEDFORM, "REDUCEDFORM" => REDUCEDFORM))
    end
    if mom_on
        norm_dMOM = norm ? _id_norm_rows(si_dMOMENTS) : 1
        merge!(ide_moments, Dict{String,Any}("ind_dMOMENTS" => ind_dMOMENTS, "norm_dMOMENTS" => norm_dMOM, "si_dMOMENTS" => si_dMOMENTS,
            "dMOMENTS" => dMOMENTS, "MOMENTS" => MOMENTS))
        if o["advanced"] != 0
            ide_moments["pars"], ide_moments["cosndMOMENTS"] = _id_bruteforce(dMOMENTS[ind_dMOMENTS, :], o["max_dim_cova_group"], tol_deriv)
        end
        S, V = _id_svd_SV(dMOMENTS[ind_dMOMENTS, :], tot)
        ide_moments["S"], ide_moments["V"] = S, V
    end
    if o["no_identification_minimal"] == 0 && err["identification_minimal"] == 0
        if norm
            ind_dMINIMAL = _id_rows_above(dMINIMAL, tol_deriv)
            norm_dMIN = _id_norm_rows(dMINIMAL[ind_dMINIMAL, :])
        else
            norm_dMIN = 1
        end
        merge!(ide_minimal, Dict{String,Any}("ind_dMINIMAL" => ind_dMINIMAL, "norm_dMINIMAL" => norm_dMIN, "dMINIMAL" => dMINIMAL))
    end
    if o["no_identification_spectrum"] == 0 && err["identification_spectrum"] == 0
        if norm
            ind_dSPECTRUM = _id_rows_above(dSPECTRUM, tol_deriv)
            tilda = zeros(size(dSPECTRUM))
            sub = dSPECTRUM[ind_dSPECTRUM, ind_dSPECTRUM]
            delta = _id_sqrt.(diag(sub))
            tilda[ind_dSPECTRUM, ind_dSPECTRUM] = sub ./ (delta * delta')
            norm_dSPEC = _id_norm_rows(dSPECTRUM[ind_dSPECTRUM, :])
        else
            tilda, norm_dSPEC = dSPECTRUM, 1
        end
        merge!(ide_spectrum, Dict{String,Any}("ind_dSPECTRUM" => ind_dSPECTRUM, "norm_dSPECTRUM" => norm_dSPEC, "tilda_dSPECTRUM" => tilda,
            "dSPECTRUM" => dSPECTRUM, "dSPECTRUM_NO_MEAN" => J[:dSPECTRUM_NO_MEAN]))
    end
    nd(a) = norm ? a ./ norm_dDYNAMIC : a
    if o["checks_via_subsets"] != 0
        _id_checks_via_subsets!(c, dDYNAMIC, ide_dynamic, ide_reducedform, ide_moments, ide_spectrum, ide_minimal, err)
        ide_minimal["minimal_state_space"] = err["identification_minimal"] != 0 ? 0.0 : 1.0
    else
        _id_merge!(ide_dynamic, _id_checks(nd(dDYNAMIC[ind_dDYNAMIC, :]), 1, tol_rank, tol_sv, modn))
        rf_on && _id_merge!(ide_reducedform, _id_checks(dREDUCEDFORM[ind_dREDUCEDFORM, :] ./ ide_reducedform["norm_dREDUCEDFORM"], 1, tol_rank, tol_sv, tot))
        mom_on && _id_merge!(ide_moments, _id_checks(dMOMENTS[ind_dMOMENTS, :] ./ ide_moments["norm_dMOMENTS"], 1, tol_rank, tol_sv, tot))
        if o["no_identification_minimal"] == 0
            if err["identification_minimal"] == 0
                _id_merge!(ide_minimal, _id_checks(dMINIMAL[ind_dMINIMAL, :] ./ ide_minimal["norm_dMINIMAL"], 2, tol_rank, tol_sv, tot))
                ide_minimal["minimal_state_space"] = 1.0
            else
                ide_minimal["minimal_state_space"] = 0.0
            end
        end
        if o["no_identification_spectrum"] == 0 && err["identification_spectrum"] == 0
            _id_merge!(ide_spectrum, _id_checks(ide_spectrum["tilda_dSPECTRUM"], 3, tol_rank, tol_sv, tot))
        end
    end
    Dict{Symbol,Any}(:ide_moments => ide_moments, :ide_spectrum => ide_spectrum, :ide_minimal => ide_minimal, :ide_hess => ide_hess,
        :ide_reducedform => ide_reducedform, :ide_dynamic => ide_dynamic, :error_indicator => err, :messages => messages, :ar => nlags)
end

# ------------------------------------------------------------------------------------------------ simulation (strength)
function _id_state_var(c::_IdCtx)
    _, sv, _ = dr_ordering(c.m.lead_lag_incidence)
    sv
end

function _id_dr(c::_IdCtx, sol)
    d = sol[:derivs]
    Dict{String,Any}("ys" => sol[:ys], "order_var" => c.order_var, "state_var" => _id_state_var(c), "ghx" => d[:ghx], "ghu" => d[:ghu], "order" => 1)
end

"`stoch_simul(M_, options_, oo_, varobs)` with `periods + 100` periods: the draws and the observables of the last `periods` periods."
function _id_simulate_data(c::_IdCtx, sol, periods)
    c.rng === nothing && throw(IdentificationError("identification: the identification strength measure draws random numbers; no random stream was given"))
    m = c.m
    n_sim = periods + 100
    Sigma = sol[:Sigma]
    active = findall(!iszero, diag(Sigma))
    shocks = zeros(n_sim, m.n_exo)
    if !isempty(active)
        R = Matrix(cholesky(Symmetric(Sigma[active, active])).U)
        shocks[:, active] = permutedims(randn(c.rng, length(active), n_sim)) * R
    end
    y = simulate(_id_dr(c, sol), shocks; order=1, pruning=false, y0=sol[:ys])
    y[101:end, Int[m.ie[v] for v in m.varobs]]
end

"`identification.simulated_moment_uncertainty`: covariance matrix of the selected moments over `replic` simulations."
function _id_simulated_moment_uncertainty(c::_IdCtx, sol, indx, periods, replic, nlags)
    m = c.m
    Sigma = sol[:Sigma]
    active = findall(!iszero, diag(Sigma))
    R = Matrix(cholesky(Symmetric(Sigma[active, active])).U)
    dr = _id_dr(c, sol)
    ivar = Int[m.ie[v] for v in m.varobs]
    mm = zeros(length(indx), replic)
    for j in 1:replic
        shocks = zeros(periods, m.n_exo)
        shocks[:, active] = permutedims(randn(c.rng, length(active), periods)) * R
        y = simulate(dr, shocks; order=1, pruning=false, y0=sol[:ys])[101:end, ivar]
        N = size(y, 1)
        mean_ = vec(sum(y; dims=1)) ./ N
        y = y .- mean_'
        var = y' * y / N
        s = vec(_id_sqrt.(sum(abs2, y; dims=1) ./ N))
        zero_ = abs.(s) .< 1e-10
        var[zero_, :] .= 0
        var[:, zero_] .= 0
        dum = Any[mean_, _id_vech(var)]
        sdv = _id_sqrt.(diag(var))
        for i in 1:nlags
            a0, a1 = y[nlags+1:end, :], y[nlags-i+1:N-i, :]
            ac = a0' * a1 ./ ((N - nlags) .* (vec(_id_std(a0; dims=1)) * vec(_id_std(a1; dims=1))'))
            ac[zero_, :] .= NaN
            ac[:, zero_] .= NaN
            push!(dum, vec(ac .* (sdv * sdv')))
        end
        mm[:, j] = reduce(vcat, dum)[indx]
    end
    _id_cov_rows(mm)
end

function _id_try_asymptotic_hessian(c::_IdCtx, theta, sol, data)
    m = c.m
    o = c.opts
    c.prior_exist || return nothing
    est = c.est
    is_bayesian(est.prior) || return nothing
    lb, ub = prior_bounds(est.prior, Float64(_id_num(get(c.raw, "prior_trunc", 1e-10))))
    (any(theta .< lb) || any(theta .> ub)) && return nothing
    Sigma = sol[:Sigma]
    (!isempty(est.groups.cx) && !_ispd(Sigma[est.Q_check, est.Q_check])) && return nothing
    d = sol[:derivs]
    n = m.n_endo
    iv = sort(union(c.indvobs, c.nstatic .+ (1:c.nspred)))
    nst, nsp = c.nstatic, c.nspred
    Tfull = zeros(n, n)
    Tfull[:, nst+1:nst+nsp] = d[:ghx]
    T = Tfull[iv, iv]
    R = d[:ghu][iv, :]
    Z = Int[findfirst(==(i), iv) for i in c.indvobs]
    tot = c.totparam_nbr
    DTf = zeros(n, n, tot)
    DTf[:, nst+1:nst+nsp, :] = d[:dghx]
    DT = DTf[iv, iv, :]
    DOm = d[:dOm][iv, iv, :]
    DYss = hcat(zeros(length(iv), c.nsc), d[:dYss][iv, :])
    Pstar = lyapunov_symm(T, R * Sigma * R')[1]
    Yss_obs = sol[:ys][Int[m.ie[v] for v in m.varobs]]
    Y = permutedims(data) .- Yss_obs
    local Hess
    try
        Hess = _id_asymptotic_hessian(Y, T, R, Sigma, Z, DT, DYss, DOm, Pstar)
    catch e
        e isa _IdSingularF || rethrow()
        uf = get(c.raw, "use_univariate_filters_if_singularity_is_detected", 1)
        (uf == false || uf == 0) && throw(IdentificationError("identification: singular prediction error covariance with use_univariate_filters_if_singularity_is_detected=0 (not ported)"))
        try
            Hess = _id_univariate_asymptotic_hessian(Y, T, R, Sigma, Z, DT, DYss, DOm, Pstar)
        catch e2
            e2 isa _IdNegativeVariance && return nothing
            rethrow()
        end
    end
    -Hess
end

function _id_strength(c::_IdCtx, theta, sol, J, si_dMOMENTS, si_dREDUCEDFORM, si_dDYNAMIC, ind_dMOMENTS, ind_dREDUCEDFORM, ind_dDYNAMIC, nlags)
    o = c.opts
    tot, sc = c.totparam_nbr, c.nsc
    tol_rank, tol_deriv = o["tol_rank"], o["tol_deriv"]
    params = Vector{Float64}(theta)
    normaliz_prior_std = c.prior_exist ? Vector{Float64}(c.est.prior.p2) : fill(NaN, tot)
    data = _id_simulate_data(c, sol, o["periods"])
    AHess = _id_try_asymptotic_hessian(c, theta, sol, data)
    ide_hess = Dict{String,Any}()
    if AHess !== nothing && !(tol_rank isa AbstractString) && minimum(eigvals(Symmetric((AHess + AHess') / 2))) < -tol_rank
        AHess = nothing
    end
    uncert = fill(NaN, tot)
    local flag_score, cmm, diag_chh, cdynamic, indok, deltaM
    if AHess !== nothing
        flag_score = 1
        ide_hess["AHess"] = AHess
        deltaM = _id_sqrt.(diag(AHess))
        iflag = any(deltaM .* deltaM .== 0)
        tildaM = AHess ./ (deltaM * deltaM')
        use = (iflag || _id_rank(AHess) > _id_rank(tildaM)) ? AHess : tildaM
        ch = _id_checks(use, 0, tol_rank, o["tol_sv"], tot)
        for k in (:cond, :rank, :ind0, :indno, :ino, :Mco, :Pco)
            ide_hess[string(k)] = ch[k]
        end
        indok = findall(==(0), vec(maximum(ch[:indno]; dims=1)))
        uncert[indok] = _id_sqrt.(diag(_id_inv(AHess[indok, indok])))
        ind1 = findall(!iszero, ch[:ind0])
        sol_ = _id_mldivide(AHess[ind1, ind1], si_dMOMENTS[:, ind1]')
        cmm = si_dMOMENTS[:, ind1] * sol_
        temp1 = _id_mldivide(AHess[ind1, ind1], si_dREDUCEDFORM[:, ind1]')
        diag_chh = vec(sum(si_dREDUCEDFORM[:, ind1]' .* temp1; dims=1))
        ind1 = filter(>(sc), ind1)
        sd_ = si_dDYNAMIC[:, ind1 .- sc]
        cdynamic = sd_ * _id_mldivide(AHess[ind1, ind1], sd_')
    else
        flag_score = 0
        replic = max(Int(o["replic"]), length(ind_dMOMENTS) * 3)
        cmm = _id_simulated_moment_uncertainty(c, sol, ind_dMOMENTS, o["periods"], replic, nlags)
        sd = _id_sqrt.(diag(cmm))
        cc = cmm ./ (sd * sd')
        DD, VV = eigen(Symmetric((cc + cc') / 2))
        idd = findall(>(tol_deriv), DD)
        siTMP = si_dMOMENTS ./ sd
        MIM = (siTMP' * VV[:, idd]) * Diagonal(1 ./ DD[idd]) * (VV[:, idd]' * siTMP)
        ide_hess["AHess"] = MIM
        deltaM = _id_sqrt.(diag(MIM))
        iflag = any(deltaM .* deltaM .== 0)
        tildaM = MIM ./ (deltaM * deltaM')
        use = (iflag || _id_rank(MIM) > _id_rank(tildaM)) ? MIM : tildaM
        ch = _id_checks(use, 0, tol_rank, o["tol_sv"], tot)
        for k in (:cond, :rank, :ind0, :indno, :ino, :Mco, :Pco)
            ide_hess[string(k)] = ch[k]
        end
        indok = findall(==(0), vec(maximum(ch[:indno]; dims=1)))
        ind1 = findall(!iszero, ch[:ind0])
        temp1 = _id_mldivide(MIM[ind1, ind1], si_dREDUCEDFORM[:, ind1]')
        diag_chh = vec(sum(si_dREDUCEDFORM[:, ind1]' .* temp1; dims=1))
        ind1 = filter(>(sc), ind1)
        sd_ = si_dDYNAMIC[:, ind1 .- sc]
        cdynamic = sd_ * _id_mldivide(MIM[ind1, ind1], sd_')
        isempty(indok) || (uncert[indok] = _id_sqrt.(diag(_id_inv(tildaM[indok, indok]))) ./ deltaM[indok])
    end
    strength = fill(NaN, tot)
    strength_prior = fill(NaN, tot)
    strength[indok] = 1.0 ./ (uncert[indok] ./ abs.(params[indok]))
    strength_prior[indok] = 1.0 ./ (uncert[indok] ./ normaliz_prior_std[indok])
    sens_zero = findall(isinf, deltaM)
    deltaM_prior = deltaM .* abs.(normaliz_prior_std)
    deltaM = deltaM .* abs.(params)
    quant = si_dMOMENTS ./ _id_sqrt.(diag(cmm))
    si_dMOMENTSnorm = size(quant, 1) == 1 ? abs.(quant) .* normaliz_prior_std' : (_id_vnorm(quant) .* normaliz_prior_std)'
    iy = findall(!iszero, diag_chh)
    ind_dREDUCEDFORM = ind_dREDUCEDFORM[iy]
    si_dREDUCEDFORM = si_dREDUCEDFORM[iy, :]
    if !isempty(iy)
        quant = si_dREDUCEDFORM ./ _id_sqrt.(diag_chh[iy])
        si_dREDUCEDFORMnorm = size(quant, 1) == 1 ? abs.(quant) .* normaliz_prior_std' : (_id_vnorm(quant) .* normaliz_prior_std)'
    else
        si_dREDUCEDFORMnorm = zeros(0, 0)
    end
    diag_cd = diag(cdynamic)
    iy = findall(!iszero, diag_cd)
    ind_dDYNAMIC = ind_dDYNAMIC[iy]
    si_dDYNAMIC = si_dDYNAMIC[iy, :]
    if !isempty(iy)
        quant = si_dDYNAMIC ./ _id_sqrt.(diag_cd[iy])
        nps = normaliz_prior_std[sc+1:end]
        si_dDYNAMICnorm = size(quant, 1) == 1 ? abs.(quant) .* nps' : (_id_vnorm(quant) .* nps)'
    else
        si_dDYNAMICnorm = zeros(0, 0)
    end
    merge!(ide_hess, Dict{String,Any}("ide_strength_dMOMENTS" => strength, "ide_strength_dMOMENTS_prior" => strength_prior,
        "deltaM" => deltaM, "deltaM_prior" => deltaM_prior,
        "sensitivity_zero_pos" => isempty(sens_zero) ? zeros(0, 1) : Float64.(sens_zero),
        "identified_parameter_indices" => isempty(indok) ? zeros(1, 0) : Float64.(indok), "flag_score" => Float64(flag_score)))
    Dict{Symbol,Any}(:ide_hess => ide_hess, :ind_dREDUCEDFORM => ind_dREDUCEDFORM, :si_dREDUCEDFORM => si_dREDUCEDFORM,
        :ind_dDYNAMIC => ind_dDYNAMIC, :si_dDYNAMIC => si_dDYNAMIC, :si_dDYNAMICnorm => si_dDYNAMICnorm,
        :si_dMOMENTSnorm => si_dMOMENTSnorm, :si_dREDUCEDFORMnorm => si_dREDUCEDFORMnorm)
end

# ------------------------------------------------------------------------------------------------ bruteforce, subsets
function _id_combinations(v::AbstractVector, k::Int)
    n = length(v)
    out = Vector{Vector{eltype(v)}}()
    k > n && return out
    k == 0 && return [eltype(v)[]]
    idx = collect(1:k)
    while true
        push!(out, v[idx])
        i = k
        while i >= 1 && idx[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for j in i+1:k
            idx[j] = idx[j-1] + 1
        end
    end
    out
end

"`identification.bruteforce`: for each parameter the group of up to `max_dim_cova_group` others whose columns best reproduce its column."
function _id_bruteforce(J, max_dim_cova_group, tol_deriv)
    tot = size(J, 2)
    cosnJ = zeros(tot, max_dim_cova_group)
    pars = Matrix{Any}(undef, tot, max_dim_cova_group)
    fill!(pars, zeros(0, 0))
    for ll in 1:max_dim_cova_group
        for ii in 1:tot
            tmp = [k for k in 1:tot if k != ii]
            combos = _id_combinations(tmp, ll)
            c2 = [_id_cosn(hcat(J[:, [ii]], J[:, cm])) for cm in combos]
            cosnJ[ii, ll] = maximum(c2)
            if cosnJ[ii, ll] > tol_deriv
                if ll > 1 && (cosnJ[ii, ll] - cosnJ[ii, ll-1]) < tol_deriv
                    pars[ii, ll] = reshape(vcat(vec(pars[ii, ll-1]), NaN), 1, :)
                    cosnJ[ii, ll] = cosnJ[ii, ll-1]
                else
                    k = findfirst(==(maximum(c2)), c2)
                    pars[ii, ll] = reshape(Float64.(combos[k]), 1, :)
                end
            else
                pars[ii, ll] = fill(NaN, 1, ll)
            end
        end
    end
    pars, cosnJ
end

"`RemoveProblematicParameterSets`: drop the subsets that contain an already known problematic set."
function _id_remove_problematic(idx, problparset)
    isempty(idx) && return idx
    iset = length(idx[1])
    for iii in 1:iset-1
        sets = problparset[iii]
        (sets === nothing || isempty(sets)) && continue
        if iii == 1
            sets = [[v] for v in sets]
        end
        for st in sets
            ss = Set(st)
            idx = [r for r in idx if count(in(ss), r) != iii]
        end
    end
    idx
end

function _id_checks_via_subsets!(c::_IdCtx, dDYNAMIC, ide_dynamic, ide_reducedform, ide_moments, ide_spectrum, ide_minimal, err)
    o = c.opts
    tot, modn = c.totparam_nbr, c.modparam_nbr
    tol_rank = o["tol_rank"]
    maxdim = o["max_dim_subsets_groups"]
    rk(X) = tol_rank isa AbstractString ? _id_rank(X) : _id_rank(X, tol_rank)
    # spectrum: Dynare compares abs(dSPECTRUM(j,j)) with tol_rank without the 'robust' test (a char vector then)
    small(x) = tol_rank isa AbstractString ? all(x .< Float64.(collect(codeunits(tol_rank)))) : x < tol_rank
    use = Dict("dynamic" => true, "reducedform" => o["no_identification_reducedform"] == 0 && err["identification_reducedform"] == 0,
               "moments" => o["no_identification_moments"] == 0 && err["identification_moments"] == 0,
               "spectrum" => o["no_identification_spectrum"] == 0 && err["identification_spectrum"] == 0,
               "minimal" => o["no_identification_minimal"] == 0 && err["identification_minimal"] == 0)
    order = ["dynamic", "reducedform", "moments", "spectrum", "minimal"]
    mats = Dict{String,Matrix{Float64}}()
    need = Dict{String,Int}()
    fixed = 0
    ind = ide_dynamic["ind_dDYNAMIC"]
    D = copy(dDYNAMIC)
    D[ind, :] = D[ind, :] ./ ide_dynamic["norm_dDYNAMIC"]
    tot > modn && (D = hcat(zeros(size(D, 1), tot - modn), D))
    mats["dynamic"], need["dynamic"] = D, tot
    if use["reducedform"]
        D = copy(ide_reducedform["dREDUCEDFORM"])
        ind = ide_reducedform["ind_dREDUCEDFORM"]
        D[ind, :] = D[ind, :] ./ ide_reducedform["norm_dREDUCEDFORM"]
        mats["reducedform"], need["reducedform"] = D, tot
    end
    if use["moments"]
        D = copy(ide_moments["dMOMENTS"])
        ind = ide_moments["ind_dMOMENTS"]
        D[ind, :] = D[ind, :] ./ ide_moments["norm_dMOMENTS"]
        mats["moments"], need["moments"] = D, tot
    end
    if use["spectrum"]
        mats["spectrum"], need["spectrum"] = ide_spectrum["tilda_dSPECTRUM"], tot
    end
    if use["minimal"]
        D = copy(ide_minimal["dMINIMAL"])
        ind = ide_minimal["ind_dMINIMAL"]
        D[ind, :] = D[ind, :] ./ ide_minimal["norm_dMINIMAL"]
        mats["minimal"] = D
        fixed = size(D, 2) - tot
        need["minimal"] = tot + fixed
    end
    ides = Dict("dynamic" => ide_dynamic, "reducedform" => ide_reducedform, "moments" => ide_moments, "spectrum" => ide_spectrum,
                "minimal" => ide_minimal)
    ks = [k for k in order if haskey(mats, k)]
    probl = Dict{String,Vector{Any}}(k => Any[nothing for _ in 1:maxdim] for k in ks)
    for k in ks
        probl[k][1] = Int[]
    end
    active, indparam, ranks = Dict{String,Bool}(), Dict{String,Vector{Int}}(), Dict{String,Int}()
    for k in ks
        ranks[k] = rk(mats[k])
        ides[k]["rank"] = Float64(ranks[k])
        active[k] = ranks[k] != need[k]
        indparam[k] = active[k] ? collect(1:tot) : Int[]
    end
    for j in 1:tot
        for k in ks
            active[k] || continue
            D = mats[k]
            if k == "spectrum"
                small(abs(D[j, j])) && push!(probl[k][1], j)
            elseif k == "minimal"
                par, rest = D[:, 1:tot], D[:, tot+1:end]
                rk(hcat(par[:, [j]], rest)) == fixed && push!(probl[k][1], j)
            else
                rk(D[:, [j]]) == 0 && push!(probl[k][1], j)
            end
        end
    end
    for k in ks
        active[k] || continue
        if length(probl[k][1]) == need[k] - ranks[k]
            active[k] = false
        else
            indparam[k] = [x for x in indparam[k] if !(x in Set(probl[k][1]))]
        end
    end
    indtot = sort!(unique(reduce(vcat, [indparam[k] for k in ks]; init=Int[])))
    for j in 2:min(length(indtot), maxdim)
        indexj = any(values(active)) ? _id_combinations(indtot, j) : Vector{Int}[]
        for k in ks
            active[k] || continue
            D = mats[k]
            idx = _id_remove_problematic(indexj, probl[k])
            rj = zeros(length(idx))
            for (r, row) in enumerate(idx)
                rj[r] = k == "spectrum" ? rk(D[row, row]) : k == "minimal" ? rk(hcat(D[:, 1:tot][:, row], D[:, tot+1:end])) : rk(D[:, row])
            end
            thr = j + (k == "minimal" ? fixed : 0)
            probl[k][j] = [row for (row, rv) in zip(idx, rj) if rv < thr]
        end
    end
    probl["dynamic"][1] = [x for x in probl["dynamic"][1] if x > tot - modn]
    for k in order
        cell = Matrix{Any}(undef, 1, maxdim)
        if !haskey(mats, k)
            fill!(cell, zeros(0, 0))
            ides[k]["problpars"] = cell
            continue
        end
        for i in 1:maxdim
            v = probl[k][i]
            if i == 1
                cell[1, i] = isempty(v) ? zeros(0, 0) : reshape(Float64.(v), :, 1)
            elseif v === nothing
                cell[1, i] = zeros(0, 0)
            else
                cell[1, i] = isempty(v) ? zeros(0, i) : permutedims(reduce(hcat, [Float64.(r) for r in v]))
            end
        end
        ides[k]["problpars"] = cell
    end
    nothing
end

# ------------------------------------------------------------------------------------------------ prior draws
function _id_prior_draw(c::_IdCtx)
    c.rng === nothing && throw(IdentificationError("identification: prior draws need a random stream"))
    e = c.est
    lb, ub = prior_bounds(e.prior, Float64(_id_num(get(c.raw, "prior_trunc", 1e-10))))
    pr = e.prior
    if c.opts["prior_range"] != 0
        (all(isfinite, lb) && all(isfinite, ub)) || throw(IdentificationError("identification: prior_range needs finite prior bounds"))
        pr = Prior(fill(5, length(pr.shape)), copy(pr.p1), copy(pr.p2), copy(lb), copy(ub), pr.p6, pr.p7, pr.names)
    end
    any(in((1, 2, 4, 6, 8)), pr.shape) && (c.not_replayed = true)
    prior_draw(pr, lb, ub, c.rng)
end

# ------------------------------------------------------------------------------------------------ Monte Carlo
function _id_ide_store(ide, S, ncol, ind)
    Dict{String,Any}(ind => ide[ind], "ino" => zeros(S, ind == "ind_dDYNAMIC" ? ncol : 1), "ind0" => zeros(S, ncol), "jweak" => zeros(S, ncol),
        "jweak_pair" => zeros(S, ncol * (ncol + 1) ÷ 2), "cond" => zeros(S, 1), "Mco" => zeros(S, ncol))
end

function _id_ide_put!(store, it, ide)
    store["cond"][it, 1] = ide["cond"]
    store["ino"][it, 1] = ide["ino"]
    store["ind0"][it, :] = vec(ide["ind0"])
    store["jweak"][it, :] = vec(ide["jweak"])
    store["jweak_pair"][it, :] = vec(ide["jweak_pair"])
    store["Mco"][it, :] = vec(ide["Mco"])
end

function _id_grow(a, n)
    size(a, 1) >= n && return a
    vcat(a, zeros(n - size(a, 1), size(a, 2)))
end

"`identification.vnorm(A)` exactly: the norm runs along the first non-singleton dimension (a single row gives one number)."
function _id_vnorm_matlab(A::AbstractMatrix)
    size(A, 1) == 1 && size(A, 2) > 1 && return fill(sqrt(sum(abs2, A)), 1)
    _id_vnorm(A)
end

function _id_std0(a, dims)
    size(a, dims) > 1 ? vec(_id_std(a; dims=dims)) : zeros(size(a, 3 - dims))
end

function _id_monte_carlo!(c::_IdCtx, out, point_res)
    o = c.opts
    S = o["prior_mc"]
    tot, modn = c.totparam_nbr, c.modparam_nbr
    o["checks_via_subsets"] != 0 && throw(IdentificationError("identification: prior_mc > 1 with checks_via_subsets=1 (Dynare's own Monte Carlo loop stops there, as checks_via_subsets does not return cond, ind0, ...)"))
    o["no_identification_reducedform"] != 0 && throw(IdentificationError("identification: prior_mc > 1 with no_identification_reducedform (Dynare's Monte Carlo loop needs the reduced form criterion)"))
    opts_ident = o
    opts_mc = copy(o); opts_mc["advanced"] = 0
    no_mom, no_spec, no_min = opts_mc["no_identification_moments"] != 0, opts_mc["no_identification_spectrum"] != 0, opts_mc["no_identification_minimal"] != 0
    opts_rerun = copy(opts_ident); opts_rerun["checks_via_subsets"] = 0
    iteration = run_index = 0
    pdraws = zeros(S, tot)
    sto = Dict{String,Any}()
    files = Dict{String,Any}[]
    max_runs = 0
    n_attempts = 0
    max_attempts = max(1000, 200 * S)
    c.opts = opts_mc
    try
        while iteration < S
            n_attempts += 1
            n_attempts > max_attempts && throw(IdentificationError("identification: no admissible parameter draw (the model does not solve) in $max_attempts attempts"))
            theta = _id_prior_draw(c)
            res = try
                _id_analysis(c, theta; init=false)
            catch e
                e isa IdentificationStop && continue
                rethrow()
            end
            ide_dyn, ide_rf, ide_mom = res[:ide_dynamic], res[:ide_reducedform], res[:ide_moments]
            ide_spec, ide_min, err = res[:ide_spectrum], res[:ide_minimal], res[:error_indicator]
            if iteration == 0
                out[:clear_sto] = true
                rows_rf = size(ide_rf["si_dREDUCEDFORM"], 1)
                max_runs = min(S, Int(ceil(1e7 / (rows_rf * tot) / 8)))
                sto["si_dDYNAMIC"] = zeros(size(ide_dyn["si_dDYNAMIC"], 1), modn, max_runs)
                sto["DYNAMIC"] = zeros(length(ide_dyn["DYNAMIC"]), S)
                sto["IDE_DYNAMIC"] = _id_ide_store(ide_dyn, S, modn, "ind_dDYNAMIC")
                sto["si_dREDUCEDFORM"] = zeros(rows_rf, tot, max_runs)
                sto["REDUCEDFORM"] = zeros(length(ide_rf["REDUCEDFORM"]), S)
                sto["IDE_REDUCEDFORM"] = _id_ide_store(ide_rf, S, tot, "ind_dREDUCEDFORM")
                if !no_mom
                    isempty(ide_mom) && throw(IdentificationError("identification: no moment criterion at the first draw"))
                    sto["si_dMOMENTS"] = zeros(size(ide_mom["si_dMOMENTS"], 1), tot, max_runs)
                    sto["MOMENTS"] = zeros(length(ide_mom["MOMENTS"]), S)
                    sto["IDE_MOMENTS"] = _id_ide_store(ide_mom, S, tot, "ind_dMOMENTS")
                    k8 = min(8, tot)
                    sto["IDE_MOMENTS"]["S"] = zeros(S, k8)
                    sto["IDE_MOMENTS"]["V"] = zeros(S, tot, k8)
                end
                if !no_spec
                    (err["identification_spectrum"] != 0 || isempty(ide_spec)) && throw(IdentificationError("identification: no spectrum criterion at the first draw"))
                    sto["dSPECTRUM"] = zeros(size(ide_spec["dSPECTRUM"], 1), size(ide_spec["dSPECTRUM"], 2), max_runs)
                end
                if !no_min
                    if err["identification_minimal"] != 0 || isempty(ide_min) || get(ide_min, "minimal_state_space", 0.0) == 0.0
                        sto["dMINIMAL"] = nothing
                    else
                        sto["dMINIMAL"] = zeros(size(ide_min["dMINIMAL"], 1), size(ide_min["dMINIMAL"], 2), max_runs)
                    end
                end
            end
            iteration += 1
            run_index += 1
            pdraws[iteration, :] = theta
            it = iteration
            sto["DYNAMIC"][:, it] = ide_dyn["DYNAMIC"]
            sto["si_dDYNAMIC"][:, :, run_index] = ide_dyn["si_dDYNAMIC"]
            _id_ide_put!(sto["IDE_DYNAMIC"], it, ide_dyn)
            sto["REDUCEDFORM"][:, it] = ide_rf["REDUCEDFORM"]
            sto["si_dREDUCEDFORM"][:, :, run_index] = ide_rf["si_dREDUCEDFORM"]
            _id_ide_put!(sto["IDE_REDUCEDFORM"], it, ide_rf)
            if !no_mom
                sto["MOMENTS"][:, it] = ide_mom["MOMENTS"]
                sto["si_dMOMENTS"][:, :, run_index] = ide_mom["si_dMOMENTS"]
                _id_ide_put!(sto["IDE_MOMENTS"], it, ide_mom)
                sto["IDE_MOMENTS"]["S"][it, :] = vec(ide_mom["S"])
                sto["IDE_MOMENTS"]["V"][it, :, :] = ide_mom["V"]
            end
            !no_spec && (sto["dSPECTRUM"][:, :, run_index] = ide_spec["dSPECTRUM"])
            if !no_min && err["identification_minimal"] == 0
                sto["dMINIMAL"] === nothing && throw(IdentificationError("identification: the minimal system is available at a later draw only (Dynare's Monte Carlo loop fails there)"))
                sto["dMINIMAL"][:, :, run_index] = ide_min["dMINIMAL"]
            end
            if run_index == max_runs || iteration == S
                n = run_index
                f = Dict{String,Any}("STO_si_dDYNAMIC" => sto["si_dDYNAMIC"][:, :, 1:n], "STO_si_dREDUCEDFORM" => sto["si_dREDUCEDFORM"][:, :, 1:n])
                !no_mom && (f["STO_si_dMOMENTS"] = sto["si_dMOMENTS"][:, :, 1:n])
                !no_spec && (f["STO_dSPECTRUM"] = sto["dSPECTRUM"][:, :, 1:n])
                !no_min && (f["STO_dMINIMAL"] = sto["dMINIMAL"] === nothing ? Matrix{Any}(undef, 0, 0) : sto["dMINIMAL"][:, :, 1:n])
                push!(files, f)
                for k in ("si_dDYNAMIC", "si_dREDUCEDFORM", "si_dMOMENTS", "dSPECTRUM", "dMINIMAL")
                    v = get(sto, k, nothing)
                    v === nothing || fill!(v, 0)
                end
                run_index = 0
            end
        end
    finally
        c.opts = opts_ident
    end
    out[:sto] = files
    norm_dyn = _id_std0(sto["DYNAMIC"], 2)
    norm_rf = _id_std0(sto["REDUCEDFORM"], 2)
    norm_mom = !no_mom ? _id_std0(sto["MOMENTS"], 2) : nothing
    normaliz1 = _id_std0(pdraws, 1)
    nsc = c.nsc
    iter_ = 0
    res_dyn = res_rf = res_mom = nothing
    for f in files
        runs = Int[size(f["STO_si_dDYNAMIC"], 3), size(f["STO_si_dREDUCEDFORM"], 3)]
        !no_mom && push!(runs, size(f["STO_si_dMOMENTS"], 3))
        !no_spec && push!(runs, size(f["STO_dSPECTRUM"], 3))
        (!no_min && ndims(f["STO_dMINIMAL"]) == 3) && push!(runs, size(f["STO_dMINIMAL"], 3))
        nr = maximum(runs)
        res_dyn = fill(NaN, nr, modn)
        res_rf = fill(NaN, nr, tot)
        res_mom = !no_mom ? fill(NaN, nr, tot) : nothing
        for irun in 1:nr
            iter_ += 1
            res_dyn = _id_grow(res_dyn, iter_)
            res_rf = _id_grow(res_rf, iter_)
            if irun <= runs[1]
                res_dyn[iter_, :] = _id_vnorm_matlab(f["STO_si_dDYNAMIC"][:, :, irun] ./ norm_dyn) .* normaliz1[nsc+1:end]
            end
            if irun <= runs[2]
                res_rf[iter_, :] = _id_vnorm_matlab(f["STO_si_dREDUCEDFORM"][:, :, irun] ./ norm_rf) .* normaliz1
            end
            if !no_mom
                res_mom = _id_grow(res_mom, iter_)
                irun <= size(f["STO_si_dMOMENTS"], 3) && (res_mom[iter_, :] = _id_vnorm_matlab(f["STO_si_dMOMENTS"][:, :, irun] ./ norm_mom) .* normaliz1)
            end
        end
    end
    sto["IDE_DYNAMIC"]["si_dDYNAMICnorm"] = res_dyn
    extra = Dict{String,Any}("pdraws" => pdraws, "IDE_DYNAMIC" => sto["IDE_DYNAMIC"], "STO_DYNAMIC" => sto["DYNAMIC"])
    sto["IDE_REDUCEDFORM"]["si_dREDUCEDFORMnorm"] = res_rf
    extra["IDE_REDUCEDFORM"] = sto["IDE_REDUCEDFORM"]; extra["STO_REDUCEDFORM"] = sto["REDUCEDFORM"]
    if !no_mom
        sto["IDE_MOMENTS"]["si_dMOMENTSnorm"] = res_mom
        extra["IDE_MOMENTS"] = sto["IDE_MOMENTS"]; extra["STO_MOMENTS"] = sto["MOMENTS"]
    end
    merge!(out[:extra], extra)
    c.not_replayed && push!(out[:messages], "prior draws of gamma, beta, inverse gamma and Weibull priors come from the inverse CDF of one uniform each: Octave's gamrnd/betarnd use randg, which Dynare does not seed, so they cannot be replayed")
    if o["advanced"] != 0
        ino = !no_mom ? vec(sto["IDE_MOMENTS"]["ino"]) : zeros(S)
        jcrit = findall(!iszero, ino)
        if length(jcrit) < S
            if isempty(jcrit)
                cnd = vec(sto["IDE_MOMENTS"]["cond"])
                jmax, jmin = argmax(cnd), argmin(cnd)
                c.opts = opts_rerun
                try
                    for (tag, j) in (("max", jmax), ("min", jmin))
                        r = _id_analysis(c, pdraws[j, :]; init=true)
                        out[:extra]["ide_hess_$tag"] = r[:ide_hess]; out[:extra]["ide_moments_$tag"] = r[:ide_moments]
                        out[:extra]["ide_spectrum_$tag"] = r[:ide_spectrum]; out[:extra]["ide_minimal_$tag"] = r[:ide_minimal]
                        out[:extra]["ide_reducedform_$tag"] = r[:ide_reducedform]; out[:extra]["ide_dynamic_$tag"] = r[:ide_dynamic]
                        out[:extra]["j$tag"] = Float64(j)
                    end
                finally
                    c.opts = opts_ident
                end
            else
                throw(IdentificationError("identification: advanced=1 with rank deficient draws in the Monte Carlo sample (struct arrays of the analyses) is not ported"))
            end
        end
    end
    nothing
end

# ------------------------------------------------------------------------------------------------ run
"""
`identification.run` for the statement: the point analysis (with up to 50 prior draws where the model does not solve) and, for
`prior_mc > 1`, the Monte Carlo analysis.  Returns nothing where the model does not solve, else a Dict with `:label`, `:point`
(the variables of `<name>_identif.mat`), `:extra` (variables appended to it), `:sto` (the `<name>_identif_<k>.mat` files),
`:clear_sto` and `:messages`.
"""
function _id_run(c::_IdCtx)
    o = c.opts
    theta, label = _id_point(c)
    messages = copy(c.messages_init)
    res = nothing
    try
        res = _id_analysis(c, theta; init=true)
    catch e
        e isa IdentificationStop || rethrow()
        push!(messages, "The model does not solve for $label: $(e.msg)")
        if c.prior_exist && is_bayesian(c.est.prior)
            for _ in 1:50
                theta = _id_prior_draw(c)
                try
                    res = _id_analysis(c, theta; init=true)
                    break
                catch e2
                    e2 isa IdentificationStop || rethrow()
                end
            end
            res === nothing || (label = "Random_prior_params")
        end
    end
    res === nothing && return nothing
    append!(messages, res[:messages])
    res[:ide_hess]["params"] = Vector{Float64}(theta)
    store = Dict{String,Any}(k => v for (k, v) in c.opts)
    point = Dict{String,Any}("ide_moments_point" => res[:ide_moments], "ide_spectrum_point" => res[:ide_spectrum],
        "ide_minimal_point" => res[:ide_minimal], "ide_hess_point" => res[:ide_hess], "ide_reducedform_point" => res[:ide_reducedform],
        "ide_dynamic_point" => res[:ide_dynamic], "store_options_ident" => store)
    out = Dict{Symbol,Any}(:label => label, :point => point, :extra => Dict{String,Any}(), :sto => Dict{String,Any}[],
                           :clear_sto => false, :messages => messages)
    o["prior_mc"] > 1 && _id_monte_carlo!(c, out, res)
    out
end

# ------------------------------------------------------------------------------------------------ the statement
const _ID_COLUMN_FIELDS = Set(["DYNAMIC", "REDUCEDFORM", "MOMENTS", "S", "Mco"])

"A value of the identification structures as the .mat file holds it (row vectors for 1-d arrays, as Dynare's)."
function _id_mat(v, key=nothing)
    v isa AbstractDict && return Dict{String,Any}(string(k) => _id_mat(x, string(k)) for (k, x) in v)
    v isa Bool && return Float64(v)
    v isa Integer && return Float64(v)
    v isa Real && return Float64(v)
    v isa AbstractString && return String(v)
    v isa Nothing && return zeros(0, 0)
    if v isa AbstractArray && eltype(v) <: Union{Real,Bool}
        a = Array{Float64}(v)
        if ndims(a) == 1
            isempty(a) && return key in _ID_COLUMN_FIELDS ? zeros(0, 1) : zeros(0, 0)
            return key in _ID_COLUMN_FIELDS ? reshape(a, :, 1) : reshape(a, 1, :)
        end
        return a
    end
    if v isa AbstractArray && eltype(v) <: Complex
        return Array{ComplexF64}(v)
    end
    v isa AbstractArray && return map(x -> _id_mat(x), v)
    throw(IdentificationError("identification: cannot write a value of type $(typeof(v))"))
end

function _id_write_mat(dir::AbstractString, fname::AbstractString, variables::AbstractDict)
    mkpath(dir)
    MAT.matwrite(joinpath(dir, fname), Dict{String,Any}(k => _id_mat(v, k) for (k, v) in variables))
end

"""
The .mod file may empty the global `estim_params_` (kim2), which makes Dynare test the current parameters instead of the priors;
any other value assigned to it is refused.
"""
function _id_no_priors(m::Model)
    m.session === nothing && return false
    v = get(m.session.sc.vars, "estim_params_", nothing)
    v === nothing && return false
    (v isa AbstractArray && isempty(v)) && return true
    throw(IdentificationError("identification: estim_params_ was assigned in the .mod file; only [] is supported"))
end

"The `identification` statement: runs the analysis and writes `<fname>/identification/*_identif.mat` (nothing is changed in `oo_`)."
function _identification_run!(m::Model, st; rng=nothing, warn=nothing)
    raw = Dict{String,Any}(string(k) => v for (k, v) in get_or(st, "options", Dict{String,Any}()))
    c = _id_context(m, raw; rng=rng, no_priors=_id_no_priors(m))
    out = _id_run(c)
    out === nothing && return nothing
    for msg in out[:messages]
        warn === nothing ? @warn("identification: " * msg) : warn(msg)
    end
    dir = joinpath(m.mod_dir, m.name, "identification")
    name = m.name
    if out[:clear_sto]
        isdir(dir) && for f in readdir(dir)
            occursin(Regex("^" * name * "_identif_\\d+\\.mat\$"), f) && rm(joinpath(dir, f); force=true)
        end
    end
    _id_write_mat(dir, "$(name)_identif.mat", merge(out[:point], out[:extra]))
    _id_write_mat(dir, "$(name)_$(out[:label])_identif.mat", out[:point])
    for (k, f) in enumerate(out[:sto])
        _id_write_mat(dir, "$(name)_identif_$k.mat", f)
    end
    out
end

"""
The `identification` statement of a .mod file (run.m): changes none of `M_`, `oo_` and `options_` (except `options_.nodisplay` and
`options_.nograph` when the statement gives them) and writes `<fname>/identification/<fname>_identif.mat` and
`<fname>_<parameter set>_identif.mat`.
"""
function _identification_statement!(ws, st, rng)
    m = ws.model
    opts = get_or(st, "options", Dict{String,Any}())
    for k in ("nodisplay", "nograph")
        haskey(opts, k) && (ws.options_[k] = _mval(opts[k]))
    end
    r = rng === nothing ? dynare_rng(ws) : rng
    _identification_run!(m, st; rng=r, warn=msg -> warn_once(m, "identification: " * msg))
    nothing
end
