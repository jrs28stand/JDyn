# Dynare's nonlinear filters for second-order approximated models (estimation(order=2, filter_algorithm=...)).
#
# Port of pydyn/nonlinear_filters.py, itself a port of matlab/nonlinear-filters/{sequential_importance_particle_filter,
# auxiliary_particle_filter, conditional_particle_filter, gaussian_filter, gaussian_filter_bank, nonlinear_kalman_filter,
# iterate_law_of_motion, measurement_equations, resample, traditional_resampling, multivariate_smooth_resampling,
# univariate_smooth_resampling, neff, probability2, reduced_rank_cholesky, unscented_sigma_points,
# spherical_radial_sigma_points}.m, matlab/estimation/qr2.m and matlab/missing/mex/local_state_space_iterations/
# local_state_space_iteration_2.m of Dynare 7.2 (Copyright (C) 2009-2025 Dynare Team, GPL-3.0-or-later), modified.
#
# The state space is the reduced form of non_linear_dsge_likelihood.m: the rows of the decision rule restricted to the
# union of the state and observed variables (`dr.restrict_var_list`), `mf0` the positions of the state variables and `mf1`
# those of the observed variables in it. Particles are columns (states x particles), as in Dynare.
#
# Every filter returns `(LIK, lik, complex)`: minus the log likelihood summed from period `start` (1-based), the log
# density of every period (NaN/-Inf where Dynare stops early) and whether Dynare's LIK would be complex (a log of a
# negative number: info 46). Random numbers come from a `ParticleStream` with MATLAB's randn(r, c)/rand(r, c) calls, in
# Dynare's order; with `OctaveStream(0)` the filters reproduce Dynare under Octave draw for draw.
#
# The hot loops (sis, apf) are allocation free after setup: one second-order transition for all particles is a single
# matrix product, G * Z, where the columns of Z hold the state deviations, the shocks and their Kronecker products.
#
# Dynare's behaviour is reproduced including what looks like slips in its code (NONLINEAR_FILTER_QUIRKS); cases where
# Dynare's code stops with an error throw `NonlinearFilterError`.

#: behaviour of Dynare 7.2 that is reproduced although it looks unintended
const NONLINEAR_FILTER_QUIRKS = (
    "resampling=generic resamples when neff < resampling_threshold * (number of periods), not * (number of particles) (sis, apf, cpf, gf with Monte Carlo distribution approximation)",
    "sis with resampling=generic: in a period without resampling the particles are not moved forward (the state vectors stay those of the previous period)",
    "resampling_method=stratified: noise.^(1./(N:-1:1)) broadcasts to an N x N matrix, so the sorted uniforms are the cumulative products of the uniforms and nearly all particles get index 1",
    "apf: the first-stage Student-t weights use z.^2 where z is already a squared Mahalanobis distance",
    "gf and nlkf with sigma points: no resampling options are used; gf adds 1e-6 to every weight",
)

"A configuration or state where Dynare's code stops with an error."
struct NonlinearFilterError <: Exception
    msg::String
end
Base.showerror(io::IO, e::NonlinearFilterError) = print(io, "NonlinearFilterError: ", e.msg)

"`options_.particle` (the fields the filters use), with Dynare's defaults."
Base.@kwdef mutable struct ParticleOptions
    filter_algorithm::String = "sis"
    number_of_particles::Int = 5000
    resampling::String = "systematic"               # systematic | none | generic
    resampling_threshold::Float64 = 0.5
    resampling_method::String = "kitagawa"          # kitagawa | stratified | smooth
    proposal_approximation::String = "unscented"    # unscented | cubature | montecarlo
    distribution_approximation::String = "unscented"
    unscented_alpha::Float64 = 1.0
    unscented_beta::Float64 = 2.0
    unscented_kappa::Float64 = 1.0
    pruning::Bool = false                           # options_.particle.pruning
    cpf_weights::String = "amisanotristani"         # amisanotristani | murrayjonesparslow
    initialization::Int = 1
    initial_state_prior_std::Float64 = 0.1
end

"Second-order reduced form (rows: `dr.restrict_var_list`, decision-rule order); `mf0`, `mf1` are 1-based."
mutable struct NLReducedForm
    ghx::Matrix{Float64}
    ghu::Matrix{Float64}
    ghxx::Matrix{Float64}
    ghuu::Matrix{Float64}
    ghxu::Matrix{Float64}
    ghs2::Vector{Float64}
    steadystate::Vector{Float64}
    constant::Vector{Float64}
    state_variables_steady_state::Vector{Float64}
    mf0::Vector{Int}
    mf1::Vector{Int}
    Q::Matrix{Float64}
    H::Matrix{Float64}
    StateVectorMean::Vector{Float64}
    StateVectorVariance::Matrix{Float64}
end

# ---------------------------------------------------------------------------------------------------- small helpers
# sequential sums, as Octave's sum (Julia's `sum` is pairwise)
@inline function _ssum(x::AbstractVector{Float64})
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]
    end
    s
end
@inline function _sdot(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    s = 0.0
    @inbounds for i in eachindex(x, y)
        s += x[i] * y[i]
    end
    s
end

neff(w::AbstractVector{Float64}) = 1.0 / _sdot(w, w)

must_resample(o::ParticleOptions, weights, sample_size) =
    (o.resampling == "generic" && neff(weights) < o.resampling_threshold * sample_size) || o.resampling == "systematic"

_finite(A) = all(isfinite, A)

"MATLAB's `chol(X)` (upper factor from the upper triangle); `nothing` if X is not positive definite."
function _chol_upper(X::AbstractMatrix{Float64})
    _finite(X) || return nothing
    F = cholesky(Symmetric(X, :U); check=false)
    issuccess(F) ? Matrix(F.U) : nothing
end
"MATLAB's `chol(X, 'lower')` (from the lower triangle); `nothing` if X is not positive definite."
function _chol_lower_from_lower(X::AbstractMatrix{Float64})
    _finite(X) || return nothing
    F = cholesky(Symmetric(X, :L); check=false)
    issuccess(F) ? Matrix(F.L) : nothing
end

function _lower_chol(X::AbstractMatrix{Float64}, what::AbstractString)
    U = _chol_upper(X)
    U === nothing && throw(NonlinearFilterError("chol: $what is not positive definite (Dynare stops with an error)"))
    Matrix(U')
end

"""
    reduced_rank_cholesky(X) -> T with T'T = X (upper Cholesky factor, or a reduced-rank factor from the eigen
    decomposition when X is only semi-definite)

Dynare's reduced_rank_cholesky; NonlinearFilterError when X is not positive semi-definite (Dynare then returns NaN, which
stops its filters with an error a few lines later).
"""
function reduced_rank_cholesky(X::AbstractMatrix{Float64})
    U = _chol_upper(X)
    U === nothing || return U
    _finite(X) && size(X, 1) == size(X, 2) ||
        throw(NonlinearFilterError("reduced_rank_cholesky:: Input matrix is not semidefinite positive!"))
    n = size(X, 1)
    E = eigen(Symmetric(X, :U))
    D, Uv = E.values, copy(E.vectors)
    for j in 1:n
        i = argmax(abs.(@view Uv[:, j]))
        Uv[i, j] < 0 && (Uv[:, j] .*= -1)
    end
    tol = sqrt(eps(maximum(D)) * length(D) * 10)
    t = findall(d -> abs(d) > tol, D)
    Dt = D[t]
    any(<(0), Dt) && throw(NonlinearFilterError("reduced_rank_cholesky:: Input matrix is not semidefinite positive!"))
    Diagonal(sqrt.(Dt)) * Uv[:, t]'
end

"R factor of `qr2(A, 0)`: economy QR with a non-negative diagonal."
function qr2_r(A::AbstractMatrix{Float64})
    R = Matrix(qr(A).R)
    for i in 1:min(size(R)...)
        R[i, i] < 0 && (R[i, :] .*= -1)
    end
    R
end

"Density N(mu, S S') (S lower triangular) at the columns of X, with Dynare's probability2 (adds 1e-99)."
function probability2(mu::AbstractVector{Float64}, S::AbstractMatrix{Float64}, X::AbstractMatrix{Float64})
    dim = size(X, 1)
    foo = X .- mu
    ldiv!(LowerTriangular(S), foo)                        # S\(X-mu) for the lower triangular S of Dynare's callers
    normfact = (2 * pi)^(dim / 2)
    dg = 1.0
    for i in 1:size(S, 1)
        dg *= S[i, i]
    end
    out = Vector{Float64}(undef, size(X, 2))
    den = abs(normfact * dg)
    for j in axes(X, 2)
        s = 0.0
        for i in 1:dim
            s += foo[i, j] * foo[i, j]
        end
        out[j] = exp(-0.5 * s) / den + 1e-99
    end
    out
end

# H\x for the measurement-error covariance (a diagonal H divides exactly, as MATLAB's mldivide)
struct HSolver
    diag::Union{Nothing,Vector{Float64}}
    fact::Any
end
function HSolver(H::AbstractMatrix{Float64})
    if H == Diagonal(diag(H))
        HSolver(Float64.(diag(H)), nothing)
    else
        F = cholesky(Symmetric(H); check=false)
        HSolver(nothing, issuccess(F) ? F : lu(H))
    end
end
"sum(PE .* (H\\PE)) for column j of PE."
@inline function _quadform(hs::HSolver, PE::AbstractMatrix{Float64}, j::Int, buf::Vector{Float64})
    p = size(PE, 1)
    s = 0.0
    if hs.diag !== nothing
        d = hs.diag
        @inbounds for i in 1:p
            s += PE[i, j] * (PE[i, j] / d[i])
        end
    else
        @inbounds for i in 1:p
            buf[i] = PE[i, j]
        end
        ldiv!(hs.fact, buf)
        @inbounds for i in 1:p
            s += PE[i, j] * buf[i]
        end
    end
    s
end

function _rcond(H::AbstractMatrix{Float64})
    _finite(H) || return 0.0
    try
        return 1.0 / cond(H, 1)
    catch
        return 0.0
    end
end

# ---------------------------------------------------------------------------------------------------- sigma points
"Nodes ((2n+1) x n) and the weights W_m, W_c of the unscented transform."
function unscented_sigma_points(n::Int, alpha=1.0, beta=1.0, kappa=1.0)
    lam = alpha^2 * (n + kappa) - n
    nodes = zeros(2n + 1, n)
    f = sqrt(n + lam)
    for i in 1:n
        nodes[1+i, i] = f
        nodes[1+n+i, i] = -f
    end
    w0 = lam / (n + lam)
    temp = fill(1 / (2 * (n + lam)), 2n)
    W_m = vcat(w0, temp)
    W_c = vcat(w0 + (1 - alpha^2 + beta), temp)
    nodes, W_m, W_c
end

"Nodes (2n x n) and the (scalar) weight of the spherical-radial cubature rule."
function spherical_radial_sigma_points(n::Int)
    nodes = zeros(2n, n)
    f = sqrt(n)
    for i in 1:n
        nodes[i, i] = f
        nodes[n+i, i] = -f
    end
    nodes, 1.0 / (2n)
end

# ---------------------------------------------------------------------------------------------------- second-order state space
"""
Workspace of the second-order transition (Dynare's local_state_space_iteration_2) for `N` columns: with
`G = [ghx ghu ghxx/2 ghuu/2 ghxu]` and `Z` the stacked deviations, shocks and their Kronecker products,
`y = constant + G Z`, one matrix product for all particles.
"""
mutable struct Transition
    rf::NLReducedForm
    N::Int
    m::Int
    n::Int
    q::Int
    G::Matrix{Float64}
    Z::Matrix{Float64}
    out::Matrix{Float64}
    pr::Matrix{Float64}          # pruned first-order state (levels), rows of the restricted variables
    ssv::Vector{Float64}
end

function Transition(rf::NLReducedForm, N::Int)
    m, n = size(rf.ghx)
    q = size(rf.ghu, 2)
    size(rf.ghxx, 2) == n * n && size(rf.ghuu, 2) == q * q && size(rf.ghxu, 2) == n * q ||
        throw(NonlinearFilterError("the second-order decision rule has inconsistent dimensions"))
    G = hcat(rf.ghx, rf.ghu, 0.5 .* rf.ghxx, 0.5 .* rf.ghuu, rf.ghxu)
    Transition(rf, N, m, n, q, G, Matrix{Float64}(undef, size(G, 2), N), Matrix{Float64}(undef, m, N),
               Matrix{Float64}(undef, 0, 0), rf.state_variables_steady_state)
end

"""
    iterate_law_of_motion!(tr, SV, eps, SVp=nothing) -> tr.out (levels of all restricted variables at t+1)

`SV` (states x N, or states x 1 for all columns), `eps` (shocks x N). With `SVp` (pruned states, levels) the quadratic
terms use the pruned state and `tr.pr` receives the pruned first-order state plus the steady state.
"""
function iterate_law_of_motion!(tr::Transition, SV::AbstractMatrix{Float64}, eps::AbstractMatrix{Float64},
                                SVp::Union{Nothing,AbstractMatrix{Float64}}=nothing)
    n, q, N, Z, ssv = tr.n, tr.q, tr.N, tr.Z, tr.ssv
    size(eps, 2) == N || throw(DimensionMismatch("iterate_law_of_motion: $(size(eps, 2)) shock columns for a workspace of $N"))
    bc = size(SV, 2) == 1 && N > 1
    o2 = n + q
    o3 = o2 + n * n
    o4 = o3 + q * q
    SQ = SVp === nothing ? SV : SVp
    bcq = size(SQ, 2) == 1 && N > 1
    @inbounds for j in 1:N
        sj = bc ? 1 : j
        sq = bcq ? 1 : j
        for a in 1:n
            Z[a, j] = SV[a, sj] - ssv[a]
        end
        for a in 1:q
            Z[n+a, j] = eps[a, j]
        end
        for a in 1:n
            ya = SQ[a, sq] - ssv[a]
            for b in 1:n
                Z[o2+(a-1)*n+b, j] = ya * (SQ[b, sq] - ssv[b])
            end
        end
        for a in 1:q
            ea = eps[a, j]
            for b in 1:q
                Z[o3+(a-1)*q+b, j] = ea * eps[b, j]
            end
        end
        for a in 1:n
            ya = SQ[a, sq] - ssv[a]
            for b in 1:q
                Z[o4+(a-1)*q+b, j] = ya * eps[b, j]
            end
        end
    end
    mul!(tr.out, tr.G, Z)
    c = tr.rf.constant
    out = tr.out
    @inbounds for j in 1:N
        for i in 1:tr.m
            out[i, j] += c[i]
        end
    end
    if SVp !== nothing
        size(tr.pr, 2) == N || (tr.pr = Matrix{Float64}(undef, tr.m, N))
        ghx, ghu, ss = tr.rf.ghx, tr.rf.ghu, tr.rf.steadystate
        @inbounds for j in 1:N
            sq = bcq ? 1 : j
            for i in 1:tr.m
                s = 0.0
                for a in 1:n
                    s += ghx[i, a] * (SVp[a, sq] - ssv[a])
                end
                for a in 1:q
                    s += ghu[i, a] * eps[a, j]
                end
                tr.pr[i, j] = s + ss[i]
            end
        end
    end
    out
end

"Observables at the given states with zero shocks (Dynare's measurement_equations)."
function measurement_equations(SV::AbstractMatrix{Float64}, rf::NLReducedForm)
    tr = Transition(rf, size(SV, 2))
    iterate_law_of_motion!(tr, SV, zeros(tr.q, size(SV, 2)))
    tr.out[rf.mf1, :]
end

# ---------------------------------------------------------------------------------------------------- resampling
"""
    kitagawa_indices!(idx, c, weights, noise)

Indices (1-based) of Kitagawa's systematic resampling (one uniform `noise`): `idx[i]` is the first `j` with
`(i-1+noise)/N <= cumsum(weights)[j]`.
"""
function kitagawa_indices!(idx::Vector{Int}, c::Vector{Float64}, weights::AbstractVector{Float64}, noise::Float64)
    N = length(weights)
    s = 0.0
    @inbounds for i in 1:N
        s += weights[i]
        c[i] = s
    end
    j = 1
    @inbounds for i in 1:N
        rv = ((i - 1) + noise) / N
        while rv > c[j]
            j += 1
            j > N && throw(NonlinearFilterError("resampling: index out of bound (the cumulated weights do not reach the resampling threshold; Dynare stops here too)"))
        end
        idx[i] = j
    end
    idx
end

"Dynare's \"stratified\" variant: one uniform per particle; the sorted uniforms are their cumulative products (a slip of Dynare's code)."
function stratified_indices!(idx::Vector{Int}, c::Vector{Float64}, weights::AbstractVector{Float64}, noise::AbstractVector{Float64})
    N = length(weights)
    s = 0.0
    @inbounds for i in 1:N
        s += weights[i]
        c[i] = s
    end
    rv = 1.0
    @inbounds for i in 1:N
        rv *= noise[i]
        k = 1
        if !isnan(c[N])
            k = searchsortedfirst(c, rv)               # 1 + sum(rv > c)
        end
        k > N && throw(NonlinearFilterError("resampling: index out of bound (the cumulated weights do not reach the resampling threshold; Dynare stops here too)"))
        idx[i] = k
    end
    idx
end

function univariate_smooth_resampling(weights::AbstractVector{Float64}, particles::AbstractVector{Float64}, nnew::Int, u::Float64)
    M = length(particles)
    lt = Vector{Float64}(undef, M + 1)
    lt[1] = 0.5 * weights[1]
    for i in 2:M
        lt[i] = 0.5 * (weights[i-1] + weights[i])
    end
    lt[M+1] = 0.5 * weights[M]
    lb = cumsum(lt)
    out = Vector{Float64}(undef, nnew)
    rj, j = 0, 1
    for i in 1:nnew
        uj = ((i - 1) + u) / nnew
        while uj > lb[j]
            rj = j
            j += 1
            j > M + 1 && throw(NonlinearFilterError("univariate_smooth_resampling: index out of bound"))
        end
        if rj == 0
            out[i] = particles[1]
        elseif rj == M
            out[i] = particles[M]
        else
            ustar = (uj - lb[rj]) / lt[rj+1]
            out[i] = (particles[rj+1] - particles[rj]) * ustar + particles[rj]
        end
    end
    out
end

"""
    multivariate_smooth_resampling(particles, weights, stream; symmetric=nothing)

`multivariate_smooth_resampling.m`; `particles` is N x s. `symmetric` forces the eigensolver (default: the symmetric one if
the product P'P/N is exactly symmetric, as MATLAB/Octave decide). Which one Dynare gets depends on the rounding of the
BLAS product, so bit-for-bit agreement with Dynare is not guaranteed for this method.
"""
function multivariate_smooth_resampling(particles::AbstractMatrix{Float64}, weights::AbstractVector{Float64}, stream::ParticleStream;
                                        symmetric::Union{Nothing,Bool}=nothing)
    N = length(weights)
    s = size(particles, 2)
    A = particles' * (particles .* (1.0 / N))
    sym = symmetric === nothing ? issymmetric(A) : symmetric
    local D, P
    if sym
        D, P = LAPACK.syev!('V', 'U', copy(A))
    else
        E = eigen(A)
        (any(!iszero ∘ imag, E.values) || any(!iszero ∘ imag, E.vectors)) &&
            throw(NonlinearFilterError("multivariate_smooth_resampling: complex eigenvalues"))
        D, P = real.(E.values), real.(E.vectors)
    end
    vectors = P .* sqrt.(D)'
    orth = (particles * vectors) ./ D'
    new = zeros(N, s)
    for j in 1:s
        order = sortperm(orth[:, j]; alg=MergeSort)
        new[:, j] = univariate_smooth_resampling(weights[order], orth[order, j], N, stream_rand1(stream))
    end
    new * vectors'
end

"""
    resample_particles!(dst, src, weights, opts, stream, c, idx)

`resample.m` for particles stored as columns: `dst[:, i] = src[:, idx[i]]`. For `resampling_method=smooth` the particles
are smoothed (`dst` and `src` are s x N).
"""
function resample_particles!(dst::Matrix{Float64}, src::Matrix{Float64}, weights::AbstractVector{Float64}, opts::ParticleOptions,
                             stream::ParticleStream, c::Vector{Float64}, idx::Vector{Int})
    m = opts.resampling_method
    if m == "smooth"
        dst .= permutedims(multivariate_smooth_resampling(permutedims(src), weights, stream))
        return dst
    end
    resample_indices!(idx, c, weights, opts, stream)
    @inbounds for i in eachindex(idx)
        k = idx[i]
        for a in axes(src, 1)
            dst[a, i] = src[a, k]
        end
    end
    dst
end

function resample_indices!(idx::Vector{Int}, c::Vector{Float64}, weights::AbstractVector{Float64}, opts::ParticleOptions, stream::ParticleStream)
    m = opts.resampling_method
    if m == "kitagawa"
        kitagawa_indices!(idx, c, weights, stream_rand1(stream))
    elseif m == "stratified"
        stratified_indices!(idx, c, weights, vec(stream_rand(stream, length(weights), 1)))
    elseif m == "smooth"
        throw(NonlinearFilterError("Particle = 0 is incompatible with the resampling_method=smooth method!"))
    else
        throw(NonlinearFilterError("Unknown sampling method!"))
    end
    idx
end

# ---------------------------------------------------------------------------------------------------- particle filters
_hmat(rf::NLReducedForm) = isempty(rf.H) ? zeros(1, 1) : rf.H

"""
    sequential_importance_particle_filter(rf, Y, start, opts, stream) -> (LIK, lik, complex)

`sequential_importance_particle_filter.m` (`filter_algorithm = sis`). `Y` is observables x periods.
"""
function sequential_importance_particle_filter(rf::NLReducedForm, Y::AbstractMatrix{Float64}, start::Int, opts::ParticleOptions, stream::ParticleStream)
    T = size(Y, 2)
    mf0, mf1 = rf.mf0, rf.mf1
    n, p, q = length(mf0), length(mf1), size(rf.Q, 1)
    N = opts.number_of_particles
    H = _hmat(rf)
    lik = fill(NaN, T)
    _rcond(H) < 1e-12 && return NaN, lik, false
    const_lik = log(2 * pi) * p + log(det(H))
    S0 = _lower_chol(rf.StateVectorVariance, "the initial state variance")
    Qc = _lower_chol(rf.Q, "Q")
    pruning = opts.pruning
    hs = HSolver(H)
    tr = Transition(rf, N)
    weights = fill(1.0 / N, N)
    wt = Vector{Float64}(undef, N)
    lnw = Vector{Float64}(undef, N)
    c = Vector{Float64}(undef, N)
    idx = Vector{Int}(undef, N)
    buf = Vector{Float64}(undef, p)
    E0 = Matrix{Float64}(undef, q, N)
    eps = Matrix{Float64}(undef, q, N)
    PE = Matrix{Float64}(undef, p, N)
    SV = S0 * stream_randn(stream, size(S0, 2), N)
    SV .+= rf.StateVectorMean
    SVp = pruning ? copy(SV) : nothing
    SVn = Matrix{Float64}(undef, n, N)
    SVpn = pruning ? Matrix{Float64}(undef, n, N) : nothing
    comb_src = pruning ? Matrix{Float64}(undef, 2n, N) : Matrix{Float64}(undef, 0, 0)
    comb_dst = pruning ? Matrix{Float64}(undef, 2n, N) : Matrix{Float64}(undef, 0, 0)
    sres = Matrix{Float64}(undef, n, N)
    for t in 1:T
        stream_randn!(stream, E0)
        mul!(eps, Qc, E0)
        iterate_law_of_motion!(tr, SV, eps, SVp)
        tmp = tr.out
        @inbounds for j in 1:N
            for i in 1:p
                PE[i, j] = Y[i, t] - tmp[mf1[i], j]
            end
            lnw[j] = -0.5 * (const_lik + _quadform(hs, PE, j, buf))
        end
        dfac = maximum(lnw)
        @inbounds for j in 1:N
            wt[j] = weights[j] * exp(lnw[j] - dfac)
        end
        sw = _ssum(wt)
        lik[t] = log(sw) + dfac
        @inbounds for j in 1:N
            weights[j] = wt[j] / sw
        end
        if must_resample(opts, weights, T)
            if pruning
                @inbounds for j in 1:N
                    for a in 1:n
                        comb_src[a, j] = tmp[mf0[a], j]
                        comb_src[n+a, j] = tr.pr[mf0[a], j]
                    end
                end
                resample_particles!(comb_dst, comb_src, weights, opts, stream, c, idx)
                @inbounds for j in 1:N, a in 1:n
                    SV[a, j] = comb_dst[a, j]
                    SVp[a, j] = comb_dst[n+a, j]
                end
            else
                @inbounds for j in 1:N, a in 1:n
                    sres[a, j] = tmp[mf0[a], j]
                end
                resample_particles!(SVn, sres, weights, opts, stream, c, idx)
                SV, SVn = SVn, SV
            end
            fill!(weights, 1.0 / N)
        elseif opts.resampling == "none"
            @inbounds for j in 1:N, a in 1:n
                SV[a, j] = tmp[mf0[a], j]
            end
            if pruning
                @inbounds for j in 1:N, a in 1:n
                    SVp[a, j] = tr.pr[mf0[a], j]
                end
            end
        end
    end
    -_ssum(@view lik[start:end]), lik, false
end

"""
    auxiliary_particle_filter(rf, Y, start, opts, stream) -> (LIK, lik, complex)

`auxiliary_particle_filter.m` (`filter_algorithm = apf`).
"""
function auxiliary_particle_filter(rf::NLReducedForm, Y::AbstractMatrix{Float64}, start::Int, opts::ParticleOptions, stream::ParticleStream)
    opts.resampling_method == "smooth" && throw(NonlinearFilterError("auxiliary_particle_filter: resampling_method=smooth is not supported (resample(0, ...) stops in Dynare)"))
    T = size(Y, 2)
    mf0, mf1 = rf.mf0, rf.mf1
    n, p, q = length(mf0), length(mf1), size(rf.Q, 1)
    N = opts.number_of_particles
    S0 = _lower_chol(rf.StateVectorVariance, "the initial state variance")
    Qc = _lower_chol(rf.Q, "Q")
    lik = fill(NaN, T)
    H = _hmat(rf)
    _rcond(H) < 1e-12 && return NaN, lik, false
    const_lik = log(2 * pi) * p + log(det(H))
    pruning = opts.pruning
    hs = HSolver(H)
    tr = Transition(rf, N)
    weights = fill(1.0 / N, N)
    ddl = 3.0
    cst = exp(SpecialFunctions.loggamma((ddl + 1) / 2) - SpecialFunctions.loggamma(ddl / 2))
    tau = Vector{Float64}(undef, N)
    w1 = Vector{Float64}(undef, N)
    w2 = Vector{Float64}(undef, N)
    c = Vector{Float64}(undef, N)
    idx = Vector{Int}(undef, N)
    buf = Vector{Float64}(undef, p)
    E0 = Matrix{Float64}(undef, q, N)
    eps = Matrix{Float64}(undef, q, N)
    zeros_q = zeros(q, N)
    PE = Matrix{Float64}(undef, p, N)
    SV = S0 * stream_randn(stream, size(S0, 2), N)
    SV .+= rf.StateVectorMean
    SVp = pruning ? copy(SV) : nothing
    SVsel = Matrix{Float64}(undef, n, N)
    SVpsel = pruning ? Matrix{Float64}(undef, n, N) : nothing
    SVn = Matrix{Float64}(undef, n, N)
    comb_src = pruning ? Matrix{Float64}(undef, 2n, N) : Matrix{Float64}(undef, 0, 0)
    comb_dst = pruning ? Matrix{Float64}(undef, 2n, N) : Matrix{Float64}(undef, 0, 0)
    for t in 1:T
        iterate_law_of_motion!(tr, SV, zeros_q, SVp)
        tmp = tr.out
        @inbounds for j in 1:N
            for i in 1:p
                PE[i, j] = Y[i, t] - tmp[mf1[i], j]
            end
            z = _quadform(hs, PE, j, buf)
            tau[j] = weights[j] * (cst / (sqrt(ddl * pi) * (1 + (z^2) / ddl)^((ddl + 1) / 2)) + 1e-99)
        end
        st = _ssum(tau)
        @inbounds for j in 1:N
            tau[j] /= st
        end
        resample_indices!(idx, c, tau, opts, stream)
        @inbounds for j in 1:N
            w1[j] = weights[idx[j]] / tau[idx[j]]
        end
        stream_randn!(stream, E0)
        mul!(eps, Qc, E0)
        @inbounds for j in 1:N, a in 1:n
            SVsel[a, j] = SV[a, idx[j]]
        end
        if pruning
            @inbounds for j in 1:N, a in 1:n
                SVpsel[a, j] = SVp[a, idx[j]]
            end
        end
        iterate_law_of_motion!(tr, SVsel, eps, pruning ? SVpsel : nothing)
        tmp = tr.out
        @inbounds for j in 1:N
            for a in 1:n
                SV[a, j] = tmp[mf0[a], j]
            end
            pruning && for a in 1:n
                SVp[a, j] = tr.pr[mf0[a], j]
            end
            for i in 1:p
                PE[i, j] = Y[i, t] - tmp[mf1[i], j]
            end
            w2[j] = w1[j] * (exp(-0.5 * (const_lik + _quadform(hs, PE, j, buf))) + 1e-99)
        end
        sw2 = _ssum(w2)
        lik[t] = log(sw2 / N)
        @inbounds for j in 1:N
            weights[j] = w2[j] / sw2
        end
        if must_resample(opts, weights, T)
            if pruning
                @inbounds for j in 1:N, a in 1:n
                    comb_src[a, j] = SV[a, j]
                    comb_src[n+a, j] = SVp[a, j]
                end
                resample_particles!(comb_dst, comb_src, weights, opts, stream, c, idx)
                @inbounds for j in 1:N, a in 1:n
                    SV[a, j] = comb_dst[a, j]
                    SVp[a, j] = comb_dst[n+a, j]
                end
            else
                resample_particles!(SVn, SV, weights, opts, stream, c, idx)
                SV, SVn = SVn, SV
            end
            fill!(weights, 1.0 / N)
        end
    end
    -_ssum(@view lik[start:end]), lik, false
end

# probability2 for a stack: Gaussian density of the vector e under N(0, S S') with S lower triangular (p x p)
function _prob2_zero_mean(S::AbstractMatrix{Float64}, e::AbstractVector{Float64}, work::Vector{Float64})
    p = length(e)
    copyto!(work, e)
    ldiv!(LowerTriangular(S), work)
    s = 0.0
    dg = 1.0
    for i in 1:p
        s += work[i] * work[i]
        dg *= S[i, i]
    end
    exp(-0.5 * s) / abs((2 * pi)^(p / 2) * dg) + 1e-99
end

"""
    conditional_particle_filter(rf, Y, start, opts, stream; model_pruning=false) -> (LIK, lik, complex)

`conditional_particle_filter.m` (`filter_algorithm = cpf`): every particle gets a Gaussian proposal from a sigma-point
Kalman step. All sigma points of all particles go through one transition; the random numbers are drawn in Dynare's order
(particle by particle).
"""
function conditional_particle_filter(rf::NLReducedForm, Y::AbstractMatrix{Float64}, start::Int, opts::ParticleOptions, stream::ParticleStream;
                                     model_pruning::Bool=false)
    opts.pruning && throw(NonlinearFilterError("conditional_particle_filter: pruning is not yet implemented."))
    model_pruning && throw(NonlinearFilterError("iterate_law_of_motion: with pruning, the components of the augmented pruned state space are an required input (Dynare's cpf passes options_.pruning without the pruned states)"))
    T = size(Y, 2)
    N = opts.number_of_particles
    mf0, mf1 = rf.mf0, rf.mf1
    ns, p, q = length(mf0), length(mf1), size(rf.Q, 1)
    H = _hmat(rf)
    Hc = _lower_chol(H, "H")
    S0 = _lower_chol(rf.StateVectorVariance, "the initial state variance")
    Qc = _lower_chol(rf.Q, "Q")
    prop = opts.proposal_approximation
    local nodes, weights, weights_c, npts
    if prop == "montecarlo"
        N % 10 == 0 || throw(NonlinearFilterError("cpf with proposal_approximation=montecarlo needs number_of_particles divisible by 10 (randn(number_of_particles/10, ...) in Dynare)"))
        npts = N ÷ 10
        weights = 1.0 / npts
        weights_c = weights
        nodes = zeros(0, 0)
    elseif prop == "cubature"
        nodes, weights = spherical_radial_sigma_points(q)
        weights_c = weights
        npts = size(nodes, 1)
    elseif prop == "unscented"
        nodes, weights, weights_c = unscented_sigma_points(q, opts.unscented_alpha, opts.unscented_beta, opts.unscented_kappa)
        npts = size(nodes, 1)
    else
        throw(NonlinearFilterError("Estimation: This approximation for the proposal is not implemented or unknown!"))
    end
    mjp = opts.cpf_weights == "murrayjonesparslow"
    mjp && prop != "unscented" && throw(Unsupported("cpf_weights=murrayjonesparslow with proposal_approximation=cubature or montecarlo: Dynare 7.2 uses a (points x points) matrix as the predicted state variance there (dState*dState' with dState points x states); not reproduced"))
    sigma_branch = prop != "unscented"
    scalar_w = prop != "unscented"
    lik = fill(NaN, T)
    SP = S0 * stream_randn(stream, size(S0, 2), N)
    SP .+= rf.StateVectorMean
    SW = fill(1.0 / N, N)
    NP = N * npts
    tr = Transition(rf, NP)
    X = Matrix{Float64}(undef, ns, NP)
    E = Matrix{Float64}(undef, q, NP)
    Enode = prop == "montecarlo" ? Matrix{Float64}(undef, q, npts) : (Qc * nodes')
    c = Vector{Float64}(undef, N); idx = Vector{Int}(undef, N)
    sqw = scalar_w ? sqrt(weights) : 0.0
    # scratch
    tS = Matrix{Float64}(undef, ns, npts); tO = Matrix{Float64}(undef, p, npts)
    PSM = Vector{Float64}(undef, ns); POM = Vector{Float64}(undef, p)
    Err = Vector{Float64}(undef, p)
    work = Vector{Float64}(undef, p)
    Wnew = Vector{Float64}(undef, N)
    Prop = Matrix{Float64}(undef, ns, N)
    SPn = Matrix{Float64}(undef, ns, N)
    Hc_zero = hcat(Hc, zeros(p, ns))
    ndraw = prop == "montecarlo" ? npts * q + ns : ns
    SVVSRs = Vector{Matrix{Float64}}(undef, N)
    SVMs = Matrix{Float64}(undef, ns, N)
    PSVSRs = mjp ? Vector{Matrix{Float64}}(undef, N) : Matrix{Float64}[]
    PSMs = mjp ? Matrix{Float64}(undef, ns, N) : zeros(0, 0)
    for t in 1:T
        y = view(Y, :, t)
        draws = stream_randn(stream, ndraw, N) |> identity
        # one transition for every sigma point of every particle (particle-major columns)
        @inbounds for i in 1:N
            if prop == "montecarlo"
                for a in 1:q, k in 1:npts
                    # nodes(k, a) = draws[(a-1)*npts + k, i]; epsilon = Qc * nodes'
                    Enode[a, k] = 0.0
                end
                for k in 1:npts
                    for a in 1:q
                        s = 0.0
                        for b in 1:q
                            s += Qc[a, b] * draws[(b-1)*npts+k, i]
                        end
                        Enode[a, k] = s
                    end
                end
            end
            for k in 1:npts
                col = (i - 1) * npts + k
                for a in 1:ns
                    X[a, col] = SP[a, i]
                end
                for a in 1:q
                    E[a, col] = Enode[a, k]
                end
            end
        end
        iterate_law_of_motion!(tr, X, E)
        tmp = tr.out
        flag = false
        @inbounds for i in 1:N
            base = (i - 1) * npts
            for k in 1:npts, a in 1:ns
                tS[a, k] = tmp[mf0[a], base+k]
            end
            for k in 1:npts, a in 1:p
                tO[a, k] = tmp[mf1[a], base+k]
            end
            if scalar_w
                for a in 1:ns
                    s = 0.0
                    for k in 1:npts
                        s += tS[a, k] * weights
                    end
                    PSM[a] = s
                end
                for a in 1:p
                    s = 0.0
                    for k in 1:npts
                        s += tO[a, k] * weights
                    end
                    POM[a] = s
                end
            else
                mul!(PSM, tS, weights)
                mul!(POM, tO, weights)
            end
            for a in 1:p
                Err[a] = y[a] - POM[a]
            end
            local SVVSR, SVM, Wi
            if sigma_branch
                big = Matrix{Float64}(undef, npts + p, p + ns)
                for k in 1:npts
                    for a in 1:p
                        big[k, a] = (tO[a, k] - POM[a]) * sqw
                    end
                    for a in 1:ns
                        big[k, p+a] = (tS[a, k] - PSM[a]) * sqw
                    end
                end
                big[npts+1:end, :] .= Hc_zero
                mat = Matrix(qr2_r(big)')
                POVSR = mat[1:p, 1:p]
                COSSR = mat[p+1:p+ns, 1:p]
                SVVSR = mat[p+1:p+ns, p+1:p+ns]
                SVM = PSM .+ (COSSR / LowerTriangular(POVSR)) * Err
                Wi = SW[i] * _prob2_zero_mean(POVSR, Err, work)
            else
                dS = tS .- PSM
                dO = tO .- POM
                wd = weights_c' .* dS
                PSV = wd * dS'
                POV = (weights_c' .* dO) * dO' + H
                PSOC = wd * dO'
                K = PSOC / POV
                SVM = PSM .+ K * Err
                SVV = PSV - K * POV * K'
                SVV = 0.5 .* (SVV + SVV')
                SVVSR = _chol_lower_from_lower(SVV)
                if SVVSR === nothing
                    flag = true
                    break
                end
                if !mjp
                    POVc = _chol_upper(POV)
                    POVc === nothing && throw(NonlinearFilterError("chol: the predicted observation variance is not positive definite (Dynare stops with an error)"))
                    Wi = SW[i] * _prob2_zero_mean(Matrix(POVc'), Err, work)
                else
                    Wi = 0.0
                    PSVs = 0.5 .* (PSV + PSV')
                    PSVc = _chol_lower_from_lower(PSVs)
                    if PSVc === nothing
                        flag = true
                        break
                    end
                    PSVSRs[i] = PSVc
                    PSMs[:, i] = PSM
                end
            end
            SVVSRs[i] = SVVSR
            SVMs[:, i] = SVM
            Wnew[i] = Wi
        end
        if flag
            lik[t] = -Inf
            return -Inf, lik, false
        end
        # the proposals, in Dynare's order of random numbers
        @inbounds for i in 1:N
            off = prop == "montecarlo" ? npts * q : 0
            S = SVVSRs[i]
            for a in 1:ns
                s = 0.0
                for b in 1:size(S, 2)
                    s += S[a, b] * draws[off+b, i]
                end
                Prop[a, i] = s + SVMs[a, i]
            end
        end
        if mjp
            Like = probability2_cols_meas(y, Hc, Prop, rf)
            wk = Vector{Float64}(undef, ns)
            @inbounds for i in 1:N
                d1 = Prop[:, i] .- PSMs[:, i]
                Prior = _prob2_zero_mean(PSVSRs[i], d1, wk)
                d2 = Prop[:, i] .- SVMs[:, i]
                Post = _prob2_zero_mean(SVVSRs[i], d2, wk)
                Wnew[i] = SW[i] * Like[i] * (Prior / Post)
            end
        end
        SP, Prop = Prop, SP
        SW .= Wnew
        s = _ssum(SW)
        lik[t] = log(s)
        SW ./= s
        if must_resample(opts, SW, T)
            resample_particles!(SPn, SP, SW, opts, stream, c, idx)
            SP, SPn = SPn, SP
            fill!(SW, 1.0 / N)
        end
    end
    -_ssum(@view lik[start:end]), lik, false
end

function probability2_cols_meas(y, Hc, Prop, rf)
    probability2(Vector{Float64}(y), Hc, measurement_equations(Prop, rf))
end

# ---------------------------------------------------------------------------------------------------- Gaussian filters
function _proposal_nodes(opts::ParticleOptions, dim::Int, stream::ParticleStream)
    prop = opts.proposal_approximation
    if prop == "montecarlo"
        N = opts.number_of_particles
        return stream_randn(stream, N, dim), 1.0 / N, 1.0 / N
    elseif prop == "cubature"
        nodes, w = spherical_radial_sigma_points(dim)
        return nodes, w, w
    elseif prop == "unscented"
        return unscented_sigma_points(dim, opts.unscented_alpha, opts.unscented_beta, opts.unscented_kappa)
    end
    throw(NonlinearFilterError("This approximation for the proposal is not implemented or unknown!"))
end

function _sigma_propagate(rf::NLReducedForm, mean, S, Qc, nodes)
    n, q = length(rf.mf0), size(rf.Q, 1)
    (size(S) == (n, n) && size(Qc) == (q, q)) ||
        throw(NonlinearFilterError("vertical dimensions mismatch ($(size(S, 1))x$(size(S, 2)) vs $(q)x$n): the state (or shock) covariance matrix has reduced rank (Dynare stops with this error)"))
    xbar = vcat(mean, zeros(q))
    sqr = zeros(n + q, n + q)
    sqr[1:n, 1:n] = S
    sqr[n+1:end, n+1:end] = Qc
    sp = xbar .+ sqr * nodes'
    tr = Transition(rf, size(sp, 2))
    iterate_law_of_motion!(tr, sp[1:n, :], sp[n+1:end, :])
    tr.out
end

_wmean(X, w) = w isa Number ? vec(sum(X .* w; dims=2)) : X * w

"One step of `gaussian_filter_bank.m`: predicted state mean and variance square root, updated mean and square root."
function gaussian_filter_bank(rf::NLReducedForm, obs, mean, S, Qc, Hc, H, opts::ParticleOptions, stream::ParticleStream)
    mf0, mf1 = rf.mf0, rf.mf1
    n, p, q = length(mf0), length(mf1), size(rf.Q, 1)
    nodes, w, wc = _proposal_nodes(opts, n + q, stream)
    tmp = _sigma_propagate(rf, mean, S, Qc, nodes)
    tS, tO = tmp[mf0, :], tmp[mf1, :]
    PSM = _wmean(tS, w)
    POM = _wmean(tO, w)
    PE = obs .- POM
    if opts.proposal_approximation != "unscented"
        dState = (tS .- PSM)' .* sqrt(w)
        dObs = (tO .- POM)' .* sqrt(w)
        PSVSR = _lower_chol(dState' * dState, "the predicted state variance")
        size(Hc) == (p, p) || throw(NonlinearFilterError("dimensions mismatch: H has reduced rank"))
        big = vcat(hcat(dObs, dState), hcat(Hc, zeros(p, n)))
        mat = Matrix(qr2_r(big)')
        POVSR = mat[1:p, 1:p]
        COSSR = mat[p+1:p+n, 1:p]
        SVVSR = mat[p+1:p+n, p+1:p+n]
        SVM = PSM .+ (COSSR / LowerTriangular(POVSR)) * PE
    else
        dState = tS .- PSM
        dObs = tO .- POM
        PSV = (dState .* wc') * dState'
        POV = (dObs .* wc') * dObs' + H
        PSOC = (dState .* wc') * dObs'
        PSVSR = _lower_chol(PSV, "the predicted state variance")
        K = PSOC / POV
        SVM = PSM .+ K * PE
        SVV = PSV - K * POV * K'
        SVV = 0.5 .* (SVV + SVV')
        SVVSR = Matrix(reduced_rank_cholesky(SVV)')
    end
    PSM, PSVSR, SVM, SVVSR
end

"""
    gaussian_filter(rf, Y, start, opts, stream) -> (LIK, lik, complex)

`gaussian_filter.m` (`filter_algorithm = gf`): Gaussian proposal from a sigma-point (or Monte Carlo) Kalman step; the
particle distribution is approximated by sigma points or draws.
"""
function gaussian_filter(rf::NLReducedForm, Y::AbstractMatrix{Float64}, start::Int, opts::ParticleOptions, stream::ParticleStream)
    opts.pruning && throw(NonlinearFilterError("gaussian_filter: pruning is not yet implemented."))
    T = size(Y, 2)
    mf0 = rf.mf0
    n = length(mf0)
    N = opts.number_of_particles
    da = opts.distribution_approximation
    local nodes2, weights2
    if da == "cubature"
        nodes2, weights2 = spherical_radial_sigma_points(n)
    elseif da == "unscented"
        nodes2, weights2, _ = unscented_sigma_points(n, opts.unscented_alpha, opts.unscented_beta, opts.unscented_kappa)
    elseif da != "montecarlo"
        throw(NonlinearFilterError("This approximation for the distribution is unknown!"))
    end
    if isempty(rf.H)
        H, Hc = zeros(1, 1), zeros(1, 1)
    else
        H = rf.H
        H == Diagonal(diag(H)) || throw(Unsupported("gaussian_filter with correlated measurement errors: Dynare takes sqrt(H) element by element (complex numbers); not reproduced"))
        Hc = Matrix(reduced_rank_cholesky(H)')
    end
    mean = copy(rf.StateVectorMean)
    S = Matrix(reduced_rank_cholesky(rf.StateVectorVariance)')
    Qc = Matrix(reduced_rank_cholesky(rf.Q)')
    lik = fill(NaN, T)
    sqrtH = sqrt.(H)
    montecarlo = da == "montecarlo"
    for t in 1:T
        obs = Vector{Float64}(Y[:, t])
        PSM, PSVSR, mean, S = gaussian_filter_bank(rf, obs, mean, S, Qc, Hc, H, opts, stream)
        (size(S) == (n, n) && _finite(S)) ||
            throw(NonlinearFilterError("gaussian_filter: the updated state variance has reduced rank or is not positive semi-definite (Dynare stops with a dimension error)"))
        SPt = montecarlo ? (S * stream_randn(stream, size(S, 2), N) .+ mean) : (mean .+ S * nodes2')
        proposal = probability2(mean, S, SPt)
        prior = probability2(PSM, PSVSR, SPt)
        like = probability2(obs, sqrtH, measurement_equations(SPt, rf))
        IW = like .* prior ./ proposal
        SWt = montecarlo ? IW ./ N : weights2 .* IW
        SWt = SWt .+ 1e-6
        s = _ssum(SWt)
        lik[t] = log(s)
        SWt ./= s
        if montecarlo && must_resample(opts, SWt, T)
            c = Vector{Float64}(undef, N); idx = Vector{Int}(undef, N)
            SPn = similar(SPt)
            resample_particles!(SPn, SPt, SWt, opts, stream, c, idx)
            SPt = SPn
            SWt = fill(1.0 / N, N)
        end
        mean = SPt * SWt
        temp = SPt .- mean
        S = Matrix(reduced_rank_cholesky((temp .* SWt') * temp')')
        _finite(S) || throw(NonlinearFilterError("reduced_rank_cholesky:: Input matrix is not semidefinite positive!"))
    end
    -_ssum(@view lik[start:end]), lik, false
end

"""
    nonlinear_kalman_filter(rf, Y, start, opts, stream) -> (LIK, lik, complex)

`nonlinear_kalman_filter.m` (`filter_algorithm = nlkf`): sigma-point (or Monte Carlo) Kalman filter; the density of period
t is the sigma-point mixture of N(y_t; y_i, H).
"""
function nonlinear_kalman_filter(rf::NLReducedForm, Y::AbstractMatrix{Float64}, start::Int, opts::ParticleOptions, stream::ParticleStream)
    opts.pruning && throw(NonlinearFilterError("nonlinear_kalman_filter: pruning is not yet implemented."))
    mf0, mf1 = rf.mf0, rf.mf1
    T = size(Y, 2)
    n, p, q = length(mf0), length(mf1), size(rf.Q, 1)
    nodes, w, wc = _proposal_nodes(opts, n + q, stream)
    H = _hmat(rf)
    Hc = _lower_chol(H, "H")
    Qc = _lower_chol(rf.Q, "Q")
    mean = copy(rf.StateVectorMean)
    S = _lower_chol(rf.StateVectorVariance, "the initial state variance")
    lik = fill(NaN, T)
    for t in 1:T
        tmp = _sigma_propagate(rf, mean, S, Qc, nodes)
        tS, tO = tmp[mf0, :], tmp[mf1, :]
        PSM = _wmean(tS, w)
        POM = _wmean(tO, w)
        y = Vector{Float64}(Y[:, t])
        if opts.proposal_approximation != "unscented"
            dState = (tS .- PSM)' .* sqrt(w)
            dObs = (tO .- POM)' .* sqrt(w)
            big = vcat(hcat(dObs, dState), hcat(Hc, zeros(p, n)))
            mat = Matrix(qr2_r(big)')
            POVSR = mat[1:p, 1:p]
            COSSR = mat[p+1:p+n, 1:p]
            S = mat[p+1:p+n, p+1:p+n]
            mean = PSM .+ (COSSR / LowerTriangular(POVSR)) * (y .- POM)
        else
            dState = tS .- PSM
            dObs = tO .- POM
            PSV = (dState .* wc') * dState'
            POV = (dObs .* wc') * dObs' + H
            PSOC = (dState .* wc') * dObs'
            K = PSOC / POV
            mean = PSM .+ K * (y .- POM)
            SVV = PSV - K * POV * K'
            Sn = _chol_lower_from_lower(SVV)
            Pc = _chol_lower_from_lower(POV)
            if Sn === nothing || Pc === nothing
                lik[t] = -Inf
                return -Inf, lik, false
            end
            S = Sn
        end
        s = _sdot(probability2(y, Hc, tO), w isa Number ? fill(w, size(tO, 2)) : w)
        if s < 0
            return NaN, lik, true                       # log of a negative number: Dynare's LIK is complex (info 46)
        end
        lik[t] = log(s)
    end
    -_ssum(@view lik[start:end]), lik, false
end

"Dynare's filter_algorithm names."
const NONLINEAR_FILTERS = Dict{String,Function}("sis" => sequential_importance_particle_filter, "apf" => auxiliary_particle_filter,
    "cpf" => conditional_particle_filter, "gf" => gaussian_filter, "nlkf" => nonlinear_kalman_filter)
