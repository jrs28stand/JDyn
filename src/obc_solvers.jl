# Solvers for the linear complementarity problem of occasionally binding constraints (docs/OBC_PROGRESS.md).
#
# Every solver has the signature of the shared contract (docs/OBC_PROGRESS.md, "Contract"):
#
#     solve(M, q; tol, kw...) -> (y, status, info)
#
# Find wedges y >= 0 with gap g = q + M y >= 0 and y .* g = 0. `status` is :ok, :no_solution, :horizon_too_short
# or :not_converged (the last two are also set by the caller, which checks the gap beyond the horizon). Every solver
# verifies its own answer with `obc_verify` before it returns :ok, and returns y = 0 at once when q >= 0.
#
# `minnorm` follows the formulation of MacroModelling.jl (Thore Kockerols, MIT licence; see NOTICE.md): the smallest
# wedges, in the 2-norm, that enforce the constraint and its complementarity over the horizon. MacroModelling.jl
# solves that programme with NLopt's SLSQP from y = 0; JDyn solves the same programme without an NLP solver:
# an exact solution of the complementarity problem by Lemke's complementary pivoting, then a descent over
# neighbouring complementary bases to a solution of locally smallest norm (see docs/OBC_PROGRESS.md for why).

"""
    obc_verify(M, q, y; tol=1e-10) -> (ok, max_violation)

The verification step every OBC solver runs before it reports `:ok`: `y >= -tol`, `q + M y >= -tol` and
`|y .* (q + M y)| <= tol`. `max_violation` is the largest of the three violations (0 when all hold exactly).
"""
function obc_verify(M::AbstractMatrix, q::AbstractVector, y::AbstractVector; tol::Real=1e-10)
    g = q .+ M * y
    v = 0.0
    for i in eachindex(y)
        v = max(v, -y[i], -g[i], abs(y[i] * g[i]))
    end
    (all(isfinite, y) && v <= tol, v)
end

# ---------------------------------------------------------------------- Lemke's algorithm
"""
    lemke(M, q; maxpivots=20*length(q)+100, piv_tol=1e-12) -> (z, status, pivots)

Lemke's complementary pivoting algorithm with covering vector `ones` and lexicographic ratio test (so it does not
cycle on degenerate problems): finds `z >= 0` with `w = q + M z >= 0`, `z' w = 0`. `status` is `:ok`, `:ray`
(secondary-ray termination: for the matrix classes Lemke's method is complete on, such as P-matrices and
copositive-plus matrices, this proves there is no solution) or `:maxpivots`. The returned `z` is recomputed from the
final complementary basis with one LU solve, which removes the rounding error accumulated over the pivots.
"""
function lemke(M::AbstractMatrix, q::AbstractVector; maxpivots::Integer=20 * length(q) + 100, piv_tol::Real=1e-12)
    n = length(q)
    all(>=(0), q) && return (zeros(n), :ok, 0)
    # tableau rows: w - M z - e z0 = q. Columns 1:n = w, n+1:2n = z, 2n+1 = z0; last column = rhs.
    # The w-columns of the current tableau hold the inverse of the basis (lexicographic ratio test).
    Tb = zeros(n, 2n + 2)
    for i in 1:n
        Tb[i, i] = 1.0
        for j in 1:n
            Tb[i, n+j] = -M[i, j]
        end
        Tb[i, 2n+1] = -1.0
        Tb[i, end] = q[i]
    end
    basis = collect(1:n)                      # basic variable of each row
    function pivot!(r, c)
        p = Tb[r, c]
        @views Tb[r, :] ./= p
        for i in 1:n
            i == r && continue
            f = Tb[i, c]
            f == 0 && continue
            @views Tb[i, :] .-= f .* Tb[r, :]
        end
        basis[r] = c
    end
    # z0 enters; the row with the most negative q leaves (first index on ties)
    r = argmin(q)
    leaving = basis[r]
    pivot!(r, 2n + 1)
    pivots = 1
    status = :maxpivots
    while pivots < maxpivots
        entering = leaving <= n ? leaving + n : leaving - n          # complement of the variable that left
        col = view(Tb, :, entering)
        # lexicographic minimum ratio over rows with a positive pivot element
        best = 0
        for i in 1:n
            col[i] > piv_tol || continue
            if best == 0
                best = i
                continue
            end
            # compare (rhs_i, Binv_i) / col_i with (rhs_b, Binv_b) / col_b lexicographically
            a, b = col[i], col[best]
            d = Tb[i, end] / a - Tb[best, end] / b
            if abs(d) <= 1e-13 * max(1.0, abs(Tb[i, end] / a))
                for j in 1:n
                    d = Tb[i, j] / a - Tb[best, j] / b
                    abs(d) > 1e-15 && break
                end
            end
            d < 0 && (best = i)
        end
        if best == 0
            status = :ray
            break
        end
        # prefer letting z0 leave when it ties for the minimum ratio (ends the algorithm)
        for i in 1:n
            if basis[i] == 2n + 1 && col[i] > piv_tol &&
               abs(Tb[i, end] / col[i] - Tb[best, end] / col[best]) <= 1e-13 * max(1.0, abs(Tb[best, end] / col[best]))
                best = i
            end
        end
        leaving = basis[best]
        pivot!(best, entering)
        pivots += 1
        if leaving == 2n + 1
            status = :ok
            break
        end
    end
    z = zeros(n)
    status == :ok || return (z, status, pivots)
    alpha = sort([basis[i] - n for i in 1:n if n < basis[i] <= 2n])
    if !isempty(alpha)
        F = lu(M[alpha, alpha]; check=false)
        if issuccess(F)
            z[alpha] = -(F \ q[alpha])
        else                                   # singular principal block: keep the tableau values
            for i in 1:n
                n < basis[i] <= 2n && (z[basis[i]-n] = Tb[i, end])
            end
        end
    end
    (z, :ok, pivots)
end

const OBC_PIVOT_RTOL = 1e-10

"""
The complementary solution for the index set `alpha` (y_alpha solves the gap equations, the rest is zero), or nothing. A block whose LU
has a pivot of at most `OBC_PIVOT_RTOL * scale` (`scale`: the largest entry of the whole M) is treated as singular: its "solution" is
rounding noise divided by a pivot (wedges of 1e14 that satisfy the complementarity conditions to rounding and so pass `obc_verify`).
"""
function _complementary_solution(M, q, alpha::Vector{Int}; scale::Real=maximum(abs, M))
    y = zeros(length(q))
    isempty(alpha) && return y
    F = lu(M[alpha, alpha]; check=false)
    issuccess(F) || return nothing
    minimum(abs, diag(F.U)) <= OBC_PIVOT_RTOL * scale && return nothing
    y[alpha] = -(F \ q[alpha])
    all(isfinite, y) ? y : nothing
end

# ---------------------------------------------------------------------- minnorm
"""
    solve_minnorm(M, q; tol=1e-10, maxpivots=..., descent=true, max_descent=50) -> (y, status, info)

MacroModelling.jl's formulation of occasionally binding constraints: minimise `‖y‖²` subject to `y >= 0`,
`q + M y >= 0` and complementarity (MacroModelling.jl imposes the last as `Σ y .* g <= 0`, which with the first
two is `y .* g = 0`), i.e. the smallest anticipated wedges that enforce the constraint over the horizon.

JDyn solves it in two exact steps instead of with an NLP solver: Lemke's algorithm finds a solution of the
complementarity problem; then, while a complementary basis that differs in one index from the current one gives a
feasible solution of smaller norm, it moves there. When the problem has a single solution (always the case when `M`
is a P-matrix) this is the solution of MacroModelling.jl's programme; otherwise it is a locally smallest one, and
`info.alternatives` counts the other solutions met in the neighbourhood (a sign of multiple equilibria).

`info`: `pivots`, `descent_steps`, `alternatives`, `max_violation` (of the verification step), `message`.
"""
function solve_minnorm(M::AbstractMatrix, q::AbstractVector; tol::Real=1e-10, maxpivots::Integer=20 * length(q) + 100,
                       descent::Bool=true, max_descent::Integer=50, kw...)
    n = length(q)
    size(M) == (n, n) || throw(DimensionMismatch("M is $(size(M)) but q has $n entries"))
    if all(>=(0), q)
        return (zeros(n), :ok, (pivots=0, descent_steps=0, alternatives=0, max_violation=0.0, message="q >= 0: constraints slack"))
    end
    y, st, piv = lemke(M, q; maxpivots=maxpivots)
    if st != :ok
        msg = st == :ray ? "Lemke's algorithm ended on a secondary ray: no solution of the complementarity problem (proof for P- and copositive-plus M)" :
                           "Lemke's algorithm did not finish in $maxpivots pivots"
        return (y, st == :ray ? :no_solution : :not_converged, (pivots=piv, descent_steps=0, alternatives=0, max_violation=Inf, message=msg))
    end
    ok, viol = obc_verify(M, q, y; tol=tol)
    ok || return (y, :not_converged, (pivots=piv, descent_steps=0, alternatives=0, max_violation=viol,
                                      message="the solution of Lemke's algorithm fails the verification step (max violation $viol)"))
    steps, alts = 0, 0
    if descent
        best = sum(abs2, y)
        alpha = findall(>(tol), y)
        improved = true
        while improved && steps < max_descent
            improved = false
            for i in 1:n
                cand = i in alpha ? setdiff(alpha, i) : sort(vcat(alpha, i))
                yc = _complementary_solution(M, q, cand)
                yc === nothing && continue
                okc, _ = obc_verify(M, q, yc; tol=tol)
                okc || continue
                isapprox(yc, y; atol=tol) && continue
                alts += 1
                nc = sum(abs2, yc)
                if nc < best - tol^2
                    y, best, alpha = yc, nc, findall(>(tol), yc)
                    improved = true
                    steps += 1
                    break
                end
            end
        end
    end
    ok, viol = obc_verify(M, q, y; tol=tol)
    (y, ok ? :ok : :not_converged, (pivots=piv, descent_steps=steps, alternatives=alts, solver=:minnorm, max_violation=viol,
                                    message=ok ? "" : "verification failed after the descent (max violation $viol)"))
end

# ---------------------------------------------------------------------- shared fast path
"""
    solve_active_set(M, q; tol=1e-10, maxit=60) -> (y, status, info)

The fast path shared by all methods (contract section 5, added 2026-11-24): guess-and-verify on the binding set, as
OccBin does on regimes. Start from the periods with `q < 0`, solve the square system on the set, drop negative
wedges, add periods whose gap is negative; repeat until nothing changes (verified with `obc_verify`), a set repeats or
`maxit` is reached (`:not_converged`: the caller then runs the method's own solver). No guarantee of success; when
the problem has several solutions, the one found need not be the method's (e.g. not the smallest in norm).
"""
function solve_active_set(M::AbstractMatrix, q::AbstractVector; tol::Real=1e-10, maxit::Integer=60, kw...)
    n = length(q)
    all(>=(0), q) && return (zeros(n), :ok, (pivots=0, iterations=0, solver=:none, max_violation=0.0, message="q >= 0: constraints slack"))
    S = q .< -tol
    scale = maximum(abs, M)
    seen = Set{BitVector}()
    for it in 1:maxit
        S in seen && break
        push!(seen, copy(S))
        alpha = findall(S)
        y = _complementary_solution(M, q, alpha; scale=scale)
        y === nothing && break
        g = q .+ M * y
        drop = S .& (y .< -tol)
        add = .!S .& (g .< -tol)
        if !any(drop) && !any(add)
            ok, v = obc_verify(M, q, y; tol=tol)
            return (y, ok ? :ok : :not_converged, (pivots=it, iterations=it, solver=:active_set, max_violation=v, message=ok ? "" : "active set failed verification"))
        end
        S = (S .& .!drop) .| add
    end
    (zeros(n), :not_converged, (pivots=0, iterations=0, solver=:active_set, max_violation=Inf, message="active-set iteration did not settle"))
end

# ---------------------------------------------------------------------- news
# The homogenised mixed-integer formulation of Holden's complementarity problem and the diagnostics on M follow
#   T. D. Holden, "Existence and uniqueness of solutions to dynamic models with occasionally binding constraints" and
#   "Computation of solutions to dynamic models with occasionally binding constraints",
# and the constraint set of DynareOBC's Core/InnerProblem/FormOptimizer.m (GPL-3.0-or-later). The code is written from those
# descriptions (and ported from pydyn's obc.py, GPL-3.0-or-later) for JDyn's own data structures; it is not a translation of
# DynareOBC. The P-matrix recursion is the one of M. J. Tsatsomeros and L. Li, "A recursive test for P-matrices", BIT 40 (2000).

"The check rows beyond the horizon: do they hold (`>= -tol`)? (`Mb`, `qb` may be `nothing`.)"
_beyond_ok(Mb, qb, y, tol) = Mb === nothing || isempty(qb) || minimum(qb .+ Mb * y) >= -tol

"""
    solve_news(M, q; tol=1e-10, Mb=nothing, qb=nothing, horizons=nothing, n_c=1, omega=1000, time_limit=nothing) -> (y, status, info)

Holden's method (DynareOBC): the complementarity problem `y >= 0, q + M y >= 0, y .* (q + M y) = 0` as a mixed-integer
programme, solved with HiGHS (`src/highs.jl`).

Homogenised formulation: maximise `alpha` subject to `0 <= yn <= z`, `alpha q + M yn >= 0` and `alpha q + M yn <= omega (1 - z)`
(on the complementarity rows) with `z` binary; then `y = yn / alpha`. `alpha > 0` at the optimum if and only if a solution
exists with wedges inside the horizon. `M` and `q` are scaled by their largest entries first (the big-M constants follow
from `q` and `M`, none is fixed). `Mb`, `qb` are extra rows (the gaps beyond the horizon): `qb + Mb y >= 0` is imposed too.
`horizons` lists the numbers of leading periods (per constraint) in which wedges are allowed, tried in order (default
`T/4, T/2, T`): the first that works is returned, so shorter binding spells are preferred.

After the integer programme the wedges are polished: with the binding set `z` the solver reports, the square system on that
set is solved exactly (this removes the solver's tolerance from the wedges), and the answer must pass `obc_verify` (and the
extra rows) before `:ok` is returned. Failure: `:horizon_too_short` when the problem without the extra rows has a solution,
`:no_solution` otherwise (`info.proof` is true when even the linear relaxation `y >= 0, q + M y >= 0` is infeasible),
`:not_converged` when HiGHS ran out of time (`time_limit`) without an answer.

`info`: `solver = :milp`, `horizon` (periods allowed), `polished`, `max_violation`, `milp_calls`, `message`.
"""
function solve_news(M::AbstractMatrix, q::AbstractVector; tol::Real=1e-10, Mb=nothing, qb=nothing, horizons=nothing, n_c::Integer=1,
                    omega::Real=1000.0, time_limit=nothing, kw...)
    n = length(q)
    size(M) == (n, n) || throw(DimensionMismatch("M is $(size(M)) but q has $n entries"))
    nb = Mb === nothing ? 0 : size(Mb, 1)
    mk(st, y, msg; kws...) = (y, st, (; solver=:milp, horizon=0, polished=false, max_violation=Inf, milp_calls=0, proof=false, message=msg, kws...))
    if all(>=(-tol), q) && (nb == 0 || all(>=(-tol), qb))
        return (zeros(n), :ok, (solver=:none, horizon=0, polished=true, max_violation=0.0, milp_calls=0, proof=false, message="q >= 0: constraints slack"))
    end
    T = n ÷ n_c
    hz = horizons === nothing ? sort(unique([max(1, T ÷ 4), max(1, T ÷ 2), T])) : collect(Int, horizons)
    qall = nb == 0 ? q : vcat(q, qb)
    Mall = nb == 0 ? Matrix{Float64}(M) : vcat(Matrix{Float64}(M), Matrix{Float64}(Mb))
    qn = maximum(abs, qall)
    msc = maximum(abs, Mall)
    (msc > 0 && qn > 0) || return mk(:no_solution, zeros(n), "M is zero and q is not non-negative")
    Mn, qs = Mall ./ msc, qall ./ qn
    nv = 2n + 1
    cobj = zeros(nv); cobj[n+1] = -1.0
    Id = Matrix{Float64}(I, n, n)
    A1 = hcat(Id, zeros(n, 1), -Id)                                         # yn - z <= 0
    A2 = hcat(Mn, qs, zeros(n + nb, n))                                     # alpha q + M yn >= 0 (every row)
    A3 = hcat(Mn[1:n, :], qs[1:n], omega .* Id)                             # alpha q_s + M_s yn <= omega (1 - z)
    A = vcat(A1, A2, A3)
    rlo = vcat(fill(-Inf, n), zeros(n + nb), fill(-Inf, n))
    rhi = vcat(zeros(n), fill(Inf, n + nb), fill(float(omega), n))
    integer = vcat(falses(n + 1), trues(n))
    timed_out, calls = false, 0
    for Tss in hz
        ub_y = zeros(n)
        for c in 1:n_c
            ub_y[(c-1)*T+1:(c-1)*T+min(Tss, T)] .= 1.0
        end
        lb = zeros(nv)
        ub = vcat(ub_y, 1e8, ub_y)
        st, x = highs_milp(cobj, A, rlo, rhi, lb, ub, integer; time_limit=time_limit, options=(mip_rel_gap=0.0,))
        calls += 1
        st == :time_limit && isempty(x) && (timed_out = true)
        isempty(x) && continue
        alpha = x[n+1]
        alpha <= 1e-9 && continue
        y = max.(x[1:n] ./ alpha, 0.0) .* (qn / msc)
        z = x[n+2:end] .> 0.5
        polished = false
        if any(z)                                    # polish: with the binding set known, the wedges solve a square system exactly
            F = lu(M[z, z]; check=false)
            if issuccess(F) && minimum(abs, diag(F.U)) > OBC_PIVOT_RTOL * maximum(abs, M)
                yp = zeros(n)
                yp[z] = -(F \ q[z])
                ok, _ = obc_verify(M, q, yp; tol=tol)
                if ok && _beyond_ok(Mb, qb, yp, tol)
                    y, polished = yp, true
                end
            end
        end
        ok, viol = obc_verify(M, q, y; tol=polished ? tol : max(tol, 1e-7))     # an unpolished MILP point is held to 1e-7 only
        ok && _beyond_ok(Mb, qb, y, polished ? tol : max(tol, 1e-7)) &&
            return (y, :ok, (solver=:milp, horizon=Tss, polished=polished, max_violation=viol, milp_calls=calls, proof=false, message=""))
    end
    timed_out && return mk(:not_converged, zeros(n), "HiGHS reached the time limit ($(time_limit) s) without a verified solution"; milp_calls=calls)
    if nb > 0                                         # is it the rows beyond the horizon that fail?
        y2, st2, _ = solve_news(M, q; tol=tol, horizons=hz, n_c=n_c, omega=omega, time_limit=time_limit)
        st2 == :ok && return mk(:horizon_too_short, y2, "a constraint still binds after the news horizon: increase obc_horizon"; milp_calls=calls)
    end
    # a proof of absence: no non-negative wedges keep the gap non-negative at all (linear programme)
    stl, _ = highs_lp(zeros(n), -vcat(Matrix{Float64}(M), nb > 0 ? Matrix{Float64}(Mb) : zeros(0, n)), fill(-Inf, n + nb), qall,
                      zeros(n), fill(Inf, n))
    proof = stl == :infeasible
    mk(:no_solution, zeros(n), proof ? "no solution returns to the reference regime within the news horizon: no non-negative wedges keep the gap " *
                                      "non-negative in every period (linear programme infeasible)" :
                                      "no solution within the news horizon (the integer programme is infeasible; the linear relaxation is feasible)";
       proof=proof, milp_calls=calls)
end

# ---------------------------------------------------------------------- diagnostics on M
"""
    _is_p_exact(A) -> Bool

P-matrix test by the recursion of Tsatsomeros and Li (2000): `A` is a P-matrix iff `a11 > 0` and both `A[2:end, 2:end]` and the
Schur complement of `a11` are P-matrices: about `2^n` small steps instead of `2^n` determinants.
"""
function _is_p_exact(A::AbstractMatrix)
    A[1, 1] > 0 || return false
    size(A, 1) == 1 && return true
    B = A[2:end, 2:end]
    _is_p_exact(B) || return false
    _is_p_exact(B .- (A[2:end, 1] * A[1, 2:end]') ./ A[1, 1])
end

"Reference implementation of the P-matrix test by enumeration of principal minors (tests only)."
function _is_p_minors(A::AbstractMatrix)
    n = size(A, 1)
    for mask in 1:(1<<n)-1
        idx = [i for i in 1:n if (mask >> (i - 1)) & 1 == 1]
        det(A[idx, idx]) > 0 || return false
    end
    true
end

"""
    m_diagnostics(Ms; n_c=1, max_exact=16) -> NamedTuple

Existence and uniqueness diagnostics on the square matrix `M` (Holden's theory paper):
- `is_s`: `M` is an S-matrix (some `y > 0` has `M y > 0`), a necessary condition for a solution to exist for every `q` (a linear programme);
- `is_p`: `true`, `false` or `nothing` (undecided). A P-matrix (all principal minors positive) gives exactly one solution for every `q`.
  Decided exactly (Tsatsomeros-Li recursion) up to `max_exact` rows; above that `true` only when the symmetric part is positive
  definite (sufficient), `false` when a leading or diagonal minor is not positive, otherwise `nothing`;
- `symmetric_part_positive_definite`, `min_diag`;
- `largest_p`: the largest `T'` (per constraint) for which the leading block is a P-matrix by the exact test, capped at `max_exact ÷ n_c`.
"""
function m_diagnostics(Ms::AbstractMatrix; n_c::Integer=1, max_exact::Integer=16)
    Ms = Matrix{Float64}(Ms)
    N = size(Ms, 1)
    T = N ÷ n_c
    # S-matrix: max t s.t. M y >= t, 0 <= y <= 1, t <= 1
    st, x = highs_lp(vcat(zeros(N), -1.0), hcat(-Ms, ones(N)), fill(-Inf, N), zeros(N), vcat(zeros(N), -Inf), vcat(ones(N), 1.0))
    is_s = st == :optimal && x[end] > 1e-9
    pd = all(>(0), eigvals(Symmetric((Ms .+ Ms') ./ 2)))
    block(t) = (idx = reduce(vcat, [collect((c-1)*T+1:(c-1)*T+t) for c in 1:n_c]); Ms[idx, idx])
    largest = 0
    for t in 1:min(T, max_exact ÷ n_c)
        _is_p_exact(block(t)) ? (largest = t) : break
    end
    if N <= max_exact
        is_p = largest == T
    elseif pd
        is_p = true
    elseif largest < min(T, max_exact ÷ n_c) || any(<=(0), diag(Ms)) || any(t -> det(block(t)) <= 0, 1:T)
        is_p = false
    else
        is_p = nothing
    end
    (is_s=is_s, is_p=is_p, symmetric_part_positive_definite=pd, largest_p=largest, min_diag=minimum(diag(Ms)), max_exact=max_exact)
end

const OBC_STATUSES = (:ok, :no_solution, :horizon_too_short, :not_converged)

"The solver of each `obc_method` other than `occbin`; every entry has the signature `solve(M, q; tol, kw...) -> (y, status, info)`."
const OBC_SOLVERS = Dict{Symbol,Function}(:minnorm => solve_minnorm, :news => solve_news)
