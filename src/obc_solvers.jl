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

"The complementary solution for the index set `alpha` (y_alpha solves the gap equations, the rest is zero), or nothing."
function _complementary_solution(M, q, alpha::Vector{Int})
    y = zeros(length(q))
    isempty(alpha) && return y
    F = lu(M[alpha, alpha]; check=false)
    issuccess(F) || return nothing
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
    (y, ok ? :ok : :not_converged, (pivots=piv, descent_steps=steps, alternatives=alts, max_violation=viol,
                                    message=ok ? "" : "verification failed after the descent (max violation $viol)"))
end

# ---------------------------------------------------------------------- news (not implemented yet)
"""
    solve_news(M, q; kw...) -> (y, status, info)

PLACEHOLDER for Tom Holden's DynareOBC method (the LCP as a mixed-integer programme with big-M constants derived from
`q` and `M`, plus the existence/uniqueness diagnostics on `M`). It is being written first in pydyn and will be ported
here: it must keep this signature, return `y`, a status in `OBC_STATUSES` and an `info` named tuple (put the
diagnostics on `M` in it), and run `obc_verify` before returning `:ok`. Register nothing else: `OBC_SOLVERS[:news]`
already points here. See docs/OBC_PROGRESS.md, "Lessons for the other port".
"""
function solve_news(M::AbstractMatrix, q::AbstractVector; kw...)
    throw(ErrorException("obc_method = news is not implemented yet in JDyn (it is being written in pydyn first); " *
                         "use obc_method = minnorm or occbin"))
end

const OBC_STATUSES = (:ok, :no_solution, :horizon_too_short, :not_converged)

"The solver of each `obc_method` other than `occbin`; every entry has the signature `solve(M, q; tol, kw...) -> (y, status, info)`."
const OBC_SOLVERS = Dict{Symbol,Function}(:minnorm => solve_minnorm, :news => solve_news)
