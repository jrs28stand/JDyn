# The full-average cubature of the higher-order news method (obc_cubature = true, obc_cubature_regions = all): one complementarity problem
#   y >= 0,  p + M y >= 0 (head rows, the first n_s = n_c T),  y'(p + M y) = 0,  and p + M y >= 0 on the check-ahead rows
# per relevant perturbed path p = q + R xi_j, averaged over all points. Written afresh for JDyn from pydyn's prototype
# (pydyn/obc_ho.py, solve_active_set_batch; the structure of DynareOBC's PerformCubature.m) and docs/OBC_HO_PERF.md of pydyn.
#
# The point solver is the scalar guess-and-verify iteration of `solve_active_set` (start: the periods with p < -tol; solve the square system on the
# set; drop negative wedges, add violated periods; stop on a repeated set), with the linear solves served from a cache of LU factorisations of
# M[A, A] keyed by the set A (M is fixed for a parameter set, so a set seen at one point or in one period is seen again at others). The sequence of sets
# per point is the one `solve_active_set` follows, hence the same solution when there are several. `cub_mode = :group` runs all points of a period in
# lockstep and solves the points that share a set in one multi-right-hand-side solve.

"One cached binding set: the indices, the LU of M[A, A] and whether the block passed the pivot guard (otherwise the iteration fails there, as `_complementary_solution`)."
struct CubEntry
    alpha::Vector{Int}
    F::LU{Float64,Matrix{Float64},Vector{Int}}
    ok::Bool
end

"""
    CubCache(cap_bytes)

LU factorisations by binding set (`Vector{UInt64}` of the packed set), memory-capped: when the estimate of the bytes held passes the cap the cache is
cleared (pydyn does the same). With one constraint the sets saturate (a few hundred to a few thousand); with several they hardly repeat.
"""
mutable struct CubCache
    d::Dict{Vector{UInt64},CubEntry}
    bytes::Int
    cap::Int
    hits::Int
    misses::Int
    clears::Int
end
CubCache(cap::Integer=256 * 2^20) = CubCache(Dict{Vector{UInt64},CubEntry}(), 0, Int(cap), 0, 0, 0)

"State of the full-average solver (one per `ObcHO`): mode, cache, work space and the failure-handling options."
mutable struct CubState
    mode::Symbol                       # :cache (default), :scalar (solve_active_set per point), :group (lockstep with multi-RHS solves)
    cache::CubCache
    S::Vector{Bool}
    y::Vector{Float64}
    g::Vector{Float64}
    rhs::Vector{Float64}
    key::Vector{UInt64}
    hist::Vector{UInt64}
    rel::Vector{Int}
    acc::Vector{Float64}
    pt::Vector{Float64}
    Mscale::Float64
    Sg::Matrix{Bool}                   # lockstep work space (mode :group)
    Hg::Matrix{UInt64}
    act::Vector{Int}
    act2::Vector{Int}
    k1::Vector{UInt64}
    k2::Vector{UInt64}
    Gbuf::Matrix{Float64}
    Bbuf::Matrix{Float64}
    Mabuf::Matrix{Float64}
    Pm::Matrix{Float64}
    Y::Matrix{Float64}
    ok::Vector{Bool}
    m_diag::Any                        # up-front diagnostic of M (`cub_m_diagnostic`), nothing when skipped
    milp_time_limit::Float64           # s per failed point (obc_cubature_milp_time_limit)
    tolerance::Float64                 # share of failed relevant points that is dropped instead of raised (obc_cubature_failure_tolerance)
    lp_head::Any                       # CubLP on the head rows / on all rows (built on first use)
    lp_full::Any
    dropped_last::Int
    relevant_last::Int
    qlog::Union{Nothing,Vector{Vector{Float64}}}      # when a vector: every gap path `q` passed to `ho_news_shocks` is appended (tests, cross-checks)
    undecided_tolerance::Float64       # share of a period's relevant points of class b or c that may be dropped (obc_cubature_undecided_tolerance)
    undecided_min_points::Int          # floor on that allowance in points (obc_cubature_undecided_min_points)
    dropped_cls::NTuple{3,Int}         # dropped points of the last period by class (a, b, c)
end
function CubState(M::AbstractMatrix, N::Integer)
    nws = size(M, 1)
    CubState(:cache, CubCache(), zeros(Bool, nws), zeros(nws), zeros(nws), zeros(nws), zeros(UInt64, cld(nws, 64)),
               UInt64[], Int[], zeros(nws), zeros(N), maximum(abs, M),
               zeros(Bool, nws, 0), zeros(UInt64, 120, 0), Int[], Int[], UInt64[], UInt64[], zeros(nws, 0), zeros(nws, 0), zeros(nws, nws),
               zeros(N, 0), zeros(nws, 0), Bool[], nothing, 0.1, 0.0, nothing, nothing, 0, 0, nothing, 0.005, 3, (0, 0, 0))
end

@inline function _pack!(key::Vector{UInt64}, S::Vector{Bool})
    fill!(key, 0)
    @inbounds for i in eachindex(S)
        S[i] && (key[(i-1)>>6+1] |= UInt64(1) << ((i - 1) & 63))
    end
    key
end

function _cub_entry!(c::CubCache, key::Vector{UInt64}, S::AbstractVector{Bool}, M::Matrix{Float64}, scale::Float64)
    e = get(c.d, key, nothing)
    if e !== nothing
        c.hits += 1
        return e
    end
    c.misses += 1
    alpha = findall(S)
    k = length(alpha)
    F = lu(M[alpha, alpha]; check=false)
    ok = k == 0 || (issuccess(F) && minimum(abs, diag(F.U)) > OBC_PIVOT_RTOL * scale)
    e = CubEntry(alpha, F, ok)
    sz = 8 * k * k + 8 * k + 160 + 8 * length(key)
    if c.bytes + sz > c.cap
        empty!(c.d); c.bytes = 0; c.clears += 1
    end
    c.d[copy(key)] = e
    c.bytes += sz
    e
end

"""
The scalar iteration at one point `p` (head rows `1:T`, `T` here is `n_s`), cached solves. Returns `true` and the wedges in `st.y` when it settles on a
verified solution, `false` otherwise (cycle, singular block, `maxit`, failed verification). `Mh` is the `n_s x n_s` block, `scale = max|Mh|`.
"""
function _cub_point!(st::CubState, Mh::Matrix{Float64}, scale::Float64, p::AbstractVector, ns::Int, tol::Float64, maxit::Int=60)
    S, y, g, rhs, key, hist = st.S, st.y, st.g, st.rhs, st.key, st.hist
    nw = length(key)
    allnn = true
    @inbounds for i in 1:ns
        p[i] < 0 && (allnn = false; break)
    end
    if allnn
        fill!(y, 0.0)
        return true
    end
    @inbounds for i in 1:ns
        S[i] = p[i] < -tol
    end
    empty!(hist)
    for it in 1:maxit
        _pack!(key, S)
        nh = length(hist) ÷ nw
        @inbounds for r in 1:nh                          # a repeated set: the iteration cycles
            same = true
            for w in 1:nw
                hist[(r-1)*nw+w] == key[w] || (same = false; break)
            end
            same && return false
        end
        append!(hist, key)
        e = _cub_entry!(st.cache, key, S, Mh, scale)
        e.ok || return false
        alpha = e.alpha
        k = length(alpha)
        fill!(y, 0.0)
        if k > 0
            @inbounds for a in 1:k
                rhs[a] = p[alpha[a]]
            end
            ldiv!(e.F, view(rhs, 1:k))
            @inbounds for a in 1:k
                v = -rhs[a]
                isfinite(v) || return false
                y[alpha[a]] = v
            end
        end
        @inbounds for i in 1:ns
            g[i] = p[i]
        end
        @inbounds for a in 1:k
            j = alpha[a]
            yj = y[j]
            for i in 1:ns
                g[i] += Mh[i, j] * yj
            end
        end
        changed = false
        @inbounds for i in 1:ns
            if S[i]
                if y[i] < -tol
                    S[i] = false; changed = true
                end
            elseif g[i] < -tol
                S[i] = true; changed = true
            end
        end
        if !changed
            v = 0.0
            @inbounds for i in 1:ns
                v = max(v, -y[i], -g[i], abs(y[i] * g[i]))
            end
            return v <= tol
        end
    end
    false
end

# ------------------------------------------------------------------------------------------------ lockstep over the points of a period
"""
    _cub_group!(st, Mh, scale, Pm, ns, tol, Y, ok, maxit=60)

The iteration of `_cub_point!` for all columns of `Pm` (`N x m`, head rows `1:ns`) in lockstep: in every pass the active columns are sorted by their set
(the first two words of the packed key; sets of more than 128 periods are not grouped by this routine), the columns that share a set are solved with one
multi-right-hand-side solve against the cached factorisation, and each column then drops / adds periods exactly as in the scalar iteration. A column's
sequence of sets is the scalar one, hence its solution. `Y` (`ns x m`) receives the wedges, `ok[j]` says whether column `j` settled on a verified solution.
"""
function _cub_group!(st::CubState, Mh::Matrix{Float64}, scale::Float64, Pm::AbstractMatrix, ns::Int, tol::Float64,
                     Y::Matrix{Float64}, ok::Vector{Bool}, maxit::Int=60)
    m = size(Pm, 2)
    nw = cld(ns, 64)
    nw <= 2 || throw(ArgumentError("cub_mode = :group handles at most 128 wedge periods"))
    S = st.Sg
    size(S, 2) < m && (st.Sg = S = zeros(Bool, ns, m))
    size(st.Gbuf, 2) < m && (st.Gbuf = zeros(ns, m); st.Bbuf = zeros(ns, m))
    H = st.Hg
    size(H, 2) < m && (st.Hg = H = zeros(UInt64, 2 * maxit, m))
    fill!(ok, false)
    act = st.act; empty!(act)
    fill!(Y, 0.0)
    @inbounds for c in 1:m
        allnn = true
        for i in 1:ns
            Pm[i, c] < 0 && (allnn = false; break)
        end
        if allnn
            ok[c] = true
        else
            for i in 1:ns
                S[i, c] = Pm[i, c] < -tol
            end
            push!(act, c)
        end
    end
    keys1, keys2 = st.k1, st.k2
    length(keys1) < m && (resize!(keys1, m); resize!(keys2, m))
    key = st.key
    for pass in 1:maxit
        isempty(act) && break
        # keys and the cycle test
        nxt = 0
        @inbounds for c in act
            fill!(key, 0)
            for i in 1:ns
                S[i, c] && (key[(i-1)>>6+1] |= UInt64(1) << ((i - 1) & 63))
            end
            k1 = key[1]; k2 = nw > 1 ? key[2] : UInt64(0)
            cyc = false
            for r in 1:pass-1
                if H[(r-1)*nw+1, c] == k1 && (nw == 1 || H[(r-1)*nw+2, c] == k2)
                    cyc = true; break
                end
            end
            if !cyc
                H[(pass-1)*nw+1, c] = k1
                nw > 1 && (H[(pass-1)*nw+2, c] = k2)
                keys1[c] = k1; keys2[c] = k2
                nxt += 1
                act[nxt] = c
            end
        end
        resize!(act, nxt)
        isempty(act) && break
        sort!(act; by=c -> (keys1[c], keys2[c]))
        newact = st.act2; empty!(newact)
        a0 = 1
        while a0 <= length(act)
            c0 = act[a0]
            a1 = a0
            while a1 < length(act) && keys1[act[a1+1]] == keys1[c0] && keys2[act[a1+1]] == keys2[c0]
                a1 += 1
            end
            cols = view(act, a0:a1)
            gsz = a1 - a0 + 1
            fill!(key, 0)
            key[1] = keys1[c0]; nw > 1 && (key[2] = keys2[c0])
            Sc = view(S, :, c0)
            e = _cub_entry!(st.cache, key, Sc, Mh, scale)       # failed block: the columns fail
            if e.ok
                alpha = e.alpha
                k = length(alpha)
                G = view(st.Gbuf, 1:ns, 1:gsz)
                @inbounds for (jj, c) in enumerate(cols), i in 1:ns
                    G[i, jj] = Pm[i, c]
                end
                if k > 0
                    B = view(st.Bbuf, 1:k, 1:gsz)
                    @inbounds for (jj, c) in enumerate(cols), a in 1:k
                        B[a, jj] = Pm[alpha[a], c]
                    end
                    ldiv!(e.F, B)
                    @inbounds for jj in 1:gsz, a in 1:k
                        B[a, jj] = -B[a, jj]
                    end
                    Ma = view(st.Mabuf, 1:ns, 1:k)
                    @inbounds for a in 1:k, i in 1:ns
                        Ma[i, a] = Mh[i, alpha[a]]
                    end
                    mul!(G, Ma, B, 1.0, 1.0)                  # g = p + M[:, A] y_A
                end
                @inbounds for (jj, c) in enumerate(cols)
                    fin = true
                    for a in 1:k
                        isfinite(st.Bbuf[a, jj]) || (fin = false; break)
                    end
                    fin || continue                           # singular-like: the column fails
                    for i in 1:ns
                        Y[i, c] = 0.0
                    end
                    for a in 1:k
                        Y[alpha[a], c] = st.Bbuf[a, jj]
                    end
                    changed = false
                    for i in 1:ns
                        if S[i, c]
                            if Y[i, c] < -tol
                                S[i, c] = false; changed = true
                            end
                        elseif G[i, jj] < -tol
                            S[i, c] = true; changed = true
                        end
                    end
                    if changed
                        push!(newact, c)
                    else
                        v = 0.0
                        for i in 1:ns
                            v = max(v, -Y[i, c], -G[i, jj], abs(Y[i, c] * G[i, jj]))
                        end
                        ok[c] = v <= tol
                    end
                end
            end
            a0 = a1 + 1
        end
        empty!(act); append!(act, newact)
    end
    nothing
end


# ------------------------------------------------------------------------------------------------ options
"`obc_m_diagnostic`: `:s` (S-matrix test, default), `:full` (also the exact P-matrix tests) or `false` (skip)."
function _obc_mdiag(x)
    v = x isa AbstractArray && length(x) == 1 ? x[1] : x
    v === false && return false
    v isa Real && return v == 0 ? false : v == 1 ? :s : throw(ArgumentError("obc_m_diagnostic must be s, full or false (got $x)"))
    v === true && return :s
    s = lowercase(strip(string(v)))
    s in ("s", "true") && return :s
    s == "full" && return :full
    s in ("false", "off", "none", "no") && return false
    throw(ArgumentError("obc_m_diagnostic must be s, full or false (got $x)"))
end

"`obc_cubature_milp_time_limit` (seconds, >= 0), `obc_cubature_failure_tolerance` and `obc_cubature_undecided_tolerance` (shares in [0, 1)), `obc_cubature_undecided_min_points` (integer >= 0)."
function _obc_float_option(x, name::AbstractString; lo::Real, hi::Real=Inf, open_hi::Bool=false)
    v = x isa AbstractArray && length(x) == 1 ? x[1] : x
    v isa AbstractString && (v = tryparse(Float64, v))
    ok = v isa Real && isfinite(v) && v >= lo && (open_hi ? v < hi : v <= hi)
    ok || throw(ArgumentError("$name must be a number in [$lo, $hi" * (open_hi ? ") " : "] ") * "(got $x)"))
    Float64(v)
end

# ------------------------------------------------------------------------------------------------ up-front diagnostic of M
const _CUB_DIAG_CACHE = Dict{Tuple{UInt,Int,Symbol},Any}()

"""
    cub_m_diagnostic(M, n_c; level=:s) -> NamedTuple | nothing

Existence diagnostic of the complementarity block of `M` when cubature is on (`m_diagnostics`, reused): `is_s` (S-matrix: some y > 0 has M y > 0, necessary for a
solution to exist for every q). Level `:s` runs the S-matrix linear programme and one leading P-block per constraint (`is_p = nothing`: not decided); `:full`
also runs the exact P-matrix tests (hundreds of ms at T = 32); `false` skips (returns `nothing`). Cached per parameter set (key: hash of `M`). One-way: a non-S matrix
means some paths have no solution; an S-matrix does not guarantee that every path has one (the borrowing-constraint example is S and still has failing points).
The named tuple adds `seconds` and `level`.
"""
function cub_m_diagnostic(M::AbstractMatrix, n_c::Integer; level=:s)
    level === false && return nothing
    key = (hash(M), Int(n_c), level)
    get!(_CUB_DIAG_CACHE, key) do
        length(_CUB_DIAG_CACHE) > 32 && empty!(_CUB_DIAG_CACHE)
        t = @elapsed d = m_diagnostics(M; n_c=n_c, max_exact=level == :full ? 16 : n_c)
        d = level == :full ? d : merge(d, (is_p=nothing,))
        merge(d, (seconds=t, level=level))
    end
end

# ------------------------------------------------------------------------------------------------ a native LP kept per parameter set
"""
    CubLP(M)

`min s  s.t.  y >= 0, s >= 0, M y + s 1 >= -p` as one native HiGHS model, built once from the fixed `M` (`m x k`); a point only changes the row lower bounds
(`-p`), and HiGHS keeps the basis from the previous point (warm start). `s* = 0` iff some `y >= 0` has `p + M y >= 0`. This is the LP of pydyn's `_lp_gap`, without
rebuilding it per point (pydyn's tolerance mode spends 2.3-5 ms per failed point on the call overhead of a freshly built `linprog`).
"""
mutable struct CubLP
    ptr::Ptr{Cvoid}
    m::Int
    k::Int
    lo::Vector{Float64}
    hi::Vector{Float64}
    calls::Int
end
function CubLP(M::AbstractMatrix)
    m, k = size(M)
    ptr = ccall((:Highs_create, libhighs), Ptr{Cvoid}, ())
    ptr == C_NULL && error("HiGHS: Highs_create failed")
    ccall((:Highs_setBoolOptionValue, libhighs), Cint, (Ptr{Cvoid}, Cstring, Cint), ptr, "output_flag", 0)
    ccall((:Highs_setBoolOptionValue, libhighs), Cint, (Ptr{Cvoid}, Cstring, Cint), ptr, "log_to_console", 0)
    start, idx, val = Cint[0], Cint[], Float64[]
    for j in 1:k+1
        for i in 1:m
            a = j <= k ? Float64(M[i, j]) : 1.0
            if a != 0
                push!(idx, i - 1); push!(val, a)
            end
        end
        push!(start, length(idx))
    end
    cost = vcat(zeros(k), 1.0)
    r = ccall((:Highs_passLp, libhighs), Cint,
              (Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
               Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
              ptr, k + 1, m, length(val), 1, 1, 0.0, cost, zeros(k + 1), fill(Inf, k + 1), fill(-Inf, m), fill(Inf, m), start, idx, val)
    r < 0 && (ccall((:Highs_destroy, libhighs), Cvoid, (Ptr{Cvoid},), ptr); error("HiGHS: Highs_passLp failed"))
    lp = CubLP(ptr, m, k, zeros(m), fill(Inf, m), 0)
    finalizer(l -> ccall((:Highs_destroy, libhighs), Cvoid, (Ptr{Cvoid},), l.ptr), lp)
end

"`s*` of the LP at `p` (its first `m` entries), or `nothing` when HiGHS does not return an optimal status."
function _cublp_gap(lp::CubLP, p::AbstractVector)
    @inbounds for i in 1:lp.m
        lp.lo[i] = -p[i]
    end
    ccall((:Highs_changeRowsBoundsByRange, libhighs), Cint, (Ptr{Cvoid}, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}), lp.ptr, 0, lp.m - 1, lp.lo, lp.hi)
    ccall((:Highs_run, libhighs), Cint, (Ptr{Cvoid},), lp.ptr)
    lp.calls += 1
    Int(ccall((:Highs_getModelStatus, libhighs), Cint, (Ptr{Cvoid},), lp.ptr)) == 7 || return nothing
    Float64(ccall((:Highs_getObjectiveValue, libhighs), Cdouble, (Ptr{Cvoid},), lp.ptr))
end

# ------------------------------------------------------------------------------------------------ failure rule
"Where the user guide of the failure rule lives (named in the warning and in the error)."
const CUBATURE_FAILURES_PAGE = "docs/OBC_CUBATURE_FAILURES.md"

"""
    cubature_limits(nrel, failure_tolerance, undecided_tolerance, min_points=3) -> (a_lim, total_lim)

The rule of one period with `nrel` relevant points (pydyn's `cubature_limits`): the period is acceptable iff `n_a <= a_lim` and `n_a + n_b + n_c <= total_lim`, where
`a_lim = failure_tolerance * nrel`, `u = max(undecided_tolerance * nrel, min_points)` if `undecided_tolerance > 0` (else 0) and `total_lim = max(a_lim, u)`.
With `undecided_tolerance = 0` this is the strict rule: no failed point unless `failure_tolerance > 0`.
"""
function cubature_limits(nrel::Integer, failure_tolerance::Real, undecided_tolerance::Real, min_points::Integer=3)
    a_lim = failure_tolerance * nrel
    u = undecided_tolerance > 0 ? max(undecided_tolerance * nrel, float(min_points)) : 0.0
    a_lim, max(a_lim, u)
end

"""
    cubature_warning(tolerance, min_points, dropped_cls, relevant, share_max, nperiods, S)

The single warning of a simulation that dropped failed cubature points: totals by class, the largest share in one period, the tolerance and floor, advice per class and the pointer to the user guide.
"""
function cubature_warning(utol::Real, min_points::Integer, cls::NTuple{3,Int}, relevant::Integer, share_max::Real, nper::Integer, S::Integer)
    a, b, c = cls
    tot = a + b + c
    "cubature: $tot of $relevant relevant perturbed points ($(round(100 * tot / max(1, relevant); digits=3))%) were dropped in $nper of $S periods " *
    "(by class: a=$a proven no solution, b=$b horizon too short, c=$c undecided; largest share in one period $(round(100 * share_max; digits=2))%; " *
    "allowed: $(100 * utol)% of a period's relevant points, never fewer than $min_points). " *
    "Classes (b) and (c) are tail points that the solver could not settle: they are NOT proven to have no solution. The remaining points are reweighted " *
    "by 1/(points - dropped), which biases the average slightly towards benign states. To reduce them: a longer `obc_horizon` for (b); a larger " *
    "`obc_cubature_milp_time_limit` for (c). `obc_cubature_undecided_tolerance = 0` makes any failed point an error. See $CUBATURE_FAILURES_PAGE."
end

# ------------------------------------------------------------------------------------------------ failure payload
"""
    CubatureFailure

The payload of the `ObcError` raised for a cubature point without a usable solution (`e.detail`): `period`, `n_failed` (a lower bound when `lower_bound`: the search stopped
at the first failure or when the tolerance was exceeded), `n_relevant`, `n_points`, `counts` = `(a, b, c)` of the examined points, `n_active_set_failed` (points on which the
scalar iteration failed over the whole period: an upper bound on the failures when all were examined; `nothing` for regions modes), `tolerance`, `advice`.
Classes: (a) proved no solution at this horizon; (b) the check-ahead rows cannot be met with wedges inside the horizon (a longer horizon is advised: this does NOT prove that one
solves); (c) undecided at the MILP time limit.
"""
struct CubatureFailure
    status::Symbol
    period::Int
    n_failed::Int
    lower_bound::Bool
    n_relevant::Int
    n_points::Int
    counts::NamedTuple{(:a, :b, :c),NTuple{3,Int}}
    n_active_set_failed::Union{Nothing,Int}
    tolerance::Float64
    milp_time_limit::Float64
    advice::Vector{String}
    undecided_tolerance::Float64
end

function _cub_failure(period, counts, nfail, unexamined, nrel, npts, n_as, tol, limit, utol=0.0)
    a, b, c = counts.a, counts.b, counts.c
    advice = String[]
    a > 0 && push!(advice, "(a) PROVEN no solution: the model has states that agents give weight to (through the shocks they expect) from which no equilibrium " *
                           "exists. This is a property of the model or its calibration, not of the solver; averaging over fewer points would hide it. " *
                           "Raise obc_cubature_failure_tolerance only to drop such points knowingly")
    b > 0 && push!(advice, "(b) horizon too short: the bound would still bind after the news horizon; try a longer obc_horizon (not proved to help)")
    c > 0 && push!(advice, "(c) undecided: the integer programme did not settle the point in $(limit) s, or found a regime sequence that fails OccBin's test; " *
                           "try a larger obc_cubature_milp_time_limit (this cannot make a point worse)")
    push!(advice, "obc_cubature_undecided_tolerance is the share of relevant points in a period that may be (b) or (c) and dropped; see $CUBATURE_FAILURES_PAGE")
    status = a > 0 ? :no_solution : b > 0 ? :horizon_too_short : :not_converged
    lb = unexamined > 0
    msg = "cubature: $(lb ? "at least " : "")$nfail of $nrel relevant perturbed points ($npts in total) have no usable solution: a=$a b=$b c=$c" *
          (tol > 0 ? "; obc_cubature_failure_tolerance=$tol" : "") * (utol > 0 ? "; obc_cubature_undecided_tolerance=$utol" : "") * ". " * join(advice, " ")
    ObcError(status, period, msg, CubatureFailure(status, period, nfail, lb, nrel, npts, counts, n_as, tol, limit, advice, utol))
end
