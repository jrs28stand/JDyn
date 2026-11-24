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
end
function CubState(M::AbstractMatrix, N::Integer)
    nws = size(M, 1)
    CubState(:cache, CubCache(), zeros(Bool, nws), zeros(nws), zeros(nws), zeros(nws), zeros(UInt64, cld(nws, 64)),
               UInt64[], Int[], zeros(nws), zeros(N), maximum(abs, M))
end

@inline function _pack!(key::Vector{UInt64}, S::Vector{Bool})
    fill!(key, 0)
    @inbounds for i in eachindex(S)
        S[i] && (key[(i-1)>>6+1] |= UInt64(1) << ((i - 1) & 63))
    end
    key
end

function _cub_entry!(c::CubCache, key::Vector{UInt64}, S::Vector{Bool}, M::Matrix{Float64}, scale::Float64)
    e = get(c.d, key, nothing)
    if e !== nothing
        c.hits += 1
        return e
    end
    c.misses += 1
    alpha = findall(S)
    k = length(alpha)
    F = lu(M[alpha, alpha]; check=false)
    ok = issuccess(F) && minimum(abs, diag(F.U)) > OBC_PIVOT_RTOL * scale
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
