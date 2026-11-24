# The general regime formulation of `obc_method = news`: constraints that switch several equations, or whose conditions are not
# the mirror image of the switched equation (docs/OBC_PROGRESS.md, "State (news port)").
#
# Each equation that a constraint switches gets its own wedge on its reference version. In a period in which the constraint is in
# its alternative regime the wedges are whatever makes the alternative versions hold; otherwise they are zero. A regime sequence is
# accepted by OccBin's own test: the bind condition is false wherever the reference regime is assumed and the relax condition is
# false wherever the alternative regime is assumed (conditions evaluated in levels). So the answer is OccBin's piecewise-linear
# solution, found differently: the responses to wedges are computed once per parameter set, a guess costs one small linear
# solve, and the guess is updated as OccBin does. When the iteration cycles, the fallbacks are (1) for one constraint with
# several equations whose wedges are tied, the equivalent complementarity problem (`_find_tie`, `lcp`, `_solve_lcp`), which also
# proves the absence of a solution when its linear relaxation is infeasible, and (2) a big-M integer programme over regime
# indicators (`_milp`), with a time limit.
#
# Ported from pydyn/obc.py (`RegimeOBC`, GPL-3.0-or-later); the homogenised integer programme of the LCP is Holden's (see
# obc_solvers.jl, NOTICE.md).

# ---------------------------------------------------------------------- pieces of the build (ported from pydyn)
"""
Rows `r_t` (t = 0..N, as `R[t+1]`) of the response of the linear functional `aC y(t-1) + aB y(t) + aA y(t+1)` to a state carried
forward by `P`: `r_0 = aB + aA P`, `r_1 = aC + aB P + aA P^2`, `r_{t+1} = r_t P`. `aC`, `aB`, `aA` are `m x n`.
"""
function _functional_rows(P::AbstractMatrix, aC, aB, aA, N::Int)
    R = Vector{Matrix{Float64}}(undef, N + 1)
    R[1] = aB + aA * P
    if N >= 1
        r = aC + aB * P + (aA * P) * P
        R[2] = r
        for t in 3:N+1
            r = r * P
            R[t] = r
        end
    end
    R
end

"Columns `G^s F e_w` (wedge block `w`, news period `s`): the response in period 0 of the wedge announced for period `s`. `n x (nw T)`, column `(w-1) T + s + 1`."
function _wedge_vectors(G::AbstractMatrix, FE::AbstractMatrix, T::Int)
    n, nw = size(FE)
    V = Matrix{Float64}(undef, n, nw * T)
    v = Matrix{Float64}(FE)
    for s in 0:T-1
        for w in 1:nw
            V[:, (w-1)*T+s+1] = view(v, :, w)
        end
        v = G * v
    end
    V
end

"""
`M[(i, t), (w, s)]` = response of functional `i` in period `t` (0-based, `< rows`) to a unit wedge `w` announced for period `s`
(row `(i-1) rows + t + 1`, column `(w-1) T + s + 1`). The response of the path to a wedge at `s + 1` equals the response to a
wedge at `s` shifted by one period plus the free response `P^t v_{s+1}`, hence along each diagonal
`M[t, s+1] = M[t-1, s] + r_t v_{s+1}` (`t >= 1`) and `M[0, s+1] = a_A v_s + r_0 v_{s+1}`; first column `M[t, 0] = r_t v_0`.
One matrix product per period for `R V`, then vector additions: no `n x n` product per `(t, s)` pair.
"""
function _response_matrix(R::Vector{Matrix{Float64}}, V::AbstractMatrix, aA::AbstractMatrix, rows::Int, T::Int)
    m = size(R[1], 1)
    nw = size(V, 2) ÷ T
    AV = aA * V
    Mt = Vector{Matrix{Float64}}(undef, rows)          # Mt[t+1]: m x (nw T)
    for t in 0:rows-1
        RV = R[t+1] * V
        if t == 0
            for w in 1:nw
                c0 = (w - 1) * T
                T > 1 && (RV[:, c0+2:c0+T] .+= view(AV, :, c0+1:c0+T-1))
            end
        else
            prev = Mt[t]
            for w in 1:nw
                c0 = (w - 1) * T
                T > 1 && (RV[:, c0+2:c0+T] .+= view(prev, :, c0+1:c0+T-1))
            end
        end
        Mt[t+1] = RV
    end
    out = Matrix{Float64}(undef, m * rows, nw * T)
    for t in 1:rows, i in 1:m
        out[(i-1)*rows+t, :] = view(Mt[t], i, :)
    end
    out
end

"Responses of the path to every unit wedge: `Y[t+1]` is `n x (nw T)` (period `t`, 0 <= t <= N), by one forward recursion over `t`."
function _full_responses(P::AbstractMatrix, V::AbstractMatrix, N::Int, T::Int)
    n = size(P, 1)
    nwT = size(V, 2)
    nw = nwT ÷ T
    out = Vector{Matrix{Float64}}(undef, N + 1)
    Y = zeros(n, nwT)
    for t in 0:N
        Y = P * Y
        if t < T
            for w in 1:nw
                c0 = (w - 1) * T
                Y[:, c0+t+1:c0+T] .+= view(V, :, c0+1:c0+T-t)
            end
        end
        out[t+1] = copy(Y)
    end
    out
end

"""
Path `y_0..y_upto` (`upto + 1` rows) of `y_t = P y_{t-1} + u_t` with `u_t = sum_d G^d F E w_{t+d}` (wedges in periods `< T`), starting
from `init_term` (= `P y_{-1}` + the shock term) in period 0; `w` has `nw T` entries, block-major.
"""
function _wedge_path(P::AbstractMatrix, G::AbstractMatrix, FE::AbstractMatrix, init_term::AbstractVector, w::AbstractVector, T::Int, upto::Int)
    n, nw = size(FE)
    H = zeros(n, T)
    h = zeros(n)
    for t in T:-1:1
        hn = G * h
        for b in 1:nw
            wb = w[(b-1)*T+t]
            wb == 0 || (hn .+= view(FE, :, b) .* wb)
        end
        h = hn
        H[:, t] = h
    end
    Y = Matrix{Float64}(undef, upto + 1, n)
    y = init_term .+ view(H, :, 1)
    Y[1, :] = y
    for t in 1:upto
        y = P * y
        t < T && (y .+= view(H, :, t + 1))
        Y[t+1, :] = y
    end
    Y
end

_obc_median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2)

# relative-scale verification of the LCP (all rows >= 0, complementarity on the first n_s rows), as pydyn's verify_lcp
function _verify_lcp(M, q, y, n_s, tol)
    g = q .+ M * y
    scale = max(1.0, maximum(abs, q))
    all(>=(-tol * scale), y) && all(>=(-tol * scale), g) || return false
    for i in 1:n_s
        min(abs(y[i]), abs(g[i])) <= tol * scale || return false
    end
    true
end

# ---------------------------------------------------------------------- the object
"""
    RegimeOBC(ob; horizon=40, check_ahead=40)

The regime formulation for the model of `ob`: `horizon` periods in which wedges may be non-zero, `check_ahead` further periods over
which the regimes are checked (`N = horizon + check_ahead` in all). Refuses (`ModelError`) a model in which an equation's
version depends on both constraints jointly. Built per `Occbin` object and `(horizon, N)` (see `regime_obc`).
"""
mutable struct RegimeOBC
    ob::Occbin
    T::Int
    N::Int
    nc::Int
    cols::Vector{Tuple{Int,Int}}          # (constraint, equation) of each wedge block
    alt_rows::Matrix{Float64}             # nb x (3n + nx + 1): alternative version of each switched equation [C B A J | D]
    ref_rows::Matrix{Float64}             # the reference versions, same layout
    cv::Vector{Int}                       # variables read by the bind/relax conditions
    jb::Vector{Int}                       # constraint of each wedge block
    G::Matrix{Float64}
    FE::Matrix{Float64}
    V0::Matrix{Float64}                   # n x (nb T): response of the next state to the wedges
    R::Matrix{Float64}                    # (nb T) x (nb T): response of the alternative residuals to the wedges
    Am::Matrix{Float64}                   # (nb T) x (n + nx): alternative residuals' dependence on [state; shock]
    a0::Vector{Float64}
    Wcv::Matrix{Float64}                  # (k N) x (nb T), row (t-1) k + i: condition variable i in period t
    Bcv::Matrix{Float64}                  # (k N) x (n + nx)
    st::Vector{Int}                       # state columns that matter
    Ps::Matrix{Float64}
    Ax::Matrix{Float64}
    Bx::Matrix{Float64}
    zbuf::Vector{Float64}
    cache::Dict{Symbol,Any}               # lazily built: Yw, Yb, tie, cond
end
Base.show(io::IO, r::RegimeOBC) = print(io, "RegimeOBC(T = ", r.T, ", N = ", r.N, ", ", r.nc, " constraint(s), ", length(r.cols), " wedge block(s))")

const _REGIME_CACHE = WeakKeyDict{Occbin,Dict{Tuple{Int,Int},RegimeOBC}}()
const _REGIME_SETUP = WeakKeyDict{Occbin,Any}()
const REGIME_MILP_TIME_LIMIT = Ref(30.0)       # seconds, for the integer programmes of the fallbacks

"The switched equations of each constraint, the reference and alternative rows, and the refusal of joint dependence."
function _regime_setup(ob::Occbin)
    get!(_REGIME_SETUP, ob) do
        n, nx, k = ob.n, ob.n_exo, ob.k
        ref_all = hcat(regime_matrices(ob, ntuple(_ -> false, k))...)
        cols, alt_rows, ref_rows = Tuple{Int,Int}[], Vector{Float64}[], Vector{Float64}[]
        per_reg = Tuple{Matrix{Float64},Vector{Int}}[]
        for j in 1:k
            alt_all = hcat(regime_matrices(ob, ntuple(i -> i == j, k))...)
            scale = [max(maximum(abs, view(ref_all, r, :)), maximum(abs, view(alt_all, r, :)), 1.0) for r in 1:n]
            rows = [r for r in 1:n if maximum(abs, view(alt_all, r, :) .- view(ref_all, r, :)) > 1e-12 * scale[r]]
            isempty(rows) && throw(ModelError("obc_method = news: constraint '$(ob.names[j])': no equation changes between the regimes. Use obc_method = occbin for this model."))
            push!(per_reg, (alt_all, rows))
            for r in rows
                push!(cols, (j, r)); push!(alt_rows, alt_all[r, :]); push!(ref_rows, ref_all[r, :])
            end
        end
        if k == 2             # an equation's version may depend on one constraint only
            both = hcat(regime_matrices(ob, (true, true))...)
            expect = copy(ref_all)
            for (alt_all, rows) in per_reg
                expect[rows, :] = alt_all[rows, :]
            end
            shared = intersect(per_reg[1][2], per_reg[2][2])
            if !isempty(shared) || maximum(abs, both .- expect) > 1e-10 * max(1.0, maximum(abs, both))
                throw(ModelError("obc_method = news: an equation's version depends on both constraints jointly; the regime formulation needs each " *
                                 "switched equation to depend on one constraint. Use obc_method = occbin for this model."))
            end
        end
        m = ob.model
        used = Set{Int}()
        for c in m.occbin_constraints, key in ("bind", "relax")
            txt = strip(string(something(get(c, key, ""), "")))
            isempty(txt) && continue
            _collect_vars!(used, _parse_constraint(txt), m)
        end
        (cols=cols, alt_rows=Matrix{Float64}(reduce(hcat, alt_rows)'), ref_rows=Matrix{Float64}(reduce(hcat, ref_rows)'),
         cv=sort!(collect(used)), jb=[j for (j, _) in cols])
    end
end

# endogenous variables named in an expression of a constraint condition
function _collect_vars!(used::Set{Int}, ex, m::Model)
    if ex isa Symbol
        haskey(m.ie, String(ex)) && push!(used, m.ie[String(ex)])
    elseif ex isa Expr
        for a in (ex.head == :call ? ex.args[2:end] : ex.args)
            if ex.head == :call && ex.args[1] isa Symbol && lowercase(String(ex.args[1])) == "steady_state"
                break
            end
            _collect_vars!(used, a, m)
        end
    end
    used
end

"The `(d_b, d_r)` of each constraint: the linearised bind/relax conditions (`d_b < 0`: bind holds; `d_r > 0`: relax holds; `d_r = d_b` without relax), `3n + nx + 1` entries each (constant last)."
function _regime_conditions(ob::Occbin)
    m, n, nx = ob.model, ob.n, ob.n_exo
    out = Tuple{Vector{Float64},Vector{Float64}}[]
    function lin(txt)
        ex = _negative_side(_parse_constraint(txt))
        ex === nothing && throw(ModelError("the condition '$txt' is not a single <, <=, > or >= comparison"))
        v, g = _linearise_condition(m, ex, ob.ys, ob.params)
        vec = zeros(3n + nx + 1)
        vec[n+1:2n] = g
        vec[end] = v
        vec
    end
    for c in m.occbin_constraints
        db = lin(string(c["bind"]))
        relaxtxt = strip(string(something(get(c, "relax", ""), "")))
        push!(out, (db, isempty(relaxtxt) ? db : -lin(String(relaxtxt))))
    end
    out
end

regime_obc(ob::Occbin; horizon::Integer=40, check_ahead::Integer=40) = regime_obc(ob, Int(horizon), Int(horizon + check_ahead))
function regime_obc(ob::Occbin, T::Int, N::Int)
    cache = get!(() -> Dict{Tuple{Int,Int},RegimeOBC}(), _REGIME_CACHE, ob)
    get!(() -> _build_regime(ob, T, N), cache, (T, N))
end
RegimeOBC(ob::Occbin; horizon::Integer=40, check_ahead::Integer=40) = regime_obc(ob; horizon=horizon, check_ahead=check_ahead)

function _build_regime(ob::Occbin, T::Int, N::Int)
    N >= T || throw(ArgumentError("the checked periods N must be at least the horizon"))
    su = _regime_setup(ob)
    n, nx = ob.n, ob.n_exo
    C0, B0, A0, _, _ = regime_matrices(ob, ntuple(_ -> false, ob.k))
    F = lu(B0 + A0 * ob.P)
    G = -(F \ A0)
    nb = length(su.cols)
    E = zeros(n, nb)
    for (b, (_, row)) in enumerate(su.cols)
        E[row, b] = 1.0
    end
    FE = F \ E
    V = _wedge_vectors(G, FE, T)
    a = su.alt_rows
    aC, aB, aA, aJ = a[:, 1:n], a[:, n+1:2n], a[:, 2n+1:3n], a[:, 3n+1:3n+nx]
    Rr = _functional_rows(ob.P, aC, aB, aA, T)
    R = _response_matrix(Rr, V, aA, T, T)
    Am = zeros(nb * T, n + nx)
    for i in 1:nb, t in 1:T
        Am[(i-1)*T+t, 1:n] = view(Rr[t+1], i, :)
        Am[(i-1)*T+t, n+1:n+nx] = ob.Q' * view(Rr[t], i, :)
        t == 1 && (Am[(i-1)*T+t, n+1:n+nx] .+= view(aJ, i, :))
    end
    a0 = repeat(a[:, end]; inner=T)
    cv = su.cv
    k = length(cv)
    Ecv = zeros(k, n)
    for (c, i) in enumerate(cv)
        Ecv[c, i] = 1.0
    end
    Z = zeros(k, n)
    Rc = _functional_rows(ob.P, Z, Ecv, Z, N)
    Wfull = _response_matrix(Rc, V, Z, N, T)                  # row (i-1) N + t
    Wcv = Matrix{Float64}(undef, k * N, nb * T)
    Bcv = Matrix{Float64}(undef, k * N, n + nx)
    for t in 1:N, i in 1:k
        Wcv[(t-1)*k+i, :] = view(Wfull, (i-1)*N+t, :)
        Bcv[(t-1)*k+i, 1:n] = view(Rc[t+1], i, :)
        Bcv[(t-1)*k+i, n+1:n+nx] = ob.Q' * view(Rc[t], i, :)
    end
    scale = max(1.0, maximum(abs, ob.P), maximum(abs, view(Am, :, 1:n)))
    used = [max(maximum(abs, view(ob.P, :, j)), maximum(abs, view(Am, :, j))) > 1e-14 * scale for j in 1:n]
    st = findall(used)
    xcols = vcat(st, n+1:n+nx)
    RegimeOBC(ob, T, N, ob.k, su.cols, su.alt_rows, su.ref_rows, cv, su.jb, G, FE, V, R, Am, a0, Wcv, Bcv, st,
              Matrix{Float64}(ob.P[:, st]), Matrix{Float64}(Am[:, xcols]), Matrix{Float64}(Bcv[:, xcols]), copy(ob.ys), Dict{Symbol,Any}())
end

# lazily built responses (only the fallbacks and the diagnostics need them)
function _Yw(rg::RegimeOBC)
    get!(rg.cache, :Yw) do
        _full_responses(rg.ob.P, rg.V0, rg.N, rg.T)
    end::Vector{Matrix{Float64}}
end
function _Yb(rg::RegimeOBC)
    get!(rg.cache, :Yb) do
        ob, n, nx = rg.ob, rg.ob.n, rg.ob.n_exo
        out = Vector{Matrix{Float64}}(undef, rg.N + 1)
        y = hcat(Matrix{Float64}(I, n, n), zeros(n, nx))
        for t in 0:rg.N
            y = ob.P * y
            t == 0 && (y[:, n+1:end] = ob.Q)
            out[t+1] = y
        end
        out
    end::Vector{Matrix{Float64}}
end
_conditions(rg::RegimeOBC) = get!(() -> _regime_conditions(rg.ob), rg.cache, :cond)::Vector{Tuple{Vector{Float64},Vector{Float64}}}

# ---------------------------------------------------------------------- regime checks and wedges
const _ZMat = AbstractMatrix{Bool}

"Regime indicators that fail OccBin's test along the path `pcv` (`k x N` deviations of the condition variables): `(binding & ~z, relax & z)`."
function _regime_violations(rg::RegimeOBC, z::_ZMat, pcv::AbstractMatrix)
    vb, vr = falses(rg.N, rg.nc), falses(rg.N, rg.nc)
    Base.invokelatest(_regime_flags!, vb, vr, rg, z, pcv)      # the compiled conditions live in a newer world
    vb, vr
end
function _regime_flags!(vb, vr, rg::RegimeOBC, z, pcv)
    ob, zr, cv = rg.ob, rg.zbuf, rg.cv
    for t in 1:rg.N
        for (c, i) in enumerate(cv)
            zr[i] = pcv[c, t] + ob.ys[i]
        end
        for (j, cn) in enumerate(ob.cons)
            b = _call_now(cn.bind, zr, ob.params, ob.ys)
            bound = b === true || (b isa Number && !isnan(b) && b != 0)
            rel = if cn.relax === nothing
                !bound
            else
                r = _call_now(cn.relax, zr, ob.params, ob.ys)
                r === true || (r isa Number && !isnan(r) && r != 0)
            end
            vb[t, j] = bound && !z[t, j]
            vr[t, j] = rel && z[t, j]
        end
    end
    nothing
end

# positions (in the stacked wedge vector) of the periods in which each block's constraint is in its alternative regime
function _regime_index(rg::RegimeOBC, z::_ZMat)
    S = Int[]
    for b in eachindex(rg.cols), t in 1:rg.T
        z[t, rg.jb[b]] && push!(S, (b - 1) * rg.T + t)
    end
    S
end

"Wedges that make the alternative equations hold in the periods of `z`; `nothing` when the system is singular."
function _regime_wedges(rg::RegimeOBC, z::_ZMat, a::AbstractVector)
    w = zeros(length(rg.cols) * rg.T)
    S = _regime_index(rg, z)
    if !isempty(S)
        F = lu(rg.R[S, S]; check=false)
        issuccess(F) || return nothing
        w[S] = -(F \ a[S])
        all(isfinite, w) || return nothing
    end
    w
end

"OccBin's guess-and-verify iteration from the guess `z`: `(ok, beyond, z, w, iterations)`; `beyond`: it wants a regime change after the horizon."
function _regime_iterate(rg::RegimeOBC, z0::_ZMat, Ycv, a, maxit::Int)
    T, N, k = rg.T, rg.N, length(rg.cv)
    z = BitMatrix(z0)
    seen = Set{BitMatrix}()
    for it in 0:maxit-1
        any(view(z, T+1:N, :)) && return (false, true, z, nothing, it)
        w = _regime_wedges(rg, z, a)
        w === nothing && return (false, false, z, nothing, it)
        pcv = any(!iszero, w) ? Ycv .+ reshape(rg.Wcv * w, k, N) : Ycv
        vb, vr = _regime_violations(rg, z, pcv)
        (any(vb) || any(vr)) || return (true, false, z, w, it)
        z in seen && return (false, false, z, nothing, it)
        push!(seen, copy(z))
        z = (z .& .!vr) .| vb
    end
    (false, false, z, nothing, maxit)
end

# ---------------------------------------------------------------------- a step: one shock period
"""
`_regime_step(rg, init, shock; maxit=60, method=:auto, z0=nothing) -> (status, z, w, info)`: regime indicators `z` (`N x nc`), wedges `w`
and `info = (used, iterations, message)` for one shock period (no path). `method`: `:auto` (guess and verify, then the fallbacks),
`:guess_verify` (no fallbacks), `:milp` (the integer programme over regime indicators only). `status` is one of `OBC_STATUSES`.
"""
function _regime_step(rg::RegimeOBC, init::AbstractVector, shock::AbstractVector; maxit::Int=60, method::Symbol=:auto, z0=nothing)
    T, N, k, nc = rg.T, rg.N, length(rg.cv), rg.nc
    xs = vcat(init[rg.st], shock)
    Ycv = reshape(rg.Bx * xs, k, N)
    a = rg.a0 .+ rg.Ax * xs
    z = falses(N, nc)
    info = (used=:none, iterations=0, message="")
    ok = beyond = false
    w = nothing
    if method != :milp
        if z0 !== nothing && any(z0)
            ok, beyond, zg, w, it = _regime_iterate(rg, z0, Ycv, a, maxit)
            ok && (z = zg; info = (used=it > 0 ? :guess_verify : :none, iterations=it, message=""))
        end
        if !ok
            ok, beyond, z, w, it = _regime_iterate(rg, z, Ycv, a, maxit)
            ok && (info = (used=it > 0 ? :guess_verify : :none, iterations=it, message=""))
        end
    end
    ok && return (:ok, z, w, info)
    horizon_msg = "a constraint still binds after the news horizon: increase obc_horizon"
    if beyond && method != :milp
        return (:horizon_too_short, z, zeros(length(rg.cols) * T), (used=:guess_verify, iterations=0, message=horizon_msg))
    end
    if method == :guess_verify
        return (:not_converged, z, zeros(length(rg.cols) * T), (used=:guess_verify, iterations=0, message="the guess-and-verify iteration cycles"))
    end
    fail(st, msg, used) = (st, falses(N, nc), zeros(length(rg.cols) * T), (used=used, iterations=0, message=msg))
    if method != :milp
        try
            got = _solve_lcp(rg, init, shock)               # throws ObcError with a proof when no solution exists
            if got !== nothing
                zl, wl, pl = got
                vb, vr = _regime_violations(rg, zl, pl)
                (any(vb) || any(vr)) || return (:ok, zl, wl, (used=:lcp, iterations=0, message=""))
            end
        catch err
            err isa ObcError || rethrow()
            return fail(err.status, err.msg, :lcp)
        end
    end
    try
        zm = _regime_milp(rg, init, shock, a)
        wm = _regime_wedges(rg, zm, a)
        wm === nothing && return fail(:not_converged, "the regime sequence found by the integer programme gives a singular system", :milp)
        pcv = Ycv .+ reshape(rg.Wcv * wm, k, N)
        vb, vr = _regime_violations(rg, zm, pcv)
        any(view(vb, T+1:N, :)) && return fail(:horizon_too_short, horizon_msg, :milp)
        (any(vb) || any(vr)) &&
            return fail(:not_converged, "the regime sequence found by the integer programme (linearised conditions) fails OccBin's test on the conditions in levels", :milp)
        return (:ok, zm, wm, (used=:milp, iterations=0, message=""))
    catch err
        err isa ObcError || rethrow()
        return fail(beyond ? :horizon_too_short : err.status, beyond ? horizon_msg : err.msg, :milp)
    end
end

# ---------------------------------------------------------------------- reduction to an LCP (tied wedges)
"""
Structure that lets a one-constraint, several-equation model be written as an LCP. If all but one of the switched equations respond
to the wedges only in the same period and with constant coefficients (`alt_i(t) = sum_b C[i, b] w_b(t) + (terms the wedges do not
move)`), then in an alternative-regime period those equations tie the other wedges to the wedge of the remaining ("primary")
equation. Returns `nothing` when the structure is absent; otherwise a named tuple with `primary`, `secondary`, the wedge direction
`d`, `Coi`, `Mblocks`, `Mraw`, `Qraw`, `q0raw`, `mirror` (the bind/relax conditions are exact mirror images of the primary
equation, so that the complementarity theory applies), `s_g`, `s_y`.
"""
function _find_tie(rg::RegimeOBC)
    nb, T, N = length(rg.cols), rg.T, rg.N
    (rg.nc == 1 && nb >= 2) || return nothing
    ob, n, nx = rg.ob, rg.ob.n, rg.ob.n_exo
    Cm = zeros(nb, nb)
    diagonal = falses(nb)
    scale = max(maximum(abs, rg.R), 1e-300)
    for i in 1:nb
        ok = true
        for b in 1:nb
            blk = rg.R[(i-1)*T+1:i*T, (b-1)*T+1:b*T]
            c = sum(blk[t, t] for t in 1:T) / T
            off = maximum((abs(blk[s, t]) for s in 1:T, t in 1:T if s != t); init=0.0)
            dev = maximum(abs(blk[t, t] - c) for t in 1:T)
            if off > 1e-9 * scale || dev > 1e-9 * scale
                ok = false
                break
            end
            Cm[i, b] = c
        end
        diagonal[i] = ok
    end
    count(diagonal) == nb - 1 || return nothing
    pb = findfirst(!, diagonal)
    sec = [i for i in 1:nb if i != pb]
    Co = Cm[sec, sec]
    abs(det(Co)) < 1e-12 && return nothing
    d = zeros(nb)
    d[pb] = 1.0
    d[sec] = -(Co \ Cm[sec, pb])
    Coi = inv(Co)
    Yw, Yb = _Yw(rg), _Yb(rg)
    a = rg.alt_rows[pb, :]
    aC, aB, aA, aJ = a[1:n], a[n+1:2n], a[2n+1:3n], a[3n+1:3n+nx]
    apply(Y, cols) = begin                                  # alt_p response, N periods
        out = zeros(N, length(cols))
        for t in 1:N
            v = view(Y[t], :, cols)' * aB + view(Y[t+1], :, cols)' * aA
            t > 1 && (v = v + view(Y[t-1], :, cols)' * aC)
            out[t, :] = v
        end
        out
    end
    Mblocks = [apply(Yw, (b-1)*T+1:b*T) for b in 1:nb]
    Mraw = sum(d[b] .* Mblocks[b] for b in 1:nb)
    Qraw = apply(Yb, 1:n+nx)
    Qraw[1, 1:n] .+= aC
    Qraw[1, n+1:end] .+= aJ
    mirror, s_g, s_y = false, 1.0, 1.0
    refs = [rg.ref_rows[b, :] for b in 1:nb]
    conds = try
        _conditions(rg)
    catch err
        err isa ModelError || rethrow()
        return nothing
    end
    d_b, d_r = conds[1]
    Xb = hcat(a, refs...)                                     # bind: k1 alt_p + reference residuals
    cb = pinv(Xb) * d_b
    Xr = hcat(refs[pb], [rg.alt_rows[b, :] for b in 1:nb]...)    # relax: c ref_p + alternative residuals
    cr = pinv(Xr) * d_r
    okb = norm(Xb * cb - d_b) <= 1e-8 * norm(d_b) && abs(cb[1]) > 1e-10
    okr = norm(Xr * cr - d_r) <= 1e-8 * norm(d_r) && abs(cr[1]) > 1e-10
    if okb && okr
        s_g, s_y = sign(cb[1]), -sign(cr[1])
        mirror = s_g * a[end] > 0
    end
    (primary=pb, secondary=sec, d=d, Coi=Coi, Mblocks=Mblocks, Mraw=Mraw, Qraw=Qraw, q0raw=a[end], mirror=mirror, s_g=s_g, s_y=s_y)
end

"The tied-wedge structure (`_find_tie`), computed on first use."
function _tie(rg::RegimeOBC)
    haskey(rg.cache, :tie) || (rg.cache[:tie] = _find_tie(rg))
    rg.cache[:tie]
end

"""
    obc_lcp(rg, init, shock; tol=1e-8) -> (M, q, G, info) or nothing

The equivalent complementarity problem of a one-constraint, several-equation model for this state, or `nothing` when the model has
not got the tie structure. With `y_t >= 0` the (normalised) wedge on the primary equation and `z_t = 1` in alternative-regime
periods, the primary gap is `g = q + M y + G z` and a solution has `g >= 0, y >= 0, y' g = 0`, `y_t > 0` only if `z_t = 1`, `g_t = 0` if
`z_t = 1`. `G` holds the state-dependent offsets of the tied wedges (`nothing` when zero, e.g. no shock enters the tied equations);
then this is a standard LCP in `(M, q)` and Holden's results on `M` apply. `M`, `q`, `G` have `N` rows (the first `T` are the
complementarity rows). `info.a`, `info.x`: the alternative residuals' constants and `[init; shock]`.
"""
function obc_lcp(rg::RegimeOBC, init::AbstractVector, shock::AbstractVector; tol::Real=1e-8)
    tie = _tie(rg)
    tie === nothing && return nothing
    T, N = rg.T, rg.N
    x = vcat(init, shock)
    a = rg.a0 .+ rg.Am * x
    base = tie.q0raw .+ tie.Qraw * x
    s_g, s_y = tie.s_g, tie.s_y
    M = (s_g * s_y) .* tie.Mraw
    q = s_g .* base
    G = zeros(N, T)
    asec = reduce(vcat, [a[(i-1)*T+1:i*T]' for i in tie.secondary])          # n_sec x T
    f = -(tie.Coi * asec)
    for (o, blk) in enumerate(tie.secondary)
        G .+= s_g .* tie.Mblocks[blk] .* f[o, :]'
    end
    maximum(abs, G) <= tol * max(maximum(abs, q), 1e-300) && (G = nothing)
    (M, q, G, (a=a, x=x))
end

"""
    obc_diagnostics(rg::RegimeOBC; max_exact=16) -> NamedTuple or nothing

Existence / uniqueness diagnostics (`m_diagnostics`) on the `M` of the equivalent LCP of the tied-wedge reduction. They are
Holden's conditions when the offsets `G` vanish; `nothing` when the structure is absent.
"""
function obc_diagnostics(rg::RegimeOBC; max_exact::Integer=16)
    tie = _tie(rg)
    (tie === nothing || !tie.mirror) && return nothing
    m_diagnostics((tie.s_g * tie.s_y) .* tie.Mraw[1:rg.T, :]; n_c=1, max_exact=max_exact)
end

# the path of the condition variables (k x N, deviations) for state/shock x and wedges w
function _regime_pcv(rg::RegimeOBC, x::AbstractVector, w::AbstractVector)
    Yb, Yw, k = _Yb(rg), _Yw(rg), length(rg.cv)
    out = Matrix{Float64}(undef, k, rg.N)
    for t in 1:rg.N
        v = Yb[t] * x .+ Yw[t] * w
        for (c, i) in enumerate(rg.cv)
            out[c, t] = v[i]
        end
    end
    out
end

"""
    single_spell_solutions(rg, init, shock; l_max=8, tol=1e-7) -> Vector{(l, k)}

All solutions with one binding spell `l .. l + k - 1` (0-based start `l <= l_max`), as `(l, k)` pairs. More than one entry means
multiple equilibria at this state.
"""
function single_spell_solutions(rg::RegimeOBC, init::AbstractVector, shock::AbstractVector; l_max::Int=8, tol::Real=1e-7)
    T, out = rg.T, Tuple{Int,Int}[]
    tie = _tie(rg)
    got = (tie !== nothing && tie.mirror) ? obc_lcp(rg, init, shock) : nothing
    if got !== nothing
        M, q, G, _ = got
        for l in 0:min(l_max, T - 1), k in 1:T-l
            idx = l+1:l+k
            qq = G === nothing ? q : q .+ vec(sum(view(G, :, idx); dims=2))
            y = zeros(T)
            F = lu(M[idx, idx]; check=false)
            issuccess(F) || continue
            y[idx] = -(F \ qq[idx])
            _verify_lcp(M, qq, y, T, tol) && push!(out, (l, k))
        end
        return out
    end
    x = vcat(init, shock)
    a = rg.a0 .+ rg.Am * x
    for l in 0:min(l_max, T - 1), k in 1:T-l
        z = falses(rg.N, rg.nc)
        z[l+1:l+k, 1] .= true
        w = _regime_wedges(rg, z, a)
        w === nothing && continue
        vb, vr = _regime_violations(rg, z, _regime_pcv(rg, x, w))
        (any(vb) || any(vr)) || push!(out, (l, k))
    end
    out
end

# solve through the equivalent complementarity problem: (z, w, pcv) or nothing; throws ObcError(:no_solution) with a proof when the relaxation is infeasible
function _solve_lcp(rg::RegimeOBC, init::AbstractVector, shock::AbstractVector)
    tie = _tie(rg)
    (tie === nothing || !tie.mirror) && return nothing
    M, q, G, info = obc_lcp(rg, init, shock)
    T, N = rg.T, rg.N
    proof = "no solution exists that returns to the reference regime within the news horizon: the constraint cannot be kept satisfied in " *
            "every period by any non-negative wedges (linear programme infeasible)"
    zsol = nothing
    if G === nothing
        M1, q1 = M[1:T, :], q[1:T]
        y, st, _ = solve_active_set(M1, q1; tol=1e-8)
        if !(st == :ok && _verify_lcp(M, q, y, T, 1e-8))
            y, st, inf = solve_news(M1, q1; tol=1e-8, Mb=M[T+1:N, :], qb=q[T+1:N], n_c=1,
                                    time_limit=REGIME_MILP_TIME_LIMIT[])
            if st != :ok
                msg = st == :no_solution && inf.proof ? proof : inf.message
                throw(ObcError(st, 0, st == :no_solution && !inf.proof ? "no solution within the news horizon (the linear complementarity problem has none)" : msg))
            end
        end
        zsol = y .> 1e-9 * max(1.0, maximum(abs, y))
    else
        st, _ = highs_lp(zeros(2T), -hcat(M, G), fill(-Inf, N), q, zeros(2T), vcat(fill(Inf, T), ones(T)))      # y >= 0, 0 <= z <= 1, q + M y + G z >= 0
        st == :infeasible && throw(ObcError(:no_solution, 0, proof))
        sc = max(maximum(abs, q), 1e-12)
        dg = abs.(diag(M[1:T, :]))
        pos = filter(>(0), dg)
        U = 1e3 * sc / max(isempty(pos) ? 1.0 : minimum(pos), 1e-12)
        Om = 1e3 * sc
        Id = Matrix{Float64}(I, T, T)
        A = vcat(hcat(Id, -U .* Id), hcat(M, G), hcat(M[1:T, :], G[1:T, :] .+ Om .* Id))
        rlo = vcat(fill(-Inf, T), -q, fill(-Inf, T))
        rhi = vcat(zeros(T), fill(Inf, N), Om .- q[1:T])
        st, x = highs_milp(vcat(zeros(T), ones(T)), A, rlo, rhi, zeros(2T), vcat(fill(U, T), ones(T)), vcat(falses(T), trues(T));
                           time_limit=REGIME_MILP_TIME_LIMIT[])
        if isempty(x)
            st == :time_limit && throw(ObcError(:not_converged, 0, "the integer programme did not finish in $(REGIME_MILP_TIME_LIMIT[]) s"))
            throw(ObcError(:no_solution, 0, "no solution found within the news horizon by the integer programme (the linear relaxation is feasible, so this is not a proof)"))
        end
        zsol = x[T+1:2T] .> 0.5
    end
    z = falses(N, 1)
    z[1:T, 1] = zsol
    w = _regime_wedges(rg, z, info.a)
    w === nothing && return nothing
    z, w, _regime_pcv(rg, info.x, w)
end

# ---------------------------------------------------------------------- the integer programme over regime indicators
"""
Regime indicators from a mixed-integer programme on the linearised conditions: minimise the number of periods spent in the
alternative regimes subject to the regime logic (big-M). Slow with several constraints or long horizons; time-limited.
"""
function _regime_milp(rg::RegimeOBC, init::AbstractVector, shock::AbstractVector, a::AbstractVector)
    T, N, n, nb, nc = rg.T, rg.N, rg.ob.n, length(rg.cols), rg.nc
    nw, nz = nb * T, nc * T
    x = vcat(init, shock)
    Yw, Yb = _Yw(rg), _Yb(rg)
    Db, Dr = zeros(nc * N, nw), zeros(nc * N, nw)
    b0, r0 = zeros(nc * N), zeros(nc * N)
    for (j, (d_b, d_r)) in enumerate(_conditions(rg))
        for (d, D, d0) in ((d_b, Db, b0), (d_r, Dr, r0))
            g = d[n+1:2n]
            for t in 1:N
                D[(j-1)*N+t, :] = g' * Yw[t]
                d0[(j-1)*N+t] = dot(g, Yb[t] * x) + d[end]
            end
        end
    end
    sc = max(maximum(abs, a), maximum(abs, b0), maximum(abs, r0), 1e-12)
    diagR = abs.(diag(rg.R))
    pos = filter(>(0), diagR)
    W = 1e3 * sc / max(isempty(pos) ? 1.0 : _obc_median(pos), 1e-12)
    Om = 1e3 * sc
    eps = 1e-9 * sc
    rows, lo, hi = Matrix{Float64}[], Vector{Float64}[], Vector{Float64}[]
    function add(Aw, Az, lower, upper)
        push!(rows, hcat(Aw, Az))
        push!(lo, fill(0.0, size(Aw, 1)) .+ lower)
        push!(hi, fill(0.0, size(Aw, 1)) .+ upper)
    end
    Zb = zeros(nw, nz)                                   # block (wedge) -> its constraint's indicator
    for (b, (j, _)) in enumerate(rg.cols)
        Zb[(b-1)*T+1:b*T, (j-1)*T+1:j*T] = Matrix{Float64}(I, T, T)
    end
    Iw = Matrix{Float64}(I, nw, nw)
    add(Iw, -W .* Zb, -Inf, 0.0)                         #  w <= W z
    add(Iw, W .* Zb, 0.0, Inf)                           # -w <= W z
    add(rg.R, Om .* Zb, -Inf, Om .- a)                   #  alt residual <= Om (1 - z)
    add(rg.R, -Om .* Zb, -Om .- a, Inf)                  #  alt residual >= -Om (1 - z)
    for j in 1:nc
        sel = zeros(T, nz)
        sel[:, (j-1)*T+1:j*T] = Matrix{Float64}(I, T, T)
        rin, rout = (j-1)*N+1:(j-1)*N+T, (j-1)*N+T+1:j*N
        add(Db[rin, :], Om .* sel, -b0[rin] .- eps, Inf)                 # reference regime: bind condition false
        add(Dr[rin, :], Om .* sel, -Inf, Om .- r0[rin] .+ eps)           # alternative regime: relax condition false
        N > T && add(Db[rout, :], zeros(N - T, nz), -b0[rout] .- eps, Inf)
    end
    A = reduce(vcat, rows)
    st, xs = highs_milp(vcat(zeros(nw), ones(nz)), A, reduce(vcat, lo), reduce(vcat, hi), vcat(fill(-W, nw), zeros(nz)),
                        vcat(fill(W, nw), ones(nz)), vcat(falses(nw), trues(nz)); time_limit=REGIME_MILP_TIME_LIMIT[])
    if st == :time_limit && isempty(xs)
        throw(ObcError(:not_converged, 0, "the integer programme over regime sequences did not finish in $(REGIME_MILP_TIME_LIMIT[]) s " *
                                          "(this big-M formulation is slow with several constraints or long horizons)"))
    end
    isempty(xs) && throw(ObcError(:no_solution, 0, "no regime sequence within the news horizon satisfies the model (linearised conditions)"))
    z = falses(N, nc)
    for j in 1:nc, t in 1:T
        z[t, j] = xs[nw+(j-1)*T+t] > 0.5
    end
    z
end

# ---------------------------------------------------------------------- simulation
"`regime_solve_period(rg, init, shock)` -> `(status, z, w, path, info)`, the path in deviations (`N + 1` rows)."
function regime_solve_period(rg::RegimeOBC, init::AbstractVector, shock::AbstractVector; maxit::Int=60, method::Symbol=:auto)
    st, z, w, info = _regime_step(rg, init, shock; maxit=maxit, method=method)
    path = _wedge_path(rg.ob.P, rg.G, rg.FE, rg.ob.P * init .+ rg.ob.Q * shock, w, rg.T, rg.N)
    st, z, w, path, info
end

"""
`_simulate_regime`: `_simulate_obc`'s loop with the regime formulation (`obc_method = news` for constraints that are not all
complementary pairs): every shock period is solved by `_regime_step`; periods without a new shock continue the last path.
`fast_path = false` forces the integer programme over regime indicators.
"""
function _simulate_regime(ob::Occbin, shocks_in::AbstractMatrix; method::Symbol, horizon::Integer, tol::Real, raise::Bool,
                          periods::Integer, check_ahead_periods::Integer, max_check_ahead_periods::Real, init, exo_pos,
                          ignored::NamedTuple, solver_options=NamedTuple(), fast_path::Bool=true)
    k, n, n_exo, m = ob.k, ob.n, ob.n_exo, ob.model
    changed = [String(key) for (key, v) in pairs(ignored) if !isequal(v, getfield(_OBC_IGNORED_DEFAULTS, key))]
    isempty(changed) || warn_once(m, "obc_method = $method: OccBin options of the regime iteration are not used (ignored): " * join(sort(changed), ", "))
    exo_idx = exo_pos === nothing ? collect(1:n_exo) : Int[_exo_index(m, p) for p in (exo_pos isa Union{AbstractString,Integer} ? [exo_pos] : exo_pos)]
    size(shocks_in, 2) == length(exo_idx) ||
        throw(ArgumentError("shocks have $(size(shocks_in, 2)) columns but exo_pos selects $(length(exo_idx)) shocks"))
    S = size(shocks_in, 1)
    S >= 1 || throw(ArgumentError("at least one shock period is needed"))
    shocks = zeros(S, n_exo)
    shocks[:, exo_idx] = Float64.(shocks_in)
    n_periods = max(Int(periods), S)
    T = Int(horizon)
    check = Int(min(check_ahead_periods, max_check_ahead_periods))
    n0 = max(check, n_periods - S)
    L = n0 + 1
    N = max(L, T + check)
    rg = regime_obc(ob, T, N)
    nb = length(rg.cols)
    rmethod = fast_path ? get(solver_options, :regime_method, :auto) : :milp
    y0 = init === nothing || isempty(init) ? zeros(n) : Float64.(vec(collect(init)))
    length(y0) == n || throw(ArgumentError("init needs $n values (declaration order)"))
    init_orig = copy(y0)
    piecewise = zeros(n_periods, n)
    history = Vector{Vector{Tuple{Vector{Int},Vector{Int}}}}()
    iters, diag = Int[], _Diag[]
    ss_out = Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}[]
    wedges = zeros(S, nb * T)
    status = Symbol[]
    infos = Any[]
    Hn = max(L, T)                                     # rows of the regime history
    w = zeros(nb * T)
    z = falses(N, k)
    Z = zeros(0, n)
    Zvalid = false
    init_last, u_last = zeros(n), zeros(n_exo)
    for sp in 1:S
        _obc_tick()
        u = shocks[sp, :]
        if sp == 1 || any(!iszero, u)
            st, z, w, info = _regime_step(rg, y0, u; method=rmethod)
            init_last, u_last, Zvalid = copy(y0), copy(u), false
            iterations = info.iterations
            if st == :ok
                ynext = rg.Ps * y0[rg.st] .+ ob.Q * u .+ rg.V0 * w
            else
                ynext = ob.P * y0 .+ ob.Q * u
            end
        else                                           # no new shock: the path solved last continues (OccBin's shift)
            if !Zvalid
                Z = _wedge_path(ob.P, rg.G, rg.FE, ob.P * init_last .+ ob.Q * u_last, w, T, L)[1:L, :]
                Zvalid = true
            end
            w = reduce(vcat, [vcat(w[(b-1)*T+2:b*T], 0.0) for b in 1:nb])
            z = vcat(z[2:end, :], falses(1, k))
            Z = vcat(Z[2:end, :], (ob.P * Z[end, :])')
            st, info, iterations = :ok, (used=:shift, iterations=0, message="no shock: previous path continued"), 0
            ynext = Z[1, :]
        end
        if st != :ok && raise
            throw(ObcError(st, sp, "obc_method = $method (regime formulation), shock period $sp: $st. $(info.message)"))
        end
        wedges[sp, :] = w
        push!(status, st)
        push!(infos, info)
        bi = z[1:Hn, :]
        push!(history, [_map_regime(view(bi, :, j)) for j in 1:k])
        push!(iters, iterations)
        Tmax = _tmax(bi)
        push!(ss_out, Tmax >= 1 ? _rules(ob, bi, Tmax)[1] : (ob.P, ob.Q, zeros(n)))
        push!(diag, (max_err=[0.0], violates=[false], n_viol=[0.0], selected=nothing))
        y0 = ynext
        piecewise[sp, :] = y0
    end
    _obc_tick()
    if n_periods > S
        Zvalid || (Z = _wedge_path(ob.P, rg.G, rg.FE, ob.P * init_last .+ ob.Q * u_last, w, T, L)[1:L, :])
        piecewise[S+1:end, :] = Z[2:n_periods-S+1, :]
    end
    linear = Matrix{Float64}(undef, n_periods, n)
    yl = init_orig
    for t in 1:n_periods
        yl = ob.P * yl + (t <= S ? ob.Q * shocks[t, :] : zeros(n))
        linear[t, :] = yl
    end
    dg = nothing
    try
        dg = obc_diagnostics(rg)
    catch err
        err isa Union{ModelError,SingularException,DomainError} || rethrow()
    end
    news = ObcNews(method, T, String[c["name"] for c in m.occbin_constraints], wedges, status, infos, :regime, dg)
    OccbinResult(piecewise .+ ob.ys', linear .+ ob.ys', Matrix{Float64}(shocks_in), history, ob.ys, iters, exo_idx, falses(S),
                 diag, m.endo_names, m.exo_names, ss_out, news)
end
