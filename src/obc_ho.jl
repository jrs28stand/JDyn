# News-shock OBC solution at order 2 and with integration over future shocks (obc_method = news, obc_order = 2, obc_cubature = true).
#
# Structure of Holden's DynareOBC (SimulateModel, ExpectedReturn, PerformCubature, ObtainRootConditionalCovariance), written afresh for
# JDyn's data structures from pydyn's prototype pydyn/obc_ho.py (executable specification) and docs/OBC_HO_PROGRESS.md; not a translation.
#
# State carried from period to period (deviations from the deterministic steady state, declaration order):
#   y1  first-order component of the pruned simulation that ignores the bound,
#   y2  second-order component (zero at order 1), including the risk correction ghs2/2,
#   o   the bound offset: the accumulated first-order effect of the news shocks, o' = P o + V0 y.
# The variables are  ys + y1 + y2 + o.  In every period:
#   1. base step: y1, y2 from the pruned decision rules and this period's shock; o_pre = P o;
#   2. the expected path E_t[y_{t+k}], k = 0..N, with the future shocks at their distribution (not at zero):
#        m1' = ghx m1s,  W' = ghxs W ghxs' + ghus Sigma ghus' (W0 = y1s y1s'),  m2' = ghx m2s + (ghxx vec(W) + ghuu vec(Sigma) + ghs2)/2,
#      plus P^k o_pre;
#   3. q = the gap (linearised) along that path; the same LCP  y >= 0, q + M y >= 0, y'(q + M y) = 0  as at order 1 (same M) gives
#      the news shocks y; with cubature the solution is the one at the mean of the relevant perturbed paths, q + R xi, times their share;
#   4. o = o_pre + V0 y.
# The path and its gap are linear in (y1, y2, y1s y1s', o_pre) and, apart from the first period's lag and shock, in nothing else, so
# everything that depends on the parameters only is computed once (`ObcHO`): the per-period work is a handful of matrix-vector products.

"""
    ObcHO

Everything of the higher-order news method that depends on the model and the options but not on the shocks: the decision rules (pruned,
declaration order), the stacked `M` over `N = horizon + check_ahead` gap periods, the affine map from the state to the gap path `q`, the
cubature root `R` and points `xi`. Build with `obc_ho(ob; ...)`, run with `obc_ho_simulate`.

Fields `R` and `xi` may be replaced (the tests inject DynareOBC's own root and Sobol points).
"""
mutable struct ObcHO
    ob::Occbin
    sys::ObcSystem
    order::Int
    T::Int                       # periods with wedges
    N::Int                       # gap periods: T + check_ahead
    n::Int
    nx::Int
    ns::Int
    sv::Vector{Int}              # state variables (declaration indices)
    ys::Vector{Float64}
    P::Matrix{Float64}
    V0::Matrix{Float64}          # n x T: next state = P x + Q u + V0 y
    Sigma::Matrix{Float64}
    ghx::Matrix{Float64}         # decision rules in declaration order (rows); ghx n x ns, ghu n x nx, ghxx n x ns^2, ghxu n x ns nx, ghuu n x nx^2
    ghu::Matrix{Float64}
    ghxx::Matrix{Float64}
    ghxu::Matrix{Float64}
    ghuu::Matrix{Float64}
    cst2::Vector{Float64}        # ghs2 / 2 (per variable)
    c2::Vector{Float64}          # (ghuu vec(Sigma) + ghs2) / 2
    ghxs::Matrix{Float64}        # ns x ns : ghx on the states
    ghus::Matrix{Float64}        # ns x nx
    SigS::Matrix{Float64}        # ghus Sigma ghus'
    aC::Vector{Float64}
    aB::Vector{Float64}
    aA::Vector{Float64}
    aJ::Vector{Float64}
    a0::Float64
    Mf::Matrix{Float64}          # N x T: sys.M over the check-ahead rows
    Mb::Matrix{Float64}          # (N - T) x T: the rows beyond the horizon
    Gq::Matrix{Float64}          # N x nx: response of the gap path to a shock in period 0
    # affine map  q = qc + Q1 y1 + Q2 y2 + Qw kron(y1s, y1s) + Qo o_pre  (+ aC' prev_total + aJ' u in the first row)
    qc::Vector{Float64}
    Q1::Matrix{Float64}
    Q2::Matrix{Float64}
    Qw::Matrix{Float64}
    Qo::Matrix{Float64}
    # cubature
    cubature::Bool
    pou::Int
    qmc_level::Int
    regions::Int
    taper::Bool
    pruning_cutoff::Union{Nothing,Float64}
    max_dim::Int
    R::Matrix{Float64}
    xi::Matrix{Float64}
    tol::Float64
    stats::Dict{Symbol,Int}
end
Base.show(io::IO, h::ObcHO) = print(io, "ObcHO(order = ", h.order, ", T = ", h.T, ", N = ", h.N, h.cubature ? ", cubature: " * string(size(h.R, 2)) * " dims, " *
                                    string(size(h.xi, 2)) * " points" : "", ")")

"Reorder the rows of a decision-rule array from decision-rule order to declaration order."
_dr_rows(A::AbstractMatrix, ov) = (B = similar(A); B[ov, :] = A; Matrix{Float64}(B))

"""
    obc_ho(ob::Occbin; order=2, horizon=40, check_ahead=40, cubature=false, periods_of_uncertainty=16, qmc_level=10, regions=1,
           taper=true, pruning_cutoff=0.01, max_cubature_dimension=128, tol=1e-10, Sigma_e=nothing) -> ObcHO

Build the higher-order news solver of a model with one complementarity constraint (the class the order-1 `news` method handles); order 1
or 2. `horizon`, `check_ahead`: periods with wedges and further periods over which the gap must stay non-negative (the gap path has
`N = horizon + check_ahead` periods). Cubature: `periods_of_uncertainty` (DynareOBC's `PeriodsOfUncertainty`), `qmc_level` (the points are
the first `2^(level+1) - 1` of the unscrambled Sobol sequence after the origin), `regions` (only 1: one region, DynareOBC's default).
"""
function obc_ho(ob::Occbin; order::Integer=2, horizon::Integer=40, check_ahead::Integer=40, cubature::Bool=false,
                periods_of_uncertainty::Integer=16, qmc_level::Integer=10, regions::Integer=1, taper::Bool=true,
                pruning_cutoff=0.01, max_cubature_dimension::Integer=128, tol::Real=1e-10, Sigma_e=nothing)
    order in (1, 2) || throw(ArgumentError("obc_order must be 1 or 2 (order 3 is not implemented)"))
    regions == 1 || throw(ArgumentError("obc_cubature_regions must be 1: several regions are not implemented"))
    periods_of_uncertainty >= 1 || throw(ArgumentError("obc_periods_of_uncertainty must be at least 1"))
    qmc_level >= 0 || throw(ArgumentError("obc_qmc_level must be non-negative"))
    m = ob.model
    T, N = Int(horizon), Int(horizon) + Int(check_ahead)
    check_ahead >= 0 || throw(ArgumentError("check_ahead_periods must be non-negative"))
    sys = obc_system(ob; horizon=T, method=:news)
    length(sys.cons) == 1 ||
        throw(ModelError("obc_order = 2 and obc_cubature are implemented for one constraint that is a complementary pair (the model has $(length(sys.cons)))"))
    n, nx = ob.n, ob.n_exo
    Sigma = Matrix{Float64}(Sigma_e === nothing ? m.Sigma_e : Sigma_e)
    dr = k_order(m; order=order, ys=ob.ys, params=ob.params, pruning=true, Sigma_e=Sigma)
    ov, sv = Vector{Int}(dr["order_var"]), Vector{Int}(dr["state_var"])
    ns = length(sv)
    ghx, ghu = _dr_rows(dr["ghx"], ov), _dr_rows(dr["ghu"], ov)
    ghxx, ghxu, ghuu = zeros(n, ns^2), zeros(n, ns * nx), zeros(n, nx^2)
    cst2, c2 = zeros(n), zeros(n)
    if order == 2
        ghxx, ghxu, ghuu = _dr_rows(dr["ghxx"], ov), _dr_rows(dr["ghxu"], ov), _dr_rows(dr["ghuu"], ov)
        g2 = zeros(n); g2[ov] = vec(dr["ghs2"])
        cst2 = 0.5 .* g2
        c2 = 0.5 .* (ghuu * vec(Sigma) .+ g2)
    end
    ghxs, ghus = ghx[sv, :], ghu[sv, :]
    SigS = ghus * Sigma * ghus'
    G, a0 = sys.G[1, :], sys.g0[1]
    aC, aB, aA, aJ = G[1:n], G[n+1:2n], G[2n+1:3n], G[3n+1:3n+nx]
    ext = _obc_ext(sys, N)
    Mf = vcat(sys.M, ext.Mb)
    Qxf = vcat(sys.Qx, ext.Qxb)
    Gq = Qxf[:, n+1:n+nx]
    h = ObcHO(ob, sys, Int(order), T, N, n, nx, ns, sv, Vector{Float64}(ob.ys), Matrix{Float64}(ob.P), Matrix{Float64}(sys.Wn), Sigma,
              ghx, ghu, ghxx, ghxu, ghuu, cst2, c2, ghxs, ghus, SigS, aC, aB, aA, aJ, a0, Mf, ext.Mb, Gq,
              zeros(N), zeros(N, n), zeros(N, n), zeros(N, ns^2), zeros(N, n),
              cubature, Int(periods_of_uncertainty), Int(qmc_level), Int(regions), taper,
              pruning_cutoff === nothing ? nothing : Float64(pruning_cutoff), Int(max_cubature_dimension), zeros(N, 0), zeros(0, 0), Float64(tol),
              Dict(:lcp => 0, :cubature_points => 0, :relevant_points => 0))
    _ho_build_q_map!(h)
    if cubature
        h.R = ho_conditional_root(h)
        h.xi = qmc_normal_points(size(h.R, 2), h.qmc_level)
    end
    h
end

# The affine map of q. X_k, the expected total at period k, is  A1_k y1 + A2_k y2 + Aw_k kron(y1s, y1s) + P^k o_pre + d_k  (all in
# declaration order); q_t = a0 + aC' X_{t-1} + aB' X_t + aA' X_{t+1}. Only the contractions with the gap coefficients are kept.
function _ho_build_q_map!(h::ObcHO)
    n, ns, N, sv, order = h.n, h.ns, h.N, h.sv, h.order
    fill!(h.qc, h.a0)
    fill!(h.Q1, 0.0); fill!(h.Q2, 0.0); fill!(h.Qw, 0.0); fill!(h.Qo, 0.0)
    coefs = ((h.aB, 0), (h.aA, -1), (h.aC, 1))              # (gap coefficients, row offset: X_k enters row k + offset)
    # k = 0: X_0 = y1 + y2 + o_pre
    for (a, off) in coefs
        r = 0 + off + 1
        1 <= r <= N || continue
        @views h.Q1[r, :] .+= a
        @views h.Q2[r, :] .+= a
        @views h.Qo[r, :] .+= a
    end
    Z = copy(h.ghx)                                         # n x ns: y1[sv] -> m1_k
    U = copy(h.ghx)                                         # y2[sv] -> m2_k
    Vw = 0.5 .* h.ghxx                                      # n x ns^2: kron(y1s, y1s) -> m2_k   (k = 1)
    e = copy(h.c2)                                          # constant of m2_k                  (k = 1)
    Phi = copy(h.ghxs)                                      # W_k = Phi_k W0 Phi_k' + Wc_k, at k = 1
    Wc = copy(h.SigS)
    Pk = copy(h.P)                                          # P^k
    for k in 1:N
        for (a, off) in coefs
            r = k + off + 1
            1 <= r <= N || continue
            @views h.Q1[r, sv] .+= Z' * a
            h.qc[r] += dot(a, e)
            if order == 2
                @views h.Q2[r, sv] .+= U' * a
                @views h.Qw[r, :] .+= Vw' * a
            end
            @views h.Qo[r, :] .+= Pk' * a
        end
        k == N && break
        # advance X_k -> X_{k+1}
        if order == 2
            Vw = h.ghx * Vw[sv, :] .+ 0.5 .* h.ghxx * kron(Phi, Phi)
            e = h.ghx * e[sv] .+ 0.5 .* (h.ghxx * vec(Wc)) .+ h.c2
            U = h.ghx * U[sv, :]
            Wc = h.ghxs * Wc * h.ghxs' .+ h.SigS
            Phi = h.ghxs * Phi
        end
        Z = h.ghx * Z[sv, :]
        Pk = h.P * Pk
    end
    h
end

"""
    ho_expected_path(h, y1, y2, o_pre) -> Matrix

The expected total deviation (declaration order, `N + 1` rows: periods 0..N) given this period's realised `y1`, `y2` (declaration
order) and the offset `o_pre`, by the recursion itself. The per-period code uses the equivalent affine map of `ObcHO`; this is the
reference it is tested against.
"""
function ho_expected_path(h::ObcHO, y1::AbstractVector, y2::AbstractVector, o_pre::AbstractVector)
    N, n, sv = h.N, h.n, h.sv
    out = zeros(N + 1, n)
    m1, m2, o = Vector{Float64}(y1), Vector{Float64}(y2), Vector{Float64}(o_pre)
    W = y1[sv] * y1[sv]'
    for k in 0:N
        if k > 0
            h.order == 2 && (m2 = h.ghx * m2[sv] .+ 0.5 .* (h.ghxx * vec(W)) .+ h.c2)
            m1 = h.ghx * m1[sv]
            W = h.ghxs * W * h.ghxs' .+ h.SigS
            o = h.P * o
        end
        out[k+1, :] = m1 .+ m2 .+ o
    end
    out
end

"The gap along an expected path (`N + 1` rows) by the definition: the reference for the affine map."
function ho_gap_path(h::ObcHO, path::AbstractMatrix, prev_total::AbstractVector, shock::AbstractVector)
    N = h.N
    q = zeros(N)
    for t in 1:N
        q[t] = h.a0 + dot(h.aB, view(path, t, :)) + dot(h.aA, view(path, t + 1, :))
        t > 1 && (q[t] += dot(h.aC, view(path, t - 1, :)))
    end
    q[1] += dot(h.aC, prev_total) + dot(h.aJ, shock)
    q
end

"""
    ho_conditional_root(h) -> Matrix

`R` with `R R'` the covariance of the gap path caused by the shocks arriving in the next `periods_of_uncertainty` periods (to first order),
with DynareOBC's cosine taper `w_i = (1 + cos(pi (i - 1) / PoU)) / 2` for a shock `i` periods ahead; then pruned as in DynareOBC's
`ObtainRootConditionalCovariance`: eigenvalues below `pruning_cutoff` times the largest are dropped, at most `max_cubature_dimension` kept.
"""
function ho_conditional_root(h::ObcHO)
    N, nx = h.N, h.nx
    w, V = eigen(Symmetric(h.Sigma))
    keep = w .> 1e-14 * max(maximum(w), 1e-300)
    L = V[:, keep] .* sqrt.(w[keep])'
    R = zeros(N, 0)
    blocks = Matrix{Float64}[]
    for i in 1:h.pou
        wi = h.taper ? 0.5 * (1 + cos(pi * (i - 1) / h.pou)) : 1.0
        B = zeros(N, size(L, 2))
        i < N && (B[i+1:N, :] = sqrt(wi) .* (h.Gq[1:N-i, :] * L))
        push!(blocks, B)
    end
    R = reduce(hcat, blocks)
    (size(R, 2) == 0 || h.pruning_cutoff === nothing) && return R
    ev, U = eigen(Symmetric(R * R'))
    ev = max.(ev, 0.0)
    ev[ev .< h.pruning_cutoff * maximum(ev)] .= 0.0
    ev[1:max(0, length(ev) - h.max_dim)] .= 0.0
    k = sqrt.(ev) .> sqrt(eps())
    U[:, k] .* sqrt.(ev[k])'
end

# ----------------------------------------------------------------------------------------------- the LCP at one q
function _ho_lcp(h::ObcHO, q::AbstractVector, period::Int)
    T, N = h.T, h.N
    h.stats[:lcp] += 1
    qmax = max(1.0, maximum(abs, q))
    all(>=(-1e-8 * qmax), q) && return zeros(T)
    qs = view(q, 1:T)
    tol = h.tol
    y, st, info = solve_active_set(view(h.Mf, 1:T, :), qs; tol=tol)
    if st == :ok && N > T && minimum(view(q, T+1:N) .+ h.Mb * y) < -tol
        st = :not_converged
    end
    if st != :ok
        hz = sort(unique([max(1, T ÷ 4), max(1, T ÷ 2), T]))
        y, st, info = solve_news(h.sys.M, Vector(qs); tol=tol, Mb=h.Mb, qb=Vector(view(q, T+1:N)), horizons=hz)
    end
    st == :ok || throw(ObcError(st, period, "obc_method = news (order $(h.order)): $st" *
                                (hasproperty(info, :message) ? ". " * info.message : "")))
    y
end

"The news shocks at gap path `q`: the LCP solution, or with cubature its average over the perturbed paths (one region)."
function ho_news_shocks(h::ObcHO, q::AbstractVector, period::Int=0)
    (!h.cubature || size(h.R, 2) == 0) && return _ho_lcp(h, q, period)
    N = h.N
    npts = size(h.xi, 2)
    pts = q .+ h.R * h.xi                                   # N x npts
    nrel = 0
    xsum = zeros(size(h.R, 2))
    for j in 1:npts
        pmax, pmin = 1.0, Inf
        for i in 1:N
            v = pts[i, j]
            pmax = max(pmax, abs(v)); pmin = min(pmin, v)
        end
        if !(pmin >= -1e-8 * pmax)                           # the path goes negative somewhere: relevant
            nrel += 1
            @views xsum .+= h.xi[:, j]
        end
    end
    h.stats[:cubature_points] += npts
    h.stats[:relevant_points] += nrel
    nrel == 0 && return zeros(h.T)
    qm = q .+ h.R * (xsum ./ nrel)
    (nrel / npts) .* _ho_lcp(h, qm, period)
end

# ----------------------------------------------------------------------------------------------------- simulation
@inline function _kron_self!(k::Vector{Float64}, x::AbstractVector)
    m = length(x)
    @inbounds for a in 1:m, b in 1:m
        k[(a-1)*m+b] = x[a] * x[b]
    end
    k
end
@inline function _kron2!(k::Vector{Float64}, x::AbstractVector, u::AbstractVector)
    m = length(u)
    @inbounds for a in 1:length(x), b in 1:m
        k[(a-1)*m+b] = x[a] * u[b]
    end
    k
end

"""
    obc_ho_simulate(h::ObcHO, shocks; init=nothing) -> NamedTuple

Simulate under the realised `shocks` (`S x n_exo`, one row per period, every period solved: DynareOBC's simulation, agents expect the
future shocks to be drawn from the model's distribution). `init`: initial state, deviations from the steady state (declaration order).
Returns levels (`S x n`, declaration order) `with_bound`, `without_bound` (the pruned simulation that ignores the bound), the news
shocks `wedges` (`S x horizon`) and `stats` (LCP solves, cubature points and relevant ones).
"""
function obc_ho_simulate(h::ObcHO, shocks::AbstractMatrix; init=nothing)
    n, nx, ns, T, sv, ys = h.n, h.nx, h.ns, h.T, h.sv, h.ys
    S = size(shocks, 1)
    size(shocks, 2) == nx || throw(ArgumentError("shocks need $nx columns (got $(size(shocks, 2)))"))
    y1, y2, o = zeros(n), zeros(n), zeros(n)
    init === nothing || (y1 .= init)
    prev = y1 .+ y2 .+ o
    with_b, without_b, wedges = Matrix{Float64}(undef, S, n), Matrix{Float64}(undef, S, n), zeros(S, T)
    y1n, y2n, opre = zeros(n), zeros(n), zeros(n)
    x1, x2, x1n = zeros(ns), zeros(ns), zeros(ns)
    kxx, kuu, kxu = zeros(ns^2), zeros(nx^2), zeros(ns * nx)
    q = zeros(h.N)
    u = zeros(nx)
    for t in 1:S
        @inbounds for j in 1:nx
            u[j] = shocks[t, j]
        end
        @inbounds for i in 1:ns
            x1[i] = y1[sv[i]]; x2[i] = y2[sv[i]]
        end
        if h.order == 2
            _kron_self!(kxx, x1); _kron_self!(kuu, u); _kron2!(kxu, x1, u)
            mul!(y2n, h.ghx, x2)
            mul!(y2n, h.ghxx, kxx, 0.5, 1.0)
            mul!(y2n, h.ghuu, kuu, 0.5, 1.0)
            mul!(y2n, h.ghxu, kxu, 1.0, 1.0)
            y2n .+= h.cst2
        end
        mul!(y1n, h.ghx, x1)
        mul!(y1n, h.ghu, u, 1.0, 1.0)
        y1, y1n = y1n, y1
        y2, y2n = y2n, y2
        @inbounds for i in 1:ns
            x1n[i] = y1[sv[i]]
        end
        h.order == 2 && _kron_self!(kxx, x1n)
        mul!(opre, h.P, o)
        copyto!(q, h.qc)
        mul!(q, h.Q1, y1, 1.0, 1.0)
        mul!(q, h.Qo, opre, 1.0, 1.0)
        if h.order == 2
            mul!(q, h.Q2, y2, 1.0, 1.0)
            mul!(q, h.Qw, kxx, 1.0, 1.0)
        end
        q[1] += dot(h.aC, prev) + dot(h.aJ, u)
        w = ho_news_shocks(h, q, t)
        mul!(o, h.V0, w)
        o .+= opre
        @inbounds for i in 1:n
            b = y1[i] + y2[i]
            without_b[t, i] = ys[i] + b
            with_b[t, i] = ys[i] + b + o[i]
            prev[i] = b + o[i]
        end
        wedges[t, :] = w
    end
    (with_bound=with_b, without_bound=without_b, wedges=wedges, stats=copy(h.stats))
end

# ------------------------------------------------------------------------------------ entry point (simulate with obc_order / obc_cubature)
const _OBC_HO_CACHE = WeakKeyDict{Occbin,Dict{Any,ObcHO}}()

"The `obc_order` value as an Int (1 or 2)."
function _obc_order(x)
    v = x isa AbstractArray && length(x) == 1 ? x[1] : x
    v isa AbstractString && (v = parse(Float64, v))
    (v isa Real && v in (1, 2)) || throw(ArgumentError("obc_order must be 1 or 2 (got $x); order 3 is not implemented"))
    Int(v)
end
"A positive integer option (`obc_periods_of_uncertainty`, `obc_cubature_regions`) or a non-negative one (`obc_qmc_level`)."
function _obc_int_option(x, name::AbstractString; min::Integer=1)
    v = x isa AbstractArray && length(x) == 1 ? x[1] : x
    v isa AbstractString && (v = parse(Float64, v))
    (v isa Real && isfinite(v) && v == round(v) && v >= min) || throw(ArgumentError("$name must be an integer >= $min (got $x)"))
    Int(v)
end

"""
The simulation of `simulate(ob, shocks; obc_method = :news, obc_order = 2 | obc_cubature = true)`: DynareOBC's loop (`obc_ho_simulate`)
instead of the OccBin-style one of `_simulate_obc`. Differences from the order-1 path: every period is solved (a period without a shock
is a period with a zero realisation, not a continuation of the last path), shocks are padded with zero rows up to `periods`, an
`ObcError` is always raised (`obc_raise = false` is not supported), and `piecewise` is the simulation with the bound, `linear` the
pruned simulation that ignores it. `ss` holds the reference regime's rule in every period; `regime_history` marks the periods with
positive wedges.
"""
function _simulate_obc_ho(ob::Occbin, shocks_in::AbstractMatrix; horizon::Integer, tol::Real, periods::Integer, check_ahead_periods::Integer,
                          max_check_ahead_periods::Real, init, exo_pos, ignored::NamedTuple, order::Integer, cubature::Bool,
                          periods_of_uncertainty::Integer, qmc_level::Integer, regions::Integer)
    m, n, n_exo = ob.model, ob.n, ob.n_exo
    changed = [String(key) for (key, v) in pairs(ignored) if !isequal(v, getfield(_OBC_IGNORED_DEFAULTS, key))]
    isempty(changed) || warn_once(m, "obc_order = $order: OccBin options of the regime iteration are not used (ignored): " * join(sort(changed), ", "))
    exo_idx = exo_pos === nothing ? collect(1:n_exo) : Int[_exo_index(m, p) for p in (exo_pos isa Union{AbstractString,Integer} ? [exo_pos] : exo_pos)]
    size(shocks_in, 2) == length(exo_idx) ||
        throw(ArgumentError("shocks have $(size(shocks_in, 2)) columns but exo_pos selects $(length(exo_idx)) shocks"))
    S = size(shocks_in, 1)
    S >= 1 || throw(ArgumentError("at least one shock period is needed"))
    n_periods = max(Int(periods), S)
    shocks = zeros(n_periods, n_exo)
    shocks[1:S, exo_idx] = Float64.(shocks_in)
    check = Int(min(check_ahead_periods, max_check_ahead_periods))
    key = (Int(horizon), check, Int(order), cubature, Int(periods_of_uncertainty), Int(qmc_level), Int(regions), Float64(tol))
    h = get!(() -> obc_ho(ob; order=order, horizon=horizon, check_ahead=check, cubature=cubature, periods_of_uncertainty=periods_of_uncertainty,
                          qmc_level=qmc_level, regions=regions, tol=tol), get!(() -> Dict{Any,ObcHO}(), _OBC_HO_CACHE, ob), key)
    foreach(k -> h.stats[k] = 0, keys(h.stats))
    y0 = init === nothing || isempty(init) ? nothing : Float64.(vec(collect(init)))
    y0 === nothing || length(y0) == n || throw(ArgumentError("init needs $n values (declaration order)"))
    sim = obc_ho_simulate(h, shocks; init=y0)
    T = h.T
    status = fill(:ok, n_periods)
    infos = Any[(solver=:ho, cubature=cubature, order=order, lcp=h.stats[:lcp], cubature_points=h.stats[:cubature_points],
                 relevant_points=h.stats[:relevant_points]) for _ in 1:n_periods]
    history = Vector{Vector{Tuple{Vector{Int},Vector{Int}}}}()
    diag = _Diag[]
    for t in 1:n_periods
        bi = falses(T, 1)
        for s in 1:T
            bi[s, 1] = sim.wedges[t, s] > tol
        end
        push!(history, [_map_regime(view(bi, :, 1))])
        push!(diag, (max_err=[0.0], violates=[false], n_viol=[0.0], selected=nothing))
    end
    ss = fill((ob.P, ob.Q, zeros(n)), n_periods)
    news = ObcNews(:news, T, [c.name for c in h.sys.cons], sim.wedges, status, infos, :lcp, nothing)
    OccbinResult(sim.with_bound, sim.without_bound, Matrix{Float64}(shocks_in), history, ob.ys, zeros(Int, n_periods), exo_idx,
                 falses(n_periods), diag, m.endo_names, m.exo_names, ss, news)
end
