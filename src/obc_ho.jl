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
    aC::Matrix{Float64}          # nc x n: gap coefficients per constraint (lag, current, lead of the alternative residual's gradient, shocks)
    aB::Matrix{Float64}
    aA::Matrix{Float64}
    aJ::Matrix{Float64}          # nc x nx
    a0::Vector{Float64}          # nc
    Mf::Matrix{Float64}          # NR x nT: sys.M over the check-ahead rows (head rows first, then the rows beyond the horizon, constraint-major)
    Mb::Matrix{Float64}          # (NR - nT) x nT: the rows beyond the horizon
    Gq::Matrix{Float64}          # NR x nx: response of the gap path to a shock in period 0
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
    xs_src::Matrix{Float64}      # cache of `xi` sorted by norm: the matrix it was built from (identity-compared), sorted points, norms
    xs::Matrix{Float64}
    xnorm::Vector{Float64}
    Rnorm::Vector{Float64}       # row norms of R
    pbuf::Matrix{Float64}        # N x npoints work space
    tol::Float64
    stats::Dict{Symbol,Int}
    cub::CubState                # work space, factorisation cache and mode of the full average (src/obc_cubature.jl)
    nc::Int                      # constraints (each a complementary pair)
    nT::Int                      # wedges: nc T, stacked constraint by constraint
    NR::Int                      # rows of q: nc N, the head rows (nT) first, then the rows beyond the horizon, constraint-major in both blocks
    rows::Matrix{Int}            # N x nc: row of q of constraint c in gap period t
end
Base.show(io::IO, h::ObcHO) = print(io, "ObcHO(order = ", h.order, ", T = ", h.T, ", N = ", h.N, h.cubature ? ", cubature: " * string(size(h.R, 2)) * " dims, " *
                                    string(size(h.xi, 2)) * " points" : "", ")")

"""
The `obc_cubature_regions` option: `1` (one region), `k > 1` (DynareOBC's k regions) or `all` (`:all`, `"all"` or `0`: the average over
every relevant point). Returned as an Int, 0 for `all`.
"""
function _obc_regions(x)
    v = x isa AbstractArray && length(x) == 1 ? x[1] : x
    (v isa Symbol || v isa AbstractString) && lowercase(strip(string(v))) == "all" && return 0
    v isa AbstractString && (v = tryparse(Float64, v); v === nothing && throw(ArgumentError("obc_cubature_regions must be 1, k or all (got $x)")))
    (v isa Real && isfinite(v) && v == round(v) && v >= 0) || throw(ArgumentError("obc_cubature_regions must be 1, k or all (got $x)"))
    Int(v)
end

"Reorder the rows of a decision-rule array from decision-rule order to declaration order."
_dr_rows(A::AbstractMatrix, ov) = (B = similar(A); B[ov, :] = A; Matrix{Float64}(B))

"""
    obc_ho(ob::Occbin; order=2, horizon=40, check_ahead=40, cubature=false, periods_of_uncertainty=16, qmc_level=11, regions="all",
           taper=true, pruning_cutoff=0.01, max_cubature_dimension=128, tol=1e-10, Sigma_e=nothing) -> ObcHO

Build the higher-order news solver of a model with one complementarity constraint (the class the order-1 `news` method handles); order 1
or 2. `horizon`, `check_ahead`: periods with wedges and further periods over which the gap must stay non-negative (the gap path has
`N = horizon + check_ahead` periods). Cubature: `periods_of_uncertainty` (DynareOBC's `PeriodsOfUncertainty`), `qmc_level` (the points are
the first `2^(level+1) - 1` of the unscrambled Sobol sequence after the origin), `regions` (only 1: one region, DynareOBC's default).
"""
function obc_ho(ob::Occbin; order::Integer=2, horizon::Integer=40, check_ahead::Integer=40, cubature::Bool=false,
                periods_of_uncertainty::Integer=16, qmc_level::Integer=11, regions="all", taper::Bool=true,
                pruning_cutoff=0.01, max_cubature_dimension::Integer=128, tol::Real=1e-10, Sigma_e=nothing,
                m_diagnostic=:s, cubature_milp_time_limit::Real=0.1, cubature_failure_tolerance::Real=0.0,
                cubature_undecided_tolerance=0.005, cubature_undecided_min_points=3)
    order in (1, 2) || throw(ArgumentError("obc_order must be 1 or 2 (order 3 is not implemented)"))
    regions = _obc_regions(regions)
    periods_of_uncertainty >= 1 || throw(ArgumentError("obc_periods_of_uncertainty must be at least 1"))
    qmc_level >= 0 || throw(ArgumentError("obc_qmc_level must be non-negative"))
    m = ob.model
    T, N = Int(horizon), Int(horizon) + Int(check_ahead)
    check_ahead >= 0 || throw(ArgumentError("check_ahead_periods must be non-negative"))
    sys = obc_system(ob; horizon=T, method=:news)
    nc = length(sys.cons)
    nT, NR = nc * T, nc * N
    rows = [t <= T ? (c - 1) * T + t : nT + (c - 1) * (N - T) + (t - T) for t in 1:N, c in 1:nc]
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
    G, a0 = sys.G, Vector{Float64}(sys.g0)
    aC, aB, aA, aJ = Matrix(G[:, 1:n]), Matrix(G[:, n+1:2n]), Matrix(G[:, 2n+1:3n]), Matrix(G[:, 3n+1:3n+nx])
    ext = _obc_ext(sys, N)
    Mf = vcat(sys.M, ext.Mb)
    Qxf = vcat(sys.Qx, ext.Qxb)
    Gq = Qxf[:, n+1:n+nx]
    h = ObcHO(ob, sys, Int(order), T, N, n, nx, ns, sv, Vector{Float64}(ob.ys), Matrix{Float64}(ob.P), Matrix{Float64}(sys.Wn), Sigma,
              ghx, ghu, ghxx, ghxu, ghuu, cst2, c2, ghxs, ghus, SigS, aC, aB, aA, aJ, a0, Mf, ext.Mb, Gq,
              zeros(NR), zeros(NR, n), zeros(NR, n), zeros(NR, ns^2), zeros(NR, n),
              cubature, Int(periods_of_uncertainty), Int(qmc_level), regions, taper,
              pruning_cutoff === nothing ? nothing : Float64(pruning_cutoff), Int(max_cubature_dimension), zeros(NR, 0), zeros(0, 0),
              zeros(0, 0), zeros(0, 0), Float64[], Float64[], zeros(0, 0), Float64(tol),
              Dict(:lcp => 0, :cubature_points => 0, :relevant_points => 0), CubState(sys.M, NR), nc, nT, NR, rows)
    _ho_build_q_map!(h)
    h.cub.milp_time_limit = _obc_float_option(cubature_milp_time_limit, "obc_cubature_milp_time_limit"; lo=0.0)
    h.cub.tolerance = _obc_float_option(cubature_failure_tolerance, "obc_cubature_failure_tolerance"; lo=0.0, hi=1.0, open_hi=true)
    h.cub.undecided_tolerance = _obc_float_option(cubature_undecided_tolerance, "obc_cubature_undecided_tolerance"; lo=0.0, hi=1.0, open_hi=true)
    h.cub.undecided_min_points = _obc_int_option(cubature_undecided_min_points, "obc_cubature_undecided_min_points"; min=0)
    if cubature
        lev = _obc_mdiag(m_diagnostic)
        h.cub.m_diag = cub_m_diagnostic(sys.M, length(sys.cons); level=lev)
        h.cub.m_diag !== nothing && !h.cub.m_diag.is_s &&
            @warn "obc_cubature: M is not an S-matrix: some paths can have no solution; cubature points may fail (see h.cub.m_diag; obc_m_diagnostic = false skips this test)"
    end
    if cubature
        h.R = ho_conditional_root(h)
        h.xi = qmc_normal_points(size(h.R, 2), h.qmc_level)
    end
    h
end

# row i of the result is the coefficient row of ghxx[i, :] on kron(x, x) after substituting x -> Phi x: vec(Phi' G_i Phi) (G_i the ns x ns
# matrix of the row), i.e. ghxx * kron(Phi, Phi) without forming the ns^2 x ns^2 matrix.
function _quad_transform(ghxx::AbstractMatrix, Phi::AbstractMatrix)
    ns = size(Phi, 1)
    out = similar(ghxx)
    for i in axes(ghxx, 1)
        G = reshape(view(ghxx, i, :), ns, ns)
        out[i, :] = vec(Phi' * G * Phi)
    end
    out
end

# The affine map of q. X_k, the expected total at period k, is  A1_k y1 + A2_k y2 + Aw_k kron(y1s, y1s) + P^k o_pre + d_k  (all in
# declaration order); q_t = a0 + aC' X_{t-1} + aB' X_t + aA' X_{t+1}. Only the contractions with the gap coefficients are kept.
function _ho_build_q_map!(h::ObcHO)
    n, ns, N, sv, order, nc = h.n, h.ns, h.N, h.sv, h.order, h.nc
    for c in 1:nc, t in 1:N
        h.qc[h.rows[t, c]] = h.a0[c]
    end
    fill!(h.Q1, 0.0); fill!(h.Q2, 0.0); fill!(h.Qw, 0.0); fill!(h.Qo, 0.0)
    coefs = ((h.aB, 0), (h.aA, -1), (h.aC, 1))              # (gap coefficients, row offset: X_k enters row k + offset)
    # k = 0: X_0 = y1 + y2 + o_pre
    for c in 1:nc, (A, off) in coefs
        r = 0 + off + 1
        1 <= r <= N || continue
        rr = h.rows[r, c]
        @views h.Q1[rr, :] .+= A[c, :]
        @views h.Q2[rr, :] .+= A[c, :]
        @views h.Qo[rr, :] .+= A[c, :]
    end
    Z = copy(h.ghx)                                         # n x ns: y1[sv] -> m1_k
    U = copy(h.ghx)                                         # y2[sv] -> m2_k
    Vw = 0.5 .* h.ghxx                                      # n x ns^2: kron(y1s, y1s) -> m2_k   (k = 1)
    e = copy(h.c2)                                          # constant of m2_k                  (k = 1)
    Phi = copy(h.ghxs)                                      # W_k = Phi_k W0 Phi_k' + Wc_k, at k = 1
    Wc = copy(h.SigS)
    Pk = copy(h.P)                                          # P^k
    for k in 1:N
        for c in 1:nc, (A, off) in coefs
            r = k + off + 1
            1 <= r <= N || continue
            rr = h.rows[r, c]
            a = view(A, c, :)
            @views h.Q1[rr, sv] .+= Z' * a
            h.qc[rr] += dot(a, e)
            if order == 2
                @views h.Q2[rr, sv] .+= U' * a
                @views h.Qw[rr, :] .+= Vw' * a
            end
            @views h.Qo[rr, :] .+= Pk' * a
        end
        k == N && break
        # advance X_k -> X_{k+1}
        if order == 2
            Vw = h.ghx * Vw[sv, :] .+ 0.5 .* _quad_transform(h.ghxx, Phi)
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

"The gap along an expected path (`N + 1` rows) by the definition (stacked like `q`): the reference for the affine map."
function ho_gap_path(h::ObcHO, path::AbstractMatrix, prev_total::AbstractVector, shock::AbstractVector)
    N = h.N
    q = zeros(h.NR)
    for c in 1:h.nc
        for t in 1:N
            r = h.rows[t, c]
            q[r] = h.a0[c] + dot(view(h.aB, c, :), view(path, t, :)) + dot(view(h.aA, c, :), view(path, t + 1, :))
            t > 1 && (q[r] += dot(view(h.aC, c, :), view(path, t - 1, :)))
        end
        q[h.rows[1, c]] += dot(view(h.aC, c, :), prev_total) + dot(view(h.aJ, c, :), shock)
    end
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
    R = zeros(h.NR, 0)
    blocks = Matrix{Float64}[]
    for i in 1:h.pou
        wi = h.taper ? 0.5 * (1 + cos(pi * (i - 1) / h.pou)) : 1.0
        B = zeros(h.NR, size(L, 2))
        if i < N                                             # per constraint: the gap i periods later responds like the gap now to a shock now
            for c in 1:h.nc
                B[h.rows[i+1:N, c], :] = sqrt(wi) .* (h.Gq[h.rows[1:N-i, c], :] * L)
            end
        end
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
    T, N = h.nT, h.NR
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
        hz = sort(unique([max(1, h.T ÷ 4), max(1, h.T ÷ 2), h.T]))
        y, st, info = solve_news(h.sys.M, Vector(qs); tol=tol, Mb=h.Mb, qb=Vector(view(q, T+1:N)), horizons=hz, n_c=h.nc)
    end
    st == :ok || throw(ObcError(st, period, "obc_method = news (order $(h.order)): $st" *
                                (hasproperty(info, :message) ? ". " * info.message : "")))
    y
end

function _ho_prepare_points!(h::ObcHO)
    h.xs_src === h.xi && length(h.Rnorm) == size(h.R, 1) && return
    ord = sortperm(vec(sqrt.(sum(abs2, h.xi; dims=1))))
    h.xs = h.xi[:, ord]
    h.xnorm = vec(sqrt.(sum(abs2, h.xs; dims=1)))
    h.Rnorm = vec(sqrt.(sum(abs2, h.R; dims=2)))
    h.pbuf = zeros(h.NR, size(h.xi, 2))
    h.xs_src = h.xi
    nothing
end

"""
    ho_news_shocks(h, q, period=0) -> Vector

The news shocks at gap path `q`: the LCP solution, or with cubature its average over the perturbed paths `q + R xi`. A path is relevant if it goes
negative (below `-1e-8 max(1, max|p|)`); the others contribute zero. `regions = 1` (DynareOBC's default): the solution at the mean of the relevant paths,
times their share (it does not price tail risk: the wedge of the average path is not the average of the wedges). `regions = 0` (option value `all`):
the average of the solutions at every relevant path (`_cub_all!`).
Exact skipping: `p_h >= q_h - |R_h| |xi|` (Cauchy-Schwarz), so when `q >= 0` every point with `|xi| < min_h q_h / |R_h|` is irrelevant
(points are sorted by norm: a prefix); and a path with `min p >= -1e-8` is irrelevant whatever its maximum.
"""
function ho_news_shocks(h::ObcHO, q::AbstractVector, period::Int=0)
    h.cub.dropped_last = 0; h.cub.relevant_last = 0; h.cub.dropped_cls = (0, 0, 0)
    h.cub.qlog === nothing || push!(h.cub.qlog, Vector{Float64}(q))
    (!h.cubature || size(h.R, 2) == 0) && return _ho_lcp(h, q, period)
    _ho_prepare_points!(h)
    N, npts = h.NR, size(h.xs, 2)
    first = 1                                               # points before `first` are irrelevant
    if minimum(q) >= 0
        thr = Inf
        for i in 1:N
            h.Rnorm[i] > 0 && (thr = min(thr, q[i] / h.Rnorm[i]))
        end
        thr *= 1 - 1e-9
        first = searchsortedfirst(h.xnorm, thr)             # first point with |xi| >= thr
        if first > npts
            h.stats[:cubature_points] += npts
            return zeros(h.nT)
        end
    end
    nc = npts - first + 1
    P = view(h.pbuf, :, 1:nc)
    mul!(P, h.R, view(h.xs, :, first:npts))
    nrel = 0
    xsum = zeros(size(h.R, 2))
    rel = h.cub.rel
    empty!(rel)
    @inbounds for j in 1:nc
        pmin = Inf
        for i in 1:N
            pmin = min(pmin, q[i] + P[i, j])
        end
        pmin >= -1e-8 && continue
        pmax = 1.0
        for i in 1:N
            pmax = max(pmax, abs(q[i] + P[i, j]))
        end
        if pmin < -1e-8 * pmax
            nrel += 1
            push!(rel, j)
            if h.regions == 1
                for k in eachindex(xsum)
                    xsum[k] += h.xs[k, first+j-1]
                end
            end
        end
    end
    h.stats[:cubature_points] += npts
    h.stats[:relevant_points] += nrel
    h.cub.relevant_last = nrel
    nrel == 0 && return zeros(h.nT)
    h.regions == 0 && return _cub_all!(h, q, P, rel, npts, period)
    h.regions > 1 && return _cub_regions!(h, q, P, rel, first, npts, period)
    qm = q .+ h.R * (xsum ./ nrel)
    _cub_one!(h, qm, nrel, npts, period)
end

"""
    _cub_classify(h, p) -> (y | nothing, class | nothing)

A point on which the scalar iteration failed (pydyn's `_classify`): the linear programme on the head rows (`s* > 1e-6 max(1, |p|)`: class `a`, proved no solution), then on all rows
(class `b`: the check-ahead rows cannot be met), then the MILP (`solve_news` at horizons T/4, T/2, T in turn; one time budget `cubature_milp_time_limit` shared by all MILP calls of the
point): a solution returns `y`; time out: `c` (undecided); no solution at any horizon: `b` if the MILP on the head rows alone (horizon T) finds one, else `a`. A point whose LP is feasible by less
than the tolerance goes to the MILP, not to `a`. Both LPs are native HiGHS models built once from `M` (`CubLP`), only the right-hand side changes per point.
"""
function _cub_classify(h::ObcHO, p::AbstractVector)
    st, T, N = h.cub, h.nT, h.NR
    st.lp_head === nothing && (st.lp_head = CubLP(h.sys.M); st.lp_full = CubLP(h.Mf))
    h.stats[:classified] = get(h.stats, :classified, 0) + 1
    tol = 1e-6 * max(1.0, maximum(abs, p))
    s_head = _cublp_gap(st.lp_head, p)
    s_head !== nothing && s_head > tol && return nothing, :a
    s_full = s_head === nothing ? nothing : _cublp_gap(st.lp_full, p)
    s_full !== nothing && s_full > tol && return nothing, :b
    t_end = time() + st.milp_time_limit
    ph = Vector{Float64}(view(p, 1:T)); pb = Vector{Float64}(view(p, T+1:N))
    undecided = false
    for hz in sort(unique([max(1, h.T ÷ 4), max(1, h.T ÷ 2), h.T]))
        left = t_end - time()
        left <= 0 && (undecided = true; break)
        h.stats[:milp] = get(h.stats, :milp, 0) + 1
        y, stt, _ = solve_news(h.sys.M, ph; tol=h.tol, Mb=h.Mb, qb=pb, horizons=[hz], time_limit=left, diagnose=false, n_c=h.nc)
        stt == :ok && return y, nothing
        stt == :not_converged && (undecided = true)
    end
    undecided && return nothing, :c
    left = max(t_end - time(), 1e-3)
    y2, s2, _ = solve_news(h.sys.M, ph; tol=h.tol, horizons=[h.T], time_limit=left, diagnose=false, n_c=h.nc)
    nothing, s2 == :ok ? :b : :a
end

"Statistics of a period that dropped `nfail` points (`counts` by class)."
function _cub_record!(h::ObcHO, counts, nfail::Int)
    nfail > 0 || return
    for c in (:a, :b, :c)
        h.stats[Symbol("failed_", c)] = get(h.stats, Symbol("failed_", c), 0) + getfield(counts, c)
    end
    h.stats[:dropped_points] = get(h.stats, :dropped_points, 0) + nfail
    h.stats[:dropped_periods] = get(h.stats, :dropped_periods, 0) + 1
    nothing
end

"Raise the `ObcError` (payload `CubatureFailure`) of a period whose failures exceed the rule."
function _cub_raise!(h::ObcHO, period, counts, nfail, unexamined, nrel, npts, n_as)
    h.stats[:no_solution] = get(h.stats, :no_solution, 0) + 1
    for c in (:a, :b, :c)
        h.stats[Symbol("failed_", c)] = get(h.stats, Symbol("failed_", c), 0) + getfield(counts, c)
    end
    throw(_cub_failure(period, counts, nfail, unexamined, nrel, npts, n_as, h.cub.tolerance, h.cub.milp_time_limit, h.cub.undecided_tolerance))
end

"""
    _cub_all!(h, q, P, rel, npts, period) -> Vector

The full average: the solution at `q + P[:, j]` for every relevant column `j` in `rel`, summed and divided by the number of points `npts`.
A point on which the scalar iteration (or the rows beyond the horizon) fails is classified (`_cub_classify`) and solved by the MILP if it can be. With `F` points without a usable
solution the result is `(sum over the solved relevant points) / (npts - F)`; `F > tolerance * (relevant points)` raises an `ObcError` with a `CubatureFailure` payload (`e.detail`),
by default (tolerance 0) at the first failed point. Non-relevant points are legitimate zeros; failed points are the only ones about which nothing is known, so they leave the sample.
"""
function _cub_all!(h::ObcHO, q::AbstractVector, P::AbstractMatrix, rel::Vector{Int}, npts::Int, period::Int)
    st, T, N = h.cub, h.nT, h.NR
    acc = fill!(st.acc, 0.0)
    pt = st.pt
    Mh = h.sys.M
    scale = st.Mscale
    tol = h.tol
    nb = N - T
    bad = Int[]                                              # columns (of P) on which the scalar iteration failed
    if st.mode == :group
        m = length(rel)
        size(st.Pm, 2) < m && (st.Pm = zeros(N, m); st.Y = zeros(T, m); resize!(st.ok, m))
        Pm = view(st.Pm, :, 1:m)
        @inbounds for (jj, j) in enumerate(rel), i in 1:N
            Pm[i, jj] = q[i] + P[i, j]
        end
        Y = view(st.Y, :, 1:m); okv = view(st.ok, 1:m)
        _cub_group!(st, Mh, scale, view(Pm, 1:T, :), T, tol, st.Y, st.ok)
        h.stats[:lcp] += m
        for jj in 1:m
            good = okv[jj]
            if good && nb > 0
                @inbounds for i in 1:nb
                    v = Pm[T+i, jj]
                    for a in 1:T
                        v += h.Mb[i, a] * Y[a, jj]
                    end
                    v < -tol && (good = false; break)
                end
            end
            if good
                @inbounds for a in 1:T
                    acc[a] += Y[a, jj]
                end
            else
                push!(bad, rel[jj])
            end
        end
    else
        for j in rel
            @inbounds for i in 1:N
                pt[i] = q[i] + P[i, j]
            end
            h.stats[:lcp] += 1
            ok = false
            if st.mode == :scalar
                y, s, _ = solve_active_set(Mh, view(pt, 1:T); tol=tol)
                s == :ok && (ok = true; copyto!(st.y, y))
            else
                ok = _cub_point!(st, Mh, scale, pt, T, tol)
            end
            if ok && nb > 0                                  # the check-ahead rows beyond the horizon
                @inbounds for i in 1:nb
                    v = pt[T+i]
                    for a in 1:T
                        v += h.Mb[i, a] * st.y[a]
                    end
                    v < -tol && (ok = false; break)
                end
            end
            if ok
                @inbounds for a in 1:T
                    acc[a] += st.y[a]
                end
            else
                push!(bad, j)
            end
        end
    end
    nfail = 0
    counts = (a=0, b=0, c=0)
    if !isempty(bad)
        h.stats[:batch_fail] = get(h.stats, :batch_fail, 0) + length(bad)
        nrel = length(rel)
        a_lim, limit = cubature_limits(nrel, st.tolerance, st.undecided_tolerance, st.undecided_min_points)
        for (i, j) in enumerate(bad)
            p = [q[r] + P[r, j] for r in 1:N]
            y, cls = _cub_classify(h, p)
            if cls === nothing
                acc .+= y
            else
                counts = merge(counts, NamedTuple{(cls,)}((getfield(counts, cls) + 1,)))
                nfail += 1
                if counts.a > a_lim || nfail > limit                 # decided: stop (the counts are then lower bounds when points are left)
                    _cub_raise!(h, period, counts, nfail, length(bad) - i, nrel, npts, length(bad))
                end
            end
        end
        _cub_record!(h, counts, nfail)
    end
    st.dropped_last = nfail
    st.dropped_cls = (counts.a, counts.b, counts.c)
    acc ./ (npts - nfail)
end

"""
    _cub_unit(h, p) -> (y, class | nothing)

One cubature unit (a region's mean point): zeros when nothing binds, else the scalar iteration (and the rows beyond the horizon), else `_cub_classify`.
"""
function _cub_unit(h::ObcHO, p::AbstractVector)
    T, N = h.nT, h.NR
    h.stats[:lcp] += 1
    qmax = max(1.0, maximum(abs, p))
    all(>=(-1e-8 * qmax), p) && return zeros(T), nothing
    y, s, _ = solve_active_set(view(h.Mf, 1:T, :), view(p, 1:T); tol=h.tol)
    if s == :ok && N > T && minimum(view(p, T+1:N) .+ h.Mb * y) < -h.tol
        s = :not_converged
    end
    s == :ok && return y, nothing
    _cub_classify(h, p)
end

"""
    _cub_one!(h, qm, nrel, npts, period) -> Vector

One region: the solution at the mean point `qm` of the relevant paths, times `nrel / npts`. The unit is the region (all its `nrel` points count as failed if it fails); a failure is
classified and raised like the points of the full average (the tolerance is below 1, so a failed region always raises).
"""
function _cub_one!(h::ObcHO, qm::AbstractVector, nrel::Int, npts::Int, period::Int)
    y, cls = _cub_unit(h, qm)
    if cls !== nothing
        counts = merge((a=0, b=0, c=0), NamedTuple{(cls,)}((1,)))
        a_lim, limit = cubature_limits(nrel, h.cub.tolerance, h.cub.undecided_tolerance, h.cub.undecided_min_points)
        if counts.a > a_lim || nrel > limit
            _cub_raise!(h, period, counts, nrel, 0, nrel, npts, nothing)
        end
        _cub_record!(h, counts, nrel)
        h.cub.dropped_last = nrel
        h.cub.dropped_cls = (counts.a, counts.b, counts.c)
        return zeros(h.nT)
    end
    (nrel / npts) .* y
end

"""
    _split_regions(pr, k, cutoff, maxdim) -> Vector{Vector{Int}}

DynareOBC's region rule (PerformCubature.m, CubatureClusteringEffort = 0): each relevant path (a column of `pr`) is described by its direction and the direction of its negative part,
reduced by principal components (eigendecomposition of the `2N x 2N` Gram matrix; components below `cutoff` times the largest variance dropped, at most `maxdim` kept), and the region
with the largest within-sum-of-squares is split along its first principal axis at Otsu's threshold, until there are `k` regions. Returns `k` index vectors (increasing) into the columns of `pr`.
The component signs are arbitrary and the split is symmetric under a sign change; the partition differs only when a point lies within rounding of a threshold.
"""
function _split_regions(pr::AbstractMatrix, k::Int, cutoff, maxdim::Int)
    m = size(pr, 2)
    X0 = pr ./ sqrt.(sum(abs2, pr; dims=1))
    X = hcat(X0', min.(0.0, X0'))                               # m x 2NR
    X .-= sum(X; dims=1) ./ m
    ev, V = eigen(Symmetric(X' * X))
    ev, V = reverse(ev), V[:, end:-1:1]
    lat = max.(ev, 0.0) ./ max(1, m - 1)
    keep = cutoff === nothing ? trues(length(lat)) : lat .>= cutoff * lat[1]
    keep[maxdim+1:end] .= false
    keep .&= lat .> eps()
    Z = X * V[:, keep]
    sse(ix) = (Zi = Z[ix, :]; sum(abs2, Zi .- sum(Zi; dims=1) ./ length(ix)))
    groups = [collect(1:m)]
    sses = [sse(groups[1])]
    for n in 2:k
        r = n == 2 ? 1 : argmax(sses)
        pi_ = groups[r]
        if n == 2
            proj = Z[:, 1]
        else
            Zp = Z[pi_, :]
            Zc = Zp .- sum(Zp; dims=1) ./ length(pi_)
            C = (Zc' * Zc) ./ max(1, length(pi_) - 1)
            ev2, V2 = eigen(Symmetric(C))
            proj = Zp * V2[:, argmax(abs.(ev2))]
        end
        proj = proj .- sum(proj) / length(proj)
        srt = sort(proj)
        cs = cumsum(srt[1:end-1])
        j = 1:length(proj)-1
        cut = srt[argmax(cs .^ 2 ./ (j .* (length(proj) .- j)))]
        right = proj .> cut
        groups[r] = pi_[.!right]
        new = pi_[right]
        push!(groups, new)
        sses[r] = sse(groups[r])
        push!(sses, isempty(new) ? 0.0 : sse(new))
    end
    groups
end

"""
    _cub_regions!(h, q, P, rel, first, npts, period) -> Vector

`regions = k`: the relevant paths are split into `k` regions (`_split_regions`); each region contributes its size times the solution at its mean point `q + R mean(xi)`; the sum is divided by
`npts`. Failed regions are classified, counted in points (a failed region counts all its points) and handled like failed points of the full average. Fewer relevant points than regions: every point is its own region.
"""
function _cub_regions!(h::ObcHO, q::AbstractVector, P::AbstractMatrix, rel::Vector{Int}, first::Int, npts::Int, period::Int)
    st, T, N = h.cub, h.nT, h.NR
    nrel = length(rel)
    k = h.regions
    xr = h.xs[:, first-1 .+ rel]
    groups = nrel <= k ? [[j] for j in 1:nrel] : _split_regions(q .+ view(P, :, rel), k, h.pruning_cutoff, h.max_dim)
    a_lim, limit = cubature_limits(nrel, st.tolerance, st.undecided_tolerance, st.undecided_min_points)
    acc = zeros(T)
    nfail = 0
    counts = (a=0, b=0, c=0)
    for (i, g) in enumerate(groups)
        isempty(g) && continue
        qm = q .+ h.R * vec(sum(view(xr, :, g); dims=2) ./ length(g))
        y, cls = _cub_unit(h, qm)
        if cls === nothing
            acc .+= length(g) .* y
        else
            counts = merge(counts, NamedTuple{(cls,)}((getfield(counts, cls) + length(g),)))
            nfail += length(g)
            if counts.a > a_lim || nfail > limit
                _cub_raise!(h, period, counts, nfail, length(groups) - i, nrel, npts, nothing)
            end
        end
    end
    _cub_record!(h, counts, nfail)
    st.dropped_last = nfail
    st.dropped_cls = (counts.a, counts.b, counts.c)
    acc ./ (npts - nfail)
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
    foreach(k -> h.stats[k] = 0, keys(h.stats))
    S = size(shocks, 1)
    size(shocks, 2) == nx || throw(ArgumentError("shocks need $nx columns (got $(size(shocks, 2)))"))
    y1, y2, o = zeros(n), zeros(n), zeros(n)
    init === nothing || (y1 .= init)
    prev = y1 .+ y2 .+ o
    with_b, without_b, wedges = Matrix{Float64}(undef, S, n), Matrix{Float64}(undef, S, n), zeros(S, h.nT)
    dropped, relevant = zeros(Int, S), zeros(Int, S)
    dcls = zeros(Int, S, 3)
    y1n, y2n, opre = zeros(n), zeros(n), zeros(n)
    x1, x2, x1n = zeros(ns), zeros(ns), zeros(ns)
    kxx, kuu, kxu = zeros(ns^2), zeros(nx^2), zeros(ns * nx)
    q = zeros(h.NR)
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
        for c in 1:h.nc
            q[h.rows[1, c]] += dot(view(h.aC, c, :), prev) + dot(view(h.aJ, c, :), u)
        end
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
        dropped[t], relevant[t] = h.cub.dropped_last, h.cub.relevant_last
        dcls[t, :] .= h.cub.dropped_cls
    end
    if any(>(0), dropped)
        share = maximum(dropped[t] / max(1, relevant[t]) for t in 1:S)
        @warn cubature_warning(h.cub.undecided_tolerance, h.cub.undecided_min_points, (sum(dcls[:, 1]), sum(dcls[:, 2]), sum(dcls[:, 3])),
                               sum(relevant), share, count(>(0), dropped), S)
    end
    (with_bound=with_b, without_bound=without_b, wedges=wedges, dropped=dropped, relevant=relevant, dropped_a=dcls[:, 1], dropped_b=dcls[:, 2], dropped_c=dcls[:, 3],
     stats=copy(h.stats))
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
                          periods_of_uncertainty::Integer, qmc_level::Integer, regions::Integer, m_diagnostic=:s,
                          milp_time_limit::Real=0.1, failure_tolerance::Real=0.0,
                          undecided_tolerance=0.005, undecided_min_points=3)
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
    key = (Int(horizon), check, Int(order), cubature, Int(periods_of_uncertainty), Int(qmc_level), Int(regions), Float64(tol), m_diagnostic, Float64(milp_time_limit), Float64(failure_tolerance),
           Float64(undecided_tolerance), Int(undecided_min_points))
    h = get!(() -> obc_ho(ob; order=order, horizon=horizon, check_ahead=check, cubature=cubature, periods_of_uncertainty=periods_of_uncertainty,
                          qmc_level=qmc_level, regions=regions, tol=tol, m_diagnostic=m_diagnostic,
                          cubature_milp_time_limit=milp_time_limit, cubature_failure_tolerance=failure_tolerance,
                          cubature_undecided_tolerance=undecided_tolerance, cubature_undecided_min_points=undecided_min_points), get!(() -> Dict{Any,ObcHO}(), _OBC_HO_CACHE, ob), key)
    foreach(k -> h.stats[k] = 0, keys(h.stats))
    y0 = init === nothing || isempty(init) ? nothing : Float64.(vec(collect(init)))
    y0 === nothing || length(y0) == n || throw(ArgumentError("init needs $n values (declaration order)"))
    sim = obc_ho_simulate(h, shocks; init=y0)
    T = h.T
    status = fill(:ok, n_periods)
    infos = Any[(solver=:ho, cubature=cubature, order=order, lcp=h.stats[:lcp], cubature_points=h.stats[:cubature_points],
                 relevant_points=h.stats[:relevant_points], relevant=sim.relevant[t], dropped=sim.dropped[t],
                 dropped_a=sim.dropped_a[t], dropped_b=sim.dropped_b[t], dropped_c=sim.dropped_c[t]) for t in 1:n_periods]
    history = Vector{Vector{Tuple{Vector{Int},Vector{Int}}}}()
    diag = _Diag[]
    for t in 1:n_periods
        bi = falses(T, h.nc)
        for c in 1:h.nc, s in 1:T
            bi[s, c] = sim.wedges[t, (c-1)*T+s] > tol
        end
        push!(history, [_map_regime(view(bi, :, c)) for c in 1:h.nc])
        push!(diag, (max_err=[0.0], violates=[false], n_viol=[0.0], selected=nothing))
    end
    ss = fill((ob.P, ob.Q, zeros(n)), n_periods)
    news = ObcNews(:news, T, [c.name for c in h.sys.cons], sim.wedges, status, infos, :lcp, nothing)
    OccbinResult(sim.with_bound, sim.without_bound, Matrix{Float64}(shocks_in), history, ob.ys, zeros(Int, n_periods), exo_idx,
                 falses(n_periods), diag, m.endo_names, m.exo_names, ss, news)
end
