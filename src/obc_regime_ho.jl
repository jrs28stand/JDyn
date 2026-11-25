# Order 2 and cubature for the regime formulation (obc_formulation = regime; docs/OBC_HO_PROGRESS.md, "Regime formulation at higher order").
#
# New design with no external reference for the classes that are not exact complementary pairs: the structure of `obc_ho` (pruned base simulation, first-order bound
# offset, expected path with the future shocks at their distribution, quasi-Monte-Carlo averaging over perturbed paths; after T. D. Holden's DynareOBC) combined with
# `RegimeOBC` (one wedge per switched equation, acceptance by OccBin's bind/relax test in levels). Written for JDyn from pydyn's prototype pydyn/obc_regime_ho.py
# (executable specification, docs/OBC_REGIME_HO.md there), not a translation.
#
# Per period the state-dependent inputs of the regime search are the alternative residuals `a` of every switched equation (T periods) and the path `Ycv` of the
# variables the bind/relax conditions read (N periods), both on the no-wedge path. At order 2 they are evaluated on the order-2 expected path of the unconstrained model
# (as `obc_ho`); the response to the wedges stays first order (`R`, `Wcv`, `V0` of `RegimeOBC`). Both are affine in z = (y1 + y2, kron(x1, x1), o_pre), precomputed per
# parameter set (`_hor_affine_maps`, the adjoint recursion of `_ho_build_q_map!` for any list of functionals).
#
# Cubature: the perturbed quantities are the alternative residuals and the condition paths at `base + Ro xi`, where `Ro = Cov(other, gap) pinv(R')` is the regression of
# every other acceptance quantity on the gap-like path that the root `R` describes (for an exact complementary pair they are +-gap, so Ro = +-R). A point is relevant when its
# reference-regime path (no wedges) fails OccBin's bind test; the others contribute zero. Relevant points go through a batched guess-and-verify, grouped by regime guess (one
# LU per distinct guess, many right-hand sides), whose iteration 0 is the relevance test itself; points it does not settle go through the complete search (tied LCP, integer
# programme) one by one. Failed points follow the rule of `cubature_limits` with classes a (an infeasible linear programme certifies it), b (horizon too short), c (rest).

"""
    ObcHORegime

The higher-order solver of the regime formulation: everything that depends on the model and the options but not on the shocks. Build with `obc_ho_regime(ob; ...)`, run
with `obc_hor_simulate`. Fields `R` and `xi` may be replaced (the tests inject DynareOBC's own root and Sobol points); `Ro` is recomputed when `R` is replaced.
"""
mutable struct ObcHORegime
    ob::Occbin
    rg::RegimeOBC
    order::Int
    T::Int
    N::Int
    n::Int
    nx::Int
    ns::Int
    nb::Int                      # wedge blocks (switched equations)
    nc::Int                      # constraints
    nk::Int                      # condition variables
    nws::Int                     # nb T
    sv::Vector{Int}
    ys::Vector{Float64}
    P::Matrix{Float64}
    Sigma::Matrix{Float64}
    ghx::Matrix{Float64}         # declaration order rows, as `ObcHO`
    ghu::Matrix{Float64}
    ghxx::Matrix{Float64}
    ghxu::Matrix{Float64}
    ghuu::Matrix{Float64}
    cst2::Vector{Float64}
    c2::Vector{Float64}
    ghxs::Matrix{Float64}
    ghus::Matrix{Float64}
    SigS::Matrix{Float64}
    rows::Matrix{Int}            # N x nc: row of the gap path (head rows first, then the rows beyond the horizon, constraint-major in both blocks)
    gapfun::Matrix{Float64}      # nc x (3n + nx + 1): the gap-like functional of each constraint
    fC::Matrix{Float64}          # (nb + nk + nc) x n: functionals of the expected path (alternative residuals, condition variables, gap-like), for the direct reference
    fB::Matrix{Float64}
    fA::Matrix{Float64}
    Gall::Matrix{Float64}        # ((nb + nk + nc) N) x nz: functional f, period t in row (f - 1) N + t;  z = (y1 + y2, kron(x1, x1), o_pre)
    c0all::Vector{Float64}
    zvec::Vector{Float64}
    vbuf::Vector{Float64}
    # cubature
    cubature::Bool
    pou::Int
    qmc_level::Int
    regions::Int
    taper::Bool
    pruning_cutoff::Union{Nothing,Float64}
    max_dim::Int
    R::Matrix{Float64}
    Lg::Matrix{Float64}          # effects of the next shocks on the gap-like path / on every other acceptance quantity (rows: alternative residuals (b-1) N + t, condition paths (t-1) k + j)
    Lo::Matrix{Float64}
    xi::Matrix{Float64}
    Ro_src::Matrix{Float64}      # the `R` that `Ro` was computed for (identity-compared)
    Ro::Matrix{Float64}
    milp_time_limit::Float64     # s per cubature point
    tolerance::Float64
    undecided_tolerance::Float64
    undecided_min_points::Int
    root_milp_time_limit::Float64   # s for the one search of a period without cubature
    tol::Float64
    stats::Dict{Symbol,Int}
    dropped_last::Int
    relevant_last::Int
    dropped_cls::NTuple{3,Int}
    trace::Union{Nothing,Vector{Any}}      # when a vector: `obc_hor_simulate` appends the inputs and outputs of every period (tests)
    last::Any                              # (v, prev, shock) of the last `hor_inputs_from_state` (for `hor_gap_path`)
end
Base.show(io::IO, h::ObcHORegime) = print(io, "ObcHORegime(order = ", h.order, ", T = ", h.T, ", N = ", h.N, ", ", h.nb, " wedge block(s)",
                                          h.cubature ? ", cubature: " * string(size(h.R, 2)) * " dims, " * string(size(h.xi, 2)) * " points" : "", ")")

# ------------------------------------------------------------------------------------------------ construction
"""
    obc_ho_regime(ob; order=2, horizon=40, check_ahead=40, cubature=false, periods_of_uncertainty=16, qmc_level=11, regions="all", taper=true, pruning_cutoff=0.01,
                  max_cubature_dimension=128, tol=1e-10, Sigma_e=nothing, cubature_milp_time_limit=0.1, cubature_failure_tolerance=0.0,
                  cubature_undecided_tolerance=0.005, cubature_undecided_min_points=3, milp_time_limit=nothing) -> ObcHORegime

The regime formulation at order 1 or 2, with or without cubature (`regions`: `all`, `1` or `k`, as `obc_ho`). `milp_time_limit` (default `REGIME_MILP_TIME_LIMIT`) is the time of
the integer programmes of the one search of a period without cubature; `cubature_milp_time_limit` that of each cubature point.
"""
function obc_ho_regime(ob::Occbin; order::Integer=2, horizon::Integer=40, check_ahead::Integer=40, cubature::Bool=false, periods_of_uncertainty::Integer=16,
                       qmc_level::Integer=11, regions="all", taper::Bool=true, pruning_cutoff=0.01, max_cubature_dimension::Integer=128, tol::Real=1e-10,
                       Sigma_e=nothing, cubature_milp_time_limit=0.1, cubature_failure_tolerance=0.0, cubature_undecided_tolerance=0.005,
                       cubature_undecided_min_points=3, milp_time_limit=nothing, m_diagnostic=false)
    order in (1, 2) || throw(ArgumentError("obc_order must be 1 or 2 (order 3 is not implemented)"))
    regions = _obc_regions(regions)
    periods_of_uncertainty >= 1 || throw(ArgumentError("obc_periods_of_uncertainty must be at least 1"))
    qmc_level >= 0 || throw(ArgumentError("obc_qmc_level must be non-negative"))
    check_ahead >= 0 || throw(ArgumentError("check_ahead_periods must be non-negative"))
    mtl = _obc_float_option(cubature_milp_time_limit, "obc_cubature_milp_time_limit"; lo=0.0)
    ftol = _obc_float_option(cubature_failure_tolerance, "obc_cubature_failure_tolerance"; lo=0.0, hi=1.0, open_hi=true)
    utol = _obc_float_option(cubature_undecided_tolerance, "obc_cubature_undecided_tolerance"; lo=0.0, hi=1.0, open_hi=true)
    umin = _obc_int_option(cubature_undecided_min_points, "obc_cubature_undecided_min_points"; min=0)
    m = ob.model
    T, N = Int(horizon), Int(horizon) + Int(check_ahead)
    rg = regime_obc(ob, T, N)
    n, nx = ob.n, ob.n_exo
    nb, nc, nk = length(rg.cols), rg.nc, length(rg.cv)
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
    rows = [t <= T ? (c - 1) * T + t : nc * T + (c - 1) * (N - T) + (t - T) for t in 1:N, c in 1:nc]
    h = ObcHORegime(ob, rg, Int(order), T, N, n, nx, ns, nb, nc, nk, nb * T, sv, Vector{Float64}(ob.ys), Matrix{Float64}(ob.P), Sigma,
                    ghx, ghu, ghxx, ghxu, ghuu, cst2, c2, ghxs, ghus, SigS, rows,
                    zeros(nc, 3n + nx + 1), zeros(0, n), zeros(0, n), zeros(0, n), zeros(0, 0), Float64[], Float64[], Float64[],
                    cubature, Int(periods_of_uncertainty), Int(qmc_level), regions, taper, pruning_cutoff === nothing ? nothing : Float64(pruning_cutoff),
                    Int(max_cubature_dimension), zeros(nc * N, 0), zeros(nc * N, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0),
                    mtl, ftol, utol, umin, milp_time_limit === nothing ? REGIME_MILP_TIME_LIMIT[] : Float64(milp_time_limit), Float64(tol),
                    Dict(:regime => 0, :cubature_points => 0, :relevant_points => 0), 0, 0, (0, 0, 0), nothing, nothing)
    _hor_build_maps!(h)
    if cubature
        _hor_init_cubature!(h)
    end
    h
end

"""
The gap-like functional of each constraint (`nc x (3n + nx + 1)`, columns `[C B A J | const]` of the Jacobian rows, as `RegimeOBC.alt_rows`): `s_g` times the alternative residual
of a complementary pair (so the root of an exact model is the LCP's), else the sign-normalised alternative residual of the primary switched equation (the tied one when
`_find_tie` applies, else the first). It builds the cubature root and the region split; the search does not use it.
"""
function _hor_gap_functionals(ob::Occbin, rg::RegimeOBC)
    cons = get!(() -> obc_constraints(ob), _OBC_CONS_CACHE, ob)
    tie = rg.nc == 1 ? _tie(rg) : nothing
    n, nx = ob.n, ob.n_exo
    out = zeros(rg.nc, 3n + nx + 1)
    for (j, co) in enumerate(cons)
        if co.class == :complementarity
            out[j, :] = vcat(co.s_g .* co.alt_resid[1].g, co.s_g * co.alt_resid[1].c)
        else
            pb = first(b for b in eachindex(rg.cols) if rg.jb[b] == j)
            tie !== nothing && (pb = tie.primary)
            r = rg.alt_rows[pb, :]
            out[j, :] = r[end] >= 0 ? r : -r
        end
    end
    out
end

# the functionals of the expected path that the search needs: alternative residuals (nb), condition variables (nk), gap-like (nc)
function _hor_functionals(h::ObcHORegime)
    rg, n = h.rg, h.n
    a = vcat(rg.alt_rows, h.gapfun)
    E = zeros(h.nk, n)
    for (c, i) in enumerate(rg.cv)
        E[c, i] = 1.0
    end
    Z = zeros(h.nk, n)
    nb = h.nb
    aC, aB, aA = a[:, 1:n], a[:, n+1:2n], a[:, 2n+1:3n]
    vcat(aC[1:nb, :], Z, aC[nb+1:end, :]), vcat(aB[1:nb, :], E, aB[nb+1:end, :]), vcat(aA[1:nb, :], Z, aA[nb+1:end, :])
end

function _hor_build_maps!(h::ObcHORegime)
    h.gapfun = _hor_gap_functionals(h.ob, h.rg)
    h.fC, h.fB, h.fA = _hor_functionals(h)
    h.Gall, h.c0all = _hor_affine_maps(h, h.fC, h.fB, h.fA, h.N)
    nz = size(h.Gall, 2)
    h.zvec = zeros(nz)
    h.vbuf = zeros(size(h.Gall, 1))
    h
end

# F' G F for every row of the (m x ns^2) array `G` read as an ns x ns matrix (any consistent orientation: the map is covariant), written into `out`
function _hor_quad_sandwich!(out::AbstractMatrix, G::AbstractMatrix, F::AbstractMatrix)
    ns = size(F, 1)
    Gr, tmp, res = Matrix{Float64}(undef, ns, ns), Matrix{Float64}(undef, ns, ns), Matrix{Float64}(undef, ns, ns)
    for r in axes(G, 1)
        @inbounds for j in 1:ns, i in 1:ns
            Gr[i, j] = G[r, (j-1)*ns+i]
        end
        mul!(tmp, F', Gr)
        mul!(res, tmp, F)
        @inbounds for j in 1:ns, i in 1:ns
            out[r, (j-1)*ns+i] = res[i, j]
        end
    end
    out
end

"""
Affine maps of the functionals `f_t = aC E_{t-1} + aB E_t + aA E_{t+1}`, `t = 1..H`, of the expected path: `f = c0 + Gz z` (row `(f - 1) H + t`; the lag of the first period
and the shock are added by the caller), with `z = (y1 + y2, kron(x1, x1), o_pre)`. The adjoint recursion of `_ho_build_q_map!` for any list of functionals (any number of rows).
"""
function _hor_affine_maps(h::ObcHORegime, aC::AbstractMatrix, aB::AbstractMatrix, aA::AbstractMatrix, H::Int)
    n, sv, ns, P = h.n, h.sv, h.ns, h.P
    two = h.order == 2
    K = H + 1
    m = size(aB, 1)
    m3 = 3m
    Ay = Matrix{Float64}(undef, m3, n)
    for c in 1:m
        Ay[3(c-1)+1, :] = aB[c, :]; Ay[3(c-1)+2, :] = aA[c, :]; Ay[3(c-1)+3, :] = aC[c, :]
    end
    F, ghx = h.ghxs, h.ghx
    rY = [zeros(m3, n) for _ in 1:K]; rY[1] .= Ay
    rO = Vector{Matrix{Float64}}(undef, K); rO[1] = copy(Ay)
    nn = ns * ns
    rW = two ? [zeros(m3, nn) for _ in 1:K] : Matrix{Float64}[]
    rc = two ? [zeros(m3) for _ in 1:K] : Vector{Float64}[]
    if two
        Gbar = Ay * h.ghxx
        Gs = h.ghxx[sv, :]
        Bm = Ay * ghx
        D = zeros(m3, nn)
        Dn, Gn = zeros(m3, nn), zeros(m3, nn)
        zc = zeros(ns)
        S = zeros(ns, ns)
    end
    Vy = Ay * ghx
    for k in 1:K-1
        rY[k+1][:, sv] = Vy
        rO[k+1] = rO[k] * P
        if two
            dc = 0.5 .* (h.ghxx * vec(S)) .+ h.c2
            rW[k+1] = 0.5 .* (Gbar .+ D)
            rc[k+1] = Ay * (ghx * zc .+ dc)
            _hor_quad_sandwich!(Dn, D, F)
            Dn .+= Bm * Gs
            _hor_quad_sandwich!(Gn, Gbar, F)
            D, Dn = Dn, D
            Gbar, Gn = Gn, Gbar
            Bm = Bm * F
            zc = F * zc .+ dc[sv]
            S = F * S * F' .+ h.SigS
        end
        Vy = Vy * F
    end
    function comb(r::Vector{<:AbstractMatrix})
        out = zeros(m * H, size(r[1], 2))
        for c in 1:m, t in 1:H
            row = (c - 1) * H + t
            out[row, :] .= view(r[t], 3(c-1)+1, :) .+ view(r[t+1], 3(c-1)+2, :)
            t >= 2 && (out[row, :] .+= view(r[t-1], 3(c-1)+3, :))
        end
        out
    end
    cols = Matrix{Float64}[comb(rY)]
    c0 = zeros(m * H)
    if two
        push!(cols, comb(rW))
        for c in 1:m, t in 1:H
            c0[(c-1)*H+t] = rc[t][3(c-1)+1] + rc[t+1][3(c-1)+2] + (t >= 2 ? rc[t-1][3(c-1)+3] : 0.0)
        end
    end
    push!(cols, comb(rO))
    Matrix{Float64}(reduce(hcat, cols)), c0
end

# ------------------------------------------------------------------------------------------------ inputs of the search
"""
    hor_inputs_from_state(h, y1, y2, o_pre, prev_total, shock) -> (a, Ycv, A)

From the precomputed maps: `a` (`nb T`, block-major) the alternative residuals of the switched equations on the no-wedge path, `Ycv` (`k x N`) the condition variables' path,
`A` (`N x nb`) the residual paths over all `N` periods.
"""
function hor_inputs_from_state(h::ObcHORegime, y1::AbstractVector, y2::AbstractVector, o_pre::AbstractVector, prev::AbstractVector, shock::AbstractVector)
    n, ns, N, T, nb, nk = h.n, h.ns, h.N, h.T, h.nb, h.nk
    z = h.zvec
    @inbounds for i in 1:n
        z[i] = y1[i] + y2[i]
    end
    if h.order == 2
        @inbounds for a in 1:ns, b in 1:ns
            z[n+(a-1)*ns+b] = y1[h.sv[a]] * y1[h.sv[b]]
        end
        z[n+ns^2+1:end] .= o_pre
    else
        z[n+1:end] .= o_pre
    end
    v = h.vbuf
    copyto!(v, h.c0all)
    mul!(v, h.Gall, z, 1.0, 1.0)
    A = Matrix{Float64}(undef, N, nb)
    @inbounds for b in 1:nb, t in 1:N
        A[t, b] = v[(b-1)*N+t] + h.rg.alt_rows[b, end]
    end
    for b in 1:nb
        A[1, b] += dot(view(h.rg.alt_rows, b, 1:n), prev) + dot(view(h.rg.alt_rows, b, 3n+1:3n+h.nx), shock)
    end
    Ycv = Matrix{Float64}(undef, nk, N)
    @inbounds for j in 1:nk, t in 1:N
        Ycv[j, t] = v[(nb+j-1)*N+t]
    end
    h.last = (copy(v), copy(prev), copy(shock))
    vec(A[1:T, :]), Ycv, A
end

"The gap-like path (`nc N`, the row order of `ObcHO`'s `q`) of the last `hor_inputs_from_state`."
function hor_gap_path(h::ObcHORegime)
    v, prev, shock = h.last
    N, nb, nk, n = h.N, h.nb, h.nk, h.n
    q = zeros(h.nc * N)
    for c in 1:h.nc, t in 1:N
        val = v[(nb+nk+c-1)*N+t] + h.gapfun[c, end]
        t == 1 && (val += dot(view(h.gapfun, c, 1:n), prev) + dot(view(h.gapfun, c, 3n+1:3n+h.nx), shock))
        q[h.rows[t, c]] = val
    end
    q
end

"The expected total deviation (`N + 1` rows, declaration order) of the unconstrained model given `y1`, `y2` and `o_pre`, by the recursion itself (the reference of the affine maps)."
function hor_expected_path(h::ObcHORegime, y1::AbstractVector, y2::AbstractVector, o_pre::AbstractVector)
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

"The same inputs as `hor_inputs_from_state` from the expected path itself (slow reference implementation for the tests)."
function hor_inputs_direct(h::ObcHORegime, y1, y2, o_pre, prev, shock)
    N, T, nb, n = h.N, h.T, h.nb, h.n
    E = hor_expected_path(h, y1, y2, o_pre)
    A = zeros(N, nb)
    for b in 1:nb, t in 1:N
        A[t, b] = h.rg.alt_rows[b, end] + dot(view(E, t, :), view(h.fB, b, :)) + dot(view(E, t + 1, :), view(h.fA, b, :))
        t > 1 && (A[t, b] += dot(view(E, t - 1, :), view(h.fC, b, :)))
    end
    for b in 1:nb
        A[1, b] += dot(view(h.rg.alt_rows, b, 1:n), prev) + dot(view(h.rg.alt_rows, b, 3n+1:3n+h.nx), shock)
    end
    vec(A[1:T, :]), Matrix{Float64}(E[1:N, h.rg.cv]'), A
end

# ------------------------------------------------------------------------------------------------ one search (the complete one)
"""
    _hor_search(h, A, Ycv, time_limit) -> (status, z, w, info)

`RegimeOBC`'s search (guess and verify, the tied LCP, the integer programme) on the inputs `A` (`N x nb`) and `Ycv` (`k x N`); `time_limit` for the integer programmes (0: none).
"""
function _hor_search(h::ObcHORegime, A::AbstractMatrix, Ycv::AbstractMatrix, time_limit::Real)
    rg, T, N, n = h.rg, h.T, h.N, h.n
    a = vec(A[1:T, :])
    lcp_fn = function ()
        tie = _tie(rg)
        (tie === nothing || !tie.mirror) && return nothing
        M, q, G = _lcp_from(rg, a, A[:, tie.primary])
        got = _solve_lcp_core(rg, M, q, G, a; milp=time_limit > 0, time_limit=time_limit)
        got === nothing && return nothing
        z, w = got
        z, w, Ycv .+ reshape(rg.Wcv * w, length(rg.cv), N)
    end
    milp_fn = function ()
        Y0 = [zeros(n) for _ in 1:N]
        for t in 1:N, (c, i) in enumerate(rg.cv)
            Y0[t][i] = Ycv[c, t]
        end
        _regime_milp_core(rg, Y0, a; time_limit=time_limit)
    end
    _regime_search(rg, Ycv, a; lcp_fn=lcp_fn, milp_fn=milp_fn, milp=time_limit > 0)
end

# class of a point the complete search could not solve: (a) only when an infeasible linear programme certifies it, (b) horizon too short, (c) the rest
_hor_class(status::Symbol, proof::Bool) = status == :horizon_too_short ? :b : (status == :no_solution && proof ? :a : :c)

# ------------------------------------------------------------------------------------------------ cubature
function _hor_init_cubature!(h::ObcHORegime)
    h.R, h.Lg, h.Lo = _hor_conditional_root(h)
    h.xi = size(h.R, 2) == 0 ? zeros(0, 0) : qmc_normal_points(size(h.R, 2), h.qmc_level)
    h.Ro_src = zeros(0, 0)
    h
end

"""
    _hor_conditional_root(h) -> (R, Lg, Lo)

`R` with `R R'` the covariance of the first-order effect, on the gap-like path of each constraint, of the shocks of the next `periods_of_uncertainty` periods (cosine taper,
eigenvalue pruning: `ho_conditional_root` on the same rows in the same order), `Lg` the effects before pruning and `Lo` those of the same shocks on every other acceptance
quantity (alternative residuals over `N` periods, rows `(b - 1) N + t`, then condition-variable paths, rows `nb N + (t - 1) k + j`).
"""
function _hor_conditional_root(h::ObcHORegime)
    ob, rg, n, nx, N, nc, nb, nk = h.ob, h.rg, h.n, h.nx, h.N, h.nc, h.nb, h.nk
    P, Q = ob.P, ob.Q
    g = h.gapfun
    a = rg.alt_rows
    Rg = _functional_rows(P, g[:, 1:n], g[:, n+1:2n], g[:, 2n+1:3n], N)
    Ra = _functional_rows(P, a[:, 1:n], a[:, n+1:2n], a[:, 2n+1:3n], N)
    Z = zeros(nk, n)
    Ecv = zeros(nk, n)
    for (c, i) in enumerate(rg.cv)
        Ecv[c, i] = 1.0
    end
    Rc = _functional_rows(P, Z, Ecv, Z, N)
    Gq = [vcat([(Rg[t][c, :]' * Q) .+ (t == 1 ? g[c, 3n+1:3n+nx]' : 0.0) for t in 1:N]...) for c in 1:nc]     # N x nx per constraint
    Sa = [vcat([(Ra[t][b, :]' * Q) .+ (t == 1 ? a[b, 3n+1:3n+nx]' : 0.0) for t in 1:N]...) for b in 1:nb]
    Sc = vcat([Rc[t] * Q for t in 1:N]...)                                  # row (t - 1) k + j
    w, V = eigen(Symmetric(h.Sigma))
    keep = w .> 1e-14 * max(maximum(w), 1e-300)
    L = V[:, keep] .* sqrt.(w[keep])'
    bg, bo = Matrix{Float64}[], Matrix{Float64}[]
    for i in 1:h.pou
        wi = h.taper ? 0.5 * (1 + cos(pi * (i - 1) / h.pou)) : 1.0
        B = zeros(nc * N, size(L, 2))
        Ba = zeros(nb * N, size(L, 2))
        Bc = zeros(N * nk, size(L, 2))
        if i < N
            for c in 1:nc
                B[h.rows[i+1:N, c], :] = sqrt(wi) .* (Gq[c][1:N-i, :] * L)
            end
            for b in 1:nb
                Ba[(b-1)*N+i+1:b*N, :] = sqrt(wi) .* (Sa[b][1:N-i, :] * L)
            end
            Bc[i*nk+1:end, :] = sqrt(wi) .* (Sc[1:(N-i)*nk, :] * L)
        end
        push!(bg, B); push!(bo, vcat(Ba, Bc))
    end
    Lg = reduce(hcat, bg)
    Lo = reduce(hcat, bo)
    R = Lg
    if size(R, 2) > 0 && h.pruning_cutoff !== nothing
        ev, U = eigen(Symmetric(R * R'))
        ev = max.(ev, 0.0)
        ev[ev .< h.pruning_cutoff * maximum(ev)] .= 0.0
        ev[1:max(0, length(ev) - h.max_dim)] .= 0.0
        kp = sqrt.(ev) .> sqrt(eps())
        R = U[:, kp] .* sqrt.(ev[kp])'
    end
    R, Lg, Lo
end

"Perturbation roots of the other acceptance quantities along the retained directions of `R`: `Ro = Cov(other, gap) pinv(R')`. Recomputed when `R` is replaced."
function _hor_other_roots(h::ObcHORegime)
    h.Ro_src === h.R && return h.Ro
    h.Ro = size(h.R, 2) > 0 ? h.Lo * (h.Lg' * pinv(h.R')) : zeros(size(h.Lo, 1), 0)
    h.Ro_src = h.R
    h.Ro
end

# OccBin's flags for one path into the buffers `vb`, `vr` (the compiled conditions live in a newer world)
function _hor_flags!(vb::BitMatrix, vr::BitMatrix, rg::RegimeOBC, z::AbstractMatrix{Bool}, pcv::AbstractMatrix)
    fill!(vb, false); fill!(vr, false)
    Base.invokelatest(_regime_flags!, vb, vr, rg, z, pcv)
    nothing
end

"""
The relevance test of every point: the reference-regime path (no wedges anywhere) fails OccBin's bind test somewhere. Returns the indices of the relevant points and, for them,
the bind violations (`N x nc`), which are the next guess of the guess-and-verify (iteration 0 of the search is this test).
"""
function _hor_relevant(h::ObcHORegime, pY::AbstractMatrix)
    rg, N, nk = h.rg, h.N, h.nk
    rel = Int[]
    vbs = BitMatrix[]
    z0 = falses(N, h.nc)
    vb, vr = falses(N, h.nc), falses(N, h.nc)
    for p in 1:size(pY, 2)
        _hor_flags!(vb, vr, rg, z0, reshape(view(pY, :, p), nk, N))
        if any(vb)
            push!(rel, p)
            push!(vbs, copy(vb))
        end
    end
    rel, vbs
end

"""
    _hor_gv_many(h, pa, pY, vb0; maxit=60) -> (W, ok)

OccBin's guess-and-verify (`_regime_iterate`) on many perturbed points at once from the reference regime: `pa` (`nws x P`), `pY` (`k N x P`), `vb0` the bind violations at
the reference regime, which are the guess of the second iteration (the first is the relevance test). Points are grouped by their guess so that each distinct regime sequence
needs one LU with many right-hand sides. `ok[p]` false: the iteration could not settle the point (it cycles, wants a regime change beyond the horizon, or has a singular
block): the caller runs the complete search on it.
"""
function _hor_gv_many(h::ObcHORegime, pa::AbstractMatrix, pY::AbstractMatrix, vb0::Vector{BitMatrix}; maxit::Int=60)
    rg, T, N, nc, nk, nws = h.rg, h.T, h.N, h.nc, h.nk, h.nws
    P = size(pa, 2)
    W = zeros(nws, P)
    Z = [copy(v) for v in vb0]
    zero_key = falses(N, nc)
    seen = [Set{BitMatrix}([zero_key]) for _ in 1:P]
    state = zeros(Int8, P)                                  # 0 active, 1 solved, 2 not settled
    vb, vr = falses(N, nc), falses(N, nc)
    pcv = Matrix{Float64}(undef, nk, N)
    for it in 1:maxit-1
        act = findall(==(0), state)
        isempty(act) && break
        for i in act
            any(view(Z[i], T+1:N, :)) && (state[i] = 2)
        end
        act = findall(==(0), state)
        isempty(act) && break
        groups = Dict{BitMatrix,Vector{Int}}()
        for i in act
            push!(get!(() -> Int[], groups, Z[i][1:T, :]), i)
        end
        for (_, idx) in groups
            S = _regime_index(rg, Z[idx[1]])
            X = nothing
            Vc = nothing
            if !isempty(S)
                F = lu(rg.R[S, S]; check=false)
                if !issuccess(F)
                    state[idx] .= 2
                    continue
                end
                X = F \ (-pa[S, idx])
                if !all(isfinite, X)
                    state[idx] .= 2
                    continue
                end
                Vc = rg.Wcv[:, S] * X                       # (k N) x |idx|
            end
            for (c, i) in enumerate(idx)
                @inbounds for r in 1:nk*N
                    pcv[r] = pY[r, i] + (Vc === nothing ? 0.0 : Vc[r, c])
                end
                _hor_flags!(vb, vr, rg, Z[i], pcv)
                if !(any(vb) || any(vr))
                    if X !== nothing
                        for (s, r) in enumerate(S)
                            W[r, i] = X[s, c]
                        end
                    end
                    state[i] = 1
                else
                    key = Z[i]
                    if key in seen[i]
                        state[i] = 2
                    else
                        push!(seen[i], copy(key))
                        Z[i] = (key .& .!vr) .| vb
                    end
                end
            end
        end
    end
    state[state .== 0] .= 2
    W, state .== 1
end

"The wedges of one period with cubature (`hor_news_wedges`) see `_hor_cubature`; this is the result record of a period: `(w, info)`."
function _hor_period(h::ObcHORegime, a, Ycv, A, period::Int)
    if h.cubature && size(h.R, 2) > 0
        return _hor_cubature(h, A, Ycv, period), (used=:cubature,)
    end
    h.stats[:regime] += 1
    st, z, w, info = _hor_search(h, A, Ycv, h.root_milp_time_limit)
    st == :ok || throw(ObcError(st, period, "obc_method = news (order $(h.order), regime formulation): $st. $(info.message)"))
    w, merge(info, (z=z,))
end

"""
    _hor_cubature(h, A, Ycv, period) -> Vector

The wedges of one period: the average over the perturbed acceptance paths `(A, Ycv) + Ro xi`, a point being relevant when its reference-regime path fails OccBin's bind test;
non-relevant points contribute zero. With `P` points and `F` failed points: `(sum over the solved relevant units of size * w) / (P - F)`; an `ObcError` with a `CubatureFailure`
payload when the failures exceed the rule (`cubature_limits`), as soon as that is decided.
"""
function _hor_cubature(h::ObcHORegime, A::AbstractMatrix, Ycv::AbstractMatrix, period::Int)
    h.dropped_last = 0; h.relevant_last = 0; h.dropped_cls = (0, 0, 0)
    rg, nb, N, T, nk, nws = h.rg, h.nb, h.N, h.T, h.nk, h.nws
    npts = size(h.xi, 2)
    Ro = _hor_other_roots(h)
    RA, RY = view(Ro, 1:nb*N, :), view(Ro, nb*N+1:size(Ro, 1), :)
    A0, Y0 = vec(A), vec(Ycv)
    pY = RY * h.xi
    pY .+= Y0
    rel, vbs = _hor_relevant(h, pY)
    nrel = length(rel)
    h.stats[:cubature_points] += npts
    h.stats[:relevant_points] += nrel
    h.relevant_last = nrel
    nrel == 0 && return zeros(nws)
    counts = (a=0, b=0, c=0)
    nfail = 0
    a_lim, limit = cubature_limits(nrel, h.tolerance, h.undecided_tolerance, h.undecided_min_points)
    acc = zeros(nws)
    unexamined = 0
    decided = false
    tl = h.milp_time_limit
    function full_search(Ap::AbstractMatrix, Yp::AbstractMatrix)
        st, z, w, info = _hor_search(h, Ap, Yp, tl)
        if st == :ok
            h.stats[:full_search_solved] = get(h.stats, :full_search_solved, 0) + 1
            return w, nothing
        end
        nothing, _hor_class(st, get(info, :proof, false) === true)
    end
    function failed(cls::Symbol, size_::Int)
        counts = merge(counts, NamedTuple{(cls,)}((getfield(counts, cls) + size_,)))
        nfail += size_
        counts.a > a_lim || nfail > limit
    end
    n_as = nothing
    if h.regions == 0
        xr = h.xi[:, rel]
        pA = RA * xr
        pA .+= A0
        pa = Matrix{Float64}(undef, nws, nrel)
        for j in 1:nrel, b in 1:nb
            copyto!(view(pa, (b-1)*T+1:b*T, j), view(pA, (b-1)*N+1:(b-1)*N+T, j))
        end
        W, ok = _hor_gv_many(h, pa, pY[:, rel], vbs)
        for j in 1:nrel
            ok[j] && (acc .+= view(W, :, j))
        end
        bad = findall(!, ok)
        n_as = length(bad)
        h.stats[:batch_fail] = get(h.stats, :batch_fail, 0) + length(bad)
        for (i, j) in enumerate(bad)
            w, cls = full_search(reshape(pA[:, j], N, nb), reshape(pY[:, rel[j]], nk, N))
            if cls === nothing
                acc .+= w
            elseif failed(cls, 1)
                unexamined = length(bad) - i
                decided = true
                break
            end
        end
    else
        xr = h.xi[:, rel]
        groups = if h.regions == 1
            [collect(1:nrel)]
        elseif nrel <= h.regions
            [[j] for j in 1:nrel]
        else
            _split_regions(hor_gap_path(h) .+ h.R * xr, h.regions, h.pruning_cutoff, h.max_dim)
        end
        for (i, g) in enumerate(groups)
            isempty(g) && continue
            xm = vec(sum(view(xr, :, g); dims=2) ./ length(g))
            Ap = reshape(A0 .+ RA * xm, N, nb)
            Yp = reshape(Y0 .+ RY * xm, nk, N)
            isempty(_hor_relevant(h, reshape(vec(Yp), :, 1))[1]) && continue          # the region's mean point needs no wedge
            w, cls = full_search(Ap, Yp)
            if cls === nothing
                acc .+= length(g) .* w
            elseif failed(cls, length(g))
                unexamined = length(groups) - i
                decided = true
                break
            end
        end
    end
    if decided || counts.a > a_lim || nfail > limit
        h.stats[:no_solution] = get(h.stats, :no_solution, 0) + 1
        for c in (:a, :b, :c)
            h.stats[Symbol("failed_", c)] = get(h.stats, Symbol("failed_", c), 0) + getfield(counts, c)
        end
        throw(_cub_failure(period, counts, nfail, unexamined, nrel, npts, n_as, h.tolerance, tl, h.undecided_tolerance))
    end
    if nfail > 0
        h.dropped_last = nfail
        h.dropped_cls = (counts.a, counts.b, counts.c)
        for c in (:a, :b, :c)
            h.stats[Symbol("failed_", c)] = get(h.stats, Symbol("failed_", c), 0) + getfield(counts, c)
        end
        h.stats[:dropped_points] = get(h.stats, :dropped_points, 0) + nfail
        h.stats[:dropped_periods] = get(h.stats, :dropped_periods, 0) + 1
    end
    acc ./ (npts - nfail)
end

# ------------------------------------------------------------------------------------------------ simulation
"""
    obc_hor_simulate(h::ObcHORegime, shocks; init=nothing) -> NamedTuple

Simulate under the realised `shocks` (`S x n_exo`; every period is solved, as `obc_ho_simulate`). Returns levels (`S x n`, declaration order) `with_bound`, `without_bound`,
the wedges of all switched equations (`S x nb T`, block-major), per period `relevant`, `dropped` and `dropped_a/b/c`, the regime of the root search (`regimes`: `T x nc`
indicator or `nothing` with cubature) and `stats`; one warning states the totals of the dropped points.
"""
function obc_hor_simulate(h::ObcHORegime, shocks::AbstractMatrix; init=nothing)
    n, nx, ns, T, sv, ys, rg = h.n, h.nx, h.ns, h.T, h.sv, h.ys, h.rg
    foreach(k -> h.stats[k] = 0, keys(h.stats))
    S = size(shocks, 1)
    size(shocks, 2) == nx || throw(ArgumentError("shocks need $nx columns (got $(size(shocks, 2)))"))
    y1, y2, o = zeros(n), zeros(n), zeros(n)
    init === nothing || (y1 .= init)
    prev = y1 .+ y2 .+ o
    with_b, without_b, wedges = Matrix{Float64}(undef, S, n), Matrix{Float64}(undef, S, n), zeros(S, h.nws)
    dropped, relevant = zeros(Int, S), zeros(Int, S)
    dcls = zeros(Int, S, 3)
    regimes = Vector{Any}(undef, S)
    x1, x2 = zeros(ns), zeros(ns)
    for t in 1:S
        u = Vector{Float64}(view(shocks, t, :))
        @inbounds for i in 1:ns
            x1[i] = y1[sv[i]]; x2[i] = y2[sv[i]]
        end
        if h.order == 2
            kxx = vec((x1 * x1')')                               # kron(x1, x1), first index major
            kuu = vec((u * u')')
            kxu = vec((x1 * u')')
            y2 = h.ghx * x2 .+ 0.5 .* (h.ghxx * kxx) .+ 0.5 .* (h.ghuu * kuu) .+ h.ghxu * kxu .+ h.cst2
        end
        y1 = h.ghx * x1 .+ h.ghu * u
        opre = h.P * o
        a, Ycv, A = hor_inputs_from_state(h, y1, y2, opre, prev, u)
        w, info = _hor_period(h, a, Ycv, A, t)
        h.trace === nothing || push!(h.trace, (y1=copy(y1), y2=copy(y2), o_pre=copy(opre), prev=copy(prev), u=copy(u), w=copy(w), z=get(info, :z, nothing)))
        o = opre .+ rg.V0 * w
        base = y1 .+ y2
        without_b[t, :] = ys .+ base
        with_b[t, :] = ys .+ base .+ o
        prev = base .+ o
        wedges[t, :] = w
        regimes[t] = haskey(info, :z) ? Matrix{Bool}(info.z[1:T, :]) : nothing
        dropped[t], relevant[t] = h.dropped_last, h.relevant_last
        dcls[t, :] .= h.dropped_cls
    end
    if any(>(0), dropped)
        share = maximum(dropped[t] / max(1, relevant[t]) for t in 1:S)
        @warn cubature_warning(h.undecided_tolerance, h.undecided_min_points, (sum(dcls[:, 1]), sum(dcls[:, 2]), sum(dcls[:, 3])),
                               sum(relevant), share, count(>(0), dropped), S)
    end
    (with_bound=with_b, without_bound=without_b, wedges=wedges, dropped=dropped, relevant=relevant, dropped_a=dcls[:, 1], dropped_b=dcls[:, 2], dropped_c=dcls[:, 3],
     regimes=regimes, stats=copy(h.stats))
end
