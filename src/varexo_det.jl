# Deterministic exogenous variables (`varexo_det`) in stochastic models: Dynare's stochastic_solvers.m ("exogenous deterministic
# variables" section: ghud, ghxud, ghuud, ghudud) and simult_varexo_det.m; port of pydyn/varexo_det.py. Everything else in
# stochastic models (moments, IRFs, simulations) ignores the varexo_det variables, which stay at their steady-state values.
# Orders 1 and 2 only (Dynare's k-order solver, used at order 3, computes no such terms); no pruning.

"`dr.exo_det_length`: the last period with a deterministic shock on a varexo_det variable in the `shocks` blocks."
function exo_det_length(m::Model)
    L = 0
    for (_, p1, p2, _) in get(m.cache, :det_shocks_exo_det, Tuple{Int,Any,Any,Float64}[])
        L = max(L, Int(_numval(p2)))
    end
    L
end

function _dense_derivs(m::Model, order::Int, ys::Vector{Float64})
    n, nx, nxd = m.n_endo, m.n_exo, n_exo_det(m)
    nk = 3n + nx + nxd
    r, c, v = dynamic_full_derivs(m, order, repeat(ys, 3), exo_steady(m), m.params, ys)
    if order == 1
        g = zeros(n, nk)
        for k in eachindex(r)
            g[r[k], c[k, 1]] += v[k]
        end
        return g
    end
    H = zeros(n, nk, nk)
    for k in eachindex(r)
        H[r[k], c[k, 1], c[k, 2]] = v[k]
        H[r[k], c[k, 2], c[k, 1]] = v[k]
    end
    H
end

"""
    exo_det_decision_rules(m, dr; length=nothing, params=nothing) -> Dict

`dr` (a decision-rule Dict from `as_dr(first_order(m))` or `k_order`) plus Dynare's `ghud` (a vector indexed from 1: `ghud[j]` is
Dynare's `ghud{j}`) and, at order 2, `ghxud`, `ghuud` and `ghudud` (a Dict keyed `(j, k)`, `j <= k`, as Dynare's `ghudud{j,k}`),
for `j = 1 .. exo_det_length`. Order 3 is refused.
"""
function exo_det_decision_rules(m::Model, dr; length=nothing)
    n_exo_det(m) == 0 && throw(ModelError("the model has no varexo_det variables"))
    dr = Dict{String,Any}(_dr_dict(dr))
    order = Int(get(dr, "order", 1))
    order > 2 && throw(ModelError("varexo_det: Dynare computes no deterministic-exogenous terms at order 3 (k_order_solver)"))
    lli = m.lead_lag_incidence
    any(lli[:, 1] .> 0) || throw(ModelError("var_exo_det not implemented for purely forward models"))
    any(lli[:, 3] .> 0) || throw(ModelError("var_exo_det not implemented for purely backwards models"))
    n, nx, nxd = m.n_endo, m.n_exo, n_exo_det(m)
    _, _, cnt = dr_ordering(lli)
    ov = Int.(dr["order_var"])
    nstatic, npred, nboth, nfwrd = cnt.nstatic, cnt.npred, cnt.nboth, cnt.nfwrd
    nspred, nsfwrd = npred + nboth, nboth + nfwrd
    ys = Vector{Float64}(dr["ys"])
    ghx = Matrix{Float64}(dr["ghx"]); ghu = Matrix{Float64}(dr["ghu"])
    gx = ghx[nstatic+npred+1:end, :]
    Gy = ghx[nstatic+1:nstatic+nspred, :]
    g1 = _dense_derivs(m, 1, ys)
    icurr = findall(>(0), lli[ov, 2])
    f1 = g1[:, 2n .+ ov[nstatic+npred+1:nstatic+npred+nsfwrd]]
    f0 = g1[:, n .+ ov[icurr]]
    fudet = g1[:, 3n+nx+1:3n+nx+nxd]
    M1 = inv(f0 + hcat(zeros(n, nstatic), f1 * gx, zeros(n, nsfwrd - nboth)))
    M2 = M1 * f1
    L = length === nothing ? exo_det_length(m) : Int(length)
    ghud = Matrix{Float64}[]
    L > 0 && push!(ghud, -M1 * fudet)
    for i in 1:L-1
        push!(ghud, -M2 * ghud[i][n-nsfwrd+1:n, :])
    end
    dr["exo_det_length"] = L
    dr["ghud"] = ghud
    if order == 2 && L > 0
        H = _dense_derivs(m, 2, ys)
        kk1 = vcat(ov[nstatic+1:nstatic+nspred], n .+ ov[icurr], 2n .+ ov[nstatic+npred+1:nstatic+npred+nsfwrd], 3n .+ (1:nx+nxd))
        K = Base.length(kk1)
        G2 = zeros(n, K * K)
        for a in 1:K, b in 1:K
            G2[:, (a-1)*K+b] = H[:, kk1[a], kk1[b]]
        end
        kf = n-nfwrd-nboth+1:n
        kp = nstatic+1:nstatic+nspred
        ghxx = Matrix{Float64}(dr["ghxx"])
        hu = ghu[kp, :]
        hud = ghud[1][kp, :]
        Z(a, b) = zeros(a, b)
        I_(k) = Matrix{Float64}(I, k, k)
        zx = vcat(I_(nspred), ghx[icurr, :], gx * Gy, Z(nx + nxd, nspred))
        zu = vcat(Z(nspred, nx), ghu[icurr, :], gx * hu, Z(nx + nxd, nx))
        zud = vcat(Z(nspred, nxd), ghud[1], gx * hud, Z(nx, nxd), I_(nxd))
        Eud = I_(nxd)
        function zudi_(i, extra=nothing)          # i 0-based
            hudi = ghud[i+1][kp, :]
            mid = extra === nothing ? gx * hudi : gx * hudi + extra
            vcat(Z(nspred, nxd), ghud[i+1], mid, Z(nx + nxd, nxd))
        end
        ghxud = Matrix{Float64}[-M1 * (G2 * kron(zx, zud) + f1 * ghxx[kf, :] * kron(ghx[kp, :], ghud[1][kp, :]))]
        for i in 1:L-1
            R2 = G2 * kron(zx, zudi_(i))
            push!(ghxud, -M2 * (ghxud[i][kf, :] * kron(Gy, Eud) + ghxx[kf, :] * kron(ghx[kp, :], ghud[i+1][kp, :])) - M1 * R2)
        end
        ghuud = Matrix{Float64}[-M1 * (G2 * kron(zu, zud) + f1 * ghxx[kf, :] * kron(ghu[kp, :], ghud[1][kp, :]))]
        for i in 1:L-1
            R2 = G2 * kron(zu, zudi_(i))
            push!(ghuud, -M2 * ghxud[i][kf, :] * kron(hu, Eud) - M1 * R2)
        end
        ghudud = Dict{Tuple{Int,Int},Matrix{Float64}}((1, 1) => -M1 * (G2 * kron(zud, zud)) - M2 * ghxx[kf, :] * kron(hud, hud))
        for i in 1:L-1                                # (i, k) below are 0-based; the keys are 1-based
            hudi = ghud[i+1][kp, :]
            zudi = zudi_(i, ghud[i][kf, :])
            R2 = G2 * kron(zudi, zudi)
            ghudud[(i + 1, i + 1)] = -M2 * (ghudud[(i, i)][kf, :] + 2 * ghxud[i][kf, :] * kron(hudi, Eud) + ghxx[kf, :] * kron(hudi, hudi)) - M1 * R2
            R2 = G2 * kron(zud, zudi)
            ghudud[(1, i + 1)] = -M2 * (ghxud[i][kf, :] * kron(hud, Eud) + ghxx[kf, :] * kron(hud, hudi)) - M1 * R2
            for j in 1:i-1
                hudj = ghud[j+1][kp, :]
                R2 = G2 * kron(zudi_(j), zudi)
                ghudud[(j + 1, i + 1)] = -M2 * (ghudud[(j, i)][kf, :] + ghxud[j][kf, :] * kron(hudi, Eud) + ghxud[i][kf, :] * kron(hudj, Eud) +
                                                ghxx[kf, :] * kron(hudj, hudi)) - M1 * R2
            end
        end
        dr["ghxud"] = ghxud; dr["ghuud"] = ghuud; dr["ghudud"] = ghudud
    end
    dr
end

"""
    simult_varexo_det(m, dr, ex, ex_det; y0=nothing, order=nothing) -> (horizon + 1, n) levels

`simult_varexo_det.m` (the path): `ex` (horizon, n_exo) stochastic shocks, `ex_det` (horizon + 1, n_exo_det) levels of the varexo_det
variables, the first row being the initial period as in Dynare; `dr` from [`exo_det_decision_rules`](@ref). The first row of the result
is `y0` (default: the steady state).
"""
function simult_varexo_det(m::Model, dr, ex::AbstractMatrix, ex_det::AbstractMatrix; y0=nothing, order=nothing)
    order = order === nothing ? Int(get(dr, "order", 1)) : Int(order)
    order > 2 && throw(ModelError("simult_varexo_det.m: Forecasting with varexo_det does not support order>2."))
    get(dr, "pruning", false) == true && order == 2 && throw(ModelError("simult_varexo_det.m: Forecasting with varexo_det does not support pruning."))
    horizon = size(ex, 1)
    size(ex_det, 1) == horizon + 1 ||
        throw(ModelError("simult_varexo_det: Size mismatch: number of forecasting periods for stochastic and deterministic exogenous don't match"))
    _, _, cnt = dr_ordering(m.lead_lag_incidence)
    nstatic, nspred = cnt.nstatic, cnt.npred + cnt.nboth
    ov = Int.(dr["order_var"])
    ys = Vector{Float64}(dr["ys"])
    yss = ys[ov]
    L = Int(dr["exo_det_length"])
    xdss = exo_det_steady(m)
    ghx = Matrix{Float64}(dr["ghx"]); ghu = Matrix{Float64}(dr["ghu"])
    n = m.n_endo
    Y = zeros(horizon + 1, n)
    Y[1, :] = y0 === nothing ? ys : Vector{Float64}(y0)
    for i in 1:horizon                                 # Dynare's period i + maximum_lag
        tempx = (Y[i, ov] .- yss)[nstatic+1:nstatic+nspred]
        tempu = Vector{Float64}(ex[i, :])
        y = yss .+ ghx * tempx .+ ghu * tempu
        if order == 2
            y = y .+ vec(Float64.(dr["ghs2"])) ./ 2 .+ 0.5 .* (dr["ghxx"] * kron(tempx, tempx) .+ dr["ghuu"] * kron(tempu, tempu)) .+
                dr["ghxu"] * kron(tempx, tempu)
        end
        for j in 0:min(L + 1 - i, L)-1
            tempud = Vector{Float64}(ex_det[i+1+j, :]) .- xdss
            y = y .+ dr["ghud"][j+1] * tempud
            if order == 2
                y = y .+ dr["ghxud"][j+1] * kron(tempx, tempud) .+ dr["ghuud"][j+1] * kron(tempu, tempud) .+
                    0.5 .* (dr["ghudud"][(j + 1, j + 1)] * kron(tempud, tempud))
                for k in 0:j-1
                    y = y .+ dr["ghudud"][(k + 1, j + 1)] * kron(Vector{Float64}(ex_det[i+1+k, :]) .- xdss, tempud)
                end
            end
        end
        Y[i+1, ov] = y
    end
    Y
end

"""
`oo_.exo_det_simul` as `make_ex_.m` builds it for the forecast commands, with the rows `forecasts.run` adds when the forecast is longer:
`maximum_lag + periods + maximum_lead` rows of steady-state values with the `shocks` of the varexo_det variables written in (a shock
beyond the last row extends the matrix with zeros, as MATLAB does), then steady-state rows up to `maximum_lag + horizon`.
Returns `(exo_det_simul, exo_det_length)`.
"""
function exo_det_paths(m::Model, periods::Integer=0, horizon=nothing)
    ml, mlead = 1, 1
    ss = exo_det_steady(m)
    X = repeat(reshape(ss, 1, :), ml + Int(periods) + mlead, 1)
    for st in m.statements
        get(st, "statementName", "") == "mshocks" || continue
        any(d -> d["var"] in exo_det_names(m), get_or(st, "deterministic_shocks", [])) &&
            throw(ModelError("mshocks on varexo_det variables are not supported by the forecast"))
    end
    for (i, p1, p2, val) in get(m.cache, :det_shocks_exo_det, Tuple{Int,Any,Any,Float64}[])
        (p1 isa Real || tryparse(Float64, string(p1)) !== nothing) ||
            throw(ModelError("forecast with dated shocks on varexo_det variables is not supported"))
        k = Int(_numval(p1)):Int(_numval(p2))
        rows = collect(k) .+ ml
        any(==(0), rows) && throw(ModelError("make_ex_: The model does not have lags, so you cannot set values for period 0"))
        mx = maximum(rows)
        mx > size(X, 1) && (X = vcat(X, zeros(mx - size(X, 1), size(X, 2))))
        X[rows, i] .= val
    end
    len = size(X, 1) - ml
    if horizon !== nothing && horizon > len
        X = vcat(X, repeat(reshape(ss, 1, :), horizon - len, 1))
        len = horizon
    end
    X, len
end
