# Simulation and impulse responses from perturbation decision rules (port of pydyn/simulate.py, after
# Dynare's simult_.m and irf.m). Decision rules are Dicts with Dynare's oo_.dr fields: "ghx", "ghu",
# "ghxx", ... (rows in decision-rule order), "ys", "order_var", "state_var" (1-based), "order".

_as_dr(dr::FirstOrderSolution) = as_dr(dr)
_as_dr(dr::AbstractDict) = dr

"""
    simulate(dr, shocks; order=dr order, pruning=false, y0=steady state) -> (T, n) levels

Simulate for a (T, n_exo) shock array, variables in declaration order. At order 2 with `pruning` this is
the Kim et al. (2008) scheme; at order 3 the Andreasen et al. (2018) scheme.
"""
function simulate(dr::AbstractDict, shocks::AbstractMatrix; order=nothing, pruning=nothing, y0=nothing)
    order = order === nothing ? get(dr, "order", 1) : order
    pruning = pruning === nothing ? get(dr, "pruning", false) == true : pruning
    order > get(dr, "order", 1) && throw(ArgumentError("decision rule only available up to order $(get(dr, "order", 1))"))
    ys, ov, sv = dr["ys"], dr["order_var"], dr["state_var"]
    T, n = size(shocks, 1), length(ys)
    y0 = y0 === nothing ? copy(ys) : Float64.(y0)
    out = zeros(T, n)
    ghx_, ghu_ = dr["ghx"], dr["ghu"]
    if order == 1
        x = y0[sv] - ys[sv]
        for t in 1:T
            yh = ghx_ * x + ghu_ * shocks[t, :]
            out[t, ov] .= ys[ov] .+ yh
            x = out[t, sv] - ys[sv]
        end
        return out
    end
    ghxx, ghxu, ghuu, ghs2 = dr["ghxx"], dr["ghxu"], dr["ghuu"], vec(dr["ghs2"])
    if order == 2
        cst = ys[ov] .+ 0.5 .* ghs2
        x = y0[sv] - ys[sv]
        x1 = copy(x)
        for t in 1:T
            u = shocks[t, :]
            xq = pruning ? x1 : x
            quad = 0.5 * ghxx * kron(xq, xq) + 0.5 * ghuu * kron(u, u) + ghxu * kron(xq, u)
            out[t, ov] .= cst + ghx_ * x + ghu_ * u + quad
            if pruning
                y1 = zeros(n)
                y1[ov] .= ghx_ * x1 + ghu_ * u
                x1 = y1[sv]
            end
            x = out[t, sv] - ys[sv]
        end
        return out
    end
    order == 3 || throw(ArgumentError("simulation above third order"))
    ghxxx, ghxxu, ghxuu, ghuuu = dr["ghxxx"], dr["ghxxu"], dr["ghxuu"], dr["ghuuu"]
    ghxss, ghuss = dr["ghxss"], dr["ghuss"]
    pos = Dict(v => i for (i, v) in enumerate(ov))
    ipred = [pos[v] for v in sv]
    if pruning
        x1 = y0[sv] - ys[sv]
        x2 = zeros(length(x1)); x3 = zeros(length(x1))
        for t in 1:T
            u = shocks[t, :]
            yu = kron(x1, u)
            y3 = ghx_ * x3 + ghxx * kron(x1, x2) + ghxu * kron(x2, u) +
                 (ghxxx * kron(kron(x1, x1), x1) + ghuuu * kron(kron(u, u), u) +
                  3 * (ghxxu * kron(x1, yu) + ghxuu * kron(yu, u) + ghxss * x1 + ghuss * u)) / 6
            y2 = ghx_ * x2 + 0.5 * (ghxx * kron(x1, x1) + ghuu * kron(u, u) + 2 * ghxu * yu + ghs2)
            y1 = ghx_ * x1 + ghu_ * u
            out[t, ov] .= ys[ov] + y1 + y2 + y3
            x1, x2, x3 = y1[ipred], y2[ipred], y3[ipred]
        end
        return out
    end
    x = y0[sv] - ys[sv]
    for t in 1:T
        u = shocks[t, :]
        xu = kron(x, u)
        yh = ghx_ * x + ghu_ * u + 0.5 * (ghxx * kron(x, x) + 2 * ghxu * xu + ghuu * kron(u, u) + ghs2) +
             (ghxxx * kron(kron(x, x), x) + ghuuu * kron(kron(u, u), u) +
              3 * (ghxxu * kron(x, xu) + ghxuu * kron(xu, u) + ghxss * x + ghuss * u)) / 6
        out[t, ov] .= ys[ov] + yh
        x = yh[ipred]
    end
    out
end

"""
    irf(dr, impulse; Sigma_e=nothing, periods=40, order=dr order, pruning=nothing, drop=100, replic=50, draws=nothing)

Impulse responses, size (periods, n_endo), deviations as Dynare reports them. At order 1 the exact
linear response; above, Dynare's simulation-based generalised IRF (`draws`, of size
(periods + drop, n shocks with non-zero variance, replic), makes it reproducible).
"""
function irf(dr::AbstractDict, impulse::AbstractVector; Sigma_e=nothing, periods=40, order=nothing, pruning=nothing,
             drop=100, replic=50, draws=nothing, rng=nothing)
    order = order === nothing ? get(dr, "order", 1) : order
    impulse = Float64.(impulse)
    n_exo = length(impulse)
    if order == 1
        ex = zeros(periods, n_exo)
        ex[1, :] .= impulse
        return simulate(dr, ex; order=1) .- dr["ys"]'
    end
    Sigma_e === nothing && throw(ArgumentError("Sigma_e is needed for impulse responses above first order"))
    active = findall(!=(0), diag(Sigma_e))
    chol_u = Matrix(cholesky(Symmetric(Sigma_e[active, active])).U)
    if draws === nothing
        rng = rng === nothing ? Random.default_rng() : rng
        draws = randn(rng, periods + drop, length(active), replic)
    else
        replic = size(draws, 3)
        size(draws, 1) == periods + drop || throw(ArgumentError("draws must have periods + drop rows"))
    end
    acc = zeros(periods, length(dr["ys"]))
    for j in 1:replic
        ex1 = zeros(periods + drop, n_exo)
        ex1[:, active] .= draws[:, :, j] * chol_u
        ex2 = copy(ex1)
        ex2[drop+1, :] .+= impulse
        y1 = simulate(dr, ex1; order=order, pruning=pruning)
        y2 = simulate(dr, ex2; order=order, pruning=pruning)
        acc .+= y2[drop+1:end, :] .- y1[drop+1:end, :]
    end
    acc ./ replic
end
irf(s::FirstOrderSolution, shock::AbstractVector, periods::Int) = irf(s, shock; periods=periods)
