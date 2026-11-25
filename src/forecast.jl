# Forecasts and conditional forecasts: Dynare's `forecast` (after stoch_simul and after estimation) and
# `conditional_forecast` / `conditional_forecast_paths` / `plot_conditional_forecast` (port of Dynare 7.2's
# matlab/+forecasts/{run, stochastic_classical}.m and matlab/+conditional_forecasts/{run, get_shock_paths, plot}.m,
# GPL-3.0-or-later, Dynare Team).
#
#     m = JDyn.load("model.mod")
#     fc = forecast(m; periods=12, conf_sig=0.8)               # point forecast and intervals from the decision rule
#     fc.Mean["y"], fc.HPDinf["y"], fc.HPDsup["y"]              # (periods,) vectors, as oo_.forecast.Mean.y
#     res = estimate(m)                                          # estimation(forecast=8): res.forecast (at the mode)
#     forecast(res.smoother, m; periods=8)                       # the same from a smoother result
#     cf = conditional_forecast(m; paths=["gy_obs" => [1.01, 0.99, NaN]], controlled_varexo=["e_a"], replic=1000)
#     cf.cond_mean["k"], cf.cond_ci["k"]                        # oo_.conditional_forecast.cond.Mean.k / .ci.k
#     conditional_forecast_table(cf, ["k"])                      # plot_conditional_forecast's numbers as a table
#
# Random numbers: forecasts at order >= 2 (intervals from simulations) and conditional forecasts (shocks that are not
# controlled) draw standard normals with Julia's generator; pass `draws` (the arrays Dynare's `randn` calls would give,
# see the docstrings) to reproduce a Dynare run exactly. The point forecasts at order 1 and 2 and the conditional
# forecasts with every shock controlled are deterministic.
#
# Dynare 7.2 defects corrected by default (`replicate_dynare_bugs=true` reproduces Dynare's numbers):
# - forecast after estimation with `prefilter` or `observation_trends`: Dynare adds the data mean / trend to the
#   mean forecast of the observed variables but not to HPDinf/HPDsup (the interval is centred on the model's
#   steady state); JDyn shifts the interval with the mean.
# - same case, a `var_list` that does not hold the observed variables in declaration order: Dynare subtracts the steady
#   state of the variable at the observed variable's position in var_list (and fails when an observed variable is
#   missing from var_list); JDyn uses the observed variable's own steady state.

# ====================================================================== helpers
"Quantiles as MATLAB's `quantile` and Octave's statistics package (method 5, piecewise linear with p = (i - 0.5)/n)."
function quantile_r5(x::AbstractVector, p::Real)
    xs = sort(filter(!isnan, collect(Float64, x)))
    N = length(xs)
    N == 0 && return NaN
    N == 1 && return xs[1]
    p < 0.5 / N && return xs[1]
    p >= (N - 0.5) / N && return xs[N]
    h = N * p + 0.5
    f = floor(Int, h)
    xs[f] + (h - f) * (xs[f+1] - xs[f])
end

_norminv(p) = -sqrt(2.0) * SpecialFunctions.erfcinv(2.0 * p)
_iround(x) = Int(_mround(Float64(x)))                                  # MATLAB round (half away from zero)

"Decision rule (oo_.dr layout) from an MStruct `oo_.dr` (shapes as MATLAB's) or a Dict."
function _dr_dict(D)
    D isa AbstractDict && return D
    get_f(k) = D[k]
    out = Dict{String,Any}("ys" => vec(Float64.(get_f("ys"))), "order_var" => Int.(vec(get_f("order_var"))),
                           "state_var" => Int.(vec(get_f("state_var"))))
    for k in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss")
        haskey(D, k) && (out[k] = Matrix{Float64}(D[k]))
    end
    haskey(D, "ghs2") && (out["ghs2"] = vec(Float64.(D["ghs2"])))
    out["order"] = haskey(out, "ghxxx") ? 3 : haskey(out, "ghxx") ? 2 : 1
    out
end

# ====================================================================== results
"""
Result of `forecast`: `Mean`, `HPDinf`, `HPDsup` (and `HPDinf_ME`, `HPDsup_ME` for observed variables when there are
measurement errors) map variable names to (periods,) vectors, as Dynare's `oo_.forecast`.
"""
struct ForecastResult
    var_list::Vector{String}
    Mean::Dict{String,Vector{Float64}}
    HPDinf::Dict{String,Vector{Float64}}
    HPDsup::Dict{String,Vector{Float64}}
    HPDinf_ME::Dict{String,Vector{Float64}}
    HPDsup_ME::Dict{String,Vector{Float64}}
    periods::Int
    conf_sig::Float64
    order::Int
    Exogenous::Dict{String,Vector{Float64}}      # paths of the varexo_det variables over the periods (`oo_.forecast.Exogenous`)
end
ForecastResult(vl, M, L, U, LM, UM, periods, conf_sig, order) =
    ForecastResult(vl, M, L, U, LM, UM, periods, conf_sig, order, Dict{String,Vector{Float64}}())
Base.show(io::IO, r::ForecastResult) = print(io, "ForecastResult(", r.periods, " periods, ", length(r.var_list), " variables, conf_sig ", r.conf_sig, ")")

"`oo_.forecast` as Dynare stores it (fields in var_list order, column vectors)."
function forecast_struct(r::ForecastResult)
    S = MStruct()
    for (f, d) in (("Mean", r.Mean), ("HPDinf", r.HPDinf), ("HPDsup", r.HPDsup), ("HPDinf_ME", r.HPDinf_ME), ("HPDsup_ME", r.HPDsup_ME))
        isempty(d) && continue
        sub = MStruct()
        for v in r.var_list
            haskey(d, v) && (sub[v] = reshape(copy(d[v]), :, 1))
        end
        S[f] = sub
    end
    if !isempty(r.Exogenous)
        sub = MStruct()
        for (v, p) in sort(collect(r.Exogenous); by=first)
            sub[v] = reshape(copy(p), :, 1)
        end
        S["Exogenous"] = sub
    end
    S
end

# ====================================================================== stochastic_classical
"""
    _forecast_paths(dr, y0, horizon, ivar; order, pruning, Sigma_e, me, conf_sig, replic, draws, me_draws, rng, max_lag)

`forecasts.stochastic_classical`: the mean path `yf` (nvar x (max_lag + horizon), first column the initial condition)
and the bands. At order 1 the bands come from the recursion of the forecast-error variance; above, from `replic`
simulations (draws: (n shocks with non-zero variance, horizon, replic) standard normals, Dynare's `randn(nxs,horizon)`
per replication) and quantiles. `me` lists (position in ivar, measurement-error variance) for the `_ME` bands
(`me_draws`: (n observed with non-zero variance, horizon + 1, replic)).
"""
function _forecast_paths(dr::AbstractDict, y0::AbstractVector, horizon::Int, ivar::Vector{Int}; order::Int=1, pruning::Bool=false,
                         Sigma_e::AbstractMatrix, me=nothing, me_obs=nothing, H=nothing, conf_sig::Real=0.9, replic::Int=1000,
                         draws=nothing, me_draws=nothing, rng=nothing, max_lag::Int=1, xd=nothing)
    nx = size(Sigma_e, 1)
    n = length(dr["ys"])
    iter = horizon + max_lag - 1                 # columns simulated after the initial one (simult_)
    sim0(ex) = (y = simulate(dr, ex; order=order, pruning=pruning, y0=y0); hcat(reshape(Float64.(y0), :, 1), permutedims(y)))
    sim = xd === nothing ? sim0 : xd.sim          # varexo_det: simult_varexo_det.m over the paths (src/varexo_det.jl), possibly longer than the horizon
    if xd !== nothing
        iter = xd.iter
        horizon = iter - max_lag + 1
    end
    yfull = sim(zeros(max(iter, 0), nx))
    yf = yfull[ivar, :]
    fact = _norminv((1 - conf_sig) / 2)
    with_me = me !== nothing && !isempty(me)
    lo = hi = lo_me = hi_me = nothing
    if order == 1
        ov, sv = Int.(dr["order_var"]), Int.(dr["state_var"])
        inv_ov = invperm(ov)
        var_pos = unique(vcat(inv_ov[sv], inv_ov[ivar]))           # union(..., 'stable')
        state_pos = [findfirst(==(p), var_pos) for p in inv_ov[sv]]
        var_list_pos = [findfirst(==(p), var_pos) for p in inv_ov[ivar]]
        ghx1 = dr["ghx"][var_pos, :]
        ghu1 = dr["ghu"][var_pos, :]
        sigma = zeros(length(var_pos), length(var_pos))
        var_yf = zeros(length(ivar), size(yf, 2))
        Q = ghu1 * Sigma_e * ghu1'
        for h in 1:horizon
            sigma = ghx1 * sigma[state_pos, state_pos] * ghx1' + Q
            var_yf[:, 1+h] = diag(sigma[var_list_pos, var_list_pos])
        end
        w = -fact .* sqrt.(var_yf)
        lo, hi = yf .- w, yf .+ w
        if with_me
            v2 = copy(var_yf)
            for (j, h2) in me
                v2[j, :] .+= h2
            end
            w2 = -fact .* sqrt.(v2)
            lo_me, hi_me = yf .- w2, yf .+ w2
        end
    else
        active = findall(!=(0), diag(Sigma_e))
        nxs = length(active)
        cS = Matrix(cholesky(Symmetric(Matrix{Float64}(Sigma_e[active, active]))).U)
        if draws === nothing
            r = rng === nothing ? Random.default_rng() : rng
            draws = randn(r, nxs, iter, replic)
        else
            size(draws, 1) == nxs && size(draws, 2) >= iter || throw(ArgumentError("draws must be (shocks with non-zero variance = $nxs, $iter, replic)"))
            replic = size(draws, 3)
        end
        ysim = zeros(n, size(yf, 2), replic)
        ex = zeros(iter, nx)
        for b in 1:replic
            ex[:, active] = permutedims(draws[:, 1:iter, b]) * cS
            ysim[:, :, b] = sim(ex)
        end
        a = (1 - conf_sig) / 2
        q(Y, i, t, p) = quantile_r5(view(Y, i, t, :), p)
        lo = [q(ysim, ivar[i], t, a) for i in eachindex(ivar), t in axes(yf, 2)]
        hi = [q(ysim, ivar[i], t, 1 - a) for i in eachindex(ivar), t in axes(yf, 2)]
        if with_me
            # Dynare draws randn(n observed, horizon+1)' * chol(H) per replication and adds it to the observed variables
            iobs = findall(!=(0), diag(H))
            cH = Matrix(cholesky(Symmetric(Matrix{Float64}(H[iobs, iobs]))).U)
            if me_draws === nothing
                r = rng === nothing ? Random.default_rng() : rng
                me_draws = randn(r, length(iobs), size(yf, 2), replic)
            end
            for b in 1:replic
                e = permutedims(me_draws[:, 1:size(yf, 2), b]) * cH            # (T, n observed with ME)
                for (jj, j) in enumerate(iobs)
                    ysim[me_obs[j], :, b] .+= e[:, jj]
                end
            end
            lo_me = [q(ysim, ivar[i], t, a) for i in eachindex(ivar), t in axes(yf, 2)]
            hi_me = [q(ysim, ivar[i], t, 1 - a) for i in eachindex(ivar), t in axes(yf, 2)]
        end
    end
    yf, lo, hi, lo_me, hi_me
end

function _forecast_result(var_list, yf, lo, hi, lo_me, hi_me, me_vars, horizon, max_lag, conf_sig, order)
    cols = max_lag .+ (1:horizon)
    M, L, U, LM, UM = (Dict{String,Vector{Float64}}() for _ in 1:5)
    for (i, v) in enumerate(var_list)
        M[v] = yf[i, cols]; L[v] = lo[i, cols]; U[v] = hi[i, cols]
        if lo_me !== nothing && v in me_vars
            LM[v] = lo_me[i, cols]; UM[v] = hi_me[i, cols]
        end
    end
    ForecastResult(collect(String, var_list), M, L, U, LM, UM, horizon, Float64(conf_sig), order)
end

"Positions in `var_list` of the observed variables and their measurement-error variances (`[loc_H, loc_varlist] = ismember(varobs, var_list)`)."
function _me_positions(var_list, varobs, H)
    (H === nothing || isempty(H) || all(iszero, H)) && return nothing, String[]
    me = Tuple{Int,Float64}[]
    names = String[]
    for (k, v) in enumerate(varobs)
        j = findfirst(==(v), var_list)
        j === nothing && continue
        push!(me, (j, Float64(H[k, k])))
        push!(names, v)
    end
    me, names
end

function _check_var_list(m::Model, var_list)
    vl = isempty(var_list) ? orig_endo_names(m) : String.(collect(var_list))
    for v in vl
        haskey(m.ie, v) || throw(ModelError("forecast: $v isn't an endogenous variable"))
    end
    vl
end

# ====================================================================== forecast after stoch_simul
"""
    forecast(m::Model; periods=5, conf_sig=0.9, var_list=[], order=1, pruning=false, y0=nothing, histval=true,
             replic=1000, draws=nothing, me_draws=nothing, rng=nothing, dr=nothing) -> ForecastResult

Dynare's `forecast` after `stoch_simul` (`forecasts.run(..., 'simul')`): the forecast from the decision rule of order
`order` (1, 2 or 3; computed at the model's current calibration unless `dr` is given) over `periods` periods (Dynare: 5 when
`periods=0`), starting from `y0` (levels; in logs for a `loglinear` model), by default the `histval` block's values when the
file has one (unset variables at 0, as Dynare) and the steady state otherwise. Intervals hold `conf_sig` of the
forecast-error distribution: at order 1 from the exact variance, above from `replic` simulations (Dynare's
`forecast_replic`; `draws` replays given standard normals, see `_forecast_paths`). Observed variables get `HPDinf_ME` /
`HPDsup_ME` intervals that include the measurement-error variance when the model declares measurement errors.
`var_list` defaults to all (non-auxiliary) endogenous variables.
"""
function forecast(m::Model; periods::Integer=5, conf_sig::Real=0.9, var_list=String[], order::Integer=1, pruning::Bool=false,
                  y0=nothing, histval::Bool=true, replic::Integer=1000, draws=nothing, me_draws=nothing, rng=nothing, dr=nothing,
                  loglinear=nothing)
    rng === nothing && (rng = dynare_rng(m))      # the model's reproducible stream (set_dynare_seed / seed=)
    loglinear = loglinear === nothing ? get(stoch_options(m), "loglinear", false) == true : loglinear
    if dr === nothing
        dr = order == 1 ? as_dr(first_order(m; loglinear=loglinear)) : k_order(m; order=order, pruning=pruning)
    end
    dr = _dr_dict(dr)
    if y0 === nothing
        h = histval ? _histval_vector(m) : nothing
        y0 = h === nothing ? copy(dr["ys"]) : (loglinear ? _log_histval(m, h) : h)
    end
    vl = _check_var_list(m, var_list)
    H = me_H(m, m.varobs)
    xd = nothing
    if n_exo_det(m) > 0
        # varexo_det (forecasts.run): orders 1 and 2 without pruning, along the paths of the `shocks` blocks (src/varexo_det.jl)
        order > 2 && throw(ModelError("forecasts.run: forecasting with varexo_det does not support order>2."))
        order == 2 && pruning && throw(ModelError("forecasts.run: forecasting with varexo_det does not support pruning."))
        X, _ = exo_det_paths(m, Int(periods), periods == 0 ? 5 : Int(periods))
        xdr = exo_det_decision_rules(m, dr)
        xd = (X=X, rules=xdr, sim=ex -> permutedims(simult_varexo_det(m, xdr, ex, X; y0=y0, order=order)), iter=size(X, 1) - 1)
    end
    res = _forecast_simul(dr, y0, vl, m; periods=periods, conf_sig=conf_sig, order=order, pruning=pruning, Sigma_e=m.Sigma_e, H=H,
                          varobs=m.varobs, replic=replic, draws=draws, me_draws=me_draws, rng=rng, xd=xd)
    xd === nothing && return res
    hz = periods == 0 ? 5 : Int(periods)
    exo = Dict{String,Vector{Float64}}(v => Float64.(xd.X[2:1+hz, i]) for (i, v) in enumerate(exo_det_names(m)))
    ForecastResult(res.var_list, res.Mean, res.HPDinf, res.HPDsup, res.HPDinf_ME, res.HPDsup_ME, res.periods, res.conf_sig, res.order, exo)
end

function _forecast_simul(dr, y0, vl, m; periods, conf_sig, order, pruning, Sigma_e, H, varobs, replic, draws, me_draws, rng, max_lag=1, xd=nothing)
    horizon = periods == 0 ? 5 : Int(periods)
    ivar = [m.ie[v] for v in vl]
    me, me_vars = _me_positions(vl, varobs, H)
    me_obs = [m.ie[v] for v in varobs]
    yf, lo, hi, lo_me, hi_me = _forecast_paths(dr, Float64.(vec(y0)), horizon, ivar; order=order, pruning=pruning, Sigma_e=Sigma_e,
                                               me=me, me_obs=me_obs, H=H, conf_sig=conf_sig, replic=replic, draws=draws,
                                               me_draws=me_draws, rng=rng, max_lag=max_lag, xd=xd)
    _forecast_result(vl, yf, lo, hi, lo_me, hi_me, me_vars, horizon, max_lag, conf_sig, order)
end

"The histval block's values (n,), unset variables at 0 as Dynare's M_.endo_histval; `nothing` without a histval block."
function _histval_vector(m::Model)
    any(st -> get(st, "statementName", "") == "histval", m.statements) || return nothing
    h = _pf_replay(m, nothing, nothing, (), ()).endo_histval
    h === nothing ? nothing : Vector{Float64}(vec(h isa AbstractMatrix ? h[:, end] : h))
end
"`log_variable` of Dynare on a histval vector (auxiliary variables of lagged exogenous variables stay in levels)."
function _log_histval(m::Model, h)
    y = Float64.(copy(h))
    mask = log_mask(m)
    y[mask] = log.(y[mask])
    y
end

# ====================================================================== forecast after estimation
"""
    forecast(sm::SmootherResult, m::Model; periods, conf_sig=0.9, var_list=[], replicate_dynare_bugs=false) -> ForecastResult

Dynare's classical forecast after estimation (`forecasts.run(..., 'smoother')`, the `forecast` option of `estimation`): the
forecast from the last smoothed state of `sm` (a smoother result at the mode, or at the initial values with
`mode_compute=0`) with the parameters, shock and measurement-error covariances `sm` was computed with. Observed variables
get their `observation_trends` trend and, under `prefilter`, the data mean back (see the head of this file for the Dynare
defect `replicate_dynare_bugs` reproduces).
"""
function forecast(sm::SmootherResult, m::Model; periods::Integer, conf_sig::Real=0.9, var_list=String[],
                  replicate_dynare_bugs::Bool=false)
    horizon = Int(periods)
    horizon > 0 || throw(ArgumentError("forecast after estimation: periods must be positive"))
    sol = sm.solution
    sol === nothing && throw(ArgumentError("the smoother result carries no solution"))
    dr = as_dr(sol)
    vl = _check_var_list(m, var_list)
    ivar = [m.ie[v] for v in vl]
    y0 = sm.smoothed[end, :] .+ sol.ys                      # last smoothed state, deviations around the (log) steady state
    H = sm.me_cov
    me, me_vars = _me_positions(vl, sm.varobs, H)
    yf, lo, hi, lo_me, hi_me = _forecast_paths(dr, y0, horizon, ivar; order=1, Sigma_e=sm.Sigma_e, me=me, conf_sig=conf_sig)
    shift = zeros(size(yf))
    gend, first = size(sm.smoothed, 1), sm.first_obs
    prefilter = get(sm.options, "prefilter", 0) == 1
    # trend of the observed variables in var_list (Dynare: oo_.Smoother.TrendCoeffs is always stored, so the trend is
    # added whenever an observed variable is in var_list, with zero slopes without an observation_trends block)
    tc = sm.TrendCoeffs
    obs_in = [(k, v, findfirst(==(v), vl)) for (k, v) in enumerate(sm.varobs) if v in vl]
    for (k, v, j) in obs_in
        c = tc[m.ie[v]]
        tr = c .* (first + gend - 1 .+ (0:horizon))
        prefilter && (tr = tr .- c * sum(first:first+gend-1) / gend)
        shift[j, :] .+= tr
    end
    if prefilter
        if replicate_dynare_bugs && length(obs_in) != length(sm.varobs)
            throw(ArgumentError("Dynare 7.2 fails here: with prefilter its forecast needs every observed variable in var_list"))
        end
        for (pos, (k, v, j)) in enumerate(obs_in)
            ysv = replicate_dynare_bugs ? sol.ys[ivar[pos]] : sol.ys[m.ie[v]]   # Dynare: oo_.dr.ys(i_var_obs), a position in var_list
            shift[j, :] .+= sm.obs_constant[k] - ysv
        end
    end
    yf = yf .+ shift
    if !replicate_dynare_bugs
        lo, hi = lo .+ shift, hi .+ shift
        lo_me === nothing || ((lo_me, hi_me) = (lo_me .+ shift, hi_me .+ shift))
    end
    _forecast_result(vl, yf, lo, hi, lo_me, hi_me, me_vars, horizon, 1, conf_sig, 1)
end

# ====================================================================== conditional forecasts
"""
Result of `conditional_forecast`, Dynare's `oo_.conditional_forecast`: means and `ci` bands (2 x (periods + 1), the
first column being the initial condition) of the conditional (`cond_*`) and unconditional (`uncond_*`) forecasts of every
endogenous variable, and of the controlled shocks (`controlled_exo_mean` / `_ci`, one column per constrained period).
"""
struct ConditionalForecastResult
    endo_names::Vector{String}               # in decision-rule order (the order of Dynare's fields)
    cond_mean::Dict{String,Vector{Float64}}
    cond_ci::Dict{String,Matrix{Float64}}
    uncond_mean::Dict{String,Vector{Float64}}
    uncond_ci::Dict{String,Matrix{Float64}}
    instruments::Vector{String}
    controlled_exo_mean::Dict{String,Vector{Float64}}
    controlled_exo_ci::Dict{String,Matrix{Float64}}
    controlled_variables::Vector{Int}        # constrained variables, positions in decision-rule order (as Dynare)
    constrained_names::Vector{String}
    title::String
    fname::String
    periods::Int
    replic::Int
    conf_sig::Float64
    max_control_error::Float64               # largest gap between a constrained path and its target
end
Base.show(io::IO, r::ConditionalForecastResult) = print(io, "ConditionalForecastResult(", r.periods, " periods, ", r.replic,
    " replications, constrained ", join(r.constrained_names, ", "), " with ", join(r.instruments, ", "), ")")

"`oo_.conditional_forecast` as Dynare stores it."
function conditional_forecast_struct(r::ConditionalForecastResult)
    S = MStruct()
    S.controlled_variables = reshape(Float64.(r.controlled_variables), 1, :)      # a row, as Dynare's inv_order_var(...)
    S.instruments = _cellcol(r.instruments)
    S.cond = MStruct(["Mean" => _ordered_cols(r.endo_names, r.cond_mean), "ci" => _ordered_mats(r.endo_names, r.cond_ci)])
    S.controlled_exo_variables = MStruct(["Mean" => _ordered_cols(r.instruments, r.controlled_exo_mean),
                                          "ci" => _ordered_mats(r.instruments, r.controlled_exo_ci)])
    S.uncond = MStruct(["Mean" => _ordered_cols(r.endo_names, r.uncond_mean), "ci" => _ordered_mats(r.endo_names, r.uncond_ci)])
    S.graph = MStruct(["title" => r.title, "fname" => r.fname])
    S
end
_ordered_cols(names, d) = (s = MStruct(); for v in names; s[v] = reshape(copy(d[v]), :, 1); end; s)
_ordered_mats(names, d) = (s = MStruct(); for v in names; s[v] = copy(d[v]); end; s)

"`conditional_forecasts.get_shock_paths`: the path of the state and of the controlled shocks."
function _cf_shock_paths!(forcs, cL, Hh, mcValue, shocks, T, R, mv, mu)
    e = zeros(size(mcValue, 1), cL)
    for t in 1:cL
        k = findall(isfinite, mcValue[:, t])
        if isempty(k)
            forcs[:, t+1] = T * forcs[:, t] + R * shocks[:, t]
            continue
        end
        A = mv[k, :] * R * mu[:, k]
        e[k, t] = inv(A) * (mcValue[k, t] - mv[k, :] * T * forcs[:, t] - mv[k, :] * R * shocks[:, t])
        forcs[:, t+1] = T * forcs[:, t] + R * (mu[:, k] * e[k, t] + shocks[:, t])
    end
    for t in cL+1:Hh
        forcs[:, t+1] = T * forcs[:, t] + R * shocks[:, t]
    end
    e
end

"""
    _conditional_forecast_core(T, R, Sigma_e, init, trend, cvars, cpaths, controlled_exo; periods, replic, conf_sig, draws, rng)

The simulation part of `conditional_forecasts.run` in decision-rule order: `T` (n x n) and `R` (n x n_exo) the first-order
rule, `init` the initial deviation, `trend` (n x (periods + 1)) the level to add back, `cvars` the constrained variables
(decision-rule positions), `cpaths` their target levels (n1 x cL, NaN = free), `controlled_exo` the instruments (indices).
`draws` (n_exo, periods, 2 replic): the standard normals of Dynare's `randn(ExoSize, periods)`, first the `replic` draws of
the conditional forecasts, then those of the unconditional ones.
"""
function _conditional_forecast_core(T, R, Sigma_e, init, trend, cvars, cpaths, controlled_exo; periods, replic, conf_sig, draws=nothing, rng=nothing)
    n, nx = size(R)
    n1 = length(cvars)
    n1 == length(controlled_exo) || throw(ModelError("conditional_forecast: The number of constrained variables doesn't match the number of controlled shocks"))
    if isdiag(Sigma_e)
        sQ = sqrt.(Matrix{Float64}(Sigma_e))
    else
        @warn "conditional_forecast: the innovations are correlated; the results depend on the ordering of the shocks (a Cholesky factor is used)"
        sQ = Matrix(cholesky(Symmetric(Matrix{Float64}(Sigma_e))).L)
    end
    mv = zeros(n1, n)
    mu = zeros(nx, n1)
    for i in 1:n1
        mv[i, cvars[i]] = 1.0
        mu[controlled_exo[i], i] = 1.0
    end
    cL = size(cpaths, 2)
    cL <= periods || throw(ModelError("conditional_forecast: the constrained paths ($cL periods) are longer than the forecast (periods=$periods)"))
    mc = cpaths .- trend[cvars, 2:1+cL]
    if draws === nothing
        r = rng === nothing ? Random.default_rng() : rng
        draws = randn(r, nx, periods, 2 * replic)
    else
        size(draws) == (nx, periods, 2 * replic) || throw(ArgumentError("draws must be (exo_nbr, periods, 2 replic) = ($nx, $periods, $(2replic))"))
    end
    F1 = zeros(n, periods + 1, replic)
    E1 = zeros(n1, cL, replic)
    for b in 1:replic
        shocks = sQ * draws[:, :, b]
        shocks[controlled_exo, :] .= 0.0
        f = zeros(n, periods + 1)
        f[:, 1] = init
        E1[:, :, b] = _cf_shock_paths!(f, cL, periods, mc, shocks, T, R, mv, mu)
        F1[:, :, b] = f .+ trend
    end
    err = 0.0
    for b in 1:replic, i in 1:n1, t in 1:cL
        isfinite(cpaths[i, t]) && (err = max(err, abs(F1[cvars[i], 1+t, b] - cpaths[i, t])))
    end
    err > 1e-4 && @warn "conditional_forecast: controlling of variables was not successful (numerical imprecision, e.g. explosive simulations, or instruments that do not allow controlling the variables)"
    F2 = zeros(n, periods + 1, replic)
    for b in 1:replic
        shocks = sQ * draws[:, :, replic+b]
        shocks[controlled_exo, :] .= 0.0
        f = zeros(n, periods + 1)
        f[:, 1] = init
        _cf_shock_paths!(f, 0, periods, mc, shocks, T, R, mv, mu)
        F2[:, :, b] = f .+ trend
    end
    tt = (1 - conf_sig) / 2
    t1 = max(1, _iround(replic * tt))
    t2 = min(replic, _iround(replic * (1 - tt)))
    band(X, i) = (s = sort(X[i, :, :]; dims=2); vcat(reshape(s[:, t1], 1, :), reshape(s[:, t2], 1, :)))
    meanr(X) = dropdims(sum(X; dims=3); dims=3) ./ size(X, 3)
    F1, E1, F2, meanr(F1), meanr(E1), meanr(F2), band, err
end

"""
    conditional_forecast(m::Model; paths, controlled_varexo, periods=40, replic=5000, conf_sig=0.8,
                         parameter_set="calibration", estimation=nothing, theta=nothing, mode=nothing, mcmc=nothing,
                         y0=nothing, histval=true, draws=nothing, rng=nothing) -> ConditionalForecastResult

Dynare's `conditional_forecast` (imcforecast): forecasts in which the variables of `paths` follow given values over the
first periods (`paths`: pairs `"var" => values`, NaN where the variable is free; ordered as Dynare orders them, by
declaration) while the shocks `controlled_varexo` (one per constrained variable) take the values that make them do so;
the other shocks are drawn (`replic` times). The unconditional forecast uses the same initial state with all shocks
except the instruments drawn.

`parameter_set="calibration"`: the model's current calibration, starting from `y0` (levels; logs under `loglinear`),
default the histval block (unset variables at 0, as Dynare) or the steady state. Otherwise (`"posterior_mode"`,
`"posterior_mean"`, `"posterior_median"`, `"mle_mode"`, `"prior_mode"`, `"prior_mean"`, or `theta=`) the parameters of
the estimation `estimation` (with `mode` / `mcmc` as `smooth` takes them) and the last smoothed state of the data; trends
of `observation_trends` are added to the observed variables. `prefilter` is refused, as in Dynare.

`draws` (exo_nbr, periods, 2 replic) replays given standard normals (see `_conditional_forecast_core`); with every shock
controlled the result does not depend on them.
"""
function conditional_forecast(m::Model; paths, controlled_varexo, periods::Integer=40, replic::Integer=5000, conf_sig::Real=0.8,
                              parameter_set::AbstractString="calibration", estimation=nothing, theta=nothing, mode=nothing, mcmc=nothing,
                              y0=nothing, histval::Bool=true, draws=nothing, rng=nothing, loglinear=nothing, smoother=nothing)
    rng === nothing && (rng = dynare_rng(m))      # the model's reproducible stream (set_dynare_seed / seed=)
    cvars_names, cpaths = _paths_matrix(m, paths)
    instruments = String.(collect(controlled_varexo))
    for e in instruments
        haskey(m.ix, e) || throw(ModelError("conditional_forecast: Declared controlled shock $e is not a shock in the model."))
    end
    estimated = parameter_set != "calibration" || theta !== nothing || smoother !== nothing
    if estimated
        sm = smoother
        if sm === nothing
            est = estimation === nothing ? throw(ModelError("conditional_forecast(parameter_set=$(repr(parameter_set))) needs estimation=<Estimation>")) : estimation
            est.options["prefilter"] != 0 && throw(ModelError("conditional_forecast: Conditional forecasting does not support the prefiltering option"))
            th = theta === nothing ? parameter_vector(est, parameter_set; mode=mode, mcmc=mcmc) : Vector{Float64}(theta)
            sm = smooth(m; estimation=est, theta=th)
        end
        get(sm.options, "prefilter", 0) == 1 && throw(ModelError("conditional_forecast: Conditional forecasting does not support the prefiltering option"))
        title = Dict("posterior_mode" => "Posterior Mode", "posterior_mean" => "Posterior Mean", "posterior_median" => "Posterior Median",
                     "mle_mode" => "ML Mode", "prior_mode" => "Prior Mode", "prior_mean" => "Prior Mean")
        return _cf_estimated(m, sm, cvars_names, cpaths, instruments; periods=periods, replic=replic, conf_sig=conf_sig, draws=draws, rng=rng,
                             title=get(title, parameter_set, "Theta"))
    end
    loglinear = loglinear === nothing ? get(stoch_options(m), "loglinear", false) == true : loglinear
    sol = first_order(m; loglinear=loglinear)
    if y0 === nothing
        h = histval ? _histval_vector(m) : nothing
        y0 = h === nothing ? nothing : h
    end
    _cf_calibrated(m, sol, y0, cvars_names, cpaths, instruments; periods=periods, replic=replic, conf_sig=conf_sig, draws=draws, rng=rng,
                   loglinear=loglinear, Sigma_e=m.Sigma_e)
end

"Constrained variables and their target matrix (n1 x path length, NaN = free) from pairs var => values."
function _paths_matrix(m::Model, paths)
    if paths isa Tuple{<:AbstractVector,<:AbstractMatrix}
        names, P = String.(paths[1]), Matrix{Float64}(paths[2])
    else
        prs = collect(paths)
        isempty(prs) && throw(ModelError("conditional_forecast: no constrained paths (conditional_forecast_paths block)"))
        names = String[String(first(p)) for p in prs]
        L = maximum(length(last(p)) for p in prs)
        P = fill(NaN, length(prs), L)
        for (i, p) in enumerate(prs)
            v = Float64.(collect(last(p)))
            P[i, 1:length(v)] = v
        end
    end
    for v in names
        haskey(m.ie, v) || throw(ModelError("conditional_forecast: $v is not an endogenous variable"))
    end
    names, P
end

function _cf_calibrated(m, sol, y0, cnames, cpaths, instruments; periods, replic, conf_sig, draws, rng, loglinear, Sigma_e, title="Calibration")
    ov = sol.order_var
    ys = sol.ys                                     # log steady state under loglinear
    yy0 = if y0 === nothing
        copy(ys)
    elseif loglinear
        # Dynare: only variables with a non-zero histval value are logged, the others stay at 0
        h = Float64.(vec(y0))
        z = zeros(length(h))
        lg = _log_histval(m, h)
        z[h .!= 0] = lg[h .!= 0]
        z
    else
        Float64.(vec(y0))
    end
    init = yy0[ov] - ys[ov]
    trend = repeat(ys[ov], 1, periods + 1)
    _cf_finish(m, sol, init, trend, cnames, cpaths, instruments, Sigma_e; periods, replic, conf_sig, draws, rng, title)
end

function _cf_estimated(m, sm::SmootherResult, cnames, cpaths, instruments; periods, replic, conf_sig, draws, rng, title)
    sol = sm.solution
    ov = sol.order_var
    gend, first = size(sm.smoothed, 1), sm.first_obs
    constant = repeat(sol.ys, 1, periods + 1)
    for (k, v) in enumerate(sm.varobs)
        c = sm.TrendCoeffs[m.ie[v]]
        constant[m.ie[v], :] .+= c .* (first + gend - 1 .+ (0:periods))
    end
    trend = constant[ov, :]
    init = sm.smoothed[end, ov]
    _cf_finish(m, sol, init, trend, cnames, cpaths, instruments, sm.Sigma_e; periods, replic, conf_sig, draws, rng, title)
end

function _cf_finish(m, sol, init, trend, cnames, cpaths, instruments, Sigma_e; periods, replic, conf_sig, draws, rng, title)
    ov = sol.order_var
    inv_ov = invperm(ov)
    T = sol.G[ov, ov]
    R = sol.H[ov, :]
    cvars = [inv_ov[m.ie[v]] for v in cnames]
    cexo = [m.ix[e] for e in instruments]
    F1, E1, F2, mF1, mE1, mF2, band, err = _conditional_forecast_core(T, R, Sigma_e, init, trend, cvars, cpaths, cexo;
        periods=Int(periods), replic=Int(replic), conf_sig=Float64(conf_sig), draws=draws, rng=rng)
    names = m.endo_names[ov]
    cm, cci, um, uci = Dict{String,Vector{Float64}}(), Dict{String,Matrix{Float64}}(), Dict{String,Vector{Float64}}(), Dict{String,Matrix{Float64}}()
    for (i, v) in enumerate(names)
        cm[v] = mF1[i, :]; cci[v] = band(F1, i)
        um[v] = mF2[i, :]; uci[v] = band(F2, i)
    end
    em, eci = Dict{String,Vector{Float64}}(), Dict{String,Matrix{Float64}}()
    for (i, e) in enumerate(instruments)
        em[e] = mE1[i, :]; eci[e] = band(E1, i)
    end
    ConditionalForecastResult(names, cm, cci, um, uci, instruments, em, eci, cvars, cnames, title, m.name, Int(periods), Int(replic),
                              Float64(conf_sig), err)
end

"""
    conditional_forecast_table(cf, variables=cf's constrained variables; periods=nothing)

The numbers `plot_conditional_forecast` draws, as rows (variable, period, cond_mean, cond_lower, cond_upper, uncond_mean,
uncond_lower, uncond_upper); period 1 is the initial condition, as in Dynare's figures. `periods` (Dynare's option) keeps
periods + 1 columns. `cf` is a `ConditionalForecastResult` or `oo_.conditional_forecast` (an MStruct). Write with `write_csv`.
"""
function conditional_forecast_table(cf, variables=nothing; periods=nothing)
    S = cf isa ConditionalForecastResult ? conditional_forecast_struct(cf) : cf
    vars = variables === nothing ? (cf isa ConditionalForecastResult ? cf.constrained_names : collect(keys(S.cond.Mean))) : String.(collect(variables))
    isempty(vars) && throw(ArgumentError("plot_conditional_forecast: no variables"))
    for v in vars
        haskey(S.cond.Mean, v) || throw(ArgumentError("plot_conditional_forecast: $v is not an endogenous variable of the conditional forecast"))
    end
    nf = length(S.cond.Mean[vars[1]])
    P = periods === nothing ? nf : Int(periods) + 1
    if P > nf
        @warn "conditional_forecasts.plot: Number of periods for plotting exceeds forecast horizon. Setting periods to $(nf - 1)."
        P = nf
    end
    rows = NamedTuple{(:variable, :period, :cond_mean, :cond_lower, :cond_upper, :uncond_mean, :uncond_lower, :uncond_upper),
                      Tuple{String,Int,Float64,Float64,Float64,Float64,Float64,Float64}}[]
    for v in vars, t in 1:P
        c, u = S.cond.ci[v], S.uncond.ci[v]
        push!(rows, (variable=v, period=t, cond_mean=S.cond.Mean[v][t], cond_lower=c[1, t], cond_upper=c[2, t],
                     uncond_mean=S.uncond.Mean[v][t], uncond_lower=u[1, t], uncond_upper=u[2, t]))
    end
    rows
end

# ====================================================================== Dynare workspace (dynare(), @dynare)
"""
    forecast!(ws=current workspace, var_list=[]; draws=nothing, me_draws=nothing, rng=nothing, options...)

Dynare's `forecast` statement on a workspace: from the decision rule in `oo_.dr` (as Dynare, the last `stoch_simul`'s),
`M_.Sigma_e`, `M_.H`, `M_.endo_histval` and `options_` (`periods`, `forecasts.conf_sig`, `order`, `pruning`, `loglinear`,
`forecast_replic`; keyword options are written into options_ first, e.g. `forecast!(ws; periods=12)`); fills `oo_.forecast`.
"""
function forecast!(ws::DynareWorkspace, var_list=String[]; draws=nothing, me_draws=nothing, rng=nothing, plots=nothing, options...)
    m = _sync_model!(ws)
    _take_options!(ws, options)
    oo = ws.oo_
    D = get(oo, "dr", nothing)
    (D isa MStruct && haskey(D, "ghx")) || throw(ModelError("forecasts.run: the decision rules have not been computed. Did you forget a stoch_simul-command?"))
    _optb(ws, "prefilter", false) && throw(ModelError("forecasts.run: The prefiltering option is not allowed without providing a dataset"))
    dr = _dr_dict(D)
    order = _optb(ws, "linear", false) ? 1 : _opti(ws, "order", 2)
    order <= dr["order"] || throw(ModelError("forecast: options_.order=$order but oo_.dr holds a decision rule of order $(dr["order"])"))
    ml = Int(get(ws.M_, "maximum_lag", 1.0))
    ml <= 1 || throw(ModelError("forecast: maximum_lag = $ml is not supported"))
    h = ws.M_.endo_histval
    y0 = isempty(h) ? copy(dr["ys"]) : (_optb(ws, "loglinear", false) ? _log_histval(m, vec(h[:, end])) : Float64.(vec(h[:, end])))
    vl = _check_var_list(m, String.(collect(var_list)))
    # M_.H; while the file runs the shocks block's measurement errors are in the model (M_ is rebuilt at the end)
    H = Matrix{Float64}(ws.M_.H)
    np = length(m.varobs)
    (size(H) != (np, np) || all(iszero, H)) && (H = me_H(m, m.varobs))
    res = _forecast_simul(dr, y0, vl, m; periods=_opti(ws, "periods", 0), conf_sig=_optf(ws, "forecasts.conf_sig", 0.9), order=order,
                          pruning=_optb(ws, "pruning", false), Sigma_e=Matrix{Float64}(ws.M_.Sigma_e), H=H,
                          varobs=m.varobs, replic=_opti(ws, "forecast_replic", 1000), draws=draws, me_draws=me_draws, rng=rng, max_lag=max(ml, 1))
    oo.forecast = forecast_struct(res)
    ws.state[:forecast] = res
    plots === true && _statement_figures!(ws, "forecast", Dict{String,Any}())       # src/plots_hooks.jl
    ws
end

"""
    conditional_forecast!(ws=current workspace; paths=nothing, draws=nothing, rng=nothing, estimation=nothing, mode=nothing,
                          mcmc=nothing, options...)

Dynare's `conditional_forecast` on a workspace: the constrained paths of the last `conditional_forecast_paths` block (or
`paths`, as `conditional_forecast` takes them), options as the statement's (`controlled_varexo`, `periods`, `replic`, `conf_sig`,
`parameter_set`; Dynare's defaults 40, 5000, 0.8 and posterior_mode/mle_mode when oo_ holds one). With
`parameter_set=calibration` the model is solved at `M_.params` and starts from `M_.endo_histval`; otherwise the estimated
parameters are read from `oo_.posterior_mode` / `oo_.posterior_mean` / `oo_.posterior_median` / `oo_.mle_mode` (Dynare's
`get_posterior_parameters`), or from `mode`/`mcmc`, and the forecast starts from the smoothed state of the data. Fills
`oo_.conditional_forecast` and the MATLAB variable `options_cond_fcst_`.
"""
function conditional_forecast!(ws::DynareWorkspace; paths=nothing, draws=nothing, rng=nothing, estimation=nothing, mode=nothing,
                               mcmc=nothing, plots=nothing, options...)
    m = _sync_model!(ws)
    o = Dict{String,Any}(String(k) => v for (k, v) in options)
    haskey(o, "conf_sig") && (o["conditional_forecast.conf_sig"] = pop!(o, "conf_sig"))
    for k in keys(o)
        k in ("controlled_varexo", "periods", "replic", "conditional_forecast.conf_sig", "parameter_set") ||
            throw(ModelError("conditional_forecast: unknown option $k"))
    end
    ocf = MStruct()                                            # options_cond_fcst_ (a fresh struct for each statement)
    for (k, v) in o
        _setpath!(ocf, k, k == "controlled_varexo" ? _cellcol(_symbol_names(v)) : _mval(v))
    end
    pset = haskey(o, "parameter_set") ? String(o["parameter_set"]) :
           haskey(ws.oo_, "posterior_mode") ? "posterior_mode" : haskey(ws.oo_, "mle_mode") ? "mle_mode" :
           throw(ModelError("conditional_forecast: No valid parameter set found"))
    periods = Int(get(o, "periods", 40)); replic = Int(get(o, "replic", 5000))
    conf_sig = Float64(get(o, "conditional_forecast.conf_sig", 0.8))
    ocf.parameter_set = pset; ocf.periods = Float64(periods); ocf.replic = Float64(replic)
    _setpath!(ocf, "conditional_forecast.conf_sig", conf_sig)
    instruments = haskey(ocf, "controlled_varexo") ? String.(collect(ocf.controlled_varexo)) : String[]
    ws.state[:options_cond_fcst] = ocf
    _set_matlab_var!(ws, "options_cond_fcst_", ocf)
    if paths === nothing
        haskey(ws.state, :cf_paths) || throw(ModelError("conditional_forecast: no conditional_forecast_paths block before the statement"))
        paths = ws.state[:cf_paths]
    end
    cnames, cpaths = _paths_matrix(m, paths)
    for e in instruments
        haskey(m.ix, e) || throw(ModelError("conditional_forecast: Declared controlled shock $e is not a shock in the model."))
    end
    res = if pset == "calibration"
        D = get(ws.oo_, "dr", nothing)
        (D isa MStruct && haskey(D, "ghx")) || throw(ModelError("You need to call stoch_simul before conditional_forecast"))
        loglinear = _optb(ws, "loglinear", false)
        sol = _with_options(() -> first_order(m; loglinear=loglinear), ws, _SOLVE_KINDS)
        _store_params!(ws, sol.params)
        h = ws.M_.endo_histval
        _cf_calibrated(m, sol, isempty(h) ? nothing : vec(h[:, end]), cnames, cpaths, instruments; periods=periods, replic=replic,
                       conf_sig=conf_sig, draws=draws, rng=rng, loglinear=loglinear, Sigma_e=Matrix{Float64}(ws.M_.Sigma_e))
    else
        est = estimation === nothing ? Estimation(m; allow_unsupported=true) : estimation
        theta = if mode !== nothing || mcmc !== nothing || pset in ("prior_mode", "prior_mean")
            parameter_vector(est, pset; mode=mode, mcmc=mcmc)
        else
            _theta_from_oo(est, ws.oo_, pset)
        end
        conditional_forecast(m; paths=(cnames, cpaths), controlled_varexo=instruments, periods=periods, replic=replic, conf_sig=conf_sig,
                             parameter_set=pset, estimation=est, theta=theta, draws=draws, rng=rng)
    end
    ws.oo_.conditional_forecast = conditional_forecast_struct(res)
    ws.state[:conditional_forecast] = res
    plots === true && _statement_figures!(ws, "plot_conditional_forecast", Dict{String,Any}())
    ws
end

_symbol_names(v) = v isa AbstractDict ? _symbol_names(get(v, "symbol_list", String[])) : v isa MStruct ? _symbol_names(v["symbol_list"]) :
                   v isa AbstractString ? [String(v)] : String[String(x) for x in v]

"Dynare's `get_posterior_parameters`: the estimated-parameter vector stored in oo_.posterior_mode (mean, median) or oo_.mle_mode."
function _theta_from_oo(est::Estimation, oo::MStruct, pset::AbstractString)
    field = pset in ("posterior_mode", "posterior_mean", "posterior_median", "mle_mode") ? pset :
            throw(ModelError("conditional_forecast: unknown parameter_set $(repr(pset))"))
    S = get(oo, field, nothing)
    S isa MStruct || throw(ModelError("conditional_forecast: parameter_set=$pset but oo_.$field does not exist (run the estimation first)"))
    m = est.model
    th = zeros(est.n)
    for (i, nm) in enumerate(est.names)
        if startswith(nm, "stderr ")
            v = nm[8:end]
            th[i] = haskey(m.ix, v) ? S.shocks_std[v] : S.measurement_errors_std[v]
        elseif startswith(nm, "corr ")
            a, b = strip.(split(nm[6:end], ","))
            th[i] = haskey(m.ix, a) ? S.shocks_corr["$(a)_$(b)"] : S.measurement_errors_corr["$(a)_$(b)"]
        else
            th[i] = S.parameters[nm]
        end
    end
    th
end

"A MATLAB variable of the .mod file's workspace (while dynare() runs the file) and of `ws.vars`."
function _set_matlab_var!(ws::DynareWorkspace, name::AbstractString, v)
    ws.vars[name] = v
    sess = ws.model.session
    sess === nothing && return
    try
        sess.sc.vars[name] = v
    catch
    end
end

"conditional_forecast_paths where it stands: the targets (values may be MATLAB expressions, scalars or vectors)."
function _cf_paths_statement!(ws::DynareWorkspace, st)
    m = ws.model
    entries = get_or(st, "paths", [])
    L = 0
    for p in entries, v in p["values"]
        L = max(L, Int(v["period2"]))
    end
    names = String[]
    P = fill(NaN, length(entries), L)
    for (i, p) in enumerate(entries)
        push!(names, String(p["var"]))
        for v in p["values"]
            p1, p2 = Int(v["period1"]), Int(v["period2"])
            val = _value(m, v["value"]; vector=true)
            vals = val isa AbstractArray ? vec(Float64.(val)) : [Float64(val)]
            length(vals) == 1 ? (P[i, p1:p2] .= vals[1]) :
                length(vals) == p2 - p1 + 1 ? (P[i, p1:p2] = vals) :
                throw(ModelError("conditional_forecast_paths: $(length(vals)) values for periods $p1:$p2 of $(p["var"])"))
        end
    end
    ws.state[:cf_paths] = (names, P)
    _set_matlab_var!(ws, "constrained_vars_", reshape(Float64[m.ie[v] for v in names], :, 1))
    _set_matlab_var!(ws, "constrained_paths_", P)
    nothing
end

"The forecast statements of a .mod file where they stand (called by `_run_in_place!`)."
function _forecast_statement!(ws::DynareWorkspace, kind, st, rng)
    if kind == "forecast"
        forecast!(ws, String.(get_or(st, "symbol_list", String[])); rng=rng)
    elseif kind == "conditional_forecast"
        opts = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in get_or(st, "options", Dict{String,Any}()))
        conditional_forecast!(ws; rng=rng, opts...)
    elseif kind == "plot_conditional_forecast"
        haskey(ws.oo_, "conditional_forecast") || throw(ModelError("Can't find conditional forecasts"))
        vars = String.(get_or(st, "symbol_list", String[]))
        tbl = conditional_forecast_table(ws.oo_.conditional_forecast, vars; periods=get(st, "periods", nothing))
        push!(get!(ws.state, :plot_conditional_forecast, Any[]), tbl)
    end
    nothing
end

forecast!(var_list::AbstractVector=String[]; kw...) = forecast!(_current(), var_list; kw...)
conditional_forecast!(; kw...) = conditional_forecast!(_current(); kw...)
