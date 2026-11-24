# Forecasts over posterior draws: the forecast branch of Dynare 7.2's estimation/prior_posterior_statistics_core.m with the
# moments of pm3.m (oo_.MeanForecast, oo_.PointForecast, oo_.PointForecastME). Validated against Dynare by replaying its
# sub-draws and its random numbers (test/test_posterior.jl). Included after posterior.jl, smoother.jl and forecast.jl.
# Adapted from Dynare (GPL-3.0-or-later, (C) the Dynare Team); see NOTICE.

"""
Result of `posterior_forecast`: for every variable of `var_list` (every observed one for `PointForecastME`) a `SeriesStats` with
one series per forecast period (Dynare's `oo_.MeanForecast`, `oo_.PointForecast`, `oo_.PointForecastME`).
`MeanForecast` is the path without future shocks from the smoothed last state, `PointForecast` one draw of the shocks along it and
`PointForecastME` that path of the observed variables plus one draw of the measurement errors.
"""
struct PosteriorForecast
    var_list::Vector{String}
    horizon::Int
    mh_conf_sig::Float64
    draws::Int
    MeanForecast::Dict{String,SeriesStats}
    PointForecast::Dict{String,SeriesStats}
    PointForecastME::Dict{String,SeriesStats}
    params::Matrix{Float64}
    mean_draws::Union{Nothing,Array{Float64,3}}          # (draws, horizon, endogenous variable)
    point_draws::Union{Nothing,Array{Float64,3}}
    me_draws::Union{Nothing,Array{Float64,3}}            # (draws, horizon, observed variable)
end
Base.show(io::IO, r::PosteriorForecast) = print(io, "PosteriorForecast(", r.draws, " draws, ", r.horizon, " periods, ", length(r.var_list), " variables)")

_fc_stats(d::Array{Float64,3}, idx, names, conf) = Dict{String,SeriesStats}(nm => series_stats(d[:, :, i], conf) for (nm, i) in zip(names, idx))

"""
    posterior_forecast(est, mc=nothing; params=nothing, horizon, var_list, sub_draws, mh_conf_sig, drop, rng, subsample_rng,
                       keep_draws=false, smoothers=nothing, smoother_kw...) -> PosteriorForecast

Forecasts over posterior draws. `params` (rows ordered as `est.names`) or the chains `mc` (`sub_draws` vectors selected as in
`prior_posterior_statistics.m`, with `subsample_rng`, default the model's stream) give the parameter vectors. For each one the
smoother runs and, from the smoothed state of the last period, `horizon` periods are forecast. `rng` supplies the normals in Dynare's
order: per draw the shocks `randn(nshocks, horizon)`, then the measurement errors; with `OctaveRNG(seed)` Dynare's numbers under
Octave are reproduced. Dynare draws the measurement errors as `chol(H) * randn` (the upper factor), so their covariance is
`chol(H) chol(H)'`, which is `H` only for a diagonal `H` (reproduced, with a warning otherwise). `smoothers` may carry ready
`SmootherResult`s. First-order models only.
"""
function posterior_forecast(est::Estimation, mc::Union{Nothing,MCMCResult}=nothing; params=nothing, horizon=nothing, var_list=nothing,
                            sub_draws=nothing, mh_conf_sig=nothing, drop=nothing, rng=nothing, subsample_rng=nothing,
                            keep_draws::Bool=false, smoothers=nothing, smoother_kw...)
    m = est.model
    isempty(m.occbin_constraints) || throw(EstimationError("posterior_forecast: OccBin models are not supported"))
    h = horizon === nothing ? _opti_forecast(est) : Int(horizon)
    h > 0 || throw(ArgumentError("posterior_forecast: no horizon (pass horizon=, or estimation(forecast=N))"))
    conf = mh_conf_sig === nothing ? Float64(est.options["mh_conf_sig"]) : Float64(mh_conf_sig)
    if params === nothing
        mc === nothing && throw(ArgumentError("posterior_forecast needs parameter draws (params=) or chains (mc)"))
        dr = drop === nothing ? mc.drop : drop
        P, _ = posterior_draws(est, mc; sub_draws=sub_draws, drop=dr, rng=subsample_rng === nothing ? dynare_rng(m) : subsample_rng)
        params = P
    end
    P = Matrix{Float64}(params isa AbstractVector ? reshape(params, 1, :) : params)
    vl = var_list === nothing ? _estimation_var_list_est(est) : String.(collect(var_list))
    isempty(vl) && (vl = orig_endo_names(m))
    for v in vl
        haskey(m.ie, v) || throw(EstimationError("posterior_forecast: $v is not an endogenous variable"))
    end
    rng === nothing && (rng = dynare_rng(m))
    B = size(P, 1)
    obs = collect(String, est.varobs)
    nobs_ = length(obs)
    mean_d = point_d = me_d = nothing
    any_me = false
    for b in 1:B
        res = smoothers === nothing ? smooth(m; estimation=est, theta=P[b, :], smoother_kw...) : smoothers[b]
        G, Hr, Sigma = res.G, res.H, res.Sigma_e
        n = size(G, 1)
        if mean_d === nothing
            mean_d = fill(NaN, B, h, n); point_d = fill(NaN, B, h, n); me_d = fill(NaN, B, h, nobs_)
        end
        ys_c = res.solution.ys
        mf = res.obs_index
        Hme = isempty(res.me_cov) ? zeros(nobs_, nobs_) : res.me_cov
        dev0 = res.smoothed[end, :]
        gend, first_obs = size(res.smoothed, 1), res.first_obs
        tc = res.TrendCoeffs[mf]
        prefilter = get(res.options, "prefilter", 0) == 1
        nz = findall(!iszero, diag(Sigma))
        chol_S = isempty(nz) ? zeros(0, 0) : Matrix(cholesky(Symmetric(Sigma[nz, nz])).U)
        e = randn(rng, length(nz), h)
        B1 = Hr[:, nz] * chol_S'
        paths = (zeros(h + 1, n), zeros(h + 1, n))
        paths[1][1, :] = dev0; paths[2][1, :] = dev0
        for t in 1:h
            paths[1][t+1, :] = G * paths[1][t, :]
            paths[2][t+1, :] = G * paths[2][t, :] + B1 * e[:, t]
        end
        t_idx = (first_obs - 1) + gend .+ (0:h)                                  # (1-maxlag:horizon), maxlag = 1
        out = Matrix{Float64}[]
        for path in paths
            if prefilter
                path[:, mf] .+= reshape(res.obs_constant, 1, :)
                path[:, mf] .+= t_idx * tc' .- reshape(sum((first_obs:first_obs+gend-1) * tc'; dims=1) ./ gend, 1, :)
            else
                path[:, mf] .+= t_idx * tc'
            end
            push!(out, (path .+ reshape(ys_c, 1, :))[2:end, :])                   # drop the starting value
        end
        mean_d[b, :, :], point_d[b, :, :] = out
        iz = findall(!iszero, diag(Hme))
        me_shocks = zeros(nobs_, h)
        if !isempty(iz)
            any_me = true
            ch = Matrix(cholesky(Symmetric(Hme[iz, iz])).U)
            isdiag(ch) || warn_once(m, "posterior_forecast: correlated measurement errors; Dynare draws them as chol(H)*randn, whose covariance is chol(H)*chol(H)', not H (reproduced here)")
            me_shocks[iz, :] = ch * randn(rng, length(iz), h)
        end
        me_d[b, :, :] = point_d[b, :, mf] .+ me_shocks'
    end
    idx = [m.ie[v] for v in vl]
    res_mean = _fc_stats(mean_d, idx, vl, conf)
    res_point = _fc_stats(point_d, idx, vl, conf)
    res_me = Dict{String,SeriesStats}()
    if any_me
        sel = [v for v in obs if v in vl]
        res_me = _fc_stats(me_d, [findfirst(==(v), obs) for v in sel], sel, conf)
    end
    PosteriorForecast(vl, h, conf, B, res_mean, res_point, res_me, P, keep_draws ? mean_d : nothing, keep_draws ? point_d : nothing,
                      keep_draws && any_me ? me_d : nothing)
end

_opti_forecast(est::Estimation) = Int(get(est.options, "forecast", 0))

"`oo_.MeanForecast`, `oo_.PointForecast`, `oo_.PointForecastME` (`Mean`, `Median`, `Var`, `HPDinf`, `HPDsup` columns, `deciles` 9 x periods)."
function store_posterior_forecast!(oo::MStruct, r::PosteriorForecast)
    for (field, d) in (("MeanForecast", r.MeanForecast), ("PointForecast", r.PointForecast), ("PointForecastME", r.PointForecastME))
        isempty(d) && (haskey(oo, field) && delete!(oo, field); continue)
        S = Dict(f => MStruct() for f in ("Mean", "Median", "Var", "deciles", "HPDinf", "HPDsup"))
        for nm in (v for v in vcat(r.var_list, collect(keys(d))) |> unique if haskey(d, v))
            s = d[nm]
            S["Mean"][nm] = _col(s.mean); S["Median"][nm] = _col(s.median); S["Var"][nm] = _col(s.var)
            S["HPDinf"][nm] = _col(s.hpd_inf); S["HPDsup"][nm] = _col(s.hpd_sup); S["deciles"][nm] = Matrix{Float64}(s.deciles)
        end
        M = MStruct()
        for f in ("Mean", "Median", "Var", "deciles", "HPDinf", "HPDsup")   # Dynare's field order: Mean, Median, Var, deciles, HPDinf, HPDsup
            M[f] = S[f]
        end
        oo[field] = M
    end
    oo
end
