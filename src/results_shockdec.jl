# oo_ of the `shock_decomposition` statement, including the decomposition of a forecast (`forecast_type=unconditional|conditional`)
# and the parameter sets of Bayesian estimation. Port of Dynare 7.2's matlab/shock_decomposition/shock_decomposition.m
# (Copyright (C) Dynare Team, GPL-3.0-or-later) and of pydyn's forecast_decomp.py.
#
# The decomposition runs the smoother of `calib_smoother` at the parameter set (`evaluate_smoother.m`) on a copy: like Dynare, the
# statement leaves oo_ as it is apart from `oo_.shock_decomposition` (or `oo_.forecast_shock_decomposition.<type>`),
# `oo_.shock_decomposition_info` and `oo_.gui.ran_shock_decomposition`.
#
# Refused: `with_epilogue` (epilogue_shock_decomposition), models with more than one lag of the state in the decomposition recursion
# (as the Model API's shock_decomposition).

"""
    forecast_decomposition(smoothed, shocks, G, H, kind, i_var, forecast_means, ys; init_state=0, controlled_exo=nothing)

The array `z` (n_endo, n_exo + 2, gend) of `shock_decomposition.m` with a forecast type. `smoothed` (T, n_endo) and `shocks` (T, n_exo)
are the smoother's (T may be 0 without data); `G`, `H` the transition from declaration order; `forecast_means[i]` the forecast path of
variable `i` (conditional paths start at the initial period, which is dropped); `ys` the steady state; `controlled_exo` the
(n_exo) paths of the controlled exogenous variables of a conditional forecast.
"""
function forecast_decomposition(smoothed::AbstractMatrix, shocks::AbstractMatrix, G, H, kind::AbstractString, i_var, forecast_means, ys;
                                init_state::Integer=0, controlled_exo=nothing)
    T, n = size(smoothed)
    nx = size(H, 2)
    i0 = first(i_var)
    L = length(forecast_means[i0])
    gend = T + L - (kind == "conditional" && T > 0 ? 1 : 0)
    eps = zeros(nx, gend)
    z = zeros(n, nx + 2, gend)
    T > 0 && (eps[:, 1:T] = shocks'; z[:, nx+2, 1:T] = smoothed')
    for i in i_var
        path = Float64.(vec(forecast_means[i]))
        kind == "conditional" && (path = path[2:end])
        length(path) == gend - T || merr("=: nonconformant arguments (op1 is 1x$(gend - T), op2 is 1x$(length(path)))")
        z[i, nx+2, T+1:end] = path .- ys[i]
    end
    if kind == "conditional"
        cp = length(controlled_exo[1])
        T + cp > gend && merr("index (_,$(T + cp)): out of bound $gend")
        for i in 1:nx
            c = Float64.(vec(controlled_exo[i]))
            length(c) == cp || merr("=: nonconformant arguments (op1 is 1x$cp, op2 is 1x$(length(c)))")
            eps[i, T+1:T+cp] = c
        end
    end
    for t in 1:gend
        t > 1 && (z[:, 1:nx, t] = G * z[:, 1:nx, t-1])
        t > init_state && (z[:, 1:nx, t] .+= H .* eps[:, t]')
        z[:, nx+1, t] = z[:, nx+2, t] - vec(sum(z[:, 1:nx, t]; dims=2))
    end
    z
end

function _sd_get(s, path::AbstractString, what::AbstractString)
    v = _getpath(s, path, nothing)
    v === nothing && merr("Reference to non-existent field '$(last(split(path, ".")))'.")
    v
end

function _sd_statement!(ws, st)
    m = _sync_model!(ws)
    O, oo = ws.options_, ws.oo_
    vars = String.(get_or(st, "symbol_list", String[]))
    isempty(vars) && (vars = m.endo_names[1:orig_endo_nbr(m)])
    for v in vars
        haskey(m.ie, v) || merr("$v is not an endogenous variable")
    end
    i_var = [m.ie[v] for v in unique(vars)]
    _optb(ws, "shock_decomp.with_epilogue", false) &&
        throw(Unsupported("shock_decomposition with_epilogue (epilogue_shock_decomposition)", "option"))
    ftype = String(_optf(ws, "shock_decomp.forecast_type", ""))
    pset = String(_optf(ws, "parameter_set", ""))
    if isempty(pset)
        pset = haskey(oo, "posterior_mean") ? "posterior_mean" : haskey(oo, "mle_mode") ? "mle_mode" :
               haskey(oo, "posterior") ? "posterior_mode" :
               merr("shock_decomposition: option parameter_set is not specified and posterior mode is not available")
    end
    uf = cf = nothing
    if ftype == "unconditional"
        if pset in ("calibration", "mle_mode")
            haskey(oo, "forecast") || merr(pset == "calibration" ? "Can't find unconditional forecasts in oo_.forecasts. Did you run the forecast command?" :
                                                                   "Can't find unconditional forecasts in oo_.forecasts. Did you run ML estimation with the forecast option?")
            uf = oo.forecast
        else
            haskey(oo, "MeanForecast") || merr("Can't find unconditional forecasts in oo_.MeanForecast. Did you run Bayesian estimation with the forecast option?")
            uf = oo.MeanForecast
        end
    elseif ftype == "conditional"
        haskey(oo, "conditional_forecast") || merr("Can't find conditional forecasts")
        cf = oo.conditional_forecast
    elseif !isempty(ftype)
        merr("shock_decomposition: unknown forecast_type $ftype")
    end
    # the smoother at the parameter set, without touching oo_
    saved = MStruct(); for k in keys(oo); saved[k] = deepcopy(oo[k]); end
    ps_saved = get(O, "parameter_set", nothing)
    O.parameter_set = pset
    res = nothing
    try
        _calib_smoother!(ws, nothing, String[], Dict{Symbol,Any}(:parameter_set => pset))
        res = ws.state[:last_smoother]
    finally
        for k in collect(keys(oo)); haskey(saved, k) || delete!(oo, k); end
        for k in keys(saved); oo[k] = saved[k]; end
        ps_saved === nothing || (O.parameter_set = ps_saved)
    end
    init_state = Int(_optf(ws, "shock_decomp.init_state", 0))
    n = length(m.endo_names)
    ys = res.SteadyState
    if isempty(ftype)
        T = size(res.smoothed, 1)
        z = zeros(n, size(res.H, 2) + 2, T)
        nx = size(res.H, 2)
        z[:, nx+2, :] = res.smoothed'
        for t in 1:T
            t > 1 && (z[:, 1:nx, t] = res.G * z[:, 1:nx, t-1])
            t > init_state && (z[:, 1:nx, t] .+= res.H .* res.shocks[t, :]')
            z[:, nx+1, t] = z[:, nx+2, t] - vec(sum(z[:, 1:nx, t]; dims=2))
        end
    else
        res.loglinear && throw(Unsupported("shock_decomposition of a forecast with loglinear (Dynare's forecast paths and dr.ys are logged: not validated)", "option"))
        present = !(isempty(String(_optf(ws, "datafile", "")))) || haskey(ws.state, :dataset)
        smoothed = present ? res.smoothed : zeros(0, n)
        shocks = present ? res.shocks : zeros(0, size(res.H, 2))
        means = Dict{Int,Vector{Float64}}()
        local ctrl
        if ftype == "unconditional"
            for i in i_var
                means[i] = vec(Float64.(_sd_get(uf, "Mean." * m.endo_names[i], "forecast")))
            end
            ctrl = nothing
        else
            for i in i_var
                means[i] = vec(Float64.(_sd_get(cf, "cond.Mean." * m.endo_names[i], "conditional forecast")))
            end
            ctrl = [vec(Float64.(_sd_get(cf, "controlled_exo_variables.Mean." * x, "conditional forecast"))) for x in m.exo_names]
        end
        z = forecast_decomposition(smoothed, shocks, res.G, res.H, ftype, i_var, means, ys; init_state=init_state, controlled_exo=ctrl)
    end
    if isempty(ftype)
        oo.shock_decomposition = z
        info = get(oo, "shock_decomposition_info", nothing)
        info isa MStruct || (info = MStruct(); oo.shock_decomposition_info = info)
        info.parameter_set = pset
    else
        fsd = get(oo, "forecast_shock_decomposition", nothing)
        fsd isa MStruct || (fsd = MStruct(); oo.forecast_shock_decomposition = fsd)
        fsd[ftype] = z
        info = get(oo, "shock_decomposition_info", nothing)
        info isa MStruct || (info = MStruct(); oo.shock_decomposition_info = info)
        f = get(info, "forecast", nothing)
        f isa MStruct || (f = MStruct(); info.forecast = f)
        t = MStruct(); t.parameter_set = pset
        f[ftype] = t
    end
    nog = _optb(ws, "no_graph.shock_decomposition", false)
    nog || (psi = MStruct(); psi.i_var = reshape(Float64.(i_var), :, 1); oo.plot_shock_decomposition_info = psi)
    _gui!(oo, "ran_shock_decomposition")
    nothing
end
