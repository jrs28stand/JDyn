# Forecast, conditional forecast and perfect-foresight (rplot) figures, after Dynare 7.2's +forecasts/graph.m, +conditional_forecasts/plot.m and rplot.m.

"""
    forecast_figures(m, r::ForecastResult; var_list=nothing) -> Vector{Figure}

`forecasts.graph`: mean forecast and the two HPD limits (all blue solid lines) of each variable, 12 panels per figure on a 3 x 4 grid; files `forcst<k>`, titles
`Forecasts (I)` for the first figure and `Forecasts (k)` after it (Dynare's naming). With measurement errors in the results a second set `forcst_ME<k>` ("Forecasts including ME")
shows the mean with the HPD bands including them.
"""
function forecast_figures(m, r::ForecastResult; var_list=nothing)
    vars = var_list === nothing ? r.var_list : String.(var_list)
    function build(stem, titlestem, vars, bands)
        figs = Figure[]
        panels = Panel[]
        for (j, v) in enumerate(vars)
            t = collect(1.0:length(r.Mean[v]))
            lo, hi = bands(v)
            push!(panels, Panel(v, [Curve(t, r.Mean[v]; color="#2a78d6"), Curve(t, lo; color="#2a78d6"), Curve(t, hi; color="#2a78d6")]))
            if length(panels) == 12 || j == length(vars)
                n = length(figs) + 1
                push!(figs, Figure(string(stem, n), n == 1 ? string(titlestem, " (I)") : string(titlestem, " (", n, ")"), 3, 4, panels))
                panels = Panel[]
            end
        end
        figs
    end
    figs = build("forcst", "Forecasts", vars, v -> (r.HPDinf[v], r.HPDsup[v]))
    if !isempty(r.HPDinf_ME)
        mev = [v for v in vars if haskey(r.HPDinf_ME, v)]
        isempty(mev) || append!(figs, build("forcst_ME", "Forecasts including ME", mev, v -> (r.HPDinf_ME[v], r.HPDsup_ME[v])))
    end
    figs
end

"""
    conditional_forecast_figures(m, cf::ConditionalForecastResult, variables=nothing; periods=nothing) -> Vector{Figure}

`conditional_forecasts.plot`: one figure per variable: the conditional band (grey area between the confidence limits) and mean (solid black), the unconditional mean (dashed)
and its limits (thin dashed); files `Conditional_forecast_<title with _>_<variable>`, titles `Conditional forecast (<title>): <variable>.`.
"""
function conditional_forecast_figures(m, cf::ConditionalForecastResult, variables=nothing; periods=nothing)
    vars = variables === nothing ? cf.constrained_names : String.(collect(variables))
    isempty(vars) && throw(ArgumentError("plot_conditional_forecast: no variables"))
    nf = length(cf.cond_mean[vars[1]])
    P = periods === nothing ? nf : min(Int(periods) + 1, nf)
    t = collect(1.0:P)
    figs = Figure[]
    for v in vars
        haskey(cf.cond_mean, v) || throw(ArgumentError("plot_conditional_forecast: $v is not an endogenous variable of the conditional forecast"))
        c, u = cf.cond_ci[v][:, 1:P], cf.uncond_ci[v][:, 1:P]
        cv = [Curve(t, c[1, :]; kind=:band, y2=c[2, :], color="#bbbbbb", label="conditional interval"),
              Curve(t, cf.cond_mean[v][1:P]; color="#000000", label="conditional mean"),
              Curve(t, cf.uncond_mean[v][1:P]; kind=:dashed, color="#000000", label="unconditional mean"),
              Curve(t, u[1, :]; kind=:dashed, color="#555555"), Curve(t, u[2, :]; kind=:dashed, color="#555555")]
        push!(figs, Figure(string("Conditional_forecast_", replace(strip(cf.title), " " => "_"), "_", v),
                           string("Conditional forecast (", cf.title, "): ", v, "."), 1, 1, [Panel(v, cv)]))
    end
    figs
end

"""
    rplot_figures(m, endo_simul, exo_simul, vars; rplottype=0, smpl=0, steady_state=nothing, exo_steady_state=nothing) -> Vector{Figure}

`rplot`: `endo_simul` (n_endo x periods + maximum_lag + lead, Dynare's `oo_.endo_simul`) and `exo_simul`. `rplottype` 0: all variables in one panel (`Plot of a b`, legend
when several, file `SimulatedTrajectory_<first>`); 1: one figure per variable; 2: one figure with `max(1, fld(n, 4))` rows of subplots with the steady state as a dotted line. x is the
period (`1 - maximum_lag` ...). `smpl` = `[first, last]` restricts the sample as `options_.smpl`.
"""
function rplot_figures(m, endo_simul::AbstractMatrix, exo_simul::AbstractMatrix, vars; rplottype::Int=0, smpl=0, steady_state=nothing, exo_steady_state=nothing,
                       maximum_lag::Int=1)
    vars = String.(collect(vars))
    ncol = size(endo_simul, 2)
    ix = collect(1-maximum_lag:ncol-maximum_lag)
    rows = Vector{Vector{Float64}}()
    for v in vars
        if haskey(m.ie, v)
            push!(rows, Float64.(endo_simul[m.ie[v], :]))
        elseif haskey(m.ix, v)
            push!(rows, Float64.(exo_simul[:, m.ix[v]]))
        else
            throw(ArgumentError("rplot: One of the variables specified does not exist"))
        end
    end
    i = smpl == 0 ? collect(max(1, maximum_lag):ncol) : collect(smpl[1]+maximum_lag:smpl[2]+maximum_lag)
    i = i[i .<= min(ncol, minimum(length, rows))]
    x = Float64.(ix[i])
    tr = [r[i] for r in rows]
    if rplottype == 0
        [Figure("SimulatedTrajectory_" * vars[1], "Simulated Trajectory", 1, 1,
                [Panel("Plot of " * join(vars, " "), [Curve(x, tr[k]; label=vars[k]) for k in eachindex(vars)])])]
    elseif rplottype == 1
        [Figure("SimulatedTrajectory_" * v, "Simulated Trajectory", 1, 1, [Panel("Plot of " * v, [Curve(x, tr[k])])]) for (k, v) in enumerate(vars)]
    elseif rplottype == 2
        nl = max(1, fld(length(vars), 4))
        nc = cld(length(vars), nl)
        panels = Panel[]
        for (k, v) in enumerate(vars)
            ss = haskey(m.ie, v) ? (steady_state === nothing ? NaN : steady_state[m.ie[v]]) : (exo_steady_state === nothing ? NaN : exo_steady_state[m.ix[v]])
            push!(panels, Panel("Plot of " * v, [Curve(x, tr[k]), Curve(x, fill(ss, length(x)); kind=:dotted, color="#d62a2a")]))
        end
        [Figure("SimulatedTrajectory_" * vars[1], "Simulated Trajectory", nl, nc, panels)]
    else
        throw(ArgumentError("rplot: rplottype must be 0, 1 or 2"))
    end
end
