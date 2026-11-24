# Figures of the estimation commands, after Dynare 7.2: plot_priors / draw_prior_density, mode_check, PlotPosteriorDistributions, trace_plot,
# mcmc_diagnostics (Brooks-Gelman), PosteriorIRF_core2, pm3_core, plot_classical_smoother_results. Each `*_figures` function returns `Figure`s
# holding the arrays Dynare draws; `plot_*` wrappers also write them (`save_figures`, folder `<fname>/graphs` unless said otherwise).

# ---------------------------------------------------------------------------------------------- names
"""
    dynare_parameter_name(est, k) -> String

`get_the_name.m`: the name of estimated parameter `k` in Dynare's figures and structures: `SE_<shock>`, `SE_EOBS_<variable>`,
`CC_<a>_<b>`, `CC_EOBS_<a>_<b>`, or the parameter's name.
"""
function dynare_parameter_name(est, k::Integer)
    nm = est.names[k]
    m = est.model
    if (mt = match(r"^stderr (.+)$", nm)) !== nothing
        v = String(mt.captures[1])
        return haskey(m.ix, v) ? "SE_" * v : "SE_EOBS_" * v
    elseif (mt = match(r"^corr (.+), (.+)$", nm)) !== nothing
        a, b = String(mt.captures[1]), String(mt.captures[2])
        return string(haskey(m.ix, a) ? "CC_" : "CC_EOBS_", a, "_", b)
    end
    nm
end

"Figures of at most 9 panels in a layout that is the same for every figure (`subplot(nr, nc, i)` with the numbers of the full figure), as plot_priors.m."
function _uniform_figures(stem, title, panels::Vector{Panel}; legend=Tuple{String,String}[], number_title::Bool=false)
    nbplt, nr, nc = pltorg(length(panels))
    [Figure(string(stem, f), number_title ? string(title, " ", f) : title, nr, nc, panels[(f-1)*9+1:min(f * 9, end)], legend) for f in 1:nbplt]
end

# ---------------------------------------------------------------------------------------------- priors
"""
    prior_density(p::Prior, i; truncprior=1e-3, steps=200) -> (x, f)

`draw_prior_density.m`: 200 points between the `truncprior` and `1 - truncprior` quantiles (the bounds for a uniform prior) and the density there; for a
non-uniform prior whose mode is at an end of the grid, density above 10 is NaN.
"""
function prior_density(p::Prior, i::Integer; truncprior::Real=1e-3, steps::Int=200)
    sh, p3, p4, p6, p7 = p.shape[i], p.p3[i], p.p4[i], p.p6[i], p.p7[i]
    lo, hi = if sh == 1
        _betainv(truncprior, p6, p7) * (p4 - p3) + p3, _betainv(1 - truncprior, p6, p7) * (p4 - p3) + p3
    elseif sh == 2
        _gaminv(truncprior, p6, p7) + p3, _gaminv(1 - truncprior, p6, p7) + p3
    elseif sh == 3
        max(p3, _norminv(truncprior, p6, p7)), min(p4, _norminv(1 - truncprior, p6, p7))
    elseif sh == 4
        1 / sqrt(_gaminv(1 - 10 * truncprior, p7 / 2, 2 / p6)) + p3, 1 / sqrt(_gaminv(10 * truncprior, p7 / 2, 2 / p6)) + p3
    elseif sh == 5
        p6, p7
    elseif sh == 6
        1 / _gaminv(1 - 10 * truncprior, p7 / 2, 2 / p6) + p3, 1 / _gaminv(10 * truncprior, p7 / 2, 2 / p6) + p3
    elseif sh == 8
        p3 + _wblinv(truncprior, p7, p6), p3 + _wblinv(1 - truncprior, p7, p6)
    else
        throw(PriorError("draw_prior_density: unknown distribution shape $sh"))
    end
    x = collect(range(lo, hi; length=steps))
    f = sh == 5 ? fill(1 / (p7 - p6), steps) : [exp(logpdf_each(p, fill(xi, length(p)))[i]) for xi in x]
    if sh != 5
        k = argmax(replace(f, NaN => -Inf))
        (k == 1 || k == length(f)) && (f[f .> 10] .= NaN)
    end
    x, f
end

"Prior figures (`plot_priors`): one panel per estimated parameter, grey line; files `<fname>_Priors<k>`."
function prior_figures(est)
    panels = Panel[]
    for k in 1:est.n
        x, f = prior_density(est.prior, k)
        push!(panels, Panel(dynare_parameter_name(est, k), [Curve(x, f; color="#b3b3b3")]))
    end
    _uniform_figures(string(est.model.name, "_Priors"), "Priors", panels)
end

# ---------------------------------------------------------------------------------------------- mode_check
"""
    mode_check_data(est, x; neighbourhood_size=0.5, number_of_points=20, symmetric_plots=true, nolik=false) -> Vector{NamedTuple}

`mode_check.m`: for every estimated parameter the grid `z` around `x[k]` and the log posterior `y1` (NaN where the model cannot be solved) and, unless `nolik`,
the log-likelihood kernel `y2` (the posterior shifted by the log prior difference), all other parameters at `x`. Returns `(name, z, logpost, loglik, xmode)`.
"""
function mode_check_data(est, x::AbstractVector; neighbourhood_size=nothing, number_of_points=nothing, symmetric_plots=nothing, nolik=nothing)
    o = est.options
    ll = Float64(neighbourhood_size === nothing ? get(o, "mode_check.neighbourhood_size", 0.5) : neighbourhood_size)
    npts = Int(number_of_points === nothing ? get(o, "mode_check.number_of_points", 20) : number_of_points)
    sym = symmetric_plots === nothing ? _est_flag(get(o, "mode_check.symmetric_plots", true)) : symmetric_plots
    nolik = nolik === nothing ? _est_flag(get(o, "mode_check.nolik", false)) : nolik
    isinf(ll) && (sym = false)
    x = Float64.(x)
    tol = 1e-8
    out = NamedTuple{(:name, :z, :logpost, :loglik, :xmode),Tuple{String,Vector{Float64},Vector{Float64},Vector{Float64},Float64}}[]
    dy = nolik ? 0.0 : logpdf(est.prior, x)
    for k in 1:est.n
        xk, lb, ub = x[k], est.lb[k], est.ub[k]
        if xk != 0 && !isinf(lb) && !isinf(ub)
            l1 = max(lb, (1 - sign(xk) * ll) * xk); m1 = false
            l2 = min(ub, (1 + sign(xk) * ll) * xk)
        else
            up = isinf(lb) ? -1e-6 * 1e8 : lb
            lo = isinf(ub) ? -1e-6 * 1e8 : ub
            l1 = max(lo, -est.prior.p2[k]); m1 = false
            l2 = min(up, est.prior.p2[k])
        end
        bl = abs(xk - lb) < tol
        bu = abs(xk - ub) < tol
        if sym && !bl && !bu
            if l2 < (1 + ll) * xk
                l1 = xk - (l2 - xk); m1 = true
            end
            if !m1 && l1 > (1 - ll) * xk && xk + (xk - l1) < ub
                l2 = xk + (xk - l1)
            end
        end
        z1 = l1 == xk ? [xk] : collect(l1:((xk - l1) / (npts / 2)):xk)
        z2 = l2 == xk ? [xk] : collect(xk:((l2 - xk) / (npts / 2)):l2)
        z = sort(unique(vcat(z1, z2)))
        y1 = fill(NaN, length(z)); y2 = fill(NaN, length(z))
        for (i, zi) in enumerate(z)
            xx = copy(x); xx[k] = zi
            r = _dsge_likelihood(est, xx)
            if r[2] == 0
                y1[i] = r[1]
                nolik || (y2[i] = r[1] + logpdf(est.prior, xx) - dy)
            end
        end
        push!(out, (name=dynare_parameter_name(est, k), z=z, logpost=-y1, loglik=-y2, xmode=xk))
    end
    out
end

"Mode-check figures: log posterior (and log-likelihood kernel) around the mode, a vertical line at the mode, red points where the model could not be solved. Files `<fname>_CheckPlots<k>`."
function mode_check_figures(est, x; kw...)
    data = mode_check_data(est, x; kw...)
    nolik = all(d -> all(isnan, d.loglik), data)
    panels = Panel[]
    for d in data
        cv = Curve[Curve(d.z, d.logpost; color="#2a78d6", label="log-post")]
        nolik || push!(cv, Curve(d.z, d.loglik; color="#3a9d5d", label="log-lik kernel"))
        push!(cv, Curve([d.xmode]; kind=:vline, color="#00b8b8"))
        bad = findall(isnan, d.logpost)
        isempty(bad) || push!(cv, Curve(d.z[bad], fill(minimum(filter(!isnan, d.logpost); init=0.0), length(bad)); kind=:scatter, color="#d62a2a"))
        push!(panels, Panel(d.name, cv))
    end
    _uniform_figures(string(est.model.name, "_CheckPlots"), "Mode check plots", panels)
end

# ---------------------------------------------------------------------------------------------- posterior densities
"""
    prior_posterior_figures(est, mc; mode=nothing, drop=mc.drop) -> Vector{Figure}

`PlotPosteriorDistributions.m`: prior density (grey), posterior kernel density (the 512-point estimate of `posterior_moments`) and, when `mode` is given, a
dashed vertical segment at the posterior mode up to `1.1 max`; 9 panels per figure on a 3 x 3 grid; files `<fname>_PriorsAndPosteriors<k>`.
"""
function prior_posterior_figures(est, mc; mode=nothing, drop=mc.drop)
    X, _ = pooled_draws(mc; drop=drop)
    st = series_stats(X, Float64(est.options["mh_conf_sig"]); density=true)
    xm = mode === nothing ? nothing : Float64.(mode isa ModeResult ? mode.x : mode)
    panels = Panel[]
    for k in 1:est.n
        x2, f2 = prior_density(est.prior, k)
        d = st.density[:, :, k]
        x1, f1 = d[:, 1], d[:, 2]
        top = maximum(filter(!isnan, vcat(f1, f2)); init=1.0)
        cv = Curve[Curve(x2, f2; color="#b3b3b3", label="prior"), Curve(x1, f1; color="#000000", label="posterior")]
        xm === nothing || push!(cv, Curve([xm[k]]; kind=:vline, color="#3a9d5d"))
        push!(panels, Panel(dynare_parameter_name(est, k), cv))
    end
    figs = _uniform_figures(string(est.model.name, "_PriorsAndPosteriors"), "Priors and posteriors", panels)
    [Figure(f.name, f.title, 3, 3, f.panels, f.legend) for f in figs]
end

# ---------------------------------------------------------------------------------------------- trace plots
"""
    trace_figures(est, mc; chains=nothing, ma=200) -> Vector{Figure}

`generate_trace_plots` / `trace_plot.m`: for the posterior density (`TracePlot_Posterior_blck_<k>`) and every estimated parameter (`TracePlot_<name>_blck_<k>`), the
draws of the chain `k` (all of them, burn-in included) in grey with their centred `2ma+1` moving average in black (one chain), or the draws of several chains
(`blocks`: `TracePlot_<name>_blck_1_2`). Figure titles as Dynare's `Trace plot for ...`.
"""
function trace_figures(est, mc; chains=nothing, ma::Integer=200)
    nb, T, n = size(mc.draws)
    chains = chains === nothing ? collect(1:nb) : collect(Int, chains)
    maximum(chains) <= nb || throw(ArgumentError("chain number is bigger than existing number of chains"))
    m = est.model
    series = Pair{String,Matrix{Float64}}["Posterior"=>reshape(Float64.(mc.log_posterior[chains, :])', T, length(chains))]
    for k in 1:n
        push!(series, dynare_parameter_name(est, k) => Float64.(mc.draws[chains, :, k]'))
    end
    figs = Figure[]
    suffix = length(chains) == 1 ? string("_blck_", chains[1]) : string("_blck_", join(chains, "_"))
    for (name, D) in series
        what = name == "Posterior" ? "the posterior density" : startswith(name, "SE_") ? "the standard deviation of structural shock " * name[4:end] :
               startswith(name, "CC_") ? "the correlation between " * name[4:end] : "parameter " * name
        ttl = string("Trace plot for ", what)
        t = collect(1.0:T)
        if length(chains) == 1
            nb > 1 && (ttl *= string(" (block number ", chains[1], ")."))
            y = D[:, 1]
            mav = fill(NaN, T)
            for s in ma+1:T-ma
                mav[s] = sum(@view y[s-ma:s+ma]) / (2ma + 1)
            end
            cv = [Curve(t, y; color="#b3b3b3", label="MCMC draw"), Curve(t, mav; color="#000000", label="$ma period moving average")]
        else
            cv = [Curve(t, D[:, j]; label="Chain $(chains[j])") for j in eachindex(chains)]
        end
        push!(figs, Figure(string("TracePlot_", name, suffix), ttl, 1, 1, [Panel(name, cv)]))
    end
    figs
end

# ---------------------------------------------------------------------------------------------- convergence diagnostics
"""
    brooks_gelman_figures(est, mc; plotrows=3, drop=mc.drop) -> Vector{Figure}

The figures of `mcmc_diagnostics`: `<fname>_udiag<k>` with `plotrows` x 3 panels (parameter by parameter, the Interval / m2 / m3 criteria, pooled in blue, within in red; a final
partial figure, 3 x 1 for one parameter), and `<fname>_mdiag` (3 x 1, Interval, m2, m3 of the posterior kernel). Empty when Dynare computes none (one chain, or at most 2000 draws).
"""
function brooks_gelman_figures(est, mc; plotrows=nothing, drop=mc.drop)
    bg = brooks_gelman(mc; drop=drop)
    bg.univariate === nothing && return Figure[]
    rows = Int(plotrows === nothing ? get(est.options, "convergence.brooksgelman.plotrows", 3) : plotrows)
    n = size(bg.univariate, 3)
    x = Float64.(bg.x)
    crit = ("Interval", "m2", "m3")
    pan(name, k, S) = Panel(name, [Curve(x, S[:, 2k-1]; color="#2a78d6", label="pooled"), Curve(x, S[:, 2k]; color="#d62a2a", label="within")])
    figs = Figure[]
    panels = Panel[]
    for j in 1:n, c in 1:3
        push!(panels, pan(string(dynare_parameter_name(est, j), " (", crit[c], ")"), c, bg.univariate[:, :, j]))
    end
    per = 3 * rows
    pages = fld(n, rows)
    for i in 1:pages
        push!(figs, Figure(string(est.model.name, "_udiag", i), "MCMC univariate convergence diagnostic (Brooks and Gelman,1998)", rows, 3, panels[(i-1)*per+1:i*per]))
    end
    reste = n - pages * rows
    if reste > 0
        nr, nc = reste == 1 ? (3, 1) : (rows, 3)
        push!(figs, Figure(string(est.model.name, "_udiag", pages + 1), "MCMC univariate convergence diagnostic (Brooks and Gelman, 1998)", nr, nc, panels[pages*per+1:end]))
    end
    if bg.multivariate !== nothing
        push!(figs, Figure(string(est.model.name, "_mdiag"), "Multivariate convergence diagnostic", 3, 1,
                           [pan(crit[c], c, bg.multivariate) for c in 1:3]))
    end
    figs
end

# ---------------------------------------------------------------------------------------------- posterior IRFs and series
"""
    bayesian_irf_figures(m, ir::PosteriorIRF; threshold=1e-10, relative=false, periods=nothing) -> Vector{Figure}

`PosteriorIRF_core2.m`: for each shock, the panels of the variables whose posterior-mean response reaches `threshold`, at most 9 per figure on a 3 x 3 grid: the HPD
band (`:band`, drawn only if the HPD limits are not NaN) and the mean (black). Files `<fname>_Bayesian_IRF_<shock>_<k>` (Dynare writes them to `<fname>/Output`),
titles `Orthogonalized shock to <shock>` / `Relative response to orthogonalized shock to <shock>`.
"""
function bayesian_irf_figures(m, ir::PosteriorIRF; threshold::Real=1e-10, relative::Bool=false)
    figs = Figure[]
    for s in ir.shocks
        panels = Panel[]
        for v in ir.variables
            k = findfirst(==("$(v)_$(s)"), ir.names)
            k === nothing && continue
            mean_ = ir.mean[:, k]
            maximum(abs, mean_) >= threshold || continue
            t = collect(1.0:length(mean_))
            cv = Curve[]
            all(isfinite, ir.hpd_inf[:, k]) && push!(cv, Curve(t, ir.hpd_inf[:, k]; kind=:band, y2=ir.hpd_sup[:, k], color="#999999", label="HPD"))
            push!(cv, Curve(t, mean_; color="#000000", label="mean"))
            push!(panels, Panel(v, cv))
        end
        for (f, i0) in enumerate(1:9:length(panels))
            push!(figs, Figure(string(m.name, "_Bayesian_IRF_", s, "_", f), string(relative ? "Relative response to orthogonalized shock to " : "Orthogonalized shock to ", s),
                               3, 3, panels[i0:min(i0 + 8, end)]))
        end
    end
    figs
end

"""
    posterior_series_figures(m, sd::SeriesDistribution, name3; first_period=1) -> Vector{Figure}

`pm3_core.m`: the nine deciles (green) and the mean (black) of each series of a posterior distribution (smoothed variables, shocks, ...), 9 panels per figure on a 3 x 3
grid; series whose mean is NaN everywhere get no panel. Files `<fname>_<name3>_<last series of the figure>`; titles `<name3> <k>`.
"""
function posterior_series_figures(m, sd::SeriesDistribution, name3::AbstractString)
    figs = Figure[]
    panels = Panel[]
    last_name = ""
    nvar = length(sd.names)
    for (i, v) in enumerate(sd.names)
        if !all(isnan, sd.mean[:, i])
            t = collect(1.0:size(sd.mean, 1))
            cv = Curve[Curve(t, sd.deciles[:, q, i]; color="#3a9d5d") for q in 1:9]
            push!(cv, Curve(t, sd.mean[:, i]; color="#000000", label="mean"))
            push!(panels, Panel(v, cv))
        end
        if length(panels) == 9 || i == nvar
            isempty(panels) || push!(figs, Figure(string(m.name, "_", name3, "_", v), string(name3, " ", length(figs) + 1), 3, 3, panels))
            panels = Panel[]
        end
    end
    figs
end

# ---------------------------------------------------------------------------------------------- smoother
"""
    smoother_figures(m, res::SmootherResult, rawdata, varobs; me_plot_tol=1e-3) -> Vector{Figure}

`plot_classical_smoother_results.m`: `<fname>_SmoothedShocks<k>` (zero line and smoothed shock, `pltorg(n_exo)` panels), `<fname>_SmoothedObservationErrors<k>` (only
variables whose smoothed measurement error exceeds `me_plot_tol`), `<fname>_HistoricalAndSmoothedVariables<k>` (smoothed observed variable in red, data dashed black).
`rawdata` is (periods x observed variables), as in the data set.
"""
function smoother_figures(m, res::SmootherResult, rawdata::AbstractMatrix, varobs; me_plot_tol::Real=1e-3)
    T = size(rawdata, 1)
    t = collect(1.0:T)
    figs = Figure[]
    panels = [Panel(e, [Curve(t, zeros(T); color="#d62a2a"), Curve(t, res.SmoothedShocks[e]; color="#000000")]) for e in m.exo_names]
    append!(figs, _uniform_figures(string(m.name, "_SmoothedShocks"), "Smoothed shocks", panels))
    me = [v for v in varobs if haskey(res.SmoothedMeasurementErrors, v) && maximum(abs, res.SmoothedMeasurementErrors[v]) > me_plot_tol]
    if !isempty(me)
        panels = [Panel(v, [Curve(t, zeros(T); color="#d62a2a"), Curve(t, res.SmoothedMeasurementErrors[v]; color="#000000")]) for v in me]
        append!(figs, _uniform_figures(string(m.name, "_SmoothedObservationErrors"), "Smoothed observation errors", panels))
    end
    panels = [Panel(v, [Curve(t, res.SmoothedVariables[v]; color="#d62a2a"), Curve(t, rawdata[:, k]; kind=:dashed, color="#000000")]) for (k, v) in enumerate(varobs)]
    append!(figs, _uniform_figures(string(m.name, "_HistoricalAndSmoothedVariables"), "Historical and smoothed variables", panels))
    figs
end
