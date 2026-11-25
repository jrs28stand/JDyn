# Figures written by the statements of a .mod file, as Dynare does: `dynare(file; plots=true)` writes `<fname>/graphs/<stem>.<fmt>` for stoch_simul (IRFs),
# forecast, plot_conditional_forecast and rplot, unless the statement says `nograph`; `graph_format` is honoured ("svg" built in; "png"/"pdf"/"eps" need
# `using Plots` and the `plots_backend=:plots` keyword). Off by default: the package does not draw anything unless asked. The files written are listed in
# `ws.state[:figure_files]`.

function _statement_figures!(ws::DynareWorkspace, kind::AbstractString, st)
    m = ws.model
    nograph = _optb(ws, "nograph", false)
    figs = Figure[]
    if kind == "stoch_simul"
        _opti(ws, "irf", 40) > 0 || return nothing
        vl = String.(get_or(st, "symbol_list", String[]))
        r = irf_figures(m; var_list=isempty(vl) ? nothing : vl, rng=Random.MersenneTwister(0))
        append!(figs, to_figure(r))
    elseif kind == "rplot"
        es, xs = get(ws.oo_, "endo_simul", nothing), get(ws.oo_, "exo_simul", nothing)
        (es isa AbstractMatrix && xs isa AbstractMatrix) || return nothing
        vars = String.(get_or(st, "symbol_list", String[]))
        isempty(vars) && (vars = copy(m.endo_names))
        append!(figs, rplot_figures(m, es, xs, vars; rplottype=_opti(ws, "rplottype", 0)))
    elseif kind == "forecast" && haskey(ws.state, :forecast)
        append!(figs, forecast_figures(m, ws.state[:forecast]))
    elseif kind == "plot_conditional_forecast" && haskey(ws.state, :conditional_forecast)
        append!(figs, conditional_forecast_figures(m, ws.state[:conditional_forecast], String.(get_or(st, "symbol_list", String[])); periods=get(st, "periods", nothing)))
    end
    isempty(figs) && return nothing
    fmt = _graph_format_option(ws)
    files = save_figures(figs, graphs_dir(m); format=fmt, backend=_hook_backend(ws, fmt), nograph=nograph)
    append!(get!(ws.state, :figure_files, String[]), files)
    nothing
end

# ----------------------------------------------------------------------------------------------------- shared helpers
_plots_on(ws) = get(ws.state, :plots, false) === true

"""
options_.graph_format as a list of names: the preprocessor stores `graph_format = (pdf, eps)` as a struct with a `symbol_list`; nothing set gives SVG, the built-in format
(Dynare's default is eps; `none` writes no files).
"""
function _graph_format_option(ws)
    g = get(ws.options_, "graph_format", nothing)
    g isa MStruct && (g = get(g, "symbol_list", nothing))
    g === nothing && return "svg"
    g isa Cell && return String[String(x) for x in g.items]
    g isa AbstractString ? String(g) : String[String(x) for x in g]
end

"""
The writer for the statement's `graph_format`: `plots_backend` as given, but with the default `:svg` the Plots.jl writer is used when `using Plots` has loaded it and a
format other than SVG is asked for.
"""
function _hook_backend(ws, fmt)
    b = get(ws.state, :plots_backend, :svg)
    fl = [lowercase(String(x)) for x in (fmt isa AbstractString ? [fmt] : fmt)]
    b == :svg && any(f -> f != "svg" && f != "none", fl) && hasmethod(_render_figure_plots, Tuple{Figure,String,String}) && return :plots
    b
end

"Write figures into `<fname>/graphs` (or `sub`), honouring `nograph` and `graph_format`; the paths go to `ws.state[:figure_files]`."
function _emit_figures!(ws::DynareWorkspace, figs; sub::AbstractString="graphs", nograph=nothing)
    isempty(figs) && return String[]
    m = ws.model
    ng = nograph === nothing ? _optb(ws, "nograph", false) : nograph
    fmt = _graph_format_option(ws)
    dir = sub == "graphs" ? graphs_dir(m) : joinpath(m.mod_dir, m.name, sub)
    files = save_figures(collect(Figure, figs), dir; format=fmt, backend=_hook_backend(ws, fmt), nograph=ng)
    append!(get!(ws.state, :figure_files, String[]), files)
    files
end

"Run `f` with figure writing switched on (`plots=true`) or off for the duration; `nothing` leaves the workspace setting alone."
function _with_plots(f, ws::DynareWorkspace, plots)
    plots === nothing && return f()
    old = get(ws.state, :plots, false)
    ws.state[:plots] = plots
    try
        f()
    finally
        ws.state[:plots] = old
    end
end

# ----------------------------------------------------------------------------------------------------- estimation
"""
Figures of `estimation` at the stages Dynare draws them: `:prior` (plot_priors, default on, Bayesian only), `:mode` (mode_check), `:mcmc` (prior and posterior
densities, Brooks-Gelman diagnostics unless `nodiagnostic`), `:birf` (bayesian_irf, in `<fname>/Output`), `:series` (posterior distributions of smoothed variables,
shocks, ...), `:smoother` (smoothed shocks, observation errors and variables against the data). Called from results_estimation.jl, mcmc_stage.jl and posterior_stage.jl.
"""
function _fig_est!(ws::DynareWorkspace, stage::Symbol, est; kw...)
    _plots_on(ws) || return nothing
    m = ws.model
    kw = Dict{Symbol,Any}(kw)
    figs = Figure[]
    sub = "graphs"
    if stage == :prior
        _optb(ws, "plot_priors", true) && is_bayesian(est.prior) && append!(figs, prior_figures(est))
    elseif stage == :mode
        if _optb(ws, "mode_check.status", false) && !_optb(ws, "mh_posterior_mode_estimation", false)
            x = kw[:mode] isa ModeResult ? kw[:mode].x : kw[:mode]
            mk = Dict{Symbol,Any}()
            haskey(ws.options_, "mode_check.nolik") && (mk[:nolik] = _optb(ws, "mode_check.nolik", false))
            haskey(ws.options_, "mode_check.symmetric_plots") && (mk[:symmetric_plots] = _optb(ws, "mode_check.symmetric_plots", true))
            haskey(ws.options_, "mode_check.number_of_points") && (mk[:number_of_points] = _opti(ws, "mode_check.number_of_points", 20))
            haskey(ws.options_, "mode_check.neighbourhood_size") && (mk[:neighbourhood_size] = _optf(ws, "mode_check.neighbourhood_size", 0.5))
            append!(figs, mode_check_figures(est, x; mk...))
        end
    elseif stage == :mcmc
        mc = kw[:mc]
        if est.options["mh_replic"] > 0 || _est_flag(get(est.options, "load_mh_file", false))
            append!(figs, prior_posterior_figures(est, mc; mode=kw[:mode]))
            _optb(ws, "nodiagnostic", false) || append!(figs, brooks_gelman_figures(est, mc))
        end
    elseif stage == :birf
        append!(figs, bayesian_irf_figures(m, kw[:ir]))
        sub = "Output"
    elseif stage == :series
        r = kw[:stats]
        for (name3, sd) in (("SmoothedVariables", r.smoothed_variables), ("UpdatedVariables", r.updated_variables), ("SmoothedShocks", r.smoothed_shocks),
                            ("SmoothedObservationErrors", r.smoothed_measurement_errors), ("FilteredVariables", r.filtered_variables))
            sd === nothing || append!(figs, posterior_series_figures(m, sd, name3))
        end
    elseif stage == :smoother
        _optb(ws, "smoother", false) && kw[:res] !== nothing && append!(figs, smoother_figures(m, kw[:res], est.rawdata, est.varobs))
    end
    _emit_figures!(ws, figs; sub=sub)
    nothing
end

# ----------------------------------------------------------------------------------------------------- decompositions and OccBin
const _PLOT_STATEMENT_KINDS = Set(["shock_decomposition", "initial_condition_decomposition", "realtime_shock_decomposition",
                                   "squeeze_shock_decomposition", "plot_shock_decomposition", "occbin_graph"])

"A statement's options with Dynare's `shock_decomp.` / `plot_shock_decomp.` / ... prefixes removed and one-element lists unwrapped."
function _stmt_opts(st)
    out = Dict{String,Any}()
    for (k, v) in get_or(st, "options", Dict{String,Any}())
        out[String(last(split(String(k), "."; limit=2)))] = v isa AbstractVector && length(v) == 1 ? v[1] : v
        startswith(String(k), "no_graph.") && (out[String(k)] = v)
    end
    out
end
_num(v) = v isa AbstractString ? parse(Float64, v) : Float64(v)
_flagv(v) = v === true || (v isa Real && v != 0) || (v isa AbstractString && !isempty(v) && v != "0")

function _decomp_smoother_kw(ws, o)
    kw = Dict{Symbol,Any}()
    last_est = get(ws.state, :est_last, nothing)
    if last_est !== nothing
        kw[:estimation] = last_est.est
        last_est.mode === nothing || (kw[:mode] = last_est.mode)
        last_est.mc === nothing || (kw[:mcmc] = last_est.mc)
    end
    haskey(o, "parameter_set") && (kw[:parameter_set] = String(o["parameter_set"]))
    kw
end

function _plot_statement!(ws::DynareWorkspace, kind::AbstractString, st)
    m = ws.model
    o = _stmt_opts(st)
    vars = String.(get_or(st, "symbol_list", String[]))
    if kind == "shock_decomposition"
        res = smooth(m; _decomp_smoother_kw(ws, o)...)
        ws.state[:sd_res] = res
        ws.state[:sd] = shock_decomposition(m, res; init_state=haskey(o, "init_state") ? Int(_num(o["init_state"])) : nothing)
        ws.state[:icd] = initial_condition_decomposition(m, res)
    elseif kind == "realtime_shock_decomposition"
        kw = _decomp_smoother_kw(ws, o)
        ws.state[:rt] = realtime_shock_decomposition(m, nothing; kw...)
    elseif kind == "squeeze_shock_decomposition"
        haskey(ws.state, :sd) && (ws.state[:sd] = squeeze_shock_decomposition(ws.state[:sd], vars))
        haskey(ws.state, :rt) && (ws.state[:rt] = squeeze_shock_decomposition(ws.state[:rt], vars))
    elseif kind in ("plot_shock_decomposition", "initial_condition_decomposition")
        icd_stmt = kind == "initial_condition_decomposition"
        if icd_stmt
            kw = _decomp_smoother_kw(ws, o)
            ws.state[:icd] = initial_condition_decomposition(m, nothing; kw...)
        end
        realtime = Int(_num(get(o, "realtime", 0)))
        vintage = Int(_num(get(o, "vintage", 0)))
        dec = icd_stmt ? ws.state[:icd] : realtime != 0 ? (haskey(ws.state, :rt) ? select_vintage(ws.state[:rt], realtime, vintage) :
                  throw(SmootherError("plot_shock_decomposition(realtime=...) needs a realtime_shock_decomposition before it"))) :
              get(ws.state, :sd, nothing)
        dec === nothing && throw(SmootherError("$kind: run shock_decomposition first"))
        pk = Dict{Symbol,Any}()
        for k in ("type", "use_shock_groups", "expand", "init2shocks")
            haskey(o, k) && (pk[Symbol(k)] = String(o[k]))
        end
        for k in ("diff", "flip", "steadystate", "screen_shocks")
            haskey(o, k) && (pk[Symbol(k)] = _flagv(o[k]))
        end
        for k in ("plot_init_date", "plot_end_date")
            haskey(o, k) && (o[k] isa Real || (o[k] isa AbstractString && occursin(r"^\d+$", o[k]))) && (pk[Symbol(k)] = Int(_num(o[k])))
        end
        haskey(pk, :init2shocks) && (pk[:icd] = get(ws.state, :icd, nothing))
        nog = _flagv(get(o, "nograph", false)) || _flagv(get(o, "no_graph.plot_shock_decomposition", false))
        d = decomposition_plot_data(dec, isempty(vars) ? nothing : vars; model=m, pk...)
        fl = realtime in (1, 3) && haskey(ws.state, :rt) && vintage != 0 ? ws.state[:rt].forecast : nothing
        figs = decomposition_figures(d, m; fig_name=String(get(o, "fig_name", "")), forecast_length=fl)
        _emit_figures!(ws, figs; nograph=nog || _optb(ws, "nograph", false))
    elseif kind == "occbin_graph"
        r = get(ws.state, :occbin_result, nothing)
        r === nothing && throw(ModelError("occbin_graph: run occbin_solver first"))
        _emit_figures!(ws, occbin_graph_figures(r, m; variables=isempty(vars) ? nothing : vars))
    end
    nothing
end
