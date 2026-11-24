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
    fmt = get(ws.options_, "graph_format", nothing)
    fmt = fmt === nothing ? "svg" : fmt isa Cell ? [String(x) for x in fmt.items] : fmt
    files = save_figures(figs, graphs_dir(m); format=fmt, backend=get(ws.state, :plots_backend, :svg), nograph=nograph)
    append!(get!(ws.state, :figure_files, String[]), files)
    nothing
end
