# Shock-decomposition figures (Dynare's graph_decomp.m): one figure per variable, stacked bars of the components (positive parts up from zero,
# negative parts down, in component order) with the smoothed line on top and the components named in a legend.

const _PARAMETER_SET_NAMES = Dict("calibration" => "Calibration", "prior_mode" => "Prior Mode", "prior_mean" => "Prior Mean",
    "posterior_mode" => "Posterior Mode", "posterior_mean" => "Posterior Mean", "posterior_median" => "Posterior Median", "mle_mode" => "MLE Mode")
_decomp_colors(n) = [_palette[mod1(k, length(_palette))] for k in 1:n]

"""
    decomposition_figures(d::DecompositionPlotData, m; fig_name="", forecast_length=nothing, vintage_title=false) -> Vector{Figure}

One `Figure` per plotted variable of `d` (those skipped by Dynare's 1e-6 rule are absent). Panel curves: one `:bar` curve per component (x = the period, bars centred on it), the smoothed series as a `:line` named "smoothed", a `:vline` at the forecast start when `forecast_length`
is given. Names follow Dynare: `<fname>_shock_decomposition_<var>[_<type>]<fig_name>_<parameter set>` (`_initval_decomposition_` for initial
conditions), window title `Historical shock decomposition<fig_name>[ type]: <var> (<Parameter set>).` The steady-state line / axis labels
(`steadystate`) are not drawn; the value is in `d.steady_state`.
"""
function decomposition_figures(d::DecompositionPlotData, m; fig_name::AbstractString="", forecast_length=nothing)
    is_icd = d.kind == "initial_conditions" || startswith(d.title, "Initial condition")
    pre = is_icd ? "Initial condition decomposition" : (startswith(d.title, "Shock decomposition") ? "Shock decomposition" : "Historical shock decomposition")
    rest = replace(d.title, r"^(Historical shock decomposition|Initial condition decomposition|Shock decomposition)" => "")
    long = string(isempty(fig_name) ? "" : " " * fig_name, rest)
    stem = replace(replace(replace(replace(replace(replace(long, "(given " => "", "(vintage " => ""), " " => "_"), "." => ""), "-" => ""), ")" => ""), "(" => "")
    mode1 = isempty(d.type) ? "" : "_" * d.type
    pset = get(_PARAMETER_SET_NAMES, d.parameter_set, "")
    figs = Figure[]
    for p in d.panels
        T = length(p.periods)
        x = Float64.(p.periods)                       # bars centred on the period, as Dynare's fill() spans [x(i), x(i+1)] + 1/2
        cols = _decomp_colors(length(p.labels))
        curves = Curve[Curve(x, p.contributions[i, :]; kind=:bar, color=cols[i], label=p.labels[i]) for i in eachindex(p.labels)]
        push!(curves, Curve(x, p.total; kind=:line, color="#000000", label="smoothed"))
        forecast_length === nothing || push!(curves, Curve([Float64(p.periods[1]) - 1 + T - forecast_length + 1.5]; kind=:vline, color="#d62a2a"))
        panel = Panel(p.name, curves; zeroline=true)
        name = string(m.name, is_icd ? "_initval_decomposition_" : "_shock_decomposition_", p.name, mode1, stem, "_", d.parameter_set)
        title = string(pre, long, replace(mode1, "_" => " "), ": ", p.name, " (", pset, ").")
        push!(figs, Figure(name, title, 1, 1, [panel]; legend=[(p.labels[i], cols[i]) for i in eachindex(p.labels)]))
    end
    figs
end

"""
    plot_shock_decomposition(m, dec, var_list=nothing; outdir=nothing, format="svg", backend=:svg, nograph=false, fig_name="", kw...) -> (figures, files)

Draw `decomposition_plot_data(dec, var_list; model=m, kw...)` (all its keywords: `use_shock_groups`, `type`, `diff`, `flip`, `init2shocks`, `screen_shocks`,
`plot_init_date`, ...) as `decomposition_figures`, written to `<fname>/graphs`. Returns the figures and the files.
"""
function plot_shock_decomposition(m, dec, var_list=nothing; outdir=nothing, format="svg", backend::Symbol=:svg, nograph::Bool=false,
                                  fig_name::AbstractString="", forecast_length=nothing, kw...)
    d = decomposition_plot_data(dec, var_list; model=m, kw...)
    figs = decomposition_figures(d, m; fig_name=fig_name, forecast_length=forecast_length)
    files = save_figures(figs, outdir === nothing ? graphs_dir(m) : outdir; format=format, backend=backend, nograph=nograph)
    figs, files
end
