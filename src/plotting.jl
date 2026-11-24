# Impulse-response figures for stoch_simul, after Dynare 7.2 (matlab/stochastic_solver/stoch_simul.m, pltorg.m).
#
# What is reproduced is the *content* of Dynare's figures, not their look: one figure per shock, the shock being
# a one-standard-deviation orthogonalised (lower Cholesky) innovation; responses below `irf_plot_threshold` are
# left out; at most 3 x 3 panels per figure, more figures when needed; Dynare's figure titles and file names.
# The plotted data are returned (`IRFResult`) and can be written as a tidy long-format CSV for other tools.
#
# Rendering is optional. The built-in writer produces SVG without any dependency; PNG and PDF (and the
# `backend = :plots` path) need Plots.jl, loaded by the user, which activates the extension in ext/.

"""
    pltorg(n) -> (nbplt, nr, nc, lr, lc, nstar)

Dynare's `pltorg`: for `n` panels, the number of figures, the rows and columns of a full figure, the rows and
columns of the last figure, and the panels per full figure (9).
"""
function pltorg(n::Integer)
    nstar = 9
    n <= 0 && return (0, 0, 0, 0, 0, nstar)
    small = Dict(1 => (1, 1), 2 => (2, 1), 3 => (3, 1), 4 => (2, 2), 5 => (3, 2), 6 => (3, 2), 7 => (3, 3), 8 => (3, 3), 9 => (3, 3))
    n <= 9 && return (1, small[n]..., 0, 0, nstar)
    nbplt = cld(n, nstar)
    rest = n - (nbplt - 1) * nstar          # in 1:9; a full last figure is 3 x 3 as in Dynare
    lr, lc = small[rest]
    (nbplt, 3, 3, lr, lc, nstar)
end

"One figure: the responses (periods x variables) of `variables` to `shock`, laid out on `nrows` x `ncols` panels."
struct IRFFigure
    shock::String
    index::Int                 # 1-based figure number for this shock
    nfigs::Int                 # number of figures for this shock
    nrows::Int
    ncols::Int
    title::String              # Dynare's figure name, e.g. "Orthogonalized shock to e"
    name::String               # Dynare's file stem, e.g. "example1_IRF_e" or "example1_IRF_e2"
    variables::Vector{String}  # panel titles, row-major
    series::Matrix{Float64}    # periods x length(variables)
end

"""
    IRFResult

The content of the IRF figures. `figures` are in Dynare's order (shock by shock); `irfs` holds every computed
response as Dynare's `oo_.irfs` does (`"y_e" => vector`), including those not plotted; `skipped` lists the
`(variable, shock)` pairs below the threshold and `skipped_shocks` the shocks with (almost) zero standard
deviation, for which Dynare draws nothing. `files` is filled by `plot_irfs`.
"""
struct IRFResult
    model::String
    periods::Int
    order::Int                 # approximation order of the responses
    threshold::Float64
    relative::Bool
    figures::Vector{IRFFigure}
    irfs::Dict{String,Vector{Float64}}
    all_series::Vector{Tuple{String,String,Vector{Float64}}}   # (shock, variable, response), every computed one
    skipped::Vector{Tuple{String,String}}
    skipped_shocks::Vector{String}
    files::Vector{String}
end

# ---------------------------------------------------------------------------------------------- content
_lower_cholesky(S, o=Dict{String,Any}()) = get_lower_cholesky_covariance(S; add_tiny=Float64(get(o, "add_tiny_number_to_cholesky", 1e-14)))

_opt(o, k, default) = (v = get(o, k, default); v === nothing ? default : v)
_truthy(v) = v === true || (v isa Real && v != 0)

"Names listed in an option (`irf_shocks = (e u)`), whichever way the preprocessor wrote them."
function _name_list(v)
    v === nothing && return String[]
    v isa AbstractDict && return _name_list(get(v, "symbol_list", nothing))
    v isa AbstractString && return String.(split(replace(v, r"[(),]" => " ")))
    String[x isa AbstractString ? x : string(x) for x in v]
end

_stoch_simul_options(m) = get(m.options, "stoch_simul", Dict{String,Any}())

"""
    irf_figures(m, sol = first_order(m); var_list, shocks, periods, threshold, relative) -> IRFResult

First-order impulse responses to one-standard-deviation orthogonalised shocks, selected and arranged as
Dynare's `stoch_simul` does. Defaults come from the model's `stoch_simul` options (`irf`, `irf_plot_threshold` (stored by the preprocessor as `impulse_responses.plot_threshold`),
`irf_shocks`, `relative_irf`, `order`, `pruning`, `drop`, `replic`) and its variable list. At order 2 and 3 the responses
are Dynare's generalised IRFs (simulation averages with `replic` replications; pass `rng` to fix the draws).
"""
function irf_figures(m, sol=first_order(m); var_list=nothing, shocks=nothing, periods=nothing, threshold=nothing,
                     relative=nothing, order=nothing, pruning=nothing, drop=nothing, replic=nothing, rng=nothing)
    o = _stoch_simul_options(m)
    ord = order === nothing ? (try Int(_opt(o, "order", 1)) catch; 1 end) : Int(order)
    pruning = pruning === nothing ? _truthy(_opt(o, "pruning", false)) : pruning
    dr = ord > 1 ? k_order(m; order=ord, pruning=pruning) : nothing
    rng = rng === nothing ? Random.MersenneTwister(0) : rng
    periods = Int(periods === nothing ? _opt(o, "irf", 40) : periods)
    threshold = Float64(threshold === nothing ? _opt(o, "impulse_responses.plot_threshold", 1e-10) : threshold)
    relative = relative === nothing ? _truthy(_opt(o, "relative_irf", false)) : relative
    shock_names = shocks === nothing ? _name_list(get(o, "irf_shocks", nothing)) : String.(shocks)
    isempty(shock_names) && (shock_names = copy(m.exo_names))
    for s in shock_names
        s in m.exo_names || throw(ArgumentError("$s is not an exogenous variable"))
    end
    ishocks = unique([findfirst(==(s), m.exo_names) for s in shock_names])
    ivars = _var_indices(m, var_list)
    vnames = m.endo_names[ivars]
    L = _lower_cholesky(m.Sigma_e, o)
    irfs = Dict{String,Vector{Float64}}()
    all_series = Tuple{String,String,Vector{Float64}}[]
    figures = IRFFigure[]
    skipped = Tuple{String,String}[]
    skipped_shocks = String[]
    for i in ishocks
        sname = m.exo_names[i]
        if !(L[i, i] > 5e-7)                       # Dynare's test on the shock's standard deviation
            push!(skipped_shocks, sname)
            continue
        end
        if ord == 1
            y = irf(sol, L[:, i]; periods=periods)     # periods x endo
            relative && (y = 100 .* y ./ L[i, i])
        else                                           # generalised IRFs from simulations (Dynare's irf.m at order > 1)
            impulse = relative ? L[:, i] ./ L[i, i] ./ 100 : L[:, i]
            y = irf(dr, impulse; Sigma_e=m.Sigma_e, periods=periods, order=ord, pruning=pruning,
                    drop=Int(drop === nothing ? _opt(o, "drop", 100) : drop), replic=Int(replic === nothing ? _opt(o, "replic", 50) : replic), rng=rng)
        end
        keep = Int[]
        for (j, v) in zip(ivars, vnames)
            irfs[string(v, "_", sname)] = y[:, j]
            push!(all_series, (sname, String(v), y[:, j]))
            if maximum(abs, y[:, j]) >= threshold
                push!(keep, j)
            else
                push!(skipped, (v, sname))
            end
        end
        nbplt, nr, nc, lr, lc, nstar = pltorg(length(keep))
        for f in 1:nbplt
            cols = keep[(f-1)*nstar+1:min(f * nstar, length(keep))]
            last = f == nbplt
            r, c = nbplt == 1 ? (nr, nc) : last ? (lr, lc) : (nr, nc)
            title = string(relative ? "Relative response to orthogonalized shock to " : "Orthogonalized shock to ", sname,
                           nbplt == 1 ? "" : " figure $f")
            stem = string(m.name, "_IRF_", sname, nbplt == 1 ? "" : string(f))
            push!(figures, IRFFigure(sname, f, nbplt, r, c, title, stem, String.(m.endo_names[cols]), y[:, cols]))
        end
    end
    IRFResult(m.name, periods, ord, threshold, relative, figures, irfs, all_series, skipped, skipped_shocks, String[])
end

# ---------------------------------------------------------------------------------------------- tidy export
"""
    irf_table(res::IRFResult; all = false)

The responses in long format, one row per shock, variable and period, as a vector of named tuples with
`shock, variable, period, value, order`. By default only the plotted series; `all = true` adds the ones below
the threshold. Periods count from 1, the impact period, as in Dynare's figures.
"""
function irf_table(res::IRFResult; all::Bool=false)
    rows = NamedTuple{(:shock, :variable, :period, :value, :order),Tuple{String,String,Int,Float64,Int}}[]
    if all
        for (s, v, y) in res.all_series, t in eachindex(y)
            push!(rows, (shock=s, variable=v, period=t, value=y[t], order=res.order))
        end
    else
        for fig in res.figures, (j, v) in enumerate(fig.variables), t in 1:size(fig.series, 1)
            push!(rows, (shock=fig.shock, variable=v, period=t, value=fig.series[t, j], order=res.order))
        end
    end
    rows
end

_csv_field(x::AbstractString) = occursin(r"[\",\n]", x) ? string('"', replace(x, "\"" => "\"\""), '"') : String(x)
_csv_field(x::Integer) = string(x)
_csv_field(x::Real) = isfinite(x) ? repr(Float64(x)) : "NA"

"Write a table of named tuples (e.g. `irf_table(res)`) as CSV with a header row; returns `path`."
function write_csv(path::AbstractString, rows)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        isempty(rows) && return
        println(io, join(string.(keys(first(rows))), ","))
        for r in rows
            println(io, join((_csv_field(x) for x in values(r)), ","))
        end
    end
    path
end

# ---------------------------------------------------------------------------------------------- SVG writer
function _nice_ticks(lo, hi; n=5)
    span = hi - lo
    span > 0 || return [lo]
    raw = span / n
    mag = 10.0^floor(log10(raw))
    step = mag * (raw / mag < 1.5 ? 1 : raw / mag < 3.5 ? 2 : raw / mag < 7.5 ? 5 : 10)
    collect(ceil(lo / step) * step:step:hi+step * 1e-9) .+ 0.0
end
_fmt_tick(x) = (s = @sprintf("%.4g", abs(x) < 1e-12 ? 0.0 : x); s)
_xml(s) = replace(replace(replace(String(s), "&" => "&amp;"), "<" => "&lt;"), ">" => "&gt;")

"Render one `IRFFigure` as a self-contained SVG string (no dependencies)."
function svg_figure(fig::IRFFigure; panel_w=320, panel_h=210)
    W, H = fig.ncols * panel_w, fig.nrows * panel_h + 34
    ink, muted, grid, line, zero = "#2b2b29", "#6b6b67", "#e6e5e1", "#2a78d6", "#9a9994"
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="Helvetica, Arial, sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="#ffffff"/>""")
    println(io, """<text x="$(W / 2)" y="22" text-anchor="middle" font-size="14" fill="$ink">$(_xml(fig.title))</text>""")
    T = size(fig.series, 1)
    for (k, v) in enumerate(fig.variables)
        r, c = fld(k - 1, fig.ncols), mod(k - 1, fig.ncols)
        x0, y0 = c * panel_w + 46, 34 + r * panel_h + 30        # plot area origin (top-left)
        pw, ph = panel_w - 66, panel_h - 66
        y = fig.series[:, k]
        lo, hi = min(minimum(y), 0.0), max(maximum(y), 0.0)
        lo == hi && (hi = lo + 1)
        pad = 0.05 * (hi - lo); lo -= pad; hi += pad
        sx(t) = x0 + (T == 1 ? 0.0 : (t - 1) / (T - 1) * pw)
        sy(val) = y0 + ph - (val - lo) / (hi - lo) * ph
        println(io, """<text x="$(x0 + pw / 2)" y="$(y0 - 10)" text-anchor="middle" font-size="12" fill="$ink">$(_xml(v))</text>""")
        for tv in _nice_ticks(lo, hi)
            println(io, """<line x1="$x0" x2="$(x0 + pw)" y1="$(sy(tv))" y2="$(sy(tv))" stroke="$grid" stroke-width="1"/>""")
            println(io, """<text x="$(x0 - 5)" y="$(sy(tv) + 3.5)" text-anchor="end" font-size="9" fill="$muted">$(_fmt_tick(tv))</text>""")
        end
        xt = T <= 1 ? [1.0] : _nice_ticks(1, T; n=5)
        for tv in xt
            println(io, """<text x="$(sx(tv))" y="$(y0 + ph + 14)" text-anchor="middle" font-size="9" fill="$muted">$(Int(round(tv)))</text>""")
        end
        println(io, """<line x1="$x0" x2="$x0" y1="$y0" y2="$(y0 + ph)" stroke="$muted" stroke-width="1"/>""")
        println(io, """<line x1="$x0" x2="$(x0 + pw)" y1="$(sy(0.0))" y2="$(sy(0.0))" stroke="$zero" stroke-width="1" stroke-dasharray="4 3"/>""")
        pts = join((@sprintf("%.2f,%.2f", sx(t), sy(y[t])) for t in 1:T), " ")
        println(io, """<polyline points="$pts" fill="none" stroke="$line" stroke-width="2" stroke-linejoin="round"/>""")
    end
    println(io, "</svg>")
    String(take!(io))
end

# Rendering hook for the Plots.jl extension: `_render_plots(fig, path, format)`; a method is added by ext/JDynPlotsExt.jl.
function _render_plots end

"""
    plot_irfs(m, sol = first_order(m); outdir, format = "svg", backend = :svg, nograph = nothing, kw...) -> IRFResult

Draw the impulse-response figures of `stoch_simul` (see `irf_figures` for the selection keywords) and save them
as `<outdir>/<name>_IRF_<shock>[<n>].<format>`, Dynare's file names; `outdir` defaults to `<model folder>/<name>/graphs`
(Dynare's `<dname>/graphs`). `nograph` defaults to the model's `stoch_simul(nograph)` option: nothing is drawn or
written then, but the content is still returned. `format` is "svg" (built in, no dependency) or, with
`backend = :plots` and Plots.jl loaded, "png", "pdf" or "svg". The returned `res.files` lists the files written.
"""
function plot_irfs(m, sol=first_order(m); outdir=nothing, format="svg", backend::Symbol=:svg,
                   nograph=nothing, kw...)
    res = irf_figures(m, sol; kw...)
    o = _stoch_simul_options(m)
    nograph = nograph === nothing ? _truthy(_opt(o, "nograph", false)) : nograph
    nograph && return res
    dir = outdir === nothing ? graphs_dir(m) : outdir
    append!(res.files, save_figures(to_figure(res), dir; format=format, backend=backend))
    res
end

"The content of an `IRFFigure` as a generic `Figure`."
function to_figure(f::IRFFigure)
    panels = [Panel(v, [Curve(1:size(f.series, 1), f.series[:, k]; kind=:line)]; zeroline=true) for (k, v) in enumerate(f.variables)]
    Figure(f.name, f.title, f.nrows, f.ncols, panels)
end
to_figure(r::IRFResult) = Figure[to_figure(f) for f in r.figures]
