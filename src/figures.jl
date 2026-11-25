# A small figure model shared by every plot of JDyn: a `Figure` is a titled grid of `Panel`s, a panel holds `Curve`s (lines, dashed lines, bands,
# stacked bars, vertical lines). Producers (impulse responses, shock decompositions, priors, densities, forecasts, ...) build `Figure`s that carry
# the data Dynare draws; `save_figures` writes them under `<fname>/graphs/` in Dynare's `graph_format` terms. The built-in writer produces SVG without any
# dependency; PNG / PDF / EPS go through the optional Plots.jl extension (ext/JDynPlotsExt.jl). Only the content matches Dynare's figures, not the look.

"""
    Curve(x, y; label="", kind=:line, color="", y2=Float64[])

One drawn object. `kind`: `:line`, `:dashed`, `:dotted`, `:band` (filled between `y` and `y2`), `:bar` (bars at `x`, stacked with the other `:bar`
curves of the panel, positive and negative parts separately, as Dynare's shock decompositions), `:vline` (a vertical line at each `x`), `:scatter`.
"""
struct Curve
    x::Vector{Float64}
    y::Vector{Float64}
    y2::Vector{Float64}
    label::String
    kind::Symbol
    color::String
end
Curve(x, y; label::AbstractString="", kind::Symbol=:line, color::AbstractString="", y2=Float64[]) =
    Curve(Float64.(collect(x)), Float64.(collect(y)), Float64.(collect(y2)), String(label), kind, String(color))

Curve(x; label::AbstractString="", kind::Symbol=:vline, color::AbstractString="") = Curve(x, Float64[]; label=label, kind=kind, color=color)

struct Panel
    title::String
    curves::Vector{Curve}
    zeroline::Bool
end
Panel(title, curves; zeroline::Bool=false) = Panel(String(title), collect(Curve, curves), zeroline)

"One figure: `name` is the file stem (Dynare's), `title` the window name, panels in row-major order on `nrows` x `ncols`; `legend` is `(label, color)`."
struct Figure
    name::String
    title::String
    nrows::Int
    ncols::Int
    panels::Vector{Panel}
    legend::Vector{Tuple{String,String}}
end
Figure(name, title, nrows, ncols, panels; legend=Tuple{String,String}[]) =
    Figure(String(name), String(title), Int(nrows), Int(ncols), collect(Panel, panels), collect(Tuple{String,String}, legend))

"""
    arrange_figures(stem, title, panels; figure_titles=nothing, number_first=false) -> Vector{Figure}

Split `panels` into figures of at most 9 as Dynare's `pltorg` does; with more than one figure the names get the figure number appended to `stem`
and the titles ` figure <k>`.
"""
function arrange_figures(stem::AbstractString, title::AbstractString, panels::Vector{Panel}; legend=Tuple{String,String}[], number_first::Bool=false)
    nbplt, nr, nc, lr, lc, nstar = pltorg(length(panels))
    figs = Figure[]
    for f in 1:nbplt
        chunk = panels[(f-1)*nstar+1:min(f * nstar, length(panels))]
        r, c = nbplt == 1 ? (nr, nc) : (f == nbplt ? (lr, lc) : (nr, nc))
        push!(figs, Figure(string(stem, (nbplt == 1 && !number_first) ? "" : string(f)), string(title, nbplt == 1 ? "" : " figure $f"), r, c, chunk, legend))
    end
    figs
end

# ---------------------------------------------------------------------------------------------- data
_palette = ["#2a78d6", "#eb6834", "#3a9d5d", "#8a5cc2", "#c9a227", "#2b2b29", "#d6477c", "#5fb3b3"]
_curve_color(c::Curve, k) = isempty(c.color) ? _palette[mod1(k, length(_palette))] : c.color

"The data behind a figure as long-format rows `figure, panel, series, kind, x, y, y2` (named tuples), for `write_csv` (ggplot2 etc.)."
function figure_table(figs::AbstractVector{Figure})
    rows = NamedTuple{(:figure, :panel, :series, :kind, :x, :y, :y2),Tuple{String,String,String,String,Float64,Float64,Float64}}[]
    for f in figs, p in f.panels, (k, c) in enumerate(p.curves), i in eachindex(c.x)
        push!(rows, (figure=f.name, panel=p.title, series=isempty(c.label) ? string("series", k) : c.label, kind=String(c.kind), x=c.x[i],
                     y=i <= length(c.y) ? c.y[i] : NaN, y2=i <= length(c.y2) ? c.y2[i] : NaN))
    end
    rows
end
figure_table(f::Figure) = figure_table([f])

# ---------------------------------------------------------------------------------------------- SVG
function _panel_range(p::Panel)
    lo, hi, xlo, xhi = Inf, -Inf, Inf, -Inf
    stack_pos = Dict{Float64,Float64}(); stack_neg = Dict{Float64,Float64}()
    for c in p.curves
        if c.kind == :bar
            for (x, y) in zip(c.x, c.y)
                isfinite(y) || continue
                y >= 0 ? (stack_pos[x] = get(stack_pos, x, 0.0) + y) : (stack_neg[x] = get(stack_neg, x, 0.0) + y)
            end
        end
        for v in c.x; isfinite(v) && (xlo = min(xlo, v); xhi = max(xhi, v)); end
        c.kind == :vline && continue
        for v in c.y; isfinite(v) && c.kind != :bar && (lo = min(lo, v); hi = max(hi, v)); end
        for v in c.y2; isfinite(v) && (lo = min(lo, v); hi = max(hi, v)); end
    end
    for v in values(stack_pos); hi = max(hi, v); lo = min(lo, 0.0); end
    for v in values(stack_neg); lo = min(lo, v); hi = max(hi, 0.0); end
    p.zeroline && (lo = min(lo, 0.0); hi = max(hi, 0.0))
    isfinite(lo) || (lo = 0.0; hi = 1.0)
    isfinite(xlo) || (xlo = 0.0; xhi = 1.0)
    lo == hi && (hi = lo + 1)
    xlo == xhi && (xhi = xlo + 1)
    (xlo, xhi, lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
end

"Render a `Figure` as a self-contained SVG string."
function svg_figure(fig::Figure; panel_w=320, panel_h=210)
    W, H = fig.ncols * panel_w, fig.nrows * panel_h + 34 + (isempty(fig.legend) ? 0 : 22)
    ink, muted, grid = "#2b2b29", "#6b6b67", "#e6e5e1"
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="Helvetica, Arial, sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="#ffffff"/>""")
    println(io, """<text x="$(W / 2)" y="22" text-anchor="middle" font-size="14" fill="$ink">$(_xml(fig.title))</text>""")
    for (k, p) in enumerate(fig.panels)
        r, c = fld(k - 1, fig.ncols), mod(k - 1, fig.ncols)
        x0, y0 = c * panel_w + 46, 34 + r * panel_h + 30
        pw, ph = panel_w - 66, panel_h - 66
        xlo, xhi, lo, hi = _panel_range(p)
        sx(t) = x0 + (t - xlo) / (xhi - xlo) * pw
        sy(v) = y0 + ph - (v - lo) / (hi - lo) * ph
        println(io, """<text x="$(x0 + pw / 2)" y="$(y0 - 10)" text-anchor="middle" font-size="12" fill="$ink">$(_xml(p.title))</text>""")
        for tv in _nice_ticks(lo, hi)
            println(io, """<line x1="$x0" x2="$(x0 + pw)" y1="$(sy(tv))" y2="$(sy(tv))" stroke="$grid" stroke-width="1"/>""")
            println(io, """<text x="$(x0 - 5)" y="$(sy(tv) + 3.5)" text-anchor="end" font-size="9" fill="$muted">$(_fmt_tick(tv))</text>""")
        end
        for tv in _nice_ticks(xlo, xhi; n=5)
            println(io, """<text x="$(sx(tv))" y="$(y0 + ph + 14)" text-anchor="middle" font-size="9" fill="$muted">$(_fmt_tick(tv))</text>""")
        end
        println(io, """<line x1="$x0" x2="$x0" y1="$y0" y2="$(y0 + ph)" stroke="$muted" stroke-width="1"/>""")
        (p.zeroline || (lo < 0 < hi)) && println(io, """<line x1="$x0" x2="$(x0 + pw)" y1="$(sy(0.0))" y2="$(sy(0.0))" stroke="#9a9994" stroke-width="1" stroke-dasharray="4 3"/>""")
        pos = Dict{Float64,Float64}(); neg = Dict{Float64,Float64}()
        nx = maximum(c -> length(c.x), p.curves; init=1)
        bw = max(1.0, 0.8 * pw / max(nx, 1))
        for (j, cv) in enumerate(p.curves)
            col = _curve_color(cv, j)
            if cv.kind in (:line, :dashed, :dotted)
                pts = join((@sprintf("%.2f,%.2f", sx(cv.x[i]), sy(cv.y[i])) for i in eachindex(cv.x) if isfinite(cv.y[i])), " ")
                dash = cv.kind == :dashed ? " stroke-dasharray=\"6 3\"" : cv.kind == :dotted ? " stroke-dasharray=\"2 3\"" : ""
                println(io, """<polyline points="$pts" fill="none" stroke="$col" stroke-width="2" stroke-linejoin="round"$dash/>""")
            elseif cv.kind == :band
                idx = [i for i in eachindex(cv.x) if isfinite(cv.y[i]) && isfinite(cv.y2[i])]
                pts = join([@sprintf("%.2f,%.2f", sx(cv.x[i]), sy(cv.y2[i])) for i in idx], " ") * " " * join([@sprintf("%.2f,%.2f", sx(cv.x[i]), sy(cv.y[i])) for i in reverse(idx)], " ")
                println(io, """<polygon points="$pts" fill="$col" fill-opacity="0.25" stroke="none"/>""")
            elseif cv.kind == :bar
                for (x, y) in zip(cv.x, cv.y)
                    isfinite(y) && y != 0 || continue
                    base = y >= 0 ? get(pos, x, 0.0) : get(neg, x, 0.0)
                    top = base + y
                    (y >= 0 ? (pos[x] = top) : (neg[x] = top))
                    ya, yb = sy(max(base, top)), sy(min(base, top))
                    println(io, """<rect x="$(@sprintf("%.2f", sx(x) - bw / 2))" y="$(@sprintf("%.2f", ya))" width="$(@sprintf("%.2f", bw))" height="$(@sprintf("%.2f", max(yb - ya, 0.0)))" fill="$col"/>""")
                end
            elseif cv.kind == :vline
                for x in cv.x
                    println(io, """<line x1="$(sx(x))" x2="$(sx(x))" y1="$y0" y2="$(y0 + ph)" stroke="$col" stroke-width="1.5" stroke-dasharray="3 3"/>""")
                end
            elseif cv.kind == :scatter
                for (x, y) in zip(cv.x, cv.y)
                    isfinite(y) && println(io, """<circle cx="$(sx(x))" cy="$(sy(y))" r="2.5" fill="$col"/>""")
                end
            end
        end
    end
    for (i, (lab, col)) in enumerate(fig.legend)
        x = 20 + (i - 1) * 140
        println(io, """<rect x="$x" y="$(H - 16)" width="12" height="8" fill="$col"/><text x="$(x + 16)" y="$(H - 8)" font-size="10" fill="$ink">$(_xml(lab))</text>""")
    end
    println(io, "</svg>")
    String(take!(io))
end

# ---------------------------------------------------------------------------------------------- saving
function _render_figure_plots end        # added by ext/JDynPlotsExt.jl: (fig::Figure, path, format)

"Dynare's `graph_format` (a name or a list) as a vector of lower-case names."
_graph_formats(v) = v === nothing ? ["svg"] : v isa AbstractString ? [lowercase(String(v))] : [lowercase(String(x)) for x in v]

"""
    save_figures(figs, dir; format="svg", backend=:svg, nograph=false) -> Vector{String}

Write the figures as `<dir>/<name>.<format>` (Dynare's `dyn_saveas`): `format` may be a list; `"none"` writes nothing. SVG is built in; `"png"`, `"pdf"`, `"eps"`
need Plots.jl (`using Plots`) and `backend=:plots`. `nograph=true` returns no files. Returns the paths written.
"""
function save_figures(figs::AbstractVector{Figure}, dir::AbstractString; format="svg", backend::Symbol=:svg, nograph::Bool=false)
    files = String[]
    (nograph || isempty(figs)) && return files
    fmts = filter(!=("none"), _graph_formats(format))
    isempty(fmts) && return files
    mkpath(dir)
    for fmt in fmts
        if backend == :svg
            fmt == "svg" || throw(ArgumentError("the built-in backend writes SVG only; for $fmt load Plots and use backend = :plots"))
        elseif backend == :plots
            hasmethod(_render_figure_plots, Tuple{Figure,String,String}) || throw(ArgumentError("backend = :plots needs the Plots package: run `using Plots` first"))
        else
            throw(ArgumentError("backend must be :svg or :plots"))
        end
        for f in figs
            path = joinpath(dir, string(f.name, ".", fmt))
            backend == :svg ? write(path, svg_figure(f)) : _render_figure_plots(f, path, fmt)
            push!(files, path)
        end
    end
    files
end
save_figures(f::Figure, dir; kw...) = save_figures([f], dir; kw...)

"Dynare's graph folder `<model folder>/<name>/graphs`."
graphs_dir(m) = joinpath(m.mod_dir, m.name, "graphs")
