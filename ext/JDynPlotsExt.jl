# Plots.jl (GR backend) rendering of IRF figures: PNG, PDF and SVG. Loaded only when the user loads Plots.
module JDynPlotsExt

using JDyn, Plots

function JDyn._render_plots(fig::JDyn.IRFFigure, path::String, format::String)
    format in ("png", "pdf", "svg") || throw(ArgumentError("format must be png, pdf or svg"))
    T = size(fig.series, 1)
    panels = map(enumerate(fig.variables)) do (k, v)
        y = fig.series[:, k]
        p = plot(1:T, y; title=v, titlefontsize=10, color="#2a78d6", linewidth=2, legend=false, framestyle=:axes,
                 grid=true, gridalpha=0.25, tickfontsize=7, xlims=(1, max(T, 2)))
        hline!(p, [0.0]; color="#9a9994", linestyle=:dash, linewidth=1)
        p
    end
    n = fig.nrows * fig.ncols
    append!(panels, [plot(framestyle=:none) for _ in length(panels)+1:n])
    out = plot(panels...; layout=(fig.nrows, fig.ncols), size=(320 * fig.ncols, 210 * fig.nrows + 30),
               plot_title=fig.title, plot_titlefontsize=12)
    savefig(out, path)
end

function JDyn._render_figure_plots(fig::JDyn.Figure, path::String, format::String)
    format in ("png", "pdf", "svg", "eps") || throw(ArgumentError("format must be png, pdf, eps or svg"))
    panels = map(fig.panels) do p
        pl = plot(; title=p.title, titlefontsize=10, legend=false, framestyle=:axes, grid=true, gridalpha=0.25, tickfontsize=7)
        pos = Dict{Float64,Float64}(); neg = Dict{Float64,Float64}()
        for (j, c) in enumerate(p.curves)
            col = isempty(c.color) ? JDyn._palette[mod1(j, length(JDyn._palette))] : c.color
            if c.kind in (:line, :dashed, :dotted)
                plot!(pl, c.x, c.y; color=col, linewidth=2, linestyle=c.kind == :line ? :solid : c.kind == :dashed ? :dash : :dot)
            elseif c.kind == :band
                plot!(pl, c.x, c.y; fillrange=c.y2, fillalpha=0.25, color=col, linealpha=0)
            elseif c.kind == :bar
                for (x, y) in zip(c.x, c.y)
                    (isfinite(y) && y != 0) || continue
                    base = y >= 0 ? get(pos, x, 0.0) : get(neg, x, 0.0)
                    (y >= 0 ? (pos[x] = base + y) : (neg[x] = base + y))
                    plot!(pl, Shape([x - 0.4, x + 0.4, x + 0.4, x - 0.4], [base, base, base + y, base + y]); color=col, linealpha=0)
                end
            elseif c.kind == :vline
                vline!(pl, c.x; color=col, linestyle=:dash)
            elseif c.kind == :scatter
                scatter!(pl, c.x, c.y; color=col, markersize=3)
            end
        end
        p.zeroline && hline!(pl, [0.0]; color="#9a9994", linestyle=:dash, linewidth=1)
        pl
    end
    append!(panels, [plot(framestyle=:none) for _ in length(panels)+1:fig.nrows*fig.ncols])
    out = plot(panels...; layout=(fig.nrows, fig.ncols), size=(320 * fig.ncols, 210 * fig.nrows + 30),
               plot_title=fig.title, plot_titlefontsize=12)
    savefig(out, path)
end

end
