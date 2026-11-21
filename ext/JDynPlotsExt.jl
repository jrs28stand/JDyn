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

end
