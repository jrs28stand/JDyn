# OccBin figures: `occbin.plot_irfs` (IRFs to positive / negative shocks, linear against piecewise) and `occbin.graph` (simulated paths).

"""
    occbin_irf_figures(res::OccbinIRF, m; variables=nothing, threshold=1e-10, ncols=3, nrows=3, simulname="", length_irf=nothing,
                       endo_scaling_factor=nothing) -> Vector{Figure}

The figures of Dynare's `occbin.plot_irfs`: for each sign (`pos`, `neg`) and shock, figures of `nrows` x `ncols` panels (default 3 x 3) with the
linear (blue) and piecewise-linear (red dashed) responses; responses below `threshold` are set to zero; x is 1:`length_irf` (default the
response length). File stems `<fname>_irf_occbin_<simulname_><shock>_<sign>_<k>`, titles `OccBin IRFs to positive e shock <simulname>`.
`add_steadystate` is not supported (refused).
"""
function occbin_irf_figures(res::OccbinIRF, m; variables=nothing, threshold::Real=1e-10, ncols::Int=3, nrows::Int=3, simulname::AbstractString="",
                            length_irf=nothing, endo_scaling_factor=nothing, add_steadystate::Bool=false)
    add_steadystate && throw(ArgumentError("occbin plot_irf add_steadystate is not supported"))
    vars = variables === nothing ? orig_endo_names(m) : unique(String.(variables))
    cols = [findfirst(==(v), res.variables) for v in vars]
    any(isnothing, cols) && throw(ArgumentError("unknown variable(s): $(join(vars[isnothing.(cols)], ", "))"))
    scale = endo_scaling_factor === nothing ? ones(length(vars)) : Float64.(endo_scaling_factor)
    save = isempty(simulname) ? "" : simulname * "_"
    npan = ncols * nrows
    figs = Figure[]
    for sign in res.signs, shock in res.shocks
        haskey(res, (shock, sign)) || continue
        r = res[(shock, sign)]
        word = sign == "pos" ? "positive" : sign == "neg" ? "negative" : throw(ArgumentError("Unknown shock sign $sign"))
        panels = Panel[]
        for (k, (v, c)) in enumerate(zip(vars, cols))
            cv = Curve[]
            for (arr, kind, color, lab) in ((r.linear, :line, "#2a78d6", "Linear"), (r.piecewise, :dashed, "#d62a2a", "Piecewise"))
                y = copy(arr[:, c]); y[abs.(y) .< threshold] .= 0
                n = min(length_irf === nothing ? length(y) : Int(length_irf), length(y))
                push!(cv, Curve(1:n, y[1:n] .* scale[k]; kind=kind, color=color, label=lab))
            end
            push!(cv, Curve(cv[1].x, zeros(length(cv[1].x)); kind=:line, color="#000000"))
            push!(panels, Panel(v, cv))
        end
        for (f, i0) in enumerate(1:npan:length(panels))
            push!(figs, Figure(string(m.name, "_irf_occbin_", save, shock, "_", sign, "_", f), "OccBin IRFs to $word $shock shock $simulname",
                               nrows, ncols, panels[i0:min(i0 + npan - 1, end)]; legend=[("Linear", "#2a78d6"), ("Piecewise", "#d62a2a")]))
        end
    end
    figs
end

"""
    occbin_graph_figures(res::OccbinResult, m; variables=nothing, noconstant=false) -> Vector{Figure}

`occbin.graph`: simulated paths, piecewise (blue) and linear (red dashed), as deviations from the steady state; `steady_state=true` (Dynare's
`graph.steady_state`) draws levels. Shock sequences of exogenous variables in `variables` follow the endogenous
panels. At most 9 panels per figure (`pltorg`); stems `<fname>_occbin_<k>`, titles `OccBin simulated paths, figure k`.
"""
function occbin_graph_figures(res::OccbinResult, m; variables=nothing, steady_state::Bool=false)
    vars = variables === nothing ? orig_endo_names(m) : unique(String.(variables))
    T = size(res.piecewise, 1)
    panels = Panel[]
    for v in vars
        c = findfirst(==(v), res.endo_names)
        c === nothing && continue
        off = steady_state ? 0.0 : res.ys[c]
        push!(panels, Panel(v, [Curve(1:T, res.piecewise[:, c] .- off; color="#2a78d6", label="Piecewise Linear"),
                                Curve(1:T, res.linear[:, c] .- off; kind=:dashed, color="#d62a2a", label="Linear")]; zeroline=true))
    end
    for v in vars
        e = findfirst(==(v), res.exo_names)
        e === nothing && continue
        col = findfirst(==(e), res.exo_pos)
        col === nothing && throw(ArgumentError("$v was not part of the shocks for OccBin"))
        seq = [t <= size(res.shocks, 1) ? res.shocks[t, col] : 0.0 for t in 1:T]
        push!(panels, Panel(v, [Curve(1:T, seq; color="#2a78d6", label="Piecewise Linear")]; zeroline=true))
    end
    figs = arrange_figures(string(m.name, "_occbin_"), "", panels; number_first=true)
    [Figure(f.name, string("OccBin simulated paths, figure ", k), f.nrows, f.ncols, f.panels, [("Piecewise Linear", "#2a78d6"), ("Linear", "#d62a2a")])
     for (k, f) in enumerate(figs)]
end
