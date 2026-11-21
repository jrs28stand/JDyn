# Historical shock decompositions on top of the Kalman smoother (port of the second half of pydyn/smoother.py): Dynare's
# `shock_decomposition`, `initial_condition_decomposition`, `realtime_shock_decomposition`, `squeeze_shock_decomposition`,
# `shock_groups` / `init2shocks` and the numbers `plot_shock_decomposition` draws (as tidy tables, not figures).
# Derived from Dynare 7.2's matlab/shock_decomposition/{shock_decomposition, initial_condition_decomposition,
# realtime_shock_decomposition, squeeze_shock_decomposition, plot_shock_decomposition (make_the_groups), expand_group, graph_decomp,
# graph_decomp_detail}.m (GPL-3.0-or-later, Dynare Team).
#
#     res = smooth(m)                              # or pass nothing: the decomposition runs the smoother itself
#     dec = shock_decomposition(m, res)            # dec.z is (n_endo, n_exo + 2, T): shocks, initial conditions, smoothed
#     dec["y"]["e_a"]                              # contribution of one shock to one variable (vector over periods)
#     g = group_shocks(dec, model=m, name="grp")   # the groups of a shock_groups block
#     d = decomposition_plot_data(dec, ["y", "pi"], m; use_shock_groups="grp")
#     write_csv("sd.csv", decomposition_table(d))  # variable, component, period, value (+ the smoothed line)

"""
    ShockDecomposition

`z` has Dynare's layout `(n_endo, n_columns, T)`: one column per shock (or per variable for the initial-condition decomposition, or per
group), then the contribution of the initial conditions, then the smoothed deviation. `columns` labels the second axis.
"""
Base.@kwdef mutable struct ShockDecomposition
    z::Array{Float64,3}
    endo_names::Vector{String}
    columns::Vector{String}
    kind::String = "shocks"                         # "shocks", "initial_conditions" or "groups"
    init_state::Int = 0
    steady_state::Union{Nothing,Vector{Float64}} = nothing   # level steady state of the variables at the parameters used
    parameter_set::String = "calibration"
    first_period::Int = 1                           # number of the first period (Dynare counts the sample from 1)
end
Base.show(io::IO, d::ShockDecomposition) = print(io, "ShockDecomposition(", d.kind, ", ", size(d.z, 1), " variables, ", size(d.z, 2) - 2,
                                                  " components, ", size(d.z, 3), " periods)")
"Contributions to one variable, as a Dict column => series."
function Base.getindex(d::ShockDecomposition, name::AbstractString)
    i = findfirst(==(name), d.endo_names)
    i === nothing && throw(SmootherError("variable $name is not in the decomposition"))
    Dict{String,Vector{Float64}}(c => d.z[i, j, :] for (j, c) in enumerate(d.columns))
end

# ====================================================================== options of the decomposition statements
function _decomp_statement_options(m::Model, kind)
    out = _statement_options(m, (kind,))
    symbols = String[]
    for st in _statements_of(m, kind)
        symbols = String.(get_or(st, "symbol_list", String[]))
    end
    out, symbols
end

"Warn once about options of `kind` statements that the decomposition does not act on."
function _report_unused(m::Model, kind, honoured, what, options)
    known = union(Set(honoured), _SM_COSMETIC, _DECOMP_COSMETIC)
    unused = sort([k for k in keys(options) if !(k in known) && !startswith(k, "no_graph.")])
    isempty(unused) || warn_once(m, "$what does not act on these $kind options from the .mod file: $(join(unused, ", "))")
    unused
end

function _file_smoother_options(m::Model, kind, kw)
    file_opts, _ = _decomp_statement_options(m, kind)
    merged = Dict{Symbol,Any}()
    for (k, v) in file_opts
        haskey(_SM_DEFAULTS, k) && (merged[Symbol(k)] = v)
    end
    for (k, v) in kw
        merged[Symbol(k)] = v
    end
    file_opts, merged
end

"""
    shock_decomposition(m, res_or_data=nothing; init_state=nothing, options...) -> ShockDecomposition

Historical shock decomposition (Dynare's `shock_decomposition`). `res`: a `SmootherResult`, or data / nothing, in which case the
smoother is run with `options` (defaults also read from the file's `shock_decomposition` statement, including `parameter_set`; the
parameter set is chosen as in `smooth`: `estimation=est, theta=vector` or `estimation=est, parameter_set="posterior_mode", mode=...`).
`init_state`: Dynare's option; shocks of periods `1..init_state` are attributed to the initial conditions (0: condition on the
smoothed state of period 0).

`z` is `(n_endo, n_exo + 2, T)`: shocks in declaration order, the initial conditions, then the smoothed deviation from the steady state (in
logs with `loglinear`). Observed variables under `prefilter` are measured from the model's steady state, not from the data mean, and
observation trends are not part of the decomposition, as in Dynare. Shocks are never grouped here: `use_shock_groups` acts when
plotting (`group_shocks`, `decomposition_plot_data`), as in Dynare where `oo_.shock_decomposition` is ungrouped.
"""
function shock_decomposition(m::Model, res::SmootherResult; init_state=nothing)
    file_opts, _ = _decomp_statement_options(m, "shock_decomposition")
    _report_unused(m, "shock_decomposition", union(keys(_SM_DEFAULTS), ["init_state", "use_shock_groups", "max_nrows"]), "shock_decomposition", file_opts)
    init_state === nothing && (init_state = Int(_tonum(_unwrap1(get(file_opts, "init_state", 0)))))
    T, n = size(res.smoothed)
    nx = size(res.shocks, 2)
    z = zeros(n, nx + 2, T)
    z[:, nx+2, :] = permutedims(res.smoothed)
    for t in 1:T
        t > 1 && (z[:, 1:nx, t] = res.G * z[:, 1:nx, t-1])
        t > init_state && (z[:, 1:nx, t] .+= res.H .* res.shocks[t, :]')
        z[:, nx+1, t] = z[:, nx+2, t] - vec(sum(z[:, 1:nx, t]; dims=2))
    end
    ShockDecomposition(z=z, endo_names=copy(res.endo_names), columns=vcat(res.exo_names, ["initial_conditions", "smoothed"]),
                       kind="shocks", init_state=Int(init_state), steady_state=copy(res.SteadyState), parameter_set=res.parameter_set)
end
function shock_decomposition(m::Model, data=nothing; init_state=nothing, kw...)
    _, merged = _file_smoother_options(m, "shock_decomposition", kw)
    shock_decomposition(m, smooth(m, data; merged...); init_state=init_state)
end

"""
    initial_condition_decomposition(m, res_or_data=nothing; options...) -> ShockDecomposition

Dynare's `initial_condition_decomposition`: contribution of each variable's smoothed value in period 1 to the later path. `z` is
`(n_endo, n_endo + 2, T)`; column `n_endo + 1` is the remainder, i.e. the contribution of all shocks from period 2 on, and the last
column is the smoothed deviation.
"""
function initial_condition_decomposition(m::Model, res::SmootherResult)
    file_opts, _ = _decomp_statement_options(m, "initial_condition_decomposition")
    _report_unused(m, "initial_condition_decomposition", ["type", "steadystate", "diff", "flip", "detail_plot", "max_nrows"],
                   "initial_condition_decomposition", file_opts)
    T, n = size(res.smoothed)
    z = zeros(n, n + 2, T)
    z[:, n+2, :] = permutedims(res.smoothed)
    for i in 1:n
        z[i, i, 1] = res.smoothed[1, i]
    end
    for t in 1:T
        t > 1 && (z[:, 1:n, t] = res.G * z[:, 1:n, t-1])
        z[:, n+1, t] = z[:, n+2, t] - vec(sum(z[:, 1:n, t]; dims=2))
    end
    ShockDecomposition(z=z, endo_names=copy(res.endo_names), columns=vcat(res.endo_names, ["shocks", "smoothed"]),
                       kind="initial_conditions", steady_state=copy(res.SteadyState), parameter_set=res.parameter_set)
end
function initial_condition_decomposition(m::Model, data=nothing; kw...)
    _, merged = _file_smoother_options(m, "initial_condition_decomposition", kw)
    initial_condition_decomposition(m, smooth(m, data; merged...))
end

# ====================================================================== groups, init2shocks, squeeze
"Groups of a `shock_groups` block as an ordered vector of `label => [shock, ...]` (the last block, or the one called `name`)."
function shock_groups(m::Model, name=nothing)
    blocks = _statements_of(m, "shock_groups")
    name === nothing || (blocks = [b for b in blocks if get(b, "name", "default") == name])
    isempty(blocks) && throw(SmootherError("no shock_groups block found" * (name === nothing ? "" : " with name $(repr(name))")))
    Pair{String,Vector{String}}[String(g["group_name"]) => String.(g["shocks"]) for g in blocks[end]["groups"]]
end

"""
    group_shocks(decomp; groups=nothing, model=nothing, name=nothing, expand=nothing) -> ShockDecomposition

Sum shock contributions by group, as Dynare does when plotting with `use_shock_groups` (`make_the_groups` in
`plot_shock_decomposition.m`). `groups` is a vector of `label => [shocks]` pairs (or a `Dict`/`NamedTuple`-like iterable of pairs); or pass
`model` (and optionally the `name` of a `shock_groups` block). The result has one column per group, then one column holding the shocks
in no group plus the initial conditions, then the smoothed deviation. That column is labelled "Others + Initial Values" when some shock
outside the groups contributes and "Initial values" otherwise (Dynare's labels). A shock listed in two groups counts in the first only.
`expand="label"` reproduces Dynare's "expand group": the shocks of that group one by one, everything else in "Others + Initial Values".
"""
function group_shocks(decomp::ShockDecomposition; groups=nothing, model=nothing, name=nothing, expand=nothing)
    decomp.kind == "shocks" || throw(SmootherError("group_shocks needs an ungrouped shock decomposition"))
    if groups === nothing
        model === nothing && throw(SmootherError("group_shocks needs groups= or model="))
        groups = shock_groups(model, name)
    end
    groups = collect(Pair{String,Vector{String}}, [String(k) => String.(v) for (k, v) in groups])
    if expand !== nothing
        i = findfirst(p -> p.first == expand, groups)
        i === nothing && throw(SmootherError("no group $(repr(expand)) to expand; groups are $(first.(groups))"))
        groups = [s => [s] for s in groups[i].second]
    end
    nshocks = length(decomp.columns) - 2
    z = copy(decomp.z)
    out = zeros(size(z, 1), length(groups) + 2, size(z, 3))
    for (i, (label, members)) in enumerate(groups), s in members
        k = findfirst(==(s), decomp.columns[1:nshocks])
        k === nothing && throw(SmootherError("shock $(repr(s)) of group $(repr(label)) is not in the decomposition"))
        out[:, i, :] .+= z[:, k, :]
        z[:, k, :] .= 0                           # a shock listed twice counts once, as in Dynare
    end
    others = any(!iszero, nan_to_zero.(dropdims(sum(z[:, 1:nshocks, :]; dims=2); dims=2)))
    out[:, end-1, :] = dropdims(sum(z[:, 1:nshocks+1, :]; dims=2); dims=2)
    out[:, end, :] = z[:, nshocks+2, :]
    ShockDecomposition(z=out, endo_names=copy(decomp.endo_names), columns=vcat(first.(groups), [others ? "Others + Initial Values" : "Initial values", "smoothed"]),
                       kind="groups", init_state=decomp.init_state, steady_state=decomp.steady_state, parameter_set=decomp.parameter_set,
                       first_period=decomp.first_period)
end
nan_to_zero(x) = isnan(x) ? zero(x) : x

_init2shocks_block(m, name) = init2shocks(m, name)
"Pairs `(endogenous, exogenous)` of an `init2shocks` block (the last one, or `name`); the exogenous entry is `nothing` when absent."
function init2shocks(m::Model, name=nothing)
    blocks = _statements_of(m, "init2shocks")
    name === nothing || (blocks = [b for b in blocks if get(b, "name", "default") == name])
    isempty(blocks) && throw(SmootherError("no init2shocks block found" * (name === nothing ? "" : " with name $(repr(name))")))
    Tuple{String,Union{Nothing,String}}[(String(g["endogenous"]), (e = get(g, "exogenous", nothing); (e === nothing || isempty(e)) ? nothing : String(e))) for g in blocks[end]["groups"]]
end

"""
    apply_init2shocks(decomp, icd, pairs; rename=true)

Attribute the effect of the initial condition of some endogenous variables to shocks (Dynare's `init2shocks` option of
`plot_shock_decomposition`). `pairs`: `(endogenous, exogenous)` tuples; `icd` the initial-condition decomposition of the same smoother
run. For a shock decomposition the contribution of the variable's period-1 value moves from the initial conditions to the shock, whose label
becomes "shock + variable" (unless `rename` is false). For an initial-condition decomposition (pass it as `decomp`) the listed variables are
merged into the remainder (all shocks), as Dynare does in `initial_condition_decomposition.m`.
"""
function apply_init2shocks(decomp::ShockDecomposition, icd, pairs; rename=true)
    z = copy(decomp.z)
    cols = copy(decomp.columns)
    if decomp.kind == "initial_conditions"
        for (endo, _) in pairs
            j = findfirst(==(endo), decomp.endo_names)
            z[:, end-1, :] .+= z[:, j, :]
            z[:, j, :] .= 0
        end
    elseif decomp.kind == "shocks"
        (icd === nothing || icd.kind != "initial_conditions") && throw(SmootherError("init2shocks needs the initial-condition decomposition (icd=...)"))
        icd.endo_names == decomp.endo_names || throw(SmootherError("init2shocks: the initial-condition decomposition holds other variables (apply init2shocks before squeezing)"))
        T = size(z, 3)
        for (endo, exo) in pairs
            j = findfirst(==(endo), icd.columns)
            j === nothing && throw(SmootherError("init2shocks: variable $endo is not in the initial-condition decomposition"))
            if exo !== nothing
                jj = findfirst(==(exo), decomp.columns)
                jj === nothing && throw(SmootherError("init2shocks: shock $exo is not in the decomposition"))
                rename && (cols[jj] = "$(cols[jj]) + $endo")
                z[:, jj, :] .+= icd.z[:, j, 1:T]
            else
                z[:, end, :] .-= icd.z[:, j, 1:T]
            end
            z[:, end-1, :] .-= icd.z[:, j, 1:T]
        end
    else
        throw(SmootherError("init2shocks applies before the shocks are grouped"))
    end
    ShockDecomposition(z=z, endo_names=copy(decomp.endo_names), columns=cols, kind=decomp.kind, init_state=decomp.init_state,
                       steady_state=decomp.steady_state, parameter_set=decomp.parameter_set, first_period=decomp.first_period)
end

# ====================================================================== realtime decomposition
"""
    RealtimeDecomposition

Output of `realtime_shock_decomposition`; the four dicts mirror Dynare's `oo_` fields and hold arrays `(n_endo, n_exo + 2, periods)` with
the columns of a shock decomposition.

- `realtime` (`oo_.realtime_shock_decomposition`): `"pool"` (`nobs + forecast` periods: period `t` as decomposed with the data up to `t`; the
  periods up to `presample + 1` come from the first vintage, the forecast periods from the last) and `"time_j"` for the saved vintages
  (data up to `j`, `j + forecast` periods);
- `conditional` (`oo_.conditional_shock_decomposition`): `"pool"` (one-step: period `t` given the smoothed state of `t - 1`, vintage `t`) and
  `"time_j"` (`forecast + 1` periods starting at `j`, conditional on the state at `j`, from vintage `j + forecast`);
- `realtime_forecast`: `"pool"` (`nobs + 1` periods: the one-step forecast of `t` made with the data up to `t - 1`) and `"time_j"` (period `j`
  and the `forecast` periods after it, as forecast in vintage `j`);
- `realtime_conditional`: `"time_j"`: realtime pool minus the forecast made at `j` (the forecast itself is added to the initial-condition column).
"""
Base.@kwdef mutable struct RealtimeDecomposition
    endo_names::Vector{String}
    columns::Vector{String}
    nobs::Int
    forecast::Int
    presample::Int
    realtime::Dict{String,Array{Float64,3}}
    conditional::Dict{String,Array{Float64,3}}
    realtime_forecast::Dict{String,Array{Float64,3}}
    realtime_conditional::Dict{String,Array{Float64,3}}
    steady_state::Union{Nothing,Vector{Float64}} = nothing
    parameter_set::String = "calibration"
end

"""
    select_vintage(rt, realtime=1, vintage=0) -> ShockDecomposition

The decomposition Dynare's `plot_shock_decomposition(realtime=..., vintage=...)` draws: `realtime` 1 (realtime), 2 (conditional), 3 (forecast);
`vintage` 0 for the pooled one.
"""
function select_vintage(rt::RealtimeDecomposition, realtime::Integer=1, vintage::Integer=0)
    realtime in (1, 2, 3) || throw(SmootherError("realtime must be 1 (realtime), 2 (conditional) or 3 (forecast)"))
    key = vintage != 0 ? "time_$vintage" : "pool"
    if realtime == 1
        store, first = rt.realtime, 1
    elseif realtime == 2
        store, first = vintage != 0 ? (rt.realtime_conditional, vintage) : (rt.conditional, 1)
    else
        store, first = rt.realtime_forecast, vintage != 0 ? vintage : 1
    end
    haskey(store, key) || throw(SmootherError("vintage $vintage was not stored (save_realtime); available: $(sort(collect(keys(store))))"))
    ShockDecomposition(z=store[key], endo_names=copy(rt.endo_names), columns=copy(rt.columns), kind="shocks",
                       steady_state=rt.steady_state, parameter_set=rt.parameter_set, first_period=first)
end

"""
    squeeze_shock_decomposition(obj, var_list)

Dynare's `squeeze_shock_decomposition`: keep the decomposition of the listed variables only (in the order given). Works on a
`ShockDecomposition` or a `RealtimeDecomposition`.
"""
function squeeze_shock_decomposition(obj, var_list)
    var_list = String.(var_list)
    miss = [v for v in var_list if !(v in obj.endo_names)]
    isempty(miss) || throw(SmootherError("variable(s) $miss not in the decomposition"))
    k = [findfirst(==(v), obj.endo_names) for v in var_list]
    if obj isa RealtimeDecomposition
        sq(d) = Dict{String,Array{Float64,3}}(key => v[k, :, :] for (key, v) in d)
        return RealtimeDecomposition(endo_names=var_list, columns=copy(obj.columns), nobs=obj.nobs, forecast=obj.forecast, presample=obj.presample,
                                     realtime=sq(obj.realtime), conditional=sq(obj.conditional), realtime_forecast=sq(obj.realtime_forecast),
                                     realtime_conditional=sq(obj.realtime_conditional),
                                     steady_state=obj.steady_state === nothing ? nothing : obj.steady_state[k], parameter_set=obj.parameter_set)
    end
    ShockDecomposition(z=obj.z[k, :, :], endo_names=var_list, columns=copy(obj.columns), kind=obj.kind, init_state=obj.init_state,
                       steady_state=obj.steady_state === nothing ? nothing : obj.steady_state[k], parameter_set=obj.parameter_set,
                       first_period=obj.first_period)
end

"""
    realtime_shock_decomposition(m, data=nothing; forecast=nothing, presample=nothing, save_realtime=nothing, fast_realtime=nothing, options...)

Dynare's `realtime_shock_decomposition`: the smoother is re-run on the samples `1..j` for `j = presample + 1, ..., nobs` and each period is
decomposed with the data available then. `forecast` (periods ahead, shocks set to zero), `presample`, `save_realtime` (vintages to keep) and
`fast_realtime` (vintages at which the smoother is really re-run; other vintages reuse the next such run truncated to `j` periods) default to
the statement in the `.mod` file. Other keywords are those of `smooth` (parameter sets included; the parameters are the same for all vintages).

Conventions of Dynare 7.2 kept here (both modes, as in pydyn): `presample` is `max(1, P - 1)` with `P` the `presample` option of the estimation, and the option given here can only
lower it (`min` of the two), so with the default `presample=0` the recursion always starts at `j = 2` whatever is asked; the shocks of period 1
are attributed to the initial conditions (in `shock_decomposition` they are not), so the pooled result differs from the full-sample
decomposition even in the last vintage.
"""
function realtime_shock_decomposition(m::Model, data=nothing; forecast=nothing, presample=nothing, save_realtime=nothing,
                                      fast_realtime=nothing, kw...)
    file_opts, merged = _file_smoother_options(m, "realtime_shock_decomposition", kw)
    _report_unused(m, "realtime_shock_decomposition", union(keys(_SM_DEFAULTS), ["forecast", "presample", "save_realtime", "fast_realtime"]),
                   "realtime_shock_decomposition", file_opts)
    ints(v) = v === nothing ? Int[] : Int[Int(_tonum(x)) for x in (v isa AbstractVector ? v : [v])]
    forecast = Int(_tonum(_unwrap1(forecast === nothing ? get(file_opts, "forecast", 0) : forecast)))
    my_presample = presample === nothing ? get(file_opts, "presample", nothing) : presample
    delete!(merged, :presample)               # the statement's presample is not the estimation's
    save = ints(save_realtime === nothing ? get(file_opts, "save_realtime", nothing) : save_realtime)
    fast = filter(>(0), ints(fast_realtime === nothing ? get(file_opts, "fast_realtime", 0) : fast_realtime))

    full = smooth(m, data; merged...)
    nobs_, n, nx = size(full.smoothed, 1), size(full.smoothed, 2), size(full.shocks, 2)
    pres = max(1, Int(full.options["presample"]) - 1)
    my_presample === nothing || (pres = min(pres, max(1, Int(_tonum(_unwrap1(my_presample))))))
    G, H = full.G, full.H
    again = Dict{Symbol,Any}(k => v for (k, v) in merged if !(k in (:estimation, :theta, :mode, :mcmc, :sol, :nobs, :datafile, :parameter_set)))
    est = get(merged, :estimation, nothing)
    run(j) = j == nobs_ ? full : smooth(m; reuse=full, estimation=est, nobs=j, again...)

    zreal = zeros(n, nx + 2, nobs_ + forecast)
    zcond = zeros(n, nx + 2, nobs_)
    zfrcst = zeros(n, nx + 2, nobs_ + 1)
    rt = Dict{String,Array{Float64,3}}(); cond = Dict{String,Array{Float64,3}}(); fc = Dict{String,Array{Float64,3}}()
    rtc = Dict{String,Array{Float64,3}}(); fc_all = Dict{Int,Array{Float64,3}}()
    init = true
    gend0 = 0
    res0 = nothing
    for j in pres+1:nobs_
        if isempty(fast)
            res = run(j)
        else
            if j < minimum(fast) && gend0 < j
                res0, gend0 = run(minimum(fast)), minimum(fast)
            end
            if j in fast && gend0 < j
                res = res0 = run(j)
                gend0 = j
            else
                if j > gend0
                    gend0 = j > maximum(fast) ? nobs_ : minimum(f for f in fast if f > j)
                    res0 = run(gend0)
                end
                res = res0
            end
        end
        gend = j
        sm = permutedims(res.smoothed[1:gend, :])                         # (n, gend)
        eps = hcat(permutedims(res.shocks[1:gend, :]), zeros(nx, forecast))
        z = zeros(n, nx + 2, gend + forecast)
        z[:, end, 1:gend] = sm
        for i in 1:gend+forecast
            if i > 1
                i > gend && (z[:, end, i] = G * z[:, end, i-1])
                z[:, 1:nx, i] = G * z[:, 1:nx, i-1] .+ H .* eps[:, i]'
            end
            z[:, nx+1, i] = z[:, end, i] - vec(sum(z[:, 1:nx, i]; dims=2))
        end
        # conditional decomposition one step ahead
        z1 = zeros(n, nx + 2)
        z1[:, end] = sm[:, gend]
        z1[:, 1:nx] = H .* eps[:, gend]'
        z1[:, nx+1] = z1[:, end] - vec(sum(z1[:, 1:nx]; dims=2))
        # conditional decomposition k steps ahead
        if forecast > 0 && forecast < j
            zn = zeros(n, nx + 2, forecast + 1)
            zn[:, end, :] = sm[:, gend-forecast:gend]
            for i in 1:forecast+1
                i > 1 && (zn[:, 1:nx, i] = G * zn[:, 1:nx, i-1] .+ H .* eps[:, i+gend-forecast-1]')
                zn[:, nx+1, i] = zn[:, end, i] - vec(sum(zn[:, 1:nx, i]; dims=2))
            end
            (j - forecast) in save && (cond["time_$(j - forecast)"] = zn)
        end
        if init
            zreal[:, :, 1:j] = z[:, :, 1:j]
        elseif j < nobs_
            zreal[:, :, j] = z[:, :, gend]
        else
            zreal[:, :, j:end] = z[:, :, gend:end]
        end
        zcond[:, :, j] = z1
        j in save && (rt["time_$j"] = z)
        if forecast > 0
            zfrcst[:, :, j+1] = z[:, :, gend+1]
            fc_all[j] = z[:, :, gend:end]
            j in save && (fc["time_$j"] = fc_all[j])
            if j > forecast + pres
                for back in vcat([forecast], j == nobs_ ? collect(forecast-1:-1:1) : Int[])
                    o = j - back
                    x = zreal[:, :, o:j] - fc_all[o][:, :, 1:back+1]
                    x[:, end-1, :] += fc_all[o][:, end, 1:back+1]
                    x[:, end, :] = zreal[:, end, o:j]
                    o in save && (rtc["time_$o"] = x)
                end
            end
        end
        init = false
    end
    rt["pool"] = zreal
    cond["pool"] = zcond
    forecast > 0 && (fc["pool"] = zfrcst)
    RealtimeDecomposition(endo_names=copy(full.endo_names), columns=vcat(full.exo_names, ["initial_conditions", "smoothed"]), nobs=nobs_,
                          forecast=forecast, presample=pres, realtime=rt, conditional=cond, realtime_forecast=fc, realtime_conditional=rtc,
                          steady_state=copy(full.SteadyState), parameter_set=full.parameter_set)
end

# ====================================================================== the numbers behind the figures
"""
    DecompositionPlotData

What `plot_shock_decomposition` draws. `z` is `(n_variables, n_components + 1, T)`: Dynare's `z(i_var,:,:)` after grouping, `diff`, `type` and
`flip` and before cropping; the rows are the components named in `labels` (the last one is the initial conditions, with the ungrouped
shocks when groups are used) and then the line (the smoothed deviation). `periods` numbers the columns (sample periods from 1; years for
`type="aoa"`); `steady_state` is the value Dynare adds to the axis labels with the `steadystate` option (after the same transformations).

`panels` holds one `NamedTuple` per figure, after cropping and screening: `name`, `labels`, `contributions` `(n_components, T)`, `total` `(T,)`,
`periods`, `steady_state`. Variables whose bars span less than 1e-6 are skipped, as in Dynare.
"""
Base.@kwdef mutable struct DecompositionPlotData
    var_list::Vector{String}
    z::Array{Float64,3}
    labels::Vector{String}
    periods::Vector{Int}
    steady_state::Vector{Float64}
    panels::Vector{Any}
    kind::String = "shocks"
    type::String = ""
    steadystate::Bool = false
    title::String = ""
    parameter_set::String = "calibration"
end

"""
    stack_bars(contributions) -> (bottom, top)

Bottom and top of every bar of a stacked decomposition: positive contributions pile up from zero in component order, negative ones pile down
(`graph_decomp.m`). The shape is that of `contributions` (components, periods); NaN stays NaN.
"""
function stack_bars(c::AbstractMatrix)
    pos = [x > 0 ? x : 0.0 for x in c]
    neg = [x > 0 ? 0.0 : nan_to_zero(x) for x in c]
    base_p = cumsum(pos; dims=1) .- pos
    base_n = cumsum(neg; dims=1) .- neg
    bottom = [c[i, j] > 0 ? base_p[i, j] : base_n[i, j] for i in axes(c, 1), j in axes(c, 2)]
    bottom, bottom .+ c
end

"""
    decomposition_plot_data(decomp, var_list=nothing; model=nothing, use_shock_groups=nothing, groups=nothing, expand=nothing, init2shocks=nothing,
                            icd=nothing, type="", diff=false, flip=false, steadystate=false, plot_init_date=nothing, plot_end_date=nothing,
                            screen_shocks=false, first_quarter=1, steady_state=nothing) -> DecompositionPlotData

The numbers of Dynare's `plot_shock_decomposition` (and of the figure of `initial_condition_decomposition` when `decomp` is one). `decomp`: a
`ShockDecomposition` (ungrouped shocks, already grouped, or initial conditions); for realtime results pass `select_vintage(rt, realtime, vintage)`.
`var_list`: variables to show (default: the model's original variables, else all). `use_shock_groups` (name of a `shock_groups` block of
`model`) or `groups` (`label => [shocks]` pairs); `expand`: label of a group to show shock by shock. `init2shocks`: name of an `init2shocks`
block or a vector of `(endogenous, shock)` pairs, with `icd` the initial-condition decomposition of the same smoother run. `diff`: first
difference (first period NaN, steady state 0). `type`: `""`/`"qoq"` (as is), `"yoy"` (sum of four consecutive periods, steady state times 4) or
`"aoa"` (every fourth period, for variables already annualised in the model; the sample is taken to start in quarter `first_quarter`). `flip`:
change the sign. `steadystate`: show the steady state on the axis. `plot_init_date` / `plot_end_date`: first and last period shown, as integers
in the numbering of `periods`. `screen_shocks`: with more than 18 components and no groups, keep the 16 shocks with the largest mean absolute
contribution for each variable and sum the rest in "Others". `steady_state`: overrides the steady state stored in `decomp` (Dynare keeps the one
of the last solved parameter set, which after an estimation is the posterior mean's, not the mode's).
"""
function decomposition_plot_data(decomp::ShockDecomposition, var_list=nothing; model=nothing, use_shock_groups=nothing, groups=nothing,
                                 expand=nothing, init2shocks=nothing, icd=nothing, type="", diff=false, flip=false, steadystate=false,
                                 plot_init_date=nothing, plot_end_date=nothing, screen_shocks=false, first_quarter=1, steady_state=nothing)
    type = type === nothing ? "" : String(type)
    type in ("", "qoq", "yoy", "aoa") || throw(SmootherError("plot_shock_decomposition: wrong type (use \"\", \"qoq\", \"yoy\" or \"aoa\")"))
    is_icd = decomp.kind == "initial_conditions"
    title = is_icd ? "Initial condition decomposition" : "Historical shock decomposition"
    pairs = nothing
    will_group = (use_shock_groups !== nothing || groups !== nothing || expand !== nothing) && decomp.kind == "shocks"
    if init2shocks !== nothing && !(init2shocks isa AbstractVector && isempty(init2shocks))
        pairs = init2shocks isa AbstractString ? _init2shocks_block(model, init2shocks) : init2shocks
        decomp = apply_init2shocks(decomp, icd, pairs; rename=!will_group)
    end
    grouped = false
    if (use_shock_groups !== nothing || groups !== nothing || expand !== nothing) && !is_icd
        decomp.kind == "shocks" || throw(SmootherError("the decomposition is grouped already"))
        decomp = group_shocks(decomp; groups=groups, model=model, name=use_shock_groups, expand=expand)
        grouped = true
        use_shock_groups === nothing || (title *= " group $use_shock_groups")
    elseif decomp.kind == "groups"
        grouped = true
    end
    if is_icd
        screen_shocks = true                          # forced by Dynare's initial_condition_decomposition
        labels = vcat(decomp.columns[1:end-2], ["Shocks"])
    elseif grouped
        labels = decomp.columns[1:end-1]
    else
        labels = vcat(decomp.columns[1:end-2], ["Initial values"])
    end
    names = decomp.endo_names
    if var_list === nothing
        orig = model === nothing ? String[] : orig_endo_names(model)
        var_list = [v for v in orig if v in names]
        isempty(var_list) && (var_list = copy(names))
    end
    var_list = unique(String.(var_list))
    miss = [v for v in var_list if !(v in names)]
    isempty(miss) || throw(SmootherError("variable(s) $miss not in the decomposition"))
    iv = [findfirst(==(v), names) for v in var_list]
    z = decomp.z[iv, :, :]
    ss = steady_state === nothing ? decomp.steady_state : Vector{Float64}(steady_state)
    ss = ss === nothing ? zeros(length(names)) : copy(ss)
    length(ss) == length(names) && (ss = ss[iv])
    periods = decomp.first_period .+ collect(0:size(z, 3)-1)
    flip && (title *= " flip")
    pairs === nothing || (title *= " init2shocks")
    if diff
        title *= " diff"
        z[:, :, 2:end] = z[:, :, 2:end] - z[:, :, 1:end-1]
        z[:, :, 1] .= NaN
        ss = ss .* 0
    end
    if type == "yoy"
        size(z, 3) >= 4 || throw(SmootherError("type=\"yoy\" needs at least four periods"))
        z = z[:, :, 1:end-3] + z[:, :, 2:end-2] + z[:, :, 3:end-1] + z[:, :, 4:end]
        periods = periods[4:end]
        ss = 4 .* ss
    elseif type == "aoa"
        first_quarter in 1:4 || throw(SmootherError("first_quarter must be 1, 2, 3 or 4"))
        z = z[:, :, 5-first_quarter:4:end]
        periods = collect(1:size(z, 3))              # years
    end
    if flip
        z = -z
        ss = -ss
    end
    a, b = 1, size(z, 3)
    if plot_init_date !== nothing
        hit = findfirst(==(Int(plot_init_date)), periods)
        hit === nothing && throw(SmootherError("plot_init_date=$plot_init_date is outside the periods $(periods[1])..$(periods[end]) of the decomposition"))
        a = hit
    end
    if plot_end_date !== nothing
        if Int(plot_end_date) <= periods[end]
            hit = findfirst(==(Int(plot_end_date)), periods)
            hit === nothing && throw(SmootherError("plot_end_date=$plot_end_date is before the first period $(periods[1])"))
            b = hit
        else
            @warn "plot_end_date=$plot_end_date is after the last period $(periods[end]): the last period is used"
        end
    end
    zc, pc = z[:, :, a:b], periods[a:b]
    ncomp = size(z, 2) - 1
    screen = screen_shocks && !grouped && ncomp > 18
    panels = Any[]
    for (k, v) in enumerate(var_list)
        z1 = zc[k, :, :]
        lab = copy(labels)
        if screen
            size_ = [sum(abs.(z1[i, :])) / size(z1, 2) for i in 1:size(z1, 1)-2]
            order = sortperm(-size_; alg=MergeSort)   # stable; NaN (diff shown from its first period) keeps the order
            z1 = vcat(z1[order[1:16], :], sum(z1[order[17:end], :]; dims=1), z1[ncomp:end, :])
            lab = vcat(labels[order[1:16]], ["Others", labels[end]])
        end
        contrib = z1[1:end-1, :]
        clean = nan_to_zero.(contrib)
        ymax = maximum(sum(max.(clean, 0.0); dims=1)) * 1.1
        ymin = minimum(sum(min.(clean, 0.0); dims=1)) * 1.1
        ymax - ymin < 1e-6 && continue
        push!(panels, (name=v, labels=lab, contributions=contrib, total=z1[end, :], periods=pc, steady_state=ss[k]))
    end
    screen && (title *= " SCREEN")
    DecompositionPlotData(var_list=var_list, z=z, labels=labels, periods=periods, steady_state=ss, panels=panels, kind=decomp.kind, type=type,
                          steadystate=steadystate, title=title, parameter_set=decomp.parameter_set)
end

"""
    decomposition_table(d::DecompositionPlotData; panels=true) -> Vector{NamedTuple}
    decomposition_table(dec::ShockDecomposition; vars=nothing) -> Vector{NamedTuple}

Tidy table of the content of a decomposition figure (for `write_csv`): one row per variable, component and period with columns
`variable, component, period, value`; the smoothed line is the component "smoothed". For a `DecompositionPlotData`, the plotted panels
(after cropping and screening); with `panels=false` the full `z`. For a `ShockDecomposition`, every column of `z` for `vars` (default all
variables), periods numbered from `first_period`.
"""
function decomposition_table(d::DecompositionPlotData; panels::Bool=true)
    rows = NamedTuple{(:variable, :component, :period, :value),Tuple{String,String,Int,Float64}}[]
    if panels
        for p in d.panels, (i, lab) in enumerate(p.labels), (t, per) in enumerate(p.periods)
            push!(rows, (variable=p.name, component=lab, period=per, value=p.contributions[i, t]))
        end
        for p in d.panels, (t, per) in enumerate(p.periods)
            push!(rows, (variable=p.name, component="smoothed", period=per, value=p.total[t]))
        end
    else
        comps = vcat(d.labels, ["smoothed"])
        for (k, v) in enumerate(d.var_list), (i, lab) in enumerate(comps), (t, per) in enumerate(d.periods)
            push!(rows, (variable=v, component=lab, period=per, value=d.z[k, i, t]))
        end
    end
    rows
end
function decomposition_table(dec::ShockDecomposition; vars=nothing)
    vars = vars === nothing ? dec.endo_names : String.(vars)
    rows = NamedTuple{(:variable, :component, :period, :value),Tuple{String,String,Int,Float64}}[]
    for v in vars
        k = findfirst(==(v), dec.endo_names)
        k === nothing && throw(SmootherError("variable $v is not in the decomposition"))
        for (i, lab) in enumerate(dec.columns), t in 1:size(dec.z, 3)
            push!(rows, (variable=v, component=lab, period=dec.first_period + t - 1, value=dec.z[k, i, t]))
        end
    end
    rows
end
