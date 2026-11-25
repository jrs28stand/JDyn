# Results-layer glue of the parity2 batch (port of pydyn's second parity batch): functions MATLAB lines of a .mod file call and
# that need the workspace of the running `dynare()` session. Registered with the interpreter (PKG_FUNCS / bi!) as pac.jl and
# pf_functions.jl do; the workspace is found through `_pfo_ws(I)`.

# ---------------------------------------------------------------------------------------------- occbin.forecast
"The `occbin.simul` options of `options_` as `simulate` takes them (the same reading as `occbin_solver!`)."
function _p2_occbin_sim_kw(O::MStruct)
    occ_opts = _getpath(O, "occbin", MStruct())
    flat = occ_opts isa MStruct ? _flat_options(occ_opts) : Dict{String,Any}()
    kw = Dict{Symbol,Any}()
    for (key, (sym, T)) in _OCCBIN_MOD_OPTIONS
        haskey(flat, key) || continue
        v = flat[key]
        kw[sym] = T === Bool ? _asbool(v) : T === Int ? Int(round(_asnum(v))) : _asnum(v)
    end
    kw
end

_p2_empty(v) = v === nothing || (v isa AbstractArray && isempty(v)) || (v isa AbstractString && isempty(v))

function _p2_stat_struct(d::Dict{String,Dict{String,Matrix{Float64}}}, names)
    out = MStruct()
    for stat in OCCBIN_FORECAST_STATISTICS
        haskey(d, stat) || continue
        s = MStruct()
        for nm in names
            s[nm] = d[stat][nm]
        end
        out[stat] = s
    end
    out
end

"`[forecast, error_flag, yf] = occbin.forecast(options_, M_, dr, ys, exo_ys, exo_det_ys, horizon)` for the workspace's model at its current parameters."
function _p2_occbin_forecast(I, a, nout)
    length(a) == 7 || merr("occbin.forecast needs seven arguments")
    ws = _pfo_ws(I)
    horizon = Int(round(scalar(a[7])))
    _model_to_M!(ws)                                   # while the file runs the model is the source of truth for M_.params / Sigma_e
    opts = _pfo_options_struct(ws, a[1])
    Mst = ws.M_
    m = ws.model
    _optb(ws, "loglinear", false) && throw(Unsupported("occbin.forecast with loglinear: Dynare 7.2 mixes the level model with log-linearised rules here (as occbin_solver)", "feature"))
    g(path, d) = (v = _getpath(opts, "occbin.forecast." * path, d); _p2_empty(v) ? d : v)
    eh = haskey(Mst, "endo_histval") ? Float64.(vec(farr(Mst["endo_histval"]))) : Float64[]
    isempty(eh) && merr("occbin.forecast: M_.endo_histval is empty (set the initial state first)")
    ys = Float64.(vec(farr(a[4])))
    replic = Int(round(Float64(scalar(g("replic", 0.0)))))
    sh = _getpath(opts, "occbin.forecast.SHOCKS0", nothing)
    shocks = nothing
    if sh isa Cell && !isempty(sh.items)
        shocks = [(String(c.items[1]), Float64.(vec(farr(c.items[2])))) for c in (first(sh.items) isa Cell && length(sh.items) == 1 && first(sh.items).items[1] isa Cell ? first(sh.items).items : sh.items)]
    elseif sh isa AbstractArray && !isempty(sh)
        shocks = Float64.(sh)
    end
    ob = Occbin(m)
    if replic == 0
        # forecast.m (replic = 0) leaves options_.occbin.simul.exo_pos at what occbin.setup put there: the columns of the
        # surprise shocks. With fewer shocks than the model has, the SHOCKS (all of them) do not fit and Dynare stops.
        pos = _getpath(opts, "occbin.simul.exo_pos", nothing)
        if _p2_empty(pos)
            occ = get(ws.oo_, "occbin", nothing)
            sim = occ isa MStruct ? get(occ, "simul", nothing) : nothing
            pos = sim isa MStruct ? get(sim, "exo_pos", nothing) : nothing
        end
        if !_p2_empty(pos) && length(pos) != m.n_exo
            throw(Unsupported("occbin.forecast: with replic = 0 Dynare's solver uses options_.occbin.simul.exo_pos left by occbin.setup ($(length(pos)) of " *
                              "$(m.n_exo) shocks) and stops with a size mismatch (a defect of +occbin/forecast.m)", "feature"))
        end
    end
    fr = _getpath(opts, "occbin.forecast.frcst_regimes", nothing)
    frcst = fr isa Cell && !isempty(fr.items) ? _regimes_from_cell(fr) : nothing
    kw = _p2_occbin_sim_kw(opts)
    for k in (:maxit, :check_ahead_periods, :periods)
        delete!(kw, k)
    end
    simopts = merge(Dict{Symbol,Any}(pairs(mod_options(ob))), kw)
    conf = _getpath(opts, "forecasts.conf_sig", 0.9)
    conf = _p2_empty(conf) ? 0.9 : Float64(scalar(conf))
    fc = occbin_forecast(ob, horizon; endo_histval=eh, shocks=shocks, frcst_regimes=frcst, replic=replic,
                         qmc=_asbool(g("qmc", false)), maxit=Int(round(Float64(scalar(g("maxit", 30.0))))),
                         check_ahead_periods=Int(round(Float64(scalar(g("check_ahead_periods", 30.0))))),
                         conf_sig=conf, sigma_e=Float64.(farr(Mst["Sigma_e"])), rng=dynare_rng(ws), simul_options=simopts)
    out = MStruct()
    if fc.yf === nothing                                   # replic = 0 and the simulation failed: Dynare returns forecast without the fields
        fc.replic == 0 && fc.error_flag != 0 && @warn "occbin.forecast: forecast simulation failed (error flag $(fc.error_flag))."
        return Any[out, Float64(fc.error_flag), zeros(0, 0)][1:max(nout, 1)]
    end
    out.linear = _p2_stat_struct(fc.linear, m.endo_names)
    out.piecewise = _p2_stat_struct(fc.piecewise, m.endo_names)
    if fc.replic == 0
        out.regimes = _regime_history_cell(fc.regimes)
    else
        T = horizon
        items = Any[zeros(0, 0) for _ in 1:(fc.replic * T)]
        for (it, rh) in enumerate(fc.regimes)
            rh === nothing && continue
            c = _regime_history_cell(rh)
            for t in 1:T
                items[it+(t-1)*fc.replic] = c.items[t]
            end
        end
        out.regimes = Cell(items, (fc.replic, T))
    end
    err = fc.replic == 0 ? 0.0 : reshape(Float64.(fc.failed), :, 1)
    Any[out, err, fc.yf][1:max(nout, 1)]
end

# ---------------------------------------------------------------------------------------------- qmc_sequence (Dynare's mex file)
function _p2_qmc_sequence(I, a, nout)
    _nargs(a, 4, 5, "qmc_sequence")
    lu = length(a) > 4 ? (a[5] isa Real ? Float64(a[5]) : Float64.(farr(a[5]))) : nothing
    seq, seed = qmc_sequence(Int(scalar(a[1])), Int(scalar(a[2])), Int(scalar(a[3])), Int(scalar(a[4])), lu)
    Any[seq, Float64(seed)][1:max(nout, 1)]
end

function _register_parity2!()
    PKG_FUNCS["occbin.forecast"] = _p2_occbin_forecast
    bi!(_p2_qmc_sequence, "qmc_sequence")
    nothing
end
_register_parity2!()

# ---------------------------------------------------------------------------------------------- send_*_to_workspace
# Dynare's send_endogenous_variables_to_workspace.m & co.: assignin('base', name, series') for every variable.
function _p2_register_send!()
    bi!("send_endogenous_variables_to_workspace") do I, a, nout
        ws = _pfo_ws(I)
        sim = get(ws.oo_, "endo_simul", nothing)
        sim === nothing && merr("Reference to non-existent field 'endo_simul'.")
        sim = Matrix{Float64}(farr(sim))
        for (i, nm) in enumerate(ws.model.endo_names)
            i <= size(sim, 1) || merr("index ($i,_): out of bound $(size(sim, 1))")
            I.base.vars[nm] = reshape(sim[i, :], :, 1)
        end
        Any[]
    end
    bi!("send_exogenous_variables_to_workspace") do I, a, nout
        ws = _pfo_ws(I)
        sim = get(ws.oo_, "exo_simul", nothing)
        sim = sim === nothing ? zeros(0, 0) : Matrix{Float64}(farr(sim))
        for (i, nm) in enumerate(ws.model.exo_names)
            i <= size(sim, 2) || merr("index (_,$i): out of bound $(size(sim, 2))")
            I.base.vars[nm] = reshape(sim[:, i], 1, :)
        end
        Any[]
    end
    bi!("send_irfs_to_workspace") do I, a, nout
        ws = _pfo_ws(I)
        irfs = get(ws.oo_, "irfs", nothing)
        if irfs isa MStruct
            for nm in keys(irfs)
                v = farr(irfs[nm])
                I.base.vars[nm] = v isa AbstractMatrix ? Matrix{Float64}(permutedims(v)) : v      # transposed, as Dynare
            end
        end
        Any[]
    end
    nothing
end
_p2_register_send!()

# ---------------------------------------------------------------------------------------------- smoother2histval
_p2_namelist(x) = x isa AbstractString ? String[x] : x isa AbstractDict ? String.(x["symbol_list"]) : String[String(v) for v in x]

"""
`smoother2histval(options)` (matlab/smoother2histval.m): the smoothed values of `period` (and the `maximum_lag - 1` periods before it)
become `M_.endo_histval` / `M_.exo_histval`, or, with `outfile`, a dseries saved in a MAT-file (`histval_file` reads it).
"""
function _smoother2histval!(ws, st)
    m = ws.model
    opts = get_or(st, "options", Dict{String,Any}())
    oo = ws.oo_
    sv, ss = get(oo, "SmoothedVariables", nothing), get(oo, "SmoothedShocks", nothing)
    sv isa MStruct || throw(ModelError("smoother2histval: Could not find smoothed variables, did you compute them? " *
        "(either by the \"smoother\" option of the \"estimation\" command, or the \"calib_smoother\" command"))
    names_v = collect(keys(sv))
    post_metropolis = !isempty(names_v) && sv[names_v[1]] isa MStruct
    if post_metropolis
        sv, ss = sv["Mean"], ss["Mean"]
        names_v = collect(keys(sv))
        length(names_v) != m.n_endo && @warn "You are using smoother2histval although smoothed values have not been computed for all endogenous and auxiliary variables. The value of these variables will be set to their steady state."
    end
    names_x = ss isa MStruct ? collect(keys(ss)) : String[]
    n = length(farr(sv[names_v[1]]))
    ml = Int(ws.M_.maximum_lag)
    oml = Int(get(ws.M_, "orig_maximum_lag", ml))
    n < Int(ws.M_.maximum_endo_lag) && throw(ModelError("smoother2histval: Not enough observations to create initial conditions"))
    invars = haskey(opts, "invars") ? _p2_namelist(opts["invars"]) : vcat(names_v, names_x)
    period = n
    if haskey(opts, "period")
        period = Int(_numval(opts["period"]))
        period > n && throw(ModelError("smoother2histval: The period that you indicated is beyond the data sample"))
        period < Int(ws.M_.maximum_endo_lag) && throw(ModelError("smoother2histval: The period that you indicated is too small to construct initial conditions"))
    end
    outvars = invars
    if haskey(opts, "outvars")
        outvars = _p2_namelist(opts["outvars"])
        length(invars) == length(outvars) || throw(ModelError("smoother2histval: The number of input and output variables is not the same"))
    end
    outfile = get(opts, "outfile", nothing)
    sm = get(oo, "Smoother", nothing)
    loglin = sm isa MStruct && haskey(sm, "loglinear") && sm["loglinear"] != 0
    ssv = vec(Float64.(farr(oo["steady_state"])))
    xsv = m.n_exo > 0 ? vec(Float64.(farr(oo["exo_steady_state"]))) : Float64[]
    local eh, xh, data
    if outfile === nothing
        eh = repeat(reshape(ssv, :, 1), 1, ml)
        xh = repeat(reshape(xsv, :, 1), 1, ml)
    else
        data = zeros(oml, length(invars))
        for (i, nm) in enumerate(outvars)
            j = findfirst(==(nm), m.endo_names)
            j === nothing || (data[:, i] .= ssv[j])
        end
    end
    for (i, nm) in enumerate(invars)
        s = if nm in m.endo_names
            haskey(sv, nm) || throw(ModelError("smoother2histval: Reference to non-existent field '$nm'."))
            x = vec(Float64.(farr(sv[nm]))); loglin ? exp.(x) : x
        elseif nm in m.exo_names
            vec(Float64.(farr(ss[nm])))
        else
            throw(ModelError("smoother2histval: unknown input variable"))
        end
        if outfile === nothing
            v = s[period-ml+1:period]
            je, jx = findfirst(==(outvars[i]), m.endo_names), findfirst(==(outvars[i]), m.exo_names)
            je === nothing || (eh[je, :] = v)
            jx === nothing || (xh[jx, :] = v)
            (je === nothing && jx === nothing) && throw(ModelError("smoother2histval: output variable $(outvars[i]) does not exist."))
        else
            data[:, i] = s[period-oml+1:period]
        end
    end
    if outfile === nothing
        ws.M_.endo_histval = eh
        ws.M_.exo_histval = m.n_exo > 0 ? xh : zeros(0, 0)
        return nothing
    end
    d, f = splitdir(String(outfile))
    base, ext = splitext(f)
    (ext == ".mat" || isempty(ext)) || throw(ModelError("smoother2hisvtval: if outfile has an extension, it must be .mat"))
    I = m.session.I
    o = ds_ctor(I, Any[data, "1Y", Cell(Any[invars...], (1, length(invars)))])
    OBJM["dseries"]["save"](I, Any[o, d * base], 0)         # Dynare writes [dir fname]: the folder is glued to the name
    nothing
end

const _P2_KINDS = Set(["smoother_2_histval", "smoother2histval", "shock_decomposition"])
"Statements of the parity2 batch, called from `_run_in_place!` (src/results.jl)."
function _p2_statement!(ws, kind, st)
    kind in ("smoother_2_histval", "smoother2histval") && return _smoother2histval!(ws, st)
    kind == "shock_decomposition" && return _sd_statement!(ws, st)
    nothing
end
