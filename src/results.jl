# Dynare-style results: M_, options_ and oo_ as MATLAB structs (MStruct), filled the way Dynare's
# commands fill them, so results arrive as in MATLAB: oo_.endo_simul, oo_.dr.ghx, oo_.irfs.y_e, ...
#
# The commands of the .mod file run after the file has been loaded (with the final calibration), in file
# order; MATLAB lines placed between commands do not see their results yet (docs/RESULTS.md). Afterwards the
# same commands (stoch_simul!, perfect_foresight_solver!, ...) re-run on the workspace without preprocessing.

"What Dynare leaves in MATLAB's workspace after `dynare model`."
struct DynareWorkspace
    model::Model
    M_::MStruct
    options_::MStruct
    oo_::MStruct
    vars::Dict{String,Any}          # other variables of the base workspace (parameters, MATLAB variables of the .mod file)
    state::Dict{Symbol,Any}         # last synchronised params / Sigma_e, MATLAB options already applied
end
DynareWorkspace(m, M, O, oo, vars) = DynareWorkspace(m, M, O, oo, vars, Dict{Symbol,Any}())
Base.show(io::IO, w::DynareWorkspace) = print(io, "DynareWorkspace(", w.model.name, ": M_, options_, oo_, ", length(w.vars), " variables)")

_col(v) = reshape(Float64.(collect(v)), :, 1)
_row(v) = reshape(Float64.(collect(v)), 1, :)
_cellcol(v) = Cell(Any[String(x) for x in v], (length(v), 1))
_cellrow(v) = Cell(Any[x for x in v], (1, length(v)))

"s.a.b.c = v, creating the intermediate structs."
function _setpath!(s::MStruct, path::AbstractString, v)
    parts = split(path, ".")
    for p in parts[1:end-1]
        sub = get(s, p, nothing)
        if !(sub isa MStruct)
            sub = MStruct()
            s[p] = sub
        end
        s = sub
    end
    s[parts[end]] = v
end

"A JSON option value as MATLAB would hold it."
function _mval(v)
    v isa Bool && return v
    v isa Real && return Float64(v)
    v isa AbstractString && return String(v)
    v === nothing && return zeros(0, 0)
    if v isa AbstractVector
        isempty(v) && return zeros(0, 0)
        all(x -> x isa Real && !(x isa Bool), v) && return _row(v)
        all(x -> x isa AbstractString, v) && return _cellrow(String.(v))
        all(x -> x isa AbstractVector && all(y -> y isa Real, x), v) && return Matrix{Float64}(reduce(vcat, [_row(x) for x in v]))
        return _cellrow(map(_mval, v))
    end
    if v isa AbstractDict
        s = MStruct()
        for (k, x) in v
            _setpath!(s, String(k), _mval(x))
        end
        return s
    end
    v
end

# ---------------------------------------------------------------------- M_
"Dynare's M_ for a loaded model (the fields JDyn knows; see docs/RESULTS.md for the full list)."
function build_M(m::Model)
    M = MStruct()
    info = m.raw["model_info"]
    M.fname = m.name
    M.dname = m.name
    M.endo_names = _cellcol(m.endo_names)
    M.exo_names = _cellcol(m.exo_names)
    M.param_names = _cellcol(m.param_names)
    M.endo_names_long = _cellcol([get(d, "longName", d["name"]) for d in m.raw["endogenous"]])
    M.exo_names_long = _cellcol([get(d, "longName", d["name"]) for d in m.raw["exogenous"]])
    M.param_names_long = _cellcol([get(d, "longName", d["name"]) for d in m.raw["parameters"]])
    M.endo_names_tex = _cellcol([get(d, "texName", d["name"]) for d in m.raw["endogenous"]])
    M.exo_names_tex = _cellcol([get(d, "texName", d["name"]) for d in m.raw["exogenous"]])
    M.param_names_tex = _cellcol([get(d, "texName", d["name"]) for d in m.raw["parameters"]])
    M.endo_nbr = Float64(m.n_endo)
    M.exo_nbr = Float64(m.n_exo)
    M.exo_det_nbr = 0.0
    M.param_nbr = Float64(length(m.param_names))
    M.orig_endo_nbr = Float64(orig_endo_nbr(m))
    M.eq_nbr = Float64(m.n_endo)
    for k in ("maximum_lag", "maximum_lead", "maximum_endo_lag", "maximum_endo_lead", "maximum_exo_lag", "maximum_exo_lead",
              "orig_maximum_endo_lag", "orig_maximum_endo_lead", "orig_maximum_exo_lag", "orig_maximum_exo_lead",
              "orig_maximum_lag", "orig_maximum_lead")
        haskey(info, k) && (M[k] = Float64(info[k]))
    end
    # lead_lag_incidence: (lags + 1 + leads) x endo_nbr, as in the MATLAB driver
    lli = reduce(vcat, [reshape(Int.(r), 1, :) for r in info["lead_lag_incidence"]])
    M.lead_lag_incidence = Matrix{Float64}(permutedims(lli))
    _, _, cnt = dr_ordering(m.lead_lag_incidence)
    M.nstatic = Float64(cnt.nstatic); M.npred = Float64(cnt.npred); M.nboth = Float64(cnt.nboth); M.nfwrd = Float64(cnt.nfwrd)
    M.nspred = Float64(cnt.npred + cnt.nboth); M.nsfwrd = Float64(cnt.nboth + cnt.nfwrd)
    M.ndynamic = Float64(cnt.npred + cnt.nboth + cnt.nfwrd)
    M.params = _col(m.params)
    M.Sigma_e = copy(m.Sigma_e)
    M.Correlation_matrix = _corr(m.Sigma_e)
    M.sigma_e_is_diagonal = isdiag(m.Sigma_e)
    M.H = Matrix{Float64}(LinearAlgebra.Diagonal([get(m.measurement_errors, v, 0.0) for v in m.varobs]))
    aux = Any[]
    for a in get_or(m.raw, "aux_vars", [])
        s = MStruct()
        for (k, v) in a
            s[k] = _mval(v)
        end
        push!(aux, s)
    end
    M.aux_vars = Cell(aux, (1, length(aux)))
    M.endo_histval = zeros(0, 0)
    M.dynare_version = "7.2"
    M
end
_corr(S) = (d = sqrt.(diag(S)); [d[i] > 0 && d[j] > 0 ? S[i, j] / (d[i] * d[j]) : (i == j ? 1.0 : 0.0) for i in axes(S, 1), j in axes(S, 2)])

# ---------------------------------------------------------------------- options_
# where the preprocessor's MATLAB driver writes each statement's options (ComputingTasks.cc); every other
# statement writes into options_ itself. Statements writing to a separate struct (options_ident, options_mom_,
# options_occbin_ for occbin_setup, ...) do not change options_ and are skipped here.
const _OPTION_PREFIX = Dict("occbin_solver" => "occbin.", "occbin_write_regimes" => "occbin.", "data" => "dataset.")
const _OPTION_ELSEWHERE = Set(["occbin_setup", "occbin_graph", "identification", "method_of_moments", "model_info",
                               "conditional_forecast", "dynare_sensitivity", "model_diagnostics", "heterogeneity_solve"])

"""options_ before the first statement: Dynare's defaults known to JDyn and the MATLAB assignments of the
.mod file. Each statement then writes its options when it runs (`_apply_options!`), as in Dynare's driver."""
function build_options(m::Model)
    O = MStruct()
    for (k, v) in OPTION_DEFAULTS
        _setpath!(O, k, v isa Bool ? v : Float64(v))
    end
    O.impulse_responses = MStruct(["plot_threshold" => 1e-10])
    O.qz_criterium = zeros(0, 0)                      # Dynare: empty until a command selects it
    for k in keys(m.matlab_options)
        O[k] = m.matlab_options[k]
    end
    O.dynare_version = "7.2"
    O
end

"Write a statement's options into options_, at the path Dynare's driver uses (the preprocessor names them by it)."
function _apply_options!(O::MStruct, st)
    kind = st["statementName"]
    opts = get_or(st, "options", nothing)
    (opts isa AbstractDict && !(kind in _OPTION_ELSEWHERE)) || return O
    pre = get(_OPTION_PREFIX, kind, "")
    for (k, v) in opts
        _setpath!(O, pre * String(k), _mval(v))
    end
    O
end

"Keyword options of a command call, written into options_ as Dynare's driver would (they persist)."
function _take_options!(ws, kw)
    for (k, v) in kw
        _setpath!(ws.options_, String(k), v isa AbstractVector{<:AbstractString} ? _cellrow(String.(v)) : _mval(v))
    end
end

function _getpath(s::MStruct, path::AbstractString, default)
    for p in split(path, ".")
        (s isa MStruct && haskey(s, p)) || return default
        s = s[p]
    end
    s
end
_optf(ws, k, d) = (v = _getpath(ws.options_, k, d); v isa AbstractMatrix && isempty(v) ? d : v)
_opti(ws, k, d) = Int(_optf(ws, k, d))
_optb(ws, k, d) = (v = _optf(ws, k, d); v isa Bool ? v : v != 0)

"""
Bring the model and M_ in line before a command. M_ is the handle Dynare users edit (`M_.params(i) = x`);
the Model API (`set_param!(ws.model, ...)`) works too: whichever side changed since the last command wins
(M_ if both did, with a warning).
"""
function _sync_model!(ws)
    m = ws.model
    pM = vec(Float64.(ws.M_.params))
    length(pM) == length(m.params) || throw(ModelError("M_.params has $(length(pM)) elements, the model has $(length(m.params)) parameters"))
    last = get(ws.state, :params, nothing)
    if last === nothing || !isequal(pM, last)
        last !== nothing && !isequal(m.params, last) && !isequal(m.params, pM) &&
            warn_once(m, "both M_.params and the model's parameters were changed since the last command: M_.params is used")
        m.params .= pM
    elseif !isequal(m.params, last)
        ws.M_.params = _col(m.params)
    end
    S = ws.M_.Sigma_e
    size(S) == size(m.Sigma_e) || throw(ModelError("M_.Sigma_e has size $(size(S)), expected $(size(m.Sigma_e))"))
    lastS = get(ws.state, :Sigma_e, nothing)
    if lastS === nothing || !isequal(S, lastS)
        m.Sigma_e .= S
    elseif !isequal(m.Sigma_e, lastS)
        ws.M_.Sigma_e = copy(m.Sigma_e)
    end
    ws.M_.Correlation_matrix = _corr(m.Sigma_e)
    _mark_synced!(ws)
    m
end
_mark_synced!(ws) = (ws.state[:params] = copy(ws.model.params); ws.state[:Sigma_e] = copy(ws.model.Sigma_e); ws)
"While a .mod file runs, the model is the source of truth (MATLAB lines write it): copy it into M_."
function _model_to_M!(ws)
    ws.M_.params = _col(ws.model.params)
    ws.M_.Sigma_e = copy(ws.model.Sigma_e)
    _mark_synced!(ws)
end

# ---- options_ as the source of truth for the solvers (R26)
"options_ flattened to the dotted keys the solvers read (as the preprocessor writes them in modfile.json)."
function _flat_options(s::MStruct, pre="", out=Dict{String,Any}())
    for k in keys(s)
        v = s[k]
        key = pre * k
        key in ("dynare_version", "occbin", "dataset") && continue
        if v isa MStruct
            _flat_options(v, key * ".", out)
            continue
        end
        jv = _jval(v)
        jv === nothing && continue
        haskey(OPTION_DEFAULTS, key) && isequal(jv, _jval(OPTION_DEFAULTS[key])) && continue   # a default: as if unset
        out[key] = jv
    end
    out
end
function _jval(v)
    v isa Bool && return v
    v isa AbstractFloat && return isinteger(v) && abs(v) < 2^53 ? Int(v) : v
    v isa Real && return v
    v isa AbstractString && return v
    v isa Cell && return Any[_jval(x) for x in v]
    if v isa AbstractMatrix
        isempty(v) && return nothing
        length(v) == 1 && return _jval(v[1])
        size(v, 1) == 1 && return Any[_jval(x) for x in vec(v)]
        return v
    end
    v
end
"Run `f` with the solvers reading `ws.options_` (in place of the .mod file's statement options)."
function _with_options(f, ws, kinds)
    m = ws.model
    flat = _flat_options(ws.options_)
    saved = Dict(k => get(m.options, k, nothing) for k in kinds)
    saved_initval = m.initval
    isempty(m.oo) || (m.initval = merge(m.initval, m.oo))          # Dynare starts from oo_.steady_state
    for k in kinds
        # steady reads only its own options (the rest of options_ would show up as "not acted on")
        m.options[k] = k == "steady" ? filter(kv -> kv.first in _STEADY_KEYS, flat) : copy(flat)
    end
    try
        return f()
    finally
        for (k, v) in saved
            v === nothing ? delete!(m.options, k) : (m.options[k] = v)
        end
        m.initval = saved_initval
    end
end
"Eigenvalues as a column; real when they all are (MATLAB shows those as real too)."
_eigcol(e) = all(x -> imag(x) == 0, e) ? reshape(Float64.(real.(e)), :, 1) : reshape(ComplexF64.(e), :, 1)
const _SOLVE_KINDS = ("options_", "check", "stoch_simul", "estimation", "calib_smoother", "steady")   # "options_": the accumulated table stoch_options reads
const _STEADY_KEYS = Set(["solve_tolf", "steadystate.nocheck", "homotopy_mode", "homotopy_steps"])

"Parameters the steady state (file or steady_state_model block) has set go back to M_.params, as in Dynare."
function _store_params!(ws, p)
    ws.model.params .= p
    ws.M_.params = _col(p)
    _mark_synced!(ws)
end

# ---------------------------------------------------------------------- oo_
function _store_dr!(oo::MStruct, dr::AbstractDict, n::Int)
    D = MStruct()
    old = get(oo, "dr", nothing)
    old isa MStruct && haskey(old, "eigval") && (D.eigval = old.eigval)
    D.ys = _col(dr["ys"])
    D.order_var = _col(dr["order_var"])
    inv = zeros(Int, n); inv[Int.(dr["order_var"])] .= 1:n
    D.inv_order_var = _col(inv)
    D.state_var = _row(dr["state_var"])
    for k in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss")
        haskey(dr, k) && (D[k] = Matrix{Float64}(dr[k]))
    end
    haskey(dr, "ghs2") && (D.ghs2 = _col(dr["ghs2"]))
    oo.dr = D
end

_shock_list(v, m) = v isa AbstractVector{<:AbstractString} ? String.(v) : v isa Cell ? String.(collect(v)) :
                    v isa MStruct && haskey(v, "symbol_list") ? _shock_list(v["symbol_list"], m) : m.exo_names

"""
    stoch_simul!(ws=current workspace, var_list=[]; shocks=nothing, rng=nothing, options...)

Dynare's `stoch_simul` on a workspace: solves at the current `M_.params` and `M_.Sigma_e` with the current
`options_` (keyword options are written into options_ first, as `stoch_simul(order=1, irf=20)` does) and
fills `oo_.dr`, `oo_.irfs`, `oo_.endo_simul`/`oo_.exo_simul` (periods > 0) and the moments. No preprocessing:
change a parameter (`M_.params[i] = x` or `set_param_value!`) and call it again.
"""
function stoch_simul!(ws::DynareWorkspace, var_list=String[]; shocks=nothing, rng=nothing, options...)
    m, oo = _sync_model!(ws), ws.oo_
    _take_options!(ws, options)
    _with_options(() -> _stoch_simul(ws, m, oo, var_list, shocks, rng), ws, _SOLVE_KINDS)
    ws
end

function _stoch_simul(ws, m, oo, var_list, shocks, rng)
    order = _opti(ws, "order", 2)
    pruning = _optb(ws, "pruning", false)
    order > 1 && m.max_deriv_order < order && throw(ModelError("stoch_simul(order=$order) needs derivatives of order $order: run dynare(...; output=\"$(order == 2 ? "second" : "third")\")"))
    sol = first_order(m)                      # also gives dr.eigval, which Dynare's resol stores at every order
    dr, p = order == 1 ? (as_dr(sol), sol.params) : (d = k_order(m; order=order, pruning=pruning); (d, d["params"]))
    _store_params!(ws, p)
    _store_dr!(oo, dr, m.n_endo)
    oo.dr.eigval = _eigcol(sol.eigenvalues)
    oo.steady_state = _col(dr["ys"])
    oo.exo_steady_state = _col(exo_steady(m))
    var_list = isempty(var_list) ? orig_endo_names(m) : String.(collect(var_list))
    oo.var_list = _cellcol(var_list)
    ivar = [m.ie[v] for v in var_list]
    nirf = _opti(ws, "irf", 40)
    if nirf > 0
        irfs = get(oo, "irfs", MStruct())
        cs = get_lower_cholesky_covariance(m.Sigma_e)
        relative = _optb(ws, "relative_irf", false)
        for sname in _shock_list(_getpath(ws.options_, "irf_shocks", nothing), m)
            i = m.ix[sname]
            cs[i, i] > 5e-7 || continue                           # stoch_simul.m: no IRF for (near) zero variance
            impulse = order > 1 && relative ? cs[:, i] ./ cs[i, i] ./ 100 : cs[:, i]   # GIRFs: shock normalised to 0.01
            y = order == 1 ? irf(dr, impulse; periods=nirf) :
                irf(dr, impulse; Sigma_e=m.Sigma_e, periods=nirf, order=order, pruning=pruning,
                    drop=_opti(ws, "drop", 100), replic=_opti(ws, "replic", 50),
                    rng=rng === nothing ? dynare_rng(ws) : rng)
            relative && order == 1 && (y = 100 .* y ./ cs[i, i])
            for j in ivar
                irfs[string(m.endo_names[j], "_", sname)] = _row(y[:, j])
            end
        end
        oo.irfs = irfs
    end
    periods = _opti(ws, "periods", 0)
    if periods > 0
        ex = if shocks !== nothing
            size(shocks) == (periods, m.n_exo) || throw(ArgumentError("shocks must be periods x exo_nbr = ($periods, $(m.n_exo))"))
            Matrix{Float64}(shocks)
        else
            r = rng === nothing ? dynare_rng(ws) : rng
            iv = findall(!=(0), diag(m.Sigma_e))
            e = zeros(periods, m.n_exo)
            e[:, iv] .= randn(r, periods, length(iv)) * Matrix(cholesky(Symmetric(m.Sigma_e[iv, iv])).U)
            e
        end
        # stoch_simul.m: the simulation starts from M_.endo_histval (histval), else from the steady state
        h = ws.M_.endo_histval
        y0 = isempty(h) ? nothing : (_optb(ws, "loglinear", false) ? log.(vec(h[:, end])) : vec(h[:, end]))
        y = simulate(dr, ex; order=order, pruning=pruning, y0=y0)
        oo.endo_simul = Matrix{Float64}(permutedims(y))        # n_endo x periods, as Dynare's simult
        oo.exo_simul = ex
    end
    if !_optb(ws, "nomoments", false) && order <= 2
        mom = try
            periods > 0 ? simulated_moments(permutedims(oo.endo_simul); ivar=ivar, shocks=oo.exo_simul, dr=dr, model=m) :
                          theoretical_moments(m, dr; var_list=var_list)
        catch e
            e isa Union{ArgumentError,ModelError} || rethrow()
            warn_once(m, "stoch_simul: moments not computed ($(sprint(showerror, e)))")
            nothing
        end
        if mom !== nothing
            oo.mean = _col(mom.mean)
            oo.var = Matrix{Float64}(mom.var)
            isempty(mom.autocorr) || (oo.autocorr = _cellrow(mom.autocorr))
            mom.variance_decomposition === nothing || (oo.variance_decomposition = Matrix{Float64}(mom.variance_decomposition))
            mom.conditional_variance_decomposition === nothing ||
                (oo.conditional_variance_decomposition = Array{Float64}(mom.conditional_variance_decomposition))
            if periods == 0
                isempty(mom.gamma_y) || (oo.gamma_y = Cell(Any[g === nothing ? zeros(0, 0) : g for g in mom.gamma_y], (length(mom.gamma_y), 1)))   # column, as Dynare
            else
                mom.skewness === nothing || (oo.skewness = _col(mom.skewness))
                mom.kurtosis === nothing || (oo.kurtosis = _col(mom.kurtosis))
            end
        end
    end
    oo
end

"""
    steady!(ws=current workspace; options...)

Dynare's `steady`: the steady state at the current `M_.params` into `oo_.steady_state`; parameters a
steady-state file or `steady_state_model` block sets go to `M_.params`.
"""
function steady!(ws::DynareWorkspace; options...)
    m = _sync_model!(ws)
    _take_options!(ws, options)
    ys, p = _with_options(() -> steady_state(m), ws, _SOLVE_KINDS)
    _store_params!(ws, p)
    for (v, y) in zip(m.endo_names, ys)
        m.oo[v] = y
    end
    ws.oo_.steady_state = _col(ys)
    ws
end

"""
Dynare's `check`: the generalised eigenvalues (complex) into `oo_.dr.eigval`. As in Dynare, a failed
Blanchard-Kahn order condition is reported (a warning) and the file goes on; other errors are thrown.
"""
function check!(ws::DynareWorkspace; options...)
    m = _sync_model!(ws)
    _take_options!(ws, options)
    eig = try
        sol = _with_options(() -> first_order(m), ws, _SOLVE_KINDS)
        _store_params!(ws, sol.params)
        sol.eigenvalues
    catch e
        (e isa BlanchardKahnError && !isempty(e.eigenvalues)) || rethrow()
        @warn "check: $(e.msg) (the order condition is NOT verified)"
        e.eigenvalues
    end
    D = get(ws.oo_, "dr", MStruct())
    D.eigval = _eigcol(eig)
    ws.oo_.dr = D
    ws
end

"`set_param_value('name', v)`: sets M_.params (and the model's parameter)."
function set_param_value!(ws::DynareWorkspace, name::AbstractString, v::Real)
    i = get(ws.model.ip, name, nothing)
    i === nothing && throw(ArgumentError("unknown parameter '$name'"))
    ws.M_.params[i] = Float64(v)
    ws.model.params[i] = Float64(v)
    ws
end

"""
    perfect_foresight_setup!(ws=current workspace; periods=options_.periods)

Dynare's `perfect_foresight_setup`: builds `oo_.endo_simul` (initial/terminal conditions and the starting
guess) and `oo_.exo_simul` (the shocks of the `shocks` block) for the current `M_.params`. Edit
`oo_.exo_simul` afterwards to try other shock paths.
"""
function perfect_foresight_setup!(ws::DynareWorkspace; options...)
    m = _sync_model!(ws)
    _take_options!(ws, options)
    T = _opti(ws, "periods", 0)
    T > 0 || throw(ArgumentError("perfect_foresight_setup: options_.periods must be positive"))
    pws = _pf_replay(m, nothing, nothing, (), ())
    endo, x = _pf_make_paths(m, pws, T)
    ws.oo_.endo_simul = Matrix{Float64}(endo)
    ws.oo_.exo_simul = Matrix{Float64}(permutedims(x))
    ws.oo_.steady_state = _col(pws.ss)
    ws.oo_.exo_steady_state = _col(pws.xss)
    pws.ss0 === nothing || (ws.oo_.initial_steady_state = _col(pws.ss0))
    ws
end

"""
    perfect_foresight_solver!(ws=current workspace; options...)

Dynare's `perfect_foresight_solver`: solves from the current `oo_.endo_simul` (initial and terminal
conditions, and the starting guess) and `oo_.exo_simul`, as Dynare does, and stores the path in
`oo_.endo_simul`. Loop over shock paths by editing `oo_.exo_simul` and calling it again.
"""
function perfect_foresight_solver!(ws::DynareWorkspace; raise_on_failure::Bool=true, options...)
    m = _sync_model!(ws)
    _take_options!(ws, _pf_option_paths(options))
    T = _opti(ws, "periods", 0)
    lag, lead = Int(m.raw["model_info"]["maximum_lag"]), Int(m.raw["model_info"]["maximum_lead"])
    rows = lag + T + lead
    oo = ws.oo_
    (haskey(oo, "endo_simul") && haskey(oo, "exo_simul") && size(oo.endo_simul) == (m.n_endo, rows) &&
     size(oo.exo_simul) == (rows, m.n_exo)) || perfect_foresight_setup!(ws)
    E = oo.endo_simul
    # every perfect-foresight option options_ holds (set by statements, the user or earlier calls)
    flat = _flat_options(ws.options_)
    kw = Dict{Symbol,Any}()
    for (name, key) in PF_SOLVER_KEYS
        haskey(flat, key) && (kw[Symbol(name)] = flat[key])
    end
    r = perfect_foresight(m; periods=T, exo=Matrix{Float64}(oo.exo_simul),
                          y_init=lag > 0 ? E[:, 1:lag] : nothing, y_term=lead > 0 ? E[:, lag+T+1:end] : nothing,
                          guess=E[:, lag+1:lag+T], raise_on_failure=raise_on_failure, kw...)
    _store_pf!(ws, r)
end
"Keyword names of perfect_foresight() (tolf, maxit, ...) as the options_ paths Dynare uses (dynatol.f, simul.maxit, ...)."
_pf_option_paths(options) = Dict(Symbol(get(PF_SOLVER_KEYS, String(k), get(PF_SETUP_KEYS, String(k), String(k)))) => v for (k, v) in options)

function _store_pf!(ws, r)
    oo = ws.oo_
    oo.endo_simul = Matrix{Float64}(r.endo_simul)
    oo.exo_simul = Matrix{Float64}(r.exo_simul)
    ds = MStruct()
    ds.status = r.status
    ds.iterations = Float64(r.iterations)
    ds.error = r.max_residual
    oo.deterministic_simulation = ds
    ws
end

"perfect_foresight_with_expectation_errors_solver (from the .mod file's declarations)."
function _perfect_foresight_wee!(ws::DynareWorkspace)
    r = perfect_foresight(_sync_model!(ws); expectation_errors=true)
    _store_pf!(ws, r)
    ws.oo_.steady_state = _col(r.steady_state)
    r.initial_steady_state === nothing || (ws.oo_.initial_steady_state = _col(r.initial_steady_state))
    ws
end

"""
Dynare's `occbin_solver` at the current `M_.params` (shocks from the `shocks(surprise)` block), with the
options in `options_.occbin` (keywords use the .mod file's names: `occbin_solver!(simul_periods=50)`).
"""
function occbin_solver!(ws::DynareWorkspace; options...)
    _sync_model!(ws)
    _take_options!(ws, Dict(Symbol("occbin." * _occbin_path(String(k))) => v for (k, v) in options))
    ob = Occbin(ws.model)
    occ_opts = _getpath(ws.options_, "occbin", MStruct())
    flat = occ_opts isa MStruct ? _flat_options(occ_opts) : Dict{String,Any}()
    kw = Dict{Symbol,Any}()
    for (key, (sym, T)) in _OCCBIN_MOD_OPTIONS
        haskey(flat, key) || continue
        v = flat[key]
        kw[sym] = T === Bool ? _asbool(v) : T === Int ? Int(round(_asnum(v))) : _asnum(v)
    end
    r = simulate(ob; kw...)
    sim = MStruct()
    sim.piecewise = Matrix{Float64}(r.piecewise)
    sim.linear = Matrix{Float64}(r.linear)
    sim.shocks_sequence = Matrix{Float64}(r.shocks)
    sim.exo_pos = _col(r.exo_pos)
    occ = get(ws.oo_, "occbin", MStruct())
    occ.simul = sim
    ws.oo_.occbin = occ
    ws
end

"simul_periods -> simul.periods (the .mod file's option names to options_.occbin's paths)."
_occbin_path(k) = (mt = match(r"^(simul|smoother|likelihood|filter)_(.+)$", k); mt === nothing ? k : mt[1] * "." * mt[2])

# Dynare's commands act on the current workspace (the last one `@dynare` defined M_/oo_ for), as MATLAB's act on globals
const CURRENT = Ref{Union{Nothing,DynareWorkspace}}(nothing)
_current() = CURRENT[] === nothing ? throw(ArgumentError("no Dynare workspace yet: run dynare(\"model.mod\") or @dynare first")) : CURRENT[]
for f in (:stoch_simul!, :steady!, :check!, :perfect_foresight_setup!, :perfect_foresight_solver!, :occbin_solver!)
    @eval $f(; kw...) = $f(_current(); kw...)
end
stoch_simul!(var_list::AbstractVector; kw...) = stoch_simul!(_current(), var_list; kw...)
set_param_value!(name::AbstractString, v::Real) = set_param_value!(_current(), name, v)

"""
    dynare(modfile; output=nothing, shocks=Dict(), rng=nothing, seed=nothing, kw...) -> DynareWorkspace

Run a .mod file as `dynare modfile` does and return M_, options_ and oo_ as MATLAB-style structs:
`ws.oo_.endo_simul`, `ws.oo_.dr.ghx`, `ws.oo_.irfs.y_e`, `ws.M_.params`, ... (Julia reads them with
MATLAB's syntax). `@dynare "model.mod"` also defines `M_`, `options_` and `oo_` in the caller's module.

Commands run where they stand in the file, with the calibration of that point, and MATLAB lines read and
write their results (`irf1 = oo_.irfs.y_e;`, `oo_.exo_simul(2, 1) = 0.1;`). Afterwards, re-run them on the
workspace without preprocessing: `M_.params[i] = x` or `set_param_value!(ws, "rho", x)`, then
`stoch_simul!(ws)`; edit `oo_.exo_simul`, then `perfect_foresight_solver!(ws)`.

`shocks` maps the position of a `stoch_simul` statement among the file's stoch_simul statements (1, 2, ...)
to a (periods x n_exo) shock matrix, to reproduce a given simulation (e.g. Dynare's `oo_.exo_simul`);
otherwise shocks are drawn from the workspace's random stream (`seed=n` seeds it as `set_dynare_seed(n)`, default
seed 0 like Dynare's 'default'; `set_dynare_seed(...)` lines of the file act as in Dynare; an explicit `rng` replaces
the stream). The draws are Julia's, reproducible run to run but not MATLAB's: see docs/RESULTS.md, "Random draws".
"""
function dynare(modfile::AbstractString; output=nothing, shocks=Dict{Int,Any}(), rng=nothing, seed=nothing, kw...)
    if output === nothing
        # derivatives up to the highest order a stoch_simul statement asks for (Dynare's default is 2)
        output = occursin(r"order\s*=\s*3", read(modfile, String)) ? "third" : "second"
    end
    wsref = Ref{Any}(nothing)
    nss = Ref(0)
    factory = function (m)
        set_dynare_seed!(m, seed === nothing ? "default" : seed)     # Dynare seeds its stream with 'default' at start-up
        ws = DynareWorkspace(m, build_M(m), build_options(m), MStruct(), Dict{String,Any}())
        ws.oo_.exo_det_steady_state = zeros(0, 0)
        ws.oo_.dynare_version = "7.2"
        ws.state[:matlab_options] = Dict{String,Any}()
        m.session.host.oo.results = ws.oo_          # MATLAB lines of the file see (and write) this oo_
        wsref[] = ws
        st -> _run_in_place!(ws, st, nss, shocks, rng)
    end
    m = load(modfile; output=output, on_command=factory, kw...)
    ws = wsref[]
    m.session.host.oo.results = nothing             # the struct now belongs to the workspace alone
    _apply_matlab_options!(ws)
    # M_ as the driver leaves it at the end of the file (declarations, final calibration); histval kept
    h = ws.M_.endo_histval
    fresh = build_M(m)
    empty!(ws.M_._keys); empty!(ws.M_._vals)
    for k in keys(fresh)
        ws.M_[k] = fresh[k]
    end
    ws.M_.endo_histval = h
    _mark_synced!(ws)
    haskey(ws.oo_, "steady_state") || (ws.oo_.steady_state = _col(Float64[get(m.oo, v, 0.0) for v in m.endo_names]))
    haskey(ws.oo_, "exo_steady_state") || (ws.oo_.exo_steady_state = _col(exo_steady(m)))
    for (k, v) in m.session.sc.vars
        v isa Tainted || v isa HostStruct || (ws.vars[k] = v)
    end
    ws
end

const _QUIET_KINDS = Set(["perfect_foresight_with_expectation_errors_setup", "occbin_setup", "occbin_graph", "rplot",
    "write_latex_dynamic_model", "write_latex_static_model", "write_latex_original_model",
    "write_latex_steady_state_model", "collect_latex_files", "write_latex_definitions", "resid"])

"A Dynare statement of the file, where it stands: options into options_, then the command."
function _run_in_place!(ws, st, nss, shocks, rng)
    m = ws.model
    kind = st["statementName"]
    _apply_matlab_options!(ws)
    _apply_options!(ws.options_, st)
    if kind in ("initval", "endval")                   # these blocks also write oo_.steady_state
        ws.oo_.steady_state = _col(Float64[get(m.oo, v, 0.0) for v in m.endo_names])
        ws.oo_.exo_steady_state = _col(exo_steady(m))
        return
    elseif kind == "histval"
        _histval!(ws)
        return
    end
    kind in COMPUTE_KINDS || return
    _model_to_M!(ws)
    if kind == "steady"
        steady!(ws)
    elseif kind == "check"
        check!(ws)
    elseif kind == "stoch_simul"
        nss[] += 1
        stoch_simul!(ws, String.(get_or(st, "symbol_list", String[])); shocks=get(shocks, nss[], nothing), rng=rng)
    elseif kind == "perfect_foresight_setup"
        perfect_foresight_setup!(ws)
    elseif kind == "perfect_foresight_solver"
        perfect_foresight_solver!(ws; raise_on_failure=false)      # Dynare stores the status and goes on
        ws.oo_.deterministic_simulation.status == true ||
            @warn "perfect_foresight_solver: the simulation failed (oo_.deterministic_simulation.status = false)"
    elseif kind == "perfect_foresight_with_expectation_errors_solver"
        _perfect_foresight_wee!(ws)
    elseif kind == "occbin_solver"
        occbin_solver!(ws)
    elseif !(kind in _QUIET_KINDS)
        warn_once(m, "dynare(): '$kind' does not fill oo_ yet")
    end
    nothing
end

"options_ fields the MATLAB lines of the file assigned since the last statement."
function _apply_matlab_options!(ws)
    mo = ws.model.matlab_options
    seen = ws.state[:matlab_options]
    for k in keys(mo)
        v = mo[k]
        isequal(get(seen, k, nothing), v) && continue
        seen[k] = deepcopy(v)
        if v isa MStruct && get(ws.options_, k, nothing) isa MStruct
            _merge_struct!(ws.options_[k], v)
        else
            ws.options_[k] = deepcopy(v)
        end
    end
end
function _merge_struct!(dst::MStruct, src::MStruct)
    for k in keys(src)
        v = src[k]
        v isa MStruct && get(dst, k, nothing) isa MStruct ? _merge_struct!(dst[k], v) : (dst[k] = deepcopy(v))
    end
end

"M_.endo_histval from the histval block (Dynare: unset values are 0), as stoch_simul starts from it."
function _histval!(ws)
    h = try
        _pf_replay(ws.model, nothing, nothing, (), ()).endo_histval
    catch e
        e isa Union{ModelError,PerfectForesightError,SteadyStateError,ArgumentError} || rethrow()
        warn_once(ws.model, "histval could not be evaluated for M_.endo_histval ($(sprint(showerror, e)))")
        nothing
    end
    h === nothing || (ws.M_.endo_histval = _col(h))
end

"""
    @dynare "model.mod" [kw...]

Run the file like MATLAB's `dynare model` and define `M_`, `options_` and `oo_` (and return the
DynareWorkspace).
"""
macro dynare(file, kws...)
    kwargs = [Expr(:kw, k.args[1], k.args[2]) for k in kws if k isa Expr && k.head == :(=)]
    esc(quote
        local ws = $JDyn.dynare($file; $(kwargs...))
        $JDyn.CURRENT[] = ws
        global M_ = ws.M_
        global options_ = ws.options_
        global oo_ = ws.oo_
        ws
    end)
end
