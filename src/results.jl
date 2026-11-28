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
    M.exo_det_nbr = Float64(n_exo_det(m))
    n_exo_det(m) > 0 && (M.exo_det_names = _cellcol(exo_det_names(m)))
    M.param_nbr = Float64(length(m.param_names))
    M.orig_endo_nbr = Float64(orig_endo_nbr(m))
    M.eq_nbr = Float64(length(get_or(m.raw, "model", Any[])) == 0 ? m.n_endo : length(m.raw["model"]))
    ri = is_ramsey(m) ? ramsey_info(m) : nothing
    M.ramsey_orig_eq_nbr = ri === nothing ? 0.0 : Float64(ri.orig_eq_nbr)
    M.ramsey_orig_endo_nbr = ri === nothing ? 0.0 : Float64(ri.orig_endo_nbr)
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
    M.Correlation_matrix = copy(declared_correlation_matrix(m))
    M.sigma_e_is_diagonal = isdiag(m.Sigma_e)
    M.H = me_H(m, m.varobs)
    M.Correlation_matrix_ME = me_corr_matrix(m, m.varobs)
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
    _semi_into_M!(M, m)                     # M_.var, M_.pac, ... (src/pac.jl)
    M
end
_corr(S) = (d = sqrt.(diag(S)); [d[i] > 0 && d[j] > 0 ? S[i, j] / (d[i] * d[j]) : (i == j ? 1.0 : 0.0) for i in axes(S, 1), j in axes(S, 2)])

# ---------------------------------------------------------------------- options_
# where the preprocessor's MATLAB driver writes each statement's options (ComputingTasks.cc); every other
# statement writes into options_ itself. Statements writing to a separate struct (options_ident, options_mom_,
# options_occbin_ for occbin_setup, ...) do not change options_ and are skipped here.
const _OPTION_PREFIX = Dict("occbin_solver" => "occbin.", "occbin_write_regimes" => "occbin.", "occbin_write_regimes_xls" => "occbin.",
                           "data" => "dataset.")
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
    O.logged_steady_state = 0.0                       # set by stoch_simul(loglinear), see `_undo_logged_ss!`
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
        kind == "extended_path" && k == "periods" && continue          # the sample size of the call, not options_.periods
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
    ws.M_.Correlation_matrix = copy(declared_correlation_matrix(m))
    if !haskey(ws.state, :H)                      # no estimation has set the measurement-error covariance
        ws.M_.H = me_H(m, m.varobs)
        ws.M_.Correlation_matrix_ME = me_corr_matrix(m, m.varobs)
    end
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
        key in ("dynare_version", "occbin", "dataset", "logged_steady_state") && continue
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
const _STEADY_KEYS = Set(["solve_tolf", "steadystate.nocheck", "homotopy_mode", "homotopy_steps", "solve_algo", "solve_tolx", "steady.maxit"])

"Parameters the steady state (file or steady_state_model block) has set go back to M_.params, as in Dynare."
function _store_params!(ws, p)
    ws.model.params .= p
    ws.M_.params = _col(p)
    _mark_synced!(ws)
end

# ---------------------------------------------------------------------- oo_
function _store_dr!(oo::MStruct, dr::AbstractDict, n::Int; sol=nothing, kos::Bool=false)
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
    ex = sol === nothing ? (;) : sol.extra
    if haskey(ex, :gx) && !kos
        # dyn_first_order_solver.m / stochastic_solvers.m: gx, Gy (= hx, the transition matrix of the states, in levels also under
        # loglinear, where Dynare converts only ghx and ghu), sdim and edim; the simplified purely backward branch sets no gx or Gy
        # but full_rank = 1
        if ex.backward
            D.full_rank = 1.0
        else
            D.gx = ex.gx
            D.Gy = ex.hx
        end
        D.sdim = Float64(ex.sdim); D.edim = Float64(ex.edim)
    elseif haskey(dr, "ghx") && haskey(dr, "state_var")      # dr.Gy: the transition matrix of the predetermined variables
        rows = findall(v -> Int(v) in Int.(dr["state_var"]), Int.(dr["order_var"]))
        length(rows) == size(dr["ghx"], 2) && (D.Gy = Matrix{Float64}(dr["ghx"])[rows, :])
    end
    if kos       # k_order_pert.m copies Dynare++'s folded tensors, and with pruning the pruned ones
        for i in 0:Int(get(dr, "order", 1))
            g = get(dr, "g_$i", nothing)
            g === nothing || (D["g_$i"] = Matrix{Float64}(g))
        end
        if get(dr, "pruning", false) === true
            P = MStruct()
            for i in 0:Int(get(dr, "order", 1))
                g = get(dr, "pruning_g_$i", nothing)
                g === nothing || (P["g_$i"] = Matrix{Float64}(g))
            end
            D.pruning = P
        end
    end
    # resol works on the dr it is given: what dynare_estimation_init left in it (restrict_var_list, restrict_columns) stays
    if old isa MStruct
        for k in ("restrict_var_list", "restrict_columns")
            haskey(old, k) && (D[k] = old[k])
        end
    end
    oo.dr = D
end

"""
`dr.fuu` of dyn_second_order_solver.m: the (equation-ordered) vector `g1(:,lead)*ghuu(lead,:)*vec(Sigma_e) + g2(:,lead,lead)*kron(ghu(lead,:),ghu(lead,:))*vec(Sigma_e)`,
the second-order terms of the forward-looking variables that enter the `ghs2` equation.
"""
function _dr_fuu(m::Model, dr::AbstractDict, J::AbstractMatrix)
    n = m.n_endo
    ys = Vector{Float64}(dr["ys"])
    params = haskey(dr, "params") ? Vector{Float64}(dr["params"]) : copy(m.params)
    ov = Int.(dr["order_var"]); inv = invperm(ov)
    ghu = Matrix{Float64}(dr["ghu"])[inv, :]                   # declaration order
    ghuu = Matrix{Float64}(dr["ghuu"])[inv, :]
    Sig = Matrix{Float64}(m.Sigma_e)
    vecS = vec(Sig)
    lead = findall(>(0), m.lead_lag_incidence[:, 3])
    fuu = zeros(n)
    for j in lead
        fuu .+= J[:, 2n+j] .* dot(ghuu[j, :], vecS)
    end
    r, c, v = dynamic_sparse_derivs(m, 2, vcat(ys, ys, ys), exo_steady(m), params, ys)
    for k in eachindex(r)
        c1, c2 = c[k, 1], c[k, 2]
        (2n < c1 <= 3n && 2n < c2 <= 3n) || continue
        a, b = c1 - 2n, c2 - 2n
        w = dot(ghu[a, :], Sig * ghu[b, :])
        fuu[r[k]] += v[k] * w * (a == b ? 1.0 : 2.0)
    end
    fuu
end

"The `varexo_det` terms of stochastic_solvers.m (dr.exo_det_length, ghud, and at order 2 ghxud, ghudud, ghuud) as Dynare stores them."
function _store_exo_det_dr!(D::MStruct, m::Model, dr::AbstractDict)
    ed = try
        exo_det_decision_rules(m, dr)
    catch e
        e isa ModelError || rethrow()
        warn_once(m, "varexo_det: the deterministic-exogenous decision rules are not stored in oo_.dr ($(e.msg))")
        return D
    end
    L = Int(ed["exo_det_length"])
    D.exo_det_length = Float64(L)
    D.ghud = Cell(Any[Matrix{Float64}(x) for x in ed["ghud"]], (L, 1))
    if haskey(ed, "ghxud")
        D.ghxud = Cell(Any[Matrix{Float64}(x) for x in ed["ghxud"]], (L, 1))
        D.ghuud = Cell(Any[Matrix{Float64}(x) for x in ed["ghuud"]], (L, 1))
        items = Any[haskey(ed["ghudud"], (j, k)) ? Matrix{Float64}(ed["ghudud"][(j, k)]) : zeros(0, 0) for k in 1:L for j in 1:L]
        D.ghudud = Cell(items, (L, L))
    end
    D
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
function stoch_simul!(ws::DynareWorkspace, var_list=String[]; shocks=nothing, rng=nothing, plots=nothing, options...)
    m, oo = _sync_model!(ws), ws.oo_
    _take_options!(ws, options)
    _with_options(() -> _stoch_simul(ws, m, oo, var_list, shocks, rng), ws, _SOLVE_KINDS)
    plots === true && _statement_figures!(ws, "stoch_simul", Dict{String,Any}("symbol_list" => String.(collect(var_list))))     # src/plots_hooks.jl
    ws
end

"""
The logged steady state of Dynare's loglinear runs. `stoch_simul` (loglinear) replaces `oo_.steady_state` (and `oo_.dr.ys`, which the
solver already holds in logs) by the logs of the variables that are log-linearised (`log_variable`) and sets
`options_.logged_steady_state = 1`; `stoch_simul`, `dynare_estimation` (so `estimation`) and `model_diagnostics` undo it first
(`exp`, flag 0). One mechanism: `ws.state[:logged_ss]` mirrors the flag, which is also written to `options_.logged_steady_state`.
A command that writes `oo_.steady_state` in levels (`steady`, `calib_smoother`, the perfect-foresight solvers) ends the logged state
but leaves the option as Dynare does (Dynare then reads the levels as logs at the next `stoch_simul`; JDyn does not).
"""
function _undo_logged_ss!(ws)
    get(ws.state, :logged_ss, false) === true || return ws
    m, oo = ws.model, ws.oo_
    mask = log_mask(m)
    unlog(v) = (x = vec(Float64.(v)); length(x) == m.n_endo ? _col([mask[i] ? exp(x[i]) : x[i] for i in 1:m.n_endo]) : v)
    haskey(oo, "steady_state") && (oo.steady_state = unlog(oo.steady_state))
    D = get(oo, "dr", nothing)
    D isa MStruct && haskey(D, "ys") && (D.ys = unlog(D.ys))
    ws.state[:logged_ss] = false
    _set_logged_flag!(ws, 0.0)
    ws
end

"`options_.logged_steady_state` in options_ and where the file's MATLAB lines read it."
function _set_logged_flag!(ws, v::Float64)
    ws.options_.logged_steady_state = v
    ws.model.matlab_options["logged_steady_state"] = v
    haskey(ws.state, :matlab_options) && (ws.state[:matlab_options]["logged_steady_state"] = v)
    sess = ws.model.session
    sess === nothing || delete!(sess.host.options.unknown, "logged_steady_state")
    ws
end

function _log_ss!(ws)
    get(ws.state, :logged_ss, false) === true && return ws
    m, oo = ws.model, ws.oo_
    ss = vec(Float64.(get(oo, "steady_state", zeros(m.n_endo))))
    mask = log_mask(m)
    length(ss) == m.n_endo && (oo.steady_state = _col([mask[i] ? log(ss[i]) : ss[i] for i in 1:m.n_endo]))
    ws.state[:logged_ss] = true
    _set_logged_flag!(ws, 1.0)
    ws
end

function _stoch_simul(ws, m, oo, var_list, shocks, rng)
    order = _opti(ws, "order", 2)
    if order != 1 && _hessian_eq_zero_dynare(m)           # stoch_simul.m: M_.hessian_eq_zero (a linear model) forces order 1
        warn_once(m, "stoch_simul: using order = 1 because Hessian is equal to zero")
        order = 1
        ws.options_.order = 1.0
    end
    pruning = _optb(ws, "pruning", false)
    order > 1 && m.max_deriv_order < order && throw(ModelError("stoch_simul(order=$order) needs derivatives of order $order: run dynare(...; output=\"$(order == 2 ? "second" : "third")\")"))
    solved = pop!(ws.state, :solved_dr, nothing)        # discretionary_policy: the rules are not the perturbation solution
    # stoch_simul.m: oo_.exo_simul is filled with the exogenous steady state over max(periods,1) + lags + leads before anything is solved
    if m.n_exo > 0
        nper = max(_opti(ws, "periods", 0), 1) + Int(m.raw["model_info"]["maximum_lag"]) + Int(m.raw["model_info"]["maximum_lead"])
        oo.exo_simul = repeat(permutedims(Float64.(exo_steady(m))), nper, 1)
    end
    _undo_logged_ss!(ws)                       # "if options_.logged_steady_state: undo this"
    kos = _optb(ws, "k_order_solver", false) || order >= 3       # Dynare solves with the k-order library (g_0 ... g_k, no gx or Gy)
    sol = nothing
    if solved !== nothing
        dr, p, eigv = solved
    else
        sol = first_order(m)                      # also gives dr.eigval, which Dynare's resol stores at every order
        dr, p = (order == 1 && !kos) ? (as_dr(sol), sol.params) :
                (d = k_order(m; order=order, pruning=pruning); (d, d["params"]))
        eigv = sol.eigenvalues
    end
    _store_params!(ws, p)
    _store_dr!(oo, dr, m.n_endo; sol=sol, kos=kos && solved === nothing)
    if solved === nothing && !kos
        order == 2 && (oo.dr.fuu = _col(_dr_fuu(m, dr, sol.jacobian)))
        n_exo_det(m) > 0 && _store_exo_det_dr!(oo.dr, m, dr)
    end
    oo.dr.eigval = _eigcol(eigv)
    if _optb(ws, "aim_solver", false) && order == 1 && !kos && solved === nothing
        # AIM_first_order_solver.m: dr.eigval = eig(kalman_transition_matrix(dr, nstatic+(1:nspred), 1:nspred)), the nspred
        # eigenvalues of the state transition, not the generalised eigenvalues of the QZ problem
        _, _, cnt = dr_ordering(m.lead_lag_incidence)
        nspred = cnt.npred + cnt.nboth
        G = Matrix{Float64}(dr["ghx"])[cnt.nstatic+1:cnt.nstatic+nspred, 1:nspred]
        oo.dr.eigval = _eigcol(LinearAlgebra.eigvals(G))
    end
    # stoch_simul.m leaves oo_.steady_state alone (the initval values, or the last `steady`): resol only fills oo_.dr.ys. Under
    # loglinear it logs oo_.steady_state (log_variable) once and flags options_.logged_steady_state, undone at the next solve.
    _optb(ws, "loglinear", false) && _log_ss!(ws)
    var_list = isempty(var_list) ? orig_endo_names(m) : String.(collect(var_list))
    oo.var_list = _cellcol(var_list)
    ivar = [m.ie[v] for v in var_list]
    periods = _opti(ws, "periods", 0)
    if periods > 0
        # simult.m: options_.simul_replic replications drawn one after the other from one stream; oo_.endo_simul is the first
        # replication, oo_.exo_simul the last, and with more than one the raw data (float64, endo_nbr x periods each) go to
        # <fname>/Output/<fname>_simul
        nrep = max(_opti(ws, "simul_replic", 1), 1)
        shocks !== nothing && nrep > 1 && throw(ArgumentError("shocks cannot be given together with simul_replic > 1"))
        r = rng === nothing ? dynare_rng(ws) : rng
        iv = findall(!=(0), diag(m.Sigma_e))
        U = isempty(iv) ? zeros(0, 0) : Matrix(cholesky(Symmetric(m.Sigma_e[iv, iv])).U)
        # stoch_simul.m: the simulation starts from M_.endo_histval (histval), else from the steady state
        h = ws.M_.endo_histval
        y0 = isempty(h) ? nothing : (_optb(ws, "loglinear", false) ? log.(vec(h[:, end])) : vec(h[:, end]))
        io = nothing
        if nrep > 1 && haskey(ws.state, :modfile)
            fn = splitext(basename(ws.state[:modfile]))[1]
            od = joinpath(dirname(ws.state[:modfile]), fn, "Output"); mkpath(od)
            io = open(joinpath(od, fn * "_simul"), "w")
        end
        yfirst = nothing; ex = nothing
        for rep in 1:nrep
            ex = if shocks !== nothing
                size(shocks) == (periods, m.n_exo) || throw(ArgumentError("shocks must be periods x exo_nbr = ($periods, $(m.n_exo))"))
                Matrix{Float64}(shocks)
            else
                e = zeros(periods, m.n_exo)
                # simult.m: transpose(randn(nxs, periods)) * chol(Sigma_e(iv, iv)): the draws fill period by period
                e[:, iv] .= permutedims(randn(r, length(iv), periods)) * U
                e
            end
            y = simulate(dr, ex; order=order, pruning=pruning, y0=y0)
            ym = Matrix{Float64}(permutedims(y))         # n_endo x periods, as Dynare's simult
            rep == 1 && (yfirst = ym)
            io === nothing || write(io, ym)
        end
        io === nothing || close(io)
        oo.endo_simul = yfirst
        oo.exo_simul = ex
    end
    if !_optb(ws, "nomoments", false) && (order <= 2 || (order == 3 && pruning && periods == 0))
        mom = try
            periods > 0 ? simulated_moments(permutedims(oo.endo_simul); ivar=ivar, shocks=oo.exo_simul, dr=dr, model=m, order=order) :
                          theoretical_moments(m, dr; var_list=var_list, order=order, pruning=pruning)
        catch e
            e isa Union{ArgumentError,ModelError} || rethrow()
            warn_once(m, "stoch_simul: moments not computed ($(sprint(showerror, e)))")
            nothing
        end
        if mom !== nothing
            oo.mean = _col(mom.mean)
            oo.var = Matrix{Float64}(mom.var)
            if _optb(ws, "contemporaneous_correlation", false) && !_optb(ws, "nocorr", false)
                # disp_th_moments.m / disp_moments.m: NaN where the variance is (numerically) zero
                V = Matrix{Float64}(mom.var); sd = sqrt.(max.(diag(V), 0.0))
                ok = periods == 0 ? diag(V) .> 1e-12 : sd .>= 1e-10
                C = fill(NaN, size(V))
                for i in eachindex(sd), j in eachindex(sd)
                    ok[i] && ok[j] && (C[i, j] = V[i, j] / (sd[i] * sd[j]))
                end
                oo.contemporaneous_correlation = C
            end
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
    # stoch_simul.m draws the simulation first, then the (generalised) IRFs: one random stream, in that order
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
    _optb(ws, "SpectralDensity.trigger", false) && _spectral_density!(ws, m, oo, ivar, order)
    oo
end

"""
UnivariateSpectralDensity.m: oo_.SpectralDensity.freqs (ngrid x 1) and .density (nvar x ngrid) for the variables of the list (HP or
band-pass filtered when hp_filter / bandpass_filter is set), on the grid 0:pi/(ngrid-1):pi, ngrid = filtered_theoretical_moments_grid.
At order >= 2 Dynare prints a message and returns (nothing is stored). A model with unit roots (Dynare keeps the variables whose
loadings on the unstable subspace of lyapunov_symm vanish) is refused.
"""
function _spectral_density!(ws, m, oo, ivar, order)
    order > 1 && return oo
    _optf(ws, "one_sided_hp_filter", 0) != 0 && throw(ModelError("stoch_simul: UnivariateSpectralDensity:: spectral density estimate not available with one-sided HP filter"))
    D = oo.dr
    ghx, ghu = Matrix{Float64}(D.ghx), Matrix{Float64}(D.ghu)
    ov = Int.(vec(D.order_var)); iv = invperm(ov)
    _, _, cnt = dr_ordering(m.lead_lag_incidence)
    nsp = cnt.npred + cnt.nboth
    ikx = cnt.nstatic .+ (1:nsp)
    A, ghu1 = ghx[ikx, :], ghu[ikx, :]
    nsp > 0 && maximum(abs, LinearAlgebra.eigvals(A)) >= 1 - 1e-12 &&
        throw(ModelError("stoch_simul: SpectralDensity.trigger for a model with unit roots is not supported"))
    iky = iv[ivar]
    aa, bb = ghx[iky, :], ghu[iky, :]
    ng = _opti(ws, "filtered_theoretical_moments_grid", 512)
    freqs = ng > 1 ? collect(range(0, pi; length=ng)) : zeros(1)
    gain = if _optb(ws, "bandpass.indicator", false)
        pb = Float64.(vec(_optf(ws, "bandpass.passband", [6.0 32.0])))
        lo, hi = pb[2], max(2.0, pb[1])
        Float64.((freqs .>= 2pi / lo) .& (freqs .<= 2pi / hi))
    elseif _optf(ws, "hp_filter", 0) != 0
        lam = Float64(_optf(ws, "hp_filter", 0))
        4lam .* (1 .- cos.(freqs)) .^ 2 ./ (1 .+ 4lam .* (1 .- cos.(freqs)) .^ 2)
    else
        ones(ng)
    end
    Sig = Matrix{Float64}(m.Sigma_e)
    IA = Matrix{Float64}(LinearAlgebra.I, nsp, nsp); IE = Matrix{Float64}(LinearAlgebra.I, m.n_exo, m.n_exo)
    nv = length(iky)
    f = zeros(nv, ng)
    for (ig, w) in enumerate(freqs)
        tp, tn = cis(w), cis(-w)
        left = nsp > 0 ? vcat((IA - A * tn) \ ghu1, IE) : IE
        right = nsp > 0 ? hcat(ghu1' / (IA - A' * tp), IE) : IE
        fo = (1 / (2pi)) * (left * Sig * right)
        L = nsp > 0 ? hcat(aa * tn, bb) : bb
        Rr = nsp > 0 ? vcat(aa' * tp, bb') : bb'
        g = L * fo * Rr
        f[:, ig] = real.(diag(g)) .* gain[ig]^2
    end
    S = MStruct(); S.freqs = _col(freqs); S.density = f
    oo.SpectralDensity = S
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
    ws.state[:logged_ss] = false                       # levels again (steady.m leaves options_.logged_steady_state as it was)
    ws.oo_.exo_steady_state = _col(exo_steady(m))      # a homotopy moves the exogenous variables too
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

"options_.periods, or the length of first_simulation_period:last_simulation_period when only dates were given."
function _pf_T(ws)
    T = _opti(ws, "periods", 0)
    if T <= 0
        dc = _pf_date_context(ws.model)
        dc.first !== nothing && dc.last !== nothing && (T = Int(dc.last[2] - dc.first[2] + 1))
    end
    T
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
    T = _pf_T(ws)
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
    T = _pf_T(ws)
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
    ws.state[:ran_pf] = true
    oo.endo_simul = Matrix{Float64}(r.endo_simul)
    oo.exo_simul = Matrix{Float64}(r.exo_simul)
    ds = MStruct()
    ds.status = r.status
    ds.iterations = Float64(r.iterations)
    ds.error = r.max_residual
    oo.deterministic_simulation = ds
    ws
end

"""
perfect_foresight_with_expectation_errors_setup: builds oo_.pfwee (information sets) and the initial guess
in oo_.endo_simul / oo_.exo_simul (src/pf_functions.jl); a failure is left to the solver command, which then
replays the declarations of the file.
"""
function _perfect_foresight_wee_setup!(ws::DynareWorkspace)
    m = _sync_model!(ws)
    delete!(ws.oo_, "pfwee")
    try
        pws = _pf_replay(m, nothing, nothing, (), ())
        ws.oo_.steady_state = _col(pws.ss)
        ws.oo_.exo_steady_state = _col(pws.xss)
        pws.ss0 === nothing ? (haskey(ws.oo_, "initial_steady_state") && delete!(ws.oo_, "initial_steady_state"),
                               haskey(ws.oo_, "initial_exo_steady_state") && delete!(ws.oo_, "initial_exo_steady_state")) :
                              (ws.oo_.initial_steady_state = _col(pws.ss0); ws.oo_.initial_exo_steady_state = _col(pws.xss0))
        _pfo_pfwee_setup!(m, ws.options_, ws.oo_, copy(m.params))
    catch e
        e isa Union{ModelError,PerfectForesightError,SteadyStateError,ArgumentError,Unsupported} || rethrow()
        delete!(ws.oo_, "pfwee")
    end
    ws
end

"perfect_foresight_with_expectation_errors_solver (from oo_.pfwee when the setup command stored it, else from the .mod file's declarations)."
function _perfect_foresight_wee!(ws::DynareWorkspace)
    if haskey(ws.oo_, "pfwee")
        m = _sync_model!(ws)
        _pfo_pfwee_solve!(m, ws.options_, ws.oo_, copy(m.params))
        return ws
    end
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
    # Dynare 7.2 runs OccBin under options_.loglinear by combining the level model's derivatives at the logged steady
    # state with log-linearised rules: neither levels nor logs. JDyn's piecewise-linear solution is in levels, so the
    # combination is refused rather than silently answered in levels.
    _optb(ws, "loglinear", false) && throw(ModelError("occbin_solver with loglinear is not supported: Dynare 7.2 mixes the " *
        "level model with log-linearised rules here, and JDyn's piecewise-linear solution is in levels; drop the loglinear option"))
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
    ws.state[:occbin_result] = r
    sim = MStruct()
    sim.piecewise = Matrix{Float64}(r.piecewise)
    sim.linear = Matrix{Float64}(r.linear)
    sim.shocks_sequence = Matrix{Float64}(r.shocks)
    sim.exo_pos = _col(r.exo_pos)
    sim.regime_history = _regime_history_cell(r.regime_history)
    sim.ys = _col(r.ys)
    sim.error_flag = 0.0
    occ = get(ws.oo_, "occbin", MStruct())
    occ.simul = sim
    # occbin.unpack_simulations.m: the simulations by variable name (and the shocks used)
    lin, pw, sss = MStruct(), MStruct(), MStruct()
    for (i, v) in enumerate(ws.model.endo_names)
        lin[v] = _col(r.linear[:, i]); pw[v] = _col(r.piecewise[:, i]); sss[v] = Float64(r.ys[i])
    end
    occ.endo_linear = lin; occ.endo_piecewise = pw; occ.endo_ss = sss
    ex = MStruct()
    for (j, p) in enumerate(r.exo_pos)
        j <= size(r.shocks, 2) && (ex[ws.model.exo_names[j]] = _col(r.shocks[:, j]))
    end
    occ.exo = ex
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
    dynare(modfile; output=nothing, shocks=Dict(), rng=nothing, seed=nothing, random=:julia, kw...) -> DynareWorkspace

Run a .mod file as `dynare modfile` does and return M_, options_ and oo_ as MATLAB-style structs:
`ws.oo_.endo_simul`, `ws.oo_.dr.ghx`, `ws.oo_.irfs.y_e`, `ws.M_.params`, ... (Julia reads them with
MATLAB's syntax). `@dynare "model.mod"` also defines `M_`, `options_` and `oo_` in the caller's module.

`random=:octave` gives Octave's random numbers bit for bit (the Mersenne twister and ziggurat of Octave 8.4): the workspace's
stream becomes an `OctaveRNG`, so the file's own `rand`/`randn`/`randi`/`randperm`/`rng`/`set_dynare_seed` calls and Dynare's
commands (simulated shocks, forecast draws, particle filters, MCMC) consume one stream in file order, as Dynare under Octave
does. The default `random=:julia` keeps the seeded Xoshiro stream (reproducible, not Octave's numbers).

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
function dynare(modfile::AbstractString; output=nothing, shocks=Dict{Int,Any}(), rng=nothing, seed=nothing, particle_rng=nothing, mcmc_rng=nothing, _json_dir=nothing, plots::Bool=false, plots_backend::Symbol=:auto, random::Symbol=:julia, kw...)
    random in (:julia, :octave) || throw(ArgumentError("random must be :julia (default) or :octave"))
    if output === nothing
        # derivatives up to the highest order a stoch_simul statement asks for (Dynare's default is 2)
        output = occursin(r"order\s*=\s*3", read(modfile, String)) ? "third" : "second"
    end
    wsref = Ref{Any}(nothing)
    nss = Ref(0)
    factory = function (m)
        random === :octave && use_octave_random!(m)                  # one Octave stream for simulations and the file's rand/randn (src/matlab/rng.jl)
        set_dynare_seed!(m, seed === nothing ? "default" : seed)     # Dynare seeds its stream with 'default' at start-up
        particle_rng === nothing || set_rng_kind!(m, particle_rng)   # :octave = Octave's numbers in the particle filters (src/octave_rng.jl)
        mcmc_rng === nothing || set_mcmc_rng!(m, mcmc_rng)           # :octave = Dynare's random numbers in the MCMC samplers (src/mcmc_driver.jl)
        ws = DynareWorkspace(m, build_M(m), build_options(m), MStruct(), Dict{String,Any}())
        ws.oo_.exo_det_steady_state = n_exo_det(m) > 0 ? _col(exo_det_steady(m)) : zeros(0, 0)
        ws.oo_.dynare_version = "7.2"
        ws.oo_.steady_state = zeros(m.n_endo, 1)           # the driver's `oo_.steady_state = zeros(endo_nbr, 1)`; initval and steady fill it
        ws.oo_.exo_steady_state = zeros(m.n_exo, 1)
        ws.state[:matlab_options] = Dict{String,Any}()
        ws.state[:modfile] = abspath(modfile)
        _install_estimation_globals!(ws)
        m.session.host.M.results = ws.M_
        m.session.host.oo.results = ws.oo_          # MATLAB lines of the file see (and write) this oo_
        _pf_register_host!(ws)                      # perfect_foresight_solver(M_, options_, oo_) & co. (src/pf_functions.jl)
        ws.state[:plots] = plots; ws.state[:plots_backend] = plots_backend
        wsref[] = ws
        st -> _run_in_place!(ws, st, nss, shocks, rng)
    end
    allow0 = ALLOW_RANDOM[]; ALLOW_RANDOM[] = true         # as in MATLAB, the file's rand/randn calls draw numbers
    m = try
        _json_dir === nothing ? load(modfile; output=output, on_command=factory, kw...) : Model(_json_dir; on_command=factory, kw...)
    finally
        ALLOW_RANDOM[] = allow0
    end
    ws = wsref[]
    m.session.host.M.results = nothing
    m.session.host.oo.results = nothing             # the struct now belongs to the workspace alone
    _apply_matlab_options!(ws)
    # M_ as the driver leaves it at the end of the file (declarations, final calibration); histval kept
    h = ws.M_.endo_histval
    h_x = get(ws.M_, "exo_histval", nothing)
    fresh = build_M(m)
    empty!(ws.M_._keys); empty!(ws.M_._vals)
    for k in keys(fresh)
        ws.M_[k] = fresh[k]
    end
    ws.M_.endo_histval = h
    h_x === nothing || (ws.M_.exo_histval = h_x)
    haskey(ws.state, :CME) && (ws.M_.Correlation_matrix_ME = ws.state[:CME])
    haskey(ws.state, :H) && (ws.M_.H = ws.state[:H])           # the measurement-error covariance estimation left (results_estimation.jl)
    _mark_synced!(ws)
    # Dynare writes <fname>/Output/<fname>_results.mat; not while precompiling (the fixture lives in the package source)
    _json_dir === nothing && (try; save_results_mat(dirname(abspath(modfile)), splitext(basename(modfile))[1], ws.M_, ws.oo_, ws.options_); catch e; @debug "results.mat not written" e; end)
    haskey(ws.oo_, "steady_state") || (ws.oo_.steady_state = _col(Float64[get(m.oo, v, 0.0) for v in m.endo_names]))
    haskey(ws.oo_, "exo_steady_state") || (ws.oo_.exo_steady_state = _col(exo_steady(m)))
    for (k, v) in m.session.sc.vars
        v isa Tainted || v isa HostStruct || (ws.vars[k] = v)
    end
    ws
end

const _QUIET_KINDS = Set(["occbin_setup", "occbin_graph", "rplot",
    "write_latex_dynamic_model", "write_latex_static_model", "write_latex_original_model",
    "write_latex_steady_state_model", "collect_latex_files", "write_latex_definitions", "resid"])

"""
endval (also endval(learnt_in=1), which the preprocessor turns into a plain block) as the driver runs it: the initial steady state is stored if there is none, then
oo_.exo_steady_state and oo_.steady_state (auxiliary variables included) take the values of the block.
"""
function _endval_learnt_in!(ws, st)
    m = ws.model
    oo = ws.oo_
    haskey(oo, "steady_state") || (oo.steady_state = _col(Float64[get(m.oo, v, 0.0) for v in m.endo_names]))
    haskey(oo, "exo_steady_state") || (oo.exo_steady_state = _col(exo_steady(m)))
    if !_pfo_has(oo, "initial_steady_state")
        oo.initial_steady_state = copy(oo.steady_state)
        oo.initial_exo_steady_state = copy(oo.exo_steady_state)
    end
    ss, xss = _pfo_vec(oo, "steady_state"), something(_pfo_vec(oo, "exo_steady_state"), Float64[])
    for d in get_or(st, "vals", [])
        name = d["name"]
        get(d, "type", "level") == "level" || throw(ModelError("endval(learnt_in=1) with add or multiply is not supported"))
        v = haskey(m.endval, name) ? Float64(m.endval[name]) : Float64(_eval_with_vars(m, d["value"]))    # as _pf_replay reads the block
        haskey(m.ie, name) ? (ss[m.ie[name]] = v) : haskey(m.ix, name) ? (xss[m.ix[name]] = v) : nothing
    end
    oo.steady_state = _col(fill_auxiliary(m, ss, xss, m.params))
    oo.exo_steady_state = _col(xss)
    nothing
end

"A Dynare statement of the file, where it stands: options into options_, then the command."
function _run_in_place!(ws, st, nss, shocks, rng)
    m = ws.model
    kind = st["statementName"]
    msg = option_policy_message(kind, get_or(st, "options", nothing))          # src/option_policy.jl: honoured or refused, never ignored
    msg === nothing || throw(ModelError("$kind: $msg"))
    _apply_matlab_options!(ws)
    if kind == "set_time"
        ws.state[:set_time] = string(get(st, "period", ""))
        return
    elseif kind == "estimation_data"                     # the `data` command: read by the next estimation (results_estimation.jl)
        o = get_or(st, "options", nothing)
        ws.state[:dataset] = o isa AbstractDict ? Dict{String,Any}(String(k) => v for (k, v) in o) : Dict{String,Any}()
        return
    end
    _apply_options!(ws.options_, st)
    if kind == "endval"                                # (learnt_in=p > 1 is news for the expectation-errors commands, M_.learnt_endval: oo_ is not touched)
        haskey(st, "learnt_in") && _pf_period(st["learnt_in"], "learnt_in") != 1 || _endval_learnt_in!(ws, st)
        return
    end
    if kind in ("initval", "endval")                   # these blocks also write oo_.steady_state
        ss0 = Float64[get(m.oo, v, 0.0) for v in m.endo_names]
        # the preprocessor's writeAuxVarInitval: initval/endval also set the auxiliary variables (leads, lags, log) from the originals
        ssf = try fill_auxiliary(m, ss0, exo_steady(m), m.params) catch; ss0 end
        ws.oo_.steady_state = _col(all(isfinite, ssf) ? ssf : ss0)
        ws.oo_.exo_steady_state = _col(exo_steady(m))
        n_exo_det(m) > 0 && (ws.oo_.exo_det_steady_state = _col(exo_det_steady(m)))
        return
    elseif kind == "histval"
        _histval!(ws)
        return
    end
    kind == "estimated_params" && (m.options["estimated_params_executed"] = get(m.options, "estimated_params_executed", 0) + 1)
    kind == "conditional_forecast_paths" && return _cf_paths_statement!(ws, st)
    if kind == "ramsey_model"
        _ramsey_model_statement!(ws, st)           # src/results_policy.jl
        return
    end
    if kind == "optim_weights"                      # in force from here to the next block (read by osr!)
        ws.state[:osr_weights] = st
        return
    end
    if kind in _PLOT_STATEMENT_KINDS && _plots_on(ws)       # decompositions and occbin_graph: only run for their figures (src/plots_hooks.jl)
        _plot_statement!(ws, kind, st)
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
        _simulated_series!(ws)
    elseif kind == "perfect_foresight_with_expectation_errors_setup"
        _perfect_foresight_wee_setup!(ws)
    elseif kind == "perfect_foresight_with_expectation_errors_solver"
        _perfect_foresight_wee!(ws)
    elseif kind == "ramsey_policy"
        ramsey_policy!(ws, String.(get_or(st, "symbol_list", String[])); rng=rng)
    elseif kind == "discretionary_policy"
        discretionary_policy!(ws, String.(get_or(st, "symbol_list", String[])); rng=rng)
    elseif kind == "evaluate_planner_objective"
        evaluate_planner_objective!(ws)
    elseif kind == "osr"
        osr!(ws, String.(get_or(st, "symbol_list", String[])); rng=rng)
    elseif kind == "extended_path"
        _extended_path_statement!(ws, st)               # src/extended_path.jl
    elseif kind == "occbin_solver"
        occbin_solver!(ws)
    elseif kind == "occbin_write_regimes_xls"
        occbin_write_regimes!(ws)                       # src/occbin_estimation.jl
    elseif kind == "estimation"
        _estimation!(ws, st, String.(get_or(st, "symbol_list", String[])), nothing, 0, (;))
    elseif kind == "calib_smoother"
        _calib_smoother!(ws, st, String.(get_or(st, "symbol_list", String[])), (;))
    elseif kind in ("forecast", "conditional_forecast", "plot_conditional_forecast")
        _forecast_statement!(ws, kind, st, rng)          # src/forecast.jl
    elseif kind == "identification"
        _identification_statement!(ws, st, rng)          # src/identification.jl
    elseif kind in _P2_KINDS
        _p2_statement!(ws, kind, st)                    # src/results_parity2.jl
    elseif !(kind in _QUIET_KINDS)
        warn_once(m, "dynare(): '$kind' does not fill oo_ yet")
    end
    get(ws.state, :plots, false) === true && _statement_figures!(ws, kind, st)      # src/plots_hooks.jl
    nothing
end

"options_ fields the MATLAB lines of the file assigned since the last statement."
function _apply_matlab_options!(ws)
    mo = ws.model.matlab_options
    seen = ws.state[:matlab_options]
    # an assignment of the value a field already had still counts when a Dynare command has changed the field since
    # (options_.lik_init=1 after an estimation that left 3)
    asg = ws.model.session === nothing ? Set{String}() : ws.model.session.host.options.assigned
    for k in keys(mo)
        v = mo[k]
        isequal(get(seen, k, nothing), v) && !(k in asg) && continue
        seen[k] = deepcopy(v)
        if v isa MStruct && get(ws.options_, k, nothing) isa MStruct
            _merge_struct!(ws.options_[k], v)
        else
            ws.options_[k] = deepcopy(v)
        end
    end
    empty!(asg)
end
function _merge_struct!(dst::MStruct, src::MStruct)
    for k in keys(src)
        v = src[k]
        v isa MStruct && get(dst, k, nothing) isa MStruct ? _merge_struct!(dst[k], v) : (dst[k] = deepcopy(v))
    end
end

"""
Dynare's `Simulated_time_series` (construct_simulation_dseries.m): a dseries of the original endogenous variables and the
exogenous ones over the simulation, which starts `maximum_lag` periods before first_simulation_period (default 1Y).
"""
function _simulated_series!(ws)
    m = ws.model
    dc = _pf_date_context(m)
    f = dc.first === nothing ? (1, 1.0) : dc.first
    first = MDates(f[1], [f[2]])
    lag = Int(m.raw["model_info"]["maximum_lag"])
    no = orig_endo_nbr(m)
    E, X = ws.oo_.endo_simul, ws.oo_.exo_simul
    data = hcat(permutedims(E[1:no, :]), X)
    names = vcat(m.endo_names[1:no], m.exo_names)
    init = MDates(first.freq, [first.time[1] - lag])
    ts = MDseries(Matrix{Float64}(data), String.(names), [name2tex(n) for n in names], mkdates(first.freq, init.time[1] .+ (0:size(data, 1)-1)),
                  fill("", length(names)), Pair{String,Vector{Any}}[])
    m.session.sc.vars["Simulated_time_series"] = ts
    nothing
end

"M_.endo_histval from the histval block (Dynare: unset values are 0), as stoch_simul starts from it."
function _histval!(ws)
    h = try
        is_ramsey(ws.model) ? _op_histval_vector(ws.model) : _pf_replay(ws.model, nothing, nothing, (), ()).endo_histval
    catch e
        e isa Union{ModelError,PerfectForesightError,SteadyStateError,ArgumentError} || rethrow()
        warn_once(ws.model, "histval could not be evaluated for M_.endo_histval ($(sprint(showerror, e)))")
        nothing
    end
    h === nothing || (ws.M_.endo_histval = _col(h))
end

"""
    @dynare "model.mod" [kw...]

Run the file like MATLAB's `dynare model`: afterwards `M_`, `options_` and `oo_`, and the other variables Dynare
leaves in MATLAB's base workspace (variables set by MATLAB lines in the `.mod` file, `send_*_to_workspace` series,
...), are plain variables in your session, as after `dynare` in MATLAB. Returns the DynareWorkspace. Keywords as
for [`dynare`](@ref), e.g. `@dynare "model.mod" plots=true extra_args=["-Dx=1"]`. To do the same for a workspace
made with `dynare(...)`, call [`to_workspace`](@ref).
"""
macro dynare(file, kws...)
    kwargs = [Expr(:kw, k.args[1], k.args[2]) for k in kws if k isa Expr && k.head == :(=)]
    esc(quote
        local ws = $JDyn.dynare($file; $(kwargs...))
        $JDyn.to_workspace(ws; mod=$__module__)
        ws
    end)
end

"""
    to_workspace(ws=current workspace; mod=Main, vars=true) -> Vector{Symbol}

Define `M_`, `options_` and `oo_` as global variables in module `mod` (your session by default), as MATLAB's
`dynare` leaves them in the base workspace, and make `ws` the workspace that Dynare-named commands act on. With
`vars=true` the workspace's other MATLAB variables are defined too (names that are not valid Julia identifiers, or
that would replace a function or constant, are skipped). Returns the names defined. `@dynare` calls this for you.
"""
function to_workspace(ws::DynareWorkspace=_current(); mod::Module=Main, vars::Bool=true)
    CURRENT[] = ws
    done = Symbol[]
    function def!(name::Symbol, v)
        if isdefined(mod, name) && !(name in (:M_, :options_, :oo_))
            old = getfield(mod, name)
            (old isa Function || old isa Type || old isa Module || isconst(mod, name)) && return
        end
        try
            Core.eval(mod, Expr(:global, Expr(:(=), name, QuoteNode(v))))
            push!(done, name)
        catch
        end
    end
    def!(:M_, ws.M_); def!(:options_, ws.options_); def!(:oo_, ws.oo_)
    if vars
        for (k, v) in ws.vars
            (Base.isidentifier(k) && !(k in ("M_", "options_", "oo_", "ans"))) || continue
            def!(Symbol(k), v)
        end
    end
    done
end
