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
end
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

"Bring the model in line with what the user may have changed in M_ (params, Sigma_e)."
function _sync_model!(ws)
    m = ws.model
    p = vec(Float64.(ws.M_.params))
    length(p) == length(m.params) || throw(ModelError("M_.params has $(length(p)) elements, the model has $(length(m.params)) parameters"))
    m.params .= p
    S = ws.M_.Sigma_e
    size(S) == size(m.Sigma_e) || throw(ModelError("M_.Sigma_e has size $(size(S)), expected $(size(m.Sigma_e))"))
    m.Sigma_e .= S
    ws.M_.Correlation_matrix = _corr(m.Sigma_e)
    m
end
"Parameters the steady state (file or steady_state_model block) has set go back to M_.params, as in Dynare."
function _store_params!(ws, p)
    ws.model.params .= p
    ws.M_.params = _col(p)
end

# ---------------------------------------------------------------------- oo_
function _store_dr!(oo::MStruct, dr::AbstractDict, n::Int)
    D = MStruct()
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
    order = _opti(ws, "order", 2)
    pruning = _optb(ws, "pruning", false)
    order > 1 && m.max_deriv_order < order && throw(ModelError("stoch_simul(order=$order) needs derivatives of order $order: run dynare(...; output=\"$(order == 2 ? "second" : "third")\")"))
    dr, p = if order == 1
        sol = first_order(m)
        as_dr(sol), sol.params
    else
        d = k_order(m; order=order, pruning=pruning)
        d, d["params"]
    end
    _store_params!(ws, p)
    _store_dr!(oo, dr, m.n_endo)
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
            abs(cs[i, i]) < 1e-6 && continue                     # Dynare skips shocks with zero variance
            y = order == 1 ? irf(dr, cs[:, i]; periods=nirf) :
                irf(dr, cs[:, i]; Sigma_e=m.Sigma_e, periods=nirf, order=order, pruning=pruning,
                    drop=_opti(ws, "drop", 100), replic=_opti(ws, "replic", 50), rng=rng)
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
            r = rng === nothing ? Random.default_rng() : rng
            iv = findall(!=(0), diag(m.Sigma_e))
            e = zeros(periods, m.n_exo)
            e[:, iv] .= randn(r, periods, length(iv)) * Matrix(cholesky(Symmetric(m.Sigma_e[iv, iv])).U)
            e
        end
        y = simulate(dr, ex; order=order, pruning=pruning)
        oo.endo_simul = Matrix{Float64}(permutedims(y))        # n_endo x periods, as Dynare's simult
        oo.exo_simul = ex
    end
    if !_optb(ws, "nomoments", false) && order <= 2
        mo = Dict{Symbol,Any}(:order => order, :pruning => pruning, :periods => periods, :drop => _opti(ws, "drop", 100),
                              :ar => _opti(ws, "ar", 5), :hp_filter => _optf(ws, "hp_filter", 0.0),
                              :nocorr => _optb(ws, "nocorr", false), :nodecomposition => _optb(ws, "nodecomposition", false))
        mom = try
            stoch_simul_moments(m; var_list=var_list, dr=dr, shocks=(periods > 0 ? oo.exo_simul : nothing), mo...)
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
            if periods == 0
                isempty(mom.gamma_y) || (oo.gamma_y = _cellrow([g === nothing ? zeros(0, 0) : g for g in mom.gamma_y]))
            else
                mom.skewness === nothing || (oo.skewness = _col(mom.skewness))
                mom.kurtosis === nothing || (oo.kurtosis = _col(mom.kurtosis))
            end
        end
    end
    ws
end

"""
    steady!(ws=current workspace; options...)

Dynare's `steady`: the steady state at the current `M_.params` into `oo_.steady_state`; parameters a
steady-state file or `steady_state_model` block sets go to `M_.params`.
"""
function steady!(ws::DynareWorkspace; options...)
    m = _sync_model!(ws)
    _take_options!(ws, options)
    ys, p = steady_state(m)
    _store_params!(ws, p)
    ws.oo_.steady_state = _col(ys)
    ws
end

"Dynare's `check`: eigenvalues into `oo_.dr.eigval` (throws BlanchardKahnError if the conditions fail)."
function check!(ws::DynareWorkspace; options...)
    m = _sync_model!(ws)
    _take_options!(ws, options)
    sol = first_order(m)
    _store_params!(ws, sol.params)
    D = get(ws.oo_, "dr", MStruct())
    D.eigval = _col(abs.(sol.eigenvalues))
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
    _take_options!(ws, options)
    T = _opti(ws, "periods", 0)
    lag, lead = Int(m.raw["model_info"]["maximum_lag"]), Int(m.raw["model_info"]["maximum_lead"])
    rows = lag + T + lead
    oo = ws.oo_
    (haskey(oo, "endo_simul") && haskey(oo, "exo_simul") && size(oo.endo_simul) == (m.n_endo, rows) &&
     size(oo.exo_simul) == (rows, m.n_exo)) || perfect_foresight_setup!(ws)
    E = oo.endo_simul
    kw = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in options if k in (:tolf, :tolx, :maxit, :homotopy, :lmmcp, :stack_solve_algo, :markowitz,
                                                                        :endval_steady, :linear_approximation, :robust_lin_solve))
    r = perfect_foresight(m; periods=T, exo=Matrix{Float64}(oo.exo_simul),
                          y_init=lag > 0 ? E[:, 1:lag] : nothing, y_term=lead > 0 ? E[:, lag+T+1:end] : nothing,
                          guess=E[:, lag+1:lag+T], raise_on_failure=raise_on_failure, kw...)
    _store_pf!(ws, r)
end

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

"Dynare's `occbin_solver` at the current `M_.params` (shocks from the `shocks(surprise)` block)."
function occbin_solver!(ws::DynareWorkspace; options...)
    _sync_model!(ws)
    _take_options!(ws, Dict(Symbol("occbin." * String(k)) => v for (k, v) in options))
    ob = Occbin(ws.model)
    r = simulate(ob)
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

# Dynare's commands act on the current workspace (the last one `dynare` made), as MATLAB's act on globals
const CURRENT = Ref{Union{Nothing,DynareWorkspace}}(nothing)
_current() = CURRENT[] === nothing ? throw(ArgumentError("no Dynare workspace yet: run dynare(\"model.mod\") or @dynare first")) : CURRENT[]
for f in (:stoch_simul!, :steady!, :check!, :perfect_foresight_setup!, :perfect_foresight_solver!, :occbin_solver!)
    @eval $f(; kw...) = $f(_current(); kw...)
end
stoch_simul!(var_list::AbstractVector; kw...) = stoch_simul!(_current(), var_list; kw...)
set_param_value!(name::AbstractString, v::Real) = set_param_value!(_current(), name, v)

"""
    dynare(modfile; output=nothing, shocks=Dict(), rng=nothing, kw...) -> DynareWorkspace

Run a .mod file as `dynare modfile` does and return M_, options_ and oo_ as MATLAB-style structs:
`ws.oo_.endo_simul`, `ws.oo_.dr.ghx`, `ws.oo_.irfs.y_e`, `ws.M_.params`, ... (Julia reads them with
MATLAB's syntax). `@dynare "model.mod"` also defines `M_`, `options_` and `oo_` in the caller's module.

`shocks` maps the position of a `stoch_simul` statement among the file's stoch_simul statements (1, 2, ...)
to a (periods x n_exo) shock matrix, to reproduce a given simulation (e.g. Dynare's `oo_.exo_simul`);
otherwise shocks are drawn with `rng` (Julia's random numbers, so paths differ from MATLAB's).

Commands run after the whole file has been loaded, with the final calibration. Afterwards, re-run them on
the workspace without preprocessing: `M_.params[i] = x` or `set_param_value!("rho", x)`, then `stoch_simul!()`;
edit `oo_.exo_simul`, then `perfect_foresight_solver!()`.
"""
function dynare(modfile::AbstractString; output=nothing, shocks=Dict{Int,Any}(), rng=nothing, kw...)
    if output === nothing
        # derivatives up to the highest order a stoch_simul statement asks for (Dynare's default is 2)
        output = occursin(r"order\s*=\s*3", read(modfile, String)) ? "third" : "second"
    end
    m = load(modfile; output=output, kw...)
    ws = DynareWorkspace(m, build_M(m), build_options(m), MStruct(), Dict{String,Any}())
    oo = ws.oo_
    oo.steady_state = _col(Float64[get(m.oo, v, 0.0) for v in m.endo_names])
    oo.exo_steady_state = _col(exo_steady(m))
    oo.exo_det_steady_state = zeros(0, 0)
    oo.dynare_version = "7.2"
    nss = 0
    CURRENT[] = ws
    for st in m.statements
        kind = st["statementName"]
        _apply_options!(ws.options_, st)
        if kind == "steady"
            steady!(ws)
        elseif kind == "check"
            check!(ws)
        elseif kind == "stoch_simul"
            nss += 1
            stoch_simul!(ws, String.(get_or(st, "symbol_list", String[])); shocks=get(shocks, nss, nothing), rng=rng)
        elseif kind == "perfect_foresight_setup"
            perfect_foresight_setup!(ws)
        elseif kind == "perfect_foresight_solver"
            perfect_foresight_solver!(ws)
        elseif kind == "perfect_foresight_with_expectation_errors_solver"
            _perfect_foresight_wee!(ws)
        elseif kind == "occbin_solver"
            occbin_solver!(ws)
        elseif kind in COMPUTE_KINDS && !(kind in ("perfect_foresight_with_expectation_errors_setup",
                                                   "occbin_setup", "occbin_graph", "rplot", "write_latex_dynamic_model",
                                                   "write_latex_static_model", "write_latex_original_model",
                                                   "write_latex_steady_state_model", "collect_latex_files", "write_latex_definitions"))
            warn_once(m, "dynare(): '$kind' does not fill oo_ yet")
        end
    end
    for (k, v) in m.session.sc.vars
        v isa Tainted || v isa HostStruct || (ws.vars[k] = v)
    end
    ws
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
        global M_ = ws.M_
        global options_ = ws.options_
        global oo_ = ws.oo_
        ws
    end)
end
