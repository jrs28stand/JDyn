# Deterministic (perfect foresight) simulation: port of pydyn/perfect_foresight.py, which ports parts of
# Dynare 7.2 (matlab/perfect-foresight-models/*.m, matlab/lmmcp/lmmcp.m and the preprocessor's handling of
# initval, endval, histval, shocks and mshocks). See NOTICE.md for provenance.
#
# `perfect_foresight(m)` does what `perfect_foresight_setup` followed by `perfect_foresight_solver` do in
# Dynare (or the `perfect_foresight_with_expectation_errors` pair when the .mod file uses it), reading
# initial and terminal conditions, shocks and options from the .mod file:
#
# * initval, endval, steady, histval, shocks, mshocks, shocks(learnt_in=...), endval(learnt_in=...) are
#   replayed in the order of the file, so that oo_.steady_state, oo_.exo_steady_state and
#   oo_.initial_steady_state at perfect_foresight_setup are reproduced. Without `steady` after `initval`
#   the path starts from (and, without `endval`, ends at) the values written in `initval`.
# * Solvers: Newton on the stacked system with a sparse LU factorisation (Dynare's `sim1`), one linear
#   solve for model(linear) / linear_approximation, period-by-period solvers for purely backward, purely
#   forward and static models, Dynare 7's homotopy on the size of the shocks with its two linearization
#   fallbacks, and the LMMCP algorithm of Kanzow and Petra for models with complementarity conditions.
#
# Layout of the result: `endo_simul` is Dynare's oo_.endo_simul, n_endo x (maximum_lag + periods +
# maximum_lead): with lags and leads column 1 is the initial condition (period 0), columns 2 .. periods+1
# the simulation, the last the terminal condition; a model without lags has no initial column, a model
# without leads no terminal column. `exo_simul` is Dynare's oo_.exo_simul (same number of rows, one column
# per exogenous variable). Leads and lags beyond one period are carried by the auxiliary variables the
# preprocessor adds (extra rows of endo_simul), as in Dynare.


"Raised when no solution is found; `result` holds the last iterate (status false)."
struct PerfectForesightError <: Exception
    msg::String
    result::Any
end
PerfectForesightError(msg::AbstractString) = PerfectForesightError(String(msg), nothing)
Base.showerror(io::IO, e::PerfectForesightError) = print(io, "PerfectForesightError: ", e.msg)

"""
Outcome of a deterministic simulation. `endo_simul` (n_endo x rows) and `exo_simul` (rows x n_exo) are
Dynare's oo_.endo_simul and oo_.exo_simul; `iterations`, `max_residual` and `status` are
oo_.deterministic_simulation.iterations, .error and .status. `res["k"]` is the path of a variable over
all columns of `endo_simul` (or of an exogenous variable).
"""
mutable struct PerfectForesightResult
    endo_simul::Matrix{Float64}
    exo_simul::Matrix{Float64}
    iterations::Int
    max_residual::Float64
    status::Bool
    endo_names::Vector{String}
    exo_names::Vector{String}
    periods::Int
    maximum_lag::Int
    maximum_lead::Int
    steady_state::Vector{Float64}                          # oo_.steady_state (terminal)
    initial_steady_state::Union{Nothing,Vector{Float64}}   # oo_.initial_steady_state (nothing without endval)
    exo_steady_state::Vector{Float64}
    initial_exo_steady_state::Union{Nothing,Vector{Float64}}
    params::Vector{Float64}
    homotopy_completion_share::Float64
    homotopy_iterations::Int
    homotopy_linearization::Bool
    homotopy_marginal_linearization::Bool
    solver::String
    info::Dict{String,Any}
end

function Base.getindex(r::PerfectForesightResult, name::AbstractString)
    i = findfirst(==(name), r.endo_names)
    i !== nothing && return r.endo_simul[i, :]
    j = findfirst(==(name), r.exo_names)
    j !== nothing && return r.exo_simul[:, j]
    throw(KeyError(name))
end

function Base.show(io::IO, r::PerfectForesightResult)
    print(io, "PerfectForesightResult(", r.status ? "solved" : "FAILED", ", ", r.periods, " periods, solver ", r.solver,
          ", ", r.iterations, " iterations, max residual ", @sprintf("%.3g", r.max_residual), ")")
end

"""
    paths(res; deviation=nothing, exo=false, simulation_only=false) -> Dict(name => vector)

Paths of the variables: levels, or deviations from the `:initial` or `:terminal` steady state
(Dynare's oo_.initial_steady_state / oo_.steady_state; without `endval` both are the same).
`simulation_only` drops the initial and terminal columns.
"""
function paths(r::PerfectForesightResult; deviation=nothing, exo::Bool=false, simulation_only::Bool=false)
    deviation in (nothing, :initial, :terminal) || throw(ArgumentError("deviation must be nothing, :initial or :terminal"))
    cols = simulation_only ? (r.maximum_lag+1:r.maximum_lag+r.periods) : (1:size(r.endo_simul, 2))
    base = zeros(length(r.endo_names)); bx = zeros(length(r.exo_names))
    if deviation !== nothing
        initial = deviation === :initial && r.initial_steady_state !== nothing
        base = initial ? r.initial_steady_state : r.steady_state
        bx = initial ? r.initial_exo_steady_state : r.exo_steady_state
    end
    out = Dict{String,Vector{Float64}}()
    for (i, v) in enumerate(r.endo_names)
        out[v] = r.endo_simul[i, cols] .- base[i]
    end
    if exo
        for (j, v) in enumerate(r.exo_names)
            out[v] = r.exo_simul[cols, j] .- bx[j]
        end
    end
    out
end

# ------------------------------------------------------------------------------------------ options
const PF_SOLVER_KEYS = Dict("tolf" => "dynatol.f", "tolx" => "dynatol.x", "maxit" => "simul.maxit",
    "stack_solve_algo" => "stack_solve_algo", "no_homotopy" => "no_homotopy",
    "linear_approximation" => "linear_approximation", "endogenous_terminal_period" => "endogenous_terminal_period",
    "lmmcp" => "lmmcp.status", "homotopy_initial_step_size" => "simul.homotopy_initial_step_size",
    "homotopy_min_step_size" => "simul.homotopy_min_step_size",
    "homotopy_step_size_increase_success_count" => "simul.homotopy_step_size_increase_success_count",
    "homotopy_max_completion_share" => "simul.homotopy_max_completion_share",
    "homotopy_linearization_fallback" => "simul.homotopy_linearization_fallback",
    "homotopy_marginal_linearization_fallback" => "simul.homotopy_marginal_linearization_fallback",
    "homotopy_exclude_varexo" => "simul.homotopy_exclude_varexo", "steady_tolf" => "simul.steady_tolf",
    "constant_simulation_length" => "pfwee.constant_simulation_length", "noprint" => "noprint")
const PF_SETUP_KEYS = Dict("periods" => "periods", "endval_steady" => "simul.endval_steady",
    "endval_steady_nocheck" => "simul.endval_steady_nocheck", "datafile" => "datafile")
const PF_DEFAULTS = Dict{String,Any}("dynatol.f" => nothing, "dynatol.x" => 1e-5, "simul.maxit" => 50,
    "stack_solve_algo" => 0, "no_homotopy" => false, "linear_approximation" => false,
    "endogenous_terminal_period" => false, "lmmcp.status" => false, "simul.homotopy_initial_step_size" => 1.0,
    "simul.homotopy_min_step_size" => 1e-3, "simul.homotopy_step_size_increase_success_count" => 3,
    "simul.homotopy_max_completion_share" => 1.0, "simul.homotopy_linearization_fallback" => false,
    "simul.homotopy_marginal_linearization_fallback" => 0.0, "simul.homotopy_exclude_varexo" => String[],
    "simul.steady_tolf" => nothing, "pfwee.constant_simulation_length" => false, "noprint" => false,
    "periods" => nothing, "simul.endval_steady" => false, "simul.endval_steady_nocheck" => false, "datafile" => nothing)
"Dynare's default tolerance on the residuals of a perfect foresight simulation (options_.dynatol.f)."
const PF_DEFAULT_TOLF = 1e-5
const PF_NOTES = Dict("markowitz" => "markowitz only matters for Dynare's bytecode solvers",
    "minimal_solving_periods" => "minimal_solving_periods only matters for Dynare's bytecode solvers",
    "simul.robust_lin_solve" => "robust_lin_solve (iterative fallbacks for singular stacked Jacobians) is not implemented: a singular Jacobian makes the simulation fail",
    "simul.steady_solve_algo" => "steady_solve_algo: JDyn has one steady-state solver",
    "simul.steady_maxit" => "steady_maxit: JDyn has one steady-state solver",
    "simul.steady_tolx" => "steady_tolx: JDyn has one steady-state solver",
    "simul.steady_markowitz" => "steady_markowitz only matters for Dynare's bytecode solvers",
    "solve_algo" => "solve_algo: period-by-period problems are solved by Newton's method (then a trust-region method)")

_pf_statement_options(m::Model, name) = (out = Dict{String,Any}(); for st in m.statements
    st["statementName"] == name && (out = Dict{String,Any}(get_or(st, "options", Dict{String,Any}()))); end; out)

_pf_bool(v) = v === true || (v isa Number && v != 0) || (v isa AbstractString && lowercase(v) in ("true", "1"))
_pf_float(v) = v isa Real ? Float64(v) : v isa AbstractString ? parse(Float64, v) : throw(ArgumentError("expected a number, got $(repr(v))"))

function _pf_resolve_options(m::Model, pfwee::Bool, kw)
    names = pfwee ? ("perfect_foresight_with_expectation_errors_setup", "perfect_foresight_with_expectation_errors_solver") :
                    ("perfect_foresight_setup", "perfect_foresight_solver")
    setup, solver = (_pf_statement_options(m, n) for n in names)
    o = copy(PF_DEFAULTS)
    for (k, v) in setup; haskey(PF_DEFAULTS, k) && (o[k] = v); end
    for (k, v) in solver; haskey(PF_DEFAULTS, k) && (o[k] = v); end
    given = Set{String}(union(keys(setup), keys(solver)))
    for (k, v) in kw
        key = get(PF_SOLVER_KEYS, String(k), get(PF_SETUP_KEYS, String(k), nothing))
        key === nothing && throw(ArgumentError("perfect_foresight() got an unexpected keyword argument '$k'"))
        if v !== nothing
            o[key] = v
            push!(given, key)
        end
    end
    excl = o["simul.homotopy_exclude_varexo"]
    excl isa AbstractDict && (excl = get(excl, "symbol_list", String[]))
    o["simul.homotopy_exclude_varexo"] = excl isa AbstractString ? [String(excl)] : String[String(x) for x in something(excl, String[])]
    o["dynatol.f"] = o["dynatol.f"] === nothing ? PF_DEFAULT_TOLF : _pf_float(o["dynatol.f"])
    o["dynatol.x"] = _pf_float(o["dynatol.x"])
    o["simul.maxit"] = Int(_pf_float(o["simul.maxit"]))
    for k in ("simul.homotopy_initial_step_size", "simul.homotopy_min_step_size", "simul.homotopy_max_completion_share")
        o[k] = _pf_float(o[k])
    end
    marg = o["simul.homotopy_marginal_linearization_fallback"]
    o["simul.homotopy_marginal_linearization_fallback"] = marg === true ? 0.01 : marg === false || marg === nothing ? 0.0 : _pf_float(marg)
    for k in ("no_homotopy", "linear_approximation", "endogenous_terminal_period", "lmmcp.status", "simul.homotopy_linearization_fallback",
              "pfwee.constant_simulation_length", "simul.endval_steady", "simul.endval_steady_nocheck")
        o[k] = _pf_bool(o[k])
    end
    o["simul.homotopy_step_size_increase_success_count"] = Int(_pf_float(o["simul.homotopy_step_size_increase_success_count"]))
    algo = Int(_pf_float(o["stack_solve_algo"]))
    o["stack_solve_algo"] = algo
    if "stack_solve_algo" in given
        0 <= algo <= 10 || throw(ArgumentError("stack_solve_algo must be between 0 and 10"))
        warn_once(m, "stack_solve_algo=$algo: JDyn has one solver for the stacked system, Newton's method with a sparse LU " *
                     "factorisation (Dynare's stack_solve_algo=0); every value of stack_solve_algo is solved that way")
    end
    for (st, src) in zip(names, (setup, solver))
        notes = ["$k ($(PF_NOTES[k]))" for k in keys(src) if haskey(PF_NOTES, k)]
        other = sort([k for k in keys(src) if !haskey(PF_DEFAULTS, k) && !haskey(PF_NOTES, k) && !(k in COSMETIC) && k != "stack_solve_algo"])
        isempty(notes) && isempty(other) ||
            warn_once(m, "perfect_foresight does not act on these $st options from the .mod file: " * join(vcat(sort(notes), other), "; "))
    end
    o
end

"Argument checks of perfect_foresight_solver.m."
function _pf_check_homotopy_options(o)
    init, mn, mx = o["simul.homotopy_initial_step_size"], o["simul.homotopy_min_step_size"], o["simul.homotopy_max_completion_share"]
    lin, marg = o["simul.homotopy_linearization_fallback"], o["simul.homotopy_marginal_linearization_fallback"]
    if o["no_homotopy"] && (init != 1 || mx != 1 || lin || marg != 0)
        throw(ArgumentError("the no_homotopy option is incompatible with homotopy_initial_step_size, homotopy_max_completion_share, " *
                            "homotopy_linearization_fallback and homotopy_marginal_linearization_fallback"))
    end
    for (nm, v) in (("homotopy_initial_step_size", init), ("homotopy_min_step_size", mn), ("homotopy_max_completion_share", mx),
                    ("homotopy_marginal_linearization_fallback", marg))
        0 <= v <= 1 || throw(ArgumentError("$nm must be in the [0,1] interval"))
    end
    init >= mn || throw(ArgumentError("homotopy_initial_step_size must be greater than or equal to homotopy_min_step_size"))
    (lin && marg > 0) && throw(ArgumentError("homotopy_linearization_fallback and homotopy_marginal_linearization_fallback cannot be used together"))
    (mx < 1 && !lin && marg == 0) && throw(ArgumentError("homotopy_max_completion_share is less than 1, so " *
        "homotopy_linearization_fallback or homotopy_marginal_linearization_fallback must also be given"))
end

# ------------------------------------------------------------------------------------------ setup
"What Dynare holds when perfect_foresight_setup runs (the oo_ and M_ fields it uses)."
mutable struct PFWorkspace
    params::Vector{Float64}
    ss::Vector{Float64}                       # oo_.steady_state
    xss::Vector{Float64}                      # oo_.exo_steady_state
    ss0::Union{Nothing,Vector{Float64}}       # oo_.initial_steady_state
    xss0::Union{Nothing,Vector{Float64}}
    det_shocks::Vector{Tuple{Int,Int,Int,Float64,String}}     # (exo, first period, last period, value, type)
    learnt_shocks::Vector{Dict{String,Any}}                   # learnt_in, exo, p1, p2, type, value
    learnt_endval::Vector{Dict{String,Any}}                   # learnt_in, exo, type, value
    endo_histval::Union{Nothing,Vector{Float64}}
    exo_histval::Union{Nothing,Vector{Float64}}
    dirty::Bool                               # values set by initval/endval since the last steady
    pfwee::Bool
    has_setup::Bool
end

_pf_is_setup(st) = st["statementName"] in ("perfect_foresight_setup", "perfect_foresight_with_expectation_errors_setup")

function _pf_period(v, what)
    v isa Integer && return Int(v)
    v isa Real && isinteger(v) && return Int(v)
    v isa AbstractString && occursin(r"^\s*-?\d+\s*$", v) && return parse(Int, v)
    throw(ModelError("$what given as a date or an expression ($(repr(v))): dates are not supported"))
end

"Steady state of the model at `x`, from the guess `guess` (variables a `steady_state_model` block does not set start there)."
function _pf_steady(m::Model, params, guess, x; tol=nothing, check=nothing)
    saved = m.initval
    d = Dict{String,Float64}()
    for (i, v) in enumerate(m.endo_names)
        isfinite(guess[i]) && (d[v] = guess[i])
    end
    for (j, v) in enumerate(m.exo_names)
        haskey(saved, v) && (d[v] = saved[v])
    end
    m.initval = d
    try
        return with_logger(NullLogger()) do
            steady_state(m; params=copy(params), guess=copy(guess), exo=Vector{Float64}(x), tol=tol, check=check)
        end
    finally
        m.initval = saved
    end
end

"Replay the statements of the .mod file up to the last perfect_foresight_setup."
function _pf_replay(m::Model, params=nothing, steady=nothing, extra_learnt_shocks=(), extra_learnt_endval=())
    sts = m.statements
    idx = [i for (i, st) in enumerate(sts) if _pf_is_setup(st)]
    last = isempty(idx) ? length(sts) : idx[end]
    ws = PFWorkspace(params === nothing ? copy(m.params) : Vector{Float64}(params), zeros(m.n_endo), zeros(m.n_exo), nothing, nothing,
                     Tuple{Int,Int,Int,Float64,String}[], Dict{String,Any}[], Dict{String,Any}[], nothing, nothing, true, false, !isempty(idx))
    force = steady === nothing ? isempty(idx) : steady === true
    never = steady === false
    ie, ix = m.ie, m.ix
    counts = Dict{Tuple{String,String},Int}()
    for st in sts[1:last]
        if st["statementName"] in ("initval", "endval")
            for d in get_or(st, "vals", [])
                k = (st["statementName"], d["name"])
                counts[k] = get(counts, k, 0) + 1
            end
        end
    end
    function run_steady!(nocheck=nothing)
        guess = fill_auxiliary(m, ws.ss, ws.xss, ws.params)
        ys, p = _pf_steady(m, ws.params, guess, ws.xss; check=nocheck === nothing ? nothing : !nocheck)
        ws.ss, ws.params = ys, p
        ws.dirty = false
    end
    function set_values!(st, stored)
        kind = st["statementName"]
        for d in get_or(st, "vals", [])
            name = d["name"]
            # the values Dynare holds after the block, as computed by the model loader; a variable set in
            # several blocks is evaluated again here (possible when the expression has no variable)
            val = (haskey(stored, name) && get(counts, (kind, name), 0) <= 1) ? Float64(stored[name]) : Float64(_eval_with_vars(m, d["value"]))
            haskey(ie, name) ? (ws.ss[ie[name]] = val) : haskey(ix, name) ? (ws.xss[ix[name]] = val) : nothing
        end
        ws.ss = fill_auxiliary(m, ws.ss, ws.xss, ws.params)     # the driver the preprocessor writes sets the auxiliary variables
        ws.dirty = true
    end
    for st in sts[1:last]
        kind = st["statementName"]
        if kind == "initval"
            set_values!(st, m.initval)
        elseif kind == "endval" && haskey(st, "learnt_in")
            for d in get_or(st, "vals", [])
                push!(ws.learnt_endval, Dict{String,Any}("learnt_in" => _pf_period(st["learnt_in"], "learnt_in"), "exo" => ix[d["name"]],
                      "type" => get(d, "type", "level"), "value" => Float64(_value(m, d["value"]))))
            end
        elseif kind == "endval"
            force && ws.dirty && ws.ss0 === nothing && run_steady!()
            if ws.ss0 === nothing
                ws.ss0, ws.xss0 = copy(ws.ss), copy(ws.xss)
            end
            set_values!(st, m.endval)
        elseif kind == "load_params_and_steady_state"
            for d in get_or(st, "values", [])
                if haskey(ie, d["name"]) && haskey(m.initval, d["name"])
                    ws.ss[ie[d["name"]]] = m.initval[d["name"]]
                elseif haskey(ix, d["name"]) && haskey(m.initval, d["name"])
                    ws.xss[ix[d["name"]]] = m.initval[d["name"]]
                end
            end
            ws.dirty = true
        elseif kind == "steady" && !never
            run_steady!()
        elseif kind == "histval"
            _pf_histval!(m, ws, st)
        elseif kind in ("shocks", "mshocks") && haskey(st, "learnt_in")
            lin = _pf_period(st["learnt_in"], "learnt_in")
            get(st, "overwrite", false) == true && filter!(d -> d["learnt_in"] != lin, ws.learnt_shocks)
            for d in get_or(st, "learnt_shocks", [])
                for v in d["values"]
                    push!(ws.learnt_shocks, Dict{String,Any}("learnt_in" => lin, "exo" => ix[d["var"]], "p1" => _pf_period(v["period1"], "period"),
                          "p2" => _pf_period(v["period2"], "period"), "type" => get(v, "type", "level"), "value" => Float64(_value(m, v["value"]))))
                end
            end
        elseif kind in ("shocks", "mshocks") && get(st, "surprise", false) != true
            get(st, "overwrite", false) == true && empty!(ws.det_shocks)
            typ = kind == "shocks" ? "level" : (get(st, "relative_to_initval", false) == true ? "multiply_initial_steady_state" : "multiply_steady_state")
            for d in get_or(st, "deterministic_shocks", [])
                haskey(ix, d["var"]) || throw(ModelError("shock on $(d["var"]): deterministic exogenous variables are not supported"))
                for v in d["values"]
                    push!(ws.det_shocks, (ix[d["var"]], _pf_period(v["period1"], "period"), _pf_period(v["period2"], "period"),
                                          Float64(_value(m, v["value"])), typ))
                end
            end
        elseif kind in ("initval_file", "histval_file", "perfect_foresight_controlled_paths", "shock_paths", "smoother2histval")
            throw(ModelError("$kind is not implemented in JDyn's perfect foresight"))
        elseif _pf_is_setup(st)
            ws.pfwee = kind != "perfect_foresight_setup"
        end
    end
    force && ws.dirty && run_steady!()
    for d in extra_learnt_shocks
        d = Dict{String,Any}(String(k) => v for (k, v) in pairs(d))
        p = pop!(d, "periods", nothing)
        if p !== nothing
            d["p1"], d["p2"] = p isa Real ? (Int(p), Int(p)) : (Int(first(p)), Int(last(p)))
        end
        d["exo"] = haskey(d, "var") ? ix[pop!(d, "var")] : d["exo"]
        get!(d, "type", "level")
        push!(ws.learnt_shocks, d)
    end
    for d in extra_learnt_endval
        d = Dict{String,Any}(String(k) => v for (k, v) in pairs(d))
        d["exo"] = haskey(d, "var") ? ix[pop!(d, "var")] : d["exo"]
        get!(d, "type", "level")
        push!(ws.learnt_endval, d)
    end
    ws
end

const _PF_SIMPLE_AUX = r"^\s*([A-Za-z_]\w*)\s*(?:\(\s*(-?\d+)\s*\))?\s*$"

"""
M_.endo_histval and M_.exo_histval as the histval block and dynamic_set_auxiliary_series build them.
Variables the block does not mention are 0 in period 0 (not their steady state).
"""
function _pf_histval!(m::Model, ws::PFWorkspace, st)
    info = m.info
    L = Int(get(info, "orig_maximum_lag_with_diffs_expanded", get(info, "orig_maximum_lag", 1)))
    info["maximum_lag"] == 0 && return                   # M_.endo_histval has no column: ignored by Dynare
    L < 1 && return
    series = Dict{String,Vector{Float64}}(v => zeros(L) for v in vcat(orig_endo_names(m), m.exo_names))     # dates -L+1 .. 0
    for d in get_or(st, "vals", [])
        lag = Int(_numval(d["lag"]))
        -L < lag <= 0 || throw(ModelError("histval: $(d["name"])($lag) is outside the lags of the model"))
        haskey(series, d["name"]) && (series[d["name"]][L+lag] = Float64(_value(m, d["value"])))
    end
    defs = Dict{Any,Any}(get(eq, "lhs", nothing) => get(eq, "rhs", nothing) for eq in get_or(m.raw, "model", []) if eq isa AbstractDict)
    for a in get_or(m.raw, "aux_vars", [])
        name = m.endo_names[Int(a["endo_index"])]
        mo = match(_PF_SIMPLE_AUX, string(get(defs, name, "")))
        (get(a, "type", -1) in (0, 1, 2, 3) && mo !== nothing && haskey(series, mo.captures[1])) ||
            throw(ModelError("histval with the auxiliary variable $name = $(get(defs, name, nothing)) is not supported"))
        src, shift = series[mo.captures[1]], mo.captures[2] === nothing ? 0 : parse(Int, mo.captures[2])
        out = fill(NaN, L)
        for t in 1:L
            1 <= t + shift <= L && (out[t] = src[t+shift])
        end
        series[name] = out
    end
    h = Float64[series[v][end] for v in m.endo_names]
    ws.endo_histval = replace(h, NaN => 0.0)
    ws.exo_histval = Float64[series[v][end] for v in m.exo_names]
    nothing
end

"make_ex_.m and make_y_.m: oo_.endo_simul (n x rows) and oo_.exo_simul transposed (n_exo x rows)."
function _pf_make_paths(m::Model, ws::PFWorkspace, periods::Int)
    lag, lead = m.info["maximum_lag"], m.info["maximum_lead"]
    rows = lag + periods + lead
    (ws.endo_histval !== nothing && ws.ss0 !== nothing) && throw(PerfectForesightError("histval and endval cannot be used simultaneously"))
    x = repeat(ws.xss, 1, rows)
    y = repeat(ws.ss, 1, rows)
    if ws.ss0 !== nothing
        x[:, 1:lag] .= ws.xss0
        y[:, 1:lag] .= ws.ss0
    end
    if ws.endo_histval !== nothing && lag > 0
        x[:, 1:lag] .= ws.exo_histval
        y[:, 1:lag] .= ws.endo_histval
    end
    late = sort(unique([m.exo_names[j] for (j, p1, p2, _, _) in ws.det_shocks if p2 > periods || p1 > periods]))
    isempty(late) || throw(PerfectForesightError("a shock is dated after the end of the simulation: at least one expected value for " *
                                                  "$(join(late, ", ")) has been declared after the terminal period"))
    for (j, p1, p2, val, typ) in ws.det_shocks
        k1, k2 = p1 + lag, p2 + lag                      # 1-based columns
        k1 < 1 && throw(PerfectForesightError("the model does not have lags, so you cannot set values for period 0"))
        if typ == "multiply_steady_state"
            val = ws.xss[j] * val
        elseif typ == "multiply_initial_steady_state"
            ws.xss0 === nothing && throw(PerfectForesightError("option relative_to_initval of mshocks cannot be used without an endval block"))
            val = ws.xss0[j] * val
        end
        x[j, k1:k2] .= val
    end
    y, x
end

"Exogenous path (rows x n_exo) from initval/endval/histval and the deterministic shocks/mshocks blocks, as perfect_foresight_setup builds oo_.exo_simul."
pf_exo_path(m::Model, periods::Integer) = permutedims(_pf_make_paths(m, _pf_replay(m, nothing, false), Int(periods))[2])

# ------------------------------------------------------------------------------------------ stacked problem
_pf_norm_inf(v) = (isempty(v) ? 0.0 : (any(isnan, v) ? NaN : maximum(abs, v)))

"""
Residuals and Jacobian of the stacked system F(y[t-1], y[t], y[t+1], x[t]) = 0, t = 1..T. The sparse
pattern is built once per T and the Jacobian is filled in place. `eq_pos[r]` is the row (within a
period) of equation r (complementarity problems reorder the equations).
"""
mutable struct PFStacked
    m::Model
    params::Vector{Float64}
    ss::Vector{Float64}
    n::Int
    nx::Int
    eq_pos::Vector{Int}
    er::Vector{Int}                 # structural non-zeros of one period's Jacobian that are endogenous (row, column in 1:3n)
    ec::Vector{Int}
    sel::Vector{Int}                # their indices in the full non-zero list of g1
    patterns::Dict{Int,Any}
    tbuf::Vector{Float64}
    rbuf::Vector{Float64}
    gbuf::Vector{Float64}
    yy::Vector{Float64}
    xx::Vector{Float64}
    evaluations::Int
end

function PFStacked(m::Model, params, ss; eq_index=nothing)
    n = m.n_endo
    f = m.fns
    sel = findall(c -> c <= 3n, f.g1_cols)
    pos = collect(1:n)
    if eq_index !== nothing
        for (j, e) in enumerate(eq_index)
            pos[e] = j
        end
    end
    PFStacked(m, Vector{Float64}(params), Vector{Float64}(ss), n, m.n_exo, pos, pos[f.g1_rows[sel]], f.g1_cols[sel], sel,
              Dict{Int,Any}(), zeros(max(f.dyn_tmp[1] + f.dyn_tmp[2], 1)), zeros(n), zeros(length(f.g1_rows)), zeros(3n),
              zeros(m.n_exo), 0)
end

"Sparse pattern of the stacked Jacobian for T periods, with the buffers filled in place and the LU factorisation (symbolic analysis reused)."
mutable struct PFPattern
    A::SparseMatrixCSC{Float64,Int}
    perm::Vector{Int}                # position in V (K x T) of each stored entry of A
    res::Vector{Float64}
    V::Matrix{Float64}
    F::Any
end

function _pf_pattern(P::PFStacked, T::Int)::PFPattern
    get!(P.patterns, T) do
        n, K = P.n, length(P.er)
        rows = Int[]; cols = Int[]; lin = Int[]
        sizehint!(rows, T * K)
        for t in 1:T, k in 1:K
            gc = (t - 2) * n + P.ec[k]
            (1 <= gc <= T * n) || continue
            push!(rows, (t - 1) * n + P.er[k]); push!(cols, gc); push!(lin, (t - 1) * K + k)
        end
        A = sparse(rows, cols, Float64.(1:length(rows)), T * n, T * n)
        nnz(A) == length(rows) || throw(ModelError("stacked Jacobian has duplicate entries"))
        perm = Int[lin[Int(v)] for v in A.nzval]                 # position in V (K x T) of each stored entry
        PFPattern(A, perm, zeros(T * n), zeros(K, T), nothing)
    end
end

# function barrier: the generated functions are called with concrete types, once per period
function _pf_eval_periods!(res, V, f_rtt, f_r, f_gtt, f_g, full, X, params, ss, T, n, nx, tbuf, rbuf, gbuf, yy, xx, sel, pos, jac)
    K = length(sel)
    for t in 1:T
        copyto!(yy, 1, full, (t - 1) * n + 1, 3n)
        nx > 0 && copyto!(xx, 1, X, (t - 1) * nx + 1, nx)
        ok = true
        try
            if jac
                f_gtt(tbuf, yy, xx, params, ss)
                f_r(tbuf, rbuf, yy, xx, params, ss)
                f_g(tbuf, gbuf, yy, xx, params, ss)
            else
                f_rtt(tbuf, yy, xx, params, ss)
                f_r(tbuf, rbuf, yy, xx, params, ss)
            end
        catch e
            e isa DomainError || rethrow()
            ok = false
        end
        base = (t - 1) * n
        if ok
            @inbounds for r in 1:n
                res[base+pos[r]] = rbuf[r]
            end
            if jac
                @inbounds for k in 1:K
                    V[k, t] = gbuf[sel[k]]
                end
            end
        else
            @inbounds for r in 1:n
                res[base+pos[r]] = NaN
            end
            jac && (V[:, t] .= NaN)
        end
    end
    nothing
end

"""
    evaluate!(P, Y, y0, yT, X, jac) -> (res, A)

`Y` is n x T (the simulated periods), `y0` and `yT` the boundary columns, `X` n_exo x T. `res` is the
residual vector of length T n; `A` the stacked Jacobian (or `nothing`). Both are buffers reused by the
next call with the same T.
"""
function evaluate!(P::PFStacked, Y::AbstractMatrix, y0, yT, X::AbstractMatrix, jac::Bool=true)
    n, T = P.n, size(Y, 2)
    P.evaluations += 1
    pat = _pf_pattern(P, T)
    A, perm, res, V = pat.A, pat.perm, pat.res, pat.V
    full = Matrix{Float64}(undef, n, T + 2)
    full[:, 1] .= y0; full[:, T+2] .= yT; full[:, 2:T+1] .= Y
    f = P.m.fns
    Base.invokelatest(_pf_eval_periods!, res, V, f.dyn_resid_tt, f.dyn_resid, f.dyn_g1_tt, f.dyn_g1, full, X, P.params, P.ss, T, n, P.nx,
                      P.tbuf, P.rbuf, P.gbuf, P.yy, P.xx, P.sel, P.eq_pos, jac)
    if jac
        nz = A.nzval
        @inbounds for i in eachindex(nz)
            nz[i] = V[perm[i]]
        end
        return res, A
    end
    res, nothing
end

"""
A \\ b with a sparse LU factorisation; a singular matrix gives NaN. With `pat` (the pattern `A` was built
from) the symbolic analysis of the first factorisation is reused by the next ones.
"""
function _pf_lu_solve(A::SparseMatrixCSC, b::AbstractVector, pat=nothing)
    try
        if pat !== nothing && pat.A === A
            if pat.F === nothing
                pat.F = lu(A; check=false)
            else
                lu!(pat.F, A; check=false)
            end
            F = pat.F
        else
            F = lu(A; check=false)
        end
        issuccess(F) || return fill(NaN, length(b))
        return F \ b
    catch e
        e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException || e isa ArgumentError || rethrow()
        return fill(NaN, length(b))
    end
end

function _pf_boundaries(m::Model, endo::AbstractMatrix, periods::Int)
    lag, lead, n = m.info["maximum_lag"], m.info["maximum_lead"], m.n_endo
    y0 = lag > 0 ? Vector{Float64}(endo[:, lag]) : fill(NaN, n)
    yT = lead > 0 ? Vector{Float64}(endo[:, lag+periods+1]) : fill(NaN, n)
    y0, yT
end

"compute_maxerror.m (the infinity norm keeps NaN, unlike max(abs()))."
function _pf_maxerror(m::Model, P::PFStacked, endo, exo, periods)
    lag = m.info["maximum_lag"]
    y0, yT = _pf_boundaries(m, endo, periods)
    res, _ = evaluate!(P, endo[:, lag+1:lag+periods], y0, yT, exo[:, lag+1:lag+periods], false)
    _pf_norm_inf(res)
end

# ------------------------------------------------------------------------------------------ solvers
"sim1.m: Newton's method on the stacked system. Returns (endo, success, err, iter)."
function _pf_sim1(m::Model, P::PFStacked, endo, exo, periods, o, log)
    lag, n = m.info["maximum_lag"], m.n_endo
    endo = copy(endo)
    y0, yT = _pf_boundaries(m, endo, periods)
    Y = endo[:, lag+1:lag+periods]
    X = exo[:, lag+1:lag+periods]
    T, tolf, maxit = periods, o["dynatol.f"], o["simul.maxit"]
    it, converged, err = 1, false, NaN
    while !converged && it <= maxit
        res, A = evaluate!(P, Y, y0, yT, X[:, 1:T])
        if o["endogenous_terminal_period"] && it > 1
            for k in 1:T
                if _pf_norm_inf(view(res, (k-1)*n+1:k*n)) < tolf / 1e7
                    if k < T
                        keep = k * n
                        res, A = res[1:keep], A[1:keep, 1:keep]
                        yT = Y[:, k+1]
                        endo[:, lag+k+1:lag+T] .= Y[:, k+1:end]
                        Y, T = Y[:, 1:k], k
                    end
                    break
                end
            end
        end
        err = _pf_norm_inf(res)
        log("Iter: $it,\t err. = $(@sprintf("%g", err))")
        if err < tolf
            converged = true
        else
            dy = _pf_lu_solve(A, res, _pf_pattern(P, periods))
            Y = Y .- reshape(dy, n, T)
        end
        it += 1
    end
    endo[:, lag+1:lag+T] .= Y
    o["endogenous_terminal_period"] && (err = _pf_maxerror(m, P, endo, exo, periods))
    success = converged && all(isfinite, Y)
    endo, success, err, it                    # Dynare's count: residual evaluations + 1
end

"sim1_linear.m: one linear solve around the (terminal) steady state."
function _pf_sim1_linear(m::Model, P::PFStacked, endo, exo, periods, ss, xss, params, o, nocheck)
    lag, n, nx = m.info["maximum_lag"], m.n_endo, m.n_exo
    y3 = repeat(ss, 3)
    d1 = dynamic_resid(m, y3, xss, params, ss)
    if !nocheck && !(maximum(abs, d1) <= DYNARE_TOLF)
        throw(PerfectForesightError("linear_approximation: the Jacobian is not evaluated at the steady state " *
                                    "(max |residual| at oo_.steady_state = $(@sprintf("%.3e", maximum(abs, d1))))"))
    end
    vals = dynamic_jac_values(m, y3, xss, params, ss)
    f = m.fns
    J = sparse(f.g1_rows, f.g1_cols, vals, n, 3n + nx)
    dy, dx = endo .- ss, exo .- xss
    y0, yT = _pf_boundaries(m, dy, periods)
    replace!(y0, NaN => 0.0); replace!(yT, NaN => 0.0)
    function residuals(Y)
        full = hcat(y0, Y, yT)
        out = zeros(n, periods)
        for t in 1:periods
            z = vcat(vec(view(full, :, t:t+2)), dx[:, lag+t])
            out[:, t] = J * z
        end
        vec(out)
    end
    Y = dy[:, lag+1:lag+periods]
    res = residuals(Y)
    K = length(P.sel)
    pat = _pf_pattern(P, periods)
    A, perm, V = pat.A, pat.perm, pat.V
    for t in 1:periods
        V[:, t] .= vals[P.sel]
    end
    nz = A.nzval
    for i in eachindex(nz)
        nz[i] = V[perm[i]]
    end
    Y = Y .- reshape(_pf_lu_solve(A, res), n, periods)
    err = _pf_norm_inf(residuals(Y))
    success = all(isfinite, Y) && isfinite(err)
    out = copy(endo)
    out[:, lag+1:lag+periods] .= Y .+ ss
    out, success, err, 1
end

"""
One period of a purely backward, purely forward or static model: dynare_solve's convention that a guess
whose residuals are already below tolf is returned unchanged, then Newton's method, then a trust-region
method.
"""
function _pf_solve_period(m::Model, guess, fixed_lag, fixed_lead, x, params, ss, o)
    n = m.n_endo
    tolf, maxit = o["dynatol.f"], o["simul.maxit"]
    stack(z) = vcat(fixed_lag, z, fixed_lead)
    f(z) = dynamic_resid(m, stack(z), x, params, ss)
    jac(z) = dynamic_jac(m, stack(z), x, params, ss)[:, n+1:2n]
    z = Vector{Float64}(guess)
    r = f(z)
    if all(isfinite, r) && maximum(abs, r) < tolf
        return z, true
    end
    for _ in 1:maxit
        all(isfinite, r) || break
        F = lu(jac(z); check=false)
        issuccess(F) || break
        z = z - F \ r
        r = f(z)
        all(isfinite, r) && maximum(abs, r) < tolf && return z, true
    end
    sol = try
        NLsolve.nlsolve((rr, zz) -> (rr .= f(zz)), (JJ, zz) -> (JJ .= jac(zz)), Vector{Float64}(guess);
                        method=:trust_region, ftol=min(tolf, 1e-10), xtol=0.0, iterations=1000).zero
    catch e
        e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException || e isa DomainError || rethrow()
        Vector{Float64}(guess)
    end
    r = f(sol)
    ok = all(isfinite, r) && maximum(abs, r) < tolf
    ok ? (sol, true) : (Vector{Float64}(guess), false)
end

"sim1_purely_backward.m, sim1_purely_forward.m, sim1_purely_static.m."
function _pf_sim1_recursive(m::Model, endo, exo, periods, params, ss, o)
    n, lag = m.n_endo, m.info["maximum_lag"]
    endo = copy(endo)
    nan = fill(NaN, n)
    backward = m.info["maximum_endo_lag"] > 0
    forward = m.info["maximum_endo_lead"] > 0
    success = true
    if backward
        for it in lag+1:lag+periods
            y = endo[:, it-1]
            z, ok = _pf_solve_period(m, y, y, nan, exo[:, it], params, ss, o)
            endo[:, it] = z
            ok || (success = false; break)
        end
    elseif forward
        for it in lag+periods:-1:lag+1
            yf = endo[:, it+1]
            z, ok = _pf_solve_period(m, yf, nan, yf, exo[:, it], params, ss, o)
            endo[:, it] = z
            ok || (success = false; break)
        end
    else
        y = endo[:, 1]
        for it in 1:periods
            y, ok = _pf_solve_period(m, y, nan, nan, exo[:, it], params, ss, o)
            success = success && ok
            endo[:, it] = y
        end
    end
    endo, success
end

# ------------------------------------------------------------------------------------------ LMMCP
"""
(lower bounds, upper bounds, equation order) from the complementarity conditions of the model
(`[mcp = 'r > 0']` tags or the `⟂` syntax), as Dynare's dynamic_complementarity_conditions and
M_.dynamic_mcp_equations_reordering. Equation `order[i]` is paired with variable `i`.
"""
function complementarity_conditions(m::Model)
    n = m.n_endo
    lb, ub = fill(-Inf, n), fill(Inf, n)
    order = collect(1:n)
    eqs = Any[eq for eq in get_or(m.raw, "model", []) if eq isa AbstractDict]
    for (i, eq) in enumerate(eqs)
        tags = get(eq, "tags", nothing)
        cc = get(eq, "complementarity_condition", nothing)
        if tags isa AbstractDict && haskey(tags, "mcp") && (cc === nothing || isempty(cc))
            throw(ModelError("an [mcp=...] tag the preprocessor did not turn into a complementarity condition"))
        end
        (cc === nothing || isempty(cc)) && continue
        (length(eqs) != n || !haskey(m.ie, cc["variable"])) &&
            throw(ModelError("complementarity condition on something else than an endogenous variable"))
        v = m.ie[cc["variable"]]
        get(cc, "lower_bound", nothing) in (nothing, "") || (lb[v] = Float64(_value(m, cc["lower_bound"])))
        get(cc, "upper_bound", nothing) in (nothing, "") || (ub[v] = Float64(_value(m, cc["upper_bound"])))
        j = findfirst(==(i), order)
        order[v], order[j] = order[j], order[v]
    end
    lb, ub, order
end

has_complementarity(m::Model) = any(eq -> eq isa AbstractDict && ((get(eq, "complementarity_condition", nothing) !== nothing &&
        !isempty(eq["complementarity_condition"])) || (get(eq, "tags", nothing) isa AbstractDict && haskey(eq["tags"], "mcp"))),
        get_or(m.raw, "model", []))

function _lmmcp_phi(x, F, lb, ub, l1, l2, idx)
    n = length(x)
    y = zeros(2n)
    for i in 1:n
        xi, Fi, li, ui = x[i], F[i], lb[i], ub[i]
        phi_u = sqrt((ui - xi)^2 + Fi^2) - ui + xi + Fi
        if idx[i] == 0
            y[i] = -l1 * Fi
            y[n+i] = -l2 * Fi
        elseif idx[i] == 1
            y[i] = l1 * (-xi + li - Fi + sqrt((xi - li)^2 + Fi^2))
            y[n+i] = l2 * max(0, xi - li) * max(0, Fi)
        elseif idx[i] == 2
            y[i] = -l1 * phi_u
            y[n+i] = l2 * max(0, ui - xi) * max(0, -Fi)
        else
            y[i] = l1 * (sqrt((xi - li)^2 + phi_u^2) - xi + li - phi_u)
            y[n+i] = l2 * (max(0, xi - li) * max(0, Fi) + max(0, ui - xi) * max(0, -Fi))
        end
    end
    y
end

"An element of the C-subdifferential of Phi (subfunction DPhi of lmmcp.m)."
function _lmmcp_dphi(x, F, DF, lb, ub, l1, l2, idx)
    n, null = length(x), 1e-8
    beta_l = [abs(x[i] - lb[i]) <= null && abs(F[i]) <= null for i in 1:n]
    beta_u = [abs(ub[i] - x[i]) <= null && abs(F[i]) <= null for i in 1:n]
    z = Float64.(beta_l .| beta_u)
    DFz = DF * z
    Da, Db, s, e = zeros(n), zeros(n), zeros(n), zeros(n)
    for i in 1:n
        xi, Fi, li, ui, zi, Di = x[i], F[i], lb[i], ub[i], z[i], DFz[i]
        alpha_l = xi - li >= -null && Fi >= -null
        alpha_u = ui - xi >= -null && Fi <= null
        denz = max(null, sqrt(zi^2 + Di^2))
        k = idx[i]
        if k == 0
            Db[i] = -1; s[i] = -1
        elseif k == 1
            den1 = max(null, sqrt((xi - li)^2 + Fi^2))
            if !beta_l[i]
                Da[i], Db[i] = (xi - li) / den1 - 1, Fi / den1 - 1
            else
                Da[i], Db[i] = zi / denz - 1, Di / denz - 1
            end
            if alpha_l; s[i] = xi - li; e[i] = Fi; end
        elseif k == 2
            den1 = max(null, sqrt((ui - xi)^2 + Fi^2))
            if !beta_u[i]
                Da[i], Db[i] = (ui - xi) / den1 - 1, -Fi / den1 - 1
            else
                Da[i], Db[i] = -zi / denz - 1, -Di / denz - 1
            end
            if alpha_u; s[i] = xi - ui; e[i] = Fi; end
        else
            phi = -ui + xi + Fi + sqrt((ui - xi)^2 + Fi^2)
            den1 = max(null, sqrt((xi - li)^2 + phi^2))
            den3 = max(null, sqrt((ui - xi)^2 + Fi^2))
            den4 = max(null, sqrt(zi^2))
            if !beta_u[i]
                ci, di = (xi - ui) / den3 + 1, Fi / den3 + 1
            else
                ci, di = 1 + zi / denz, 1 + Di / denz
            end
            if !beta_l[i]
                ai, bi = (xi - li) / den1 - 1, phi / den1 - 1
            else
                ai, bi = zi / den4 - 1, (ci * zi + di * Di) / den4 - 1
            end
            Da[i], Db[i] = ai + bi * ci, bi * di
            if alpha_l && alpha_u
                s[i], e[i] = 2xi - li - ui, 2Fi
            elseif alpha_l
                s[i], e[i] = xi - li, Fi
            elseif alpha_u
                s[i], e[i] = xi - ui, Fi
            end
        end
    end
    H1 = spdiagm(Db) * DF + spdiagm(Da)
    H2 = spdiagm(s) * DF + spdiagm(e)
    vcat(l1 * H1, l2 * H2)
end

"""
min |H d - rhs| (H is 2n x n). Small problems (n < 100, as in lmmcp.m) use the regularised least-squares
problem `[H; sqrt(mu) I] d = [rhs; 0]`; large ones the sparse least-squares solution (sparse QR, as MATLAB's
backslash does).
"""
function _lmmcp_least_squares(H::SparseMatrixCSC, rhs, mu)
    n = size(H, 2)
    if mu !== nothing
        M = vcat(Matrix(H), sqrt(mu) * Matrix(1.0I, n, n))
        all(isfinite, M) || return fill(NaN, n)
        return M \ vcat(rhs, zeros(n))
    end
    all(isfinite, nonzeros(H)) || return fill(NaN, n)
    try
        d = qr(H) \ rhs
        all(isfinite, d) && return d
    catch e
        e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException || e isa ErrorException || e isa ArgumentError || rethrow()
    end
    N = H' * H
    F = lu(N; check=false)
    if issuccess(F)
        d = F \ (H' * rhs)
        d = d + F \ (H' * (rhs - H * d))
        all(isfinite, d) && return d
    end
    fill(NaN, n)
end

"""
    lmmcp(fun, x, lb, ub; tolfun=nothing, maxiter=500, log=nothing) -> (x, F, exitflag, iterations)

Port of Dynare's `lmmcp.m` (Kanzow and Petra's semismooth least-squares method for mixed complementarity
problems, with the projected Levenberg-Marquardt preprocessor). `fun(x, jac)` returns `(F, DF)` (`DF`
sparse, or `nothing` when `jac` is false). Solves lb = x => F >= 0, lb < x < ub => F = 0, x = ub => F <= 0.
`exitflag` is 1 when the merit function 0.5 |Phi|^2 is below 0.5 tolfun^2.
"""
function lmmcp(fun, x0, lb, ub; tolfun=nothing, maxiter=500, log=nothing)
    beta, Big, delta0, deltamin, eta, kwatch, lambda1, m = 0.55, 1e10, 5.0, 1.0, 0.95, 20, 0.1, 10
    null, presteps, sigma, sigma1, sigma2, tmin = 1e-10, 20, 1e-4, 0.5, 2.0, 1e-12
    tolfun = tolfun === nothing ? sqrt(eps(Float64)) : Float64(tolfun)
    eps2, lambda2 = 0.5 * tolfun^2, 1 - lambda1
    log === nothing && (log = s -> nothing)
    x = max.(lb, min.(Vector{Float64}(x0), ub))
    n = length(x)
    idx = zeros(Int, n)
    for i in 1:n
        (lb[i] > -Big && ub[i] > Big) && (idx[i] = 1)
        (lb[i] < -Big && ub[i] < Big) && (idx[i] = 2)
        (lb[i] > -Big && ub[i] < Big) && (idx[i] = 3)
    end
    small = n < 100
    phi(x, F) = _lmmcp_phi(x, F, lb, ub, lambda1, lambda2, idx)
    dphi(x, F, DF) = _lmmcp_dphi(x, F, DF, lb, ub, lambda1, lambda2, idx)
    function psi(P)
        v = 0.5 * dot(P, P)
        isfinite(v) ? v : Inf
    end
    function direction(DPhi, Phi, k, scale)
        mu = nothing
        if small
            mu = 1e-16
            N = Matrix(DPhi' * DPhi)
            if !all(isfinite, N) || cond(N, 1) > 1e25
                mu = scale / (k + 1)
            end
        end
        _lmmcp_least_squares(DPhi, -Phi, mu)
    end
    Fx, DFx = fun(x, true)
    Phix = phi(x, Fx)
    normPhix, Psix = norm(Phix), psi(Phix)
    DPhix = dphi(x, Fx, DFx)
    DPsix = DPhix' * Phix
    start = (copy(x), Fx, DFx, Phix, Psix, DPhix, DPsix)
    k = k_main = 0
    aux = zeros(m)
    aux[1] = Psix
    MaxPsi = Psix
    best = (k, copy(x), Fx, Phix, Psix, DPhix, DPsix)
    delta = delta0
    # ---- preprocessor: projected Levenberg-Marquardt steps
    normp = 1.0
    while k < presteps && Psix > eps2 && normp > null
        k += 1
        p = direction(DPhix, Phix, k, 1e-6)
        normp = norm(p)
        isfinite(normp) || break
        lbnew = max.(min.(lb .- x, 0), -delta)
        ubnew = min.(max.(ub .- x, 0), delta)
        d = max.(lbnew, min.(p, ubnew))
        x = x .+ d
        Fx, DFx = fun(x, true)
        Phinew = phi(x, Fx)
        normnew = norm(Phinew)
        if normnew <= eta * normPhix
            delta = max(deltamin, sigma2 * delta)
        elseif normnew > 5 * eta * normPhix
            delta = max(deltamin, sigma1 * delta)
        end
        Phix, Psix, normPhix = Phinew, psi(Phinew), normnew
        DPhix = dphi(x, Fx, DFx)
        DPsix = DPhix' * Phix
        log(@sprintf("%4d %24.5e (preprocessor)", k, Psix))
    end
    if Psix < eps2
        return x, Fx, 1, k
    end
    x, Fx, DFx, Phix, Psix, DPhix, DPsix = start
    x = copy(x)
    # ---- main algorithm: Levenberg-Marquardt direction with a nonmonotone Armijo rule
    while k < maxiter && Psix > eps2
        d = direction(DPhix, Phix, k, 1e-1)
        all(isfinite, d) || break
        t = 1.0
        xnew = x .+ d
        Fnew, _ = fun(xnew, false)
        Phinew = phi(xnew, Fnew)
        Psinew = psi(Phinew)
        const_ = sigma * dot(DPsix, d)
        while !(Psinew <= MaxPsi + const_ * t) && t > tmin
            t *= beta
            xnew = x .+ t .* d
            Fnew, _ = fun(xnew, false)
            Phinew = phi(xnew, Fnew)
            Psinew = psi(Phinew)
        end
        x, Phix, Psix = xnew, Phinew, Psinew
        Fx, DFx = fun(x, true)
        DPhix = dphi(x, Fx, DFx)
        DPsix = DPhix' * Phix
        k += 1
        k_main += 1
        aux[mod(k_main, m)+1] = Psix
        MaxPsi = k_main <= 5 ? Psix : maximum(aux)
        if Psix < best[5]
            best = (k, copy(x), Fx, Phix, Psix, DPhix, DPsix)
        elseif k - best[1] > kwatch
            _, x, Fx, Phix, Psix, DPhix, DPsix = best
            x = copy(x)
            MaxPsi = Psix
        end
        log(@sprintf("%4d %24.5e %11.7g", k, Psix, t))
    end
    flag = Psix <= eps2 ? 1 : (k >= maxiter ? 0 : -1)
    x, Fx, flag, k
end

"solve_stacked_problem.m with solve_algo=10 (LMMCP), then its own treatment of the residuals of equations whose variable sits on a bound."
function _pf_solve_mcp(m::Model, P::PFStacked, endo, exo, periods, lb, ub, o, log)
    lag, n = m.info["maximum_lag"], m.n_endo
    y0, yT = _pf_boundaries(m, endo, periods)
    X = exo[:, lag+1:lag+periods]
    LB, UB = repeat(lb, periods), repeat(ub, periods)
    function fun(z, jac)
        res, A = evaluate!(P, reshape(z, n, periods), y0, yT, X, jac)
        copy(res), A === nothing ? nothing : copy(A)
    end
    z, F, flag, it = lmmcp(fun, vec(endo[:, lag+1:lag+periods]), LB, UB; tolfun=get(o, "lmmcp.tolfun", nothing),
                           maxiter=get(o, "lmmcp.maxiter", 500), log=log)
    e = eps(Float64)
    bounded = isfinite.(LB) .| isfinite.(UB)
    F = [(bounded[i] && (z[i] <= LB[i] + e || z[i] >= UB[i] - e)) ? 0.0 : F[i] for i in eachindex(F)]
    out = copy(endo)
    out[:, lag+1:lag+periods] .= reshape(z, n, periods)
    out, flag == 1 && all(isfinite, z), _pf_norm_inf(F), it
end

# ------------------------------------------------------------------------------------------ driver
"State of one perfect_foresight_solver call (the fields of oo_ and M_ it reads)."
mutable struct PFSim
    m::Model
    params::Vector{Float64}
    o::Dict{String,Any}
    endo::Matrix{Float64}
    exo::Matrix{Float64}
    ss::Vector{Float64}
    xss::Vector{Float64}
    ss0::Union{Nothing,Vector{Float64}}
    xss0::Union{Nothing,Vector{Float64}}
    verbose::Bool
    lag::Int
    lead::Int
    mcp::Any
    problems::Dict{Vector{Float64},PFStacked}
    solver::String
end

function PFSim(m::Model, params, o, endo, exo, ss, xss, ss0, xss0, verbose)
    mcp = nothing
    if o["lmmcp.status"]
        (o["linear_approximation"] || get(o, "linear", false)) &&
            throw(ModelError("lmmcp together with model(linear) or linear_approximation is not implemented"))
        mcp = complementarity_conditions(m)
    end
    PFSim(m, Vector{Float64}(params), o, endo, exo, ss, xss, ss0, xss0, verbose, m.info["maximum_lag"], m.info["maximum_lead"], mcp,
          Dict{Vector{Float64},PFStacked}(), "")
end

_pf_log(s::PFSim) = msg -> (s.verbose && println(msg); nothing)

function _pf_problem(s::PFSim, ss)
    get!(s.problems, Vector{Float64}(ss)) do
        length(s.problems) > 4 && empty!(s.problems)
        PFStacked(s.m, s.params, ss; eq_index=s.mcp === nothing ? nothing : s.mcp[3])
    end
end

"perfect_foresight_solver_core.m: returns (endo, success, maxerror, iterations)."
function _pf_core(s::PFSim, endo, exo, ss, xss, periods)
    m, o, info = s.m, s.o, s.m.info
    has_lag, has_lead = info["maximum_endo_lag"] > 0, info["maximum_endo_lead"] > 0
    linear = o["linear_approximation"]
    if (linear || o["endogenous_terminal_period"]) && !(has_lag && has_lead)
        name = linear ? "linear_approximation" : "endogenous_terminal_period"
        throw(ArgumentError("option $name is not available with purely backward, purely forward or static models"))
    end
    get(o, "linear", false) && has_lag && has_lead && (linear = true)
    P = _pf_problem(s, ss)
    if s.mcp !== nothing
        s.solver = "lmmcp"
        return _pf_solve_mcp(m, P, endo, exo, periods, s.mcp[1], s.mcp[2], o, _pf_log(s))
    end
    if !(has_lag && has_lead)
        s.solver = "period by period"
        y, success = _pf_sim1_recursive(m, endo, exo, periods, s.params, ss, o)
        return y, success, _pf_maxerror(m, P, y, exo, periods), 0
    end
    if linear
        s.solver = "linear"
        nocheck = get(get(m.options, "steady", Dict{String,Any}()), "steadystate.nocheck", false) == true
        return _pf_sim1_linear(m, P, endo, exo, periods, ss, xss, s.params, o, nocheck)
    end
    s.solver = "stacked Newton"
    _pf_sim1(m, P, endo, exo, periods, o, _pf_log(s))
end

function _pf_steady_at(s::PFSim, guess, x)
    o = s.o
    try
        tol = o["simul.steady_tolf"] === nothing ? DYNARE_TOLF : _pf_float(o["simul.steady_tolf"])
        ys, _ = _pf_steady(s.m, s.params, guess, x; tol=tol, check=!o["simul.endval_steady_nocheck"])
        return all(isfinite, ys) ? ys : nothing
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
end

"create_scenario of perfect_foresight_solver.m: (ok, endo, exo, ss, xss)."
function _pf_scenario(s::PFSim, share, shareorig, endoorig, exoorig, endobase, exobase, recompute, endo, ss, xss, periods)
    lag, lead = s.lag, s.lead
    w = share / shareorig
    exo = exoorig .* w .+ exobase .* (1 - w)
    for name in s.o["simul.homotopy_exclude_varexo"]
        haskey(s.m.ix, name) || throw(ArgumentError("homotopy_exclude_varexo must only contain exogenous variable names"))
        j = s.m.ix[name]
        exo[j, :] .= exoorig[j, :]
    end
    endo = copy(endo)
    endo[:, 1:lag] .= w .* endoorig[:, 1:lag] .+ (1 - w) .* endobase[:, 1:lag]
    ok = true
    term = lag+periods+1:lag+periods+lead
    if recompute
        saved = endo[:, lag+periods+1:end]
        for j in term
            ys = _pf_steady_at(s, endo[:, j], exo[:, j])
            ys === nothing && (ys = _pf_steady_at(s, s.ss0 === nothing ? s.ss : s.ss0, exo[:, j]))
            if ys === nothing
                endo[:, lag+periods+1:end] .= saved
                ok = false
                break
            end
            endo[:, j] = ys
        end
        ss, xss = endo[:, end], exo[:, end]
    else
        endo[:, lag+periods+1:end] .= w .* endoorig[:, lag+periods+1:end] .+ (1 - w) .* endobase[:, lag+periods+1:end]
    end
    ok, endo, exo, ss, xss
end

"homotopy_loop of perfect_foresight_solver.m."
function _pf_homotopy(s::PFSim, max_share, shareorig, endoorig, exoorig, endobase, exobase, recompute, periods)
    o, lag, lead = s.o, s.lag, s.lead
    log = _pf_log(s)
    completed, step, counter, iteration = 0.0, min(o["simul.homotopy_initial_step_size"], max_share), 0, 0
    endo, exo, ss, xss = copy(endoorig), copy(exoorig), s.ss, s.xss
    maxerror, solver_iter = NaN, 0
    sim = lag+1:lag+periods
    while step > o["simul.homotopy_min_step_size"]
        iteration += 1
        saved_endo, saved_exo = endo, exo
        new_share = completed + step
        if new_share > max_share
            new_share = max_share
            step = new_share - completed
        end
        ok, endo, exo, ss, xss = _pf_scenario(s, new_share, shareorig, endoorig, exoorig, endobase, exobase, recompute, endo, ss, xss, periods)
        success = false
        if ok
            if completed == 0 && !(iteration == 1 && new_share == shareorig)
                # guess: the terminal steady state (the base scenario for models without leads)
                endo[:, sim] .= lead > 0 ? endo[:, lag+periods+1] : endobase[:, sim]
            end
            endo, success, maxerror, solver_iter = _pf_core(s, endo, exo, ss, xss, periods)
        else
            maxerror = NaN
        end
        if o["no_homotopy"] || (iteration == 1 && success && new_share == 1)
            success && (completed = new_share)
            break
        end
        iteration == 1 && log("Entering the homotopy method iterations...")
        log("$iteration\t | $(@sprintf("%1.5f", new_share))\t | $(success ? "succeeded" : "failed")\t | $(@sprintf("%e", maxerror))")
        if success
            completed = new_share
            completed >= max_share && break
            counter += 1
            n_inc = o["simul.homotopy_step_size_increase_success_count"]
            if n_inc > 0 && counter >= n_inc
                counter = 0
                step *= 2
            end
        else
            endo, exo = saved_endo, saved_exo
            counter = 0
            step /= 2
        end
    end
    completed, endo, exo, ss, xss, iteration, maxerror, solver_iter
end

"perfect_foresight_solver.m: returns a Dict with the outcome."
function _pf_solve(s::PFSim, periods::Int)
    m, o, lag, lead = s.m, s.o, s.lag, s.lead
    _pf_check_homotopy_options(o)
    rows = lag + periods + lead
    (size(s.endo) == (m.n_endo, rows) && size(s.exo) == (m.n_exo, rows)) ||
        throw(ArgumentError("endo and exo must have $rows columns (maximum_lag + periods + maximum_lead)"))
    base_ss, base_x = s.ss0 === nothing ? (s.ss, s.xss) : (s.ss0, s.xss0)
    endobase, exobase = repeat(base_ss, 1, rows), repeat(base_x, 1, rows)
    if o["simul.endval_steady"]
        lead == 0 && throw(ArgumentError("option endval_steady cannot be used on a purely backward or static model"))
        recompute = true
    elseif s.ss0 !== nothing
        recompute = true
        tol = o["simul.steady_tolf"] === nothing ? DYNARE_TOLF : _pf_float(o["simul.steady_tolf"])
        for j in lag+periods+1:rows
            r = _pf_norm_inf(static_resid(m, s.endo[:, j], s.exo[:, j], s.params))
            if !(r <= tol)
                recompute = false
                break
            end
        end
    else
        recompute = false
    end
    endoorig, exoorig = s.endo, s.exo
    max_share = o["simul.homotopy_max_completion_share"]
    completed, endo, exo, ss, xss, iteration, maxerror, solver_iter = _pf_homotopy(s, max_share, 1.0, endoorig, exoorig, endobase, exobase, recompute, periods)
    # oo_.exo_simul is left as perfect_foresight_setup built it unless the simulation fails
    out = Dict{String,Any}("endo" => endo, "exo" => exoorig, "ss" => s.ss, "xss" => s.xss, "status" => false, "linearization" => false,
                           "marginal" => false, "share" => completed, "homotopy_iterations" => iteration, "iterations" => solver_iter,
                           "recompute" => recompute)
    marg = o["simul.homotopy_marginal_linearization_fallback"]
    if completed == 1
        out["status"] = true
        recompute && (out["ss"] = ss)
    elseif o["simul.homotopy_linearization_fallback"] && completed > 0
        out["endo"] = endobase .+ (endo .- endobase) ./ completed
        recompute && (out["ss"] = out["endo"][:, end])
        maxerror = _pf_maxerror(m, _pf_problem(s, out["ss"]), out["endo"], out["exo"], periods)
        out["status"] = true; out["linearization"] = true
    elseif marg > 0 && completed > marg
        extra_share = completed - marg
        ok, e_endo, e_exo, e_ss, e_xss = _pf_scenario(s, extra_share, 1.0, endoorig, exoorig, endobase, exobase, recompute, endo, ss, xss, periods)
        if ok
            e_endo, ok, _, _ = _pf_core(s, e_endo, e_exo, e_ss, e_xss, periods)
        end
        if !ok
            share2, e_endo, e_exo, e_ss, e_xss, _, _, _ = _pf_homotopy(s, extra_share, 1.0, endoorig, exoorig, endobase, exobase, recompute, periods)
            ok = share2 == extra_share
        end
        if ok
            out["endo"] = endo .+ (endo .- e_endo) .* ((1 - completed) / marg)
            recompute && (out["ss"] = out["endo"][:, end])
            maxerror = _pf_maxerror(m, _pf_problem(s, out["ss"]), out["endo"], out["exo"], periods)
            out["marginal"] = true
        else
            out["ss"], out["xss"], out["exo"] = ss, xss, exo
        end
        out["status"] = ok
    else
        out["ss"], out["xss"], out["exo"] = ss, xss, exo
    end
    out["maxerror"] = maxerror
    out
end

# ------------------------------------------------------------------------------------------ expectation errors
"""
perfect_foresight_with_expectation_errors_setup.m: `shocks_info[exo, real time, information time]` and
`terminal_info[exo, information time]`; `datafile` is the CSV file of the datafile option.
"""
function _pf_pfwee_information(m::Model, ws::PFWorkspace, periods, datafile)
    nx = m.n_exo
    terminal = fill(NaN, nx, periods)
    shocks = fill(NaN, nx, periods, periods)
    if datafile !== nothing
        path = isabspath(datafile) ? datafile : joinpath(m.mod_dir, datafile)
        lines = [strip(l) for l in readlines(path) if !isempty(strip(l))]
        names = [strip(s) for s in split(lines[1], ",")][2:end]
        raw = Matrix{Float64}(undef, length(lines) - 1, length(names))
        for (r, ln) in enumerate(lines[2:end])
            for (c, f) in enumerate(split(ln, ",")[2:end])
                raw[r, c] = isempty(strip(f)) ? NaN : parse(Float64, f)
            end
        end
        size(raw, 1) - 2 == periods || throw(ArgumentError("the number of rows in $datafile does not match the periods setting"))
        for (i, name) in enumerate(names)
            haskey(m.ix, name) || throw(ArgumentError("$datafile: $name is not an exogenous variable"))
            j, p = m.ix[name], Int(raw[1, i])
            shocks[j, p:periods, p] = raw[p+1:end-1, i]
            terminal[j, p] = raw[end, i]
        end
        return shocks, terminal
    end
    terminal[:, 1] = ws.xss
    for p in 1:periods
        shocks[:, p, 1] = ws.xss
    end
    for (j, p1, p2, val, typ) in ws.det_shocks
        if typ == "multiply_steady_state"
            val = ws.xss[j] * val
        elseif typ == "multiply_initial_steady_state"
            ws.xss0 === nothing && throw(PerfectForesightError("option relative_to_initval of mshocks cannot be used without an endval block"))
            val = ws.xss0[j] * val
        end
        (p1 < 1 || p2 > periods) && throw(PerfectForesightError("a shock is dated outside the simulation"))
        shocks[j, p1:p2, 1] .= val
    end
    for p in 1:periods
        if p > 1
            terminal[:, p] = terminal[:, p-1]
            shocks[:, :, p] = shocks[:, :, p-1]
        end
        for d in ws.learnt_endval
            d["learnt_in"] == p || continue
            p > 1 || throw(PerfectForesightError("endval(learnt_in=1) must be a plain endval block"))
            j, v, typ = d["exo"], d["value"], d["type"]
            if typ == "level"
                terminal[j, p] = v
            elseif typ == "add"
                terminal[j, p] = terminal[j, p-1] + v
            elseif typ == "multiply"
                terminal[j, p] = terminal[j, p-1] * v
            else
                throw(ArgumentError("unknown type $typ in endval(learnt_in=...)"))
            end
            shocks[j, p:periods, p] .= terminal[j, p]
        end
        for d in ws.learnt_shocks
            d["learnt_in"] == p || continue
            j, v = d["exo"], d["value"]
            (d["p1"] < 1 || d["p2"] > periods) && throw(PerfectForesightError("a learnt shock is dated outside the simulation"))
            sl = d["p1"]:d["p2"]
            typ = d["type"]
            if typ == "level"
                shocks[j, sl, p] .= v
            elseif typ == "add"
                p > 1 || throw(PerfectForesightError("shocks(learnt_in=1) with add must be a plain shocks block"))
                shocks[j, sl, p] = shocks[j, sl, p-1] .+ v
            elseif typ == "multiply"
                p > 1 || throw(PerfectForesightError("shocks(learnt_in=1) with multiply must be a plain shocks block"))
                shocks[j, sl, p] = shocks[j, sl, p-1] .* v
            elseif typ == "multiply_steady_state"
                shocks[j, sl, p] .= terminal[j, p] * v
            else
                throw(ArgumentError("unknown type $typ in shocks(learnt_in=...)"))
            end
        end
    end
    shocks, terminal
end

"perfect_foresight_with_expectation_errors_solver.m."
function _pf_solve_pfwee(m::Model, ws::PFWorkspace, periods, o, verbose, datafile)
    lag, lead = m.info["maximum_lag"], m.info["maximum_lead"]
    shocks, terminal = _pf_pfwee_information(m, ws, periods, datafile)
    saved = ws.det_shocks
    ws.det_shocks = []                       # the setup command builds the paths without the shocks
    endo_simul, exo_simul = _pf_make_paths(m, ws, periods)
    ws.det_shocks = saved
    o = copy(o)
    lead > 0 && (o["simul.endval_steady"] = true)
    ss, xss = copy(ws.ss), copy(ws.xss)
    ss0, xss0 = ws.ss0 === nothing ? (copy(ss), copy(xss)) : (ws.ss0, ws.xss0)
    const_len = o["pfwee.constant_simulation_length"]
    info_period, increment, share0 = 1, 0, nothing
    out = nothing
    solves = Tuple{Int,Int,Float64}[]
    while info_period <= periods
        p = info_period - 1
        xss = terminal[:, info_period]
        if const_len && increment > 0
            endo_simul = hcat(endo_simul, repeat(ss, 1, increment))
            exo_simul = hcat(exo_simul, fill(NaN, m.n_exo, increment))
        end
        endo = endo_simul[:, p+1:end]
        exo = exo_simul[:, p+1:end]
        sim_length = const_len ? periods : periods - p
        known = periods - p
        exo[:, lag+1:lag+known] = shocks[:, info_period:periods, info_period]
        exo[:, lag+known+1:end] .= xss
        sim = PFSim(m, ws.params, o, endo, exo, ss, xss, ss0, xss0, verbose)
        out = _pf_solve(sim, sim_length)
        out["solver"] = sim.solver
        push!(solves, (info_period, Int(out["iterations"]), Float64(out["maxerror"])))
        if !out["status"]
            out["endo_full"], out["exo_full"] = endo_simul, exo_simul
            throw(PerfectForesightError("perfect_foresight_with_expectation_errors_solver: failed to compute the solution " *
                                        "for information available at period $info_period", out))
        end
        out["marginal"] && throw(ModelError("marginal linearization together with expectation errors is not implemented"))
        if share0 === nothing
            share0 = out["share"]
            o["simul.homotopy_max_completion_share"] = share0
        elseif out["share"] != share0
            throw(PerfectForesightError("could not find a solution for information available at period $info_period " *
                                        "with the same homotopy completion share as period 1"))
        end
        ss = out["ss"]
        endo_simul[:, p+1:end] = out["endo"]
        exo_simul[:, p+1:end] = out["exo"]
        increment = 1
        while info_period + increment <= periods && terminal[:, info_period] == terminal[:, info_period+increment] &&
              isequal(shocks[:, info_period+increment:periods, info_period], shocks[:, info_period+increment:periods, info_period+increment])
            increment += 1
        end
        info_period += increment
    end
    out = copy(out)
    merge!(out, Dict{String,Any}("endo" => endo_simul, "exo" => exo_simul, "ss" => ss, "xss" => xss, "ss0" => ss0, "xss0" => xss0,
                                 "solves" => solves, "shocks_info" => shocks, "terminal_info" => terminal))
    out
end

# ------------------------------------------------------------------------------------------ entry points
"`model(linear)` is not in the preprocessor's JSON: read it from the .mod file."
function _pf_model_is_linear(m::Model)
    get!(m.cache, :pf_linear) do
        file = joinpath(m.mod_dir, m.name * ".mod")
        isfile(file) || return false
        text = replace(read(file, String), r"/\*.*?\*/"s => " ")
        text = replace(text, r"(//|%)[^\n]*" => " ")
        any(mo -> occursin(r"(?<![\w=])linear(?![\w=])", mo.captures[1]), eachmatch(r"(?<![\w.])model(?:_options)?\s*\(([^)]*)\)", text))
    end::Bool
end

"""
    perfect_foresight(m; periods, params, exo, y_init, y_term, guess, steady, expectation_errors, learnt_shocks,
                      learnt_endval, raise_on_failure=true, verbose=false, lmmcp_tolfun, lmmcp_maxiter, options...)

Deterministic simulation: Dynare's `perfect_foresight_setup` + `perfect_foresight_solver` (or the
`perfect_foresight_with_expectation_errors` pair). Everything is read from the .mod file and can be
overridden:

* `periods`, `params`; `exo` (full path, `rows x n_exo` like oo_.exo_simul, `rows = maximum_lag + periods +
  maximum_lead`), `y_init` / `y_term` (initial and terminal conditions, n_endo x lag / n_endo x lead),
  `guess` (n_endo x periods);
* `tolf` (or `tol`; default: the file's `tolf`, else Dynare's 1e-5), `maxit`, `tolx`, `no_homotopy`,
  `homotopy_initial_step_size`, `homotopy_min_step_size`, `homotopy_step_size_increase_success_count`,
  `homotopy_max_completion_share`, `homotopy_linearization_fallback`, `homotopy_marginal_linearization_fallback`,
  `homotopy_exclude_varexo`, `linear_approximation`, `endogenous_terminal_period`, `lmmcp`, `endval_steady`,
  `endval_steady_nocheck`, `steady_tolf`, `constant_simulation_length`, `stack_solve_algo`.

`steady`: `nothing` follows the .mod file when it has a `perfect_foresight_setup` command (a `steady` command
after `initval`/`endval` makes the path start from/end at a steady state, otherwise the values of the block
are used as they are) and computes the steady states when the file has no such command; `true` always
computes them; `false` never does.

`learnt_shocks` (dicts with `learnt_in`, `var`, `periods`, `value` and optionally `type`: level, add, multiply,
multiply_steady_state) and `learnt_endval` (`learnt_in`, `var`, `value`, `type`) add news from Julia, like
`shocks(learnt_in=...)` / `endval(learnt_in=...)`; they turn on expectation errors.

With complementarity conditions in the model and the `lmmcp` option the stacked problem is solved with the
LMMCP algorithm (`lmmcp_tolfun`: tolerance on the 2-norm of its merit vector, default sqrt(eps) as in Dynare).

Throws `PerfectForesightError` when no solution is found (`raise_on_failure=false` returns the result with
`status` false instead, as Dynare does).
"""
function perfect_foresight(m::Model; periods=nothing, params=nothing, exo=nothing, y_init=nothing, y_term=nothing, guess=nothing,
                           steady=nothing, expectation_errors=nothing, learnt_shocks=(), learnt_endval=(), raise_on_failure::Bool=true,
                           verbose::Bool=false, lmmcp_tolfun=nothing, lmmcp_maxiter=nothing, tol=nothing, kw...)
    options = Dict{Symbol,Any}(kw)
    tol !== nothing && !haskey(options, :tolf) && (options[:tolf] = tol)
    periods !== nothing && (options[:periods] = periods)
    ws = _pf_replay(m, params, steady, learnt_shocks, learnt_endval)
    pfwee = expectation_errors === nothing ? ws.pfwee : expectation_errors === true
    expectation_errors === nothing && (!isempty(learnt_shocks) || !isempty(learnt_endval)) && (pfwee = true)
    if !pfwee && (!isempty(ws.learnt_shocks) || !isempty(ws.learnt_endval)) && ws.has_setup
        warn_once(m, "shocks(learnt_in=...) / endval(learnt_in=...) blocks are only used by the perfect_foresight_with_expectation_errors " *
                     "commands; they are ignored here, as in Dynare")
    end
    o = _pf_resolve_options(m, pfwee, options)
    if o["periods"] === nothing
        o["periods"] = get(_pf_statement_options(m, "perfect_foresight_setup"), "periods", nothing)
        o["periods"] === nothing && (o["periods"] = get(_pf_statement_options(m, "perfect_foresight_with_expectation_errors_setup"), "periods", nothing))
    end
    o["periods"] === nothing && throw(ArgumentError("periods not given and not set in the .mod file"))
    T = Int(_pf_float(o["periods"]))
    o["linear"] = _pf_model_is_linear(m)
    if lmmcp_tolfun !== nothing
        o["lmmcp.tolfun"] = Float64(lmmcp_tolfun)
    else
        try                                       # options_.lmmcp.TolFun = ... written in the .mod file
            o["lmmcp.tolfun"] = Float64(scalar(m.matlab_options["lmmcp"]["TolFun"], "options_.lmmcp.TolFun"))
        catch
        end
    end
    lmmcp_maxiter !== nothing && (o["lmmcp.maxiter"] = Int(lmmcp_maxiter))
    if has_complementarity(m) && !o["lmmcp.status"]
        warn_once(m, "the model has complementarity conditions but perfect_foresight_solver is used without the lmmcp option: " *
                     "as in Dynare the conditions are ignored and the unconstrained path is computed (pass lmmcp=true to impose them)")
    end
    lag, lead = m.info["maximum_lag"], m.info["maximum_lead"]
    o["datafile"] !== nothing && !pfwee && throw(ModelError("perfect_foresight_setup(datafile=...) (an initval_file) is not implemented"))
    local out, ss0, xss0
    if pfwee
        (exo !== nothing || y_init !== nothing || y_term !== nothing || guess !== nothing) &&
            throw(ArgumentError("exo, y_init, y_term and guess cannot be combined with expectation errors"))
        out = _pf_solve_pfwee(m, ws, T, o, verbose, o["datafile"])
        ss0, xss0 = out["ss0"], out["xss0"]
    else
        endo, x = _pf_make_paths(m, ws, T)
        rows = lag + T + lead
        if exo !== nothing
            size(exo) == (rows, m.n_exo) || throw(ArgumentError("exo must have size ($rows, $(m.n_exo))"))
            x = Matrix{Float64}(permutedims(exo))
        end
        if y_init !== nothing
            lag > 0 || throw(ArgumentError("the model has no lag: there is no initial condition"))
            endo[:, 1:lag] .= y_init
        end
        if y_term !== nothing
            lead > 0 || throw(ArgumentError("the model has no lead: there is no terminal condition"))
            endo[:, lag+T+1:end] .= y_term
        end
        if guess !== nothing
            size(guess) == (m.n_endo, T) || throw(ArgumentError("guess must have size ($(m.n_endo), $T)"))
            endo[:, lag+1:lag+T] .= guess
        end
        sim = PFSim(m, ws.params, o, endo, x, ws.ss, ws.xss, ws.ss0, ws.xss0, verbose)
        out = _pf_solve(sim, T)
        out["solver"] = sim.solver
        ss0, xss0 = ws.ss0, ws.xss0
    end
    info = Dict{String,Any}(k => out[k] for k in ("solves", "shocks_info", "terminal_info", "recompute") if haskey(out, k))
    info["tolf"] = o["dynatol.f"]
    res = PerfectForesightResult(out["endo"], Matrix{Float64}(permutedims(out["exo"])), Int(something(out["iterations"], 0)),
                                 Float64(out["maxerror"]), out["status"] === true, copy(m.endo_names), copy(m.exo_names), T, lag, lead,
                                 out["ss"], ss0, out["xss"], xss0, ws.params, Float64(out["share"]), Int(out["homotopy_iterations"]),
                                 out["linearization"] === true, out["marginal"] === true, get(out, "solver", ""), info)
    if !res.status && raise_on_failure
        throw(PerfectForesightError("failed to solve the perfect foresight model (max residual $(@sprintf("%.2e", res.max_residual)) after " *
              "$(res.iterations) iterations of the last attempt, homotopy completed $(@sprintf("%.3g", res.homotopy_completion_share)) " *
              "of the shocks in $(res.homotopy_iterations) iterations)", res))
    end
    res
end

"""
    unanticipated_shocks(m, surprises; periods=nothing, kw...)

A sequence of unanticipated shocks: `surprises` is a list of `(period, Dict(exo => value))`; each shock hits
in `period` and is learnt in that same period (agents expect the exogenous variables to return to their
steady state afterwards). Shorthand for `perfect_foresight(m; learnt_shocks=[...])`, i.e. Dynare's
`shocks(learnt_in=p)` blocks with `perfect_foresight_with_expectation_errors_solver`.
"""
function unanticipated_shocks(m::Model, surprises; periods=nothing, kw...)
    learnt = [Dict{String,Any}("learnt_in" => Int(p), "var" => String(name), "periods" => Int(p), "value" => Float64(v), "type" => "level")
              for (p, shocks) in surprises for (name, v) in shocks]
    perfect_foresight(m; periods=periods, learnt_shocks=learnt, expectation_errors=true, kw...)
end
