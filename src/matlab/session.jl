# The model seen from MATLAB (M_, options_, oo_), the .mod-file session that runs MATLAB lines in
# order with the Dynare statements, and MATLAB steady-state files (port of pydyn/matlab.py).

# ====================================================================== hosts
mutable struct HostM <: HostStruct
    h::Any
    cells::Dict{String,Cell}
    results::Any                   # dynare(): the workspace's M_ (an MStruct); the fields below are read from it
end
HostM(h, cells) = HostM(h, cells, nothing)
mutable struct HostOptions <: HostStruct
    store::MStruct                 # model.matlab_options: what the .mod file assigned
    unknown::Set{String}           # fields a Dynare command has overwritten since
    defaults_valid::Bool
    assigned::Set{String}          # fields the MATLAB lines assigned since the workspace last took them over (dynare())
end
HostOptions(store, unknown, valid) = HostOptions(store, unknown, valid, Set{String}())
mutable struct HostOO <: HostStruct
    session::Any
    results::Any                   # dynare(): the workspace's oo_ (an MStruct), which MATLAB lines read and write
end
HostOO(session) = HostOO(session, nothing)

"What MATLAB code can see of a model. A live host writes through to the model."
mutable struct Host
    model::Any
    params::Vector{Float64}
    sigma::Matrix{Float64}
    sigma_tainted::Bool
    params_row::Bool
    readonly::Bool                  # true for the M_ a steady-state file or helper sees through "global"
    tainted::Set{Int}               # parameters whose value in Dynare is unknown to JDyn
    taint_why::String
    nwrites::Int
    M::HostM
    options::HostOptions
    oo::HostOO
end
function Host(model; params=nothing, live=false)
    p = live ? model.params : Vector{Float64}(params === nothing ? model.params : params)
    s = live ? model.Sigma_e : copy(model.Sigma_e)
    h = Host(model, p, s, false, false, false, Set{Int}(), "", 0, HostM(nothing, Dict{String,Cell}()),
             HostOptions(model.matlab_options, Set{String}(), true), HostOO(nothing))
    h.M.h = h
    h
end

host_M(h::Host) = h.M
host_options(h::Host) = h.options
host_oo(h::Host) = h.oo
host_is_param(h::Host, name) = haskey(h.model.ip, name)

function _pidx(h::Host, name, what)
    i = get(h.model.ip, name, nothing)
    i === nothing && merr("$what: parameter name $name doesn't exist")
    i
end
function host_set_param(h::Host, name, value)
    i = _pidx(h, name, "set_param_value")
    h.readonly && _unsup("a steady-state file or helper function calls set_param_value")
    h.params[i] = scalar(value, "parameter value")
    delete!(h.tainted, i)
    h.nwrites += 1
end
function host_get_param(h::Host, name)
    i = _pidx(h, name, "get_param_by_name")
    i in h.tainted && throw(Unsupported("the value of parameter $name is unknown ($(h.taint_why))", "tainted"))
    h.params[i]
end
function host_set_shock_stderr(h::Host, name, value)
    i = get(h.model.ix, name, nothing)
    i === nothing && merr("Shock name $name doesn't exist")
    h.readonly && _unsup("a steady-state file or helper function calls set_shock_stderr_value")
    h.sigma[i, i] = scalar(value)^2
    h.nwrites += 1
end

# ---- M_
const _M_SCALARS = ["param_nbr", "endo_nbr", "exo_nbr", "orig_endo_nbr", "exo_det_nbr", "maximum_lag", "maximum_lead",
                    "maximum_endo_lag", "maximum_endo_lead"]
const _M_NAMES = ["param_names", "endo_names", "exo_names"]
const _M_HISTVAL = ("endo_histval", "exo_histval")
field_names(M::HostM) = vcat(["params", "Sigma_e", "fname", "dname"], _M_SCALARS, _M_NAMES)
function snapshot(M::HostM)
    s = MStruct()
    s._kind = "M_"
    for f in field_names(M)
        s[f] = get_field(M, f)
    end
    s
end
function restore!(M::HostM, value)
    (value isa MStruct && value._kind == "M_") || _unsup("replacing M_ by something that is not a copy of M_")
    cur = Dict(f => peek_field(M, f; raw=true) for f in field_names(M) if !(f in ("params", "Sigma_e")))
    for f in keys(value)
        f in ("params", "Sigma_e") && continue
        v = value[f]
        ok = haskey(cur, f) && (v === cur[f] || (is_num(v) && is_num(cur[f]) && arr(v) == arr(cur[f])) ||
                                (v isa AbstractString && v == cur[f]) || (v isa Cell && cur[f] isa Cell && v.items == cur[f].items))
        ok || _unsup("M_ is replaced by a struct whose field '$f' was changed")
    end
    isempty(setdiff(Set(keys(cur)), Set(keys(value)))) || _unsup("M_ is replaced by a struct with fewer fields")
    set_field!(M, "params", value["params"])
    set_field!(M, "Sigma_e", value["Sigma_e"])
end
has_field(M::HostM, name) = name in field_names(M) || name in ("H", "Correlation_matrix_ME", "Correlation_matrix") ? true : (name in _M_HISTVAL && M.results !== nothing) ? true : _semi_has(M.h.model, name) ? true : throw(Unsupported("JDyn does not know whether M_ has a field '$name'", "results"))
get_field(M::HostM, name) = peek_field(M, name)
function peek_field(M::HostM, name; raw=false)
    h = M.h; m = h.model
    if name == "params"
        if !isempty(h.tainted) && !raw
            bad = [m.param_names[i] for i in sort(collect(h.tainted))][1:min(end, 5)]
            throw(Unsupported("M_.params: the value of $bad is unknown ($(h.taint_why))", "tainted"))
        end
        length(h.params) == 1 && return h.params[1]
        return h.params_row ? reshape(copy(h.params), 1, :) : reshape(copy(h.params), :, 1)
    end
    if name == "Sigma_e"
        (h.sigma_tainted && !raw) && throw(Unsupported("M_.Sigma_e is unknown ($(h.taint_why))", "tainted"))
        return ret(copy(h.sigma))
    end
    name == "param_nbr" && return Float64(length(m.param_names))
    name == "endo_nbr" && return Float64(m.n_endo)
    name == "exo_nbr" && return Float64(m.n_exo)
    name == "orig_endo_nbr" && return Float64(orig_endo_nbr(m))
    name == "exo_det_nbr" && return Float64(n_exo_det(m))
    name in ("maximum_lag", "maximum_lead", "maximum_endo_lag", "maximum_endo_lead") && return Float64(m.info[name])
    if name in ("endo_names_tex", "exo_names_tex", "param_names_tex", "endo_names_long", "exo_names_long", "param_names_long")
        return get!(M.cells, name) do
            sec = startswith(name, "endo") ? "endogenous" : startswith(name, "exo") ? "exogenous" : "parameters"
            key = endswith(name, "tex") ? "texName" : "longName"
            Cell(Any[String(get(d, key, d["name"])) for d in m.raw[sec]], (length(m.raw[sec]), 1))
        end
    end
    if name in _M_NAMES
        return get!(M.cells, name) do
            names = name == "param_names" ? m.param_names : name == "endo_names" ? m.endo_names : m.exo_names
            Cell(Any[names...], (length(names), 1))
        end
    end
    name in ("fname", "dname") && return m.name
    if name in _M_HISTVAL && M.results !== nothing         # dynare(): M_.endo_histval / exo_histval live in the workspace's M_
        return haskey(M.results, name) ? deepcopy(M.results[name]) : zeros(0, 0)
    end
    _semi_has(m, name) && return semi_M(m)[name]          # M_.var, M_.pac, ... (src/pac.jl)
    if name in ("H", "Correlation_matrix_ME", "Correlation_matrix")      # read-only views of the covariance structure
        if M.results !== nothing && haskey(M.results, name)
            return deepcopy(M.results[name])
        end
        name == "H" && return me_H(m, m.varobs)
        name == "Correlation_matrix_ME" && return me_corr_matrix(m, m.varobs)
        return copy(declared_correlation_matrix(m))
    end
    throw(Unsupported("M_.$name is not available in JDyn", "results"))
end
function set_field!(M::HostM, name, value)
    h = M.h
    h.readonly && _unsup("a steady-state file or helper function writes to the global M_")
    if name == "params"
        v = farr(value)
        n = length(h.params)
        if size(v) == (1, n) && n > 1
            h.params_row = true
        elseif size(v) == (n, 1) || (length(v) == 1 && n == 1)
            h.params_row = false
        else
            merr("M_.params must remain a vector of $n elements (assigned $(size(v,1))x$(size(v,2)))")
        end
        h.params .= vec(v)
        empty!(h.tainted)
        h.nwrites += 1
    elseif name == "Sigma_e"
        v = farr(value)
        n = size(h.sigma, 1)
        size(v) == (n, n) || merr("M_.Sigma_e must remain a $(n)x$(n) matrix")
        h.sigma .= v
        h.sigma_tainted = false
        h.nwrites += 1
    elseif name in _M_HISTVAL && M.results !== nothing
        M.results[name] = deepcopy(value)
    else
        _unsup("assignment to M_.$name is not supported")
    end
end
can_assign_sub(::HostM) = true
function assign_sub!(M::HostM, name, args, value)
    h = M.h
    h.readonly && _unsup("a steady-state file or helper function writes to the global M_")
    if name == "params"
        n = length(h.params)
        col = h.params_row ? reshape(copy(h.params), 1, :) : reshape(copy(h.params), :, 1)
        new = arr(assign_index(col, args, value))
        size(new) != size(col) && merr("M_.params must not change size")
        length(args) in (1, 2) || merr("bad index into M_.params")
        k = _lin_index(args[1], n)
        h.params .= vec(new)
        setdiff!(h.tainted, k)
        h.nwrites += 1
    elseif name == "Sigma_e"
        new = farr(assign_index(h.sigma, args, value))
        size(new) != size(h.sigma) && merr("M_.Sigma_e must not change size")
        h.sigma .= new
        h.nwrites += 1
    elseif name in _M_HISTVAL && M.results !== nothing
        cur = haskey(M.results, name) ? M.results[name] : zeros(0, 0)
        M.results[name] = assign_index(cur, args, value)
    else
        _unsup("assignment to M_.$name is not supported")
    end
end

# ---- options_
"Dynare 7.2 defaults (matlab/default_option_values.m) valid until a Dynare command runs."
const OPTION_DEFAULTS = Dict{String,Any}("order" => 2.0, "irf" => 40.0, "periods" => 0.0, "drop" => 100.0, "replic" => 50.0,
    "ar" => 5.0, "solve_algo" => 4.0, "solve_tolf" => eps(Float64)^(1 / 3), "solve_tolx" => eps(Float64)^(2 / 3),
    "hp_filter" => 0.0, "pruning" => false, "loglinear" => false, "linear" => false, "nocorr" => false,
    "nomoments" => false, "k_order_solver" => false, "logged_steady_state" => 0.0)
field_names(O::HostOptions) = copy(O.store._keys)
function snapshot(O::HostOptions)
    s = MStruct()
    for k in O.store._keys
        k in O.unknown || (s[k] = O.store[k])
    end
    s._kind = "options_"
    s
end
function restore!(O::HostOptions, value)
    (value isa MStruct && value._kind == "options_") || _unsup("replacing options_ by something that is not a copy of options_")
    empty!(O.store._keys); empty!(O.store._vals)
    for k in keys(value); O.store[k] = value[k]; end
    empty!(O.unknown)
end
function has_field(O::HostOptions, name)
    (haskey(O.store, name) || (O.defaults_valid && haskey(OPTION_DEFAULTS, name))) && return true
    throw(Unsupported("JDyn does not know Dynare's default for options_.$name", "results"))
end
"A Dynare command was given these options: the fields of options_ it sets are no longer known."
forget!(O::HostOptions, options) = (for k in keys(options); push!(O.unknown, split(string(k), ".")[1]); end)
function get_field(O::HostOptions, name)
    name in O.unknown && throw(Unsupported("options_.$name was changed by a Dynare command; its value is not known to JDyn", "results"))
    haskey(O.store, name) && return O.store[name]
    if O.defaults_valid && haskey(OPTION_DEFAULTS, name)
        v = OPTION_DEFAULTS[name]
        return v isa Bool ? v : Float64(v)
    end
    throw(Unsupported("options_.$name has not been set in the .mod file (Dynare's defaults are not known to JDyn)", "results"))
end
function peek_field(O::HostOptions, name; raw=false)
    name in O.unknown && throw(Unsupported("options_.$name was changed by a Dynare command; its value is not known to JDyn", "results"))
    get(O.store, name, nothing)
end
set_field!(O::HostOptions, name, value) = (O.store[name] = value; delete!(O.unknown, name); push!(O.assigned, name))

# ---- oo_
# While loading (JDyn.load) results do not exist; under dynare() commands run in place and oo_ is the
# workspace's struct, read and written by MATLAB lines with value semantics.
_oo_results(O::HostOO) = O.results
_no_oo(what) = throw(Unsupported(what, "results"))
snapshot(O::HostOO) = O.results === nothing ? _no_oo("oo_ (results) is not available while the model is loaded") : deepcopy(O.results)
function restore!(O::HostOO, value)
    O.results === nothing && _no_oo("assignment to oo_")
    value isa MStruct || merr("oo_ must be a struct")
    r = O.results
    empty!(r._keys); empty!(r._vals)
    for k in keys(value)
        r[k] = deepcopy(value[k])
    end
end
has_field(O::HostOO, name) = O.results === nothing ? _no_oo("oo_ (results) is not available while the model is loaded") : haskey(O.results, name)
field_names(O::HostOO) = O.results === nothing ? _no_oo("oo_ (results) is not available while the model is loaded") : copy(keys(O.results))
function get_field(O::HostOO, name)
    if O.results !== nothing && haskey(O.results, name)
        return deepcopy(O.results[name])
    end
    (O.session !== nothing && name in ("steady_state", "exo_steady_state")) && return oo_vector(O.session, name)
    (O.results === nothing && O.session !== nothing && _semi_oo_has(O.session.model, name)) && return O.session.model.cache[:semi_oo][name]
    O.results !== nothing && merr("Reference to non-existent field '$name'.")
    throw(Unsupported("oo_.$name: results are not available while the model is loaded", "results"))
end
function peek_field(O::HostOO, name; raw=false)
    O.results === nothing && _no_oo("oo_.$name: results are not available while the model is loaded")
    haskey(O.results, name) ? O.results[name] : nothing
end
set_field!(O::HostOO, name, value) = O.results === nothing ? _no_oo("assignment to oo_.$name") : (O.results[name] = value)

# ====================================================================== .mod-file session
struct EscapeSig <: Exception
    u::Unsupported
end
struct TryUnknown <: Exception
    u::Unsupported
end

mutable struct Info
    writes::Set{String}
    calib::Union{Nothing,String}
    dyn::Vector{Any}
    all::Bool
    paths::Bool
    has_return::Bool
    has_break::Bool
    solves::Bool
    count::Int
end
Info() = Info(Set{String}(), nothing, Any[], false, false, false, false, false, 0)

"Dynare statements that only declare or store something (they never call the steady-state file)."
const DECL_KINDS = Set(["param_init", "initval", "endval", "histval", "shocks", "mshocks", "heteroskedastic_shocks", "varobs",
    "estimated_params", "estimated_params_init", "estimated_params_bounds", "estimated_params_remove",
    "occbin_constraints", "observation_trends", "native", "verbatim", "write_latex_dynamic_model",
    "write_latex_static_model", "write_latex_original_model", "write_latex_steady_state_model",
    "collect_latex_files", "write_latex_definitions", "set_time", "estimation_data", "planner_objective",
    "osr_params", "optim_weights", "matched_moments", "matched_irfs", "matched_irfs_weights", "var_model",
    "pac_model", "trend_component_model", "var_expectation_model", "perfect_foresight_setup",
    "perfect_foresight_solver", "initval_file", "histval_file", "homotopy", "occbin_setup",
    "load_params_and_steady_state", "save_params_and_steady_state", "options", "model_info", "conditional_forecast_paths"])
"Dynare statements that compute or display: inside a block whose condition is unknown they are skipped."
const COMPUTE_KINDS = Set(["steady", "check", "stoch_simul", "estimation", "osr", "forecast", "calib_smoother", "identification",
    "model_diagnostics", "model_info", "resid", "rplot", "perfect_foresight_setup",
    "perfect_foresight_solver", "shock_decomposition", "realtime_shock_decomposition",
    "plot_shock_decomposition", "initial_condition_decomposition", "conditional_forecast",
    "plot_conditional_forecast", "sensitivity", "dynare_sensitivity", "extended_path", "simul",
    "write_latex_dynamic_model", "write_latex_static_model", "write_latex_original_model",
    "write_latex_steady_state_model", "collect_latex_files", "write_latex_definitions", "occbin_solver",
    "occbin_graph", "occbin_write_regimes_xls", "method_of_moments", "ramsey_policy", "discretionary_policy",
    "evaluate_planner_objective", "save_params_and_steady_state", "smoother2histval", "smoother_2_histval", "dynasave",
    "dynatype", "bvar_density", "bvar_forecast", "bvar_irf", "ms_sbvar_estimation", "ms_sbvar_simulation",
    "ms_sbvar_compute_mdd", "ms_sbvar_compute_probabilities", "ms_sbvar_irf", "ms_sbvar_forecast",
    "ms_sbvar_variance_decomposition", "sbvar", "prior_posterior_function", "generate_irfs",
    "perfect_foresight_with_expectation_errors_setup", "perfect_foresight_with_expectation_errors_solver",
    "heterogeneity_solve", "heterogeneity_simulate", "heterogeneity_compute_steady_state"])
const _OPTION_FREE_KINDS = Set(["param_init", "initval", "endval", "histval", "shocks", "mshocks", "native", "verbatim",
    "estimated_params", "estimated_params_init", "estimated_params_bounds", "varobs"])
const _EVALUATED_KINDS = Set(["param_init", "initval", "endval", "shocks", "load_params_and_steady_state"])
const _ESTIMATING = Set(["estimation", "method_of_moments", "osr", "dynare_sensitivity", "sensitivity", "ms_sbvar_estimation"])
const _OO_INPUTS = ("steady_state", "exo_steady_state", "exo_det_steady_state")

"Runs the statements of a .mod file: Dynare statements through the model, native MATLAB lines through
the interpreter, in source order and under the same control flow."
mutable struct ModSession
    model::Any
    host::Host
    I::Interp
    sc::Scope
    skipped::Vector{Tuple{String,String,String,Int}}
    skipped_keys::Set{Tuple{String,String}}
    path_edits::Bool
    est_params::Set{String}
    current::Any
    in_try::Int
    ss_params_::Union{Nothing,Set{Int}}
    any_taint::Bool
    ss_warned::Bool
    oo_numeric::Bool
    oo_written::Set{String}
    pending::Any                    # (params, oo) of a 'steady' whose result has not been needed yet
    oo_failed::Union{Nothing,String}
    statements::Vector{Any}
    on_command::Any                 # dynare(): called after every Dynare statement, to run commands in place
end

function ModSession(model)
    host = Host(model; live=true)
    I = Interp(model.mod_dir, host)
    I.jlfuncs = Dict{String,Any}(k => v for (k, v) in model.functions if !haskey(model.matlab_functions, k))
    s = ModSession(model, host, I, I.base, Tuple{String,String,String,Int}[], Set{Tuple{String,String}}(), false,
                   Set{String}(), nothing, 0, nothing, false, false, false, Set{String}(), nothing, nothing, Any[], nothing)
    I.session = s
    I.load_hook = _mod_load
    host.oo.session = s
    pre = _PRESET_VARS[]                             # variables of a driver script that ran `dynare ... noclearall`
    pre === nothing || merge!(I.base.vars, pre)
    s
end

session_error(s::ModSession, msg, cause=nothing) = ModelError(String(msg), cause)

function record!(s::ModSession, text, kind, why, n=1)
    key = (first(join(split(string(text)), " "), 120), string(kind))
    if !(key in s.skipped_keys)              # a statement skipped in every pass of a loop is reported once
        push!(s.skipped_keys, key)
        push!(s.skipped, (key[1], key[2], first(_whymsg(why), 160), n))
    end
    nothing
end
_whymsg(w) = w isa Exception ? (hasfield(typeof(w), :msg) ? w.msg : sprint(showerror, w)) : string(w)

function warn_skipped(s::ModSession)
    s.path_edits && @warn "MATLAB code in the .mod file writes simulation paths or results (oo_.*); JDyn ignores these assignments, so a perfect-foresight simulation may differ from Dynare's"
    isempty(s.skipped) && return
    kinds = Dict{String,Int}()
    for (_, k, _, n) in s.skipped
        kinds[k] = get(kinds, k, 0) + n
    end
    label = Dict("plot" => "plotting", "results" => "use of results (oo_, options_ defaults, ...)", "function" => "unknown functions",
                 "file" => "file input/output", "random" => "random numbers", "syntax" => "unsupported syntax",
                 "tainted" => "dependent on skipped statements", "eval" => "dynamic code", "environment" => "MATLAB environment",
                 "display" => "display", "block" => "blocks with an unknown condition", "feature" => "unsupported features")
    parts = join(["$n $(get(label, k, k))" for (k, n) in sort(collect(kinds); by=kv -> -kv[2])], ", ")
    firsts = join(["$(repr(t)) ($w)" for (t, _, w, _) in s.skipped[1:min(end, 3)]], "; ")
    @warn "MATLAB code in the .mod file: $(sum(values(kinds))) statement(s) not executed ($parts). In them JDyn found no write to M_ or to a parameter; variables they assign are treated as unknown. First: $firsts. Full list: model.matlab_skipped"
end

# ---- static analysis
function _scan_expr!(s::ModSession, e, info::Info, seen::Set{String})
    nodes = walk(e)
    literal_eval = Set(objectid(n.base) for n in nodes if n isa IndexE && n.base isa Id && n.base.name == "eval" &&
                       length(n.args) == 1 && n.args[1] isa Str)
    for n in nodes
        objectid(n) in literal_eval && continue
        name = nothing
        if n isa IndexE && n.base isa Id
            name = n.base.name
        elseif n isa Id
            name = n.name
        elseif n isa FieldE
            path = _dotted(n)
            if path !== nothing && path in MUTATOR_DOTTED && !has_var(s.I, s.sc, split(path, ".")[1])
                info.calib === nothing && (info.calib = "it calls the Dynare function $path, which writes to M_")
            end
        end
        (name === nothing || has_var(s.I, s.sc, name)) && continue
        if name == "eval" && n isa IndexE && length(n.args) == 1 && n.args[1] isa Str
            try
                body, funcs = mparse(n.args[1].v)
                isempty(funcs) || throw(ParseError("function"))
                _scan!(s, body, info, seen)
            catch e
                e isa ParseError || rethrow()
                info.calib === nothing && (info.calib = "eval of text that cannot be parsed")
            end
            continue
        end
        name == "load" && (info.all = true)
        name in SOLVE_FUNCS && (info.solves = true)
        if name in ("clear", "clearvars")
            info.calib === nothing && (info.calib = "it clears variables in a way JDyn cannot follow")
        end
        if name in MUTATORS
            info.calib === nothing && (info.calib = "it calls $name")
        elseif haskey(dir_files(s.I), name) && !(name in seen)
            push!(seen, name)
            u = unit(s.I, name)
            if u isa ParseError
                info.calib === nothing && (info.calib = "$name.m cannot be parsed")
            elseif is_script(u)
                _scan!(s, u.body, info, seen)
            else
                sub = Info()
                for f in values(u.funcs)
                    _scan!(s, f.body, sub, seen)
                    if any(x -> x isa GlobalS, walk(f.body))
                        sub.calib === nothing && (sub.calib = "$name.m declares global variables")
                    end
                end
                info.calib === nothing && (info.calib = sub.calib)
            end
        end
    end
end

function _target!(s::ModSession, t, info::Info)
    e, firstf = t, nothing
    while !(e isa Id)
        (e isa FieldE && e.base isa Id) && (firstf = e.name)
        e = e.base
    end
    if e.name == "M_"
        info.calib === nothing && (info.calib = "it assigns to M_")
    elseif e.name == "oo_"
        if firstf in _OO_INPUTS
            info.calib === nothing && (info.calib = "it assigns to oo_.$firstf")
        end
        info.paths = true
    elseif e.name != "options_"
        push!(info.writes, e.name)
    end
end

function _scan!(s::ModSession, stmts, info::Info, seen::Set{String})
    for st in stmts
        if st isa DynS
            push!(info.dyn, st)
            continue
        end
        info.count += 1
        if st isa Assign
            for x in st.targets
                x === nothing && continue
                _target!(s, x, info)
                _scan_expr!(s, Any[a for n in walk(x) if n isa IndexE for a in n.args], info, seen)
            end
            _scan_expr!(s, st.rhs, info, seen)
        elseif st isa ExprStmt
            _scan_expr!(s, st.e, info, seen)
            if st.e isa Id && !has_var(s.I, s.sc, st.e.name) && !haskey(BUILTINS, st.e.name) &&
               !haskey(LAZY, st.e.name) && !haskey(dir_files(s.I), st.e.name)
                info.all = true          # could be a script that is not next to the .mod file
            end
        elseif st isa CommandS
            st.name in ("load", "clear", "clearvars") && (info.all = true)
            if st.name in ("dynare", "run") || (st.name in ("clear", "clearvars") && !isempty(strip(st.args)))
                info.calib === nothing && (info.calib = "it runs '$(first(join(split(something(st.text, st.name)), " "), 60))'")
            end
        elseif st isa Unparsed
            txt = replace(st.text, r"'[^']*'" => "''")
            ids = Set(m.captures[1] for m in eachmatch(r"(?<![\w.])([A-Za-z]\w*)", txt))
            am = match(r"(?<![=<>~!])=(?!=)", txt)
            plain = am !== nothing && !occursin(r"\+\+|--|[-+*/^|&\\]=", txt) && !occursin("unterminated", st.why)
            target = plain ? txt[1:am.offset-1] : txt
            if occursin(r"\b(M_|oo_|global|persistent)\b", target) || !isempty(intersect(ids, MUTATORS)) ||
               any(d -> occursin(d, txt), MUTATOR_DOTTED)
                info.calib === nothing && (info.calib = "it cannot be parsed and may write to the model")
            end
            written = setdiff(Set(m.captures[1] for m in eachmatch(r"(?<![\w.])([A-Za-z]\w*)", target)), Set(HOST_GLOBALS))
            union!(info.writes, written)
            (!plain && am === nothing && occursin(r"^\s*[A-Za-z]\w*\s*$", txt)) && (info.all = true)
        elseif st isa IfS
            for (c, b) in st.clauses
                _scan_expr!(s, c, info, seen)
                _scan!(s, b, info, seen)
            end
            _scan!(s, st.orelse, info, seen)
        elseif st isa ForS
            st.var === nothing || push!(info.writes, st.var)
            _scan_expr!(s, st.e, info, seen)
            sub = Info()
            _scan!(s, st.body, sub, seen)
            _merge!(info, sub; loop=true)
        elseif st isa WhileS
            _scan_expr!(s, st.cond, info, seen)
            sub = Info()
            _scan!(s, st.body, sub, seen)
            _merge!(info, sub; loop=true)
        elseif st isa SwitchS
            _scan_expr!(s, st.e, info, seen)
            for (c, b) in st.cases
                _scan_expr!(s, c, info, seen)
                _scan!(s, b, info, seen)
            end
            _scan!(s, something(st.default, Any[]), info, seen)
        elseif st isa TryS
            _scan!(s, st.body, info, seen)
            _scan!(s, st.handler, info, seen)
            st.ident === nothing || push!(info.writes, st.ident)
        elseif st isa ReturnS
            info.has_return = true
        elseif st isa BreakS || st isa ContinueS
            info.has_break = true
        end
    end
end

function _merge!(info::Info, sub::Info; loop=false)
    union!(info.writes, sub.writes)
    info.calib === nothing && (info.calib = sub.calib)
    append!(info.dyn, sub.dyn)
    info.all |= sub.all
    info.paths |= sub.paths
    info.count += sub.count
    info.has_return |= sub.has_return
    info.solves |= sub.solves
    loop || (info.has_break |= sub.has_break)
end

analyze(s::ModSession, stmts) = (info = Info(); _scan!(s, stmts, info, Set{String}()); info)

# ---- taints
function taint!(s::ModSession, info::Info, why)
    ws = s.sc.vars
    s.any_taint = true
    if info.solves && !isempty(ss_params(s))
        union!(s.host.tainted, ss_params(s))
        s.host.taint_why = "a Dynare solution function was called in code that was not executed"
    end
    if info.all
        s.I.open_world = true
        for k in collect(keys(ws))
            ws[k] = Tainted(why)
        end
    end
    for k in info.writes
        ws[k] = Tainted(why)
    end
    info.paths && (s.path_edits = true)
end

_root_name(t) = (while !(t isa Id); t = t.base; end; t.name)

"Indices of the parameters a steady-state computation may change."
function ss_params(s::ModSession)
    if s.ss_params_ === nothing
        m = s.model
        idx = Set{Int}()
        if m.steady_state_fn !== nothing
            idx = Set(1:length(m.param_names))
        elseif m.steady_state_model !== nothing
            names = Set{String}()
            for (lhs, _) in m.steady_state_model
                lhs isa String ? push!(names, lhs) : union!(names, lhs)
            end
            idx = Set(m.ip[n] for n in names if haskey(m.ip, n))
        elseif m.steady_state_file !== nothing
            idx = Set(1:length(m.param_names))
            try
                body, funcs = mparse(read(m.steady_state_file, String))
                main = _first_func(funcs)
                main === nothing && throw(ParseError("no function"))
                out = length(main.outs) >= 3 ? main.outs[2] : nothing
                direct = any(n -> n isa Assign && any(x -> x !== nothing && !(x isa Id) && _root_name(x) in (out, "M_"), n.targets), walk(main.body))
                if !direct
                    assigned = Set(_root_name(x) for n in walk(main.body) if n isa Assign for x in n.targets if x !== nothing)
                    idx = Set(m.ip[n] for n in assigned if haskey(m.ip, n))
                end
            catch e
                (e isa ParseError || e isa SystemError) || rethrow()
            end
        end
        s.ss_params_ = idx
    end
    s.ss_params_
end

function after_compute!(s::ModSession, kind, why)
    h, m = s.host, s.model
    new = copy(ss_params(s))
    if kind in _ESTIMATING
        union!(new, Set(m.ip[n] for n in s.est_params if haskey(m.ip, n)))
        h.sigma_tainted = true
        h.taint_why = why
    end
    if !isempty(new)
        union!(h.tainted, new)
        h.taint_why = why
    end
    shadow_results!(s, kind)
end

# ---- execution
function run!(s::ModSession, statements)
    s.statements = collect(Any, statements)
    m = s.model
    local body, funcs
    if any(st -> st["statementName"] in ("native", "verbatim"), statements)
        lines = Any[st["statementName"] in ("native", "verbatim") ? st["string"] : (:dyn, k) for (k, st) in enumerate(statements)]
        try
            body, funcs = mparse(lines)
        catch e
            e isa ParseError || rethrow()
            throw(session_error(s, "MATLAB code in the .mod file cannot be parsed: $(e.msg)"))
        end
    else
        body, funcs = Any[DynS(k, nothing) for k in 1:length(statements)], Dict{String,Any}()
    end
    s.sc.funcs = funcs
    for (i, st) in enumerate(body)
        try
            run_stmt!(s, st)
        catch e
            if e isa ReturnSig
                break
            elseif e isa BreakSig || e isa ContinueSig
                throw(session_error(s, "MATLAB break/continue outside a loop in the .mod file"))
            elseif e isa EscapeSig
                rest = body[i+1:end]
                info = analyze(s, rest)
                _refuse_if_calibrating(s, info, "the rest of the file after a 'return' whose condition is unknown", e.u)
                taint!(s, info, "not executed")
                record!(s, "(rest of the file after a conditional return)", "block", e.u, info.count)
                break
            elseif e isa Unsupported
                throw(session_error(s, "MATLAB code in the .mod file cannot be interpreted at $(_where(s)): $(e.msg)", e))
            elseif e isa MatlabError
                throw(session_error(s, "MATLAB error in the .mod file at $(_where(s)): $(e.msg)", e))
            else
                rethrow()
            end
        end
    end
    m.matlab_skipped = Any[(t, k, w) for (t, k, w, _) in s.skipped]
    warn_skipped(s)
end

_where(s::ModSession) = s.current === nothing ? "?" : repr(first(join(split(string(something(s.current.text, ""))), " "), 100))

function _refuse_if_calibrating(s::ModSession, info::Info, what, u)
    bad = [s.statements[d.k]["statementName"] for d in info.dyn if !(s.statements[d.k]["statementName"] in COMPUTE_KINDS)]
    if info.calib !== nothing || !isempty(bad)
        why = info.calib !== nothing ? info.calib : "it contains the Dynare statement(s) $(sort(unique(bad)))"
        throw(session_error(s, "MATLAB code changes the model and cannot be interpreted: $what: $why. Reason: $(_whymsg(u))"))
    end
end

function undecidable!(s::ModSession, st, u)
    info = analyze(s, Any[st])
    _refuse_if_calibrating(s, info, "block $(repr(first(join(split(st.text), " "), 100)))", u)
    taint!(s, info, "block $(repr(first(st.text, 50))) was not executed")
    for d in info.dyn
        kind = s.statements[d.k]["statementName"]
        after_compute!(s, kind, "'$kind' inside a block that was not executed")
        forget!(s.host.options, get_or(s.statements[d.k], "options", Dict()))
        if kind == "steady"
            s.pending, s.oo_failed = nothing, "a 'steady' statement is inside a block that was not executed"
        end
    end
    record!(s, st.text, "block", u, info.count)
    if info.has_return || (info.has_break && !(st isa ForS || st isa WhileS))
        throw(EscapeSig(u))
    end
end

run_block!(s::ModSession, stmts) = (for st in stmts; run_stmt!(s, st); end)

function run_stmt!(s::ModSession, st)
    I, sc = s.I, s.sc
    tick(I)
    if st isa DynS
        s.current = st
        return exec_dyn!(s, st)
    end
    if st isa Assign || st isa ExprStmt || st isa CommandS || st isa GlobalS || st isa Unparsed
        s.current = st
        n0 = s.host.nwrites
        try
            if st isa Assign && st.rhs isa IndexE && st.rhs.base isa Id && st.rhs.base.name in SOLVE_FUNCS &&
               !has_var(I, sc, st.rhs.base.name) && !haskey(BUILTINS, st.rhs.base.name)
                s.in_try > 0 && throw(TryUnknown(Unsupported("$(st.rhs.base.name) is not run while loading", "function")))
                info = analyze(s, Any[st])
                info.calib = nothing
                taint!(s, info, "output of $(st.rhs.base.name)")
                on_solve!(s, false)
                info.paths && (s.pending = nothing; s.oo_failed = "oo_ was assigned from $(st.rhs.base.name)")
                return record!(s, st.text, "function", "$(st.rhs.base.name) is not run while loading")
            end
            return exec_simple(I, st, sc)
        catch e
            e isa Unsupported || rethrow()
            s.host.nwrites != n0 && throw(session_error(s, "MATLAB statement changed the model before failing: $(_where(s)): $(e.msg)"))
            info = analyze(s, Any[st])
            info.calib !== nothing && throw(session_error(s, "MATLAB statement changes the model and cannot be interpreted: $(_where(s)): $(info.calib). Reason: $(e.msg)"))
            s.in_try > 0 && throw(TryUnknown(e))
            taint!(s, info, "$(repr(first(join(split(something(st.text, "")), " "), 60))) was not executed")
            return record!(s, something(st.text, ""), e.kind, e)
        end
    end
    if st isa IfS || st isa SwitchS
        try
            if st isa IfS
                for (cond, body) in st.clauses
                    s.current = st
                    truth(ev(I, cond, sc)) && return run_block!(s, body)
                end
                return run_block!(s, st.orelse)
            else
                s.current = st
                body = switch_body(I, st, sc)
                return (body === nothing || isempty(body)) ? nothing : run_block!(s, body)
            end
        catch e
            (e isa Unsupported && s.current === st) || rethrow()
            return undecidable!(s, st, e)
        end
    end
    if st isa ForS
        s.current = st
        values = try
            for_values(I, st, sc)
        catch e
            e isa Unsupported || rethrow()
            return undecidable!(s, st, e)
        end
        isempty(values) && (sc.vars[st.var] = I.empty_loop_value)
        for v in values
            tick(I)
            sc.vars[st.var] = v
            try
                run_block!(s, st.body)
            catch e
                e isa BreakSig && break
                e isa ContinueSig && continue
                e isa EscapeSig && return undecidable!(s, st, e.u)
                rethrow()
            end
        end
        return nothing
    end
    if st isa WhileS
        while true
            tick(I)
            s.current = st
            c = try
                truth(ev(I, st.cond, sc))
            catch e
                e isa Unsupported || rethrow()
                return undecidable!(s, st, e)
            end
            c || break
            try
                run_block!(s, st.body)
            catch e
                e isa BreakSig && break
                e isa ContinueSig && continue
                e isa EscapeSig && return undecidable!(s, st, e.u)
                rethrow()
            end
        end
        return nothing
    end
    if st isa TryS
        n0 = s.host.nwrites
        s.in_try += 1
        try
            return run_block!(s, st.body)
        catch e
            if e isa TryUnknown
                s.in_try -= 1
                try
                    s.host.nwrites != n0 && throw(session_error(s, "MATLAB try block changed the model before a statement that cannot be interpreted: $(e.u.msg)"))
                    return undecidable!(s, st, e.u)
                finally
                    s.in_try += 1
                end
            elseif e isa MatlabError
                s.in_try -= 1
                try
                    st.ident === nothing || (sc.vars[st.ident] = MStruct(["message" => e.msg, "identifier" => e.id, "stack" => MStruct()]))
                    return run_block!(s, st.handler)
                finally
                    s.in_try += 1
                end
            else
                rethrow()
            end
        finally
            s.in_try -= 1
        end
    end
    st isa BreakS && throw(BreakSig())
    st isa ContinueS && throw(ContinueSig())
    st isa ReturnS && throw(ReturnSig())
    throw(session_error(s, "unsupported MATLAB statement in the .mod file: $(repr(something(st.text, "")))"))
end

function _json_text(obj)
    out = String[]
    stack = Any[obj]
    while !isempty(stack)
        o = pop!(stack)
        if o isa AbstractString
            push!(out, o)
        elseif o isa AbstractDict
            append!(stack, collect(values(o)))
        elseif o isa AbstractVector
            append!(stack, o)
        end
    end
    join(out, " ")
end

function exec_dyn!(s::ModSession, st)
    stj = s.statements[st.k]
    kind = stj["statementName"]
    m, ws, h = s.model, s.sc.vars, s.host
    st.text = "<$(kind)$(kind == "param_init" ? " " * stj["name"] : "")>"
    if kind in _EVALUATED_KINDS
        payload = kind == "param_init" ? get(stj, "value", "") : stj
        txt = payload isa AbstractString ? payload : _json_text(payload)
        ids = setdiff(Set(mt.match for mt in eachmatch(r"[A-Za-z_]\w*", txt)), union(Set(keys(m.ie)), Set(keys(m.ix))))
        bad = sort([n for n in ids if get(ws, n, nothing) isa Tainted && !haskey(m.ip, n)])
        isempty(bad) || throw(session_error(s, "the Dynare statement $(st.text) uses the MATLAB variable(s) $bad, whose value JDyn could not compute: $(ws[bad[1]].why)"))
        badp = sort([n for n in ids if haskey(m.ip, n) && m.ip[n] in h.tainted])
        isempty(badp) || throw(session_error(s, "the Dynare statement $(st.text) uses parameter(s) $badp, whose value at this point JDyn does not know: $(h.taint_why)"))
    end
    _exec_statement!(m, stj)
    if kind == "param_init"
        i = m.ip[stj["name"]]
        delete!(h.tainted, i)
        h.nwrites += 1
        ws[stj["name"]] = m.params[i]          # Dynare's driver also creates this workspace variable
    elseif kind in ("estimated_params", "osr_params")
        get(stj, "overwrite", false) == true && empty!(s.est_params)
        union!(s.est_params, Set(mt.match for mt in eachmatch(r"[A-Za-z_]\w*", _json_text(stj)) if haskey(m.ip, mt.match)))
    end
    if !(kind in _OPTION_FREE_KINDS)
        h.options.defaults_valid = false
        forget!(h.options, get_or(stj, "options", Dict()))
    end
    # Dynare's steady and stoch_simul write oo_ and M_ only; the workspace variables named after the model's variables
    # (and <endo>_<shock> irfs) appear when the .mod file calls send_endogenous_variables_to_workspace or
    # send_irfs_to_workspace, which the interpreter runs itself. Their results were tainted before (shadow_results!), which
    # made `n0 = n0_fss` in an endval block after 'steady' refuse when the model has an endogenous variable n0.
    if kind == "steady"
        on_solve!(s, true)
    elseif kind == "stoch_simul"
        on_solve!(s, false)
    elseif kind in ("check", "resid")
    elseif !(kind in DECL_KINDS)
        after_compute!(s, kind, "it may have been changed by '$kind'")
    end
    s.on_command === nothing || s.on_command(stj)
end

# ---- what 'steady' and 'stoch_simul' leave in M_ and oo_
"Steady state for the calibration and starting values Dynare had at a 'steady' statement."
function _steady_at(s::ModSession, params, oo)
    m = s.model
    saved = m.initval
    m.initval = copy(oo)
    try
        return with_logger(NullLogger()) do
            steady_state(m; params=copy(params), exo=Float64[get(oo, v, 0.0) for v in m.exo_names])
        end
    finally
        m.initval = saved
    end
end

"""Dynare's steady and stoch_simul store in M_.params the parameters the steady state sets; steady
also stores the steady state in oo_.steady_state (computed only if a later statement needs it)."""
function on_solve!(s::ModSession, is_steady)
    m, h = s.model, s.host
    idx = ss_params(s)
    if is_steady && s.pending !== nothing && !issubset(Set(m.endo_names), s.oo_written)
        resolve!(s)                    # the new computation starts from the previous steady state
    end
    if !isempty(idx)
        local ys, p
        try
            ys, p = _steady_at(s, m.params, m.oo)
        catch e
            e isa InterruptException && rethrow()
            why = "$(nameof(typeof(e))): $(first(_whymsg(e), 150))"
            union!(h.tainted, idx)
            h.taint_why = "the steady state could not be computed while loading ($why)"
            if is_steady
                s.pending, s.oo_failed = nothing, h.taint_why
                s.oo_written = Set{String}()
            end
            names = [m.param_names[i] for i in sort(collect(idx))]
            if !s.ss_warned
                s.ss_warned = true
                @warn "the steady state could not be computed while loading ($why): the parameters it sets ($(names[1:min(end, 8)])$(length(names) > 8 ? "..." : "")) keep the values of the .mod file, where Dynare's M_.params holds the computed ones"
            end
            return
        end
        for i in idx
            m.params[i] = p[i]
        end
        setdiff!(h.tainted, idx)
        h.nwrites += 1
        if is_steady
            for (name, v) in zip(m.endo_names, ys)
                m.oo[name] = v
            end
            s.pending, s.oo_failed, s.oo_written, s.oo_numeric = nothing, nothing, Set{String}(), false
        end
    elseif is_steady
        s.pending = (copy(m.params), copy(m.oo))
        s.oo_failed, s.oo_written, s.oo_numeric = nothing, Set{String}(), false
    end
end

_analytical(m) = m.steady_state_fn !== nothing || m.steady_state_model !== nothing || m.steady_state_file !== nothing

"Compute the steady state of the pending 'steady' statement."
function resolve!(s::ModSession)
    m = s.model
    params, oo = s.pending
    s.pending = nothing
    _analytical(m) || (s.oo_numeric = true)
    ys = try
        _steady_at(s, params, oo)[1]
    catch e
        e isa InterruptException && rethrow()
        s.oo_failed = "$(nameof(typeof(e))): $(first(_whymsg(e), 150))"
        return
    end
    for (name, v) in zip(m.endo_names, ys)
        name in s.oo_written || (m.oo[name] = v)
    end
end

"oo_.steady_state / oo_.exo_steady_state for MATLAB code, when JDyn knows them exactly."
function oo_vector(s::ModSession, name)
    m = s.model
    name == "exo_steady_state" && return ret(reshape(Float64[get(m.oo, v, 0.0) for v in m.exo_names], :, 1))
    if s.pending !== nothing && !issubset(Set(m.endo_names), s.oo_written)
        _analytical(m) || throw(Unsupported("oo_.steady_state after 'steady': a numerically computed steady state is not reproduced to the last digit", "results"))
        resolve!(s)
    end
    (s.oo_failed !== nothing || s.oo_numeric) && throw(Unsupported("oo_.steady_state is not known exactly ($(something(s.oo_failed, "computed numerically")))", "results"))
    ret(reshape(Float64[get(m.oo, v, 0.0) for v in m.endo_names], :, 1))
end

"oo_.steady_state(name) as a Dynare statement would read it now."
function oo_value(s::ModSession, name)
    if !(name in s.oo_written)
        s.pending !== nothing && resolve!(s)
        s.oo_failed !== nothing && throw(session_error(s, "an initval/endval block refers to '$name' after 'steady': in Dynare that is the computed steady state, which JDyn could not compute ($(s.oo_failed))"))
    end
    get(s.model.oo, name, 0.0)
end

"Dynare commands create workspace variables named after the model's variables."
function shadow_results!(s::ModSession, kind)
    ws, m = s.sc.vars, s.model
    for k in collect(keys(ws))
        hit = haskey(m.ie, k) || haskey(m.ix, k) || (occursin('_', k) && haskey(m.ie, rsplit(k, "_"; limit=2)[1]))
        if hit && !(ws[k] isa Tainted) && !haskey(m.ip, k)
            ws[k] = Tainted("Dynare's '$kind' may overwrite this workspace variable")
            s.any_taint = true
        end
    end
end

"Value of an expression in a Dynare statement. Parameter names mean M_.params, as in Dynare's driver."
function eval_value(s::ModSession, text::AbstractString, locals=nothing; vector::Bool=false)
    m, h = s.model, s.host
    try
        parsed = get(_EVAL_CACHE, text, nothing)
        if parsed === nothing
            parsed = mparse(String(text))
            length(_EVAL_CACHE) < 5000 && (_EVAL_CACHE[String(text)] = parsed)
        end
        body, funcs = parsed
        (!isempty(funcs) || length(body) != 1 || !(body[1] isa ExprStmt)) && throw(ParseError("not an expression"))
        sc = Scope(copy(s.sc.vars); funcs=s.sc.funcs)
        sc.global_names = Set(HOST_GLOBALS)
        for n in walk(body[1].e)
            n isa Id || continue
            name = n.name
            if haskey(m.ip, name) && !(locals !== nothing && haskey(locals, name))
                m.ip[name] in h.tainted && throw(Unsupported("the value of parameter $name is unknown ($(h.taint_why))", "tainted"))
                sc.vars[name] = m.params[m.ip[name]]
            end
        end
        if locals !== nothing
            for (k, v) in locals
                sc.vars[k] = Float64(v)
            end
        end
        push!(s.I.scope_stack, sc)
        try
            val = ev(s.I, body[1].e, sc)
            return vector && val isa AbstractArray && length(val) > 1 ? Float64.(vec(val)) : scalar(val, "value")
        finally
            pop!(s.I.scope_stack)
        end
    catch e
        (e isa ParseError || e isa Unsupported) && throw(session_error(s, "cannot evaluate $(repr(first(text, 80))): $(_whymsg(e))"))
        e isa MatlabError && throw(session_error(s, "MATLAB error while evaluating $(repr(first(text, 80))): $(e.msg)"))
        rethrow()
    end
end

"`load file` in a .mod file: numeric and char variables of a MAT-file next to it."
function _mod_load(I::Interp, args, sc)
    s = I.session
    name = strip(args[1], ['\'', '"'])
    names = [strip(a, ['\'', '"']) for a in args[2:end]]
    if any(n -> startswith(n, "-"), names) || !occursin(r"^[\w.-]+$", name) || occursin("..", name) || I.dir === nothing
        throw(Unsupported("load with options or a path", "file"))
    end
    path = nothing
    for p in (joinpath(I.dir, name), joinpath(I.dir, name * ".mat"))
        isfile(p) && (path = p; break)
    end
    path === nothing && throw(Unsupported("load: $name does not exist next to the .mod file (yet)", "file"))
    data = try
        MAT.matread(path)
    catch
        throw(Unsupported("load: $(basename(path)) is not a MAT-file that can be read", "file"))
    end
    wanted = [n for n in keys(data) if isempty(names) || n in names]
    hit = intersect(Set(wanted), Set(HOST_GLOBALS))
    isempty(hit) || throw(session_error(s, "'load $name' would replace $(sort(collect(hit))): JDyn cannot follow this"))
    for n in wanted
        v = data[n]
        if v isa AbstractMatrix{<:Real} || v isa Real
            sc.vars[n] = v isa AbstractMatrix{Bool} || v isa Bool ? ret(v isa Bool ? v : Matrix{Bool}(v)) : ret(Float64.(v isa Real ? fill(v, 1, 1) : v))
        elseif v isa AbstractString
            sc.vars[n] = String(v)
        else
            sc.vars[n] = Tainted("loaded from $(basename(path)) with a class JDyn does not read")
        end
    end
    nothing
end

# ====================================================================== steady-state files
"`<model>_steadystate.m` as a steady-state function: `f(model, params, exo) -> (ys, params)` (interpreted)."
mutable struct SteadyStateFile
    path::String
    unit::Any
end
SteadyStateFile(path::AbstractString) = SteadyStateFile(String(path), nothing)

function (f::SteadyStateFile)(model, params, exo)
    if f.unit === nothing
        body, funcs = try
            mparse(read(f.path, String))
        catch e
            e isa ParseError || rethrow()
            throw(Unsupported("$(basename(f.path)) cannot be parsed: $(e.msg)", "syntax"))
        end
        f.unit = Unit(f.path, body, funcs)
        f.unit.main === nothing && throw(Unsupported("$(basename(f.path)) is not a function file", "function"))
    end
    main = f.unit.main
    host = Host(model; params=params)
    host.readonly = true
    I = Interp(dirname(f.path), host)
    M_ = snapshot(host.M)
    ys0 = reshape(Float64[get(model.initval, v, 0.0) for v in model.endo_names], :, 1)
    # Dynare 7 calls [ys, params, info] = f(ys, exo, M_, options_)
    args = Any[ret(ys0), ret(reshape(Float64.(exo), :, 1)), M_, snapshot(host.options)]
    outs = try
        call_user(I, main, f.unit.funcs, args, 3)
    catch e
        e isa MatlabError && throw(SteadyStateError("$(basename(f.path)): MATLAB error: $(e.msg)"))
        rethrow()
    end
    ys, new, check = outs
    truth(check) && throw(SteadyStateError("$(basename(f.path)) signalled failure (check = $(scalar(check)))"))
    newv = vec(farr(new))
    length(newv) != length(model.param_names) && throw(SteadyStateError("$(basename(f.path)) returned $(length(newv)) parameters, the model has $(length(model.param_names))"))
    vec(farr(ys)), newv
end

"Names of MATLAB function files next to the .mod file that the steady_state_model block uses."
function helper_functions(mod_dir, names_used, taken)
    out = Dict{String,Any}()
    isdir(mod_dir) || return out
    for f in sort(readdir(mod_dir))
        endswith(f, ".m") || continue
        n = f[1:end-2]
        (n in names_used && !(n in taken) && occursin(r"^[A-Za-z_]\w*$", n) && !endswith(n, "_steadystate")) || continue
        head = try
            read(joinpath(mod_dir, f), String)
        catch
            continue
        end
        stripped = lstrip(replace(head, r"(?m)^\s*%.*$" => ""))
        occursin(r"^\s*function\b", first(stripped, 200)) && (out[n] = joinpath(mod_dir, f))
    end
    out
end

"""
    workspace(m) -> Dict{String,Float64}

Scalar numeric MATLAB variables of the .mod file's base workspace at the end of loading (variables
whose value JDyn could not compute are left out).
"""
function workspace(m)
    s = m.session
    s === nothing && return Dict{String,Float64}()
    Dict{String,Float64}(k => Float64(v) for (k, v) in s.sc.vars if v isa Real)
end
