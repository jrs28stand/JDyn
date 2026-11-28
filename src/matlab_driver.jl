# MATLAB driver scripts that call Dynare: run_matlab("driver.m") / @run_matlab.
#
# The script runs in JDyn's MATLAB interpreter. `dynare model [options]` (command or function syntax) runs the .mod
# file with JDyn.dynare and then, as in MATLAB, leaves M_, options_, oo_ and the file's own MATLAB variables in the
# script's workspace, so later lines can post-process them. As in Dynare:
#   - without `noclearall`, the script's variables are cleared before the .mod file runs;
#   - with `noclearall`, they are visible to the .mod file's MATLAB lines (calibration computed in the script);
#   - `-Dname=value` and other preprocessor options are passed to the preprocessor.

# Variables a driver script hands to the .mod file it runs (`dynare ... noclearall`); read by ModSession (session.jl).
const _PRESET_VARS = Ref{Any}(nothing)
# Workspace variables that belong to Dynare's own structures: never handed from the script to a new run.
const _DYNARE_GLOBALS = Set(["M_", "options_", "oo_", "estim_params_", "bayestopt_", "dataset_", "dataset_info",
                             "estimation_info", "oo_recursive_", "ans"])
# Options of MATLAB's dynare command that only change MATLAB-side behaviour (logging, console, timing): accepted.
const _DRIVER_HARMLESS = Set(["nolog", "console", "nointeractive", "notime", "nowarn", "nopreprocessoroutput",
                              "nopathchange", "noemptylinemacro", "fast", "nograph"])

# The dynare() keywords of the current run_matlab call (plots, random, ...), used by the `dynare` builtin.
const _DRIVER_KW = Ref{Any}(nothing)

function _driver_dynare(I, a, nout)
    kw = _DRIVER_KW[]
    kw === nothing && throw(Unsupported("calling dynare from MATLAB code outside run_matlab", "function"))
    words = String[strip(String(x), ['\'', '"']) for x in a if x isa AbstractString]
    length(words) == length(a) && !isempty(words) || merr("dynare: arguments must be char arrays")
    name = words[1]
    isempty(splitext(name)[2]) && (name *= ".mod")
    path = isabspath(name) ? name : joinpath(something(I.dir, pwd()), name)
    isfile(path) || merr("dynare: can't open $(words[1])")
    noclear = false
    extra = String[]
    for w in words[2:end]
        if w == "noclearall"
            noclear = true
        elseif w in _DRIVER_HARMLESS
            continue
        elseif startswith(w, "-D") || startswith(w, "-I") || occursin('=', w) || w in ("nostrict", "savemacro",
                    "onlymacro", "linemacro", "notmpterms", "nolinemacro", "stochastic", "minimal_workspace",
                    "compute_xrefs", "transform_unary_ops", "onlyclearglobals", "debug")
            push!(extra, w)
        else
            throw(Unsupported("dynare option '$w' in a driver script", "function"))
        end
    end
    base = I.base.vars
    if noclear
        _PRESET_VARS[] = Dict{String,Any}(k => v for (k, v) in base if !(k in _DYNARE_GLOBALS))
    else
        empty!(base)                                    # Dynare's `clear all` before running the file
        empty!(I.globals)
        _PRESET_VARS[] = nothing
    end
    ws = try
        dynare(path; extra_args=vcat(extra, get(kw, :extra_args, String[])), (k => v for (k, v) in kw if !(k in (:extra_args, :_workspaces)))...)
    finally
        _PRESET_VARS[] = nothing
    end
    CURRENT[] = ws
    for (k, v) in ws.vars
        k in HOST_GLOBALS || (base[k] = v)
    end
    _driver_sync_out!(I, ws)
    push!(get!(() -> Any[], kw, :_workspaces), ws)
    nout > 0 && merr("dynare does not return a value")
    Any[]
end
bi!(_driver_dynare, "dynare")

"""
    run_matlab(script; variables=Dict(), mod=nothing, kw...) -> (vars, workspaces)

Run a MATLAB script that calls Dynare, such as a driver that calibrates, runs `dynare model noclearall` and then
post-processes `oo_`. The script runs in JDyn's MATLAB interpreter; each `dynare` call runs the `.mod` file with
[`dynare`](@ref) (keywords `kw...` such as `plots=true` or `random=:octave` are passed on) and leaves `M_`, `options_`,
`oo_` and the file's variables in the script's workspace, as in MATLAB. Without `noclearall` the script's variables
are cleared first, as Dynare does; with it they are visible to the `.mod` file's MATLAB lines. Code the interpreter
cannot follow stops with an error naming it.

Returns the script's final variables (a `Dict`) and the DynareWorkspace of each `dynare` call. With `mod` (a module)
the variables are also defined there; `@run_matlab "driver.m"` does that in your session.
"""
function run_matlab(script::AbstractString; variables=Dict{String,Any}(), mod::Union{Nothing,Module}=nothing, kw...)
    path = abspath(script)
    isfile(path) || throw(ArgumentError("no file $script"))
    kwd = Dict{Symbol,Any}(kw)
    kwd[:_workspaces] = Any[]
    old = _DRIVER_KW[]
    _DRIVER_KW[] = kwd
    vars = try
        cd(dirname(path)) do
            I = Interp(dirname(path))
            merge!(I.base.vars, Dict{String,Any}(variables))
            body, funcs = mparse(read(path, String))
            I.base.funcs = funcs
            try
                exec_block(I, body, I.base)
            catch e
                e isa ReturnSig || rethrow()
            end
            out = copy(I.base.vars)
            for nm in HOST_GLOBALS
                haskey(I.globals, nm) && (out[nm] = I.globals[nm])
            end
            out
        end
    finally
        _DRIVER_KW[] = old
    end
    wss = kwd[:_workspaces]
    mod === nothing || _define_in(mod, vars)
    (vars=vars, workspaces=wss)
end

function _define_in(mod::Module, vars)
    for (k, v) in vars
        Base.isidentifier(k) || continue
        name = Symbol(k)
        if isdefined(mod, name) && !(k in ("M_", "options_", "oo_"))
            old = getfield(mod, name)
            (old isa Function || old isa Type || old isa Module || isconst(mod, name)) && continue
        end
        try
            Core.eval(mod, Expr(:global, Expr(:(=), name, QuoteNode(v))))
        catch
        end
    end
end

"""
    @run_matlab "driver.m" [kw...]

Run a MATLAB driver script with [`run_matlab`](@ref) and define its final variables (`M_`, `oo_`, `options_`,
anything the script computed) in your session, as after running the script in MATLAB.
"""
macro run_matlab(file, kws...)
    kwargs = [Expr(:kw, k.args[1], k.args[2]) for k in kws if k isa Expr && k.head == :(=)]
    esc(quote
        $JDyn.run_matlab($file; mod=$__module__, $(kwargs...))
    end)
end

# ------------------------------------------------------------------------------------------------------------------
# Dynare commands called from the driver script after `dynare`: they act on the last workspace (CURRENT[]), after
# taking in edits the script made to M_ / options_ / oo_ (e.g. `M_.params(3) = 0.9;`), and hand the results back.

_in_driver(I) = I.session === nothing && _DRIVER_KW[] !== nothing && CURRENT[] !== nothing

function _driver_sync_in!(I, ws)
    for (nm, field) in (("M_", ws.M_), ("options_", ws.options_), ("oo_", ws.oo_))
        v = get(I.globals, nm, nothing)             # M_, options_, oo_ are globals of the base workspace (interp.jl)
        (v isa MStruct && v !== field) || continue
        for k in keys(v)
            x = v[k]
            # MATLAB scalars are 1x1 matrices: the interpreter hands back a Float64 for a one-element M_.params etc.
            (x isa Real && k in ("params", "Sigma_e", "H", "steady_state", "exo_steady_state")) && (x = fill(Float64(x), 1, 1))
            field[k] = x
        end
    end
end

function _driver_sync_out!(I, ws)
    for (nm, v) in (("M_", ws.M_), ("options_", ws.options_), ("oo_", ws.oo_))
        I.globals[nm] = v
        delete!(I.base.vars, nm)
    end
end

_driver_varlist(x) = x isa AbstractString ? (isempty(x) ? String[] : [String(x)]) :
                     x isa Cell ? String[String(s) for s in x.items] :
                     x isa AbstractArray ? String[String(s) for s in vec(x)] : String[]

let f0 = BUILTINS["set_param_value"]
    bi!("set_param_value") do I, a, nout
        _in_driver(I) || return f0(I, a, nout)
        _nargs(a, 2, 2, "set_param_value")
        ws = CURRENT[]
        _driver_sync_in!(I, ws)
        set_param_value!(ws, _str(a[1], "set_param_value"), Float64(only(a[2])))
        _driver_sync_out!(I, ws)
        I.base.vars[a[1]] = a[2]
        Any[]
    end
end

# [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list) or stoch_simul(var_list)
bi!("stoch_simul") do I, a, nout
    _in_driver(I) || throw(Unsupported("unknown function or variable 'stoch_simul'", "function"))
    ws = CURRENT[]
    _driver_sync_in!(I, ws)
    if length(a) == 4
        for (x, field) in zip(a[1:3], (ws.M_, ws.options_, ws.oo_))
            x isa MStruct && x !== field && foreach(k -> (field[k] = x[k]), keys(x))
        end
        vl = _driver_varlist(a[4])
    elseif length(a) <= 1
        vl = isempty(a) ? String[] : _driver_varlist(a[1])
    else
        merr("stoch_simul: expected stoch_simul(var_list) or stoch_simul(M_, options_, oo_, var_list)")
    end
    kw = _DRIVER_KW[]
    stoch_simul!(ws, vl; plots=get(kw, :plots, nothing))
    _driver_sync_out!(I, ws)
    out = Any[0.0, ws.oo_, ws.options_, ws.M_]
    out[1:min(max(nout, 1), 4)]
end

# steady; (command) or [ys, params, info] = steady_(M_, options_, oo_)
bi!("steady") do I, a, nout
    _in_driver(I) || throw(Unsupported("unknown function or variable 'steady'", "function"))
    ws = CURRENT[]
    _driver_sync_in!(I, ws)
    steady!(ws)
    _driver_sync_out!(I, ws)
    Any[]
end
bi!("steady_") do I, a, nout
    _in_driver(I) || throw(Unsupported("unknown function or variable 'steady_'", "function"))
    ws = CURRENT[]
    _driver_sync_in!(I, ws)
    steady!(ws)
    _driver_sync_out!(I, ws)
    Any[ws.oo_.steady_state, ws.M_.params, 0.0][1:min(max(nout, 1), 3)]
end

# The other Dynare commands, under the names and signatures Dynare 7.2's generated driver uses for them
# (preprocessor/src/ComputingTasks.cc). The perfect-foresight, backward-model, extended-path and send_*_to_workspace
# functions already exist as builtins; they find the driver's workspace through `_pfo_ws` (pf_functions.jl).

# A builtin (or package function) that runs in a driver script and keeps whatever was registered before otherwise.
function _driver_cmd!(f, table::Dict, name)
    f0 = get(table, name, nothing)
    table[name] = function (I, a, nout)
        if !_in_driver(I)
            f0 === nothing && throw(Unsupported("unknown function or variable '$name'", "function"))
            return f0(I, a, nout)
        end
        ws = CURRENT[]
        _driver_sync_in!(I, ws)
        out = f(I, ws, a, nout)
        _driver_sync_out!(I, ws)
        out
    end
end
_plots_kw() = get(_DRIVER_KW[], :plots, nothing)
_firstn(v, nout) = v[1:min(max(nout, 1), length(v))]

# oo_recursive_ = dynare_estimation(var_list_)           (estimation; options from options_ and the .mod file)
_driver_cmd!(BUILTINS, "dynare_estimation") do I, ws, a, nout
    estimation!(ws, isempty(a) ? String[] : _driver_varlist(a[1]); plots=_plots_kw())
    Any[get(ws.vars, "oo_recursive_", Cell(Any[], (0, 0)))]
end

# [oo_, M_, options_, bayestopt_] = evaluate_smoother('calibration', var_list_, M_, oo_, options_, bayestopt_, estim_params_)
_driver_cmd!(BUILTINS, "evaluate_smoother") do I, ws, a, nout
    isempty(a) && merr("evaluate_smoother: not enough input arguments")
    p = a[1]
    p isa AbstractString && p == "calibration" ||
        throw(Unsupported("evaluate_smoother at '$(p isa AbstractString ? p : "a parameter vector")' from a driver script (only 'calibration')", "function"))
    calib_smoother!(ws, length(a) >= 2 ? _driver_varlist(a[2]) : String[]; plots=_plots_kw())
    nout > 4 && throw(Unsupported("evaluate_smoother with more than 4 outputs from a driver script", "function"))
    _firstn(Any[ws.oo_, ws.M_, ws.options_, get(ws.vars, "bayestopt_", MStruct())], nout)
end

# oo_.forecast = forecasts.run(var_list_, M_, options_, oo_, 'simul')
_driver_cmd!(PKG_FUNCS, "forecasts.run") do I, ws, a, nout
    length(a) >= 5 && a[5] isa AbstractString && a[5] != "simul" &&
        throw(Unsupported("forecasts.run with task '$(a[5])' from a driver script (only 'simul')", "function"))
    forecast!(ws, isempty(a) ? String[] : _driver_varlist(a[1]); plots=_plots_kw())
    Any[ws.oo_.forecast]
end

# [eigenvalues_, result, info] = check(M_, options_, oo_)
_driver_cmd!(BUILTINS, "check") do I, ws, a, nout
    ok = try
        check!(ws); true
    catch e
        e isa BlanchardKahnError || rethrow()
        false
    end
    _firstn(Any[ws.oo_.dr.eigval, ok ? 1.0 : 0.0, ok ? 0.0 : 1.0], nout)
end

# oo_.planner_objective_value = evaluate_planner_objective(M_, options_, oo_)
_driver_cmd!(BUILTINS, "evaluate_planner_objective") do I, ws, a, nout
    evaluate_planner_objective!(ws)
    Any[ws.oo_.planner_objective_value]
end

# [info, oo_, options_, M_] = discretionary_policy(M_, options_, oo_, var_list_)
_driver_cmd!(BUILTINS, "discretionary_policy") do I, ws, a, nout
    discretionary_policy!(ws, length(a) >= 4 ? _driver_varlist(a[4]) : String[])
    _firstn(Any[0.0, ws.oo_, ws.options_, ws.M_], nout)
end

# [oo_.dr, oo_.occbin.simul] = occbin.solver(M_, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state)
_driver_cmd!(PKG_FUNCS, "occbin.solver") do I, ws, a, nout
    occbin_solver!(ws)
    _firstn(Any[ws.oo_.dr, ws.oo_.occbin.simul], nout)
end
