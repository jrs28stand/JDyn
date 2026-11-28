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
        _PRESET_VARS[] = nothing
    end
    ws = try
        dynare(path; extra_args=vcat(extra, get(kw, :extra_args, String[])), (k => v for (k, v) in kw if !(k in (:extra_args, :_workspaces)))...)
    finally
        _PRESET_VARS[] = nothing
    end
    CURRENT[] = ws
    for (k, v) in ws.vars
        base[k] = v
    end
    base["M_"] = ws.M_; base["options_"] = ws.options_; base["oo_"] = ws.oo_
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
            mrun(read(path, String); variables=Dict{String,Any}(variables), directory=dirname(path))
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
