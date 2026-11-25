# Model: symbols and calibration from the preprocessor's JSON, residuals and derivatives from the
# Julia code the preprocessor writes (language=julia), statements run by the MATLAB session.

struct ModelError <: Exception
    msg::String
    cause::Any          # the exception (e.g. a MatlabError raised by the file's own error()) this wraps, or nothing
end
ModelError(msg::AbstractString) = ModelError(String(msg), nothing)
Base.showerror(io::IO, e::ModelError) = print(io, "ModelError: ", e.msg)

"Locate the Dynare 7.2 preprocessor: explicit path, \$DYNARE_PREPROCESSOR, or `dynare-preprocessor` on the PATH."
function find_preprocessor(explicit=nothing)
    for cand in (explicit, get(ENV, "DYNARE_PREPROCESSOR", nothing), Sys.which("dynare-preprocessor"))
        (cand !== nothing && isfile(cand)) && return String(cand)
    end
    throw(ArgumentError("dynare-preprocessor not found; pass its path or set DYNARE_PREPROCESSOR"))
end

"""
    run_preprocessor(modfile; preprocessor=nothing, output=nothing, extra_args=String[]) -> json_dir

Run the Dynare preprocessor on `modfile` (JSON and Julia output) and return the folder holding the JSON
files. `output="second"` or `"third"` asks for derivatives up to that order whatever the `.mod` file says.
"""
function run_preprocessor(modfile::AbstractString; preprocessor=nothing, output=nothing, extra_args=String[])
    modfile = abspath(modfile)
    exe = find_preprocessor(preprocessor)
    args = [basename(modfile), "json=compute", "language=julia"]
    output !== nothing && push!(args, "output=$(output)")
    append!(args, extra_args)
    out = IOBuffer(); err = IOBuffer()
    proc = run(pipeline(ignorestatus(Cmd(`$exe $args`; dir=dirname(modfile))); stdout=out, stderr=err))
    name = splitext(basename(modfile))[1]
    if proc.exitcode != 0
        msg = String(take!(out)) * "\n" * String(take!(err))
        if occursin("`use_dll` option is not compatible with Julia", msg)
            # use_dll only says how Dynare compiles the model under MATLAB: drop it and preprocess again
            _preprocess_without_use_dll(exe, modfile, name, args, extra_args)
        else
            error("preprocessor failed:\n$msg")
        end
    end
    jd = joinpath(dirname(modfile), name, "model", "json")
    isfile(joinpath(jd, "modfile.json")) || error("no JSON output in $jd")
    _semi_prepare(exe, modfile, extra_args, jd)       # src/pac.jl: MATLAB-mode output for var_model / pac_model / ... (a no-op otherwise)
    jd
end

"""
Preprocess `modfile` without its `use_dll` option (which the preprocessor refuses with language=julia):
expand the macros, remove `use_dll` from the `model(...)` options and the `// --+ options: +--` line,
preprocess that text in a scratch folder, and move the output next to the original file.
"""
function _preprocess_without_use_dll(exe, modfile, name, args, extra_args)
    tmp = mktempdir()
    try
        expanded = joinpath(tmp, name * ".mod")
        out = IOBuffer(); err = IOBuffer()
        proc = run(pipeline(ignorestatus(Cmd(`$exe $(basename(modfile)) savemacro=$expanded onlymacro $extra_args`; dir=dirname(modfile)));
                            stdout=out, stderr=err))
        (proc.exitcode == 0 && isfile(expanded)) || error("preprocessor failed (macro expansion):\n$(String(take!(out)))\n$(String(take!(err)))")
        text = read(expanded, String)
        strip_opt(opts) = join(filter(o -> strip(o) != "use_dll", split(opts, ",")), ",")
        text = replace(text, r"\bmodel\s*\(([^()]*)\)" => m -> begin
            inner = strip_opt(match(r"\(([^()]*)\)", m).captures[1])
            isempty(strip(inner)) ? "model" : "model($inner)"
        end)
        text = replace(text, r"(--\+\s*options:[^\n]*?)\buse_dll\b" => s"\1")
        write(expanded, text)
        out = IOBuffer(); err = IOBuffer()
        proc = run(pipeline(ignorestatus(Cmd(`$exe $args`; dir=tmp)); stdout=out, stderr=err))
        proc.exitcode == 0 || error("preprocessor failed:\n$(String(take!(out)))\n$(String(take!(err)))")
        dest = joinpath(dirname(modfile), name)
        rm(dest; recursive=true, force=true)
        mv(joinpath(tmp, name), dest)
    finally
        rm(tmp; recursive=true, force=true)
    end
end

# ---------------------------------------------------------------------- generated model functions
"Functions the preprocessor's Julia output refers to."
const RUNTIME_DEFS = quote
    # MATLAB's max/min (used in Dynare's generated model files) ignore a NaN operand
    max(a::Real, b::Real) = isnan(a) ? b : isnan(b) ? a : (a > b ? a : b)
    min(a::Real, b::Real) = isnan(a) ? b : isnan(b) ? a : (a < b ? a : b)
    function get_power_deriv(x::Real, p::Real, k::Integer)
        if abs(x) < 1e-12 && p >= 0 && k > p && abs(p - round(p)) < 1e-12
            return 0.0
        end
        dxp = x^(p - k)
        for i in 0:k-1
            dxp *= p - i
        end
        dxp
    end
    normcdf(x::Real) = 0.5 * $(SpecialFunctions.erfc)(-x / sqrt(2.0))
    # the 7.2 preprocessor's Julia output writes the three-argument forms as normcdf(mu, sigma, x)
    # (ExprNode.cc: "Julia API is normcdf(mu, sigma, x)"), not MATLAB's normcdf(x, mu, sigma)
    normcdf(mu::Real, sigma::Real, x::Real) = normcdf((x - mu) / sigma)
    normpdf(x::Real) = exp(-0.5 * x * x) / sqrt(2pi)
    normpdf(mu::Real, sigma::Real, x::Real) = normpdf((x - mu) / sigma) / sigma
    erf(x::Real) = $(SpecialFunctions.erf)(x)
    erfc(x::Real) = $(SpecialFunctions.erfc)(x)
    ln(x::Real) = log(x)
    cbrt(x::Real) = $(Base.cbrt)(x)
end

# While the package is precompiled (src/precompile.jl) a model may only be loaded into modules of the package itself:
# Core.eval into an anonymous module is refused then.
_generating() = ccall(:jl_generating_output, Cint, ()) == 1
module _WorkloadModel end
baremodule _WorkloadSSM end

# ---------------------------------------------------------------------- the steady_state_model module
# The preprocessor's Julia text of a steady_state_model block runs in a bare module: no Base, so a MATLAB function
# name that Julia also has cannot get Julia's meaning (round(2.5) is 3, mod(x, 0) is x) and one that only Julia has
# (exit, sleep, ...) is undefined, which sends the block to the interpreter (src/matlab), where it is unknown. What
# the module sees is listed here: the operators and types of the generated text, and MATLAB's functions.
_ssm_round(x::Real) = _mround(Float64(x))
_ssm_sign(x::Real) = (y = Float64(x); y == 0 ? 0.0 : _msign(y))
_ssm_mod(x::Real, y::Real) = _mmod1(Float64(x), Float64(y), false)
_ssm_rem(x::Real, y::Real) = _mmod1(Float64(x), Float64(y), true)
_ssm_pow(x::Real, y::Real) = Base.:^(Float64(x), Float64(y))      # a negative base with a fractional exponent: DomainError
_ssm_periodic(f) = x -> (y = Float64(x); isinf(y) ? NaN : f(y))
_ssm_gamma(x::Real) = _mgamma(Float64(x))
_ssm_gammaln(x::Real) = _mgammaln(Float64(x))
_ssm_atan2(y::Real, x::Real) = Base.atan(Float64(y), Float64(x))
_ssm_fix(x::Real) = Base.trunc(Float64(x))

const SSM_NAMES = Dict{Symbol,Any}(
    # syntax the generated text uses
    :+ => Base.:+, :- => Base.:-, :* => Base.:*, :/ => Base.:/, :^ => _ssm_pow, :(:) => Base.:(:),
    :< => Base.:<, :> => Base.:>, :<= => Base.:<=, :>= => Base.:>=, :(==) => Base.:(==), :(!=) => Base.:(!=),
    :! => Base.:!, :Real => Base.Real, :Vector => Base.Vector, :Inf => Base.Inf, :NaN => Base.NaN, :pi => Base.pi,
    :Integer => Base.Integer, :Bool => Base.Bool, :Float64 => Base.Float64, :Int => Base.Int,
    # MATLAB functions: the same on real arguments in Julia (a complex result is a DomainError: the interpreter decides)
    :exp => Base.exp, :log => Base.log, :log10 => Base.log10, :log2 => Base.log2, :log1p => Base.log1p,
    :expm1 => Base.expm1, :sqrt => Base.sqrt, :cbrt => Base.cbrt, :abs => Base.abs, :asin => Base.asin,
    :acos => Base.acos, :atan => Base.atan, :sinh => Base.sinh, :cosh => Base.cosh, :tanh => Base.tanh,
    :asinh => Base.asinh, :acosh => Base.acosh, :atanh => Base.atanh, :floor => Base.floor, :ceil => Base.ceil,
    :hypot => Base.hypot, :isnan => Base.isnan, :isinf => Base.isinf, :isfinite => Base.isfinite,
    # MATLAB's own semantics where Julia differs
    :round => _ssm_round, :fix => _ssm_fix, :sign => _ssm_sign, :mod => _ssm_mod, :rem => _ssm_rem,
    :sin => _ssm_periodic(Base.sin), :cos => _ssm_periodic(Base.cos), :tan => _ssm_periodic(Base.tan),
    :gamma => _ssm_gamma, :lgamma => _ssm_gammaln, :gammaln => _ssm_gammaln, :atan2 => _ssm_atan2,
    :power => _ssm_pow)

"A module with only what a steady_state_model block may use (see SSM_NAMES), the functions of RUNTIME_DEFS, and `helpers`."
function steady_state_module(helpers=Dict{String,Any}())
    _generating() && return _WorkloadSSM          # precompile workload: no steady_state_model block, nothing to define
    mod = Module(gensym(:JDynSSM), false, false)
    for (n, f) in SSM_NAMES
        Core.eval(mod, Expr(:const, Expr(:(=), n, QuoteNode(f))))
    end
    Core.eval(mod, :(macro inbounds(ex); return $(GlobalRef(Base, :esc))(ex); end))
    Core.eval(mod, RUNTIME_DEFS)                      # max, min (NaN operands ignored), normcdf, normpdf, erf, erfc, ln, cbrt
    for (name, f) in helpers
        Core.eval(mod, Expr(:const, Expr(:(=), Symbol(name), QuoteNode(f))))
    end
    mod
end
"Names the steady-state module defines itself: a helper function of the same name (it takes precedence in MATLAB) forces the interpreter."
const SSM_RESERVED = Set(vcat(string.(keys(SSM_NAMES)), ["max", "min", "normcdf", "normpdf", "erf", "erfc", "ln", "cbrt", "get_power_deriv", "steady_state!"]))

"Compiled residual and derivative functions of a model, with the sparse layout of the derivatives."
mutable struct ModelFunctions
    mod::Module
    dyn_resid::Function
    dyn_resid_tt::Function
    dyn_g1::Function
    dyn_g1_tt::Function
    dyn_g2::Union{Nothing,Function}
    dyn_g2_tt::Union{Nothing,Function}
    dyn_g3::Union{Nothing,Function}
    dyn_g3_tt::Union{Nothing,Function}
    sta_resid::Function
    sta_resid_tt::Function
    sta_g1::Function
    sta_g1_tt::Function
    steady_state::Union{Nothing,Function}
    dyn_tmp::Vector{Int}            # number of temporary terms per derivative order (cumulated in T)
    sta_tmp::Vector{Int}
    g1_rows::Vector{Int}            # dynamic Jacobian: entry k is at (g1_rows[k], g1_cols[k])
    g1_cols::Vector{Int}
    sg1_rows::Vector{Int}           # static Jacobian
    sg1_cols::Vector{Int}
    g2_idx::Matrix{Int}             # nnz x 3: (equation, column 1, column 2), each symmetric entry once
    g3_idx::Matrix{Int}             # nnz x 4
end

function _csc_cols(colptr::Vector{Int})
    cols = Int[]
    for j in 1:length(colptr)-1
        append!(cols, fill(j, colptr[j+1] - colptr[j]))
    end
    cols
end

function _idx_matrix(v, k)
    isempty(v) && return zeros(Int, 0, k)
    M = zeros(Int, length(v), k)
    for (i, row) in enumerate(v)
        M[i, :] .= Int.(row)
    end
    M
end

"""
Rewrite chained comparisons `a < b < c` in the generated code as MATLAB evaluates them, `(a < b) < c`
(Julia would read `a < b && b < c`; the preprocessor prints MATLAB's left-associative chains unchanged).
"""
function _matlab_comparisons(ex)
    ex isa Expr || return ex
    args = map(_matlab_comparisons, ex.args)
    if ex.head == :comparison && length(args) > 3
        acc = Expr(:call, args[2], args[1], args[3])
        for i in 4:2:length(args)
            acc = Expr(:call, args[i], acc, args[i+1])
        end
        return acc
    end
    Expr(ex.head, args...)
end

function load_model_functions(json_dir::AbstractString, raw::Dict{String,Any}; helpers=Dict{String,Any}())
    jdir = joinpath(dirname(json_dir), "julia")
    isdir(jdir) || throw(ModelError("no Julia output from the preprocessor in $jdir (run it with language=julia)"))
    mod = _generating() ? _WorkloadModel : Module(gensym(:JDynModel))
    Core.eval(mod, RUNTIME_DEFS)
    for (name, f) in helpers                          # MATLAB helper functions called in steady_state_model
        Core.eval(mod, :($(Symbol(name))(args...) = $(f)(args...)))
    end
    files = readdir(jdir)
    ssm_ok = !any(n -> n in SSM_RESERVED, keys(helpers))
    ssmod = steady_state_module(ssm_ok ? helpers : Dict{String,Any}())
    for f in files
        endswith(f, ".jl") || continue
        if f == "SteadyState2.jl"
            ssm_ok || continue
            # the steady_state_model block: the 7.2 preprocessor writes invalid Julia for a call with several
            # outputs ([ys_[1], kk, info] = f(...)); such a block is interpreted instead (steady.jl)
            try
                Base.include(_matlab_comparisons, ssmod, joinpath(jdir, f))
            catch e
                e isa LoadError || rethrow()
            end
            continue
        end
        path = joinpath(jdir, f)
        text = read(path, String)
        if _has_external(text)                         # external functions (src/external.jl): rewrite the MATLAB-style lines
            isdefined(mod, :_EXT) || _install_external!(mod, ExternalFunctions(helpers, dirname(dirname(dirname(json_dir)))))
            Base.include_string(_matlab_comparisons, mod, _rewrite_external(text), path)
        else
            Base.include(_matlab_comparisons, mod, path)
        end
    end
    get(n, m=mod) = Base.invokelatest(isdefined, m, Symbol(n)) ? Base.invokelatest(getfield, m, Symbol(n)) : nothing
    req(n) = (f = get(n); f === nothing ? throw(ModelError("generated function $n is missing")) : f)
    dtmp = Int.(get_or(raw, "dynamic_tmp_nbr", [0, 0, 0, 0]))
    stmp = Int.(get_or(raw, "static_tmp_nbr", [0, 0, 0, 0]))
    g1r = Int.(raw["dynamic_g1_sparse_rowval"]); g1c = _csc_cols(Int.(raw["dynamic_g1_sparse_colptr"]))
    sg1r = Int.(raw["static_g1_sparse_rowval"]); sg1c = _csc_cols(Int.(raw["static_g1_sparse_colptr"]))
    ModelFunctions(mod, req("SparseDynamicResid!"), req("SparseDynamicResidTT!"), req("SparseDynamicG1!"),
                   req("SparseDynamicG1TT!"), get("SparseDynamicG2!"), get("SparseDynamicG2TT!"),
                   get("SparseDynamicG3!"), get("SparseDynamicG3TT!"),
                   req("SparseStaticResid!"), req("SparseStaticResidTT!"), req("SparseStaticG1!"), req("SparseStaticG1TT!"),
                   get("steady_state!", ssmod), dtmp, stmp, g1r, g1c, sg1r, sg1c,
                   _idx_matrix(get_or(raw, "dynamic_g2_sparse_indices", []), 3),
                   _idx_matrix(get_or(raw, "dynamic_g3_sparse_indices", []), 4))
end
get_or(d, k, default) = haskey(d, k) && d[k] !== nothing ? d[k] : default

# ---------------------------------------------------------------------- the model
mutable struct Model
    json_dir::String
    mod_dir::String
    name::String
    raw::Dict{String,Any}
    endo_names::Vector{String}
    exo_names::Vector{String}
    param_names::Vector{String}
    ie::Dict{String,Int}
    ix::Dict{String,Int}
    ip::Dict{String,Int}
    n_endo::Int
    n_exo::Int
    lead_lag_incidence::Matrix{Int}
    info::Dict{String,Any}
    varobs::Vector{String}
    statements::Vector{Any}
    fns::ModelFunctions
    functions::Dict{String,Any}
    matlab_functions::Dict{String,Any}
    steady_state_model::Union{Nothing,Vector{Any}}     # (lhs, rhs) pairs; lhs a name or a vector of names
    ssm_compiled_ok::Bool
    steady_state_file::Union{Nothing,String}
    steady_state_fn::Any
    # calibration and statement data
    params::Vector{Float64}
    initval::Dict{String,Float64}
    endval::Dict{String,Float64}
    det_shocks::Vector{Tuple{Int,Int,Int,Float64}}
    Sigma_e::Matrix{Float64}
    measurement_errors::Dict{String,Float64}
    occbin_constraints::Vector{Any}
    options::Dict{String,Any}
    matlab_options::MStruct
    matlab_skipped::Vector{Any}
    unhandled_statements::Vector{String}
    oo::Dict{String,Float64}            # what Dynare holds in oo_.steady_state / oo_.exo_steady_state
    homotopy::Any
    n_surprise::Int
    warned::Set{String}
    max_deriv_order::Int
    session::Any
    cache::Dict{Symbol,Any}
end

"""
    Model(json_dir; functions=Dict(), steady_state_fn=nothing)

Load a model from the preprocessor output in `json_dir` (see `run_preprocessor`) and run its
statements (calibration, shocks, initval, ... and the MATLAB lines between them).

`functions` maps names of functions called in a `steady_state_model` block to Julia callables; they
take precedence over MATLAB function files next to the `.mod` file. `steady_state_fn(model, params, exo)`
replaces a `steady_state_model` block or a `_steadystate.m` file; it returns `ys` or `(ys, params)`.
"""
function Model(json_dir::AbstractString; functions=Dict{String,Any}(), steady_state_fn=nothing, on_command=nothing)
    json_dir = abspath(json_dir)
    mod_dir = dirname(dirname(dirname(json_dir)))
    name = basename(dirname(dirname(json_dir)))
    raw = _read_modfile_json(joinpath(json_dir, "modfile.json"))
    endo = String[d["name"] for d in raw["endogenous"]]
    exo = String[d["name"] for d in raw["exogenous"]]
    par = String[d["name"] for d in raw["parameters"]]
    isempty(get_or(raw, "heterogeneity_dimension", [])) || throw(ModelError("heterogeneous-agent models (heterogeneity_dimension) are not supported"))
    info = raw["model_info"]
    n = length(endo)
    lli_raw = info["lead_lag_incidence"]
    lli = reduce(vcat, [reshape(Int.(r), 1, :) for r in lli_raw])
    cols = Vector{Vector{Int}}()
    push!(cols, info["maximum_endo_lag"] > 0 ? lli[:, 1] : zeros(Int, n))
    push!(cols, lli[:, info["maximum_endo_lag"] > 0 ? 2 : 1])
    push!(cols, info["maximum_endo_lead"] > 0 ? lli[:, end] : zeros(Int, n))
    lead_lag = hcat(cols...)
    (info["maximum_endo_lag"] > 1 || info["maximum_endo_lead"] > 1) &&
        throw(ModelError("expected the preprocessor to reduce the model to one lead/lag"))
    ssm = nothing
    ssm_path = joinpath(json_dir, "steady_state_model.json")
    if isfile(ssm_path)
        js = JSON.parse(read(ssm_path, String); dicttype=Dict{String,Any})
        ssm = Any[(d["lhs"] isa AbstractVector ? String.(d["lhs"]) : String(d["lhs"]), String(d["rhs"])) for d in js["steady_state_model"]]
    end
    fnames = Dict{String,Any}(String(k) => v for (k, v) in functions)
    mfuns = Dict{String,Any}()
    if ssm !== nothing
        used = Set(m.match for m in eachmatch(r"[A-Za-z_]\w*", read(ssm_path, String)))
        mfuns = helper_functions(mod_dir, used, union(Set(keys(fnames)), Set(keys(BUILTINS))))
    end
    fns = try
        load_model_functions(json_dir, raw; helpers=fnames)
    catch e
        e isa LoadError || rethrow()
        throw(ModelError("the preprocessor's Julia code for the model could not be loaded (a construct the Julia output does not support, such as external functions): " *
                         first(sprint(showerror, e.error), 200)))
    end
    ssfile = joinpath(mod_dir, name * "_steadystate.m")
    m = Model(json_dir, mod_dir, name, raw, endo, exo, par,
              Dict(v => i for (i, v) in enumerate(endo)), Dict(v => i for (i, v) in enumerate(exo)),
              Dict(v => i for (i, v) in enumerate(par)), n, length(exo), lead_lag, info,
              String.(get_or(raw, "varobs", String[])), collect(Any, get_or(raw, "statements", [])), fns,
              fnames, mfuns, ssm, isempty(mfuns), isfile(ssfile) ? ssfile : nothing, steady_state_fn,
              fill(NaN, length(par)), Dict{String,Float64}(), Dict{String,Float64}(), Tuple{Int,Int,Int,Float64}[],
              zeros(length(exo), length(exo)), Dict{String,Float64}(), Any[], Dict{String,Any}(), MStruct(), Any[],
              String[], Dict{String,Float64}(), nothing, 0, Set{String}(),
              1 + (fns.dyn_g2 !== nothing) + (fns.dyn_g3 !== nothing),
              nothing, Dict{Symbol,Any}())
    _semi_init!(m)                                       # src/pac.jl: M_.var, M_.pac, ... of the preprocessor's MATLAB output
    _run_statements!(m; on_command=on_command)
    m
end

"""
    load(modfile; kwargs...) -> Model

Run the preprocessor on `modfile` and load the model (`Model(run_preprocessor(modfile))`). `extra_args` are
command-line options of the preprocessor (e.g. `["-DA=3"]` macro definitions, `"-I path"`), as in `dynare file -DA=3`.
"""
function load(modfile::AbstractString; preprocessor=nothing, output=nothing, extra_args=String[], kwargs...)
    Model(run_preprocessor(modfile; preprocessor=preprocessor, output=output, extra_args=String.(extra_args)); kwargs...)
end

"Read modfile.json, repairing the invalid JSON the 7.2 preprocessor writes for a few statements."
function _read_modfile_json(path)
    text = read(path, String)
    try
        return JSON.parse(text; dicttype=Dict{String,Any})
    catch
    end
    text = replace(text, r"(\"dsge_var\":\s*)(-?Inf)\b" => s"\1\"\2\"")     # dsge_var=Inf
    text = replace(text, r"(\"period2\":\s*)\}(\d+)" => s"\1\2")
    text = replace(text, r"\"(lower_bound|upper_bound)\":\s*\((-[^)\"]*)\)" => s"\"\1\": \"\2\"")
    text = replace(text, r"\\(.)" => m -> (m[2] in "\"\\/bfnrtu") ? m : "\\\\" * m[2:2])
    text = replace(text, r"(\"period2\"\s*:\s*\"[^\"\n]*\")\"" => s"\1")
    text = replace(text, r"(\"name\":\s*\"[^\"\n]*\")(\"prior\":)" => s"\1, \2")
    text = replace(text, r"(\"exo\d\":\s*)([A-Za-z_]\w*\")" => s"\1\"\2")
    text = replace(text, r"(\"[\w.]*colormap\":\s*)([A-Za-z_]\w*)(?=\s*[,}])" => s"\1\"\2\"")
    text = replace(text, r"([:\[,]\s*)\.(\d)" => s"\g<1>0.\2")
    text = replace(text, r"([:\[,]\s*)0+(\d+)(?=\s*[,\]}])" => s"\1\2")
    JSON.parse(text; dicttype=Dict{String,Any})
end

orig_endo_nbr(m::Model) = Int(get_or(m.raw, "orig_endo_nbr", m.n_endo))
orig_endo_names(m::Model) = m.endo_names[1:orig_endo_nbr(m)]
aux_names(m::Model) = m.endo_names[orig_endo_nbr(m)+1:end]
param_dict(m::Model) = Dict(zip(m.param_names, m.params))

get_param(m::Model, name::AbstractString) = m.params[m.ip[name]]
"Counterpart of Dynare's `set_param_value`; takes effect at the next solve."
set_param!(m::Model, name::AbstractString, value::Real) = (m.params[m.ip[name]] = Float64(value); m)
"All-or-nothing: every name is checked before any parameter is changed."
function set_params!(m::Model, values=Dict(); kw...)
    pairs = Tuple{String,Float64}[(String(k), Float64(v)) for (k, v) in Iterators.flatten((values, kw))]
    for (k, _) in pairs
        haskey(m.ip, k) || throw(ModelError("set_params!: $k is not a parameter of the model"))
    end
    for (k, v) in pairs; m.params[m.ip[k]] = v; end
    m
end
"""
Counterpart of Dynare's `set_shock_stderr_value`: sets the diagonal element `Sigma_e(i,i) = value^2` only. As in
Dynare, covariances given earlier (`corr`/`covariance` in a `shocks` block) are kept, so the implied correlation changes.
"""
function set_shock_stderr!(m::Model, name::AbstractString, value::Real)
    haskey(m.ix, name) || throw(ModelError("set_shock_stderr!: $name is not an exogenous variable of the model"))
    i = m.ix[name]
    m.Sigma_e[i, i] = Float64(value)^2
    m
end

exo_steady(m::Model; terminal=false) = Float64[get(terminal ? merge(m.initval, m.endval) : m.initval, v, 0.0) for v in m.exo_names]

# ---------------------------------------------------------------------- varexo_det
"Names of the `varexo_det` variables (they follow the stochastic exogenous variables in the generated code's `x`)."
exo_det_names(m::Model) = String[d["name"] for d in get_or(m.raw, "exogenous_deterministic", [])]
n_exo_det(m::Model) = length(get_or(m.raw, "exogenous_deterministic", []))
"Steady-state values of the `varexo_det` variables (initval, else 0), as `oo_.exo_det_steady_state`."
exo_det_steady(m::Model) = Float64[get(m.initval, v, 0.0) for v in exo_det_names(m)]
"""
The exogenous vector the generated functions take: `x` (n_exo) followed, for a model with `varexo_det`, by the deterministic
exogenous variables at their steady-state values (or `xd`); a vector that already has all of them is left alone.
"""
function _xfull(m::Model, x, xd=nothing)
    x = x === nothing ? zeros(m.n_exo) : Vector{Float64}(x)
    nd = n_exo_det(m)
    (nd == 0 || length(x) == m.n_exo + nd) && return x
    vcat(x, xd === nothing ? exo_det_steady(m) : Vector{Float64}(xd))
end

# ---------------------------------------------------------------------- evaluation
_tmp(n::Int) = zeros(max(n, 1))

"Call a generated function; a DomainError (log of a negative number, ...) gives NaN values, as in MATLAB."
function _safe!(f, out, args...)
    try
        Base.invokelatest(f, args...)
    catch e
        e isa DomainError || rethrow()
        fill!(out, NaN)
    end
    out
end

function static_resid(m::Model, y::AbstractVector, x=nothing, params=nothing)
    f = m.fns
    x = _xfull(m, x)
    p = params === nothing ? m.params : Vector{Float64}(params)
    y = Vector{Float64}(y)
    T = _tmp(f.sta_tmp[1])
    neq = _neq(m)
    r = zeros(neq)
    try
        Base.invokelatest(f.sta_resid_tt, T, y, x, p)
        Base.invokelatest(f.sta_resid, T, r, y, x, p)
    catch e
        e isa DomainError || rethrow()
        fill!(r, NaN)
    end
    neq == m.n_endo ? r : vcat(r, zeros(m.n_endo - neq))
end
"Number of equations written in the model block (fewer than the variables under `discretionary_policy`)."
_neq(m::Model) = (v = get(m.raw, "model", nothing); v isa AbstractVector && !isempty(v) && length(v) < m.n_endo ? length(v) : m.n_endo)

function static_jac(m::Model, y::AbstractVector, x=nothing, params=nothing)
    f = m.fns
    x = _xfull(m, x)
    p = params === nothing ? m.params : Vector{Float64}(params)
    y = Vector{Float64}(y)
    T = _tmp(f.sta_tmp[1] + f.sta_tmp[2])
    v = zeros(length(f.sg1_rows))
    try
        Base.invokelatest(f.sta_g1_tt, T, y, x, p)
        Base.invokelatest(f.sta_g1, T, v, y, x, p)
    catch e
        e isa DomainError || rethrow()
        fill!(v, NaN)
    end
    J = zeros(m.n_endo, m.n_endo)
    for k in eachindex(v)
        J[f.sg1_rows[k], f.sg1_cols[k]] += v[k]
    end
    J
end

function dynamic_resid(m::Model, yy::AbstractVector, x=nothing, params=nothing, ss=nothing)
    f = m.fns
    x = _xfull(m, x)
    p = params === nothing ? m.params : Vector{Float64}(params)
    ss = ss === nothing ? zeros(m.n_endo) : Vector{Float64}(ss)
    T = _tmp(f.dyn_tmp[1])
    neq = _neq(m)
    r = zeros(neq)
    try
        Base.invokelatest(f.dyn_resid_tt, T, Vector{Float64}(yy), x, p, ss)
        Base.invokelatest(f.dyn_resid, T, r, Vector{Float64}(yy), x, p, ss)
    catch e
        e isa DomainError || rethrow()
        fill!(r, NaN)
    end
    neq == m.n_endo ? r : vcat(r, zeros(m.n_endo - neq))
end

"Structural non-zeros of the dynamic Jacobian (at `(m.fns.g1_rows, m.fns.g1_cols)`)."
function dynamic_jac_values(m::Model, yy::AbstractVector, x=nothing, params=nothing, ss=nothing)
    f = m.fns
    x = _xfull(m, x)
    p = params === nothing ? m.params : Vector{Float64}(params)
    ss = ss === nothing ? zeros(m.n_endo) : Vector{Float64}(ss)
    T = _tmp(f.dyn_tmp[1] + f.dyn_tmp[2])
    v = zeros(length(f.g1_rows))
    try
        Base.invokelatest(f.dyn_g1_tt, T, Vector{Float64}(yy), x, p, ss)
        Base.invokelatest(f.dyn_g1, T, v, Vector{Float64}(yy), x, p, ss)
    catch e
        e isa DomainError || rethrow()
        fill!(v, NaN)
    end
    v
end

"Dense dynamic Jacobian, columns ordered [y(-1), y, y(+1), exogenous] (each of length n or n_exo)."
function dynamic_jac(m::Model, yy::AbstractVector, x=nothing, params=nothing, ss=nothing)
    v = dynamic_jac_values(m, yy, x, params, ss)
    f = m.fns
    J = zeros(m.n_endo, 3 * m.n_endo + m.n_exo)
    for k in eachindex(v)
        f.g1_cols[k] > size(J, 2) && continue          # varexo_det columns: see dynamic_full_derivs
        J[f.g1_rows[k], f.g1_cols[k]] += v[k]
    end
    J
end

"""
    dynamic_sparse_derivs(m, order, yy, x, params, ss) -> (rows, cols, vals)

Derivatives of order 1 to 3 of the dynamic model: `cols` is nnz x order, 1-based, columns ordered
[y(-1), y, y(+1), exogenous]; for order >= 2 each symmetric entry appears once.
"""
function dynamic_sparse_derivs(m::Model, order::Int, yy, x=nothing, params=nothing, ss=nothing)
    r, c, v = _dynamic_sparse_derivs(m, order, yy, x, params, ss)
    n_exo_det(m) == 0 && return r, c, v
    keep = findall(k -> all(<=(3 * m.n_endo + m.n_exo), view(c, k, :)), eachindex(r))      # the stochastic view: no varexo_det columns
    r[keep], c[keep, :], v[keep]
end

"`dynamic_sparse_derivs` with the `varexo_det` columns (`[y(-1), y, y(+1), x, x_det]`)."
dynamic_full_derivs(m::Model, order::Int, yy, x=nothing, params=nothing, ss=nothing) = _dynamic_sparse_derivs(m, order, yy, x, params, ss)

function _dynamic_sparse_derivs(m::Model, order::Int, yy, x=nothing, params=nothing, ss=nothing)
    f = m.fns
    if order == 1
        return f.g1_rows, reshape(f.g1_cols, :, 1), dynamic_jac_values(m, yy, x, params, ss)
    end
    x = _xfull(m, x)
    p = params === nothing ? m.params : Vector{Float64}(params)
    ss = ss === nothing ? zeros(m.n_endo) : Vector{Float64}(ss)
    if order == 2
        f.dyn_g2 === nothing && throw(ModelError("second derivatives were not computed by the preprocessor (use output=second)"))
        T = _tmp(sum(f.dyn_tmp[1:3]))
        v = zeros(size(f.g2_idx, 1))
        Base.invokelatest(f.dyn_g2_tt, T, Vector{Float64}(yy), x, p, ss)
        Base.invokelatest(f.dyn_g2, T, v, Vector{Float64}(yy), x, p, ss)
        return f.g2_idx[:, 1], f.g2_idx[:, 2:3], v
    elseif order == 3
        f.dyn_g3 === nothing && throw(ModelError("third derivatives were not computed by the preprocessor (use output=third)"))
        T = _tmp(sum(f.dyn_tmp[1:4]))
        v = zeros(size(f.g3_idx, 1))
        Base.invokelatest(f.dyn_g3_tt, T, Vector{Float64}(yy), x, p, ss)
        Base.invokelatest(f.dyn_g3, T, v, Vector{Float64}(yy), x, p, ss)
        return f.g3_idx[:, 1], f.g3_idx[:, 2:4], v
    end
    throw(ArgumentError("derivatives above third order"))
end

# ---------------------------------------------------------------------- statements
"Apply the statements of the .mod file in order, with the MATLAB lines between them."
function _run_statements!(m::Model; on_command=nothing)
    m.params = fill(NaN, length(m.param_names))
    sess = ModSession(m)
    m.session = sess
    sess.on_command = on_command === nothing ? nothing : on_command(m)
    on_command === nothing && warn_refused_options(m)         # src/option_policy.jl (dynare() refuses the statements instead)
    run!(sess, m.statements)
    m.n_surprise > 1 && @warn "several shocks(surprise) blocks: they are merged; Dynare's 'overwrite' option is not visible in the preprocessor output"
    isempty(m.unhandled_statements) || @warn "statements not handled: $(sort(unique(m.unhandled_statements)))"
    m
end

"""
(first period, last period, value) entries of a `values` expression over `p1:p2`: a scalar gives the range as one entry, a
vector as long as the range gives one entry per period (MATLAB's `x(p1:p2) = a`).
"""
function _period_values(m::Model, expr, p1::Int, p2::Int, what)
    v = _value(m, expr; vector=true)
    v isa Real && return [(p1, p2, Float64(v))]
    length(v) == p2 - p1 + 1 || throw(ModelError("$what: $(length(v)) values for the periods $p1:$p2"))
    [(p1 + i - 1, p1 + i - 1, Float64(v[i])) for i in eachindex(v)]
end

"Value of an expression in a Dynare statement (Dynare passes these to MATLAB)."
_value(m::Model, s; locals=nothing, vector::Bool=false) = eval_value(m.session, string(s), locals; vector=vector)

"Value of an initval/endval right-hand side: references to variables read oo_.steady_state."
function _eval_with_vars(m::Model, s)
    s = string(s)
    locals = Dict{String,Float64}()
    for mt in eachmatch(r"[A-Za-z_]\w*", s)
        name = mt.match
        if haskey(m.ie, name)
            locals[name] = oo_value(m.session, name)
        elseif haskey(m.ix, name)
            locals[name] = get(m.oo, name, 0.0)
        end
    end
    _value(m, s; locals=locals)
end

const _STORED_KINDS = Set(["stoch_simul", "estimation", "occbin_solver", "estimated_params",
    "estimated_params_init", "varobs", "perfect_foresight_setup", "perfect_foresight_solver", "calib_smoother",
    "shock_decomposition", "initial_condition_decomposition", "shock_groups", "observation_trends",
    "estimated_params_bounds", "plot_shock_decomposition", "realtime_shock_decomposition",
    "squeeze_shock_decomposition", "init2shocks", "filter_initial_state",
    "perfect_foresight_with_expectation_errors_setup", "perfect_foresight_with_expectation_errors_solver",
    "histval", "mshocks", "forecast", "conditional_forecast_paths", "conditional_forecast", "plot_conditional_forecast",
    "occbin_write_regimes_xls"])
const _LATEX_KINDS = Set(["write_latex_dynamic_model", "write_latex_static_model", "write_latex_original_model",
    "write_latex_steady_state_model", "collect_latex_files", "write_latex_definitions"])

# commands whose options live in Dynare's shared `options_`
const _ACCUMULATED_KINDS = Set(["stoch_simul", "estimation", "calib_smoother"])
"`options_` of the solution commands: the options of all their statements, in file order (key \"options_\" of `m.options`)."
_accumulate_options!(m::Model, o) = (m.options["options_"] = merge(get(m.options, "options_", Dict{String,Any}()), o))

"One Dynare statement of the .mod file."
function _exec_statement!(m::Model, st)
    kind = st["statementName"]
    if kind == "param_init"
        m.params[m.ip[st["name"]]] = _value(m, st["value"])
    elseif kind == "endval" && haskey(st, "learnt_in")
        # endval(learnt_in=...) is news for perfect_foresight_with_expectation_errors, not a terminal condition
        # (perfect_foresight.jl reads these blocks from the statements)
    elseif kind == "initval" || kind == "endval"
        target = kind == "initval" ? m.initval : m.endval
        for d in st["vals"]
            v = _eval_with_vars(m, d["value"])
            target[d["name"]] = v
            m.oo[d["name"]] = v
            push!(m.session.oo_written, d["name"])
        end
    elseif kind == "load_params_and_steady_state"
        for d in get_or(st, "values", [])
            if haskey(m.ip, d["name"])
                m.params[m.ip[d["name"]]] = _value(m, d["value"])
            else
                v = _value(m, d["value"])
                m.initval[d["name"]] = v; m.oo[d["name"]] = v
                push!(m.session.oo_written, d["name"])
            end
        end
    elseif kind == "save_params_and_steady_state"
        _save_params_and_steady_state(m, st["filename"])
    elseif kind in ("ramsey_model", "ramsey_policy")
        o = get_or(st, "options", Dict{String,Any}())
        m.options[kind] = o
        kind == "ramsey_policy" && _accumulate_options!(m, o)
        ins = get(o, "instruments", nothing)
        # options_.instruments as Dynare holds it (a column cell array), for _steadystate.m files that read it
        ins === nothing || (m.matlab_options["instruments"] = Cell(Any[String(v) for v in (ins isa AbstractDict ? ins["symbol_list"] : ins)], (length(ins isa AbstractDict ? ins["symbol_list"] : ins), 1)))
    elseif kind == "shocks" && get(st, "surprise", false) == true
        m.n_surprise += 1
        push!(get!(m.options, "shock_paths", Any[]), st)
    elseif kind == "shocks"
        _shocks!(m, st)
    elseif kind == "occbin_constraints"
        m.occbin_constraints = collect(Any, st["constraints"])
    elseif kind == "homotopy" && isempty(get_or(st, "values", []))
        # homotopy_setup(from_initval_to_endval): the exogenous variables move from their initval to their endval
        # values (Dynare's steady.m adds one row per exogenous variable not listed explicitly)
        m.options["homotopy_from_initval_to_endval"] = true
    elseif kind == "homotopy" && all(d -> haskey(m.ip, d["name"]) || haskey(m.ix, d["name"]), st["values"])
        m.homotopy = st["values"]
        delete!(m.options, "homotopy_from_initval_to_endval")
    elseif kind in ("occbin_setup", "occbin_solver")
        # Dynare applies every occbin_setup / occbin_solver statement to options_.occbin in order: merge them
        o = get_or(st, "options", Dict{String,Any}())
        m.options[kind] = merge(get(m.options, kind, Dict{String,Any}()), o)
        m.options["occbin_options"] = merge(get(m.options, "occbin_options", Dict{String,Any}()), o)
    elseif kind in ("steady", "check", "occbin_graph", "rplot")
        # Dynare's options_ persists from one statement to the next: later statements add to (and override) earlier ones
        o = get_or(st, "options", Dict{String,Any}())
        m.options[kind] = merge(get(m.options, kind, Dict{String,Any}()), o)
        kind == "check" && _accumulate_options!(m, o)
        # homotopy_setup + steady(homotopy_mode=...): stepped as Dynare's homotopy1/2/3 (src/homotopy.jl); moves the parameters and
        # exogenous variables to their final values and leaves the steady state there
        kind == "steady" && haskey(get_or(st, "options", Dict()), "homotopy_mode") && _steady_homotopy!(m, merge(get(m.options, "steady", Dict()), get_or(st, "options", Dict())))
    elseif kind in _STORED_KINDS
        if kind == "estimated_params"
            m.options[kind] = st
        elseif kind in _ACCUMULATED_KINDS
            # options_ persists: an option given to an earlier stoch_simul (hp_filter, nomoments, ...) still applies to
            # the later ones (the preprocessor writes an explicit `order` for every stoch_simul, which resets it)
            o = get_or(st, "options", Dict{String,Any}())
            m.options[kind] = merge(get(m.options, kind, Dict{String,Any}()), o)
            _accumulate_options!(m, o)
        else
            m.options[kind] = get_or(st, "options", Dict{String,Any}())
        end
    elseif kind in ("planner_objective", "evaluate_planner_objective", "discretionary_policy", "osr", "osr_params", "optim_weights",
                    "osr_params_bounds", "ramsey_constraints")
        # read by src/optimal_policy.jl and src/results_policy.jl; the solution commands' options persist in options_
        o = get_or(st, "options", Dict{String,Any}())
        m.options[kind] = merge(get(m.options, kind, Dict{String,Any}()), o)
        kind in ("discretionary_policy", "osr") && _accumulate_options!(m, o)
    elseif kind == "initval_file" && any(k -> haskey(get_or(st, "options", Dict{String,Any}()), k), ("series", "first_obs", "last_obs", "first_simulation_period", "last_simulation_period"))
        throw(ModelError("initval_file with the options series, first_obs, last_obs, first_simulation_period or last_simulation_period is not implemented " *
                         "(only datafile/filename and nobs are)"))
    elseif kind == "extended_path"
        # run where it stands by dynare() (src/extended_path.jl); its options are written into options_.ep there
    elseif kind in ("estimation_data", "set_time")
        # the `data` command and set_time are read by dynare() where they stand (src/results.jl, src/results_estimation.jl)
    elseif kind in ("native", "verbatim", "perfect_foresight_controlled_paths", "initval_file") || kind in _LATEX_KINDS || kind in _SEMI_KINDS
        # native lines are run by the MATLAB session
    else
        push!(m.unhandled_statements, kind)
    end
    nothing
end

"""
    declared_correlation_matrix(m) -> Matrix

Dynare's `M_.Correlation_matrix` as the preprocessor's code leaves it: the identity, with the entries of `corr` statements of `shocks`
blocks (calibrated ones included; `var a, b = value` covariances do not enter it) and reset by `shocks(overwrite)`.
"""
declared_correlation_matrix(m::Model) = get!(() -> Matrix{Float64}(LinearAlgebra.I, m.n_exo, m.n_exo), m.options, "Correlation_matrix")::Matrix{Float64}

"Calibrated covariances of pairs of measurement errors (`M_.H` off-diagonal entries), keyed by the two names (both orders)."
me_offdiag(m::Model) = get!(() -> Dict{Tuple{String,String},Float64}(), m.options, "ME_offdiag")::Dict{Tuple{String,String},Float64}
"Declared correlations of pairs of measurement errors (`M_.Correlation_matrix_ME` off-diagonal entries), keyed by the two names (both orders)."
me_corr_entries(m::Model) = get!(() -> Dict{Tuple{String,String},Float64}(), m.options, "ME_corr")::Dict{Tuple{String,String},Float64}

"`M_.H` for the observed variables `names`: calibrated variances on the diagonal, covariances of `shocks` blocks off it."
function me_H(m::Model, names::AbstractVector{<:AbstractString})
    p = length(names)
    H = zeros(p, p)
    for (i, a) in enumerate(names)
        H[i, i] = get(m.measurement_errors, a, 0.0)
    end
    for ((a, b), v) in me_offdiag(m)
        i, j = findfirst(==(a), names), findfirst(==(b), names)
        (i === nothing || j === nothing) &&
            throw(ModelError("measurement-error covariance of $a and $b: both must be observed variables (varobs)"))
        H[i, j] = v
    end
    H
end
"`M_.Correlation_matrix_ME` for the observed variables `names` (identity plus the declared correlations)."
function me_corr_matrix(m::Model, names::AbstractVector{<:AbstractString})
    p = length(names)
    C = Matrix{Float64}(LinearAlgebra.I, p, p)
    for ((a, b), v) in me_corr_entries(m)
        i, j = findfirst(==(a), names), findfirst(==(b), names)
        (i === nothing || j === nothing) && continue
        C[i, j] = v
    end
    C
end

"One shocks block, applied in Dynare's order: reset, variances, covariances, correlations."
function _shocks!(m::Model, st)
    ix, S = m.ix, m.Sigma_e
    if get(st, "overwrite", false) == true
        S .= 0
        m.options["Correlation_matrix"] = Matrix{Float64}(LinearAlgebra.I, m.n_exo, m.n_exo)
        empty!(m.measurement_errors)
        empty!(me_offdiag(m)); empty!(me_corr_entries(m))
    end
    for (key, pw) in (("variance", 1), ("stderr", 2))
        for d in get_or(st, key, [])
            val = _value(m, d[key])^pw
            if haskey(ix, d["name"])
                S[ix[d["name"]], ix[d["name"]]] = val
            else
                m.measurement_errors[d["name"]] = val
            end
        end
    end
    for key in ("covariance", "correlation")
        for d in get_or(st, key, [])
            if !haskey(ix, d["name"]) && !haskey(ix, d["name2"])
                # covariance / correlation of two measurement errors: M_.H(i,j) and M_.Correlation_matrix_ME(i,j) (kept by name)
                val = _value(m, d[key])
                a, b = d["name"], d["name2"]
                if key == "correlation"
                    me_corr_entries(m)[(a, b)] = val
                    me_corr_entries(m)[(b, a)] = val
                    val *= sqrt(get(m.measurement_errors, a, 0.0) * get(m.measurement_errors, b, 0.0))
                end
                me_offdiag(m)[(a, b)] = val
                me_offdiag(m)[(b, a)] = val
                continue
            elseif !haskey(ix, d["name"]) || !haskey(ix, d["name2"])
                push!(m.unhandled_statements, "shocks: covariance between a structural shock and a measurement error")
                continue
            end
            i, j = ix[d["name"]], ix[d["name2"]]
            val = _value(m, d[key])
            if key == "correlation"
                C = declared_correlation_matrix(m)
                C[i, j] = C[j, i] = val
                val *= sqrt(S[i, i] * S[j, j])
            end
            S[i, j] = S[j, i] = val
        end
    end
    for d in get_or(st, "deterministic_shocks", [])
        if !haskey(ix, d["var"]) && d["var"] in exo_det_names(m)
            # shocks on varexo_det variables (Dynare's M_.det_shocks with exo_det = true): periods and values for the forecasts
            for v in d["values"]
                push!(get!(m.cache, :det_shocks_exo_det, Tuple{Int,Any,Any,Float64}[]),
                      (findfirst(==(d["var"]), exo_det_names(m)), v["period1"], v["period2"], _value(m, v["value"])))
            end
            continue
        end
        for v in d["values"]
            # dated periods are resolved against first_simulation_period when the perfect foresight statements are replayed
            (v["period1"] isa AbstractString && startswith(v["period1"], "dates(")) && continue
            p1, p2 = Int(_numval(v["period1"])), Int(_numval(v["period2"]))
            for (q1, q2, val) in _period_values(m, v["value"], p1, p2, "shocks: $(d["var"])")       # `values (a)` with a vector a: one value per period
                push!(m.det_shocks, (ix[d["var"]], q1, q2, val))
            end
        end
    end
    for key in ("skewness", "coskewness")
        isempty(get_or(st, key, [])) || push!(m.unhandled_statements, "shocks: $key")
    end
end
_numval(x::Real) = x
function _numval(x::AbstractString)
    v = tryparse(Float64, x)
    v === nothing && throw(ModelError("shock period $(repr(x)) is not a number (dates are not supported in `shocks` blocks of the load API)"))
    v
end

warn_once(m::Model, msg::AbstractString) = (msg in m.warned) || (push!(m.warned, msg); @warn msg)
