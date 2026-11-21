# Model: symbols and calibration from the preprocessor's JSON, residuals and derivatives from the
# Julia code the preprocessor writes (language=julia), statements run by the MATLAB session.

struct ModelError <: Exception
    msg::String
end
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
    cbrt(x::Real) = Base.cbrt(x)
end

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

function load_model_functions(json_dir::AbstractString, raw::Dict{String,Any}; helpers=Dict{String,Any}())
    jdir = joinpath(dirname(json_dir), "julia")
    isdir(jdir) || throw(ModelError("no Julia output from the preprocessor in $jdir (run it with language=julia)"))
    mod = Module(gensym(:JDynModel))
    Core.eval(mod, RUNTIME_DEFS)
    for (name, f) in helpers                          # MATLAB helper functions called in steady_state_model
        Core.eval(mod, :($(Symbol(name))(args...) = $(f)(args...)))
    end
    files = readdir(jdir)
    for f in files
        endswith(f, ".jl") || continue
        if f == "SteadyState2.jl"
            # the steady_state_model block: the 7.2 preprocessor writes invalid Julia for a call with several
            # outputs ([ys_[1], kk, info] = f(...)); such a block is interpreted instead (steady.jl)
            try
                Base.include(mod, joinpath(jdir, f))
            catch e
                e isa LoadError || rethrow()
            end
            continue
        end
        Base.include(mod, joinpath(jdir, f))
    end
    get(n) = Base.invokelatest(isdefined, mod, Symbol(n)) ? Base.invokelatest(getfield, mod, Symbol(n)) : nothing
    req(n) = (f = get(n); f === nothing ? throw(ModelError("generated function $n is missing")) : f)
    dtmp = Int.(get_or(raw, "dynamic_tmp_nbr", [0, 0, 0, 0]))
    stmp = Int.(get_or(raw, "static_tmp_nbr", [0, 0, 0, 0]))
    g1r = Int.(raw["dynamic_g1_sparse_rowval"]); g1c = _csc_cols(Int.(raw["dynamic_g1_sparse_colptr"]))
    sg1r = Int.(raw["static_g1_sparse_rowval"]); sg1c = _csc_cols(Int.(raw["static_g1_sparse_colptr"]))
    ModelFunctions(mod, req("SparseDynamicResid!"), req("SparseDynamicResidTT!"), req("SparseDynamicG1!"),
                   req("SparseDynamicG1TT!"), get("SparseDynamicG2!"), get("SparseDynamicG2TT!"),
                   get("SparseDynamicG3!"), get("SparseDynamicG3TT!"),
                   req("SparseStaticResid!"), req("SparseStaticResidTT!"), req("SparseStaticG1!"), req("SparseStaticG1TT!"),
                   get("steady_state!"), dtmp, stmp, g1r, g1c, sg1r, sg1c,
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
function Model(json_dir::AbstractString; functions=Dict{String,Any}(), steady_state_fn=nothing)
    json_dir = abspath(json_dir)
    mod_dir = dirname(dirname(dirname(json_dir)))
    name = basename(dirname(dirname(json_dir)))
    raw = _read_modfile_json(joinpath(json_dir, "modfile.json"))
    endo = String[d["name"] for d in raw["endogenous"]]
    exo = String[d["name"] for d in raw["exogenous"]]
    par = String[d["name"] for d in raw["parameters"]]
    isempty(get_or(raw, "exogenous_deterministic", [])) || throw(ModelError("deterministic exogenous variables are not supported"))
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
    # external functions in the model block are not supported (the generated code would call them)
    dynj = joinpath(json_dir, "dynamic.json")
    if isfile(dynj) && occursin("external_functions_temporary_terms\":[{", replace(read(dynj, String), " " => ""))
        throw(ModelError("external functions in the model block are not supported"))
    end
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
    fns = load_model_functions(json_dir, raw; helpers=fnames)
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
    _run_statements!(m)
    m
end

"""
    load(modfile; kwargs...) -> Model

Run the preprocessor on `modfile` and load the model (`Model(run_preprocessor(modfile))`).
"""
function load(modfile::AbstractString; preprocessor=nothing, output=nothing, kwargs...)
    Model(run_preprocessor(modfile; preprocessor=preprocessor, output=output); kwargs...)
end

"Read modfile.json, repairing the invalid JSON the 7.2 preprocessor writes for a few statements."
function _read_modfile_json(path)
    text = read(path, String)
    try
        return JSON.parse(text; dicttype=Dict{String,Any})
    catch
    end
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
function set_params!(m::Model, values=Dict(); kw...)
    for (k, v) in values; set_param!(m, String(k), v); end
    for (k, v) in kw; set_param!(m, String(k), v); end
    m
end
function set_shock_stderr!(m::Model, name::AbstractString, value::Real)
    i = m.ix[name]
    m.Sigma_e[i, :] .= 0.0; m.Sigma_e[:, i] .= 0.0
    m.Sigma_e[i, i] = Float64(value)^2
    m
end

exo_steady(m::Model; terminal=false) = Float64[get(terminal ? merge(m.initval, m.endval) : m.initval, v, 0.0) for v in m.exo_names]

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
    x = x === nothing ? zeros(m.n_exo) : Vector{Float64}(x)
    p = params === nothing ? m.params : Vector{Float64}(params)
    y = Vector{Float64}(y)
    T = _tmp(f.sta_tmp[1])
    r = zeros(m.n_endo)
    try
        Base.invokelatest(f.sta_resid_tt, T, y, x, p)
        Base.invokelatest(f.sta_resid, T, r, y, x, p)
    catch e
        e isa DomainError || rethrow()
        fill!(r, NaN)
    end
    r
end

function static_jac(m::Model, y::AbstractVector, x=nothing, params=nothing)
    f = m.fns
    x = x === nothing ? zeros(m.n_exo) : Vector{Float64}(x)
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
    x = x === nothing ? zeros(m.n_exo) : Vector{Float64}(x)
    p = params === nothing ? m.params : Vector{Float64}(params)
    ss = ss === nothing ? zeros(m.n_endo) : Vector{Float64}(ss)
    T = _tmp(f.dyn_tmp[1])
    r = zeros(m.n_endo)
    try
        Base.invokelatest(f.dyn_resid_tt, T, Vector{Float64}(yy), x, p, ss)
        Base.invokelatest(f.dyn_resid, T, r, Vector{Float64}(yy), x, p, ss)
    catch e
        e isa DomainError || rethrow()
        fill!(r, NaN)
    end
    r
end

"Structural non-zeros of the dynamic Jacobian (at `(m.fns.g1_rows, m.fns.g1_cols)`)."
function dynamic_jac_values(m::Model, yy::AbstractVector, x=nothing, params=nothing, ss=nothing)
    f = m.fns
    x = x === nothing ? zeros(m.n_exo) : Vector{Float64}(x)
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
    f = m.fns
    if order == 1
        return f.g1_rows, reshape(f.g1_cols, :, 1), dynamic_jac_values(m, yy, x, params, ss)
    end
    x = x === nothing ? zeros(m.n_exo) : Vector{Float64}(x)
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
function _run_statements!(m::Model)
    m.params = fill(NaN, length(m.param_names))
    sess = ModSession(m)
    m.session = sess
    run!(sess, m.statements)
    m.n_surprise > 1 && @warn "several shocks(surprise) blocks: they are merged; Dynare's 'overwrite' option is not visible in the preprocessor output"
    isempty(m.unhandled_statements) || @warn "statements not handled: $(sort(unique(m.unhandled_statements)))"
    m
end

"Value of an expression in a Dynare statement (Dynare passes these to MATLAB)."
_value(m::Model, s; locals=nothing) = eval_value(m.session, string(s), locals)

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
    "histval", "mshocks"])
const _LATEX_KINDS = Set(["write_latex_dynamic_model", "write_latex_static_model", "write_latex_original_model",
    "write_latex_steady_state_model", "collect_latex_files", "write_latex_definitions"])

"One Dynare statement of the .mod file."
function _exec_statement!(m::Model, st)
    kind = st["statementName"]
    if kind == "param_init"
        m.params[m.ip[st["name"]]] = _value(m, st["value"])
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
    elseif kind == "shocks" && get(st, "surprise", false) == true
        m.n_surprise += 1
        push!(get!(m.options, "shock_paths", Any[]), st)
    elseif kind == "shocks"
        _shocks!(m, st)
    elseif kind == "occbin_constraints"
        m.occbin_constraints = collect(Any, st["constraints"])
    elseif kind == "homotopy" && !isempty(get_or(st, "values", [])) &&
           all(d -> haskey(m.ip, d["name"]) || haskey(m.ix, d["name"]), st["values"])
        m.homotopy = st["values"]
    elseif kind in ("steady", "check", "occbin_setup", "occbin_graph", "rplot")
        haskey(m.options, kind) || (m.options[kind] = get_or(st, "options", Dict{String,Any}()))
        if kind == "steady" && get(get_or(st, "options", Dict()), "homotopy_mode", nothing) !== nothing && m.homotopy !== nothing
            # Dynare's steady moves these parameters and exogenous variables to their final values
            for d in m.homotopy
                val = _value(m, d["final_value"])
                if haskey(m.ip, d["name"])
                    m.params[m.ip[d["name"]]] = val
                else
                    m.initval[d["name"]] = val; m.oo[d["name"]] = val
                end
            end
        end
    elseif kind in _STORED_KINDS
        m.options[kind] = kind == "estimated_params" ? st : get_or(st, "options", Dict{String,Any}())
    elseif kind in ("native", "verbatim") || kind in _LATEX_KINDS
        # native lines are run by the MATLAB session
    else
        push!(m.unhandled_statements, kind)
    end
    nothing
end

"One shocks block, applied in Dynare's order: reset, variances, covariances, correlations."
function _shocks!(m::Model, st)
    ix, S = m.ix, m.Sigma_e
    if get(st, "overwrite", false) == true
        S .= 0
        empty!(m.measurement_errors)
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
            if !haskey(ix, d["name"]) || !haskey(ix, d["name2"])
                push!(m.unhandled_statements, "shocks: measurement-error covariance")
                continue
            end
            i, j = ix[d["name"]], ix[d["name2"]]
            val = _value(m, d[key])
            key == "correlation" && (val *= sqrt(S[i, i] * S[j, j]))
            S[i, j] = S[j, i] = val
        end
    end
    for d in get_or(st, "deterministic_shocks", [])
        for v in d["values"]
            push!(m.det_shocks, (ix[d["var"]], Int(_numval(v["period1"])), Int(_numval(v["period2"])), _value(m, v["value"])))
        end
    end
    for key in ("skewness", "coskewness")
        isempty(get_or(st, key, [])) || push!(m.unhandled_statements, "shocks: $key")
    end
end
_numval(x::Real) = x
_numval(x::AbstractString) = parse(Float64, x)

warn_once(m::Model, msg::AbstractString) = (msg in m.warned) || (push!(m.warned, msg); @warn msg)
