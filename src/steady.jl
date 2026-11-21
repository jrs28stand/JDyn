# Steady-state computation (port of pydyn/steady.py).

struct SteadyStateError <: Exception
    msg::String
end
Base.showerror(io::IO, e::SteadyStateError) = print(io, "SteadyStateError: ", e.msg)

"Dynare's default tolerance on the steady-state residuals (options_.solve_tolf = eps^(1/3))."
const DYNARE_TOLF = eps(Float64)^(1 / 3)

_initval_vector(m::Model) = Float64[get(m.initval, v, 0.0) for v in m.endo_names]

"""
Evaluate the `steady_state_model` block: compiled (the preprocessor's Julia `steady_state!`) when it
only uses model symbols and Julia-known functions, otherwise interpreted. May update parameters
(returns copies). As in Dynare, variables the block does not set keep their initval (default 0).
"""
function _from_steady_state_model(m::Model, params::Vector{Float64}, x::Vector{Float64})
    params = copy(params)
    y = _initval_vector(m)
    if m.ssm_compiled_ok && m.fns.steady_state !== nothing && !_ssm_assigns_params(m) && !_ssm_multi_output(m)
        try
            Base.invokelatest(m.fns.steady_state, y, copy(x), params)
            return y, params
        catch e
            if e isa UndefVarError || e isa MethodError
                m.ssm_compiled_ok = false            # e.g. a MATLAB helper with several outputs: interpreted
                y = _initval_vector(m)
            elseif e isa DomainError
                y = _initval_vector(m)               # complex value in MATLAB: let the interpreter decide
            else
                rethrow()
            end
        end
    end
    _from_steady_state_model_interpreted(m, params, x, y)
end

_ssm_assigns_params(m::Model) = get!(m.cache, :ssm_params) do
    any(lhs -> (lhs isa String ? haskey(m.ip, lhs) : any(n -> haskey(m.ip, n), lhs)), first.(m.steady_state_model))
end
_ssm_multi_output(m::Model) = get!(m.cache, :ssm_multi) do
    any(lhs -> !(lhs isa String), first.(m.steady_state_model))
end

function _ssm_interp(m::Model, params)
    host = Host(m; params=params)
    host.readonly = true                    # helpers may read "global M_", not write it
    I = Interp(m.mod_dir, host)
    I.jlfuncs = m.functions
    I
end

function _from_steady_state_model_interpreted(m::Model, params, x, y)
    I = _ssm_interp(m, params)
    sc = Scope(; funcs=Dict{String,Any}())
    sc.global_names = Set(HOST_GLOBALS)
    for (i, v) in enumerate(m.param_names); sc.vars[v] = params[i]; end
    for (i, v) in enumerate(m.exo_names); sc.vars[v] = x[i]; end
    for (i, v) in enumerate(m.endo_names); sc.vars[v] = y[i]; end
    function store(lhs, val)
        val = val isa Real ? Float64(val) : scalar(val, "steady_state_model value")
        if haskey(m.ie, lhs)
            y[m.ie[lhs]] = val
        elseif haskey(m.ip, lhs)
            params[m.ip[lhs]] = val
        end
        sc.vars[lhs] = val
    end
    push!(I.scope_stack, sc)
    try
        for (lhs, rhs) in m.steady_state_model
            body, funcs = get!(() -> mparse(rhs), _EVAL_CACHE, rhs)
            (length(body) == 1 && body[1] isa ExprStmt) || throw(SteadyStateError("steady_state_model: cannot parse $(repr(rhs))"))
            if lhs isa String
                store(lhs, ev(I, body[1].e, sc))
            else
                vals = ev_multi(I, body[1].e, sc, length(lhs))
                length(vals) < length(lhs) && throw(SteadyStateError("steady_state_model: $(first(rhs, 60)) does not return $(length(lhs)) values"))
                for (nm, v) in zip(lhs, vals)
                    store(nm, v)
                end
            end
        end
    catch e
        e isa Unsupported && throw(SteadyStateError("steady_state_model: a MATLAB function cannot be interpreted ($(e.msg)); pass a Julia function in Model(...; functions=...)"))
        e isa MatlabError && throw(SteadyStateError("steady_state_model: MATLAB error: $(e.msg)"))
        rethrow()
    finally
        pop!(I.scope_stack)
    end
    y, params
end

"""
Set auxiliary variables from their defining equations, which are linear in the variable itself: two
Gauss-Seidel sweeps of one Newton step per auxiliary variable (as pydyn). The coefficient of an auxiliary
variable in its own equation is constant, so the Jacobian is evaluated once.
"""
function fill_auxiliary(m::Model, ys, x, params)
    aux = get!(m.cache, :aux_endo_index) do
        Int[Int(d["endo_index"]) for d in get_or(m.raw, "aux_vars", [])]
    end::Vector{Int}
    ys = copy(ys)
    isempty(aux) && return ys
    x = Vector{Float64}(x); params = Vector{Float64}(params)
    for i in aux
        isfinite(ys[i]) || (ys[i] = 0.0)
    end
    J = static_jac(m, ys, x, params)
    diagc = Float64[J[i, i] for i in aux]
    exact = all(isfinite, diagc) && all(!iszero, diagc)
    for _ in 1:2
        for (k, i) in enumerate(aux)
            isfinite(ys[i]) || (ys[i] = 0.0)
            r = static_resid(m, ys, x, params)[i]
            ys[i] -= r / (exact ? diagc[k] : static_jac(m, ys, x, params)[i, i])
        end
    end
    ys
end

"""
    steady_state(m; params=m.params, guess=nothing, tol=nothing, check=nothing, exo=nothing) -> (ys, params)

Use a Julia steady-state function, the `steady_state_model` block or a `_steadystate.m` file when the
model has one, otherwise solve the static model numerically from `initval` (or `guess`). The residuals
are then checked against `tol`: by default the `tolf` given to `steady` in the .mod file, else Dynare's
default (about 6e-6). `check=false` (or `steady(nocheck)`) skips the test.
"""
function steady_state(m::Model; params=nothing, guess=nothing, tol=nothing, check=nothing, exo=nothing)
    so = get(m.options, "steady", Dict{String,Any}())
    tol = tol === nothing ? Float64(get(so, "solve_tolf", DYNARE_TOLF)) : tol
    check = check === nothing ? !(get(so, "steadystate.nocheck", false) == true) : check
    warn_unused(m, "steady", Set(["solve_tolf", "steadystate.nocheck", "homotopy_mode", "homotopy_steps"]), "steady_state")
    params = params === nothing ? copy(m.params) : Vector{Float64}(params)
    x = exo === nothing ? exo_steady(m) : Vector{Float64}(exo)
    ys, params = _steady_state(m, params, guess, x)
    if check
        res = maximum(abs, static_resid(m, ys, x, params); init=0.0)
        res < tol || throw(SteadyStateError("steady state not found: max |residual| = $(round(res; sigdigits=4))"))
        res > 1e-8 && warn_once(m, "steady state accepted with max |residual| = $(round(res; sigdigits=2)) (tolerance $(round(tol; sigdigits=2)), as in Dynare); results are only that accurate")
    end
    ys, params
end

function _steady_state(m::Model, params, guess, x)
    fn = m.steady_state_fn
    if fn === nothing && m.steady_state_model === nothing && m.steady_state_file !== nothing
        fn = steady_state_file_fn(m)
    end
    if fn !== nothing
        out = try
            fn(m, copy(params), x)
        catch e
            e isa Unsupported && throw(SteadyStateError("$(basename(something(m.steady_state_file, "steady_state_fn"))) cannot be interpreted ($(e.msg)): pass an equivalent function as Model(...; steady_state_fn=...)"))
            rethrow()
        end
        if out isa Tuple
            ys, params = Vector{Float64}(vec(out[1])), Vector{Float64}(vec(out[2]))
        else
            ys = Vector{Float64}(vec(out))
        end
        no = orig_endo_nbr(m)
        if length(ys) == no && no < m.n_endo
            ys = vcat(ys, fill(NaN, m.n_endo - no))
        end
        length(ys) == m.n_endo || throw(SteadyStateError("steady_state_fn returned $(length(ys)) values, expected $no or $(m.n_endo)"))
        ys = fill_auxiliary(m, ys, x, params)
    elseif m.steady_state_model !== nothing
        ys, params = _from_steady_state_model(m, params, x)
        if !all(isfinite, ys)
            bad = [v for (v, y) in zip(m.endo_names, ys) if !isfinite(y)]
            throw(SteadyStateError("steady_state_model gives non-finite values for: $bad"))
        end
    else
        y0 = guess === nothing ? fill_auxiliary(m, _initval_vector(m), x, params) : Vector{Float64}(guess)
        all(isfinite, y0) || throw(SteadyStateError("initial guess for the steady state is not finite"))
        ys = _solve_static(m, y0, x, params)
    end
    ys, params
end

function _solve_static(m::Model, y0, x, params)
    f!(r, y) = (r .= static_resid(m, y, x, params))
    j!(J, y) = (J .= static_jac(m, y, x, params))
    ys = try
        sol = NLsolve.nlsolve(f!, j!, copy(y0); method=:trust_region, ftol=1e-13, xtol=0.0, iterations=1000)
        sol.zero
    catch e
        e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException || rethrow()
        copy(y0)
    end
    for _ in 1:50                                 # polish with Newton steps
        r = static_resid(m, ys, x, params)
        (all(isfinite, r) && all(isfinite, ys)) || throw(SteadyStateError("steady state not found: non-finite values during the search"))
        maximum(abs, r; init=0.0) < 1e-14 && break
        J = static_jac(m, ys, x, params)
        F = lu(J; check=false)
        ys = issuccess(F) ? ys - F \ r : ys - pinv(J) * r
    end
    ys
end

"The model's `_steadystate.m` as a steady-state function, or nothing."
function steady_state_file_fn(m::Model)
    m.steady_state_file === nothing && return nothing
    get!(m.cache, :ssfile) do
        SteadyStateFile(m.steady_state_file)
    end
end

# ---------------------------------------------------------------------- options
const COSMETIC = Set(["nograph", "nodisplay", "graph_format", "noprint", "nocorr", "nomoments", "nofunctions", "tex",
    "nodecomposition", "contemporaneous_correlation", "TeX", "no_graph.posterior", "no_graph.shock_decomposition",
    "nodiagnostic", "console_mode", "silent_optimizer", "debug", "plot_priors", "no_graph.plot_shock_decomposition",
    "graph_save_formats", "sub_draws", "irf_plot_threshold", "dirname", "verbosity", "simul.debug"])

"Options of the model's solution commands, later commands taking precedence as in Dynare."
function stoch_options(m::Model)
    out = Dict{String,Any}()
    for kind in ("check", "stoch_simul", "estimation", "calib_smoother")
        merge!(out, get(m.options, kind, Dict{String,Any}()))
    end
    out
end
qz_criterium(m::Model, value=nothing; default=1.0 + 1e-6) = value !== nothing ? Float64(value) : Float64(get(stoch_options(m), "qz_criterium", default))

"Warn (once per model) about options of `statement` that `what` does not act on."
function warn_unused(m::Model, statement, honoured, what; extra=Set{String}())
    opts = get(m.options, statement, Dict{String,Any}())
    unused = sort([k for k in keys(opts) if !(k in honoured) && !(k in COSMETIC) && !(k in extra)])
    isempty(unused) || warn_once(m, "$what does not act on these $statement options from the .mod file: $(join(unused, ", "))")
    unused
end
