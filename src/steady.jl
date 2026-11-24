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
            if e isa UndefVarError || e isa MethodError || e isa TypeError
                # a name the bare steady-state module does not have (a MATLAB helper with several outputs, a function
                # JDyn does not know, exit(...)): interpreted, where it is resolved or refused
                m.ssm_compiled_ok = false
                y = _initval_vector(m)
            elseif e isa DomainError || e isa Unsupported || e isa MatlabError
                y = _initval_vector(m)               # complex value or unsupported case: let the interpreter decide
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
    # var(log): the auxiliary variable is log of its original variable (the equation is not linear in it)
    logaux = Int[]
    for d in get_or(m.raw, "aux_vars", [])
        (get(d, "type", -1) == 7 && haskey(d, "orig_index")) || continue
        i, o = Int(d["endo_index"]), Int(d["orig_index"])
        if isfinite(ys[o]) && ys[o] > 0
            ys[i] = log(ys[o]); push!(logaux, i)
        end
    end
    aux = [i for i in aux if !(i in logaux)]
    isempty(aux) && return ys
    J = static_jac(m, ys, x, params)
    diagc = Float64[J[i, i] for i in aux]
    exact = all(isfinite, diagc) && all(!iszero, diagc)
    for _ in 1:2
        for (k, i) in enumerate(aux)
            isfinite(ys[i]) || (ys[i] = 0.0)
            r = static_resid(m, ys, x, params)[i]
            d = exact ? diagc[k] : static_jac(m, ys, x, params)[i, i]
            (isfinite(d) && d != 0) || continue          # equation without its own variable (a lead-lag auxiliary of a unit-root / MCE model): keep the start value
            ys[i] -= r / d
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
    ys, params = _steady_state(m, params, guess, x; tolf=tol)
    if check
        res = maximum(abs, static_resid(m, ys, x, params); init=0.0)
        res < tol || throw(SteadyStateError("steady state not found: max |residual| = $(round(res; sigdigits=4))"))
        res > 1e-8 && warn_once(m, "steady state accepted with max |residual| = $(round(res; sigdigits=2)) (tolerance $(round(tol; sigdigits=2)), as in Dynare); results are only that accurate")
    end
    ys, params
end

function _steady_state(m::Model, params, guess, x; tolf=DYNARE_TOLF)
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
        ys = _solve_static(m, y0, x, params; tolf=tolf)
    end
    ys, params
end

"""
Dynare's `dogleg.m`: the Gauss-Newton step `r \\ b` (minimum-norm least squares when `r` is singular, as Octave's
backslash falls back) cut to the trust region of radius `delta`.
"""
function _dogleg(r::AbstractMatrix, b::AbstractVector, d::AbstractVector, delta::Real)
    x = try
        F = lu(r; check=false)
        issuccess(F) && size(r, 1) == size(r, 2) && rcond_ok(F, r) ? F \ b : pinv(r) * b
    catch
        pinv(r) * b
    end
    qnorm = norm(d .* x)
    qnorm <= delta && return x
    s = (r' * b) ./ d
    gnorm = norm(s)
    if gnorm > 0
        s = (s ./ gnorm) ./ d
        temp0 = norm(r * s)
        sgnorm = gnorm / (temp0 * temp0)
        if sgnorm < delta
            bnorm = norm(b)
            temp1 = delta / qnorm
            temp2 = sgnorm / delta
            temp0 = bnorm * bnorm * temp2 / (gnorm * qnorm)
            temp0 = temp0 - temp1 * temp2 * temp2 + sqrt((temp0 - temp1)^2 + (1.0 - temp1 * temp1) * (1.0 - temp2 * temp2))
            alpha = temp1 * (1.0 - temp2 * temp2) / temp0
        else
            alpha = 0.0
        end
    else
        alpha = delta / qnorm
        sgnorm = 0.0
    end
    alpha > 0 ? alpha .* x .+ (1.0 - alpha) * min(sgnorm, delta) .* s : min(sgnorm, delta) .* s
end
rcond_ok(F, r) = (c = try; 1 / cond(r, 1); catch; 0.0; end; c > eps())

"""
Dynare's `trust_region.m` (the default `solve_algo = 4` solver) on the equations `j1` and the unknowns `j2` of
`fun(x)` (residual vector) with Jacobian `jac(x)`. Returns `(x, errorflag)`.
"""
function _dynare_trust_region(fun, jac, x::Vector{Float64}, j1, j2; tolf, tolx, maxiter, factor=1.0)
    radiusfactor = 0.5; actualreductionthreshold = 0.001; relativereductionthreshold = 0.1
    x = copy(x)
    errorflag = true
    iter = 1; info = 0; delta = 0.0; ncsucc = 0; ncslow = 0
    xinit = copy(x)
    fval = fun(x)[j1]
    fjac = jac(x)[j1, j2]
    fnorm = norm(fval)
    (all(isfinite, fval) && all(isfinite, fjac)) || return x, true
    fnorm < tolf && return x, false
    fjacnorm_ = ones(length(j2))
    xnorm = 0.0
    while iter <= maxiter && info == 0
        fjacnorm = vec(sqrt.(sum(abs2, fjac; dims=1)))
        if iter == 1
            fjacnorm_ = copy(fjacnorm)
            fjacnorm_[fjacnorm .< eps(1.0)] .= 1.0
            xnorm = norm(fjacnorm_ .* x[j2])
            delta = xnorm > 0 ? xnorm * factor : factor
        else
            fjacnorm_ = max.(0.1 .* fjacnorm_, fjacnorm)
            xnorm = norm(fjacnorm_ .* x[j2])
        end
        p = _dogleg(fjac, fval, fjacnorm_, delta)
        pnorm = norm(fjacnorm_ .* p)
        xx = copy(x); x0 = copy(x)
        xx[j2] .= x[j2] .- p
        ww = fval .- fjac * p
        fval1 = fun(xx)[j1]
        if !all(isfinite, fval1)
            delta *= radiusfactor
            iter += 1
            continue
        end
        fnorm1 = norm(fval1)
        if fnorm1 < tolf
            x = xx; errorflag = false; info = 1
            continue
        end
        actualreduction = fnorm1 < fnorm ? 1.0 - (fnorm1 / fnorm)^2 : -1.0
        tt = norm(ww)
        predictedreduction = tt < fnorm ? 1.0 - (tt / fnorm)^2 : 0.0
        ratio = predictedreduction > 0 ? actualreduction / predictedreduction : 0.0
        iter == 1 && (delta = min(delta, pnorm))
        if ratio < relativereductionthreshold
            ncsucc = 0
            delta *= radiusfactor
        else
            ncsucc += 1
            if abs(ratio - 1.0) < relativereductionthreshold
                delta = pnorm / radiusfactor
            elseif ratio >= radiusfactor || ncsucc > 1
                delta = max(delta, pnorm / radiusfactor)
            end
        end
        if ratio > 1e-4
            x = xx; fval = fval1
            xnorm = norm(fjacnorm_ .* x[j2])
            fnorm = fnorm1
        end
        ncslow += 1
        actualreduction >= actualreductionthreshold && (ncslow = 0)
        iter += 1
        if iter == maxiter
            info = 2; x = copy(xinit)
            continue
        end
        if delta < tolx * xnorm
            info = 3; x = copy(xinit); errorflag = true
            continue
        end
        if tolx > 0 && max(0.1 * delta, pnorm) <= 10 * eps(xnorm) * xnorm
            info = 5; x = copy(xinit); errorflag = true
            continue
        end
        if ncslow == 15
            info = 4; x = copy(xinit); errorflag = true
            continue
        end
        fjac0 = fjac
        fjac = jac(x)[j1, j2]
        if !all(isfinite, fjac)
            x = x0; fjac = fjac0
            delta *= radiusfactor
        end
    end
    x, errorflag
end

"""
Dynare's `dynare_solve.m` for the default `solve_algo = 4` with a Jacobian: returns `(x, ok, rows, cols, why)`: the equations and
unknowns of the blocks that were actually solved, and why it failed (`:singular` for an over-determined block). The Jacobian at the initial guess is split with the
Dulmage-Mendelsohn decomposition and the blocks are solved from the last to the first; a non-square block whose
equations already hold at the initial guess is skipped (so the variables that only appear in identically satisfied
equations stay at their initial values); `j2(j)` is indexed with the *row* positions of the block, as in Dynare.
"""
function _dynare_solve(fun, jac, x0::Vector{Float64}; tolf, tolx, maxit, rng=Random.default_rng())
    x = copy(x0)
    nn = length(x)
    fvec = fun(x)
    if all(isfinite, fvec) && maximum(abs, fvec; init=0.0) < tolf
        return x, true, Int[], Int[], :none
    end
    fjac = jac(x)
    if !(all(isfinite, fvec) && all(isfinite, fjac))
        # options_.solve_randomize_initial_guess (true by default): replace the default (zero) initial values by random
        # numbers, first positive (rand*10), then normal (randn*10), then negative, until the model can be evaluated
        idx = any(!iszero, x) ? findall(iszero, x) : collect(1:nn)
        in0 = length(idx)
        good = false
        for gen in (() -> rand(rng, in0) .* 10, () -> randn(rng, in0) .* 10, () -> -rand(rng, in0) .* 10)
            for _ in 0:in0*10
                x[idx] .= gen()
                fvec = fun(x); fjac = jac(x)
                good = all(isfinite, fvec) && all(isfinite, fjac)
                good && break
            end
            good && break
        end
        good || return x0, false, Int[], Int[], :failed
    end
    j1, j2, r, s = dmperm(fjac)
    srows = Int[]; moved = Int[]
    for i in length(r)-1:-1:1
        j = r[i]:r[i+1]-1
        blocklength = r[i+1] - r[i]
        blockcolumns = s[i+1] - s[i]
        if blockcolumns != blocklength
            fval_check = fun(x)
            if norm(fval_check[j1[j]]) < tolf
                continue
            end
        end
        if blockcolumns >= blocklength
            x, errorflag = _dynare_trust_region(fun, jac, x, j1[j], j2[j]; tolf=tolf, tolx=tolx, maxiter=maxit)
            errorflag && return x, false, srows, moved, :failed
            append!(srows, j1[j]); append!(moved, j2[j])
        else
            return x, false, srows, moved, :singular     # over-determined block: the Jacobian is singular
        end
    end
    fvec = fun(x)
    if !(maximum(abs, fvec; init=0.0) <= tolf)
        x, errorflag = _dynare_trust_region(fun, jac, x, collect(1:nn), collect(1:nn); tolf=tolf, tolx=tolx, maxiter=maxit)
        errorflag && return x, false, srows, moved, :failed
        srows = collect(1:nn); moved = collect(1:nn)
    end
    x, true, srows, moved, :none
end

function _solve_static(m::Model, y0, x, params; tolf=DYNARE_TOLF, tolx=eps(Float64)^(2 / 3), maxit=50)
    f(y) = static_resid(m, y, x, params)
    jac(y) = static_jac(m, y, x, params)
    y0 = Vector{Float64}(y0)
    # Dynare's solve_algo = 4 first; the old robust search below is only used where it fails
    ys, ok, srows, moved, why = try
        _dynare_solve(f, jac, y0; tolf=tolf, tolx=tolx, maxit=maxit, rng=dynare_rng(m))
    catch e
        e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException || e isa DomainError || rethrow()
        y0, false, Int[], Int[], :failed
    end
    if ok
        # polish the unknowns that were solved for (Dynare stops at |f| < tolf; JDyn goes on to machine precision)
        if !isempty(moved)
            for _ in 1:3
                r = f(ys)
                (all(isfinite, r) && maximum(abs, r[srows]) > 1e-14) || break
                J = jac(ys)
                Jb = J[srows, moved]
                F = lu(Jb; check=false)
                issuccess(F) || break
                ys[moved] .-= F \ r[srows]
            end
        end
        return ys
    end
    why == :singular && warn_once(m, "the static Jacobian is singular at the initial values (Dulmage-Mendelsohn decomposition: an over-determined block, which Dynare's solve_algo=4 refuses): falling back to a general least-squares search; variables the equations do not pin down may move away from initval")
    _solve_static_robust(m, y0, x, params)
end

function _solve_static_robust(m::Model, y0, x, params)
    f!(r, y) = (r .= static_resid(m, y, x, params))
    j!(J, y) = (J .= static_jac(m, y, x, params))
    ys = try
        sol = NLsolve.nlsolve(f!, j!, copy(y0); method=:trust_region, ftol=1e-13, xtol=0.0, iterations=1000)
        sol.zero
    catch e
        e isa NLsolve.IsFiniteException && throw(SteadyStateError("steady state not found: the static equations are not finite along the search (Dynare's solvers would also fail from this initial point)"))
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
    "graph_save_formats", "sub_draws", "irf_plot_threshold", "impulse_responses.plot_threshold", "dirname", "verbosity", "simul.debug"])

"Options of the model's solution commands: those of every `check`, `stoch_simul`, `estimation` and `calib_smoother` statement in file order, later ones taking precedence (Dynare's `options_` persists)."
stoch_options(m::Model) = copy(get(m.options, "options_", Dict{String,Any}()))
"""
Option `name` of the model's solution commands, else a value the .mod file assigned to `options_.name` in
MATLAB code (Dynare's commands read that field too, e.g. select_qz_criterium_value.m), else `default`.
"""
function solver_option(m::Model, name::AbstractString, default)
    so = stoch_options(m)
    haskey(so, name) && return so[name]
    if haskey(m.matlab_options, name)
        v = m.matlab_options[name]
        try
            return scalar(v, "options_.$name")
        catch
        end
    end
    default
end
qz_criterium(m::Model, value=nothing; default=1.0 + 1e-6) = value !== nothing ? Float64(value) : Float64(solver_option(m, "qz_criterium", default))
qz_zero_threshold(m::Model) = Float64(solver_option(m, "qz_zero_threshold", QZ_ZERO_THRESHOLD))

"Warn (once per model) about options of `statement` that `what` does not act on."
function warn_unused(m::Model, statement, honoured, what; extra=Set{String}())
    opts = get(m.options, statement, Dict{String,Any}())
    unused = sort([k for k in keys(opts) if !(k in honoured) && !(k in COSMETIC) && !(k in extra)])
    isempty(unused) || warn_once(m, "$what does not act on these $statement options from the .mod file: $(join(unused, ", "))")
    unused
end
