# Optimal policy after Dynare 7.2: Ramsey (`ramsey_model`, `ramsey_policy`), welfare (`evaluate_planner_objective`),
# discretion (`discretionary_policy`) and optimal simple rules (`osr`). A port of pydyn/optimal_policy.py, after
# Dynare's matlab/optimal_policy/{dyn_ramsey_static, evaluate_planner_objective, get_optimal_policy_discount_factor}.m,
# the Ramsey branch of evaluate_steady_state.m, matlab/discretionary_policy/*.m, matlab/+osr/{run,objective,
# get_variance_of_endogenous_variables}.m and matlab/matrix_solver/lyapunov_symm.m (fixed-point method 3).
#
# Ramsey models need nothing special beyond the steady state: the preprocessor writes the planner's first-order
# conditions (with the Lagrange multipliers MULT_*) into the model, `steady_state` routes such models to
# `ramsey_steady_state` (dyn_ramsey_static.m), and `first_order`, `k_order`, moments, simulations and IRFs then
# work on the Ramsey model as on any other.
#
# Dynare 7.2 defect (corrected by default; `replicate_dynare_bugs=true` reproduces Dynare's number):
# evaluate_planner_objective at order 2 builds Wyu from Uyy*kron(gu,gy) and Wyy*kron(Gu,Gy), which multiply
# kron(u,yhat), but evaluates Wyu*kron(yhat,u) (the layout of ghxu). With one shock or one predetermined
# variable the two agree; with several of both Dynare's conditional welfare mixes up the cross terms.

"Dynare stops with an error in the optimal-policy code (its `info` code is in `.info`)."
struct OptimalPolicyError <: Exception
    msg::String
    info::Union{Nothing,Int}
end
OptimalPolicyError(msg::AbstractString) = OptimalPolicyError(String(msg), nothing)
OptimalPolicyError(msg::AbstractString, info::Integer) = OptimalPolicyError(String(msg), Int(info))
Base.showerror(io::IO, e::OptimalPolicyError) = print(io, "OptimalPolicyError: ", e.msg)

"Raised by `osr` (Dynare's own `error('OSR: ...')`)."
struct OSRError <: Exception
    msg::String
end
Base.showerror(io::IO, e::OSRError) = print(io, "OSRError: ", e.msg)

"Raised by `discretionary_policy`."
struct DiscretionaryPolicyError <: Exception
    msg::String
    info::Union{Nothing,Int}
end
DiscretionaryPolicyError(msg::AbstractString) = DiscretionaryPolicyError(String(msg), nothing)
DiscretionaryPolicyError(msg::AbstractString, info::Integer) = DiscretionaryPolicyError(String(msg), Int(info))
Base.showerror(io::IO, e::DiscretionaryPolicyError) = print(io, "DiscretionaryPolicyError: ", e.msg)

const RAMSEY_MAXIT = 500         # options_.ramsey.maxit
const GSTEP1 = 1e-2              # options_.gstep(1): floor of the step of the forward-difference Jacobians

# ---------------------------------------------------------------------- statements
_statement(m::Model, kinds...) = (i = findfirst(st -> get(st, "statementName", "") in kinds, m.statements); i === nothing ? nothing : m.statements[i])
_statements(m::Model, kinds...) = [st for st in m.statements if get(st, "statementName", "") in kinds]

"The model is the Ramsey problem written by the preprocessor (`ramsey_model` or `ramsey_policy`)."
is_ramsey(m::Model) = get!(m.cache, :is_ramsey) do
    any(a -> get(a, "type", -1) == 6, get_or(m.raw, "aux_vars", [])) && _statement(m, "ramsey_model", "ramsey_policy") !== nothing
end::Bool
is_discretionary(m::Model) = _statement(m, "discretionary_policy") !== nothing

"Number of equations of the model as written (fewer than the variables under `discretionary_policy`: the instruments' equations are missing)."
eq_nbr(m::Model) = _neq(m)

"What Dynare keeps in `M_.ramsey_orig_endo_nbr`, `M_.ramsey_orig_eq_nbr`, `options_.instruments`."
struct RamseyInfo
    orig_endo_nbr::Int          # M_.ramsey_orig_endo_nbr: variables (and FOC equations) before the multipliers
    orig_eq_nbr::Int            # M_.ramsey_orig_eq_nbr: equations of the original model = number of multipliers
    mult::Vector{Int}           # endogenous indices (1-based) of the multipliers, in equation order
    instruments::Vector{String}
    inst_idx::Vector{Int}
    recursive::Vector{Int}      # auxiliary variables created after the multipliers (chain rule in A)
    discount_index::Int         # parameter index of optimal_policy_discount_factor
    inst_nbr::Int
end

function _instrument_names(m::Model)
    out = String[]
    for st in _statements(m, "ramsey_model", "ramsey_policy", "discretionary_policy")
        ins = get(get_or(st, "options", Dict{String,Any}()), "instruments", nothing)
        ins === nothing && continue
        out = String.(ins isa AbstractDict ? ins["symbol_list"] : ins)
    end
    out
end

function ramsey_info(m::Model)::RamseyInfo
    get!(m.cache, :ramsey_info) do
        is_ramsey(m) || throw(OptimalPolicyError("the model has no ramsey_model / ramsey_policy statement"))
        aux = get_or(m.raw, "aux_vars", [])
        mults = sort([a for a in aux if get(a, "type", -1) == 6]; by=a -> Int(a["endo_index"]))
        mult = Int[Int(a["endo_index"]) for a in mults]
        n_r = minimum(mult) - 1
        mult == collect(n_r+1:n_r+length(mult)) ||
            throw(ModelError("Lagrange multipliers that are not contiguous in the list of variables are not supported"))
        eqs = [a["eq_nbr"] for a in mults]
        eqs == sort(eqs) || throw(ModelError("Lagrange multipliers are not in equation order"))
        rec = sort(Int[Int(a["endo_index"]) for a in aux if get(a, "type", -1) != 6 && Int(a["endo_index"]) > maximum(mult)])
        names = _instrument_names(m)
        for nme in names
            haskey(m.ie, nme) || throw(OptimalPolicyError("instrument $nme is not an endogenous variable"))
        end
        haskey(m.ip, "optimal_policy_discount_factor") || throw(OptimalPolicyError("optimal_policy_discount_factor is not a parameter of the model"))
        RamseyInfo(n_r, length(mult), mult, names, Int[m.ie[v] for v in names], rec, m.ip["optimal_policy_discount_factor"], n_r - length(mult))
    end
end

"`get_optimal_policy_discount_factor`: the parameter `optimal_policy_discount_factor`."
function planner_discount(m::Model, params=nothing)
    haskey(m.ip, "optimal_policy_discount_factor") ||
        throw(OptimalPolicyError("the model has no planner discount factor (optimal_policy_discount_factor): declare planner_discount in the Ramsey, discretionary or planner_objective statement"))
    Float64((params === nothing ? m.params : params)[m.ip["optimal_policy_discount_factor"]])
end

# ---------------------------------------------------------------------- Ramsey steady state
"Rows of the static model holding the defining equations of auxiliary variables `idx` (1-based variable indices)."
function _aux_rows(m::Model, idx::AbstractVector{Int})
    is_ramsey(m) || return collect(idx)
    ri = ramsey_info(m)
    # a variable created before the multipliers has its equation among the original ones (after the FOCs): in the
    # same relative position minus the instruments; those created after are last
    Int[i <= ri.orig_endo_nbr ? i + ri.orig_endo_nbr - ri.inst_nbr : i for i in idx]
end

"Run `f` with `ys_init` as the starting values the steady-state file or block sees (Dynare's `ys_init` argument)."
function _with_initval(f, m::Model, ys_init::AbstractVector)
    saved = m.initval
    m.initval = merge(saved, Dict(zip(m.endo_names, Float64.(ys_init))))
    try
        return f()
    finally
        m.initval = saved
    end
end

"`evaluate_steady_state_file.m` for a Ramsey model (auxiliary variables are not set, no check). Returns `(ys, params)`."
function _evaluate_steady_state_file(m::Model, ys_init, x, params)
    _with_initval(m, ys_init) do
        fn = m.steady_state_fn
        if fn === nothing && m.steady_state_model === nothing
            fn = steady_state_file_fn(m)
        end
        if fn !== nothing
            out = try
                fn(m, copy(params), x)
            catch e
                e isa Unsupported && throw(SteadyStateError("steady-state file cannot be interpreted ($(e.msg))"))
                rethrow()
            end
            ys, p = out isa Tuple ? (Vector{Float64}(vec(out[1])), Vector{Float64}(vec(out[2]))) : (Vector{Float64}(vec(out)), copy(params))
            # MATLAB grows ys_init(1:k) = ...: the rest is ys_init
            length(ys) < m.n_endo && (ys = vcat(ys, Float64.(ys_init)[length(ys)+1:end]))
            ys, p
        else
            _from_steady_state_model(m, copy(params), x)
        end
    end
end

_has_steady_state_file(m::Model) = m.steady_state_fn !== nothing || m.steady_state_model !== nothing || m.steady_state_file !== nothing

"""
`ramsey_multipliers_static_g1`: derivatives of the first-order conditions with respect to the multipliers, through the
auxiliary variables created after them (chain rule, as the preprocessor's computeRamseyMultipliersDerivatives).
"""
function _multiplier_jacobian(m::Model, ri::RamseyInfo, xx, x, params)
    J = static_jac(m, xx, x, params)
    F = 1:ri.orig_endo_nbr
    A = J[F, ri.mult]
    if !isempty(ri.recursive)
        rows = _aux_rows(m, ri.recursive)
        dR = -(J[rows, ri.recursive] \ J[rows, ri.mult])
        A = A + J[F, ri.recursive] * dR
    end
    A
end

"MATLAB's `A \\ y`: LU for square matrices (least squares when singular), QR least squares otherwise."
function _mldivide(A::AbstractMatrix, y::AbstractVector)
    if size(A, 1) == size(A, 2)
        F = lu(A; check=false)
        issuccess(F) && rcond_ok(F, A) && return F \ y
    end
    pinv(A) * y
end

"The residuals of `dyn_ramsey_static_1`: returns `(resids, steady_state, params)` at `z` (original variables, or the instruments with a steady-state file)."
function _ramsey_static_1(m::Model, ri::RamseyInfo, z::AbstractVector, x, ys_init, params, ssflag::Bool)
    n_orig = orig_endo_nbr(m)
    has_aux = any(a -> get(a, "type", -1) != 6, get_or(m.raw, "aux_vars", []))
    if ssflag
        ys = copy(ys_init)
        ys[ri.inst_idx] = z
        full = try
            y, params = _evaluate_steady_state_file(m, ys, x, params)
            y
        catch e
            e isa SteadyStateError || rethrow()
            return ones(ri.inst_nbr) .+ sum(abs, ys[1:n_orig]), fill(NaN, m.n_endo), params
        end
        # MATLAB would see complex values here (penalty 1 + sum|imag|); JDyn's NaN gets the unit penalty
        all(isfinite, full[1:n_orig]) || return ones(ri.inst_nbr), fill(NaN, m.n_endo), params
        z = full
    end
    xx = zeros(m.n_endo)
    xx[1:n_orig] = z[1:n_orig]
    has_aux && (xx = fill_auxiliary(m, xx, x, params))
    res = static_resid(m, xx, x, params)
    A = _multiplier_jacobian(m, ri, xx, x, params)
    y = res[1:ri.orig_endo_nbr]
    mult = (all(isfinite, A) && all(isfinite, y)) ? -_mldivide(A, y) : fill(NaN, length(ri.mult))
    resids1 = y + A * mult
    if ri.inst_nbr == 1
        r1 = [sqrt(resids1' * resids1)]
    elseif ri.inst_nbr <= 0
        r1 = Float64[]
    elseif all(isfinite, A) && all(isfinite, y)
        R = qr(Matrix(transpose(hcat(A, y)))).R
        d = abs.(diag(R)[1:min(size(R)..., size(A, 2))])
        if any(<=(1e-10 * max(1.0, maximum(abs, A))), d)
            warn_once(m, "Ramsey steady state: the residual dyn_ramsey_static takes from a QR factorisation is not unique for this model (dependent first-order conditions come first); the steady-state search can differ from Dynare's. Declaring the instruments last can avoid this")
        end
        r1 = R[end, size(A, 1)-ri.inst_nbr+1:size(A, 1)]
    else
        r1 = fill(NaN, ri.inst_nbr)
    end
    resids = ssflag ? r1 : vcat(res[ri.orig_endo_nbr+1:ri.orig_endo_nbr+n_orig-ri.inst_nbr], r1)
    ss = zeros(m.n_endo)
    ss[1:ri.orig_endo_nbr] = xx[1:ri.orig_endo_nbr]
    ss[ri.mult] = mult
    has_aux && (ss = fill_auxiliary(m, ss, x, params))
    resids, ss, params
end

"The forward-difference Jacobian of dynare_solve.m / trust_region.m without analytic Jacobian."
function _numjac(fun, z::Vector{Float64}, f0)
    n = length(z)
    dh = max.(abs.(z), GSTEP1) .* eps(Float64)^(1 / 3)
    J = zeros(length(f0), n)
    for j in 1:n
        zz = copy(z)
        zz[j] += dh[j]
        J[:, j] = (fun(zz) .- f0) ./ dh[j]
    end
    J
end

"csolve.m's numerical gradient (`gradfun = ''`)."
function _csolve_numgrad(fun)
    z -> begin
        f0 = fun(z)
        n = length(z)
        g = zeros(length(f0), n)
        for i in 1:n
            e = zeros(n); e[i] = 1e-6
            g[:, i] = (fun(z + e) - f0) / 1e-6
        end
        f0, g
    end
end

"""
    ramsey_steady_state(m; params=m.params, guess=nothing, exo=nothing, tolf, tolx, solve_algo, maxit) -> (ys, params)

Steady state of a Ramsey model as Dynare's `evaluate_steady_state` computes it (`dyn_ramsey_static.m`). `guess` is Dynare's
`ys_init` (default the `initval` values, 0 elsewhere).

* With a `steady_state_model` block, a `_steadystate.m` file or a `steady_state_fn`, the block gives the steady state
  conditional on the instruments (their starting values come from `initval`); the instruments are found by `csolve`
  (one instrument) or `dynare_solve` (several) so that the first-order conditions hold with least-squares multipliers.
* Otherwise the original variables are solved for by `dynare_solve` (`solve_algo` 4 or 9; forward-difference Jacobian),
  the multipliers again by least squares; a starting point that already solves the whole static model is kept.
"""
function ramsey_steady_state(m::Model, params=nothing, guess=nothing, x=nothing; tolf=nothing, tolx=nothing, solve_algo=nothing, maxit=nothing)
    ri = ramsey_info(m)
    so = get(m.options, "steady", Dict{String,Any}())
    tolf = tolf === nothing ? Float64(get(so, "solve_tolf", DYNARE_TOLF)) : Float64(tolf)
    tolx = tolx === nothing ? Float64(get(so, "solve_tolx", eps(Float64)^(2 / 3))) : Float64(tolx)
    solve_algo = Int(solve_algo === nothing ? solver_option(m, "solve_algo", get(so, "solve_algo", 4)) : solve_algo)
    maxit = Int(maxit === nothing ? solver_option(m, "ramsey.maxit", RAMSEY_MAXIT) : maxit)
    params = Vector{Float64}(params === nothing ? m.params : params)
    x = x === nothing ? exo_steady(m) : Vector{Float64}(x)
    isfinite(params[ri.discount_index]) || @warn "evaluate_steady_state: the planner_discount is NaN/Inf. That will cause problems."
    ssflag = _has_steady_state_file(m)
    ys_init = guess === nothing ? _initval_vector(m) : Vector{Float64}(guess)
    n_orig = orig_endo_nbr(m)
    has_aux = any(a -> get(a, "type", -1) != 6, get_or(m.raw, "aux_vars", []))
    has_aux && !ssflag && (ys_init = fill_auxiliary(m, ys_init, x, params))
    lo, hi = ri.orig_endo_nbr + 1, ri.orig_endo_nbr + ri.orig_eq_nbr
    ys = nothing
    if ssflag
        isempty(ri.instruments) && throw(OptimalPolicyError("a Ramsey model with a steady-state file needs instruments (ramsey_model(instruments=...))"))
        ys0, p1 = try
            _evaluate_steady_state_file(m, ys_init, x, params)
        catch e
            e isa SteadyStateError || rethrow()
            throw(OptimalPolicyError("steady-state file failed: $(e.msg)", 19))
        end
        resids = static_resid(m, ys0, x, p1)
        any(isnan, resids[lo:hi]) &&
            throw(OptimalPolicyError("The steady state file computation for the Ramsey problem resulted in NaNs (Dynare's error 84): if the initial values of the instruments are not admissible, change them in an initval block", 84))
        maximum(abs, resids[lo:hi]) > tolf &&
            throw(OptimalPolicyError("The steady state file does not solve the steady state for the Ramsey problem (Dynare's error 85)", 85))
    end
    check = 0
    p_out = params
    nl_ss = z -> _ramsey_static_1(m, ri, z, x, ys_init, params, ssflag)
    nl = z -> nl_ss(z)[1]
    if !ssflag && norm(static_resid(m, ys_init, x, params)) < tolf
        ys = ys_init
    elseif ssflag
        z0 = ys_init[ri.inst_idx]
        if length(ri.instruments) != ri.inst_nbr
            # dyn_ramsey_static solves for the declared instruments, dyn_ramsey_static_1 returns inst_nbr residuals: a
            # non-square system that Dynare's solvers only handle when the starting point already solves it
            f0 = nl(z0)
            (all(isfinite, f0) && maximum(abs, f0) < tolf) ||
                throw(ModelError("$(length(ri.instruments)) instruments declared for $(ri.inst_nbr) missing equation(s), and the initial values of the instruments do not solve the Ramsey steady state: Dynare would run its solver on a non-square system"))
            z = z0
        elseif length(ri.instruments) == 1
            z, rc = csolve(nl, z0, _csolve_numgrad(nl), tolf, maxit; rng=dynare_rng(m))
            (rc == 1 || rc == 3) && (check = 81)
            rc == 4 && (check = 87)
        else
            z, err = _ramsey_dynare_solve(m, nl, z0, tolf, tolx, maxit, solve_algo)
            err && (check = 81)
        end
        ys_tmp = copy(ys_init); ys_tmp[ri.inst_idx] = z
        _, p_out = _evaluate_steady_state_file(m, ys_tmp, x, params)     # run the file again to update the parameters
        ys = _ramsey_static_1(m, ri, z, x, ys_init, p_out, ssflag)[2]
    else
        z, err = _ramsey_dynare_solve(m, nl, ys_init[1:n_orig], tolf, tolx, maxit, solve_algo)
        err && (check = 81)
        ys = nl_ss(z)[2]
    end
    check != 0 && throw(OptimalPolicyError("The steady state for the Ramsey problem could not be computed (Dynare's error $check)", check))
    resids = static_resid(m, ys, x, p_out)
    any(isnan, resids[lo:end]) &&
        throw(OptimalPolicyError("The steady state computation for the Ramsey problem resulted in NaNs (Dynare's error 82)", 82))
    any(isnan, resids[1:lo-1]) &&
        throw(OptimalPolicyError("The steady state computation for the Ramsey problem resulted in NaNs in the auxiliary equations (Dynare's error 83)", 83))
    maximum(abs, resids) > tolf &&
        throw(OptimalPolicyError("The steady state for the Ramsey problem could not be computed (Dynare's error 81): max |residual| = $(round(maximum(abs, resids); sigdigits=4))", 81))
    ys, p_out
end

"`dynare_solve.m` with `options_.jacobian_flag = false` (as dyn_ramsey_static sets it): returns `(x, errorflag)`."
function _ramsey_dynare_solve(m::Model, fun, z0, tolf, tolx, maxit, solve_algo)
    # solve_algo 10/11 (mixed complementarity solvers, used with ramsey_constraints) concern the perfect foresight problem;
    # Dynare computes the Ramsey steady state before the file sets them, with the default algorithm
    solve_algo in (10, 11) && (solve_algo = 4)
    if solve_algo == 4
        jac = z -> _numjac(fun, Vector{Float64}(z), fun(z))
        z, ok, = _dynare_solve(fun, jac, Vector{Float64}(z0); tolf=tolf, tolx=tolx, maxit=maxit, rng=dynare_rng(m), jacflag=false)
        return z, !ok
    elseif solve_algo == 9
        jac = z -> _numjac(fun, Vector{Float64}(z), fun(z))
        n = length(z0)
        z0 = Vector{Float64}(z0)
        f0 = fun(z0)
        all(isfinite, f0) && maximum(abs, f0) < tolf && return z0, false
        z, err = _dynare_trust_region(fun, jac, z0, collect(1:n), collect(1:n); tolf=tolf, tolx=tolx, maxiter=maxit)
        return z, err
    end
    throw(ModelError("Ramsey steady state with solve_algo=$solve_algo: only solve_algo 4 and 9 are implemented for the Ramsey problem"))
end

# ---------------------------------------------------------------------- planner objective
const _JL_KEYWORDS = Set(["begin", "end", "function", "if", "else", "elseif", "for", "while", "in", "let", "do", "try", "catch",
                          "finally", "return", "break", "continue", "global", "local", "const", "module", "struct", "macro",
                          "quote", "abstract", "primitive", "mutable", "type", "true", "false", "using", "import", "export"])

"Julia code of a Dynare expression: integer literals become floats (2^-1 would otherwise fail), comparisons as MATLAB reads them."
function _objective_expr(s::AbstractString)
    ex = _matlab_comparisons(Meta.parse(replace(String(s), "~=" => "!=", "~" => "!")))
    _floatify(ex)
end
_floatify(x::Integer) = Float64(x)
_floatify(x::Expr) = x.head === :ref ? x : Expr(x.head, map(_floatify, x.args)...)
_floatify(x) = x

"""
The `planner_objective` expression as the preprocessor writes it (`static_model_simple` of the statement): value, gradient and
Hessian with respect to the endogenous variables (declaration order).
"""
struct PlannerObjective
    n::Int
    value_fn::Function
    jac_fn::Function
    hess_fn::Function
    has_hessian::Bool          # false when the preprocessor computed derivatives of order 1 only (stoch_simul order=1)
end

function _compile_objective(m::Model)
    st = _statement(m, "planner_objective")
    st === nothing && throw(OptimalPolicyError("there is no planner_objective statement in the .mod file"))
    blk = get(st, "static_model_simple", nothing)
    blk === nothing && throw(ModelError("planner_objective without the preprocessor's static_model_simple block"))
    isempty(get_or(blk, "external_functions_temporary_terms_", [])) || throw(ModelError("external functions in the planner objective are not supported"))
    for nm in vcat(m.endo_names, m.param_names, m.exo_names)
        nm in _JL_KEYWORDS && throw(ModelError("the planner objective cannot be compiled: the symbol $nm is a Julia keyword"))
    end
    pre = String[]
    for (i, nm) in enumerate(m.endo_names); push!(pre, "$nm = __y[$i]"); end
    for (i, nm) in enumerate(m.param_names); push!(pre, "$nm = __p[$i]"); end
    for nm in m.exo_names; push!(pre, "$nm = 0.0"); end
    for d in get_or(blk, "model_local_variables", [])
        push!(pre, "$(d["variable"]) = $(d["value"])")
    end
    temps(key) = String["$(d["temporary_term"]) = $(d["value"])" for d in get_or(blk, key, [])]
    t0, t1, t2 = temps("temporary_terms_"), temps("temporary_terms_jacobian"), temps("temporary_terms_hessian")
    resid = blk["residuals"][1]["residual"]
    value = "($(resid["lhs"])) - ($(resid["rhs"]))"
    jac = Tuple{Int,String}[(Int(e["indices"][2]), e["val"]) for e in get_or(get_or(blk, "jacobian", Dict()), "entries", [])]
    hes = Tuple{Int,Int,String}[(Int(e["indices"][2]), Int(e["indices"][3]), e["val"]) for e in get_or(get_or(blk, "hessian", Dict()), "entries", [])]
    mk(body_lines, ret) = begin
        src = "function (__y, __p)\n" * join(vcat(pre, body_lines), "\n") * "\n" * ret * "\nend"
        ex = _floatify(_matlab_comparisons(Meta.parse(src)))
        mod = get!(m.cache, :objective_module) do
            md = _generating() ? _WorkloadModel : Module(gensym(:JDynObjective))
            Core.eval(md, RUNTIME_DEFS)
            md
        end
        Core.eval(mod, ex)
    end
    parsed(s) = replace(s, r"\bget_power_deriv\b" => "get_power_deriv")
    vfn = mk(t0, parsed(value))
    jfn = mk(vcat(t0, t1), "Tuple{Int,Float64}[" * join(("($(c), Float64($(parsed(v))))" for (c, v) in jac), ", ") * "]")
    hfn = mk(vcat(t0, t1, t2), "Tuple{Int,Int,Float64}[" * join(("($(i), $(j), Float64($(parsed(v))))" for (i, j, v) in hes), ", ") * "]")
    PlannerObjective(m.n_endo, vfn, jfn, hfn, !isempty(hes))
end

planner_objective(m::Model) = get!(() -> _compile_objective(m), m.cache, :planner_objective)::PlannerObjective

function _po_call(f, y, p)
    try
        Base.invokelatest(f, Vector{Float64}(y), Vector{Float64}(p))
    catch e
        e isa DomainError || rethrow()
        nothing
    end
end
objective_value(po::PlannerObjective, y, p) = (v = _po_call(po.value_fn, y, p); v === nothing ? NaN : Float64(v))
function objective_gradient(po::PlannerObjective, y, p)
    g = zeros(po.n)
    ent = _po_call(po.jac_fn, y, p)
    ent === nothing && return fill(NaN, po.n)
    for (c, v) in ent
        g[c] += v
    end
    g
end
function objective_hessian(po::PlannerObjective, y, p)
    H = zeros(po.n, po.n)
    ent = _po_call(po.hess_fn, y, p)
    ent === nothing && return fill(NaN, po.n, po.n)
    for (i, j, v) in ent
        H[i, j] += v
        i != j && (H[j, i] += v)
    end
    H
end

"Second derivatives of the objective, refused when the preprocessor only computed the first (order-1 `stoch_simul` in the file)."
function _objective_hessian_checked(m::Model, po::PlannerObjective, y, p)
    if !po.has_hessian
        # an objective with a constant gradient has an empty Hessian: compare gradients at two points
        g0 = objective_gradient(po, y, p)
        g1 = objective_gradient(po, y .* 1.01 .+ 0.01, p)
        isapprox(g0, g1; rtol=1e-12, atol=1e-12) ||
            throw(ModelError("the preprocessor computed the planner objective's derivatives to order 1 only (the file asks for order 1): reload the file with stoch_simul(order=2) or an order-2 evaluate_planner_objective"))
    end
    objective_hessian(po, y, p)
end

# ---------------------------------------------------------------------- evaluate_planner_objective
"`A * kron(B, C)` (A_times_B_kronecker_C.m)."
_akc(A, B, C) = A * kron(B, C)

"`options_.lyapunov_fixed_point_tol`, `lyapunov_complex_threshold` defaults and the persistent state of lyapunov_symm.m (method 3)."
mutable struct _LyapState
    X::Union{Nothing,Matrix{Float64}}
    method::Int
end

"""
lyapunov_symm.m with method 3 (fixed-point iterations, as evaluate_planner_objective calls it). MATLAB keeps the last solution in
a persistent variable and starts the next call (of the same size) from it with at most 300 iterations (2000 for the first call);
here it is kept per model. When the iterations do not converge Dynare switches to the Schur method for the rest of the session.
"""
function _lyapunov_fixed_point(m::Model, a::AbstractMatrix, b::AbstractMatrix; tol=1e-10, qz_criterium=QZ_CRITERIUM, complex_threshold=1e-15)
    st = get!(() -> _LyapState(nothing, 3), m.cache, :lyap_fp)::_LyapState
    if st.method == 3
        if st.X === nothing || size(st.X, 1) != size(b, 1)
            X, max_it = Matrix{Float64}(b), 2000
        else
            X, max_it = st.X, 300
        end
        it, evol = 0, 100.0
        at = transpose(a)
        while evol > tol && it < max_it
            Xold = X
            X = a * X * at + b
            evol = maximum(sum(abs, X - Xold; dims=1))
            it += 1
        end
        st.X = X
        if it >= max_it
            @info "lyapunov_symm:: convergence not achieved in solution of Lyapunov equation after $it iterations, switching method from 3 to 0"
            st.method = 0
        else
            return X
        end
    end
    lyapunov_symm(a, b; qz_criterium=qz_criterium, complex_threshold=complex_threshold)[1]
end

"`oo_.planner_objective_value`. `conditional` is a number, or under Ramsey policy the two values with the multipliers initially 0 and at their steady state."
struct PlannerObjectiveValue
    unconditional::Float64
    conditional::Union{Float64,@NamedTuple{zero_initial_multiplier::Float64, steady_initial_multiplier::Float64}}
end
function Base.getproperty(v::PlannerObjectiveValue, s::Symbol)
    c = getfield(v, :conditional)
    s === :zero_initial_multiplier && return c isa Float64 ? throw(ArgumentError("no Ramsey welfare values: not a Ramsey model")) : c.zero_initial_multiplier
    s === :steady_initial_multiplier && return c isa Float64 ? throw(ArgumentError("no Ramsey welfare values: not a Ramsey model")) : c.steady_initial_multiplier
    getfield(v, s)
end
Base.propertynames(::PlannerObjectiveValue) = (:unconditional, :conditional, :zero_initial_multiplier, :steady_initial_multiplier)

function format_planner_objective(v::PlannerObjectiveValue; ramsey::Bool, discretionary::Bool=false)
    if ramsey
        return @sprintf("\nApproximated value of unconditional welfare:  %10.8f\n\nApproximated value of conditional welfare:\n    - with initial Lagrange multipliers set to 0: %10.8f\n    - with initial Lagrange multipliers set to steady state: %10.8f\n",
                        v.unconditional, v.zero_initial_multiplier, v.steady_initial_multiplier)
    end
    kind = discretionary ? " with discretionary policy" : ""
    @sprintf("\nApproximated value of unconditional welfare%s:  %10.8f\n\nApproximated value of conditional welfare%s: %10.8f\n", kind, v.unconditional, kind, v.conditional)
end

"""
M_.endo_histval as the histval block builds it (unset variables are 0), or nothing without histval. Built from the block alone: a
Ramsey model's steady state cannot be computed where the block stands (the planner is declared after it).
"""
function _op_histval_vector(m::Model)
    sts = _statements(m, "histval")
    isempty(sts) && return nothing
    ws = PFWorkspace(copy(m.params), zeros(m.n_endo), zeros(m.n_exo), nothing, nothing, Tuple{Int,Int,Int,Float64,String}[],
                     Dict{String,Any}[], Dict{String,Any}[], nothing, nothing, false, false, false)
    _pf_histval!(m, ws, sts[end])
    ws.endo_histval
end

"get_initial_state of evaluate_planner_objective.m: `(yhat_L_SS, yhat_L_0, u)`."
function _initial_state(m::Model, ys, order_var, nstatic, nspred; histval=nothing, u=nothing)
    mult = Int[Int(a["endo_index"]) for a in get_or(m.raw, "aux_vars", []) if get(a, "type", -1) == 6]
    nonmult = trues(m.n_endo); nonmult[mult] .= false
    y_ss = Vector{Float64}(ys)
    y_0 = zeros(m.n_endo)
    hv = histval === nothing ? _op_histval_vector(m) : Vector{Float64}(histval)
    if hv !== nothing
        y_ss[nonmult] = hv[nonmult]
        y_0[nonmult] = hv[nonmult]
    else
        y_0[nonmult] = ys[nonmult]
    end
    sv = order_var[nstatic+1:nstatic+nspred]
    yhat_0 = y_0[sv] - ys[sv]
    yhat_ss = y_ss[sv] - ys[sv]
    if u === nothing
        u = exo_steady(m)
        if !isempty(m.det_shocks)
            # M_.det_shocks holds one entry per exogenous variable and period
            ent = Tuple{Int,Int,Float64}[(e[1], p, e[4]) for e in m.det_shocks for p in e[2]:e[3]]
            periods = [e[2] for e in ent]
            any(==(0), periods) && println("evaluate_planner_objective: M_.det_shocks contains values for the predetermined t=0 period. Dynare will ignore them. Use histval to set the value of lagged innovations.")
            any(>(1), periods) && println("evaluate_planner_objective: Shock values for periods not contained in the initial information set (t=1) have been provided. Note that they will be ignored.")
            for (ix, p, v) in ent
                p == 1 && (u[ix] = v)
            end
        end
    end
    yhat_ss, yhat_0, Vector{Float64}(u)
end

_hessian_eq_zero(m::Model) = m.fns.dyn_g2 === nothing || isempty(m.fns.g2_idx) || m.max_deriv_order < 2 && false

"""
    evaluate_planner_objective(m, dr=nothing; order=nothing, u=nothing, histval=nothing, ramsey=nothing, discretionary=nothing,
                               params=nothing, Sigma_e=nothing, replicate_dynare_bugs=false, print_results=false) -> PlannerObjectiveValue

Welfare as `evaluate_planner_objective.m` approximates it, from decision rules `dr` (the Dict of `k_order` / `as_dr(first_order(m))`
or `discretionary_policy(m).dr`; solved at `order`, default the .mod file's `stoch_simul` order, else 2, when omitted). Orders
1 and 2 (Dynare needs its `k_order_welfare` MEX file beyond). `u` is the shock vector of the first period of the conditional
welfare: Dynare takes `oo_.exo_simul(1,:)`, which is the first *simulated* shock after a `stoch_simul` with `periods > 0`, else
the exogenous steady state (the default here), or the deterministic shocks of period 1. `histval` (default the .mod file's
`histval` block) sets the initial state.

Unconditional welfare is `E[U]/(1-beta)`; conditional welfare starts from the deterministic steady state (or `histval`); under
Ramsey policy it is reported with the multipliers initially at their steady state and at zero.

Dynare 7.2 defect, corrected by default (see the head of this file); `replicate_dynare_bugs=true` reproduces Dynare's value.
"""
function evaluate_planner_objective(m::Model, dr=nothing; order=nothing, u=nothing, histval=nothing, ramsey=nothing, discretionary=nothing,
                                    params=nothing, Sigma_e=nothing, replicate_dynare_bugs::Bool=false, print_results::Bool=false)
    ramsey = ramsey === nothing ? is_ramsey(m) : ramsey
    discretionary = discretionary === nothing ? is_discretionary(m) : discretionary
    so = stoch_options(m)
    dr isa FirstOrderSolution && (dr = as_dr(dr))
    if u === nothing && isempty(m.det_shocks) && Int(get(so, "periods", 0)) > 0
        warn_once(m, "evaluate_planner_objective: the .mod file simulates (periods > 0), so Dynare's conditional welfare starts from the first simulated shock (oo_.exo_simul(1,:)); JDyn uses the exogenous steady state unless u= is given (the dynare() workspace commands pass it)")
    end
    order === nothing && (order = dr !== nothing && haskey(dr, "order") ? Int(dr["order"]) : Int(get(so, "order", 2)))
    if dr === nothing
        dr = discretionary ? discretionary_policy(m; warm_start=false).dr :
             order == 1 ? as_dr(first_order(m; loglinear=false)) : k_order(m; order=order, loglinear=false)
    end
    params = Vector{Float64}(params === nothing ? (haskey(dr, "params") && dr["params"] !== nothing ? dr["params"] : m.params) : params)
    Sigma = Matrix{Float64}(Sigma_e === nothing ? m.Sigma_e : Sigma_e)
    beta = planner_discount(m, params)
    beta >= 1 && println("evaluate_planner_objective: the planner discount factor is not strictly smaller than 1. Unconditional welfare will not be finite.")
    po = planner_objective(m)
    ys = Vector{Float64}(dr["ys"])
    get(dr, "loglinear", false) == true && throw(ModelError("evaluate_planner_objective with loglinear rules is not supported"))
    _, _, cnt = dr_ordering(m.lead_lag_incidence)
    nstatic, nspred = cnt.nstatic, cnt.npred + cnt.nboth
    ov = Vector{Int}(dr["order_var"])
    n = m.n_endo
    ghx, ghu = Matrix{Float64}(dr["ghx"]), Matrix{Float64}(dr["ghu"])
    nx = size(ghu, 2)
    gy = zeros(n, nspred); gu = zeros(n, nx)
    gy[ov, :] = ghx; gu[ov, :] = ghu
    Gy = ghx[nstatic+1:nstatic+nspred, :]
    Gu = ghu[nstatic+1:nstatic+nspred, :]
    U = objective_value(po, ys, params)
    Uy = reshape(objective_gradient(po, ys, params), 1, :)
    I_ = Matrix{Float64}(I, nspred, nspred)
    lqhessian = _hessian_eq_zero_dynare(m)
    local EW, W_SS, W_0
    if order == 1 && !discretionary
        EW = U / (1 - beta)
        Wbar = U / (1 - beta)
        Wy = (Uy * gy) / (I_ - beta * Gy)
        Wu = Uy * gu + beta * Wy * Gu
        yss, y0, uu = _initial_state(m, ys, ov, nstatic, nspred; histval=histval, u=u)
        W_SS = Wbar + (Wy * yss)[1] + (Wu * uu)[1]
        W_0 = Wbar + (Wy * y0)[1] + (Wu * uu)[1]
    elseif order == 2 || discretionary
        lq = discretionary || lqhessian
        Uyy = reshape(_objective_hessian_checked(m, po, ys, params), 1, :)
        Uyygygy, Uyygugu = _akc(Uyy, gy, gy), _akc(Uyy, gu, gu)
        # Dynare 7.2: kron(gu, gy) (u-major) where the layout of Wyu*kron(yhat, u) is y-major: kron(gy, gu) is the consistent one
        Uyygugy = replicate_dynare_bugs ? _akc(Uyy, gu, gy) : _akc(Uyy, gy, gu)
        var_list = [m.endo_names[i] for i in ov[nstatic+1:nstatic+nspred]]
        mom_order = lq ? 1 : 2
        if !lq
            for k in ("ghxx", "ghxu", "ghuu", "ghs2")
                haskey(dr, k) || throw(OptimalPolicyError("evaluate_planner_objective at order 2 needs second-order rules"))
            end
            get(so, "pruning", false) == true && println("evaluate_planner_objective: pruning option is not supported and will be ignored")
        end
        if nspred > 0
            mo = Logging.with_logger(Logging.NullLogger()) do
                theoretical_moments(m, dr; var_list=var_list, Sigma_e=Sigma, order=mom_order, pruning=false, nodecomposition=true,
                                    conditional_variance_decomposition=Int[], ar=0, nocorr=true, hp_filter=0)
            end
            Ey = Vector{Float64}(mo.mean); Vy = Matrix{Float64}(mo.var)
        else
            Ey, Vy = zeros(0), zeros(0, 0)
        end
        if !lq && (any(isnan, Ey) || any(isnan, Vy))
            println("evaluate_planner_objective: encountered NaN moments in the endogenous variables; they are replaced by a large number (options_.huge_number)")
        end
        Ey = replace(Ey, NaN => 1e7); Vy = replace(Vy, NaN => 1e7)
        Euu = vec(Sigma)
        Wbar = U / (1 - beta)
        Wy = (Uy * gy) / (I_ - beta * Gy)
        Wu = Uy * gu + beta * Wy * Gu
        qz = qz_criterium(m)
        local Wyy, Wuu, Wss, Wyu
        if !lq
            gyy = zeros(n, nspred^2); gyy[ov, :] = dr["ghxx"]
            gyu = zeros(n, nspred * nx); gyu[ov, :] = dr["ghxu"]
            guu = zeros(n, nx^2); guu[ov, :] = dr["ghuu"]
            gss = zeros(n); gss[ov] = vec(dr["ghs2"])
            Gyy = Matrix{Float64}(dr["ghxx"])[nstatic+1:nstatic+nspred, :]
            Gyu = Matrix{Float64}(dr["ghxu"])[nstatic+1:nstatic+nspred, :]
            Guu = Matrix{Float64}(dr["ghuu"])[nstatic+1:nstatic+nspred, :]
            Gss = vec(dr["ghs2"])[nstatic+1:nstatic+nspred]
            Eyhat = Ey - ys[ov[nstatic+1:nstatic+nspred]]
            EU = U + (Uy*gy*Eyhat)[1] + 0.5 * (((Uyygygy + Uy * gyy) * vec(Vy))[1] + ((Uyygugu + Uy * guu) * Euu)[1] + (Uy*gss)[1])
            rhs = reshape(Uyygygy + Uy * gyy + beta * Wy * Gyy, nspred, nspred)
            Wyy = reshape(_lyapunov_fixed_point(m, sqrt(beta) * transpose(Gy), rhs; qz_criterium=qz), 1, :)
            Wuu = Uyygugu + Uy * guu + beta * (_akc(Wyy, Gu, Gu) + Wy * Guu)
            Wss = (Uy * gss + beta * (Wy * Gss + Wuu * Euu)) / (1 - beta)
            Wyu = Uyygugy + Uy * gyu + beta * ((replicate_dynare_bugs ? _akc(Wyy, Gu, Gy) : _akc(Wyy, Gy, Gu)) + Wy * Gyu)
        else
            EU = U + 0.5 * (Uyygugu * Euu)[1]
            if get(m.info, "maximum_endo_lag", 0) > 0 && nspred > 0
                Eyhat = Ey - ys[ov[nstatic+1:nstatic+nspred]]
                EU = EU + (Uy*gy*Eyhat)[1] + 0.5 * (Uyygygy * vec(Vy))[1]
            end
            Wyy = nspred > 0 ? reshape(_lyapunov_fixed_point(m, sqrt(beta) * transpose(Gy), reshape(Uyygygy, nspred, nspred); qz_criterium=qz), 1, :) : zeros(1, 0)
            Wuu = Uyygugu + beta * _akc(Wyy, Gu, Gu)
            Wss = beta * Wuu * Euu / (1 - beta)
            Wyu = Uyygugy + beta * (replicate_dynare_bugs ? _akc(Wyy, Gu, Gy) : _akc(Wyy, Gy, Gu))
        end
        EW = Float64(EU isa AbstractArray ? EU[1] : EU) / (1 - beta)
        yss, y0, uu = _initial_state(m, ys, ov, nstatic, nspred; histval=histval, u=u)
        Wq(yh) = (Wbar + (Wy * yh)[1] + (Wu * uu)[1] + (_akc(Wyu, yh, uu) + 0.5 * (Wss + _akc(Wyy, yh, yh) + _akc(Wuu, uu, uu)))[1])
        Wq(yh::Vector) = begin
            yv = reshape(yh, :, 1); uv = reshape(uu, :, 1)
            Wbar + (Wy * yv)[1] + (Wu * uv)[1] + (_akc(Wyu, yv, uv))[1] + 0.5 * ((Wss isa AbstractArray ? Wss[1] : Wss) + (_akc(Wyy, yv, yv))[1] + (_akc(Wuu, uv, uv))[1])
        end
        W_SS, W_0 = Wq(yss), Wq(y0)
    else
        throw(ModelError("evaluate_planner_objective beyond order 2 needs Dynare's k_order_welfare (a MEX file JDyn does not have)"))
    end
    out = ramsey ? PlannerObjectiveValue(EW, (zero_initial_multiplier=Float64(W_0), steady_initial_multiplier=Float64(W_SS))) :
                   PlannerObjectiveValue(EW, Float64(W_SS))
    print_results && print(format_planner_objective(out; ramsey=ramsey, discretionary=discretionary))
    out
end

"M_.hessian_eq_zero: the dynamic model has no second derivatives (a linear model)."
_hessian_eq_zero_dynare(m::Model) = get!(m.cache, :hessian_eq_zero) do
    m.fns.dyn_g2 === nothing ? false : isempty(m.fns.g2_idx)
end::Bool

"""
`evaluate_planner_objective` after a perfect-foresight simulation of a Ramsey model: the planner objective summed backwards along the
path (`endo` is `endo_nbr x T`, as `oo_.endo_simul`), from `U(T-lead)/(1-beta)`. `conditional` is the welfare of the path.
"""
function evaluate_planner_objective_perfect_foresight(m::Model, endo::AbstractMatrix; params=nothing)
    params = Vector{Float64}(params === nothing ? m.params : params)
    beta = planner_discount(m, params)
    beta >= 1 && println("evaluate_planner_objective: the planner discount factor is not strictly smaller than 1. Unconditional welfare will not be finite.")
    po = planner_objective(m)
    lead, lag = Int(m.raw["model_info"]["maximum_lead"]), Int(m.raw["model_info"]["maximum_lag"])
    T = size(endo, 2)
    EW = objective_value(po, endo[:, T-lead], params) / (1 - beta)
    W = EW
    for t in T-lead:-1:1+lag
        W = objective_value(po, endo[:, t], params) + beta * W
    end
    PlannerObjectiveValue(EW, W)
end

# ---------------------------------------------------------------------- discretionary policy
"""
Outcome of `discretionary_policy`: `dr` (Dynare's `oo_.dr`: `ys`, `ghx`, `ghu`, `order_var`, `state_var`, order 1), the transition `H`
and impact `G` in declaration order (y_t = H y_{t-1} + G e_t), the welfare and the number of iterations.
"""
struct DiscretionaryResult
    dr::Dict{String,Any}
    H::Matrix{Float64}
    G::Matrix{Float64}
    instruments::Vector{String}
    welfare::Union{Nothing,PlannerObjectiveValue}
    iterations::Int
end

function _sylvester_doubling(d, g, h, tol, maxit)
    v = copy(d)
    for _ in 1:maxit
        vadd = g * v * h
        v = v + vadd
        opnorm(vadd, 1) <= tol * opnorm(v, 1) && break
        g = g * g
        h = h * h
    end
    v
end

"Fallback of discretionary_policy_engine: `v = d + g v h` solved directly."
function _sylvester_direct(d, g, h)
    n, mm = size(d)
    K = Matrix{Float64}(I, n * mm, n * mm) - kron(transpose(h), g)
    reshape(K \ vec(d), n, mm)
end

"""
discretionary_policy_engine.m (Dennis 2007): iterates on the policy rule and value function of the linear-quadratic problem until the
rule converges. `instr_id` is 1-based. Returns `(H, G, retcode, iterations)`.
"""
function discretionary_policy_engine(AAlag, AA0, AAlead, BB, bigw, instr_id, beta, maxit, tol, qz_criterium, H00=nothing)
    eq_nbr_, endo_nbr = size(AAlag)
    exo_nbr = size(BB, 2)
    y = setdiff(1:endo_nbr, instr_id)
    instr_nbr = length(instr_id)
    A0, A1, A2 = AA0[:, y], -AAlag[:, y], -AAlead[:, y]
    A3, A4, A5 = -AA0[:, instr_id], -AAlead[:, instr_id], -BB
    W, Q = bigw[y, y], bigw[instr_id, instr_id]
    A6 = -AAlag[:, instr_id]
    aux = vec(any(A6 .!= 0; dims=1))
    naux = count(aux)
    ny, mm = eq_nbr_, eq_nbr_ + naux
    A00 = zeros(mm, mm); A00[1:ny, 1:ny] = A0; A00[ny+1:end, ny+1:end] = Matrix{Float64}(I, naux, naux)
    A11 = zeros(mm, mm); A11[1:ny, 1:ny] = A1; A11[1:ny, ny+1:end] = A6[:, aux]
    A22 = zeros(mm, mm); A22[1:ny, 1:ny] = A2
    A33 = zeros(mm, instr_nbr); A33[1:ny, :] = A3; A33[ny+1:end, aux] = Matrix{Float64}(I, naux, naux)
    A44 = zeros(mm, instr_nbr); A44[1:ny, :] = A4
    A55 = zeros(mm, exo_nbr); A55[1:ny, :] = A5
    WW = zeros(mm, mm); WW[1:ny, 1:ny] = W
    endo_augm_id = setdiff(1:endo_nbr+naux, instr_id)
    H0 = zeros(endo_nbr + naux, endo_nbr + naux)
    H00 === nothing || (H0[1:endo_nbr, 1:endo_nbr] = H00)
    H10 = H0[endo_augm_id, endo_augm_id]
    F10 = H0[instr_id, endo_augm_id]
    it = 0
    H1, F1 = H10, F10
    local A3DPD, Dinv, rcode
    while true
        it += 1
        P = _sylvester_doubling(WW + beta * transpose(F1) * Q * F1, beta * transpose(H1), H1, tol, maxit)
        if !all(isfinite, P)
            P = _sylvester_direct(WW + beta * transpose(F1) * Q * F1, beta * transpose(H1), H1)
            all(isfinite, P) || return nothing, nothing, 2, it
        end
        D = A00 - A22 * H1 - A44 * F1
        Dinv = inv(D)
        A3DPD = transpose(A33) * transpose(Dinv) * P * Dinv
        F1 = -((Q + A3DPD * A33) \ (A3DPD * A11))
        H1 = Dinv * (A11 + A33 * F1)
        NQ = maximum(abs, vcat(H1, F1) - vcat(H10, F10))
        rcode = isnan(NQ) ? 3 : it > maxit ? 2 : NQ < tol ? 1 : 0
        rcode != 0 && break
        H10, F10 = H1, F1
    end
    rcode == 3 && return nothing, nothing, 63, it
    rcode == 2 && return nothing, nothing, 61, it
    maximum(abs, eigvals(H1)) > qz_criterium && return nothing, nothing, 62, it
    F2 = -((Q + A3DPD * A33) \ (A3DPD * A55))
    H2 = Dinv * (A55 + A33 * F2)
    H = zeros(endo_nbr + naux, endo_nbr + naux)
    G = zeros(endo_nbr + naux, exo_nbr)
    H[endo_augm_id, endo_augm_id] = H1
    H[instr_id, endo_augm_id] = F1
    G[endo_augm_id, :] = H2
    G[instr_id, :] = F2
    if naux > 0                                  # lagged instruments: their auxiliary columns
        H[:, instr_id[aux]] = H[:, size(H, 2)-naux+1:end]
    end
    H[1:endo_nbr, 1:endo_nbr], G[1:endo_nbr, :], 0, it
end

const _DP_MESSAGES = Dict(61 => "Discretionary policy: maximum number of iterations has been reached. Procedure failed.",
    62 => "Discretionary policy: some eigenvalues greater than options_.qz_criterium. Model potentially unstable.",
    63 => "Discretionary policy: NaN elements are present in the solution. Procedure failed.",
    64 => "discretionary_policy: the derivatives of the objective function contain NaN.",
    65 => "discretionary_policy: the model must be written in deviation form and not have constant terms or an analytical steady state needs to be provided.",
    66 => "discretionary_policy: the objective function must have zero first order derivatives.",
    2 => "discretionary_policy: the Sylvester equation of the value function could not be solved.")

"""
    discretionary_policy(m; params=nothing, tol, maxit, qz_criterium, H0, welfare=true, u=nothing, warm_start=true) -> DiscretionaryResult

`discretionary_policy` (Dynare 7.2's discretionary_policy.m / discretionary_policy_1.m): the optimal policy under discretion of a
linear-quadratic problem (Dennis 2007), with the `instruments` and `planner_objective` of the .mod file. `tol`
(`discretionary_tol`, default 1e-7) and `maxit` (`dp.maxit`, 3000) default to the file's options.

As in Dynare the objective's first derivatives and Hessian are taken at zero and the model's at the steady state given by a
steady-state file or block, else at zero (the model must then be written in deviations). Like Dynare's persistent variable, the
previous solution of the same model starts the iterations (`warm_start=false`, or `H0`, changes that). `welfare` adds
`evaluate_planner_objective` (`u`: see there).
"""
function discretionary_policy(m::Model; params=nothing, tol=nothing, maxit=nothing, qz_criterium=nothing, H0=nothing, welfare::Bool=true, u=nothing,
                              warm_start::Bool=true, Sigma_e=nothing)
    opts = get(m.options, "discretionary_policy", Dict{String,Any}())
    so = stoch_options(m)
    (get(opts, "loglinear", false) == true || get(so, "loglinear", false) == true) &&
        throw(DiscretionaryPolicyError("discretionary_policy is not compatible with `loglinear` option set to 1"))
    Int(get(opts, "order", 1)) == 1 || throw(DiscretionaryPolicyError("discretionary_policy: only linear-quadratic problems can be solved"))
    tol = Float64(tol === nothing ? get(opts, "discretionary_tol", get(so, "discretionary_tol", 1e-7)) : tol)
    maxit = Int(maxit === nothing ? get(opts, "dp.maxit", get(so, "dp.maxit", 3000)) : maxit)
    qz = JDyn.qz_criterium(m, qz_criterium)
    params = Vector{Float64}(params === nothing ? m.params : params)
    n, nx = m.n_endo, m.n_exo
    neq = eq_nbr(m)
    instr_nbr = n - neq
    instr_nbr == 0 && throw(DiscretionaryPolicyError("discretionary_policy:: There are no available instruments, because the model has as many equations as variables."))
    names = _instrument_names(m)
    length(names) < instr_nbr && throw(DiscretionaryPolicyError("discretionary_policy:: There are fewer declared instruments than omitted equations."))
    length(names) > instr_nbr && throw(DiscretionaryPolicyError("discretionary_policy:: There are more declared instruments than omitted equations."))
    for v in names
        haskey(m.ie, v) || throw(DiscretionaryPolicyError("discretionary_policy:: instrument $v not found"))
    end
    instr_id = Int[m.ie[v] for v in names]
    beta = planner_discount(m, params)
    x = exo_steady(m)
    if _has_steady_state_file(m)
        ys0 = _initval_vector(m)
        ys, params = try
            _evaluate_steady_state_file(m, ys0, x, params)
        catch e
            e isa SteadyStateError || rethrow()
            throw(DiscretionaryPolicyError("steady-state file failed: $(e.msg)", 19))
        end
        any(a -> true, get_or(m.raw, "aux_vars", [])) && (ys = fill_auxiliary(m, ys, x, params))
        if !(get(get(m.options, "steady", Dict()), "steadystate.nocheck", false) == true)
            res = static_resid(m, ys, x, params)
            (maximum(abs, res) <= DYNARE_TOLF && !any(isnan, res)) ||
                throw(DiscretionaryPolicyError("the steady state given by the steady-state file does not solve the static model (Dynare's error 19)", 19))
        end
    else
        ys = zeros(n)
    end
    po = planner_objective(m)
    y0 = zeros(n)
    Uy = objective_gradient(po, y0, params)
    any(isnan, Uy) && throw(DiscretionaryPolicyError(_DP_MESSAGES[64], 64))
    any(!=(0), Uy) && throw(DiscretionaryPolicyError(_DP_MESSAGES[66], 66))
    Wm = _objective_hessian_checked(m, po, y0, params)
    tolf = Float64(get(get(m.options, "steady", Dict()), "solve_tolf", DYNARE_TOLF))
    resid = dynamic_resid(m, repeat(ys, 3), zeros(nx), params, ys)[1:neq]
    maximum(abs, resid) <= tolf || throw(DiscretionaryPolicyError(_DP_MESSAGES[65], 65))
    J = dynamic_jac(m, repeat(ys, 3), zeros(nx), params, ys)[1:neq, :]
    Hold = H0 === nothing && warm_start ? get(m.cache, :dp_Hold, nothing) : H0
    H, G, info, it = discretionary_policy_engine(J[:, 1:n], J[:, n+1:2n], J[:, 2n+1:3n], J[:, 3n+1:end], Wm, instr_id, beta, maxit, tol, qz, Hold)
    info != 0 && throw(DiscretionaryPolicyError(get(_DP_MESSAGES, info, "discretionary_policy failed (info $info)"), info))
    m.cache[:dp_Hold] = H
    order_var, state_var, _ = dr_ordering(m.lead_lag_incidence)
    ghx = get(m.info, "maximum_endo_lag", 0) > 0 ? H[order_var, state_var] : zeros(n, 0)
    dr = Dict{String,Any}("ys" => ys, "order_var" => order_var, "state_var" => state_var, "ghx" => ghx, "ghu" => G[order_var, :],
                          "order" => 1, "params" => params)
    w = welfare ? evaluate_planner_objective(m, dr; order=1, u=u, ramsey=false, discretionary=true, params=params, Sigma_e=Sigma_e) : nothing
    DiscretionaryResult(dr, H, G, names, w, it)
end

# ---------------------------------------------------------------------- optimal simple rules
"""
`oo_.osr`: optimal parameters, objective value at the optimum, optimiser information; `dr` holds the rules at the optimum and
`error_indicator` is 0 when the model could be solved there.
"""
struct OSRResult
    optim_params::Vector{Pair{String,Float64}}      # in the order of osr_params
    objective_function::Float64
    x::Vector{Float64}
    initial_objective::Float64
    exitflag::Int
    message::String
    iterations::Int
    func_count::Int
    dr::Union{Nothing,Dict{String,Any}}
    error_indicator::Int
end
Base.getindex(r::OSRResult, name::AbstractString) = last(r.optim_params[findfirst(p -> first(p) == name, r.optim_params)])

# resol's error codes that osr.objective penalises with info(2) (when finite)
const _OSR_INFO2 = Set([3, 4, 5, 6, 19, 20, 21, 23, 26, 81, 84, 85])

"M_.osr.variable_weights and M_.osr.variable_indices (1-based), as the preprocessor writes them: a covariance weight is set at (name1, name2) only."
function _osr_weights(m::Model)
    W = zeros(m.n_endo, m.n_endo)
    idx = Int[]
    # each optim_weights block starts afresh (M_.osr.variable_weights is reset): the one in force at the osr statement, else the last
    sts = haskey(m.cache, :osr_weights_stmt) ? [m.cache[:osr_weights_stmt]] : last(_statements(m, "optim_weights"), 1)
    for st in sts
        for d in st["weights"]
            v = Float64(_value(m, d["value"]))
            if haskey(d, "name2")
                i, j = m.ie[d["name1"]], m.ie[d["name2"]]
                W[i, j] = v
                push!(idx, i, j)
            else
                i = m.ie[d["name"]]
                W[i, i] = v
                push!(idx, i)
            end
        end
    end
    W, idx
end

"osr.get_variance_of_endogenous_variables: the variance of the variables in rows `i_var` of the decision rules (DR order), Inf where they are not stationary."
function variance_of_endogenous_variables(m::Model, dr, i_var; Sigma_e=nothing, qz_criterium=nothing, schur_vec_tol=1e-11)
    Sigma = Matrix{Float64}(Sigma_e === nothing ? m.Sigma_e : Sigma_e)
    ghx = Matrix{Float64}(dr["ghx"])[i_var, :]
    ghu = Matrix{Float64}(dr["ghu"])[i_var, :]
    _, _, cnt = dr_ordering(m.lead_lag_incidence)
    ns, nspred = cnt.nstatic, cnt.npred + cnt.nboth
    A = Matrix{Float64}(dr["ghx"])[ns+1:ns+nspred, :]
    B = Matrix{Float64}(dr["ghu"])[ns+1:ns+nspred, :]
    vx, u = lyapunov_symm(A, B * Sigma * transpose(B); qz_criterium=JDyn.qz_criterium(m, qz_criterium))
    n = length(i_var)
    i_stat = size(u, 2) > 0 ? findall(i -> any(abs.((ghx*u)[i, :]) .< schur_vec_tol), 1:n) : collect(1:n)
    out = fill(Inf, n, n)
    out[i_stat, i_stat] = ghx[i_stat, :] * vx * transpose(ghx[i_stat, :]) + ghu[i_stat, :] * Sigma * transpose(ghu[i_stat, :])
    out
end

"`resol.m`: steady state and decision rules at `params`. Returns `(dr, info)`; `info = [code, penalty]`."
function _resol_dr(m::Model, params, order::Int)
    try
        with_logger(NullLogger()) do
            ys, p = steady_state(m; params=params)
            dr = if order == 1
                d = as_dr(first_order(m; ys=ys, params=p, loglinear=false)); d["params"] = p; d
            else
                k_order(m; order=order, ys=ys, params=p, loglinear=false)
            end
            return dr, [0.0]
        end
    catch e
        e isa BlanchardKahnError && return nothing, [3.0, e.penalty]
        e isa OptimalPolicyError && return nothing, [Float64(something(e.info, 20)), NaN]
        (e isa SteadyStateError || e isa KOrderError || e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException ||
         e isa DomainError || e isa LoglinearError) && return nothing, [20.0, NaN]
        rethrow()
    end
end

"The set-up of osr.run and the objective osr.objective."
struct OSRProblem
    m::Model
    param_names::Vector{String}
    i_params::Vector{Int}
    order::Int
    use_planner::Bool
    i_var::Vector{Int}
    W::Matrix{Float64}
    params0::Vector{Float64}
    qz::Float64
    replicate_dynare_bugs::Bool
end

function OSRProblem(m::Model, param_names, order::Int; replicate_dynare_bugs=false)
    isempty(param_names) && throw(OSRError("OSR: no parameters to optimise (osr_params)"))
    for p in param_names
        haskey(m.ip, p) || throw(OSRError("OSR: $p is not a parameter of the model"))
    end
    kinds = [st["statementName"] for st in m.statements]
    iosr = findfirst(==("osr"), kinds)
    ipo = findfirst(==("planner_objective"), kinds)
    # the preprocessor's planner_discount_declared: a planner_objective statement before osr
    use_planner = ipo !== nothing && (iosr === nothing || ipo < iosr)
    i_var, W = Int[], zeros(0, 0)
    if !use_planner
        order == 1 || throw(OSRError("OSR: OSR without planner_objective only supports order=1."))
        Wf, iv = _osr_weights(m)
        any(isinf, Wf) && throw(OSRError("OSR: At least one of the optim_weights is infinite."))
        isempty(iv) && throw(OSRError("OSR: there is neither a planner_objective nor optim_weights"))
        i_var = sort(unique(iv))
        W = Wf[i_var, i_var]
    end
    get(m.info, "maximum_lead", 1) == 0 && throw(OSRError("OSR: Backward or static model: no point in using OSR"))
    p0 = copy(m.params)
    any(isnan, p0[[m.ip[p] for p in param_names]]) && throw(OSRError("OSR: At least one of the initial parameter values for osr_params is NaN"))
    OSRProblem(m, String.(param_names), Int[m.ip[p] for p in param_names], order, use_planner, i_var, W, p0, qz_criterium(m), replicate_dynare_bugs)
end

"`(loss, info)` as osr.objective returns them (loss 1e8 and info[4] the penalty on failure)."
function (pr::OSRProblem)(x::AbstractVector)
    m = pr.m
    p = copy(pr.params0)
    p[pr.i_params] = x
    dr, info = _resol_dr(m, p, pr.order)
    if info[1] != 0
        i2 = length(info) > 1 ? info[2] : NaN
        pen = (Int(info[1]) in _OSR_INFO2 && isfinite(i2)) ? i2 : 0.1
        return 1e8, [info[1], i2, 0.0, pen]
    end
    p = dr["params"]
    loss = if !pr.use_planner
        inv = zeros(Int, m.n_endo); inv[Int.(dr["order_var"])] .= 1:m.n_endo
        vx = variance_of_endogenous_variables(m, dr, inv[pr.i_var]; qz_criterium=pr.qz)
        nz = pr.W .!= 0                        # sparse weights: 0 * Inf does not occur
        sum(pr.W[nz] .* vx[nz])
    else
        -evaluate_planner_objective(m, dr; order=pr.order, ramsey=false, discretionary=false, params=p,
                                    replicate_dynare_bugs=pr.replicate_dynare_bugs).unconditional
    end
    isinf(loss) && return 1e8, [174.0, 0.0, 0.0, 0.1]
    isnan(loss) && return 1e8, [175.0, 0.0, 0.0, 0.1]
    loss, [0.0]
end

"osr_params_bounds as a matrix of (lower, upper) rows in the order of `names` (nothing when the file has none)."
function _osr_bounds(m::Model, names)
    sts = _statements(m, "osr_params_bounds")
    isempty(sts) && return nothing
    B = hcat(fill(-Inf, length(names)), fill(Inf, length(names)))
    for st in sts, d in st["bounds"]
        i = findfirst(==(d["parameter"]), names)
        i === nothing && throw(OSRError("OSR: osr_params_bounds names $(d["parameter"]), which is not in osr_params"))
        B[i, :] = [Float64(_value(m, d["bounds"][1])), Float64(_value(m, d["bounds"][2]))]
    end
    B
end

function _osr_param_names(m::Model)
    names = String[]
    for st in _statements(m, "osr_params")
        get(st, "overwrite", false) == true && empty!(names)
        append!(names, String.(st["symbol_list"]))
    end
    names
end

"The loss (weighted moments, or minus welfare) at parameter `values` (osr.objective; 1e8 when the model cannot be solved there)."
function osr_objective(m::Model, values, param_names=nothing; order=nothing, replicate_dynare_bugs=false)
    names = param_names === nothing ? _osr_param_names(m) : String.(param_names)
    order = order === nothing ? Int(get(stoch_options(m), "order", 1)) : Int(order)
    OSRProblem(m, names, order; replicate_dynare_bugs=replicate_dynare_bugs)(Vector{Float64}(values))[1]
end

const _OSR_OPTIM_KEYS = Dict(4 => ["MaxIter", "TolFun", "InitialInverseHessian", "NumgradAlgorithm", "NumgradEpsilon", "verbosity", "SaveFiles"],
                             7 => ["MaxIter", "MaxFunEvals", "TolFun", "TolX", "Display"],
                             8 => ["MaxIter", "TolFun", "TolX", "MaxFunEvals", "MaxFunEvalFactor", "InitialSimplexSize", "verbosity"],
                             9 => ["MaxIter", "TolFun", "TolX", "MaxFunEvals", "verbosity", "SaveFiles"])

"""
    osr(m; param_names=nothing, opt_algo=nothing, order=nothing, bounds=nothing, optim_opt=nothing, maxit=nothing, tolf=nothing,
        rng=nothing, replicate_dynare_bugs=false, apply=true) -> OSRResult

`osr` (Dynare 7.2's osr.run / osr.objective): the values of `osr_params` that minimise the weighted (co)variances of `optim_weights`
(order 1), or, when the .mod file has a `planner_objective` before `osr`, maximise the unconditional welfare
(`evaluate_planner_objective`; orders 1 and 2). Options default to the file's `osr_params`, `osr_params_bounds`, `optim_weights`
and `osr` statement.

Optimisers (`opt_algo`): 4 (default, csminwel), 7 (fminsearch), 8 (Dynare's simplex), 9 (CMA-ES, random, the only one taking bounds
here). Dynare's other values (MATLAB toolboxes 1, 2, 3; 5, 6, 10, 11 refused by Dynare itself) raise, as does
`analytic_derivation`. `optim_opt` is a Dict (or the .mod file's string) of Dynare's optim options. Parameters are set to the
optimum in `m` unless `apply=false` (as Dynare does).
"""
function osr(m::Model; param_names=nothing, opt_algo=nothing, order=nothing, bounds=nothing, optim_opt=nothing, maxit=nothing, tolf=nothing,
             rng=nothing, replicate_dynare_bugs::Bool=false, apply::Bool=true, options=nothing)
    opts = options === nothing ? Dict{String,Any}(string(k) => v for (k, v) in get(m.options, "osr", Dict{String,Any}())) : Dict{String,Any}(options)
    so = stoch_options(m)
    names = param_names === nothing ? _osr_param_names(m) : String.(param_names)
    opt_algo === nothing && (opt_algo = get(opts, "osr.opt_algo", 4))
    order === nothing && (order = Int(get(opts, "order", get(so, "order", 1))))
    prob = OSRProblem(m, names, Int(order); replicate_dynare_bugs=replicate_dynare_bugs)
    get(opts, "analytic_derivation", false) == true &&
        throw(ModelError("osr(analytic_derivation): the analytic derivatives of the moments (identification.get_perturbation_params_derivs) are not implemented"))
    unknown = [k for k in keys(opts) if !(k in ("order", "osr.opt_algo", "optim_opt", "osr.maxit", "osr.tolf", "analytic_derivation") || k in COSMETIC || startswith(k, "osr."))]
    isempty(unknown) || throw(ModelError("osr does not act on these options: $(join(sort(unknown), ", "))"))
    bnds = bounds === nothing ? _osr_bounds(m, names) : Matrix{Float64}(bounds)
    oo = Dict{String,Any}()
    if optim_opt isa AbstractDict
        merge!(oo, Dict{String,Any}(string(k) => v for (k, v) in optim_opt))
    else
        s = optim_opt === nothing ? get(opts, "optim_opt", get(so, "optim_opt", "")) : optim_opt
        isempty(strip(string(s))) || merge!(oo, _parse_optim_opt(s))
    end
    maxit === nothing && haskey(opts, "osr.maxit") && (maxit = opts["osr.maxit"])
    tolf === nothing && haskey(opts, "osr.tolf") && (tolf = opts["osr.tolf"])
    if maxit !== nothing || tolf !== nothing
        @warn "OSR: The use of maxit and tolf is now deprecated. Please use the optim-option instead."
        opt_algo == 4 || throw(OSRError("OSR: The maxit and tolf options are not supported when not using the default opt_algo=4. Use the optim-option instead."))
        maxit === nothing || (oo["MaxIter"] = Float64(maxit))
        tolf === nothing || (oo["TolFun"] = Float64(tolf))
    end
    x0 = prob.params0[prob.i_params]
    loss0, info0 = prob(x0)
    info0[1] != 0 && @warn "OSR: the model cannot be solved at the initial values (Dynare's error $(Int(info0[1])))"
    isinf(loss0) && throw(OSRError("OSR: Initial likelihood is infinite"))
    opt_algo in (5, 6, 10, 11) && throw(OSRError("OSR: OSR does not support opt_algo=$opt_algo."))
    if bnds !== nothing && !(opt_algo in (1, 2, 5, 9))
        throw(OSRError("OSR: OSR with bounds on parameters requires a constrained optimizer, i.e. opt_algo= 1,2 or 9."))
    end
    opt_algo in (4, 7, 8, 9) ||
        throw(ModelError("osr with opt_algo=$opt_algo is not implemented (JDyn has 4 csminwel, 7 fminsearch, 8 simplex and 9 CMA-ES; Dynare's 1, 2 and 3 need MATLAB's optimisation toolbox)"))
    for k in keys(oo)
        k in _OSR_OPTIM_KEYS[Int(opt_algo)] || throw(ModelError("osr optim option $k is not supported with opt_algo=$opt_algo (supported: $(join(_OSR_OPTIM_KEYS[Int(opt_algo)], ", ")))"))
    end
    xh = x0; fval = loss0; iters = 0; nfev = 0; exitflag = 0; msg = ""
    if opt_algo == 4
        fcn = th -> (r = prob(th); (r[1], Int(r[2][1]), length(r[2]) > 3 ? r[2][4] : 0.0))
        H0 = get(oo, "InitialInverseHessian", nothing)
        fh, xh_, _, _, iters, nfev, exitflag, msg = csminwel(fcn, x0; H0=H0 === nothing ? nothing : Matrix{Float64}(H0),
            crit=Float64(get(oo, "TolFun", 1e-7)), nit=Int(get(oo, "MaxIter", 1000)), method=Int(get(oo, "NumgradAlgorithm", 2)),
            epsilon=Float64(get(oo, "NumgradEpsilon", 1e-6)), rng=rng === nothing ? dynare_rng(m) : rng)
        xh_ === nothing && throw(OSRError("OSR: csminwel: bad initial parameter"))
        xh, fval = xh_, fh
    elseif opt_algo == 7
        fcn = th -> prob(th)[1]
        xh, fval, exitflag, iters, nfev = fminsearch(fcn, x0; maxiter=Int(get(oo, "MaxIter", 6000)), maxfunevals=Int(get(oo, "MaxFunEvals", 1_000_000)),
                                                     tolx=Float64(get(oo, "TolX", 1e-6)), tolf=Float64(get(oo, "TolFun", 1e-8)))
        exitflag = exitflag == 1 ? 0 : exitflag
        msg = "fminsearch"
    elseif opt_algo == 8
        fcn = th -> (r = prob(th); (r[1], r[2][1] == 0))
        mf = get(oo, "MaxFunEvals", nothing)
        xh, fval, exitflag, iters, nfev, msg = simplex_optimization(fcn, x0; maxiter=Int(get(oo, "MaxIter", 10000)), tolx=Float64(get(oo, "TolX", 1e-4)),
            tolf=Float64(get(oo, "TolFun", 1e-4)), maxfcall=mf === nothing ? nothing : Int(mf), maxfcallfactor=Float64(get(oo, "MaxFunEvalFactor", 500)),
            delta_factor=Float64(get(oo, "InitialSimplexSize", 0.05)))
    else
        fcn = th -> prob(th)[1]
        lb = bnds === nothing ? fill(-Inf, length(x0)) : bnds[:, 1]
        ub = bnds === nothing ? fill(Inf, length(x0)) : bnds[:, 2]
        mf = get(oo, "MaxFunEvals", nothing)
        xh, fval, iters, nfev, msg, ok = cmaes(fcn, x0, lb, ub; rng=rng === nothing ? dynare_rng(m) : rng, maxiter=get(oo, "MaxIter", nothing),
                                                tolfun=Float64(get(oo, "TolFun", 1e-7)), tolx=Float64(get(oo, "TolX", 1e-7)), maxfunevals=mf)
        exitflag = ok ? 0 : 1
    end
    xh = Vector{Float64}(xh)
    apply && (m.params[prob.i_params] = xh)
    pfinal = copy(prob.params0); pfinal[prob.i_params] = xh
    dr, info = _resol_dr(m, pfinal, prob.order)
    optim = Pair{String,Float64}[names[i] => xh[i] for i in eachindex(names)]
    OSRResult(optim, Float64(fval), xh, loss0, Int(exitflag), String(msg), Int(iters), Int(nfev), dr, info[1] == 0 ? 0 : 1)
end
