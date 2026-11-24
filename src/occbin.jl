# Occasionally binding constraints: piecewise-linear solution (Guerrieri and Iacoviello, 2015).
#
# Julia port of pydyn/occbin.py, which is a port of Dynare 7.2's matlab/+occbin/{solve_one_constraint,
# solve_two_constraints, mkdatap_anticipated_dyn, mkdatap_anticipated_2constraints_dyn, map_regime, mkdata}.m
# (public domain: original authors Luca Guerrieri and Matteo Iacoviello, adapted by the Dynare Team). The
# option handling (set_default_options.m, setup.m, solver.m), the IRF routine (irf.m) and the constraint-error
# defaults (preprocessor ComputingTasks.cc, occbin_difference) follow Dynare Team files, GPL-3.0-or-later.
# See NOTICE.md.
#
# For each period in which an unanticipated shock arrives, the solver guesses the periods in which each
# constraint binds, builds the implied time-varying decision rules by backward recursion from the
# reference-regime rule, simulates, checks the constraints and updates the guess until it is verified.
#
# The 7.2 preprocessor has already folded the regime-specific equations into the model: each equation tagged
# [bind='X'] / [relax='X'] is multiplied by the parameter occbin_X_bind (or 1 - occbin_X_bind), so a regime is
# a vector of 0/1 values of those parameters, and the constraints themselves are strings in the
# occbin_constraints statement (m.occbin_constraints: name, bind, relax, error_bind, error_relax).

# ---------------------------------------------------------------------- errors
"""
Failure of the OccBin solver. `error_flag(e)` is Dynare's code: 310 (guesses alternate between two regimes
and `periodic_solution` is off), 311 (no convergence within `maxit`), 312 (a constraint still binds at the end
of the check-ahead horizon: increase `check_ahead_periods`), 313 (infinite loop of regime guesses).
`e.period` is the 1-based shock period that failed.
"""
abstract type OccbinError <: Exception end
struct OccbinPeriodicError <: OccbinError; msg::String; period::Int; end
struct OccbinMaxitError <: OccbinError; msg::String; period::Int; end
struct OccbinHorizonError <: OccbinError; msg::String; period::Int; end
struct OccbinLoopError <: OccbinError; msg::String; period::Int; end
error_flag(::OccbinPeriodicError) = 310
error_flag(::OccbinMaxitError) = 311
error_flag(::OccbinHorizonError) = 312
error_flag(::OccbinLoopError) = 313
Base.showerror(io::IO, e::OccbinError) = print(io, typeof(e), " (error_flag ", error_flag(e), "): ", e.msg)

"""
A case in which Dynare 7.2's OccBin fails (a defect of Dynare), raised only with `replicate_dynare_bugs=true`;
by default JDyn does what Dynare evidently intends (see docs/STATUS.md).
"""
struct OccbinDynareBug <: Exception
    msg::String
end
Base.showerror(io::IO, e::OccbinDynareBug) = print(io, "OccbinDynareBug (Dynare 7.2 fails here): ", e.msg)

# ---------------------------------------------------------------------- results
const _Diag = NamedTuple{(:max_err, :violates, :n_viol, :selected),Tuple{Vector{Float64},Vector{Bool},Vector{Float64},Union{Nothing,Int}}}

"""
    OccbinResult

Result of `simulate(ob, shocks; ...)`: `piecewise` and `linear` are `(periods, n_endo)` paths in levels (the
linear one ignores the constraints), `regime_history[t][j] = (regime, start)` for shock period `t` and constraint
`j` (vectors of 0/1 regimes and their 1-based start periods, Dynare's `oo_.occbin.simul.regime_history`),
`exo_pos` the (1-based) positions of the columns of `shocks` among the exogenous variables, `iterations` the
number of iterations per shock period, `periodic[t]` whether a periodic solution was selected, and
`diagnostics[t]` the per-iteration `max_err`, `violates`, `n_viol` and the 1-based `selected` iteration.
`ss[t] = (T, R, C)` is the rule of shock period `t` under its verified regime, `y_t - ys = T (y_{t-1} - ys) + R u_t + C`
(declaration order, `R` on all shocks; Dynare's `ss.T/R/C(:,:,1)` of `occbin.solver`, used by the OccBin filters).
"""
struct OccbinResult
    piecewise::Matrix{Float64}
    linear::Matrix{Float64}
    shocks::Matrix{Float64}
    regime_history::Vector{Vector{Tuple{Vector{Int},Vector{Int}}}}
    ys::Vector{Float64}
    iterations::Vector{Int}
    exo_pos::Vector{Int}
    periodic::Vector{Bool}
    diagnostics::Vector{_Diag}
    endo_names::Vector{String}
    exo_names::Vector{String}
    ss::Vector{Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}}   # per shock period: (T, R, C), see below
    news::Any                    # obc_method minnorm / news: an ObcNews (src/obc.jl); nothing for occbin
end
OccbinResult(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) = OccbinResult(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, nothing)

"""
    OccbinIRF

OccBin impulse responses (Dynare's `occbin.irf`): `res[(shock, sign)]` is a named tuple `(piecewise, linear)` of
`(periods + 1, n_endo)` deviations from the steady state; `res.shocks`, `res.signs` give the order, `res.size`
the shock sizes.
"""
struct OccbinIRF
    model::String
    periods::Int
    shocks::Vector{String}
    signs::Vector{String}
    size::Vector{Float64}
    variables::Vector{String}
    responses::Dict{Tuple{String,String},NamedTuple{(:piecewise, :linear),Tuple{Matrix{Float64},Matrix{Float64}}}}
end
Base.getindex(r::OccbinIRF, k::Tuple{<:AbstractString,<:AbstractString}) = r.responses[(String(k[1]), String(k[2]))]
Base.keys(r::OccbinIRF) = [(s, g) for g in r.signs for s in r.shocks]
Base.haskey(r::OccbinIRF, k) = haskey(r.responses, (String(k[1]), String(k[2])))

# ---------------------------------------------------------------------- the solver object
struct _Constraint
    bind::Function
    ebind::Function
    relax::Union{Nothing,Function}
    erelax::Function
end

"""
    Occbin(m::Model; params=nothing, qz_criterium=nothing)

Piecewise-linear solver for a model with an `occbin_constraints` block. The reference regime is the one with all
the `occbin_X_bind` parameters at 0 (steady state and first-order rules of that regime are computed here, from
the parameters at construction time: build a new `Occbin` after `set_param!`).
"""
mutable struct Occbin
    model::Model
    names::Vector{String}
    k::Int
    switch::Vector{Int}
    ys::Vector{Float64}
    params::Vector{Float64}
    ref::FirstOrderSolution
    P::Matrix{Float64}
    Q::Matrix{Float64}
    n::Int
    n_exo::Int
    regimes::Dict{Tuple,Any}
    cons::Vector{_Constraint}
end

function Occbin(m::Model; params=nothing, qz_criterium=nothing)
    isempty(m.occbin_constraints) && throw(ModelError("the model has no occbin_constraints block"))
    names = String[c["name"] for c in m.occbin_constraints]
    k = length(names)
    k <= 2 || throw(ModelError("only up to two constraints are supported in OccBin (as in Dynare)"))
    switch = Int[]
    for nm in names
        pn = "occbin_$(nm)_bind"
        haskey(m.ip, pn) || throw(ModelError("parameter $pn not found: no equation tagged for constraint '$nm' in the model"))
        push!(switch, m.ip[pn])
    end
    base = Vector{Float64}(params === nothing ? m.params : params)
    base[switch] .= 0.0
    ys, p = steady_state(m; params=base)
    ref = first_order(m; ys=ys, params=p, loglinear=false, qz_criterium=qz_criterium)       # the piecewise rules are in levels
    ob = Occbin(m, names, k, switch, ys, p, ref, ref.G, ref.H, m.n_endo, m.n_exo, Dict{Tuple,Any}(), _Constraint[])
    _compile_constraints!(ob)
    ob
end

"""
Dynare's `(C, B, A, J, D)` for a regime (a tuple of 0/1 values of the binding-indicator parameters): the
derivatives of the dynamic model at the reference steady state with respect to y(-1), y, y(+1) and the shocks, and
the residual (zero in the reference regime).
"""
function regime_matrices(ob::Occbin, regime::Tuple)
    get!(ob.regimes, regime) do
        p = copy(ob.params)
        for (s, on) in zip(ob.switch, regime)
            p[s] = on ? 1.0 : 0.0
        end
        n, m = ob.n, ob.model
        yy = vcat(ob.ys, ob.ys, ob.ys)
        x0 = exo_steady(m)
        J = dynamic_jac(m, yy, x0, p, ob.ys)
        D = any(regime) ? dynamic_resid(m, yy, x0, p, ob.ys) : zeros(n)
        (J[:, 1:n], J[:, n+1:2n], J[:, 2n+1:3n], J[:, 3n+1:end], D)
    end
end

# ---------------------------------------------------------------------- constraints
const _COMPARISONS = (:<, :>, :<=, :>=, :(==), :(!=))

"Translate the Julia-parsed text of a constraint into code on `z` (levels of the endogenous variables in one period), `p`, `ss`."
function _translate(ex, m::Model)
    if ex isa Symbol
        s = String(ex)
        haskey(m.ie, s) && return :(z[$(m.ie[s])])
        haskey(m.ip, s) && return :(p[$(m.ip[s])])
        haskey(m.ix, s) && throw(ModelError("exogenous variable $s in an occbin constraint is not supported"))
        return ex
    elseif ex isa Expr
        if ex.head == :call && ex.args[1] isa Symbol
            f = String(ex.args[1])
            if lowercase(f) == "steady_state"
                length(ex.args) == 2 && ex.args[2] isa Symbol && haskey(m.ie, String(ex.args[2])) ||
                    throw(ModelError("STEADY_STATE in an occbin constraint takes an endogenous variable"))
                return :(ss[$(m.ie[String(ex.args[2])])])
            elseif haskey(m.ie, f) || haskey(m.ip, f)
                throw(ModelError("leads and lags of variables in occbin constraints are not supported ($f)"))
            end
            return Expr(:call, ex.args[1], (_translate(a, m) for a in ex.args[2:end])...)
        end
        return Expr(ex.head, (_translate(a, m) for a in ex.args)...)
    end
    ex
end

_parse_constraint(s::AbstractString) = Meta.parse(replace(String(s), "~=" => "!=", "~" => "!"))

# Compiled constraint conditions are a function of the model's text only: memoise them per model (Core.eval costs milliseconds,
# and a new Occbin is built for every parameter set of an estimation).
const _CONDITION_CACHE = WeakKeyDict{Model,Dict{Expr,Function}}()
function _compiled_condition(m::Model, mod, body)
    cache = get!(() -> Dict{Expr,Function}(), _CONDITION_CACHE, m)
    ex = _translate(body, m)
    get!(cache, ex) do
        Base.invokelatest(Core.eval, mod, :((z, p, ss) -> $ex))
    end
end

function _compile_constraints!(ob::Occbin)
    m = ob.model
    mod = m.fns.mod
    mk(body) = _compiled_condition(m, mod, body)
    function absdiff(ex)
        (ex isa Expr && ex.head == :call && length(ex.args) == 3 && ex.args[1] in _COMPARISONS) ||
            throw(ModelError("occbin bind/relax condition must be a comparison: $ex"))
        Expr(:call, :abs, Expr(:call, :-, ex.args[2], ex.args[3]))
    end
    empty(s) = isempty(strip(string(s)))
    for c in m.occbin_constraints
        bind = _parse_constraint(c["bind"])
        relax = empty(get(c, "relax", "")) ? nothing : _parse_constraint(c["relax"])
        eb = empty(get(c, "error_bind", "")) ? nothing : _parse_constraint(c["error_bind"])
        er = empty(get(c, "error_relax", "")) ? nothing : _parse_constraint(c["error_relax"])
        # the defaults of ComputingTasks.cc (OccbinConstraintsStatement::writeOutput)
        ebind = eb !== nothing ? eb : absdiff(bind)
        erelax = er !== nothing ? er : relax !== nothing ? absdiff(relax) : absdiff(bind)
        push!(ob.cons, _Constraint(mk(bind), mk(ebind), relax === nothing ? nothing : mk(relax), mk(erelax)))
    end
    ob
end

_call(f, args...) = try
    Base.invokelatest(f, args...)
catch e
    e isa DomainError || rethrow()
    NaN
end

"""
    check_constraints(ob, zlevels) -> (binding, relax, err_bind, err_relax)

Dynare's `occbin_difference` on a `(T, n_endo)` array of levels: boolean `(T, k)` arrays saying where each constraint
binds and where it can be relaxed, and the errors `err.binding_constraint_j` / `err.relax_constraint_j` (the
`error_bind` / `error_relax` expressions of the block, by default the absolute difference between the two sides of
the bind / relax condition).
"""
function check_constraints(ob::Occbin, z::AbstractMatrix)
    # the compiled conditions live in a newer world than this method: run the whole loop in the latest world once, instead of
    # one Base.invokelatest per evaluation
    Base.invokelatest(_check_constraints_loop, ob, z)
end

@inline function _call_now(f, args...)
    try
        f(args...)
    catch e
        e isa DomainError || rethrow()
        NaN
    end
end

function _check_constraints_loop(ob::Occbin, z::AbstractMatrix)
    T, k = size(z, 1), ob.k
    binding, relax = falses(T, k), falses(T, k)
    eb, er = zeros(T, k), zeros(T, k)
    for t in 1:T
        zt = view(z, t, :)
        for (j, c) in enumerate(ob.cons)
            b = _call_now(c.bind, zt, ob.params, ob.ys)
            binding[t, j] = b === true || (b isa Number && !isnan(b) && b != 0)
            if c.relax === nothing
                relax[t, j] = !binding[t, j]
            else
                r = _call_now(c.relax, zt, ob.params, ob.ys)
                relax[t, j] = r === true || (r isa Number && !isnan(r) && r != 0)
            end
            eb[t, j] = _call_now(c.ebind, zt, ob.params, ob.ys)
            er[t, j] = _call_now(c.erelax, zt, ob.params, ob.ys)
        end
    end
    binding, relax, eb, er
end

# ---------------------------------------------------------------------- path for a given guess
"1-based start of the last regime of a 0/1 column (`regime_start(end)` in map_regime)."
function _last_regime_start(col)
    j = findlast(i -> col[i+1] != col[i], 1:length(col)-1)
    j === nothing ? 1 : j + 1
end

"Dynare's `Tmax = max(regime_start(end)) - 1`: last period of the last binding regime."
_tmax(ind::AbstractMatrix) = maximum(_last_regime_start(view(ind, :, j)) for j in 1:size(ind, 2)) - 1

# The rule of period i depends only on the regimes of periods i..Tmax (backward recursion from the reference rule), so rules are
# memoised per Occbin object in a trie keyed from the end: node (parent, regime code) -> rule. Every rule is computed by exactly
# the arithmetic of the uncached recursion, so results are bit for bit the same.
struct _RuleTrie
    child::Dict{Tuple{Int,Int},Int}
    rules::Vector{Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}}
end
const _RULE_TRIES = WeakKeyDict{Occbin,_RuleTrie}()

"Time-varying rules `y_t = T_t y_{t-1} + R_t u_t + C_t` for t = 1..Tmax (vector of tuples)."
function _rules(ob::Occbin, ind::AbstractMatrix, Tmax::Int)
    n, k = ob.n, ob.k
    trie = get!(() -> _RuleTrie(Dict{Tuple{Int,Int},Int}(), Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}[]), _RULE_TRIES, ob)
    out = Vector{Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}}(undef, Tmax)
    node = 0
    for i in Tmax:-1:1
        reg = ntuple(j -> Bool(ind[i, j]), k)
        if i == Tmax
            # Dynare takes the regime of the last binding period from the guess for two constraints
            # (10, 11, else 01) and as "binding" for one constraint
            reg = k == 1 ? (true,) : (any(reg) ? reg : (false, true))
        end
        code = 0
        for (j, r) in enumerate(reg)
            r && (code |= 1 << (j - 1))
        end
        key = (node, code)
        id = get(trie.child, key, 0)
        if id == 0
            Tn, Cn = node == 0 ? (ob.P, zeros(n)) : (trie.rules[node][1], trie.rules[node][3])
            C, B, A, J, D = regime_matrices(ob, reg)
            sol = -((B + A * Tn) \ hcat(C, J, A * Cn + D))
            push!(trie.rules, (sol[:, 1:n], sol[:, n+1:n+ob.n_exo], sol[:, end]))
            id = length(trie.rules)
            trie.child[key] = id
        end
        out[i] = trie.rules[id]
        node = id
    end
    out
end

"Deviations from steady state for `size(ind, 1)` periods given a regime guess `ind` (periods x constraints)."
_path(ob::Occbin, ind::AbstractMatrix, init::AbstractVector, shock::AbstractVector, Tmax::Int) = _path_rules(ob, ind, init, shock, Tmax)[1]

"`_path` and the time-varying rules it used (Dynare's `ss.T/R/C` of `occbin.solver`)."
function _path_rules(ob::Occbin, ind::AbstractMatrix, init::AbstractVector, shock::AbstractVector, Tmax::Int)
    rules = _rules(ob, ind, Tmax)
    N = size(ind, 1)
    z = Matrix{Float64}(undef, N, ob.n)
    y = Vector{Float64}(init)
    tmp = similar(y)
    for t in 1:N
        if t <= length(rules)
            T, R, C = rules[t]
            mul!(tmp, T, y); tmp .+= C                 # = T * y + C, without the temporaries
            y, tmp = tmp, y
            t == 1 && (y .+= R * shock)
        else
            mul!(tmp, ob.P, y)
            y, tmp = tmp, y
            t == 1 && (y .+= ob.Q * shock)
        end
        z[t, :] = y
    end
    z, rules
end

"Dynare's `map_regime`: regimes and their 1-based start periods for a 0/1 column."
function _map_regime(col)
    regime, start = [Int(col[1])], [1]
    for i in 2:length(col)
        if Int(col[i]) != regime[end]
            push!(regime, Int(col[i])); push!(start, i)
        end
    end
    regime, start
end

"Dynare's `init_regime` guess for one shock period: fill the columns of `bi` from [(regime, start), ...], one pair per constraint."
function _apply_regime_guess(bi::BitMatrix, per)
    k = size(bi, 2)
    length(per) == k || throw(ArgumentError("init_regime needs one (regime, start) pair per constraint"))
    bi = copy(bi)
    for (j, (regime, start)) in enumerate(per)
        bi[:, j] .= regime[end] != 0
        for ir in 1:length(regime)-1
            a, b = Int(start[ir]), Int(start[ir+1]) - 1
            if b > size(bi, 1)                  # like MATLAB's indexed assignment, a start beyond the rows grows the matrix
                bi = vcat(bi, falses(b - size(bi, 1), k))
            end
            bi[a:b, j] .= regime[ir] != 0
        end
    end
    bi
end

# index of the smallest non-NaN value (first one on ties; the first element when all are NaN), as MATLAB's min
function _argmin_nan(v)
    best = 0
    for i in eachindex(v)
        isnan(v[i]) && continue
        (best == 0 || v[i] < v[best]) && (best = i)
    end
    best == 0 ? 1 : best
end

_exo_index(m::Model, p::AbstractString) = (haskey(m.ix, p) ? m.ix[p] : throw(ArgumentError("unknown exogenous variable $p")))
_exo_index(m::Model, p::Integer) = (1 <= p <= m.n_exo ? Int(p) : throw(ArgumentError("exo_pos $p out of range")))

# ---------------------------------------------------------------------- simulation
"""
    simulate(ob::Occbin, shocks; periods=100, check_ahead_periods=200, maxit=30, init=nothing, curb_retrench=false,
             periodic_solution=false, periodic_solution_threshold=1, periodic_solution_strict=true, init_regime=nothing,
             init_binding_indicator=nothing, max_check_ahead_periods=Inf, reset_check_ahead_periods_in_new_period=false,
             reset_regime_in_new_period=false, exo_pos=nothing, algo_truncation=1, replicate_dynare_bugs=false) -> OccbinResult

Piecewise-linear simulation under a sequence of unanticipated shocks (Dynare's `occbin.solver` with the `simul`
options, defaults as in `set_default_options.m`). `shocks` is `(n_shock_periods, n_exo)`, or
`(n_shock_periods, length(exo_pos))` when `exo_pos` (1-based positions or names of the exogenous variables,
Dynare's `exo_pos`) is given. The simulation runs for `periods` periods (at least the number of shock periods).
`init` is `endo_init`, deviations from the steady state in declaration order.

`init_regime` (Dynare's `init_regime`) is one entry per leading shock period, each a vector of `(regime, start)`
pairs, one per constraint (the format of `OccbinResult.regime_history`); `init_binding_indicator` is a
`(rows, n_constraints)` 0/1 array, the initial guess for the first shock period.

Failures throw an `OccbinError` whose `error_flag` is Dynare's (310-313).

Dynare 7.2 crashes when `check_ahead_periods` exceeds `max_check_ahead_periods` (its message refers to an undefined
variable `simul`); JDyn does what the message announces and lowers `check_ahead_periods` to the maximum, unless
`replicate_dynare_bugs=true`, which throws `OccbinDynareBug`.
"""
function simulate(ob::Occbin, shocks_in::AbstractMatrix; periods::Integer=100, check_ahead_periods::Integer=200, maxit::Integer=30,
                  init=nothing, curb_retrench::Bool=false, periodic_solution::Bool=false, periodic_solution_threshold::Integer=1,
                  periodic_solution_strict::Bool=true, init_regime=nothing, init_binding_indicator=nothing,
                  max_check_ahead_periods::Real=Inf, reset_check_ahead_periods_in_new_period::Bool=false,
                  reset_regime_in_new_period::Bool=false, exo_pos=nothing, algo_truncation::Integer=1,
                  replicate_dynare_bugs::Bool=false, obc_method=:occbin, obc_horizon=40, obc_tol::Real=1e-10,
                  obc_raise::Bool=true, obc_solver_options=NamedTuple(), obc_fast_path::Bool=true,
                  obc_order=1, obc_cubature::Bool=false, obc_periods_of_uncertainty=16, obc_qmc_level=10, obc_cubature_regions=1)
    method = _obc_method(obc_method)
    if _obc_order(obc_order) != 1 || obc_cubature       # src/obc_ho.jl: order 2 and / or cubature (news only)
        method == :news || throw(ArgumentError("obc_order = 2 and obc_cubature need obc_method = news"))
        return _simulate_obc_ho(ob, shocks_in; horizon=_obc_horizon(obc_horizon), tol=obc_tol, periods=periods, check_ahead_periods=check_ahead_periods,
                                max_check_ahead_periods=max_check_ahead_periods, init=init, exo_pos=exo_pos, order=_obc_order(obc_order),
                                cubature=obc_cubature, periods_of_uncertainty=_obc_int_option(obc_periods_of_uncertainty, "obc_periods_of_uncertainty"),
                                qmc_level=_obc_int_option(obc_qmc_level, "obc_qmc_level"; min=0), regions=_obc_int_option(obc_cubature_regions, "obc_cubature_regions"),
                                ignored=(maxit=maxit, curb_retrench=curb_retrench, periodic_solution=periodic_solution,
                                         periodic_solution_threshold=periodic_solution_threshold, periodic_solution_strict=periodic_solution_strict,
                                         init_regime=init_regime, init_binding_indicator=init_binding_indicator,
                                         reset_check_ahead_periods_in_new_period=reset_check_ahead_periods_in_new_period,
                                         reset_regime_in_new_period=reset_regime_in_new_period, algo_truncation=algo_truncation))
    end
    if method != :occbin                            # src/obc.jl: the same loop with an LCP solver instead of the regime iteration
        return _simulate_obc(ob, shocks_in; method=method, horizon=_obc_horizon(obc_horizon), tol=obc_tol, raise=obc_raise,
                             periods=periods, check_ahead_periods=check_ahead_periods, max_check_ahead_periods=max_check_ahead_periods,
                             init=init, exo_pos=exo_pos, solver_options=obc_solver_options, fast_path=obc_fast_path,
                             ignored=(maxit=maxit, curb_retrench=curb_retrench, periodic_solution=periodic_solution,
                                      periodic_solution_threshold=periodic_solution_threshold, periodic_solution_strict=periodic_solution_strict,
                                      init_regime=init_regime, init_binding_indicator=init_binding_indicator,
                                      reset_check_ahead_periods_in_new_period=reset_check_ahead_periods_in_new_period,
                                      reset_regime_in_new_period=reset_regime_in_new_period, algo_truncation=algo_truncation))
    end
    k, n, n_exo, m = ob.k, ob.n, ob.n_exo, ob.model
    exo_idx = exo_pos === nothing ? collect(1:n_exo) : Int[_exo_index(m, p) for p in (exo_pos isa Union{AbstractString,Integer} ? [exo_pos] : exo_pos)]
    size(shocks_in, 2) == length(exo_idx) ||
        throw(ArgumentError("shocks have $(size(shocks_in, 2)) columns but exo_pos selects $(length(exo_idx)) shocks"))
    S = size(shocks_in, 1)
    S >= 1 || throw(ArgumentError("at least one shock period is needed"))
    shocks = zeros(S, n_exo)
    shocks[:, exo_idx] = Float64.(shocks_in)
    n_periods = max(Int(periods), S)
    max_check = max_check_ahead_periods
    if check_ahead_periods > max_check
        replicate_dynare_bugs && throw(OccbinDynareBug("simul_check_ahead_periods ($check_ahead_periods) exceeds simul_max_check_ahead_periods ($max_check): " *
                                                       "Dynare 7.2's occbin.solver references an undefined variable 'simul' when it prints its warning"))
        check_ahead_periods = Int(max_check)
    end
    n0 = max(Int(check_ahead_periods), n_periods - S)                  # nperiods_0
    y0 = init === nothing || isempty(init) ? zeros(n) : Float64.(vec(collect(init)))
    length(y0) == n || throw(ArgumentError("init needs $n values (declaration order)"))
    init_orig = copy(y0)

    bi = init_binding_indicator === nothing ? falses(0, k) : BitMatrix(reshape(Bool.(collect(init_binding_indicator)), :, k))
    size(bi, 1) > n0 + 1 && throw(ArgumentError("init_binding_indicator has more rows than check_ahead_periods + 1"))
    size(bi, 1) < n0 + 1 && (bi = vcat(bi, falses(n0 + 1 - size(bi, 1), k)))

    piecewise = zeros(n_periods, n)
    history = Vector{Vector{Tuple{Vector{Int},Vector{Int}}}}()
    iters, periodic_flags, diag = Int[], Bool[], _Diag[]
    z = zeros(0, n)
    used = bi
    ss_out = Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}[]
    rules_cur = Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}[]
    first_rule(r) = isempty(r) ? (ob.P, ob.Q, zeros(n)) : r[1]

    for sp in 1:S
        _obc_tick()
        changed, it, increased = true, 0, false
        is_periodic = is_loop = false
        guess_it = init_regime !== nothing && sp <= length(init_regime)
        hist = BitMatrix[]
        max_err = fill(NaN, maxit)
        viol_exp = falses(maxit)
        n_viol = zeros(maxit)
        shock_here = sp == 1 || any(!iszero, view(shocks, sp, :))
        chosen_periodic, selected = false, nothing

        while changed && it < maxit && !is_periodic && !is_loop
            it += 1
            if any(view(bi, size(bi, 1), :)) && n0 < max_check
                bi = vcat(bi, falses(1, k))
                n0 += 1
                increased = true
            elseif n0 >= max_check
                bi = bi[1:min(size(bi, 1), Int(max_check) + 1), :]
                bi[end, :] .= false
            end
            size(bi, 1) < n0 + 1 && (bi = vcat(bi, falses(n0 + 1 - size(bi, 1), k)))
            if it == 1 && guess_it
                bi = _apply_regime_guess(bi, init_regime[sp])
                n0 = size(bi, 1) - 1
            end
            used = copy(bi)
            push!(hist, used)

            if shock_here
                if it == 1 && reset_regime_in_new_period
                    if reset_check_ahead_periods_in_new_period
                        n0 = max(Int(check_ahead_periods), n_periods - S)
                        bi = falses(n0 + 1, k)
                    else
                        bi = falses(size(bi)...)
                    end
                    used = copy(bi)
                    hist[end] = used
                end
                z, rules_cur = _path_rules(ob, used, y0, view(shocks, sp, :), _tmax(used))
                binding, relax, eb, er = check_constraints(ob, z .+ ob.ys')
                L = size(bi, 1)
                if isfinite(max_check) && max_check < L
                    E, pad = Int(max_check), L - Int(max_check)
                else
                    E, pad = L, 0
                end
                new_viol = binding[1:E, :] .& .!bi[1:E, :]
                new_rel = relax[1:E, :] .& bi[1:E, :]
                if any(new_viol) || any(new_rel)
                    max_err[it] = maximum(abs.(vcat(eb[1:E, :][new_viol], er[1:E, :][new_rel])))
                    changed = true
                    viol_exp[it] = any(new_viol)
                else
                    changed = false
                    max_err[it] = 0.0
                end
                # enforce the unconstrained regime beyond max_check_ahead_periods
                bind_new = vcat(binding[1:E, :], falses(pad, k))
                rel_new = vcat(relax[1:E, :], trues(pad, k))
                if k == 1     # solve_one_constraint: sum(bi - next guess), can be negative
                    n_viol[it] = sum(Int.(bi) .- Int.((bi .| bind_new) .& .!(bi .& rel_new)))
                else          # solve_two_constraints: newly binding periods, the larger of the two constraints
                    n_viol[it] = maximum(sum(.!bi[:, c] .& bind_new[:, c]) for c in 1:k)
                end
                if curb_retrench
                    # slow the update down: release at most the last binding period per constraint
                    retrench = falses(size(bi)...)
                    for j in 1:k
                        rb = findall(relax[1:E, j] .& bi[1:E, j])
                        r_all, b_all = findall(relax[1:E, j]), findall(bi[1:E, j])
                        if !isempty(rb) && r_all[end] >= b_all[end]
                            retrench[rb[end], j] = true
                        end
                    end
                    bi = (bi .| bind_new) .& .!retrench
                else
                    bi = (bi .| bind_new) .& .!(bi .& rel_new)
                end

                if it > 1 && changed && !increased
                    is_loop = any(h -> size(h) == size(bi) && h == bi, hist[1:it-1])
                    prev = hist[it-1]
                    same = size(prev) == size(bi) && prev == bi
                    small = all(sum(Int.(used[:, c]) .- Int.(bi[:, c])) <= periodic_solution_threshold for c in 1:k)
                    is_periodic = same && small
                    is_per_all = falses(it - 1)
                    is_per_all[it-1] = is_periodic
                    strict = is_periodic
                    if is_loop && !is_periodic && !periodic_solution_strict
                        cand = n_viol[1:it][.!viol_exp[1:it]]
                        ok = k == 1 ? any(c -> c <= periodic_solution_threshold, cand) : any(c -> c < periodic_solution_threshold, cand)
                        if ok
                            is_periodic = true
                            is_per_all = falses(it - 1)
                            is_per_all[1] = true         # consider all guesses
                        end
                    end
                    if is_periodic && periodic_solution
                        inx = collect(findfirst(is_per_all):it)
                        inx1 = [i for i in inx if !viol_exp[i]]
                        isempty(inx1) || (inx = inx1)
                        vals = (strict || isempty(inx1)) ? max_err[inx] : n_viol[inx]
                        best = inx[_argmin_nan(vals)]
                        bi = copy(hist[best])
                        chosen_periodic, selected = true, best
                        if best < it
                            used = copy(bi)
                            z, rules_cur = _path_rules(ob, used, y0, view(shocks, sp, :), _tmax(used))
                        end
                    end
                end
            else
                changed = false
                rules_cur = rules_cur[2:end]          # Dynare: SS = SS(2:end), the rule of the next period
                z = vcat(z[2:end, :], (ob.P * z[end, :])')
                size(z, 1) < size(bi, 1) && (z = vcat(z, (ob.P * z[end, :])'))
            end
        end

        if changed
            if maxit > algo_truncation
                if is_periodic
                    periodic_solution || throw(OccbinPeriodicError("period $sp: regime guesses alternate between two regimes (error 310); " *
                                                                   "use periodic_solution=true to pick the guess with the smallest error", sp))
                elseif is_loop
                    throw(OccbinLoopError("period $sp: regime guesses cycle without converging (error 313)", sp))
                else
                    throw(OccbinMaxitError("period $sp: no convergence in $maxit iterations (error 311)", sp))
                end
            else
                bi = copy(hist[end])
            end
        end
        any(view(used, size(used, 1), :)) &&
            throw(OccbinHorizonError("period $sp: a constraint binds at the end of the horizon; increase check_ahead_periods (error 312)", sp))
        push!(history, [_map_regime(view(used, :, j)) for j in 1:k])
        push!(iters, it)
        push!(ss_out, first_rule(rules_cur))
        push!(periodic_flags, changed && chosen_periodic)
        push!(diag, (max_err=max_err[1:it], violates=Vector{Bool}(viol_exp[1:it]), n_viol=n_viol[1:it], selected=selected))
        y0 = z[1, :]
        piecewise[sp, :] = y0
        bi = vcat(bi[2:end, :], falses(1, k))
    end
    _obc_tick()
    n_periods > S && (piecewise[S+1:end, :] = z[2:n_periods-S+1, :])

    linear = Matrix{Float64}(undef, n_periods, n)
    y = init_orig
    for t in 1:n_periods
        y = ob.P * y + (t <= S ? ob.Q * shocks[t, :] : zeros(n))
        linear[t, :] = y
    end
    OccbinResult(piecewise .+ ob.ys', linear .+ ob.ys', Matrix{Float64}(shocks_in), history, ob.ys, iters, exo_idx, periodic_flags,
                 diag, m.endo_names, m.exo_names, ss_out)
end
simulate(ob::Occbin, shocks::AbstractVector; kw...) = simulate(ob, reshape(Float64.(shocks), :, 1); kw...)

# ---------------------------------------------------------------------- options of the .mod file
# option names of occbin_setup / occbin_solver (JSON keys) -> simulate keywords
const _OCCBIN_MOD_OPTIONS = Dict{String,Tuple{Symbol,Type}}(
    "simul.periods" => (:periods, Int),
    "simul.maxit" => (:maxit, Int),
    "simul.curb_retrench" => (:curb_retrench, Bool),
    "simul.check_ahead_periods" => (:check_ahead_periods, Int),
    "simul.max_check_ahead_periods" => (:max_check_ahead_periods, Float64),
    "simul.periodic_solution" => (:periodic_solution, Bool),
    "simul.periodic_solution_threshold" => (:periodic_solution_threshold, Int),
    "simul.periodic_solution_strict" => (:periodic_solution_strict, Bool),
    "simul.reset_check_ahead_periods_in_new_period" => (:reset_check_ahead_periods_in_new_period, Bool),
    "simul.reset_regime_in_new_period" => (:reset_regime_in_new_period, Bool),
)

_asbool(v::Bool) = v
_asbool(v::Real) = v != 0
_asbool(v::AbstractString) = lowercase(strip(v)) in ("true", "1")
_asnum(v::Real) = Float64(v)
_asnum(v::AbstractString) = lowercase(strip(v)) in ("inf", "+inf") ? Inf : parse(Float64, v)

"""
    mod_options(ob) -> NamedTuple

`simulate` keywords set by the `.mod` file's `occbin_setup` / `occbin_solver` statements (applied in order, as Dynare
does: a later statement overrides an earlier one). Unset options keep Dynare's defaults. `simul_debug` only prints diagnostics in Dynare and is accepted;
the options of OccBin estimation and smoothing (`smoother_*`, `likelihood_*`, `filter_*`) are read by `occbin_filter_options`
(a warning says the simulation does not use them);
other options produce a warning.
"""
function mod_options(ob::Occbin)
    m = ob.model
    o = get(m.options, "occbin_options", Dict{String,Any}())     # every occbin_setup / occbin_solver statement, in order
    out = Dict{Symbol,Any}()
    unused, filt = String[], String[]
    for (key, v) in o
        if haskey(_OCCBIN_MOD_OPTIONS, key)
            kw, T = _OCCBIN_MOD_OPTIONS[key]
            out[kw] = T === Bool ? _asbool(v) : T === Int ? Int(round(_asnum(v))) : _asnum(v)
        elseif any(pre -> startswith(key, pre), ("smoother.", "likelihood.", "filter."))
            push!(filt, key)                          # read by occbin_filter_options (estimation, smoother)
        elseif key != "simul.debug"
            push!(unused, key)
        end
    end
    merge!(out, _obc_mod_options(m))                 # options_.occbin.obc_method / obc_horizon (native lines, src/obc.jl)
    isempty(unused) || warn_once(m, "occbin_setup/occbin_solver options not implemented in JDyn (ignored): $(join(sort(unused), ", "))")
    isempty(filt) || warn_once(m, "occbin_setup options of the OccBin filters and smoother, not used by the simulation (they apply to " *
                                  "estimation and smoothing): $(join(sort(filt), ", "))")
    (; (k => out[k] for k in sort(collect(keys(out))))...)
end

"""
    surprise_shocks(ob) -> (shocks, exo_pos)

The shock sequence of the model's `shocks(surprise)` blocks as Dynare's `occbin.setup` builds it: the columns that
are not entirely zero, and their (1-based) positions among the exogenous variables.
"""
function surprise_shocks(ob::Occbin)
    m = ob.model
    blocks = [st for st in get(m.options, "shock_paths", Any[]) if get(st, "surprise", false) == true]
    isempty(blocks) && throw(ModelError("the model has no shocks(surprise) block"))
    last = maximum(Int(_numval(v["period2"])) for st in blocks for s in st["surprise_shocks"] for v in s["values"])
    out = zeros(last, m.n_exo)
    for st in blocks, s in st["surprise_shocks"], v in s["values"]
        out[Int(_numval(v["period1"])):Int(_numval(v["period2"])), m.ix[s["var"]]] .= _value(m, v["value"])
    end
    cols = [j for j in 1:m.n_exo if any(!iszero, out[:, j])]
    out[:, cols], cols
end

"""
    simulate(ob; kw...) -> OccbinResult

Simulate as the `.mod` file specifies: its `shocks(surprise)` blocks and the options given to `occbin_setup` /
`occbin_solver` (`simul_periods`, `simul_maxit`, `simul_curb_retrench`, `simul_check_ahead_periods`,
`simul_max_check_ahead_periods`, `simul_periodic_solution`, `simul_periodic_solution_threshold`,
`simul_periodic_solution_strict`, `simul_reset_check_ahead_periods_in_new_period`,
`simul_reset_regime_in_new_period`). Keyword arguments override them.
"""
function simulate(ob::Occbin; kw...)
    opts = Dict{Symbol,Any}(pairs(mod_options(ob)))
    merge!(opts, Dict{Symbol,Any}(kw))
    sh, pos = surprise_shocks(ob)
    haskey(opts, :exo_pos) || (opts[:exo_pos] = pos)
    simulate(ob, sh; opts...)
end

# ---------------------------------------------------------------------- impulse responses
"""
    irf(ob::Occbin, impulse; periods=40, baseline_shocks=nothing, kw...) -> (periods, n_endo)

Piecewise-linear impulse response: the simulation with `impulse` added to the first period's shocks minus the
simulation without it. Unlike a linear IRF this depends on the size and sign of the impulse and on the baseline.
See `irf_occbin` for Dynare's own definition (shock size, signs, horizon).
"""
function irf(ob::Occbin, impulse::AbstractVector; periods::Integer=40, baseline_shocks=nothing, kw...)
    base = baseline_shocks === nothing ? zeros(1, ob.n_exo) : copy(Float64.(baseline_shocks isa AbstractVector ? reshape(baseline_shocks, 1, :) : baseline_shocks))
    hit = copy(base)
    hit[1, :] .+= Float64.(impulse)
    a = simulate(ob, hit; periods=periods, kw...)
    b = simulate(ob, base; periods=periods, kw...)
    a.piecewise .- b.piecewise
end

"""
    irf_occbin(ob; periods=40, shocksize=nothing, exo_names=nothing, shocksigns=("pos","neg"), check_ahead_periods=30,
               maxit=30, init_regime=nothing, replicate_dynare_bugs=false, kw...) -> OccbinIRF

OccBin IRFs as Dynare's `occbin.irf` computes them (`t0 = 0`, no smoother). For each shock in `exo_names` (default
all) and each sign: one simulation with `+/- shocksize` in period 1 and zero shocks for `periods + 1` shock periods
(Dynare passes `options_.irf + 1` rows of shocks), minus the simulation without the shock. `shocksize` defaults to
the shock's standard deviation (0.01 if that is below 1e-9) and may be a scalar or one value per shock.
`check_ahead_periods` (30) and `maxit` (30) are the `irf` options, which replace the `simul` ones; further keywords
(`curb_retrench`, `periodic_solution`, ...) go to `simulate`. Each response has `periods + 1` rows, in deviations
from the steady state.

Dynare 7.2's `occbin.irf` fails for a model with more than one shock when `occbin_setup` had set `exo_pos` from a
`shocks(surprise)` block covering fewer shocks (SHOCKS then has one column per shock but `exo_pos` is stale).
JDyn computes the responses regardless; `replicate_dynare_bugs=true` throws `OccbinDynareBug` in that case.
"""
function irf_occbin(ob::Occbin; periods::Integer=40, shocksize=nothing, exo_names=nothing, shocksigns=("pos", "neg"),
                    check_ahead_periods::Integer=30, maxit::Integer=30, init_regime=nothing,
                    replicate_dynare_bugs::Bool=false, kw...)
    m = ob.model
    names = String.(exo_names === nothing ? m.exo_names : (exo_names isa AbstractString ? [exo_names] : collect(exo_names)))
    idx = [_exo_index(m, nm) for nm in names]
    if replicate_dynare_bugs && haskey(m.options, "shock_paths")
        _, stale = surprise_shocks(ob)
        length(stale) != m.n_exo && throw(OccbinDynareBug("occbin.irf passes SHOCKS with $(m.n_exo) columns but exo_pos (set by occbin_setup from the " *
                                                          "shocks(surprise) block) has $(length(stale)) entries"))
    end
    if shocksize === nothing
        size_ = sqrt.(diag(m.Sigma_e)[idx])
        size_[size_ .< 1e-9] .= 0.01
    else
        size_ = Float64.(shocksize isa Real ? [shocksize] : collect(shocksize))
        length(size_) == 1 && (size_ = fill(size_[1], length(idx)))
        length(size_) == length(idx) || throw(ArgumentError("shocksize needs one value or one per shock"))
    end
    signs = String.(shocksigns isa AbstractString ? [shocksigns] : collect(shocksigns))
    for s in signs
        (occursin("pos", s) || occursin("neg", s)) || throw(ArgumentError("unknown shock sign $s"))
    end
    sim = (; periods=periods, check_ahead_periods=check_ahead_periods, maxit=maxit, init_regime=init_regime,
           replicate_dynare_bugs=replicate_dynare_bugs, _obc_mod_options(m)..., kw...)
    base_shocks = zeros(periods + 1, ob.n_exo)
    base = simulate(ob, base_shocks; sim...)
    out = Dict{Tuple{String,String},NamedTuple{(:piecewise, :linear),Tuple{Matrix{Float64},Matrix{Float64}}}}()
    for sign in signs, (nm, j, sz) in zip(names, idx, size_)
        sh = copy(base_shocks)
        sh[1, j] += occursin("pos", sign) ? sz : -sz
        r = simulate(ob, sh; sim...)
        out[(nm, sign)] = (piecewise=r.piecewise .- base.piecewise, linear=r.linear .- base.linear)
    end
    OccbinIRF(m.name, periods, names, signs, size_, copy(m.endo_names), out)
end

# ---------------------------------------------------------------------- plotted content
"""
    irf_table(res::OccbinIRF; variables=nothing)

The OccBin responses in long format, one row per shock, sign, variable, solution and period, as a vector of
named tuples `shock, sign, variable, period, solution, value`; `solution` is `"piecewise"` or `"linear"`. Periods
count from 1, the impact period (Dynare's `occbin.plot_irfs` plots the same arrays against 1:periods+1). Write it
with `write_csv`. `variables` selects variables (default all).
"""
function irf_table(res::OccbinIRF; variables=nothing)
    vars = variables === nothing ? res.variables : String.(variables)
    cols = [findfirst(==(v), res.variables) for v in vars]
    any(isnothing, cols) && throw(ArgumentError("unknown variable(s): $(join(vars[isnothing.(cols)], ", "))"))
    rows = NamedTuple{(:shock, :sign, :variable, :period, :solution, :value),Tuple{String,String,String,Int,String,Float64}}[]
    for sign in res.signs, shock in res.shocks
        r = res.responses[(shock, sign)]
        for (v, c) in zip(vars, cols), (sol, arr) in (("piecewise", r.piecewise), ("linear", r.linear)), t in 1:size(arr, 1)
            push!(rows, (shock=shock, sign=sign, variable=v, period=t, solution=sol, value=arr[t, c]))
        end
    end
    rows
end

"""
    irf_table(res::OccbinResult; variables=nothing, noconstant=false)

The simulated paths of `simulate(ob, ...)` in long format, `variable, period, solution, value` with `solution` equal
to `"piecewise"` or `"linear"`, plus the shock sequence of each requested exogenous variable (`solution = "shock"`),
the content of Dynare's `occbin_graph`. Values are levels, or deviations from the steady state with
`noconstant=true` (Dynare's `graph.steady_state` option). Exogenous variables among `variables` must be among the
shocks of the simulation, as in Dynare.
"""
function irf_table(res::OccbinResult; variables=nothing, noconstant::Bool=false)
    vars = variables === nothing ? res.endo_names[1:min(end, length(res.endo_names))] : String.(variables)
    rows = NamedTuple{(:variable, :period, :solution, :value),Tuple{String,Int,String,Float64}}[]
    T = size(res.piecewise, 1)
    for v in vars
        if (c = findfirst(==(v), res.endo_names)) !== nothing
            off = noconstant ? res.ys[c] : 0.0
            for (sol, arr) in (("piecewise", res.piecewise), ("linear", res.linear)), t in 1:T
                push!(rows, (variable=v, period=t, solution=sol, value=arr[t, c] - off))
            end
        elseif (e = findfirst(==(v), res.exo_names)) !== nothing
            col = findfirst(==(e), res.exo_pos)
            col === nothing && throw(ArgumentError("$v was not part of the shocks for OccBin"))
            for t in 1:T
                push!(rows, (variable=v, period=t, solution="shock", value=t <= size(res.shocks, 1) ? res.shocks[t, col] : 0.0))
            end
        else
            throw(ArgumentError("unknown variable $v"))
        end
    end
    rows
end

"""
    occbin_graph_table(ob, res) -> rows

`irf_table(res)` for the variable list and options of the `.mod` file's `occbin_graph` statement (all original
endogenous variables when the list is empty; `noconstant` is honoured).
"""
function occbin_graph_table(ob::Occbin, res::OccbinResult)
    m = ob.model
    st = findlast(s -> get(s, "statementName", "") == "occbin_graph", m.statements)
    list = st === nothing ? String[] : String.(get(m.statements[st], "symbol_list", String[]))
    opts = st === nothing ? Dict{String,Any}() : get(m.statements[st], "options", Dict{String,Any}())
    noconst = any(kv -> (occursin("steady_state", kv[1]) && kv[2] == false) || (occursin("noconstant", kv[1]) && kv[2] != false), opts)
    isempty(list) && (list = orig_endo_names(m))
    irf_table(res; variables=list, noconstant=noconst)
end
