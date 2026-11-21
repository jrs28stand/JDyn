# Likelihood filters and smoother of OccBin models (port of pydyn/occbin_filter.py, which follows Dynare 7.2's
# matlab/+occbin/{IVF_core, match_function} (public domain: Cuba-Borda, Guerrieri, Iacoviello and Zhong, adapted by
# the Dynare Team), matlab/+occbin/map_regime.m (public domain: Guerrieri and Iacoviello), and the Dynare Team files
# (GPL-3.0-or-later) matlab/+occbin/{IVF_posterior, kalman_update_engine, kalman_update_algo_1, kalman_update_algo_3,
# DSGE_smoother, check_regimes, backward_map_regime, shock_decomposition, squeeze_shock_decomposition,
# write_regimes_to_xls, set_default_options, setup}.m, matlab/kalman/likelihood/missing_observations_kalman_filter.m
# (OccBin branch), matlab/kalman/missing_DiffuseKalmanSmootherH3_Z.m and DsgeSmoother.m (OccBin branches) and
# matlab/optimization/csolve.m (Christopher Sims, Dynare Team). See NOTICE.md.
#
# Two filters, both on the piecewise-linear solution of `Occbin`:
#   * the inversion filter (IVF): as many shocks as observed variables, no measurement error; in each period the
#     shocks that reproduce the observations are found through the piecewise-linear solution (csolve Newton steps)
#     and the likelihood is the shock density times the Jacobian of the shock-to-observable map;
#   * the piecewise Kalman filter (PKF): the Kalman update of each period is iterated with the regime-dependent
#     transition implied by the smoothed shocks of the two-period problem until the regime is consistent.
# Plus the OccBin smoother (DSGE_smoother), the OccBin shock decomposition and the regime table of occbin_write_regimes.
#
# A regime is Dynare's regime_history entry: one (regime, regimestart) pair of Int vectors per constraint, e.g.
# [([0, 1, 0], [1, 3, 5])] for one constraint binding in periods 3 and 4 (the type of OccbinResult.regime_history[t]).

const OccRegime = Vector{Tuple{Vector{Int},Vector{Int}}}
const _TRC = Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}

# ====================================================================== options
"set_default_options.m (flags likelihood, filter, smoother, solver; the fields the filters read)."
const OCCBIN_FILTER_DEFAULTS = Dict{String,Any}(
    "likelihood.brute_force_regime_guess" => true,
    "likelihood.brute_force_extra_regime_guess" => true,
    "likelihood.curb_retrench" => false,
    "likelihood.first_period_binding_regime_allowed" => 1,
    "likelihood.first_period_occbin_update" => 1,
    "likelihood.IF_likelihood" => false,
    "likelihood.inversion_filter" => false,
    "likelihood.IVF_shock_observable_mapping" => nothing,
    "likelihood.loss_function_regime_guess" => false,
    "likelihood.maxit" => 30,
    "likelihood.max_number_of_iterations" => 10,
    "likelihood.max_check_ahead_periods" => Inf,
    "likelihood.number_of_initial_periods_with_extra_regime_guess" => 0,
    "likelihood.periods" => 3,
    "likelihood.check_ahead_periods" => 200,
    "likelihood.periodic_solution" => false,
    "likelihood.restrict_state_space" => true,
    "likelihood.use_updated_regime" => true,
    "likelihood.init_regime_history" => nothing,
    "likelihood.init_binding_indicator" => nothing,
    "filter.periodic_solution" => true,
    "filter.use_relaxation" => false,
    "filter.use_relaxation_tol_period" => 1,
    "filter.particle.status" => false,
    "filter.init_periods_using_particles" => false,
    "smoother.curb_retrench" => false,
    "smoother.debug" => false,
    "smoother.fast" => false,
    "smoother.first_period_occbin_update" => 1,
    "smoother.inversion_filter" => false,
    "smoother.linear_smoother" => true,
    "smoother.maxit" => 30,
    "smoother.max_check_ahead_periods" => Inf,
    "smoother.max_number_of_iterations" => 10,
    "smoother.periods" => 3,
    "smoother.check_ahead_periods" => 200,
    "smoother.periodic_solution" => false,
    "smoother.plot" => true,
    "smoother.init_regime_history" => nothing,
    "smoother.init_binding_indicator" => nothing,
    "simul.periodic_solution_threshold" => 1,
    "simul.periodic_solution_strict" => true,
    "simul.reset_check_ahead_periods_in_new_period" => false,
    "simul.reset_regime_in_new_period" => false,
    "solver.maxit" => 10,
    "solver.solve_tolf" => 1e-5,
    "solver.solve_tolx" => 1e-10,
    "solver.solve_algo" => 3,
)

"The occbin_setup options that Dynare 7.2's occbin.setup copies into options_.occbin; the others are parsed and dropped."
const _OCC_SETUP_COPIED = Set(["simul.periods", "simul.curb_retrench", "simul.maxit", "simul.check_ahead_periods", "simul.debug",
    "simul.periodic_solution", "smoother.periods", "smoother.curb_retrench", "smoother.maxit", "smoother.check_ahead_periods",
    "smoother.debug", "smoother.periodic_solution", "smoother.inversion_filter", "likelihood.periods", "likelihood.curb_retrench",
    "likelihood.maxit", "likelihood.check_ahead_periods", "likelihood.periodic_solution", "likelihood.max_number_of_iterations",
    "filter.use_relaxation", "likelihood.inversion_filter"])

# options the filters accept only at their default value (Dynare acts on them; JDyn does not implement them)
const _OCC_REFUSED = Dict{String,Any}(
    "likelihood.loss_function_regime_guess" => false, "likelihood.number_of_initial_periods_with_extra_regime_guess" => 0,
    "likelihood.restrict_state_space" => true, "filter.particle.status" => false, "filter.init_periods_using_particles" => false,
    "likelihood.init_regime_history" => nothing, "likelihood.init_binding_indicator" => nothing,
    "smoother.init_regime_history" => nothing, "smoother.init_binding_indicator" => nothing,
    "smoother.first_period_occbin_update" => 1, "solver.solve_algo" => 3)

"Option name of the manual (`likelihood_maxit`) or of options_.occbin (`likelihood.maxit`) -> key."
function _occ_key(name::AbstractString)
    occursin('.', name) && return String(name)
    name == "likelihood_max_kalman_iterations" && return "likelihood.max_number_of_iterations"
    for pre in ("likelihood_", "smoother_", "filter_", "simul_", "solver_")
        startswith(name, pre) && return pre[1:end-1] * "." * name[length(pre)+1:end]
    end
    throw(ArgumentError("unknown OccBin option $name"))
end

function _occ_convert(key, v)
    d = get(OCCBIN_FILTER_DEFAULTS, key, missing)
    (d === missing || d === nothing || v === nothing) && return v
    d isa Bool && return _asbool(v)
    d isa Integer && return Int(round(_asnum(v)))
    d isa AbstractFloat && return _asnum(v)
    v
end

function _occ_set!(o::Dict{String,Any}, k::AbstractString, v)
    if k in ("likelihood_piecewise_kalman_filter", "smoother_piecewise_kalman_filter")
        o[split(k, "_")[1] * ".inversion_filter"] = !_asbool(v)
        return o
    end
    key = _occ_key(String(k))
    o[key] = _occ_convert(key, v)
    o
end

"""
    occbin_filter_options(m::Model, overrides=nothing; what="OccBin filtering") -> Dict{String,Any}

OccBin likelihood, filter and smoother options as Dynare 7.2 has them in `options_.occbin` after the .mod file's
statements: `set_default_options.m`, then for each `occbin_setup` in order the options `occbin.setup` copies (the
others have no effect in Dynare 7.2 and are reported once with a warning), and the `simul_*` options of
`occbin_solver`. `overrides` (keys `"likelihood_maxit"` or `"likelihood.maxit"`) are applied last; options Dynare acts
on but JDyn does not implement (`loss_function_regime_guess`, particle filters, `init_regime_history`, ...) raise.
"""
function occbin_filter_options(m::Model, overrides=nothing; what="OccBin filtering")
    o = copy(OCCBIN_FILTER_DEFAULTS)
    dropped = String[]
    for st in m.statements
        kind = get(st, "statementName", "")
        kind in ("occbin_setup", "occbin_solver") || continue
        for (k, v) in get_or(st, "options", Dict{String,Any}())
            if kind == "occbin_solver"
                startswith(k, "simul.") && (o[k] = _occ_convert(k, v))
            elseif k in _OCC_SETUP_COPIED
                o[k] = _occ_convert(k, v)
            elseif k in ("likelihood.piecewise_kalman_filter", "smoother.piecewise_kalman_filter")
                push!(dropped, k)                 # not copied by occbin.setup either
            elseif !startswith(k, "simul.")
                push!(dropped, k)
            end
        end
    end
    isempty(dropped) || warn_once(m, "$what: these occbin_setup options have no effect (Dynare 7.2's occbin.setup does not copy them " *
                                     "into options_.occbin, so they are ignored here too; pass them as overrides to use them): " *
                                     join(sort(unique(dropped)), ", "))
    if overrides !== nothing
        for (k, v) in overrides
            _occ_set!(o, string(k), v)
        end
    end
    _occ_check_options(o)
end

function _occ_check_options(o::Dict{String,Any})
    for (k, dflt) in _OCC_REFUSED
        v = get(o, k, dflt)
        same = dflt === nothing ? (v === nothing || (v isa AbstractArray && isempty(v))) : isequal(v, dflt)
        same || throw(EstimationError("OccBin option $k = $v is not implemented in JDyn (only $(dflt === nothing ? "empty" : dflt))"))
    end
    for k in ("likelihood.max_check_ahead_periods", "smoother.max_check_ahead_periods")
        o[k] === nothing && (o[k] = Inf)
    end
    o
end

"occbin.solver options of the likelihood or the smoother (set_occbin_options of dsge_likelihood.m, DSGE_smoother.m)."
function _occ_sim(o::Dict{String,Any}, prefix::AbstractString)
    Dict{Symbol,Any}(:periods => Int(o["$prefix.periods"]), :maxit => Int(o["$prefix.maxit"]),
        :check_ahead_periods => Int(o["$prefix.check_ahead_periods"]),
        :max_check_ahead_periods => Float64(o["$prefix.max_check_ahead_periods"]),
        :curb_retrench => Bool(o["$prefix.curb_retrench"]), :periodic_solution => Bool(o["$prefix.periodic_solution"]),
        :periodic_solution_threshold => Int(o["simul.periodic_solution_threshold"]),
        :periodic_solution_strict => Bool(o["simul.periodic_solution_strict"]),
        :reset_check_ahead_periods_in_new_period => Bool(o["simul.reset_check_ahead_periods_in_new_period"]),
        :reset_regime_in_new_period => Bool(o["simul.reset_regime_in_new_period"]))
end

# ====================================================================== regimes
base_regime(k::Integer) = OccRegime([([0], [1]) for _ in 1:k])
_copy_regime(r) = OccRegime([(copy(a), copy(b)) for (a, b) in r])

"map_regime.m: 0/1 vector -> (regime, regimestart)."
map_regime(ind::AbstractVector) = isempty(ind) ? ([0], [1]) : _map_regime(ind)

"backward_map_regime.m: binding indicator of length `start[end]` (the last regime's own period is left at 0, as in Dynare)."
function backward_map_regime(regime::AbstractVector, start::AbstractVector)
    ind = falses(Int(start[end]))
    for i in 1:length(regime)-1
        ind[Int(start[i]):Int(start[i+1])-1] .= regime[i] != 0
    end
    ind
end

"Regime one period later (kalman_update_algo_1.m: regimes0(2:3) from regimes0(1))."
_shift_regime(reg) = OccRegime([map_regime(backward_map_regime(r, s)[2:end]) for (r, s) in reg])

_max_start(regs) = maximum(maximum(s) for reg in regs for (_, s) in reg)

"Binding indicator (periods x constraints): whether each constraint binds in the period a regime-history entry refers to."
regime_indicator(rh) = isempty(rh) ? zeros(Int, 0, 0) : [rh[t][j][1][1] for t in eachindex(rh), j in eachindex(rh[1])]

# ====================================================================== solver
"occbin.solver for the filters: `(error_flag, result)`, Dynare's error flag instead of an exception."
function _occ_solve(ob::Occbin, shocks::AbstractMatrix, init, sim::Dict{Symbol,Any}; init_regime=nothing, exo_pos=nothing)
    try
        return 0, simulate(ob, shocks; init=init, init_regime=init_regime, exo_pos=exo_pos, sim...)
    catch e
        e isa OccbinError && return error_flag(e), nothing
        e isa Union{LinearAlgebra.SingularException,LinearAlgebra.LAPACKException,DomainError} && return 311, nothing
        rethrow()
    end
end

"(T, R, C) of the first period of regime `reg` (check_regimes.m: occbin.solver with `init_regime` and `maxit = 1`)."
function _regime_rule(ob::Occbin, reg, sim::Dict{Symbol,Any})
    reg == base_regime(ob.k) && return (ob.P, ob.Q, zeros(ob.n))
    nper = _max_start([reg])
    kw = copy(sim); kw[:maxit] = 1; kw[:periods] = nper
    flag, res = _occ_solve(ob, zeros(1, ob.n_exo), nothing, kw; init_regime=[reg])
    flag == 0 && return res.ss[1]
    # Dynare keeps the matrices computed before the error (312: binding at the horizon)
    ind = falses(nper + 1, ob.k)
    for (j, (r, s)) in enumerate(reg)
        b = backward_map_regime(r, s)
        ind[1:length(b), j] = b
    end
    rules = _rules(ob, ind, _tmax(ind))
    isempty(rules) ? (ob.P, ob.Q, zeros(ob.n)) : rules[1]
end

# ====================================================================== csolve
"""
csolve.m (Sims): Newton steps with backtracking on sum(|f|). `gradfun(x)` returns (f, grad). Returns (x, rc), rc = 0 on
success. The random search (when no progress is made) draws from `rng`.
"""
function csolve(fun, x0::AbstractVector, gradfun, crit, itmax; rng=Random.default_rng())
    delta, alpha = 1e-6, 1e-3
    x = Vector{Float64}(x0)
    nv = length(x)
    tvec = delta * Matrix{Float64}(I, nv, nv)
    f0 = fun(x)
    af0 = sum(abs, f0)
    af00 = af0
    itct = 0
    rc = 0
    while true
        if itct > 3 && af00 - af0 < crit * max(1.0, af0) && isodd(itct)
            randomize = true
            dx0 = zeros(nv)
        else
            f0, grad = gradfun(x)
            grad = Matrix{Float64}(grad)
            randomize = false
            rc_ = all(isfinite, grad) ? _rcond(grad) : 0.0
            (!isfinite(rc_) || rc_ < 1e-12) && (grad = grad + tvec)
            dx0 = if all(isfinite, grad)
                try
                    -(grad \ f0)
                catch e
                    e isa LinearAlgebra.SingularException || rethrow()
                    -(pinv(grad) * f0)
                end
            else
                fill(NaN, nv)
            end
        end
        randomize && (dx0 = norm(x) ./ randn(rng, nv))
        lam = 1.0
        fmin, xmin, afmin = f0, x, af0
        dxsize = norm(dx0)
        factor, shrink, subdone = 0.6, true, false
        while !subdone
            dx = lam * dx0
            f = fun(x + dx)
            af = sum(abs, f)
            if af < afmin
                afmin, fmin, xmin = af, f, x + dx
            end
            if (lam > 0 && af0 - af < alpha * lam * af0) || (lam < 0 && af0 - af < 0)
                if !shrink
                    factor = factor^0.6
                    shrink = true
                end
                if abs(lam * (1 - factor)) * dxsize > 0.1 * delta
                    lam = factor * lam
                elseif lam > 0 && factor == 0.6
                    lam = -0.3
                else
                    subdone = true
                    rc = lam > 0 ? (factor == 0.6 ? 2 : 1) : 3
                end
            elseif lam > 0 && af - af0 > (1 - alpha) * lam * af0
                if shrink
                    factor = factor^0.6
                    shrink = false
                end
                lam = lam / factor
            else
                subdone, rc = true, 0
            end
            (!isfinite(lam) || !isfinite(dxsize)) && (subdone = true; rc = 3)
        end
        itct += 1
        x, f0 = xmin, fmin
        af00, af0 = af0, afmin
        itct >= itmax && return x, 4
        af0 < crit && return x, 0
    end
end

# ====================================================================== inversion filter
"""
    InversionFilterResult

Output of `inversion_filter` (Dynare's IVF_core): `filtered_errs` (T, n_obs) shocks (columns as `err_index`), `resids`
(T, n_obs) observation residuals at the solution, `Emat[t]` (n_endo, n_obs) response of the variables to the shocks,
`stateval` (T, n_endo) deviations from the steady state (declaration order), `regime_history`, `error_code` (Dynare's
303/304/306, 0 if fine) and `error_period`. After `inversion_filter_loglik`: `LIK` (minus the log likelihood) and `lik`.
"""
mutable struct InversionFilterResult
    filtered_errs::Matrix{Float64}
    resids::Matrix{Float64}
    Emat::Vector{Matrix{Float64}}
    stateval::Matrix{Float64}
    regime_history::Vector{OccRegime}
    error_code::Int
    error_period::Int
    penalty::Float64
    LIK::Float64
    lik::Vector{Float64}
    occbin::Union{Nothing,Occbin}
end

"""
    inversion_filter(ob, data, obs_idx, err_index, options; rng) -> InversionFilterResult

IVF_core.m: shocks that reproduce `data` (T x n_obs, levels, NaN = missing) period by period through the piecewise-linear
solution. `obs_idx`: declaration indices of the observed variables; `err_index`: exogenous indices of the shocks mapped to
them (Dynare's `IVF_shock_observable_mapping`). Each period is solved with `csolve` (`solver.maxit` iterations, tolerance
`solver.solve_tolf` on sum(|residual|)) from zero shocks, as Dynare's `dynare_solve` with `solve_algo = 3`.
"""
function inversion_filter(ob::Occbin, data::AbstractMatrix, obs_idx::AbstractVector{<:Integer}, err_index::AbstractVector{<:Integer},
                          o::Dict{String,Any}=copy(OCCBIN_FILTER_DEFAULTS); rng=Random.default_rng())
    T, nobs = size(data)
    n = ob.n
    sim = _occ_sim(o, "likelihood")
    tolf, maxit_solve = Float64(o["solver.solve_tolf"]), Int(o["solver.maxit"])
    resids = zeros(T, nobs)
    stateval = zeros(T, n)
    Emat = [zeros(n, nobs) for _ in 1:T]
    errs = zeros(T, nobs)
    regimes = OccRegime[]
    init = zeros(n)
    out(code, t, pen) = InversionFilterResult(errs, resids, Emat, stateval, regimes, code, t, pen, Inf, Float64[], ob)
    for t in 1:T
        inan = .!isnan.(view(data, t, :))
        cur = data[t, inan]
        oi, ep = obs_idx[inan], err_index[inan]
        y0 = copy(init)
        cache = Dict{Vector{Float64},Any}()
        function match(err)
            key = Vector{Float64}(err)
            haskey(cache, key) && return cache[key]
            flag, res = _occ_solve(ob, reshape(key, 1, :), y0, sim; exo_pos=ep)
            r = if flag != 0
                (fill(100.0, length(oi)), fill(NaN, length(oi), length(ep)), nothing, nothing, nothing)
            else
                _, Rr, _ = res.ss[1]
                (res.piecewise[1, oi] .- cur, Rr[oi, ep], res.piecewise[1, :] .- ob.ys, Rr[:, ep], res)
            end
            cache[key] = r
            r
        end
        x0 = zeros(count(inan))
        f0 = match(x0)[1]
        x, rc = if all(isfinite, f0) && maximum(abs, f0; init=0.0) < tolf
            x0, 0                                     # dynare_solve: the initial guess solves the problem
        else
            csolve(e -> match(e)[1], x0, e -> match(e)[1:2], tolf, maxit_solve; rng=rng)
        end
        rc != 0 && return out(304, t, 1000.0)
        errs[t, inan] = x
        r_, _, state, E, res = match(x)
        res === nothing && return out(304, t, 1000.0)
        resids[t, inan] = r_
        stateval[t, :] = state
        Emat[t][:, inan] = E
        init = state
        push!(regimes, res.regime_history[1])
        maximum(abs, x; init=0.0) > 1e8 && return out(306, t, maximum(abs, x) / 1000)
        maximum(abs, view(resids, t, :); init=0.0) > 1e-3 && return out(303, t, maximum(abs, view(resids, t, :)) * 100)
    end
    out(0, 0, 0.0)
end

"""
    inversion_filter_loglik!(ivf, data, obs_idx, Sigma_e, err_index; presample=0) -> (code, penalty)

IVF_posterior.m: sets `ivf.LIK` (minus the log likelihood after `presample`) and `ivf.lik` (per-period contributions);
returns Dynare's code (0, or 301 Inf, 302 NaN, 303 residuals above 1e-3) and its penalty.
"""
function inversion_filter_loglik!(ivf::InversionFilterResult, data::AbstractMatrix, obs_idx, Sigma_e::AbstractMatrix, err_index;
                                  presample::Integer=0)
    COV = Sigma_e[err_index, err_index]
    e = ivf.filtered_errs
    T = size(e, 1)
    likei = fill(NaN, T)
    logdet_(A) = (d = det(A); d > 0 ? log(d) : d == 0 ? -Inf : NaN)
    if !any(isnan, data)
        ldc = logdet_(COV)
        for t in 1:T
            G = ivf.Emat[t][obs_idx, :]
            likei[t] = ldc + 2 * log(abs(det(G))) + dot(e[t, :], COV \ e[t, :])
        end
        likei .+= length(obs_idx) * LOG2PI
    else
        for t in 1:T
            inan = .!isnan.(view(data, t, :))
            C = COV[inan, inan]
            G = ivf.Emat[t][obs_idx[inan], findall(inan)]
            likei[t] = logdet_(C) + 2 * log(abs(det(G))) + dot(e[t, inan], C \ e[t, inan]) + count(inan) * LOG2PI
        end
    end
    like = 0.5 * sum(likei[presample+1:end])
    ivf.LIK, ivf.lik = like, 0.5 .* likei
    isinf(like) && return 301, 1000.0
    isnan(like) && return 302, 1000.0
    maximum(abs, ivf.resids; init=0.0) > 1e-3 && return 303, sum(abs2, ivf.resids)
    0, 0.0
end

# ====================================================================== piecewise Kalman filter
"What the PKF needs besides the filter quantities: solver, state space (`r`: declaration indices of the filter states in
Dynare's decision-rule order), selection matrix `Z`, covariances and options; `prefix` selects `likelihood.*` or `smoother.*`."
mutable struct _PKFContext
    ob::Occbin
    r::Vector{Int}
    Z::Matrix{Float64}
    H::Matrix{Float64}
    Q::Matrix{Float64}
    o::Dict{String,Any}
    k::Int
    base::OccRegime
    sim::Dict{Symbol,Any}
    T0::Matrix{Float64}
    R0::Matrix{Float64}
    mm::Int
    nexo::Int
    n::Int
    kalman_tol::Float64
    rescale::Bool
end
function _PKFContext(ob::Occbin, r, Z, H, Q, o::Dict{String,Any}; prefix="likelihood", kalman_tol=1e-10, rescale=false)
    r = Vector{Int}(r)
    _PKFContext(ob, r, Matrix{Float64}(Z), Matrix{Float64}(H), Matrix{Float64}(Q), o, ob.k, base_regime(ob.k), _occ_sim(o, prefix),
                ob.P[r, r], ob.Q[r, :], length(r), ob.n_exo, ob.n, Float64(kalman_tol), rescale)
end
_rule(ctx::_PKFContext, trc) = (trc[1][ctx.r, ctx.r], trc[2][ctx.r, :], trc[3][ctx.r])

_rank_deficient(F) = !isempty(F) && rank(F) < size(F, 1)
_logdet_(A) = (d = det(A); d > 0 ? log(d) : d == 0 ? -Inf : NaN)

"The two-period problem of one PKF update (t-1 filtered, t new): kalman_update_algo_1 (multivariate) or _3 (univariate)."
mutable struct _Step
    ctx::_PKFContext
    a_prev::Vector{Float64}
    a1_1::Vector{Float64}
    P_prev::Matrix{Float64}
    P1_1::Matrix{Float64}
    di1::Vector{Int}
    di2::Vector{Int}
    y2::Vector{Float64}
    TT2::Matrix{Float64}
    RR2::Matrix{Float64}
    CC2::Vector{Float64}
    v1::Vector{Float64}
    Fi1::Vector{Float64}
    Ki1::Matrix{Float64}
    univariate::Bool
    pre_error::Int
    iF1::Matrix{Float64}
    L1::Matrix{Float64}
end
function _Step(ctx, a_prev, a1_1, P_prev, P1_1, di1, di2, y2, TT2, RR2, CC2, v1, Fi1, Ki1, univariate::Bool)
    s = _Step(ctx, Vector{Float64}(a_prev), Vector{Float64}(a1_1), Matrix{Float64}(P_prev), Matrix{Float64}(P1_1), Vector{Int}(di1),
              Vector{Int}(di2), Vector{Float64}(y2), TT2, RR2, CC2, Vector{Float64}(v1), Vector{Float64}(Fi1), Matrix{Float64}(Ki1),
              univariate, 0, zeros(0, 0), zeros(0, 0))
    if !univariate
        Z, H, mm = ctx.Z, ctx.H, ctx.mm
        ZZ1 = Z[s.di1, :]
        F = ZZ1 * s.P1_1 * ZZ1' + H[s.di1, s.di1]
        if any(isnan, F)
            s.pre_error = 325
        elseif _rank_deficient(F)
            s.pre_error = 326
        elseif !isempty(s.di1)
            sig = sqrt.(diag(F))
            ss = sig * sig'
            s.iF1 = inv(F ./ ss) ./ ss
            s.L1 = Matrix{Float64}(I, mm, mm) - s.P1_1 * ZZ1' * s.iF1 * ZZ1
        else
            s.iF1, s.L1 = zeros(0, 0), Matrix{Float64}(I, mm, mm)
        end
    end
    s
end

_update(s::_Step, T2, R2, C2) = s.univariate ? _update_uni(s, T2, R2, C2) : _update_multi(s, T2, R2, C2)

"occbin_kalman_update0 of kalman_update_algo_1.m."
function _update_multi(s::_Step, T2, R2, C2)
    ctx = s.ctx
    Z, H, Q, mm = ctx.Z, ctx.H, ctx.Q, ctx.mm
    IFl = Bool(ctx.o["likelihood.IF_likelihood"])
    di2 = s.di2
    QQ = R2 * Q * R2'
    a1_2 = T2 * s.a_prev + C2
    a_2 = copy(a1_2)
    P1_2 = T2 * s.P_prev * T2' + QQ
    P_2 = copy(P1_2)
    v2 = zeros(size(Z, 1))
    iF2 = zeros(0, 0)
    lik = 0.0
    if !isempty(di2)
        ZZ = Z[di2, :]
        v2[di2] = s.y2[di2] - ZZ * a_2
        F = ZZ * P_2 * ZZ' + H[di2, di2]
        any(isnan, F) && return (error_flag=325, lik=Inf)
        _rank_deficient(F) && return (error_flag=326, lik=Inf)
        if ctx.rescale
            sig = sqrt.(diag(F))
            ss = sig * sig'
            log_dF = _logdet_(F ./ ss) + 2 * sum(log.(sig))
            iF2 = inv(F ./ ss) ./ ss
        else
            log_dF = _logdet_(F)
            iF2 = inv(F)
        end
        lik = IFl ? Inf : log_dF + dot(v2[di2], iF2 * v2[di2]) + length(di2) * LOG2PI
        PZI = P_2 * ZZ' * iF2
        a_2 = a_2 + PZI * v2[di2]
        P_2 = P_2 - PZI * ZZ * P_2
    end
    r2 = isempty(di2) ? zeros(mm) : Z[di2, :]' * (iF2 * v2[di2])
    alpha2 = a1_2 + P1_2 * r2
    eta2 = Q * R2' * r2
    if IFl && !isempty(di2)
        ZZ = Z[di2, :]
        ish = vec(any(ZZ * R2 .!= 0; dims=1)) .& (eta2 .!= 0)
        G = ZZ * R2[:, ish]
        lik = if size(G, 1) == size(G, 2)
            Qs = Q[ish, ish]
            _logdet_(Qs) + 2 * log(abs(det(G))) + dot(eta2[ish], Qs \ eta2[ish]) + length(di2) * LOG2PI
        else
            Inf
        end
    end
    r1 = s.L1' * (T2' * r2)
    isempty(s.di1) || (r1 = r1 + Z[s.di1, :]' * (s.iF1 * s.v1[s.di1]))
    alpha1 = s.a1_1 + s.P1_1 * r1
    (error_flag=0, a=hcat(s.a_prev, a_2), a1=hcat(s.a1_1, a1_2), P_upd=P_2, P1_2=P1_2, v2=v2, Fi2=nothing, Ki2=nothing,
     alphahat=hcat(alpha1, alpha2), etahat2=eta2, lik=lik)
end

"occbin_kalman_update of kalman_update_algo_3.m (observations processed one by one)."
function _update_uni(s::_Step, T2, R2, C2)
    ctx = s.ctx
    Z, H, Q, mm = ctx.Z, ctx.H, ctx.Q, ctx.mm
    tol = ctx.kalman_tol
    Hd = diag(H)
    di2 = s.di2
    pp = size(Z, 1)
    QQ = R2 * Q * R2'
    a1_2 = T2 * s.a_prev + C2
    a_2 = copy(a1_2)
    P1_2 = T2 * s.P_prev * T2' + QQ
    P_2 = copy(P1_2)
    v2, Fi2, Ki2 = zeros(pp), zeros(pp), zeros(mm, pp)
    lik = 0.0
    vv, F, sig = Float64[], zeros(0, 0), Float64[]
    if !isempty(di2)
        vv = s.y2[di2] - Z[di2, :] * a_2
        F = Z[di2, :] * P_2 * Z[di2, :]' + Diagonal(Hd[di2])
        sig = sqrt.(diag(F))
    end
    for i in di2
        Zi = Z[i, :]
        v2[i] = s.y2[i] - dot(Zi, a_2)
        Fi2[i] = dot(Zi, P_2 * Zi) + Hd[i]
        Ki2[:, i] = P_2 * Zi
        if Fi2[i] > tol
            a_2 = a_2 + Ki2[:, i] * v2[i] / Fi2[i]
            P_2 = P_2 - Ki2[:, i] * Ki2[:, i]' / Fi2[i]
        end
    end
    if !isempty(di2)
        ss = sig * sig'
        log_dF = _logdet_(F ./ ss) + 2 * sum(log.(sig))
        iF = inv(F ./ ss) ./ ss
        lik = log_dF + dot(vv, iF * vv) + length(di2) * LOG2PI
    end
    ri = zeros(mm)
    for i in reverse(di2)
        if Fi2[i] > tol
            Zi = Z[i, :]
            ri = Zi * v2[i] / Fi2[i] + ri - Zi * dot(Ki2[:, i], ri) / Fi2[i]
        end
    end
    alpha2 = a1_2 + P1_2 * ri
    eta2 = Q * R2' * ri
    ri = T2' * ri
    for i in reverse(s.di1)
        if s.Fi1[i] > tol
            Zi = Z[i, :]
            ri = Zi * s.v1[i] / s.Fi1[i] + ri - Zi * dot(s.Ki1[:, i], ri) / s.Fi1[i]
        end
    end
    alpha1 = s.a1_1 + s.P1_1 * ri
    (error_flag=0, a=hcat(s.a_prev, a_2), a1=hcat(s.a1_1, a1_2), P_upd=P_2, P1_2=P1_2, v2=v2, Fi2=Fi2, Ki2=Ki2,
     alphahat=hcat(alpha1, alpha2), etahat2=eta2, lik=lik)
end

"Output of kalman_update_algo_1/3 and kalman_update_engine for one period."
Base.@kwdef mutable struct _PKFUpdate
    error_flag::Int = 0
    lik::Float64 = Inf
    regimes::Vector{OccRegime} = OccRegime[]       # periods t, t+1, t+2
    a::Union{Nothing,Matrix{Float64}} = nothing      # (mm, 2): updated state at t, predicted state at t+1 (piecewise)
    a1::Union{Nothing,Matrix{Float64}} = nothing     # (mm, 2): a1[:, 2] = prediction for t under the final regime
    P::Union{Nothing,Tuple{Matrix{Float64},Matrix{Float64}}} = nothing
    P1_2::Union{Nothing,Matrix{Float64}} = nothing
    v2::Union{Nothing,Vector{Float64}} = nothing
    Fi2::Union{Nothing,Vector{Float64}} = nothing
    Ki2::Union{Nothing,Matrix{Float64}} = nothing
    T::Any = nothing                                 # (T_t, T_t+1), filter state space
    R::Any = nothing
    C::Any = nothing
    etahat::Union{Nothing,Vector{Float64}} = nothing
    alphahat::Union{Nothing,Matrix{Float64}} = nothing
end

"kalman_update_algo_1.m / kalman_update_algo_3.m. `regimes0`: nothing (start from the linear model) or regimes (period t first)."
function _algo(ctx::_PKFContext, step::_Step; regimes0=nothing, guess_regime::Bool=false)
    o, base = ctx.o, ctx.base
    uni = step.univariate
    fail(code) = _PKFUpdate(error_flag=code, lik=Inf, regimes=[base, base, base])
    step.pre_error != 0 && return fail(step.pre_error)
    if regimes0 !== nothing
        regimes0 = OccRegime[_copy_regime(r) for r in regimes0]
        if guess_regime
            T2, R2, C2 = _rule(ctx, _regime_rule(ctx.ob, regimes0[1], ctx.sim))
            r1 = _shift_regime(regimes0[1])
            regimes0 = [regimes0[1], r1, _shift_regime(r1)]
        else
            T2, R2, C2 = step.TT2, step.RR2, step.CC2
        end
    else
        T2, R2, C2 = ctx.T0, ctx.R0, zeros(ctx.mm)
        regimes0 = [base, base, base]
    end
    up = _update(step, T2, R2, C2)
    up.error_flag != 0 && return fail(up.error_flag)
    r = ctx.r
    sim = copy(ctx.sim)
    shocks = zeros(3, ctx.nexo)
    shocks[1, :] = up.etahat2
    init_of(alpha1) = (y0 = zeros(ctx.n); y0[r] = alpha1; y0)
    init = init_of(up.alphahat[:, 1])
    if guess_regime
        flag, out = _occ_solve(ctx.ob, shocks, init, sim; init_regime=regimes0)
        flag != 0 && ((flag, out) = _occ_solve(ctx.ob, shocks, init, sim))
    else
        flag, out = _occ_solve(ctx.ob, shocks, init, sim)
        flag != 0 && ((flag, out) = _occ_solve(ctx.ob, shocks, init, sim; init_regime=regimes0))
    end
    flag != 0 && return fail(flag)
    regs = out.regime_history
    regime_hist = OccRegime[regimes0[1]]
    lik_hist = Float64[up.lik]
    niter = 1
    is_periodic = false
    maxiter = Int(o["likelihood.max_number_of_iterations"])
    any_binding = any(any(!=(0), rg) for reg in regs for (rg, _) in reg)
    if any_binding || regs[1] != regimes0[1]
        while regs[1] != regimes0[1] && !is_periodic && niter <= maxiter
            niter += 1
            T2, R2, C2 = _rule(ctx, out.ss[1])
            push!(regime_hist, regs[1])
            up = _update(step, T2, R2, C2)
            up.error_flag != 0 && return _PKFUpdate(error_flag=up.error_flag, lik=up.lik, regimes=regs)
            push!(lik_hist, up.lik)
            shocks[1, :] = up.etahat2
            init = init_of(up.alphahat[:, 1])
            sim[:periods] = max(sim[:periods], _max_start(regs))
            flag, out2 = _occ_solve(ctx.ob, shocks, init, sim; init_regime=[regs[1]])
            (flag != 0 && !uni) && ((flag, out2) = _occ_solve(ctx.ob, shocks, init, sim))     # algo_3 does not retry
            flag != 0 && return _PKFUpdate(error_flag=flag, lik=Inf, regimes=regs)
            out = out2
            regimes0 = regs
            regs = out.regime_history
            hits = [kk for kk in 1:niter-1 if regime_hist[kk] == regs[1]]
            if !isempty(hits)
                is_periodic = true
                if Bool(o["filter.periodic_solution"])
                    i0 = hits[1]
                    im = argmin(lik_hist[i0:end]) - 1
                    sim[:periods] = max(sim[:periods], _max_start(regimes0))
                    sim[:maxit] = 1
                    flag, out2 = _occ_solve(ctx.ob, shocks, init, sim; init_regime=[regime_hist[i0+im]])
                    flag != 0 && return _PKFUpdate(error_flag=flag, lik=Inf, regimes=regs)
                    out = out2
                    regs = out.regime_history
                    T2, R2, C2 = _rule(ctx, out.ss[1])
                    up = _update(step, T2, R2, C2)
                    up.error_flag != 0 && return _PKFUpdate(error_flag=up.error_flag, lik=up.lik, regimes=regs)
                else
                    return _PKFUpdate(error_flag=330, lik=Inf, regimes=regs)
                end
            end
        end
    end
    error_flag = (niter > maxiter && regs[1] != regimes0[1]) ? 331 : 0
    lik = error_flag != 0 ? Inf : up.lik
    a = Matrix((out.piecewise[1:2, r] .- ctx.ob.ys[r]')')
    T1, R1, C1 = _rule(ctx, out.ss[1])
    T2n, R2n, C2n = _rule(ctx, out.ss[2])
    P_t = up.P_upd
    P_next = T2n * P_t * T2n' + R2n * ctx.Q * R2n'
    _PKFUpdate(error_flag=error_flag, lik=lik, regimes=regs[1:3], a=a, a1=up.a1, P=(P_t, P_next), P1_2=up.P1_2, v2=up.v2,
               Fi2=up.Fi2, Ki2=up.Ki2, T=(T1, T2n), R=(R1, R2n), C=(C1, C2n), etahat=up.etahat2, alphahat=up.alphahat)
end

"kalman_update_engine.m: `regimes_pair` are the regimes predicted for t and t+1 by the previous period."
function _engine(ctx::_PKFContext, t::Int, step::_Step, regimes_pair)
    o, base = ctx.o, ctx.base
    brute = Bool(o["likelihood.brute_force_regime_guess"])
    brute_extra = Bool(o["likelihood.brute_force_extra_regime_guess"])
    loss_fn = Bool(o["likelihood.loss_function_regime_guess"])
    best = _algo(ctx, step)
    likvec, regvec = Float64[], OccRegime[]
    if best.error_flag == 0
        likvec, regvec = [best.lik], [best.regimes[1]]
    else
        best.lik = Inf
    end
    info0 = best.error_flag
    info1 = 0
    is_base_pair = regimes_pair[1] == base && regimes_pair[2] == base
    if best.error_flag != 0
        is_base_pair || (best = _algo(ctx, step; regimes0=[regimes_pair[1], regimes_pair[2]]))
        if best.error_flag == 0
            likvec, regvec = [best.lik], [best.regimes[1]]
        else
            best.lik = Inf
        end
        info1 = best.error_flag
    else
        if !is_base_pair
            alt = _algo(ctx, step; regimes0=[regimes_pair[1], regimes_pair[2]])
            info1 = alt.error_flag
            if alt.error_flag == 0 && alt.regimes != best.regimes
                push!(likvec, alt.lik); push!(regvec, alt.regimes[1])
            end
            (alt.error_flag == 0 && alt.lik < best.lik) && (best = alt)
        else
            if t > Int(o["likelihood.number_of_initial_periods_with_extra_regime_guess"])
                info1 = 0
            else
                info1 = 1
                brute, loss_fn = true, true
            end
        end
    end
    t < Int(o["likelihood.first_period_binding_regime_allowed"]) && return best
    diffstart, which = 0, 1
    if best.error_flag == 0 && length(regvec) == 1
        diffs = [best.regimes[1][j][2][end] - regimes_pair[1][j][2][end] for j in 1:ctx.k]
        which = argmax(diffs)
        diffstart = diffs[which]
    end
    if Bool(o["filter.use_relaxation"]) && diffstart > o["filter.use_relaxation_tol_period"]
        guess = _copy_regime(best.regimes[1])
        regR = best.regimes
        while guess == regR[1]
            regR[1][which][2][end] <= 1 && break
            st = guess[which][2]
            st[end] = regR[1][which][2][end] - 1
            if st[end] == 1
                guess[which] = ([0], [st[end]])
            elseif length(st) > 1 && st[end-1] == st[end]
                st[end-1] -= 1
            end
            resR = _algo(ctx, step; regimes0=[_copy_regime(guess), base], guess_regime=true)
            regR = resR.regimes
            if !(regR[1] in regvec)
                push!(likvec, resR.lik); push!(regvec, regR[1])
            end
            (resR.error_flag == 0 && resR.lik < best.lik) && (best = resR)
        end
        brute = loss_fn = false
    end
    if (brute && (info0 != 0 && info1 != 0)) || (brute_extra && (info0 != 0 || info1 != 0))
        use, info2 = nothing, 1
        if ctx.k == 1
            for kk in 1:5
                c1 = _algo(ctx, step; regimes0=[OccRegime([([0, 1, 0], [1, 5, 5 + 4kk])]), base], guess_regime=true)
                info2 = c1.error_flag
                if info2 == 0
                    use = c1
                    (best.error_flag == 0 && c1.regimes == best.regimes) || break
                end
                c2 = _algo(ctx, step; regimes0=[OccRegime([([1, 0], [1, 1 + 4kk])]), base], guess_regime=true)
                info2 = c2.error_flag
                info2 == 0 && (use = c2)
                if use !== nothing
                    info2 = 0
                    break
                end
            end
        else
            use, info2 = _brute_force_two(ctx, step, best)
        end
        if info2 == 0 && use !== nothing
            info0 = info1 = 0
            if !(use.regimes[1] in regvec)
                push!(regvec, use.regimes[1]); push!(likvec, use.lik)
            end
            use.lik < best.lik && (best = use)
        end
    end
    (loss_fn && (info0 != 0 || info1 != 0)) &&
        throw(EstimationError("the loss-function regime guess of the piecewise Kalman filter (occbin.findmin) is not implemented"))
    if length(likvec) > 1
        mn = minimum(likvec)
        best.lik = -2 * log(sum(exp.(-(likvec .- mn) ./ 2))) + mn
    end
    if best.error_flag == 0
        if isnan(best.lik)
            best.error_flag = 323
        elseif any(isnan, best.a)
            best.error_flag = 324
        end
    end
    best
end

"The two-constraint brute-force regime guesses of kalman_update_engine.m."
function _brute_force_two(ctx::_PKFContext, step::_Step, best::_PKFUpdate)
    base1 = ([0], [1])
    use, info2, c = nothing, 1, nothing
    new_solution(c) = info2 == 0 && !(best.error_flag == 0 && c.regimes == best.regimes)
    for jk in 0:1
        for kk in 1:5
            for jr in 0:1
                for kind in 1:2
                    cur = kind == 1 ? ([0, 1, 0], [1, 5, 5 + 4kk]) : ([1, 0], [1, 1 + 4kk])
                    for kj in 1:(1 + (jk > 0 ? 1 : 0))
                        other = jk == 0 ? base1 : kj == 1 ? ([1, 0], [1, 1 + 4jk]) : ([0, 1, 0], [1, 5, 5 + 4jk])
                        g = jr == 0 ? OccRegime([cur, other]) : OccRegime([other, cur])
                        c = _algo(ctx, step; regimes0=[g, ctx.base], guess_regime=true)
                        info2 = c.error_flag
                        if info2 == 0
                            use = c
                            new_solution(c) && break
                        end
                    end
                    new_solution(c) && break
                end
                new_solution(c) && break
            end
            use !== nothing && return use, 0
        end
        use !== nothing && return use, 0
    end
    use, info2
end

"""
    PKFResult

Output of the piecewise Kalman filter: `LIK` (minus the log likelihood after the presample), `lik` per-period contributions,
`a0` (mm, T) updated states a_t|t, `a1` (mm, T + 1) predictions a_t|t-1, `P0`/`P1` (T vectors) their covariances, `regimes`
the regime of each period, `error_code`/`error_period` (Dynare's 320-331 OccBin codes), and the regime-dependent matrices
`TT[p]`, `RR[p]`, `CC[p]` of period p (filter state space), innovations `v`, `Fi`, `Ki` (univariate pass).
"""
Base.@kwdef mutable struct PKFResult
    LIK::Float64
    lik::Vector{Float64}
    a0::Matrix{Float64}
    a1::Matrix{Float64}
    P0::Vector{Matrix{Float64}}
    P1::Vector{Matrix{Float64}}
    regimes::Vector{OccRegime}
    error_code::Int = 0
    error_period::Int = 0
    TT::Vector{Matrix{Float64}} = Matrix{Float64}[]
    RR::Vector{Matrix{Float64}} = Matrix{Float64}[]
    CC::Vector{Vector{Float64}} = Vector{Float64}[]
    v::Matrix{Float64} = zeros(0, 0)
    Fi::Matrix{Float64} = zeros(0, 0)
    Ki::Vector{Matrix{Float64}} = Matrix{Float64}[]
    ainit::Vector{Float64} = Float64[]
    Pinit::Matrix{Float64} = zeros(0, 0)
    aK::Matrix{Float64} = zeros(0, 0)
    occbin::Union{Nothing,Occbin} = nothing
end

_all_index(Y) = [findall(!isnan, view(Y, :, t)) for t in 1:size(Y, 2)]

"""
Forward pass of the piecewise Kalman filter: missing_observations_kalman_filter.m with `occbin_.status` (multivariate,
likelihood) or the forward loop of missing_DiffuseKalmanSmootherH3_Z.m with OccBin (univariate, smoother).
`Y` (pp x T) observations minus their steady state (and trends); `a`, `P` the initial prediction and covariance.
"""
function _pkf_forward(ctx::_PKFContext, Y::AbstractMatrix, a::AbstractVector, P::AbstractMatrix; data_index=nothing, presample::Int=0,
                      univariate::Bool=false, regimes_init=nothing, first_period_occbin_update=nothing, check_inf::Bool=true)
    o = ctx.o
    Z, H, Q = ctx.Z, ctx.H, ctx.Q
    pp, last = size(Y)
    mm = ctx.mm
    tol = ctx.kalman_tol
    data_index === nothing && (data_index = _all_index(Y))
    fpou = first_period_occbin_update === nothing ? Int(o["likelihood.first_period_occbin_update"]) : first_period_occbin_update
    use_updated = univariate ? true : Bool(o["likelihood.use_updated_regime"])
    fbra = Int(o["likelihood.first_period_binding_regime_allowed"])
    TT = [copy(ctx.T0) for _ in 1:last+2]
    RR = [copy(ctx.R0) for _ in 1:last+2]
    CC = [zeros(mm) for _ in 1:last+2]
    base = ctx.base
    regimes_ = OccRegime[base for _ in 1:last+3]
    if regimes_init !== nothing
        for (i, r) in enumerate(regimes_init)
            regimes_[i] = r
        end
    end
    a0 = zeros(mm, last)
    a1 = zeros(mm, last + 1)
    aK = zeros(mm, last + 1)          # Dynare's aK(1,:,:): piecewise one-step-ahead predictions
    P0 = [zeros(mm, mm) for _ in 1:last]
    P1 = [zeros(mm, mm) for _ in 1:last]
    vv = zeros(pp, last)
    Fi = zeros(pp, last)
    Ki = [zeros(mm, pp) for _ in 1:last]
    lik = zeros(last)
    P = Matrix{Float64}(tril(P) + tril(P, -1)')
    a = Vector{Float64}(a)
    fail(code, t) = PKFResult(LIK=Inf, lik=lik, a0=a0, a1=a1, P0=P0, P1=P1, regimes=regimes_[1:last], error_code=code,
                              error_period=t, TT=TT, RR=RR, CC=CC, v=vv, Fi=Fi, Ki=Ki)
    ainit, Pinit = zeros(mm), zeros(mm, mm)
    for t in 1:last
        a1[:, t] = a
        P1[t] = P
        C, R, T = CC[t+1], RR[t+1], TT[t+1]
        QQ = R * Q * R'
        t == 1 && (Pinit = copy(P1[1]); ainit = a1[:, 1])
        di = data_index[t]
        if t < fpou || !use_updated
            # the linear branch (Dynare uses it before first_period_occbin_update)
            if isempty(di)
                a = T * a + C
                P = T * P * T' + QQ
            else
                z = Z[di, :]
                v = Y[di, t] - z * a
                F = z * P * z' + H[di, di]
                if t < fpou && _rcond(F) < tol
                    sig = sqrt.(diag(F))
                    (any(diag(F) .< tol) || _rcond(F ./ (sig * sig')) < tol) && return fail(320, t)
                end
                iF = inv(F)
                lik[t] = _logdet_(F) + dot(v, iF * v) + length(di) * LOG2PI
                if t < fpou
                    K = P * z' * iF
                    P0[t] = P - K * z * P
                    a0[:, t] = a + K * v
                    vv[di, t] = v
                    P = T * (P - K * z * P) * T' + QQ
                    a = T * (a + K * v) + C
                    aK[:, t+1] = a
                end
            end
        end
        if t >= fpou
            step = if t == 1
                _Step(ctx, ainit, ainit, Pinit, Pinit, Int[], data_index[1], Y[:, 1], TT[1], RR[1], CC[1], zeros(pp), zeros(pp),
                      zeros(mm, pp), univariate)
            else
                _Step(ctx, a0[:, t-1], a1[:, t-1], P0[t-1], P1[t-1], data_index[t-1], di, Y[:, t], TT[t], RR[t], CC[t], vv[:, t-1],
                      Fi[:, t-1], Ki[t-1], univariate)
            end
            res = _engine(ctx, t, step, (regimes_[t], regimes_[t+1]))
            likx = res.lik
            (!univariate && t < fbra && res.regimes[1] != base) && (likx = Inf)
            res.error_flag != 0 && return fail(res.error_flag, t)
            (check_inf && isinf(likx)) && return fail(320, t)
            use_updated && (lik[t] = likx)
            regimes_[t:t+2] = res.regimes
            a0[:, t] = res.a[:, 1]
            a1[:, t] = res.a1[:, 2]
            a = res.a[:, 2]
            aK[:, t+1] = a
            vv[di, t] = res.v2[di]
            if univariate
                Fi[di, t] = res.Fi2[di]
                Ki[t][:, di] = res.Ki2[:, di]
            end
            TT[t], TT[t+1] = res.T
            RR[t], RR[t+1] = res.R
            CC[t], CC[t+1] = res.C
            P0[t] = res.P[1]
            P1[t] = res.P1_2
            P = res.P[2]
        end
    end
    a1[:, last+1] = a
    lik .*= 0.5
    LIK = sum(lik[presample+1:end])
    PKFResult(LIK=LIK, lik=lik, a0=a0, a1=a1, P0=P0, P1=P1, regimes=regimes_[1:last+1], TT=TT, RR=RR, CC=CC, v=vv, Fi=Fi, Ki=Ki,
              ainit=ainit, Pinit=Pinit, aK=aK)
end

"""
    piecewise_kalman_filter(ctx, Y, a, P; data_index=nothing, presample=0) -> PKFResult

missing_observations_kalman_filter.m with `occbin_.status` (the likelihood of dsge_likelihood.m): `Y` (pp x T)
observations minus their steady state (and trends), `a`/`P` the initial prediction and covariance (period 1). `LIK` is
minus the log likelihood; `regimes` has one entry per period.
"""
function piecewise_kalman_filter(ctx::_PKFContext, Y::AbstractMatrix, a, P; data_index=nothing, presample::Integer=0)
    res = _pkf_forward(ctx, Y, a, P; data_index=data_index, presample=Int(presample))
    res.error_code == 0 && (res.regimes = res.regimes[1:size(Y, 2)])
    res.occbin = ctx.ob
    res
end

# ====================================================================== OccBin smoother
"Backward pass of missing_DiffuseKalmanSmootherH3_Z.m (non-diffuse part) with time-varying TT/RR (TT[t]: rule of period t)."
function _univariate_backward(ctx::_PKFContext, a1, P1, v, Fi, Ki, TT, RR, data_index, ainit, Pinit)
    Z, Q, tol = ctx.Z, ctx.Q, ctx.kalman_tol
    mm, T = size(a1, 1), length(data_index)
    alphahat = zeros(mm, T)
    etahat = zeros(ctx.nexo, T)
    ri = zeros(mm)
    for t in T:-1:1
        for i in reverse(data_index[t])
            if Fi[i, t] > tol
                Zi = Z[i, :]
                ri = Zi * v[i, t] / Fi[i, t] + ri - Zi * dot(Ki[t][:, i], ri) / Fi[i, t]
            end
        end
        alphahat[:, t] = a1[:, t] + P1[t] * ri
        etahat[:, t] = Q * RR[t]' * ri
        ri = TT[t]' * ri
    end
    alphahat0 = ainit + Pinit * ri
    alphahat, etahat, alphahat0
end

"Forward univariate filter of missing_DiffuseKalmanSmootherH3_Z.m with given time-varying matrices (TT[t]: rule of period t)."
function _tv_forward(ctx::_PKFContext, Y, a_init, P_init, data_index, TT, RR, CC)
    Z, Q, tol = ctx.Z, ctx.Q, ctx.kalman_tol
    Hd = diag(ctx.H)
    pp, T = size(Y)
    mm = ctx.mm
    a = zeros(mm, T)
    a1 = zeros(mm, T + 1)
    P1 = [zeros(mm, mm) for _ in 1:T]
    v, Fi, Ki = zeros(pp, T), zeros(pp, T), [zeros(mm, pp) for _ in 1:T]
    an = TT[1] * a_init + CC[1]
    Pn = TT[1] * P_init * TT[1]' + RR[1] * Q * RR[1]'
    for t in 1:T
        a1[:, t] = an
        P1[t] = Pn
        at, Pt = copy(an), copy(Pn)
        for i in data_index[t]
            Zi = Z[i, :]
            v[i, t] = Y[i, t] - dot(Zi, at)
            Fi[i, t] = dot(Zi, Pt * Zi) + Hd[i]
            Ki[t][:, i] = Pt * Zi
            if Fi[i, t] > tol
                at = at + Ki[t][:, i] * v[i, t] / Fi[i, t]
                Pt = Pt - Ki[t][:, i] * Ki[t][:, i]' / Fi[i, t]
                Pt = (Pt + Pt') / 2
            end
        end
        a[:, t] = at
        Tn, Cn = TT[t+1], CC[t+1]
        an = Tn * at + Cn
        Pn = Tn * Pt * Tn' + RR[t+1] * Q * RR[t+1]'
        Pn = (Pn + Pn') / 2
    end
    a1[:, T+1] = an
    a, a1, P1, v, Fi, Ki
end

"""
    OccbinSmootherResult

Output of `occbin_smoother` (Dynare's occbin.DSGE_smoother). States are deviations from the steady state in declaration
order, (T, n_endo); shocks (T, n_exo): `smoothed` (oo_.SmoothedVariables - ys), `shocks` (oo_.SmoothedShocks), `updated` and
`filtered` (the real-time PKF's a_t|t and a_t|t-1), `smoothed0` (period 0), `regime_history` (oo_.occbin.smoother.regime_history),
`realtime_regime_history`, `linear_smoothed`/`linear_shocks` (oo_.occbin.linear_smoother), `error_flag`/`warning_flag`
(oo_.occbin.smoother), `iterations`, `piecewise`/`linear` (simulation with the smoothed shocks, levels;
oo_.occbin.smoother.simul), `TT`/`RR`/`CC` (regime matrices of each period in declaration order, `TT[t]` the rule of period t).
"""
Base.@kwdef mutable struct OccbinSmootherResult
    smoothed::Matrix{Float64}
    shocks::Matrix{Float64}
    updated::Matrix{Float64}
    filtered::Matrix{Float64}
    smoothed0::Vector{Float64}
    regime_history::Vector{OccRegime}
    realtime_regime_history::Vector{OccRegime}
    linear_smoothed::Union{Nothing,Matrix{Float64}} = nothing
    linear_shocks::Union{Nothing,Matrix{Float64}} = nothing
    error_flag::Int = 0
    warning_flag::Int = 0
    iterations::Int = 1
    regime_history_iter::Vector{Vector{OccRegime}} = Vector{OccRegime}[]
    piecewise::Union{Nothing,Matrix{Float64}} = nothing
    linear::Union{Nothing,Matrix{Float64}} = nothing
    ys::Vector{Float64} = Float64[]
    TT::Union{Nothing,Vector{Matrix{Float64}}} = nothing
    RR::Union{Nothing,Vector{Matrix{Float64}}} = nothing
    CC::Union{Nothing,Vector{Vector{Float64}}} = nothing
    converged::Bool = true
    inversion_filter::Bool = false
    endo_names::Vector{String} = String[]
    exo_names::Vector{String} = String[]
end
Base.show(io::IO, r::OccbinSmootherResult) = print(io, "OccbinSmootherResult(", size(r.smoothed, 1), " periods, ",
    r.inversion_filter ? "inversion filter" : "piecewise Kalman smoother", ", error_flag ", r.error_flag, ", warning_flag ", r.warning_flag, ")")

"""
    occbin_smoother(ob, Y, Q, H, obs_idx, options; data_index=nothing, lik_init=1, qz_criterium=1-1e-6,
                    Harvey_scale_factor=10, kalman_tol=1e-10) -> OccbinSmootherResult

Dynare's occbin.DSGE_smoother (with `smoother_redux = false`: all endogenous variables are filter states). `Y` (pp x T) is
the data minus the steady state of the observed variables, `Q` the shock covariance, `H` the measurement-error covariance,
`obs_idx` the declaration indices of the observed variables.

1. linear smoother (`oo_.occbin.linear_smoother`);
2. real-time smoother: univariate piecewise Kalman filter (kalman_update_algo_3), then the backward pass with the
   regime-dependent matrices;
3. simulation with the smoothed shocks from the smoothed initial state and, while its regimes differ from those the
   smoother used, a smoother with the matrices of the new regimes (at most `smoother.max_number_of_iterations` passes),
   with Dynare's "conditional smoother" attempt.
"""
function occbin_smoother(ob::Occbin, Y::AbstractMatrix, Q::AbstractMatrix, H::AbstractMatrix, obs_idx::AbstractVector{<:Integer},
                         options::Dict{String,Any}=copy(OCCBIN_FILTER_DEFAULTS); data_index=nothing, lik_init::Integer=1,
                         qz_criterium::Real=1 - 1e-6, Harvey_scale_factor::Real=10.0, kalman_tol::Real=1e-10)
    o = options
    m = ob.model
    order_var, _, counts = dr_ordering(m.lead_lag_incidence)
    inv = invperm(order_var)
    pp, T = size(Y)
    n = ob.n
    Z = zeros(pp, n)
    for (i, d) in enumerate(obs_idx)
        Z[i, inv[d]] = 1.0
    end
    ctx = _PKFContext(ob, order_var, Z, H, Q, o; prefix="smoother", kalman_tol=kalman_tol)
    data_index === nothing && (data_index = _all_index(Y))
    state_dr = collect(counts.nstatic+1:counts.nstatic+counts.npred+counts.nboth)
    Int(o["smoother.first_period_occbin_update"]) == 1 ||
        throw(EstimationError("smoother_first_period_occbin_update other than 1 is not implemented"))
    T0, R0 = ctx.T0, ctx.R0
    P_init = if lik_init == 1
        lyapunov_symm(T0, R0 * ctx.Q * R0'; qz_criterium=qz_criterium)[1]
    elseif lik_init == 2
        Harvey_scale_factor * Matrix{Float64}(I, n, n)
    else
        throw(EstimationError("the OccBin smoother is available for lik_init = 1, 2"))
    end
    a_init = zeros(n)
    base = ctx.base
    to_decl(x::AbstractMatrix) = x[inv, :]
    to_decl(x::AbstractVector) = x[inv]
    # 1. linear smoother
    TTl = [T0 for _ in 1:T+2]
    RRl = [R0 for _ in 1:T+2]
    CCl = [zeros(n) for _ in 1:T+2]
    la, la1, lP1, lv, lFi, lKi = _tv_forward(ctx, Y, a_init, P_init, data_index, TTl, RRl, CCl)
    lin_alpha, lin_eta, lin_alpha0 = _univariate_backward(ctx, la1, lP1, lv, lFi, lKi, TTl, RRl, data_index, a_init, P_init)
    # 2. real-time smoother
    rt = _pkf_forward(ctx, Y, a_init, P_init; data_index=data_index, univariate=true, check_inf=false)
    rt_ok = rt.error_code == 0
    if rt_ok
        alphahat, etahat, alphahat0 = _univariate_backward(ctx, rt.a1, rt.P1, rt.v, rt.Fi, rt.Ki, rt.TT, rt.RR, data_index,
                                                           rt.ainit, rt.Pinit)
        regime_history = rt.regimes[1:T+1]
        ahat, aK = rt.a0, rt.aK
    else
        alphahat, etahat, alphahat0 = lin_alpha, lin_eta, lin_alpha0
        regime_history = OccRegime[base for _ in 1:T+1]
        ahat, aK = la, la1
    end
    realtime_regimes = copy(regime_history)
    regime_history0 = [copy(regime_history)]
    sim = _occ_sim(o, "smoother")
    init_regime = copy(regime_history)
    function simulate_(eta, alpha0)
        shocks = vcat(Matrix(eta'), zeros(1, ctx.nexo))
        kw = copy(sim); kw[:periods] = size(shocks, 1)
        _occ_solve(ob, shocks, to_decl(alpha0), kw; init_regime=init_regime)
    end
    matrices(res) = ([t_[1][order_var, order_var] for t_ in res.ss], [t_[2][order_var, :] for t_ in res.ss], [t_[3][order_var] for t_ in res.ss])
    error_flag, warning_flag = 0, 0
    flag, out = simulate_(etahat, alphahat0)
    last_sim_ok = true
    fbr = Int(o["likelihood.first_period_binding_regime_allowed"])
    TT = RR = CC = nothing
    if flag != 0
        last_sim_ok, error_flag = false, 321
        new_regimes = regime_history
    else
        new_regimes = out.regime_history
        if new_regimes[1:fbr-1] != regime_history[1:fbr-1]
            error_flag, new_regimes = 322, regime_history
        end
        TT, RR, CC = matrices(out)
    end
    regime_history = new_regimes
    is_changed = regime_history0[1] != regime_history
    rt_converged = !(is_changed || error_flag != 0)
    sto = (alphahat=alphahat, etahat=etahat, alphahat0=alphahat0, regime_history=copy(regime_history))
    sto_TT = (TT, RR, CC)
    cond_converged = false
    if !rt_converged && !(lik_init == 2 && Harvey_scale_factor == 0) && TT !== nothing
        # conditional smoother: initial state known (the smoothed one), Harvey_scale_factor = 0
        a_c = zeros(n)
        a_c[state_dr] = alphahat0[state_dr]
        rc = _pkf_forward(ctx, Y, a_c, zeros(n, n); data_index=data_index, univariate=true, check_inf=false,
                          regimes_init=vcat([regime_history[1]], regime_history))
        if rc.error_code != 0
            error_flag = 322
        else
            c_alpha, c_eta, c_alpha0 = _univariate_backward(ctx, rc.a1, rc.P1, rc.v, rc.Fi, rc.Ki, rc.TT, rc.RR, data_index,
                                                            rc.ainit, rc.Pinit)
            c_regimes = rc.regimes[1:T+1]
            regime_history = c_regimes
            flag, outc = simulate_(c_eta, alphahat0)
            error_flag = 0
            if flag != 0
                error_flag = 321
                c_new = regime_history
            else
                out = outc
                c_new = outc.regime_history
                if c_new[1:fbr-1] != regime_history[1:fbr-1]
                    error_flag, c_new = 322, regime_history
                end
            end
            is_changed = c_regimes != c_new
            if !(is_changed || error_flag != 0)
                cond_converged = true
                TT, RR, CC = matrices(outc)
                alphahat, etahat, alphahat0 = c_alpha, c_eta, c_alpha0
                regime_history = c_regimes
            end
        end
    end
    maxiter = Int(o["smoother.max_number_of_iterations"])
    it, is_periodic = 1, false
    fast = Bool(o["smoother.fast"])
    is_changed_start = false
    cur_TT = sto_TT
    while is_changed && maxiter > it && !is_periodic && last_sim_ok
        if it == 1
            rt_ok && (regime_history = copy(sto.regime_history))
            error_flag = 0
            TT, RR, CC = cur_TT
        end
        it += 1
        _, a1_, P1_, v_, Fi_, Ki_ = _tv_forward(ctx, Y, a_init, P_init, data_index, TT, RR, CC)
        alphahat, etahat, alphahat0 = _univariate_backward(ctx, a1_, P1_, v_, Fi_, Ki_, TT, RR, data_index, a_init, P_init)
        push!(regime_history0, copy(regime_history))
        sto_TT = (TT, RR, CC)
        flag, out2 = simulate_(etahat, alphahat0)
        if flag != 0
            error_flag = 321
            break
        end
        new = out2.regime_history
        if new[1:fbr-1] != regime_history[1:fbr-1]
            error_flag = 322
            break
        end
        out = out2
        regime_history = new
        TT, RR, CC = matrices(out2)
        prev = regime_history0[end]
        is_changed = prev != regime_history
        d_reg = any(prev[kk][j][1] != regime_history[kk][j][1] for kk in eachindex(prev) for j in 1:ctx.k)
        is_changed_start = any(prev[kk][j][2] != regime_history[kk][j][2] for kk in eachindex(prev) for j in 1:ctx.k)
        fast && (is_changed = d_reg)
        is_tmp = any(regime_history0[kk] == regime_history for kk in 1:it-1)
        is_periodic = is_changed && is_tmp
    end
    push!(regime_history0, copy(regime_history))
    final_regimes = regime_history0[end]
    converged = false
    if (((maxiter == it) || !last_sim_ok) && is_changed) || is_periodic || error_flag != 0
        if is_periodic
            converged, error_flag, warning_flag = true, 0, 321
        elseif !rt_ok
            error_flag = 322
        end
    else
        converged, error_flag = true, 0
        if !cond_converged && !rt_converged
            warning_flag = 320
        elseif fast && is_changed_start
            warning_flag = 323
        end
    end
    if rt_ok && !cond_converged && !rt_converged && !converged
        warning_flag, error_flag = 322, 0
        alphahat, etahat, alphahat0 = sto.alphahat, sto.etahat, sto.alphahat0
        final_regimes = realtime_regimes
        is_changed = false
    end
    if is_changed && is_periodic
        TT, RR, CC = sto_TT
        final_regimes = regime_history0[end-1]
    end
    res = OccbinSmootherResult(smoothed=Matrix(to_decl(alphahat)'), shocks=Matrix(etahat'), updated=Matrix(to_decl(ahat)'),
        filtered=Matrix(to_decl(aK[:, 2:T+1])'), smoothed0=to_decl(alphahat0), regime_history=final_regimes,
        realtime_regime_history=realtime_regimes, linear_smoothed=Matrix(to_decl(lin_alpha)'), linear_shocks=Matrix(lin_eta'),
        error_flag=error_flag, warning_flag=warning_flag, iterations=it, regime_history_iter=regime_history0, ys=copy(ob.ys),
        converged=converged, endo_names=copy(m.endo_names), exo_names=copy(m.exo_names))
    if out !== nothing
        res.piecewise = out.piecewise[1:end-1, :]
        res.linear = out.linear[1:end-1, :]
    end
    if TT !== nothing
        res.TT = [zeros(n, n) for _ in TT]
        res.RR = [zeros(n, ctx.nexo) for _ in TT]
        res.CC = [zeros(n) for _ in TT]
        for kk in eachindex(TT)
            res.TT[kk][order_var, order_var] = TT[kk]
            res.RR[kk][order_var, :] = RR[kk]
            res.CC[kk][order_var] = CC[kk]
        end
    end
    res
end

"""
Smoother output of the inversion filter (`smoother_inversion_filter`; Dynare's IVF_posterior `atT`/`innov`): the filtered
states and shocks are the smoothed ones, the system starts at the steady state. `updated`/`filtered` are NaN (Dynare does
not provide them with the inversion filter).
"""
function inversion_filter_smoother(ivf::InversionFilterResult, ys, err_index, n_exo; endo_names=String[], exo_names=String[])
    T, n = size(ivf.stateval)
    shocks = zeros(T, n_exo)
    shocks[:, err_index] = ivf.filtered_errs
    nan = fill(NaN, T, n)
    OccbinSmootherResult(smoothed=copy(ivf.stateval), shocks=shocks, updated=nan, filtered=copy(nan), smoothed0=zeros(n),
                         regime_history=copy(ivf.regime_history), realtime_regime_history=copy(ivf.regime_history),
                         error_flag=ivf.error_code, ys=Vector{Float64}(ys), inversion_filter=true,
                         endo_names=String.(endo_names), exo_names=String.(exo_names))
end

# ====================================================================== shock decomposition
"""
    occbin_shock_decomposition(sm, T0, R0; tinit=1, regime_shock=nothing) -> Dict{String,ShockDecomposition}

occbin/shock_decomposition.m on an OccBin smoother result (with `TT`/`RR`/`CC`); `T0`, `R0` the linear rules (declaration
order). Layout `(n_endo, n_exo + 2, T)` (shocks, initial conditions, smoothed variable; deviations from the steady state):

* `"decomp"` (oo_.occbin.smoother.decomp): piecewise decomposition; the effect of the regime constants goes to the
  initial-condition column, unless the model has a shock named `EPS_REGIME` (or `regime_shock`) that receives it;
* `"wdecomp"` (oo_.occbin.smoother.wdecomp): the regime effect spread over the columns in proportion to their absolute
  contributions;
* `"init_decomp"` (oo_.occbin.smoother.init_decomp): initial conditions by variable, then the shocks and the smoothed
  variable (Dynare's field keeps only the n variable columns);
* `"linear"`, `"linear_init"` (oo_.occbin.linear_smoother.decomp / init_decomp).

`tinit` (1-based) is the first period decomposed (Dynare's TINIT).
"""
function occbin_shock_decomposition(sm::OccbinSmootherResult, T0::AbstractMatrix, R0::AbstractMatrix; tinit::Integer=1,
                                    regime_shock=nothing)
    (sm.TT === nothing || sm.linear_smoothed === nothing) &&
        throw(SmootherError("the smoother result has no regime matrices / linear smoother (inversion filter?)"))
    alpha, eta = Matrix(sm.smoothed'), Matrix(sm.shocks')
    la, le = Matrix(sm.linear_smoothed'), Matrix(sm.linear_shocks')
    n, gend = size(alpha)
    nexo = size(eta, 1)
    t0 = Int(tinit)
    L = gend - t0 + 1
    # linear
    as_lin, att, deco = zeros(n, L), zeros(n, L), zeros(n, nexo + 2, L)
    deco_init = zeros(n, n + 2, L)
    deco_init[:, end, :] = la[:, t0:end]
    for i in 1:n
        deco_init[i, i, 1] = la[i, t0]
    end
    att[:, 1] = as_lin[:, 1] = la[:, t0]
    for j in 2:L
        e = le[:, j+t0-1]
        as_lin[:, j] = T0 * as_lin[:, j-1] + R0 * e
        att[:, j] = T0 * att[:, j-1]
        deco[:, 1:nexo, j] = R0 .* e' + T0 * deco[:, 1:nexo, j-1]
        deco_init[:, 1:n, j] = T0 * deco_init[:, 1:n, j-1]
        deco_init[:, n+1, j] = deco_init[:, n+2, j] - vec(sum(deco_init[:, 1:n, j]; dims=2))
    end
    deco[:, nexo+1, :] = att
    deco[:, nexo+2, :] = as_lin
    # piecewise
    as_p, att_p, reg_p = zeros(n, L), zeros(n, L), zeros(n, L)
    deco_p = zeros(n, nexo + 2, L)
    deco_p_init = zeros(n, n + 2, L)
    deco_p_init[:, end, :] = alpha[:, t0:end]
    for i in 1:n
        deco_p_init[i, i, 1] = alpha[i, t0]
    end
    att_p[:, 1] = as_p[:, 1] = alpha[:, t0]
    for j in 2:L
        TM, RM, CONST = sm.TT[j+t0-1], sm.RR[j+t0-1], sm.CC[j+t0-1]
        e = eta[:, j+t0-1]
        as_p[:, j] = TM * as_p[:, j-1] + RM * e + CONST
        att_p[:, j] = TM * att_p[:, j-1]
        reg_p[:, j] = TM * reg_p[:, j-1] + CONST
        deco_p[:, 1:nexo, j] = RM .* e' + TM * deco_p[:, 1:nexo, j-1]
        deco_p_init[:, 1:n, j] = TM * deco_p_init[:, 1:n, j-1]
        deco_p_init[:, n+1, j] = deco_p_init[:, n+2, j] - vec(sum(deco_p_init[:, 1:n, j]; dims=2))
    end
    exo = sm.exo_names
    i_reg = regime_shock !== nothing && regime_shock in exo ? findfirst(==(regime_shock), exo) : findfirst(==("EPS_REGIME"), exo)
    if i_reg !== nothing
        deco_p[:, i_reg, :] = reg_p
    else
        att_p = att_p + reg_p
    end
    deco_p[:, nexo+1, :] = att_p
    deco_p[:, nexo+2, :] = as_p
    decomp = copy(deco_p)
    i_reg === nothing && (deco_p[:, nexo+1, :] = att_p - reg_p)
    rr = abs.(deco_p[:, 1:end-1, :])
    i_reg === nothing || (rr[:, i_reg, :] .= 0)
    tot = sum(rr; dims=2)
    tot[tot .< 1e-10] .= 1
    ww = rr ./ tot
    wdeco = copy(deco_p)
    i_reg === nothing || (wdeco[:, i_reg, :] .= 0)
    for kk in 1:L
        if any(!=(0), reg_p[:, kk])
            wdeco[:, 1:end-1, kk] = wdeco[:, 1:end-1, kk] + reg_p[:, kk] .* ww[:, :, kk]
        end
    end
    cols = vcat(exo, ["initial_conditions", "smoothed"])
    icols = vcat(sm.endo_names, ["shocks", "smoothed"])
    mk(z, c, kind) = ShockDecomposition(z=z, endo_names=copy(sm.endo_names), columns=c, kind=kind, steady_state=sm.ys, first_period=t0)
    Dict("decomp" => mk(decomp, cols, "shocks"), "wdecomp" => mk(wdeco, cols, "shocks"),
         "init_decomp" => mk(deco_p_init, icols, "initial_conditions"), "linear" => mk(deco, cols, "shocks"),
         "linear_init" => mk(deco_init, icols, "initial_conditions"))
end

"occbin/squeeze_shock_decomposition.m: keep the listed variables in every decomposition."
squeeze_occbin_shock_decomposition(decomps::AbstractDict, var_list) =
    Dict(k => squeeze_shock_decomposition(v, var_list) for (k, v) in decomps)

# ====================================================================== regime output
"""
    regime_table(regime_history; periods=nothing, dynare_indexing=false) -> rows

The table written by `occbin_write_regimes` (write_regimes_to_xls.m) as named tuples per row: `time`, `regime_sequence`,
`starting_period_of_regime` (suffixes `_1`, `_2` with two constraints), the sequences written as MATLAB's `int2str` does
("0  1  0"). `periods` (1-based, default all) selects rows. Dynare 7.2 labels the rows with `periods` but fills them with
the first `length(periods)` regimes (it indexes `regime_history(tp)` with the row number): `dynare_indexing=true`
reproduces that; by default row `t` holds the regime of period `t`.
"""
function regime_table(rh::AbstractVector; periods=nothing, dynare_indexing::Bool=false)
    Ts = periods === nothing ? collect(1:length(rh)) : Int.(vec(collect(periods)))
    k = isempty(rh) ? 1 : length(rh[1])
    rows = Dict{String,Any}[]
    for (tp, t) in enumerate(Ts)
        reg = dynare_indexing ? rh[tp] : rh[t]
        row = Dict{String,Any}("time" => t)
        for (j, (r, s)) in enumerate(reg)
            suf = k == 1 ? "" : "_$j"
            row["regime_sequence$suf"] = join(string.(Int.(r)), "  ")
            row["starting_period_of_regime$suf"] = join(string.(Int.(s)), "  ")
        end
        push!(rows, row)
    end
    rows
end

"Write `regime_table` as a CSV file (instead of Dynare's .xls)."
function write_regime_table(path::AbstractString, rh::AbstractVector; periods=nothing, dynare_indexing::Bool=false)
    rows = regime_table(rh; periods=periods, dynare_indexing=dynare_indexing)
    k = isempty(rh) ? 1 : length(rh[1])
    cols = ["time"]
    for j in 1:k
        suf = k == 1 ? "" : "_$j"
        push!(cols, "regime_sequence$suf", "starting_period_of_regime$suf")
    end
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join([string(r[c]) for c in cols], ","))
        end
    end
    rows
end
