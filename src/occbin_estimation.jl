# Estimation and smoothing of OccBin models (port of pydyn/occbin_estimation.py, after the OccBin branches of Dynare 7.2's
# matlab/estimation/{dsge_likelihood, dynare_estimation_init, initial_estimation_checks, dynare_estimation_1}.m,
# matlab/+occbin/{IVF_posterior, DSGE_smoother}.m and matlab/kalman/evaluate_smoother.m; Dynare Team, GPL-3.0-or-later).
#
# Dynare sets options_.occbin.likelihood.status and smoother.status for every model with an occbin_constraints block, so:
#   * `Estimation` of such a model uses the piecewise Kalman filter (PKF), or the inversion filter (IVF) when
#     `occbin_setup(likelihood_inversion_filter)` (or `occbin=Dict("likelihood_inversion_filter" => true)`) asks for it;
#     everything built on `_dsge_likelihood` (posterior_mode, hessian, laplace, sample_posterior, estimate) follows;
#   * `smooth` (calib_smoother, estimation(smoother)) runs the OccBin smoother (occbin.DSGE_smoother) or the inversion
#     filter (`smoother_inversion_filter`); the linear smoother is kept in `res.occbin.linear_*`.

"Options of the estimation statement and Dynare checks for an OccBin model (called by the Estimation constructor)."
function _setup_occbin!(est::Estimation, overrides; obc=NamedTuple())
    m = est.model
    if isempty(m.occbin_constraints)
        overrides === nothing || throw(EstimationError("occbin options were given but the model has no occbin_constraints block"))
        any(!isnothing, values(obc)) && throw(EstimationError("obc_* options were given but the model has no occbin_constraints block"))
        return est
    end
    o = est.options
    overrides, live_obc = _split_obc_overrides(overrides)          # obc_method, obc_horizon, obc_selection in `occbin=Dict(...)` (src/obc_filter.jl)
    oc = occbin_filter_options(m, overrides; what="OccBin estimation")
    ivf = Bool(oc["likelihood.inversion_filter"])
    o["loglinear"] && throw(EstimationError("loglinear is not available with OccBin (the piecewise-linear solution is in levels)"))
    o["lik_init"] == 3 && throw(EstimationError("the diffuse filter (lik_init=3, diffuse_filter, kalman_algo 3/4) is not implemented with OccBin"))
    # initial_estimation_checks.m
    o["prefilter"] != 0 && throw(EstimationError("initial_estimation_checks: OccBin is incompatible with the prefilter option due to the sample " *
                                                 "mean generally not corresponding to the steady state with an occasionally binding constraint."))
    (!ivf && o["kalman_algo"] in (2, 4)) &&
        throw(EstimationError("initial_estimation_checks: OccBin is incompatible with the selected univariate Kalman filter."))
    if ivf || Bool(oc["smoother.inversion_filter"])
        oc["likelihood.IVF_shock_observable_mapping"] = _ivf_mapping(est, oc)
        (any(!=(0), est.H0) || !isempty(est.groups.vn) || !isempty(est.groups.cn)) &&
            throw(EstimationError("IVF-filter: Measurement errors are not allowed with the inversion filter."))
        length(oc["likelihood.IVF_shock_observable_mapping"]) == length(est.varobs) ||
            throw(EstimationError("The IVF requires exactly as many shocks as observables."))
    end
    o["occbin"] = oc
    o["obc"] = _obc_est_settings(m, live_obc; obc...)
    o["obc"].method == :news && !ivf && throw(EstimationError(_PKF_NEWS_MSG))
    o["obc"].formulation == :lcp && _check_lcp_model(m)                # an error unless every constraint is one complementary pair
    est
end

"IVF_shock_observable_mapping (dynare_estimation_init.m): by default the shocks with a non-zero calibrated variance."
function _ivf_mapping(est::Estimation, oc)
    m = est.model
    mp = oc["likelihood.IVF_shock_observable_mapping"]
    zero_var = findall(diag(est.Sigma_e0) .== 0)
    (mp === nothing || (mp isa AbstractArray && isempty(mp))) && return findall(diag(est.Sigma_e0) .!= 0)
    idx = Int[x isa AbstractString ? _exo_index(m, x) : Int(x) for x in (mp isa AbstractArray ? vec(mp) : [mp])]
    any(in(zero_var), idx) && throw(EstimationError("IVF-filter: an observable is mapped to a zero variance shock."))
    idx
end

"OccBin options of an estimation (nothing for a model without occbin_constraints)."
occbin_options(est::Estimation) = get(est.options, "occbin", nothing)

"""
    occbin_at(est, theta) -> Occbin

The piecewise-linear model at `theta` (steady state and reference-regime rules at the estimated parameters).
"""
occbin_at(est::Estimation, theta::AbstractVector) = Occbin(est.model; params=model_params(est, theta), qz_criterium=est.options["qz_criterium"])

"PKF state space at `theta`: `_PKFContext` on the filter states (`est.restrict`, Dynare's restrict_var_list)."
function _pkf_context(est::Estimation, ob::Occbin, Q, Hm)
    oc = occbin_options(est)
    mm = length(est.restrict)
    Z = zeros(length(est.varobs), mm)
    for (i, k) in enumerate(est.mf)
        Z[i, k] = 1.0
    end
    _PKFContext(ob, est.restrict, Z, Hm, Q, oc; kalman_tol=est.options["kalman_tol"],
                rescale=est.options["rescale_prediction_error_covariance"])
end

"Initial prediction and covariance of the PKF (dsge_likelihood.m, lik_init 1 and 2)."
function _pkf_initial_state(est::Estimation, ctx::_PKFContext)
    o, oc = est.options, ctx.o
    P = if o["lik_init"] == 1
        lyapunov_symm(ctx.T0, ctx.R0 * ctx.Q * ctx.R0'; qz_criterium=o["qz_criterium"], complex_threshold=o["lyapunov_complex_threshold"])[1]
    else
        Ph = o["Harvey_scale_factor"] * Matrix{Float64}(I, ctx.mm, ctx.mm)
        (Int(oc["likelihood.first_period_occbin_update"]) != 1 && o["Harvey_scale_factor"] == 0) &&
            (Ph = ctx.T0 * Ph * ctx.T0' + ctx.R0 * ctx.Q * ctx.R0')
        Ph
    end
    zeros(ctx.mm), P
end

"Observations minus the steady state (and trends) of the observed variables, (pp x T)."
function _occbin_observations(est::Estimation, ob::Occbin)
    Y = permutedims(est.data) .- ob.ys[est.obs_decl]
    trend = _trend(est, ob.params, est.nobs)
    trend === nothing ? Y : Y .- trend
end

"""
    occbin_filter(est, theta) -> PKFResult or InversionFilterResult

Run the OccBin filter of the estimation at `theta` (no bounds or prior): the piecewise Kalman filter (`PKFResult`), or the
inversion filter (`InversionFilterResult`, with `LIK` and `lik` set) under `likelihood_inversion_filter`. Throws the solver's
errors (`SteadyStateError`, `BlanchardKahnError`).
"""
# Dynare's inversion filter solves each period with csolve, whose random search (when no progress is made) draws from the
# global stream; whether an evaluation succeeds then depends on how far that stream has advanced, not only on the
# parameters (a Dynare 7.2 defect: the same vector can give a likelihood on one call and code 304 on the next). Corrected
# by default: each evaluation draws from a fresh generator seeded with the model's seed (set_dynare_seed), so the
# likelihood is a function of the parameters. `replicate_dynare_bugs=true` draws from the shared stream, as Dynare does.
function _ivf_rng(m::Model)
    b = get(m.options, "replicate_dynare_bugs", false)
    (b === true || (b isa Real && b != 0)) && return dynare_rng(m)
    Random.Xoshiro(_streams(m).seed)
end

function occbin_filter(est::Estimation, theta::AbstractVector; rng=_ivf_rng(est.model))
    oc = occbin_options(est)
    oc === nothing && throw(EstimationError("the model has no occbin_constraints block"))
    theta = Float64.(theta)
    Q, Hm = covariances(est, theta)
    ob = occbin_at(est, theta)
    if Bool(oc["likelihood.inversion_filter"])
        err = oc["likelihood.IVF_shock_observable_mapping"]
        st = obc_settings(est)
        st.last_failure = nothing
        ivf = inversion_filter(ob, est.data, est.obs_decl, err, oc; rng=rng, solver=_obc_solver(est, ob))
        ivf.failure === nothing || (st.last_failure = ivf.failure)          # news solver: why the failing period failed (src/obc_filter.jl)
        if ivf.error_code == 0
            code, pen = inversion_filter_loglik!(ivf, est.data, est.obs_decl, Q, err; presample=est.options["presample"])
            if code != 0
                ivf.error_code, ivf.penalty = code, pen
            end
        end
        return ivf
    end
    obc_settings(est).method == :occbin || throw(EstimationError(_PKF_NEWS_MSG))
    ctx = _pkf_context(est, ob, Q, Hm)
    a, P = _pkf_initial_state(est, ctx)
    di = est.data_index === nothing ? nothing : est.data_index
    piecewise_kalman_filter(ctx, _occbin_observations(est, ob), a, P; data_index=di, presample=est.options["presample"])
end

"dsge_likelihood.m / IVF_posterior.m for an OccBin model, after the bounds and covariance checks; same outputs as `_dsge_likelihood`."
function _occbin_likelihood(est::Estimation, theta::AbstractVector, with_lik::Bool)
    oc = occbin_options(est)
    reject(info, pen, msg) = (est.last_info = msg; est.last_info_code = info; (Inf, info, pen, NaN, NaN, Float64[]))
    ivf = Bool(oc["likelihood.inversion_filter"])
    est.last_filter = ivf ? "inversion filter" : "piecewise Kalman filter"
    res = try
        occbin_filter(est, theta)
    catch e
        if e isa SteadyStateError
            return reject(22, 0.1, "steady state: $(e.msg)")
        elseif e isa BlanchardKahnError
            return reject(3, isfinite(e.penalty) ? e.penalty : 0.1, "solution: $(e.msg)")
        elseif e isa DomainError || e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException
            return reject(22, 0.1, "steady state or solution: $(sprint(showerror, e))")
        end
        rethrow()
    end
    if ivf
        res.error_code != 0 && return reject(res.error_code, res.penalty, "inversion filter failed in period $(res.error_period) (info $(res.error_code))" *
                                              (res.failure === nothing ? "" : "; news solver: $(res.failure.primary) ($(res.failure.advice))"))
    else
        res.error_code != 0 && return reject(320, 0.1, "piecewise Kalman filter failed in period $(res.error_period) (OccBin error $(res.error_code); info 320)")
        isfinite(res.LIK) || return reject(320, 0.1, "the likelihood is NaN or Inf (info 320)")
    end
    LIK = res.LIK
    lnprior = logpdf(est.prior, theta)
    if isinf(lnprior) || isnan(lnprior)
        info = isnan(lnprior) ? 47 : 40
        est.last_info = "the prior density is zero, infinite or NaN (info $info)"
        est.last_info_code = info
        return Inf, info, ivf ? 1000.0 : 0.1, -LIK, lnprior, Float64[]
    end
    contrib = with_lik ? res.lik[est.options["presample"]+1:end] : Float64[]
    LIK - lnprior, 0, 0.0, -LIK, lnprior, contrib
end

# ====================================================================== smoother
"""
OccBin part of a `SmootherResult` (`res.occbin`): the `OccbinSmootherResult` (`smoother`; Dynare's oo_.occbin.smoother), the
linear smoother's result (`linear`; oo_.occbin.linear_smoother) and the OccBin options used.
"""
struct OccbinSmoothing
    smoother::OccbinSmootherResult
    linear::Any
    options::Dict{String,Any}
end

# smoother options that need the regime-dependent variances, not implemented with OccBin
const _OCC_SMOOTHER_REFUSED = ("filter_covariance", "smoothed_state_uncertainty")

"""
`smooth` for a model with occbin_constraints (Dynare's evaluate_smoother / dynare_estimation_1 with
options_.occbin.smoother.status): the linear smoother (all of smooth's options and parameter sets) followed by the OccBin
smoother at the same parameters, or by the inversion filter with `smoother_inversion_filter`. `occbin` overrides OccBin options.
"""
function _occbin_smooth(m::Model, data=nothing; occbin=nothing, estimation=nothing, kw...)
    for k in _OCC_SMOOTHER_REFUSED
        _flag(get(kw, Symbol(k), false)) && throw(SmootherError("$k is not implemented for the OccBin smoother"))
    end
    est = estimation
    occbin, live_obc = _split_obc_overrides(occbin)              # obc_* entries of the overrides (src/obc_filter.jl)
    oc = (est !== nothing && occbin === nothing && occbin_options(est) !== nothing) ? occbin_options(est) :
         occbin_filter_options(m, occbin; what="OccBin smoother")
    lin = _smooth_linear(m, data; estimation=est, kw...)
    for (k, v) in (("filter_covariance", lin.FilteredVariablesKStepAheadVariances), ("smoothed_state_uncertainty", lin.State_uncertainty))
        v === nothing || throw(SmootherError("$k is not implemented for the OccBin smoother"))
    end
    lin.loglinear && throw(SmootherError("loglinear is not available with OccBin (the piecewise-linear solution is in levels)"))
    _flag(get(lin.options, "prefilter", 0)) && throw(SmootherError("OccBin is incompatible with the prefilter option (initial_estimation_checks)"))
    any(!iszero, lin.TrendCoeffs) && throw(SmootherError("observation_trends are not implemented for the OccBin smoother"))
    lin.kalman_algo in (3, 4) && throw(SmootherError("the diffuse filter is not implemented for the OccBin smoother"))
    fsa = lin.filter_step_ahead
    (isempty(fsa) || fsa == [1]) || throw(SmootherError("filter_step_ahead beyond 1 is not implemented for the OccBin smoother"))
    lik_init = Int(get(lin.options, "lik_init", 1))
    qz = (est !== nothing && lik_init == est.options["lik_init"]) ? est.options["qz_criterium"] :
         _select_qz(m, lik_init, get(lin.options, "qz_criterium", nothing))
    ob = Occbin(m; params=lin.params, qz_criterium=qz)
    maximum(abs.(ob.ys .- lin.SteadyState); init=0.0) < 1e-8 * max(1.0, maximum(abs, ob.ys; init=0.0)) ||
        throw(SmootherError("the steady state of the reference regime differs from the linear smoother's (occbin_X_bind parameters set?)"))
    n, T = m.n_endo, size(lin.data, 1)
    obs = lin.obs_index
    if Bool(oc["smoother.inversion_filter"])
        err = (est !== nothing && haskey(oc, "likelihood.IVF_shock_observable_mapping") && oc["likelihood.IVF_shock_observable_mapping"] !== nothing) ?
              oc["likelihood.IVF_shock_observable_mapping"] : findall(diag(lin.Sigma_e) .!= 0)
        length(err) == length(obs) || throw(SmootherError("The IVF requires exactly as many shocks as observables."))
        any(!iszero, lin.me_cov) && throw(SmootherError("IVF-filter: Measurement errors are not allowed with the inversion filter."))
        sol = if est !== nothing
            _obc_solver(est, ob)
        else                                           # no estimation object: the native options_.occbin.obc_* lines
            s = _obc_est_settings(m, live_obc)
            s.method == :occbin ? nothing : news_filter_solver(ob; horizon=s.horizon, selection=s.selection, horizon_max=s.horizon_max, formulation=s.formulation)
        end
        ivf = inversion_filter(ob, lin.data .+ ob.ys[obs]', obs, err, oc; rng=_ivf_rng(m), solver=sol)
        sm = inversion_filter_smoother(ivf, ob.ys, err, ob.n_exo; endo_names=m.endo_names, exo_names=m.exo_names)
    else
        (est !== nothing ? obc_settings(est).method : _obc_est_settings(m, live_obc).method) == :occbin ||
            throw(SmootherError(_PKF_NEWS_MSG))
        ktol = Float64(something(get(lin.options, "kalman_tol", nothing), 1e-10))
        sm = occbin_smoother(ob, permutedims(lin.data), lin.Sigma_e, lin.me_cov, obs, oc; lik_init=lik_init, qz_criterium=qz,
                             Harvey_scale_factor=Float64(something(get(lin.options, "Harvey_scale_factor", nothing), 10.0)), kalman_tol=ktol)
    end
    res = deepcopy(lin)
    res.occbin = OccbinSmoothing(sm, lin, oc)
    sm.error_flag == 0 || return res              # Dynare writes nothing to oo_ (store_smoother_results is skipped)
    nm = m.endo_names
    cols(M, names) = Dict{String,Vector{Float64}}(v => M[:, i] for (i, v) in enumerate(names))
    ys = ob.ys
    res.smoothed, res.shocks, res.init_state = sm.smoothed, sm.shocks, sm.smoothed0
    res.SmoothedVariables = cols(sm.smoothed .+ ys', nm)
    res.SmoothedShocks = cols(sm.shocks, m.exo_names)
    res.updated = sm.updated
    res.UpdatedVariables = cols(sm.updated .+ ys', nm)
    res.Init_State = Dict{String,Float64}(v => sm.smoothed0[i] + ys[i] for (i, v) in enumerate(nm))
    res.Init_State_Vector = sm.smoothed0 .+ ys
    eps = lin.data .- sm.smoothed[:, obs]
    res.measurement_errors = eps
    res.SmoothedMeasurementErrors = isempty(lin.SmoothedMeasurementErrors) ? Dict{String,Vector{Float64}}() : cols(eps, lin.varobs)
    if sm.inversion_filter
        res.FilteredVariables = Dict{String,Vector{Float64}}()
        res.FilteredVariablesKStepAhead = nothing
        res.filter_step_ahead = Int[]
    else
        res.FilteredVariables = cols(sm.filtered .+ ys', nm)
        if fsa == [1]
            k = zeros(1, n, T + 1)
            k[1, :, 2:T+1] = permutedims(sm.filtered .+ ys')
            k[1, :, 1] = ys
            res.FilteredVariablesKStepAhead = k
        end
    end
    res
end

"""
    occbin_shock_decomposition(m, res::SmootherResult; tinit=1) -> Dict{String,ShockDecomposition}

Dynare's `occbin.shock_decomposition` on the OccBin smoother result of `smooth` (fields `decomp`, `wdecomp`, `init_decomp`,
`linear`, `linear_init`; see the method on `OccbinSmootherResult`).
"""
function occbin_shock_decomposition(m::Model, res::SmootherResult; tinit::Integer=1)
    res.occbin === nothing && throw(SmootherError("not an OccBin smoother result"))
    sm = res.occbin.smoother
    sm.error_flag == 0 || throw(SmootherError("the OccBin smoother failed (error_flag $(sm.error_flag))"))
    ob = Occbin(m; params=res.params)
    occbin_shock_decomposition(sm, ob.P, ob.Q; tinit=tinit)
end

# ====================================================================== oo_ (dynare(), estimation!, calib_smoother!)
"OccBin options set in options_.occbin (by MATLAB lines of the file or by keywords): overrides for `occbin_filter_options`."
function _occbin_overrides(ws)
    isempty(ws.model.occbin_constraints) && return nothing
    oc = _getpath(ws.options_, "occbin", nothing)
    oc isa MStruct || return nothing
    out = Dict{String,Any}()
    for (k, v) in _flat_options(oc)
        haskey(OCCBIN_FILTER_DEFAULTS, k) || continue
        v isa AbstractArray && length(v) == 1 && (v = first(v))
        v isa AbstractArray && isempty(v) && (v = nothing)
        out[k] = v
    end
    for (k, v) in _flat_options(oc)                               # obc_method, obc_horizon, obc_selection (src/obc_filter.jl)
        k in ("obc_method", "obc_horizon", "obc_selection", "obc_horizon_max", "obc_formulation") || continue
        v isa AbstractArray && length(v) == 1 && (v = first(v))
        out[k] = v
    end
    isempty(out) ? nothing : out
end

_rowf(v) = reshape(Float64.(collect(v)), 1, :)

"A regime history as Dynare's struct array (fields regime/regimestart, or regime1, regimestart1, regime2, regimestart2): a row Cell of structs, as JDyn writes M_.aux_vars."
function _regime_history_cell(rh::AbstractVector)
    items = Any[]
    for reg in rh
        s = MStruct()
        if length(reg) == 1
            s.regime = _rowf(reg[1][1]); s.regimestart = _rowf(reg[1][2])
        else
            for (j, (r, st)) in enumerate(reg)
                s["regime$j"] = _rowf(r); s["regimestart$j"] = _rowf(st)
            end
        end
        push!(items, s)
    end
    Cell(items, (1, length(items)))
end

"""
oo_.occbin.smoother and oo_.occbin.linear_smoother as occbin.DSGE_smoother writes them (evaluate_smoother.m / dynare_estimation_1.m):
error and warning flags, regime histories, the smoothed and updated variables and shocks of both smoothers (with Dynare's
alphahat/etahat/ahat in decision-rule order), the simulation with the smoothed shocks and the regime matrices.
"""
function _store_occbin_smoother!(oo::MStruct, m::Model, res::SmootherResult)
    occ = get(oo, "occbin", nothing)
    occ isa MStruct || (occ = MStruct(); oo.occbin = occ)
    sm, lin = res.occbin.smoother, res.occbin.linear
    ov = res.solution.order_var
    S = MStruct()
    S.error_flag = Float64(sm.error_flag)
    S.regime_history = _regime_history_cell(sm.regime_history)
    if !sm.inversion_filter
        S.warning_flag = Float64(sm.warning_flag)
        S.realtime_regime_history = _regime_history_cell(sm.realtime_regime_history)
    end
    if sm.error_flag == 0
        tmp = MStruct()
        r2 = deepcopy(res); r2.occbin = nothing
        _store_smoother_results!(tmp, m, r2)
        for f in ("SmoothedVariables", "UpdatedVariables", "SmoothedShocks")
            S[f] = tmp[f]
        end
        S.alphahat = Matrix(sm.smoothed[:, ov]')
        S.etahat = Matrix(sm.shocks')
        S.ahat = Matrix(sm.updated[:, ov]')
        S.SteadyState = _col(sm.ys)
        if !sm.inversion_filter
            S.epsilonhat = Matrix(res.measurement_errors')
            S.trend_coeff = zeros(length(res.varobs), 1)
            sm.TT === nothing || (S.T0 = cat([T[ov, ov] for T in sm.TT]...; dims=3); S.R0 = cat([R[ov, :] for R in sm.RR]...; dims=3);
                                  S.C0 = hcat([C[ov] for C in sm.CC]...))
            if sm.piecewise !== nothing
                sim = MStruct(); sim.piecewise = sm.piecewise; sim.linear = sm.linear
                S.simul = sim
            end
        end
    end
    occ.smoother = S
    if lin !== nothing && !sm.inversion_filter
        L = MStruct()
        tmp = MStruct()
        _store_smoother_results!(tmp, m, lin)
        for f in ("SmoothedVariables", "UpdatedVariables", "SmoothedShocks")
            L[f] = tmp[f]
        end
        L.alphahat = Matrix(lin.smoothed[:, ov]')
        L.etahat = Matrix(lin.shocks')
        L.ahat = Matrix(lin.updated[:, ov]')
        L.SteadyState = _col(lin.SteadyState)
        L.T0 = lin.G[ov, ov]
        L.R0 = lin.H[ov, :]
        occ.linear_smoother = L
    end
    oo
end

"Regimes back from a regime_history Cell of structs (oo_.occbin.simul / smoother)."
function _regimes_from_cell(c::Cell)
    out = OccRegime[]
    for s in c.items
        if haskey(s, "regime")
            push!(out, OccRegime([(Int.(vec(s.regime)), Int.(vec(s.regimestart)))]))
        else
            k = count(f -> startswith(f, "regimestart"), keys(s))
            push!(out, OccRegime([(Int.(vec(s["regime$j"])), Int.(vec(s["regimestart$j"]))) for j in 1:k]))
        end
    end
    out
end

"""
    occbin_write_regimes!(ws; type=..., periods=..., filename=...) -> rows

Dynare's `occbin_write_regimes` (write_regimes_to_xls.m): the regime history of `oo_.occbin.simul` (type `simul`, the default)
or `oo_.occbin.smoother` (type `smoother`) as a table `time, regime_sequence, starting_period_of_regime` (suffixes `_1`, `_2`
with two constraints), written as CSV to `<dir>/Output/<filename>.csv` (Dynare: `.xls`; default filename
`<model>_occbin_regimes`). Options come from options_.occbin.write_regimes (keywords are written there first).
Dynare 7.2 labels the rows with `periods` but fills them with the first regimes; JDyn writes the regimes of the labelled
periods unless `options_.replicate_dynare_bugs` is set.
"""
function occbin_write_regimes!(ws; options...)
    _take_options!(ws, Dict(Symbol("occbin.write_regimes." * String(k)) => v for (k, v) in options))
    m = ws.model
    wr = _getpath(ws.options_, "occbin.write_regimes", MStruct())
    typ = lowercase(String(get(wr, "type", "simul")))
    typ in ("simul", "smoother") || throw(ModelError("write_regimes_to_xls: output type can only be simul or smoother."))
    occ = get(ws.oo_, "occbin", nothing)
    src = occ isa MStruct ? get(occ, typ, nothing) : nothing
    (src isa MStruct && haskey(src, "regime_history")) || throw(ModelError("write_regimes_to_xls: the required field does not exist"))
    rh = _regimes_from_cell(src.regime_history)
    per = get(wr, "periods", zeros(0, 0))
    periods = (per isa AbstractArray && isempty(per)) ? nothing : Int.(round.(vec(collect(per))))
    fname = String(get(wr, "filename", m.name * "_occbin_regimes"))
    dir = joinpath(m.mod_dir, "Output")
    mkpath(dir)
    bugs = _optb(ws, "replicate_dynare_bugs", false)
    write_regime_table(joinpath(dir, fname * ".csv"), rh; periods=periods, dynare_indexing=bugs)
end
