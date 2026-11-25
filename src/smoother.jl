# Kalman filter and smoother of a calibrated or estimated first-order model: Dynare's `calib_smoother`, the smoother
# `estimation` runs at the posterior mode (`smoother` option) and `evaluate_smoother`'s parameter sets (port of
# pydyn/smoother.py). Derived from Dynare 7.2's matlab/kalman/{DsgeSmoother, evaluate_smoother, store_smoother_results,
# missing_DiffuseKalmanSmootherH1_Z, missing_DiffuseKalmanSmootherH3_Z}.m (GPL-3.0-or-later, Dynare Team).
#
#     m = JDyn.load("model.mod")                    # varobs, calib_smoother(datafile=...) in the file
#     res = smooth(m)                               # what calib_smoother does; options from the file, keywords override
#     res.SmoothedVariables["y"], res.SmoothedShocks["e"], res.UpdatedVariables, res.FilteredVariables
#     est = Estimation(m); smooth(m; estimation=est, theta=mode.x)           # at an estimated parameter vector
#     smooth(m; estimation=est, parameter_set="posterior_mode", mode=mode)  # evaluate_smoother's parameter sets
#
# The state vector holds every endogenous variable in declaration order (Dynare's default, `smoother_redux` off).
# State space (deviations from the steady state, or from its log with `loglinear`):
#     a_{t+1} = T a_t + R eps_{t+1},  y_t = Z a_t + eta_t
# Conventions are Dynare's: variables come back as deviation plus steady state (log steady state under `loglinear`);
# observed variables under `prefilter` get the data mean instead and the trend of an `observation_trends` block;
# `FilteredVariables[v][t]` is the forecast of period t+1 made in period t (Dynare's aK(1,:,2:end)).
#
# Defects of Dynare 7.2 (docs/GAPS.md of pydyn): the multivariate diffuse smoother (kalman_algo=3) drops terms of the
# backward recursion in a diffuse period without observations or with F_inf = 0; the univariate diffuse smoother's
# State_uncertainty is wrong in that situation too; `filter_initial_state` centres the period-0 state on T a. These are
# corrected by default; `replicate_dynare_bugs=true` reproduces Dynare's numbers.

struct SmootherError <: Exception
    msg::String
end
Base.showerror(io::IO, e::SmootherError) = print(io, "SmootherError: ", e.msg)

# ====================================================================== options
const _SM_DEFAULTS = Dict{String,Any}("datafile" => nothing, "first_obs" => 1, "nobs" => nothing, "prefilter" => 0,
    "loglinear" => false, "logdata" => false, "presample" => 0, "diffuse_filter" => false, "lik_init" => 1,
    "kalman_algo" => 0, "kalman_tol" => 1e-10, "diffuse_kalman_tol" => 1e-6, "qz_criterium" => nothing,
    "filtered_vars" => false, "filter_step_ahead" => nothing, "filter_covariance" => false,
    "smoothed_state_uncertainty" => false, "filter_decomposition" => false, "parameter_set" => "calibration", "varobs" => nothing,
    "replicate_dynare_bugs" => false, "Harvey_scale_factor" => 10.0,
    "use_univariate_smoother_if_singularity_is_detected" => false, "noprint" => false, "heteroskedastic_filter" => false)
"Set by `_smoother_multivariate` when it stops because F is singular in a stationary period (Dynare's error flag 422)."
const _SINGULAR_F = Ref(false)
const PARAMETER_SETS = ("calibration", "posterior_mode", "posterior_mean", "posterior_median", "mle_mode", "prior_mode", "prior_mean")
# options of the estimation statement that concern the optimiser, the sampler or its output, not the smoother
const _ESTIMATION_ONLY = Set(["mode_compute", "mode_file", "mode_check", "mode_check.status", "mh_replic", "mh_nblck", "mh_nblocks",
    "mh_jscale", "mh_drop", "mh_init_scale", "mh_init_scale_factor", "mh_tune_jscale", "optim", "cova_compute", "posterior_sampling_method",
    "bayesian_irf", "moments_varendo", "irf", "forecast", "conf_sig", "mh_conf_sig", "load_mh_file", "prior_trunc", "smoother", "order",
    "posterior_sampler_options", "additional_optimizer_steps", "nobs_vec", "lyapunov", "sub_draws", "mh_recover", "mcmc_jumping_covariance",
    "use_univariate_filters_if_singularity_is_detected", "rescale_prediction_error_covariance", "riccati_tol", "gstep",
    "lyapunov_complex_threshold", "use_univariate_smoother_if_singularity_is_detected", "dsge_var", "dsge_varlag", "endogenous_prior",
    "mh_tune_guess", "load_results_after_load_mh"])
const _SM_COSMETIC = union(COSMETIC, Set(["xls_sheet", "xls_range", "use_univariate_smoother_if_singularity_is_detected",
    "consider_all_endogenous", "consider_only_observed", "selected_variables_only", "plot_priors"]))
# options of the decomposition statements that only concern Dynare's figure files and menus
const _DECOMP_COSMETIC = Set(["colormap", "fig_name", "graph_format", "nodisplay", "interactive", "nograph",
    "shock_decomposition", "plot_shock_decomposition"])
# options the code cannot honour: refused rather than ignored
const _SM_REFUSED = Dict("smoother_redux" => "smoother_redux",
    "occbin" => "OccBin smoothing", "noconstant" => "noconstant (Dynare's store_smoother_results ignores it, so its output mixes two conventions)", "partial_information" => "partial information", "kalman_algo5" => "the pruned skewed filter")

_unwrap1(v) = (v isa AbstractArray && length(v) == 1) ? v[1] : v
_flag(v) = !(v === false || v === 0 || v === 0.0 || v === nothing || v == "")
_tonum(v) = v isa AbstractString ? parse(Float64, v) : Float64(v)

"""
Options of the statements of `kinds` of the .mod file merged in file order (Dynare accumulates `options_`); the
`shock_decomp.` / `plot_shock_decomp.` / `initial_condition_decomp.` prefixes are removed.
"""
function _statement_options(m::Model, kinds)
    out = Dict{String,Any}()
    for st in m.statements
        get(st, "statementName", "") in kinds || continue
        for (k, v) in get_or(st, "options", Dict{String,Any}())
            out[last(split(k, "."; limit=2)) == k ? k : (startswith(k, "no_graph.") ? k : String(last(split(k, "."; limit=2))))] = v
        end
    end
    out
end

function _resolve_options(m::Model, user::Dict{String,Any}; extra_kinds=())
    opts = copy(_SM_DEFAULTS)
    from_file = _statement_options(m, ("estimation", "calib_smoother", extra_kinds...))
    for (k, v) in from_file
        haskey(opts, k) && (opts[k] = v)
    end
    for k in ("logdata", "kalman_tol", "diffuse_kalman_tol", "Harvey_scale_factor")   # MATLAB-side `options_.x = ...;`
        if !haskey(from_file, k) && haskey(m.matlab_options, k)
            try
                opts[k] = scalar(m.matlab_options[k], "options_.$k")
            catch
            end
        end
    end
    for k in keys(from_file)
        haskey(_SM_REFUSED, k) && _flag(_unwrap1(from_file[k])) && throw(SmootherError("the .mod file sets $k; $(_SM_REFUSED[k]) is not implemented"))
    end
    dropped = sort([k for k in keys(from_file) if !haskey(opts, k) && !(k in _SM_COSMETIC) && !(k in _ESTIMATION_ONLY) &&
                    !(k in _DECOMP_COSMETIC) && !startswith(k, "no_graph.") && !haskey(_SM_REFUSED, k) &&
                    !(k in ("init_state", "use_shock_groups", "max_nrows", "forecast", "save_realtime", "fast_realtime",
                            "type", "steadystate", "diff", "flip", "detail_plot", "realtime", "vintage", "init2shocks", "expand", "fig_name"))])
    isempty(dropped) || warn_once(m, "the smoother does not act on these options from the .mod file: " * join(dropped, ", "))
    unknown = sort([k for k in keys(user) if !haskey(opts, k)])
    isempty(unknown) || throw(SmootherError("unknown smoother option(s): $(join(unknown, ", "))"))
    for (k, v) in user
        (v !== nothing || k in ("nobs", "filter_step_ahead", "datafile", "qz_criterium", "varobs")) && (opts[k] = v)
    end
    for k in ("first_obs", "nobs", "presample", "lik_init", "kalman_algo", "prefilter")
        v = _unwrap1(opts[k])
        (k == "nobs" && v === nothing) && continue
        v isa AbstractVector && throw(SmootherError("vector-valued $k (recursive estimation) is not supported"))
        opts[k] = Int(_tonum(v))
    end
    for k in ("kalman_tol", "diffuse_kalman_tol", "Harvey_scale_factor")
        opts[k] = _tonum(_unwrap1(opts[k]))
    end
    for k in ("loglinear", "logdata", "diffuse_filter", "filtered_vars", "filter_covariance", "smoothed_state_uncertainty", "filter_decomposition",
              "replicate_dynare_bugs")
        opts[k] = _flag(_unwrap1(opts[k]))
    end
    opts["datafile"] = _unwrap1(opts["datafile"])
    opts["qz_criterium"] = opts["qz_criterium"] === nothing ? nothing : _tonum(_unwrap1(opts["qz_criterium"]))
    opts["parameter_set"] = opts["parameter_set"] === nothing ? "calibration" : String(_unwrap1(opts["parameter_set"]))
    opts["parameter_set"] in PARAMETER_SETS || throw(SmootherError("unknown parameter_set $(repr(opts["parameter_set"])); use one of $(join(PARAMETER_SETS, ", "))"))
    opts["prefilter"] in (0, 1) || throw(SmootherError("prefilter must be 0 or 1"))
    fsa = opts["filter_step_ahead"]
    if fsa !== nothing
        fsa = sort(unique(Int[Int(_tonum(x)) for x in (fsa isa AbstractVector ? fsa : [fsa])]))
        (!isempty(fsa) && fsa[1] < 1) && throw(SmootherError("filter_step_ahead must hold positive integers"))
    end
    (fsa === nothing || isempty(fsa)) && (fsa = opts["filtered_vars"] ? [1] : Int[])
    opts["filter_step_ahead"] = fsa
    opts
end

# ====================================================================== results
"""
    SmootherResult

Smoother output named after the fields of Dynare's `oo_`. The `Dict`s are keyed by variable or shock name and hold vectors of
length `nobs`. Arrays keep Dynare's layout: `FilteredVariablesKStepAhead` is (steps, n_endo, nobs + max step),
`FilteredVariablesKStepAheadVariances` (steps, n_endo, n_endo, nobs + max step), `Variance` (n_endo, n_endo, nobs + 1) and
`State_uncertainty` (n_endo, n_endo, nobs), all in declaration order. The working arrays `smoothed`, `updated`, `shocks`,
`measurement_errors` and `data` are (periods x variables), in deviations from the constant.
"""
Base.@kwdef mutable struct SmootherResult
    endo_names::Vector{String}
    exo_names::Vector{String}
    varobs::Vector{String}
    SmoothedVariables::Dict{String,Vector{Float64}}
    UpdatedVariables::Dict{String,Vector{Float64}}
    FilteredVariables::Dict{String,Vector{Float64}}
    SmoothedShocks::Dict{String,Vector{Float64}}
    SmoothedMeasurementErrors::Dict{String,Vector{Float64}}
    Constant::Dict{String,Vector{Float64}}
    Init_State::Dict{String,Float64}
    Init_State_Vector::Vector{Float64}
    SteadyState::Vector{Float64}                    # levels
    loglinear::Bool
    FilteredVariablesKStepAhead::Union{Nothing,Array{Float64,3}} = nothing
    FilteredVariablesKStepAheadVariances::Union{Nothing,Array{Float64,4}} = nothing
    FilteredVariablesShockDecomposition::Union{Nothing,Array{Float64,4}} = nothing   # (steps, n_endo, n_shocks, nobs + max step)
    filter_step_ahead::Vector{Int} = Int[]
    Variance::Union{Nothing,Array{Float64,3}} = nothing
    State_uncertainty::Union{Nothing,Array{Float64,3}} = nothing
    Init_State_uncertainty::Union{Nothing,Matrix{Float64}} = nothing
    smoothed::Matrix{Float64} = zeros(0, 0)         # (T, n_endo) smoothed deviations
    updated::Matrix{Float64} = zeros(0, 0)
    init_state::Vector{Float64} = Float64[]         # smoothed deviation in period 0
    shocks::Matrix{Float64} = zeros(0, 0)           # (T, n_exo)
    measurement_errors::Matrix{Float64} = zeros(0, 0)   # (T, n_obs); NaN where the observation is missing
    data::Matrix{Float64} = zeros(0, 0)             # (T, n_obs): data after log / prefilter, constant and trend removed
    obs_constant::Vector{Float64} = Float64[]
    me_variance::Vector{Float64} = Float64[]
    G::Matrix{Float64} = zeros(0, 0)                # dev_t = G dev_{t-1} + H u_t
    H::Matrix{Float64} = zeros(0, 0)
    obs_index::Vector{Int} = Int[]
    kalman_algo::Int = 0                            # filter actually used: 1, 2 (stationary), 3, 4 (diffuse)
    diffuse_periods::Int = 0
    options::Dict{String,Any} = Dict{String,Any}()
    Trend::Dict{String,Vector{Float64}} = Dict{String,Vector{Float64}}()
    TrendCoeffs::Vector{Float64} = Float64[]
    parameter_set::String = "calibration"
    theta::Union{Nothing,Vector{Float64}} = nothing
    params::Vector{Float64} = Float64[]
    Sigma_e::Matrix{Float64} = zeros(0, 0)
    me_cov::Matrix{Float64} = zeros(0, 0)
    solution::Any = nothing
    sample::Matrix{Float64} = zeros(0, 0)           # observations of the sample: after logs, before prefilter and trends
    first_obs::Int = 1
    occbin::Any = nothing                           # OccBin models: OccbinSmoothing (src/occbin_estimation.jl)
end
nobs(r::SmootherResult) = size(r.smoothed, 1)
Base.show(io::IO, r::SmootherResult) = print(io, "SmootherResult(", nobs(r), " periods, ", length(r.endo_names), " variables, ",
    length(r.exo_names), " shocks, filter ", r.kalman_algo, ", ", r.diffuse_periods, " diffuse periods)")

# ====================================================================== core: multivariate smoother
_rank_(A::AbstractMatrix, tol) = isempty(A) ? 0 : count(>(tol), svdvals(A))
_lowersym(A) = tril(A) + tril(A, -1)'

"""
Port of `missing_DiffuseKalmanSmootherH1_Z`. `Y` is (n_obs, T), `di[t]` the observed rows of period t. Returns `nothing` when Dynare's
routine reports a singular forecast-error variance (the caller then switches to the univariate smoother). `compat=true` reproduces
three defects of Dynare 7.2 in the diffuse phase. `a_pred` is the state forecast of period 1, `a_zero` the mean of the period-0
state (zero unless `filter_initial_state` is used).
"""
function _smoother_multivariate(T, Z, R, Q, H, Pinf1, Pstar1, Y, di, nk, kalman_tol, dk_tol, want_V, want_PK, compat, a_pred, a_zero; want_decomp=false)
    mm = size(T, 1)
    pp, smpl = size(Y)
    Tt = T'
    a = zeros(mm, smpl + 1)
    atilde = zeros(mm, smpl)
    v = zeros(pp, smpl)
    aK = zeros(nk, mm, smpl + nk)
    PK = want_PK ? zeros(nk, mm, mm, smpl + nk) : nothing
    iFv = Vector{Matrix{Float64}}(undef, smpl)
    L = Vector{Matrix{Float64}}(undef, smpl)
    P = zeros(mm, mm, smpl + 1)
    # heteroskedastic_filter: Q is (n_exo, n_exo, smpl + 1); period t predicts with Q[:, :, t + 1] and its smoothed shocks use Q[:, :, t]
    qvec = ndims(Q) == 3
    Qv = Q
    qvec && (Q = Qv[:, :, 1])
    QQ = R * Q * R'
    QRt = Q * R'
    QQ_at(t) = qvec ? R * Qv[:, :, t+1] * R' : QQ
    QRt_at(t) = qvec ? Qv[:, :, t] * R' : QRt
    a0 = a_zero === nothing ? zeros(mm) : Vector{Float64}(a_zero)
    P0 = copy(Pstar1)
    a_pred === nothing || (a[:, 1] = a_pred)
    diffuse = Pinf1 !== nothing
    Pstar = Dict{Int,Matrix{Float64}}(1 => copy(Pstar1))
    Pinf = diffuse ? Dict{Int,Matrix{Float64}}(1 => copy(Pinf1)) : Dict{Int,Matrix{Float64}}()
    Linf = Dict{Int,Matrix{Float64}}(); Lstar = Dict{Int,Matrix{Float64}}(); Kstar = Dict{Int,Matrix{Float64}}()
    iFinf = Dict{Int,Matrix{Float64}}(); iFstar = Dict{Int,Matrix{Float64}}(); Fstar = Dict{Int,Matrix{Float64}}()
    kind = Dict{Int,Symbol}()
    Pinf_init = Pstar_init = nothing
    a_init = nothing

    t = 1
    new_rank = diffuse ? _rank_(Z * Pinf[1] * Z', dk_tol) : 0
    if new_rank > 0
        Pinf_init, Pstar_init, a_init = copy(Pinf[1]), copy(Pstar[1]), a[:, 1]
        a[:, 1] = T * a[:, 1]
        Pstar[1] = T * Pstar[1] * Tt + QQ
    end
    while new_rank > 0 && t <= smpl
        d_ = di[t]
        QQ = QQ_at(t)
        store_forecast = true
        if isempty(d_)
            kind[t] = :missing
            atilde[:, t] = a[:, t]
            Linf[t] = T
            Pstar[t+1] = T * Pstar[t] * Tt + QQ
            Pinf[t+1] = T * Pinf[t] * Tt
            store_forecast = !compat                     # Dynare leaves aK at zero here
        else
            ZZ = Z[d_, :]
            vt = Y[d_, t] - ZZ * a[:, t]
            v[d_, t] = vt
            Finf = ZZ * Pinf[t] * ZZ'
            # Dynare tests rcond(Finf) first; a Finf that is zero up to rounding but not exactly zero then passes as regular
            # and is inverted. Test for zero first instead.
            finf_zero = all(x -> abs(x) < dk_tol, Finf)
            iFi, rcF = finf_zero ? (nothing, 0.0) : _inv_rcond(Finf)
            if finf_zero || rcF < dk_tol
                finf_zero || return nothing                                   # error 420
                Fs = ZZ * Pstar[t] * ZZ' + H[d_, d_]
                Fstar[t] = Fs
                iFs, rcs = _inv_rcond(Fs)
                if rcs < kalman_tol
                    all(x -> abs(x) < kalman_tol, Fs) || return nothing        # error 421
                    kind[t] = :skip
                    Pstar[t+1] = T * Pstar[t] * Tt + QQ
                    Pinf[t+1] = T * Pinf[t] * Tt
                    upd = a[:, t]
                    iFstar[t] = zeros(size(Fs))
                    Kstar[t] = zeros(mm, length(d_))
                    Lstar[t] = compat ? zeros(mm, mm) : Matrix(T)
                else
                    kind[t] = :star
                    iFstar[t] = iFs
                    Kstar[t] = Pstar[t] * ZZ' * iFs
                    Pinf[t+1] = T * Pinf[t] * Tt
                    Lstar[t] = T - T * Kstar[t] * ZZ
                    Pstar[t+1] = T * Pstar[t] * Lstar[t]' + QQ
                    upd = a[:, t] + Kstar[t] * vt
                end
                # Dynare overwrites the state forecast with T*atilde where atilde was never set
                atilde[:, t] = compat ? zeros(mm) : upd
            else
                kind[t] = :inf
                iFinf[t] = iFi
                Kinf = Pinf[t] * ZZ' * iFi
                atilde[:, t] = a[:, t] + Kinf * vt
                Linf[t] = T - T * Kinf * ZZ
                Fs = ZZ * Pstar[t] * ZZ' + H[d_, d_]
                Fstar[t] = Fs
                Kstar[t] = (Pstar[t] * ZZ' - Kinf * Fs) * iFi
                Pstar[t+1] = T * Pstar[t] * Linf[t]' - T * Kinf * Finf * Kstar[t]' * Tt + QQ
                Pinf[t+1] = T * Pinf[t] * Linf[t]'
            end
        end
        a[:, t+1] = T * atilde[:, t]
        if store_forecast
            aK[1, :, t+1] = a[:, t+1]
            for j in 1:nk-1
                aK[j+1, :, t+1+j] = T * aK[j, :, t+j]
            end
        end
        t += 1
        new_rank = _rank_(Z * Pinf[t] * Z', dk_tol)
    end
    d = t - 1
    P[:, :, d+1] = Pstar[d+1]

    while t <= smpl
        P[:, :, t] = _lowersym(P[:, :, t])
        d_ = di[t]
        QQ = QQ_at(t)
        if isempty(d_)
            atilde[:, t] = a[:, t]
            L[t] = Matrix(T)
            P[:, :, t+1] = T * P[:, :, t] * Tt + QQ
        else
            ZZ = Z[d_, :]
            vt = Y[d_, t] - ZZ * a[:, t]
            v[d_, t] = vt
            F = ZZ * P[:, :, t] * ZZ' + H[d_, d_]
            dF = diag(F)
            any(<(kalman_tol), dF) && (_SINGULAR_F[] = true; return nothing)      # error 422
            sig = sqrt.(dF)
            ss = sig * sig'
            iFs, rc = _inv_rcond(F ./ ss)
            rc < kalman_tol && (_SINGULAR_F[] = true; return nothing)             # error 422
            iFt = iFs ./ ss
            iFv[t] = iFt
            PZI = P[:, :, t] * ZZ' * iFt
            atilde[:, t] = a[:, t] + PZI * vt
            L[t] = T - T * PZI * ZZ
            P[:, :, t+1] = T * P[:, :, t] * L[t]' + QQ
        end
        a[:, t+1] = T * atilde[:, t]
        Pf = P[:, :, t+1]
        aK[1, :, t+1] = a[:, t+1]
        for j in 0:nk-1
            if want_PK
                j > 0 && (Pf = T * Pf * Tt + QQ)
                PK[j+1, :, :, t+1+j] = Pf
            end
            j > 0 && (aK[j+1, :, t+1+j] = T * aK[j, :, t+j])
        end
        t += 1
    end

    # ---- filter decomposition (missing_DiffuseKalmanSmootherH1_Z): shock contributions to the k-step ahead filtered values
    decomp = nothing
    if want_decomp
        rr = size(R, 2)
        decomp = zeros(nk, mm, rr, smpl + nk)
        ZRQinv = inv(Z * (R * Matrix(qvec ? Qv[:, :, 1] : Q) * R') * Z')
        for t in max(d, 1)+1:smpl
            d_ = di[t]
            isempty(d_) && continue
            ZZ = Z[d_, :]
            eta = QRt_at(t) * (ZZ' * (iFv[t] * v[d_, t]))
            AAA = P[:, :, t] * ZZ' * ZRQinv[d_, d_] * ((ZZ * R) .* eta')
            decomp[1, :, :, t+1] = AAA
            for h in 2:nk
                AAA = T * AAA
                decomp[h, :, :, t+h] = AAA
            end
        end
    end

    # ---- backward pass
    alphahat = zeros(mm, smpl)
    etahat = zeros(size(Q, 1), smpl)
    r = zeros(mm, smpl + 1)
    N = want_V ? [zeros(mm, mm) for _ in 1:smpl+1] : nothing
    V = want_V ? zeros(mm, mm, smpl) : nothing
    V0 = nothing
    alphahat0 = nothing
    for t in smpl:-1:d+1
        d_ = di[t]
        if isempty(d_)
            r[:, t] = L[t]' * r[:, t+1]
            want_V && (N[t] = L[t]' * N[t+1] * L[t])
        else
            ZZ = Z[d_, :]
            iFt = iFv[t]
            r[:, t] = ZZ' * iFt * v[d_, t] + L[t]' * r[:, t+1]
            want_V && (N[t] = ZZ' * iFt * ZZ + L[t]' * N[t+1] * L[t])
        end
        alphahat[:, t] = a[:, t] + P[:, :, t] * r[:, t]
        etahat[:, t] = QRt_at(t) * r[:, t]
        want_V && (V[:, :, t] = P[:, :, t] - P[:, :, t] * N[t] * P[:, :, t])
    end
    if d == 0
        alphahat0 = a0 + P0 * (Tt * r[:, 1])
        want_V && (V0 = P0 - P0 * (Tt * N[1] * T) * P0)
    else
        r0 = zeros(mm, d + 1)
        r1 = zeros(mm, d + 1)
        r0[:, d+1] = r[:, d+1]
        N1 = want_V ? [zeros(mm, mm) for _ in 1:d+1] : nothing
        N2 = want_V ? [zeros(mm, mm) for _ in 1:d+1] : nothing
        for t in d:-1:1
            d_ = di[t]
            k = kind[t]
            if k === :missing || (k === :skip && !compat)
                r1[:, t] = Tt * r1[:, t+1]
                if !compat
                    r0[:, t] = Tt * r0[:, t+1]
                    if want_V
                        N[t] = Tt * N[t+1] * T
                        N1[t] = Tt * N1[t+1] * T
                        N2[t] = Tt * N2[t+1] * T
                    end
                end
            elseif k === :inf
                ZZ = Z[d_, :]
                Li = Linf[t]
                r0[:, t] = Li' * r0[:, t+1]
                r1[:, t] = ZZ' * (iFinf[t] * v[d_, t] - Kstar[t]' * (Tt * r0[:, t+1])) + Li' * r1[:, t+1]
                if want_V
                    L_1 = -T * Kstar[t] * ZZ
                    N[t] = Li' * N[t+1] * Li
                    N1[t] = ZZ' * iFinf[t] * ZZ + Li' * N1[t+1] * Li + L_1' * N[t+1] * Li
                    N2[t] = (ZZ' * (-iFinf[t] * Fstar[t] * iFinf[t]) * ZZ + Li' * N2[t+1] * Li +
                             Li' * N1[t+1] * L_1 + L_1' * N1[t+1]' * Li + L_1' * N[t+1] * L_1)
                end
            else                                                  # F_inf = 0 in this period
                ZZ = Z[d_, :]
                Ls = Lstar[t]
                sgn = compat ? -1.0 : 1.0                         # Dynare has "- Lstar' * r0"
                r0[:, t] = ZZ' * iFstar[t] * v[d_, t] + sgn * (Ls' * r0[:, t+1])
                r1[:, t] = Tt * r1[:, t+1]
                if want_V
                    N[t] = ZZ' * iFstar[t] * ZZ + Ls' * N[t+1] * Ls
                    N1[t] = Tt * N1[t+1] * Ls
                    N2[t] = Tt * N2[t+1] * (compat ? Tt : T)      # Dynare has T' twice
                end
            end
            alphahat[:, t] = a[:, t] + Pstar[t] * r0[:, t] + Pinf[t] * r1[:, t]
            etahat[:, t] = QRt_at(t) * r0[:, t]
            if want_V
                x = Pinf[t] * N1[t] * Pstar[t]
                V[:, :, t] = Pstar[t] - Pstar[t] * N[t] * Pstar[t] - x' - x - Pinf[t] * N2[t] * Pinf[t]
            end
        end
        alphahat0 = a_init + Pstar_init * (Tt * r0[:, 1]) + Pinf_init * (Tt * r1[:, 1])
        if want_V
            x = Pinf_init * (Tt * N1[1] * T) * Pstar_init
            V0 = Pstar_init - Pstar_init * (Tt * N[1] * T) * Pstar_init - x' - x - Pinf_init * (Tt * N2[1] * T) * Pinf_init
        end
    end
    (alphahat=alphahat, etahat=etahat, updated=atilde, P=P, aK=aK, PK=PK, V=V, V0=V0, d=d, alphahat0=alphahat0, decomp=decomp)
end

# ====================================================================== core: univariate smoother
"""
Port of `missing_DiffuseKalmanSmootherH3_Z` (no OccBin): observations are processed one at a time. `Hd` is the diagonal of the
measurement-error covariance (correlated errors are handled by the caller through an enlarged state vector, as Dynare).
`compat=true` keeps a defect of Dynare 7.2 in the smoothed state uncertainty of the diffuse phase.
"""
function _smoother_univariate(T, Z, R, Q, Hd, Pinf1, Pstar1, Y, di, nk, kalman_tol, dk_tol, want_V, want_PK, compat, a_pred, a_zero; want_decomp=false)
    mm = size(T, 1)
    pp, smpl = size(Y)
    Tt = T'
    Id = Matrix{Float64}(I, mm, mm)
    a = zeros(mm, smpl)
    a1 = zeros(mm, smpl + 1)
    v = zeros(pp, smpl)
    aK = zeros(nk, mm, smpl + nk)
    PK = want_PK ? zeros(nk, mm, mm, smpl + nk) : nothing
    Fstar = zeros(pp, smpl); Finf = zeros(pp, smpl); Fi = zeros(pp, smpl)
    Ki = zeros(mm, pp, smpl)
    Kstar = Dict{Int,Matrix{Float64}}(); Kinf = Dict{Int,Matrix{Float64}}()
    P = zeros(mm, mm, smpl + 1)
    P1 = zeros(mm, mm, smpl + 1)
    QQ = R * Q * R'
    QRt = Q * R'
    a_pred === nothing || (a1[:, 1] = a_pred)
    ainit = a_zero === nothing ? a1[:, 1] : Vector{Float64}(a_zero)
    Pinit = copy(Pstar1)
    diffuse = Pinf1 !== nothing
    Pstar = Dict{Int,Matrix{Float64}}(1 => copy(Pstar1))
    Pinf = diffuse ? Dict{Int,Matrix{Float64}}(1 => copy(Pinf1)) : Dict{Int,Matrix{Float64}}()
    Pstar_pred = Dict{Int,Matrix{Float64}}(); Pinf_pred = Dict{Int,Matrix{Float64}}()
    Pinf_init = Pstar_init = nothing

    t = 1
    new_rank = diffuse ? _rank_(Z * Pinf[1] * Z', dk_tol) : 0
    if new_rank > 0
        Pinf_init, Pstar_init = copy(Pinf[1]), copy(Pstar[1])
        Pstar[1] = T * Pstar[1] * Tt + QQ
    end
    while new_rank > 0 && t <= smpl
        a[:, t] = a1[:, t]
        Pstar_pred[t], Pinf_pred[t] = copy(Pstar[t]), copy(Pinf[t])
        Kstar[t], Kinf[t] = zeros(mm, pp), zeros(mm, pp)
        Ps, Pi = Pstar[t], Pinf[t]
        for i in di[t]
            Zi = Z[i, :]
            v[i, t] = Y[i, t] - dot(Zi, a[:, t])
            Fstar[i, t] = dot(Zi, Ps * Zi) + Hd[i]
            Finf[i, t] = dot(Zi, Pi * Zi)
            Ks = Ps * Zi
            Kstar[t][:, i] = Ks
            if Finf[i, t] > dk_tol
                Kf = Pi * Zi
                Kinf[t][:, i] = Kf
                KF = Kf / Finf[i, t]
                a[:, t] = a[:, t] + KF * v[i, t]
                Ps = Ps + (Kf * KF') * (Fstar[i, t] / Finf[i, t]) - Ks * KF' - KF * Ks'
                Pi = Pi - Kf * Kf' / Finf[i, t]
            elseif Fstar[i, t] > kalman_tol
                a[:, t] = a[:, t] + Ks * v[i, t] / Fstar[i, t]
                Ps = Ps - Ks * Ks' / Fstar[i, t]
            end
        end
        a1[:, t+1] = T * a[:, t]
        aK[1, :, t+1] = a1[:, t+1]
        for j in 1:nk-1
            aK[j+1, :, t+1+j] = T * aK[j, :, t+j]
        end
        Pstar[t+1] = T * Ps * Tt + QQ
        Pinf[t+1] = T * Pi * Tt
        t += 1
        new_rank = _rank_(Z * Pinf[t] * Z', dk_tol)
    end
    d = t - 1
    P[:, :, d+1] = Pstar[d+1]

    while t <= smpl
        a[:, t] = a1[:, t]
        P1[:, :, t] = P[:, :, t]
        Pt = P[:, :, t]
        for i in di[t]
            Zi = Z[i, :]
            v[i, t] = Y[i, t] - dot(Zi, a[:, t])
            Fi[i, t] = dot(Zi, Pt * Zi) + Hd[i]
            Ki[:, i, t] = Pt * Zi
            if Fi[i, t] > kalman_tol
                a[:, t] = a[:, t] + Ki[:, i, t] * v[i, t] / Fi[i, t]
                Pt = Pt - Ki[:, i, t] * Ki[:, i, t]' / Fi[i, t]
                Pt = (Pt + Pt') / 2
            end
        end
        a1[:, t+1] = T * a[:, t]
        P[:, :, t+1] = T * Pt * Tt + QQ
        P[:, :, t+1] = (P[:, :, t+1] + P[:, :, t+1]') / 2
        Pf = P[:, :, t+1]
        aK[1, :, t+1] = a1[:, t+1]
        for j in 0:nk-1
            if want_PK
                j > 0 && (Pf = T * Pf * Tt + QQ)
                PK[j+1, :, :, t+1+j] = Pf
            end
            j > 0 && (aK[j+1, :, t+1+j] = T * aK[j, :, t+j])
        end
        t += 1
    end
    P1[:, :, t] = P[:, :, t]

    # ---- filter decomposition (missing_DiffuseKalmanSmootherH3_Z)
    decomp = nothing
    if want_decomp
        rr = size(R, 2)
        decomp = zeros(nk, mm, rr, smpl + nk)
        ZRQinv = inv(Z * QQ * Z')
        for t in max(d, 1)+1:smpl
            ri_d = zeros(mm)
            for i in reverse(di[t])
                if Fi[i, t] > kalman_tol
                    Zi = Z[i, :]
                    ri_d = Zi / Fi[i, t] * v[i, t] + ri_d - Zi * (dot(Ki[:, i, t], ri_d) / Fi[i, t])
                end
            end
            eta = QRt * ri_d
            AAA = P1[:, :, t] * Z' * ZRQinv * Z * R
            Ttok = Matrix{Float64}(I, mm, mm)
            for h in 1:nk
                BBB = Ttok * AAA
                for j in 1:rr
                    decomp[h, :, j, t+h] = eta[j] * BBB[:, j]
                end
                Ttok = T * Ttok
            end
        end
    end

    # ---- backward pass
    alphahat = zeros(mm, smpl)
    etahat = zeros(size(Q, 1), smpl)
    V = want_V ? zeros(mm, mm, smpl) : nothing
    V0 = nothing
    alphahat0 = nothing
    ri = zeros(mm)
    Ni = zeros(mm, mm)
    for t in smpl:-1:d+1
        for i in reverse(di[t])
            if Fi[i, t] > kalman_tol
                Zi = Z[i, :]
                Li = Id - Ki[:, i, t] * Zi' / Fi[i, t]
                ri = Zi / Fi[i, t] * v[i, t] + Li' * ri
                want_V && (Ni = Zi * Zi' / Fi[i, t] + Li' * Ni * Li)
            end
        end
        alphahat[:, t] = a1[:, t] + P1[:, :, t] * ri
        etahat[:, t] = QRt * ri
        ri = Tt * ri
        if want_V
            V[:, :, t] = P1[:, :, t] - P1[:, :, t] * Ni * P1[:, :, t]
            Ni = Tt * Ni * T
        end
    end
    if d == 0
        alphahat0 = ainit + Pinit * ri
        want_V && (V0 = Pinit - Pinit * Ni * Pinit)
    else
        r0, r1 = ri, zeros(mm)
        N0, N1, N2 = Ni, zeros(mm, mm), zeros(mm, mm)
        for t in d:-1:1
            for i in reverse(di[t])
                Zi = Z[i, :]
                if Finf[i, t] > dk_tol
                    Kf, Ks = Kinf[t][:, i], Kstar[t][:, i]
                    Li = Id - Kf * Zi' / Finf[i, t]
                    L0 = (Kf * (Fstar[i, t] / Finf[i, t]) - Ks) * Zi' / Finf[i, t]
                    r1 = Zi * v[i, t] / Finf[i, t] + L0' * r0 + Li' * r1
                    r0 = Li' * r0
                    if want_V
                        zz = Zi * Zi'
                        N2 = (-zz / Finf[i, t]^2 * Fstar[i, t] + Li' * N2 * Li + Li' * N1 * L0 + L0' * N1' * Li + L0' * N0 * L0)
                        N1 = zz / Finf[i, t] + Li' * N1 * Li + L0' * N0 * Li
                        N0 = Li' * N0 * Li
                    end
                elseif Fstar[i, t] > kalman_tol
                    Ks = Kstar[t][:, i]
                    Li = Id - Ks * Zi' / Fstar[i, t]
                    r0 = Zi / Fstar[i, t] * v[i, t] + Li' * r0
                    if want_V
                        N0 = Zi * Zi' / Fstar[i, t] + Li' * N0 * Li
                        compat || (N1 = N1 * Li)                  # Dynare keeps N_1 fixed here
                    end
                end
            end
            Ps, Pi = Pstar_pred[t], Pinf_pred[t]
            alphahat[:, t] = a1[:, t] + Ps * r0 + Pi * r1
            etahat[:, t] = QRt * r0
            if want_V
                x = Pi * N1 * Ps
                V[:, :, t] = Ps - Ps * N0 * Ps - x' - x - Pi * N2 * Pi
            end
            r0, r1 = Tt * r0, Tt * r1
            want_V && ((N0, N1, N2) = (Tt * N0 * T, Tt * N1 * T, Tt * N2 * T))
        end
        alphahat0 = ainit + Pstar_init * r0 + Pinf_init * r1
        if want_V
            x = Pinf_init * N1 * Pstar_init
            V0 = Pstar_init - Pstar_init * N0 * Pstar_init - x' - x - Pinf_init * N2 * Pinf_init
        end
    end
    (d == smpl && diffuse) && @warn "the diffuse filter never left the diffuse phase: the data do not identify the initial conditions of the non-stationary variables"
    (alphahat=alphahat, etahat=etahat, updated=a, P=P1, aK=aK, PK=PK, V=V, V0=V0, d=d, alphahat0=alphahat0, decomp=decomp)
end

# ====================================================================== parameter sets
"""
    prior_mode(prior) -> Vector

Mode of each prior (Dynare's `bayestopt_.p5`, `compute_prior_mode.m` as called by `set_prior.m`); NaN for a uniform prior,
as in Dynare.
"""
function prior_mode(prior::Prior)
    out = fill(NaN, length(prior))
    for i in eachindex(prior.shape)
        sh, a, b, lo, hi = prior.shape[i], prior.p6[i], prior.p7[i], prior.p3[i], prior.p4[i]
        if sh == 1
            mo = if a > 1 && b > 1
                (a - 1) / (a + b - 2)
            elseif a < 1 && b < 1
                0.0                                     # two modes, 0 and 1: Dynare keeps the first
            elseif (a < 1 && b > 1 - eps()) || (abs(a - 1) < 2eps() && b > 1)
                0.0
            elseif (a > 1 && b < 1 + eps()) || (abs(a - 1) < 2eps() && b < 1)
                1.0
            else
                0.5
            end
            out[i] = mo * (hi - lo) + lo
        elseif sh == 2
            out[i] = (a < 1 ? 0.0 : (a - 1) * b) + lo
        elseif sh == 3
            out[i] = prior.p1[i]
        elseif sh == 4
            out[i] = 1 / sqrt((b + 1) / a) + lo
        elseif sh == 6
            out[i] = a / (b + 2) + lo
        elseif sh == 8
            out[i] = (a <= 1 ? 0.0 : b * ((a - 1) / a)^(1 / a)) + lo
        end
    end
    out
end

function _median_(x)
    s = sort(x)
    n = length(s)
    iseven(n) ? (s[n÷2] + s[n÷2+1]) / 2 : s[(n+1)÷2]
end

"""
    parameter_vector(est, parameter_set; mode=nothing, mcmc=nothing) -> Vector

The estimated-parameter vector Dynare's `evaluate_smoother` uses for `parameter_set`: `mode` (a `ModeResult` or a vector) for
`posterior_mode` / `mle_mode`, `mcmc` (an `MCMCResult`: mean / median of the draws kept after `mh_drop`, or a vector) for
`posterior_mean` / `posterior_median`, the priors for `prior_mean` / `prior_mode`.
"""
function parameter_vector(est::Estimation, pset::AbstractString; mode=nothing, mcmc=nothing)
    asvec(x, what) = x === nothing ? throw(SmootherError("parameter_set=$(repr(pset)): pass $what")) : Vector{Float64}(x isa ModeResult ? x.x : x)
    bayes = is_bayesian(est.prior)
    if pset in ("posterior_mode", "mle_mode")
        (pset == "mle_mode") == bayes && throw(SmootherError("parameter_set=$(repr(pset)) does not fit $(bayes ? "a Bayesian" : "a maximum-likelihood") estimation"))
        return asvec(mode, "mode=<ModeResult from posterior_mode(est)> or mode=<vector>")
    elseif pset in ("posterior_mean", "posterior_median")
        bayes || throw(SmootherError("parameter_set=$(repr(pset)) needs priors"))
        if mcmc isa MCMCResult
            kept = kept_draws(mcmc)
            return pset == "posterior_mean" ? vec(sum(kept; dims=1)) ./ size(kept, 1) : [_median_(kept[:, j]) for j in axes(kept, 2)]
        end
        return asvec(mcmc, "mcmc=<MCMCResult from sample_posterior(est)> or mcmc=<vector>")
    elseif pset in ("prior_mean", "prior_mode")
        bayes || throw(SmootherError("parameter_set=$(repr(pset)) needs priors"))
        x = pset == "prior_mean" ? copy(est.prior.p1) : prior_mode(est.prior)
        if !all(isfinite, x)
            bad = est.names[.!isfinite.(x)]
            throw(SmootherError("parameter_set=$(repr(pset)) is not defined for $bad (Dynare's mode of a uniform prior is NaN and its smoother then fails too)"))
        end
        return x
    end
    throw(SmootherError("unknown parameter_set $(repr(pset))"))
end

# ====================================================================== model-side inputs
_statements_of(m::Model, kind) = [st for st in m.statements if get(st, "statementName", "") == kind]

"Slopes of the `observation_trends` block in `varobs` order (`compute_trend_coefficients.m`); `nothing` without a block."
function _trend_coefficients(m::Model, varobs, params)
    blocks = _statements_of(m, "observation_trends")
    isempty(blocks) && return nothing
    coeff = zeros(length(varobs))
    locals = Dict{String,Float64}(zip(m.param_names, params))
    for st in blocks
        tr = get_or(st, "trends", Dict{String,Any}())
        unknown = [v for v in keys(tr) if !(v in varobs)]
        isempty(unknown) || throw(SmootherError("observation_trends on non-observed variable(s) $unknown"))
        for (v, e) in tr
            coeff[findfirst(==(v), varobs)] = Float64(_value(m, e; locals=locals))
        end
    end
    coeff
end

"""
Deviation of the period-0 state set by a `filter_initial_state` block (zero elsewhere); `nothing` without a block. Only state
variables are used, as in Dynare.
"""
function _initial_state(m::Model, sol::FirstOrderSolution, params, loglinear)
    blocks = _statements_of(m, "filter_initial_state")
    isempty(blocks) && return nothing
    a = zeros(m.n_endo)
    states = Set(sol.state_var)
    locals = Dict{String,Float64}(zip(m.param_names, params))
    for st in blocks, d in get_or(st, "states", [])
        Int(get(d, "lag", 0)) == 0 || throw(SmootherError("filter_initial_state on a lagged variable is not supported"))
        i = m.ie[d["var"]]
        i in states || continue
        val = Float64(_value(m, rstrip(string(d["value"]), ';'); locals=locals))
        a[i] = loglinear ? log(val) - sol.ys[i] : val - sol.ys[i]
    end
    a
end

const _FROM_ESTIMATION = ("datafile", "first_obs", "nobs", "prefilter", "loglinear", "logdata", "presample", "lik_init",
    "diffuse_filter", "kalman_algo", "kalman_tol", "diffuse_kalman_tol", "Harvey_scale_factor", "qz_criterium", "heteroskedastic_filter")

"select_qz_criterium_value.m"
function _select_qz(m::Model, lik_init, user_qz)
    qz = user_qz === nothing ? solver_option(m, "qz_criterium", nothing) : user_qz
    qz = qz === nothing ? nothing : Float64(qz)
    if qz === nothing
        return lik_init == 1 ? 1 - 1e-6 : 1 + 1e-6
    elseif lik_init == 1
        qz > 1 - eps() && throw(SmootherError("option qz_criterium is too large for estimating a stationary model. If your model contains unit roots, use option diffuse_filter"))
        return qz
    end
    qz <= 1 ? 1 + 1e-6 : qz
end

_blkdiag(A, B) = [A zeros(size(A, 1), size(B, 2)); zeros(size(B, 1), size(A, 2)) B]

# ====================================================================== the smoother
"""
    smooth(m::Model, data=nothing; sol=nothing, estimation=nothing, theta=nothing, mode=nothing, mcmc=nothing, options...) -> SmootherResult

Kalman filter and smoother of a first-order model (Dynare's `calib_smoother` and `evaluate_smoother`).

`data`: a matrix (columns in `varobs` order), a `Dict` of series or a file name; default the `datafile` option (relative to the
`.mod` file's folder). `sol`: a first-order solution (computed if omitted; calibration only).

Parameters used: the model's current calibration (`parameter_set="calibration"`). With `estimation=` (an `Estimation` of the same
model): `theta=vector` (ordered as `estimation.names`, mapped into structural parameters, shock standard deviations and
measurement-error parameters as `set_all_parameters`), or `parameter_set=` `"posterior_mode"` / `"mle_mode"` (with
`mode=posterior_mode(est)` or a vector), `"posterior_mean"` / `"posterior_median"` (with `mcmc=sample_posterior(est)` or a vector),
`"prior_mean"`, `"prior_mode"`. The model object is never modified.

Options are Dynare's and default to those of the file's `estimation` / `calib_smoother` statements (merged in file order);
keywords take precedence: `datafile first_obs nobs prefilter loglinear logdata diffuse_filter lik_init (1, 2, 3) Harvey_scale_factor
kalman_algo (0-4) filtered_vars filter_step_ahead filter_covariance smoothed_state_uncertainty kalman_tol diffuse_kalman_tol
qz_criterium varobs`. `presample` is accepted and has no effect on the smoother (as in Dynare). Options of Dynare's
statement that the code does not act on give one warning; ones that would change the numbers raise a `SmootherError`.

Blocks of the file: `observation_trends` (linear trends removed from the data, demeaned under `prefilter`; returned in `Trend` /
`TrendCoeffs`), `filter_initial_state` (the filter starts from the given state; stationary initialisations only).

For a model with `occbin_constraints`, the OccBin smoother runs after the linear one (Dynare's occbin.DSGE_smoother, or the
inversion filter with `smoother_inversion_filter`); `res.occbin` holds its output and `occbin=Dict(...)` overrides OccBin options
(src/occbin_estimation.jl).

`replicate_dynare_bugs=true` reproduces Dynare 7.2's defects instead of correcting them (see the head of this file):
the multivariate diffuse smoother (`kalman_algo=3`) when a period of the diffuse phase has no observation or a zero `F_inf`,
and the `filter_initial_state` centring of `Init_State`.
"""
smooth(m::Model, data=nothing; kw...) = isempty(m.occbin_constraints) ? _smooth_linear(m, data; kw...) : _occbin_smooth(m, data; kw...)

function _smooth_linear(m::Model, data=nothing; sol=nothing, estimation=nothing, theta=nothing, mode=nothing, mcmc=nothing,
                        reuse=nothing, kw...)
    est = estimation
    user = Dict{String,Any}(string(k) => v for (k, v) in kw)
    kw_qz = Dict{String,Any}(k => v for (k, v) in user if k == "qz_criterium" && v !== nothing)
    if est !== nothing
        est.model === m || est.model.endo_names == m.endo_names || throw(SmootherError("estimation= was built for another model"))
        for k in _FROM_ESTIMATION
            haskey(user, k) || (haskey(est.options, k) && est.options[k] !== nothing && (user[k] = est.options[k]))
        end
    end
    reuse === nothing || (est = nothing)
    opts = _resolve_options(m, user)
    isempty(m.exo_names) && throw(SmootherError("the model has no shocks"))
    varobs = opts["varobs"] === nothing ? copy(m.varobs) : String.(opts["varobs"])
    isempty(varobs) && throw(SmootherError("no observed variables: declare varobs or pass varobs=[...]"))
    for v in varobs
        haskey(m.ie, v) || throw(SmootherError("observed variable $v is not an endogenous variable"))
    end
    n, nx, pp = m.n_endo, m.n_exo, length(varobs)
    mf = Int[m.ie[v] for v in varobs]
    loglinear = opts["loglinear"]
    pset = opts["parameter_set"]
    lik_init = opts["lik_init"]
    algo = opts["kalman_algo"]
    if opts["diffuse_filter"] || algo in (3, 4)
        lik_init == 2 && throw(SmootherError("options diffuse_filter, lik_init and/or kalman_algo have contradictory settings"))
        lik_init = 3
    end
    lik_init in (1, 2, 3) || throw(SmootherError("lik_init=$lik_init is not implemented (1, 2 and 3 are)"))
    algo in (0, 1, 2, 3, 4) || throw(SmootherError("kalman_algo=$algo is not implemented"))
    qz = (est !== nothing && !haskey(kw_qz, "qz_criterium") && lik_init == est.options["lik_init"]) ? est.options["qz_criterium"] :
         _select_qz(m, lik_init, get(kw_qz, "qz_criterium", nothing))

    # ---- parameters (Dynare's set_all_parameters when a vector is used)
    if reuse !== nothing
        sol, Q, Hme, theta, pset = reuse.solution, reuse.Sigma_e, reuse.me_cov, reuse.theta, reuse.parameter_set
    else
        if theta === nothing && pset != "calibration"
            est === nothing && throw(SmootherError("parameter_set=$(repr(pset)) needs estimation=<Estimation> (or pass the vector with theta=...)"))
            theta = parameter_vector(est, pset; mode=mode, mcmc=mcmc)
        elseif theta !== nothing
            est === nothing && throw(SmootherError("theta= needs estimation=<Estimation> to map the vector into the model"))
            haskey(user, "parameter_set") || (pset = "theta")
        end
        if theta !== nothing
            sol === nothing || throw(SmootherError("sol= cannot be combined with a parameter vector"))
            varobs == est.varobs || throw(SmootherError("varobs differs from the observed variables of the estimation"))
            theta = Vector{Float64}(theta)
            length(theta) == length(est.names) || throw(SmootherError("theta has $(length(theta)) entries for the $(length(est.names)) estimated parameters"))
            Q, Hme = covariances(est, theta)
            sol = _first_order_checked(m, model_params(est, theta), qz, loglinear, lik_init)
        else
            Q = Matrix{Float64}(m.Sigma_e)
            extra = sort([v for v in keys(m.measurement_errors) if !(v in varobs)])
            isempty(extra) || throw(SmootherError("measurement error declared on non-observed variable(s) $extra"))
            Hme = me_H(m, varobs)
            if sol === nothing
                sol = _first_order_checked(m, nothing, qz, loglinear, lik_init)
            elseif sol.loglinear != loglinear
                throw(SmootherError("sol= was computed with loglinear=$(sol.loglinear), the smoother asks for loglinear=$loglinear"))
            end
        end
    end
    opts["parameter_set"] = pset
    params = Vector{Float64}(sol.params)
    ys_level = Vector{Float64}(sol.steady_state)
    const_all = copy(sol.ys)                   # log steady state under loglinear (for the variables that are logged)
    G, Hs = Matrix{Float64}(sol.G), Matrix{Float64}(sol.H)

    # ---- data (Dynare's makedataset)
    first = opts["first_obs"]
    if reuse !== nothing
        j = something(opts["nobs"], size(reuse.sample, 1))
        j <= size(reuse.sample, 1) || throw(SmootherError("nobs=$j exceeds the sample of the result being re-run"))
        obs = reuse.sample[1:j, :]
        first = reuse.first_obs
    else
        src = data === nothing ? opts["datafile"] : data
        src === nothing && throw(SmootherError("no data: pass data=... or a datafile option"))
        raw = try
            load_data(src, varobs; base_dir=m.mod_dir)
        catch e
            e isa EstimationError ? throw(SmootherError(e.msg)) : rethrow()
        end
        if loglinear && !opts["logdata"]
            any(x -> !isnan(x) && x <= 0, raw) && throw(SmootherError("loglinear without logdata: the data contain non-positive values"))
            raw = log.(raw)
        end
        nob = opts["nobs"] === nothing ? size(raw, 1) - first + 1 : opts["nobs"]
        (first < 1 || nob < 1 || first - 1 + nob > size(raw, 1)) &&
            throw(SmootherError("first_obs=$first, nobs=$nob do not fit the $(size(raw, 1)) observations in the data"))
        obs = raw[first:first+nob-1, :]
    end
    smpl = size(obs, 1)
    obs_const = const_all[mf]
    prefilter = opts["prefilter"] == 1
    if prefilter
        any(c -> abs(c) > 1e-9, const_all[mf]) &&
            throw(SmootherError("prefilter=1 is inconsistent with observed variables that have a non-zero steady state (same check as Dynare)"))
        obs_const = Float64[(c = filter(!isnan, obs[:, j]); isempty(c) ? 0.0 : sum(c) / length(c)) for j in 1:pp]
    end
    trend_coeff = _trend_coefficients(m, varobs, params)
    with_trend = trend_coeff !== nothing
    trend_coeff === nothing && (trend_coeff = zeros(pp))
    trend = collect(Float64, first:first+smpl-1) * trend_coeff'              # (T, pp)
    prefilter && (trend = trend .- sum(trend; dims=1) ./ smpl)
    Yd = permutedims(obs .- obs_const' .- trend)                           # (pp, T), deviations
    di = [findall(!isnan, Yd[:, t]) for t in 1:smpl]

    # ---- state space and initial conditions
    Z = zeros(pp, n)
    for (i, k) in enumerate(mf)
        Z[i, k] = 1.0
    end
    Hdiagonal = all(abs.(Hme .- Diagonal(diag(Hme))) .< 1e-14)
    ktol, dtol = opts["kalman_tol"], opts["diffuse_kalman_tol"]
    compat = opts["replicate_dynare_bugs"]
    Pstar = Pinf = nothing
    ST, R1, Qx, Zx = G, Hs, Q, Z
    expanded = false
    if lik_init == 1
        algo = algo == 2 ? 2 : 1
        Pstar = lyapunov_symm(G, Hs * Q * Hs'; qz_criterium=qz, complex_threshold=Float64(solver_option(m, "lyapunov_complex_threshold", 1e-15)))[1]
    elseif lik_init == 2
        opts["Harvey_scale_factor"] == 0 && throw(SmootherError("Harvey_scale_factor=0 is not implemented"))
        algo = algo == 2 ? 2 : 1
        Pstar = opts["Harvey_scale_factor"] * Matrix{Float64}(I, n, n)
    else
        my_mf = mf
        if algo != 4
            algo = 3
        elseif !Hdiagonal                   # Dynare augments the state vector (Section 6.4.3 of Durbin and Koopman 2012)
            expanded = true
            ST, Qx, R1, Zx = _blkdiag(G, zeros(pp, pp)), _blkdiag(Q, Hme), _blkdiag(Hs, Matrix{Float64}(I, pp, pp)), [Z Matrix{Float64}(I, pp, pp)]
            my_mf = findall(vec(any(Zx .!= 0; dims=1)))
        end
        Pstar, Pinf = compute_Pinf_Pstar(my_mf, ST, R1, Qx, qz)
    end
    a_init = _initial_state(m, sol, params, loglinear)
    a_pred = a_zero = nothing
    if a_init !== nothing && any(!iszero, a_init)
        lik_init == 3 && throw(SmootherError("filter_initial_state with the diffuse filter is not supported (Dynare 7.2 applies the transition matrix twice to the prescribed state there)"))
        a_pred = G * a_init
        a_zero = compat ? a_pred : a_init
    end
    fsa = opts["filter_step_ahead"]
    nk = isempty(fsa) ? 1 : maximum(fsa)
    want_V, want_PK = opts["smoothed_state_uncertainty"], opts["filter_covariance"]
    want_decomp = _flag(_unwrap1(opts["filter_decomposition"]))
    want_decomp && isempty(fsa) && (want_decomp = false)         # stored only with filter_step_ahead (store_smoother_results.m)

    # heteroskedastic_filter (src/heteroskedastic.jl): time-varying Q, multivariate smoother only
    het = _flag(_unwrap1(opts["heteroskedastic_filter"]))
    het && algo in (2, 4) && throw(SmootherError("heteroskedastic_filter: only the multivariate smoother is implemented"))
    Qs = het ? heteroskedastic_qvec(Q, heteroskedastic_arrays(m, smpl, first)...) : Q
    out = nothing
    if algo in (1, 3)
        _SINGULAR_F[] = false
        out = _smoother_multivariate(G, Z, Hs, Qs, Hme, Pinf, Pstar, Yd, di, nk, ktol, dtol, want_V, want_PK, compat, a_pred, a_zero; want_decomp)
        if out === nothing
            # DsgeSmoother.m: stochastic singularity in a stationary period is an error (print_info) unless
            # use_univariate_smoother_if_singularity_is_detected is set (or noprint)
            if _SINGULAR_F[] && !_flag(_unwrap1(opts["use_univariate_smoother_if_singularity_is_detected"])) && !_flag(_unwrap1(opts["noprint"]))
                throw(SmootherError("Kalman filter: F is singular in stationary period. Stochastic singularity detected. " *
                                    "(set use_univariate_smoother_if_singularity_is_detected to switch to the univariate smoother)"))
            end
            algo += 1                              # Dynare: switch to the univariate filter
        end
    end
    if out === nothing
        het && throw(SmootherError("heteroskedastic_filter: only the multivariate smoother is implemented (the filter needs the univariate one here)"))
        if !Hdiagonal && !expanded
            expanded = true
            Zx = [Z Matrix{Float64}(I, pp, pp)]
            ST = _blkdiag(G, zeros(pp, pp))
            Qx = _blkdiag(Q, Hme)
            R1 = _blkdiag(Hs, Matrix{Float64}(I, pp, pp))
            if algo == 4
                Pstar, Pinf = compute_Pinf_Pstar(mf, ST, R1, Qx, qz)
            else
                Pstar = _blkdiag(Pstar, Hme)
                Pinf === nothing || (Pinf = _blkdiag(Pinf, zeros(pp, pp)))
            end
        end
        Hd = expanded ? zeros(pp) : diag(Hme)
        ap = a_pred === nothing ? nothing : (expanded ? vcat(a_pred, zeros(pp)) : a_pred)
        az = a_zero === nothing ? nothing : (expanded ? vcat(a_zero, zeros(pp)) : a_zero)
        out = _smoother_univariate(ST, Zx, R1, Qx, Hd, Pinf, Pstar, Yd, di, nk, ktol, dtol, want_V, want_PK, compat, ap, az; want_decomp)
    end
    alphahat, etahat, updated, aK, Pout = out.alphahat, out.etahat, out.updated, out.aK, out.P
    PKout, Vout, V0out, alpha0 = out.PK, out.V, out.V0, out.alphahat0
    decomp = out.decomp
    eps = permutedims(Yd) .- permutedims(alphahat[mf, :])                 # NaN where the observation is missing
    if expanded && algo in (2, 4)
        k = 1:n
        eps = permutedims(etahat[end-pp+1:end, :])
        etahat = etahat[1:end-pp, :]
        alphahat, updated, alpha0 = alphahat[k, :], updated[k, :], alpha0[k]
        aK = aK[:, k, :]
        PKout === nothing || (PKout = PKout[:, k, k, :])
        decomp === nothing || (decomp = decomp[:, k, :, :])
        Pout = Pout[k, k, :]
        Vout === nothing || (Vout = Vout[k, k, :])
        V0out === nothing || (V0out = V0out[k, k])
    end

    # ---- Dynare's store_smoother_results
    alpha = permutedims(alphahat)                     # (T, n)
    upd = permutedims(updated)
    cst = repeat(const_all', smpl)
    for (i, k) in enumerate(mf)
        cst[:, k] .= obs_const[i]
    end
    full_trend = zeros(smpl, n)
    full_trend[:, mf] = trend
    step = zeros(n)
    step[mf] = trend_coeff
    sv, uv = alpha .+ cst .+ full_trend, upd .+ cst .+ full_trend
    fv = permutedims(aK[1, :, 2:smpl+1]) .+ cst .+ full_trend .+ step'
    kstep = kvar = kdec = nothing
    if !isempty(fsa)
        mx = maximum(fsa)
        kc = reshape(const_all, 1, n, 1)
        kstep = aK[fsa, :, :] .+ kc
        for (j, kk) in enumerate(fsa)                   # observed variables: constant and trend of the data
            hi = size(kstep, 3) - (mx - kk)
            for (i, k) in enumerate(mf)
                kstep[j, k, kk+1:hi] = aK[kk, k, kk+1:hi] .+ obs_const[i] .+ trend[:, i] .+ kk * trend_coeff[i]
            end
        end
        want_PK && (kvar = PKout[fsa, :, :, :])
        kdec = decomp === nothing ? nothing : decomp[fsa, :, :, :]
    end
    nm = m.endo_names
    cols(M, names) = Dict{String,Vector{Float64}}(v => M[:, i] for (i, v) in enumerate(names))
    me_present = any(!iszero, Hme)
    res = SmootherResult(
        endo_names=copy(nm), exo_names=copy(m.exo_names), varobs=varobs,
        SmoothedVariables=cols(sv, nm), UpdatedVariables=cols(uv, nm), FilteredVariables=cols(fv, nm),
        SmoothedShocks=cols(permutedims(etahat), m.exo_names),
        SmoothedMeasurementErrors=me_present ? cols(eps, varobs) : Dict{String,Vector{Float64}}(),
        Constant=cols(cst, nm),
        Init_State=Dict{String,Float64}(v => alpha0[i] + const_all[i] for (i, v) in enumerate(nm)),
        Init_State_Vector=alpha0 .+ ys_level,        # Dynare adds the level steady state, also with loglinear
        SteadyState=ys_level, loglinear=loglinear,
        FilteredVariablesKStepAhead=kstep, FilteredVariablesKStepAheadVariances=kvar, FilteredVariablesShockDecomposition=kdec, filter_step_ahead=copy(fsa),
        Variance=want_PK ? Pout : nothing, State_uncertainty=want_V ? Vout : nothing,
        Init_State_uncertainty=want_V ? V0out : nothing,
        smoothed=alpha, updated=upd, init_state=alpha0, shocks=permutedims(etahat), measurement_errors=eps,
        data=permutedims(Yd), obs_constant=obs_const, me_variance=diag(Hme), G=G, H=Hs, obs_index=mf,
        kalman_algo=algo, diffuse_periods=out.d, options=opts,
        Trend=cols(full_trend, nm), TrendCoeffs=step,
        parameter_set=pset, theta=theta === nothing ? nothing : Vector{Float64}(theta), params=params,
        Sigma_e=Q, me_cov=Hme, solution=sol, sample=obs, first_obs=first)
    res
end

"First-order solution at the steady state with Dynare's checks; a Blanchard-Kahn failure of a stationary initialisation mentions unit roots."
function _first_order_checked(m::Model, params, qz, loglinear, lik_init)
    sol = try
        first_order(m; params=params, qz_criterium=qz, loglinear=loglinear)
    catch e
        if e isa BlanchardKahnError && lik_init != 3
            throw(SmootherError("the model cannot be solved with qz_criterium=$qz ($(e.msg)); if it has unit roots, use diffuse_filter"))
        end
        rethrow()
    end
    (all(isfinite, sol.G) && all(isfinite, sol.H)) || throw(SmootherError("the decision rule contains NaN or Inf"))
    sol
end

"Internal-consistency errors of a smoother result (rounding level): transition, simulation, measurement (and decomposition if given)."
function check_smoother(res::SmootherResult, decomp=nothing; sol=nothing)
    G, Hh = sol === nothing ? (res.G, res.H) : (Matrix(sol.G), Matrix(sol.H))
    (sol !== nothing && res.loglinear) && throw(SmootherError("loglinear result: use the stored (log) decision rules, not sol"))
    prev = vcat(res.init_state', res.smoothed[1:end-1, :])
    one_step = prev * G' + res.shocks * Hh'
    y = copy(res.init_state)
    sim = similar(res.smoothed)
    for t in 1:nobs(res)
        y = G * y + Hh * res.shocks[t, :]
        sim[t, :] = y
    end
    out = Dict{String,Float64}("transition" => maximum(abs.(one_step - res.smoothed)), "simulation" => maximum(abs.(sim - res.smoothed)))
    exact = res.me_variance .== 0
    seen = .!isnan.(res.data[:, exact])
    ee = res.measurement_errors[:, exact]
    out["measurement"] = any(seen) ? maximum(abs.(ee[seen])) : 0.0
    if decomp !== nothing
        z = decomp.z
        nc = size(z, 2) - 2
        out["decomposition"] = max(maximum(abs.(dropdims(sum(z[:, 1:end-1, :]; dims=2); dims=2) - z[:, end, :])),
                                   maximum(abs.(z[:, end, :] - permutedims(res.smoothed))))
        if decomp.kind == "shocks" && decomp.init_state == 0
            y, err = copy(res.init_state), 0.0
            for t in 1:nobs(res)
                y = G * y
                err = max(err, maximum(abs.(z[:, nc+1, t] - y)))
            end
            out["decomposition_initial"] = err
        end
    end
    out
end
