# Estimation of second-order approximated models with Dynare's nonlinear filters: `estimation(order=2, filter_algorithm=...)`.
#
# Port of pydyn/nonlinear_estimation.py, after Dynare 7.2's matlab/estimation/non_linear_dsge_likelihood.m,
# check_particle_filter_options.m, the particle branches of dynare_estimation_1.m, dynare_estimation_init.m,
# initial_estimation_checks.m and select_qz_criterium_value.m, and kalman_transition_matrix.m.
#
# An `Estimation` with `order=2` carries the particle options in `est.options["particle"]` (a `ParticleOptions`) and the
# bookkeeping of the reduced form in `est.options["nl"]`; `_dsge_likelihood` then calls `_nonlinear_dsge_likelihood`:
# the model is solved at order 2 (Dynare's k-order solver), restricted to the state and observed variables, and one of the
# filters of src/nonlinear_filters.jl is run on the (undemeaned) data. As in Dynare, every filter starts its random numbers
# afresh with the 'default' seed (0), so the likelihood is a deterministic function of the parameters (common random
# numbers) and the optimisers and the Metropolis-Hastings sampler work unchanged. With `particle_rng=:octave` the numbers
# are Octave's, so the likelihood equals Dynare's under Octave (see src/octave_rng.jl); `:julia` (default, or the model's
# `set_rng_kind!`) uses Julia's Xoshiro with the same seeding.
#
# Not implemented, refused with an error: order 3 and above, `filter_algorithm=gmf`, `nonlinear_filter_initialization=2`,
# `resampling_method` residual/multinomial (Dynare stops with "Unknown sampling method!"), `loglinear`, prefilter, trends,
# missing observations (Dynare's filters do not handle them), OccBin, `cpf_weights=murrayjonesparslow` with a
# cubature/Monte Carlo proposal and correlated measurement errors with gf, and the combinations with which Dynare's code
# stops with an error. Dynare computes no smoother or forecast with particle filters.

const _PARTICLE_KW = Set(["filter_algorithm", "number_of_particles", "resampling", "resampling_threshold", "resampling_method",
    "proposal_approximation", "distribution_approximation", "cpf_weights", "nonlinear_filter_initialization",
    "particle_filter_options", "initial_state_prior_std", "unscented_alpha", "unscented_beta", "unscented_kappa", "particle_pruning"])

"The bookkeeping of the reduced form of an order-2 `Estimation` (`est.options[\"nl\"]`)."
struct NLSetup
    po::ParticleOptions
    kind::Any                         # :julia, :octave, a ParticleStream or a function returning one; nothing = the model's
    seed::Float64
    model_pruning::Bool               # options_.pruning
    k2::Vector{Int}                   # dr.restrict_var_list (positions in decision-rule order)
    mf0::Vector{Int}                  # bayestopt_.mf0: the state variables in the restricted list
    mf1::Vector{Int}                  # bayestopt_.mf1: the observed variables in the restricted list
    order_var::Vector{Int}
end

"Move the particle-filter options out of the raw option dictionary of an order-2 `Estimation`."
function _pop_particle_options!(raw::Dict{String,Any})
    out = Dict{String,Any}()
    for k in collect(keys(raw))
        if startswith(k, "particle.") || k in _PARTICLE_KW || k in ("particle_rng", "particle_seed", "pruning")
            out[k] = pop!(raw, k)
        end
    end
    out
end

_nl_unwrap1(v) = (v isa AbstractVector && length(v) == 1) ? v[1] : v

"The `order` of a raw option dictionary (1 if absent)."
function _est_order(raw::AbstractDict)
    v = _nl_unwrap1(get(raw, "order", 1))
    v isa Real ? Int(round(v)) : 1
end

"""
Options of an order-2 estimation that Dynare answers with an error or that have no meaning with a particle filter: no
smoother, filtered variables or forecasts (\"Particle Smoothers are not yet implemented\"), no Metropolis-Hastings run
without a mode search.
"""
function _check_nonlinear_options!(est::Estimation; smoother=false, filtered=false, forecast=0, mh=false)
    smoother && throw(EstimationError("Estimation:: Particle Smoothers are not yet implemented."))
    filtered && throw(EstimationError("estimation: filtered_vars / filter_step_ahead with a nonlinear filter: Particle Smoothers are not yet implemented."))
    forecast > 0 && throw(EstimationError("estimation: forecast with a nonlinear filter needs a smoother, which Dynare does not have for particle filters"))
    mh && est.options["mode_compute"] == 0 && is_bayesian(est.prior) &&
        throw(Unsupported("mode_compute=0 with mh_replic>0 and a nonlinear filter needs a mode_file (the proposal covariance); not implemented"))
    nothing
end

"`read_key_value_string`: \"'key', value, 'key', value\" with numbers or true/false."
function _parse_key_values(s::AbstractString)
    s = replace(s, "''" => "'")
    parts = String[strip(p) for p in split(s, ",") if !isempty(strip(p))]
    isodd(length(parts)) && throw(EstimationError("read_key_value_string: Wrong number of Key-Value pairs!"))
    out = Dict{String,Any}()
    for i in 1:2:length(parts)
        k = strip(parts[i], ['\'', '"'])
        v = parts[i+1]
        lv = lowercase(v)
        out[k] = lv == "true" ? true : lv == "false" ? false : something(tryparse(Float64, v), strip(v, ['\'', '"']))
    end
    out
end

"""
    particle_options(raw) -> ParticleOptions

The particle options of an estimation statement (the preprocessor's `particle.*` names, flags and all) or of keywords
(`filter_algorithm`, `number_of_particles`, `resampling`, `resampling_threshold`, `resampling_method`,
`proposal_approximation`, `distribution_approximation`, `cpf_weights`, `nonlinear_filter_initialization`,
`particle_filter_options`, `unscented_alpha`, ...; keywords win over the statement), as check_particle_filter_options.m.
"""
function particle_options(raw::AbstractDict)
    po = ParticleOptions()
    pfo = nothing
    kws = Set(k for k in keys(raw) if k in _PARTICLE_KW)
    for (k, v0) in raw
        startswith(k, "particle.") || continue
        v = _nl_unwrap1(v0)
        sub = k[length("particle.")+1:end]
        if sub == "filter_algorithm"
            po.filter_algorithm = string(v)
        elseif sub == "number_of_particles"
            po.number_of_particles = Int(v)
        elseif sub == "resampling.threshold"
            po.resampling_threshold = Float64(v)
        elseif sub == "initialization"
            po.initialization = Int(v)
        elseif sub == "particle_filter_options"
            pfo = v
        elseif sub == "status"
        elseif split(sub, ".")[1] in ("resampling", "proposal_approximation", "distribution_approximation", "cpf_weights_method")
        else
            throw(Unsupported("particle option $k is not supported"))
        end
    end
    for (group, field, kw) in (("resampling.status", :resampling, "resampling"), ("resampling.method", :resampling_method, "resampling_method"),
                               ("proposal_approximation", :proposal_approximation, "proposal_approximation"),
                               ("distribution_approximation", :distribution_approximation, "distribution_approximation"),
                               ("cpf_weights_method", :cpf_weights, "cpf_weights"))
        kw in kws && continue
        on = sort([split(k, ".")[end] for (k, v) in raw if startswith(k, "particle.$group.") && _est_flag(_nl_unwrap1(v))])
        isempty(on) || setfield!(po, field, String(on[end]))
    end
    for k in sort(collect(kws))
        v = _nl_unwrap1(raw[k])
        if k == "particle_filter_options"
            pfo = v
        elseif k == "nonlinear_filter_initialization"
            po.initialization = Int(v)
        elseif k == "number_of_particles"
            po.number_of_particles = Int(v)
        elseif k == "resampling_threshold"
            po.resampling_threshold = Float64(v)
        elseif k == "particle_pruning"
            po.pruning = _est_flag(v)
        elseif k in ("unscented_alpha", "unscented_beta", "unscented_kappa", "initial_state_prior_std")
            setfield!(po, Symbol(k), Float64(v))
        else
            setfield!(po, Symbol(k), string(v))
        end
    end
    if pfo !== nothing && !(pfo isa AbstractString && isempty(pfo))
        for (k, v) in (pfo isa AbstractString ? _parse_key_values(pfo) : Dict{String,Any}(string(a) => b for (a, b) in pfo))
            if k == "pruning"
                v isa Bool || throw(EstimationError("check_particle_filter_options:: the pruning options takes only true or false"))
                po.pruning = v
            elseif k in ("unscented_alpha", "unscented_beta", "unscented_kappa", "initial_state_prior_std")
                ((k in ("unscented_alpha", "initial_state_prior_std") && v <= 0) || v < 0) &&
                    throw(EstimationError("check_particle_filter_options:: the $k option takes a positive argument"))
                setfield!(po, Symbol(k), Float64(v))
            else
                throw(Unsupported("particle_filter_options: $k is not supported"))
            end
        end
    end
    _check_particle_options(po)
    po
end

function _check_particle_options(po::ParticleOptions)
    if !haskey(NONLINEAR_FILTERS, po.filter_algorithm)
        po.filter_algorithm == "gmf" && throw(Unsupported("filter_algorithm=gmf (Gaussian mixture filter) is not implemented"))
        throw(EstimationError("Estimation: Unknown filter $(po.filter_algorithm)"))
    end
    po.resampling in ("systematic", "none", "generic") || throw(EstimationError("unknown resampling option $(po.resampling)"))
    po.resampling_method in ("kitagawa", "stratified", "smooth") ||
        throw(Unsupported("resampling_method=$(po.resampling_method): Dynare 7.2 stops with 'Unknown sampling method!' (resample.m only has kitagawa, stratified and smooth)"))
    for (nm, v) in (("proposal_approximation", po.proposal_approximation), ("distribution_approximation", po.distribution_approximation))
        v in ("unscented", "cubature", "montecarlo") || throw(EstimationError("unknown $nm $v"))
    end
    po.cpf_weights in ("amisanotristani", "murrayjonesparslow") || throw(EstimationError("unknown cpf_weights $(po.cpf_weights)"))
    po.initialization == 2 && throw(Unsupported("nonlinear_filter_initialization=2 (ergodic variance from a 5000-period simulation) is not implemented"))
    po.initialization in (1, 3) || throw(EstimationError("Unknown initialization option!"))
    po.number_of_particles >= 1 || throw(EstimationError("number_of_particles must be positive"))
    if po.pruning && po.filter_algorithm in ("gf", "nlkf", "cpf")
        nm = Dict("gf" => "gaussian_filter", "nlkf" => "nonlinear_kalman_filter", "cpf" => "conditional_particle_filter")[po.filter_algorithm]
        throw(Unsupported("$nm: pruning is not yet implemented (Dynare 7.2 stops too)"))
    end
    po.filter_algorithm == "cpf" && po.cpf_weights == "murrayjonesparslow" && po.proposal_approximation != "unscented" &&
        throw(Unsupported("cpf_weights=murrayjonesparslow with proposal_approximation=cubature or montecarlo: Dynare 7.2 uses a (points x points) matrix as the predicted state variance there"))
    po.filter_algorithm == "apf" && po.resampling_method == "smooth" &&
        throw(Unsupported("auxiliary_particle_filter: resampling_method=smooth is not supported (Dynare 7.2)"))
    po
end

"""
Complete an order-2 `Estimation`: the particle options, the checks of initial_estimation_checks.m and
select_qz_criterium_value.m, and the restricted variable lists of dynare_estimation_init.m.
"""
function _setup_nonlinear!(est::Estimation, raw::Dict{String,Any})
    o = est.options
    m = est.model
    o["loglinear"] && throw(EstimationError("non_linear_dsge_likelihood: It is not possible to use a non linear filter with the option loglinear!"))
    o["prefilter"] == 1 && throw(EstimationError("initial_estimation_checks:: particle filtering does not support the prefilter option"))
    o["prefilter"] != 0 && throw(EstimationError("prefilter with a nonlinear filter is not supported"))
    est.trend_expr === nothing || throw(EstimationError("initial_estimation_checks:: particle filtering does not support trends"))
    any(isnan, est.data) && throw(EstimationError("initial_estimation_checks:: particle filtering does not support missing observations"))
    isempty(m.occbin_constraints) || throw(Unsupported("OccBin constraints with a second-order approximation (nonlinear filters) are not a Dynare feature"))
    po = particle_options(raw)
    model_pruning = _est_flag(get(raw, "pruning", false))
    po.filter_algorithm == "cpf" && model_pruning &&
        throw(Unsupported("filter_algorithm=cpf with the pruning option: Dynare 7.2 stops with an error (iterate_law_of_motion called with pruning but without the pruned states)"))
    kind = get(raw, "particle_rng", nothing)
    seed = Float64(_nl_unwrap1(get(raw, "particle_seed", 0)))
    user_qz = solver_option(m, "qz_criterium", nothing)
    user_qz = user_qz === nothing ? nothing : Float64(user_qz)
    # select_qz_criterium_value.m, particle branch
    if po.initialization == 3
        o["qz_criterium"] = (user_qz === nothing || user_qz <= 1) ? 1 + 1e-6 : user_qz
    else
        o["qz_criterium"] = user_qz === nothing ? 1 - 1e-6 : user_qz
    end
    order_var, _, cnt = dr_ordering(m.lead_lag_incidence)
    obs_dr = invperm(order_var)[est.obs_decl]
    state_dr = collect(cnt.nstatic+1:cnt.nstatic+cnt.npred+cnt.nboth)
    k2 = sort(union(obs_dr, state_dr))
    mf0 = Int[findfirst(==(s), k2) for s in state_dr]
    mf1 = Int[findfirst(==(s), k2) for s in obs_dr]
    o["particle"] = po
    o["nl"] = NLSetup(po, kind, seed, model_pruning, k2, mf0, mf1, order_var)
    o["order"] = 2
    est.last_filter = "nonlinear ($(po.filter_algorithm))"
    est
end

struct _NLReject <: Exception
    info::Int
    penalty::Float64
    msg::String
end

"""
    _nl_reduced_form(est, theta, Q, H) -> NLReducedForm

Second-order reduced form and initial state distribution at `theta` (the `ReducedForm` of non_linear_dsge_likelihood.m).
Throws `_NLReject` with Dynare's info codes.
"""
function _nl_reduced_form(est::Estimation, theta::AbstractVector, Q::Matrix{Float64}, H::Matrix{Float64})
    m = est.model
    o = est.options
    nl::NLSetup = o["nl"]
    po = nl.po
    local ys, params
    try
        ys, params = steady_state(m; params=model_params(est, theta))
    catch e
        e isa SteadyStateError && throw(_NLReject(20, 0.1, "steady state: $(e.msg)"))
        (e isa DomainError || e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException) &&
            throw(_NLReject(20, 0.1, "steady state: $(sprint(showerror, e))"))
        rethrow()
    end
    params = Vector{Float64}(params)
    all(isfinite, params) || throw(_NLReject(20, 0.1, "parameters are NaN or Inf after the steady-state computation"))
    local dr
    try
        dr = k_order(m; order=2, ys=Vector{Float64}(ys), params=params, pruning=false, loglinear=false,
                     qz_criterium=o["qz_criterium"], Sigma_e=Q)
    catch e
        e isa KOrderError || rethrow()
        pen = 0.1
        try                                                        # Dynare's penalty of the failed first-order solution
            first_order(m; params=model_params(est, theta), qz_criterium=o["qz_criterium"])
        catch e2
            e2 isa BlanchardKahnError && isfinite(e2.penalty) && (pen = e2.penalty)
            e2 isa Union{BlanchardKahnError,SteadyStateError} || rethrow()
        end
        throw(_NLReject(3, pen, "solution: $(e.msg)"))
    end
    k2, mf0 = nl.k2, nl.mf0
    G(nm) = Matrix{Float64}(dr[nm][k2, :])
    ghx, ghu, ghxx, ghuu, ghxu = G("ghx"), G("ghu"), G("ghxx"), G("ghuu"), G("ghxu")
    (all(isfinite, ghx) && all(isfinite, ghu) && all(isfinite, ghxx) && all(isfinite, ghuu) && all(isfinite, ghxu)) ||
        throw(_NLReject(6, 0.1, "the decision rule contains NaN or Inf"))
    ghs2 = Vector{Float64}(vec(dr["ghs2"])[k2])
    yss = Vector{Float64}(ys)[nl.order_var[k2]]
    rf = NLReducedForm(ghx, ghu, ghxx, ghuu, ghxu, ghs2, yss, yss .+ 0.5 .* ghs2, yss[mf0], nl.mf0, nl.mf1, Q, H,
                       Float64[], zeros(0, 0))
    rf.StateVectorMean = rf.constant[mf0]
    if po.initialization == 1
        # kalman_transition_matrix(dr, restrict_var_list, restrict_columns); lyapunov_symm
        A = zeros(length(k2), length(k2))
        A[:, mf0] = ghx
        V = try
            lyapunov_symm(A, ghu * Q * ghu'; qz_criterium=o["qz_criterium"], complex_threshold=o["lyapunov_complex_threshold"])[1]
        catch e
            (e isa LinearAlgebra.SingularException || e isa LinearAlgebra.LAPACKException || e isa ErrorException || e isa DomainError) || rethrow()
            throw(_NLReject(201, 0.1, "the initial state covariance matrix could not be computed (info 201)"))
        end
        rf.StateVectorVariance = Matrix{Float64}(V[mf0, mf0])
    else
        rf.StateVectorVariance = po.initial_state_prior_std * Matrix{Float64}(I, length(mf0), length(mf0))
    end
    (_ispd(rf.StateVectorVariance) && all(isfinite, rf.StateVectorVariance)) ||
        throw(_NLReject(201, 0.1, "the initial state covariance matrix is not positive definite (info 201)"))
    rf
end

"The random-number stream of one likelihood evaluation (reset at every evaluation, as Dynare's filters reseed with 'default')."
function _nl_stream(est::Estimation)
    nl::NLSetup = est.options["nl"]
    kind = nl.kind === nothing ? rng_kind(est.model) : nl.kind
    make_stream(kind isa AbstractString ? Symbol(kind) : kind, nl.seed)
end

"""
    _nonlinear_dsge_likelihood(est, theta; bounds, with_lik) -> (fval, info, penalty, loglik, logprior, lik)

Dynare's `non_linear_dsge_likelihood` with the layout of `_dsge_likelihood`.
"""
function _nonlinear_dsge_likelihood(est::Estimation, theta::AbstractVector; bounds=(est.lb, est.ub), with_lik=false, rf_out=nothing)
    o = est.options
    nl::NLSetup = o["nl"]
    theta = Float64.(theta)
    est.last_info = ""
    est.last_info_code = 0
    reject(info, pen, msg) = (est.last_info = msg; est.last_info_code = info; (Inf, info, pen, NaN, NaN, Float64[]))
    lb, ub = bounds
    if any(theta .< lb)
        k = findall(theta .< lb)
        return reject(41, sum(abs2, lb[k] - theta[k]), "below the lower bound (info 41)")
    end
    if any(theta .> ub)
        k = findall(theta .> ub)
        return reject(42, sum(abs2, theta[k] - ub[k]), "above the upper bound (info 42)")
    end
    Q, Hm = covariances(est, theta)
    if !isempty(est.groups.cx) && !_ispd(Q[est.Q_check, est.Q_check])
        return reject(43, _ispd_penalty(Q[est.Q_check, est.Q_check]), "covariance matrix of the structural shocks is not positive definite (info 43)")
    end
    if !isempty(est.groups.cn) && !_ispd(Hm[est.H_check, est.H_check])
        return reject(44, _ispd_penalty(Hm[est.H_check, est.H_check]), "covariance matrix of the measurement errors is not positive definite (info 44)")
    end
    local rf
    try
        rf = _nl_reduced_form(est, theta, Q, Hm)
    catch e
        e isa _NLReject || rethrow()
        return reject(e.info, e.penalty, e.msg)
    end
    rf_out === nothing || (rf_out[] = rf)
    Y = permutedims(est.data)
    start = Int(o["presample"]) + 1
    filter = NONLINEAR_FILTERS[nl.po.filter_algorithm]
    stream = _nl_stream(est)
    LIK, lik, cplx = try
        if nl.po.filter_algorithm == "cpf"
            filter(rf, Y, start, nl.po, stream; model_pruning=nl.model_pruning)
        else
            filter(rf, Y, start, nl.po, stream)
        end
    catch e
        (e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException || e isa LinearAlgebra.LAPACKException) &&
            throw(NonlinearFilterError("$(nl.po.filter_algorithm): $(sprint(showerror, e))"))
        rethrow()
    end
    cplx && return reject(46, 0.1, "the likelihood is complex (info 46)")
    isnan(LIK) && return reject(45, 0.1, "the likelihood is NaN (info 45)")
    isinf(LIK) && return reject(50, 0.1, "the likelihood is Inf (info 50)")
    lnprior = logpdf(est.prior, theta)
    if isinf(lnprior) || isnan(lnprior)
        info = isnan(lnprior) ? 47 : 40
        est.last_info = "the prior density is zero, infinite or NaN (info $info)"
        est.last_info_code = info
        return Inf, info, 0.1, -LIK, lnprior, Float64[]
    end
    contrib = with_lik ? lik[Int(o["presample"])+1:end] : Float64[]
    LIK - lnprior, 0, 0.0, -LIK, lnprior, contrib
end

"""
    nonlinear_filter_likelihood(est, theta) -> (LIK, lik, rf)

The filter alone at `theta` (no prior, no bounds): minus the log likelihood from period presample+1, the log density of
every period and the reduced form used. Throws `EstimationError` when the point is rejected.
"""
function nonlinear_filter_likelihood(est::Estimation, theta::AbstractVector)
    Int(est.options["order"]) == 2 || throw(EstimationError("not an order-2 estimation"))
    nl::NLSetup = est.options["nl"]
    theta = Float64.(theta)
    Q, Hm = covariances(est, theta)
    rf = try
        _nl_reduced_form(est, theta, Q, Hm)
    catch e
        e isa _NLReject && throw(EstimationError("the model cannot be solved at theta: $(e.msg)"))
        rethrow()
    end
    filter = NONLINEAR_FILTERS[nl.po.filter_algorithm]
    kw = nl.po.filter_algorithm == "cpf" ? (; model_pruning=nl.model_pruning) : (;)
    LIK, lik, _ = filter(rf, permutedims(est.data), Int(est.options["presample"]) + 1, nl.po, _nl_stream(est); kw...)
    LIK, lik, rf
end

"The checks of `initial_estimation_checks.m` for a nonlinear filter."
function _initial_checks_nonlinear(est::Estimation)
    m = est.model
    # the measurement-error variances at the initial values (estimated or calibrated)
    _, Hm = covariances(est, est.x0)
    if !any(!iszero, Hm)
        throw(EstimationError("initial_estimation_checks:: particle filtering requires measurement error on the observables"))
    elseif count(>(0), diag(Hm)) < length(est.varobs)
        throw(EstimationError("initial_estimation_checks:: particle filtering requires as many measurement errors as observed variables"))
    elseif !_ispd(Hm)
        throw(EstimationError("initial_estimation_checks:: the measurement error matrix must be positive definite"))
    end
    fval, info, = _nonlinear_dsge_likelihood(est, est.x0; bounds=(fill(-Inf, est.n), fill(Inf, est.n)))
    if info == 45
        throw(EstimationError("The initial value of the likelihood is NaN"))
    elseif info == 46
        throw(EstimationError("The initial value of the likelihood is complex"))
    elseif info > 0
        if info in (3, 4, 5)
            ev = try
                first_order(m; params=model_params(est, est.x0), qz_criterium=est.options["qz_criterium"]).eigenvalues
            catch e
                e isa BlanchardKahnError ? e.eigenvalues : ComplexF64[]
            end
            any(abs.(1 .- abs.(ev)) .< abs(est.options["qz_criterium"] - 1)) &&
                throw(EstimationError("Your model has at least one unit root and you are using a nonlinear filter. Please set nonlinear_filter_initialization=3."))
        end
    end
    nothing
end
