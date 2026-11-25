# Posterior distributions of theoretical moments (Dynare's moments_varendo: compute_moments_varendo.m, prior_posterior_statistics,
# dsge_simulated_theoretical_*, covariance_mc_analysis, correlation_mc_analysis, variance_decomposition_mc_analysis,
# conditional_variance_decomposition_mc_analysis, check_list_of_variables.m of Dynare 7.2) and of smoothed / filtered series
# (prior_posterior_statistics_core.m, pm3.m). Adapted from Dynare (GPL-3.0-or-later, (C) the Dynare Team); see NOTICE.
# Validated against Dynare by replaying its sub-draws (test/test_posterior_moments.jl).

# ====================================================================== statistics of arrays of draws
"""
Distribution of one kind of moment over draws: arrays shaped like the moment (`S...`), `deciles` (`S..., 9`) and `density`
(`S..., gridpoints, 2`) or `nothing`.
"""
struct MomentFamily
    mean::Array{Float64}
    median::Array{Float64}
    variance::Array{Float64}
    hpd_inf::Array{Float64}
    hpd_sup::Array{Float64}
    deciles::Array{Float64}
    density::Union{Nothing,Array{Float64}}
end

"`posterior_moments.m` of every series of `x` (draws first, any trailing shape `S`)."
function moment_family(x::AbstractArray, mh_conf_sig::Real; density::Bool=true, gridpoints::Int=512)
    nd = size(x, 1)
    S = size(x)[2:end]
    st = series_stats(reshape(Float64.(x), nd, :), mh_conf_sig; density=density, gridpoints=gridpoints)
    rs(v) = reshape(v, S...)
    dec = reshape(permutedims(st.deciles), S..., 9)
    dens = st.density === nothing ? nothing : reshape(permutedims(st.density, (3, 1, 2)), S..., gridpoints, 2)
    MomentFamily(rs(st.mean), rs(st.median), rs(st.var), rs(st.hpd_inf), rs(st.hpd_sup), dec, dens)
end

"Fill the lower triangle (first two dimensions) from the upper one: Dynare stores covariances for the variable listed first."
function _symmetrise(fam::MomentFamily)
    function sym(a)
        a === nothing && return nothing
        a = copy(a)
        n = size(a, 1)
        rest = ntuple(_ -> Colon(), ndims(a) - 2)
        for i in 1:n, j in i+1:n
            a[j, i, rest...] = a[i, j, rest...]
        end
        a
    end
    MomentFamily(sym(fam.mean), sym(fam.median), sym(fam.variance), sym(fam.hpd_inf), sym(fam.hpd_sup), sym(fam.deciles), sym(fam.density))
end

# ====================================================================== variable list and options
"""
    moment_variable_list(est; var_list=nothing, consider=nothing) -> Vector{String}

Variables whose moments are computed (`check_list_of_variables.m`): `var_list` (default: the list after the `estimation` command),
made unique; without a list the `consider_all_endogenous` / `consider_all_endogenous_and_auxiliary` / `consider_only_observed`
option; with neither Dynare asks, and `nointeractive` (as here) selects all original endogenous variables. A list together with
a `consider_*` option is an error, as in Dynare.
"""
function moment_variable_list(est::Estimation; var_list=nothing, consider=nothing)
    m = est.model
    stmt = _estimation_var_list_est(est)       # the list after the estimation command in the .mod file
    given = var_list === nothing ? stmt : String.(collect(var_list))
    flag = consider
    if flag === nothing
        for (k, v) in (("consider_all_endogenous", :all), ("consider_all_endogenous_and_auxiliary", :aux), ("consider_only_observed", :obs))
            _est_flag(get(est.options, k, false)) && (flag = v)
        end
    end
    # Dynare's rule concerns the .mod file's list; a list passed here explicitly takes precedence over the consider_* options
    var_list === nothing && !isempty(stmt) && flag !== nothing &&
        throw(EstimationError("You cannot use the consider_all_endogenous, consider_all_endogenous_and_auxiliary or consider_only_observed options when listing variables after the estimation command"))
    var_list === nothing || (flag = nothing)
    if isempty(given)
        given = flag === nothing || flag == :all ? orig_endo_names(m) :
                flag == :aux ? copy(m.endo_names) :
                flag == :obs ? copy(est.varobs) : throw(EstimationError("unknown endogenous variable selection $flag"))
    end
    unknown = [v for v in given if !haskey(m.ie, v)]
    isempty(unknown) || throw(EstimationError("The following symbols are not endogenous variables: " * join(unknown, " ")))
    out = String[]
    for v in given
        v in out && (@warn "$v is specified twice in the variable list and is considered only once"; continue)
        push!(out, v)
    end
    out
end

_eopt(est::Estimation, name, value, default) = value !== nothing ? value : (v = get(est.options, name, nothing); v === nothing ? default : v)

function _moment_options(est::Estimation, ar::Int)
    m = est.model
    o = _mom_options(m, Dict{String,Any}("ar" => ar, "qz_criterium" => est.options["qz_criterium"],
                                         "lyapunov_complex_threshold" => est.options["lyapunov_complex_threshold"]))
    (o["hp_filter"] != 0 || o["one_sided_hp_filter"] != 0 || o["bandpass"]) &&
        throw(EstimationError("a stoch_simul statement of the .mod file sets a filter (hp_filter, one_sided_hp_filter or bandpass_filter): Dynare would carry it into the posterior moments depending on the order of the statements; remove it from the file"))
    o["order"] = 1
    o
end

# ====================================================================== moments at one parameter vector
"""
    theoretical_moments_at(est, theta; var_list, ar, steps, nodecomposition) -> NamedTuple

Moments entering the posterior distributions at one parameter vector: `variance` (n x n), `autocorr` (n x n x ar), `decomposition`
(n x nexo, fractions, or `nothing`), `decomposition_me` (observed variables of the list x (nexo + 1)), `conditional` (n x nexo x steps)
and `conditional_me`. Unit-root variables give NaN.
"""
function theoretical_moments_at(est::Estimation, theta::AbstractVector; var_list=nothing, ar=nothing, steps=nothing, nodecomposition::Bool=false)
    vl = moment_variable_list(est; var_list=var_list)
    ar = Int(_eopt(est, "ar", ar, 5))
    st = _eopt(est, "conditional_variance_decomposition", steps, Int[])
    st = st isa Real ? [Int(st)] : Int.(collect(st))
    ivar = Int[est.model.ie[v] for v in vl]
    _moments_one(est, Float64.(theta), ivar, _moment_options(est, ar), ar, st, nodecomposition)
end

function _moments_one(est::Estimation, theta::Vector{Float64}, ivar::Vector{Int}, o, ar::Int, steps::Vector{Int}, nodecomposition::Bool)
    m = est.model
    n, nexo = length(ivar), length(m.exo_names)
    sol = first_order(m; params=model_params(est, theta), qz_criterium=est.options["qz_criterium"], loglinear=est.options["loglinear"])
    Sigma, Hm = covariances(est, theta)
    Hd = Float64.(diag(Hm))
    R = Rules(m, as_dr(sol))
    vid = est.obs_decl
    oo = copy(o)
    oo["nodecomposition"] = nodecomposition || nexo < 2
    gamma, stat = _th_autocovariances(R, Sigma, ivar, oo)
    ac = fill(NaN, n, n, ar)
    for i in 1:ar
        ac[:, :, i] = gamma[i+1]
    end
    dec = dec_me = cvd = cvd_me = nothing
    if nexo > 1
        if !nodecomposition
            dec = gamma[ar+2] === nothing ? fill(NaN, n, nexo) : gamma[ar+2]
            obs_pos, isub, iobs = _intersect_stable(ivar, vid)
            if any(!=(0), Hd) && !isempty(obs_pos)
                v = diag(gamma[1])[isub]
                me = dec[isub, :] .* (v ./ (v .+ Hd[iobs]))
                dec_me = hcat(me, 1.0 .- sum(me; dims=2))
            end
        end
        if !isempty(steps)
            cvd, cvd_me = conditional_variance_decomposition(m, R, steps; var_list=ivar, Sigma_e=Sigma, H=Hd, varobs_id=vid)
        end
    end
    (variance=gamma[1], autocorr=ac, decomposition=dec, decomposition_me=dec_me, conditional=cvd, conditional_me=cvd_me)
end

# ====================================================================== posterior distribution of the moments
"""
Dynare's `oo_.PosteriorTheoreticalMoments.dsge` as arrays. Variables in `var_names` order, shocks in declaration order.
`covariance` and `contemporaneous_correlation` are (n, n) with the upper triangle as Dynare stores it, mirrored; `correlation` is
(n, n, ar) with entry (a, b, k) the distribution of corr(y_a,t, y_b,t-k); `variance_decomposition` (n, nexo) in fractions and
`variance_decomposition_me` (observed variables of the list, nexo + 1; the last column is the measurement error);
`conditional_variance_decomposition` (n, nexo, steps) and `..._me` (observed, nexo + 1, steps). A family is `nothing` when Dynare
does not compute it. `draws` (with `keep_draws`) has the per-draw moments.
"""
Base.@kwdef struct PosteriorTheoreticalMoments
    var_names::Vector{String}
    exo_names::Vector{String}
    me_var_names::Vector{String}
    steps::Vector{Int}
    ar::Int
    mh_conf_sig::Float64
    n_draws::Int
    covariance::Union{Nothing,MomentFamily} = nothing
    contemporaneous_correlation::Union{Nothing,MomentFamily} = nothing
    correlation::Union{Nothing,MomentFamily} = nothing
    variance_decomposition::Union{Nothing,MomentFamily} = nothing
    variance_decomposition_me::Union{Nothing,MomentFamily} = nothing
    conditional_variance_decomposition::Union{Nothing,MomentFamily} = nothing
    conditional_variance_decomposition_me::Union{Nothing,MomentFamily} = nothing
    kind::String = "Posterior"
    params::Matrix{Float64} = zeros(0, 0)
    draws::Union{Nothing,NamedTuple} = nothing
end
Base.show(io::IO, r::PosteriorTheoreticalMoments) = print(io, "PosteriorTheoreticalMoments(", r.n_draws, " draws, ", length(r.var_names), " variables, ar ", r.ar, ")")

function _theta_draws(est::Estimation, mc, params, sub_draws, drop, rng, what)
    sub_draws !== nothing && Int(sub_draws) < 1 && throw(EstimationError("$what: sub_draws must be a positive integer, got $sub_draws"))
    if params === nothing
        mc === nothing && throw(ArgumentError("$what needs the chains (mc) or parameter draws (params)"))
        P, _ = posterior_draws(est, mc; sub_draws=sub_draws, drop=drop === nothing ? mc.drop : drop, rng=rng === nothing ? dynare_rng(est.model) : rng)
        return Matrix{Float64}(P)
    end
    Matrix{Float64}(params isa AbstractVector ? reshape(params, 1, :) : params)
end

"""
    posterior_theoretical_moments(est, mc=nothing; params=nothing, var_list, sub_draws, ar, steps, nodecomposition,
        contemporaneous_correlation, mh_conf_sig, density=true, drop, rng, keep_draws=false, consider) -> PosteriorTheoreticalMoments

`moments_varendo`: the model is solved at `sub_draws` parameter vectors drawn from the kept draws of `mc` (or at the rows of
`params`), the first-order theoretical moments of the variables of the list computed at each, and the distribution of every moment
over the draws summarised as `posterior_moments.m` does. Defaults come from the `estimation` statement (`ar` 5,
`conditional_variance_decomposition`, `nodecomposition`, `contemporaneous_correlation`, `mh_conf_sig`, `sub_draws`, the variable list
or `consider_*`). Decompositions need more than one shock (Dynare skips them otherwise) and are given for stationary variables.
"""
function posterior_theoretical_moments(est::Estimation, mc::Union{Nothing,MCMCResult}=nothing; params=nothing, var_list=nothing,
                                       sub_draws=nothing, ar=nothing, steps=nothing, nodecomposition=nothing,
                                       contemporaneous_correlation=nothing, mh_conf_sig=nothing, density=nothing, drop=nothing,
                                       rng=nothing, keep_draws::Bool=false, consider=nothing, kind::String="Posterior")
    m = est.model
    est.options["order"] == 1 || throw(EstimationError("posterior theoretical moments: only first-order models are supported"))
    vl = moment_variable_list(est; var_list=var_list, consider=consider)
    ar = Int(_eopt(est, "ar", ar, 5))
    st = _eopt(est, "conditional_variance_decomposition", steps, Int[])
    st = st isa Real ? [Int(st)] : Int.(collect(st))
    any(<=(0), st) && throw(EstimationError("Conditional variance decomposition: All periods must be strictly positive"))
    nodec = _est_flag(_eopt(est, "nodecomposition", nodecomposition, false))
    ccorr = _est_flag(_eopt(est, "contemporaneous_correlation", contemporaneous_correlation, false))
    conf = Float64(_eopt(est, "mh_conf_sig", mh_conf_sig, 0.9))
    dens_on, gp = _density_options(est)
    density === nothing || (dens_on = density)
    theta = _theta_draws(est, mc, params, sub_draws, drop, rng, "posterior_theoretical_moments")
    nd = size(theta, 1)
    _iround((1 - conf) * nd) < 2 && @warn "posterior_theoretical_moments: mh_conf_sig times the number of draws is too small to generate HPD intervals"
    o = _moment_options(est, ar)
    ivar = Int[m.ie[v] for v in vl]
    n, nexo = length(ivar), length(m.exo_names)
    obs_pos, isub, iobs = _intersect_stable(ivar, est.obs_decl)
    # measurement errors present: Dynare looks at M_.H before the loop; here the first draw's
    _, H1 = covariances(est, theta[1, :])
    me_present = any(!=(0), diag(H1)) && !isempty(obs_pos)
    me_names = me_present ? String[m.endo_names[i] for i in obs_pos] : String[]
    V = zeros(nd, n, n)
    AC = zeros(nd, n, n, ar)
    VD = (nexo > 1 && !nodec) ? zeros(nd, n, nexo) : nothing
    VDME = (VD !== nothing && me_present) ? zeros(nd, length(obs_pos), nexo + 1) : nothing
    CV = (nexo > 1 && !isempty(st)) ? zeros(nd, n, nexo, length(st)) : nothing
    CVME = (CV !== nothing && me_present) ? zeros(nd, length(obs_pos), nexo + 1, length(st)) : nothing
    for k in 1:nd
        r = try
            _moments_one(est, theta[k, :], ivar, o, ar, st, nodec)
        catch e
            e isa InterruptException && rethrow()
            throw(EstimationError("posterior moments: unable to compute the moments at draw $k ($(sprint(showerror, e)))"))
        end
        V[k, :, :] = r.variance
        AC[k, :, :, :] = r.autocorr
        if VD !== nothing
            VD[k, :, :] = r.decomposition
            VDME === nothing || (VDME[k, :, :] = r.decomposition_me)
        end
        if CV !== nothing
            CV[k, :, :, :] = permutedims(r.conditional, (1, 3, 2))
            CVME === nothing || (CVME[k, :, :, :] = permutedims(r.conditional_me, (1, 3, 2)))
        end
    end
    fam(x) = moment_family(x, conf; density=dens_on, gridpoints=gp)
    cov = _symmetrise(fam(V))
    cc = nothing
    if ccorr
        sd = [sqrt(V[k, i, i]) for k in 1:nd, i in 1:n]
        c = V ./ reshape(sd, nd, n, 1) ./ reshape(sd, nd, 1, n)
        c[V .== 0] .= 0.0                     # covariance_mc_analysis: 0 correlations where the covariance is 0
        cc = _symmetrise(fam(c))
    end
    PosteriorTheoreticalMoments(var_names=vl, exo_names=copy(m.exo_names), me_var_names=me_names, steps=st, ar=ar, mh_conf_sig=conf,
        n_draws=nd, covariance=cov, contemporaneous_correlation=cc, correlation=ar > 0 ? fam(AC) : nothing,
        variance_decomposition=VD === nothing ? nothing : fam(VD), variance_decomposition_me=VDME === nothing ? nothing : fam(VDME),
        conditional_variance_decomposition=CV === nothing ? nothing : fam(CV),
        conditional_variance_decomposition_me=CVME === nothing ? nothing : fam(CVME), kind=kind, params=theta,
        draws=keep_draws ? (variance=V, autocorr=AC, decomposition=VD, decomposition_me=VDME, conditional=CV, conditional_me=CVME) : nothing)
end

# ---------------------------------------------------------------------- oo_ layout
const _MOMENT_STATS = (("Mean", :mean), ("Median", :median), ("Variance", :variance), ("HPDinf", :hpd_inf), ("HPDsup", :hpd_sup),
                       ("deciles", :deciles), ("density", :density))

function _fill_family!(D::MStruct, fam::MomentFamily, rows, cols; pair_only::Bool=false, vector::Bool=false, steps::Bool=false)
    for (key, f) in _MOMENT_STATS
        a = getfield(fam, f)
        a === nothing && continue
        S = MStruct()
        for (i, r) in enumerate(rows)
            T = get(S, r, nothing)
            T isa MStruct || (T = MStruct(); S[r] = T)
            for (j, c) in enumerate(cols)
                pair_only && j < i && continue
                T[c] = if !(vector || steps)
                    key == "deciles" ? _col(a[i, j, :]) : key == "density" ? Matrix{Float64}(a[i, j, :, :]) : Float64(a[i, j])
                elseif vector       # (n, n, ar): ar x 1 columns, deciles / density in an ar x 1 cell
                    key == "deciles" ? Cell(Any[_col(a[i, j, k, :]) for k in axes(a, 3)], (size(a, 3), 1)) :
                    key == "density" ? Cell(Any[Matrix{Float64}(a[i, j, k, :, :]) for k in axes(a, 3)], (size(a, 3), 1)) :
                    _col(a[i, j, :])
                else                # (n, nexo, steps): row over the steps, deciles 9 x steps, density grid x 2 x steps
                    key == "deciles" ? Matrix{Float64}(permutedims(a[i, j, :, :])) :
                    key == "density" ? Array{Float64,3}(permutedims(a[i, j, :, :, :], (2, 3, 1))) :
                    reshape(Float64.(a[i, j, :]), 1, :)
                end
            end
        end
        D[key] = S
    end
    D
end

"`oo_.PosteriorTheoreticalMoments.dsge` (`PriorTheoreticalMoments` for `kind=\"Prior\"`), field by field as Dynare writes it."
function store_moments!(oo::MStruct, r::PosteriorTheoreticalMoments)
    top = r.kind * "TheoreticalMoments"
    P = get(oo, top, nothing)
    P isa MStruct || (P = MStruct(); oo[top] = P)
    D = MStruct()
    v, x = r.var_names, r.exo_names
    r.covariance === nothing || (D["covariance"] = _fill_family!(MStruct(), r.covariance, v, v; pair_only=true))
    r.contemporaneous_correlation === nothing ||
        (D["contemporeaneous_correlation"] = _fill_family!(MStruct(), r.contemporaneous_correlation, v, v; pair_only=true))
    r.correlation === nothing || (D["correlation"] = _fill_family!(MStruct(), r.correlation, v, v; vector=true))
    r.variance_decomposition === nothing || (D["VarianceDecomposition"] = _fill_family!(MStruct(), r.variance_decomposition, v, x))
    r.variance_decomposition_me === nothing ||
        (D["VarianceDecompositionME"] = _fill_family!(MStruct(), r.variance_decomposition_me, r.me_var_names, vcat(x, "ME")))
    for (name, fam, rows, cols) in (("ConditionalVarianceDecomposition", r.conditional_variance_decomposition, v, x),
                                    ("ConditionalVarianceDecompositionME", r.conditional_variance_decomposition_me, r.me_var_names, vcat(x, "ME")))
        fam === nothing && continue
        S = _fill_family!(MStruct(), fam, rows, cols; steps=true)
        S["Steps"] = reshape(Float64.(r.steps), 1, :)
        D[name] = S
    end
    P["dsge"] = D
    oo
end

# ====================================================================== smoothed, updated and filtered series
"""
Distribution over draws of a set of time series (one call of `pm3.m`): `names` and arrays in Dynare's layout: `mean`, `median`,
`var`, `hpd_inf`, `hpd_sup` (periods x series), `deciles` (9 x periods x series), `density` (gridpoints x 2 x periods x series) or `nothing`.
"""
struct SeriesDistribution
    names::Vector{String}
    mean::Matrix{Float64}
    median::Matrix{Float64}
    var::Matrix{Float64}
    hpd_inf::Matrix{Float64}
    hpd_sup::Matrix{Float64}
    deciles::Array{Float64,3}
    density::Union{Nothing,Array{Float64,4}}
end

"`x` is (draws, periods, series)."
function series_distribution(names, x::AbstractArray{Float64,3}, mh_conf_sig::Real; density::Bool=true, gridpoints::Int=512)
    f = moment_family(x, mh_conf_sig; density=density, gridpoints=gridpoints)
    SeriesDistribution(String.(names), f.mean, f.median, f.variance, f.hpd_inf, f.hpd_sup, permutedims(f.deciles, (3, 1, 2)),
                       f.density === nothing ? nothing : permutedims(f.density, (3, 4, 1, 2)))
end

"""
Distributions of the smoothed and filtered objects of `prior_posterior_statistics` with `smoother` / `filtered_vars` /
`filter_step_ahead`. `smoothed_variables`, `updated_variables`, `constant`, `trend`, `filtered_variables` are over the variables of the
list, `smoothed_shocks` over all shocks, `smoothed_measurement_errors` over the observed variables (`SE_EOBS_<name>`; only when a
measurement-error standard deviation is estimated), `trend_coeff` and `init_state` are one-period series. `filtered_k_step_ahead[k]` is
the distribution of the k-step-ahead forecast; `filtered_k_step_ahead_mean` is Dynare's `FilteredVariablesKStepAhead` (steps x variables x
(T + max step)) and `..._variances` its twin (Dynare 7.2 stores the means there too).
"""
Base.@kwdef struct PosteriorSmootherStatistics
    var_names::Vector{String}
    mh_conf_sig::Float64
    n_draws::Int
    smoothed_variables::Union{Nothing,SeriesDistribution} = nothing
    updated_variables::Union{Nothing,SeriesDistribution} = nothing
    smoothed_shocks::Union{Nothing,SeriesDistribution} = nothing
    smoothed_measurement_errors::Union{Nothing,SeriesDistribution} = nothing
    constant::Union{Nothing,SeriesDistribution} = nothing
    trend::Union{Nothing,SeriesDistribution} = nothing
    trend_coeff::Union{Nothing,SeriesDistribution} = nothing
    init_state::Union{Nothing,SeriesDistribution} = nothing
    filtered_variables::Union{Nothing,SeriesDistribution} = nothing
    filtered_k_step_ahead::Dict{Int,SeriesDistribution} = Dict{Int,SeriesDistribution}()
    filtered_k_step_ahead_mean::Union{Nothing,Array{Float64,3}} = nothing
    filtered_k_step_ahead_variances::Union{Nothing,Array{Float64,3}} = nothing
    filter_step_ahead::Vector{Int} = Int[]
    params::Matrix{Float64} = zeros(0, 0)
    draws::Union{Nothing,Dict{String,Any}} = nothing
    loglinear::Bool = false
end
Base.show(io::IO, r::PosteriorSmootherStatistics) = print(io, "PosteriorSmootherStatistics(", r.n_draws, " draws, ", length(r.var_names), " variables)")

"What `prior_posterior_statistics_core.m` stores for one draw (all variables in declaration order)."
function _draw_series(est::Estimation, res::SmootherResult, fsa::Vector{Int})
    nm = res.endo_names
    n = length(nm)
    mf = res.obs_index
    T = size(res.smoothed, 1)
    mat(d) = reduce(hcat, [d[v] for v in nm])
    cst, sv, uv = mat(res.Constant), mat(res.SmoothedVariables), mat(res.UpdatedVariables)
    trend = fill(NaN, T, n)
    for i in mf
        trend[:, i] = res.Trend[nm[i]]
    end
    tcoef = zeros(n)
    tcoef[mf] = res.TrendCoeffs[mf]
    init = Float64[res.Init_State[v] for v in nm]
    level = init .- res.init_state                 # the constant added to deviations (steady state, in logs under loglinear)
    init[mf] = res.init_state[mf] .+ res.obs_constant .+ trend[1, mf]
    fa = nothing
    if !isempty(fsa)
        # Dynare adds the observed variables' constant and trend to every column of the k-step forecasts; the smoother adjusts k+1:hi only
        fa = copy(res.FilteredVariablesKStepAhead)
        mx = size(fa, 3) - T
        trend_ext = vcat(trend[:, mf], trend[end:end, mf] .+ (1:mx) .* reshape(tcoef[mf], 1, :))
        for (j, k) in enumerate(res.filter_step_ahead)
            hi = size(fa, 3) - (mx - k)
            for c in vcat(1:k, hi+1:size(fa, 3))
                fa[j, mf, c] = fa[j, mf, c] .- level[mf] .+ res.obs_constant .+ trend_ext[c, :]
            end
        end
    end
    # prior_posterior_statistics_core.m adds log(SteadyState) to every variable under loglinear, also to the auxiliary variables of lagged
    # and led exogenous variables, whose steady state is 0 and which store_smoother_results.m (the frequentist smoother) leaves in levels:
    # their draws are -Inf there. A Dynare 7.2 defect: corrected by default (the smoothed shock, as the frequentist smoother), reproduced
    # under replicate_dynare_bugs=true.
    if _est_flag(get(est.options, "loglinear", false)) && _est_flag(get(res.options, "replicate_dynare_bugs", false))
        for i in findall(.!log_mask(est.model))
            cst[:, i] .= -Inf; sv[:, i] .= -Inf; uv[:, i] .= -Inf; init[i] = -Inf
        end
    end
    (smooth=sv, update=uv, innov=res.shocks, error=isempty(res.measurement_errors) ? nothing : res.measurement_errors,
     constant=cst, trend=trend, trend_coeff=tcoef, init_state=init, filter_step_ahead=fa)
end

"""
    posterior_smoother_statistics(est, mc=nothing; params=nothing, var_list, sub_draws, smoother, filtered_vars, filter_step_ahead,
        mh_conf_sig, density, drop, rng, keep_draws=false, smoother_kw...) -> PosteriorSmootherStatistics

Posterior distributions of smoothed, updated and filtered variables, shocks and measurement errors (the smoother branch of
`prior_posterior_statistics`, summarised by `pm3`): the Kalman smoother runs at every draw. `smoother`, `filtered_vars` and
`filter_step_ahead` default to the statement's options (`filtered_vars` alone means `filter_step_ahead = 1`, and
`filter_step_ahead` alone switches `filtered_vars` on). `filter_covariance` and `smoothed_state_uncertainty` are refused.
"""
function posterior_smoother_statistics(est::Estimation, mc::Union{Nothing,MCMCResult}=nothing; params=nothing, var_list=nothing,
                                       sub_draws=nothing, smoother=nothing, filtered_vars=nothing, filter_step_ahead=nothing,
                                       mh_conf_sig=nothing, density=nothing, drop=nothing, rng=nothing, keep_draws::Bool=false,
                                       smoother_kw...)
    m = est.model
    smo = _est_flag(_eopt(est, "smoother", smoother, false))
    fsa0 = _eopt(est, "filter_step_ahead", filter_step_ahead, Int[])
    fsa = fsa0 isa Real ? [Int(fsa0)] : Int.(collect(fsa0))
    fv = _est_flag(_eopt(est, "filtered_vars", filtered_vars, false))
    fv && isempty(fsa) && (fsa = [1])
    !isempty(fsa) && (fv = true)
    (smo || fv) || throw(ArgumentError("neither smoother nor filtered_vars / filter_step_ahead requested"))
    for key in ("filter_covariance", "smoothed_state_uncertainty")
        (_est_flag(get(est.options, key, false)) || _est_flag(get(smoother_kw, Symbol(key), false))) &&
            throw(EstimationError("posterior distributions of $key are not implemented"))
    end
    conf = Float64(_eopt(est, "mh_conf_sig", mh_conf_sig, 0.9))
    dens_on, gp = _density_options(est)
    density === nothing || (dens_on = density)
    theta = _theta_draws(est, mc, params, sub_draws, drop, rng, "posterior_smoother_statistics")
    nd = size(theta, 1)
    names = moment_variable_list(est; var_list=var_list)       # check_list_of_variables.m: also consider_all_endogenous(_and_auxiliary)
    sel = Int[m.ie[v] for v in names]
    if fv && fsa[1] != 1
        @warn "oo_.FilteredVariables of Dynare 7.2 is the forecast $(fsa[1]) period(s) ahead shifted by one period (pm3.m reads the first element of filter_step_ahead as if it were 1); reproduced here"
    end
    srt = sort(unique(fsa))
    stores = []
    kw = Dict{Symbol,Any}(smoother_kw)
    isempty(fsa) || (kw[:filter_step_ahead] = srt)
    for k in 1:nd
        res = try
            smooth(m; estimation=est, theta=theta[k, :], kw...)
        catch e
            e isa InterruptException && rethrow()
            throw(EstimationError("posterior smoother: unable to smooth at draw $k ($(sprint(showerror, e)))"))
        end
        push!(stores, _draw_series(est, res, srt))
    end
    T = size(stores[1].smooth, 1)
    stack(f) = permutedims(cat((getfield(s, f) for s in stores)...; dims=3), (3, 1, 2))     # (draws, T, n) from (T, n) pieces
    stack1(f) = reshape(permutedims(reduce(hcat, [getfield(s, f) for s in stores])), nd, 1, :)  # (draws, 1, n) from n-vectors
    SD(nms, x) = series_distribution(nms, x, conf; density=dens_on, gridpoints=gp)
    kw2 = Dict{Symbol,Any}(:var_names => names, :mh_conf_sig => conf, :n_draws => nd, :params => theta, :filter_step_ahead => fsa)
    if smo
        kw2[:smoothed_variables] = SD(names, stack(:smooth)[:, :, sel])
        kw2[:smoothed_shocks] = SD(m.exo_names, stack(:innov))
        kw2[:trend_coeff] = SD(names, stack1(:trend_coeff)[:, :, sel])
        kw2[:init_state] = SD(names, stack1(:init_state)[:, :, sel])
        kw2[:constant] = SD(names, stack(:constant)[:, :, sel])
        kw2[:trend] = SD(names, stack(:trend)[:, :, sel])
        kw2[:updated_variables] = SD(names, stack(:update)[:, :, sel])
        if !isempty(est.groups.vn)               # estimated measurement-error standard deviations
            kw2[:smoothed_measurement_errors] = SD(["SE_EOBS_$v" for v in est.varobs], stack(:error))
        end
    end
    fa = nothing
    if fv
        # (draws, steps in the user's order, n, T + max)
        fa = permutedims(cat((s.filter_step_ahead for s in stores)...; dims=4), (4, 1, 2, 3))
        fa = fa[:, [findfirst(==(k), srt) for k in fsa], :, :]
        mx = maximum(fsa)
        kw2[:filtered_variables] = SD(names, permutedims(fa[:, 1, sel, 2:1+T], (1, 3, 2)))
        means = zeros(length(fsa), length(names), T + mx)
        kd = Dict{Int,SeriesDistribution}()
        for (j, k) in enumerate(fsa)
            sd = SD(names, permutedims(fa[:, j, sel, k+1:T+k], (1, 3, 2)))
            kd[k] = sd
            means[j, :, k+1:T+k] = permutedims(sd.mean)
        end
        kw2[:filtered_k_step_ahead] = kd
        kw2[:filtered_k_step_ahead_mean] = means
        kw2[:filtered_k_step_ahead_variances] = copy(means)            # Dynare 7.2 stores the means here too
    end
    if keep_draws
        d = Dict{String,Any}(string(f) => (f in (:trend_coeff, :init_state) ? stack1(f) : stack(f)) for f in (:smooth, :update, :innov, :constant, :trend, :trend_coeff, :init_state))
        stores[1].error === nothing || (d["error"] = stack(:error))
        fa === nothing || (d["filter_step_ahead"] = fa)
        kw2[:draws] = d
    end
    kw2[:loglinear] = _est_flag(get(est.options, "loglinear", false))
    PosteriorSmootherStatistics(; kw2...)
end

function _store_series!(S::MStruct, sd::SeriesDistribution; vector_density::Bool=true)
    for f in ("Mean", "Median", "Var", "deciles", "HPDinf", "HPDsup", "density")
        f == "density" && sd.density === nothing && continue
        haskey(S, f) || (S[f] = MStruct())
    end
    for (i, nm) in enumerate(sd.names)
        S["Mean"][nm] = _col(sd.mean[:, i]); S["Median"][nm] = _col(sd.median[:, i]); S["Var"][nm] = _col(sd.var[:, i])
        S["deciles"][nm] = Matrix{Float64}(sd.deciles[:, :, i])
        S["HPDinf"][nm] = _col(sd.hpd_inf[:, i]); S["HPDsup"][nm] = _col(sd.hpd_sup[:, i])
        sd.density === nothing || (S["density"][nm] = Array{Float64,3}(sd.density[:, :, :, i]))
    end
    S
end

"`oo_.SmoothedVariables`, `UpdatedVariables`, `SmoothedShocks`, `SmoothedMeasurementErrors`, `Constant`, `FilteredVariables`, `Filtered_Variables_<k>_step_ahead`, `FilteredVariablesKStepAhead(Variances)` and `oo_.Smoother.{Trend, TrendCoeff, Init_State}`."
function store_smoother_statistics!(oo::MStruct, r::PosteriorSmootherStatistics)
    for (key, sd) in (("SmoothedVariables", r.smoothed_variables), ("UpdatedVariables", r.updated_variables),
                      ("SmoothedShocks", r.smoothed_shocks), ("SmoothedMeasurementErrors", r.smoothed_measurement_errors),
                      ("Constant", r.constant), ("FilteredVariables", r.filtered_variables))
        sd === nothing || (oo[key] = _store_series!(MStruct(), sd))
    end
    for k in sort(collect(keys(r.filtered_k_step_ahead)))
        oo["Filtered_Variables_$(k)_step_ahead"] = _store_series!(MStruct(), r.filtered_k_step_ahead[k])
    end
    if any(x -> x !== nothing, (r.trend, r.trend_coeff, r.init_state))
        Sm = get(oo, "Smoother", nothing)
        Sm isa MStruct || (Sm = MStruct(); oo["Smoother"] = Sm)
        Sm["loglinear"] = r.loglinear          # prior_posterior_statistics.m (read by smoother2histval)
        for (key, sd) in (("Trend", r.trend), ("TrendCoeff", r.trend_coeff), ("Init_State", r.init_state))
            sd === nothing || (Sm[key] = _store_series!(MStruct(), sd))
        end
    end
    if r.filtered_k_step_ahead_mean !== nothing
        oo["FilteredVariablesKStepAhead"] = r.filtered_k_step_ahead_mean
        oo["FilteredVariablesKStepAheadVariances"] = r.filtered_k_step_ahead_variances
    end
    oo
end
