# Dynare-style results of `estimation` and `calib_smoother`: oo_.posterior_mode, oo_.MarginalDensity, oo_.Smoother, ...
# written the way Dynare 7.2's dynare_estimation_1.m, display_estimation_results_table.m,
# GetPosteriorParametersStatistics.m, store_smoother_results.m and evaluate_smoother.m write them (layout map in
# docs/RESULTS.md). The numerical work is `Estimation`, `posterior_mode`, `sample_posterior` and `smooth`; this file
# only reads options_, calls them and fills M_, options_ and oo_.
#
#     estimation!(ws, var_list; options...)        # `estimation(options) var_list;` where options_ stands
#     calib_smoother!(ws, var_list; options...)    # `calib_smoother(options) var_list;`
#
# `dynare()` calls them where the statements stand (`_run_in_place!`).

const _SMOOTHER_OPTS = ("smoother", "filtered_vars", "filter_step_ahead", "filter_covariance", "smoothed_state_uncertainty")

"oo_.gui.ran_<command> = true (Dynare creates the six flags, all false, at the start of a run)."
function _gui!(oo::MStruct, flag::AbstractString)
    g = get(oo, "gui", nothing)
    if !(g isa MStruct)
        g = MStruct()
        for k in ("ran_estimation", "ran_stoch_simul", "ran_calib_smoother", "ran_perfect_foresight",
                  "ran_shock_decomposition", "ran_realtime_shock_decomposition")
            g[k] = false
        end
        oo.gui = g
    end
    g[flag] = true
    oo
end

# ---------------------------------------------------------------------- options_ -> keywords
_isnan_scalar(v) = v isa AbstractFloat && isnan(v)

"filter_step_ahead as options_ holds it (a column in Dynare; a row if a MATLAB line or a keyword wrote it): Int vector, empty if unset."
function _fsa(ws)
    v = _getpath(ws.options_, "filter_step_ahead", nothing)
    (v === nothing || (v isa AbstractMatrix && isempty(v))) && return Int[]
    v isa Real && return v == 0 ? Int[] : [Int(round(v))]
    sort(unique(Int.(round.(vec(Float64.(v))))))
end

# ---- the `data` command (Dynare's options_.dataset): file, first_obs, last_obs, nobs, dates
const _DATE_EXPR = r"^\s*(?:dates\(\s*'([^']+)'\s*\)|'?([0-9]+[A-Za-z][0-9]*|-?[0-9]+-[0-9]+-[0-9]+)'?)\s*$"

function _date_of(v)
    v isa AbstractVector && length(v) == 1 && (v = v[1])
    v isa AbstractString || return nothing
    mt = match(_DATE_EXPR, v)
    mt === nothing && return nothing
    string2date(String(something(mt.captures[1], mt.captures[2])))
end

"First date (freq, time) of a data file: INIT__ of a .m or .mat file, the first column of a .csv file; else `nothing`."
function _data_start_date(path::AbstractString)
    isfile(path) || return nothing
    ext = lowercase(splitext(path)[2])
    if ext == ".m"
        mt = match(r"^\s*INIT__\s*=\s*['\"]([^'\"]+)['\"]"m, read(path, String))
        return mt === nothing ? nothing : string2date(String(mt.captures[1]))
    elseif ext == ".mat"
        t = try MAT.matread(path) catch; return nothing end
        v = get(t, "INIT__", nothing)
        return v isa AbstractString ? string2date(v) : nothing
    elseif ext == ".csv"
        rows = filter(!isempty, strip.(readlines(path)))
        length(rows) > 1 || return nothing
        return string2date(String(strip(split(rows[2], ',')[1])))
    end
    nothing
end

"Rewrite the datafile, first_obs and nobs keywords as the `data` command says (observation numbers counted from 1)."
function _apply_dataset!(ws, kw)
    d = get(ws.state, :dataset, nothing)
    d === nothing && return
    for k in ("series", "xls_sheet", "xls_range")
        haskey(d, k) && throw(EstimationError("data($k=...) is not supported: use a .m, .mat or .csv file"))
    end
    haskey(d, "file") || throw(EstimationError("the data command needs a file"))
    haskey(kw, :datafile) && String(kw[:datafile]) != String(d["file"]) &&
        throw(EstimationError("You cannot simultaneously use the data command and the datafile option"))
    file = String(d["file"])
    kw[:datafile] = file
    p = isabspath(file) ? file : joinpath(ws.model.mod_dir, file)
    if !(lowercase(splitext(p)[2]) in (".m", ".mat", ".csv"))
        found = [p * e for e in (".m", ".mat", ".csv") if isfile(p * e)]
        length(found) == 1 && (p = found[1])
    end
    start = _data_start_date(p)
    if start === nothing && haskey(ws.state, :set_time)
        start = _date_of(String(ws.state[:set_time]))        # set_time: the first date of the sample
    end
    function index(v, what)
        dt = _date_of(v)
        if dt === nothing
            v isa AbstractVector && length(v) == 1 && (v = v[1])
            return Int(round(Float64(v isa AbstractString ? parse(Float64, v) : v)))
        end
        start === nothing && throw(EstimationError("data($what=...): the data file has no dates (INIT__ in a .m/.mat file, a first column of dates in a .csv file) and set_time was not used"))
        dt[1] == start[1] || throw(EstimationError("data($what=...): the date is at another frequency than the data"))
        Int(round(dt[2] - start[2])) + 1
    end
    first = haskey(d, "first_obs") ? index(d["first_obs"], "first_obs") : 1
    first < 1 && throw(EstimationError("makedataset: first_obs cannot be less than the first date in the dataset"))
    nobs = haskey(d, "nobs") ? index(d["nobs"], "nobs") : nothing
    if haskey(d, "last_obs")
        last = index(d["last_obs"], "last_obs")
        nobs !== nothing && last != first + nobs - 1 && throw(EstimationError("makedataset: Options last_obs, first_obs and nobs are not consistent!"))
        nobs = last - first + 1
    end
    kw[:first_obs] = first
    nobs === nothing ? delete!(kw, :nobs) : (kw[:nobs] = nobs)
    nothing
end

"""
Keywords of `Estimation` from options_: the options estimation reads (the ones it acts on or ignores on purpose, those of
the file's `estimation` statements, and `user`), in the form `_flat_options` gives them. The smoother options, which this
file acts on, and `frequentist_smoother` are left out.
"""
function _estimation_kwargs(ws, user)
    flat = _flat_options(ws.options_)
    stmt = Set{String}()
    for st in ws.model.statements
        get(st, "statementName", "") == "estimation" || continue
        o = get_or(st, "options", nothing)
        o isa AbstractDict && union!(stmt, String.(keys(o)))
    end
    allowed = union(_HONOURED_EST, _COSMETIC_EST, stmt, Set(String.(collect(user))))
    kw = Dict{Symbol,Any}()
    for (k, v) in flat
        (k in allowed && !(k in _SMOOTHER_OPTS) && k != "frequentist_smoother") || continue
        v isa AbstractMatrix && (v = size(v, 1) == 1 || size(v, 2) == 1 ? Any[x for x in vec(v)] : v)
        _isnan_scalar(v) && continue                          # first_obs = NaN is Dynare's "not set"
        kw[Symbol(k)] = v
    end
    _apply_dataset!(ws, kw)
    # options_.order is always read (Dynare's estimation resets it to 1; the table of defaults would hide a 2)
    kw[:order] = _opti(ws, "order", 1)
    (occ = _occbin_overrides(ws)) === nothing || (kw[:occbin] = occ)
    kw
end

"Keywords of `smooth` from options_: what the smoother reads that is not part of an Estimation (and everything for calib_smoother)."
function _smoother_kwargs(ws; full::Bool)
    kw = Dict{Symbol,Any}(:filtered_vars => _optb(ws, "filtered_vars", false),
                          :filter_covariance => _optb(ws, "filter_covariance", false),
                          :smoothed_state_uncertainty => _optb(ws, "smoothed_state_uncertainty", false),
                          :filter_step_ahead => _fsa(ws))
    for k in ("replicate_dynare_bugs", "noprint", "use_univariate_smoother_if_singularity_is_detected")
        v = _optf(ws, k, nothing)
        v === nothing || (kw[Symbol(k)] = v isa Real ? v != 0 : v)
    end
    if full
        for k in ("first_obs", "nobs", "prefilter", "presample", "lik_init", "kalman_algo", "kalman_tol", "diffuse_kalman_tol", "Harvey_scale_factor")
            v = _optf(ws, k, nothing)
            (v === nothing || _isnan_scalar(v)) && continue
            kw[Symbol(k)] = v
        end
        for k in ("loglinear", "logdata", "diffuse_filter")
            kw[Symbol(k)] = _optb(ws, k, false)
        end
        df = _optf(ws, "datafile", nothing)
        df === nothing || (kw[:datafile] = String(df))
    end
    (occ = _occbin_overrides(ws)) === nothing || (kw[:occbin] = occ)
    kw
end

# ---------------------------------------------------------------------- oo_ pieces
"The estimated parameters as Dynare names them in oo_: (group, field name, position in theta), group by group."
function _slots(est::Estimation)
    m, g = est.model, est.groups
    nvx, nvn, ncx, ncn = length(g.vx), length(g.vn), length(g.cx), length(g.cn)
    out = Tuple{String,String,Int}[]
    for (i, (_, k, _)) in enumerate(g.p)
        push!(out, ("parameters", m.param_names[k], nvx + nvn + ncx + ncn + i))
    end
    for (i, (_, k, _)) in enumerate(g.vx)
        push!(out, ("shocks_std", m.exo_names[k], i))
    end
    for (i, (_, k, _)) in enumerate(g.vn)
        push!(out, ("measurement_errors_std", est.varobs[k], nvx + i))
    end
    for (i, (_, (k1, k2), _)) in enumerate(g.cx)
        push!(out, ("shocks_corr", string(m.exo_names[k1], "_", m.exo_names[k2]), nvx + nvn + i))
    end
    for (i, (_, (k1, k2), _)) in enumerate(g.cn)
        push!(out, ("measurement_errors_corr", string(est.varobs[k1], "_", est.varobs[k2]), nvx + nvn + ncx + i))
    end
    out
end

"oo_.<field>.<group>.<name> = vals[i] for every estimated parameter (display_estimation_results_table.m, fill_mh_mode.m)."
function _fill_by_name!(oo::MStruct, field::AbstractString, est::Estimation, vals; f=identity)
    S = get(oo, field, nothing)
    S isa MStruct || (S = MStruct(); oo[field] = S)
    for (grp, nm, i) in _slots(est)
        G = get(S, grp, nothing)
        G isa MStruct || (G = MStruct(); S[grp] = G)
        G[nm] = f(vals[i])
    end
    S
end

"The estimated-parameter vector Dynare's get_posterior_parameters.m reads back from oo_.<prefix><type> (type: mode, mean, median)."
function _xparam_from_oo(ws, est::Estimation, prefix::AbstractString, type::AbstractString)
    S = get(ws.oo_, prefix * type, nothing)
    S isa MStruct || throw(EstimationError("oo_.$(prefix * type) does not exist: run estimation first (get_posterior_parameters)"))
    x = zeros(est.n)
    for (grp, nm, i) in _slots(est)
        G = get(S, grp, nothing)
        (G isa MStruct && haskey(G, nm)) || throw(EstimationError("oo_.$(prefix * type).$grp.$nm does not exist"))
        x[i] = Float64(G[nm])
    end
    x
end

"oo_.prior (dynare_estimation_init.m): prior mean, mode, variance and hyperparameters of the estimated parameters (empty without estimated_params)."
function _store_prior!(oo::MStruct, est::Union{Nothing,Estimation})
    P = MStruct()
    H = MStruct()
    if est === nothing
        P.mean = zeros(0, 0); P.mode = zeros(0, 0); P.variance = zeros(0, 0)
        H.first = zeros(0, 0); H.second = zeros(0, 0)
    else
        pr = est.prior
        n = est.n
        if is_bayesian(pr)
            P.mean = _col(pr.p1); P.mode = _col(prior_mode(pr)); P.variance = Matrix{Float64}(Diagonal(pr.p2 .^ 2))
            H.first = _col(pr.p6); H.second = _col(pr.p7)
        else                                # maximum likelihood: bayestopt_.p1, p2, p6, p7 are NaN, p5 is 0
            P.mean = fill(NaN, n, 1); P.mode = zeros(n, 1); P.variance = Matrix{Float64}(Diagonal(fill(NaN, n)))
            H.first = fill(NaN, n, 1); H.second = fill(NaN, n, 1)
        end
    end
    P.hyperparameters = H
    oo.prior = P
end

"""
oo_.dr after estimation / smoothing: the fields Dynare's dynare_estimation_init.m and the resolution inside DsgeSmoother leave
(order_var, inv_order_var (a row), restrict_var_list, restrict_columns; with a solution also ys, ghx, ghu, state_var, sdim, edim;
eigval, gx and Gy are not written: Dynare's come from the QZ solution in a different ordering). Fields an earlier command stored stay.
"""
function _store_dr_estimation!(oo::MStruct, m::Model, sol=nothing)
    D = get(oo, "dr", nothing)
    D isa MStruct || (D = MStruct())
    order_var, _, cnt = dr_ordering(m.lead_lag_incidence)
    nspred = cnt.npred + cnt.nboth
    D.order_var = _col(order_var)
    D.inv_order_var = _row(invperm(order_var))
    obs_dr = invperm(order_var)[Int[m.ie[v] for v in m.varobs]]
    k2 = sort(union(obs_dr, collect(cnt.nstatic+1:cnt.nstatic+nspred)))
    D.restrict_var_list = _col(k2)
    # dynare_estimation_init.m sets npred = M_.nspred here, so the columns are those of all the state variables
    D.restrict_columns = _col(Int[findfirst(==(cnt.nstatic + j), k2) for j in 1:nspred])
    if sol !== nothing
        gh = Matrix{Float64}(ghx(sol))
        ss = get(oo, "steady_state", nothing)   # levels, as Dynare's dr.ys (sol.ys is in logs under loglinear)
        D.ys = (ss isa AbstractArray && length(ss) == length(sol.ys)) ? _col(Float64.(vec(ss))) : _col(sol.ys)
        D.ghx = gh
        D.ghu = Matrix{Float64}(ghu(sol))
        D.state_var = _row(sol.state_var)
        D.sdim = Float64(nspred)
        D.edim = Float64(cnt.nboth + cnt.nfwrd)
    end
    oo.dr = D
end

const _SMOOTHER_FIELDS = ("Smoother", "SmoothedVariables", "UpdatedVariables", "FilteredVariables", "FilteredVariablesKStepAhead",
                          "FilteredVariablesShockDecomposition", "FilteredVariablesKStepAheadVariances", "SmoothedShocks",
                          "SmoothedMeasurementErrors")

"""
oo_.Smoother, oo_.SmoothedVariables, oo_.UpdatedVariables, oo_.FilteredVariables, oo_.FilteredVariablesKStepAhead(Variances),
oo_.SmoothedShocks and oo_.SmoothedMeasurementErrors from a `SmootherResult` (store_smoother_results.m): every variable
a (nobs x 1) column, fields created in decision-rule order as Dynare does, the smoother's own results removed first.
"""
function _store_smoother_results!(oo::MStruct, m::Model, res::SmootherResult)
    if res.occbin !== nothing                   # OccBin: oo_.occbin.smoother; nothing else is written if the smoother failed
        _store_occbin_smoother!(oo, m, res)
        res.occbin.smoother.error_flag == 0 || return oo
    end
    for f in _SMOOTHER_FIELDS
        delete!(oo, f)
    end
    nm = res.endo_names
    ord = res.solution.order_var
    varobs = res.varobs
    nk = !isempty(res.filter_step_ahead)
    S = MStruct()
    S.occbin = res.occbin !== nothing
    S.loglinear = res.loglinear
    S.SteadyState = _col(res.SteadyState)
    S.TrendCoeffs = _col(res.TrendCoeffs)
    T = MStruct()
    for v in nm
        v in varobs || (T[v] = _col(res.Trend[v]))
    end
    for v in varobs
        T[v] = _col(res.Trend[v])
    end
    S.Trend = T
    res.Variance === nothing || (S.Variance = res.Variance)
    C, I = MStruct(), MStruct()
    SV, FV, UV = MStruct(), MStruct(), MStruct()
    for i in ord
        v = nm[i]
        C[v] = _col(res.Constant[v])
        I[v] = res.Init_State[v]
        SV[v] = _col(res.SmoothedVariables[v])
        nk && (FV[v] = _col(res.FilteredVariables[v]))
        UV[v] = _col(res.UpdatedVariables[v])
    end
    S.Constant = C
    S.Init_State = I
    S.Init_State_Vector = _col(res.Init_State_Vector)
    if res.State_uncertainty !== nothing
        S.State_uncertainty = res.State_uncertainty
        S.Init_State_uncertainty = res.Init_State_uncertainty
    end
    oo.Smoother = S
    nk && res.FilteredVariablesKStepAhead !== nothing && (oo.FilteredVariablesKStepAhead = res.FilteredVariablesKStepAhead)
    nk && res.FilteredVariablesKStepAheadVariances !== nothing && (oo.FilteredVariablesKStepAheadVariances = res.FilteredVariablesKStepAheadVariances)
    oo.SmoothedVariables = SV
    nk && (oo.FilteredVariables = FV)
    oo.UpdatedVariables = UV
    sh = MStruct()
    for v in res.exo_names
        sh[v] = _col(res.SmoothedShocks[v])
    end
    oo.SmoothedShocks = sh
    if !isempty(res.SmoothedMeasurementErrors)
        me = MStruct()
        for v in varobs
            me[v] = _col(res.SmoothedMeasurementErrors[v])
        end
        oo.SmoothedMeasurementErrors = me
    end
    oo
end

"M_ as set_all_parameters.m leaves it: params, Sigma_e, Correlation_matrix, H (the model follows)."
function _apply_theta!(ws, est::Estimation, theta)
    m = ws.model
    apply_estimates!(m, est, theta)
    _, H, CME = _covariances(est, theta)
    _model_to_M!(ws)
    ws.M_.H = Matrix{Float64}(H)
    ws.M_.Correlation_matrix_ME = Matrix{Float64}(CME)
    ws.state[:CME] = copy(ws.M_.Correlation_matrix_ME)
    ws.state[:H] = copy(ws.M_.H)                 # dynare() rebuilds M_ at the end of the file and keeps this H
    C = copy(declared_correlation_matrix(m))          # estimated correlations replace their entries (set_all_parameters.m)
    Cs = _corr(m.Sigma_e)
    for (_, (k1, k2), _) in est.groups.cx
        C[k1, k2] = C[k2, k1] = Cs[k1, k2]
    end
    declared_correlation_matrix(m) .= C                  # M_ keeps them afterwards (the model's synchronisation reads this matrix)
    ws.M_.Correlation_matrix = copy(C)
    ws
end

"options_ fields estimation / calib_smoother set on the way (dynare_estimation_init.m): data size, observed variables, jumping scale, filter horizon."
function _set_estimation_options!(ws, m::Model, nobs, n_est, fsa, first_obs, lik_init)
    O = ws.options_
    O.nobs = Float64(nobs)
    O.first_obs = first_obs
    O.varobs_id = _row(Int[m.ie[v] for v in m.varobs])
    O.number_of_observed_variables = Float64(length(m.varobs))
    isempty(_optf(ws, "mh_jscale", zeros(0, 0))) && (O.mh_jscale = 2.38 / sqrt(n_est))
    fv = _optb(ws, "filtered_vars", false)
    (!fv && !isempty(fsa)) && (fv = true)
    if fv && isempty(fsa)
        fsa = [1]
    end
    O.filtered_vars = fv
    O.filter_step_ahead = isempty(fsa) ? zeros(0, 0) : _col(fsa)
    O.nk = isempty(fsa) ? zeros(0, 0) : Float64(maximum(fsa))
    O.lik_init = Float64(lik_init)
    O
end

# ---------------------------------------------------------------------- estimation
"Dynare's exitflag of csminwel (its return code) from the message `posterior_mode` stores."
function _csminwel_exitflag(msg)
    for (k, v) in _CSMINWEL_MESSAGES
        v == msg && return Float64(k)
    end
    0.0
end

"""
    estimation!(ws=current workspace, var_list=[]; data=nothing, seed=0, options...)

Dynare's `estimation` statement on a workspace: the options of options_ (keywords are written into it first, as
`estimation(mode_compute=4, ...)` does) drive `Estimation`, `posterior_mode`, `sample_posterior` and `smooth`, and the results
go where Dynare 7.2 puts them: `oo_.posterior_mode` / `oo_.mle_mode` and `oo_.posterior_std_at_mode`,
`oo_.posterior.optimization`, `oo_.MarginalDensity`, `oo_.prior`, with an MCMC run `oo_.posterior_mean`, `_median`, `_hpdinf`,
`_hpdsup`, `_variance`, `_std`, `_deciles` and `oo_.posterior.metropolis`, and the (frequentist) smoother's `oo_.Smoother`,
`oo_.SmoothedVariables`, `oo_.UpdatedVariables`, `oo_.FilteredVariables`, `oo_.SmoothedShocks`,
`oo_.SmoothedMeasurementErrors`, `oo_.dr`. `M_.params`, `M_.Sigma_e` and `M_.H` are set to the mode (to the posterior mean
after an MCMC run), as in Dynare. Options the code cannot act on raise an `EstimationError`, as `Estimation` does.
`data` replaces the datafile; `seed` seeds the Metropolis-Hastings chains.
"""
function estimation!(ws::DynareWorkspace, var_list=String[]; data=nothing, seed=0, plots=nothing, options...)
    _with_plots(ws, plots) do
        _estimation!(ws, nothing, var_list, data, seed, options)
    end
end

function _estimation!(ws::DynareWorkspace, st, var_list, data, seed, options)
    m = _sync_model!(ws)
    O, oo = ws.options_, ws.oo_
    _take_options!(ws, options)
    stopts = st === nothing ? Dict{String,Any}() : (o = get_or(st, "options", nothing); o isa AbstractDict ? o : Dict{String,Any}())
    # EstimationStatement::writeOutput: options_.order = 1 unless the statement gives an order
    (haskey(stopts, "order") || haskey(options, :order)) || (O.order = 1.0)
    _optb(ws, "selected_variables_only", false) &&
        throw(EstimationError("selected_variables_only is not supported by the oo_ writer (oo_.SmoothedVariables would not have Dynare's layout)"))
    vl = String.(collect(var_list))
    for v in vl
        haskey(m.ie, v) || throw(EstimationError("estimation: $v is not an endogenous variable"))
    end
    fsa = _fsa(ws)
    smoother_flag = _optb(ws, "smoother", false)
    wants_filter = _optb(ws, "filtered_vars", false) || !isempty(fsa)
    first_obs0 = let v = _optf(ws, "first_obs", NaN); v isa Real ? Float64(v) : NaN end
    # dynare_estimation_1.m returns before it resets first_obs to NaN when only the smoother runs (mode_compute=0, order 1)
    early_return = (_optf(ws, "mode_compute", 4.0) == 0 && isempty(string(_optf(ws, "mode_file", ""))) && _opti(ws, "order", 1) == 1) ||
                   _opti(ws, "order", 1) > 1        # with a particle filter dynare_estimation_1.m returns before the reset too
    result = _with_options(ws, _SOLVE_KINDS) do
        kw = _estimation_kwargs(ws, keys(options))
        est = Estimation(m; data=data, statement=0, kw...)
        o = est.options
        haskey(o, "nl") && _check_nonlinear_options!(est; smoother=smoother_flag, filtered=wants_filter,
                                                     forecast=_opti(ws, "forecast", 0), mh=o["mh_replic"] > 0)
        bayes = is_bayesian(est.prior)
        # mode file, samplers without a mode search, load_mh_file (src/mcmc_stage.jl)
        mfile = String(get(o, "mode_file", ""))
        load_mh = _est_flag(get(o, "load_mh_file", false))
        skip_mode = bayes && samples_without_mode(est)
        mc0 = skip_mode || (o["mode_compute"] == 0 && isempty(mfile))
        mcmc_wanted = bayes && (skip_mode ? (sampling_method(est) in SMC_METHODS || o["mh_replic"] > 0 || load_mh) :
                                            (!mc0 && (o["mh_replic"] > 0 || load_mh)))
        if mc0 && !skip_mode && (o["mh_replic"] > 0 || load_mh) && bayes
            @warn "estimation: mode_compute=0 without a mode_file: Dynare returns before the Metropolis-Hastings run, so mh_replic=$(o["mh_replic"]) is not used"
        end
        initial_estimation_checks(est)
        if _optb(ws, "debug", false)         # initial_estimation_checks.m: with options_.debug the objective at the initial (or mode-file) values
            theta_init = isempty(mfile) ? est.x0 : match_mode_file(est, read_mode_file(m, mfile); mode_compute=o["mode_compute"])[1]
            fv, finfo = _dsge_likelihood(est, theta_init; bounds=(est.lb, est.ub))[1:2]
            finfo == 0 || throw(EstimationError("the likelihood at the initial parameters cannot be computed: $(est.last_info)"))
            oo.likelihood_at_initial_parameters = Float64(fv)
        end
        # Dynare sets M_ to the initial values first; the steady state it stores is the one at those values
        sol0 = try
            first_order(m; params=model_params(est, est.x0), qz_criterium=o["qz_criterium"], loglinear=o["loglinear"])
        catch e
            e isa Union{SteadyStateError,BlanchardKahnError,LoglinearError} || rethrow()
            nothing
        end
        sol0 === nothing || (oo.steady_state = _col(sol0.steady_state))
        O.noconstant = 0.0
        _store_prior!(oo, est)
        _store_dr_estimation!(oo, m)
        _fig_est!(ws, :prior, est)                  # src/plots_hooks.jl (figures only with plots=true)
        theta = copy(est.x0)
        mode = nothing
        mc = nothing
        post = nothing
        mhm = NaN
        smooth_at = nothing                       # parameter vector of the final smoother, if it runs
        if mc0
            if mcmc_wanted       # a sampler that needs no mode (slice from the prior mode, SMC)
                mc, post, theta = _mcmc_block!(ws, m, est, nothing, vl, seed)
                smoother_flag && !(o["mh_replic"] > 0 || load_mh) && (smooth_at = theta)
            else
                smoother_flag && (smooth_at = theta)
            end
        else
            t0 = time()
            from_file = o["mode_compute"] == 0
            x0m = est.x0
            if !from_file && !isempty(mfile)       # mode_compute > 0: the mode file only supplies the starting values (check_mode_file.m)
                x0m = match_mode_file(est, read_mode_file(m, mfile); mode_compute=o["mode_compute"])[1]
            end
            mode = from_file ? mode_from_file(est, mfile) : posterior_mode(est; x0=x0m)
            runtime = time() - t0
            theta = copy(mode.x)
            _fig_est!(ws, :mode, est; mode=mode)
            if !from_file                      # dynare_estimation_1.m: <fname>/Output/<fname>_mode.mat (xparam1, hh, parameter_names, fval)
                try
                    out = joinpath(m.mod_dir, m.name, "Output"); mkpath(out)
                    md = Dict{String,Any}("xparam1" => reshape(copy(mode.x), :, 1), "parameter_names" => reshape(copy(est.names), :, 1),
                                          "fval" => -mode.log_posterior)
                    o["cova_compute"] != 0 && mode.hessian !== nothing && (md["hh"] = Matrix{Float64}(mode.hessian))
                    MAT.matwrite(joinpath(out, m.name * "_mode.mat"), md)
                catch e
                    @debug "mode file not written" e
                end
            end
            P = get(oo, "posterior", nothing)
            P isa MStruct || (P = MStruct(); oo.posterior = P)
            haskey(P, "optimization") && delete!(P, "optimization")
            opt = MStruct()
            if !from_file          # no optimisation, no optimization_info with a mode file (mode_compute=0)
                inf = MStruct()
                inf.runtime = runtime
                inf.iterations = Float64(mode.iterations)
                inf.funcCount = Float64(mode.nfev)
                inf.exitflag = _csminwel_exitflag(mode.message)
                inf.message = mode.message
                opt.optimization_info = inf
            end
            cova = o["cova_compute"] != 0
            opt.mode = _col(mode.x)
            opt.Variance = (cova && mode.inv_hessian !== nothing) ? Matrix{Float64}(mode.inv_hessian) : zeros(0, 0)
            opt.log_density = from_file ? zeros(0, 0) : mode.log_posterior     # Dynare has no fval with a mode file
            P.optimization = opt
            cova && mode.inv_hessian === nothing && @warn "estimation: the Hessian at the mode is not positive definite (check_hessian_at_the_mode.m): no standard errors, no Laplace approximation"
            stdh = (cova && mode.inv_hessian !== nothing) ? mode_std(mode) : fill(NaN, est.n)
            tag = bayes ? "posterior" : "mle"
            _fill_by_name!(oo, tag * "_mode", est, mode.x)
            _fill_by_name!(oo, tag * "_std_at_mode", est, stdh)
            if bayes && cova
                MD = get(oo, "MarginalDensity", nothing)
                MD isa MStruct || (MD = MStruct(); oo.MarginalDensity = MD)
                MD.LaplaceApproximation = mode.inv_hessian === nothing ? NaN : mode.laplace
            end
            bayes && dsge_var_active(est) && _store_dsge_var!(oo, est, mode.x)        # oo_.dsge_var.posterior_mode (dynare_estimation_1.m)
            if mcmc_wanted
                (mode.inv_hessian !== nothing || String(get(o, "mcmc_jumping_covariance", "hessian")) != "hessian") ||
                    throw(EstimationError("mh_replic > 0 needs a posterior mode and its Hessian (mode_compute != 0 and cova_compute != 0)"))
                mc, post, theta = _mcmc_block!(ws, m, est, mode, vl, seed)      # Dynare then sets M_ to the posterior mean
            end
            # the frequentist smoother runs at the mode / posterior mean unless a Bayesian smoother was asked for
            # (Dynare skips it when the Bayesian smoother ran over the posterior draws)
            _optb(ws, "frequentist_smoother", true) && !haskey(o, "nl") && !(smoother_flag && mcmc_wanted) && (smooth_at = theta)     # no smoother with a particle filter
        end
        res = nothing
        if smooth_at !== nothing
            sk = _smoother_kwargs(ws; full=false)
            res = smooth(m; estimation=est, theta=smooth_at, sk...)
        end
        ws.state[:est_last] = (est=est, mode=mode, mc=mc)
        _fig_est!(ws, :smoother, est; res=res)
        (est, theta, res, est.nobs, o["lik_init"])
    end
    est, theta, res, nobs, lik_init = result
    _apply_theta!(ws, est, theta)
    if res !== nothing
        _store_smoother_results!(oo, m, res)
        _store_dr_estimation!(oo, m, res.solution)
    end
    _estimation_forecast!(ws, m, est, theta, res, vl)
    _set_estimation_options!(ws, m, nobs, est.n, _fsa(ws), early_return && isnan(first_obs0) ? 1.0 : first_obs0, lik_init)
    O.varlist = _cellcol(isempty(vl) && (smoother_flag || wants_filter) ? orig_endo_names(m) : vl)
    _gui!(oo, "ran_estimation")
    ws
end

"""
    _posterior_density(draws; ngrid=512) -> (ngrid x 2) [abscissa density]

The Gaussian-kernel density estimate of `posterior_moments.m` (kernel_density_estimate.m with the rule-of-thumb bandwidth of
mh_optimal_bandwidth.m, bandwidth option 0, corrected for repeated draws), on a grid of `ngrid` points; NaN for a variance
below 1e-12. The FFT convolution of Dynare is done directly.
"""
function _posterior_density(draws::AbstractVector; ngrid::Int=512)
    xx = sort(Float64.(draws))
    n = length(xx)
    mu = sum(xx) / n
    v = sum(abs2, xx .- mu) / (n - 1)
    v > 1e-12 || return fill(NaN, ngrid, 2)
    corr = 0.0                                              # correction_for_repeated_draws
    for i in 1:n
        j = i
        while j <= n && xx[j] - xx[i] < 2 * eps()
            j += 1
        end
        corr += 2 * (j - i) - 1
    end
    corr /= n
    mu02, mu21 = 1 / (2 * sqrt(pi)), 1.0
    h = 2 * sqrt(v) * (sqrt(pi) * mu02 / (12 * mu21^2 * n))^(1 / 5) * corr^(1 / 5)
    lo = xx[1] - (xx[end] - xx[1]) / 3
    hi = xx[end] + (xx[end] - xx[1]) / 3
    absc = collect(range(lo, hi; length=ngrid))
    inc = absc[2] - absc[1]
    xi = zeros(ngrid)
    for x in xx
        xa = (x - lo) / (hi - lo) * ngrid
        idx = floor(Int, xa)
        t = xa - idx
        xi[idx+1] += 1 - t
        xi[idx+2] += t
    end
    kern(l) = exp(-0.5 * (l * inc / h)^2) / sqrt(2 * pi)
    total = sum(kern(l) for l in -ngrid:ngrid-1)
    scale = 1 / (total * inc * n)
    f = zeros(ngrid)
    for t in 1:ngrid, s in 1:ngrid
        f[t] += xi[s] * kern(t - s)
    end
    hcat(absc, f .* scale)
end

"oo_.posterior_mean ... and oo_.posterior.metropolis after an MCMC run (GetPosteriorParametersStatistics.m, GetPosteriorMeanVariance.m)."
function _store_posterior_statistics!(oo::MStruct, est::Estimation, mc::MCMCResult, post, mhm)
    names = est.names
    nm(f) = Float64[f(post[n]) for n in names]
    _fill_by_name!(oo, "posterior_mean", est, nm(p -> p.mean))
    _fill_by_name!(oo, "posterior_hpdinf", est, nm(p -> p.hpd[1]))
    _fill_by_name!(oo, "posterior_hpdsup", est, nm(p -> p.hpd[2]))
    _fill_by_name!(oo, "posterior_median", est, nm(p -> p.median))
    _fill_by_name!(oo, "posterior_variance", est, nm(p -> p.var))
    _fill_by_name!(oo, "posterior_std", est, nm(p -> p.std))
    _fill_by_name!(oo, "posterior_deciles", est, [post[n].deciles for n in names]; f=_col)
    X = kept_draws(mc)
    _fill_by_name!(oo, "posterior_density", est, [_posterior_density(X[:, j]) for j in 1:est.n])
    MD = get(oo, "MarginalDensity", nothing)
    MD isa MStruct || (MD = MStruct(); oo.MarginalDensity = MD)
    MD.ModifiedHarmonicMean = mhm
    N = size(X, 1)
    mu = vec(sum(X; dims=1)) ./ N
    Xc = X .- mu'
    M = MStruct()
    M.mean = _col(mu)
    M.Variance = (Xc' * Xc) ./ N
    P = get(oo, "posterior", nothing)
    P isa MStruct || (P = MStruct(); oo.posterior = P)
    P.metropolis = M
    oo
end

# ---------------------------------------------------------------------- calib_smoother
"""
    calib_smoother!(ws=current workspace, var_list=[]; options...)

Dynare's `calib_smoother` (evaluate_smoother.m) on a workspace: the Kalman smoother at the calibration, or at the parameter set
`options_.parameter_set` (`posterior_mode`, `posterior_mean`, `posterior_median`, `mle_mode`, `prior_mode`, `prior_mean`; the
first four are read back from oo_ as Dynare's get_posterior_parameters.m does). Fills `oo_.Smoother`, `oo_.SmoothedVariables`,
`oo_.UpdatedVariables`, `oo_.FilteredVariables`, `oo_.SmoothedShocks`, `oo_.SmoothedMeasurementErrors`, `oo_.dr` and `oo_.prior`;
`M_.params` and the other M_ fields stay as they are, as in Dynare.
"""
function calib_smoother!(ws::DynareWorkspace, var_list=String[]; plots=nothing, options...)
    _with_plots(ws, plots) do
        _calib_smoother!(ws, nothing, var_list, options)
    end
end

function _calib_smoother!(ws::DynareWorkspace, st, var_list, options)
    m = _sync_model!(ws)
    O, oo = ws.options_, ws.oo_
    _take_options!(ws, options)
    stopts = st === nothing ? Dict{String,Any}() : (o = get_or(st, "options", nothing); o isa AbstractDict ? o : Dict{String,Any}())
    # CalibSmootherStatement::writeOutput
    (haskey(stopts, "parameter_set") || haskey(options, :parameter_set)) || (O.parameter_set = "calibration")
    O.smoother = true
    O.order = 1.0
    _optb(ws, "selected_variables_only", false) &&
        throw(SmootherError("selected_variables_only is not supported by the oo_ writer (oo_.SmoothedVariables would not have Dynare's layout)"))
    for v in String.(collect(var_list))
        haskey(m.ie, v) || throw(SmootherError("calib_smoother: $v is not an endogenous variable"))
    end
    pset = String(_optf(ws, "parameter_set", "calibration"))
    first_obs0 = let v = _optf(ws, "first_obs", NaN); v isa Real ? Float64(v) : NaN end
    has_est = any(st -> get(st, "statementName", "") == "estimated_params", m.statements)
    res, est = _with_options(ws, _SOLVE_KINDS) do
        sk = _smoother_kwargs(ws; full=true)
        est = nothing
        if has_est
            kw = _estimation_kwargs(ws, ())
            for k in (:parameter_set,)
                delete!(kw, k)
            end
            est = try
                Estimation(m; statement=0, kw...)
            catch e
                e isa Union{EstimationError,PriorError} || rethrow()
                pset == "calibration" || rethrow()
                warn_once(m, "calib_smoother: oo_.prior is not filled ($(sprint(showerror, e)))")
                nothing
            end
        end
        if pset == "calibration"
            res = smooth(m; sk...)
        else
            est === nothing && throw(SmootherError("parameter_set=$(repr(pset)) needs an estimated_params block"))
            theta = if pset == "posterior_mode"
                _xparam_from_oo(ws, est, "posterior_", "mode")
            elseif pset == "mle_mode"
                _xparam_from_oo(ws, est, "mle_", "mode")
            elseif pset == "posterior_mean"
                _xparam_from_oo(ws, est, "posterior_", "mean")
            elseif pset == "posterior_median"
                _xparam_from_oo(ws, est, "posterior_", "median")
            else
                parameter_vector(est, pset)
            end
            res = smooth(m; estimation=est, theta=theta, parameter_set=pset, sk...)
        end
        res, est
    end
    _store_prior!(oo, est)
    # dynare_estimation_init.m stores the steady state at the initial values of the estimated parameters
    ys0 = res.solution.steady_state
    if est !== nothing
        sol0 = try
            first_order(m; params=model_params(est, est.x0), qz_criterium=est.options["qz_criterium"], loglinear=est.options["loglinear"])
        catch e
            e isa Union{SteadyStateError,BlanchardKahnError,LoglinearError} || rethrow()
            nothing
        end
        sol0 === nothing || (ys0 = sol0.steady_state)
    end
    oo.steady_state = _col(ys0)
    O.noconstant = 0.0
    _store_smoother_results!(oo, m, res)
    _store_dr_estimation!(oo, m, res.solution)
    nobs = size(res.smoothed, 1)
    # evaluate_smoother.m does not reset first_obs to NaN as dynare_estimation_1.m does
    _set_estimation_options!(ws, m, nobs, est === nothing ? 0 : est.n, _fsa(ws), isnan(first_obs0) ? 1.0 : first_obs0, res.options["lik_init"])
    _gui!(oo, "ran_calib_smoother")
    est === nothing || _fig_est!(ws, :smoother, est; res=res)       # src/plots_hooks.jl
    ws
end

# the commands act on the current workspace as MATLAB's act on the globals
estimation!(; kw...) = estimation!(_current(); kw...)
estimation!(var_list::AbstractVector; kw...) = estimation!(_current(), var_list; kw...)
calib_smoother!(; kw...) = calib_smoother!(_current(); kw...)
calib_smoother!(var_list::AbstractVector; kw...) = calib_smoother!(_current(), var_list; kw...)

"""
The `forecast=` option of `estimation` (dynare_estimation_1.m): the classical forecast from the smoother at the mode
(with `mode_compute=0` only when the smoother ran) into `oo_.forecast`; with `mh_replic > 0` the forecasts over the posterior draws
(`oo_.MeanForecast` ...) come from `_posterior_stage!`.
"""
function _estimation_forecast!(ws, m, est, theta, res, vl)
    nf = _opti(ws, "forecast", 0)
    nf > 0 || return
    mcmc = _opti(ws, "mh_replic", 0) > 0 && _opti(ws, "mode_compute", 5) != 0
    mcmc && return                          # forecasts over the posterior draws: _posterior_stage!
    sm = res
    if sm === nothing
        _opti(ws, "mode_compute", 5) == 0 && return         # Dynare forecasts at the initial values only with `smoother`
        sm = smooth(m; estimation=est, theta=theta)
    end
    fc = forecast(sm, m; periods=nf, conf_sig=Float64(_optf(ws, "forecasts.conf_sig", 0.9)), var_list=String.(collect(vl)))
    ws.oo_.forecast = forecast_struct(fc)
    nothing
end
