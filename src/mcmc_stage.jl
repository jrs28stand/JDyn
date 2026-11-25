# The sampling stage of `estimation` (dynare_estimation_1.m, from the mode file / mode search to the posterior statistics) for the
# samplers of src/samplers.jl, src/smc.jl and src/mcmc_driver.jl, with Dynare's `mode_file`, `load_mh_file` and the MH files on disk.
# Hooks are called from results_estimation.jl.

"The workspace's seed (`options_.DynareRandomStreams.seed`) after syncing its stream."
function _ws_seed(ws)
    dynare_rng(ws)
    _streams(ws.model).seed
end

function _estopt(est::Estimation, key, default)
    v = get(est.options, key, default)
    v === nothing ? default : v
end

"The posterior sampling method of the estimation (default random walk)."
sampling_method(est::Estimation) = String(_estopt(est, "posterior_sampling_method", RWMH))

"Does Dynare skip the mode search for this sampler? SMC samplers and the slice sampler that does not start from the mode."
function samples_without_mode(est::Estimation)
    _est_flag(get(est.options, "mh_posterior_mode_estimation", false)) && return true     # dynare_estimation_1.m: no smoother-only step, no mode search
    meth = sampling_method(est)
    meth in SMC_METHODS && return true
    meth == SLICE || return false
    !_est_flag(get(_kv_dict(_estopt(est, "posterior_sampler_options", nothing)), "slice_initialize_with_mode", 0))
end
_kv_dict(spec) = Dict{String,Any}(_kv_pairs(spec))

"""
    mode_from_file(est, name) -> ModeResult

`mode_compute=0` with `mode_file`: the mode `xparam1` and Hessian `hh` of Dynare's file (`<fname>_mode.mat`; names matched as
`check_mode_file.m`), with the inverse Hessian and Laplace approximation as `dynare_estimation_1.m` computes them.
"""
function mode_from_file(est::Estimation, name::AbstractString)
    mf = read_mode_file(est.model, name)
    x, hh = match_mode_file(est, mf; mode_compute=0)
    out = ModeResult(copy(x), est.names, log_posterior(est, x), nothing, nothing, NaN, true, "mode_file", 0, 0)
    if est.options["cova_compute"] != 0
        hh === nothing && throw(EstimationError("mode_file $name has no Hessian (variable hh): use cova_compute=0 or a mode file written with the Hessian"))
        out.hessian = Matrix{Float64}(hh)
        if all(isfinite, hh) && all(>(0), diag(hh))
            out.inv_hessian = inverse_hessian(hh)
            any(<=(0), diag(out.inv_hessian)) && (out.inv_hessian = nothing)
        end
        out.inv_hessian !== nothing && is_bayesian(est.prior) && (out.laplace = laplace(est, x, out.inv_hessian))
    end
    out
end

"The earlier MCMC of `load_mh_file` as an MCMCResult (Dynare's `load_last_mh_history_file` and the `_mh<k>_blck<j>` files)."
function load_previous_mcmc(est::Estimation)
    m = est.model
    rec = load_last_mh_history(m)
    (!isnan(rec.concluded) && rec.concluded == 0) &&
        throw(EstimationError("You are trying to load an MCMC that did not finish successfully. Please use mh_recover."))
    draws, lp = load_mh_draws(m, rec.nblck, Int(sum(rec.mhdraws[:, 2])))
    size(draws, 3) == est.n || throw(EstimationError("load_mh_file: the stored draws are for $(size(draws, 3)) parameters, the model estimates $(est.n)"))
    info = Dict{Symbol,Any}(:method => rec.sampler, :loaded => true)
    if rec.last_seeds !== nothing
        info[:last_seeds] = rec.last_seeds
    end
    if !isempty(rec.proposal_covariance) && !isempty(rec.proposal_scale)
        info[:proposal_chol] = rec.proposal_covariance
        info[:jscale_vector] = rec.proposal_scale
        info[:jscale] = rec.proposal_scale[1]
    end
    x0 = size(rec.initial_parameters, 2) == est.n ? rec.initial_parameters : draws[:, 1, :]
    acc = isempty(rec.acceptance) ? fill(NaN, rec.nblck) : rec.acceptance
    MCMCResult(draws, lp, acc, est.names, x0, est.options["mh_drop"], falses(rec.nblck, size(draws, 2)), nothing, info), rec
end

"""
Run the sampler of the estimation (`posterior_sampler`, or an SMC sampler), write Dynare's MH files, compute the posterior
statistics, diagnostics and the posterior analyses of `_posterior_stage!`; returns `(mc, post, theta)` with `theta` the posterior mean.
`mode` is nothing for samplers that do not start from a mode.
"""
function _mcmc_block!(ws, m::Model, est::Estimation, mode, vl, seed)
    o = est.options
    oo = ws.oo_
    method = sampling_method(est)
    kind = _mcmc_kind(est, nothing)
    sd = kind == :octave ? _ws_seed(ws) : seed
    load_mh = _est_flag(get(o, "load_mh_file", false))
    _est_flag(get(o, "mh_recover", false)) && throw(EstimationError("mh_recover (recovering a crashed MCMC from its temporary files) is not implemented; use load_mh_file to continue a finished one"))
    mhm = NaN
    local mc, smc
    smc = nothing
    if method in SMC_METHODS
        smc = sample_smc(est; seed=sd)
        mc = as_mcmc(smc)
    else
        previous = nothing; rec = nothing
        if load_mh
            previous, rec = load_previous_mcmc(est)
        end
        jcov = String(get(o, "mcmc_jumping_covariance", "hessian"))
        spec = _kv_dict(_estopt(est, "posterior_sampler_options", nothing))
        plain = method == RWMH && kind == :julia && previous === nothing && jcov == "hessian" && mode isa ModeResult &&
                mode.inv_hessian !== nothing && _tune_target(o) === nothing && get(spec, "proposal_distribution", "rand_multivariate_normal") == "rand_multivariate_normal" &&
                !_est_flag(get(spec, "use_mh_covariance_matrix", 0)) && !haskey(spec, "scale_file")
        if plain          # the default sampler with Julia's generators: JDyn's original chains (sample_posterior)
            mc = sample_posterior(est; mode=mode, seed=seed)
            js, base = JDyn.jscale(est)
            mc.info[:method] = RWMH; mc.info[:jscale_vector] = js; mc.info[:jscale] = base
            mc.info[:proposal_chol] = Matrix(cholesky(Symmetric(Matrix{Float64}(mode.inv_hessian))).U)
        else
            mc = sample_mcmc(est; mode=mode, previous=previous, seed=sd, rng=kind)
        end
        if o["mh_replic"] > 0
            try
                write_mh_files(m, mc, mc.info; previous=rec, new_draws=Int(o["mh_replic"]))
            catch e
                @warn "estimation: the MH files could not be written" exception = e
            end
        end
    end
    post = posterior_statistics(est, mc)
    if smc === nothing
        mhm = marginal_density_mhm(mc)
        _optb(ws, "nodiagnostic", false) || _posterior_diagnostics!(ws, est, mc)
    end
    _store_posterior_statistics!(oo, est, mc, post, mhm)
    MD = oo["MarginalDensity"]
    if smc !== nothing
        haskey(MD, "ModifiedHarmonicMean") && delete!(MD, "ModifiedHarmonicMean")
        setfield_md!(MD, smc.method, smc.log_marginal_density)
    end
    # marginal_density.m / compute_posterior_covariance_matrix.m: without a mode (or with the slice sampler) the posterior mode
    # is the draw with the highest kernel
    if !haskey(oo, "posterior_mode") || method == SLICE
        best = -Inf; xb = nothing
        for j in 1:size(mc.draws, 1), t in 1:size(mc.draws, 2)
            if mc.log_posterior[j, t] > best
                best = mc.log_posterior[j, t]; xb = mc.draws[j, t, :]
            end
        end
        xb === nothing || (_fill_by_name!(oo, "posterior_mode", est, xb); _fill_by_name!(oo, "posterior_std_at_mode", est, fill(NaN, est.n)))
    end
    _posterior_stage!(ws, m, est, mc, vl)
    _fig_est!(ws, :mcmc, est; mc=mc, mode=mode)       # src/plots_hooks.jl
    theta = Float64[post[nm].mean for nm in est.names]
    mc, post, theta
end

setfield_md!(MD::MStruct, name::AbstractString, v) = (MD[name] = v)
