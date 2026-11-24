# The posterior stage of `estimation` (dynare_estimation_1.m, after the Metropolis-Hastings run): MCMC convergence diagnostics,
# posterior IRFs (`bayesian_irf`), posterior theoretical moments (`moments_varendo`) and the posterior distributions of smoothed
# / filtered variables and forecasts (`prior_posterior_statistics`), in Dynare's order. Hooks are called from results_estimation.jl.

"Convergence diagnostics of the chains into `oo_.convergence` (Dynare: `mcmc_diagnostics.m`, skipped by `nodiagnostic`)."
function _posterior_diagnostics!(ws, est::Estimation, mc::MCMCResult)
    store_convergence!(ws.oo_, mcmc_diagnostics(est, mc))
    nothing
end

"""
Posterior computations that follow the sampler: `bayesian_irf`, `moments_varendo`, then smoother / `filtered_vars` /
`filter_step_ahead` / `forecast` over the same kind of sub-draws. Each of them draws its own sub-sample from the model's stream, as
Dynare's separate calls do. `sub_draws = 0` skips them all.
"""
function _posterior_stage!(ws, m::Model, est::Estimation, mc::MCMCResult, vl)
    o = est.options
    oo = ws.oo_
    sd = get(o, "sub_draws", nothing)
    if sd !== nothing && !(sd isa AbstractVector) && Int(sd) == 0
        @info "estimation: sub_draws was set to 0. Skipping posterior computations."
        return
    end
    flag(k) = _est_flag(get(o, k, false))
    fsa = get(o, "filter_step_ahead", Int[])
    wants_stats = flag("smoother") || flag("filtered_vars") || !(fsa === nothing || (fsa isa AbstractVector && isempty(fsa)))
    nf = Int(get(o, "forecast", 0))
    isempty(m.occbin_constraints) || ((flag("bayesian_irf") || flag("moments_varendo") || wants_stats || nf > 0) &&
        throw(EstimationError("posterior IRFs, moments, smoothers and forecasts with OccBin are not supported")))
    vls = String.(collect(vl))
    flag("bayesian_irf") && store_posterior_irf!(oo, posterior_irf(est, mc; var_list=vls))
    flag("moments_varendo") && store_moments!(oo, posterior_theoretical_moments(est, mc; var_list=vls))
    if wants_stats || nf > 0
        sk = _smoother_kwargs(ws; full=false)
        for k in (:filtered_vars, :filter_covariance, :smoothed_state_uncertainty, :filter_step_ahead, :occbin)
            delete!(sk, k)
        end
        P, _ = posterior_draws(est, mc; sub_draws=nothing, drop=mc.drop, rng=dynare_rng(m))
        wants_stats && store_smoother_statistics!(oo, posterior_smoother_statistics(est; params=P, var_list=isempty(vls) ? nothing : vls, sk...))
        nf > 0 && store_posterior_forecast!(oo, posterior_forecast(est; params=P, horizon=nf, var_list=isempty(vls) ? nothing : vls, sk...))
    end
    nothing
end
