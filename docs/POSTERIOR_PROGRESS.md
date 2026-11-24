# Posterior analyses over MCMC draws: status and next steps

Branch `posterior`. All first order. Validation replays Dynare 7.2's (Octave 8.4) own saved chains, sub-draws and random numbers
(references and generators: test/ref_posterior, test/ref_posterior/gen; tests: test/test_posterior.jl).

| Feature | API | oo_ | Agreement with Dynare (replayed draws) |
|---|---|---|---|
| Convergence diagnostics | `mcmc_diagnostics(est, mc)`, `store_convergence!` | `oo_.convergence` | ifac 8e-12 rel; Geweke 1.5e-15; chi2 p 2e-14; Brooks-Gelman (univariate, multivariate) 1.5e-12; Raftery-Lewis exact |
| Posterior IRFs | `posterior_irf` | `oo_.PosteriorIRF.dsge` | 3e-14 (post_nk; per-draw IRFs 2e-15) |
| Posterior moments | `posterior_theoretical_moments` | `oo_.PosteriorTheoreticalMoments.dsge` | 1.5e-14 (NK, 4 cases); 4e-11 (Schorfheide, a root near 0.99) |
| Smoothed / updated / filtered distributions | `posterior_smoother_statistics` | `oo_.SmoothedVariables` ... | 1e-13 (NK, 2 cases); 3e-11 (Schorfheide, same near-unit root) |
| Forecasts | `posterior_forecast` | `oo_.MeanForecast`, `PointForecast`, `PointForecastME` | 2e-14 (two cases, one with correlated measurement errors) |

## Wiring
`_estimation!` (src/results_estimation.jl) calls `_posterior_diagnostics!` and `_posterior_stage!` (src/posterior_stage.jl) after the
sampler, in Dynare's order: diagnostics, IRFs, moments, smoother statistics / forecasts. Each draws its own sub-sample from the model's
stream `dynare_rng`. The frequentist smoother is skipped when the Bayesian smoother ran (as Dynare). `sub_draws=0` skips them all.
Options now honoured: `sub_draws`, `nodecomposition`, `ar`, `conditional_variance_decomposition`, `contemporaneous_correlation`,
`moments_varendo`, `bayesian_irf`, `irf`, `irf_shocks`, `relative_irf`, `consider_all_endogenous(_and_auxiliary)`, `consider_only_observed`,
`posterior_max_subsample_draws`.

## Dynare 7.2 behaviours reproduced
- `oo_.FilteredVariables` is the first listed `filter_step_ahead` element shifted by one period; `FilteredVariablesKStepAheadVariances`
  holds the means; the posterior IRF has no density; the correlation density is an ar x 1 cell; the conditional decomposition has `Steps`.
- Measurement errors in forecasts are drawn as `chol(H) * randn` (upper factor), a covariance of `chol(H) chol(H)'` (warning if H is not diagonal).
- With B equal to the kept draws of one chain Dynare uses the first B pooled rows rather than a random sub-sample (`posterior_draws`).
- Inefficiency factor: Parzen window shifted by one lag including lag 0 (about 3 for iid draws); Geweke whole-chain moments include burn-in.

## Refused / not done
- `order > 1`, OccBin, `filter_covariance` and `smoothed_state_uncertainty` over draws, hp/bandpass filters set in a `stoch_simul` statement
  (raise); `moments_posterior_density` with a kernel other than gaussian / rule-of-thumb bandwidth, or gridpoints other than 512 (raise).
- `mode_file` is not read (mode_compute=0 with `mh_replic > 0` still returns before the sampler); `load_mh_file`; prior moments / prior draws;
  plots; Dynare's `consider_*` question without a list (nointeractive behaviour: all original endogenous variables).
- The smoother statistics and the forecast run the smoother separately (twice per draw when both are asked).

## Next steps
1. Read `mode_file` so the replay `.mod` files (mode_compute=0, mode_file=...) run end to end through `dynare()`.
2. Share smoother runs between smoother statistics and forecasts.
3. Prior-draw versions (`prior moments(distribution)`).
