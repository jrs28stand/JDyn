# Parity batch 3 (branch parity3): pac.estimate, shock_decomposition(forecast_type), loglinear results layer

Clone /home/claude/work/jdyn-parity3 (base 1bb4530). Spec: pydyn main 42fcf84. Validation: Dynare 7.2 under Octave 8.4
(references test/ref_parity2/oo/, Octave store /home/claude/accept_ref/accept_ref).
Dev loop: a long-lived Julia session that `Base.include`s the changed files into JDyn, then runs test/test_parity2_results.jl
(`JDYN_TEST_ONLY=pe_t12,sd_fc_pm` selects cases).

## Status

| # | item | status |
|---|------|--------|
| 1 | pac.estimate.nls / iterative_ols | runs; all 11 pe_* cases within their tolerance (below); registered in runtests |
| 2 | shock_decomposition forecast_type | sd_fc_cal / ml / pm / pmm all agree with Octave to 3e-13 (see fixes) |
| 3 | loglinear results layer | done; test/test_parity3_loglinear.jl (22 checks), ll_est and example4_loglinear agree with Octave to 1e-9 |

## Item 1: pac.estimate

* Bug: `Set \ Set` (matrix division) in `_pe_set_auxiliary_series`: replaced by `setdiff`.
* Differences from the references (relative, floor 1): t12 6e-12, t9 6e-12, t19g 4e-11, t18 9e-10, t6 6e-10, t7 1.4e-9, t17 5e-9 (GaussNewton),
  t8a 9e-9 (annealing), v9 8e-8 (fmincon), t25 1.3e-7 (fmincon), t15 2e-5 (fminunc).
  fmincon/fminunc: own Levenberg-Marquardt to the minimiser; the references stop at the routines' own tolerance (1e-7 level), hence the gap.
  Annealing: same path as Dynare's (draws from Octave's stream); the last accepted points differ by 5e-9 because near the optimum the
  accept test compares sums of squares that differ in the 13th digit between Octave and Julia.

## Item 2: shock_decomposition(forecast_type=...)

Cases sd_fc_cal, sd_fc_ml, sd_fc_pm, sd_fc_pmm (test/test_parity2_results.jl, tolerance 1e-10 / 1e-9), measured agreement 3e-13.
Fixes:
* `conditional_forecast_paths` values are evaluated with endogenous names bound to oo_.steady_state (`_cf_eval`, src/forecast.jl).
* After estimation, oo_.steady_state is stored at the mode file's xparam1 when `mode_file` is given (as dynare_estimation_init does), dr.ys at the smoother's parameters.
* Posterior forecasts store `density` (pm3 kernel density, 512 points) in MeanForecast / PointForecast.
* With `mh_replic`, the Octave generator state after the last chain is copied into the global stream, so the later subsample and forecast-shock draws continue from it as in Dynare.

## Item 3: loglinear results layer (single mechanism: ws.state[:logged_ss])

* stoch_simul(loglinear): oo_.steady_state and oo_.dr.ys in logs (aux variables of exogenous lags/leads, types 2/3, not logged), options_.logged_steady_state = 1
  (readable from MATLAB lines, `_set_logged_flag!`, OPTION_DEFAULTS). stoch_simul and estimation undo it first (exp, flag 0); calib_smoother and `steady` leave levels.
* oo_.dr.Gy is the solver's hx for loglinear models.
* After estimation dr.ys is the steady state at the last resolution (levels), not at the initial values (the 8.5 offset in shock_decomposition/fs2000_est, conditional_forecasts/2/fs2000_est is gone).
* Posterior smoother with loglinear auxiliaries (sm_ll_aux): Dynare adds log(ss) to every variable, giving -Inf for aux-exo-lag variables; reproduced only under `replicate_dynare_bugs=true`, otherwise those series stay finite.
* Differences kept: Dynare's stale flag after steady/calib_smoother makes the next stoch_simul read levels as logs; JDyn does not reproduce it (steady! leaves flag 1 as Dynare does, ss in levels).
  Reading options_.logged_steady_state before any command is not defined (Dynare: 0).

## Refusals / approximations

* particleswarm (pac.estimate): refused.
* fmincon, fminunc, fminsearch, lsqnonlin: JDyn's own Levenberg-Marquardt (same minimiser, not the same iterates).
* dr.eigval of loglinear sm_ll_aux: infinite eigenvalues differ in noise (1e16 level); skipped in that test.
