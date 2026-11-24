# Status (2026-11-21)

Reference: Dynare 7.2 under GNU Octave 8.4 (the same reference results as pydyn). All figures below are
from `test/runtests.jl` (5,176 checks pass, 0 fail; 4 skipped or marked broken: two references without IRFs,
an auxiliary-variable perfect-foresight test not yet enabled, and an estimation point exactly on the QZ
criterion where Dynare's and JDyn's QZ round differently). The higher-order tests need the k-order library
(`deps/korder/build.sh`) and skip with a warning without it.

## Implemented and checked against Dynare

| Area | Check | Worst difference |
|---|---|---|
| Calibration of `.mod` files with MATLAB code | the 157 files of Dynare's test suite in pydyn's corpus: `M_.params`, `M_.Sigma_e` (and `oo_.steady_state` for 67 of them) | 157 identical, 0 different, 0 refused (rtol 1e-10) |
| MATLAB semantics | 82 cases of pydyn plus 376 new ones against Octave (122 from the second review: logical class, shapes, empty arrays, deletion, `any`/`all`/`unique` with NaN, `scalar^matrix`, `sprintf` of char data) | all match or are refused (60 of the 376 refused) |
| Steady state | analytical blocks, `_steadystate.m` files (interpreted), helper functions, numerical solve (Dynare's `solve_algo=4`: Dulmage-Mendelsohn blocks and trust region; variables of identically satisfied equations stay at `initval`) | exact; 1.6e-6 on collard, where Dynare stops its solver early (JDyn's residual 1.8e-15) |
| First-order decision rules | collard, rbc_occbin, schorfheide (ghx, ghu, order_var) | 8e-15 |
| IRFs, parameter loops | schorfheide IRFs; 6 parameter sets in one load; nk_baseline with Julia or `.m` steady state | 5e-16 (IRFs), 4e-14 (rules), 3e-9 (nk_baseline) |
| Shipped examples (first order) | gali_2015, schorfheide_2000, bkk_1992, rbc_occbin, aguiar_gopinath_2007_trend, nk_baseline, collard_* | 1e-14 to 4e-10 |
| Moments (order 1) | theoretical (plain, HP, band-pass, unit roots), variance decompositions (incl. correlated shocks, conditional, measurement error), simulated moments on Dynare's draws | 3e-13 (theoretical), 2e-10 (simulated skewness) |
| Higher-order decision rules (Dynare's k-order library through `ccall`) | collard (order 2 and 3), aux_a (order 2): `ghx ghu ghxx ghxu ghuu ghs2 ghxxx ghxxu ghxuu ghuuu ghxss ghuss`, folded tensors | 1e-13 (order 3), 7e-13 (aux_a) |
| Simulation at orders 2 and 3, with and without pruning; generalised IRFs on Dynare's draws | collard at order 2 and 3, aux_a at order 2; irf.m of Dynare with fixed seed | 5e-15 (paths), 1.3e-15 (IRFs) |
| Moments at order 2 | theoretical (plain: mean correction; pruned state space) incl. variance and conditional decompositions with measurement error; simulated moments of order-2 paths | 6e-13 (theoretical), 2e-10 (simulated skewness) |
| Shipped examples at order 2 | the five examples whose `.mod` asks for order 2: `ghxx ghxu ghuu ghs2` | 7e-14 |
| Auxiliary and model-local variables, `loglinear` | against Dynare | 3e-14, 5e-13 |
| OccBin: piecewise-linear simulation (one and two constraints), `simul_*` options, error flags 310-313, periodic solutions, `error_bind`/`error_relax` | rbc_occbin (169 shock periods, 2 constraints), borrcon_sim/curb, occ_borrcon (two shocks; maxit sweeps, curb_retrench, endo_init, exo_pos, max/reset check-ahead, init_regime, init_binding_indicator), occ_rbc_cyc/occ_rbc1/occ_rbc_err (52 + 24 cycling cases): paths, regime histories and error flags | 3e-15 (paths), regime histories and flags identical |
| OccBin IRFs (`occbin.irf`: both signs, all shocks, `shocksize`, `check_ahead_periods`, `maxit`) | occ_borrcon (2 shocks), occ_rbc_cyc (2 constraints) | 9e-16 |
| OccBin likelihoods (`src/occbin_filter.jl`): inversion filter (IVF_core/IVF_posterior) and piecewise Kalman filter (kalman_update_engine, algo_1) | test/ref_occbin: occest_nkm (ZLB NK model, 24 variables, 4 parameter vectors), occest_rbc (two constraints, 3 vectors), occest_nkms (3 vectors), occest_nkmopt (filter_use_relaxation, max_kalman_iterations, periodic_solution, check_ahead_periods): log posterior, per-period likelihood, filtered states, shocks, regime histories | IVF 1.8e-11, PKF 3.5e-10 (relative 2.3e-13); states 4e-12; regimes identical except three rounding ties (see OccBin) |
| OccBin smoother (occbin.DSGE_smoother: linear, real-time and iterated smoothers, conditional smoother, warning 320) and inversion-filter smoother; `occbin.shock_decomposition` (decomp, wdecomp, init_decomp, linear ones), squeeze | occest_nkm, occest_rbc (2 iterations) | 1e-12 (smoothed variables and shocks), 2e-15 (decompositions) |
| OccBin estimation: Hessian and Laplace at Dynare's mode (mode_compute=4, nkms); posterior mode from the prior mean (slow test); Metropolis-Hastings end to end; `dynare()` estimation(smoother) into oo_ | occest_nkms_mode, occest_nkm_smoother | Hessian 9e-10 relative, Laplace 1515.166990 to 2e-7; mode (csminwel, 18 iterations, 30 s; `JDYN_SLOW_TESTS=1`) within 2.2e-4 posterior std, log posterior 3.6e-8; oo_ smoother fields 1e-10 |
| Estimation, first order (`src/priors.jl`, `kalman.jl`, `estimation.jl`) | prior log density on all families (est_priors); log likelihood and log prior at 13 reference files (est_schorfheide, est_nk v0-v9, est_ur v0-v6: missing data, measurement errors incl. correlated, first_obs/presample, prefilter, lik_init 1-3, diffuse and univariate filters, trends); Hessian, Laplace approximation, csminwel modes, log posterior at Dynare's 40 stored draws | 1.4e-11 (log likelihood), 6e-13 (log prior), 3.6e-10 (log posterior at the draws), 6e-6 (Hessian, relative), 2e-8 (mode value), 5e-5 std (mode location) |
| Kalman smoother (`src/smoother.jl`): `calib_smoother`, smoother at calibration, posterior mode/mean/median, ML mode, prior mode/mean, or a given theta; multivariate and univariate, diffuse (`kalman_algo` 3/4) and stationary, missing data, measurement errors (correlated too), trends, prefilter, loglinear, filtered/updated variables, `filter_step_ahead`, state uncertainty, `filter_initial_state` | 14 smo_* cases (both defect modes), sd_*, est_* start points, Smets-Wouters-scale sw_* (4 cases) | 1.4e-14 (smo_*), 5.8e-14 (sd_*, est_*), 5e-7 absolute on values of order 100 (SW, diffuse), 1.8e-8 relative (SW, stationary) |
| Shock decompositions (`src/shock_decomposition.jl`): `shock_decomposition`, `initial_condition_decomposition`, groups, `init2shocks`, `realtime_shock_decomposition` (pooled, vintage, conditional, forecast), `squeeze`, plotted numbers (yoy/aoa, diff, flip, crop, screening) as tables | smo_*, sd_nk, sd_ur references | 1e-12 or better |
| Forecasts (`src/forecast.jl`): `forecast` after stoch_simul (orders 1-3, histval, var lists, `conf_sig`, measurement-error bands) and after estimation (from the smoother at the mode; prefilter, observation trends, diffuse filter); `conditional_forecast` with calibration, posterior mode, prior mean; `plot_conditional_forecast` as a table | test/forecast: fc_cal.mod (4 variants, run in place by `dynare()`), est_fc_sch (loglinear, at Dynare's mode), est_fc_nk (prefilter, measurement error), est_fc_ur (unit root, trends) | 1e-11 or better (fc_cal); 1e-10 (estimation cases; trends 1e-9). Intervals from random draws (order 2, conditional forecasts with free shocks) match exactly when Dynare's draws are replayed |
| Optimisers (`src/optimizers.jl`): `mode_compute` 5 (newrat), 7 (fminsearch), 8 (Dynare's simplex), 9 (CMA-ES) | test/optim: est_opt_nk, est_opt_sch (test/test_optimizers.jl) | simplex: Dynare's iterates (same mode, iteration and function counts on est_opt_nk); newrat: first step to 1e-5 relative, mode within 3e-6 of Dynare's log posterior (0.003 posterior std); fminsearch 6e-10, CMA-ES 2e-11 in log posterior |
| Estimation at order 2: nonlinear filters (`src/octave_rng.jl`, `nonlinear_filters.jl`, `nonlinear_estimation.jl`): `filter_algorithm` sis, apf, cpf, gf, nlkf with their resampling, proposal/distribution approximation, pruning, initialization 1 and 3 options; mode finding (csminwel, simplex `mode_compute=8`, ...), Hessian, Metropolis-Hastings, `oo_` as Dynare | Dynare 7.2 under Octave on the RBC model of tests/particle (test/test_nonlinear_estimation.jl, references in test/ref_nonlinear): per-period log likelihoods on Dynare's reduced form, 35 filter configurations at 3 parameter vectors, seeds 1..30, simplex path (same calls, same mode), whole `estimation` runs (`oo_`) | 4e-10 per period (reduced form given), 7e-10 relative to sum\|lik\| (own solution), Octave RNG bit for bit, simplex iterates identical, `oo_` 5e-8 |
| Perfect foresight: stacked Newton (initval/endval/histval, `shocks`, `mshocks`, `endogenous_terminal_period`) | 8 models of pydyn's pf_* set, Dynare's `perfect_foresight_rbc` | 1e-13 absolute (1e-14 relative), same Newton iteration count |
| Perfect foresight: purely backward, purely forward and static models; `model(linear)` and `linear_approximation` | 5 references | 1e-15 (period by period), 7e-15 (linear) |
| Perfect foresight: homotopy (3 size/step options, linearization and marginal-linearization fallbacks) | 5 references | 2e-15 relative (levels of 1e7: 7e-8 absolute); same number of homotopy iterations and completion share |
| Perfect foresight: LMMCP (`[mcp=...]` tag and `⟂` syntax) | zero lower bound, irreversible investment (lower and upper bound) | 1e-16 and 3e-14 |
| Perfect foresight with expectation errors (`learnt_in` shocks, mshocks and endval; unanticipated shock sequence; datafile) | `pf_pfwee`, `pf_surprise`, Dynare's `perfect_foresight_expectation_errors` | 7e-15, 1e-14, 2e-14 |

Speed of one `set_param!` + steady state + first-order solve: 0.04-0.08 ms (15 variables), 0.4 ms
(28 variables, Julia steady-state function), 1.4 ms (42 variables, numerical steady state), 4 to 6 times
faster than pydyn on the same machine. An interpreted `_steadystate.m` costs about 3 ms per call.

## Not implemented yet

- Theoretical moments at order 3 and filtered (HP, band-pass) theoretical moments at order 2 (refused, as in Dynare);
  loglinear models above order 1 (refused, as in Dynare); unit roots with pruning at order 2 (refused, as Dynare).
- Estimation beyond second order (order 3, `gmf`, `nonlinear_filter_initialization=2`, residual/multinomial resampling, prefilter/trends/missing data with particle filters: refused); Bayesian smoothed distributions over posterior draws (see the smoother section).
- OccBin: forecasts (`occbin.forecast`), particle filters, the loss-function regime guess, `init_regime_history` /
  `init_binding_indicator` of the filters, `smoother_first_period_occbin_update` other than 1 (refused); the Bayesian
  smoother over posterior draws. OccBin figures are not drawn: the plotted content is available as tables (see below).
- Plotting beyond first-order IRFs (OccBin, smoother, decompositions, ...). First-order IRF figures are
  done: Dynare's content rules, tidy CSV export, SVG built in, PNG/PDF through an optional Plots.jl
  extension (see docs/plotting.md).
- Optimal policy: not done are `ramsey_constraints` (Dynare's inequality constraints, `solve_algo=10`), perfect-foresight Ramsey
  welfare with `ramsey_policy` after `perfect_foresight_solver` beyond Dynare's backward recursion, `osr(analytic_derivation)`,
  `osr` with `opt_algo` 1, 2, 3 (MATLAB toolboxes), `stoch_simul` orders above 3 for Ramsey models, `mult_elimination` as a
  separate option (the preprocessor does it). All refuse with an error. Heterogeneous agents, `dseries`: refused with an error. Semi-structural toolbox: see "Semi-structural models" below for what is and is not done (`pac.estimate.*`, legacy `get_companion_matrix_legacy`, `pac.print`/`var_expectation.print`).

## Optimal policy (src/optimal_policy.jl, src/results_policy.jl; docs/RAMSEY_PROGRESS.md)

`ramsey_model` / `ramsey_policy` (instruments, planner_discount, the Ramsey steady state `dyn_ramsey_static` with multipliers found
by least squares, steady-state files, `histval`, several instruments, auxiliary variables), `evaluate_planner_objective` (orders 1
and 2, conditional welfare with multipliers at 0 and at their steady state, unconditional welfare, `u` and `histval`),
`discretionary_policy` (Dennis' algorithm) and `osr` (`osr_params`, `osr_params_bounds`, `optim_weights`, weighted variances or
minus the welfare; csminwel, fminsearch, simplex, CMA-ES). They run through `dynare()`; `ramsey_policy!`, `discretionary_policy!`,
`osr!`, `evaluate_planner_objective!` re-run them on a workspace, and `osr`, `discretionary_policy`, `evaluate_planner_objective`,
`ramsey_steady_state` work on a `Model`. test/test_optimal_policy.jl compares with Dynare 7.2 (Octave) on 18 Ramsey models, 5
discretionary models, 5 osr models and a 2-shock order-2 welfare: steady state, rules, welfare and osr objective to 1e-9 or better
(csminwel optima to the optimiser's tolerance). Dynare's order-2 welfare mixes the Kronecker order of its `Wyu` term (matters with
several shocks and states, and a nonzero first shock); JDyn corrects it by default and reproduces Dynare with
`replicate_dynare_bugs=true` (checked against Octave).

## OccBin

`Occbin(m)` builds the piecewise-linear solver from a model with an `occbin_constraints` block (reference regime:
all `occbin_X_bind` parameters at 0; it uses the parameters at construction time). `simulate(ob)` runs the file's
`shocks(surprise)` blocks with the options of its `occbin_setup` / `occbin_solver` statements (applied in statement
order, as Dynare does); `simulate(ob, shocks; kw...)` takes shocks and every `simul_*` option as keywords;
`irf_occbin(ob)` is `occbin.irf`; `irf(ob, impulse)` is impulse-minus-baseline. Failures throw `OccbinPeriodicError` (310),
`OccbinMaxitError` (311), `OccbinHorizonError` (312) or `OccbinLoopError` (313); `error_flag(e)` gives Dynare's code.
Plotted content: `irf_table(irf_occbin(ob))` (columns `shock, sign, variable, period, solution, value`, `solution` =
`piecewise` or `linear`), `irf_table(simulate(ob))` and `occbin_graph_table(ob, res)` (the variables and `noconstant`
option of the file's `occbin_graph`; shocks as `solution = "shock"`), written with `write_csv`.

Dynare 7.2 defects (see pydyn's docs/GAPS.md), corrected by default as pydyn does, reproduced with
`replicate_dynare_bugs=true` (throws `OccbinDynareBug`):
- `simul_check_ahead_periods` above `simul_max_check_ahead_periods`: Dynare's `solver.m` prints a message that uses an
  undefined variable (`simul`) and fails; JDyn lowers `check_ahead_periods` to the maximum, as the message says.
- `occbin.irf` on a model with several shocks when `shocks(surprise)` covers fewer of them: `occbin_setup` leaves
  `exo_pos` at the shocked variables while `irf` passes one column per shock, and Dynare fails (the reference run
  stores that failure). JDyn computes the responses for all shocks.

`OccbinResult.ss[t]` holds the rule of each shock period under its verified regime (Dynare's `ss` of occbin.solver).

Differences from pydyn: `simulate(ob)` keeps only the non-zero shock columns of `shocks(surprise)` and their
positions (`exo_pos`) as Dynare's `setup.m` does (pydyn used all columns; same paths); several `occbin_setup` /
`occbin_solver` statements are merged in order (pydyn kept the first `occbin_setup`); `smoother_*`-type options warn that the simulation does not use them
(pydyn ignored them silently); leads and lags of variables in a bind/relax condition are refused with an error, as in pydyn.
The constraint expressions are evaluated period by period from the preprocessor's JSON text (not vectorised MATLAB code).
Also run, without a Dynare reference: Dynare's `tests/occbin/model_irrcap_twoconstraints/dynrbc.mod` (lower-case
`steady_state`, model-local variables, `simul_periodic_solution`, `curb_retrench`, 71 iterations in one period) agrees with pydyn to 4e-16.

### OccBin likelihood, smoother and estimation

For a model with `occbin_constraints` Dynare uses OccBin's likelihood and smoother (options_.occbin.likelihood.status and
smoother.status are set by the constraints block), and so does JDyn: `Estimation(m; occbin=Dict(...))` uses the piecewise
Kalman filter, or the inversion filter with `occbin_setup(likelihood_inversion_filter)` (or the `occbin` keyword, which takes
the manual's names `likelihood_maxit` or options_ paths `likelihood.maxit`); `posterior_mode`, `hessian`, `laplace`,
`sample_posterior`, `estimate` and `dynare()` follow. `smooth` runs the OccBin smoother after the linear one (`res.occbin`:
`OccbinSmoothing` with the `OccbinSmootherResult` and the linear result); `occbin_shock_decomposition(m, res)` is
`occbin.shock_decomposition`. Lower level: `occbin_filter(est, theta)`, `inversion_filter`, `piecewise_kalman_filter`,
`occbin_smoother`, `regime_table` / `write_regime_table` / `occbin_write_regimes!`. Options: `occbin_filter_options(m)`
(Dynare's defaults, then what `occbin.setup` copies from each occbin_setup; options it drops are reported once, as pydyn).

- Rounding ties: in a few periods of the two-constraint model two self-consistent regimes have likelihoods within 2e-15
  of each other (the second constraint expected to bind until period 113 or 116, ...). Both programs combine the two
  densities (kalman_update_engine.m), so the likelihood is the same; the regime reported follows the strict `<` between the two
  values, i.e. rounding: JDyn differs from Dynare at rbc period 50 (vector 1 and smoother) and 53-54 (vector 2).
- csolve's random search (inversion filter, when Newton makes no progress) draws from the model's stream (`dynare_rng`).
- Dynare's `shock_decomposition` / `realtime_shock_decomposition` / `forecast` commands on an OccBin model use the linear
  rules with the OccBin-smoothed shocks (no OccBin branch in Dynare 7.2); JDyn's do the same, without a Dynare reference.
- Dynare 7.2 defect: `occbin_write_regimes(periods=...)` labels the rows with the periods but fills them with the first
  regimes; corrected by default, reproduced with `options_.replicate_dynare_bugs`. The results.jl prefix table named the
  statement `occbin_write_regimes` (the preprocessor's name is `occbin_write_regimes_xls`), so its options went to
  options_.write_regimes; fixed.
- Initial checks of Dynare enforced: prefilter, univariate filters (PKF), measurement errors and shock count (IVF).

## Smoother and shock decompositions

`smooth(m, data=nothing; parameter_set=..., replicate_dynare_bugs=false, kw...)` is Dynare's `DsgeSmoother` as used by
`calib_smoother` and by `estimation(smoother)`; options come from the `.mod` file's statements and can be overridden by
keywords. `estimate()` runs it at the mode (or the start point with `mode_compute=0`) when `smoother` is set. Output follows
Dynare's layout (`SmoothedVariables`, `SmoothedShocks`, `SmoothedMeasurementErrors`, `UpdatedVariables`, `FilteredVariables`,
k-step arrays, `Variance`, `State_uncertainty`, `Init_State_Vector`). `check_smoother` tests the identities.

`dynare()` fills `oo_` for `estimation` and `calib_smoother` exactly as Dynare 7.2 does (`src/results_estimation.jl`; field
map and validation in docs/RESULTS.md); `estimation!(ws, var_list; options...)` and `calib_smoother!(ws, var_list; options...)`
re-run them on a workspace. Known gaps: `oo_.dr.{eigval, gx, Gy}` are not written; `optimization_info` iterations/funcCount are JDyn's own;
`selected_variables_only` and a smoother together with MCMC raise.
`shock_decomposition`, `initial_condition_decomposition`, `realtime_shock_decomposition`, `group_shocks`, `squeeze_shock_decomposition`
and `decomposition_plot_data` / `decomposition_table` (CSV through `write_csv`) give the figures' content as tables.

- Dynare 7.2 defects (multivariate diffuse smoother with a diffuse period lacking data or with F_inf = 0; the same case for
  the univariate State_uncertainty; `filter_initial_state` centring) are corrected by default; `replicate_dynare_bugs=true`
  reproduces Dynare's output. Both modes are tested against the references (corrected equals Dynare's own univariate output).
  Dynare's realtime decomposition starting at period 2 is kept in both modes.
- Univariate smoother with correlated measurement errors: supported (augmented state); validated only against the multivariate
  smoother, not a Dynare reference.
- Smets-Wouters scale (sw_* references): see the timing lines of test/test_smoother.jl output (about 0.02 s for the stationary
  cases including the first-order solve; diffuse cases use the univariate filter and match both Dynare algorithms).
- Refused with an error: `heteroskedastic_filter`, `smoother_redux`, `noconstant`, `partial_information`,
  `kalman_algo=5`, `lik_init` 4/5, `Harvey_scale_factor=0`, `filter_initial_state` with the diffuse filter. Variable lists of
  decomposition statements are ignored with a warning; the drawn-figure references (drawn_*) are not compared.
- `Init_State_uncertainty` is trimmed to the state (Dynare leaves it untrimmed under the expanded state).

## Estimation: scope and limits

- Supported: `varobs`, one `estimated_params` block (plus `estimated_params_init`, `_bounds`, `use_calibration`),
  `observation_trends`, data from `.m`/`.csv`/`.mat`/matrix, `first_obs`, `nobs`, `presample`, `prefilter`,
  `loglinear`/`logdata`, `lik_init` 1-3, `diffuse_filter`, `kalman_algo` 0-4, `qz_criterium`, `riccati_tol`,
  `mode_compute` 0, 4 (csminwel), 5 (newrat), 7 (fminsearch), 8 (simplex), 9 (CMA-ES), `forecast`, `mh_replic`/`mh_nblck`/`mh_jscale`/`mh_drop`
  (random-walk Metropolis-Hastings), Laplace and modified-harmonic-mean marginal densities, posterior moments.
  `estimate(m)` runs what the `.mod` file's `estimation` statement asks. Unsupported options raise an
  `EstimationError` (or warn once with `allow_unsupported=true`); other `mode_compute` values (1, 3, 12, 13: MATLAB toolboxes;
  2, 6, 10, 101, 102: Dynare's own, not ported yet) raise an `EstimationError`, as do newrat's `optim('Hessian', 0 or 2)`.
- Not implemented: forecasts over the posterior draws (`forecast` with `mh_replic > 0`, oo_.PointForecast / MeanForecast:
  refused), posterior IRFs, recursive estimation, other samplers, orders above 1,
  particle filters, more than one `estimated_params` block.
- The sampler is checked against Dynare's target density (40 stored draws) and statistically (2 x 4000 draws:
  means within 0.17 posterior std of Dynare's 2 x 20000, modified harmonic mean within 0.19); it is not
  draw-for-draw because Octave's random numbers cannot be reproduced.
- Differences from pydyn: csminwel instead of scipy optimisers; correlated measurement errors in the univariate
  filter (pydyn refuses them); Blanchard-Kahn failures carry Dynare's penalty.
- Known difference: a root within rounding of `qz_criterium` (Schorfheide with rho = 0.999999, Dynare info 3) is
  classified as stable by JDyn's QZ; marked as broken in the tests.

## Forecasts and optimisers

- `forecast(m; periods, conf_sig, var_list, order, ...)`, `forecast(smoother_result, m; periods, ...)`, `conditional_forecast(m; paths,
  controlled_varexo, parameter_set, ...)`, `conditional_forecast_table`; in `dynare()` the `forecast`, `conditional_forecast_paths`,
  `conditional_forecast` and `plot_conditional_forecast` statements run in place (`forecast!`, `conditional_forecast!` re-run them).
  `estimate()` returns the forecast of the `forecast` option in `EstimationResult.forecast` (Dynare's `oo_.forecast`; with
  `mode_compute=0` only together with `smoother`, as Dynare).
- Random numbers: order-2/3 forecast intervals and conditional forecasts with free shocks use Julia's generator (`rng`); `draws=`
  replays Dynare's (the tests record them in Octave after `set_dynare_seed` and match exactly). Quantiles follow MATLAB's (method 5).
- Dynare 7.2 defects corrected by default, reproduced with `replicate_dynare_bugs=true`: after estimation with `prefilter` or
  `observation_trends`, Dynare adds the data mean / trend to the mean forecast of the observed variables but not to HPDinf/HPDsup;
  with prefilter it subtracts the steady state of the variable at the observed variable's position in var_list, and fails when an
  observed variable is missing from var_list.
- Not implemented: `varexo_det` forecasts (refused), forecasts over posterior draws, OccBin forecasts.
- newrat: the gradient steps are chosen so that the posterior moves by 1e-5, so the 1e-11 rounding differences between JDyn's and
  Dynare's likelihoods change the gradient by about 1e-6 relative and the univariate steps of `mr_gstep` (whose second differences
  are dominated by rounding in both programs) after the first iteration: the first step matches, the iterates then differ, the
  optimum agrees to newrat's tolerance. With univariate filters JDyn's outer-product Hessian uses per-period contributions
  (Dynare: per observation), so the path differs further there.
- simplex: comparisons only, so Dynare's iterates are reproduced (est_opt_nk: same 3609 iterations and 5988 evaluations), except where
  two vertices tie to rounding (est_opt_sch from the prior means: both programs stall far from the mode, at different points).
- fminsearch (7) is MATLAB's algorithm; Dynare under Octave calls Octave's different `fminsearch`, so only the optimum is compared.
  CMA-ES (9) uses cmaes.m's strategy parameters, boundary handling and stopping rules (simplified restart/stagnation logic).

## Perfect foresight

`perfect_foresight(m)` (src/perfect_foresight.jl) runs what `perfect_foresight_setup` and
`perfect_foresight_solver` (or the `..._with_expectation_errors_` pair) do in the .mod file; it reads
`initval`, `endval`, `steady`, `histval`, `shocks`, `mshocks`, `shocks(learnt_in=...)`, `endval(learnt_in=...)`
and the options of both commands, and returns a `PerfectForesightResult` whose `endo_simul` (n_endo x
maximum_lag + periods + maximum_lead) and `exo_simul` (rows x n_exo) have Dynare's `oo_.endo_simul` /
`oo_.exo_simul` layout, with `iterations`, `max_residual` (`oo_.deterministic_simulation.error`) and `status`.
`JDyn.load` does not run the simulation. Keywords override the file (`periods`, `tolf`, `exo`, `steady`, ...).

- Defaults follow Dynare: `tolf` 1e-5 when neither the file nor the call gives one (pydyn defaults to 1e-9).
  Newton iterates and the iteration count are then Dynare's own; pass `tolf=1e-12` for tighter paths.
- Every `stack_solve_algo` is solved by Newton with a sparse LU (Dynare's algorithm 0, UMFPACK); a warning says so.
  Dynare's other linear solvers (LBJ, GMRES, bicgstab) give the same Newton step up to round-off or their own
  iterative tolerance, which JDyn does not reproduce. `robust_lin_solve`, bytecode, block and `solve_algo` are warned about.
- Refused with an error: `perfect_foresight_controlled_paths`, `shock_paths` blocks, `initval_file`/`histval_file`, dates in
  `periods`/`learnt_in`, `lmmcp` together with `model(linear)`, deterministic exogenous variables (`varexo_det`), marginal
  linearization with expectation errors. `shocks(surprise)` blocks are OccBin's (Dynare's perfect foresight does not read
  them) and are ignored here too.
- Stacked evaluation: one sparse pattern per number of periods, Jacobian filled in place, the factorisation's
  symbolic analysis reused between Newton steps. 28 variables x 200 periods: 6 Newton steps in 0.04 s;
  54 variables: 0.06 s; 142 variables: 0.15 s (best of 10, with `tolf=1e-9`, 2 cores shared with other jobs).
  pydyn on the same machine: 0.04, 0.09, 0.31 s.
- Where JDyn differs from pydyn: Dynare's default `tolf` (above); the lmmcp step for 100 or more unknowns is a sparse
  QR least-squares solve as in MATLAB's backslash (pydyn: normal equations); `endval(learnt_in=1)` with `add` or
  `multiply` is refused rather than reading the last information set (the preprocessor turns `learnt_in=1` into plain blocks).

## Behaviour to know

- Orders 2 and 3 need `deps/korder/libjdyn_korder.so` (see README; `JDYN_KORDER_LIB` overrides the path); `k_order`
  raises `KOrderError` if it is missing or if the loaded model has fewer derivative orders than asked for
  (`JDyn.load(file; output="third")`). The library keeps process-wide state: calls are serialised with a lock.

- `.mod` files: MATLAB code that could change the model and cannot be followed makes loading fail; code
  that cannot change it is skipped with one warning (`model.matlab_skipped` lists it).
- Options the code does not act on produce a warning (once per model), never a silent change.
- Options persist from one statement to the next as in Dynare's `options_`: `stoch_simul(hp_filter=1600)` followed by
  `stoch_simul(irf=3)` keeps the filter (the preprocessor writes an explicit `order` for each `stoch_simul`).
- `set_shock_stderr!` changes the variance only (covariances stay, as Dynare's `set_shock_stderr_value`); `set_params!` is
  all-or-nothing; `load(file; extra_args=["-DA=1"])` passes options to the preprocessor.
- `homotopy_setup(from_initval_to_endval)` with `steady(homotopy_mode=...)` gives the steady state at the `endval` exogenous
  values (solved directly, not stepped). The smoother and `estimate` refuse a stochastically singular model as Dynare does
  (set `use_univariate_smoother_if_singularity_is_detected` for the smoother).
- A parameter that no equation uses may stay unset (NaN): `k_order` accepts it; `irf_figures` warns when the file asks for order > 1.
- The 7.2 preprocessor refuses `use_dll` with Julia output: JDyn drops that option (it only concerns
  MATLAB compilation) and preprocesses again.
- The 7.2 preprocessor's Julia output writes `normcdf(mu, sigma, x)` (reversed argument order) and
  chained comparisons `a < b < c` that Julia would read differently from MATLAB; JDyn corrects both.
- Where MATLAB and Octave differ, the interpreter follows MATLAB; `isoctave` is false. Two places where it is more
  permissive than Octave on purpose: a deletion subscript that spans its dimension counts as a colon
  (`A(1:end,2) = []`, `A(1,2) = []` on a row), and `A(1:end) = []` style forms follow MATLAB.
- `steady_state_model` is evaluated in a bare Julia module (`steady_state_module`, src/model.jl): only the operators
  the preprocessor writes and MATLAB's functions (`round` half away from zero, `mod`/`rem` with MATLAB's conventions,
  `sign`, `fix`, `max`/`min` ignoring NaN, `x^y`, ...) exist there. A name it does not have (`exit`, any other Julia
  function, a MATLAB function not listed) sends the block to the interpreter, which resolves or refuses it.
  Non-ASCII characters in string literals are kept (MATLAB: UTF-16 code units; the rest of a line is read as ASCII).
- Julia exceptions raised inside an interpreter builtin (or inside a MATLAB `try`) are MATLAB errors: a `try/catch`
  catches them and the message says what Julia reported.
- JDyn fixes several MATLAB-semantics bugs that pydyn has (`strrep` with overlapping matches,
  `strsplit` with repeated delimiters, `sprintf('%.3d')`, `num2str(x, fmt)` blanks).

## Open checks

- Homotopy can accept a wrong terminal steady state on badly scaled models: Dynare's own defect (reference-environment.md),
  reproduced because the iterates must match Dynare's; `endval_steady_nocheck`, `steady_tolf` and the residual check at the
  terminal state are the guards.
- Files of Dynare's test suite outside pydyn's corpus have not been compared: in particular a file that
  pydyn refuses but JDyn loads has no Dynare reference yet.
- `options_.qz_criterium` set by MATLAB code and a `qz_criterium` statement option: the statement option
  wins regardless of order in the file.


## Semi-structural models (VAR, trend component, var_expectation, PAC; src/pac.jl)

Statements `var_model`, `trend_component_model`, `var_expectation_model`, `pac_model`, `pac_target_info` and the model-block terms
`var_expectation(...)`, `pac_expectation(...)`, `pac_target_nonstationary(...)` (also with MCE) load. The preprocessor's Julia output
already substitutes the auxiliary variables and the h/expectation parameters into the model; what only its MATLAB output holds
(`M_.var.<name>`, `M_.trend_component.<name>`, `M_.var_expectation.<name>`, `M_.pac.<name>`, `varmatrices.m`,
`trend_component_ar_a0.m`, `pac_target_coefficients.m`) is read from a second preprocessor run in MATLAB mode that `run_preprocessor`
makes when the file has these statements (kept under `<model>/model/semi/`). `M_.var`, `M_.pac`, ... then have Dynare's layout in
`ws.M_` and in the MATLAB lines of the file.

Run-time functions, ported from Dynare 7.2's `matlab/`: `get_companion_matrix` (VAR with and without constant, structural VAR,
trend component; outputs `A0, A0star, AR, B`; `oo_.var.<n>.CompanionMatrix`, `oo_.trend_component.<n>.CompanionMatrix`,
`M_.<type>.<n>.isconstant` and `list_of_variables_in_companion_var`), `<fname>.varmatrices`, `<fname>.trend_component_ar_a0`,
`<fname>.pac_target_coefficients`, `get_aux_variable_id`, `get_difference_order`, `isauxiliary`, `var_expectation.initialize` / `.update`
(all horizons and `time_shift`), `pac.initialize`, `pac.update.expectation` / `.equation` (h vectors for `ll`, `dd`, `dl`, VAR with or without
constant, trend component, components of `pac_target_info`, growth-neutrality corrections with `additive`, `optim_additive`,
`non_optimizing_behaviour` and the bgp flags), `pac.mce.parameters`, `pac.bgp.set` / `.get`. Three-dimensional arrays exist only as
results of these functions (`AR(:,:,i)` indexing, `index3` in src/pac.jl).

| Check | What | Worst difference |
|---|---|---|
| `M_.params` after the update functions, companion matrices, eigenvalues of the MCE models | 32 fixtures cut from Dynare's tests (`test/ref_pac`, `test/test_pac.jl`): var-expectations 1, 1-with-time-shift-a, 2, 3, 3-with-time-shift, 5, 6, 9, 11, 12; pac var-1, 3, 6, 8, 9, 10e, 12 (6 files); trend-component 1, 2, 10, 19 (2), 19-growth-lin-comb, 28, 1-mce, 2-mce | 1e-9 relative (parameters, in practice 1e-15), 1e-10 (companion matrices), 1e-6 (eigenvalue moduli) |
| The files' own checks (AR matrices, companion matrices, expectation weights) | trend-component-and-var-models (tcm1, tcm3, vm2 ...), var-expectations 1, 11, 1-with-time-shift-a, 3 | pass |

Not implemented (refused or skipped with a clear message): `pac.estimate.nls` / `.iterative_ols` / `.init` (they need dseries and a
nonlinear least-squares estimator), `pac.print` / `var_expectation.print` (they write `.inc` files and a dseries `evaluate.m`; JDyn does nothing, so the
`substitution.mod` tests, which include those files, fail in the preprocessor), `get_companion_matrix_legacy` / `get_ar_ec_matrices`
(legacy matrices from the model Jacobian), `pac.update.parameters` / `var_expectation.update_parameters` called directly, `pac.check`.
Dynare 7.2 quirk reproduced: `var_expectation.update_parameters` uses `expr.params` (parameter indices) as coefficients.
Dynare 7.2 defect: `pac.mce.parameters` with a parameter-valued bgp flag in `optim_additive` uses an undefined variable; JDyn raises that error.
