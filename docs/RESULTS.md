# Results as in MATLAB: M_, options_ and oo_

Goal: after running a .mod file, results are where a Dynare user expects them (`oo_.endo_simul`,
`oo_.dr.ghx`, `oo_.irfs.y_e`, `M_.params`, `options_.periods`), in the same shapes, readable from Julia with
MATLAB's syntax and from the MATLAB lines of the .mod file.

## Usage (prototype)

```julia
using JDyn
@dynare "rbc.mod"                 # like MATLAB's `dynare rbc`: defines M_, options_, oo_
oo_.endo_simul                    # 3x202 Matrix, as Dynare
oo_.dr.ghx; oo_.irfs.y_e          # 1x40 row, as Dynare
M_.endo_names[1]; M_.params       # cell (column), n x 1 column vector
propertynames(oo_)                 # tab completion works on the structs

ws = dynare("rbc.mod")            # the same without defining globals: ws.M_, ws.options_, ws.oo_, ws.model
ws = dynare("rbc.mod"; shocks=Dict(1 => exo))   # reuse given shocks (e.g. Dynare's oo_.exo_simul) for stoch_simul #1
```

The structs are JDyn's `MStruct` (the interpreter's struct type, ordered fields) and `Cell`, so the very
same objects are what MATLAB code in the .mod file sees. Matrices keep MATLAB's shapes: column vectors are
n x 1, rows 1 x n, scalars `Float64`, logicals `Bool`, strings `String`, cells `Cell`. `display(oo_)` prints
MATLAB's struct summary (`endo_simul: 3x202 double`).

## Re-running without preprocessing

The workspace stays live: change `M_`, `options_` or `oo_` and call Dynare's commands again. Keyword
options are written into options_, as `stoch_simul(order=1, irf=20)` does in MATLAB, and persist.

```julia
@dynare "rbc.mod"
for rho in (0.5, 0.9, 0.95)
    set_param_value!("rho", rho)          # or M_.params[i] = rho
    stoch_simul!(; order=1, irf=20)       # new oo_.dr, oo_.irfs (about 0.3 s each on collard_2001)
    push!(irfs, copy(oo_.irfs.y_e))
end

x0 = copy(oo_.exo_simul)                  # after perfect_foresight_setup/solver
for s in (0.5, 1.0, 2.0)
    oo_.exo_simul .= s .* x0              # another shock path
    perfect_foresight_solver!()           # starts from oo_.endo_simul, as Dynare (about 0.2 s each)
    push!(paths, oo_.endo_simul[1, :])
end
```

Commands: `steady!`, `check!`, `stoch_simul!(var_list; shocks, rng, options...)`, `forecast!(var_list; draws, rng, options...)`,
`conditional_forecast!(; paths, draws, rng, options...)`, `perfect_foresight_setup!`,
`perfect_foresight_solver!`, `occbin_solver!`, `set_param_value!`. Each takes the workspace as first
argument, or acts on the last one `dynare`/`@dynare` made (as MATLAB's commands act on the globals).
Parameters a steady-state file or `steady_state_model` block sets go back to `M_.params`, as in Dynare.

The existing API (`load`, `first_order`, `perfect_foresight`, `estimate`, ...) is unchanged: `dynare()` is a
layer over it, and `ws.model` is the usual `Model`.

## Random draws and seeds

Every random draw in JDyn comes from one stream per workspace (per `Model`), `JDyn.dynare_rng(ws)`, a Julia
`Xoshiro` generator. It is seeded as Dynare seeds its global stream: with 'default' (seed 0) when `dynare()` starts,
so **a run is reproducible by default**, and consecutive commands continue the same stream (as MATLAB's global stream).

```julia
ws = dynare("model.mod")               # seed 0: the same shocks, GIRF draws, ... on every run
ws = dynare("model.mod"; seed=123)     # as set_dynare_seed(123) before the first statement
set_dynare_seed!(ws, 123)              # re-seed the workspace; options_.DynareRandomStreams.{algo,seed} follow
stoch_simul!(ws; periods=1000)         # draws what the first stoch_simul of dynare(...; seed=123) draws
ws.options_.DynareRandomStreams.seed = 7   # also works: the stream is re-seeded at the next draw
```

In the .mod file `set_dynare_seed(n)`, `set_dynare_seed('default')`, `set_dynare_seed('clock')`, `set_dynare_seed('reset')`
and `set_dynare_seed(algo, n)` (algo recorded, JDyn's generator is always Xoshiro) are MATLAB builtins of the interpreter
and write `options_.DynareRandomStreams`; `'clock'` stores the seed it chose so the run can be repeated with it.
An explicit `rng=` (to `dynare`, `stoch_simul!`, `irf`, `estimate`, ...) or `shocks=` replaces the stream.

Used by: `stoch_simul(periods>0)` shocks, the generalised IRFs at order > 1, the random initial guesses of the
steady-state solver (`solve_randomize_initial_guess`), `estimate` (csminwel's random Hessian perturbation; the
Metropolis-Hastings chains are seeded from the model's seed, `Xoshiro(seed * 7919 + chain)`, as before with seed 0).

**The draws differ from MATLAB's and Octave's.** Dynare's `set_dynare_seed(n)` selects a Mersenne-twister (or Octave
`state`) stream that JDyn does not reproduce; the same seed gives the same numbers on every JDyn run, never MATLAB's.
To compare simulations with Dynare use its `oo_.exo_simul` (`dynare(file; shocks=Dict(1 => exo))`), not the seed.
`rand`/`randn` lines in a .mod file are still refused by the interpreter.

### Octave's generators for the particle filters

Dynare reseeds its particle filters with the 'default' seed (0) before every likelihood evaluation, so the likelihood is a
deterministic function of the parameters. JDyn does the same; the generator is chosen per model: `JDyn.set_rng_kind!(m, :octave)`,
`dynare(file; particle_rng=:octave)` or `Estimation(m; particle_rng=:octave)` give Octave's own Mersenne-twister/ziggurat numbers
(`JDyn.OctaveRNG`, bit for bit; `particle_seed=` changes the seed), so `estimation(order=2, ...)` reproduces Dynare-under-Octave
likelihoods; the default `:julia` uses Xoshiro with the same seeding. Everything else (`stoch_simul` shocks, GIRFs, MH chains)
keeps using the model's Xoshiro stream. There is no smoother or forecast with a particle filter (Dynare has none).

## Performance notes

* Filtered theoretical moments (`hp_filter`, `bandpass_filter` with theoretical moments) transform the spectral
  density to autocovariances with an FFT (`src/fft.jl`; radix 2 and Bluestein for other grid lengths). It was a direct
  O(grid^2) sum: with `filtered_theoretical_moments_grid=8192` Dynare's `tests/moments/example1_hp_test.mod` took 835 s;
  it takes about 6 s now (the first call in a session adds Julia's compilation, see below).
* The first `dynare()` call in a Julia session compiles most of JDyn; later calls are fast (collard: 0.5 s).

## Checked against Dynare 7.2 (test/test_results.jl)

| | |
|---|---|
| perfect_foresight_rbc: `oo_.endo_simul` (3x202) | 2.8e-14 |
| `oo_.exo_simul` (202x1), `M_.params`, `oo_.steady_state` | exact / 1e-15 |
| collard_2001 `stoch_simul(periods=2000)` with Dynare's shocks: `oo_.endo_simul` (6x2000), `oo_.dr.gh*` | 2e-6 / 1.4e-5 |
| test/forecast/fc_cal.mod (4 variants: levels, measurement errors, loglinear, order 2): `oo_.forecast` and `oo_.conditional_forecast` after each statement, run in place (test/test_forecast.jl) | < 1e-11 (deterministic statements; stochastic ones with Dynare's draws replayed) |

The collard gap is Dynare's: its steady state stops at a residual of 1e-6 (default tolf), JDyn's is exact.
At Dynare's steady state the decision rules agree to 1e-12 (test_korder.jl).

## What fills what

| command | Dynare fills | prototype |
|---|---|---|
| (driver) | `M_` (~80 fields) | names (+long/tex), counts, maximum_lag/lead (+orig, endo/exo), lead_lag_incidence, nstatic/npred/nboth/nfwrd/nspred/nsfwrd/ndynamic, params, Sigma_e, Correlation_matrix, H, aux_vars, endo_histval |
| (driver) | `options_` (~600 defaults) | JDyn's small default table + every option the .mod sets (the preprocessor names them by their options_ path) + MATLAB assignments |
| steady | steady_state | yes |
| check | dr.eigval | yes |
| stoch_simul | dr (ys, order_var, inv_order_var, state_var, gh*), irfs, endo_simul, exo_simul, var_list, mean, var, autocorr, gamma_y, variance_decomposition, skewness, kurtosis | yes (orders 1-3; moments orders 1-2) |
| perfect_foresight_solver | endo_simul, exo_simul, deterministic_simulation.{status, iterations, error}, initial_steady_state | yes |
| occbin_solver | oo_.occbin.simul.{piecewise, linear, shocks_sequence, exo_pos, regime_history} | yes (regime_history: a 1 x T Cell of structs `regime1, regimestart1, ...`, MATLAB's struct array) |
| occbin_write_regimes | (an .xls file in Output/) | the same table as CSV in Output/<filename>.csv (`occbin_write_regimes!`) |
| estimation | prior, posterior.optimization, posterior_mode / mle_mode (+ `_std_at_mode`), MarginalDensity, posterior_mean/median/var/hpdinf/hpdsup/deciles/density, posterior.metropolis, Smoother, SmoothedVariables, UpdatedVariables, FilteredVariables(+KStepAhead), SmoothedShocks, SmoothedMeasurementErrors, dr (order_var ... edim), steady_state | yes, see "estimation and calib_smoother" below; `estimation!(ws; ...)` re-runs it |
| calib_smoother | the smoother fields above, prior (with an estimated_params block), dr, steady_state | yes (`calib_smoother!(ws; ...)` re-runs it) |
| forecast | forecast.{Mean, HPDinf, HPDsup, HPDinf_ME, HPDsup_ME} | yes (`forecast!`; after stoch_simul, orders 1-3; after estimation the `EstimationResult.forecast` field, copied into oo_ by `estimation`) |
| conditional_forecast_paths, conditional_forecast | conditional_forecast.{cond, uncond, controlled_exo_variables, controlled_variables, instruments, graph}; options_cond_fcst_ | yes (`conditional_forecast!`; estimated parameter sets read oo_.posterior_mode / _mean / _median / mle_mode) |
| plot_conditional_forecast | (figures) | the plotted numbers as a table (`conditional_forecast_table`) |
| ramsey_model, ramsey_policy | M_.ramsey_orig_endo_nbr/ramsey_orig_eq_nbr, aux_vars of type 6 (multipliers MULT_i), options_.instruments, options_.ramsey_policy, oo_.dr (steady state with multipliers), oo_.steady_state, then the stoch_simul fields and planner_objective_value | yes (`ramsey_policy!`) |
| evaluate_planner_objective | planner_objective_value.{unconditional, conditional} (a number, or under Ramsey `conditional.zero_initial_multiplier` and `.steady_initial_multiplier`) | yes (`evaluate_planner_objective!`) |
| discretionary_policy | oo_.dr (order 1, `order_var` ... `ghu`), the stoch_simul fields, planner_objective_value | yes (`discretionary_policy!`) |
| osr | M_.params (optimum), M_.osr.{param_names, param_indices}, oo_.osr.{objective_function, optim_params, error_indicator, optimization_info.{exitflag, iterations, funcCount, message}}, then stoch_simul at the optimum | yes (`osr!`) |
| identification, ... | | not supported by JDyn |

## estimation and calib_smoother (what Dynare 7.2 writes, and what JDyn writes)

Source: `estimation/dynare_estimation_1.m`, `dynare_estimation_init.m`, `display_estimation_results_table.m`,
`store_smoother_results.m`, `evaluate_smoother.m` (calib_smoother), `posterior_moments.m`. Implemented in
`src/results_estimation.jl`; `dynare()` runs both commands where they stand, and `estimation!(ws, var_list; options...)`
and `calib_smoother!(ws, var_list; options...)` do the same from Julia (options as for `stoch_simul!`, read from `options_`).
Names in `oo_.posterior_*` follow `display_estimation_results_table.m`: `parameters.<name>`, `shocks_std.<exo>`,
`measurement_errors_std.<obs>`, `shocks_corr.<a>_<b>`, `measurement_errors_corr.<a>_<b>`. Values are scalars except
`posterior_deciles` (9x1) and `posterior_density` (512x2: grid, density).

| Field | Shape | Written when | JDyn |
|---|---|---|---|
| `oo_.prior.{mean, mode, variance, hyperparameters.first, .second}` | nx1, nx1, nxn, nx1, nx1 | estimated_params present (ML: mean/mode/variance are NaN/0/NaN) | yes |
| `oo_.posterior.optimization.{mode, Variance, log_density}` | nx1, nxn, 1x1 | mode_compute ~= 0 | yes |
| `oo_.posterior.optimization.optimization_info.{runtime, iterations, funcCount, exitflag, message}` | scalars, char | mode_compute ~= 0 | yes; iterations/funcCount/runtime are JDyn's own |
| `oo_.posterior_mode` / `oo_.mle_mode`, `_std_at_mode` | named scalars | mode_compute ~= 0 (std needs cova_compute) | yes |
| `oo_.MarginalDensity.LaplaceApproximation` | 1x1 | mode_compute ~= 0 with a Hessian | yes |
| `oo_.MarginalDensity.ModifiedHarmonicMean` | 1x1 | mh_replic > 0 | yes |
| `oo_.posterior_mean/median/var/hpdinf/hpdsup/deciles/density`, `oo_.posterior.metropolis.{mean, Variance}` | as above | mh_replic > 0 | yes (structure exact; values are a different chain) |
| `oo_.Smoother.{occbin, loglinear, SteadyState, TrendCoeffs, Trend, Variance, Constant, Init_State, Init_State_Vector, State_uncertainty, Init_State_uncertainty}` | see store_smoother_results.m | smoother (or calib_smoother) | yes (State_uncertainty only with smoothed_state_uncertainty) |
| `oo_.SmoothedVariables.<v>`, `UpdatedVariables.<v>`, `FilteredVariables.<v>` | nobs x 1 each (filtered: nobs+1 with filtered_vars) | smoother; filtered_vars or filter_step_ahead for the last two | yes, one field per endogenous variable in dr.order_var order |
| `oo_.FilteredVariablesKStepAhead`, `...Variances` | nk x endo x (nobs+max(nk)), nk x endo x endo x ... | filter_step_ahead (variances with filter_covariance) | yes |
| `oo_.SmoothedShocks.<e>`, `oo_.SmoothedMeasurementErrors.<y>` | nobs x 1 | smoother | yes |
| `oo_.dr.{order_var, inv_order_var (row), restrict_var_list, restrict_columns}` | | always | yes |
| `oo_.dr.{ys, ghx, ghu, state_var (row), sdim, edim}` | | after a solution | yes |
| `oo_.dr.{eigval, gx, Gy}` | | after a solution | no: Dynare's come from the QZ factorisation in a different ordering than ghx's rows |
| `oo_.steady_state` | | at the starting point (`x0`) of the estimation, or the calibration (calib_smoother) | yes |
| `oo_.gui.ran_estimation`, `ran_calib_smoother` | logical | | yes |
| `M_.params`, `M_.Sigma_e`, `M_.H`, `M_.Correlation_matrix` | | set to the posterior mode (mean when mh_replic > 0 and no mode) | yes (mode_compute=0: the starting values) |
| `options_.{nobs, first_obs, varobs_id, number_of_observed_variables, mh_jscale, filtered_vars, filter_step_ahead, nk, lik_init, varlist, parameter_set}` | | | yes. `first_obs` stays 1 after a smoother-only run (mode_compute=0), as in Dynare, whose early return skips the reset to NaN |

OccBin models (occbin_constraints): the likelihood is the piecewise Kalman filter or the inversion filter, the smoother
Dynare's occbin.DSGE_smoother (src/occbin_estimation.jl). In addition to the fields above (OccBin smoother values,
`oo_.Smoother.occbin = true`):

| Field | JDyn |
|---|---|
| `oo_.occbin.smoother.{error_flag, warning_flag, regime_history, realtime_regime_history}` | yes (regime histories as 1 x T+1 Cells of structs) |
| `oo_.occbin.smoother.{SmoothedVariables, UpdatedVariables, SmoothedShocks, alphahat, etahat, ahat, epsilonhat, SteadyState, trend_coeff, T0, R0, C0, simul.{piecewise, linear}}` | yes; T0/R0/C0 are the regime matrices of every period in decision-rule order (layout not compared with Dynare) |
| `oo_.occbin.linear_smoother.{SmoothedVariables, UpdatedVariables, SmoothedShocks, alphahat, etahat, ahat, SteadyState, T0, R0}` | yes (`aK`, `decomp`, `alphahat0`, `state_uncertainty0`, `regime_history_iter` not written) |
| when the OccBin smoother fails (error_flag ~= 0) | only oo_.occbin.smoother is written, as Dynare ("No results will be written to oo_") |

OccBin options come from the file's occbin_setup / occbin_solver statements and from `options_.occbin.*` (MATLAB lines,
keywords). Refused with OccBin: prefilter, univariate filters with the PKF, the diffuse filter, loglinear, filter_covariance,
smoothed_state_uncertainty, filter_step_ahead beyond 1, observation_trends in the smoother.

Not written, and refused where the option asks for them: `selected_variables_only`, `oo_.FilteredVariablesShockDecomposition`, `filter_covariance` / `smoothed_state_uncertainty` over posterior draws, `bayestopt_`/`estim_params_` (JDyn keeps
them in the `Estimation` object, not in the workspace), recursive estimation, the other samplers. `mh_replic > 0` with
`mode_compute=0` warns and is ignored, as Dynare returns before the sampler there.

Validation (Octave 8.4 running Dynare 7.2 on `test/ref_oo_estimation`, `test/test_results_estimation.jl`):
smoother-only, calib_smoother, no-smoother and measurement-error cases match to 1e-12 or better (oo_, M_.params, Sigma_e,
H); mode_compute=4 cases agree to the csminwel stopping point (about 1e-5 to 1e-9 on modes; layout identical); a 300-draw
chain agrees in field names and shapes; the kernel density agrees to 2e-15. Also compared once (not in the suite, they take
minutes): est_schorfheide with the smoother (loglinear, 1e-13) and two est_nk runs.

## Decisions for the full version

1. **Commands run in place.** The prototype loads the whole file, then runs the commands in order with
   the final calibration. Dynare runs each command where it stands, so a later `rho = 0.5; stoch_simul;`
   gets a second, different `oo_.irfs`, and MATLAB lines after a command can read `oo_`. The session already
   stops at every command (it does so to track `M_.params` and `oo_.steady_state`); the full version calls
   the command there and lets the interpreter's `oo_` be the MStruct, so lines like
   `y = oo_.endo_simul(1, :); plot(y)` or `ghx = oo_.dr.ghx;` work inside the .mod. `load()` keeps its
   declare-only behaviour.
2. **options_ defaults.** Dynare sets ~600 defaults in `default_option_values.m`. Proposal: run it once
   through JDyn's own MATLAB interpreter at build time (it is plain assignments) and store the result as a
   data file; options_ then has Dynare's full tree, and the interpreter no longer refuses reads of unset
   options.
3. **Random draws differ.** `stoch_simul(periods>0)` draws shocks with Julia's RNG, so simulated paths
   and moments are not MATLAB's. `shocks=` (or a seed option) is the way to reproduce a run.
4. **Workspace variables.** MATLAB's `dynare` also leaves the parameters and the .mod's own MATLAB
   variables in the base workspace. `ws.vars` holds them; `@dynare` could define them too (opt-in, to avoid
   clobbering the caller's names).
5. **Shapes, not types.** Counts are `Float64` (as MATLAB doubles); `M_.endo_names` is a column cell.
   Julia code that wants native types keeps using the existing API.

## Semi-structural objects (src/pac.jl)

`ws.M_.var.<name>`, `ws.M_.trend_component.<name>`, `ws.M_.var_expectation.<name>` and `ws.M_.pac.<name>` hold Dynare's fields
(from the preprocessor's driver.m: `eqtags`, `lhs`, `max_lag`, `rhs.vars_at_eq`, `h_param_indices`, `ec`, `ar`, `additive`, `components` (a Julia
`Vector{MStruct}`), `mce.alpha`, ...), plus what the run-time functions add (`isconstant`, `list_of_variables_in_companion_var`,
`model_consistent_expectations`, `additive.bgp`, ...). `ws.oo_.var.<name>.CompanionMatrix` and `ws.oo_.trend_component.<name>.CompanionMatrix`
are filled by `get_companion_matrix` / `pac.initialize` / `var_expectation.initialize`; `M_.params` carries the h / expectation parameters after
`pac.update.expectation` / `var_expectation.update`. (`oo_.var` is overwritten by a later `stoch_simul` as in Dynare.)
`oo_.var.<name>.ar` and `oo_.trend_component.<name>.ar` (legacy) are not filled.

## Posterior analyses over MCMC draws (src/mcmc_diagnostics.jl, posterior.jl, posterior_moments.jl, posterior_forecast.jl, posterior_stage.jl)

With `mh_replic > 0` `estimation` fills, as Dynare 7.2 does: `oo_.convergence` (`geweke(b).<SE_name>`, `raftery_lewis(b)`, Brooks-Gelman
series, inefficiency factors; skipped by `nodiagnostic`), `oo_.PosteriorIRF.dsge` (`bayesian_irf`), `oo_.PosteriorTheoreticalMoments.dsge`
(`moments_varendo`: covariance, `contemporeaneous_correlation`, correlation, VarianceDecomposition(ME), ConditionalVarianceDecomposition(ME)
with `Steps`), the posterior distributions of `SmoothedVariables`, `UpdatedVariables`, `SmoothedShocks`, `SmoothedMeasurementErrors`,
`Constant`, `FilteredVariables`, `Filtered_Variables_<k>_step_ahead`, `Smoother.{Trend, TrendCoeff, Init_State}`,
`FilteredVariablesKStepAhead(Variances)` (`smoother`, `filtered_vars`, `filter_step_ahead`), and `oo_.MeanForecast`, `PointForecast`,
`PointForecastME` (`forecast`). Dynare 7.2 quirks reproduced on purpose: see docs/POSTERIOR_PROGRESS.md.
