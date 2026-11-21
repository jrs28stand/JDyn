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
| Steady state | analytical blocks, `_steadystate.m` files (interpreted), helper functions, numerical solve | exact; 1.6e-6 on collard, where Dynare stops its solver early (JDyn's residual 1.8e-15) |
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
| Estimation, first order (`src/priors.jl`, `kalman.jl`, `estimation.jl`) | prior log density on all families (est_priors); log likelihood and log prior at 13 reference files (est_schorfheide, est_nk v0-v9, est_ur v0-v6: missing data, measurement errors incl. correlated, first_obs/presample, prefilter, lik_init 1-3, diffuse and univariate filters, trends); Hessian, Laplace approximation, csminwel modes, log posterior at Dynare's 40 stored draws | 1.4e-11 (log likelihood), 6e-13 (log prior), 3.6e-10 (log posterior at the draws), 6e-6 (Hessian, relative), 2e-8 (mode value), 5e-5 std (mode location) |
| Kalman smoother (`src/smoother.jl`): `calib_smoother`, smoother at calibration, posterior mode/mean/median, ML mode, prior mode/mean, or a given theta; multivariate and univariate, diffuse (`kalman_algo` 3/4) and stationary, missing data, measurement errors (correlated too), trends, prefilter, loglinear, filtered/updated variables, `filter_step_ahead`, state uncertainty, `filter_initial_state` | 14 smo_* cases (both defect modes), sd_*, est_* start points, Smets-Wouters-scale sw_* (4 cases) | 1.4e-14 (smo_*), 5.8e-14 (sd_*, est_*), 5e-7 absolute on values of order 100 (SW, diffuse), 1.8e-8 relative (SW, stationary) |
| Shock decompositions (`src/shock_decomposition.jl`): `shock_decomposition`, `initial_condition_decomposition`, groups, `init2shocks`, `realtime_shock_decomposition` (pooled, vintage, conditional, forecast), `squeeze`, plotted numbers (yoy/aoa, diff, flip, crop, screening) as tables | smo_*, sd_nk, sd_ur references | 1e-12 or better |
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
- Estimation beyond first order; Bayesian smoothed distributions over posterior draws (see the smoother section).
- OccBin estimation, smoothing, forecasts, shock decompositions and `occbin_write_regimes`: not implemented (the
  `smoother_*`, `likelihood_*`, `filter_*` options of `occbin_setup`/`occbin_solver` give a warning; `occbin_write_regimes`
  is reported as an unhandled statement). OccBin figures themselves are not drawn: the plotted content is
  available as tables (see below).
- Plotting beyond first-order IRFs (OccBin, smoother, decompositions, ...). First-order IRF figures are
  done: Dynare's content rules, tidy CSV export, SVG built in, PNG/PDF through an optional Plots.jl
  extension (see docs/plotting.md).
- Optimal policy (Ramsey, OSR), heterogeneous agents, PAC/VAR models, `dseries`: refused with an error.

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

Differences from pydyn: `simulate(ob)` keeps only the non-zero shock columns of `shocks(surprise)` and their
positions (`exo_pos`) as Dynare's `setup.m` does (pydyn used all columns; same paths); several `occbin_setup` /
`occbin_solver` statements are merged in order (pydyn kept the first `occbin_setup`); `smoother_*`-type options warn
(pydyn ignored them silently); leads and lags of variables in a bind/relax condition are refused with an error, as in pydyn.
The constraint expressions are evaluated period by period from the preprocessor's JSON text (not vectorised MATLAB code).
Also run, without a Dynare reference: Dynare's `tests/occbin/model_irrcap_twoconstraints/dynrbc.mod` (lower-case
`steady_state`, model-local variables, `simul_periodic_solution`, `curb_retrench`, 71 iterations in one period) agrees with pydyn to 4e-16.

## Smoother and shock decompositions

`smooth(m, data=nothing; parameter_set=..., replicate_dynare_bugs=false, kw...)` is Dynare's `DsgeSmoother` as used by
`calib_smoother` and by `estimation(smoother)`; options come from the `.mod` file's statements and can be overridden by
keywords. `estimate()` runs it at the mode (or the start point with `mode_compute=0`) when `smoother` is set. Output follows
Dynare's layout (`SmoothedVariables`, `SmoothedShocks`, `SmoothedMeasurementErrors`, `UpdatedVariables`, `FilteredVariables`,
k-step arrays, `Variance`, `State_uncertainty`, `Init_State_Vector`). `check_smoother` tests the identities.
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
- Refused with an error: `heteroskedastic_filter`, `smoother_redux`, OccBin smoothing, `noconstant`, `partial_information`,
  `kalman_algo=5`, `lik_init` 4/5, `Harvey_scale_factor=0`, `filter_initial_state` with the diffuse filter. Variable lists of
  decomposition statements are ignored with a warning; the drawn-figure references (drawn_*) are not compared.
- `Init_State_uncertainty` is trimmed to the state (Dynare leaves it untrimmed under the expanded state).

## Estimation: scope and limits

- Supported: `varobs`, one `estimated_params` block (plus `estimated_params_init`, `_bounds`, `use_calibration`),
  `observation_trends`, data from `.m`/`.csv`/`.mat`/matrix, `first_obs`, `nobs`, `presample`, `prefilter`,
  `loglinear`/`logdata`, `lik_init` 1-3, `diffuse_filter`, `kalman_algo` 0-4, `qz_criterium`, `riccati_tol`,
  `mode_compute` 0 and 4 (csminwel, ported from Dynare), `mh_replic`/`mh_nblck`/`mh_jscale`/`mh_drop`
  (random-walk Metropolis-Hastings), Laplace and modified-harmonic-mean marginal densities, posterior moments.
  `estimate(m)` runs what the `.mod` file's `estimation` statement asks. Unsupported options raise an
  `EstimationError` (or warn once with `allow_unsupported=true`); other `mode_compute` values are replaced by csminwel with a warning.
- Not implemented: forecasts, posterior IRFs, recursive estimation, other samplers, orders above 1,
  particle filters, more than one `estimated_params` block.
- The sampler is checked against Dynare's target density (40 stored draws) and statistically (2 x 4000 draws:
  means within 0.17 posterior std of Dynare's 2 x 20000, modified harmonic mean within 0.19); it is not
  draw-for-draw because Octave's random numbers cannot be reproduced.
- Differences from pydyn: csminwel instead of scipy optimisers; correlated measurement errors in the univariate
  filter (pydyn refuses them); Blanchard-Kahn failures carry Dynare's penalty.
- Known difference: a root within rounding of `qz_criterium` (Schorfheide with rho = 0.999999, Dynare info 3) is
  classified as stable by JDyn's QZ; marked as broken in the tests.

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
