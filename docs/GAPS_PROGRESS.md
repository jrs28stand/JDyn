# Gap closure progress (JDyn vs pydyn), branch `gaps`

Validation reference: Dynare 7.2 under Octave 8.4. Nothing pushed or merged.

## Item 1: measurement-error covariances and correlations (done, mostly validated)
- `shocks` blocks accept `covariance` and `corr` between measurement errors. Results are in `M_.H` (off-diagonal) and `M_.Correlation_matrix_ME`.
- Covers the multivariate likelihood, smoother, forecasts, estimated ME correlations, and Dynare's info 44 (not PD) and info 72 (inconsistent) checks.
- Also added: `oo_.likelihood_at_initial_parameters` under `options_.debug`, and the `<fname>/Output/<fname>_mode.mat` writer.
- Agreement with Octave on the pydyn `me_cov` references: max difference 1.8e-12 (me_alg1/2/3, me_d3, me_lik2), 4e-17 (me_cal1/3).
- Not yet done:
  - Dynare's own `fs2000ns_*corr_ME*` tests, run end to end (needs `sed 's/filter_decomposition,//'` and the generated data files).
  - A diffuse univariate smoother with correlated ME (pydyn refuses it too).
  - A check_model-style ME consistency error.
  - Tests with references under `test/ref_gaps/`.

## Item 2: interpreter parity (in progress)
- New files: `src/matlab/nd.jl` (N-d arrays with Octave rules) and `src/matlab/rng.jl` (Octave-exact rand/randn/randi/randperm/rng/set_dynare_seed).
- `dynare(...; random=:octave)` makes the workspace stream an `OctaveRNG`, shared with the simulations. The default stays Xoshiro (`random=:julia`).
- Case-set status (match / refused / differ), last full run before the current fix batch:

| Set | Match | Refused | Differ |
|---|---|---|---|
| matlab_extra | 318 | 58 | 0 |
| pydyn_extra | 232 | 15 | 27 |
| pydyn_nd | 478 | 6 | 4 |
| pydyn_blk1 | 213 | 23 | 8 |
| pydyn_rng | 88 | 10 | 6 |

- A fix batch is on disk and being re-validated: indexing shapes, concatenation of empties, sprintf corners, strsplit, error/assert identifiers, flip/xor.
- Still open: the blk1 differs, the rng differs (73, 75, 81-84), struct-valued `rng` (76-80), a test file for the sets, an `addpath` helper check, and date helpers.
- Still to validate: `random=:octave` draws against an Octave run.

## Item 2 update
Final case-set status: matlab_extra 318/58/0, pydyn_extra 258/16/0, pydyn_nd 484/4/0, pydyn_blk1 218/23/3 (the 3 are the `d.freq()` member-call error and 2 harness cases now handled), pydyn_rng (`random=:octave`) 99/5/0 (match/refused/differ). Test: `test/test_gaps_interp.jl` (cases in `test/ref_gaps/cases`). `addpath` (function and command syntax) works. pydyn has no datenum/weekday helpers, so none were added. Not yet done: `random=:octave` simulation draws against an Octave run.

## Item 5 done: ramsey_constraints
`solve_algo=10` in the file now turns on lmmcp on the perfect foresight problem. The Ramsey steady state falls back to algorithm 4 when solve_algo is 10 or 11. Octave 8.4 test1.mod: max difference 3.6e-10. Test: `test/test_gaps_ramsey_constraints.jl`.

## Item 6 partly done
`data(file, first_obs, last_obs, nobs)` with dates (start date from INIT__/csv or `set_time`). Octave fs2000_data (nobs=40): alp 0.3570877 vs JDyn 0.3571060 (optimizer-level). `data(series=...)` is refused. trend_var and the log version: steady state, ghs2 and parameters equal the store; IRFs and means at order 2 are simulation-based in Dynare, so they cannot match exactly. Test: `test/test_gaps_data.jl`.

## Item 3 (backward models): implemented, validated on the Dynare test folders
Files: `src/backward.jl` (new), small hooks elsewhere (see "Shared files touched"). Julia API: `simul_backward_model(ws, T; innovations, initialconditions)`,
`simul_static_model`, `backward_model_irf`, `backward_model_forecast`, `set_historical_values`, `model_inversion` (all also MATLAB-callable from the .mod file,
as `simul_backward_nonlinear_model` / `simul_backward_linear_model` with Dynare's argument lists).
- Auxiliary variables (diff, lags above one, lagged exogenous variables) of the initial conditions are rebuilt as `dynamic_set_auxiliary_series` does (`_aux_series`).
- `options_.solve_algo` (4, 9, 13, 14; others refused), `options_.dynatol.*`, `options_.bnlms.set_dynare_seed_to_default`, `bnlms.innovation_distribution` are read.
- `isoctave` is true in a `random=:octave` workspace (the inversion tests have Octave tolerances), `oo_.bnlms.shocks`, `oo_.endo_simul`, `oo_.exo_simul` as Dynare.
- Interpreter additions needed by these tests: date literals (`2000Q1`, `1Y`) lex as `dates('..')` (before, every statement with a literal was silently skipped), `cumsum`/`cumprod`.
- Dynare 7.2 defects (corrected by default, reproduced under `replicate_dynare_bugs=true`):
  1. `dynare_solve` repairs a non-evaluable initial guess by redrawing `find(x)` (the NON-zero entries, NaN included) although its comment says the default ones.
     Default: only the non-finite entries are redrawn (when none is, the zero ones). Bug mode: Dynare's choice (with `random=:octave` the same draws).
  2. `backward_model.initialize` reads the history of the auxiliary variable of a lagged exogenous variable at `M_.exo_histval(M_.orig_endo_nbr+i)`, an element of another
     variable (`model-inversion/bk-7`: u1 instead of u3, so the initial u3 is 0 instead of -0.1). Default: the variable's own histval; bug mode: Dynare's element.
- Refused: forward-looking `model_inversion` (nk-1, nk-2: deterministic inversion through `perfect_foresight_solver_core`; pydyn refuses it too), `backward_model_irf` with deterministic
  shocks given as dseries, `simul_backward_model` for a maximum lag above one without auxiliary variables.

## Item 4 (extended path): order 0 and the stochastic full tree
File `src/extended_path.jl`; hook in `src/results.jl` (`extended_path` statement), `extended_path(ws, T; ...)` and the MATLAB-callable `extended_path(initial, T, exo, options_, M_, oo_)`.
- Order 0 as Dynare: stacked trust region (`ep.stack_solve_algo = 7`, `ep.solve_algo = 9`, sparse port of trust_region.m, same iterates), JDyn's PF core for
  `stack_solve_algo = 0`, linear models, purely backward models (`ep.periods = 1`) and lmmcp (`ep.solve_algo = 10`); homotopy method 2 on failure.
- Order > 0: full tree (`ep.stochastic.algo = 0`), Tensor-Gaussian-Quadrature and Unscented (also with no positive-variance shock), models with maximum lag and lead 1.
- `oo_.extended_path.status`, `oo_.endo_simul` (`n x (lag+T)`), `oo_.exo_simul`, `Simulated_time_series` (dseries) as Dynare.
- Refused: sparse tree (`tree=sparse`), hybrid (`hybrid_order > 1`), Stroud cubatures, `use_first_order_solution_as_initial_guess`, calibrated innovations, `extended_path_mc`, `ep.solve_algo = 11`.
- Agreement with Octave 8.4 (`random=:octave`): rbc2 (order 0 then 1) 3e-16, rbc_bytecode 6e-16, rbcii 1.4e-11, rbcii_MCP 1.4e-14, linearmodel1 2e-15, ar_backward exact;
  a custom run (order 2 unscented, order 2 and 3 tensor with 3 nodes on the rbc2 model) 1e-15.

## Older list (items 3-4 are now done, see above)
5. `ramsey_constraints` (still refused in `src/model.jl`).
6. `data(first_obs=<date>)`, and `trend_var`/`deflator` against the store.

## Shared files touched
(items 3-4) `src/JDyn.jl` (includes, exports), `src/results.jl` (extended_path branch, `periods` of extended_path not written to options_), `src/model.jl` (extended_path is a handled statement), `src/matlab/parser.jl` (date literals), `src/matlab/builtins.jl` (cumsum/cumprod).

`src/results.jl`, `src/results_estimation.jl`, `src/JDyn.jl`, `src/matlab/session.jl`, `src/seeds.jl`, `src/octave_rng.jl`, `src/model.jl`, `src/estimation.jl`, `src/nonlinear_estimation.jl`, `src/forecast.jl`, `src/smoother.jl`.
No existing test has been changed yet. The full suite has not been run yet.
