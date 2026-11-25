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

## Items 3-4: not started
3. Backward models (simul_backward_model, irf, forecast, set_historical_values, simul_static_model, model_inversion). Spec: pydyn `backward.py` and `results_bwd.py`.
4. Extended path order 0 and stochastic EP.
5. `ramsey_constraints` (still refused in `src/model.jl`).
6. `data(first_obs=<date>)`, and `trend_var`/`deflator` against the store.

## Shared files touched
`src/results.jl`, `src/results_estimation.jl`, `src/JDyn.jl`, `src/matlab/session.jl`, `src/seeds.jl`, `src/octave_rng.jl`, `src/model.jl`, `src/estimation.jl`, `src/nonlinear_estimation.jl`, `src/forecast.jl`, `src/smoother.jl`.
No existing test has been changed yet. The full suite has not been run yet.
