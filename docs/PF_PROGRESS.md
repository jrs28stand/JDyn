# Perfect-foresight gaps: progress log

Branch `pf-fixes` (base main 65ed314). Acceptance files: Dynare 7.2 `tests/deterministic_simulations`.

## Done

- **Function forms** (src/pf_functions.jl, new, self-contained): `make_ex_`, `make_y_`, `perfect_foresight_setup`,
  `perfect_foresight_solver`, `perfect_foresight_with_expectation_errors_setup` / `_solver` and
  `evaluate_steady_state(ys, exo, M_, options_[, flag])` as MATLAB builtins (`bi!`, as pac.jl does). They read
  their inputs from the `oo_` they are given (endo_simul / exo_simul as hand-edited, steady states, `oo_.pfwee`),
  options from `options_` of the workspace (MATLAB lines applied) plus fields the caller's copy changed.
  The workspace behind a session is found through `_pf_register_host!` (called by `dynare()`).
- **Expectation errors in place** now follow Dynare: the `..._setup` statement stores `oo_.pfwee.{terminal_info,
  shocks_info}` and the initial guess in `oo_.endo_simul`/`oo_.exo_simul`; the `..._solver` statement (and the function
  form) solves from `oo_` (hand edits of the guess are honoured). If the setup fails (dates, ...) or `oo_.pfwee`
  is absent the old replay path is used, as before.
- `endval` (and `endval(learnt_in=1)`, which the preprocessor makes a plain block) now writes `oo_.initial_steady_state`,
  `oo_.steady_state` / `oo_.exo_steady_state` (auxiliary variables included) as Dynare's driver does; the tests
  take `orig_steady_state = oo_.steady_state` after it.
- **robust_lin_solve / allow_nonfinite_values** (`_pf_lin_solve_robust` in perfect_foresight.jl): `A\b` (with a regularised
  refinement and dense least-squares fallback for singular matrices, after pydyn's `_backslash`), BiCGSTAB, GMRES, `pinv`.
- src/matlab/session.jl: one condition added in `run_stmt!` (a name in `SOLVE_FUNCS` that has a builtin is executed, not
  skipped), needed for `oo_.steady_state = evaluate_steady_state(...)`.

## Phase 2 (branch pf-fixes-2)

- **perfect_foresight_controlled_paths** (perfect_foresight.jl): `PFCP`/`_pf_cp_by_period` (controlled_paths_by_period.m),
  `_pf_cp_substitute` (controlled_paths_substitute_stacked_jacobian.m: the Jacobian columns of the exogenised variables'
  equations are swapped with the derivatives w.r.t. the endogenised exogenous variables), solved exogenous variables
  returned in `oo_.exo_simul`; linear case (`sim1_linear`), homotopy (the controlled values follow the share, the guess of
  the solved exogenous variables is `exobase`), linearization fallback (also on the exogenous path), and
  expectation errors (`learnt_in`, per-information-period controlled paths, increment test). All stack_solve_algo values
  use the same Newton solver, as before. In pf_functions.jl the function forms read the same data.
- `_pf_steady_at` rejects a "solution" more than 1e8 times larger than the guess (Dynare's `steady_solve_algo=13`, used by
  pfwee_homotopy_linearization_condpaths, avoids this runaway; JDyn's single steady solver accepted values with
  1/C -> 0 satisfying the absolute tolerance). The caller then retries from the initial steady state, as Dynare does.
- **initval_file / perfect_foresight_setup(datafile=...)** (`_pf_initval_series`, port of histvalf_initvalf.m for numeric
  first_obs/last_obs/nobs/first_simulation_period/last_simulation_period, data from .m/.mat/.csv through `ds_ctor`;
  auxiliary variable of an exogenous lead is rebuilt, other auxiliary variables refused). `_pf_make_paths` follows
  make_y_.m/make_ex_.m with `oo_.initval_series` (also used by the function forms). `options_.periods` comes from the file
  when the setup does not give it. The harness (test/acceptance/worker.jl) now runs `*_data.m` scripts, as Dynare's test
  order does.
- Statements `perfect_foresight_controlled_paths` and `initval_file` no longer produce the "statements not handled" warning.

### Not done (refused with a reason)

- `perfect_foresight_with_expectation_errors` with `homotopy_marginal_linearization_fallback`
  (pfwee_homotopy_marglin_condpaths, pfwee_homotopy_marginal_linearization): needs oo_.deterministic_simulation.sim1/sim2
  carried across information periods.
- `shock_paths` (3 tests): the preprocessor writes invalid JSON for the block (stray quote), and the `table`/database
  evaluation functions are not available in julia mode.
- `initval_file(series=...)`, dates-valued first/last_obs, `histval_file`, spreadsheets (.xls/.xlsx: dseries cannot read them),
  `oo_.initval_series` as a MATLAB-visible field (sim_exo_lead_lag_initvalf, ramst_initval_file_with_dseries).
- ramst_initval_file: its `if ispc` block holds initval_file statements the replay cannot skip (and .xlsx), so it is refused.

### Evidence (acceptance)

controlled paths: linear_state_space_arma_condpaths, homotopy_linearization_condpaths, homotopy_marglinearization_condpaths,
pfwee_homotopy_linearization_condpaths, ramst_pf_controlled_paths (+_lbj, _ssa7, _check), ramst_pfwee_controlled_paths:
REFUSED -> OK/OK_WARN (the files' own strict equality checks pass). ramst_datafile: CHECK_FAILED -> OK.
