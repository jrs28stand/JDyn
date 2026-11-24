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

## Not done / next

- `ramst_datafile`: `perfect_foresight_setup(datafile=...)` reads the data with `dseries(file)` (histvalf_initvalf.m,
  `oo_.initval_series`), and the file is a `.mat` written by a data script (`ramst_initval_file_data.m`) that the
  acceptance harness skips. Left for the dates/dseries work.
- `perfect_foresight_controlled_paths` (9 tests) and `shock_paths` (3 tests): still refused.

## Evidence

Acceptance (deterministic_simulations): pfwee, pfwee_constant_sim_length, pfwee_learnt_in, pfwee_learnt_in_bare_1st_period,
Irreversible_investment, Irreversible_investment_leadlag: CHECK_FAILED -> OK (the files' own strict-equality checks pass);
pfwee_homotopy_linearization OK_WARN -> OK. Solow_no_varexo unchanged (OK_WARN) after an endval fix.
Full test suite: all files pass (test_perfect_foresight.jl: the expectation that robust_lin_solve is warned about was
changed to markowitz only, since it is now implemented).
ramst_datafile: main now has dseries; the needed pieces are `dseries(file)` for the .mat/.m datafile and
`oo_.initval_series` in make_ex_/make_y_ (see histvalf_initvalf.m); the data .mat is produced by a script the harness skips.
