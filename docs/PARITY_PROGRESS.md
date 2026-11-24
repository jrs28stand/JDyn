# Parity bundle (pydyn features missing in JDyn): progress log

Branch `parity` (base main ce58c84). References: Dynare 7.2 under Octave 8.4; generators in `test/ref_parity/`.

| # | Item | Status |
|---|------|--------|
| 1 | steady-state `solve_algo` 9, 13, 14 | done: `src/steady_algos.jl`, `test/test_parity_steady.jl`; iterates equal Dynare's to <1e-13 relative |
| 2 | homotopy stepping, `save_params_and_steady_state` | done: `src/homotopy.jl`, `test/test_parity_homotopy.jl`; 12 Dynare runs (modes 1/2/3, from_initval_to_endval, failing and halving mode-3 paths): steady states 1e-15 relative (with solve_algo=9), same number of solves/failed steps; file identical to Dynare's |
| 3 | pruned order-3 theoretical moments | |
| 4 | `varexo_det` | |
| 5 | `epilogue` | |
| 6 | `external_function` | |
| 7 | heteroskedastic shocks / filter | |
| 8 | `M_.Correlation_matrix` | |

## Notes
- Explicit `solve_algo` 9/13/14 return Dynare's iterate (no polishing to machine precision, no fallback; failure is a `SteadyStateError`,
  as Dynare's "Impossible to find the steady state"). The default (4) keeps JDyn's polish. `steady(maxit=, tolx=)` are now honoured.
- Other `solve_algo` values (0-3, 5-8, 10-12) still warn and use 4.
- Homotopy: `steady(homotopy_mode=1|2|3, homotopy_steps=n)` runs at statement time (`_steady_homotopy!`, called from `_exec_statement!`), from
  the previous steady state (a pending lazy `steady` is resolved first); each step calls `steady_state` with the auxiliary variables reset
  (Dynare's `set_auxiliary_variables`). Failure is an error ("Homotopy step failed") unless `homotopy_force_continue`. `homotopy_mode=0` no
  longer moves the parameters (Dynare only warns). With the default solver JDyn's steps are polished (see item 1), so only
  `solve_algo` 9/13/14 reproduce Dynare's path to 1e-15; the default agrees to the solver tolerance.
- `save_params_and_steady_state(file)` writes `name %.16g` for parameters, all endogenous (auxiliary variables included) and exogenous variables.
