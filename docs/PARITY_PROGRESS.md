# Parity bundle (pydyn features missing in JDyn): progress log

Branch `parity` (base main ce58c84). References: Dynare 7.2 under Octave 8.4; generators in `test/ref_parity/`.

| # | Item | Status |
|---|------|--------|
| 1 | steady-state `solve_algo` 9, 13, 14 | done: `src/steady_algos.jl`, `test/test_parity_steady.jl`; iterates equal Dynare's to <1e-13 relative |
| 2 | homotopy stepping, `save_params_and_steady_state` | |
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
