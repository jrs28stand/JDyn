# Parity bundle (pydyn features missing in JDyn): progress log

Branch `parity` (base main ce58c84). References: Dynare 7.2 under Octave 8.4; generators in `test/ref_parity/`.

| # | Item | Status |
|---|------|--------|
| 1 | steady-state `solve_algo` 9, 13, 14 | done: `src/steady_algos.jl`, `test/test_parity_steady.jl`; iterates equal Dynare's to <1e-13 relative |
| 2 | homotopy stepping, `save_params_and_steady_state` | done: `src/homotopy.jl`, `test/test_parity_homotopy.jl`; 12 Dynare runs (modes 1/2/3, from_initval_to_endval, failing and halving mode-3 paths): steady states 1e-15 relative (with solve_algo=9), same number of solves/failed steps; file identical to Dynare's |
| 3 | pruned order-3 theoretical moments | done: `src/moments_order3.jl`, `test/test_parity_moments3.jl`; 3 models x 2 cases x (JDyn, Dynare decision rules), max scaled diff 4.5e-15 with Dynare's rules; no variance decomposition at order 3 (as Dynare); order 3 without `pruning` and with filters refused (ArgumentError) |
| 4 | `varexo_det` | done: `src/varexo_det.jl`, `test/test_parity_varexo_det.jl`; ghud/ghxud/ghuud/ghudud (orders 1-2) and simult_varexo_det paths 5e-14, forecast with exo_det paths (order 1 3e-15, order 2 with replayed draws 6e-15); perfect foresight, order 3 and pruning refused |
| 5 | `epilogue` | done as a Julia API (src/epilogue.jl); not hooked into the MATLAB interpreter or shock decompositions. Dynare numbers: bit-identical with `replicate_dynare_bugs=true`, <1e-6 without (static: 1e-14) |
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
- varexo_det: the generated functions take `x = [exo; exo_det]` (`_xfull` appends the steady-state values of the varexo_det variables), the derivative
  layout of the stochastic solvers has the varexo_det columns filtered out (`dynamic_sparse_derivs`; `dynamic_full_derivs` keeps them).
  `shocks` on varexo_det variables go to `m.cache[:det_shocks_exo_det]` (not to `m.det_shocks`). `forecast` with exo_det: `ForecastResult.Exogenous`.
  `M_.exo_det_nbr`, `M_.exo_det_names`, `oo_.exo_det_steady_state` are set. Not done: `stoch_simul` itself ignores varexo_det (as Dynare),
  `forecast` after `estimation`, simulation of `simult_` with exo_det outside forecast, estimation with varexo_det (not tested).
- Epilogue: `epilogue(m)` reads the block from the macro-expanded .mod (the JSON has none); `epilogue_dynamic(ep, series; params, replicate_dynare_bugs)`,
  `epilogue_static`, `epilogue_from_paths`. Dynare's generated epilogue_dynamic rounds every assigned series to 6 dp (dseries `sprintf('%f')`) and
  feeds the rounded values to later definitions: a Dynare defect, so exact values are the default; `replicate_dynare_bugs=true` reproduces Dynare
  bit for bit (also the lead-from-`t-lead` quirk of non-recursive definitions). Not done: calling `<mod>.epilogue_dynamic(params, ds)` from the
  MATLAB interpreter, `shock_decomposition(..., with_epilogue)`.
