# Parity bundle (pydyn features missing in JDyn): progress log

Branch `parity` (base main ce58c84). References: Dynare 7.2 under Octave 8.4; generators in `test/ref_parity/`.

| # | Item | Status |
|---|------|--------|
| 1 | steady-state `solve_algo` 9, 13, 14 | done: `src/steady_algos.jl`, `test/test_parity_steady.jl`; iterates equal Dynare's to <1e-13 relative |
| 2 | homotopy stepping, `save_params_and_steady_state` | done: `src/homotopy.jl`, `test/test_parity_homotopy.jl`; 12 Dynare runs (modes 1/2/3, from_initval_to_endval, failing and halving mode-3 paths): steady states 1e-15 relative (with solve_algo=9), same number of solves/failed steps; file identical to Dynare's |
| 3 | pruned order-3 theoretical moments | done: `src/moments_order3.jl`, `test/test_parity_moments3.jl`; 3 models x 2 cases x (JDyn, Dynare decision rules), max scaled diff 4.5e-15 with Dynare's rules; no variance decomposition at order 3 (as Dynare); order 3 without `pruning` and with filters refused (ArgumentError) |
| 4 | `varexo_det` | done: `src/varexo_det.jl`, `test/test_parity_varexo_det.jl`; ghud/ghxud/ghuud/ghudud (orders 1-2) and simult_varexo_det paths 5e-14, forecast with exo_det paths (order 1 3e-15, order 2 with replayed draws 6e-15); perfect foresight, order 3 and pruning refused |
| 5 | `epilogue` | done as a Julia API (src/epilogue.jl); not hooked into the MATLAB interpreter or shock decompositions. Dynare numbers: bit-identical with `replicate_dynare_bugs=true`, <1e-6 without (static: 1e-14) |
| 6 | `external_function` | done: orders 1-2, .m files (also +pkg), Julia callables, numerical derivatives as jacob_element/hess_element; decision rules agree with Dynare to 1.8e-15 (scaled) |
| 7 | heteroskedastic shocks / filter | done: likelihood (multivariate, univariate, diffuse, missing data) and multivariate smoother; max rel diff vs Dynare 4e-14 |
| 8 | `M_.Correlation_matrix` | done (exact match with Dynare 7.2 on corr8.mod) |

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
- `M_.Correlation_matrix`: Dynare 7.2's preprocessor writes it from the `corr` statements of `shocks` blocks, calibrated ones included (the brief's
  "identity unless estimated" is not what 7.2 does), not from `var a, b = cov` covariances; `shocks(overwrite)` resets it to the identity;
  estimated correlations replace their entries. Kept in `m.options["Correlation_matrix"]` (`declared_correlation_matrix(m)`); JDyn used to derive it
  from Sigma_e. test_results_estimation's smo_me case now also compares Correlation_matrix (it was excluded as a known deviation).
- Heteroskedastic filter: `heteroskedastic_shocks` blocks (src/heteroskedastic.jl: `heteroskedastic_arrays`, `heteroskedastic_qvec`, `heteroskedastic_rqr`) and
  the `heteroskedastic_filter` estimation option. The Kalman filters accept `QQ` as an (n, n, nobs + 1) path (`kalman.jl`, `_qq`; period t predicts with
  slice t + 1; no steady-state switch). The initial covariance and the diffuse setup use the constant Sigma_e. The smoother (multivariate only) takes
  the Q path (prediction slice t + 1, shocks slice t); with the univariate smoother it is refused. Periods given as dates are refused; correlated
  measurement errors with the univariate filter and the heteroskedastic filter are refused. Without the option the blocks are ignored, as in Dynare.
- external_function: the preprocessor's Julia output has MATLAB-style TEF lines (`[TEF_0, TEFD_0] = f(..)`, `jacob_element('f',2,{..})`) that do not parse as Julia;
  `load_model_functions` rewrites them (src/external.jl, `_rewrite_external`) to `_ext_*` calls of a per-model `ExternalFunctions` table. Julia callables come from
  `Model(...; functions=Dict(name => f))` (Float64 arguments; a function returning its own derivatives returns a tuple); otherwise `name.m` / `+pkg/name.m` next to the
  .mod runs in the MATLAB interpreter through `matlab_function_caller` (src/matlab/extern.jl, the only addition under src/matlab besides one line in session.jl).
  Order 3 is impossible (the preprocessor cannot differentiate external functions three times), as in Dynare. Not done: external functions in the planner objective
  of ramsey models (still refused), and perfect foresight with external functions is untested.
