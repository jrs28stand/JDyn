# Status (2026-11-21)

Reference: Dynare 7.2 under GNU Octave 8.4 (the same reference results as pydyn). All figures below are
from `test/runtests.jl` (1,739 checks pass, 0 fail; 3 skipped: perfect foresight, and references
without IRFs); the perfect foresight rows are from `test/test_perfect_foresight.jl` (307 checks, this branch).

## Implemented and checked against Dynare

| Area | Check | Worst difference |
|---|---|---|
| Calibration of `.mod` files with MATLAB code | the 157 files of Dynare's test suite in pydyn's corpus: `M_.params`, `M_.Sigma_e` (and `oo_.steady_state` for 67 of them) | 157 identical, 0 different, 0 refused (rtol 1e-10) |
| MATLAB semantics | 82 cases of pydyn plus 254 new ones against Octave | all match or are refused (61 refused) |
| Steady state | analytical blocks, `_steadystate.m` files (interpreted), helper functions, numerical solve | exact; 1.6e-6 on collard, where Dynare stops its solver early (JDyn's residual 1.8e-15) |
| First-order decision rules | collard, rbc_occbin, schorfheide (ghx, ghu, order_var) | 8e-15 |
| IRFs, parameter loops | schorfheide IRFs; 6 parameter sets in one load; nk_baseline with Julia or `.m` steady state | 5e-16 (IRFs), 4e-14 (rules), 3e-9 (nk_baseline) |
| Shipped examples (first order) | gali_2015, schorfheide_2000, bkk_1992, rbc_occbin, aguiar_gopinath_2007_trend, nk_baseline, collard_* | 1e-14 to 4e-10 |
| Moments (order 1) | theoretical (plain, HP, band-pass, unit roots), variance decompositions (incl. correlated shocks, conditional, measurement error), simulated moments on Dynare's draws | 3e-13 (theoretical), 2e-10 (simulated skewness) |
| Auxiliary and model-local variables, `loglinear` | against Dynare | 3e-14, 5e-13 |
| Perfect foresight: stacked Newton (initval/endval/histval, `shocks`, `mshocks`, `endogenous_terminal_period`) | 8 models of pydyn's pf_* set, Dynare's `perfect_foresight_rbc` | 1e-13 absolute (1e-14 relative), same Newton iteration count |
| Perfect foresight: purely backward, purely forward and static models; `model(linear)` and `linear_approximation` | 5 references | 1e-15 (period by period), 7e-15 (linear) |
| Perfect foresight: homotopy (3 size/step options, linearization and marginal-linearization fallbacks) | 5 references | 2e-15 relative (levels of 1e7: 7e-8 absolute); same number of homotopy iterations and completion share |
| Perfect foresight: LMMCP (`[mcp=...]` tag and `⟂` syntax) | zero lower bound, irreversible investment (lower and upper bound) | 1e-16 and 3e-14 |
| Perfect foresight with expectation errors (`learnt_in` shocks, mshocks and endval; unanticipated shock sequence; datafile) | `pf_pfwee`, `pf_surprise`, Dynare's `perfect_foresight_expectation_errors` | 7e-15, 1e-14, 2e-14 |

Speed of one `set_param!` + steady state + first-order solve: 0.04-0.08 ms (15 variables), 0.4 ms
(28 variables, Julia steady-state function), 1.4 ms (42 variables, numerical steady state), 4 to 6 times
faster than pydyn on the same machine. An interpreted `_steadystate.m` costs about 3 ms per call.

## Not implemented yet

- Orders 2 and 3 (plan: Dynare's C++ k-order library through `ccall`, fed with the preprocessor's
  G1-G3; the preprocessor output and `simulate` at orders 2-3 are already in place).
- OccBin, Kalman filter and smoother, shock decompositions, estimation.
- Plotting beyond first-order IRFs (OccBin, smoother, decompositions, ...). First-order IRF figures are
  done: Dynare's content rules, tidy CSV export, SVG built in, PNG/PDF through an optional Plots.jl
  extension (see docs/plotting.md).
- Optimal policy (Ramsey, OSR), heterogeneous agents, PAC/VAR models, `dseries`: refused with an error.
- Theoretical moments at order 2 (refused with an error).

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

- `.mod` files: MATLAB code that could change the model and cannot be followed makes loading fail; code
  that cannot change it is skipped with one warning (`model.matlab_skipped` lists it).
- Options the code does not act on produce a warning (once per model), never a silent change.
- The 7.2 preprocessor refuses `use_dll` with Julia output: JDyn drops that option (it only concerns
  MATLAB compilation) and preprocesses again.
- The 7.2 preprocessor's Julia output writes `normcdf(mu, sigma, x)` (reversed argument order) and
  chained comparisons `a < b < c` that Julia would read differently from MATLAB; JDyn corrects both.
- Where MATLAB and Octave differ, the interpreter follows MATLAB; `isoctave` is false.
- JDyn fixes several MATLAB-semantics bugs that pydyn has (`strrep` with overlapping matches,
  `strsplit` with repeated delimiters, `sprintf('%.3d')`, `num2str(x, fmt)` blanks).

## Open checks

- Files of Dynare's test suite outside pydyn's corpus have not been compared: in particular a file that
  pydyn refuses but JDyn loads has no Dynare reference yet.
- `options_.qz_criterium` set by MATLAB code and a `qz_criterium` statement option: the statement option
  wins regardless of order in the file.
