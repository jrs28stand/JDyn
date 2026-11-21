# Status (2026-11-21)

Reference: Dynare 7.2 under GNU Octave 8.4 (the same reference results as pydyn). All figures below are
from `test/runtests.jl` (2,596 checks pass, 0 fail; 3 skipped: perfect foresight, and references
without IRFs). The higher-order tests need the k-order library (`deps/korder/build.sh`) and skip with a
warning without it.

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
| Higher-order decision rules (Dynare's k-order library through `ccall`) | collard (order 2 and 3), aux_a (order 2): `ghx ghu ghxx ghxu ghuu ghs2 ghxxx ghxxu ghxuu ghuuu ghxss ghuss`, folded tensors | 1e-13 (order 3), 7e-13 (aux_a) |
| Simulation at orders 2 and 3, with and without pruning; generalised IRFs on Dynare's draws | collard at order 2 and 3, aux_a at order 2; irf.m of Dynare with fixed seed | 5e-15 (paths), 1.3e-15 (IRFs) |
| Moments at order 2 | theoretical (plain: mean correction; pruned state space) incl. variance and conditional decompositions with measurement error; simulated moments of order-2 paths | 6e-13 (theoretical), 2e-10 (simulated skewness) |
| Shipped examples at order 2 | the five examples whose `.mod` asks for order 2: `ghxx ghxu ghuu ghs2` | 7e-14 |
| Auxiliary and model-local variables, `loglinear` | against Dynare | 3e-14, 5e-13 |
| OccBin: piecewise-linear simulation (one and two constraints), `simul_*` options, error flags 310-313, periodic solutions, `error_bind`/`error_relax` | rbc_occbin (169 shock periods, 2 constraints), borrcon_sim/curb, occ_borrcon (two shocks; maxit sweeps, curb_retrench, endo_init, exo_pos, max/reset check-ahead, init_regime, init_binding_indicator), occ_rbc_cyc/occ_rbc1/occ_rbc_err (52 + 24 cycling cases): paths, regime histories and error flags | 3e-15 (paths), regime histories and flags identical |
| OccBin IRFs (`occbin.irf`: both signs, all shocks, `shocksize`, `check_ahead_periods`, `maxit`) | occ_borrcon (2 shocks), occ_rbc_cyc (2 constraints) | 9e-16 |

Speed of one `set_param!` + steady state + first-order solve: 0.04-0.08 ms (15 variables), 0.4 ms
(28 variables, Julia steady-state function), 1.4 ms (42 variables, numerical steady state), 4 to 6 times
faster than pydyn on the same machine. An interpreted `_steadystate.m` costs about 3 ms per call.

## Not implemented yet

- Theoretical moments at order 3 and filtered (HP, band-pass) theoretical moments at order 2 (refused, as in Dynare);
  loglinear models above order 1 (refused, as in Dynare); unit roots with pruning at order 2 (refused, as Dynare).
- Perfect foresight, Kalman filter and smoother, shock decompositions, estimation.
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
- Where MATLAB and Octave differ, the interpreter follows MATLAB; `isoctave` is false.
- JDyn fixes several MATLAB-semantics bugs that pydyn has (`strrep` with overlapping matches,
  `strsplit` with repeated delimiters, `sprintf('%.3d')`, `num2str(x, fmt)` blanks).

## Open checks

- Files of Dynare's test suite outside pydyn's corpus have not been compared: in particular a file that
  pydyn refuses but JDyn loads has no Dynare reference yet.
- `options_.qz_criterium` set by MATLAB code and a `qz_criterium` statement option: the statement option
  wins regardless of order in the file.
