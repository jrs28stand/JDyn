# Status (2026-11-21)

Reference: Dynare 7.2 under GNU Octave 8.4 (the same reference results as pydyn). All figures below are
from `test/runtests.jl` (1,739 checks pass, 0 fail; 3 skipped: perfect foresight, and references
without IRFs).

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

Speed of one `set_param!` + steady state + first-order solve: 0.04-0.08 ms (15 variables), 0.4 ms
(28 variables, Julia steady-state function), 1.4 ms (42 variables, numerical steady state), 4 to 6 times
faster than pydyn on the same machine. An interpreted `_steadystate.m` costs about 3 ms per call.

## Not implemented yet

- Orders 2 and 3 (plan: Dynare's C++ k-order library through `ccall`, fed with the preprocessor's
  G1-G3; the preprocessor output and `simulate` at orders 2-3 are already in place).
- OccBin, perfect foresight, Kalman filter and smoother, shock decompositions, estimation.
- Plotting beyond first-order IRFs (OccBin, smoother, decompositions, ...). First-order IRF figures are
  done: Dynare's content rules, tidy CSV export, SVG built in, PNG/PDF through an optional Plots.jl
  extension (see docs/plotting.md).
- Optimal policy (Ramsey, OSR), heterogeneous agents, PAC/VAR models, `dseries`: refused with an error.
- Theoretical moments at order 2 (refused with an error).

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
