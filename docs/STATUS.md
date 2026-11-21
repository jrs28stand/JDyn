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
| Estimation, first order (`src/priors.jl`, `kalman.jl`, `estimation.jl`) | prior log density on all families (est_priors); log likelihood and log prior at 13 reference files (est_schorfheide, est_nk v0-v9, est_ur v0-v6: missing data, measurement errors incl. correlated, first_obs/presample, prefilter, lik_init 1-3, diffuse and univariate filters, trends); Hessian, Laplace approximation, csminwel modes, log posterior at Dynare's 40 stored draws | 1.4e-11 (log likelihood), 6e-13 (log prior), 3.6e-10 (log posterior at the draws), 6e-6 (Hessian, relative), 2e-8 (mode value), 5e-5 std (mode location) |

Speed of one `set_param!` + steady state + first-order solve: 0.04-0.08 ms (15 variables), 0.4 ms
(28 variables, Julia steady-state function), 1.4 ms (42 variables, numerical steady state), 4 to 6 times
faster than pydyn on the same machine. An interpreted `_steadystate.m` costs about 3 ms per call.

## Not implemented yet

- Orders 2 and 3 (plan: Dynare's C++ k-order library through `ccall`, fed with the preprocessor's
  G1-G3; the preprocessor output and `simulate` at orders 2-3 are already in place).
- OccBin, perfect foresight, Kalman smoother, shock decompositions, estimation beyond first order.
- Plotting beyond first-order IRFs (OccBin, smoother, decompositions, ...). First-order IRF figures are
  done: Dynare's content rules, tidy CSV export, SVG built in, PNG/PDF through an optional Plots.jl
  extension (see docs/plotting.md).
- Optimal policy (Ramsey, OSR), heterogeneous agents, PAC/VAR models, `dseries`: refused with an error.
- Theoretical moments at order 2 (refused with an error).

## Estimation: scope and limits

- Supported: `varobs`, one `estimated_params` block (plus `estimated_params_init`, `_bounds`, `use_calibration`),
  `observation_trends`, data from `.m`/`.csv`/`.mat`/matrix, `first_obs`, `nobs`, `presample`, `prefilter`,
  `loglinear`/`logdata`, `lik_init` 1-3, `diffuse_filter`, `kalman_algo` 0-4, `qz_criterium`, `riccati_tol`,
  `mode_compute` 0 and 4 (csminwel, ported from Dynare), `mh_replic`/`mh_nblck`/`mh_jscale`/`mh_drop`
  (random-walk Metropolis-Hastings), Laplace and modified-harmonic-mean marginal densities, posterior moments.
  `estimate(m)` runs what the `.mod` file's `estimation` statement asks. Unsupported options raise an
  `EstimationError` (or warn once with `allow_unsupported=true`); other `mode_compute` values are replaced by csminwel with a warning.
- Not implemented: smoother, forecasts, posterior IRFs, recursive estimation, other samplers, orders above 1,
  particle filters, more than one `estimated_params` block.
- The sampler is checked against Dynare's target density (40 stored draws) and statistically (2 x 4000 draws:
  means within 0.17 posterior std of Dynare's 2 x 20000, modified harmonic mean within 0.19); it is not
  draw-for-draw because Octave's random numbers cannot be reproduced.
- Differences from pydyn: csminwel instead of scipy optimisers; correlated measurement errors in the univariate
  filter (pydyn refuses them); Blanchard-Kahn failures carry Dynare's penalty.
- Known difference: a root within rounding of `qz_criterium` (Schorfheide with rho = 0.999999, Dynare info 3) is
  classified as stable by JDyn's QZ; marked as broken in the tests.

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
