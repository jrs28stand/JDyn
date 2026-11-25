# Parity batch 3 (branch parity3): pac.estimate, shock_decomposition(forecast_type), loglinear results layer

Clone /home/claude/work/jdyn-parity3 (base 1bb4530). Spec: pydyn main 42fcf84. Validation: Dynare 7.2 under Octave 8.4
(references test/ref_parity2/oo/, Octave store /home/claude/accept_ref/accept_ref).
Dev loop: a long-lived Julia session that `Base.include`s the changed files into JDyn, then runs test/test_parity2_results.jl
(`JDYN_TEST_ONLY=pe_t12,sd_fc_pm` selects cases).

## Status

| # | item | status |
|---|------|--------|
| 1 | pac.estimate.nls / iterative_ols | runs; all 11 pe_* cases within their tolerance (below); registered in runtests |
| 2 | shock_decomposition forecast_type | sd_fc_cal / ml / pm / pmm all agree with Octave to 3e-13 (see fixes) |
| 3 | loglinear results layer | (in progress) |

## Item 1: pac.estimate

* Bug: `Set \ Set` (matrix division) in `_pe_set_auxiliary_series`: replaced by `setdiff`.
* Differences from the references (relative, floor 1): t12 6e-12, t9 6e-12, t19g 4e-11, t18 9e-10, t6 6e-10, t7 1.4e-9, t17 5e-9 (GaussNewton),
  t8a 9e-9 (annealing), v9 8e-8 (fmincon), t25 1.3e-7 (fmincon), t15 2e-5 (fminunc).
  fmincon/fminunc: own Levenberg-Marquardt to the minimiser; the references stop at the routines' own tolerance (1e-7 level), hence the gap.
  Annealing: same path as Dynare's (draws from Octave's stream); the last accepted points differ by 5e-9 because near the optimum the
  accept test compares sums of squares that differ in the 13th digit between Octave and Julia.
