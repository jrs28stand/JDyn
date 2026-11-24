# PAC / VAR / trend-component progress log

Branch `pac` (base a487e5d). Implementation: src/pac.jl; hooks (all small): `PKG_FUNCS` + `_pkg_builtin` (src/matlab/builtins.jl, interp.jl),
`index3` call in `mindex` (values.jl), M_/oo_ field fallbacks (session.jl), `_semi_into_M!` (results.jl), `_semi_prepare` / `_semi_init!` (model.jl),
a guard in `fill_auxiliary` (steady.jl: auxiliaries without their own variable on the static diagonal, as in unit-root MCE models, kept their start value instead of NaN).
Tests: test/test_pac.jl (fixtures + Octave references in test/ref_pac; `make_fixtures.py`, `gen_refs.sh`, `extract.m`).

## Acceptance (test/acceptance/run_dynare_tests.jl on the three folders)

| folder | before | after |
|---|---|---|
| pac (73) | OK_WARN 7, REFUSED 60, ERROR 6 | OK 18, OK_WARN 26, REFUSED 23, ERROR 6 |
| var-expectations (23) | CHECK_FAILED 1, REFUSED 17, ERROR 4, XFAIL 1 | OK 8, OK_WARN 10, ERROR 4, XFAIL 1 |
| trend-component-and-var-models (25) | OK_WARN 14, REFUSED 11 | OK 13, OK_WARN 5, REFUSED 7 |

OK_WARN = ran to the end but dseries/`search`/`cherrypick` lines were skipped (the dseries agent's work: `dseries(...)`, `simul_backward_model`, `2000Q1` dates):
- pac/trend-component-1/example1.mod
- pac/trend-component-11/example1.mod
- pac/trend-component-14/example1.mod
- pac/trend-component-2/example1.mod
- pac/trend-component-27/example1.mod
- pac/trend-component-28/example1.mod
- pac/trend-component-28/example2.mod
- pac/trend-component-28/example3.mod
- pac/trend-component-28/example4.mod
- pac/trend-component-28/example5.mod
- pac/trend-component-28/example6.mod
- pac/trend-component-29/example1.mod
- pac/trend-component-29/example2.mod
- pac/trend-component-3/example1.mod
- pac/trend-component-4/example1.mod
- pac/trend-component-5/example1.mod
- pac/var-1/example1.mod
- pac/var-12/example5.mod
- pac/var-2/example1.mod
- pac/var-3/example1.mod
- pac/var-4/example1.mod
- pac/var-5/example1.mod
- pac/var-6/example1.mod
- pac/var-7/example1.mod
- pac/var-8/example1.mod
- pac/var-9/example1.mod
- trend-component-and-var-models/legacy/tcm2.mod
- trend-component-and-var-models/legacy/tcm3.mod
- trend-component-and-var-models/legacy/tcm4.mod
- trend-component-and-var-models/legacy/tcm5.mod
- trend-component-and-var-models/vm4.mod
- var-expectations/10/example1.mod
- var-expectations/12/example1.mod
- var-expectations/2-with-time-shift/example1.mod
- var-expectations/2/example1.mod
- var-expectations/4-with-time-shift/example1.mod
- var-expectations/4/example1.mod
- var-expectations/6-with-time-shift/example1.mod
- var-expectations/6/example1.mod
- var-expectations/7/example1.mod
- var-expectations/8/example1.mod

REFUSED (clear reasons): `pac.estimate.nls` / `iterative_ols` (23 tests, need dseries and an estimator), legacy `get_companion_matrix_legacy` / `oo_.var.toto.ar`
(vm1-3, legacy tcm1, tcm6-8 which also read `tcm6_data.mat`/feval of a package function).
ERROR: the 10 `substitution.mod` tests (`@#include` of the `.inc` files that `pac.print` / `var_expectation.print` write; not implemented).

## Next steps
- `pac.print`/`var_expectation.print` (print_expectations.m, write_expectations.m, rewrite_aux_variable.m) writing the `.inc` files and `evaluate.m`.
- `pac.estimate.*` after dseries lands; legacy `get_ar_ec_matrices`.
- Once dseries is merged: run the 26 OK_WARN pac tests again (their simulations and `pac.estimate` checks use dseries).
