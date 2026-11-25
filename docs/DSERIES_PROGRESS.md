# dates / dseries progress log

Branch `dseries` (base a487e5d).

## Done
- src/matlab/dates.jl, dseries.jl, objects.jl: native `dates` and `dseries` (handle semantics), interpreter hooks in interp.jl/values.jl.
- 132 Octave-checked cases (test/dseries_cases.txt, r_dseries.m, dseries_ref.mat, test_dseries.jl, in ALL_TESTS): 0 mismatches.
- Dynare's embedded unit tests (test/dseries_unit_tests.jl): 366 pass / 22 fail / 97 error (rest are refused features, harness artefacts).
- .mod side: dated shocks/endval/periods/learnt_in/first_simulation_period in perfect foresight; `Simulated_time_series` dseries.
- Acceptance: ramst_dseries REFUSED -> OK; other dseries files are blocked by causes outside this work (docs/ACCEPTANCE.md).

## Not done (refused)
initval_file/histval_file with series or datafile, xls/xlsx, data(first_obs=dates) in estimation, cumsum/cumprod base period,
fill_/exist/chain_/lineartrend/projection/one-sided HP, weekly/daily ydiff, save format m, ds{regex}=..., tags in insert, struct2dseries.
