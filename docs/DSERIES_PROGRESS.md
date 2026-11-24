# dates / dseries progress log

Branch `dseries` (base a487e5d). Plan: native Julia `dates` and `dseries` objects in the MATLAB interpreter, checked against Octave running Dynare 7.2's dseries toolbox.

## Done
- src/matlab/dates.jl: MDates type, calendar helpers, constructor, operators, methods (written, not yet compiled).
- src/matlab/dseries.jl: MDseries type, constructor, readers (csv/mat/m), arithmetic, lag/lead/diff/log/exp/..., extract, merge, concat, rename, tag, save (written, not yet compiled).

## Next
1. Include both files in src/JDyn.jl; add interpreter hooks (field access, method calls, binary ops, ranges, concatenation, mcall dispatch, assignment) in interp.jl/values.jl.
2. Register constructors dates(), dseries(), isdates, isdseries, ... in BUILTINS.
3. Differential tests against Octave (test/dseries_cases.txt + generator).
4. .mod side: date literals in shocks/periods, perfect_foresight_setup dates, data(first_obs), initval_file/histval_file.
5. Acceptance sweep on dseries/dates files; update docs/ACCEPTANCE.md, STATUS.md.
