## Triage (hand-written; appended to the generated tables)

How to re-run: `nice julia --project=. test/acceptance/run_dynare_tests.jl --timeout=90 --shard=K/2` (K = 1, 2 in parallel; resumable;
`--merge` rebuilds results.csv and this file). `OK_WARN` is not a pass: JDyn skipped MATLAB lines or ignored commands
(mostly `estimation` and friends, which do not fill oo_ yet), so the file's own checks may not have run.

### CHECK_FAILED: diagnosis (12 tests, all JDyn or missing feature; none is a test-environment fault)

| tests | check | diagnosis | owner |
|---|---|---|---|
| homotopy1/2/3_test (were 4, fixed) | `oo_.exo_steady_state` after `steady(homotopy_mode)` | JDyn kept the initval x = 1; Octave gives 2. Fixed in `steady!` (results.jl, one line), verified against Octave | done |
| pfwee x4 | pfwee run equals the plain run exactly | the file calls `oo_ = perfect_foresight_solver(M_, options_, oo_)` (function form) and compares exact equality | perfect foresight |
| Irreversible_investment (2) | `oo_.deterministic_simulation.status` | `perfect_foresight_solver(robust_lin_solve)` on 400 periods does not converge in JDyn | perfect foresight |
| example2_simul_replic, example_seeds | `isequal(oo_.var, ...)` after `set_dynare_seed('default')` | needs Dynare's seeded generator streams | seeds |
| moments/example1_var_decomp | simulated (100000 periods, order 2) vs theoretical variance decomposition, tolerance 3 | simulated order-2 moments differ more than Dynare's | moments |
| irfs/example1_unit_std | order-2 relative IRFs (generalised IRFs), tolerance 2e-4 | needs Dynare's random draws (seed) | seeds |
| var-expectations/11 | `oo_.var.toto.CompanionMatrix` | VAR model objects are not implemented | not scheduled |
| ramst_datafile | `perfect_foresight_setup(datafile=...)` | datafile option not applied (its data .m script is a SKIPPED .m test) | perfect foresight |

### Causes not fixed here (count = tests, from the tables above)

- estimation, smoother, forecast and method-of-moments results in oo_ (`oo_.posterior`, `oo_.SmoothedVariables`, `oo_.MarginalDensity`,
  `oo_.likelihood_at_initial_parameters`, `estimation` does not fill oo_): about 100 REFUSED ("non-existent field") plus 150+ OK_WARN: estimation stream.
- PAC / VAR / trend-component models and `var_expectation` / `pac.*` objects: about 90 REFUSED.
- `dseries` / `dates` objects and date literals (`1Y`, `2000Q1`): the main cause of skipped MATLAB lines (130 files mention dseries): needs a dseries class in the interpreter.
- Ramsey / optimal policy (ramsey_model, ramsey_policy): 23, now a clear ModelError (was a raw SteadyStateError "initial guess ... not finite").
- heterogeneous-agent models (22), external functions in the model block (9), perfect_foresight_controlled_paths (9), shock_paths (3),
  deterministic exogenous variables in forecasts (8), ms-sbvar/bvar (11, the preprocessor has no Julia output).
- `moments/example1_bp_test`, `example1_hp_test`: TIMEOUT over 90 s (band-pass / HP simulated moments: performance stream);
  `moments/example1_theoretical_simulated` needed about 4 GB and was killed by the OOM killer in one sweep.
- tests that read output of an earlier test in the same entry now work for `_results.mat` and `save`d files; those needing `dseries` data do not.

### Fixed here

Package functions (`+pkg/f.m`: walsh), `varlist_indices`, `tic/toc/cputime`, `rand/randn/randi/randperm/normrnd/unifrnd/mvnrnd` under `dynare()`
(the loader still refuses them), `cell2mat`, `struct2cell`, `struct2array`, `num2cell`, `cellstr`, `isequaln`, `union/intersect/setdiff`,
plotting as a no-op under `dynare()`, `load` with an output, `fullfile/fileparts/pwd/isfile/isfolder/getenv/mkdir`, `save` (function and command
form), Dynare's `<fname>/Output/<fname>_results.mat`, `MatlabError.user` and `ModelError.cause` (so the file's own `error()` is reported as such),
Ramsey refusal, `oo_.exo_steady_state` after homotopy.
