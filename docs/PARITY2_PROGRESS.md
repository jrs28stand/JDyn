# Parity batch 2 (branch parity2): bring JDyn level with pydyn 0471ce9

Clone: /home/claude/work/jdyn-parity2 (base 4603c03). Spec: pydyn main 0471ce9 (/home/claude/work/pydyn-rl). Validation: Dynare 7.2 under
Octave 8.4 (references copied to test/ref_parity2/ from pydyn's reference store; Octave store /home/claude/accept_ref/accept_ref).
Environment: see the task notes; `Manifest.toml` and `deps/korder/libjdyn_korder.so` are git-ignored and were copied from other clones.
Tests: test/test_parity2_*.jl, registered in test/runtests.jl.

## Status

| # | item | status | validated against Octave | refused / not done |
|---|------|--------|--------------------------|--------------------|
| 1 | OccBin forecasts, Gaussian and Sobol bands (`occbin.forecast`, `qmc_sequence`) | done: src/occbin_forecast.jl, src/sobol.jl, src/results_parity2.jl | draw for draw with Octave's stream: 1.2e-14 to 6.4e-14 (NKM cases A-E, 60 Gaussian and 127 Sobol draws, fc1.mod through dynare()); qmc_sequence 1e-15 | loglinear; replic=0 with `shocks(surprise)` covering fewer shocks than the model (Dynare's own size mismatch) |
| 2 | OccBin findmin regime guess, smoother `first_period_occbin_update` | done: src/occbin_filter.jl, src/fminsearch_octave.jl (Octave's fminsearch) | test/test_parity2_occbin.jl against the Octave runs in test/ref_parity2 | none |
| 3 | pac.estimate.nls / iterative_ols (csminwel, simplex, GaussNewton, annealing; lsqnonlin/fmincon/fminunc/fminsearch through own Levenberg-Marquardt, minimiser only) | code written, registered, loads: src/pac_estimate.jl. NOT validated at runtime | blocked: the Dynare tests and pydyn's pe_* cases build their data with `simul_backward_model`, which this branch lacks (it is in jdyn-gaps); test/test_parity2_results.jl and test/ref_parity2/oo/pe_*.{mod,mat} are ready but not registered | particleswarm (refused) |
| 4 | shock decomposition with forecasts | `oo_.shock_decomposition`, `oo_.forecast_shock_decomposition.<type>`, `shock_decomposition_info`, `plot_shock_decomposition_info` written (src/results_shockdec.jl, any parameter set, via the calib_smoother machinery on a copy of oo_). fs2000_cal_groups matches the Octave reference completely; the forecast_type paths are NOT validated (cases sd_fc_* in test/ref_parity2/oo need conditional_forecast / forecast chain) | partly | with_epilogue; forecast_type with loglinear; realtime_shock_decomposition oo_ fields not done |
| 5 | smoother2histval, send_*_to_workspace | done (src/results_parity2.jl); tests/smoother2histval/fs2000_calib_smoother runs and its asserts pass (verdict partial: only dr.eigval/gx/Gy missing) | accept comparator | other smoother2histval tests are refused by estimation options (endo_vars_for_moment_computations_in_estimation) and `M_.params=results_estimation.M_.params` (accfix territory) |
| 6 | loglinear in the results layer | not done | | findings below |
| 7 | dseries assignment | tested only: no source change needed (test/test_parity2_dseries.jl, 15 checks against Dynare's class) | | |

## Findings and notes

- Date literals (`2005Q1:2000Q1+200`) are now lexed (src/matlab/parser.jl, `dates('..')`); before, a MATLAB line with a range of literals could not be parsed.
- accept_jdyn.py: work folders under /var/tmp were shared by all clones (two clones fed each other's jobs); `ACCEPT_TAG` now names them.
- loglinear: against the Octave store, `oo_.dr.Gy` of loglinear runs differs from ghx's state rows by 1.3 (Dynare keeps the solver's hx; JDyn takes ghx rows) and, after `estimation` of a loglinear model, `dr.ys` differs by 8.5 (fs2000_est, conditional_forecasts/2/fs2000_est). Not resolved here.
- The estimated_params blank-bounds fix of JDyn main (aeba2eb) is cherry-picked.
