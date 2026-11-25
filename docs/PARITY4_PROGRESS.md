# Parity4 progress (branch parity4, from main 15c8e50 + merge of acc2)

Environment notes: copy Manifest.toml and deps/korder/libjdyn_korder.so from ../jdyn-acc2 (both untracked); any edit under src/ triggers
a 5-9 minute precompile, so develop a changed file as a copy in /tmp and `Base.include(JDyn, copy)`, then copy back.

## Done
- Merged acc2 (conflict in src/forecast.jl: kept `_cf_eval`, made it vector-capable).
- (3a) Interpreter: src/matlab/builtins_parity4.jl (regexp per lo-regexp.cc on PCRE, intersect, cell2mat double/logical mix,
  chol [R,p], sparse as full, char(), isfile/isfolder/exist/pwd confined to the model folder). pydyn case sets in
  test/ref_gaps/cases, run by test/test_gaps_interp.jl: pydyn_fn 124 match / 9 refused / 0 differ; pydyn_files 37/4/0
  (4 refused = paths outside the model folder, as pydyn); pydyn_mi 54/0/0.
- tools/options_probe.jl + tools/options_probe_cases.jl (differential probe; first stoch_simul run recorded below).

## Probe findings (stoch_simul, JDyn): SAME where Dynare acts
nocorr, contemporaneous_correlation (no field difference at order 1 without periods: check), simul_replic (ignored: to implement),
qz_zero_threshold (ignored: to implement), replic at order 1 (expected), SpectralDensity (preprocessor syntax is `SpectralDensity.trigger`? probe syntax to fix),
dr=cycle_reduction/logarithmic_reduction (harmless at default tolerance).

## To do
1. src/option_policy.jl (table for all 65 grammar statements from DynareBison.yy via pydyn tools/options_audit.py; enforcement in `_run_in_place!`,
   warning at model load), test/test_option_policy.jl.
2. simul_replic, qz_zero_threshold, SpectralDensity.trigger.
3. Model's own generated functions interpreted from the preprocessor text; estim_params_ opaque, bayestopt_ index fields.
4. Cross-port runner on pac/, var-expectations/, kalman/lik_init; full suite (log /tmp/parity4_suite.log), git checkout -- test/ref_parity.
