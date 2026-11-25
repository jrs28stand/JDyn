# Parity batch 2 (branch parity2): bring JDyn level with pydyn 0471ce9

Clone: /home/claude/work/jdyn-parity2 (base 4603c03). Spec: pydyn main 0471ce9 (/home/claude/work/pydyn-rl). Validation: Dynare 7.2 under
Octave 8.4 (references copied to test/ref_parity2/ from pydyn's reference store; Octave store /home/claude/accept_ref/accept_ref).
Environment: see the task notes; `Manifest.toml` and `deps/korder/libjdyn_korder.so` are git-ignored and were copied from other clones.
Tests: test/test_parity2_*.jl, registered in test/runtests.jl.

## Status

| # | item | status | validated against Octave | refused / not done |
|---|------|--------|--------------------------|--------------------|
| 1 | OccBin forecasts, Sobol and Gaussian bands | code written (src/occbin_forecast.jl, src/sobol.jl, src/results_parity2.jl) | pending | loglinear |
| 2 | OccBin findmin, smoother start period | code written (src/occbin_filter.jl, src/fminsearch_octave.jl) | pending | |
| 3 | pac.estimate | not started | | |
| 4 | shock decomposition with forecasts | not started | | |
| 5 | smoother2histval, send_*_to_workspace | not started | | |
| 6 | loglinear in the results layer | not started | | |
| 7 | dseries assignment | not started | | |

## Notes
