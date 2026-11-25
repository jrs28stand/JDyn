# Second round of acceptance-sweep fixes (branch acc2)

Restart notes: clone /home/claude/work/jdyn-acc2 (branch acc2 from main 1bb4530); baseline worktree of the same commit plus the
runner fix at /home/claude/work/jdyn-acc2-base (precompiled; used for the baseline sweep into /home/claude/acc2_work/base_new,
tag acc2b). My own sweeps use ACCEPT_TAG=acc2. Manifest.toml and deps/korder/libjdyn_korder.so are copied from jdyn-parity3 (git-ignored).
Comparator: pydyn-rl/tools/accept_compare.py (always use --only / head: its full output is huge).

## Status per item
1. Runner: DONE (test/acceptance/accept_jdyn.py: a test never starts with Dynare's own outputs; earlier files of the entry that did
   not complete get Dynare's outputs, no-overwrite; --ref-artifacts restores the old behaviour). "no .mat written": see below.
2. MATLAB statements that change the model: cellfun/arrayfun/regexp/regexpi/regexprep added (done); the rest in progress.
3. filter_decomposition: DONE (smoother.jl, both filters; oo_.FilteredVariablesShockDecomposition). Recursive estimation: DONE
   structurally (results_estimation.jl `_estimation_recursive!`), values depend on optimiser end points.
4. Data files without dates: DONE (`_data_start_date`: year one, annual, as Dynare's load_*_file_data).
