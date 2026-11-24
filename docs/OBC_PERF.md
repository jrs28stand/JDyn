# OBC methods: performance and reliability (branch `obc-perf`)

Protocol: /home/claude/work/bench/BENCH_PROTOCOL.md. Harness: `bench/obc_bench.jl`.

## State at interruption (2026-11-24)

Done and measured:
- Baseline tests (before any change, fresh process, `bench/results/baseline_tests.log`): test_obc.jl 264/264, test_occbin.jl 37/37, test_occbin_options.jl 456/456, test_occbin_estimation.jl 203/203.
- Harness `bench/obc_bench.jl` (P1-P4, R1) written and smoke-tested on zlb and sw. A tick hook (`OBC_TICKS`, `_obc_tick()`) was added to `src/occbin.jl` and `src/obc.jl` (timing only, no effect on results).
- BEFORE run in progress/partial: `bench/results/obc_bench_before_partial.out` (zlb, zlb2 complete; nkm partly; sw not yet). The full run writes `bench/results/obc_bench_before.md` if it finishes (started with `--tag before --reps 5`).
- Before/after in one process: the baseline source is copied to /home/claude/work/jdynbase as package `JDynBase` (module renamed, same tick hook, BASE_COMMIT file); env /home/claude/work/benchenv has both JDyn (this clone) and JDynBase.
- sw: OccBin fails (310/311/313) in about 10% of periods at every shock scale; no scale gives 30-60% binding without absurd shocks. Harness uses s = 1 with failing draws redrawn (see `bench/results/obc_bench_smoke_sw_scan.md`).
- Findings from smoke runs: Occbin() costs ~5-9 ms of which steady state + first order is 0.25 ms (the rest is `_compile_constraints!` Core.eval); `obc_constraints` costs 45-60 ms (Core.eval in `_linearise_condition`); `news` is a stub, so only occbin, minnorm, minnorm_pure are measured; sw is OccBin only.

Not started (planned, no code written): memoise compiled constraint closures per model (occbin.jl `_compile_constraints!`, obc.jl `_linearise_condition`); gemm-recursion build of M/Qx plus next-state and beyond-horizon blocks so the per-period loop needs no full path; lazy `ss` rules for minnorm results; in-place active set; rules trie memo in `_rules` for OccBin.

Commands:
- `source /root/jl/jdyn_env.sh`
- tests: `cd /home/claude/work/jdyn-perf && nice julia --project=. test/runtests.jl test_obc.jl test_occbin.jl test_occbin_options.jl test_occbin_estimation.jl`
- benchmark: `nice julia --project=/home/claude/work/benchenv bench/obc_bench.jl --tag NAME --reps 5 [--models zlb,nkm --parts p1,p2]`

Next three steps: (1) finish the BEFORE run and save it; (2) closure memoisation (P2), then the per-period rewrite; (3) OccBin profile, re-run the harness as `after`, fill this document.
