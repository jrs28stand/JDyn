# OBC methods: performance and reliability (branch `obc-perf`)

Protocol: /home/claude/work/bench/BENCH_PROTOCOL.md. Harness: `bench/obc_bench.jl`. Raw output: `bench/results/obc_bench_before.md`
(BEFORE run, taken before any optimisation) and `bench/results/obc_bench_after.md` (AFTER run; in it the `before` rows are the baseline
source commit 5670539, measured in the same process and interleaved with the `after` rows, so each table is a within-run comparison; the
BEFORE run file is a second, independent measurement of the same baseline). Test logs: `bench/results/baseline_tests.log`, `tests_after.log`.

Method: two copies of the code in one Julia process (`JDynBase` = baseline source with the module renamed, `JDyn` = this branch), one warm-up
call, `GC.gc()` before each timed call, 5 interleaved repeats with rotating order, minimum / median reported. BLAS on one thread. Two cores were
shared with two other workers (load average 1 to 3): treat differences below about 15% as noise. Horizon 40, 1000 shock periods, seed 20261124.
Every table row comes with the maximum path difference to OccBin and whether regime histories agree (in the raw files).

## Summary of what each change bought (zlb / zlb2 / nkm / sw where applicable)

| Change | Where | Effect |
|---|---|---|
| 1. Memoise compiled `bind`/`relax`/error closures and the post-processor's linearised conditions per model (`_compiled_condition`) | occbin.jl, obc.jl | P2: `Occbin()` 2.8 / 4.6 / 3.2 / 2.7 ms -> 0.17 / 0.21 / 0.41 / 0.54 ms; `obc_constraints` 25-41 ms -> 0.3-0.4 ms. By far the largest P2/P3 item: it was `Core.eval` of 4+ closures per constraint (the steady state plus first-order solution is only 0.2-0.7 ms). |
| 2. Period loop computes only the next state (`P y + Q u + Wn y`, `Wn` precomputed), the beyond-horizon gaps from precomputed blocks (`ObcExt`), and builds the full path only for no-shock periods and the tail | obc.jl | minnorm P1 allocation 62 -> 7.4 KiB/period (zlb); path build removed from every period |
| 3. `M`, `Qx`, `Wn` and the beyond-horizon blocks from the diagonal recursion `g_{t+1}(s+1) = g_t(s) + a_{d+s} c_s` with one matrix product for all terms (`_obc_blocks`), `F^r V` by repeated products (no explicit inverse; `V` from the LU factorisation) | obc.jl | O(T^2 n) instead of one path per column (O(T^2 n^2)); agrees with the previous column-by-column `M` to 7e-17 and with the stacked system (test, 1e-10); beyond-horizon gaps agree to 2e-18 |
| 4. `q` as a map from state and shocks only | obc.jl | already the case (`q = q0 + Qx [y; u]`); now also used for the beyond-horizon check (no path needed) |
| 5. OccBin: rule memo (trie keyed by the regime suffix, `_rules`), one latest-world loop for the constraint checks instead of one `invokelatest` per evaluation, allocation-free path recursion | occbin.jl | OccBin P1 2.2-3.5x faster (below), results bit for bit (paths and regime histories, 50 parameter sets x 200 periods per model, `isequal`; plus the full OccBin test files) |
| 6. Warm start from last period's binding set | not implemented | the Python port measured it slower; with the active-set start from `q < 0` already solving every period here, not tried |

## P1: microseconds per shock period (min / median), horizon 40, from the AFTER run

| model | s, share binding | method | before | after | speed-up (min) | KiB/period before -> after |
|---|---|---|---|---|---|---|
| zlb | 1.0, 30% | occbin | 89.3 / 97.1 | 39.2 / 41.1 | 2.3x | 70.4 -> 27.7 |
| | | minnorm | 47.6 / 51.6 | 6.5 / 6.7 | 7.3x | 62.0 -> 7.4 |
| | | minnorm, no fast path | 67.4 / 71.5 | 26.3 / 27.6 | 2.6x | 98.9 -> 44.2 |
| zlb2 | 1.0, 34% | occbin | 234.2 / 238.3 | 111.6 / 115.6 | 2.1x | 130.4 -> 46.4 |
| | | minnorm | 59.2 / 67.6 | 13.0 / 13.3 | 4.6x | 82.2 -> 12.8 |
| | | minnorm, no fast path | 174.7 / 184.6 | 123.8 / 124.9 | 1.4x | 223.4 -> 153.9 |
| nkm | 3.0, 31% | occbin | 260.9 / 283.8 | 72.5 / 76.1 | 3.6x | 272.4 -> 49.6 |
| | | minnorm | 111.7 / 120.2 | 9.1 / 9.9 | 12x | 206.1 -> 10.8 |
| | | minnorm, no fast path | 146.5 / 158.8 | 40.7 / 42.3 | 3.6x | 256.6 -> 61.3 |
| sw (OccBin only) | 1.0 (see note), 21% | occbin | 201.7 / 205.5 | 68.5 / 89.1 | 2.9x | 252.8 -> 51.6 |

Split by whether something binds on the path (min, microseconds), after: zlb minnorm 5.2 quiet / 8.0 binding, occbin 31.2 / 55.4; nkm minnorm 6.3 / 11.8,
occbin 50.8 / 109.7 (before: nkm occbin 89.0 / 580.3, minnorm 57.8 / 209.7). Other splits are in the raw files.
Correctness in every row: path difference to OccBin 0 for OccBin itself (bit for bit before vs after), <= 1e-15 for minnorm on zlb/zlb2; nkm minnorm differs
from OccBin by 2.0e-5 with different regime histories BEFORE and AFTER alike: nkm is the approximate class (the linearised residual versus OccBin's
nonlinear condition, contract rule 6), not a regression.

Note on `sw`: OccBin fails (errors 310/311/313) in about 10% of periods at every scale tried (0.25 to 100) and no scale gives 30-60% binding without absurd
shocks (`bench/results/obc_bench_smoke_sw_scan.md`). The harness uses s = 1 and redraws the failing shock periods (about 5% of the 1000; the sequence is
therefore not an iid sample); 21% of periods bind. Scale and redraws are in the raw files.

## P2: milliseconds per parameter set (min), AFTER run

| model | steady state | first order | `Occbin()` before -> after | `obc_constraints` before -> after | M, q (+ beyond-horizon blocks) before -> after | ready for occbin before -> after | ready for minnorm before -> after |
|---|---|---|---|---|---|---|---|
| zlb | 0.05 | 0.18 | 2.79 -> 0.17 | 24.9 -> 0.28 | 25.1 -> 0.27 | 2.82 -> 0.19 | 27.9 -> 0.48 |
| zlb2 | 0.05 | 0.19 | 4.58 -> 0.21 | 40.6 -> 0.38 | 40.5 -> 0.49 | 4.62 -> 0.24 | 45.1 -> 0.71 |
| nkm | 0.27 | 0.37 | 3.16 -> 0.41 | 29.1 -> 0.36 | 29.0 -> 0.50 | 3.22 -> 0.45 | 32.4 -> 0.96 |
| sw | 0.24 | 0.53 | 2.67 -> 0.54 | n/a | n/a | 2.72 -> 0.60 | n/a |

"After" memoises per model object: a perturbed parameter vector reuses the compiled closures (they depend only on the model text), which is the estimation
situation. A freshly loaded model pays the compile cost once. The "after" M, q figure includes the beyond-horizon blocks for `check_ahead_periods = horizon`
(for the default `check_ahead_periods = 200` that block has 200 more rows per constraint and is built lazily at the first simulation).
Medians are in the raw files (similar).

## P3: 50 iterations (perturb two parameters, redo P2, simulate 200 periods), seconds per loop (min), AFTER run

| model | occbin before -> after | minnorm before -> after | minnorm, no fast path before -> after | failed iterations |
|---|---|---|---|---|
| zlb | 2.78 -> 0.43 (6.5x) | 2.03 -> 0.085 (24x) | 2.36 -> 0.29 (8x) | 0 |
| zlb2 | 5.98 -> 1.09 (5.5x) | 3.33 -> 0.176 (19x) | 4.66 -> 1.31 (3.6x) | 0 |
| nkm | 4.63 -> 0.90 (5.2x) | 3.32 -> 0.185 (18x) | 4.05 -> 0.64 (6.3x) | 0 |
| sw | 3.65 -> 0.81 (4.5x) | n/a | n/a | 0 |

Per iteration (minimum): zlb minnorm 40.6 -> 1.7 ms, nkm minnorm 66.5 -> 3.7 ms, nkm occbin 92.5 -> 17.9 ms. Over the 50 parameter sets, minnorm differs from
OccBin by at most 1.4e-15 (zlb), 2.2e-15 (zlb2), 2.9e-5 (nkm, approximate class, 42 of 50 iterations with at least one different regime history); OccBin before vs
after identical bit for bit on all four models.

## P4: scaling with the horizon (nkm, s = 3; sw, s = 1), minimum

| model | H | occbin P1 us before -> after | minnorm P1 us before -> after | ready (occbin) ms | ready (minnorm) ms before -> after |
|---|---|---|---|---|---|
| nkm | 20 | 182.9 -> 48.4 | 94.1 -> 8.8 | 3.21 -> 0.47 | 32.4 -> 0.89 |
| nkm | 40 | 273.9 -> 78.4 | 125.1 -> 10.7 | 3.38 -> 0.56 | 38.5 -> 1.07 |
| nkm | 80 | 383.5 -> 127.9 | 206.6 -> 14.7 | 3.10 -> 0.43 | 37.7 -> 1.15 |
| sw | 20 | 144.2 -> 40.0 | n/a | 2.71 -> 0.59 | n/a |
| sw | 40 | 224.9 -> 67.4 | n/a | 2.74 -> 0.59 | n/a |
| sw | 80 | 285.2 -> 127.3 | n/a | 2.72 -> 0.59 | n/a |

(Rows for H = 40 differ slightly from the main P1/P2 tables because they are separate runs on a loaded machine.) After the change the set-up cost grows
about linearly with H (0.89, 1.07, 1.15 ms) instead of being dominated by a constant; the per-period cost of minnorm grows with H through the N x N active-set
solve (8.8, 10.7, 14.7 us). The fast-path-free minnorm (Lemke) scales worse (nkm 21, 45, 134 us when something binds).

## R1: reliability (AFTER code; 2000 states from a long unconstrained simulation at s = 1, one period solved from each, shock scales 0.5, 1, 1.5, 2)

Solved / failed (failure types), horizon 40, `check_ahead_periods` 40. No method has a time limit; OccBin's own limit is `maxit = 30` (error 311).

| model | s | occbin solved / failed | minnorm solved | minnorm, no fast path solved | different regime sequences (of those both solve) | max path diff |
|---|---|---|---|---|---|---|
| zlb | 0.5 / 1 / 1.5 / 2 | 2000 / 0 at every scale | 2000 | 2000 | 0 | 4.4e-16 |
| zlb2 | 0.5 / 1 / 1.5 / 2 | 2000 / 0 at every scale | 2000 | 2000 | 0 | 1.3e-15 |
| nkm | 0.5 / 1 | 2000 / 0 | 2000 | 2000 | 0 | 5.1e-15 |
| nkm | 1.5 / 2 | 2000 / 0 | 2000 | 2000 | 1 per scale (both minnorm variants) | 2.1e-5 / 1.1e-5 |
| sw | 0.5 | 1810 / 190 (310: 8, 311: 51, 313: 131) | refused (multi_equation) | refused | - | - |
| sw | 1.0 | 1735 / 265 (310: 15, 311: 69, 313: 181) | - | - | - | - |
| sw | 1.5 | 1647 / 353 (310: 18, 311: 86, 313: 249) | - | - | - | - |
| sw | 2.0 | 1585 / 415 (310: 27, 311: 139, 313: 249) | - | - | - | - |

No state was found where one method fails and another solves on zlb, zlb2 and nkm (the "solves k states where the other fails" lines in the raw file are all 0).
The single nkm disagreement per scale is a state near the threshold of the approximate constraint (contract rule 6). The sw failures are OccBin's regime
iteration cycling or not converging; the main session's analysis says most of these states have no solution at all and the binding ones several, so they are
counted, not fixed. Failures described, not fixed.

## Profile

Before (measured as timings of the pieces, P2 tables above): per parameter set 25-41 ms of `obc_constraints` and 2.7-4.6 ms of `Occbin()`, of which the
steady state and first-order solution are 0.2-0.7 ms; the rest was `Core.eval` of closures (4 per constraint for OccBin, 2 per condition in the post-processor).
Per period: a full path (`obc_gaps`, `max(L, T + check)` rows) built for every shock period, `_rules` recomputed (a backward sequence of `n x n` solves) for the
`ss` output, `Base.invokelatest` per condition evaluation in `check_constraints` (4 per constraint per row).
After (`Profile`, flat, nkm, 1000 periods x 5): OccBin: `check_constraints` 127 of 306 samples, `_path_rules` 60, matrix-vector products and their allocations most
of the rest; minnorm: 53 samples in total, nothing above 1 sample per frame worth reporting at this resolution. `@code_warntype` was not run (see below).

## Lessons for the Python port

- The largest items are not numerical: compiling constraint closures per parameter set (here 25-40 ms of Core.eval; in Python `compile`/`exec` or sympy-lambdify equivalents)
  and recomputing paths that are not needed. Memoise anything that depends only on the model text.
- Next-state-only period loop: the full path is needed only when the path is continued (no-shock periods) and after the last shock period.
- Build the beyond-horizon check as extra rows of the same affine map (`q0b + Qxb [y; u] + Mb y`) so it needs no path either.
- The diagonal recursion for `M` is a one-line identity: response of gap i at t to a wedge at s is `sum_{r<s} a_{t-s+r} c_r`, with `a_m = Gf P^(m+1) + Gc P^m + Gl P^(m-1)` and `c_r = F^r V`; all terms `a_m c_r` come from one matrix product and each diagonal is a running sum.
- OccBin's rules are memoisable bit for bit by keying on the regime suffix (rule of period i depends on regimes i..Tmax only); in a simulation that revisits few regime patterns this removes the `n x n` solves.
- In Julia, one `Base.invokelatest` around the whole loop of evaluations beats one per call; allocation per period fell 5-20x without changing a digit.

## Not measured / open

- Warm start from last period's binding set: not implemented (Python port: slower).
- `@code_warntype` on the hot functions was not run; no type-stability fixes were made beyond removing temporaries. The remaining allocation (7-13 KiB/period for minnorm) is the active-set solve (`M[alpha, alpha]` copy and LU) and `q`.
- `news` is a stub in JDyn, so it is not measured; `sw` is measured for OccBin only (multi_equation constraint refused by the news methods).
- The profile "before" is by timing the pieces, not a sampling profile of the baseline; the "after" profile is a coarse flat sample.
- Closure memoisation depends on model identity: a model reloaded from disk compiles again once.
- R1: no time limit exists in either method, so no time-outs are counted; R1 was run on the after code only (the paths agree with OccBin bit for bit, so the before results are the same).
- Timings carry the noise of a shared 2-core machine (load average up to 3); P1 and P3 minima are the reliable numbers, medians show the spread.
- Estimation filters (inversion filter etc.) were not timed; P3 is the stand-in the protocol defines.

## Tests

Before any change: test_obc.jl 264/264, test_occbin.jl 37/37, test_occbin_options.jl 456/456, test_occbin_estimation.jl 203/203 (`bench/results/baseline_tests.log`).
After all changes: identical counts, all passed (`bench/results/tests_after.log`). test_plotting.jl was not run (known to fail on this machine for a hard-coded path).

## Reproduce

`source /root/jl/jdyn_env.sh`; tests: `nice julia --project=. test/runtests.jl test_obc.jl test_occbin.jl test_occbin_options.jl test_occbin_estimation.jl`;
benchmark: `nice julia --project=/home/claude/work/benchenv bench/obc_bench.jl --tag NAME --reps 5` (the env has this clone as `JDyn` and the baseline copy
/home/claude/work/jdynbase as `JDynBase`; without `JDynBase` only the current code is measured).

## Remaining speed ideas (estimates)

1. Replace the `M[alpha, alpha]` copy plus LU in the active set by an updating factorisation or a cached factorisation per binding set (binding sets repeat): minnorm per period 6-15 us -> about 3-6 us.
2. Make `ss` (the per-period rule for the filters) lazy for the news methods and skip `_tmax`/`bi` construction when no one reads it: about 1-2 us per period for minnorm, little for OccBin.
3. Reduce OccBin's remaining cost (check loop is dynamic dispatch on abstractly typed closures): compile the constraint conditions into one typed vectorised function per model returning all four values; estimated further 1.5-2x on OccBin P1.

## News port (branch `obc-news`): `news` and the regime formulation, P1 and P2

Run `bench/results/obc_bench_news.md` (2026-11-24 12:44, commit d4972af, `nice julia --project=. bench/obc_bench.jl --tag news --models zlb,zlb2,nkm,sw --parts p1,p2 --reps 5`).
Same protocol as above (one process, warm-up, `GC.gc()` before each call, 5 interleaved repeats, min / median, 1000 shock periods, horizon 40, check-ahead 40, seed 20261124, load average 1.5 to 2.8 on 2 shared cores: treat differences below about 15% as noise).
Methods: `minnorm` (fast path), `minnorm_pure` (Lemke + descent), `news` (active set in front of the HiGHS integer programme), `news_pure` (`obc_fast_path=false`: every binding period through the integer programme), `regime` (the regime formulation, forced with `obc_solver_options = (formulation = :regime,)` on the complementarity models; the only news method that applies to sw). Integer programmes inside the regime formulation's fallbacks were limited to 0.5 s in the benchmark (30 s by default).
sw is run at s = 0.25 (pydyn's choice; above that OccBin and the regime formulation fail on about a tenth of the states, see OBC_PERF.md of pydyn) with one of the 1000 draws replaced (a failing one, reported in the raw file); the earlier JDyn runs above used s = 1 with redraws.

### P1: microseconds per shock period (min / median)

| model (s, share binding) | method | µs/period | nothing binds | something binds | KiB/period | max path diff vs OccBin | same regime histories as OccBin |
|---|---|---|---|---|---|---|---|
| zlb (s=1.0, 30.4%) | occbin | 47.1 / 66.9 | 37.8 / 53.4 | 66.8 / 95.4 | 27.7 | 0.0e+00 | true |
|  | minnorm | 7.1 / 7.2 | 5.6 / 5.8 | 8.6 / 9.1 | 7.4 | 2.5e-16 | true |
|  | minnorm_pure | 28.5 / 30.2 | 6.5 / 6.9 | 77.1 / 82.0 | 44.2 | 2.5e-16 | true |
|  | news | 7.7 / 8.3 | 6.1 / 6.6 | 9.3 / 10.0 | 8.7 | 2.5e-16 | true |
|  | news_pure | 1753.2 / 1835.1 | 9.0 / 9.4 | 5744.4 / 6013.1 | 141.2 | 2.5e-16 | true |
|  | regime | 26.2 / 42.0 | 19.2 / 31.2 | 40.1 / 63.7 | 6.5 | 2.5e-16 | true |
| zlb2 (s=1.0, 33.8%) | occbin | 126.9 / 153.4 | 90.8 / 111.1 | 191.5 / 233.9 | 46.4 | 0.0e+00 | true |
|  | minnorm | 12.8 / 14.4 | 9.3 / 10.6 | 17.1 / 18.7 | 12.8 | 3.9e-16 | true |
|  | minnorm_pure | 127.0 / 135.7 | 11.4 / 11.8 | 351.6 / 375.6 | 153.9 | 3.9e-16 | true |
|  | news | 13.8 / 14.8 | 10.7 / 11.5 | 18.1 / 19.3 | 14.9 | 3.9e-16 | true |
|  | news_pure | 31815.7 / 32714.6 | 14.6 / 17.1 | 94098.6 / 96753.1 | 601.9 | 3.9e-16 | true |
|  | regime | 86.5 / 94.6 | 53.6 / 57.5 | 147.0 / 164.3 | 10.9 | 3.9e-16 | true |
| nkm (s=3.0, 31.4%) | occbin | 80.5 / 93.2 | 56.5 / 66.0 | 123.3 / 140.5 | 49.6 | 0.0e+00 | true |
|  | minnorm | 9.7 / 10.9 | 6.5 / 7.7 | 12.5 / 14.4 | 10.8 | 2.0e-05 | false |
|  | minnorm_pure | 41.5 / 45.0 | 7.4 / 8.1 | 103.0 / 111.4 | 61.3 | 2.0e-05 | false |
|  | news | 9.9 / 10.5 | 7.2 / 7.5 | 12.8 / 13.3 | 12.0 | 2.0e-05 | false |
|  | news_pure | 7660.2 / 8157.9 | 9.1 / 10.7 | 21992.5 / 23418.8 | 196.5 | 2.0e-05 | false |
|  | regime | 50.3 / 53.6 | 31.1 / 32.2 | 83.5 / 90.5 | 10.3 | 2.5e-12 | true |
| sw (s=0.25, 2.6%) | occbin | 52.1 / 82.9 | 50.0 / 79.6 | 92.4 / 145.6 | 39.4 | 0.0e+00 | true |
|  | regime | 25.4 / 29.6 | 21.6 / 24.7 | 47.3 / 54.1 | 10.9 | 3.8e-13 | true |

### P2: milliseconds per parameter set (min / median), horizon 40

| model | RegimeOBC set-up (N = 2H) | `obc_system` (M, q) | diagnostics on M | ready: occbin | ready: minnorm | ready: news (LCP + diagnostics) |
|---|---|---|---|---|---|---|
| zlb | 0.45 / 0.53 | 0.29 / 0.34 | 11.08 / 11.49 | 0.21 / 0.21 | 0.48 / 0.54 | 11.68 / 11.96 |
| zlb2 | 0.83 / 0.86 | 0.47 / 0.54 | 12.25 / 13.27 | 0.24 / 0.29 | 0.70 / 0.78 | 12.95 / 14.05 |
| nkm | 0.52 / 0.60 | 0.52 / 0.56 | 11.75 / 12.24 | 0.47 / 0.51 | 0.95 / 1.02 | 12.77 / 13.19 |
| sw | 0.77 / 0.78 | - | - | 0.59 / 0.61 | - | - |

Reading
- The LCP news method with its fast path costs the same as `minnorm` (zlb 7.7 vs 7.1 µs, zlb2 13.8 vs 12.8, nkm 9.9 vs 9.7 µs per period): the same precomputed blocks and
  next-state-only loop; the extra work is the beyond-horizon check done once for the fast-path answer. Against OccBin: 6.1x (zlb), 9.2x (zlb2), 8.1x (nkm) by the minima.
- Forced through the integer programme (`news_pure`) a binding period costs 5.7 ms (zlb), 94 ms (zlb2, 2 x 40 binaries) and 22 ms (nkm), against 9 to 18 µs through the
  fast path: the integer programme is a guarantee and a fallback, not a speed path. (`minnorm_pure`, Lemke, is 77 / 352 / 103 µs per binding period.)
- The regime formulation is 1.8x (zlb), 1.5x (zlb2), 1.6x (nkm) and 2.1x (sw) faster than OccBin by the minima (absolute times differ from the tables above because of load: compare
  ratios inside a table only). It is slower than the LCP formulation (3.4x on zlb, 6.3x on zlb2): it checks OccBin's conditions in levels along the path.
- nkm (approximate class): `minnorm`, `news` and `news_pure` differ from OccBin by 2.0e-5 and have a different regime history in one of the 1000 periods, period 786 of this
  seeded sample: OccBin binds in periods 2-3 only (regime `[0 1 0]`, starts `[1 2 4]`), the LCP binds in periods 1-3 (`[1 0]`, `[1 4]`), because the unconstrained gap of period 1 is +4.7e-5
  and the wedges that enforce periods 2-3 spill into period 1 (wedge 1.3e-5). This is the known knife-edge of that model (the linearised residual and OccBin's condition differ by the term in `mp`,
  `approx` = 0.57%). The regime formulation, which uses OccBin's own condition, agrees with OccBin on all 1000 periods (2.5e-12).
- P2: the set-up of the LCP formulation (`obc_system`) is 0.3 to 0.5 ms; the regime formulation's 0.45 to 0.83 ms; the diagnostics on M (S-matrix linear programme and the exact
  P-matrix recursion up to 16 rows) take 11 to 12 ms per parameter set, computed once per `ObcSystem` when `news` is used (not decomposed further).
- Not run for the news methods: P3, P4, R1 (not asked for).
