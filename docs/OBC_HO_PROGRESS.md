# Higher-order news OBC (obc_order = 2) and cubature: progress log

Branch `obc-ho` (pushed to `origin wip/obc-ho`), based on `wip/obc` at 5f0d3a0. Specification: pydyn's prototype
`pydyn/obc_ho.py` (`HigherOrderNews`), design in pydyn `docs/OBC_PROGRESS.md` and `reference/dynareobc_octave/README.md`.

## State / next step

MAINTENANCE TASK (three jobs, started 2026-11-25). Job 1 (bound the OccBin rule memo): DONE, see "Rule memo budget" below. Job 2 (news-filter "horizon too short" status, opt-in
`obc_horizon_max`): DONE, see "Horizon too short" below. Job 3 (merge `main` into `obc-ho`, tighten the `bf` replay): merged, see "Merge of main" below.
Resume: `source /root/jl/jdyn_env.sh; julia --project=. test/<file>.jl` (one file at a time; first run after a `src` edit precompiles for ~4 minutes). Push only to `origin obc-ho:wip/obc-ho`.

(Older state, cubature port:)

CUBATURE PORT (current task; spec: pydyn docs/OBC_HO_PERF.md). Steps 1-6 DONE (full average with cached factorisations; defaults full average + level 11; up-front M diagnostic; failure handling; two
constraints at higher order; regions = k with DynareOBC's splitting, validated against ref_o2cub4.mat). Cross-port checks vs pydyn done for steps 1, 4, 5. NEXT: final report to the coordinator (see the sections
"Step 1" timing table, "Steps 4 and cross-port checks", "Step 5", "Step 6" below).

State: H1, H2, C1 done. Follow-up (rule 7, regime selection, end-to-end estimation) done, see docs/OBC_ESTIMATION.md. Task 1 (mod-file/workspace/keyword routes and refusals for the higher-order options) and Task 2 (estimation under obc_method, steps a-d, docs/OBC_ESTIMATION.md) done; test_obc_ho.jl (85), test_obc_estimation.jl and the existing OccBin/OBC suites pass.
Next step: review by the coordinator; the cubature step is untouched (being redesigned elsewhere).
Perf items from pydyn docs/OBC_HO_PERF.md adopted (affine state-to-gap map, Cauchy-Schwarz skipping with norm-sorted points, mean-point
identity, Sobol cache). Active-set singular-block guard added (src/obc_solvers.jl, `OBC_PIVOT_RTOL`).
Next (optional): wire obc_order & co into mod-file options / results.jl; several regions; tied-wedge class; order 3.
To resume cold: `source /root/jl/jdyn_env.sh; julia --project=. test/runtests.jl test_obc_ho.jl` (first run precompiles JDyn: ~4 minutes).
Quick iteration without precompiling: load JDyn, then `Base.include(JDyn, "src/obc_qmc.jl"); Base.include(JDyn, "src/obc_ho.jl")`.

## Files
- `src/obc_ho.jl`: `ObcHO`, `obc_ho(ob; ...)`, `obc_ho_simulate`, `ho_expected_path`, `ho_gap_path`, `ho_conditional_root`, `ho_news_shocks`,
  `_simulate_obc_ho` (entry point), option parsers `_obc_order`, `_obc_int_option`.
- `src/obc_qmc.jl` + `src/obc_sobol_data.jl`: Dynare's unscrambled Sobol generator (direction-number tables generated from
  `mex/sources/sobol/sobol.f08`) and Dynare's inverse normal CDF; `qmc_normal_points(dim, level)`.
- `src/occbin.jl` (`simulate`): new keywords `obc_order=1`, `obc_cubature=false`, `obc_periods_of_uncertainty=16`, `obc_qmc_level=10`,
  `obc_cubature_regions=1`; routed to `_simulate_obc_ho` when `obc_order != 1` or `obc_cubature`. `src/JDyn.jl`: two `include`s.
- `test/test_obc_ho.jl` (registered in `test/runtests.jl`), data in `test/obc/ho/` (DynareOBC output copied from pydyn's
  `reference/dynareobc_octave`, plus `scipy_x.mat`: SciPy's unscrambled Sobol points mapped to normals, for one comparison).

## Design notes
- State (y1, y2, o) as in the prototype; everything in declaration order (decision-rule rows are permuted once).
- The expected path and the gap along it are affine in (y1, y2, kron(y1s, y1s), o_pre) and in the first period's lag and shock, so the map
  (`qc, Q1, Q2, Qw, Qo`) is built once (`_ho_build_q_map!`); a period costs five matrix-vector products, the LCP and, with cubature, one
  N x npoints product. `ho_expected_path` / `ho_gap_path` (the recursion itself) are kept as the reference the map is tested against.
- LCP: `solve_active_set` on the T x T block, then the rows beyond the horizon are checked, then `solve_news` with `Mb`, `qb`
  (horizons T/4, T/2, T) as in the order-1 loop. Trivial test as in the prototype: all q >= -1e-8 max(1, max|q|) gives zero wedges.
- One constraint, a complementary pair (the class `obc_system(...; method = :news)` accepts). Tied-wedge (regime formulation) is not done.
- `simulate(...)` through the new loop solves every period (zero shock rows are real zero draws); `piecewise` = with bound, `linear` = the
  pruned simulation without bound. `obc_raise = false` is not supported there (an `ObcError` is always raised).
- Not done: order 3, several regions, CATCH, mod-file / results.jl wiring of the new options.

## Results (nk model, horizon 32, check ahead 32, 300 shocks)
- order 1 / order 2 vs DynareOBC: 4.4e-16; M: 8e-17; R R' vs Rcc Rcc': 3.7e-19 (R columns are in the same order as Rcc, signs differ).
- cubature, DynareOBC's root and points injected: 6.7e-16. Own root with signs aligned to Rcc and own Dynare-Sobol points: ~1e-15.
- Sobol: `qmc_normal_points(12, 9)` equals `sob_o2cub.mat` to 1.8e-15 (Dynare's own generator ported, not SciPy's).
- Own root, unaligned eigenvector signs: 1e-5 to 1.4e-4 (QMC error of the one-region rule: depends on the arbitrary signs). pydyn's 1.49e-6 is
  reproduced exactly (1.4878498e-6) with pydyn's sign pattern and SciPy points: the pydyn "< 5e-6" is luck of the signs, not a property of the rule.
- us per period (min of 5, 300 periods): order 1 1.3, order 2 1.5, cubature level 9 80, level 10 200; pydyn same machine: 26, 39, 372, 677.
- Singular active-set block (simple1_occ.mod, shock -0.02): the old solver returned :ok with wedges of 1.8e14 (and `simulate` then died with
  SingularException); now :no_solution (ObcError), as pydyn.


## Cubature port, step 1: the full average (`obc_cubature_regions = all`)

Option: `obc_cubature_regions = all` (also `:all`, `"all"`, `0`; mod file `options_.occbin.obc_cubature_regions = 'all';`), internally `regions = 0`; `1` keeps working; k > 1 is refused
until step 6. The result is `(sum over relevant points of y) / (number of points)`; points that need no wedge contribute zero. Code: `src/obc_cubature.jl` (point solver, cache, lockstep
solver), `_cub_all!` and `ho_news_shocks` in `src/obc_ho.jl`. `h.cub.mode` selects the solver: `:scalar` (the existing `solve_active_set` per point), `:cache` (default), `:group`.

* `:cache`: the same scalar guess-and-verify iteration (same sequence of sets per point, hence the same solution when there are several), but the linear solve uses an LU of
  M[A, A] cached by the set A (packed key), memory-capped at 256 MB (cleared when exceeded), with the pivot guard of `_complementary_solution` applied once when the entry is made (a block
  that fails the guard fails the iteration at that point, as in the scalar code). LU instead of a stored inverse: in Julia a solve with a stored LU is O(k^2) and has the accuracy of the scalar
  solve, so no conditioning threshold is needed (pydyn applies its inverse only when cond_1 <= 1e4).
* `:group`: the points of a period in lockstep; points with the same set are solved with one multi-right-hand-side `ldiv!` against the cached LU. Sets up to 128 wedge periods.
* Verified column by column (test_obc_ho.jl, 4,400 assertions): 1,500 gap paths, ok flag and wedges equal `solve_active_set` (cached iteration and lockstep, <= 1e-12); a 20 kB cache that is cleared
  repeatedly changes nothing; the three modes give the same simulation (<= 1e-12); the average equals a hand-written loop over the relevant points; vanishing variance gives the solution at q.

Timing (microseconds per period of `obc_ho_simulate`, order 2, horizon 40 + check-ahead 40, 100 periods, minimum of 5 repeats, one BLAS thread, `cache` figures are cold start
including building the cache over the 100 periods; shared 2-core machine with a soak test running at nice 19, load average 1.6-1.9 during the run; script /tmp/ho/bench1.jl, not in the repository).
nk = test/obc/ho/nk_occ.mod with the DynareOBC shock sequence (100 periods: all bind); NKM = test/ref_occbin/occest_nkm.mod (Dynare's tests/occbin/filter) with shocks 5 sd, seed 3
(93% of the periods have wedges at level 11). "relevant" = average relevant points per period.

| model | level | points | relevant | loop only (existing scalar solver) | + cache | + grouping | pydyn (full average, level 11; its figures: nk ~14 ms, NKM ~3.5 ms) |
|---|---|---|---|---|---|---|---|
| nk | 9 | 1,023 | 735 | 4,950 | 1,455 | 1,651 | 4,270 (L9, regions all, nk) |
| nk | 11 | 4,095 | 2,943 | 20,217 | 5,820 | 6,199 | ~14,000 |
| nk | 13 | 16,383 | 11,744 | 79,748 | 23,359 | 25,163 | - |
| NKM | 9 | 1,023 | 370 | 2,569 | 589 | 563 | 1,958 (L9) |
| NKM | 11 | 4,095 | 1,484 | 9,728 | 2,261 | 2,285 | ~3,500 (65% relevant there; here 36%) |
| NKM | 13 | 16,383 | 5,939 | 38,137 | 9,459 | 9,948 | - |

Per relevant point: nk 6.9 us (loop), 2.0 us (cache); NKM 6.6 us (loop), 1.5 us (cache). The cache buys 3.4-4.2x over the loop; grouping buys nothing in Julia (it is 0-10% slower on nk, equal on NKM)
because the per-point solve is already a few microseconds; the default is `:cache`. Sets in the cache after 100 periods: nk 2,931 / 6,418 / 12,362 (levels 9/11/13), NKM 655 / 1,129 / 1,806.
Maximum difference of the simulated wedges between the three modes: 1.7e-16.
Note: the NKM rows are not at pydyn's shock scale (pydyn has 65% of points relevant): per relevant point the NKM cost is ~1.5 us against pydyn's ~2.4 us (3.5 ms / ~2,660 points).

### Steps 2 and 3
Defaults with cubature on: `obc_cubature_regions = all`, `obc_qmc_level = 11` (4,095 points); `obc_ho(...; qmc_level=11, regions="all")`. Tests that relied on one region at level 9/10 now pass `regions = 1` (and the level) explicitly; the DynareOBC checks (regions 1, level 9,
injected root and points) are unchanged and hold. The M diagnostic: `cub_m_diagnostic(M, n_c; level)` reuses `m_diagnostics` (level `:s`: S-matrix LP and one leading P block, `is_p = nothing`; `:full`: exact P tests); key = hash of M. It is one-way: a non-S matrix means some paths have no
solution; an S-matrix does not guarantee that every path has one (pydyn's borrowing-constraint example is S and still has failing points).


### Steps 4 and cross-port checks
Failure handling (`_cub_classify`, `_cub_all!`, `_cub_one!`, `CubLP`, `CubatureFailure` in src/obc_cubature.jl and src/obc_ho.jl): as pydyn's `_classify`: LP on the head rows (class a), LP on all rows (class b), then the MILP at horizons T/4, T/2, T
(`solve_news(...; diagnose=false)`, new keyword) with one time budget per point (`obc_cubature_milp_time_limit`, default 1 s; 0 gives class c at once), a solution returns y, a time-out is class c, otherwise b if the head-rows MILP at
horizon T finds a solution, else a. Tolerance 1e-6 max(1, |p|); an LP feasible by less than that goes to the MILP. The two LPs are native HiGHS models built once per parameter set (`CubLP`: only the row lower bounds change per point,
HiGHS keeps the basis), through the C API already used in src/highs.jl. `ObcError` gained a `detail` field (the `CubatureFailure` payload: period, n_failed (lower bound when the search stopped early), n_relevant, n_points, counts a/b/c,
n_active_set_failed (upper bound on the failures; `nothing` for one region), tolerance, advice); the 3-argument constructor is unchanged. Tolerance mode: result = (sum over solved relevant points) / (P - F); F > tolerance x (relevant points)
raises; `obc_ho_simulate` returns `dropped` and `relevant` per period, one warning per simulation, per-period counts in `news.infos`, class totals in `stats` (`failed_a/b/c`, `dropped_points`, `dropped_periods`).
What (b) does not prove: it says that no y >= 0 meets the gap conditions on all N rows while the head rows alone can be met; it does not establish that a longer horizon has a solution (on borrcon with horizon 96 + 96 no point fails; that is evidence, not proof).
One-way diagnostic note: an S-matrix M (borrcon) does not guarantee that every path has a solution.

Cross-port against pydyn run directly (pydyn 0abe8e7), the same root R and points xi injected from JDyn, the same gap paths q from JDyn's simulation (20 per case), full average, horizon 40 + 40 (nk, NKM) or 32 + 32:
| case | max abs difference of the wedges (tolerance 0) | tolerance 0.95: wedges / dropped counts | tolerance-0 failure payload (a, b, c, n_failed, lower bound, active-set failures, relevant) |
|---|---|---|---|
| nk level 9 | 1.5e-16 | 1.5e-16 / - | no failures |
| nk level 11 | 7.8e-16 | 7.8e-16 / - | no failures |
| NKM level 9 | 4.4e-13 | 4.4e-13 / - | no failures |
| NKM level 11 | 4.4e-13 | 4.4e-13 / - | no failures |
| borrcon level 5 (class b) | all 20 q raise in both | 6.9e-17 / dropped counts equal in 20 of 20 | identical in 20 of 20 |
| sw_cp level 5 (class a) | all 20 q raise in both | 1.9e-13 / dropped counts equal in 20 of 20 | identical in 20 of 20 |
(NKM: 4e-13 is the difference of the two ports' M and q, not of the solvers.) Script: /tmp/ho/xport_jl.jl + xport_py.py (not in the repository).

Failure-handling costs (level 11, 4,095 points, 8 periods, minimum of 5 repeats, one BLAS thread, shared machine, load average 1.6-1.9; pydyn's figures from its doc in brackets):
| model | mode | JDyn | pydyn |
|---|---|---|---|
| borrcon (13% of points fail, class b) | raise (tolerance 0) | raised in period 1 after 33 ms (526 active-set failures) | 359 ms |
| | tolerance 0.2 / 0.7 | 114 / 110 ms per period; 4,318 of 32,760 relevant points dropped (13.2%) | 2.5 s per period; 13.1% (4,275 of 32,749) |
| sw_cp (class a) | raise | raised in period 1 after 4 ms (1,341 active-set failures) | 42 ms |
| | tolerance 0.2 | raised in period 1 after 76 ms (>= 450 failed points counted) | 1.14 s (507) |
| | tolerance 0.7 | 268 ms per period; 12,905 of 22,512 dropped (57%) | 3.5 s per period; 58.2% |
Per classified point (tolerance mode, everything included): 0.17-0.21 ms, against pydyn's 2.3-5 ms. No MILP was needed in these runs (as in pydyn); the MILP step is tested by forcing it (an always-feasible LP).


### Step 5: two constraints at higher order
`obc_ho` accepts every model with complementary pairs (OccBin allows two constraints). Layout: wedge vector `nT = nc T` stacked constraint by constraint (as `sys.M`); the rows of `q`: the head rows (`nT`, constraint-major) then the rows
beyond the horizon (`nc (N - T)`, constraint-major) = the layout of `Mf = [M; Mb]`; `h.rows[t, c]` is the row of constraint `c` in gap period `t`, `h.aC/aB/aA/aJ` are `nc x n` (`nc x nx`), `h.a0` a vector, `h.nT`, `h.NR`.
The root is built per constraint (the gap i periods later responds like the gap now to a shock now). Single-constraint results are unchanged (the layout reduces to the old one; all earlier tests pass).
Tests (test/test_obc_ho.jl, obc_zlb2.mod, horizon 24 + 24, shocks 5 sd): order 1 equals JDyn's own order-1 news simulation (levels 1.0e-14, wedges 1.4e-14 at 100 periods, both constraints bind together in 46); the affine map of q equals the recursion + gap
definition for both constraints (< 1e-12); the three cubature modes agree (<= 9e-16), the full average equals the loop over the relevant points. Cross-port, pydyn with the same root and points, level 7, 20 gap paths: 1.8e-15.
Timing, zlb2 (two constraints), per period, 100 periods, minimum of 5: level 9: loop 22.1 ms, + cache 6.0 ms, + grouping 6.7 ms (764 relevant points, 15,579 sets cached); level 11: loop 81.1 ms, + cache 26.0 ms, + grouping 26.0 ms
(3,053 relevant, 42,657 sets); pydyn's figure for zlb2 at level 11: 99.8 ms. As pydyn found, the sets hardly repeat with two constraints (the cache still gives 3x here: 15,579 sets for 76,400 solves).

## Step 6: regions = k (DynareOBC's splitting)
`obc_cubature_regions = k > 1` (`_split_regions`, `_cub_regions!`, `_cub_unit` in src/obc_ho.jl): the relevant paths are described by their direction and the direction of their negative part, reduced by Gram-matrix principal components
(components below pruning_cutoff x the largest variance dropped, at most max_cubature_dimension kept), the region with the largest within-sum-of-squares is split at Otsu's threshold along its first principal axis until there are k regions;
each region contributes (its size) x (solution at q + R mean(xi of the region)); failures classified as for points, a failed region counts all its points (`unexamined` = regions not yet solved). Fewer relevant points than regions: each is its own region.
Not default; like one region it does not price tail risk (documented). Validation (test_obc_ho.jl "four regions", nk, horizon 32+32, level 9, regions=4, DynareOBC's root and points injected, ref_o2cub4.mat): periods agreeing to < 1e-12 are counted against
the reference (JDyn: 294 of 300 agree to < 1e-12, six differ up to 0.0061; pydyn: 294 of 300, six differ by up to 0.006 from near-ties of the Otsu split); the test requires >= 290 and max < 0.01.


## Rule memo budget (maintenance job 1)
Background (pydyn soak test, `soak/repro/occbin_rule_trie_memory.py`): at extreme parameter vectors the inversion filter's root search visits hundreds of thousands of distinct regime
guesses and `_RuleTrie` memoised them all. Reproduced in JDyn: NKM (Dynare's tests/occbin/filter, `occest_nkm.mod`), vector (109.65435, 2.234926, 0.645373, 0.660980, 0.9999999229, 0.78819, 0.0042891, 0.0101122, 6.0573e-05), news not involved
(`obc_method = occbin`), inversion filter: **231,173 nodes, `Base.summarysize` 1,294 MB, peak RSS 2.3-2.6 GB, 15-23 s, ends with code 304** (pydyn: 232,740 nodes; the count varies with the random root search).
Note: pydyn builds `Occbin` with its own default `qz_criterium` (1 + 1e-6), JDyn passes the estimation's (1 - 1e-6, Dynare's choice for lik_init 1); at the vector above rhos = 0.99999992 exceeds
1 - 1e-6, so JDyn stops with a Blanchard-Kahn failure (code 3) unless `qz_criterium` is set to 1 + 1e-6 as in the reproducer (done in the test and the scripts). A difference between the ports, not changed.

Change (`src/occbin.jl`): `OCCBIN_RULE_MEMO_BYTES` (a `Ref`, default 128 MiB) is the budget per Occbin object, counted as 8 (n^2 + n n_exo + n) bytes per stored rule. A rule that would exceed it is computed
by the same arithmetic and used, but not stored; the rest of that guess is computed the same way (the parent is not a node), as the un-memoised code. `_RuleTrie` gained `bytes` and `refused`;
`rule_memo_stats(ob)` gives `(nodes, bytes, refused)`; `_RULE_MEMO_PEAK`/`reset_rule_memo_peak!()` the high-water mark of one trie. Nothing else changed.
The Python port (read only) has the same idea with a byte budget of 384 MiB (`rule_memo_bytes`); JDyn's default is 128 MiB (see below), the semantics are the same.

Numbers (machine shared with another agent, load average 2-3 on 2 cores, noted per measurement):
- Largest trie of any Occbin object over `test/test_occbin_memo.jl`'s battery (NKM 4 vectors, small NKM 3, rbc two constraints with PKF and smoother, bf 6 vectors, simulations of rbc_occbin, obc_zlb, nk_zlb2, obc_zlb2): **858 nodes, 0.52 MiB**.
  Over 300 NKM evaluations in a random walk (1% perturbations): 15-16 nodes (80-86 kB) per evaluation. Default 128 MiB = 245x the largest ordinary trie (>= 20x asked); about 24,000 nodes at the NKM's 5.4 kB per node.
- Extreme vector, one evaluation (including the same random search; times vary by +-30% with it): before 15.5 s / peak RSS 2,338 MB; default 128 MiB 11.1 s / 951 MB; 16 MiB 19.4 s / 737 MB (3,120 nodes);
  budget 0 (no memo at all) 39.8 s / 746 MB. The baseline RSS of a JDyn session is about 700 MB, so the memo adds at most the budget plus garbage. The memo still pays off (no memo is 3x slower),
  so the budget is not set small. The failure itself (code 304) still takes 11 s: the budget fixes memory, not the length of the search.
- Lifetime: `_RULE_TRIES` is a WeakKeyDict on the Occbin object, and the filter builds a new Occbin per evaluation, so tries are released with it: after 300 evaluations (full GC every 50) live tries 0, GC live bytes flat at 36.6 MB, RSS flat at 714 MB. No accumulation.
- Normal path, one `log_posterior` (min of 5, ms, NKM vectors 1-4 / small NKM 1-3), before (f477d7d) vs after, load average about 2: 68.9 75.2 65.0 69.2 / 47.2 46.8 48.7 against 78.1 63.8 64.1 67.6 / 43.7 48.0 47.9: no measurable difference (noise +-10%).
- Bit for bit: `test/test_occbin_memo.jl` (12 + 3 + 3 assertions, ~2 min) runs the whole battery (57 results: log posterior and its failure code, every field of the inversion filter result with a fixed `rng`, PKF result, smoother, simulations)
  under the default budget, budget 0, 40 kB (about 7 nodes) and 400 kB, and compares with `same` (Float64 bit patterns, `isequal` structure): all identical, and the trie never exceeds the budget. The last testset is the reproducer
  (budget 16 MiB: code 304, trie <= 16 MiB, 12 s). It is not "fast" (the failing search itself takes 11-12 s); the whole file takes about 2 minutes.


## Horizon too short in the news inversion filter (maintenance job 2)
Semantics follow what pydyn's other agent implemented (pydyn-soak commit 68826e3: `pydyn/obc_filter.py`, `tests/test_obc_horizon_cause.py`), ported to JDyn.

Failure causes of a trial point (`OBC_FAILURE_CAUSES`, src/obc_filter.jl): `:horizon_too_short` (LCP: the linear programme on all rows is infeasible while the one on the head rows is feasible; regime solver / integer programme:
its own status), `:no_solution` (proved: the head rows cannot be met either; regime solver: no solution, integer programme attempted), `:undecided` (active set did not decide, no integer programme run, the default; or the
integer programme found nothing without a proof), `:milp_timeout`, `:numerical` (code 311). The numeric code of the filter stays 313 per trial point and 304 for the period; defaults, values and codes unchanged (news log posteriors at
all stored NKM / small NKM vectors compared bit for bit with commit f477d7d: identical, also the filter paths and the extreme vector's -Inf/304).
- `_SolverStats` (`solver.stats`) gained `causes` (cumulative), `period_causes`, `retries`, `retried_ok`, `retry_seconds`, `period`, `last_cause`.
- `InversionFilterResult.failure` (new last field, `nothing` unless code 304 with a news solver): `ObcFailure(period, primary, counts, horizon, horizon_max, advice)`. `primary` = `:horizon_too_short` if any trial point of the failing
  period failed so, else the most frequent cause, or `:root_search` when no trial point failed in the solver. `advice` e.g. "news horizon 40 too short at period 66: try obc_horizon >= 80 (or set obc_horizon_max to retry with a doubled
  horizon)" (pydyn's wording; JDyn adds "(obc_horizon_max = X was tried)" when a maximum was set).
- Estimation: `obc_failure(est)` returns it (`nothing` after a success and for obc_method = occbin); `est.last_info` becomes "inversion filter failed in period 66 (info 304); news solver: undecided (...)" (only for the news solver);
  `est.last_info_code` is still 304. `begin_period!(solver, t)` / `failure_info(solver)` are the hooks of `inversion_filter` (no-ops for OccBin's own solver).
- `obc_horizon_max` (default off = 0): a trial point that fails with `:horizon_too_short` or `:undecided` is solved again with the horizon doubled (40 -> 80 -> ... up to the maximum, `check_ahead = max(h, check_ahead)`), by a larger
  solver built once and kept in `solver.retry.big` (per parameter vector, since the solver is rebuilt with the Occbin object); a retried success is not counted as failure (errors[313] and causes decremented, as pydyn),
  `retries`, `retried_ok` and `retry_seconds` (time in retries, building the larger solvers included) report the cost. A retry stops when the larger solver fails for another cause. Routes: keyword `obc_horizon_max=` of `Estimation`,
  `occbin=Dict("obc_horizon_max" => 80)`, native `options_.occbin.obc_horizon_max = 80;`, `set_obc!(est; horizon_max=80)` (0 = off). Refusals (ArgumentError): not a positive integer or 0, below `obc_horizon` (also when the
  horizon is raised later), a model without constraints (EstimationError). It is not an option of `simulate` (a simulation raises `:horizon_too_short`).
- Test: `test/test_obc_horizon.jl` (13 + 10 + 28 + 279 assertions, ~35 s).

Numbers (load average ~2). pydyn's A#75 vector (NKM): OccBin -210551.27794614976 (pydyn's: ...947312, difference 1.2e-6 = 5e-12 relative: the ports' arithmetic); news horizon 40: -Inf, code 304 at period 66 (pydyn: period 66), cause
`undecided` for all 110 failed trial points (pydyn: horizon_too_short or undecided); `obc_horizon_max = 80`: -210551.27794476267, **|news - OccBin| = 1.4e-6 = 6.6e-12 relative**, 161 retries all solved, 0.04 s in retries (0.27 s for the
evaluation). `obc_horizon_max = 160` at the four stored NKM vectors: bit for bit the default's values. borrcon (long binding spells), horizon 20, 400 random trial points: 121 failures, all `horizon_too_short`, in the LCP and in the regime
solver alike; with `obc_horizon_max = 80`: 28 failures left, 93 of 192 retries solved.
Not proved: that `undecided` at the extreme vector is a horizon problem (it is the empirical finding that horizon 80 solves them); the status does not say so because the cheap checks cannot decide it.
Differences from pydyn: the integer-programme outcomes `:horizon_too_short` / proved `:no_solution` are used when `obc_milp_time_limit > 0` (pydyn counts them `undecided`); `ObcFailure` has `horizon_max`; `retry_seconds`.
Not aligned: pydyn builds `Occbin` with its default `qz_criterium` (1 + 1e-6) in the estimation, JDyn with the estimation's (1 - 1e-6 for lik_init 1, as Dynare).


## Merge of main (maintenance job 3)
Merge commit of `origin/main` 493b529 (76 commits since a487e5d). Conflicts: `src/JDyn.jl` (export list: kept both), `test/runtests.jl` (test list: kept both, plus the two new test files), `src/occbin_estimation.jl`
(OccBin smoother's inversion filter: main's `rng=_ivf_rng(m)` together with this branch's news `solver=` hook and `horizon_max`). Everything else merged automatically (`occbin_filter(est, theta; rng=_ivf_rng(est.model))` merged cleanly with the failure bookkeeping).
One fix of main's code, needed by a test (src/estimation.jl, 2 lines): blank bounds in `estimated_params` are NaN and main's `est.lb, est.ub = max.(plb, lb), min.(pub, ub)` propagated NaN, so `sample_posterior` (which now checks proposals against `est.lb/est.ub`)
rejected every proposal (acceptance 0.0 in the small-NKM chain test; chain stuck at its start). Now NaN means "use the prior bound". Main's session should take or replace this.
`occbin_solver` refusing `loglinear`: no OBC/OccBin test sets loglinear except `test_occbin_estimation.jl`, which already expects the refusal.
`obc_method = occbin` is bit for bit main's own OccBin: log posterior (hex) and filter results (filtered shocks, states, residuals, regimes, code; fixed rng) at the 4 NKM, 3 small-NKM stored vectors and 6 bf vectors, 26 comparisons, identical between main 493b529 and the merge.
bf replay in `test/test_obc_estimation.jl`: `nmis <= 10` tightened to `nmis == 0`; it holds (0 disagreements in 'fails or not' over the 100 replayed states) now that each evaluation's random search is seeded.
