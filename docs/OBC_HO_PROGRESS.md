# Higher-order news OBC (obc_order = 2) and cubature: progress log

Branch `obc-ho` (pushed to `origin wip/obc-ho`), based on `wip/obc` at 5f0d3a0. Specification: pydyn's prototype
`pydyn/obc_ho.py` (`HigherOrderNews`), design in pydyn `docs/OBC_PROGRESS.md` and `reference/dynareobc_octave/README.md`.

## State / next step

CUBATURE PORT (current task; spec: pydyn docs/OBC_HO_PERF.md). DONE and pushed: steps 1-4 (full average with cached factorisations; defaults full average + level 11; up-front M diagnostic;
failure handling: classes a/b/c, raise by default with payload `e.detail::CubatureFailure`, tolerance mode, native HiGHS LP per parameter set). Cross-port checks vs pydyn done for steps 1 and 4 (see
"Steps 4 and cross-port checks" below). NEXT: step 5 (two constraints at higher order: gap maps stacked per constraint, wedge vector n_c T, root per constraint; validate order 1 against the order-1 news
simulation on obc_zlb2.mod, cubature batch vs scalar), then step 6 (regions = k, lowest priority; reference ref_o2cub4.mat in pydyn's reference/dynareobc_octave), then a mod-file route test for the new options
(`obc_m_diagnostic`, `obc_cubature_milp_time_limit`, `obc_cubature_failure_tolerance`) and the final report.

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
