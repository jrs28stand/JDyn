# Higher-order news OBC (obc_order = 2) and cubature: progress log

Branch `obc-ho` (pushed to `origin wip/obc-ho`), based on `wip/obc` at 5f0d3a0. Specification: pydyn's prototype
`pydyn/obc_ho.py` (`HigherOrderNews`), design in pydyn `docs/OBC_PROGRESS.md` and `reference/dynareobc_octave/README.md`.

## State / next step

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
