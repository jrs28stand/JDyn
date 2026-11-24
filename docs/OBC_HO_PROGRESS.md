# Higher-order news OBC (obc_order = 2) and cubature: progress log

Branch `obc-ho` (pushed to `origin wip/obc-ho`), based on `wip/obc` at 5f0d3a0. Specification: pydyn's prototype
`pydyn/obc_ho.py` (`HigherOrderNews`), design in pydyn `docs/OBC_PROGRESS.md` and `reference/dynareobc_octave/README.md`.

## State / next step

State: H1 (order 2), H2 (tests) and C1 (cubature) are implemented; see the results below. Next: see "Open" at the end.
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
