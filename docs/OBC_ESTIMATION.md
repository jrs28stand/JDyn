# obc_method in estimation (news solver in the inversion filter), JDyn port of pydyn's `obc-est`

Specification: pydyn (merged branch, read-only) `pydyn/obc_filter.py`, `pydyn/occbin_estimation.py`, `pydyn/occbin.py` (`canonical_obc`),
`pydyn/obc.py` (selection), `docs/OBC_ESTIMATION.md`, `docs/OBC_CONTRACT.md` (rules 5a, 7), `tests/test_obc_estimation.py`, `tests/test_obc_selection.py`.
Branch `obc-ho` (pushed to `origin wip/obc-ho` only).

## State / next step

See "Progress" at the end. Plan (written before coding):

1. Options (step a). `obc_method` (`occbin` default, `news`, `minnorm`), `obc_horizon` (40), `obc_selection` (`occbin`, `shortest`, `longest`, `min_wedge`),
   plus `obc_milp_time_limit` (0). Routes: keywords of `Estimation(...)`, `occbin=Dict("obc_method" => ...)` / the workspace's `options_.occbin`, native
   `options_.occbin.obc_* = ...;` lines. Stored as `est.options["obc"]` (`ObcEstSettings`), changed with `set_obc!`. `minnorm` in the *estimation* is pydyn's
   alias: `news` + `min_wedge` (a conflicting explicit selection is an error). JDyn's simulation `minnorm` (Kockerols-style programme) is untouched.
   News needs the inversion filter (refused with the piecewise Kalman filter, at construction, in `set_obc!`, in `occbin_filter` and in the smoother).
   `obc_selection` other than `occbin` with `obc_method = occbin` is an error at filter time. Selection needs the single-spell enumeration (new in JDyn).
2. Filter solver (step b). `inversion_filter(...; solver=)`; `LCPFilterSolver` (`filter_solve`) = pydyn's `LCPFilterSolver`: `q = q0 + Qx [state; shock]`, active set,
   the check rows beyond the horizon, then the linear programme proof of non-existence, the integer programme only if `obc_milp_time_limit > 0`; failure = 313;
   (T, R, C) cached per binding set.
3. Regime solver (step c). `RegimeFilterSolver` on `RegimeOBC._regime_step` (new keyword `milp=false` skips the integer programmes), rule cached per binding positions.
4. Orientation (step d): compare JDyn's classifier with pydyn's `test_orientation_rule_7` cases.
5. Validation: JDyn's own OccBin values on the NKM example (`test/ref_occbin`), pydyn run directly, bf in Dynare's `tests/estimation/slice`; timing.
Also (Task 1): the higher-order options `obc_order`, `obc_cubature`, ... from the mod file, workspace and keywords.

## Progress and results

Steps a-d are implemented in `src/obc_filter.jl` (new), with small wiring in `src/estimation.jl`, `src/occbin_estimation.jl`, `src/occbin_filter.jl`
(`inversion_filter(...; solver=)`), `src/obc.jl`, `src/obc_regime.jl` (`milp` keyword), `src/occbin.jl`, `src/results.jl`. Tests: `test/test_obc_estimation.jl`
(options, minnorm alias, native lines, NKM small/full, bf/bf_ppf, selection, rule 7, timing) and the Task 1 testsets of `test/test_obc_ho.jl`.

Numbers (this machine):
- NKM (small and full), news vs JDyn OccBin, log posterior at the stored vectors: 1.4e-12 .. 6.4e-12; against Dynare's stored `ivf_fval` within 1e-11 relative;
  same regime histories, filtered shocks and states (< 1e-9). Against pydyn run directly (news): same values to about 1e-11 (e.g. NKM vector 4: 1533.33221647431 both).
- Vector 4 of the full NKM: solver statistics equal pydyn's (fallbacks 2, proved infeasible 1, integer programme skipped 1, rescued 0, errors 313 x 2).
- bf and bf_ppf: `RegimeFilterSolver`, 8 parameter vectors each: log posterior within 1e-9 of OccBin (observed <= 2.3e-13), same regimes, first-period (T, R, C) equal.
- One likelihood evaluation, NKM: JDyn OccBin 72-93 ms, news 3.7-4.0 ms (about 19-23x; pydyn on the same machine: 146 ms vs 34 ms, 4x). Small NKM: pydyn 121 vs 33 ms.
- `obc_method = occbin` (default) is bit-identical to the pre-change code (hex floats compared at 7 vectors against commit 5f0d3a0 of jdyn-news).

Selection: `longest` fails in the NKM filter (root finder hops to the 28-period trap solution), as in pydyn; `shortest` and `min_wedge` equal the default to 1e-9.

Deviations from pydyn:
- `obc_selection` other than `occbin` is refused for the regime solver (bf) and for the higher-order simulation route; pydyn supports it in the regime solver.
- Simulation `selection` is implemented only for the LCP news loop with one constraint.

`minnorm`: in the estimation it is an alias of news + min_wedge. JDyn's simulation `minnorm` was left alone. On `test/obc/obc_zlb.mod` the simulation
`minnorm` and news + `min_wedge` agree exactly (0.0); that is one example, not a general equivalence (one is a quadratic programme over all wedge profiles,
the other picks among single-spell solutions), so aliasing the simulation could change results elsewhere; not checked further.

Not done: MILP-on filtering is only tested on NKM vector 4 (`obc_milp_time_limit = 2`); no estimation run with the optimiser under news (only likelihood evaluations).
