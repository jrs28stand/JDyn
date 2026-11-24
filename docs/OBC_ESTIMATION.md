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
