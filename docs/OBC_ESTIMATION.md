# obc_method in estimation (news solver in the inversion filter), JDyn port of pydyn's `obc-est`

Specification: pydyn (merged branch, read-only) `pydyn/obc_filter.py`, `pydyn/occbin_estimation.py`, `pydyn/occbin.py` (`canonical_obc`),
`pydyn/obc.py` (selection), `docs/OBC_ESTIMATION.md`, `docs/OBC_CONTRACT.md` (rules 5a, 7), `tests/test_obc_estimation.py`, `tests/test_obc_selection.py`.
Branch `obc-ho` (pushed to `origin wip/obc-ho` only).

## State / next step

State: steps a-d, the follow-up items 1-4 (rule-7 cases, `obc_selection` in the regime solver, estimation end to end on NKM and bf) are done and tested; see
"Progress and results" and "End-to-end estimation" below. Next step: none open; piecewise Kalman filter / smoother under news are not wanted.
Plan (written before coding):

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

## Follow-up: rule 7, selection in the regime solver, estimation end to end

1. Rule 7. pydyn's five `test_orientation_rule_7` cases (bind `rn < rbar` with relax `rn >= rbar`, none, `rn < rbar`, `rn <= rbar`, `rbar > rn`) give the same
   classes in JDyn (complementarity, complementarity, then three times inconsistent): no case differs. Aligned: the reason now says "the relax condition points the wrong way"
   (pydyn's text), and the tests check it. Tolerance: JDyn uses 1e-6 x max(|a|,|c|) for both inequalities, pydyn a tolerance per term; they can differ only for a model that
   sits within rounding of the boundary of the rule. Not tested.
2. `obc_selection` in `RegimeFilterSolver` (`_regime_step(...; selection)`, `_regime_spells`; one constraint). Values on bf at 6 deterministic vectors
   (x0 * exp(0.05 sin(1.3 k j + k)), k = 0..5) equal pydyn's run directly to <1e-8 for all four selections: 566.5512559450814, 476.0457580673122, 536.164073136107,
   -Inf, -Inf, 564.0576327606426 (JDyn/pydyn OccBin: ...0813 / ...0814). All four selections coincide on bf (the solutions are unique there), so this checks plumbing and the
   spell enumeration, not a choice between different equilibria. The simulation with the regime formulation still refuses `obc_selection`.
3. End to end (a script outside the repository; csminwel as pydyn: crit 1e-7, `mode_compute=4`; MCMC with identical random numbers, proposal and starting points; two cores, runs timed
   one after another while another job was running):

   Mode, small NKM, start prior mean: OccBin f = 1518.4359618, 99 evaluations, 17 iterations, 16.8 s; news f = 1518.4359618, 118 evaluations, 19 iterations, 2.5 s (6.7x; 20.7 vs 167.6 ms per
   evaluation). Modes differ by 5.6e-6 (pydyn: 5.5e-6; 115 vs 133 evaluations). Paths separate at evaluation 5 (pydyn: 5). The log posterior differs by up to 0.0265 at 4 of the visited
   vectors (knife edge). No vector fails in one solver only.

   Mode, full NKM, start prior mean: the searches end at different points (as pydyn): OccBin 1692.8769 (1226 evaluations, 52 iterations, 136.6 s), news 1692.9898 (859 evaluations, 41 iterations, retcode 6, 8.9 s;
   15x). Modes up to 0.06 apart (the surface is flat in several parameters; the Hessian is not positive definite). Paths separate at evaluation 30, but along OccBin's own path the first
   likelihood difference >1e-9 is at evaluation 170 (0.124 at most), so the separation at 30 is rounding (1e-12) amplified by the finite-difference gradient, not a knife-edge period.
   Replayed at the 1226 vectors of OccBin's search news agrees on all but 52 (max 0.124) and on which vectors fail (40 fail in both, with the same codes); at the 859 vectors of the news search OccBin differs at 65 (max 0.124),
   again with the same failures. The news solver in its own search hit 340 fallbacks, 298 proofs of infeasibility, 42 skipped integer programmes (all 340 inside the filter's root search, none changed the final code).
   Start at the first stored vector: OccBin 1687.0596331 (433 evaluations, 18 iterations, 39.5-51.9 s), news 1687.0596438 (437 evaluations, 18 iterations, 2.9-5.0 s; 11-14x); modes 3.8e-7 apart.

   Metropolis-Hastings, small NKM: 2 chains x 2000 draws, seed 20261124, jscale 1.3 (pilot acceptance 0.77/0.49/0.34/0.22 at 0.4/0.9/1.3/1.8), proposal inverse Hessian at OccBin's mode. Draws identical
   draw for draw in both chains, acceptance 0.331 and 0.338 for both solvers (pydyn 0.357, 0.338), 31 and 24 evaluations rejected with Dynare's code 3 (prior bounds, both). Wall time per chain
   143.5 / 154.7 s (OccBin) vs 10.0 / 10.3 s (news): 14.3x and 15.0x. Evaluations at identical vectors: 2000 of 2000; likelihood differs by more than 1e-9 at 5 and 2 of them (max 0.0243 and
   0.0446; first at evaluation 198 and 549: 1514.210225 vs 1514.233739, 1516.986911 vs 1517.031478; pydyn: 0.0244 at evaluations 56 and 46). None flips an accept decision. Fallbacks 0.
   Full NKM: 2 chains x 1500, jscale 0.07, prior variance proposal, start around OccBin's stored-start mode: identical draws, acceptance 0.219 / 0.223 for both (pydyn 0.19, 0.20), 134-136 s vs 9.8 s (13.5-13.8x);
   differences >1e-9 at 10 and 3 evaluations (max 0.0274, 0.0374); no evaluation fails in one solver only.
4. bf with the regime solver: 2 chains x 1000, jscale 0.15, prior variance: identical draws, acceptance 0.331 / 0.354 for both, likelihood differences <= 2.3e-13 at every evaluation,
   55-63 evaluations per chain fail with code 304 in both. Wall time 32.7/35.4 s (OccBin) vs 13.3/12.0 s (news): only 2.5-3x, because every failing evaluation costs about the same in both and the
   regime solver cannot prove infeasibility by a linear programme there (about 3000 fallbacks per chain, all with the integer programme skipped).

Reproducibility note: whether a trial point fails (304) is not reproducible even for one solver: the filter's root search restarts from random points drawn from the model's Dynare generator, which
advances with every call. Repeated calls at the same vector changed the outcome at 4 of 100 visited states (OccBin or news). The sampler comparisons are unaffected because both
chains make the same sequence of calls.

Tests (test/test_obc_estimation.jl): the small-NKM mode search and a 2 x 150 chain pair under both solvers (about 55 s, 206 assertions: modes within 1e-3 posterior sd, f equal to 1e-6, draws identical),
and a 2 x 150 bf chain pair (about 30 s, 100 assertions).
