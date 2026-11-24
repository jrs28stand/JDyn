# OBC methods (`obc_method`): progress log (branch `obc`)

Clone: /home/claude/work/jdyn-obc (base `main` a487e5d). Contract: /home/claude/work/OBC_CONTRACT.md (2026-11-24,
amended the same day after pydyn's `news`). Backup: `git push origin obc:wip/obc`. Scope: order 1, the shared
machinery and `minnorm`; `news` is a marked stub (`solve_news` in src/obc_solvers.jl) that throws.

## Status (2026-11-24)

| Deliverable | State |
|---|---|
| A. Workspace | done. Julia 1.12.7 (conda-forge), k-order library built. Before any change: `test_occbin.jl` 37/37 and `test_occbin_options.jl` 456/456 passed |
| B. Post-processor and classification | done: `obc_constraints`, `obc_classification` (src/obc.jl); table of Dynare's 11 OccBin files in docs/OBC_CLASSIFICATION.md, locked in by test/test_obc.jl (also checks the doc equals the generated table) |
| C. `M` and `q` | done: `obc_system`, `obc_q`, `obc_gaps`. Against a stacked perfect-foresight system: < 1e-15 (test bound 1e-10). Against Dynare 7.2 (`test/obc/obc_zlb.mat`, 60 x 20): `M` 1.1e-14, `q` 1e-16 (test bound 1e-12) |
| D. `minnorm` | done: `solve_minnorm` (Lemke + norm descent), shared fast path `solve_active_set`, `obc_verify`; stub `solve_news` |
| E. Interface | done: keywords `obc_method`, `obc_horizon`, `obc_tol`, `obc_raise`, `obc_fast_path` on `simulate(ob, ...)`, `simulate(ob)`, `irf_occbin`, `irf(ob, ...)`; native lines `options_.occbin.obc_method/obc_horizon`; `occbin_solver!(ws; obc_method=...)`; `oo_.occbin.news` |
| F. Tests | done: test/test_obc.jl (wired into test/runtests.jl) |
| G. Timing | done (below) |

Default `occbin` is bit-for-bit unchanged: simulations (file shocks, 200 random periods), `irf_occbin` and `dynare()` on
rbc_occbin, borrcon_sim, nk_zlb2, obc_zlb were serialised from `main` (a487e5d) and from this branch and compared with
`isequal`: all identical (including the two OccBin failures on rbc_occbin's random/IRF runs, same errors).

Full suite on this branch (`julia --project=. test/runtests.jl`, 2026-11-24, fresh process): 27 of 28 files pass,
including test_obc.jl (264 checks), test_occbin.jl (37), test_occbin_options.jl (456), test_occbin_estimation.jl (203),
test_results.jl (41). test_plotting.jl fails at load because it reads a hard-coded Dynare.jl checkout
(`/root/jl/src/Dynare.jl/test/models/example1/example1.mod`) absent on this machine; the file is unchanged from `main`.

## What is where

- src/obc.jl: `ObcConstraint`/`ObcAffine` (post-processor), `ObcSystem` (M, q0, Qx, cached per `Occbin` object and
  horizon in a `WeakKeyDict`), `obc_gaps` (path and gaps for given wedges), `_simulate_obc` (OccBin's loop with an LCP
  solver), `ObcNews`, `ObcError`, option parsing.
- src/obc_solvers.jl: `obc_verify`, `lemke`, `solve_minnorm`, `solve_active_set`, `solve_news` (stub), `OBC_SOLVERS`.
- Shared files (additive only): src/occbin.jl (`OccbinResult.news` field with the old 12-argument constructor kept; the
  `obc_*` keywords of `simulate` dispatch to `_simulate_obc` before any OccBin code runs; `mod_options` and
  `irf_occbin` read the native lines), src/results.jl (`occbin_solver!` reads `options_.occbin.obc_method/obc_horizon`
  and writes `oo_.occbin.news`), src/JDyn.jl (includes, exports), test/runtests.jl, NOTICE.md.
- test/obc/: test models (nk_zlb.mod: lower bound without `relax`, the contract's worked case; nk_zlb2.mod: two
  constraints with `relax`), pydyn's obc_zlb*.mod / obc_zlb.mat / r_obc_zlb.m (copied, read-only reference),
  classify_dynare.jl (generator of docs/OBC_CLASSIFICATION.md).

## Decisions

1. Construction of `M`: first-order solution with anticipated shocks (`h_t = F h_{t+1} - (B+AP)^{-1} f_t`), not a stacked
   system. Wedge of constraint j enters its reference equation as `ref_resid = s_y * y_j(t)`. `q = q0 + Qx*[state; shock]`
   with `Qx` precomputed (`n + n_exo` columns; the full state, not only predetermined variables).
2. Simulation semantics are OccBin's, not MacroModelling.jl's (see findings): each shock period re-solves from the
   endogenous state with all wedges new; periods without a shock continue the last path (OccBin's shift), including the
   wedge vector shifted by one. Paths have OccBin's length (`max(check_ahead_periods, periods - S) + 1`).
3. Beyond-horizon check: gaps over `T .. T + check_ahead_periods - 1` (`check_ahead_periods` capped by
   `max_check_ahead_periods`); a negative gap there gives `:horizon_too_short`.
4. Failures: `ObcError(status, period, msg)` by default; `obc_raise=false` keeps going and records the status in
   `news.status`. Refusal of unsupported models: `ModelError` naming the constraint, its class and reason, and pointing to
   `obc_method = occbin`.
5. `OccbinResult` for the new methods: `iterations` = solver iterations (0 in shift periods), `periodic` all false,
   `diagnostics[t].max_err` = verification violation, `ss[t]` = OccBin's time-varying rule for the regime found (computed
   with occbin.jl's `_rules`, so the OccBin filters could consume it). Regime history from `y > obc_tol`.
6. OccBin's regime-iteration options (`maxit`, `curb_retrench`, `periodic_solution*`, `init_regime`,
   `init_binding_indicator`, `reset_*`, `algo_truncation`) are recognised and not used: one warning listing them.
7. Tolerance `obc_tol = 1e-10` (absolute) for verification and for "binding".
8. No new dependency. `minnorm` needs only LinearAlgebra (Lemke's algorithm and LU solves on principal blocks).
9. `news` stub: `OBC_SOLVERS[:news] = solve_news`, which throws "not implemented yet". `_simulate_obc` calls the
   method's solver once on `(M, q0)` before simulating (q0 > 0, so any real solver returns y = 0 at once); this makes the
   stub throw even though the active-set fast path would otherwise solve every period without it.
10. Classification precedence: multi_equation, then shared_equation, then the fits (entry, exit, orientation), gap at
    the steady state. Equations = rows whose residual or Jacobian at the steady state changes with the parameter
    (`inert_equations` records rows that name it without changing): this is what order 1 switches, and it matches pydyn.
11. Gradients of the bind/relax conditions: complex step (exact for analytic expressions), central differences if
    the expression rejects complex input (e.g. `abs`).

## Contract amendments (proposed)

- 2026-11-24 (after the coordinator's amendment of rule 3): no change needed in substance. The first JDyn version read
  rule 3 as "`d_r < 0` means leave, `d_r = -d_b` without relax, `s_y = sign(c)`", which is the amended rule with `d_r`
  negated (identical `s_y`); it gave the contract's worked-case signs for the lower bound without `relax` (y = r - rn)
  from the start. The code now uses the amended wording (`d_r > 0` leave, `d_r = d_b`, `s_y = -sign(c)`) so that fitted
  coefficients compare directly with pydyn's.
- 2026-11-24: rule 1 should say "equations whose linearisation at the steady state changes" rather than "contain the
  parameter". Text containment makes rbc_occbin's INEG `multi_equation` (three tagged equations); at order 1 only the
  Euler equation changes (the budget constraint's extra term is quadratic, the investment equation's INEG factor is
  identical in both branches). pydyn already does this; JDyn now too, and lists the others as `inert_equations`.
- 2026-11-24: new class `shared_equation`: the constraint's equation also changes with another constraint's parameter
  (a model like `r = max(lo, min(hi, rule))` written with two constraints on one equation). Wedges on one equation from
  two constraints do not superpose (the both-binding version is a fourth equation), so it is refused like
  `multi_equation`. No Dynare file is in this class now (rbc_occbin's investment equation is the near miss, see above).
- 2026-11-24: rule 7, orientation. With fits passing, require `b*s_y <= 0` and `e*s_g >= 0` (`d_b = a alt + b ref`,
  `d_r = c ref + e alt`): carried into the other regime, the bind condition must not say "leave" while the wedge is
  positive, nor the relax condition "stay" while the gap is positive. Found on a two-constraint test model whose floor had
  `relax rl > r + sp` (i.e. "leave when the floor is above the shadow rate", the reverse of the bind condition): both fits
  pass, OccBin cycles (error 313) and Lemke ends on a secondary ray (no solution). It is now `inconsistent` with an
  explicit reason (test "OBC: refusals"). pydyn does not have this check.
- 2026-11-24: rule 6 tolerances adopted as amended (exact below 1e-6, `approx` up to 0.05). `approx` is set only on
  `complementarity` constraints (0 for others).
- Section 4, "wedge path zero from T on": kept. MacroModelling.jl has T + 1 wedges and checks T gaps (see findings); the
  contract's square version is what OccBin's fixed point corresponds to.
- Section 5, fast path: implemented as asked, with `obc_fast_path=false` to force the method's own solver. Note for
  `minnorm`: when the LCP has several solutions, the active-set answer need not be the smallest; with
  `obc_fast_path=false` the minimum-norm selection applies. In every test and timing run both gave the same answer.

## MacroModelling.jl findings (read at commit e8d444a, 2026-11-21; scratch clone /home/claude/scratch/mm)

Where: src/occasionally_binding_constraints.jl (`parse_occasionally_binding_constraints`, `write_obc_violation_equations`,
`set_up_obc_violation_function!`, `obc_objective_optim_fun`, `obc_constraint_optim_fun`, `obc_state_update`); called per
period from src/impulse_response_function.jl (IRFs and `simulate` go through `obc_state_update`).

- Specification: `max(a, b)` / `min(a, b)` in an equation. The equation is rewritten (symbolically solved for the
  max/min term) as `0 = max(χl, χr)`, then `Χ = max(χl, χr)` and the equation becomes `Χ - ϵ = 0`, so at the steady state
  (first order) it reads `χr = ϵ` for the active branch, e.g. `R = max(R̄, T)` gives `T - R = ϵ`.
- News shocks: per constraint, `max_obc_horizon + 1` shocks `ϵ⁽ᵏ⁾`, k = 0..H, delivered through a chain of auxiliary
  lag variables so that `ϵ⁽ᵏ⁾` drawn today hits the equation k periods later. Default horizon `max_obc_horizon = 40`
  (macro option of `@model`), so 41 wedges per constraint.
- Objective: `sum(abs2, x)` over the news shocks (gradient `2x`): squared wedges, in the units of the rewritten equation.
- Constraints (all `<= eps()` as NLopt inequality constraints, `upper_bounds = fill(eps(), ...)`), per constraint and
  for periods 1..H of the forecast (`Y[:, 1:periods]`, periods = H): `sum(χl .* χr)`, `s*χl`, `s*χr` with s = +1 for
  max, -1 for min. Since `s*χl <= 0` and `s*χr <= 0`, `sum(χl .* χr) <= 0` forces each product to zero: complementarity
  IS imposed, as one aggregate inequality. For `R = max(R̄, T)`: `R >= R̄`, `R >= T`, `(R - R̄)(R - T) = 0`.
- The wedge of period H (the (H+1)-th shock) is in the objective but no constraint mentions it or the gap in period
  H: it is sign-free and only enters through expectations of earlier periods.
- Solver: NLopt `LD_SLSQP`, `xtol_abs = ftol_abs = eps(Float32)`, `maxeval = 500`, start at `x = 0`, analytic
  Jacobian of the constraints (`compute_obc_analytical_jacobian!`). Only run when the unconstrained forecast violates a
  constraint by more than `eps(Float32)`; success = no constraint value above `eps(Float32)` afterwards.
- Simulation: the solved shocks are added to the period's shocks and the state is updated; the auxiliary lag chain
  carries announced future wedges into the next period's state, so the next solve adds new shocks on top of the old
  announced ones (it minimises the norm of the new shocks only).
- On failure: `@warn "No solution in period: t"` and the IRF/simulation loop `break`s; remaining periods are left at the
  zero-initialised value, i.e. the steady state, in the returned array.
- Higher orders and pruning use the same programme on the nonlinear forecast (out of scope here).

JDyn's `minnorm` and where it departs:
- Same programme on the contract's objects: minimise `‖y‖²` s.t. `y >= 0`, `g = q + M y >= 0`, `y .* g = 0`
  (complementarity imposed per period rather than as one sum; equivalent given the signs).
- Solver: no NLP. Lemke's complementary pivoting (covering vector 1, lexicographic ratio test) gives an exact LCP
  solution, recomputed by one LU solve on its basis; then a descent over neighbouring complementary bases (one index
  in or out) while a feasible solution of smaller norm exists. Unique solution (e.g. `M` a P-matrix): this is the global
  minimiser of MacroModelling.jl's programme. Several solutions: a locally smallest one; `info.alternatives` counts the
  other solutions met. SLSQP on this degenerate programme (MFCQ fails everywhere on the feasible set) gives no better
  guarantee, and its answer depends on its iterates.
- Units: wedges are `s_y * ref_resid` (the residual of Dynare's tagged reference equation) rather than MacroModelling.jl's
  rewritten equation; only matters for the selection among several solutions.
- T wedges and T checked gaps (contract), not H + 1 wedges with the last one unconstrained. Beyond T the gap is checked
  over the check-ahead length (MacroModelling.jl does not check beyond H).
- OccBin's simulation semantics (re-solve from the endogenous state, all wedges new) rather than "new shocks on top of
  announced ones". With a unique solution both give the same path.
- Failure: `ObcError` (or status recorded with `obc_raise=false`), not a warning and steady-state padding.

## Tests (test/test_obc.jl), all run 2026-11-24

1. Classification of the 11 Dynare files (23 checks; also that docs/OBC_CLASSIFICATION.md equals the generated table).
   Matches pydyn's table on every file both load (borrcon s_y = -1; NKM approx 0.0057; rbc_occbin IRR complementarity,
   INEG inconsistent; bf/bf_ppf multi_equation). JDyn also loads dynrbc*.mod and swpkfppf.mod (pydyn does not):
   dynrbc = rbc_occbin's classes, swpkfppf multi_equation.
2. Worked cases: lower bound without relax (g = r - rbar, y = r - rn), Dynare's borrowing constraint (g = lb, y = M*y - b).
3. `M`, `q` vs stacked system on nk_zlb (T=40), nk_zlb2 (30), obc_zlb2 (25), borrcon (40): < 1e-10 (observed ~1e-15);
   vs Dynare 7.2 (obc_zlb.mat): `M` 1.1e-14, `q` 1e-16; wedges vs GLPK 7e-17; path vs Dynare's OccBin 2e-17; regime start equal.
4. Solvers: 50 random P-matrix LCPs (Lemke with and without descent, active set agree); a 2x2 problem with three
   solutions (minnorm returns (1/3, 1/3)); a problem without solution (`:no_solution`; active set `:not_converged`);
   `solve_news` throws.
5. `minnorm` = OccBin on every complementarity model, with the fast path and with the pure `minnorm` solver: file shocks
   and 200 seeded random periods, paths to 1e-12 absolute and identical regime histories:

   | Model | file shocks: max diff, periods binding | random 200: max diff, periods binding |
   |---|---|---|
   | nk_zlb (own, no relax) | 9e-17, 7 | 3.9e-16, 45 |
   | nk_zlb2 (own, 2 constraints) | 1.9e-16, 7 | 8.9e-16, 59 |
   | obc_zlb (pydyn) | 1.7e-16, 1 | 6.7e-16, 66 |
   | obc_zlb2 (pydyn, 2 constraints, one without relax) | 1.9e-16, 4 | 1.2e-15, 70 |
   | Dynare borrcon | 2.2e-16, 3 | 5.1e-15, 152 |
   | Dynare borrcon_0_std_shocks (shocks sd 0.02) | 2.2e-16, 3 | 4.4e-16, 106 |
   | Dynare NKM_0_std_shocks (approx class; 3x the shocks' sd) | no shocks(surprise) block | 1.2e-13, 53 |
   | borrcon_sim, borrcon_curb (pydyn refs) | 2.2e-16, 3 | 5.1e-15, 152 |
   | occ_borrcon (pydyn ref; no surprise block) | - | 5.3e-15, 141 |

   No disagreement between `minnorm` and OccBin was found on any complementarity model.
6. IRFs (`irf_occbin`, `irf`) with `obc_method`; refusals (rbc_occbin: INEG inconsistent; bf_ppf: multi_equation; the
   wrong-way floor: orientation), unknown method, bad horizon, `news` stub, `horizon_too_short` (obc_zlb with T = 3) with
   and without `obc_raise`, warning on unused OccBin options.
7. Native lines (`JDyn.load` + `simulate(ob)`, `dynare()`), `occbin_solver!(ws; obc_method=...)`, `oo_.occbin.news`,
   default workspace has no `news`.

## Timing (2026-11-24, 2 shared cores, under `nice`, best of 3 after a warm-up; 1000 random shock periods, horizon 40)

| Model | n | constraints | periods binding | build M, q | OccBin | minnorm (fast path) | minnorm (pure: Lemke + descent) |
|---|---|---|---|---|---|---|---|
| obc_zlb (lower-bound example) | 5 | 1 | 356 | 0.046 s | 0.498 s | 0.116 s | 0.170 s |
| nk_zlb | 5 | 1 | 259 | 0.035 s | 0.341 s | 0.147 s | 0.159 s |
| nk_zlb2 | 7 | 2 | 340 | 0.082 s | 1.182 s | 0.156 s | 0.294 s |
| Dynare borrcon | 8 | 1 | 733 | 0.059 s | 1.413 s | 0.218 s | 0.337 s |
| Dynare NKM_0_std_shocks | 24 | 1 | 398 | 0.103 s | 0.727 s | 0.261 s | 0.333 s |

`rbc_occbin` (asked for by the contract) cannot be timed under `minnorm`: it is refused (INEG inconsistent). Its file
simulation (169 shock periods) takes 0.031 s under OccBin. Paths of the timing runs agree with OccBin (max 2.5e-12 on
NKM over 1000 periods, regimes identical). The `minnorm` times include `ss` (OccBin's time-varying rule for the regime
found), which is a sizeable share; it could be made lazy.

## Known failures and limits

- No disagreement found. Cases where they would differ, by construction: (i) the LCP has several solutions (OccBin's
  answer depends on its guesses, `minnorm` takes the locally smallest wedges; with the fast path, the active set's);
  (ii) OccBin fails (310/311/313) where the LCP has a solution; (iii) approximate constraints (NKM): OccBin switches on
  the nonlinear condition evaluated on the linear path, `minnorm` on the linearised residual, so near the threshold the
  regimes can differ by a term of order (steady-state gap) x mp; (iv) spells longer than `obc_horizon` (reported).
- Ties: a gap of exactly zero in the unconstrained path (or `|y| <= obc_tol`) counts as not binding; OccBin's strict `<`
  agrees for a zero gap, but tiny negative gaps (-1e-14) would bind in OccBin and give a wedge below `obc_tol` here.
- Nonlinear bind/relax expressions are linearised once at the steady state; OccBin evaluates them on the path.
- Lemke's method is complete for P- and copositive-plus matrices; for other `M` a secondary ray is reported as
  `:no_solution` although a solution may exist (`news` will settle such cases).
- Not done (out of scope): orders 2-3, estimation/filters with the new methods, `multi_equation`/`shared_equation`
  constraints, `news`.

## Lessons for the other port

- The sign conventions are the trap. Check with a constraint WITHOUT `relax` (nk_zlb.mod here, obc_zlb2.mod's floor in
  pydyn) and with the reference regime binding (Dynare's borrcon, s_y = -1). Lower bound with `relax` hides errors.
- Add the orientation test (rule 7): a "relax" that is the reverse of "bind" passes both fits but produces an LCP
  without a solution and an OccBin cycle. Better refused with a reason.
- Detect switched equations by the change of the linearisation, not by the parameter's name in the text.
- Do not let the fast path hide an unimplemented solver: JDyn probes the method's solver once on `(M, q0)`.
- Test `M` with an independent stacked system (both ports did) and against Dynare with news shocks written by hand
  (obc_zlb.mat): agreement to 1e-14 shows the wedge enters the reference residual with the right sign and scale.
- Recompute the wedges from the final complementary basis (one LU solve): paths then equal OccBin's to rounding
  (1e-16 to 1e-15), not to solver tolerance.
- Use a lexicographic ratio test in Lemke's method (JDyn does): problems from lower bounds can be degenerate (a gap
  exactly zero in the unconstrained path), where a plain ratio test can cycle. Not observed in the tests; precaution.
- For porting `news` into JDyn: implement `solve_news(M, q; tol, kw...) -> (y, status, info)` in src/obc_solvers.jl
  (replace the stub body; `OBC_SOLVERS[:news]` already points to it), return a status in `OBC_STATUSES`, end with
  `obc_verify`, put the diagnostics on `M` into `info` (they appear in `news.info[t]`; add a field to `_news_struct` for
  `oo_.occbin.news`). HiGHS would be a new dependency: make it a weak dependency (package extension) like Plots. The
  pure solver is reachable with `obc_fast_path=false` for testing and timing.
- MacroModelling.jl's formulation has H + 1 wedges and H checks, the last wedge sign-free; do not copy that if paths
  must equal OccBin's.
