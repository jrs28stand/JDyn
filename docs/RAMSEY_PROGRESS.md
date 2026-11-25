# Optimal policy (Ramsey / evaluate_planner_objective / discretionary_policy / osr): status

Branch `ramsey`. Source: src/optimal_policy.jl (numerics), src/results_policy.jl (workspace commands). Tests:
test/test_optimal_policy.jl (in runtests.jl) against test/ref_optimal_policy (Dynare 7.2 under Octave 8.4; generators
r_policy.sh, pol_export.m, pol_nk_ramsey_2shocks_extra.m).

| feature | state | max difference to Dynare |
|---|---|---|
| Ramsey steady state (`dyn_ramsey_static`: multipliers by least squares, one or several instruments, steady-state file / block, `steady` twice, aux variables, histval) | done, 18 models | steady state 1e-11 or better |
| Ramsey rules, orders 1 and 2 | done | ghx, ghu, ghxx, ghxu, ghuu, ghs2 within 1e-9 relative (mostly 1e-12) |
| `evaluate_planner_objective` orders 1-2 (unconditional, conditional with multipliers 0 / steady state) | done | 1e-12 relative |
| order-2 Wyu defect of Dynare 7.2 | corrected by default, `replicate_dynare_bugs=true` reproduces Dynare | Dynare values reproduced to 1e-11 (u = 0 and u = [0.01 -0.02], two shocks) |
| `discretionary_policy` (Dennis) | done, 5 models | rules 1e-11, welfare 1e-12 |
| `osr` (weighted variances, planner objective at orders 1-2, bounds with opt_algo 9) | done, 5 models | objective at Dynare's optimum 1e-12; csminwel path identical for pol_osr_planner (20 iterations, 76 function calls) |
| perfect-foresight Ramsey welfare (backward recursion after `perfect_foresight_solver`) | done in `evaluate_planner_objective!` | only checked on nk_ramsey_det / neo_ramsey_det acceptance runs (no Dynare number) |
| `ramsey_constraints` (inequality constraints, `solve_algo=10`) | refused (RamseyConstraints/test1) | |
| `osr(analytic_derivation)`, opt_algo 1/2/3, `stoch_simul(order>3)` of Ramsey models, orders above 2 in `evaluate_planner_objective` | refused | |

Persistent state mirrors Dynare's: `lyapunov_symm` method 3 keeps its fixed-point iterate (m.cache[:lyap_fp]) and the discretionary
engine keeps `Hold` (m.cache[:dp_Hold]); a second call in the same session therefore differs from a cold call by the iteration
tolerance (about 1e-8), as it does in Dynare. The `dynare()` path solves discretionary models from a cold start.

Acceptance (test/acceptance, whole manifest, before -> after): REFUSED 376 -> 345, OK 117 -> 138, OK_WARN 283 -> 292, CHECK_FAILED 12 -> 13
(osr_obj_corr_algo_9, was REFUSED), the 38 optimal-policy, discretionary, osr and ramsey-named entries that changed: 36 improve (31 were
REFUSED), osr_example goes OK_WARN -> REFUSED (it asks for analytic_derivation, which was silently skipped before). Full test suite: all
files pass (test_examples.jl's expectation for nk_ramsey_osr changed from "load failed" to "not supported": the example loads and runs now).
Known remaining: osr_obj_corr_algo_9 (CMA-ES is random: its two runs agree
to 2e-4, Dynare's own check asks for 1e-4), `osr(analytic_derivation)`, opt_algo 1 and 3, ramsey_constraints, the k_order Ramsey
test (preprocessor JSON), estimation of Ramsey/discretionary models (outside this work).
