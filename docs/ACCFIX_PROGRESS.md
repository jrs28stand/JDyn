# Acceptance fixes (branch `accfix`): progress log

Started 2026-11-25 from main 493b529 + gaps merge 4603c03. Baseline sweep: /home/claude/accept_jdyn (90 match / 210 partial / 176 refused / 201 wrong / 55 port-error).
Own sweep output: /home/claude/accept_accfix. Setup notes: the clone needs `Manifest.toml` (copied from the gaps clone, untracked) and
`deps/korder/libjdyn_korder.so` (copied from /home/claude/jdyn, untracked).

Method: per item, edit, check with `--only` on affected ids (runner `python3 test/acceptance/accept_jdyn.py --out /home/claude/accept_accfix --ids ...`
then the comparator), commit.

## Status per item

| item | status | notes |
|---|---|---|
| 1 oo_.exo_simul | edited (not yet verified) | stoch_simul.m: ones(max(periods,1)+lag+lead) * exo_steady_state' when exo_nbr > 0 |
| 1 dr.gx, Gy, sdim, edim, full_rank, fuu, g_i, pruning, exo-det rules | edited (not yet verified) | from the QZ step (`FirstOrderSolution.extra`), Dynare++ tensors with k_order_solver / order 3 |
| 1 other (bnlms, endo_simul of backward models, mom, FilteredVariablesShockDecomposition) | todo | |
| 2 options refused | todo | |
| 3 port-error fields | todo | |
| 4 numerical | todo | |
| 5 interpreter | todo | |

## Log
- Baseline re-run with the newer comparator (pydyn 0471ce9, end-point rule): totals unchanged (90/210/176/201/55).
- Findings that explain many verdicts: (1) stoch_simul.m never writes oo_.steady_state (initval values or the last `steady` stay; zeros
  initially): JDyn overwrote it with the solved one (71 "steady_state wrong"); now kept, loglinear logs it once (small piece, overlaps
  the parity2 agent's "logged steady state in the results layer": reconcile at merge). (2) Dynare's default order is 2, so the default IRFs are
  GIRFs that draw random numbers after the simulation: JDyn drew them before; moments were computed at order 1 inside the workspace
  because `_flat_options` drops default-valued options (order 2): `order=` is now passed explicitly (order-2 mean: burnside_1, neo_growth).
  (3) simult.m draws `randn(nxs, periods)'` (period by period): JDyn drew `randn(periods, nxs)`. (4) compute_variance_decomposition.m returns
  `ones(size(ghu,1),1)` for one shock (all endo_nbr rows): gamma_y{nar+2}/variance_decomposition for Ramsey and one-shock models.
- With JDYN_ACCEPT_RNG=octave (worker now uses random=:octave and writes "rng":"octave") exo_simul/endo_simul/irfs/moments of simulated runs
  are compared and match for arima/mod1, stochastic_simulations/example1, burnside_1, neo_growth, multi_leads.
- Dev flow: edits in /tmp/devsrc, tested via JDYN_PATCH overlays (worker loads patch files after `using JDyn`), merged to src at milestones.

## Final state
- Items done: 1 (exo_simul, dr.gx/Gy/sdim/edim/g_i/fuu/exo-det rules, variance decomposition size, Octave RNG layout/order of draws), 2 (endo_vars_for_moment_computations,
  mh_posterior_mode_estimation, analytic_derivation with a note, selected_variables_only, optim NumgradAlgorithm & unused keys now warn), 3 (occbin.* fields).
- Not done: filter_decomposition / FilteredVariablesShockDecomposition (design: 4D decomposition from P, iF, v, Z, R, Q inside the smoother passes), recursive
  estimation (vector nobs/first_obs), prior/posterior_function (needs calling a user MATLAB function), oo_.trend_component.*.ar, oo_.var.*, Ramsey steady_state,
  dr after estimation at the estimated point, dsge-var smoother length, SingularException in kalman_filter_smoother algo3/algoH1/algoH3, interpreter items (5),
  backward-model fields (gaps agent), AIM eigval (needs aim_solver), partial_information (PCL solver).
- Tests changed: test_estimation.jl and test_optimizers.jl (unused optim keys warn instead of EstimationError), test_results.jl (steady-state re-run test reads
  oo_.dr.ys: stoch_simul does not touch oo_.steady_state, as Dynare). Failing and not caused by this branch: test_matlab.jl (sprintf('%d\n') and zeros(2,2,2)
  refusal tests, from the gaps merge), test_plots*.jl (Plots not installed here).
- Harness note: 5 ids (AIM/fs2000_b1L1L, steady_state/homotopy/homotopy2_test, ms-dsge/test_ms_dsge, Trend_prefilter_MC and _first_obs_MC) report "no .mat written"
  even with --jobs 1 (setup-entry interplay in accept_jdyn.py: e.g. looks for pac/var-6/example1.mod); not a JDyn result.
