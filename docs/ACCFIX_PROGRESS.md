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
