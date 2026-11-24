# OBC methods (`obc_method`): progress log (branch `obc`)

Clone: /home/claude/work/jdyn-obc (base `main` a487e5d). Contract: /home/claude/work/OBC_CONTRACT.md (2026-11-24).
Backup: `git push origin obc:wip/obc` after each deliverable. Scope: order 1, `minnorm` and the shared machinery;
`news` is a stub that throws.

## Status

| Deliverable | State |
|---|---|
| A. Workspace, Julia 1.12.7, k-order library, baseline OccBin tests | done: `test_occbin.jl` (37 pass) and `test_occbin_options.jl` (456 pass) before any change |
| B. Post-processor and classification (src/obc.jl) | in progress |
| C. M and q (src/obc.jl), test against a stacked system | in progress (first check: 8.5e-16 on borrcon) |
| D. `minnorm` solver (src/obc_solvers.jl) | in progress |
| E. Interface (`obc_method`, `obc_horizon`, `oo_.occbin.news`) | in progress |
| F. Tests (test/test_obc.jl) | not started |
| G. Timing | not started |

## Decisions

## Contract amendments (proposed)

## MacroModelling.jl findings

## Known failures and disagreements

## Lessons for the other port
