# Hostile review, second pass (branch `review2`, 2026-11-21)

Scope: `review` (pass 1, docs/REVIEW.md) merged with the current `main` (smoother and shock decompositions), plus the
`results` branch (`src/results.jl`: `dynare()`, `@dynare`, `stoch_simul!`, `steady!`, `check!`,
`perfect_foresight_setup!`, `perfect_foresight_solver!`, `occbin_solver!`, `set_param_value!`), reviewed in a separate
worktree. Nothing in `src/` was changed.

Method: every pass-1 finding was reproduced with a new minimal case (not pass 1's test). Ground truth: Dynare 7.2
under GNU Octave 8.4 on this machine (`dynare file nograph nolog`, then a few lines printing `oo_`/`M_` fields with
`%.15g`), Dynare's MATLAB source where the behaviour is a rule (cited), and for interpreter semantics a 209-snippet
differential run against Octave 8.4 (plus MATLAB documentation where Octave and MATLAB differ).

Tests: `test/test_review2_findings.jl` (standalone, about 40 s plus a child Julia for R13): one `@test_broken` per
confirmed finding (each checked to fail for the stated reason, not by error), `@test` for things checked clean. The
results-branch tests (R26-R31) run only when `JDyn.dynare` exists. On `results`: 15 pass, 45 broken; on `review2` (no results.jl): 11 pass, 30 broken.

Severity: S1 silently wrong result; S2 crash or wrong refusal on valid input; S3 misleading message or footgun;
S4 cosmetic.

## Summary

| sev | count | ids |
|---|---|---|
| S1 | 14 | R01 R02 R03 R05 R09 R11 R19 R20 (pass 1); R22 R23 R26 R27 R28 R30 (new) |
| S2 | 5 | R04 R06 R10 R21 (pass 1); R29 (new) |
| S3 | 10 | R08 R12 R13 R14 R16 R17 R18 (pass 1); R24 R25 R31 (new) |
| S4 | 3 | R07 R15 (pass 1, downgraded from S3); R32 (new) |

Pass 1: 21 of 21 confirmed, 0 false positives, 0 duplicates. Two severities lowered (R07, R15). R01 confirmed but
reframed (below): the problem is indeterminate, yet Dynare's answer is determined by its algorithm and, in the
common case, by the user's `initval`; JDyn's is not. R09's `%ld` sub-case and R07 were verified against Octave only.

| id | sev | status | one line |
|---|---|---|---|
| R01 | S1 | confirmed (reframed) | unit-root steady state: a variable whose static equation is an identity leaves its `initval` (Dynare keeps it) |
| R02 | S1 | confirmed | logical arrays become double on indexed assignment/deletion |
| R03 | S1 | confirmed | `A(logical row)` on a matrix returns a column |
| R05 | S1 | confirmed | growing an array with a zero dimension drops the zero dimension |
| R09 | S1 | confirmed | `sprintf('%d', 'abc')` prints `abc` (MATLAB `979899`) |
| R11 | S1 | confirmed | `set_shock_stderr!` zeroes the covariances |
| R19 | S1 | confirmed | `zeros(0,3); r(2,:) = 5` fills one column |
| R20 | S1 | confirmed | `unique` merges NaNs |
| R22 | S1 | **new** | `steady_state_model` runs with Julia's `Base`: `round(2.5)` = 2, `mod(x,0)` = NaN |
| R23 | S1 | **new** | load API keeps only the last `stoch_simul` statement's options (e.g. `hp_filter` of an earlier one is lost) |
| R26 | S1 | **new** (results) | options given to the re-run commands (or set in `options_`) are stored but not used |
| R27 | S1 | **new** (results) | commands run after the whole file with the final calibration: `oo_`/`M_` differ from Dynare |
| R28 | S1 | **new** (results) | `stoch_simul!` IRFs: `relative_irf` ignored at order >= 2; skip threshold 1e-6 instead of 5e-7 |
| R30 | S1 | **new** (results) | `stoch_simul(periods=...)` ignores `histval` |
| R04 | S2 | confirmed | `any`/`all` raise an error on NaN |
| R06 | S2 | confirmed | `A(:,:) = []`, `A(2:end,[]) = []` refused |
| R10 | S2 | confirmed | `scalar ^ matrix` refused |
| R21 | S2 | confirmed | `k_order` refuses a model with an unused NaN parameter (Smets-Wouters) |
| R29 | S2 | **new** (results) | `dynare()` aborts where Dynare continues (`check` with BK failure, failed perfect foresight); `oo_.dr.eigval` holds moduli |
| R08 | S3 | confirmed | Julia exceptions escape MATLAB `try/catch` |
| R12 | S3 | confirmed | `set_params!` not atomic |
| R13 | S3 | confirmed | `steady_state_model; y = exit(3)` ends the Julia process |
| R14 | S3 | confirmed | raw `DomainError` from moments for a root inside the QZ criterion (in `dynare()`: aborts, S2 there) |
| R16 | S3 | confirmed | `irf_figures` gives order-1 IRFs without a word when the file asks for order 2 |
| R17 | S3 | confirmed | `load` does not forward `extra_args` |
| R18 | S3 | confirmed | some refusals are raw Julia exceptions |
| R24 | S3 | **new** | `homotopy_setup(from_initval_to_endval)` ignored (warned as unhandled); steady state at the wrong exogenous value |
| R25 | S3 | **new** | stochastic singularity accepted silently by the smoother and estimation (Dynare refuses) |
| R31 | S3 | **new** (results) | no-argument commands act on the last workspace created, not on the globals; `set_param!(ws.model, ...)` is overwritten |
| R07 | S4 | confirmed, S3 -> S4 | `[1 2 3, zeros(2,0)]` accepted (Octave: error) |
| R15 | S4 | confirmed, S3 -> S4 | spurious "does not act on impulse_responses.plot_threshold" warning |
| R32 | S4 | **new** | non-ASCII characters become `?` (`double('é')` = 63) |

---

## Pass-1 findings, re-verified

### R01 (S1, confirmed, reframed) Steady state of models with a continuum of steady states

Repro (`y = y(-1) + e; c = 0.5*c(+1) + gam*exp(y)`, `initval y = 0.2, c = 1`, `steady`):
Dynare 7.2 `oo_.steady_state = [0.2, 0.7328416548961019]`, `ghx = [1; 0.73284]`; JDyn `[0.32432, 0.82986]`, `ghx(c) = 0.830`.
Both are exact steady states (JDyn's residual is 0); the problem is indeterminate.

Why Dynare's answer is still the expected one: Dynare's default `solve_algo = 4` (`matlab/optimization/dynare_solve.m:224-276`)
splits the static Jacobian with a Dulmage-Mendelsohn decomposition, solves the blocks in turn, and skips a non-square block
whose equations already hold (`norm(fval_check(j1(j))) < tolf ... continue`). A variable that appears only in identically
satisfied equations is therefore never moved: `initval` is the user's handle on which point of the continuum is used, and
Dynare honours it. JDyn's trust-region solve plus minimum-norm Newton polishing (`src/steady.jl:184-203`, `pinv` at :200)
moves it by an amount that depends on the other variables' initial values.

The rule is algorithmic, not "keep the unit-root variable": in a second case (`x = x(-1) + e; z = x + 0.5*(z(-1)-x(-1)) + u`,
`initval x = 0.2, z = 0.1`) Dynare moves `x` to 0.1 and keeps `z`; JDyn ends at `x = z = 0.2308`. There no choice is
"right", but JDyn's is still not reproducible from the file and nothing says the Jacobian is singular.

Impact: every level-dependent object (decision rules, IRFs, moments of the stationary variables) of models written with a
random walk in levels. Suggested fix: emulate `solve_algo=4` (DM blocks, skip blocks already solved, `\` within blocks)
or at least solve only the rows/columns with a non-zero Jacobian, keep variables whose column is zero in all non-identity
rows at `initval`, and warn when the static Jacobian is rank deficient at the solution.

### R02 R03 R05 R09 R19 R20 (S1), R04 R06 R10 (S2), R07 R08 (S3/S4): MATLAB interpreter

All reproduced with new snippets against Octave 8.4 (my differential run of 209 snippets; values in the test file):
`l = false(1,3); l(3) = 1; a(l)` -> JDyn error (R02); `A(logical([1 0 1]))` 2x1 (R03; a column mask is right);
`zeros(0,3); r(:,10) = 5` 1x10 (R05); `sprintf('%d|','ab')` -> `ab|`, MATLAB/Octave `97|98|` (R09); `zeros(0,3); r(2,:) = 5`
-> `[0 0 0; 5 0 0]` (R19; growth from `[]` is right); `unique([3 NaN 1 NaN])` has 3 elements, MATLAB 4 (R20);
`if any([NaN 1])` -> MatlabError (R04); `r(:,:) = []` (R06); `2^[1 2;3 4]` (R10); `trace(ones(2,3))` inside `try`
-> `DimensionMismatch` escapes (R08); `[1 2 3; zeros(0,2)]` accepted (R07). Causes as in REVIEW.md (they are right).

- R07 lowered to S4: accepting a concatenation that MATLAB rejects only changes programs that rely on the error.
  MATLAB itself not checked (Octave errors).
- R09: `sprintf('%ld', 5)` is `5` in Octave; whether MATLAB accepts the `l` modifier was not checked. Note that
  `sprintf('%d', [1 2.5])` gives `12.500000e+00` in JDyn and `12.5` in Octave: JDyn follows MATLAB (`%e` for
  non-integers), so that is right.
- Fix suggestions as in REVIEW.md; for R08 wrap every builtin call in `try ... catch e; e isa MatlabError || merr(...)`.

**Fixed (branch `fix-interp`).** R02 (logical stays logical on assignment and deletion; NaN into a logical is an error;
an undefined variable takes the class of the right-hand side), R03 (a logical subscript gives `find(mask)`'s orientation:
a vector source keeps its own), R05/R19 (a colon takes the extent of the array, only a 0x0 array takes it from the
right-hand side, as Octave's `zero_dims_inquire`), R20 (`unique` keeps every NaN, at the end; `-0 == 0`; logical in,
logical out; `unique([])` is 0x0), R04 (`any`/`all` ignore NaN as non-zero/zero tests: `any(NaN)` false, `all(NaN)` true),
R06 (two colons delete every row, empty subscripts delete nothing, a subscript spanning its dimension counts as a colon;
`A(i) = []` on a matrix gives a row; Octave refuses `A(1:2,2) = []`, JDyn follows MATLAB there), R10 (`s^A` = `expm(log(s)*A)` for
`s > 0`; `s < 0` is refused as complex, `0^A` as unsupported), R07 (Octave's `dim_vector::hvcat` rule: 0x0 skipped, 1x0 and
0x1 skipped when they do not fit, anything else is a dimension error), R08 (builtins and `try` turn Julia exceptions such as
`DimensionMismatch`, `InexactError`, `BoundsError` into MATLAB errors), R09 (a char array met by a numeric or `%c`
conversion is used one character at a time; `%ld`, `%lu`, `%hd`, ... accepted; `%c` of a non-integer prints `%e` as `%s` does).
Tests: test/matlab_extra_cases.txt cases 255-376 (Octave 8.4 reference regenerated), test/test_matlab.jl "interpreter fixes",
the flipped `@test_broken` in test/test_review2_findings.jl.

### R11 (S1, confirmed) `set_shock_stderr!` zeroes the covariances

`shocks; var e = 1; var u = 4; corr e,u = 0.5; end;` then `set_shock_stderr!(m, "u", 3.0)`: JDyn `[1 0; 0 9]`.
Dynare's `matlab/accessors/set_shock_stderr_value.m` is `M_.Sigma_e(i,i) = value^2` -> `[1 1; 1 9]` (it keeps the
covariance, so the correlation changes; neither keeps the correlation). The function is exported and undocumented.
Cause `src/model.jl:356-361`. Fix: diagonal only, as Dynare and as JDyn's own `.mod` path (`src/matlab/session.jl:64`);
document it (optionally `keep_correlation=true`).

### R12 (S3), R17 (S3), R18 (S3) confirmed as described (`src/model.jl:351-355`, `:320-322`, `:284` and the dates parse).

### R13 (S3, confirmed) `steady_state_model` reaches Julia's namespace

`steady_state_model; y = exit(7); end;` then `steady_state(m)` in a child process: the process exits with code 7, no
output. See R22 for the silently-wrong side of the same cause. Severity stays S3 (denial of service; the reachable
functions take numbers only).

**Fixed (branch `fix-interp`)** with R22 (bare module; the child process in the test now prints `ALIVE`).

### R14 (S3, confirmed) `DomainError` from moments near a unit root

Reproduced (`rho = 1.0000005`). On the results branch `stoch_simul!` only catches `ArgumentError`/`ModelError`
(`src/results.jl:273`), so `dynare()` aborts on a file Dynare runs (S2 there). Fix: NaN plus a warning.

### R15 (S4, confirmed, lowered) The warning is true (moments do not use the option) and harmless: cosmetic.

### R16 (S3, confirmed) `irf_figures` order 1 without a warning (`stoch_simul(irf=4)`, default order 2): reproduced.

### R21 (S2, confirmed) `k_order` with an unused NaN parameter

RBC plus `parameters unused;` (never set): `first_order` works, `k_order(m; order=2)` throws `KOrderError ... NaN`.
Cause `src/korder.jl:90-91` (finiteness of all `params`). Fix: test only the parameters that appear in the
derivatives (or replace unused NaNs by 0 before the `ccall`, as Dynare effectively does).

---

## New findings: load API

### R22 (S1) `steady_state_model` is compiled with Julia's `Base`, not MATLAB semantics

```
var y; varexo e; parameters a; a = 2.5;
model; y = 0.5*y(-1) + e; end;
steady_state_model; y = round(a); end;
```
MATLAB/Octave `round(2.5) = 3` (and JDyn's own interpreter gives 3); `steady_state(m)` gives **2** (Julia rounds half
to even). `y = mod(a, 0)`: MATLAB 2.5, JDyn `SteadyStateError` (Julia `mod(2.5, 0.0) = NaN`). Any MATLAB function name
that also exists in `Base` with other semantics is affected; names unknown to `Base` fall back to the interpreter.
Cause: the preprocessor's Julia `steady_state!` is evaluated in `Module(gensym(...))` (imports `Base`,
`src/model.jl:169-170`) and used whenever it runs without `UndefVarError`/`MethodError` (`src/steady.jl:21-34`).
Fix (also closes R13): evaluate it in a bare module (`Module(name, false)`) that defines only the functions with MATLAB
semantics (`round` half away from zero, `mod`, `rem`, `fix`, `max`/`min`, ...); anything else goes to the interpreter.

**Fixed (branch `fix-interp`), R22 and R13.** `steady_state_module` (src/model.jl) builds a bare module (no Base, no
`eval`/`include`) with the operators of the generated text and a table of MATLAB functions (`SSM_NAMES`); the 7.2 text
of `SteadyState2.jl` is included there, not in the model module. Compiled and interpreted results agree with Octave for
a 70-expression sweep (`round`, `mod`, `rem`, `sign`, `fix`, `floor`, `ceil`, `max`, `min`, powers, `gamma`, `atan2`, ...);
`exit`, `sleep` and any other name not in the table are an `UndefVarError`, which sends the block to the interpreter
(unknown function there). A user-supplied helper with the same name as one of the module's names forces the interpreter.
`steady_state_model` evaluation time: 0.55 us per call before, 0.57 us after (est_schorfheide, 20000 calls).

### R23 (S1) The load API keeps only the last `stoch_simul` statement's options

```
<RBC>; steady;
stoch_simul(order=1, irf=0, hp_filter=1600);
stoch_simul(order=1, irf=3) c;
```
Dynare 7.2: options persist in `options_` (the driver writes only the options given, plus an explicit `order`), so the
second `stoch_simul` uses `hp_filter = 1600`: `oo_.var = 0.00347420017853539`. `stoch_simul_moments(m)`:
`0.0076601` (unfiltered). Same for every option of `stoch_simul`/`estimation`/`calib_smoother` written once and relied on
later. `dynare()` on the results branch gets this right (it accumulates `options_`).
Cause: `src/model.jl:589-590` stores each statement's options by replacing the previous ones; `stoch_options`
(`src/steady.jl:220-226`) merges kinds in a fixed order, not statements in file order (`steady`/`check` keep the first
statement instead, `src/model.jl:577`). Fix: accumulate options per statement in file order into one `options_`-like
dict (with the preprocessor's explicit `order` default), and read every option from it.

### R24 (S3) `homotopy_setup(from_initval_to_endval)` is ignored

Dynare's own `tests/steady_state/homotopy/homotopy_from_initval_to_endval.mod`: Dynare 7.2
`oo_.steady_state = [6.12244879311, 51.0204019802, 2]`, `oo_.exo_steady_state = 2`. JDyn `steady_state(m)`:
`[1.5306, 12.7551, 1]` (at `x = 1`), with only a load-time "statements not handled: homotopy" warning; `m.oo` meanwhile
says `x = 2`. Cause: `src/model.jl:568-570` handles `homotopy_setup` only with explicit value lists. Fix: implement the
option (final values = `endval`), or refuse with a `ModelError`.
(`homotopy1_test`/`homotopy3_test`, explicit values: JDyn's direct solve agrees with Dynare's homotopy to Dynare's
tolerance, 1.2e-5 relative for mode 1. Homotopy stepping, pass-1 suspicion 1, is still not stepped, but I found no model
where that changes the answer: with `x` 1 -> 1000 or 1e5 Dynare's own homotopy fails.)

### R25 (S3) Stochastic singularity is accepted silently

- `calib_smoother` with `varobs y w` and `w = y` (singular forecast-error covariance): Dynare 7.2 stops with
  "Kalman filter: F is singular in stationary period. Stochastic singularity detected." (`DsgeSmoother.m:317-323`;
  `use_univariate_smoother_if_singularity_is_detected` defaults to false, and even when switching Dynare prints "This may
  be a sign of stochastic singularity"). JDyn returns a `SmootherResult` without a message.
- Estimation with `w = y + 1e-7*x`: Dynare's `initial_estimation_checks.m:225-235` disables the univariate fallback for
  one evaluation and errors ("forecast error variance ... became singular"); `estimate(m)` runs without a word.
- Cause: `src/smoother.jl:44` lists the option as cosmetic and `:921-923` always falls back to the univariate smoother;
  `estimate` has no equivalent of the initial check. Fix: honour the option (default false -> `SmootherError`), and
  add the initial singularity check to `estimate`.

### R32 (S4) Non-ASCII characters become `?`
`double('é')` is 63 in JDyn (MATLAB 233; Octave the UTF-8 bytes 195 169). Affects labels and `long_name`s only.

**Fixed (branch `fix-interp`).** String literals keep their characters (`double('é')` is 233, MATLAB's UTF-16 code unit;
Octave counts UTF-8 bytes). Only string literals are changed: the rest of a line is still read as ASCII for tokenizing.

---

## New findings: results branch (`src/results.jl`)

### R26 (S1) Options given to the re-run commands are stored in `options_` but not used

The docstrings say the commands solve "with the current `options_`" and that keyword options "persist". They are
written to `options_` (`_take_options!`, :151) but the solvers read the `.mod` file's statements (`m.options`):

- `stoch_simul!(ws; loglinear=true)`: `options_.loglinear = true`, IRFs and `oo_.dr.ys` unchanged.
- `stoch_simul!(ws; qz_criterium=0.5)` on a model with root 0.8: solved (Dynare: BK error).
- `stoch_simul!(ws; conditional_variance_decomposition=[1,4])`: no `oo_.conditional_variance_decomposition`.
- `occbin_solver!(ws; simul_periods=7)` on `examples/occbin/rbc_occbin.mod`: `options_.occbin.simul_periods = 7`, the
  simulation still has 200 periods (:398-402: `simulate(ob)` reads `mod_options`). The option paths also differ from
  Dynare's (`options_.occbin.simul.periods`).
- `steady!(; ...)` (:299-302) and `perfect_foresight_solver!` (:368: only keywords of the current call are passed;
  values persisted in `options_` by an earlier call, or set by the user, are ignored).

Fix: make `options_` the source of truth: before each command, translate `ws.options_` into the option dicts the
solvers read (or pass them explicitly), and warn on options no solver acts on.

### R27 (S1) Commands run after loading, with the final calibration

Documented as a prototype limitation (docs/RESULTS.md, decision 1), but `oo_`/`M_` are presented as Dynare's and differ:

- A steady-state block that sets a parameter, read later by MATLAB code:
  ```
  a = 0.5; b = 3; ... steady_state_model; b = 2*a; y = b/(1-a); end;
  steady;  a2 = M_.params(2) * 2;  set_param_value('a', a2/10);
  ```
  Dynare 7.2: `M_.params = [0.2; 1]`, `oo_.steady_state = 2`. `JDyn.load` gets exactly this (its session runs the
  steady state in place). `dynare()`: `M_.params = [0.2; 0.4]`, `oo_.steady_state = 0.5` (it reruns `steady` at the end).
- Two `stoch_simul` with `rho = 0.5;` -> `rho = 0.9;` between them and a shorter `var_list` the second time:
  Dynare keeps `oo_.irfs.x_e` from the first run (rho = 0.5): `[1.76905, 0.93810, ...]`; `dynare()`: `[2.25390, 1.90779, ...]`.
- Dynare's test suite: a scan of the 177 test files that recalibrate (parameters, `shocks`, `set_param_value`,
  `options_`) after a command found none that JDyn loads where the *final* `oo_` differs for this reason alone
  (`steady_state/Gali_2015_chapter_6_4.mod` would, but it is Ramsey and refused). Many store intermediate results
  (`oo_baseline = oo_`, `unit_irf = ...oo_.irfs...`, e.g. `irfs/example1_unit_std.mod`,
  `kalman_skewed/ireland2004_stoch_simul.mod`): those lines are skipped with a warning, so the file's own checks do not
  run, and every intermediate `oo_` would be computed at the wrong calibration.
- Fix: decision 1 of RESULTS.md (run each command where it stands in the session). Until then, document that `oo_`
  reflects the final calibration and warn when the calibration changes after the first command.

### R28 (S1) `stoch_simul!` IRF rules differ from `stoch_simul.m`

- `relative_irf` at order >= 2: Dynare normalises the impulse to `cs(:,i)/cs(i,i)/100` (`stoch_simul.m:279-283`);
  JDyn applies `relative_irf` only at order 1 (:243). Dynare's `tests/irfs/example1_unit_std.mod` (final statement:
  order 2, `relative_irf`, stderr 0.0095): Dynare `oo_.irfs.a_e(1) = 0.01`, `dynare()` 0.0095 (5 percent off
  everywhere).
- Skip rule: Dynare computes an IRF when `cs(i,i) > 5e-7` (`stoch_simul.m:276`); JDyn skips below 1e-6 (:239), so a
  shock with stderr 8e-7 has no `oo_.irfs` field (plotting.jl:109 has Dynare's rule).
- Fix: port both rules (and divide by `cs(i,i)` at order 1 only, as Dynare).

### R29 (S2) `dynare()` aborts where Dynare continues; `oo_.dr.eigval` is wrong

- `check` on a model violating Blanchard-Kahn: Dynare prints the eigenvalues and "The order condition is NOT verified"
  and the file continues (`check.m` only errors for info other than 0, 2, 3, 4); `dynare()` throws `BlanchardKahnError`
  (:309-312).
- `perfect_foresight_solver(maxit=1, no_homotopy)` that fails: Dynare stores
  `oo_.deterministic_simulation.status = 0` (iterations 2) and continues; `dynare()` throws `PerfectForesightError`
  (`raise_on_failure=true` default, :358).
- `oo_.dr.eigval` holds moduli (:315: `abs.(...)`); Dynare stores the complex eigenvalues (`0.6 +/- 0.3742i, 0.9, 2` in
  the test). A later `stoch_simul!` replaces `oo_.dr` and drops `eigval` (:198); Dynare's `resol` keeps it.
- Fix: `check!` stores `sol.eigenvalues` as complex and does not throw for BK failures; `dynare()` calls the PF solver
  with `raise_on_failure=false` and stores the status; `_store_dr!` keeps `eigval`.

### R30 (S1) `stoch_simul(periods=...)` ignores `histval`

RBC with `histval; k(0) = 0.15; a(0) = 0.02; end;` and `stoch_simul(order=1, periods=150)` with Dynare's own shocks:
Dynare's first simulated period starts from `histval` (`a(1) = 0.8*0.02 + e(1)`); `dynare()` starts from the steady
state (difference 0.016 in `a`, 0.018 in `c` in period 1, decaying). Cause: `M.endo_histval = zeros(0,0)` (:108) and
`simulate(dr, ex; ...)` from the steady state (:262). Fix: fill `M_.endo_histval` from `histval` and start there.

### R31 (S3) Which workspace a no-argument command acts on

`ws = dynare(...)` (documented as "without defining globals") still sets the global `CURRENT` (:415-416). After
`@dynare "a.mod"` and `dynare("b.mod")`, editing `M_.params` (A's) and calling `stoch_simul!()` re-solves B; A's `oo_`
is unchanged, silently. Also `set_param!(ws.model, ...)` / `set_shock_stderr!(ws.model, ...)` (the documented Model
API on `ws.model`) are overwritten from `M_` by the next command (`_sync_model!`, :169-173). Fix: only `@dynare` sets
`CURRENT`; sync in both directions or document `M_` as the only handle.

---

## Checked clean

| what | how | result |
|---|---|---|
| Order 3 at a unit root (rho = 1), at 1 - 1e-7 and at 0.999 (3 variables, exp and squared terms, two shocks) | `k_order` vs Dynare `oo_.dr` | all of ghx ... ghuss within 3e-15 (relative to max(1, abs)); same `order_var` |
| Perfect foresight with a unit root and with rho = 0.99999 (60 periods) | `perfect_foresight` vs `oo_.endo_simul` | 5e-15, same Newton iterations (7) |
| Perfect-foresight options one by one (RBC-type, 50 periods) | vs Dynare | `linear_approximation` 5e-14; large shock with homotopy 5e-12 (7 iterations, as Dynare); `maxit=1, no_homotopy`: same failure (status 0, 2 iterations, paths 5e-14); `tolf=1e-14, tolx=1e-3`: 6e-14 (6 iterations vs 5: Dynare cannot reach that tolf); `stack_solve_algo=1`: 4e-5 (documented: Newton instead of LBJ); `endval` + `steady` and `endval_steady`: 1.2e-3, entirely Dynare's terminal steady state (k = 34.67079 vs exact and JDyn 34.67196, Dynare stops at its 6e-6 residual tolerance) |
| `dynare()` perfect foresight (linear approximation; large shocks) | vs Dynare `oo_.endo_simul` | 5e-14, 7e-12 |
| Diffuse smoother (`kalman_algo` 0, 3, 4) on a unit-root model with missing values in both observables, including a period with none | vs Dynare `oo_.SmoothedVariables`, `SmoothedShocks` | 5e-12 |
| Diffuse likelihood, same model | vs `oo_.likelihood_at_initial_parameters` | 90.1848565996967 to 1e-14 |
| `kalman_algo=2` with a unit root; `lik_init=1` with a unit root | | same refusal as Dynare; -Inf likelihood (Dynare: BK error, as `lik_init=1` uses qz 1-1e-6) |
| `stoch_simul` inside a MATLAB `for` loop over `set_param_value` | `load` and `dynare()` vs Dynare | final `M_.params` and `oo_.var` identical; the loop's `v(i) = oo_.var(1,1)` skipped with a warning |
| `steady_state_model` that sets a parameter read by later MATLAB lines | `load` vs Dynare | `M_.params` and `oo_.steady_state` identical (R27 is `dynare()` only) |
| `dynare()` options accumulation across `stoch_simul` statements (`hp_filter` persists, `order` reset to 2 by the driver) | vs Dynare driver.m and results | right (R23 is the load API) |
| MATLAB interpreter, 209 new snippets (integer classes, cells, structs, strings, linear algebra, reductions, NaN rules) | vs Octave 8.4 | outside R02-R10, R19, R20, R32: equal or refused (`Unsupported`); `5 && []` is an error as in MATLAB |

## Not reached

- OccBin option sweeps against Dynare (only `occbin_solver!` keywords, R26). Shock decompositions, posterior sampling,
  estimation at Smets-Wouters scale and near-singular `F` beyond R25.
- Badly scaled steady states (Dynare and JDyn both test residuals with the same absolute tolerance; no case built).
- Homotopy stepping in `steady` (see R24), `steady`/`check` options of later statements (pass-1 suspicion 2, part of
  R23's cause, no case built for `steady`).
- Order 2/3 IRFs and moments in `dynare()` against Dynare draw-for-draw (Octave's draws are not reproducible).
- The k-order library lock (pass-1 suspicion 8).
