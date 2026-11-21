# Hostile review of JDyn (branch `review`, 2026-11-21)

Scope: JDyn at `847a9ee`. Method: differential testing against Dynare 7.2 and GNU Octave 8.4 on this machine
(Dynare run under Octave as in `reference-environment.md`), pydyn's `docs/GAPS.md` as a hazard list, reading of `src/`.
Nothing was fixed. Every confirmed finding has a `@test_broken` in `test/test_review_findings.jl` under the same
number (R01 ...); run it with `julia --project=. test/test_review_findings.jl` (about 1.5 minutes; R13 starts a
child Julia process). The file is not part of `runtests.jl`. When a finding is fixed, its `@test_broken` reports an
"unexpected pass": change it to `@test`.

Severity: S1 silently wrong result, S2 crash or wrong refusal on valid input, S3 misleading message, footgun,
pathology, S4 cosmetic.

Counts: **S1 8, S2 4, S3 9, S4 0** (21 findings). A "not confirmed" section is at the end.

MATLAB reference: where MATLAB and Octave differ I did not count the difference. All S1/S2 interpreter findings
below are cases where Octave 8.4 and the MATLAB documentation agree and JDyn differs.

| id | sev | one line |
|---|---|---|
| R01 | S1 | steady state of a model with a continuum of steady states (unit root, `y = y(-1) + e`) drifts away from `initval`; Dynare stays |
| R02 | S1 | logical arrays become double after indexed assignment or deletion (`l(2) = 5`, `l(end) = []`) |
| R03 | S1 | `A(logical row)` on a matrix returns a column, MATLAB a row |
| R05 | S1 | growing an array that has a zero dimension (`zeros(0,3); r(:,10) = 5`) drops the zero dimension |
| R09 | S1 | `sprintf('%d', 'abc')` prints `abc`, MATLAB `979899` |
| R11 | S1 | `set_shock_stderr!` zeroes the covariances of the shock, `set_shock_stderr_value` keeps them |
| R19 | S1 | `r = zeros(0,3); r(2,:) = 5` fills only the first column |
| R20 | S1 | `unique([NaN NaN])` returns one NaN |
| R04 | S2 | `any`/`all` raise an error when the array contains NaN |
| R06 | S2 | `A(:,:) = []` and `A(2:end,[]) = []` are refused with a MATLAB error |
| R10 | S2 | `2^[1 2;3 4]` (scalar to a matrix power) raises an error |
| R21 | S2 | `k_order` refuses a model with an unused, uninitialised parameter, Dynare solves it |
| R07 | S3 | `[1 2 3, zeros(2,0)]` is accepted (MATLAB: dimension error) |
| R08 | S3 | Julia exceptions leak from builtins (`trace` of a non-square matrix, `sprintf('%c', 65.5)`) and cannot be caught by MATLAB `try` |
| R12 | S3 | `set_params!` is not atomic |
| R13 | S3 | `y = exit(0)` in `steady_state_model` ends the Julia process |
| R14 | S3 | raw `DomainError` from `stoch_simul_moments` when a root lies inside the QZ criterion |
| R15 | S3 | spurious warning about `impulse_responses.plot_threshold` |
| R16 | S3 | `irf_figures`/`plot_irfs` return first-order IRFs without a warning when `stoch_simul` asks for order 2 (Dynare 7's default) |
| R17 | S3 | `JDyn.load` cannot pass `-D` macro variables (`extra_args`) to the preprocessor |
| R18 | S3 | unsupported models are refused with raw Julia exceptions (`LoadError`, `ArgumentError`) instead of `ModelError` |

---

## S1: silently wrong

### R01 (S1) Steady state moves variables whose static equation is an identity

Models with a unit root or any variable whose steady-state equation is identically satisfied (random-walk
processes, `a = rho*a(-1)` with `rho = 1`) have a continuum of steady states. Dynare, started at `initval`, keeps
those variables at their `initval` and moves only the variables it needs to. JDyn's solver takes minimum-norm
steps on the singular Jacobian and moves the identity variable too, so `oo_.steady_state`, `ghx` and every
second-order object differ from Dynare's.

```
var y c; varexo e; parameters gam; gam = 0.3;
model; y = y(-1) + e; c = 0.5*c(+1) + gam*exp(y); end;
initval; y = 0.2; c = 1; end;
shocks; var e = 0.01; end;
steady;
```

- Dynare 7.2 (Octave, `oo_.steady_state`, `oo_.dr.ghx`): `ys = [0.2, 0.7328416548961019]`, `ghx = [1; 0.7328416548961019]`.
- JDyn: `ys = [0.3243, 0.8299]` (y moved from 0.2 to 0.324), `ghx = [1; 0.830]`, about 10 percent off in `c`.
- Not specific to this toy: the shipped `tests/model_diagnostics/example2_block_model_diag.mod` (rho = 1) also ends at a different point
  from Dynare's (see "not confirmed": Dynare's own block and non-block runs differ from each other there, so the
  expectation is only clear when the identity variable appears alone, as above).

Cause: `src/steady.jl:184-203` (`_solve_static`): `NLsolve` trust region from `initval`, then Newton polishing with
`pinv(J)` on a singular Jacobian (line 200), both minimum-norm. Dynare's solver uses MATLAB's backslash, which gives a basic solution (leaves
the zero-column variable where it is). The `steady` result is then used by every solver. No warning is given when
the Jacobian is singular at the solution.

### R02 (S1) Logical arrays lose their class on indexed assignment and deletion

```matlab
l = [true false true]; l(2) = 5;      % MATLAB, Octave 8.4: logical [1 1 1]; JDyn: double [1 5 1]
l = [true false true]; l(end) = [];   % logical [1 0]; JDyn: double [1 0]
l = false(1,3); l(2) = 1; a = [10 20 30]; r = a(l);   % 20; JDyn: error (the mask became [0 1 0] as numbers: index 0)
```

Found by fuzzing (about 50 hits in 7,000 cases). Besides the wrong values, a mask built this way can no longer be
used as a mask.

Cause: `src/matlab/values.jl:507-511` in `assign_index`: `if eltype(a) == Bool && eltype(r) != Bool; a = Matrix{Float64}(a)`.
MATLAB converts the right-hand side to the class of the left (logical stays logical; NaN is an error).

### R03 (S1) `A(logical row vector)` on a matrix returns a column

```matlab
A = [1 2 3; 4 5 6]; r = A(logical([1 0 1]));    % Octave 8.4: 1x2 [1 2]; JDyn: 2x1 [1; 2]
```

The result of linear indexing with a vector takes the shape of the index; a logical index is treated as
`find(index)`. Values are right, the orientation is not (a later `[r, 1]` or `r'` silently changes meaning).
Cause: `src/matlab/values.jl` `_lin_index`/the one-index branch of indexing returns the shape of the source for logical
indices (not isolated to a line; `_lin_index` is at line 409).

### R05 (S1) Growing an array with a zero dimension

```matlab
r = zeros(0,3); r(:,10) = 5;     % Octave 8.4: 0x10; JDyn: 1x10 with a 5 in the last place
r = zeros(3,0); r(2,:) = 1;      % Octave 8.4: 3x0; JDyn: 3x1 [0;1;0]
```

Cause: `src/matlab/values.jl:565,568` take the extent of a colon from `size(r, ...)` of the right-hand side when `length(a) == 0`,
although `a` has a non-zero extent in the other dimension (see R19).

### R09 (S1) `sprintf` with numeric conversions applied to char data

```matlab
sprintf('%d', 'abc')         % MATLAB, Octave 8.4: '979899'; JDyn: 'abc'
sprintf('%g', 'Hello')       % '72101108108111'; JDyn: 'Hello'
sprintf('%u', '1e3')         % '4910151'; JDyn: '1e3'
sprintf('%ld', 5)            % '5'; JDyn: '%ld' (the length modifier is not understood)
```

(`sprintf('%-4.3g|', 'ab')` gives `97  |98  |` in MATLAB; JDyn consumes the whole string at once.) Strings built for
`eval`, parameter names or file names with `%d` on a char variable come out wrong without an error.
Cause: `src/matlab/builtins.jl:680` (`_sprintf`) hands a whole char array to a numeric conversion as text.

### R11 (S1) `set_shock_stderr!` zeroes the covariances

The Julia API `set_shock_stderr!(m, name, value)` (`src/model.jl:356-361`) zeroes the whole row and column of the
shock and then sets the variance. MATLAB's `set_shock_stderr_value` (Dynare `matlab/accessors/set_shock_stderr_value.m`:
`M_.Sigma_e(i,i) = value^2`) changes the diagonal only, and the same call written in a `.mod` file is handled
correctly by JDyn (`host_set_shock_stderr`, `src/matlab/session.jl:64-70`). The two entry points of the same
documented name disagree.

```
shocks; var e = 1; var u = 4; corr e,u = 0.5; end;
```
Dynare 7.2 after `set_shock_stderr_value('u', 3)`: `Sigma_e = [1 1; 1 9]`. JDyn `set_shock_stderr!(m, "u", 3.0)`: `[1 0; 0 9]`.
A parameter-loop over a shock size in a model with correlated shocks silently loses the correlation.

### R19 (S1) `A(i,:) = scalar` on an array with zero rows fills one column

```matlab
r = zeros(0,3); r(2,:) = 5;            % Octave 8.4 and MATLAB: [0 0 0; 5 5 5]; JDyn: [0 0 0; 5 0 0]
r = zeros(0,3); r(end+1,:) = 7;        % [7 7 7]; JDyn: [7 0 0]
a = [3 1 4]; r = zeros(0,3); r(a>2,:) = true;
```

A preallocated empty matrix grown row by row (common in calibration loops) is wrong from the first row. Row-vector
right-hand sides are right.
Cause: `src/matlab/values.jl:565` and `:568`: `ci = collect(1:(length(a) > 0 ? size(a, 2) : size(r, 2)))`; `length(a) > 0` is false for a 0x3 array so
the colon takes the width of the scalar right-hand side (1). Should use `size(a, 2)` when it is non-zero.

### R20 (S1) `unique` merges NaNs

`unique([NaN NaN])` is `[NaN NaN]` in MATLAB (since R2013a) and Octave 8.4 (NaNs are distinct); JDyn returns `NaN`.
Cause: `src/matlab/builtins.jl:335`, Julia's `unique` uses `isequal`.

---

## S2: crash or wrong refusal on valid input

### R04 (S2) `any`/`all` with NaN

```matlab
any([1 NaN 3])   % 1;  any(NaN) -> 0 (any ignores NaN);  all([1 NaN]) -> 1;  all(NaN) -> 1
```
JDyn raises a MatlabError for all of these. `if any(x)` on data with a NaN is the realistic trigger.
Cause: `src/matlab/builtins.jl:295`: `x = _logical(a[1])` converts to logical first (NaN is an error in `logical`, not in `any`/`all`).

### R06 (S2) Deletion with colon subscripts

`r = [1 2 3]; r(:,:) = []` is `zeros(0,3)` in MATLAB; `r(2:end,[]) = []` is a no-op; `x(2,:)` style deletion of
zero rows. JDyn: MatlabError "a null assignment can only have one non-colon index" (`src/matlab/values.jl:562`).

### R10 (S2) `scalar ^ matrix`

`2 ^ [1 2; 3 4]` is defined (`expm(log(2)*A)`; Octave 8.4: `[10.4827393896286 14.1518788283206; 21.227818242481 31.7105576321096]`).
JDyn: MatlabError "operator ^: one argument must be a square matrix and the other a scalar" (`src/matlab/values.jl:263`).
Also `[1 2;3 4]^0.5` is refused as complex, which is fine.

### R21 (S2) `k_order` refuses models with unused uninitialised parameters

Smets-Wouters (`sw_big.mod`, and the real SW_2007 file) declares `ccs`, `cinvs`, `crdpi` as "Unused parameter"
and never sets them (NaN, as in Dynare). `first_order` works; Dynare 7.2 solves order 2 too (reference run of
`sw_big` with `stoch_simul(order=2)` is OK); `k_order(m; order=2)` throws
`KOrderError: steady state, parameters or Sigma_e contain NaN or Inf`. Workaround: set the unused parameters to 0.
Repro with a three-line model in the test file (R21).
Cause: `src/korder.jl:93-95`: the finiteness test covers all of `params`, not only parameters the derivatives
depend on.

---

## S3: misleading message, footgun, pathology

### R07 (S3) Incompatible empty operands in concatenation are accepted
`[1 2 3, zeros(2,0)]` is an error in MATLAB ("Dimensions of arrays being concatenated are not consistent") and in
Octave; JDyn returns `[1 2 3]`. Only the 0x0 empty may be skipped. Code that relies on the error inside `try` takes the wrong branch.

### R08 (S3) Julia exceptions leak out of the interpreter
`trace(ones(2,3))` raises `DimensionMismatch` (`src/matlab/builtins.jl:472`: `LinearAlgebra.tr(farr(a[1]))`); MATLAB raises an error that
`try ... catch` catches. `sprintf('%c', 65.5)` raises `InexactError` (Octave prints `65.5`). In both cases the exception is not a
`MatlabError`, so a MATLAB `try/catch` does not catch it and `JDyn.load` aborts with a Julia stack trace. Found by fuzzing; there will be more
(`Int(...)` conversions and `x^Int(p)` in the builtins are candidates).

### R12 (S3) `set_params!` is not atomic
`set_params!(m, ["alpha" => 0.9, "nonexistent" => 1.0])` applies `alpha`, then throws `KeyError`; the parameters are
left half changed (`src/model.jl:351-355`). With a `Dict` argument which one is applied before the failure depends on the hash order.

### R13 (S3) `steady_state_model` code reaches Julia's namespace
The 7.2 preprocessor accepts any function name in `steady_state_model`; JDyn evaluates the preprocessor's Julia
text in a module that sees all of Base. `steady_state_model; y = exit(3); end;` followed by `steady_state(m)` ends the Julia process
(no exception, no output; reproduced in a child process in the test). `error(1)` is run as Julia's `error`. Files that
call `system(...)` etc. as MATLAB native lines are, correctly, never executed (checked: 30 variants of `system`, `unix`,
`!`, `eval`, `feval`, `builtin`, `str2func`, `fopen`, `save`, `delete`, `copyfile`, `setenv`, `ccall`, `run` and
`exit` in native lines and in `M_`-assigning lines: all skipped or refused, no side effect), so the hole is only the
compiled steady-state path (`src/steady.jl:20-26`, `src/model.jl:169`, module created with `Module(gensym(...))`, which imports Base). The reachable
functions take numeric arguments only (`exit`, `sleep`, `GC.gc`), so this is a denial-of-service hole rather than code execution.

### R14 (S3) Raw `DomainError` from `stoch_simul_moments` near a unit root

```
var y; varexo e; parameters rho; rho = 1.0000005; model; y = rho*y(-1) + e; end; shocks; var e = 1; end; steady; stoch_simul(order=1);
```
The root is inside the QZ criterion (1 + 1e-6), so the model is solved (ghx matches Dynare to 0). Dynare prints a variance of -1.0e6 (and a complex standard
deviation). JDyn throws `DomainError: sqrt was called with a negative real argument` from `sqrt.(diag(Var_y))` (`src/moments.jl:586`), with no
hint that the cause is the root. Same with `qz_criterium=1.05` and `rho=1.02` (Dynare: variance -24.75). Better: NaN plus a warning, as Dynare's
"unit root" handling does for roots outside the criterion.

### R15 (S3) Spurious warning
`stoch_simul(..., irf_plot_threshold=0.1)` makes `stoch_simul_moments` warn that it "does not act on ... impulse_responses.plot_threshold". The
option is an IRF plotting option and is honoured by `irf_figures` (`src/plotting.jl:91`); `COSMETIC` in `src/steady.jl:215` lists
`irf_plot_threshold` but the preprocessor's key is `impulse_responses.plot_threshold`.

### R16 (S3) IRFs at order 2 are first-order IRFs, silently
With `stoch_simul(irf=4)` and no `order` (Dynare 7's default is 2), `irf_figures(m)` returns the first-order responses with `res.order == 1` and no
warning (`src/plotting.jl:84` documents it in the docstring only). Dynare returns the order-2 generalised IRFs. `plot_irfs` titles carry no order.
`stoch_simul_moments` for the same file runs at order 2, so the two functions answer for different orders.

### R17 (S3) `JDyn.load` cannot pass macro variables
`JDyn.load(file; extra_args=["-DA=2"])` raises `MethodError ... Model(...; extra_args)`: `run_preprocessor` has `extra_args` but `load`
(`src/model.jl:320-322`) does not forward it. Reloading one `.mod` file with different `@#` macro values (the pydyn
`sw_big` stress cases do this) needs `JDyn.Model(JDyn.run_preprocessor(file; extra_args=...))`, which README does not show.
Checked and fine: two models from the same file with different `-D` values coexist in one session (each keeps its own compiled derivatives).

### R18 (S3) Refused models surface as Julia exceptions
Of the 733 `.mod` files of Dynare's `tests/` and `examples/` outside pydyn's 157-file corpus, 56 end in a "crash" rather than a `ModelError`. Most are preprocessor errors
on files that Dynare itself needs special options for. The JDyn-side ones:

- 9 files with `external_function` in the model block (`tests/external_function/*`): `LoadError: syntax: invalid assignment location "[TEF_0, TEFD_0, TEFDD_0]"` or `ParseError` from the preprocessor's Julia
  (`src/model.jl:284` only catches `external_functions_temporary_terms":[{` in `dynamic.json`).
- 2 files with `dates('2018Y')` in `perfect_foresight_setup` or `shocks`: `ArgumentError: cannot parse "dates('2000Q2')" as Float64`
  (`tests/deterministic_simulations/ramst_dseries.mod`, `pfwee_learnt_in_dseries_bare_1st_period.mod`).
- `tests/optimal_policy/neo_growth_ramsey_k_order.mod`: invalid JSON from the preprocessor, raw `ArgumentError`.

---

## Checked and found clean

- **Corpus outside pydyn's 157 files.** All 733 `.mod` files of Dynare 7.2's `tests/` and `examples/` that are not in
  pydyn's list were loaded: 446 load, 231 are refused with a `ModelError` (heterogeneous agents, `varexo_det`, PAC, unsupported MATLAB such as
  `dyn_ols`, `sur`, `cellfun` on names ...), 56 crash (R18, preprocessor errors). I ran Dynare 7.2 under Octave on 116 of them (all files with
  `native`/`verbatim` statements or `_steadystate.m`, plus a random 64 others) and obtained 60 references (the rest failed in Octave itself because
  they read results of other test files, or timed out at 90 s: 9 of the 60 contain MATLAB code, the other 51 are calibration only). `M_.params` and
  `M_.Sigma_e` agree on all 60 (rtol 1e-8); `oo_.steady_state` agrees where Dynare ran `steady` and nothing else changed it; the five that differ are explained
  (Dynare's state after perfect-foresight setup, `initval` kept because no `steady` was run, a continuum of steady states: see "not confirmed").
  The remaining ~40 MATLAB-code files in the "candidates" list were not compared because Dynare could not run them standalone.
- **MATLAB interpreter, differential fuzzing.** About 11,000 random snippets (indexing with `end`/logical/deletion/growth, operators and broadcasting, empties,
  strings, cells, control flow, 60 builtins) plus 150 hand-written function-file and idiom cases against Octave 8.4. Apart from R02-R10, R19, R20 and a few
  shape-only differences of empty results (see "not confirmed") JDyn matched Octave or refused (`Unsupported`). Function files: `nargin/nargout`, `varargin/varargout`, subfunctions,
  recursion, early return, globals, `end` in function-call indices, `try/catch` with identifiers, anonymous-function capture, range lengths (`0:0.1:1`, `0:1/3:1`, ...) all agree.
  `M_.param_names` as a cellstr (Dynare 7), `strmatch`, `get_param_by_name`, `M_.params(idx)=...`, `verbatim` blocks, `for` loops writing workspace
  copies of parameters, `set_param_value`: identical to Dynare (calibration idioms test, 37 values).
- **Numerics.** Smets-Wouters scale (`sw_big`, 40 variables, roots up to 0.9999995, cases 1-4): `ghx`, `ghu` to 3e-13, variances to 4e-11. Order 2 on `sw_big`: `ghx/ghu` 2e-14
  (ghxx not in the linear reference). Near-unit roots 1+5e-7 (solved, as Dynare), 1+2e-6 (BK error, as Dynare), `qz_criterium=1.05` with root 1.02 (solved, `ghx` identical),
  0.9999 (variance identical). Perfectly correlated and zero-variance shocks: moments and IRF skipping identical to Dynare's rule. Models with leads/lags up to 4, `EXPECTATION(-2)`,
  `diff`, exogenous leads and lags (18 variables with auxiliaries): `ghx` 7e-16, `ghu` 3e-16, same `order_var`. `loglinear` (IRFs, rules, variances: identical up to Dynare's 1e-6 steady-state tolerance, 2.5e-7).
  Moments: `hp_filter`, `ar`, `conditional_variance_decomposition`, variance decomposition, autocorrelations, `irf_shocks`, `relative_irf`: 1e-16.
  Shock blocks: later blocks, `overwrite`, `corr`/`stderr` mixing, `set_shock_stderr_value` in the `.mod`: identical to Dynare (6 files).
- **State and API.** `set_param!` is seen by `steady_state`, `first_order`, `k_order` (orders 2 and 3), `stoch_simul_moments`, `irf_figures`, `perfect_foresight`. Two models from one file with
  different macro values are independent. `load` + `first_order` in one function (world age) works. Folders and file names with spaces, unicode and hyphens load.
  Concurrent `first_order` and `k_order` from 4 threads (separate models, and one shared model for `first_order`): results identical to serial
  (an apparent mismatch in a first try was my own exact-equality test on roundoff-size `ghs2` values).
- **Sandbox.** See R13: MATLAB text cannot reach `system`, the file system, the environment, `ccall` or Julia's `eval` (30 variants).

## Not confirmed (suspicions, no reproduction)

1. `steady(homotopy_mode=..., homotopy_steps=...)` is listed as honoured (`src/steady.jl:135`) but `steady_state` jumps to the final values at once
   (`src/model.jl:578-587`, no stepping). A model whose direct solve fails but whose homotopy works in Dynare would be refused or land elsewhere. No such model built.
2. `steady`/`check` options: only the first statement's options are kept (`src/model.jl:578`: `haskey(m.options, kind) || ...`); `stoch_options` merges
   `check`, `stoch_simul`, `estimation`, `calib_smoother` in that fixed order, not in file order (known open item in STATUS.md for `qz_criterium`).
3. `tests/model_diagnostics/example2_block_model_diag.mod` (rho = 1): Dynare's block run gives `a = b = 0`, Dynare without `block` gives `b = -0.0035`,
   JDyn `a = 0.0099, b = -0.0069`: all valid steady states of a continuum; related to R01 but no unique Dynare answer.
4. Shapes of some empty results: `l(false)` is 1x0 in JDyn and 0x0 in Octave, `c = []; c(2:end) = []` is 0x1 vs 0x0, `diff(a(k))`, `find(0)`. MATLAB's own shapes
   were not established; these can turn a legal vertical concatenation into an error or the reverse.
5. `num2str(1e16)`: JDyn `10000000000000000`, Octave `1e+16`; MATLAB not checked. `str2double('i')` is `NaN` in JDyn (complex in MATLAB). `switch []` raises an error in JDyn, Octave runs `otherwise`.
   `nargin('f1')` (function name argument) raises an error.
6. `irf(dr, impulse; Sigma_e=...)` at order 2 raises `PosDefException` for a singular `Sigma_e` (perfectly correlated shocks); Dynare's own handling was not run.
7. Estimation, OccBin and perfect foresight were only read, not stress-tested (options warn as claimed in STATUS.md: spot checks of `stack_solve_algo`, `smoother_*`).
8. `k_order` releases `_korder_lock` between the solve and the `fetch`/`free` calls (`src/korder.jl:110-130`); I could not make this fail with 4 threads
   (results identical to serial), but the library's process-wide state makes it a candidate if it ever does.

## What I did not get to

Kalman smoother (in progress elsewhere), shock decompositions, OccBin option sweeps against Dynare beyond reading `src/occbin.jl`, estimation stress tests
(missing data with near-singular `F`, `lik_init` on unit-root models), posterior sampling, plotting output, and a Dynare comparison of order 3 on large or unit-root models.
