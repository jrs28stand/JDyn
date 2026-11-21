# Results as in MATLAB: M_, options_ and oo_

Goal: after running a .mod file, results are where a Dynare user expects them (`oo_.endo_simul`,
`oo_.dr.ghx`, `oo_.irfs.y_e`, `M_.params`, `options_.periods`), in the same shapes, readable from Julia with
MATLAB's syntax and from the MATLAB lines of the .mod file.

## Usage (prototype)

```julia
using JDyn
@dynare "rbc.mod"                 # like MATLAB's `dynare rbc`: defines M_, options_, oo_
oo_.endo_simul                    # 3x202 Matrix, as Dynare
oo_.dr.ghx; oo_.irfs.y_e          # 1x40 row, as Dynare
M_.endo_names[1]; M_.params       # cell (column), n x 1 column vector
propertynames(oo_)                 # tab completion works on the structs

ws = dynare("rbc.mod")            # the same without defining globals: ws.M_, ws.options_, ws.oo_, ws.model
ws = dynare("rbc.mod"; shocks=Dict(1 => exo))   # reuse given shocks (e.g. Dynare's oo_.exo_simul) for stoch_simul #1
```

The structs are JDyn's `MStruct` (the interpreter's struct type, ordered fields) and `Cell`, so the very
same objects are what MATLAB code in the .mod file sees. Matrices keep MATLAB's shapes: column vectors are
n x 1, rows 1 x n, scalars `Float64`, logicals `Bool`, strings `String`, cells `Cell`. `display(oo_)` prints
MATLAB's struct summary (`endo_simul: 3x202 double`).

The existing API (`load`, `first_order`, `perfect_foresight`, `estimate`, ...) is unchanged: `dynare()` is a
layer over it, and `ws.model` is the usual `Model`.

## Checked against Dynare 7.2 (test/test_results.jl)

| | |
|---|---|
| perfect_foresight_rbc: `oo_.endo_simul` (3x202) | 2.8e-14 |
| `oo_.exo_simul` (202x1), `M_.params`, `oo_.steady_state` | exact / 1e-15 |
| collard_2001 `stoch_simul(periods=2000)` with Dynare's shocks: `oo_.endo_simul` (6x2000), `oo_.dr.gh*` | 2e-6 / 1.4e-5 |

The collard gap is Dynare's: its steady state stops at a residual of 1e-6 (default tolf), JDyn's is exact.
At Dynare's steady state the decision rules agree to 1e-12 (test_korder.jl).

## What fills what

| command | Dynare fills | prototype |
|---|---|---|
| (driver) | `M_` (~80 fields) | names (+long/tex), counts, maximum_lag/lead (+orig, endo/exo), lead_lag_incidence, nstatic/npred/nboth/nfwrd/nspred/nsfwrd/ndynamic, params, Sigma_e, Correlation_matrix, H, aux_vars, endo_histval |
| (driver) | `options_` (~600 defaults) | JDyn's small default table + every option the .mod sets (the preprocessor names them by their options_ path) + MATLAB assignments |
| steady | steady_state | yes |
| check | dr.eigval | yes |
| stoch_simul | dr (ys, order_var, inv_order_var, state_var, gh*), irfs, endo_simul, exo_simul, var_list, mean, var, autocorr, gamma_y, variance_decomposition, skewness, kurtosis | yes (orders 1-3; moments orders 1-2) |
| perfect_foresight_solver | endo_simul, exo_simul, deterministic_simulation.{status, iterations, error}, initial_steady_state | yes |
| occbin_solver | oo_.occbin.simul.{piecewise, linear, shocks_sequence} | yes |
| estimation | posterior_mode, posterior_std_at_mode, MarginalDensity, posterior.*, SmoothedVariables, ... | not yet (the results exist in `EstimationResult`) |
| calib_smoother | SmoothedVariables, SmoothedShocks, FilteredVariables, UpdatedVariables | not yet (smoother being written) |
| forecast, osr, identification, shock_decomposition, ... | | not supported by JDyn |

## Decisions for the full version

1. **Commands run in place.** The prototype loads the whole file, then runs the commands in order with
   the final calibration. Dynare runs each command where it stands, so a later `rho = 0.5; stoch_simul;`
   gets a second, different `oo_.irfs`, and MATLAB lines after a command can read `oo_`. The session already
   stops at every command (it does so to track `M_.params` and `oo_.steady_state`); the full version calls
   the command there and lets the interpreter's `oo_` be the MStruct, so lines like
   `y = oo_.endo_simul(1, :); plot(y)` or `ghx = oo_.dr.ghx;` work inside the .mod. `load()` keeps its
   declare-only behaviour.
2. **options_ defaults.** Dynare sets ~600 defaults in `default_option_values.m`. Proposal: run it once
   through JDyn's own MATLAB interpreter at build time (it is plain assignments) and store the result as a
   data file; options_ then has Dynare's full tree, and the interpreter no longer refuses reads of unset
   options.
3. **Random draws differ.** `stoch_simul(periods>0)` draws shocks with Julia's RNG, so simulated paths
   and moments are not MATLAB's. `shocks=` (or a seed option) is the way to reproduce a run.
4. **Workspace variables.** MATLAB's `dynare` also leaves the parameters and the .mod's own MATLAB
   variables in the base workspace. `ws.vars` holds them; `@dynare` could define them too (opt-in, to avoid
   clobbering the caller's names).
5. **Shapes, not types.** Counts are `Float64` (as MATLAB doubles); `M_.endo_names` is a column cell.
   Julia code that wants native types keeps using the existing API.
