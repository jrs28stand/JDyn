# Second round of acceptance-sweep fixes (branch acc2)

Restart notes: clone /home/claude/work/jdyn-acc2 (branch acc2 from main 1bb4530); baseline worktree of the same commit plus the
runner fix at /home/claude/work/jdyn-acc2-base (precompiled; used for the baseline sweep into /home/claude/acc2_work/base_new,
tag acc2b). My own sweeps use ACCEPT_TAG=acc2. Manifest.toml and deps/korder/libjdyn_korder.so are copied from jdyn-parity3 (git-ignored).
Comparator: pydyn-rl/tools/accept_compare.py (always use --only / head: its full output is huge).
Precompiling next to the other agent fails transiently ("Failed to precompile" without a message): run
`Base.compilecache(Base.identify_package("JDyn"))` and retry.

Baseline (main code, new runner, current comparator): match 321, partial 49, refused 155, wrong 145, port-error 49, dynare-failed 61,
no-comparison 12, xfail 28, excluded 26 (846 tests).

## Status per item
1. Runner: DONE (test/acceptance/accept_jdyn.py: a test never starts with Dynare's own outputs; earlier files of the entry that did
   not complete get Dynare's outputs, no-overwrite; --ref-artifacts restores the old behaviour). "no .mat written" did not reproduce
   with the new runner (0 cases in a full --jobs 2 sweep); the old pre-copy of artifacts is the probable cause.
2. MATLAB statements that change the model / other interpreter refusals:
   - cellfun, arrayfun, regexp, regexpi, regexprep (builtins_extra.jl); chol(A,'lower'|'upper'); datatomfile (writes the .m data file
     of oo_.endo_simul; 20 .mod files use it: arima, dsge-var, kalman/likelihood_from_dynare, kalman_filter_smoother, ...).
   - `values (a)` with a vector a in shocks / mshocks / shocks(surprise) / conditional_forecast_paths: one value per period.
   - A MATLAB re-assignment of an options_ field to the value it already had counts after a Dynare command changed the field
     (`options_.lik_init=1` between estimations): before, the second assignment was silently lost (results.jl `_apply_matlab_options!`).
   - steady / stoch_simul no longer taint workspace variables named after the model's variables (Dynare writes none; only
     send_*_to_workspace does): block_bytecode/lola_* refused on `n0 = n0_fss` in endval.
   Still refused (reasons): sur/dyn_ols/pooled_ols/pooled_fgls/surgibbs/olsgibbs (19 files, a port of ols.m/sur.m is needed),
   M_.matched_moments and method_of_moments (7), dynare_estimation_init objects (8 kalman/lik_init files), set_prior
   (analytic_derivatives/*PertParamsDerivs), set_parameters_locally + functional stoch_simul (moments/fs2000_post_moments),
   `prior simulate|plot|table` command forms (gsa/ls2003*), set_shock_skew_value, feval of <fname>.trend_component_ar_a0 (tcm6).
3. filter_decomposition: DONE (smoother.jl, both filters; oo_.FilteredVariablesShockDecomposition). selected_variables_only: the
   k-step arrays keep the saved variables only (Dynare's layout), also for calib_smoother. Recursive estimation: DONE
   structurally (results_estimation.jl `_estimation_recursive!`), values depend on optimiser end points (recursive/ls2003 differs
   at the posterior mode: log density -633.87 in Octave, -634.34 in JDyn). Several estimated_params blocks (overwrite flag read
   from the .mod text), estimation without estimated_params and mode_compute=0 + smoother routes to calib_smoother.
4. Data files without dates: DONE (`_data_start_date`: year one, annual, as Dynare's load_*_file_data).
5. Numerical:
   - newrat singular outer-product Hessian: Octave's minimum-norm solve (`_octave_solve`) instead of SingularException
     (kalman_filter_smoother/algo3, algoH1, algoH3 now run).
   - Negative forecast-error determinant: Dynare's log is complex and the likelihood is Inf (info 46); JDyn threw DomainError.
   - stoch_simul on a linear model (M_.hessian_eq_zero) reverts to order 1 as Dynare; gamma_y cell count 8 vs 7 fixed
     (ls2003_hessian_zero, linear_exo_det_forecast).
   - The `forecast` statement ignored varexo_det paths (oo_.forecast.Exogenous missing, means flat): now as forecast(model).
   - oo_.steady_state after initval/endval carries the auxiliary variables (leads, lags, log) as the preprocessor writes them
     (stochastic_simulations/example1long, log_transform/example1, model_info/bkk, ...).
   - aim_solver: dr.eigval = eig of the state transition, as AIM_first_order_solver.m (AIM/*_AIM).
   - estimation options accepted with the standard algorithm: fast_kalman_filter (Chandrasekhar recursions, same likelihood),
     lyapunov=doubling|fixed_point|square_root_solver (same solution).
   - estimated_params_init / _bounds: `corr a, b` is `corr b, a`.
   Not JDyn's: Ramsey ramsey_ex_aux (5e-8 relative in higher moments: Dynare's 1e-5 steady-state tolerance), OSR opt_algo=4 end points,
   the walsh model (20 blocks of a near-unit-root money model: ghx differs, not investigated), the loglinear results layer
   (filter_step_ahead/*ML dr.ys, owned by the other agent).
6. Interpreter refusals not done: see item 2 list; pac.print / var_expectation.print are no-ops, so the 10
   pac/*/substitution and var-expectations/*/substitution files (which @#include the files print writes) still fail at the
   preprocessor (a port of print_expectations.m / write_expectations.m is needed).
