"""
JDyn: Dynare models in Julia, with the same `.mod` files as MATLAB/Octave Dynare.

The Dynare 7.2 preprocessor (C++) reads the `.mod` file and writes JSON plus Julia code for the model
equations and their derivatives; JDyn runs the statements of the file in order (including the MATLAB
lines between them, through a MATLAB-subset interpreter) and does the numerical work in Julia.
No MATLAB or Octave is needed at run time.
"""
module JDyn

using LinearAlgebra
using SparseArrays
using LinearAlgebra: lu, issuccess, schur, ordschur!, qr, UpperTriangular, opnorm, pinv
using Printf
using Logging
using Random
import JSON
import MAT
import NLsolve
import SpecialFunctions

include("matlab/values.jl")
include("matlab/parser.jl")
include("matlab/builtins.jl")
include("matlab/builtins_extra.jl")
include("matlab/interp.jl")
include("matlab/dates.jl")
include("matlab/dseries.jl")
include("matlab/objects.jl")
include("octave_rng.jl")
include("model.jl")
include("matlab/session.jl")
include("pac.jl")
include("dmperm.jl")
include("steady.jl")
include("perturbation.jl")
include("simulate.jl")
include("korder.jl")
include("fft.jl")
include("moments.jl")
include("priors.jl")
include("kalman.jl")
include("nonlinear_filters.jl")
include("estimation.jl")
include("nonlinear_estimation.jl")
include("optimizers.jl")
include("smoother.jl")
include("shock_decomposition.jl")
include("perfect_foresight.jl")
include("optimal_policy.jl")
include("plotting.jl")
include("occbin.jl")
include("occbin_filter.jl")
include("occbin_estimation.jl")
include("results.jl")
include("results_policy.jl")
include("pf_functions.jl")
include("results_estimation.jl")
include("seeds.jl")
include("forecast.jl")
include("mcmc_diagnostics.jl")
include("posterior.jl")
include("posterior_forecast.jl")
include("posterior_moments.jl")
include("posterior_stage.jl")
include("mh_files.jl")
include("samplers.jl")
include("mcmc_driver.jl")
include("smc.jl")
include("mcmc_stage.jl")
include("dsge_var.jl")

export posterior_irf_dsgevar, dsge_var_posterior, endogenous_log_prior, sample_mcmc, sample_smc, set_mcmc_rng!, mode_from_file, mcmc_diagnostics, posterior_irf, posterior_forecast, posterior_subsample, posterior_theoretical_moments,
       posterior_smoother_statistics,
       run_preprocessor, load, Model, ModelError, steady_state, SteadyStateError, first_order, FirstOrderSolution,
       BlanchardKahnError, ghx, ghu, irf, simulate, set_param!, set_params!, get_param, set_shock_stderr!,
       stoch_simul_moments, theoretical_moments, k_order, KOrderError, korder_library, plot_irfs, irf_figures, irf_table, write_csv, pltorg, IRFResult, IRFFigure,
       Occbin, OccbinResult, OccbinIRF, OccbinError, OccbinPeriodicError, OccbinMaxitError, OccbinHorizonError, OccbinLoopError,
       OccbinDynareBug, error_flag, irf_occbin, occbin_graph_table, surprise_shocks, mod_options, check_constraints,
       occbin_filter_options, inversion_filter, piecewise_kalman_filter, occbin_smoother, occbin_shock_decomposition,
       squeeze_occbin_shock_decomposition, regime_table, write_regime_table, OccbinSmootherResult, PKFResult,
       InversionFilterResult, occbin_filter, occbin_at, occbin_write_regimes!,
       Estimation, EstimationError, estimate, posterior_mode, sample_posterior, log_likelihood, log_prior, log_posterior,
       smooth, SmootherResult, SmootherError, check_smoother, parameter_vector, prior_mode,
       shock_decomposition, initial_condition_decomposition, ShockDecomposition, shock_groups, group_shocks, init2shocks, apply_init2shocks,
       squeeze_shock_decomposition, realtime_shock_decomposition, RealtimeDecomposition, select_vintage, decomposition_plot_data,
       DecompositionPlotData, decomposition_table, stack_bars,
       perfect_foresight, PerfectForesightResult, PerfectForesightError, unanticipated_shocks,
       dynare, @dynare, DynareWorkspace, build_M, build_options, MStruct, Cell,
       stoch_simul!, ramsey_policy!, discretionary_policy!, evaluate_planner_objective!, osr!, steady!, check!, perfect_foresight_setup!, perfect_foresight_solver!, occbin_solver!, set_param_value!,
       estimation!, calib_smoother!, set_dynare_seed!,
       ramsey_info, is_ramsey, is_discretionary, planner_discount, ramsey_steady_state, planner_objective, evaluate_planner_objective,
       PlannerObjectiveValue, discretionary_policy, DiscretionaryResult, osr, osr_objective, OSRResult, OptimalPolicyError, OSRError,
       DiscretionaryPolicyError, variance_of_endogenous_variables,
       forecast, ForecastResult, conditional_forecast, ConditionalForecastResult, conditional_forecast_table, forecast_struct,
       conditional_forecast_struct, forecast!, conditional_forecast!

include("precompile.jl")

end
