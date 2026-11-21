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
include("matlab/interp.jl")
include("model.jl")
include("matlab/session.jl")
include("steady.jl")
include("perturbation.jl")
include("simulate.jl")
include("korder.jl")
include("moments.jl")
include("priors.jl")
include("kalman.jl")
include("estimation.jl")
include("smoother.jl")
include("shock_decomposition.jl")
include("perfect_foresight.jl")
include("plotting.jl")
include("occbin.jl")
include("results.jl")

export run_preprocessor, load, Model, ModelError, steady_state, SteadyStateError, first_order, FirstOrderSolution,
       BlanchardKahnError, ghx, ghu, irf, simulate, set_param!, set_params!, get_param, set_shock_stderr!,
       stoch_simul_moments, theoretical_moments, k_order, KOrderError, korder_library, plot_irfs, irf_figures, irf_table, write_csv, pltorg, IRFResult, IRFFigure,
       Occbin, OccbinResult, OccbinIRF, OccbinError, OccbinPeriodicError, OccbinMaxitError, OccbinHorizonError, OccbinLoopError,
       OccbinDynareBug, error_flag, irf_occbin, occbin_graph_table, surprise_shocks, mod_options, check_constraints,
       Estimation, EstimationError, estimate, posterior_mode, sample_posterior, log_likelihood, log_prior, log_posterior,
       smooth, SmootherResult, SmootherError, check_smoother, parameter_vector, prior_mode,
       shock_decomposition, initial_condition_decomposition, ShockDecomposition, shock_groups, group_shocks, init2shocks, apply_init2shocks,
       squeeze_shock_decomposition, realtime_shock_decomposition, RealtimeDecomposition, select_vintage, decomposition_plot_data,
       DecompositionPlotData, decomposition_table, stack_bars,
       perfect_foresight, PerfectForesightResult, PerfectForesightError, unanticipated_shocks,
       dynare, @dynare, DynareWorkspace, build_M, build_options, MStruct, Cell

end
