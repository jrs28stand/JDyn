# Licence and provenance

JDyn is free software, released under the GNU General Public License, version 3 (see `LICENSE`). It comes
with no warranty. It is an independent project and is not affiliated with or endorsed by the Dynare Team.

The repository contains no copy of Dynare's or Octave's program code. It is a Julia port of parts of
pydyn (GPL v3 or later, same owner), and through it parts are derived from Dynare 7.2
(https://www.dynare.org), distributed under the GPL, version 3 or later (Copyright 1996-2026 Dynare Team,
see Dynare's `license.txt`):

| File here | Derived from (Dynare 7.2) | Upstream terms |
|---|---|---|
| `src/perturbation.jl` | `matlab/stochastic_solver/dyn_first_order_solver.m`, `stochastic_solvers.m` (loglinear) | GPL-3.0-or-later, Dynare Team |
| `deps/korder/jdyn_korder_capi.cc` | `mex/sources/k_order_perturbation/k_order_perturbation.cc` (a C API replacing the MEX gateway; derived through pydyn's `korder_capi.cc`) | GPL-3.0-or-later, Dynare Team |
| `src/korder.jl` | `matlab/stochastic_solver/k_order_perturbation.m` (the call of the library; derivatives are passed in as computed by the preprocessor's Julia output) | GPL-3.0-or-later, Dynare Team |
| `src/simulate.jl` | `matlab/stochastic_solver/simult_.m`, `irf.m` | GPL-3.0-or-later, Dynare Team |
| `src/moments.jl` | `matlab/moments/*.m` (including `th_autocovariances.m` at order 2), `matlab/+pruned_SS/pruned_state_space_system.m` (order 2), `mex/sources/disclyap_fast/disclyap_fast.cc`, `matlab/moments/*.m` (theoretical and simulated moments, variance decompositions), `matlab/matrix_solver/lyapunov_symm.m`, `matlab/get_lower_cholesky_covariance.m`, `matlab/sample_hp_filter.m`, `matlab/one_sided_hp_filter.m`, the Baxter-King filter of dseries | GPL-3.0-or-later, Dynare Team |
| `src/occbin.jl` | `matlab/+occbin/solve_one_constraint.m`, `solve_two_constraints.m`, `mkdatap_anticipated_dyn.m`, `mkdatap_anticipated_2constraints_dyn.m`, `map_regime.m`, `mkdata.m` (via pydyn's `occbin.py`) | public domain (original authors Luca Guerrieri and Matteo Iacoviello, adapted by the Dynare Team) |
| `src/occbin.jl` (options, IRFs, constraint errors) | `matlab/+occbin/set_default_options.m`, `setup.m`, `solver.m`, `irf.m`, `plot_irfs.m`, `graph.m`; the constraint-error defaults of the preprocessor's `ComputingTasks.cc` (`occbin_difference`) | GPL-3.0-or-later, Dynare Team |
| `src/occbin_filter.jl` | `matlab/+occbin/IVF_core.m`, `match_function.m` (public domain: Cuba-Borda, Guerrieri, Iacoviello and Zhong, adapted by the Dynare Team); `matlab/+occbin/{IVF_posterior, kalman_update_engine, kalman_update_algo_1, kalman_update_algo_3, DSGE_smoother, check_regimes, backward_map_regime, shock_decomposition, squeeze_shock_decomposition, write_regimes_to_xls, set_default_options, setup}.m`, the OccBin branches of `matlab/kalman/likelihood/missing_observations_kalman_filter.m`, `missing_DiffuseKalmanSmootherH3_Z.m`, `DsgeSmoother.m`, and `matlab/optimization/csolve.m` (Christopher Sims, Dynare Team) (via pydyn's `occbin_filter.py`) | GPL-3.0-or-later, Dynare Team (public domain where stated) |
| `src/occbin_estimation.jl` | the OccBin branches of `matlab/estimation/{dsge_likelihood, dynare_estimation_init, initial_estimation_checks, dynare_estimation_1}.m`, `matlab/kalman/evaluate_smoother.m`, `matlab/+occbin/DSGE_smoother.m` (via pydyn's `occbin_estimation.py`) | GPL-3.0-or-later, Dynare Team |
| `test/ref_occbin/` | `occest_nkm.mod` / `occest_nkms.mod` / `occest_rbc.mod` adapted from Dynare's `tests/occbin/filter/NKM*.mod` and `tests/occbin/model_irrcap_twoconstraints`, with their data; `.mat` files are numerical output of Dynare 7.2 under Octave 8.4 (pydyn's references) | GPL-3.0-or-later, Dynare Team (models); not program code (.mat) |
| `test/ref_pac/` | Modified copies of Dynare 7.2's tests in `tests/pac/`, `tests/var-expectations/` and `tests/trend-component-and-var-models/` (cut before their first dseries/simulation line and renamed); `.mat` files are numerical output of Dynare 7.2 under Octave 8.4 | GPL-3.0-or-later, Dynare Team (models); not program code (.mat) |
| `src/pac.jl` | Ports of Dynare 7.2's `matlab/+pac/`, `matlab/+var_expectation/`, `get_companion_matrix.m` and related run-time functions | GPL-3.0-or-later, Dynare Team |
| `src/priors.jl` | `matlab/distributions/*.m` (`beta_specification`, `gamma_specification`, `inverse_gamma_specification`, `lpdf_*`, `priordens`), `matlab/estimation/set_prior.m`, `prior_bounds.m` | GPL-3.0-or-later, Dynare Team |
| `src/kalman.jl` | `matlab/kalman/likelihood/*.m` (`kalman_filter`, `kalman_filter_d`, `univariate_kalman_filter`, `*_d`, `missing_observations_kalman_filter`), `matlab/compute_Pinf_Pstar.m`, `matlab/lyapunov_symm.m` | GPL-3.0-or-later, Dynare Team |
| `src/estimation.jl` | `matlab/estimation/dsge_likelihood.m`, `dynare_estimation_init.m`, `get_estimated_parameters.m`, `posterior_sampler*.m`, `posterior_moments.m`, `marginal_density.m`, `matlab/optimization/csminwel1.m`, `csminit1.m`, `bfgsi1.m`, `numgrad2.m`, `hessian.m` | GPL-3.0-or-later, Dynare Team |
| `src/smoother.jl` | `matlab/kalman/DsgeSmoother.m`, `matlab/kalman/evaluate_smoother.m`, `store_smoother_results.m`, `matlab/kalman/smoother/missing_DiffuseKalmanSmootherH1_Z.m`, `missing_DiffuseKalmanSmootherH3_Z.m` and the other smoother files of `matlab/kalman`, `matlab/estimation/compute_prior_mode.m` (via pydyn's `smoother.py`) | GPL-3.0-or-later, Dynare Team |
| `src/shock_decomposition.jl` | `matlab/shock_decomposition/*.m` (`shock_decomposition`, `initial_condition_decomposition`, `realtime_shock_decomposition`, `squeeze_shock_decomposition`, `plot_shock_decomposition`, `make_the_groups`, `annualized_shock_decomposition` ...) (via pydyn) | GPL-3.0-or-later, Dynare Team |
| `src/perfect_foresight.jl` | `matlab/perfect-foresight-models/*.m` (perfect_foresight_setup, make_ex_, make_y_, perfect_foresight_solver and its core, sim1, sim1_linear, sim1_purely_*, solve_stacked_problem, compute_maxerror, the expectation-errors setup and solver) and the preprocessor's handling of `initval`, `endval`, `histval`, `shocks`, `mshocks` | GPL-3.0-or-later, Dynare Team |
| `src/perfect_foresight.jl` (complementarity solver `lmmcp`) | `matlab/lmmcp/lmmcp.m` | Copyright 2005 Christian Kanzow and Stefania Petra, 2013 Christophe Gouel, 2014-2022 Dynare Team; permissive terms ("Unlimited permission is granted to everyone to use, copy, modify or distribute this software") |
| `src/matlab/session.jl` (option defaults) | `matlab/default_option_values.m` | GPL-3.0-or-later, Dynare Team |
| `test/*.jl` | the models and checks of Dynare's examples and test suite, read from a Dynare source tree at test time (not copied) | GPL-3.0-or-later, Dynare Team |
| `test/matlab_extra.mat` | numerical output of GNU Octave 8.4 | not program code |

All of these were written or modified in 2026. Everything else was written for this project.

`deps/korder/build.sh` compiles Dynare 7.2's `mex/sources/libkorder` (the k-order perturbation, Sylvester and tensor
libraries, GPL-3.0-or-later, Dynare Team) together with `jdyn_korder_capi.cc` into `libjdyn_korder.so`. The library
sources are read from a Dynare source tree given at build time; they are not included in this repository, and the
built library (git-ignored) is a work derived from Dynare under the GPL. `deps/korder/stub/mex.h` is a stand-in for
the MEX header so that the library builds without MATLAB or Octave.

The Julia files the Dynare preprocessor writes for a model (under `<model>/model/julia/`) are generated
at run time and are not part of this repository.

The OccBin authors ask that use of the algorithm be acknowledged by citing: Luca Guerrieri and Matteo
Iacoviello (2015), "OccBin: A toolkit for solving dynamic models with occasionally binding constraints
easily", Journal of Monetary Economics 70, 22-38.

This file records provenance; it is not legal advice.
