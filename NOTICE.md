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
| `src/simulate.jl` | `matlab/stochastic_solver/simult_.m`, `irf.m` | GPL-3.0-or-later, Dynare Team |
| `src/moments.jl` | `matlab/moments/*.m` (theoretical and simulated moments, variance decompositions), `matlab/matrix_solver/lyapunov_symm.m`, `matlab/get_lower_cholesky_covariance.m`, `matlab/sample_hp_filter.m`, `matlab/one_sided_hp_filter.m`, the Baxter-King filter of dseries | GPL-3.0-or-later, Dynare Team |
| `src/priors.jl` | `matlab/distributions/*.m` (`beta_specification`, `gamma_specification`, `inverse_gamma_specification`, `lpdf_*`, `priordens`), `matlab/estimation/set_prior.m`, `prior_bounds.m` | GPL-3.0-or-later, Dynare Team |
| `src/kalman.jl` | `matlab/kalman/likelihood/*.m` (`kalman_filter`, `kalman_filter_d`, `univariate_kalman_filter`, `*_d`, `missing_observations_kalman_filter`), `matlab/compute_Pinf_Pstar.m`, `matlab/lyapunov_symm.m` | GPL-3.0-or-later, Dynare Team |
| `src/estimation.jl` | `matlab/estimation/dsge_likelihood.m`, `dynare_estimation_init.m`, `get_estimated_parameters.m`, `posterior_sampler*.m`, `posterior_moments.m`, `marginal_density.m`, `matlab/optimization/csminwel1.m`, `csminit1.m`, `bfgsi1.m`, `numgrad2.m`, `hessian.m` | GPL-3.0-or-later, Dynare Team |
| `src/matlab/session.jl` (option defaults) | `matlab/default_option_values.m` | GPL-3.0-or-later, Dynare Team |
| `test/*.jl` | the models and checks of Dynare's examples and test suite, read from a Dynare source tree at test time (not copied) | GPL-3.0-or-later, Dynare Team |
| `test/matlab_extra.mat` | numerical output of GNU Octave 8.4 | not program code |

All of these were written or modified in 2026. Everything else was written for this project.

The Julia files the Dynare preprocessor writes for a model (under `<model>/model/julia/`) are generated
at run time and are not part of this repository.

This file records provenance; it is not legal advice.
