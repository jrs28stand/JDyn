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
| `src/occbin.jl` | `matlab/+occbin/solve_one_constraint.m`, `solve_two_constraints.m`, `mkdatap_anticipated_dyn.m`, `mkdatap_anticipated_2constraints_dyn.m`, `map_regime.m`, `mkdata.m` (via pydyn's `occbin.py`) | public domain (original authors Luca Guerrieri and Matteo Iacoviello, adapted by the Dynare Team) |
| `src/occbin.jl` (options, IRFs, constraint errors) | `matlab/+occbin/set_default_options.m`, `setup.m`, `solver.m`, `irf.m`, `plot_irfs.m`, `graph.m`; the constraint-error defaults of the preprocessor's `ComputingTasks.cc` (`occbin_difference`) | GPL-3.0-or-later, Dynare Team |
| `src/matlab/session.jl` (option defaults) | `matlab/default_option_values.m` | GPL-3.0-or-later, Dynare Team |
| `test/*.jl` | the models and checks of Dynare's examples and test suite, read from a Dynare source tree at test time (not copied) | GPL-3.0-or-later, Dynare Team |
| `test/matlab_extra.mat` | numerical output of GNU Octave 8.4 | not program code |

All of these were written or modified in 2026. Everything else was written for this project.

The Julia files the Dynare preprocessor writes for a model (under `<model>/model/julia/`) are generated
at run time and are not part of this repository.

The OccBin authors ask that use of the algorithm be acknowledged by citing: Luca Guerrieri and Matteo
Iacoviello (2015), "OccBin: A toolkit for solving dynamic models with occasionally binding constraints
easily", Journal of Monetary Economics 70, 22-38.

This file records provenance; it is not legal advice.
