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

This file records provenance; it is not legal advice.
