# JDyn

Dynare models in Julia, from the same `.mod` files as MATLAB/Octave Dynare, including the MATLAB code
people write between Dynare statements. No MATLAB or Octave is needed at run time.

JDyn is the Julia counterpart of [pydyn](https://github.com/jrs28stand/pydyn) and is validated against the
same Dynare 7.2 reference results. It is an independent project, not affiliated with the Dynare Team; for
the Dynare team's own Julia implementation see [Dynare.jl](https://github.com/DynareJulia/Dynare.jl),
which uses a Julia-flavoured syntax in `.mod` files.

**Status (2026-11-21): first order.** Calibration, steady state, first-order decision rules, IRFs,
simulation and moments are implemented and match Dynare 7.2; higher order, OccBin, perfect foresight,
the Kalman smoother is not. First-order estimation (priors, likelihood, posterior mode, Metropolis-Hastings) is implemented and checked against Dynare. See [docs/STATUS.md](docs/STATUS.md).

## How it works

1. The Dynare 7.2 preprocessor (C++) reads the `.mod` file and writes JSON (symbols, statements) plus Julia
   code for the residuals and derivatives of the model (`language=julia`, up to third order).
2. JDyn runs the statements in order. MATLAB lines in the `.mod` file (calibration code, loops,
   `set_param_value`, `M_.params(i) = ...`, helper functions, `_steadystate.m` files) go through a
   MATLAB-subset interpreter. Code it cannot follow is refused with an error when it could change the
   model, and skipped with a warning otherwise; it never silently gives a different calibration.
3. The numerical work (steady state, QZ-based first-order solution as Dynare's `dyn_first_order_solver`,
   Lyapunov-based moments) is in Julia.

## Requirements

- Julia 1.10 or later (tested with 1.12).
- The Dynare 7.2 preprocessor, built from the Dynare 7.2 sources without Octave or MATLAB:

```sh
curl -fLO http://archive.ubuntu.com/ubuntu/pool/universe/d/dynare/dynare_7.2.orig.tar.xz
mkdir -p dynare-7.2 && tar -xf dynare_7.2.orig.tar.xz -C dynare-7.2 --strip-components=1
cd dynare-7.2/preprocessor && meson setup --buildtype=release ../../pp && meson compile -C ../../pp
export DYNARE_PREPROCESSOR=$PWD/../../pp/src/dynare-preprocessor
```

(build dependencies on Ubuntu: `g++ meson ninja-build flex libfl-dev bison libboost-dev`).

## Use

```julia
using JDyn
m = JDyn.load("model.mod")            # preprocess and run the statements of the file
ys, params = steady_state(m)
sol = first_order(m)                   # FirstOrderSolution: y - ys = G (y(-1) - ys) + H u
ghx(sol), ghu(sol)                     # Dynare's oo_.dr.ghx / ghu (decision-rule order)
irf(sol, [0.01, 0.0]; periods=40)      # deviations, (periods, n_endo)
mom = stoch_simul_moments(m)           # what stoch_simul reports (order 1)

res = JDyn.irf_figures(m, sol)         # Dynare's IRF figures as data (shocks, variables, threshold rules)
JDyn.write_csv("irfs.csv", JDyn.irf_table(res))   # tidy: shock,variable,period,value,order (for ggplot2)
plot_irfs(m, sol; nograph=false)       # SVG files; PNG/PDF with `using Plots` (optional extension)

# parameter loops without re-running the preprocessor
for b in 0.98:0.005:0.995
    set_param!(m, "beta", b)
    sol = first_order(m)
end
```

`Model(...; steady_state_fn=f)` and `functions=Dict("helper" => f)` replace a `_steadystate.m` file or
MATLAB helper functions with Julia code when wanted.

## Tests

`julia --project=. test/runtests.jl` (about 4 minutes). The references are pydyn's (Dynare 7.2 run under
Octave): set `JDYN_REF` to pydyn's `reference/` folder, `JDYN_DYNARE_SRC` to the Dynare 7.2 source tree
(its `examples/` and `tests/` are used) and `DYNARE_PREPROCESSOR` to the preprocessor.

## Licence

GPL v3 (see `LICENSE`). Parts are derived from Dynare 7.2 and from pydyn; see [NOTICE.md](NOTICE.md).
