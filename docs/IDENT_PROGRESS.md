# Identification at order 1 (branch ident)

Port of pydyn's order-1 identification (pydyn/identification.py, params_derivs.py, results_ident.py) to JDyn.
Spec: Dynare 7.2 matlab/+identification, identification_analysis.m.  All code is in `src/identification.jl`; the statement is
hooked into `_run_in_place!` (src/results.jl, 2 lines).  References: `test/ref_identification/` (pydyn's Octave 8.4 results of
Dynare 7.2), tests in `test/test_identification.jl`.

## Status
- [x] plan, generic (Real-typed) model evaluation: the preprocessor's Julia functions are generic in `Real`, so parameter
      derivatives use a small forward-mode dual number type (`_IdDual`, exact, no complex step: the generated functions are typed
      `Vector{<:Real}`, which excludes Complex).  Non-smooth operators (sign, max, min, comparisons, round, floor, ceil, mod,
      external functions) are refused by a text scan of the generated functions; abs, erf, erfc, normcdf, normpdf are differentiated.
- [x] perturbation_params_derivs (modes 0, 1 (Kronecker), -1 (finite differences of everything), -2), steady state derivatives
- [x] checks, cosn, vnorm, minimal state representation (staircase), pruned_first_order, duplication/commutation
- [x] get_jacobians, asymptotic Hessian (multivariate, univariate), analysis, strength, bruteforce, checks_via_subsets
- [x] prior Monte Carlo (prior_mc), advanced, run, statement hook, `_identif.mat` writer
- [ ] all reference cases pass (see below)
- [ ] scope file (orders 2-3 parked), runtests.jl, full suite, cross-port sweep

## Development notes
- Editing src/identification.jl triggers a ~9 minute precompile of JDyn.  For development, comment the include in src/JDyn.jl
  (`#IDENT-DEV`) and run the tests with `ID_DEV=1` (the test file then includes the source into JDyn at run time).
