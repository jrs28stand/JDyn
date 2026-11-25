# Identification at order 1 (branch ident)

Port of pydyn's order-1 identification (pydyn/identification.py, params_derivs.py, results_ident.py) to JDyn.
Spec: Dynare 7.2 matlab/+identification, identification_analysis.m.

## Plan / status
- [ ] 1. progress doc, generic (Real-typed) model evaluation wrappers
- [ ] 2. perturbation_params_derivs (modes 0, 1, -1, -2), steady state derivatives
- [ ] 3. helpers: checks, cosn, vnorm, minimal state representation, pruned_first_order, duplication/commutation
- [ ] 4. get_jacobians, asymptotic Hessian (multivariate, univariate)
- [ ] 5. analysis, strength, checks_via_subsets, bruteforce, prior MC, run
- [ ] 6. statement hook, `_identif.mat` writer
- [ ] 7. scope file (orders 2-3 parked), tests in runtests.jl, full suite, cross-port sweep

## Notes
