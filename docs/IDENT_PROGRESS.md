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
- [x] reference cases: the 12 pydyn references plus 4 new Octave 8.4 / Dynare 7.2 runs (forward_looking_empty_ghx, ident_unit_root,
      rbc_ident_varexo_only, rbc_ident_std_as_structural_par) pass; registered in test/runtests.jl
- [ ] scope file (orders 2-3 parked), full suite, cross-port sweep (see Log)

## Behaviour worth knowing
- Order 2 and 3 are refused with an `IdentificationError` ("orders 2 and 3 are not ported"); checks_via_subsets together with
  prior_mc>1 is refused; non-smooth model functions are refused; a steady state file that changes parameters is refused for
  analytic_derivation_mode 0/1 (Dynare is silently wrong there).
- Unit roots: the moments of non-stationary observables are NaN as in Dynare (stationary_vars, schur_vec_tol); the diffuse filter
  has no analytic Hessian, so the strength measure falls back to simulated moment uncertainty (ident_unit_root matches Octave).
- Dynare quirk kept on purpose: for a model without state variables (isempty(dr.ghx)) `simult_` returns the first period in the
  column that `simult` drops as the "initial value", so the simulated samples lose one observation (forward_looking_empty_ghx).
- Not produced: .tex/.aux collinearity files (`tex`), plots (`nograph`-like; JDyn treats these as cosmetic).
- `parameter_set` is limited to calibration and prior_mean here.
- `replicate_dynare_bugs`: no Dynare defect found that needed it.

## Known differences against fresh Octave runs (not bugs)
- Fields decided by rank / exact-zero tests on matrices with condition number ~1e16-1e20: `indno` (all blocks of
  as2007_kronflags), `jweak`, `jweak_pair`, spectrum/dynamic `indno` (as2007_QT_equal_autocorr), cond, Mco, Pco, V.  Agreement
  of the rest is 1e-6 (rbc_ident_*: 1e-7 rounding).
- `ide_minimal_point.dMINIMAL` for fs2000_corr: the minimal state-space basis is not unique (the criterion built from it agrees).
- Prior Monte Carlo (as2007, fs2000_corr: pdraws, ide_*_max/min, IDE_*): gamma/beta/inverse-gamma draws are not reproducible
  (Octave randg is not seeded); uniform and normal draws replay exactly.


## Development notes
- Editing src/identification.jl triggers a ~9 minute precompile of JDyn.  For development, comment the include in src/JDyn.jl
  (`#IDENT-DEV`) and run the tests with `ID_DEV=1` (the test file then includes the source into JDyn at run time).
