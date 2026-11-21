# OccBin estimation work: progress log (branch occbin-est)

Clone: /home/claude/work/jdyn-occbin-est (base main 7fd0a4d). Spec: pydyn fd14d30 (pydyn/occbin_filter.py, occbin_estimation.py).
References: test/ref_occbin/ (pydyn's occest_* files; generator r_occbin_estimation.m, Dynare 7.2 / Octave 8.4).

## Done
- (1) IVF and PKF likelihoods (src/occbin_filter.jl): IVF 1.8e-11, PKF 3.5e-10 (rel 2.3e-13) at every reference point;
  states 4e-12; regimes identical except three rounding ties (rbc periods 50, 53, 54).
- (2) OccBin smoother (PKF, IVF), occbin shock decomposition, squeeze, regime table: 1e-12 / 2e-15.
- (3) Estimation: hooks in Estimation/_dsge_likelihood (src/occbin_estimation.jl); Hessian 9e-10 rel, Laplace to 2e-7;
  posterior mode (slow test) 2.2e-4 sd; MH end to end (0.2 s per draw on the NKM-small model).
- (4) oo_: Smoother.occbin, oo_.occbin.smoother / linear_smoother, occbin_solver regime_history, occbin_write_regimes,
  options_.occbin overrides; dynare() test against the NKM smoother reference.
- test/test_occbin_estimation.jl in runtests.jl.

## Open
- Full suite (test/runtests.jl, 25 files): all passed.
- Not implemented (refused): occbin.forecast, particle filters, loss-function regime guess, init_regime_history,
  smoother_first_period_occbin_update != 1, filter_covariance / smoothed_state_uncertainty / filter_step_ahead > 1 with OccBin,
  trends in the OccBin smoother, Bayesian smoother over draws.
