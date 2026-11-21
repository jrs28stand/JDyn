# OccBin estimation work: progress log (branch occbin-est)

Clone: /home/claude/work/jdyn-occbin-est (base main 7fd0a4d). Spec: pydyn fd14d30 (pydyn/occbin_filter.py, occbin_estimation.py).
References: test/ref_occbin/ (pydyn's occest_* files; generator r_occbin_estimation.m, Dynare 7.2 / Octave 8.4).

## Status
- (1) IVF and PKF likelihoods: src/occbin_filter.jl, checked in scratch runs against all reference points:
  IVF max 1.8e-11, PKF max 3.5e-10 (rel 2.3e-13); filtered states 4e-12; regimes identical except rounding ties
  (rbc point 1 period 50; point 2 periods 53-54: two candidate regimes with likelihoods within 2e-15).
- (2) OccBin smoother (PKF and IVF) and occbin shock decomposition: scratch runs match Dynare to <1e-12.
- Hooks: Estimation(...; occbin=Dict) -> _setup_occbin!; _dsge_likelihood -> _occbin_likelihood; smooth() dispatches to
  _occbin_smooth for models with occbin_constraints (src/occbin_estimation.jl). OccbinResult has `ss` (per-period rules).
- test/test_occbin_estimation.jl passes (190 checks); oo_ fields, options_.occbin overrides, occbin_write_regimes done.

## Next
- run test file; commit per piece; oo_ fields (oo_.occbin.smoother/linear_smoother, oo_.Smoother.occbin) in results_estimation;
  options_.occbin overrides in dynare(); occbin_write_regimes; posterior mode slow test; full suite; docs/STATUS.md.
