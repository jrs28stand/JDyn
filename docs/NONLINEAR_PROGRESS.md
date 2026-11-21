# Nonlinear estimation (order 2) - running log

Branch nonlinear-est (base main 7fd0a4d). Spec: pydyn nonlinear_filters.py / nonlinear_estimation.py / octave_rng.py.

## Done (uncommitted-tested state as of this entry)
- src/octave_rng.jl: Octave MT19937 + ziggurat, bit for bit vs test/ref_nonlinear/octave_rng.mat (OctaveRNG, streams, set_rng_kind!).
- src/nonlinear_filters.jl: sis, apf, cpf, gf, nlkf; checked on Dynare's reduced form (dev harness /tmp/dev/t1.jl): per-period diff <= 4e-10.
  Timing (reduced form only): sis N=20000 0.07 s, apf 0.22 s, cpf N=1000 0.18 s, gf/nlkf ~ms.
- src/nonlinear_estimation.jl + small hooks in estimation.jl (order 2 dispatch), results_estimation.jl, results.jl (dynare(particle_rng=)), korder.jl (Sigma_e kw), JDyn.jl.
- test/test_nonlinear_estimation.jl written (NOT yet run); e2e Octave refs being generated in /tmp/r2/e2e (copy *_oo.mat, *_M.mat into test/ref_nonlinear/e2e).

## Next
1. Run smoke (/tmp/dev/smoke.jl) and test file; fix. 2. Add test to runtests.jl ALL_TESTS. 3. Docs (STATUS.md, RESULTS.md). 4. Full suite once. 5. Final report.
