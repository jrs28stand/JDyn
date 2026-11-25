# JDyn posterior samplers: progress

Branch `samplers` (base main ce58c84). Tests: `test/test_samplers.jl` (references in `test/ref_samplers`).

## Done (validated against Dynare 7.2 under Octave 8.4)

| Item | Where | Validation |
| --- | --- | --- |
| `mode_file` (names matched as `check_mode_file.m`, `stderr x` vs `x`, order, Hessian), `mode_compute=0` + `mh_replic>0` end to end in `dynare()` | `src/mh_files.jl`, `src/mcmc_stage.jl`, `src/results_estimation.jl` | complete run (`smp_nk.mod -DCASE=20`): MH files, `LastSeeds`, posterior mean, MHM equal Dynare's |
| MH files and history (`<fname>/metropolis/*_mh<k>_blck<j>.mat`, `_mh_history_<id>.mat`), `load_mh_file` | `src/mh_files.jl` | second session (CASE 21) continues from Octave's files and from JDyn's own to 2e-15; history fields as Dynare's |
| Draw-for-draw replay with `OctaveRNG` (`dynare(...; mcmc_rng=:octave)`, `set_mcmc_rng!`): chain j seeded `seed+1+..+j`, starting points from the global stream, generator states saved as `LastSeeds` | `src/mcmc_driver.jl` | RWMH 1e-15, Student 3e-16, IMH 9e-16 |
| RWMH variants: Student proposal, `mh_tune_jscale` (Dynare's calibration loop), `mh_jscale`, `mcmc_jumping_covariance` (hessian, prior_variance, identity_matrix, file), `mh_init_scale_factor`, `mh_drop`, `mh_conf_sig`, estimated_params bounds in the proposal check | `src/samplers.jl`, `src/mcmc_driver.jl`, `src/estimation.jl` | Octave runs CASE 22 (tuned scale 0.5617289772 identical), 23, 24 |
| Slice sampler (univariate, rotated with dsyev eigenvectors, multimodal, from prior draws/prior mode/mode) | `src/samplers.jl` | cases 1-4, CASE 25 end to end (1e-10) |
| Independent MH, normal and Student | | cases 6, 10 |
| TaRB-MH (blocks, csminwel on blocks, Student) | | blocks replayed from Dynare's states: all agree to csminwel's accuracy except path-dependent blocks (as in pydyn) |
| SMC `hssmc`, `dsmh` from Dynare's initial particles; `oo_.MarginalDensity.hssmc/.dsmh` | `src/smc.jl` | cases 11, 12: particles 2e-10, mdd 1e-13 |
| `estimate(m)` uses mode files and all samplers | `src/estimation.jl` | test_samplers.jl |

## Behaviour kept
The default RWMH with Julia's generators still goes through the original `sample_posterior`, so existing seeds and tests are unchanged.

## Refused (EstimationError), not ignored
`mh_recover`; the DIME sampler; slice `maximize*`, `draw_init_state_*`, `fast_likelihood_evaluation_for_rejection`, `dime`/online options.

## Branch samplers-2: DSGE-VAR and endogenous_prior (src/dsge_var.jl, test/test_dsge_var.jl, refs in test/ref_dsge_var)
| Item | Validation |
| --- | --- |
| `dsge_var=VALUE` / estimated weight / `dsge_var=Inf`, `dsge_varlag`, sample moments with initial conditions, info 51/52/53, `check_dsge_var_model` refusals | kernels equal Dynare's `dsge_var_likelihood` to 1e-12 (pydyn's est_dvar_v0..2) |
| `oo_.dsge_var.posterior_mode.{PHI_tilde, SIGMA_u_tilde, iXX, prior}` | Octave run at its own mode, 1e-10; Laplace approximation |
| `bayesian_irf` under dsge_var: `oo_.PosteriorIRF.bvardsge.*` (`posterior_irf_dsgevar`, `dsge_var_irf_draw`, rand_inverse_wishart, rand_matrix_normal, qr2) | one draw replays PosteriorIRF_core1.m with randn('state',5) to 1e-13; complete run (CASE 11) chains within 5e-5 of Octave's, DSGE IRFs exact on Dynare's sub-draws |
| `endogenous_prior` (stationary models; missing data, first_obs/nobs, correlated measurement errors) | kernel 1e-15, posterior mode |
| `dsge_var=Inf` in the preprocessor JSON (unquoted `Inf`) | repaired in `_read_modfile_json` |

Limits: the sub-draws and the randn stream of `dynare()`'s posterior IRFs are JDyn's own replay (the chains are Dynare's with `mcmc_rng=:octave`, the sub-draw selection is not), so `bvardsge` agrees with Dynare statistically, not draw for draw; smoother/forecast statistics with dsge_var are not provided.

## Not done
`mh_recover`; Octave's `gamrnd`/`betarnd` cannot be replayed (prior-draw starting points of slice/SMC with several chains use inverse-CDF draws; Dynare's particles are passed in the tests).

## Changes to existing tests (both deliberate, both documented in the test file)
- `test_estimation.jl`, "sampler uses the prior support...": asserted that a proposal outside the estimated_params bounds is accepted. Dynare rejects it (proved by the Octave run smp_case8: draws never exceed rhoa's bound, replayed to 1e-15). The test now asserts the rejection; the kernel part (evaluated with the prior's support) is unchanged.
- `test_estimation.jl`, "option checks": `mode_file` was refused; it is now honoured, so the test asserts the option is accepted and that a missing file is an error.
