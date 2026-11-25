# estimation / calib_smoother probes: a small model estimated on simulated data; the base run samples (mh_replic) so that
# post-sampling options act. julia --project=. tools/options_probe_est.jl
isdefined(Main, :probe) || include(joinpath(@__DIR__, "options_probe.jl"))
using Random
const EST_MOD = """var y c k a; varexo e;
parameters alpha beta delta rho sig;
alpha=0.36;beta=0.99;delta=0.025;rho=0.9;sig=0.01;
model;
c^(-1)=beta*c(+1)^(-1)*(alpha*a(+1)*k^(alpha-1)+1-delta);
k=a*k(-1)^alpha+(1-delta)*k(-1)-c;
y=a*k(-1)^alpha;
log(a)=rho*log(a(-1))+e;
end;
initval; k=30;c=2.5;a=1;y=3;e=0;end;
steady;
shocks; var e; stderr 0.01; end;
varobs y;
estimated_params;
rho, beta_pdf, 0.8, 0.1;
stderr e, inv_gamma_pdf, 0.01, inf;
end;
"""
rng = MersenneTwister(1)
dat = "y = [" * join([string(3 + 0.02 * randn(rng)) for _ in 1:80], ";") * "];\nc = [" * join([string(2.5 + 0.02 * randn(rng)) for _ in 1:80], ";") * "];\n"
files = Dict("dd.m" => dat)
base = "estimation(datafile=dd,mode_compute=4,mh_replic=200,mh_nblocks=1,nograph,mh_jscale=0.5);"
opts = ["smoother", "filtered_vars", "filter_step_ahead=[1,2]", "filter_covariance", "smoothed_state_uncertainty", "filter_decomposition", "forecast=4",
        "moments_varendo", "bayesian_irf", "irf=5", "conditional_variance_decomposition=[1,3]", "ar=2", "contemporaneous_correlation", "nodecomposition",
        "relative_irf", "sub_draws=10", "mh_conf_sig=0.8", "mh_drop=0.2", "mh_init_scale=0.001", "mh_init_scale_factor=1", "posterior_max_subsample_draws=50",
        "raftery_lewis_diagnostics", "geweke_interval=[0.2,0.5]", "taper_steps=[4,8,15]",
        "brooks_gelman_plotrows=2", "nodiagnostic", "prior_trunc=1e-4", "lik_init=2", "kalman_algo=2", "kalman_tol=1e-6", "diffuse_filter", "prefilter=1",
        "presample=2", "first_obs=3", "nobs=70", "loglinear", "logdata", "noconstant", "cova_compute=0", "solve_algo=1", "qz_zero_threshold=1e-5",
        "aim_solver", "partial_information", "huge_number=1e5", "analytic_derivation", "additional_optimizer_steps=[4]", "optim=('MaxIter',3)",
        "no_init_estimation_check_first_obs", "use_univariate_filters_if_singularity_is_detected=0", "smoother_redux", "heteroskedastic_filter",
        "mode_check", "mode_check_neighbourhood_size=0.1", "mode_check_number_of_points=10", "plot_priors=0", "rescale_prediction_error_covariance",
        "selected_variables_only", "mh_posterior_mode_estimation", "lyapunov=doubling", "lyapunov_fixed_point_tol=1e-8",
        "lyapunov_doubling_tol=1e-10", "dsge_var=0.5", "frequentist_smoother", "endogenous_prior", "mh_drop=0.1", "xls_sheet=a", "hessian_use_penalized_objective"]
if abspath(PROGRAM_FILE) == @__FILE__
probe([(base, [replace(base, ");" => ",$o);") for o in opts])]; pre=EST_MOD, files=files)
    base2 = "estimation(datafile=dd,mode_compute=4,mh_replic=200,mh_nblocks=2,nograph,mh_jscale=0.5,smoother,filtered_vars,moments_varendo,bayesian_irf,irf=4,sub_draws=5,order=1);"
    opts2 = ["filter_decomposition", "filter_covariance", "irf=6", "ar=2", "conditional_variance_decomposition=[1,3]", "contemporaneous_correlation", "nodecomposition",
             "relative_irf", "sub_draws=10", "selected_variables_only", "mh_init_scale_factor=1", "nodiagnostic", "posterior_max_subsample_draws=20", "mh_conf_sig=0.8", "mh_drop=0.2",
             "kalman_algo=3", "smoothed_state_uncertainty", "filter_step_ahead=[1]", "logdata,loglinear", "forecast=3", "consider_all_endogenous", "irf_shocks=(e)"]
    probe([(base2, [replace(base2, ");" => ",$o);") for o in opts2])]; pre=EST_MOD, files=files)
end
