// est_ur.mod (VARIANT 5: observation trends) of pydyn's references with forecasts (test/test_forecast.jl)
@#define VARIANT = 5
var g y x yobs xobs;
varexo eg ey ex;
parameters rho lam phi mu;
rho = 0.6; lam = 0.3; phi = 0.5; mu = 0.05;

model(linear);
g = rho*g(-1) + eg;
y = y(-1) + g + ey;
x = phi*x(-1) + lam*g + ex;
yobs = y;
xobs = x;
end;

shocks;
var eg; stderr 0.4;
var ey; stderr 0.2;
var ex; stderr 0.3;
var xobs; stderr 0.05;
end;

@#if VARIANT == 4
// maximum likelihood: initial values and bounds only (stderr ey stays calibrated: the
// likelihood of this sample is maximised at ey -> 0, on the bound)
estimated_params;
stderr eg, 0.3, 0.01, 3;
stderr ex, 0.25, 0.01, 3;
rho, 0.5, 0, 0.99;
lam, 0.2, -2, 2;
phi, 0.4, 0, 0.99;
end;
@#else
estimated_params;
stderr eg, inv_gamma_pdf, 0.4, inf;
stderr ey, inv_gamma_pdf, 0.2, inf;
stderr ex, 0.25, inv_gamma2_pdf, 0.3, 0.5;
rho, beta_pdf, 0.5, 0.2;
lam, normal_pdf, 0.2, 0.2;
phi, 0.4, beta_pdf, 0.5, 0.2;
@#if VARIANT == 5
mu, normal_pdf, 0.05, 0.05;
@#endif
end;
@#endif

varobs yobs xobs;

@#if VARIANT == 5
// linear trends in the measurement equations, with an estimated slope
observation_trends;
yobs (mu);
xobs (2*mu - 0.1);
end;
@#endif
estimation(datafile=est_ur_data, diffuse_filter, first_obs=5, mode_compute=0, mh_replic=0, smoother, forecast=6, nodisplay, nograph) y x yobs g;
FE = oo_.forecast;
conditional_forecast_paths;
var yobs;
periods 1:3;
values 2.5;
end;
@#ifdef OCTAVE
set_dynare_seed(31); CF1_draws = fc_draws(M_.exo_nbr, 6, 2*25); set_dynare_seed(31);
@#endif
conditional_forecast(parameter_set=prior_mean, controlled_varexo=(eg), replic=25, periods=6);
CF1 = oo_.conditional_forecast;
