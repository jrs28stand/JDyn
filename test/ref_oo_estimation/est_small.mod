// Small NK model for checking the oo_ layout of estimation / calib_smoother (JDyn vs Dynare 7.2)
@#ifndef VARIANT
@#define VARIANT = 0
@#endif
var y pi r a u yobs piobs robs;
varexo ea eu er;
parameters beta kappa phipi phiy rhoa rhou rhor sig pibar rbar;

beta = 0.99; kappa = 0.1; phipi = 1.5; phiy = 0.25; rhoa = 0.8; rhou = 0.5; rhor = 0.7;
sig = 1.5; pibar = 0.5; rbar = 0;

model;
y = y(+1) - (1/sig)*(r - pi(+1)) + a;
pi = beta*pi(+1) + kappa*y + u;
r = rhor*r(-1) + (1-rhor)*(phipi*pi + phiy*y) + er;
a = rhoa*a(-1) + ea;
u = rhou*u(-1) + eu;
yobs = y;
piobs = pibar + pi;
robs = rbar + pibar + r;
end;

steady_state_model;
rbar = 100*(1/beta - 1);
y = 0; pi = 0; r = 0; a = 0; u = 0;
yobs = 0;
piobs = pibar;
robs = rbar + pibar;
end;

shocks;
var ea; stderr 0.5;
var eu; stderr 0.3;
var er; stderr 0.2;
corr ea, eu = 0.1;
var yobs; stderr 0.1;
var piobs; stderr 0.05;
end;

@#if VARIANT == 5
estimated_params;
rhoa, 0.7, 0.01, 0.99;
stderr ea, 0.4, 0.01, 3;
stderr eu, 0.3, 0.01, 3;
stderr yobs, 0.1, 0.001, 1;
end;
@#else
estimated_params;
stderr ea, inv_gamma_pdf, 0.5, 2;
stderr eu, inv_gamma2_pdf, 0.1, 1;
stderr yobs, 0.12, uniform_pdf, , , 0, 1;
corr ea, eu, 0.05, -0.9, 0.9, normal_pdf, 0, 0.3;
corr yobs, piobs, 0.2, uniform_pdf, , , -0.9, 0.9;
rhoa, beta_pdf, 0.7, 0.1;
end;
@#endif

varobs yobs piobs robs;

@#if VARIANT == 0
estimation(datafile=est_nk_data, mode_compute=4, mh_replic=0, nodisplay) y pi r a u yobs piobs robs;
@#endif
@#if VARIANT == 1
estimation(datafile=est_nk_data, mode_compute=0, mh_replic=0, smoother, filtered_vars, filter_step_ahead=[1 3], filter_covariance, smoothed_state_uncertainty, nodisplay) y pi r a u yobs piobs robs;
@#endif
@#if VARIANT == 2
estimation(datafile=est_nk_data, mode_compute=0, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 3
estimation(datafile=est_nk_data, mode_compute=4, mh_replic=300, mh_nblocks=2, mh_drop=0.5, mh_jscale=0.5, nodiagnostic, nodisplay) y pi r a u yobs piobs robs;
@#endif
@#if VARIANT == 4
calib_smoother(datafile=est_nk_data, filtered_vars, filter_step_ahead=[1 2], smoothed_state_uncertainty) y pi r a u yobs piobs robs;
@#endif
@#if VARIANT == 5
estimation(datafile=est_nk_data, mode_compute=4, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 6
estimation(datafile=est_nk_miss, mode_compute=4, mh_replic=0, nodisplay, smoother) y pi r a u yobs piobs robs;
@#endif
