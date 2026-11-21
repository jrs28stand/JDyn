// Forecasts and conditional forecasts of the calibrated Schorfheide (2000) model (Dynare's
// tests/conditional_forecasts/1/fs2000_cal.mod) checked against Dynare 7.2 (test/forecast/r_forecast.m).
// VARIANT 0: levels; 1: measurement errors on the observed variables; 2: loglinear; 3: order 2.
// Under Octave (-DOCTAVE) the random numbers of the stochastic statements are recorded first.
@#ifndef VARIANT
@#define VARIANT = 0
@#endif
var m P c e W R k d n l gy_obs gp_obs y dA;
varexo e_a e_m;

parameters alp bet gam mst rho psi del;

alp = 0.33;
bet = 0.99;
gam = 0.003;
mst = 1.011;
rho = 0.7;
psi = 0.787;
del = 0.02;

model;
dA = exp(gam+e_a);
log(m) = (1-rho)*log(mst) + rho*log(m(-1))+e_m;
-P/(c(+1)*P(+1)*m)+bet*P(+1)*(alp*exp(-alp*(gam+log(e(+1))))*k^(alp-1)*n(+1)^(1-alp)+(1-del)*exp(-(gam+log(e(+1)))))/(c(+2)*P(+2)*m(+1))=0;
W = l/n;
-(psi/(1-psi))*(c*P/(1-n))+l/n = 0;
R = P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(-alp)/W;
1/(c*P)-bet*P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)/(m*l*c(+1)*P(+1)) = 0;
c+k = exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)+(1-del)*exp(-(gam+e_a))*k(-1);
P*c = m;
m-1+d = l;
e = exp(e_a);
y = k(-1)^alp*n^(1-alp)*exp(-alp*(gam+e_a));
gy_obs = dA*y/y(-1);
gp_obs = (P/P(-1))*m(-1)/dA;
end;

steady_state_model;
  dA = exp(gam);
  gst = 1/dA;
  m = mst;
  khst = ( (1-gst*bet*(1-del)) / (alp*gst^alp*bet) )^(1/(alp-1));
  xist = ( ((khst*gst)^alp - (1-gst*(1-del))*khst)/mst )^(-1);
  nust = psi*mst^2/( (1-alp)*(1-psi)*bet*gst^alp*khst^alp );
  n  = xist/(nust+xist);
  P  = xist + nust;
  k  = khst*n;
  l  = psi*mst*n/( (1-psi)*(1-n) );
  c  = mst/P;
  d  = l - mst + 1;
  y  = k^alp*n^(1-alp)*gst^alp;
  R  = mst/bet;
  W  = l/n;
  e = 1;
  gp_obs = m/dA;
  gy_obs = dA;
end;

@#if VARIANT == 1
varobs gp_obs gy_obs;
@#endif

shocks;
var e_a; stderr 0.014;
var e_m; stderr 0.005;
@#if VARIANT == 1
var gy_obs; stderr 0.002;
var gp_obs; stderr 0.003;
@#endif
end;

steady;

@#if VARIANT == 2
stoch_simul(order=1, irf=0, loglinear);
@#define LV1 = "0.01"
@#define LV2 = "-0.004"
@#define LV3 = "0.003"
@#define LP = "0.012"
@#define LP2 = "0.009"
@#elseif VARIANT == 3
stoch_simul(order=2, irf=0);
@#else
stoch_simul(order=1, irf=0);
@#define LV1 = "1.01"
@#define LV2 = "0.996"
@#define LV3 = "1.003"
@#define LP = "1.012"
@#define LP2 = "1.009"
@#endif

@#if VARIANT == 3
options_.forecast_replic = 200;
@#ifdef OCTAVE
set_dynare_seed(5); F1_draws = fc_draws(M_.exo_nbr, 6, options_.forecast_replic); set_dynare_seed(5);
@#endif
forecast(periods=6);
F1 = oo_.forecast;
histval;
k(0) = 6;
m(0) = 1.01;
end;
@#ifdef OCTAVE
set_dynare_seed(6); F2_draws = fc_draws(M_.exo_nbr, 5, options_.forecast_replic); set_dynare_seed(6);
@#endif
forecast(periods=5, conf_sig=0.8) k gy_obs;
F2 = oo_.forecast;
@#else
forecast(periods=12, conf_sig=0.8);
F1 = oo_.forecast;
forecast(periods=6) gy_obs k gp_obs;
F2 = oo_.forecast;

// all shocks controlled: deterministic
conditional_forecast_paths;
var gy_obs;
periods 1 2 3:5;
values @{LV1} @{LV2} @{LV3};
var gp_obs;
periods 1:7;
values @{LP};
end;
conditional_forecast(parameter_set=calibration, controlled_varexo=(e_a,e_m), replic=20, periods=12);
CF1 = oo_.conditional_forecast;

// one controlled shock, the other drawn
conditional_forecast_paths;
var gp_obs;
periods 1:2 4;
values @{LP2} @{LP};
end;
@#ifdef OCTAVE
set_dynare_seed(11); CF2_draws = fc_draws(M_.exo_nbr, 10, 2*50); set_dynare_seed(11);
@#endif
conditional_forecast(parameter_set=calibration, controlled_varexo=(e_m), replic=50, periods=10, conf_sig=0.7);
CF2 = oo_.conditional_forecast;

histval;
k(0) = 6;
m(0) = 1.01;
P(0) = 2.25;
y(0) = 0.58;
end;
forecast(periods=8, conf_sig=0.95) k gy_obs;
F3 = oo_.forecast;
@#ifdef OCTAVE
set_dynare_seed(12); CF3_draws = fc_draws(M_.exo_nbr, 8, 2*30); set_dynare_seed(12);
@#endif
conditional_forecast(parameter_set=calibration, controlled_varexo=(e_m), replic=30, periods=8);
CF3 = oo_.conditional_forecast;
@#endif
