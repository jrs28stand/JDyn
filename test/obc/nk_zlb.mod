// Three-equation New Keynesian model with a lower bound on the policy rate (test model of test/test_obc.jl).
// The contract's worked case "lower bound": reference regime r = rn, alternative r = rbar, bind rn < rbar.
var y pi r rn d;
varexo ed em;
parameters beta sigma kappa phipi phiy rhod rss rbar;
beta = 0.99; sigma = 1; kappa = 0.1; phipi = 1.5; phiy = 0.25; rhod = 0.8; rss = 0.01; rbar = 0;

model;
y = y(+1) - sigma*(r - pi(+1) - rss) + d;
pi = beta*pi(+1) + kappa*y;
rn = rss + phipi*pi + phiy*y + em;
d = rhod*d(-1) + ed;
[name='zlb', relax='zlb']
r = rn;
[name='zlb', bind='zlb']
r = rbar;
end;

occbin_constraints;
name 'zlb'; bind rn < rbar;
end;

steady_state_model;
y = 0; pi = 0; r = rss; rn = rss; d = 0;
end;

shocks;
var ed; stderr 0.01;
var em; stderr 0.0025;
end;

steady;
check;

shocks(surprise);
var ed;
periods 1 2:3 10;
values -0.02 -0.004 0.01;
var em;
periods 5;
values -0.002;
end;

occbin_setup(simul_periods = 60);
occbin_solver;
