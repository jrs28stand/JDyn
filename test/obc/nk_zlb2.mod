// New Keynesian model with two complementarity constraints (test model of test/test_obc.jl): a lower bound on the
// policy rate r, and a floor rlbar on the lending rate rl = r + sp that households face.
var y pi r rn rl sp d;
varexo ed es em;
parameters beta sigma kappa phipi phiy rhod rhos rss rbar rlbar;
beta = 0.99; sigma = 1; kappa = 0.1; phipi = 1.5; phiy = 0.25; rhod = 0.8; rhos = 0.7; rss = 0.01; rbar = 0; rlbar = 0.004;

model;
y = y(+1) - sigma*(rl - pi(+1) - rss) + d;
pi = beta*pi(+1) + kappa*y;
rn = rss + phipi*pi + phiy*y + em;
d = rhod*d(-1) + ed;
sp = rhos*sp(-1) + es;
[name='zlb', relax='zlb']
r = rn;
[name='zlb', bind='zlb']
r = rbar;
[name='floor', relax='floor']
rl = r + sp;
[name='floor', bind='floor']
rl = rlbar;
end;

occbin_constraints;
name 'zlb'; bind rn < rbar;
name 'floor'; bind r + sp < rlbar; relax r + sp > rlbar;
end;

steady_state_model;
y = 0; pi = 0; r = rss; rn = rss; rl = rss; sp = 0; d = 0;
end;

shocks;
var ed; stderr 0.01;
var es; stderr 0.003;
var em; stderr 0.0025;
end;

steady;
check;

shocks(surprise);
var ed;
periods 1 2 8;
values -0.02 -0.006 0.012;
var es;
periods 1 4;
values -0.012 -0.004;
end;

occbin_setup(simul_periods = 60);
occbin_solver;
