var y pi r rn a m s sn;
varexo e em;
parameters beta kappa phi rho rbar rss sige sigm km sbar;
beta=0.99; kappa=0.1; phi=1.5; rho=0.7; rbar=0.002; rss=0.01; sige=0.003; sigm=0.05; km=1.25; sbar=-0.006;
model;
y = y(+1) - (r - rss - pi(+1)) + 0.3*s + a;
pi = beta*pi(+1) + kappa*y;
rn = rss + phi*pi + 0.2*y;
[name='zlb', relax='zlb']
r = rn*(1 + km*m);
[name='zlb', bind='zlb']
r = rbar*(1 + km*m);
sn = 0.5*y;
[name='floor', relax='floor']
s = sn;
[name='floor', bind='floor']
s = sbar;
a = rho*a(-1) + sige*e;
m = sigm*em;
end;
occbin_constraints;
name 'zlb'; bind rn < rbar; relax rn >= rbar;
name 'floor'; bind sn < sbar;
end;
shocks; var e = 1; var em = 1; end;
steady_state_model;
    y = 0; pi = 0; rn = rss; r = rss; a = 0; m = 0; s = 0; sn = 0;
end;
steady; check;

varobs pi r;
estimated_params;
phi,,,,NORMAL_PDF,1.5,0.25;
rho,,,,BETA_PDF,0.7,0.1;
sige,,,,INV_GAMMA_PDF,0.003,0.003;
sigm,,,,INV_GAMMA_PDF,0.05,0.025;
end;
occbin_setup(likelihood_inversion_filter);
estimation(datafile=dummy, mode_compute=4, nobs=150, first_obs=1, mh_replic=0, plot_priors=0, nograph);
