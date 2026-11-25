# probes of the other statements on the small model (steady, check, forecast, perfect_foresight_*, osr/ramsey on stoch_simul options).
#     julia --project=. tools/options_probe_other.jl
include(joinpath(@__DIR__, "options_probe.jl"))
V(base, extra) = [replace(base, ");" => ",$e);") for e in extra]
probe([
 ("steady;", ["steady(solve_algo=2);", "steady(solve_algo=3);", "steady(maxit=3);", "steady(tolf=1e-3);", "steady(tolx=1e-3);", "steady(nocheck);", "steady(homotopy_mode=1);", "steady(markowitz=0.5);"]),
 ("check;", ["check(qz_zero_threshold=1e-3);", "check(solve_algo=3);", "check(qz_zero_threshold=1e3);"]),
 ("stoch_simul(order=1,irf=0,periods=0,nograph); forecast(periods=8);", ["stoch_simul(order=1,irf=0,periods=0,nograph); forecast(periods=8,conf_sig=0.5);",
     "stoch_simul(order=1,irf=0,periods=0,nograph); forecast(periods=4);", "stoch_simul(order=1,irf=0,periods=0,nograph); forecast(periods=8,nograph);"]),
 ("perfect_foresight_setup(periods=20); perfect_foresight_solver;", ["perfect_foresight_setup(periods=30); perfect_foresight_solver;",
     "perfect_foresight_setup(periods=20); perfect_foresight_solver(maxit=2);", "perfect_foresight_setup(periods=20); perfect_foresight_solver(tolf=1e-3);",
     "perfect_foresight_setup(periods=20); perfect_foresight_solver(stack_solve_algo=1);", "perfect_foresight_setup(periods=20); perfect_foresight_solver(lmmcp);",
     "perfect_foresight_setup(periods=20); perfect_foresight_solver(no_homotopy);", "perfect_foresight_setup(periods=20); perfect_foresight_solver(noprint);",
     "perfect_foresight_setup(periods=20); perfect_foresight_solver(robust_lin_solve);", "perfect_foresight_setup(periods=20); perfect_foresight_solver(linear_approximation);"]),
 ("osr_params alpha; optim_weights; y 1; end; osr(irf=0,nograph) y;", ["osr_params alpha; optim_weights; y 1; end; osr(irf=0,nograph,order=1,periods=20) y;", "osr_params alpha; optim_weights; y 1; end; osr(irf=0,nograph,opt_algo=9) y;",
     "osr_params alpha; optim_weights; y 1; end; osr(irf=0,nograph,hp_filter=1600) y;"]),
])
