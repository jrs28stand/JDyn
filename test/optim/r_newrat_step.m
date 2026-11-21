% Regenerates newrat_step.mat for test/test_optimizers.jl: the first steps of Dynare 7.2's newrat (mode_compute=5) on
% est_opt_sch.mod (mr_hessian's gradient and step sizes, the first csminit1 iterate), from the estimated parameters' initial
% values. Run in a scratch copy of test/optim holding est_schorfheide_data.m of pydyn's references.
addpath /root/ref/dynare-7.2/matlab
addpath /root/ref/stubs
addpath /root/ref/dynare-7.2/matlab/optimization
dynare est_opt_sch -DMC=0 nograph nolog
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
      dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
args = {dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state};
[f0, ef, dlik] = penalty_objective_function(xparam1, @dsge_likelihood, 1e8, args{:});
hess_info.gstep = options_.gstep; hess_info.htol = 1e-5; hess_info.htol_rel = NaN; hess_info.h1 = 1e-6*ones(numel(xparam1),1); hess_info.robust = false;
[dum, gg, htol0, igg, hhg, h1, hess_info2, fc] = mr_hessian(xparam1, @dsge_likelihood, f0, 0, 1e-5, hess_info, [bounds.lb bounds.ub], bayestopt_.p2, 0, args{:});
[fval1, x1, fc1] = csminit1(@dsge_likelihood, xparam1, f0, f0, gg, 0, igg, 0, args{:});
save('-v7', 'newrat_step.mat', 'xparam1', 'f0', 'dlik', 'gg', 'h1', 'igg', 'fc', 'fval1', 'x1', 'fc1');
disp('OCT_OK')
