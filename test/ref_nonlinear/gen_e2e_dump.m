r = oo_;
for f = {'forecast','gui'}, if isfield(r,f{1}), r = rmfield(r,f{1}); end, end
save('-mat7-binary', [M_.fname '_oo.mat'], '-struct', 'r');
M = struct('params', M_.params, 'Sigma_e', M_.Sigma_e, 'H', M_.H, 'Correlation_matrix', M_.Correlation_matrix);
save('-mat7-binary', [M_.fname '_M.mat'], '-struct', 'M');
o = struct('order', options_.order, 'first_obs', options_.first_obs, 'nobs', options_.nobs, 'mode_compute', options_.mode_compute, 'qz_criterium', options_.qz_criterium, 'mh_jscale', options_.mh_jscale);
save('-mat7-binary', [M_.fname '_opt.mat'], '-struct', 'o');
