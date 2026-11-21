r = oo_;
for f = {'forecast'}, if isfield(r,f{1}), r = rmfield(r,f{1}); end, end
save('-mat7-binary','oo_ref.mat','-struct','r');
M = struct('params', M_.params, 'Sigma_e', M_.Sigma_e, 'H', M_.H, 'Correlation_matrix', M_.Correlation_matrix);
save('-mat7-binary','M_ref.mat','-struct','M');

O = struct();
for k = {'nobs','first_obs','varobs_id','number_of_observed_variables','mh_jscale','nk','filtered_vars','filter_step_ahead','lik_init','smoother','order','noconstant','parameter_set','kalman_algo','prefilter','loglinear','datafile','mode_compute','mh_replic','mh_drop','mh_nblck'}
  if isfield(options_, k{1}), O.(k{1}) = options_.(k{1}); end
end
if isfield(options_,'varlist'), O.varlist = strjoin(options_.varlist', ';'); end
save('-mat7-binary','opt_ref.mat','-struct','O');
fid = fopen('oo_dump.txt','w');
fprintf(fid,'-- oo_\n'); dumpstruct(oo_, 'oo_', fid);
fprintf(fid,'-- M_ (selected)\n');
for k = {'params','Sigma_e','H','Correlation_matrix','sigma_e_is_diagonal','dname'}
  v = M_.(k{1});
  if ischar(v), fprintf(fid,'M_.%s char %s\n', k{1}, v); else fprintf(fid,'M_.%s %s %s = %s\n', k{1}, class(v), mat2str(size(v)), mat2str(v,12)); end
end
fprintf(fid,'-- options_ (selected)\n');
for k = {'nobs','first_obs','varobs_id','number_of_observed_variables','mh_jscale','nk','filtered_vars','filter_step_ahead','lik_init','smoother','order','mode_compute','mh_replic','varlist','qz_criterium','kalman_algo','nograph','parameter_set','varobs','mh_nblck','mh_drop','filter_covariance','smoothed_state_uncertainty','noconstant','prefilter','loglinear','datafile','sub_draws','mode_file','first_obs'}
  if isfield(options_, k{1})
    v = options_.(k{1});
    if ischar(v), fprintf(fid,'options_.%s char "%s"\n', k{1}, v);
    elseif iscell(v), fprintf(fid,'options_.%s cell %s\n', k{1}, mat2str(size(v)));
    elseif isstruct(v), fprintf(fid,'options_.%s struct\n', k{1});
    else fprintf(fid,'options_.%s %s %s = %s\n', k{1}, class(v), mat2str(size(v)), mat2str(v,12)); end
  else
    fprintf(fid,'options_.%s MISSING\n', k{1});
  end
end
fprintf(fid,'-- bayestopt_ fields\n'); fprintf(fid,'%s ', fieldnames(bayestopt_){:}); fprintf(fid,'\n');
try, fprintf(fid,'-- estim_params_ fields\n'); dumpstruct(estim_params_, 'estim_params_', fid); catch, end
fclose(fid);
