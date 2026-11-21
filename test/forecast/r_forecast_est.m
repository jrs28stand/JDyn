% Regenerates the est_fc_*.mat references of test/test_forecast.jl (Dynare 7.2, Octave 8.4); run in a scratch
% copy of test/forecast that also holds est_schorfheide_data.m, est_nk_data.m and est_ur_data.m of pydyn's references.
addpath /root/ref/dynare-7.2/matlab
addpath /root/ref/stubs
cases = {'est_fc_sch', 'est_fc_nk', 'est_fc_ur'};
for c = 1:numel(cases)
  clearvars -except c cases
  pause(1.5); rehash;
  eval(sprintf('dynare %s -DOCTAVE nograph nolog', cases{c}));
  vars = who();
  keep = vars(~cellfun(@isempty, regexp(vars, '^(C?F[\dE](_draws)?|XMODE)$')));
  params = M_.params; Sigma_e = M_.Sigma_e; H = M_.H; ys = oo_.dr.ys;
  SmoothedVariables = oo_.SmoothedVariables;
  save('-v7', sprintf('%s.mat', cases{c}), keep{:}, 'params', 'Sigma_e', 'H', 'ys', 'SmoothedVariables');
  disp(sprintf('saved %s.mat', cases{c}));
end
disp('OCT_OK')
