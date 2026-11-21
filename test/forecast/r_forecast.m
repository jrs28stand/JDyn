% Regenerates the fc_*.mat references of test/test_forecast.jl with Dynare 7.2 under Octave 8.4:
%   cd <scratch copy of test/forecast>; GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_forecast.m
% (with /root/ref/stubs on the path for the figure functions). Each case stores what the .mod file put in its
% workspace variables (F*, CF* = oo_.forecast / oo_.conditional_forecast after each statement) and the
% random numbers the stochastic statements used (*_draws, recorded under -DOCTAVE with the same seed).
addpath /root/ref/dynare-7.2/matlab
addpath /root/ref/stubs
for v = 0:3
  clearvars -except v
  pause(1.5); rehash;
  eval(sprintf('dynare fc_cal -DVARIANT=%d -DOCTAVE nograph nolog', v));
  vars = who();
  keep = vars(~cellfun(@isempty, regexp(vars, '^(C?F\d(_draws)?)$')));
  ys = oo_.dr.ys; params = M_.params; ghx = oo_.dr.ghx; ghu = oo_.dr.ghu; order_var = oo_.dr.order_var;
  save('-v7', sprintf('fc_cal_v%d.mat', v), keep{:}, 'ys', 'params', 'ghx', 'ghu', 'order_var');
  disp(sprintf('saved fc_cal_v%d.mat', v));
end
disp('OCT_OK')
