% Regenerates the opt_*.mat references of test/test_optimizers.jl (Dynare 7.2, Octave 8.4); run in a scratch copy of
% test/optim holding est_schorfheide_data.m and est_nk_data.m of pydyn's references.
addpath /root/ref/dynare-7.2/matlab
addpath /root/ref/stubs
cases = {'est_opt_sch', 5; 'est_opt_sch', 50; 'est_opt_nk', 5; 'est_opt_sch', 8; 'est_opt_nk', 8; 'est_opt_nk', 9; 'est_opt_sch', 9; 'est_opt_sch', 7};
for c = 1:rows(cases)
  clearvars -except c cases
  pause(1.5); rehash;
  try
    eval(sprintf('dynare %s -DMC=%d nograph nolog', cases{c,1}, cases{c,2}));
    opt = oo_.posterior.optimization;
    names = strjoin(bayestopt_.name', ';');
    hh = []; try, load(sprintf('%s/Output/%s_mode.mat', M_.dname, M_.fname), 'hh'); catch, end
    save('-v7', sprintf('opt_%s_%d.mat', cases{c,1}(9:end), cases{c,2}), 'opt', 'names', 'hh');
    disp(sprintf('saved opt_%s_%d.mat', cases{c,1}(9:end), cases{c,2}));
  catch err
    disp(sprintf('FAILED %s %d: %s', cases{c,1}, cases{c,2}, err.message));
  end
end
disp('OCT_OK')
