% Octave reference for test_dseries.jl: every snippet of dseries_cases.txt (separated by lines holding only %%)
% is run with Dynare 7.2's dseries toolbox and its variable r saved to dseries_ref.mat; a failing snippet is
% stored as 'ERROR: <message>'. Run in this directory:  octave --no-gui --quiet r_dseries.m
% (needs DYNARE_MATLAB, the matlab/ folder of Dynare 7.2, default /root/ref/dynare-7.2/matlab)
1;
function r = run_case__(code__)
  r = [];
  eval(code__);
  if islogical(r), r = double(r); end
end
warning('off', 'all');
root = getenv('DYNARE_MATLAB'); if isempty(root), root = '/root/ref/dynare-7.2/matlab'; end
addpath(root); dynare_config;
warning('off', 'all');
txt = fileread('dseries_cases.txt');
cases = strsplit(txt, sprintf('\n%%%%\n'));
results = cell(numel(cases), 1);
for k = 1:numel(cases)
  try
    results{k} = run_case__(cases{k});
  catch err
    results{k} = ['ERROR: ' err.message];
  end
end
save('-mat7-binary', 'dseries_ref.mat', 'results');
printf('%d cases\n', numel(cases));
