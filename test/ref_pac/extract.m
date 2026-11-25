% saved after `dynare <name>` by gen_refs.sh: what the semi-structural functions computed
S = struct();
S.params = M_.params;
for t = {'var', 'trend_component'}
  if isfield(oo_, t{1}) && isstruct(oo_.(t{1}))
    f = fieldnames(oo_.(t{1}));
    for i = 1:numel(f)
      if isstruct(oo_.(t{1}).(f{i})) && isfield(oo_.(t{1}).(f{i}), 'CompanionMatrix')
        S.(['C_' t{1} '_' f{i}]) = oo_.(t{1}).(f{i}).CompanionMatrix;
      end
    end
  end
end
if isfield(oo_, 'dr') && isfield(oo_.dr, 'eigval')
  S.eigabs = sort(abs(oo_.dr.eigval(isfinite(oo_.dr.eigval))));
end
save('-v6', 'ref.mat', '-struct', 'S');
