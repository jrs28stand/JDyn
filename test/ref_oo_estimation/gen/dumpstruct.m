function dumpstruct(s, prefix, fid)
  f = fieldnames(s);
  for i = 1:numel(f)
    v = s.(f{i}); p = [prefix '.' f{i}];
    if isstruct(v) && numel(v)==1
      fprintf(fid, '%s struct\n', p); dumpstruct(v, p, fid);
    elseif isstruct(v)
      fprintf(fid, '%s structarray %s\n', p, mat2str(size(v)));
    elseif iscell(v)
      fprintf(fid, '%s cell %s\n', p, mat2str(size(v)));
    elseif ischar(v)
      fprintf(fid, '%s char %s "%s"\n', p, mat2str(size(v)), v);
    elseif isa(v,'function_handle')
      fprintf(fid, '%s fhandle\n', p);
    elseif ~(isnumeric(v)||islogical(v))
      fprintf(fid, '%s object:%s %s\n', p, class(v), mat2str(size(v)));
    else
      sz = mat2str(size(v));
      if numel(v) <= 12 && isreal(v) && ~islogical(v) || (islogical(v) && numel(v)<=12)
        fprintf(fid, '%s %s %s = %s\n', p, class(v), sz, mat2str(double(v), 10));
      else
        fprintf(fid, '%s %s %s%s\n', p, class(v), sz, ifelse_(isreal(v),'',' complex'));
      end
    end
  end
end
function r = ifelse_(c, a, b)
  if c, r = a; else r = b; end
end
