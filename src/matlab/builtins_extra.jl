# More MATLAB built-ins for the interpreter, added for Dynare's test suite (see docs/ACCEPTANCE.md): timing,
# random numbers (when the file is run by `dynare()`), cell/struct conversions, set operations.

import Random

# ---------------------------------------------------------------------- timing
"Start of the last `tic` (seconds); `t = tic` returns it."
const TIC = Ref(time())
bi!("tic") do I, a, nout
    t = time()
    nout == 0 && (TIC[] = t; return Any[])
    t       # `id = tic`: the id is the start time; it does not touch the global timer
end
bi!("toc") do I, a, nout
    t0 = isempty(a) ? TIC[] : scalar(a[1], "toc")
    dt = time() - t0
    nout == 0 && return Any[]       # MATLAB prints "Elapsed time is ... seconds"; nothing is printed here
    dt
end
bi!("cputime") do I, a, nout
    time_ns() / 1e9
end

# ---------------------------------------------------------------------- random numbers
"""
Random numbers exist only when the file is run by `dynare()` (the loader of `load(...)` refuses them, as the
numbers of MATLAB's generators cannot be reproduced). `MRNG` is the generator they draw from.
"""
const ALLOW_RANDOM = Ref(false)
const MRNG = Ref{Random.AbstractRNG}(Random.default_rng())

function _rand_dims(a, name)
    isempty(a) && return (1, 1)
    if length(a) == 1
        x = a[1]
        if x isa Real
            n = as_index_int(x, name); return (max(n, 0), max(n, 0))
        end
        v = vec(farr(x)); length(v) == 2 || _unsup("$name with a size vector of $(length(v)) elements")
        return (max(Int(v[1]), 0), max(Int(v[2]), 0))
    end
    length(a) == 2 || _unsup("$name with more than two dimensions")
    (max(as_index_int(a[1], name), 0), max(as_index_int(a[2], name), 0))
end

"rand('seed'|'state'|'twister', x) and friends set the generator: reproducible within JDyn, not equal to MATLAB's."
function _rand_seed!(a)
    if !isempty(a) && a[1] isa AbstractString && lowercase(a[1]) in ("seed", "state", "twister", "default")
        if length(a) >= 2 && is_num(a[2])
            MRNG[] = Random.MersenneTwister(hash(Tuple(vec(farr(a[2])))) % UInt32)
        elseif lowercase(a[1]) == "default"
            MRNG[] = Random.MersenneTwister(0)
        end
        return true
    end
    false
end

function _rand_gen(name, gen)
    BUILTINS[name] = function (I, a, nout)
        ALLOW_RANDOM[] || throw(Unsupported("random numbers are not reproduced by JDyn", "random"))
        _rand_seed!(a) && return Any[]
        any(x -> x isa AbstractString, a) && _unsup("$name with a class argument")
        r, c = _rand_dims(a, name)
        ret(gen(MRNG[], r, c))
    end
end
_rand_gen("rand", (g, r, c) -> rand(g, r, c))
_rand_gen("randn", (g, r, c) -> randn(g, r, c))

bi!("randi") do I, a, nout
    ALLOW_RANDOM[] || throw(Unsupported("random numbers are not reproduced by JDyn", "random"))
    isempty(a) && merr("randi: not enough input arguments")
    lim = vec(farr(a[1]))
    lo, hi = length(lim) == 1 ? (1.0, lim[1]) : (lim[1], lim[2])
    hi >= lo || merr("randi: require lower bound <= upper bound")
    r, c = _rand_dims(a[2:end], "randi")
    ret(Float64.(rand(MRNG[], Int(lo):Int(hi), r, c)))
end
bi!("randperm") do I, a, nout
    ALLOW_RANDOM[] || throw(Unsupported("random numbers are not reproduced by JDyn", "random"))
    n = as_index_int(a[1], "randperm")
    k = length(a) > 1 ? as_index_int(a[2], "randperm") : n
    reshape(Float64.(Random.randperm(MRNG[], n)[1:k]), 1, k) |> ret
end
bi!("normrnd") do I, a, nout
    ALLOW_RANDOM[] || throw(Unsupported("random numbers are not reproduced by JDyn", "random"))
    length(a) >= 2 || merr("normrnd: not enough input arguments")
    mu, sg = a[1], a[2]
    r, c = length(a) > 2 ? _rand_dims(a[3:end], "normrnd") : (max(size(arr(mu), 1), size(arr(sg), 1)), max(size(arr(mu), 2), size(arr(sg), 2)))
    ret(farr(mu) .+ farr(sg) .* randn(MRNG[], r, c))
end
bi!("unifrnd") do I, a, nout
    ALLOW_RANDOM[] || throw(Unsupported("random numbers are not reproduced by JDyn", "random"))
    length(a) >= 2 || merr("unifrnd: not enough input arguments")
    lo, hi = a[1], a[2]
    r, c = length(a) > 2 ? _rand_dims(a[3:end], "unifrnd") : (max(size(arr(lo), 1), size(arr(hi), 1)), max(size(arr(lo), 2), size(arr(hi), 2)))
    ret(farr(lo) .+ (farr(hi) .- farr(lo)) .* rand(MRNG[], r, c))
end
bi!("mvnrnd") do I, a, nout
    ALLOW_RANDOM[] || throw(Unsupported("random numbers are not reproduced by JDyn", "random"))
    length(a) >= 2 || merr("mvnrnd: not enough input arguments")
    mu = vec(farr(a[1])); S = farr(a[2]); n = length(a) > 2 ? as_index_int(a[3], "mvnrnd") : 1
    (size(S) == (length(mu), length(mu))) || merr("mvnrnd: dimensions of mu and sigma do not match")
    L = try
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(S)).L
    catch
        merr("mvnrnd: sigma must be symmetric positive semi-definite")
    end
    ret(reshape(mu, 1, :) .+ randn(MRNG[], n, length(mu)) * L')
end
bi!("gamrnd", "betarnd") do I, a, nout
    ALLOW_RANDOM[] || throw(Unsupported("random numbers are not reproduced by JDyn", "random"))
    _unsup("gamrnd/betarnd")
end

# ---------------------------------------------------------------------- cells and structs
bi!("cell2mat") do I, a, nout
    _nargs(a, 1, 1, "cell2mat")
    c = a[1]
    c isa Cell || merr("cell2mat: C must be a cell array")
    (0 in c.shape) && return zeros(0, 0)
    any(x -> x isa Cell || x isa MStruct || x isa Handle, c.items) && _unsup("cell2mat of cells, structs or function handles")
    kinds = Set(x isa AbstractString ? :char : x isa Bool || x isa AbstractMatrix{Bool} ? :logical : :double for x in c.items)
    length(kinds) == 1 || merr("cell2mat: wrong type elements or mixed cells")
    r, k = c.shape
    rows = Any[mhcat(Any[c.items[i + (j - 1) * r] for j in 1:k]) for i in 1:r]
    mvcat(rows)
end
bi!("struct2cell") do I, a, nout
    _nargs(a, 1, 1, "struct2cell")
    s = a[1]
    s isa MStruct || merr("struct2cell: argument must be of type structure")
    s._kind === nothing || _snapshot_field_names(s)
    Cell(Any[s[k] for k in s._keys], (length(s._keys), 1))
end
bi!("struct2array") do I, a, nout       # Dynare's matlab/missing/struct2array: vertcat of vec() of the fields
    _nargs(a, 1, 1, "struct2array")
    s = a[1]
    s isa MStruct || merr("struct2array:: Argument has to be a structure!")
    s._kind === nothing || _snapshot_field_names(s)
    parts = Any[]
    for k in s._keys
        v = s[k]
        is_num(v) || _unsup("struct2array of non-numeric fields")
        push!(parts, ret(reshape(arr(v), :, 1)))
    end
    isempty(parts) ? zeros(0, 1) : mvcat(parts)
end
bi!("num2cell") do I, a, nout
    _nargs(a, 1, 1, "num2cell")
    x = a[1]
    is_num(x) || _unsup("num2cell of a $(mclass(x))")
    m = arr(x)
    Cell(Any[ret(fill(m[i], 1, 1)) for i in eachindex(m)], size(m))
end
bi!("cellstr") do I, a, nout
    _nargs(a, 1, 1, "cellstr")
    x = a[1]
    x isa Cell && return x
    x isa AbstractString || merr("cellstr: argument STRING must be a 2-D character array")
    Cell(Any[rstrip(x)], (1, 1))
end
bi!("isequaln") do I, a, nout
    length(a) < 2 && merr("isequaln: at least two arguments")
    function same(x, y)
        if x isa AbstractString || y isa AbstractString
            return x isa AbstractString && y isa AbstractString && x == y
        end
        if x isa Cell || y isa Cell
            return x isa Cell && y isa Cell && x.shape == y.shape && all(same(p, q) for (p, q) in zip(x.items, y.items))
        end
        if x isa MStruct || y isa MStruct
            return x isa MStruct && y isa MStruct && Set(keys(x)) == Set(keys(y)) && all(same(x[k], y[k]) for k in keys(x))
        end
        (is_num(x) && is_num(y)) || _unsup("isequaln on this kind of value")
        p, q = arr(x), arr(y)
        size(p) == size(q) && all(isequal.(Float64.(p), Float64.(q)) .| (Float64.(p) .== Float64.(q)))
    end
    all(same(a[1], y) for y in a[2:end])
end

# ---------------------------------------------------------------------- set operations
"Sorted unique elements (numbers, or the strings of a cell), with the index of the last... MATLAB: first occurrence."
function _set_items(x, name)
    if x isa Cell
        all(c -> c isa AbstractString, x.items) || _unsup("$name of a cell that is not a cell array of strings")
        return (String[c for c in x.items], true, x.shape)
    elseif x isa AbstractString
        return (String[x], true, (1, 1))
    end
    is_num(x) || _unsup("$name of a $(mclass(x))")
    m = farr(x)
    (Float64[m...], false, size(m))
end
_setout(vals, iscell, rowlike) = iscell ? Cell(Any[vals...], rowlike ? (1, length(vals)) : (length(vals), 1)) :
                                 ret(rowlike ? reshape(Float64.(vals), 1, :) : reshape(Float64.(vals), :, 1))
function _setop(name, a, nout)
    length(a) < 2 && merr("$name: not enough input arguments")
    length(a) > 2 && _unsup("$name with options ('rows', 'stable', ...)")
    xa, ca, sa = _set_items(a[1], name); xb, cb, sb = _set_items(a[2], name)
    ca == cb || (isempty(xa) || isempty(xb)) || merr("$name: cell array of strings cannot be combined with a double")
    iscell = ca || cb
    # orientation of the result: a row if both inputs are rows, a column if both are columns (an empty or scalar
    # input follows the other one). MATLAB and Octave disagree when a row meets a column, and on empty results of
    # empty inputs: refused.
    isrow(s) = s[1] == 1 && s[2] >= 1
    iscol(s) = s[2] == 1 && s[1] > 1
    isneutral(s) = s == (1, 1) || 0 in s
    if name != "setdiff"
        (isempty(xa) && isempty(xb)) && _unsup("$name of two empty arrays (MATLAB and Octave return different sizes)")
        ((isrow(sa) && iscol(sb)) || (iscol(sa) && isrow(sb))) && _unsup("$name of a row and a column (MATLAB and Octave return different orientations)")
    end
    rowlike = name == "setdiff" ? !iscol(sa) : !((iscol(sa) || isneutral(sa)) && (iscol(sb) || isneutral(sb)) && (iscol(sa) || iscol(sb)))
    ua = sort(unique(xa)); ub = sort(unique(xb))
    if name == "union"
        r = sort(unique(vcat(xa, xb)))
        nout > 1 && _unsup("union with several outputs")
        return _setout(r, iscell, rowlike)
    elseif name == "intersect"
        r = [x for x in ua if x in Set(ub)]
        outs = Any[_setout(r, iscell, rowlike)]
        if nout > 1
            ia = [findfirst(==(x), xa) for x in r]; ib = [findfirst(==(x), xb) for x in r]
            push!(outs, _setout(Float64.(ia), false, false), _setout(Float64.(ib), false, false))
        end
        return outs
    else  # setdiff
        r = [x for x in ua if !(x in Set(xb))]
        outs = Any[_setout(r, iscell, rowlike)]
        if nout > 1
            push!(outs, _setout(Float64[findfirst(==(x), xa) for x in r], false, false))
        end
        return outs
    end
end
bi!("union") do I, a, nout; _setop("union", a, nout); end
bi!("intersect") do I, a, nout; _setop("intersect", a, nout); end
bi!("setdiff") do I, a, nout; _setop("setdiff", a, nout); end

# ---------------------------------------------------------------------- Dynare helper functions
# varlist_indices(sublist, list [, nocheck_dummy]) -> [i_var, nvar, index_unique_present]
# (Dynare's matlab/varlist_indices.m): positions of the names of `sublist` in the cell `list`; a name that is not
# in the list is an error unless nocheck_dummy; a name given twice is kept once (with a warning).
bi!("varlist_indices") do I, a, nout
    _nargs(a, 2, 3, "varlist_indices")
    sub, list = a[1], a[2]
    nocheck = length(a) > 2 && truth(a[3])
    list isa Cell || merr("varlist_indices: LIST must be a cell array of names")
    names = Any[x for x in list.items]
    if _isemptyarr(sub) || (sub isa Cell && isempty(sub.items))
        return Any[zeros(0, 0), 0.0, zeros(0, 1)]
    end
    subs = sub isa Cell ? sub.items : (sub isa AbstractString ? Any[sub] : merr("varlist_indices: SUBLIST must be a cell array or a string"))
    shp = sub isa Cell ? sub.shape : (1, 1)
    i_var = Int[let k = findfirst(==(s), names); k === nothing ? 0 : k end for s in subs]
    check = i_var .!= 0
    if !all(check) && !nocheck
        merr("The following symbols are not endogenous variables:" * join(" " .* [string(subs[k]) for k in findall(.!check)]))
    end
    not_present = findall(==(0), i_var)
    nvar_present = count(check)
    seen = Dict{Int,Int}()
    for (k, v) in enumerate(i_var)
        haskey(seen, v) || (seen[v] = k)
    end
    first_pos = sort([k for (v, k) in seen if !(k in not_present)])
    uniq = i_var[first_pos]
    if length(uniq) != nvar_present
        dup = [k for k in eachindex(i_var) if !(k in first_pos) && i_var[k] != 0]
        @warn "The following symbols are specified twice in the variable list and are considered only once:" * join(" " .* [string(subs[k]) for k in dup])
    end
    out = shp[1] == 1 ? reshape(Float64.(uniq), 1, :) : reshape(Float64.(uniq), :, 1)
    Any[ret(out), Float64(length(uniq)), ret(reshape(Float64.(first_pos), :, 1))]
end

# ---------------------------------------------------------------------- files (only when the file is run by dynare())
_need_exec(what) = ALLOW_RANDOM[] || throw(Unsupported("$what: file-system access is only available when the file is run by dynare()", "file"))
_mat_value(v::Bool) = v
_mat_value(v::Real) = Float64(v)
_mat_value(v::AbstractString) = String(v)
_mat_value(v::AbstractMatrix{<:Real}) = v isa AbstractMatrix{Bool} ? ret(Matrix{Bool}(v)) : ret(Matrix{Float64}(v))
_mat_value(v::AbstractVector{<:Real}) = ret(reshape(Float64.(v), :, 1))
function _mat_value(v::AbstractDict)
    s = MStruct()
    for k in sort(collect(keys(v)))
        s[String(k)] = _mat_value(v[k])
    end
    s
end
_mat_value(v::AbstractArray) = Cell(Any[_mat_value(x) for x in vec(v)], ndims(v) == 1 ? (length(v), 1) : (size(v, 1), size(v, 2)))
_mat_value(v) = throw(Unsupported("load: a value of class $(typeof(v)) in the MAT-file", "file"))

"`s = load(file)`: the variables of a MAT-file (or an ASCII matrix file) as a struct."
function _load_to_struct(I, a)
    name = String(a[1])
    cands = isabspath(name) ? [name, name * ".mat"] : [joinpath(I.dir === nothing ? pwd() : I.dir, name), joinpath(I.dir === nothing ? pwd() : I.dir, name * ".mat"), name, name * ".mat"]
    path = findfirst(isfile, cands)
    path === nothing && throw(Unsupported("load: unable to find file $name (it may be written by another run)", "file"))
    p = cands[path]
    data = try
        MAT.matread(p)
    catch
        throw(Unsupported("load: $(basename(p)) is not a MAT-file that can be read", "file"))
    end
    names = String[x for x in a[2:end]]
    s = MStruct()
    for k in sort(collect(keys(data)))
        (isempty(names) || k in names) && (s[String(k)] = _mat_value(data[k]))
    end
    s
end

bi!("fullfile") do I, a, nout
    all(x -> x isa AbstractString, a) || merr("fullfile: all arguments must be strings")
    parts = [x for x in a if !isempty(x)]
    isempty(parts) && return ""
    r = parts[1]
    for x in parts[2:end]
        r = (endswith(r, "/") ? r : r * "/") * lstrip(x, '/')
    end
    String(r)
end
bi!("fileparts") do I, a, nout
    _nargs(a, 1, 1, "fileparts")
    a[1] isa AbstractString || merr("fileparts: FILENAME must be a single string")
    f = String(a[1])
    i = findlast('/', f); dir = i === nothing ? "" : (i == 1 ? "/" : f[1:i-1]); rest = i === nothing ? f : f[i+1:end]
    j = findlast('.', rest)
    (j === nothing) ? Any[dir, rest, ""] : Any[dir, rest[1:j-1], rest[j:end]]
end
bi!("pwd") do I, a, nout
    _need_exec("pwd"); pwd()
end
bi!("isfile") do I, a, nout
    _need_exec("isfile"); a[1] isa AbstractString || merr("isfile: argument must be a string"); isfile(a[1])
end
bi!("isfolder", "isdir") do I, a, nout
    _need_exec("isfolder"); a[1] isa AbstractString || merr("isfolder: argument must be a string"); isdir(a[1])
end
bi!("getenv") do I, a, nout
    _need_exec("getenv"); a[1] isa AbstractString || merr("getenv: argument must be a string"); String(get(ENV, a[1], ""))
end
bi!("mkdir") do I, a, nout
    _need_exec("mkdir"); mkpath(String(a[end])); nout > 0 ? Any[true, "", ""] : Any[]
end

# ---------------------------------------------------------------------- save
_to_mat(v::Union{Bool,Real,AbstractString}) = v isa Real && !(v isa Bool) ? Float64(v) : v
_to_mat(v::AbstractMatrix{<:Real}) = v isa AbstractMatrix{Bool} ? Matrix{Bool}(v) : Matrix{Float64}(v)
function _to_mat(v::MStruct)
    d = Dict{String,Any}()
    for k in v._keys
        x = v[k]
        (x isa Handle || x isa HostStruct || x isa Tainted) && continue
        try
            d[k] = _to_mat(x)
        catch e
            e isa Unsupported || rethrow()          # a field of a class MAT-files cannot hold here is left out
        end
    end
    d
end
function _to_mat(v::Cell)
    a = Array{Any}(undef, v.shape...)
    for (i, x) in enumerate(v.items)
        a[i] = _to_mat(x)
    end
    a
end
_to_mat(v) = throw(Unsupported("save of a $(mclass(v)) value", "file"))

"Write the named variables of a scope into a MAT-file (MATLAB's `save file var1 var2`; no names: all variables)."
function _save_mat(path::String, vars::Dict{String,Any}, names; append=false)
    d = Dict{String,Any}()
    if append && isfile(path)
        merge!(d, MAT.matread(path))
    end
    for n in names
        haskey(vars, n) || merr("save: no such variable '$n'")
        d[n] = _to_mat(vars[n])
    end
    MAT.matwrite(path, d)
end

bi!("save") do I, a, nout
    _need_exec("save")
    isempty(a) && merr("save: not enough input arguments")
    all(x -> x isa AbstractString, a) || _unsup("save with computed variable names or a struct argument")
    opts = [x for x in a[2:end] if startswith(x, "-") || x in ("mat", "ascii")]
    names = [x for x in a[2:end] if !(x in opts)]
    any(o -> o in ("-ascii", "ascii", "-struct", "-text", "-binary"), opts) && _unsup("save with option $(join(opts, ' '))")
    f = String(a[1]); occursin('.', basename(f)) || (f *= ".mat")
    path = isabspath(f) ? f : joinpath(I.dir === nothing ? pwd() : I.dir, f)
    vars = I.scope_stack[end].vars
    isempty(names) && (names = sort(collect(keys(vars))))
    _save_mat(path, vars, names; append="-append" in opts)
    Any[]
end

"Dynare writes <fname>/Output/<fname>_results.mat (M_, oo_, options_) at the end of a run; other tests load it."
function save_results_mat(dir::AbstractString, fname::AbstractString, M_, oo_, options_)
    out = joinpath(dir, fname, "Output"); mkpath(out)
    d = Dict{String,Any}()
    for (n, v) in (("M_", M_), ("oo_", oo_), ("options_", options_))
        try
            d[n] = _to_mat(v)
        catch
            @debug "results.mat: $n not saved"
        end
    end
    MAT.matwrite(joinpath(out, fname * "_results.mat"), d)
end
