# N-dimensional arrays for the interpreter (port of the N-d layer of pydyn/matlab.py; semantics are Octave 8.4's).
#
# A value with more than two dimensions is an `Array{Float64,N}` or `Array{Bool,N}` with N >= 3 and no trailing singleton
# dimension (MATLAB's dimensions: `zeros(2,3,1)` is 2x3, a Matrix). Two-dimensional values keep their representation and
# their code paths; this file adds the paths that are taken when an operand has more than two dimensions or an
# indexing expression has more than two subscripts, and `nd_hook`, which `mcall` consults before running a builtin.
# Where Octave's result is not reproduced the code raises `Unsupported` (as pydyn does), never a different value.

is_nd(v) = (v isa Array{Float64} || v isa Array{Bool}) && ndims(v) > 2

"MATLAB's dimensions for a Julia size: at least two, no trailing singleton beyond the second."
function canon_dims(sz)
    d = collect(Int, sz)
    while length(d) < 2
        push!(d, 1)
    end
    while length(d) > 2 && d[end] == 1
        pop!(d)
    end
    Tuple(d)
end
_dimstr(sz) = join(sz, "x")
_padn(sz, n) = Tuple(vcat(collect(Int, sz), ones(Int, max(0, n - length(sz)))))

"Normalise an array result: trailing singletons beyond the second dimension removed; 2-D results go through `ret`."
function ndret(a::AbstractArray)
    T = eltype(a) == Bool ? Bool : Float64
    a isa Array{T} || (a = Array{T}(a))
    sz = canon_dims(size(a))
    size(a) == sz || (a = reshape(a, sz))
    ndims(a) > 2 ? a : ret(a)
end

"Numeric value as a Julia array of any number of dimensions (at least two)."
function ndarr(v)
    is_nd(v) && return v
    arr(v)
end
function ndfarr(v)
    a = ndarr(v)
    eltype(a) == Float64 ? a : Array{Float64}(a)
end

nd_shape(v) = is_nd(v) ? size(v) : shape_of(v)

function _check_elements(sz)
    prod(Int128.(collect(sz))) > MAX_ELEMENTS && _unsup("array with more than $MAX_ELEMENTS elements")
    nothing
end

# ---------------------------------------------------------------------- subscripts
function _nd_pos(ix, n::Int; grow::Bool=false)
    ix === COLON && return collect(1:n)
    ix isa AbstractString && _unsup("char array used as an index")
    is_nd(ix) && (ix = reshape(ix, :, 1))
    _lin_index(ix, n; grow=grow)
end

"(1-based positions, extent, is_scalar) of one subscript along a dimension of length `n` (positions may lie beyond `n`)."
function _idx_info(ix, n::Int)
    ix === COLON && return (collect(1:n), n, false)
    k = _nd_pos(ix, n; grow=true)
    ext = isempty(k) ? n : max(n, maximum(k))
    isscal = !(ix isa Bool || ix isa AbstractArray{Bool}) && length(ix) == 1
    (k, ext, isscal)
end

_is_vec(sh) = length(sh) == 2 && ((sh[1] != 1) != (sh[2] != 1))
_is_nd_vec(sh) = count(!=(1), sh) == 1

"Octave's dim_vector::redim: the dimensions seen by `n` subscripts (the last spans the remaining dimensions)."
function redim(sh, n::Int)
    d = collect(Int, sh)
    L = length(d)
    L == n && return Tuple(d)
    L < n && return Tuple(vcat(d, ones(Int, n - L)))
    n <= 1 && return n == 1 ? (prod(d), 1) : Tuple(d)
    Tuple(vcat(d[1:n-1], prod(d[n:end])))
end

function _index_dims(ia::AbstractArray, n::Int)
    eltype(ia) == Bool || return size(ia)
    size(ia) == (1, 1) && return (n, n)
    ndims(ia) > 2 && _is_nd_vec(size(ia)) && _unsup("a logical index that is an N-d vector (1x1xn): MATLAB's result shape is not documented")
    _is_vec(size(ia)) && return size(ia, 1) == 1 ? (1, n) : (n, 1)
    (n, 1)
end

function _lin_result_shape(src, ia, n::Int)
    rd = _index_dims(ia, n)
    if (length(src) > 2 || length(rd) > 2) && n != 1 && _is_nd_vec(src) && _is_nd_vec(rd)
        _unsup("linear indexing of a vector with a vector when either is an N-d vector (1x1xn)")
    end
    if src[1] * src[2] != 1 && _is_vec(src) && n != 1 && _is_vec(rd)
        return src[2] == 1 ? (n, 1) : (1, n)
    end
    rd
end

"`v(args)` where `v` or an index has more than two dimensions, or there are more than two subscripts."
function nd_index(v, args)
    a = ndarr(v)
    n = length(args)
    n == 0 && return ndret(a)
    if n == 1
        ix = args[1]
        if ix isa Real && !(ix isa Bool)
            (ix != round(ix) || ix < 1) && merr("subscript indices must be positive integers or logicals")
            k = Int(ix)
            k > length(a) && merr("index ($k) out of bounds (array has $(length(a)) element(s))")
            return ndret(reshape([a[k]], 1, 1))
        end
        flat = vec(a)
        k = _nd_pos(ix, length(flat))
        out = flat[k]
        ix === COLON && return ndret(reshape(out, :, 1))
        return ndret(reshape(out, _lin_result_shape(size(a), ndarr(ix), length(k))))
    end
    dims = redim(size(a), n)
    b = reshape(a, dims)
    idx = Any[_nd_pos(x, dims[k]) for (k, x) in enumerate(args)]
    ndret(b[idx...])
end

# ---------------------------------------------------------------------- assignment and deletion
_chop(sz) = (d = Int[x for x in sz if x != 1]; while length(d) < 2; push!(d, 1); end; d)

function _to_class_of(a, r, defined::Bool)
    defined || return zeros(eltype(r), 0, 0), r
    if eltype(a) == Bool && eltype(r) != Bool
        any(isnan, r) && merr("invalid conversion from NaN to logical")
        return a, r .!= 0
    elseif eltype(a) != Bool && eltype(r) == Bool
        return a, Array{Float64}(r)
    end
    a, r
end

function _delete_nd(a::Array, args)
    n = length(args)
    T = eltype(a)
    if n == 1
        ix = args[1]
        len = length(a)
        ix === COLON && return zeros(T, 0, 0)
        k, ext, is_scalar = _idx_info(ix, len)
        isempty(k) && return ndret(a)
        ext != len && merr("A(I) = []: index out of bounds: value $ext out of bound $len")
        keep = trues(len)
        keep[k] .= false
        out = vec(a)[keep]
        if ndims(a) == 2 && (size(a, 1) == 1 || size(a, 2) == 1)
            return ret((size(a, 2) == 1 && size(a, 1) != 1) ? reshape(out, :, 1) : reshape(out, 1, :))
        end
        is_scalar && return ret(reshape(out, 1, :))
        _unsup("deleting several elements of a matrix with a linear index")
    end
    n != ndims(a) && _unsup("null assignment with fewer or more subscripts than the array has dimensions (MATLAB and Octave differ)")
    cols = Bool[x === COLON for x in args]
    if all(cols)
        return ndret(zeros(T, 0, size(a)[2:end]...))
    end
    if count(!, cols) == 1
        d = findfirst(!, cols)
        k, ext, _ = _idx_info(args[d], size(a, d))
        isempty(k) && return ndret(a)
        ext != size(a, d) && merr("A(..,I,..) = []: index out of bounds: value $ext out of bound $(size(a, d))")
        keep = trues(size(a, d))
        keep[k] .= false
        return ndret(copy(selectdim(a, d, findall(keep))))
    end
    info = [_idx_info(x, size(a, d)) for (d, x) in enumerate(args)]
    any(t -> isempty(t[1]), info) && return ndret(a)
    for (d, x) in enumerate(args)
        x !== COLON && sort(unique(info[d][1])) == collect(1:size(a, d)) &&
            _unsup("null assignment with a subscript that covers a whole dimension (MATLAB and Octave differ)")
    end
    merr("a null assignment can only have one non-colon index")
end

function _assign_nd_core(a::Array, args, r::Array)
    n = length(args)
    if n == 1
        k, nx, _ = _idx_info(args[1], length(a))
        nx != length(a) && merr("Octave:index-out-of-bounds: A(I) = X: X must have the same size as I; resizing an N-d array with a linear index is ambiguous")
        (length(r) != 1 && length(k) != length(r)) && merr("=: nonconformant arguments (op1 is 1x$(length(k)), op2 is $(_dimstr(size(r))))")
        b = copy(a)
        length(r) == 1 ? (b[k] .= r[1]) : (b[k] .= vec(r))
        return ndret(b)
    end
    dv = redim(size(a), n)
    local pos, rdv
    if all(==(0), size(a))                       # Octave's zero_dims_inquire
        cols = Bool[x === COLON for x in args]
        scal = Bool[!cols[i] && _idx_info(args[i], 0)[3] for i in 1:n]
        rdv = Int[cols[i] ? 0 : _idx_info(args[i], 0)[2] for i in 1:n]
        rhdv = collect(Int, size(r))
        nonsc = count(!, scal)
        if all(cols)
            rdv = vcat(rhdv, ones(Int, n))[1:n]
        elseif nonsc == length(rhdv)
            j = 1
            for i in 1:n
                scal[i] && continue
                cols[i] && (rdv[i] = rhdv[j])
                j += 1
            end
        else
            rh0 = _chop(size(r))
            j = 1
            for i in 1:n
                scal[i] && continue
                if cols[i]
                    rdv[i] = j <= length(rh0) ? rh0[j] : 1
                    j += 1
                end
            end
        end
        pos = Any[cols[i] ? collect(1:rdv[i]) : _idx_info(args[i], 0)[1] for i in 1:n]
    else
        info = [_idx_info(x, dv[i]) for (i, x) in enumerate(args)]
        pos = Any[t[1] for t in info]
        rdv = Int[t[2] for t in info]
    end
    lens = Int[length(p) for p in pos]
    rh = _chop(size(r))
    isfill = length(r) == 1
    j = 1
    match = true
    for ln in lens
        ln == 1 && continue
        match = match && j <= length(rh) && ln == rh[j]
        j += 1
    end
    match = (match && (j > length(rh) || rh[j] == 1)) || isfill
    if !match
        (0 in lens && isempty(r)) && return ndret(a)
        merr("=: nonconformant arguments (op1 is $(_dimstr(canon_dims(lens))), op2 is $(_dimstr(size(r))))")
    end
    rdv_t = Tuple(rdv)
    if rdv_t != dv
        n < ndims(a) && merr("Octave:index-out-of-bounds: resizing an N-d array through fewer subscripts than it has dimensions is ambiguous (A(I,J,...) = X: dimensions mismatch)")
        _check_elements(rdv_t)
        b = zeros(eltype(a), rdv_t)
        b[ntuple(i -> 1:dv[i], n)...] = reshape(a, dv)
        shape = canon_dims(rdv_t)
    else
        b = copy(reshape(a, dv))
        shape = size(a)
    end
    if all(!=(0), lens)
        isfill ? (b[pos...] .= r[1]) : (b[pos...] = reshape(r, Tuple(lens)))
    end
    ndret(reshape(b, shape))
end

"`v(args) = rhs` with more than two subscripts or an N-d operand; `v === nothing` is an undefined variable."
function nd_assign(v, args, rhs)
    a = v === nothing ? zeros(0, 0) : ndarr(v)
    r = ndarr(rhs)
    if rhs isa AbstractMatrix && size(r) == (0, 0)           # A(I) = []: deletion
        v === nothing && merr("A(I) = []: A is undefined")
        return _delete_nd(a, args)
    end
    a, r = _to_class_of(a, r, v !== nothing)
    isempty(args) && _unsup("assignment with an empty subscript list A() = X")
    _assign_nd_core(a, args, r)
end

# ---------------------------------------------------------------------- element-wise operations
function _nd_bcast(x::AbstractArray, y::AbstractArray)
    if size(x) != size(y)
        n = max(ndims(x), ndims(y))
        sx, sy = _padn(size(x), n), _padn(size(y), n)
        for (p, q) in zip(sx, sy)
            (p != q && p != 1 && q != 1) &&
                merr("nonconformant arguments ($(_dimstr(canon_dims(size(x)))) and $(_dimstr(canon_dims(size(y)))))")
        end
    end
    nothing
end

function nd_ew(fn, a, b)
    x, y = ndfarr(a), ndfarr(b)
    _nd_bcast(x, y)
    ndret(fn.(x, y))
end
function nd_cmp(fn, a, b)
    (a isa AbstractString || b isa AbstractString) && _unsup("comparison of char arrays with an N-d array")
    x, y = ndarr(a), ndarr(b)
    _nd_bcast(x, y)
    ndret(fn.(x, y))
end
function _nd_logical(a)
    x = ndarr(a)
    (eltype(x) != Bool && any(isnan, x)) && merr("NaN cannot be converted to a logical value")
    x .!= 0
end
function nd_and_or(fn, a, b)
    x, y = _nd_logical(a), _nd_logical(b)
    _nd_bcast(x, y)
    ndret(fn.(x, y))
end
nd_uminus(a) = ndret(-ndfarr(a))
function nd_lnot(a)
    x = ndarr(a)
    (eltype(x) != Bool && any(isnan, x)) && merr("NaN cannot be converted to a logical value")
    ndret(x .== 0)
end
function _nd_hvcat(vals, d)
    any(v -> v isa AbstractString, vals) && _unsup("concatenation of char arrays with N-d arrays")
    nz = Any[v for v in vals if nd_shape(v) != (0, 0)]
    nd_concat(isempty(nz) ? vals[1:1] : nz, d == 1 ? 1 : 2, true)
end
nd_transpose(a) = merr("transpose not defined for N-D objects")
function nd_mtimes(a, b)
    (a isa Real || b isa Real || length(ndarr(a)) == 1 || length(ndarr(b)) == 1) && return nd_ew(*, a, b)
    sa, sb = nd_shape(a), nd_shape(b)
    ra, ca = sa[1], prod(sa[2:end])
    rb, cb = sb[1], prod(sb[2:end])
    ca != rb && merr("operator *: nonconformant arguments (op1 is $(ra)x$(ca), op2 is $(rb)x$(cb))")
    _unsup("matrix product with an N-d array")
end
function nd_mdivide(a, b, what)
    (a isa Real || b isa Real) && return what == "/" ? nd_ew(/, a, b) : nd_ew((x, y) -> y / x, a, b)
    length(ndarr(b)) == 1 && what == "/" && return nd_ew(/, a, b)
    length(ndarr(a)) == 1 && what == "\\" && return nd_ew((x, y) -> y / x, a, b)
    _unsup("matrix division with an N-d array")
end
nd_mpower(a, b) = merr("for x^y, only square matrix arguments are permitted and one argument must be scalar")

# ---------------------------------------------------------------------- concatenation
function _dv_concat(dv, dvb, dim::Int)        # dim is 1-based
    dv = collect(Int, dv); dvb = collect(Int, dvb)
    orig_nd, ndb = length(dv), length(dvb)
    new_nd = dim <= ndb ? ndb : dim
    if new_nd > orig_nd
        append!(dv, ones(Int, new_nd - orig_nd))
    else
        new_nd = orig_nd
    end
    match = all(i -> i == dim || dv[i] == dvb[i], 1:ndb) && all(i -> i == dim || dv[i] == 1, ndb+1:new_nd)
    if match
        dv[dim] += dim <= ndb ? dvb[dim] : 1
    elseif ndb == 2 && dvb == [0, 0]
        match = true
    elseif orig_nd == 2 && dv[1] == 0 && dv[2] == 0
        match, dv = true, dvb
    end
    match, collect(canon_dims(dv))
end

function _dv_hvcat(dv, dvb, dim::Int)
    ok, new = _dv_concat(dv, dvb, dim)
    ok && return ok, new
    dv = collect(canon_dims(dv)); dvb = collect(Int, dvb)
    if length(dv) == 2 && length(dvb) == 2
        dvb[1] + dvb[2] == 1 && return true, (dv[1] + dv[2] == 1 ? [0, 0] : dv)
        dv[1] + dv[2] == 1 && return true, dvb
    end
    false, dv
end

"Concatenation along dimension `d` (1-based) when an operand or the result has more than two dimensions; `hv` is the rule of `[a, b]`."
function nd_concat(vals, d::Int, hv::Bool)
    arrays = Any[ndarr(v) for v in vals]
    n = length(arrays)
    istart = 1
    if !hv && n > 2 && d > 2
        while istart <= n && size(arrays[istart]) == (0, 0)
            istart += 1
        end
        istart > n && (istart = 1)
    end
    dv = collect(Int, size(arrays[istart]))
    for a in arrays[istart+1:end]
        ok, new = hv ? _dv_hvcat(dv, collect(size(a)), d) : _dv_concat(dv, collect(size(a)), d)
        if !ok
            merr(hv ? "concatenation operator not implemented for '$(_dimstr(dv))' by '$(_dimstr(size(a)))' operations" :
                      "cat: dimension mismatch in dimension $d")
        end
        dv = new
    end
    (!hv && d > 2 && all(a -> size(a) == (0, 0), arrays)) &&
        _unsup("cat along a dimension above 2 of empty 0x0 arrays only (the result's shape is Octave's)")
    logical = all(a -> eltype(a) == Bool, arrays)
    shape = Tuple(dv)
    _check_elements(shape)
    parts = [a for a in arrays if !isempty(a)]
    T = logical ? Bool : Float64
    isempty(parts) && return ndret(zeros(T, shape))
    nd = max(length(shape), d)
    out = cat([reshape(a, _padn(size(a), nd)) for a in parts]...; dims=d)
    canon_dims(size(out)) != canon_dims(shape) && merr("concatenation: dimension mismatch")
    ndret(logical ? out : Array{Float64}(out))
end

# ---------------------------------------------------------------------- builtins
function _nd_dim_arg(x::AbstractArray, dimarg)
    if dimarg !== nothing
        d = as_index_int(dimarg, "dimension")
        d < 1 && merr("dimension must be a positive integer")
        return d <= ndims(x) ? d : nothing
    end
    ndims(x) == 2 ? (size(x, 1) == 1 ? 2 : 1) : findfirst(!=(1), size(x))
end

"Apply `f` (slice vector -> number) along dimension `d`; the result has size 1 there."
function _slices(f, x::AbstractArray, d::Int, empty_value)
    rsz = collect(size(x)); rsz[d] = 1
    prod(rsz) == 0 && return zeros(Float64, rsz...)
    size(x, d) == 0 && return fill(Float64(empty_value), rsz...)
    out = Array{Float64}(undef, rsz...)
    for I in CartesianIndices(Tuple(rsz))
        idx = Any[Tuple(I)...]
        idx[d] = Colon()
        out[I] = f(vec(x[idx...]))
    end
    out
end
function _slices2(f, x::AbstractArray, d::Int)      # f returns (value, index)
    rsz = collect(size(x)); rsz[d] = 1
    v = Array{Float64}(undef, rsz...); k = Array{Float64}(undef, rsz...)
    for I in CartesianIndices(Tuple(rsz))
        idx = Any[Tuple(I)...]
        idx[d] = Colon()
        v[I], k[I] = f(vec(x[idx...]))
    end
    v, k
end

_nd_median(v) = (any(isnan, v) ? NaN : (s = sort(v); n = length(s); isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2))
_nd_var(v, ddof) = (n = length(v); m = sum(v) / n; sum(abs2, v .- m) / (n - ddof))

const _ND_REDUCERS = Dict{String,Tuple{Function,Float64}}(
    "sum" => (v -> sum(v), 0.0), "prod" => (v -> prod(v), 1.0),
    "mean" => (v -> sum(v) / length(v), NaN), "median" => (_nd_median, NaN))

_isemptyarg(x) = x isa AbstractMatrix && isempty(x)

function _nd_reduce(name, a)
    _nargs(a, 1, 2, name)
    x = ndfarr(a[1])
    size(x) == (0, 0) && return name == "sum" ? 0.0 : name == "prod" ? 1.0 : NaN
    d = _nd_dim_arg(x, length(a) > 1 ? a[2] : nothing)
    d === nothing && return ndret(x)
    f, ev = _ND_REDUCERS[name]
    ndret(_slices(f, x, d, ev))
end

function _nd_stdvar(name, a)
    x = ndfarr(a[1])
    w = 0.0
    if length(a) > 1 && !_isemptyarg(a[2])
        w = scalar(a[2])
        w in (0.0, 1.0) || _unsup("weighted variance")
    end
    d = _nd_dim_arg(x, length(a) > 2 ? a[3] : nothing)
    if isempty(x)
        ndims(x) > 2 && _unsup("std/var of an empty N-d array")
        return NaN
    end
    d === nothing && return ndret(zeros(size(x)))
    n = size(x, d)
    ddof = (w == 1.0 || n == 1) ? 0 : 1
    r = _slices(v -> _nd_var(v, ddof), x, d, NaN)
    ndret(name == "std" ? sqrt.(r) : r)
end

function _nd_cum(name, a)
    _nargs(a, 1, 2, name)
    length(a) == 2 && a[2] isa AbstractString && _unsup("cumsum/cumprod with an option")
    x = ndfarr(a[1])
    d = _nd_dim_arg(x, length(a) > 1 ? a[2] : nothing)
    (d === nothing || isempty(x)) && return ndret(x)
    ndret(name == "cumsum" ? cumsum(x; dims=d) : cumprod(x; dims=d))
end

_fmax(x, y) = isnan(x) ? y : isnan(y) ? x : max(x, y)
_fmin(x, y) = isnan(x) ? y : isnan(y) ? x : min(x, y)

function _nd_minmax(name, a, nout)
    _nargs(a, 1, 3, name)
    if length(a) >= 2 && !_isemptyarg(a[2])
        length(a) == 3 && merr("$name: with a dimension argument the second argument must be []")
        nout > 1 && merr("$name: two outputs are not available with two inputs")
        return nd_ew(name == "max" ? _fmax : _fmin, a[1], a[2])
    end
    x = ndfarr(a[1])
    if isempty(x)
        ndims(x) > 2 && _unsup("$name of an empty N-d array")
        return Any[zeros(0, 0), zeros(0, 0)][1:max(nout, 1)]
    end
    d = _nd_dim_arg(x, length(a) > 2 ? a[3] : nothing)
    d === nothing && return Any[ndret(x), ndret(ones(size(x)))][1:max(nout, 1)]
    ismax = name == "max"
    v, k = _slices2(x, d) do s
        allnan = all(isnan, s)
        y = map(t -> isnan(t) ? (ismax ? -Inf : Inf) : t, s)
        idx = ismax ? argmax(y) : argmin(y)
        (allnan ? NaN : y[idx], allnan ? 1.0 : Float64(idx))
    end
    nout <= 1 ? ndret(v) : Any[ndret(v), ndret(k)]
end

function _nd_anyall(name, a)
    _nargs(a, 1, 2, name)
    x = ndarr(a[1])
    d = _nd_dim_arg(x, length(a) > 1 ? a[2] : nothing)
    if eltype(x) != Bool
        nan = isnan.(x)
        if name == "all" && any(nan) && (d === nothing || (d == 2 && size(x, 1) > 1) || (d != 1 && ndims(x) > 2))
            _unsup("all(X, DIM) of an array containing NaN (MATLAB and Octave differ)")
        end
        x = name == "any" ? ((x .!= 0) .& .!nan) : (x .!= 0)
    end
    size(x) == (0, 0) && return name == "all"
    d === nothing && return ndret(x)
    rsz = collect(size(x)); rsz[d] = 1
    ndret(name == "any" ? Array{Bool}(any(x; dims=d)) : Array{Bool}(all(x; dims=d)))
end

function _nd_find(a, nout)
    _nargs(a, 1, 3, "find")
    nout > 1 && _unsup("find with several outputs")
    x = ndarr(a[1])
    nz = eltype(x) == Bool ? vec(x) : map(t -> isnan(t) || t != 0, vec(x))
    k = Float64.(findall(nz))
    if length(a) >= 2
        n = as_index_int(a[2], "find: N")
        n < 0 && merr("find: N must be a non-negative integer")
        last = false
        if length(a) == 3
            dd = lowercase(String(a[3]))
            dd in ("first", "last") || merr("find: DIRECTION must be \"first\" or \"last\"")
            last = dd == "last"
        end
        k = last ? k[max(length(k) - n, 0)+1:end] : k[1:min(n, length(k))]
    end
    row = ndims(x) == 2 && size(x, 1) == 1 && size(x, 2) != 1
    if !isempty(k)
        return ret(row ? reshape(k, 1, :) : reshape(k, :, 1))
    end
    (size(x) == (0, 0) || size(x) == (1, 1)) && return zeros(0, 0)
    row ? zeros(1, 0) : zeros(0, 1)
end

function _nd_flip(name, a)
    x = ndarr(a[1])
    d = name == "fliplr" ? 2 : name == "flipud" ? 1 : _nd_dim_arg(x, length(a) > 1 ? a[2] : nothing)
    (d === nothing || d > ndims(x)) && return ndret(x)
    ndret(reverse(x; dims=d))
end

function _nd_size(a, nout)
    isempty(a) && merr("Invalid call to size")
    shp = collect(Int, nd_shape(a[1]))
    if length(a) >= 2
        ds = length(a) == 2 ? vec(ndfarr(a[2])) : Float64[scalar(x, "size: DIM") for x in a[2:end]]
        dims = Int[]
        for x in ds
            (x != floor(x) || !(x >= 1) || !isfinite(x)) && merr("size: requested dimension DIM (= $(x)) out of range")
            push!(dims, Int(x))
        end
        vals = Float64[d <= length(shp) ? shp[d] : 1.0 for d in dims]
        nout <= 1 && return length(vals) == 1 ? vals[1] : reshape(vals, 1, :)
        nout != length(vals) && merr("size: nargout > 1 but does not match number of requested dimensions")
        return Any[v for v in vals]
    end
    nout <= 1 && return reshape(Float64.(shp), 1, :)
    d = redim(Tuple(shp), nout)
    Any[Float64(x) for x in d]
end

function _nd_squeeze(a)
    v = a[1]
    is_nd(v) || return v
    d = Int[x for x in size(v) if x != 1]
    length(d) < 2 && (d = vcat(d, ones(Int, 2 - length(d))))
    ndret(reshape(v, Tuple(d)))
end

function _perm_vec(p, nd, name)
    p = vec(ndfarr(p))
    (length(p) < nd || any(x -> x != floor(x), p) || sort(Int.(p)) != collect(1:length(p))) &&
        merr("$name: PERM is not a valid permutation vector")
    Int.(p)
end
function _nd_permute(a, inverse::Bool)
    _nargs(a, 2, 2, "permute")
    is_num(a[1]) || is_nd(a[1]) || _unsup("permute of a $(mclass(a[1])) value")
    x = ndarr(a[1])
    p = _perm_vec(a[2], ndims(x), "permute")
    inverse && (p = sortperm(p))
    ndret(permutedims(reshape(x, _padn(size(x), length(p))), p))
end

"Dimensions given as `n`, `m, n, ...` or `[m n ...]`; negative ones are 0."
function _nd_dims(a, name)
    isempty(a) && return (1, 1)
    if length(a) == 1
        d = vec(ndfarr(a[1]))
        if length(d) == 1
            n = max(0, as_index_int(d[1], "dimension"))
            return (n, n)
        end
        dd = Int[max(0, as_index_int(x, "dimension")) for x in d]
    else
        dd = Int[max(0, as_index_int(x, "dimension")) for x in a]
    end
    sz = canon_dims(dd)
    _check_elements(sz)
    sz
end

function _nd_const(value, a, T=Float64)
    ndret(fill(T(value), _nd_dims(a, "zeros")))
end

function _nd_repmat(a)
    length(a) < 2 && merr("Invalid call to repmat")
    reps = _nd_dims(a[2:end], "repmat")
    a[1] isa AbstractString && _unsup("repmat of a char array with N-d replication")
    a[1] isa Cell && _unsup("repmat of a cell array")
    x = ndarr(a[1])
    n = max(ndims(x), length(reps))
    reps, xs = _padn(reps, n), _padn(size(x), n)
    _check_elements(Tuple(collect(reps) .* collect(xs)))
    ndret(repeat(reshape(x, xs), reps...))
end

function _nd_reshape(a)
    (a[1] isa AbstractString || a[1] isa Cell) && _unsup("reshape of a char or cell array")
    length(a) < 2 && merr("Invalid call to reshape")
    x = ndarr(a[1])
    dims = a[2:end]
    length(dims) == 1 && (dims = Any[Float64(v) for v in vec(ndfarr(dims[1]))])
    length(dims) < 2 && merr("reshape: SIZE must have 2 or more dimensions")
    d = Union{Nothing,Int}[_isemptyarg(v) ? nothing : as_index_int(v) for v in dims]
    count(isnothing, d) > 1 && merr("reshape: only a single dimension can be unknown")
    any(v -> v !== nothing && v < 0, d) && merr("reshape: SIZE must be non-negative")
    if any(isnothing, d)
        known = prod(Int[v for v in d if v !== nothing]; init=1)
        (known == 0 || length(x) % known != 0) && merr("reshape: SIZE is not divisible by the product of known dimensions (= $known)")
        d[findfirst(isnothing, d)] = length(x) ÷ known
    end
    dd = Int[v for v in d]
    prod(dd) != length(x) && merr("reshape: can't reshape array to the requested dimensions")
    ndret(reshape(x, canon_dims(dd)))
end

function _nd_cat(a)
    d = as_index_int(a[1])
    d < 1 && merr("cat: DIM must be a valid dimension")
    vals = a[2:end]
    isempty(vals) && return zeros(0, 0)
    for v in vals
        (is_num(v) || is_nd(v)) || (v isa Tainted && arr(v); _unsup("cat along dimension $d of a $(mclass(v)) value"))
    end
    length(vals) == 1 && return vals[1]
    nd_concat(vals, d, false)
end

function _nd_sub2ind(a)
    length(a) < 2 && merr("Invalid call to sub2ind")
    dims = Int[max(0, as_index_int(x, "sub2ind: DIMS")) for x in vec(ndfarr(a[1]))]
    length(dims) < 2 && merr("sub2ind: DIMS must contain integers")
    subs = Any[ndfarr(x) for x in a[2:end]]
    any(x -> size(x) != size(subs[1]), subs) && merr("sub2ind: all subscripts must be of the same size")
    dv = redim(Tuple(dims), length(subs))
    out = zeros(size(subs[1]))
    mult = 1
    for (x, n) in zip(subs, dv)
        (any(t -> t != floor(t), x) || any(<(1), x)) && merr("sub2ind: subscripts must be either integers 1 to (2^63)-1 or logicals")
        any(>(n), x) && merr("index ($(Int(maximum(x)))...): out of bound $n (dimensions are $(_dimstr(Tuple(dims))))")
        out = out .+ (x .- 1) .* mult
        mult *= n
    end
    ndret(out .+ 1)
end

function _nd_ind2sub(a, nout)
    _nargs(a, 2, 2, "ind2sub")
    dims = Int[max(0, as_index_int(x, "ind2sub: DIMS")) for x in vec(ndfarr(a[1]))]
    length(dims) < 2 && merr("ind2sub: DIMS must contain integers")
    k = ndfarr(a[2])
    nout = max(nout, 1)
    dv = nout > 1 ? redim(Tuple(dims), nout) : (prod(dims),)
    (any(t -> t != floor(t), k) || any(<(1), k)) && merr("ind2sub: subscripts must be either integers 1 to (2^63)-1 or logicals")
    any(>(prod(dims)), k) && merr("ind2sub: index out of range")
    rest = Int.(k) .- 1
    out = Any[]
    for n in dv[1:nout]
        if length(out) == nout - 1
            push!(out, ndret(Float64.(rest .+ 1)))
            break
        end
        push!(out, ndret(Float64.(rest .% n .+ 1)))
        rest = rest .÷ n
    end
    out
end

function _nd_isequal(a)
    length(a) < 2 && merr("isequal: at least two arguments")
    function same(x, y)
        (x isa AbstractString || y isa AbstractString) && _unsup("isequal of char and N-d numeric")
        ((is_num(x) || is_nd(x)) && (is_num(y) || is_nd(y))) || _unsup("isequal on this kind of value")
        p, q = ndarr(x), ndarr(y)
        size(p) == size(q) && all(p .== q)
    end
    all(same(a[1], y) for y in a[2:end])
end

# unary builtins applied to the elements: run the 2-D builtin on the flattened column and reshape
const _ND_UNARY = ("exp", "log", "log2", "log10", "sqrt", "abs", "sign", "floor", "ceil", "fix", "round", "sin", "cos", "tan", "asin", "acos",
                   "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "gamma", "gammaln", "isnan", "isinf", "isfinite",
                   "double", "logical", "real", "uplus", "not", "uminus")
const _ND_BINARY = ("mod", "rem", "atan2", "hypot", "power", "plus", "minus", "times", "rdivide", "ldivide", "and", "or", "xor",
                    "eq", "ne", "lt", "le", "gt", "ge")

function _nd_unary(I, name, x)
    f = BUILTINS[name]
    r = f(I, Any[reshape(x, :, 1)], 1)
    r = r isa Vector{Any} ? r[1] : r
    ndret(reshape(Array(arr(r)), size(x)))
end
function _nd_binary(I, name, a, b)
    f = BUILTINS[name]
    x, y = ndarr(a), ndarr(b)
    _nd_bcast(x, y)
    n = max(ndims(x), ndims(y))
    sz = Tuple(max.(collect(_padn(size(x), n)), collect(_padn(size(y), n))))
    (0 in _padn(size(x), n) || 0 in _padn(size(y), n)) && (sz = Tuple(ifelse.(collect(_padn(size(x), n)) .== 1, collect(_padn(size(y), n)), collect(_padn(size(x), n)))))
    X = repeat_to(x, sz); Y = repeat_to(y, sz)
    r = f(I, Any[reshape(X, :, 1), reshape(Y, :, 1)], 1)
    r = r isa Vector{Any} ? r[1] : r
    ndret(reshape(Array(arr(r)), sz))
end
"Expand singleton dimensions of `x` to `sz`."
function repeat_to(x::AbstractArray, sz)
    xs = _padn(size(x), length(sz))
    xr = reshape(x, xs)
    repeat(xr, (Int[xs[i] == sz[i] ? 1 : sz[i] for i in eachindex(sz)])...)
end

"""
    nd_hook(I, name, args, nout) -> nothing | result

Called by `mcall` before a builtin runs: handles the builtins whose arguments have more than two dimensions, or whose
result does. Returns `nothing` when the call is none of that.
"""
function nd_hook(I, name::String, a::Vector{Any}, nout::Int)
    haskey(_ND_NAMES, name) || return nothing
    anynd = any(is_nd, a)
    # creation functions: three or more dimensions
    if name in ("zeros", "ones", "nan", "NaN", "Inf", "inf", "true", "false")
        (isempty(a) || any(x -> x isa AbstractString, a) || anynd) && return nothing
        all(x -> x isa Real || x isa AbstractMatrix{<:Real}, a) || return nothing
        nd = length(a) == 1 ? length(a[1]) : length(a)
        nd >= 3 || return nothing
        val = name in ("zeros", "false") ? 0.0 : name == "ones" || name == "true" ? 1.0 : name in ("nan", "NaN") ? NaN : Inf
        sz = _nd_dims(a, name)
        return ndret(name in ("true", "false") ? fill(val != 0, sz) : fill(val, sz))
    end
    if name == "eye"
        (length(a) >= 3 && all(x -> x isa Real, a)) && merr("Invalid call to eye.")
        return nothing
    end
    if name == "cat"
        (length(a) >= 1 && a[1] isa Real && (anynd || a[1] >= 3)) || return nothing
        return _nd_cat(a)
    end
    if name == "repmat"
        (anynd || length(a) >= 4 || (length(a) == 3 && !_isscalar_arg(a[2])) || (length(a) == 2 && a[2] isa AbstractMatrix && length(a[2]) >= 3)) || return nothing
        return _nd_repmat(a)
    end
    if name == "reshape"
        (length(a) >= 2 && !(a[1] isa AbstractString || a[1] isa Cell)) || return nothing
        dims = a[2:end]
        multi_empty = count(_isemptyarg, dims) > 1 || (length(dims) == 1 && length(dims[1]) >= 3)
        (anynd || length(dims) >= 3 || multi_empty) || return nothing
        return _nd_reshape(a)
    end
    if name in ("permute", "ipermute")
        return _nd_permute(a, name == "ipermute")
    end
    if name == "squeeze"
        return anynd ? _nd_squeeze(a) : nothing
    end
    if name == "size"
        (anynd || nout > 2 || length(a) > 1) || return nothing
        (length(a) >= 1 && (a[1] isa Cell || a[1] isa AbstractString || a[1] isa MStruct || a[1] isa HostStruct || a[1] isa Handle || a[1] isa MObject)) && return nothing
        return _nd_size(a, nout)
    end
    if name in ("ndims", "numel", "length", "isempty", "isscalar", "isvector", "isrow", "iscolumn", "ismatrix", "issquare", "rows", "columns")
        (length(a) == 1 && anynd) || return nothing
        s = size(a[1])
        name == "ndims" && return Float64(length(s))
        name == "numel" && return Float64(prod(s))
        name == "length" && return Float64(0 in s ? 0 : maximum(s))
        name == "isempty" && return 0 in s
        name in ("isscalar", "isvector", "isrow", "iscolumn", "ismatrix", "issquare") && return false
        name == "rows" && return Float64(s[1])
        return Float64(s[2])
    end
    if name in ("sum", "prod", "mean", "median")
        (anynd || (length(a) >= 2 && a[2] isa Real && a[2] >= 3)) || return nothing
        return _nd_reduce(name, a)
    end
    if name in ("std", "var")
        (anynd || (length(a) >= 3 && a[3] isa Real && a[3] >= 3)) || return nothing
        return _nd_stdvar(name, a)
    end
    if name in ("cumsum", "cumprod")
        (anynd || (length(a) >= 2 && a[2] isa Real && a[2] >= 3)) || return nothing
        return _nd_cum(name, a)
    end
    if name in ("max", "min")
        anynd || (length(a) == 3 && a[3] isa Real && a[3] >= 3) || return nothing
        return _nd_minmax(name, a, nout)
    end
    if name in ("any", "all")
        (anynd || (length(a) >= 2 && a[2] isa Real && a[2] >= 3)) || return nothing
        return _nd_anyall(name, a)
    end
    if name == "find"
        anynd || return nothing
        return _nd_find(a, nout)
    end
    if name == "nnz"
        anynd || return nothing
        x = ndarr(a[1])
        return Float64(count(t -> isnan(t) || t != 0, x))
    end
    if name in ("flip", "fliplr", "flipud")
        (anynd || (name == "flip" && length(a) == 2 && a[2] isa Real && a[2] >= 3)) || return nothing
        return _nd_flip(name, a)
    end
    if name == "sub2ind"
        (anynd || length(a) >= 4 || (length(a) == 3 && a[1] isa AbstractMatrix && length(a[1]) >= 3) ||
         (length(a) >= 2 && a[1] isa AbstractMatrix && length(a[1]) > length(a) - 1)) || return nothing
        return _nd_sub2ind(a)
    end
    if name == "ind2sub"
        (nout > 2 || (length(a) == 2 && a[1] isa AbstractMatrix && length(a[1]) >= 3)) || return nothing
        return _nd_ind2sub(a, nout)
    end
    if name == "isequal"
        anynd || return nothing
        return _nd_isequal(a)
    end
    if name == "bsxfun"
        anynd || return nothing
        length(a) == 3 && a[1] isa Handle && a[1].name !== nothing && a[1].name in _ND_BINARY &&
            return _nd_binary(I, a[1].name, a[2], a[3])
        _unsup("bsxfun with an N-d array and a function other than an element-wise built-in")
    end
    if name in _ND_UNARY
        (anynd && length(a) == 1) || return nothing
        return _nd_unary(I, name, a[1])
    end
    if name in _ND_BINARY
        (anynd && length(a) == 2) || return nothing
        return _nd_binary(I, name, a[1], a[2])
    end
    if name in ("transpose", "ctranspose")
        anynd && return nd_transpose(a[1])
        return nothing
    end
    nothing
end
_isscalar_arg(x) = x isa Real || (x isa AbstractMatrix && length(x) == 1)

const _ND_NAMES = Dict{String,Bool}(n => true for n in
    ("zeros", "ones", "nan", "NaN", "Inf", "inf", "true", "false", "eye", "cat", "repmat", "reshape", "permute", "ipermute", "squeeze",
     "size", "ndims", "numel", "length", "isempty", "isscalar", "isvector", "isrow", "iscolumn", "ismatrix", "issquare", "rows", "columns",
     "sum", "prod", "mean", "median", "std", "var", "cumsum", "cumprod", "max", "min", "any", "all", "find", "nnz", "flip", "fliplr", "flipud",
     "sub2ind", "ind2sub", "isequal", "bsxfun", "transpose", "ctranspose",
     "exp", "log", "log2", "log10", "sqrt", "abs", "sign", "floor", "ceil", "fix", "round", "sin", "cos", "tan", "asin", "acos", "atan",
     "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "gamma", "gammaln", "isnan", "isinf", "isfinite", "double", "logical", "real", "uplus",
     "not", "uminus", "mod", "rem", "atan2", "hypot", "power", "plus", "minus", "times", "rdivide", "ldivide", "and", "or", "xor",
     "eq", "ne", "lt", "le", "gt", "ge"))
