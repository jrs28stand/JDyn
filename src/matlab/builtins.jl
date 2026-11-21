# Built-in MATLAB functions for the interpreter (port of pydyn/matlab.py).
# A builtin is f(I, args::Vector{Any}, nout::Int) and returns one value, or a Vector{Any} of values
# (several outputs, or none). LAZY builtins receive their arguments unevaluated:
# f(I, argnodes, scope, nout).

const BUILTINS = Dict{String,Function}()
const LAZY = Dict{String,Function}()

bi!(f, names...) = (for n in names; BUILTINS[n] = f; end; f)
lazy!(f, names...) = (for n in names; LAZY[n] = f; end; f)

"""Argument count check. Too few arguments is a MATLAB error; more than JDyn implements may be a valid
MATLAB call (round(x, n), sum(x, 'omitnan'), ...), so it is refused, not raised as a MATLAB error that a
try/catch would catch."""
function _nargs(a, lo, hi=lo, name="function")
    length(a) < lo && merr("$name: not enough input arguments ($(length(a)))")
    length(a) > hi && throw(Unsupported("$name with $(length(a)) arguments", "feature"))
    true
end

_isemptyarr(x) = x isa AbstractMatrix && isempty(x)

# ---------------------------------------------------------------------- element-wise functions
"Apply a real function element-wise; a DomainError (MATLAB would return a complex number) is refused."
function _apply_elem(fn, x)
    try
        if x isa Real
            return ret(fn(Float64(x)))
        end
        return ret(fn.(farr(x)))
    catch e
        e isa DomainError && _cplx()
        rethrow()
    end
end
function _elem!(name, fn)
    BUILTINS[name] = (I, a, nout) -> begin
        _nargs(a, 1, 1, name)
        _apply_elem(fn, a[1])
    end
end

function _mround(x::Float64)
    t = trunc(x)
    t + sign(x) * (abs(x - t) >= 0.5)
end
_mgamma(x::Float64) = (x <= 0 && x == round(x)) ? (x == 0 ? Inf : NaN) : SpecialFunctions.gamma(x)
_mgammaln(x::Float64) = x < 0 ? _cplx() : (x == 0 ? Inf : SpecialFunctions.loggamma(x))
_ndtri(p::Float64) = (p < 0 || p > 1) ? NaN : sqrt(2.0) * SpecialFunctions.erfinv(2p - 1)
_msign(x::Float64) = isnan(x) ? NaN : Float64(sign(x))
"Julia's sin/cos/tan throw a DomainError at ±Inf; MATLAB returns NaN (a real result, not a complex one)."
_periodic(f) = x -> isinf(x) ? NaN : f(x)

for (n, f) in Dict("exp" => exp, "log" => log, "log2" => log2, "log10" => log10, "sqrt" => sqrt, "abs" => abs,
                   "sign" => _msign, "floor" => floor, "ceil" => ceil, "fix" => trunc, "round" => _mround,
                   "sin" => _periodic(sin), "cos" => _periodic(cos), "tan" => _periodic(tan), "asin" => asin, "acos" => acos, "atan" => atan,
                   "sinh" => sinh, "cosh" => cosh, "tanh" => tanh, "asinh" => asinh, "acosh" => acosh,
                   "atanh" => atanh, "gamma" => _mgamma, "gammaln" => _mgammaln,
                   "erf" => SpecialFunctions.erf, "erfc" => SpecialFunctions.erfc,
                   "log1p" => log1p, "expm1" => expm1,
                   "real" => identity, "conj" => identity, "uplus" => identity,
                   "factorial" => (x -> round(_mgamma(x + 1))))
    _elem!(n, f)
end
for (n, f) in Dict("isnan" => isnan, "isinf" => isinf, "isfinite" => isfinite)
    BUILTINS[n] = (I, a, nout) -> (_nargs(a, 1, 1, n); x = a[1]; x isa Real ? f(Float64(x)) : ret(f.(farr(x))))
end
for (n, f) in Dict("plus" => plus, "minus" => minus, "times" => times, "rdivide" => rdivide, "ldivide" => ldivide,
                   "power" => power, "mtimes" => mtimes, "mrdivide" => mrdivide, "mldivide" => mldivide, "mpower" => mpower,
                   "lt" => mlt, "gt" => mgt, "le" => mle, "ge" => mge, "eq" => meq, "ne" => mne, "and" => el_and, "or" => el_or,
                   "atan2" => ((a, b) -> _ew(atan, a, b)), "hypot" => ((a, b) -> _ew(hypot, a, b)))
    BUILTINS[n] = (I, a, nout) -> (_nargs(a, 2, 2, n); f(a[1], a[2]))
end

BUILTINS["double"] = function (I, a, nout)
    x = a[1]
    if x isa AbstractString
        isempty(x) && return zeros(0, 0)
        return length(x) == 1 ? Float64(Int(x[1])) : reshape(_chars(x), 1, :)
    end
    ret(farr(x))
end
BUILTINS["class"] = (I, a, nout) -> mclass(a[1])
BUILTINS["uminus"] = (I, a, nout) -> uminus(a[1])
BUILTINS["not"] = (I, a, nout) -> lnot(a[1])
BUILTINS["transpose"] = BUILTINS["ctranspose"] = (I, a, nout) -> mtranspose(a[1])
BUILTINS["colon"] = (I, a, nout) -> colon(a...)
BUILTINS["horzcat"] = (I, a, nout) -> mhcat(a)
BUILTINS["vertcat"] = (I, a, nout) -> mvcat(a)
BUILTINS["pi"] = (I, a, nout) -> _const(Float64(pi), a)
BUILTINS["eps"] = (I, a, nout) -> (isempty(a) || !is_num(a[1]) || a[1] isa AbstractMatrix) ? _const(eps(Float64), a) : eps(abs(scalar(a[1])))
BUILTINS["Inf"] = BUILTINS["inf"] = (I, a, nout) -> _const(Inf, a)
BUILTINS["NaN"] = BUILTINS["nan"] = (I, a, nout) -> _const(NaN, a)
BUILTINS["true"] = (I, a, nout) -> isempty(a) ? true : ret(arr(_const(1.0, a)) .!= 0)
BUILTINS["false"] = (I, a, nout) -> isempty(a) ? false : ret(arr(_const(0.0, a)) .!= 0)
BUILTINS["zeros"] = (I, a, nout) -> _const(0.0, a)
BUILTINS["ones"] = (I, a, nout) -> _const(1.0, a)
BUILTINS["realmax"] = (I, a, nout) -> floatmax(Float64)
BUILTINS["realmin"] = (I, a, nout) -> floatmin(Float64)

function _dims(a, name="size")
    a = [x for x in a if !(x isa AbstractString) || _unsup("$name: class or 'like' arguments")]
    isempty(a) && return (1, 1)
    if length(a) == 1
        d = arr(a[1])
        if length(d) == 1
            n = max(0, as_index_int(d[1], "dimension"))
            return (n, n)
        end
        d = [max(0, as_index_int(x, "dimension")) for x in vec(d)]
    else
        d = [max(0, as_index_int(x, "dimension")) for x in a]
    end
    (length(d) > 2 && any(!=(1), d[3:end])) && _unsup("arrays with more than two dimensions")
    d[1] * d[2] > MAX_ELEMENTS && _unsup("array with more than $MAX_ELEMENTS elements")
    (d[1], d[2])
end
_const(value, a) = isempty(a) ? Float64(value) : ret(fill(Float64(value), _dims(a)))

bi!("eye") do I, a, nout
    r, c = _dims(a)
    ret(Matrix{Float64}(LinearAlgebra.I, r, c))
end
# set_dynare_seed(n | 'default' | 'clock' | 'reset' | algo, n): seeds the model's own stream (src/seeds.jl)
bi!("set_dynare_seed") do I, a, nout
    nout > 0 && merr("set_dynare_seed: too many output arguments")
    _matlab_set_dynare_seed(I, a)
    Any[]
end
bi!("i", "j", "complex", "imag", "angle") do I, a, nout
    _unsup("complex numbers")
end

"mod(x, y) / rem(x, y) with MATLAB's treatment of round-off (mod(1, 0.1) is 0)."
function _mmod1(x::Float64, y::Float64, isrem::Bool)
    isinf(y) && _unsup("mod/rem with an infinite divisor (MATLAB and Octave disagree)")
    (isnan(x) || isnan(y) || isinf(x)) && return NaN
    ys = y == 0 ? 1.0 : y
    q = x / ys
    n = isrem ? trunc(q) : floor(q)
    r = x - ys * n
    nq = round(q)
    near = (round(ys) != ys) && abs((q - nq) / (nq == 0 ? 1.0 : nq)) < eps(Float64) && nq != 0
    near && (r = 0.0)
    if isrem
        r = x != ys ? copysign(abs(r), x) : 0.0 * x
        y == 0 && (r = NaN)
    else
        r = x != ys ? (r == 0 ? 0.0 : copysign(abs(r), ys)) : 0.0
        y == 0 && (r = x)
    end
    r
end
matlab_mod(x, y; isrem=false) = _ew((p, q) -> _mmod1(p, q, isrem), x, y)

bi!("mod") do I, a, nout
    _nargs(a, 2, 2, "mod")
    _bcast(arr(a[1]), arr(a[2]))
    matlab_mod(a[1], a[2])
end
bi!("rem") do I, a, nout
    _nargs(a, 2, 2, "rem")
    _bcast(arr(a[1]), arr(a[2]))
    matlab_mod(a[1], a[2]; isrem=true)
end
BUILTINS["lgamma"] = BUILTINS["gammaln"]
BUILTINS["ln"] = BUILTINS["log"]
BUILTINS["cbrt"] = (I, a, nout) -> _apply_elem(cbrt, a[1])
BUILTINS["beta"] = (I, a, nout) -> _ew((p, q) -> SpecialFunctions.beta(p, q), a[1], a[2])
BUILTINS["betaln"] = (I, a, nout) -> _ew((p, q) -> SpecialFunctions.logbeta(p, q), a[1], a[2])

bi!("run") do I, a, nout
    name = length(a) == 1 ? _str(a[1], "run") : ""
    name = endswith(name, ".m") ? name[1:end-2] : name
    u = (occursin(r"^[A-Za-z]\w*$", name) && haskey(dir_files(I), name)) ? unit(I, name) : nothing
    if u === nothing || u isa ParseError || !is_script(u)
        throw(Unsupported("run: $(isempty(a) ? "" : repr(a[1])) is not a script next to the model file", "eval"))
    end
    run_script(I, u, I.scope_stack[end])
    Any[]
end

bi!("nthroot") do I, a, nout
    _nargs(a, 2, 2, "nthroot")
    x, n = farr(a[1]), scalar(a[2])
    n != round(n) && merr("nthroot: N must be an integer")
    (mod(n, 2) == 0 && any(<(0), x)) && merr("nthroot: negative X with even N")
    ret(sign.(x) .* abs.(x) .^ (1.0 / n))
end

"Dimension to reduce along (1 or 2): argument k if given, else the first non-singleton one; nothing if > 2."
function _dim_arg(x, a, k)
    if length(a) >= k
        d = as_index_int(a[k], "dimension")
        if !(d in (1, 2))
            d > 2 && return nothing
            merr("dimension must be a positive integer")
        end
        return d
    end
    size(x, 1) == 1 ? 2 : 1
end
function _median(v)
    any(isnan, v) && return NaN
    isempty(v) && return NaN
    s = sort(v)
    n = length(s)
    isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
end
function _reducer!(name, fn, empty)
    BUILTINS[name] = function (I, a, nout)
        _nargs(a, 1, 2, name)
        x = farr(a[1])
        (isempty(x) && size(x) == (0, 0)) && return Float64(empty)
        ax = _dim_arg(x, a, 2)
        ax === nothing && return ret(x)
        ret(fn(x, ax))
    end
end
_reducer!("sum", (x, d) -> sum(x; dims=d), 0.0)
_reducer!("prod", (x, d) -> prod(x; dims=d), 1.0)
_reducer!("mean", (x, d) -> size(x, d) == 0 ? fill(NaN, d == 1 ? (1, size(x, 2)) : (size(x, 1), 1)) : sum(x; dims=d) ./ size(x, d), NaN)
_reducer!("median", (x, d) -> d == 1 ? reshape([_median(x[:, j]) for j in 1:size(x, 2)], 1, :) : reshape([_median(x[i, :]) for i in 1:size(x, 1)], :, 1), NaN)

function _var(I, a, nout)
    x = farr(a[1])
    w = 0.0
    if length(a) > 1 && !_isemptyarr(a[2])
        w = scalar(a[2])
        w in (0.0, 1.0) || _unsup("weighted variance")
    end
    ax = _dim_arg(x, a, 3)
    isempty(x) && return NaN
    ax === nothing && return ret(zeros(size(x)))
    n = size(x, ax)
    ddof = (w == 1 || n == 1) ? 0 : 1
    m = sum(x; dims=ax) ./ n
    ret(sum(abs2.(x .- m); dims=ax) ./ (n - ddof))
end
BUILTINS["var"] = _var
BUILTINS["std"] = (I, a, nout) -> ret(sqrt.(farr(_var(I, a, nout))))
bi!("cumsum", "cumprod") do I, a, nout
    _unsup("cumsum/cumprod")
end

# diff(x), diff(x, n), diff(x, n, dim): differences along the first non-singleton dimension (data files of estimation)
BUILTINS["diff"] = function (I, a, nout)
    _nargs(a, 1, 3, "diff")
    x = farr(a[1])
    n = length(a) >= 2 && !_isemptyarr(a[2]) ? Int(scalar(a[2])) : 1
    n >= 0 || merr("diff: order must be a non-negative integer")
    dim = length(a) == 3 ? Int(scalar(a[3])) : (size(x, 1) != 1 ? 1 : 2)
    dim in (1, 2) || merr("diff: dimension must be 1 or 2")
    for _ in 1:n
        size(x, dim) <= 1 && return zeros(dim == 1 ? 0 : size(x, 1), dim == 1 ? size(x, 2) : 0)
        x = dim == 1 ? x[2:end, :] .- x[1:end-1, :] : x[:, 2:end] .- x[:, 1:end-1]
    end
    ret(x)
end

function _minmax!(name, better)
    BUILTINS[name] = function (I, a, nout)
        _nargs(a, 1, 3, name)
        if length(a) >= 2 && !_isemptyarr(a[2])
            length(a) == 3 && merr("$name: with a dimension argument the second argument must be []")
            nout > 1 && merr("$name: two outputs are not available with two inputs")
            return _ew((p, q) -> isnan(p) ? q : isnan(q) ? p : (better(q, p) ? q : p), a[1], a[2])
        end
        x = farr(a[1])
        isempty(x) && return nout > 1 ? Any[zeros(0, 0), zeros(0, 0)][1:max(nout, 1)] : zeros(0, 0)
        ax = _dim_arg(x, a, 3)
        if ax === nothing
            return nout > 1 ? Any[ret(x), ret(ones(size(x)))] : ret(x)
        end
        slices = ax == 1 ? [x[:, j] for j in 1:size(x, 2)] : [x[i, :] for i in 1:size(x, 1)]
        vals = Float64[]; idx = Float64[]
        for s in slices
            best, bi = NaN, 0
            for (k, v) in enumerate(s)
                isnan(v) && continue
                if isnan(best) || better(v, best)
                    best, bi = v, k
                end
            end
            push!(vals, best); push!(idx, bi == 0 ? 1.0 : Float64(bi))
        end
        shp = ax == 1 ? (1, length(vals)) : (length(vals), 1)
        nout <= 1 && return ret(reshape(vals, shp))
        Any[ret(reshape(vals, shp)), ret(reshape(idx, shp))]
    end
end
_minmax!("max", >)
_minmax!("min", <)

function _anyall!(name, fn)
    BUILTINS[name] = function (I, a, nout)
        _nargs(a, 1, 2, name)
        # any and all look at the non-zero entries; a NaN counts as neither (any(NaN) is false, all(NaN) true)
        x0 = arr(a[1])
        x = eltype(x0) == Bool ? x0 : (name == "any" ? ((x0 .!= 0) .& .!isnan.(x0)) : ((x0 .!= 0) .| isnan.(x0)))
        (isempty(x) && size(x) == (0, 0)) && return name == "all"
        ax = _dim_arg(x, a, 2)
        ax === nothing && return ret(x)
        ret(Matrix{Bool}(fn(x; dims=ax)))
    end
end
_anyall!("any", any)
_anyall!("all", all)

bi!("find") do I, a, nout
    _nargs(a, 1, 2, "find")
    nout > 1 && _unsup("find with several outputs")
    x = arr(a[1])
    k = Float64.(findall(!=(0), vec(x)))
    length(a) == 2 && (k = k[1:min(end, as_index_int(a[2]))])
    isrow = size(x, 1) == 1 && size(x, 2) != 1
    isempty(k) && return isrow ? zeros(1, 0) : zeros(0, 1)
    ret(isrow ? reshape(k, 1, :) : reshape(k, :, 1))
end

bi!("sort") do I, a, nout
    _nargs(a, 1, 2, "sort")
    x = farr(a[1])
    desc = length(a) == 2 && a[2] isa AbstractString && lowercase(a[2]) == "descend"
    (length(a) == 2 && !(a[2] isa AbstractString)) && _unsup("sort along a given dimension")
    (minimum(size(x)) != 1 && !isempty(x)) && _unsup("sort of a matrix")
    f = vec(x)
    key = [isnan(v) ? Inf : v for v in f]
    order = sortperm(desc ? -key : key; alg=MergeSort)
    if desc && any(isnan, f)
        order = vcat(order[isnan.(f[order])], order[.!isnan.(f[order])])
    end
    out = reshape(f[order], size(x))
    nout > 1 ? Any[ret(out), ret(reshape(Float64.(order), size(x)))] : ret(out)
end

bi!("unique") do I, a, nout
    (nout > 1 || length(a) != 1 || !is_num(a[1])) && _unsup("unique with several outputs or non-numeric input")
    x0 = arr(a[1])
    x = farr(x0)
    # NaNs are all different (MATLAB since R2013a, Octave): they are kept, at the end; -0 and 0 are equal
    v = vec(x)
    nn = sort(filter(!isnan, v))
    u = isempty(nn) ? Float64[] : [nn[i] for i in eachindex(nn) if i == 1 || nn[i] != nn[i-1]]
    append!(u, fill(NaN, count(isnan, v)))
    if isempty(x)
        return size(x) == (0, 0) ? zeros(0, 0) : zeros(0, 1)
    end
    res = size(x, 1) == 1 ? reshape(u, 1, :) : reshape(u, :, 1)
    eltype(x0) == Bool ? ret(Matrix{Bool}(res .!= 0)) : ret(res)
end
bi!("fliplr") do I, a, nout
    a[1] isa AbstractString ? reverse(a[1]) : ret(arr(a[1])[:, end:-1:1])
end
bi!("flipud") do I, a, nout
    ret(arr(a[1])[end:-1:1, :])
end
bi!("size") do I, a, nout
    _nargs(a, 1, 2, "size")
    shp = shape_of(a[1])
    if length(a) == 2
        d = as_index_int(a[2], "dimension")
        d < 1 && merr("size: dimension must be positive")
        return d <= 2 ? Float64(shp[d]) : 1.0
    end
    nout <= 1 && return [Float64(shp[1]) Float64(shp[2])]
    Any[Float64(shp[1]), Float64(shp[2]), fill(1.0, nout - 2)...]
end
BUILTINS["length"] = (I, a, nout) -> (s = shape_of(a[1]); 0 in s ? 0.0 : Float64(maximum(s)))
BUILTINS["numel"] = (I, a, nout) -> (s = shape_of(a[1]); Float64(s[1] * s[2]))
BUILTINS["ndims"] = (I, a, nout) -> 2.0
BUILTINS["rows"] = (I, a, nout) -> Float64(shape_of(a[1])[1])
BUILTINS["columns"] = (I, a, nout) -> Float64(shape_of(a[1])[2])
BUILTINS["isempty"] = (I, a, nout) -> 0 in shape_of(a[1])
BUILTINS["isscalar"] = (I, a, nout) -> shape_of(a[1]) == (1, 1)
BUILTINS["isvector"] = (I, a, nout) -> (s = shape_of(a[1]); 1 in s && !(0 in s))
BUILTINS["isrow"] = (I, a, nout) -> shape_of(a[1])[1] == 1
BUILTINS["iscolumn"] = (I, a, nout) -> shape_of(a[1])[2] == 1
BUILTINS["ismatrix"] = (I, a, nout) -> true
BUILTINS["issquare"] = (I, a, nout) -> (s = shape_of(a[1]); s[1] == s[2])
BUILTINS["isnumeric"] = BUILTINS["isfloat"] = (I, a, nout) -> mclass(a[1]) == "double"
BUILTINS["isreal"] = (I, a, nout) -> mclass(a[1]) in ("double", "logical", "char")
BUILTINS["ischar"] = (I, a, nout) -> a[1] isa AbstractString
BUILTINS["isstring"] = (I, a, nout) -> false
BUILTINS["iscell"] = (I, a, nout) -> a[1] isa Cell
BUILTINS["iscellstr"] = (I, a, nout) -> a[1] isa Cell && all(x -> x isa AbstractString, a[1].items)
BUILTINS["isstruct"] = (I, a, nout) -> mclass(a[1]) == "struct"
BUILTINS["islogical"] = (I, a, nout) -> mclass(a[1]) == "logical"
BUILTINS["is_function_handle"] = (I, a, nout) -> a[1] isa Handle
BUILTINS["squeeze"] = BUILTINS["full"] = (I, a, nout) -> a[1]
BUILTINS["vec"] = (I, a, nout) -> mindex(a[1], Any[COLON])
BUILTINS["nnz"] = (I, a, nout) -> Float64(count(!=(0), arr(a[1])))

bi!("isa") do I, a, nout
    _nargs(a, 2, 2, "isa")
    c = mclass(a[1])
    a[2] == c || (a[2] in ("float", "numeric") && c == "double")
end
bi!("logical") do I, a, nout
    ret(_logical(a[1]))
end
bi!("isequal") do I, a, nout
    length(a) < 2 && merr("isequal: at least two arguments")
    function same(x, y)
        (x isa Tainted || y isa Tainted) && arr(x isa Tainted ? x : y)
        if x isa AbstractString || y isa AbstractString
            (x isa AbstractString && y isa AbstractString) && return x == y
            return (x isa Cell || x isa MStruct || y isa Cell || y isa MStruct) ? false : _unsup("isequal of char and numeric")
        end
        if x isa Cell || y isa Cell
            return x isa Cell && y isa Cell && x.shape == y.shape && all(same(p, q) for (p, q) in zip(x.items, y.items))
        end
        if x isa MStruct || y isa MStruct
            return x isa MStruct && y isa MStruct && Set(keys(x)) == Set(keys(y)) && all(same(x[k], y[k]) for k in keys(x))
        end
        (is_num(x) && is_num(y)) || _unsup("isequal on this kind of value")
        p, q = arr(x), arr(y)
        size(p) == size(q) && all(p .== q)
    end
    all(same(a[1], y) for y in a[2:end])
end
function _linspace(a)
    _nargs(a, 2, 3, "linspace")
    n = length(a) == 3 ? as_index_int(a[3]) : 100
    lo, hi = scalar(a[1]), scalar(a[2])
    if n < 2
        return n == 1 ? hi : zeros(1, 0)
    end
    out = lo .+ (hi - lo) .* (0:n-1) ./ (n - 1)
    out = collect(out); out[end] = hi
    reshape(out, 1, n)
end
BUILTINS["linspace"] = (I, a, nout) -> _linspace(a)
bi!("logspace") do I, a, nout
    n = length(a) == 3 ? as_index_int(a[3]) : 50
    ret(10.0 .^ farr(_linspace(Any[a[1], a[2], Float64(n)])))
end
bi!("repmat") do I, a, nout
    _nargs(a, 2, 3, "repmat")
    r, c = _dims(a[2:end])
    if a[1] isa AbstractString
        r > 1 && _unsup("char matrices")
        return repeat(a[1], c)
    end
    a[1] isa Cell && _unsup("repmat of a cell array")
    length(arr(a[1])) * r * c > MAX_ELEMENTS && _unsup("array with more than $MAX_ELEMENTS elements")
    ret(repeat(arr(a[1]), r, c))
end
bi!("reshape") do I, a, nout
    x = arr(a[1])
    dims = collect(Any, a[2:end])
    length(dims) == 1 && (dims = Any[Float64(v) for v in vec(arr(dims[1]))])
    if length(dims) > 2 && any(d -> _isemptyarr(d) || scalar(d) != 1, dims[3:end])
        _unsup("arrays with more than two dimensions")
    end
    dims = dims[1:min(2, end)]
    length(dims) != 2 && merr("reshape: bad size")
    d = Any[_isemptyarr(v) ? nothing : as_index_int(v) for v in dims]
    count(isnothing, d) > 1 && merr("reshape: only one dimension can be []")
    if any(isnothing, d)
        k = findfirst(isnothing, d)
        known = d[3-k]
        (known == 0 || length(x) % known != 0) && merr("reshape: the number of elements must not change")
        d[k] = length(x) ÷ known
    end
    d[1] * d[2] != length(x) && merr("reshape: the number of elements must not change")
    ret(reshape(x, d[1], d[2]))
end
function _diagm(v::AbstractVector, k::Int)
    n = length(v) + abs(k)
    M = zeros(n, n)
    for (i, x) in enumerate(v)
        k >= 0 ? (M[i, i+k] = x) : (M[i-k, i] = x)
    end
    M
end
bi!("diag") do I, a, nout
    _nargs(a, 1, 2, "diag")
    x = farr(a[1])
    k = length(a) == 2 ? as_index_int(a[2]) : 0
    minimum(size(x)) == 1 && return ret(_diagm(vec(x), k))
    ret(reshape(collect(LinearAlgebra.diag(x, k)), :, 1))
end
BUILTINS["triu"] = (I, a, nout) -> ret(LinearAlgebra.triu(farr(a[1]), length(a) > 1 ? as_index_int(a[2]) : 0))
BUILTINS["tril"] = (I, a, nout) -> ret(LinearAlgebra.tril(farr(a[1]), length(a) > 1 ? as_index_int(a[2]) : 0))
BUILTINS["trace"] = (I, a, nout) -> Float64(LinearAlgebra.tr(farr(a[1])))
BUILTINS["kron"] = (I, a, nout) -> ret(kron(farr(a[1]), farr(a[2])))

function _square(x, name)
    size(x, 1) != size(x, 2) && merr("$name: argument must be a square matrix")
    all(isfinite, x) || _unsup("$name of a matrix with NaN or Inf")
    x
end
bi!("inv") do I, a, nout
    x = _square(farr(a[1]), "inv")
    LinearAlgebra.cond(x) > 1e15 && _unsup("inverse of a singular or badly conditioned matrix")
    ret(inv(x))
end
BUILTINS["det"] = (I, a, nout) -> isempty(arr(a[1])) ? 1.0 : Float64(LinearAlgebra.det(_square(farr(a[1]), "det")))
bi!("norm") do I, a, nout
    _nargs(a, 1, 2, "norm")
    x = farr(a[1])
    p = length(a) == 2 ? a[2] : 2.0
    isempty(x) && return 0.0
    if p isa AbstractString
        p == "fro" && return LinearAlgebra.norm(x)
        lowercase(p) == "inf" ? (p = Inf) : merr("norm: unknown norm type")
    end
    p = scalar(p)
    minimum(size(x)) == 1 && return LinearAlgebra.norm(vec(x), p)
    p in (1.0, 2.0, Inf) && return LinearAlgebra.opnorm(x, p)
    merr("norm: unsupported matrix norm")
end
bi!("chol") do I, a, nout
    (nout > 1 || length(a) != 1) && _unsup("chol with options or several outputs")
    x = _square(farr(a[1]), "chol")
    xs = isapprox(x, permutedims(x); rtol=1e-5, atol=1e-8) ? (x + permutedims(x)) / 2 : x
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(xs); check=false)
    LinearAlgebra.issuccess(C) || merr("chol: matrix must be positive definite")
    ret(Matrix(C.U))
end
bi!("dot") do I, a, nout
    x, y = farr(a[1]), farr(a[2])
    (minimum(size(x)) != 1 || minimum(size(y)) != 1 || length(x) != length(y)) && _unsup("dot of matrices")
    LinearAlgebra.dot(vec(x), vec(y))
end
bi!("cat") do I, a, nout
    d = as_index_int(a[1])
    d == 1 && return mvcat(a[2:end])
    d == 2 && return mhcat(a[2:end])
    _unsup("arrays with more than two dimensions")
end
bi!("nchoosek") do I, a, nout
    n, k = scalar(a[1]), scalar(a[2])
    Float64(round(SpecialFunctions.gamma(n + 1) / (SpecialFunctions.gamma(k + 1) * SpecialFunctions.gamma(n - k + 1))))
end
bi!("normcdf") do I, a, nout
    _nargs(a, 1, 3, "normcdf")
    mu = length(a) > 1 ? a[2] : 0.0
    sg = length(a) > 2 ? a[3] : 1.0
    z = rdivide(minus(a[1], mu), sg)
    _apply_elem(v -> 0.5 * SpecialFunctions.erfc(-v / sqrt(2)), z)
end
bi!("normpdf") do I, a, nout
    _nargs(a, 1, 3, "normpdf")
    mu = length(a) > 1 ? a[2] : 0.0
    sg = length(a) > 2 ? a[3] : 1.0
    z = farr(rdivide(minus(a[1], mu), sg))
    ret(exp.(-0.5 .* z .* z) ./ (farr(sg) .* sqrt(2pi)))
end
bi!("norminv") do I, a, nout
    _nargs(a, 1, 3, "norminv")
    mu = length(a) > 1 ? a[2] : 0.0
    sg = length(a) > 2 ? a[3] : 1.0
    plus(mu, times(sg, _apply_elem(_ndtri, a[1])))
end
bi!("sub2ind") do I, a, nout
    _nargs(a, 3, 3, "sub2ind")
    r, c = _dims(Any[a[1]])
    i, j = farr(a[2]), farr(a[3])
    (any(<(1), i) || any(>(r), i) || any(<(1), j) || any(>(c), j)) && merr("sub2ind: index out of range")
    ret((j .- 1) .* r .+ i)
end

# ---------------------------------------------------------------------- strings, cells, structs
function _str(v, name)
    if !(v isa AbstractString)
        _isemptyarr(v) && return ""
        v isa Tainted && throw(Unsupported("unknown value", "tainted"))
        # many MATLAB text functions also take cell arrays (or string objects): refused, not a MATLAB error
        (v isa Cell || v isa MStruct || v isa HostStruct) && throw(Unsupported("$name with a $(mclass(v)) argument", "feature"))
        merr("$name: a char array is required")
    end
    String(v)
end
function _strcmp(fn)
    function f(I, a, nout)
        x, y = a[1], a[2]
        n = length(a) > 2 ? as_index_int(a[3]) : 0
        one(p, q) = p isa AbstractString && q isa AbstractString && fn(p, q, n)
        (x isa Tainted || y isa Tainted) && throw(Unsupported("unknown value", "tainted"))
        if x isa Cell && y isa Cell
            x.shape != y.shape && merr("strcmp: cell arrays must have the same size")
            return ret(reshape(Bool[one(p, q) for (p, q) in zip(x.items, y.items)], x.shape))
        end
        x isa Cell && return ret(reshape(Bool[one(p, y) for p in x.items], x.shape))
        y isa Cell && return ret(reshape(Bool[one(x, q) for q in y.items], y.shape))
        one(x, y)
    end
end
BUILTINS["strcmp"] = _strcmp((p, q, n) -> p == q)
BUILTINS["strcmpi"] = _strcmp((p, q, n) -> lowercase(p) == lowercase(q))
BUILTINS["strncmp"] = _strcmp((p, q, n) -> length(p) >= n && length(q) >= n && p[1:n] == q[1:n])
BUILTINS["strncmpi"] = _strcmp((p, q, n) -> length(p) >= n && length(q) >= n && lowercase(p[1:n]) == lowercase(q[1:n]))
_cellmap(f, x, name) = x isa Cell ? Cell(Any[f(_str(v, name)) for v in x.items], x.shape) : f(_str(x, name))
BUILTINS["upper"] = BUILTINS["toupper"] = (I, a, nout) -> (_nargs(a, 1, 1, "upper"); _cellmap(uppercase, a[1], "upper"))
BUILTINS["lower"] = BUILTINS["tolower"] = (I, a, nout) -> (_nargs(a, 1, 1, "lower"); _cellmap(lowercase, a[1], "lower"))
const _WS0 = (' ', '\t', '\n', '\v', '\f', '\r', '\0')
BUILTINS["strtrim"] = (I, a, nout) -> String(strip(c -> c in _WS0, _str(a[1], "strtrim")))
BUILTINS["deblank"] = (I, a, nout) -> String(rstrip(c -> c in _WS0, _str(a[1], "deblank")))
"MATLAB's strrep: every occurrence is replaced, overlapping ones included (strrep('aaa', 'aa', 'b') is 'bb')."
function _strrep(s::String, p::String, r::String)
    isempty(p) && return s
    n = length(p)
    cs = collect(s)
    pos = [i for i in 1:(length(cs) - n + 1) if String(cs[i:i+n-1]) == p]
    isempty(pos) && return s
    out = IOBuffer()
    write(out, String(cs[1:pos[1]-1]))
    for (k, i) in enumerate(pos)
        k > 1 && i >= pos[k-1] + n && write(out, String(cs[pos[k-1]+n:i-1]))
        write(out, r)
    end
    write(out, String(cs[pos[end]+n:end]))
    String(take!(out))
end
bi!("strrep") do I, a, nout
    _nargs(a, 3, 3, "strrep")
    p, r = _str(a[2], "strrep"), _str(a[3], "strrep")
    a[1] isa Cell && return Cell(Any[_strrep(_str(x, "strrep"), p, r) for x in a[1].items], a[1].shape)
    _strrep(_str(a[1], "strrep"), p, r)
end
BUILTINS["blanks"] = (I, a, nout) -> " "^as_index_int(a[1])
BUILTINS["strcat"] = (I, a, nout) -> join(rstrip(c -> c in (' ', '\t', '\n', '\v', '\f', '\r'), _str(x, "strcat")) for x in a)
BUILTINS["int2str"] = (I, a, nout) -> _num2str(_mround(scalar(a[1], "int2str argument")))
BUILTINS["str2double"] = (I, a, nout) -> _str2double(a[1])
BUILTINS["filesep"] = (I, a, nout) -> "/"

function _str2double(s)
    if s isa Cell
        isempty(s.items) && return zeros(0, 0)
        return ret(reshape(Float64[_str2double(x) for x in s.items], s.shape))
    end
    s isa AbstractString || return NaN
    t = replace(replace(strip(s), 'd' => 'e'), 'D' => 'e')
    occursin(r"^[+-]?\d{1,3}(,\d{3})+(\.\d*)?([eE][+-]?\d+)?$", t) && (t = replace(t, "," => ""))
    if occursin(r"^([+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|[+-]?(inf|Inf|nan|NaN))$", t)
        return parse(Float64, t)
    end
    NaN
end

"C-style formatting of one number with MATLAB's spelling of Inf/NaN."
function _cfmt(spec::AbstractString, x::Float64)
    conv = spec[end]
    if isnan(x) || isinf(x)
        s = isnan(x) ? "NaN" : (x > 0 ? "Inf" : "-Inf")
        m = match(r"%([-+ 0#]*)(\d*)", spec)
        w = isempty(m.captures[2]) ? 0 : parse(Int, m.captures[2])
        return occursin('-', m.captures[1]) ? rpad(s, w) : lpad(s, w)
    end
    if conv in "di"
        if x != round(x)
            sp = occursin('.', spec) ? spec[1:end-1] * "e" : replace(spec[1:end-1], r"\.\d+" => "") * "e"
            return Printf.format(Printf.Format(sp), x)
        end
        return Printf.format(Printf.Format(spec[1:end-1] * "d"), BigInt(x))      # %.3d: at least 3 digits
    end
    if conv in "xXou"
        (x != round(x) || x < 0) && return Printf.format(Printf.Format(spec[1:end-1] * "e"), x)
        return Printf.format(Printf.Format(spec), BigInt(x))
    end
    if conv == 'c'
        (x == round(x) && 0 <= x < 0x110000 && !(0xd800 <= x < 0xe000)) ||
            return Printf.format(Printf.Format(replace(spec[1:end-1], r"\.\d+" => "") * "e"), x)   # as %s of a non-integer
        return Printf.format(Printf.Format(spec[1:end-1] * "s"), string(Char(Int(x))))
    end
    Printf.format(Printf.Format(spec), x)
end

function _num2str(x, fmt=nothing)
    x isa AbstractString && return x
    x isa Bool && return x ? "1" : "0"
    if !(x isa Real)
        a = arr(x)
        length(a) != 1 && _unsup("num2str of an array")
        x = a[1]
    end
    x = Float64(x)
    if fmt !== nothing
        fmt isa AbstractString && return String(lstrip(_sprintf(fmt, Any[x])))     # num2str trims leading blanks
        return _cfmt("%." * string(as_index_int(fmt)) * "g", x)
    end
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    x == round(x) && return string(BigInt(x))
    dgt = Int(floor(log10(abs(x)))) + 5
    dgt = min(max(dgt, 5), 16)
    String(strip(_cfmt("%" * string(dgt + 7 + (x < 0)) * "." * string(dgt) * "g", x)))
end

const _FMT_RE = r"%([-+ 0#]*)(\*|\d+)?(?:\.(\*|\d+))?(?:hh|h|ll|l|L|q|j|z|t)?([diouxXfeEgGcs%])"
const _ESC = Dict('n' => "\n", 't' => "\t", '\\' => "\\", 'r' => "\r", 'a' => "\a", 'b' => "\b", 'f' => "\f", 'v' => "\v")
_unescape(s) = replace(s, r"\\(.)" => m -> get(_ESC, m[2], m))

_shortg(x::Float64) = (isnan(x) ? "NaN" : isinf(x) ? (x > 0 ? "Inf" : "-Inf") : Printf.format(Printf.Format("%.5g"), x))

function _sprintf(fmt::AbstractString, args)
    fmt = _unescape(fmt)
    flat = Any[]
    for v in args
        if v isa AbstractString
            push!(flat, String(v))
        elseif v isa Cell || v isa MStruct || v isa Handle
            merr("sprintf: wrong type of argument")
        else
            append!(flat, Float64.(vec(arr(v))))
        end
    end
    specs = collect(eachmatch(_FMT_RE, fmt))
    nconv = count(m -> m.captures[4] != "%", specs)
    out = IOBuffer()
    k = 1
    firstpass = true
    while firstpass || (k <= length(flat) && nconv > 0)
        firstpass = false
        pos = 1
        broke = false
        for m in specs
            write(out, fmt[pos:prevind(fmt, m.offset)])
            pos = m.offset + ncodeunits(m.match)
            if m.captures[4] == "%"
                write(out, "%")
                continue
            end
            (m.captures[2] == "*" || m.captures[3] == "*") && _unsup("sprintf with '*' field width")
            if k > length(flat)
                if !isempty(flat) || k > 1
                    if k > 1
                        broke = true
                        break
                    end
                end
                continue
            end
            v = flat[k]
            if v isa String && length(v) > 1 && m.captures[4] != "s"
                # a char array met by a numeric (or %c) conversion is used one character at a time: %d of 'ab' is 97, 98
                splice!(flat, k, [string(c) for c in v])
                v = flat[k]
            end
            k += 1
            spec = "%" * m.captures[1] * something(m.captures[2], "") * (m.captures[3] === nothing ? "" : "." * m.captures[3]) * m.captures[4]
            if v isa String
                if m.captures[4] in ("s", "c")
                    write(out, Printf.format(Printf.Format(spec[1:end-1] * "s"), v))
                elseif length(v) == 1
                    write(out, _cfmt(spec, Float64(Int(v[1]))))
                else
                    write(out, v)
                end
            elseif m.captures[4] == "s"
                x = Float64(v)
                if x == round(x) && 0 <= x < 0x110000
                    write(out, Printf.format(Printf.Format(spec[1:end-1] * "s"), string(Char(Int(x)))))
                else
                    # "a text conversion for a numeric value: MATLAB overrides the conversion and uses %e"
                    write(out, _cfmt(replace(spec[1:end-1], r"\.\d+" => "") * "e", x))
                end
            else
                write(out, _cfmt(spec, Float64(v)))
            end
        end
        if broke
            break
        end
        write(out, fmt[pos:end])
    end
    String(take!(out))
end

bi!("num2str") do I, a, nout
    _nargs(a, 1, 2, "num2str")
    _num2str(a[1], length(a) == 2 ? a[2] : nothing)
end
bi!("sprintf") do I, a, nout
    isempty(a) && merr("sprintf: format missing")
    _sprintf(_str(a[1], "sprintf"), a[2:end])
end
bi!("mat2str") do I, a, nout
    x = arr(a[1])
    prec = length(a) > 1 ? as_index_int(a[2]) : 15
    one(v) = v isa Bool ? (v ? "true" : "false") : _cfmt("%.$(prec)g", Float64(v))
    length(x) == 1 && return one(x[1])
    "[" * join([join([one(v) for v in x[i, :]], " ") for i in 1:size(x, 1)], ";") * "]"
end
bi!("char") do I, a, nout
    (length(a) == 1 && a[1] isa AbstractString) && return a[1]
    (length(a) == 1 && a[1] isa Cell && length(a[1].items) == 1 && a[1].items[1] isa AbstractString) && return a[1].items[1]
    if length(a) == 1 && is_num(a[1])
        x = arr(a[1])
        size(x, 1) <= 1 && return String([Char(as_index_int(v)) for v in vec(x)])
    end
    _unsup("char matrices")
end
bi!("strsplit") do I, a, nout
    s = _str(a[1], "strsplit")
    ds = length(a) < 2 ? [" "] : a[2] isa Cell ? [_unescape(_str(x, "strsplit")) for x in a[2].items] : [_unescape(_str(a[2], "strsplit"))]
    collapse = true                       # MATLAB's default: consecutive delimiters count as one
    rest = a[3:end]
    while length(rest) >= 2
        key = lowercase(_str(rest[1], "strsplit"))
        key == "collapsedelimiters" || _unsup("strsplit option $(repr(rest[1]))")
        collapse = truth(rest[2])
        rest = rest[3:end]
    end
    isempty(rest) || merr("strsplit: options must come in name-value pairs")
    any(isempty, ds) && merr("strsplit: delimiters must not be empty")
    alt = join([replace(d, r"([\\^$.|?*+()\[\]{}])" => s"\\\1") for d in sort(ds; by=length, rev=true)], "|")
    re = Regex(collapse ? "(?:$alt)+" : "(?:$alt)")
    Cell(Any[String(x) for x in split(s, re)])
end
bi!("strjoin") do I, a, nout
    a[1] isa Cell || merr("strjoin: a cell array is required")
    d = length(a) > 1 ? _str(a[2], "strjoin") : " "
    join([_str(x, "strjoin") for x in a[1].items], _unescape(d))
end
bi!("strfind") do I, a, nout
    s, p = _str(a[1], "strfind"), _str(a[2], "strfind")
    k = isempty(p) ? Float64[] : Float64[i for i in 1:(length(s)-length(p)+1) if startswith(s[i:end], p)]
    isempty(k) ? zeros(0, 0) : ret(reshape(k, 1, :))
end
BUILTINS["contains"] = (I, a, nout) -> occursin(_str(a[2], "contains"), _str(a[1], "contains"))
BUILTINS["startsWith"] = (I, a, nout) -> startswith(_str(a[1], "startsWith"), _str(a[2], "startsWith"))
BUILTINS["endsWith"] = (I, a, nout) -> endswith(_str(a[1], "endsWith"), _str(a[2], "endsWith"))
bi!("strmatch") do I, a, nout
    _nargs(a, 2, 3, "strmatch")
    s = _str(a[1], "strmatch")
    exact = length(a) == 3 && a[3] == "exact"
    cands = a[2] isa Cell ? a[2].items : a[2] isa AbstractString ? Any[a[2]] : _unsup("strmatch on a char matrix")
    k = Float64[i for (i, c) in enumerate(cands) if c isa AbstractString && (exact ? c == s : startswith(c, s))]
    isempty(k) ? zeros(0, 1) : ret(reshape(k, :, 1))
end
bi!("ismember") do I, a, nout
    _nargs(a, 2, 2, "ismember")
    x, s = a[1], a[2]
    if x isa AbstractString && s isa Cell
        hit = [Float64(i) for (i, c) in enumerate(s.items) if c isa AbstractString && c == x]
        nout > 1 && return Any[!isempty(hit), isempty(hit) ? 0.0 : hit[1]]
        return !isempty(hit)
    end
    if x isa AbstractString && s isa AbstractString
        return length(x) != 1 ? x == s : occursin(x, s)
    end
    if x isa Cell && s isa Cell
        nout > 1 && _unsup("ismember with two outputs on cell arrays")
        return ret(reshape(Bool[any(c2 -> c2 isa AbstractString && c isa AbstractString && c == c2, s.items) for c in x.items], x.shape))
    end
    if is_num(x) && is_num(s)
        nout > 1 && _unsup("ismember with two outputs")
        S = Set(vec(farr(s)))
        return ret(map(v -> v in S, farr(x)))
    end
    _unsup("ismember on these arguments")
end
bi!("cell") do I, a, nout
    r, c = _dims(a)
    Cell(Any[zeros(0, 0) for _ in 1:(r*c)], (r, c))
end
bi!("struct") do I, a, nout
    length(a) % 2 == 1 && merr("struct: arguments must be field/value pairs")
    s = MStruct()
    for i in 1:2:length(a)
        a[i+1] isa Cell && _unsup("struct arrays")
        s[_str(a[i], "struct")] = a[i+1]
    end
    s
end
bi!("fieldnames") do I, a, nout
    v = a[1]
    names = v isa HostStruct ? field_names(v) : v isa MStruct ? (v._kind === nothing ? copy(v._keys) : _snapshot_field_names(v)) : merr("fieldnames: a struct is required")
    Cell(Any[names...], (length(names), 1))
end
_snapshot_field_names(v::MStruct) = throw(Unsupported("the complete list of fields of $(v._kind) is not known to JDyn", "results"))
function _has_field(s, name)
    s isa HostStruct && return has_field(s, name)
    if s isa MStruct && name isa AbstractString
        haskey(s, name) && return true
        s._kind !== nothing && throw(Unsupported("JDyn does not know whether $(s._kind) has a field '$name'", "results"))
        return false
    end
    false
end
bi!("isfield") do I, a, nout
    _nargs(a, 2, 2, "isfield")
    s, f = a[1], a[2]
    f isa Cell && return ret(reshape(Bool[_has_field(s, x) for x in f.items], f.shape))
    _has_field(s, f)
end
bi!("rmfield") do I, a, nout
    a[1] isa MStruct || _unsup("rmfield on this value")
    s = clone(a[1])
    for name in (a[2] isa Cell ? a[2].items : Any[a[2]])
        haskey(s, name) || merr("rmfield: no field named $(repr(name))")
        delete!(s, name)
    end
    s
end
bi!("getfield") do I, a, nout
    _nargs(a, 2, 2, "getfield")
    getfield_(I, a[1], _str(a[2], "getfield"))
end
bi!("setfield") do I, a, nout
    _nargs(a, 3, 3, "setfield")
    a[1] isa MStruct || _unsup("setfield on this value")
    s = clone(a[1])
    s[_str(a[2], "setfield")] = a[3]
    s
end
bi!("deal") do I, a, nout
    length(a) == 1 && return nout > 1 ? Any[a[1] for _ in 1:nout] : a[1]
    length(a) != max(nout, 1) && merr("deal: number of inputs and outputs must match")
    nout > 1 ? collect(Any, a) : a[1]
end

# ---------------------------------------------------------------------- display, errors, misc
function _noop(I, argnodes, sc, nout)
    # nothing is displayed, but the arguments are evaluated (they may call functions or fail)
    for a in argnodes
        (a isa Str || a isa Num || a isa ColonAll) || ev_multi(I, a, sc, 1)
    end
    nout > 0 && throw(Unsupported("output of a display or timing function", "display"))
    Any[]
end
lazy!(_noop, "disp", "display", "fprintf", "printf", "fdisp", "dprintf", "skipline", "warning", "more", "beep", "fflush",
      "rng", "format", "clc", "diary", "echo", "commandwindow", "dbstop", "profile",
      "close", "hold", "drawnow", "pause", "fclose", "print_info")

const PLOT_FUNCS = split("figure plot subplot title legend xlabel ylabel zlabel axis print saveas line bar histogram hist grid box " *
    "plot3 surf mesh colorbar suptitle sgtitle xlim ylim annotation text orient hgsave savefig fill area " *
    "scatter stairs semilogy semilogx loglog clf set get gca gcf colormap shading dyn_saveas dyn_figure " *
    "errorbar patch contour quiver fplot ezplot rplot")
function _plot(I, argnodes, sc, nout)
    ALLOW_RANDOM[] || throw(Unsupported("plotting", "plot"))
    # run by dynare() (as with the `nograph` option): drawing is skipped, the arguments are still evaluated
    for a in argnodes
        (a isa Str || a isa Num || a isa ColonAll) || ev_multi(I, a, sc, 1)
    end
    nout > 0 ? Any[1.0] : Any[]
end
lazy!(_plot, PLOT_FUNCS...)

const FILE_FUNCS = split("fopen fread fwrite fgetl fgets fscanf textscan importdata xlsread xlswrite csvread csvwrite " *
    "dlmread dlmwrite readtable writetable readmatrix writematrix delete copyfile movefile rmdir " *
    "addpath rmpath cd dir ls system unix dos exist_file " *
    "datatomfile matfile which websave urlread unzip zip tempdir tempname setenv " *
    "open winopen edit type input keyboard dbstack mfilename genpath path userpath")
_file(I, argnodes, sc, nout) = throw(Unsupported("file-system, shell or interactive functions are not available", "file"))
lazy!(_file, FILE_FUNCS...)

bi!("clock", "now", "date", "datestr", "etime", "datetime", "version", "ver", "computer",
    "ispc", "isunix", "ismac", "matlab_ver_less_than", "octave_ver_less_than", "user_has_matlab_license",
    "user_has_octave_package", "matlabroot", "OCTAVE_VERSION", "dynare_version", "feature", "gcp", "maxNumCompThreads") do I, a, nout
    nout == 0 && return Any[]
    throw(Unsupported("the value depends on the MATLAB installation or the clock", "environment"))
end

"Value returned by `isoctave`. JDyn follows the MATLAB branch by default."
const ISOCTAVE = Ref(false)
BUILTINS["isoctave"] = (I, a, nout) -> ISOCTAVE[]

bi!("error") do I, a, nout
    isempty(a) && merr("error")
    a[1] isa MStruct && throw(MatlabError(string(get(a[1], "message", "error")), string(get(a[1], "identifier", "")), true))
    fmt = a[1] isa AbstractString ? a[1] : "error"
    rest = a[2:end]
    id = ""
    if !isempty(rest) && occursin(r"^[A-Za-z]\w*(:[\w-]+)+$", fmt) && rest[1] isa AbstractString
        id = fmt
        fmt, rest = rest[1], rest[2:end]
    end
    # error(msg) alone prints msg as it is; with an identifier or arguments it is a format
    msg = try
        (length(a) > 1) ? _sprintf(fmt, rest) : fmt
    catch
        fmt
    end
    throw(MatlabError(msg, id, true))
end
bi!("rethrow") do I, a, nout
    _nargs(a, 1, 1, "rethrow")
    e = a[1]
    e isa MStruct || merr("rethrow: the argument must be an error struct")
    throw(MatlabError(string(get(e, "message", "")), string(get(e, "identifier", "")), true))
end
bi!("assert") do I, a, nout
    isempty(a) && merr("assert: argument missing")
    (length(a) >= 2 && !(a[2] isa AbstractString)) && _unsup("assert(observed, expected, tol) form")
    truth(a[1]) || throw(MatlabError(length(a) > 1 ? _sprintf(a[2], a[3:end]) : "assert (cond) failed", "", true))
    Any[]
end
bi!("exist") do I, a, nout
    name = _str(a[1], "exist")
    kind = length(a) > 1 ? a[2] : nothing
    sc = I.scope_stack[end]
    has_var(I, sc, name) && return 1.0
    (kind == "var" && !I.open_world) && return 0.0
    (kind in (nothing, "file") && haskey(dir_files(I), name) && !haskey(BUILTINS, name) && !haskey(LAZY, name)) && return 2.0
    throw(Unsupported("exist() for files or functions depends on the MATLAB path", "environment"))
end
BUILTINS["isvarname"] = (I, a, nout) -> a[1] isa AbstractString && occursin(r"^[A-Za-z]\w*$", a[1]) && !(a[1] in KEYWORDS)
bi!("nargin", "nargout") do I, a, nout
    merr("nargin/nargout used outside a function")
end

# ---------------------------------------------------------------------- solvers
function _options(a)
    (a === nothing || _isemptyarr(a)) && return Dict{String,Any}()
    a isa MStruct && return Dict{String,Any}(lowercase(k) => a[k] for k in keys(a))
    merr("solver options must be a struct created by optimset or optimoptions")
end
bi!("optimset", "optimoptions") do I, a, nout
    s = MStruct()
    a = collect(Any, a)
    if !isempty(a) && a[1] isa MStruct
        for k in keys(a[1]); s[k] = a[1][k]; end
        a = a[2:end]
    elseif !isempty(a) && length(a) % 2 == 1 && (a[1] isa AbstractString || a[1] isa Handle)
        a = a[2:end]
    end
    length(a) % 2 == 1 && merr("optimset: arguments must be name/value pairs")
    for i in 1:2:length(a)
        s[_str(a[i], "optimset")] = a[i+1]
    end
    s
end
function _opt_num(o, names...)
    for n in names
        v = get(o, lowercase(n), nothing)
        (v !== nothing && is_num(v) && length(arr(v)) == 1) && return scalar(v)
    end
    nothing
end
struct SolverNaN <: Exception end

function _callable(I, f)
    (f isa Handle || f isa AbstractString) && return mcallable(I, f)
    f isa Function && return f
    merr("a function handle or function name is required")
end

"Brent's method on [a, b] (scipy.optimize.brentq)."
function _brentq(f, a, b; xtol=2e-12, rtol=8.9e-16, maxiter=1000)
    fa, fb = f(a), f(b)
    fa == 0 && return a
    fb == 0 && return b
    (fa > 0) == (fb > 0) && throw(ArgumentError("no sign change"))
    xpre, xcur, fpre, fcur = a, b, fa, fb
    xblk, fblk, spre, scur = 0.0, 0.0, 0.0, 0.0
    for _ in 1:maxiter
        if fpre * fcur < 0
            xblk, fblk = xpre, fpre
            spre = scur = xcur - xpre
        end
        if abs(fblk) < abs(fcur)
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur
        end
        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        (fcur == 0 || abs(sbis) < delta) && return xcur
        if abs(spre) > delta && abs(fcur) < abs(fpre)
            if xpre == xblk
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            end
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta)
                spre, scur = scur, stry
            else
                spre, scur = sbis, sbis
            end
        else
            spre, scur = sbis, sbis
        end
        xpre, fpre = xcur, fcur
        xcur += abs(scur) > delta ? scur : (sbis > 0 ? delta : -delta)
        fcur = f(xcur)
    end
    throw(ErrorException("brentq: no convergence"))
end

bi!("fzero") do I, a, nout
    _nargs(a, 2, 3, "fzero")
    f = _callable(I, a[1])
    opts = _options(length(a) > 2 ? a[3] : nothing)
    tolx = _opt_num(opts, "TolX")
    function F(x)
        v = try
            scalar(f(Float64(x)), "fzero: function value")
        catch e
            (e isa Unsupported && e.kind == "complex") && return NaN
            rethrow()
        end
        Float64(v)
    end
    outs(x, fx, flag) = nout > 1 ? Any[x, fx, Float64(flag), MStruct()][1:max(nout, 1)] : x
    fail(flag) = outs(NaN, NaN, flag)
    x0 = vec(farr(a[2]))
    local lo, hi
    if length(x0) == 2
        lo, hi = x0[1], x0[2]
        flo, fhi = F(lo), F(hi)
        (isfinite(flo) && isfinite(fhi)) || merr("fzero: function values at the interval endpoints must be finite and real")
        if flo == 0 || fhi == 0
            x = flo == 0 ? lo : hi
            return outs(x, 0.0, 1)
        end
        (flo > 0) == (fhi > 0) && merr("fzero: the function values at the interval endpoints must differ in sign")
    elseif length(x0) == 1
        x = x0[1]
        fx = F(x)
        isfinite(fx) || merr("fzero: function value at starting guess must be finite and real")
        fx == 0 && return outs(x, 0.0, 1)
        dx = x != 0 ? x / 50 : 1 / 50
        lo = hi = x
        flo = fhi = fx
        while (flo > 0) == (fhi > 0)
            dx *= sqrt(2)
            lo = x - dx
            flo = F(lo)
            (!isfinite(flo) || !isfinite(lo)) && return fail(isnan(flo) || isinf(flo) ? -3 : -6)
            (flo > 0) != (fhi > 0) && break
            hi = x + dx
            fhi = F(hi)
            (!isfinite(fhi) || !isfinite(hi)) && return fail(-3)
        end
    else
        merr("fzero: second argument must be a scalar or a two-element vector")
    end
    G(x) = (v = F(x); isnan(v) && throw(SolverNaN()); v)
    x = try
        _brentq(G, lo, hi; xtol=tolx !== nothing && tolx > 0 ? tolx : 4e-16 * max(abs(lo), abs(hi), 1e-300), rtol=8.9e-16)
    catch e
        e isa SolverNaN && return fail(-3)
        (e isa ErrorException || e isa ArgumentError) && return fail(-5)
        rethrow()
    end
    outs(x, F(x), 1)
end

function _vector_fun(I, fh, shape, extra=())
    f = _callable(I, fh)
    function F(v)
        x = ret(reshape(Float64.(v), shape))
        out = try
            vec(farr(f(x, extra...)))
        catch e
            (e isa Unsupported && e.kind == "complex") && throw(SolverNaN())
            rethrow()
        end
        all(isfinite, out) || throw(SolverNaN())
        out
    end
end

"Solve F(x) = 0 from x0 (trust-region dogleg, as MINPACK's hybrd); returns (x, f)."
function _root(F, x0)
    function f!(r, x)
        r .= F(x)
    end
    function j!(J, x)                     # central differences (NLsolve's own default no longer works with NLSolversBase)
        xp = copy(x)
        for k in eachindex(x)
            h = cbrt(eps(Float64)) * max(abs(x[k]), 1.0)
            xp[k] = x[k] + h; fp = F(xp)
            xp[k] = x[k] - h; fm = F(xp)
            xp[k] = x[k]
            J[:, k] .= (fp .- fm) ./ (2h)
        end
        J
    end
    sol = NLsolve.nlsolve(f!, j!, Float64.(x0); method=:trust_region, ftol=1e-14, xtol=0.0, iterations=2000)
    x = sol.zero
    fx = F(x)
    x, fx
end

bi!("fsolve") do I, a, nout
    _nargs(a, 2, 3, "fsolve")
    opts = _options(length(a) > 2 ? a[3] : nothing)
    x0 = farr(a[2])
    F = _vector_fun(I, a[1], size(x0))
    f0 = try
        F(vec(x0))
    catch e
        e isa SolverNaN && merr("fsolve: the objective function returns undefined values at the initial point")
        rethrow()
    end
    length(f0) != length(x0) && _unsup("fsolve for a non-square system")
    tolfun = something(_opt_num(opts, "TolFun", "FunctionTolerance"), 1e-6)
    x, f, flag = try
        x, f = _root(F, vec(x0))
        x, f, (LinearAlgebra.dot(f, f) <= sqrt(tolfun) ? 1 : -2)
    catch e
        e isa SolverNaN || rethrow()
        vec(x0), f0, -2
    end
    out = Any[ret(reshape(x, size(x0))), ret(reshape(f, :, 1)), Float64(flag), MStruct()]
    nout > 1 ? out[1:max(nout, 1)] : out[1]
end
bi!("csolve") do I, a, nout
    length(a) < 5 && merr("csolve: five arguments are required")
    x0 = farr(a[2])
    crit = scalar(a[4])
    F = _vector_fun(I, a[1], size(x0), Tuple(a[6:end]))
    x, rc = try
        x, f = _root(F, vec(x0))
        x, (sum(abs, f) < crit ? 0 : 4)
    catch e
        e isa SolverNaN || rethrow()
        vec(x0), 4
    end
    out = Any[ret(reshape(x, size(x0))), Float64(rc)]
    nout > 1 ? out[1:max(nout, 1)] : out[1]
end
bi!("feval") do I, a, nout
    isempty(a) && merr("feval: function missing")
    a[1] isa Handle && return call_handle(I, a[1], collect(Any, a[2:end]), nout)
    name = _str(a[1], "feval")
    occursin(r"^[A-Za-z_]\w*$", name) || throw(Unsupported("feval of $(repr(name))", "function"))
    mcall(I, name, collect(Any, a[2:end]), nout, I.scope_stack[end])
end
bi!("str2func") do I, a, nout
    s = strip(_str(a[1], "str2func"))
    occursin(r"^[A-Za-z_]\w*$", s) && return Handle(I, nothing, nothing, nothing, String(s), I.scope_stack[end].funcs)
    body, _ = mparse(String(s))
    (length(body) == 1 && body[1] isa ExprStmt && body[1].e isa AnonFunc) && return ev(I, body[1].e, Scope())
    merr("str2func: invalid function string")
end
bi!("func2str") do I, a, nout
    (a[1] isa Handle && a[1].name !== nothing) && return a[1].name
    _unsup("func2str of an anonymous function")
end
const _EVAL_CACHE = Dict{String,Any}()
bi!("eval") do I, a, nout
    _nargs(a, 1, 2, "eval")
    code = a[1]
    (code isa Tainted || !(code isa AbstractString)) && throw(Unsupported("eval of a string that cannot be determined", "eval"))
    sc = I.scope_stack[end]
    cached = get(_EVAL_CACHE, code, nothing)
    parsed = try
        cached !== nothing ? cached : mparse(String(code))
    catch e
        e isa ParseError || rethrow()
        length(a) == 2 && return run_eval(I, a[2], sc, nout)
        throw(Unsupported("eval: cannot parse $(repr(first(code, 60))): $(e.msg)", "eval"))
    end
    body, funcs = parsed
    (cached === nothing && length(_EVAL_CACHE) < 5000) && (_EVAL_CACHE[code] = parsed)
    (!isempty(funcs) || any(s -> s isa Unparsed, body)) && throw(Unsupported("eval: cannot parse $(repr(first(code, 60)))", "eval"))
    if length(a) == 2
        try
            return run_parsed(I, body, sc, nout)
        catch e
            e isa MatlabError || rethrow()
            return run_eval(I, a[2], sc, nout)
        end
    end
    run_parsed(I, body, sc, nout)
end
bi!("evalin", "assignin", "evalc", "source", "dynare", "inputname", "builtin", "cellfun", "arrayfun", "structfun", "str2num") do I, a, nout
    throw(Unsupported("this function manipulates workspaces or runs code that cannot be analysed", "eval"))
end

# ---------------------------------------------------------------------- Dynare interface
bi!("set_param_value") do I, a, nout
    _nargs(a, 2, 2, "set_param_value")
    host_set_param(need_host(I, "set_param_value"), _str(a[1], "set_param_value"), a[2])
    I.base.vars[a[1]] = a[2]          # set_param_value also does assignin('base', name, value)
    Any[]
end
bi!("get_param_by_name") do I, a, nout
    _nargs(a, 1, 1, "get_param_by_name")
    host_get_param(need_host(I, "get_param_by_name"), _str(a[1], "get_param_by_name"))
end
bi!("set_shock_stderr_value") do I, a, nout
    _nargs(a, 2, 2, "set_shock_stderr_value")
    host_set_shock_stderr(need_host(I, "set_shock_stderr_value"), _str(a[1], "set_shock_stderr_value"), a[2])
    Any[]
end
bi!("send_endogenous_variables_to_workspace", "send_exogenous_variables_to_workspace", "send_irfs_to_workspace") do I, a, nout
    throw(Unsupported("simulation results are not available while the model is loaded", "results"))
end

"Functions known to modify the model or the workspace behind the interpreter's back."
const MUTATORS = Set(["evalin", "assignin", "run", "source", "dynare", "set_shock_skew_value", "set_all_parameters",
    "set_parameters_locally", "load_params_and_steady_state", "dynare_estimation", "dynare_estimation_1",
    "set_dynare_random_generator_state", "initvalf", "histvalf", "global_initialization", "clearvars",
    "set_local_param_value", "set_shock_stderr_value", "set_param_value", "eval", "evalc", "builtin",
    "str2num", "cellfun", "arrayfun", "structfun", "feval", "set_measurement_errors", "set_exogenous_variables_for_simulation",
    "sur", "surgibbs", "dyn_ols", "olseqs", "pooled_ols", "pooled_fgls", "olsgibbs", "dynare_estimation_init",
    "set_prior", "osr1", "simulated_moments_estimation", "prior_posterior_statistics",
    "discretionary_policy", "draw_init_state_from_smoother", "dsge_conditional_likelihood_1", "dsge_likelihood",
    "evaluate_smoother", "non_linear_dsge_likelihood", "prior", "set_param_value_locally", "set_parameters",
    "set_shock_stderr_value_locally", "shock_decomposition", "solve_model_for_online_filter", "write_latex_prior_table"])
const MUTATOR_DOTTED = Set(["backward_model.initialize", "cli.evaluate.smoother", "estimate.nls", "gsa.set_shocks_param",
    "mom.run", "occbin.IVF_posterior", "occbin.kalman_update_algo_1", "occbin.kalman_update_algo_3",
    "occbin.kalman_update_engine", "occbin.match_function", "occbin.setup", "osr.run",
    "pac.estimate.iterative_ols", "pac.estimate.nls", "pac.mce.parameters", "pac.update.expectation",
    "pac.update.parameters", "pac.update.equation", "var_expectation.update",
    "var_expectation.update_parameters", "gsa.run", "gsa.map_calibration", "gsa.stability_mapping"])
const MUTATOR_PACKAGES = Set([split(d, ".")[1] for d in MUTATOR_DOTTED])
"Dynare functions whose M_/M_.params output only reflects what the steady-state file changed."
const SOLVE_FUNCS = Set(["resol", "evaluate_steady_state", "dynare_resolve", "stoch_simul", "steady_", "resid", "check",
    "evaluate_steady_state_file", "dyn_first_order_solver", "stochastic_solvers", "compute_decision_rules"])

"'pac.estimate.nls' for the syntax tree of that name, else nothing."
function _dotted(n)
    parts = String[]
    while n isa FieldE && n.name !== nothing
        push!(parts, n.name)
        n = n.base
    end
    n isa Id || return nothing
    join(vcat([n.name], reverse(parts)), ".")
end
