# MATLAB values, arithmetic and indexing for the interpreter (port of pydyn/matlab.py, first part).
#
# Representation: real double scalars are Float64, logical scalars Bool, arrays Matrix{Float64} or
# Matrix{Bool} (always 2-D; a 1x1 result is returned as a scalar, see `ret`), char row vectors
# String, cell arrays `Cell`, scalar structs `MStruct`, function handles `Handle`.

struct MatlabError <: Exception
    msg::String
    id::String          # the identifier of error('pkg:id', ...), else ""
    user::Bool          # raised by the program's own error()/assert() (not by the interpreter's runtime checks)
end
MatlabError(msg::AbstractString, id::AbstractString) = MatlabError(String(msg), String(id), false)
MatlabError(msg::AbstractString) = MatlabError(String(msg), "", false)
Base.showerror(io::IO, e::MatlabError) = print(io, "MatlabError: ", e.msg)

"The interpreter cannot evaluate this code; `kind` classifies the reason."
struct Unsupported <: Exception
    msg::String
    kind::String
end
Unsupported(msg::AbstractString) = Unsupported(String(msg), "feature")
Base.showerror(io::IO, e::Unsupported) = print(io, "Unsupported: ", e.msg)

struct ParseError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ParseError) = print(io, "ParseError: ", e.msg)

"Struct-like views of the model (M_, options_, oo_) provided by the Dynare session."
abstract type HostStruct end

"Evaluation steps allowed per interpreter (endless-loop guard); a Ref so that tests can lower it."
const MAX_STEPS = Ref(20_000_000)
const MAX_ELEMENTS = 20_000_000

"""
Julia exceptions that an operation of the interpreter can raise on input MATLAB rejects with an error of its own
(`trace(ones(2,3))`, an integer conversion of a huge value, ...): they become MatlabErrors, which a MATLAB
try/catch can catch. JDyn's own error types and the interpreter's control signals are left alone.
"""
const _JULIA_ERRORS = Union{DimensionMismatch,InexactError,BoundsError,ArgumentError,DomainError,OverflowError,
                            KeyError,DivideError,LinearAlgebra.SingularException,LinearAlgebra.LAPACKException,
                            StringIndexError,TypeError,MethodError,UndefRefError,InvalidStateException}
matlab_error(e::_JULIA_ERRORS) = MatlabError(first(split(sprint(showerror, e), '\n')))
matlab_error(e) = e

_cplx() = throw(Unsupported("the result is a complex number in MATLAB; complex numbers are not supported", "complex"))
_unsup(msg, kind="feature") = throw(Unsupported(msg, kind))
merr(msg) = throw(MatlabError(msg))

"A cell array (2-D, stored column-major)."
struct Cell
    items::Vector{Any}
    shape::Tuple{Int,Int}
end
Cell(items::AbstractVector) = Cell(collect(Any, items), isempty(items) ? (0, 0) : (1, length(items)))

"A scalar struct with ordered fields. `kind` is set for snapshots of M_/options_ (see `has_field`)."
mutable struct MStruct
    _keys::Vector{String}
    _vals::Dict{String,Any}
    _kind::Union{Nothing,String}
end
MStruct() = MStruct(String[], Dict{String,Any}(), nothing)
function MStruct(pairs::AbstractVector)
    s = MStruct()
    for (k, v) in pairs
        s[k] = v
    end
    s
end
Base.haskey(s::MStruct, k::AbstractString) = haskey(s._vals, k)
Base.getindex(s::MStruct, k::AbstractString) = s._vals[k]
Base.get(s::MStruct, k::AbstractString, d) = get(s._vals, k, d)
function Base.setindex!(s::MStruct, v, k::AbstractString)
    haskey(s._vals, k) || push!(s._keys, String(k))
    s._vals[String(k)] = v
end
function Base.delete!(s::MStruct, k::AbstractString)
    if haskey(s._vals, k)
        delete!(s._vals, k)
        filter!(!=(k), s._keys)
    end
    s
end
Base.keys(s::MStruct) = s._keys
Base.length(s::MStruct) = length(s._keys)
clone(s::MStruct) = MStruct(copy(s._keys), copy(s._vals), s._kind)

# Julia code reads and writes MATLAB structs with MATLAB's syntax: oo_.dr.ghx, M_.params, s.f = 1.
# (MATLAB field names start with a letter, so they cannot collide with the internal _keys/_vals/_kind.)
@inline function Base.getproperty(s::MStruct, f::Symbol)
    (f === :_keys || f === :_vals || f === :_kind) && return getfield(s, f)
    v = get(getfield(s, :_vals), String(f), _NOFIELD)
    v === _NOFIELD && throw(ArgumentError("reference to non-existent field '$f'"))
    v
end
function Base.setproperty!(s::MStruct, f::Symbol, v)
    (f === :_keys || f === :_vals || f === :_kind) && return setfield!(s, f, v)
    s[String(f)] = v
end
Base.propertynames(s::MStruct, private::Bool=false) = Tuple(Symbol.(getfield(s, :_keys)))
Base.hasproperty(s::MStruct, f::Symbol) = haskey(getfield(s, :_vals), String(f))
Base.getindex(s::MStruct, f::Symbol) = getproperty(s, f)
struct _NoField end
const _NOFIELD = _NoField()
function Base.show(io::IO, ::MIME"text/plain", s::MStruct)
    ks = getfield(s, :_keys)
    print(io, "struct with ", length(ks), " field", length(ks) == 1 ? "" : "s")
    isempty(ks) || print(io, ":")
    w = isempty(ks) ? 0 : maximum(length, ks)
    for k in ks
        v = getfield(s, :_vals)[k]
        d = v isa AbstractMatrix ? "$(size(v, 1))x$(size(v, 2)) $(eltype(v) == Bool ? "logical" : "double")" :
            v isa MStruct ? "[1x1 struct]" : v isa Cell ? "{$(v.shape[1])x$(v.shape[2]) cell}" :
            v isa AbstractString ? repr(v) : sprint(show, v)
        print(io, "\n  ", lpad(k, w), ": ", d)
    end
end
Base.show(io::IO, s::MStruct) = print(io, "struct(", join(getfield(s, :_keys), ", "), ")")

# cells index and iterate like Julia vectors (column-major), e.g. M_.endo_names[1]
Base.getindex(c::Cell, i::Integer) = c.items[i]
Base.getindex(c::Cell, i::Integer, j::Integer) = c.items[(j - 1) * c.shape[1] + i]
Base.length(c::Cell) = length(c.items)
Base.iterate(c::Cell, st=1) = st > length(c.items) ? nothing : (c.items[st], st + 1)
Base.size(c::Cell) = c.shape
Base.collect(c::Cell) = copy(c.items)
Base.show(io::IO, c::Cell) = print(io, "{", join(map(x -> x isa AbstractString ? repr(x) : sprint(show, x), c.items), ", "), "}")

"A function handle: anonymous (`params`/`body`/`env`) or named (`name`)."
struct Handle
    interp::Any
    params::Union{Nothing,Vector{String}}
    body::Any
    env::Union{Nothing,Dict{String,Any}}
    name::Union{Nothing,String}
    files::Any
end

"Placeholder for a variable whose value is unknown because its statement was not executed."
struct Tainted
    why::String
end

struct ColonT end
const COLON = ColonT()
Base.show(io::IO, ::ColonT) = print(io, ":")

const MScalar = Union{Float64,Bool}
const MArray = Union{Matrix{Float64},Matrix{Bool}}

is_num(v) = v isa Real || v isa AbstractMatrix{<:Real}
isscal(v) = v isa Real

"2-D array view of a numeric value."
function arr(v)
    v isa Matrix{Float64} && return v
    v isa Matrix{Bool} && return v
    v isa AbstractMatrix{Bool} && return Matrix{Bool}(v)
    v isa AbstractMatrix{<:Real} && return Matrix{Float64}(v)
    v isa Bool && return fill(v, 1, 1)
    v isa Real && return fill(Float64(v), 1, 1)
    v isa AbstractString && _unsup("arithmetic or numeric use of a char array")
    v isa Tainted && _unsup("value depends on a statement that was not executed: $(v.why)", "tainted")
    merr("a $(mclass(v)) was used where a number is required")
end

function farr(v)
    a = arr(v)
    a isa Matrix{Float64} ? a : Matrix{Float64}(a)
end

"Normalise an array result: 1x1 becomes a scalar."
function ret(a::AbstractArray)
    a = ndims(a) == 2 ? a : reshape(a, size(a, 1), :)
    if size(a) == (1, 1)
        x = a[1, 1]
        return x isa Bool ? x : Float64(x)
    end
    eltype(a) == Bool ? Matrix{Bool}(a) : (a isa Matrix{Float64} ? a : Matrix{Float64}(a))
end
ret(x::Real) = x isa Bool ? x : Float64(x)

function scalar(v, what="value")
    v isa Float64 && return v
    v isa Real && return Float64(v)
    a = arr(v)
    length(a) == 1 || merr("$what must be a scalar")
    Float64(a[1])
end

function mclass(v)
    v isa AbstractString && return "char"
    v isa Cell && return "cell"
    (v isa MStruct || v isa HostStruct) && return "struct"
    v isa Handle && return "function_handle"
    (v isa Bool || v isa AbstractMatrix{Bool}) && return "logical"
    "double"
end

function shape_of(v)
    v isa AbstractMatrix && return size(v)
    v isa AbstractString && return isempty(v) ? (0, 0) : (1, length(v))
    v isa Cell && return v.shape
    v isa Tainted && arr(v)
    (1, 1)
end

"MATLAB truth value of an if/while condition."
function truth(v)
    if v isa Real
        v != v && merr("NaN cannot be converted to a logical value")
        return v != 0
    end
    if v isa AbstractString
        return !isempty(v) && all(c -> c != '\0', v)
    end
    a = arr(v)
    (eltype(a) != Bool && any(isnan, a)) && merr("NaN cannot be converted to a logical value")
    length(a) > 0 && all(!=(0), a)
end

function as_index_int(x, what="index")
    x = scalar(x, what)
    (x != floor(x) || !isfinite(x)) && merr("$what must be an integer")
    Int(x)
end

# ---------------------------------------------------------------------- arithmetic
function _bcast(a, b)
    if size(a) != size(b)
        for (p, q) in zip(size(a), size(b))
            (p != q && p != 1 && q != 1) && merr("nonconformant arguments ($(size(a,1))x$(size(a,2)) and $(size(b,1))x$(size(b,2)))")
        end
    end
end

_fl(x::Real) = Float64(x)

function _ew(fn, a, b)
    if a isa Real && b isa Real
        return fn(Float64(a), Float64(b))
    end
    x, y = farr(a), farr(b)
    _bcast(x, y)
    ret(fn.(x, y))
end

plus(a, b) = _ew(+, a, b)
minus(a, b) = _ew(-, a, b)
times(a, b) = _ew(*, a, b)
rdivide(a, b) = _ew(/, a, b)
ldivide(a, b) = _ew(/, b, a)

"x^y with MATLAB's real-number semantics; a complex result is refused."
function _pow(x::Float64, y::Float64)
    (isnan(x) || isnan(y)) && return (x == 1.0 ? 1.0 : (y == 0.0 ? 1.0 : NaN))
    if x < 0 && isfinite(y) && y != round(y)
        _cplx()
    end
    if x < 0 && isinf(y)
        return abs(x) == 1 ? 1.0 : (abs(x) > 1 ? (y > 0 ? Inf : 0.0) : (y > 0 ? 0.0 : Inf))
    end
    x^y
end
power(a, b) = _ew(_pow, a, b)

function mtimes(a, b)
    (a isa Real || b isa Real) && return times(a, b)
    x, y = farr(a), farr(b)
    (length(x) == 1 || length(y) == 1) && return ret(x .* y)
    size(x, 2) != size(y, 1) && merr("operator *: nonconformant arguments ($(size(x,1))x$(size(x,2)) and $(size(y,1))x$(size(y,2)))")
    ret(x * y)
end

function _solve(A, B)
    F = lu(A; check=false)
    issuccess(F) || _unsup("linear system with a singular matrix")
    F \ B
end

function mrdivide(a, b)
    b isa Real && return rdivide(a, b)
    x, y = farr(a), farr(b)
    length(y) == 1 && return ret(x ./ y[1])
    size(x, 2) != size(y, 2) && merr("operator /: nonconformant arguments")
    size(y, 1) != size(y, 2) && _unsup("least-squares solution with a non-square matrix")
    ret(permutedims(_solve(permutedims(y), permutedims(x))))
end

function mldivide(a, b)
    a isa Real && return ldivide(a, b)
    x, y = farr(a), farr(b)
    length(x) == 1 && return ret(y ./ x[1])
    size(x, 1) != size(y, 1) && merr("operator \\: nonconformant arguments")
    size(x, 1) != size(x, 2) && _unsup("least-squares solution with a non-square matrix")
    ret(_solve(x, y))
end

function mpower(a, b)
    (a isa Real && b isa Real) && return power(a, b)
    x, y = farr(a), farr(b)
    (length(x) == 1 && length(y) == 1) && return _pow(x[1], y[1])
    if length(y) == 1 && size(x, 1) == size(x, 2)
        p = y[1]
        if p == round(p) && isfinite(p)
            if p < 0
                F = lu(x; check=false)
                issuccess(F) || _unsup("negative power of a singular matrix")
                return ret(inv(F)^Int(-p))
            end
            return ret(x^Int(p))
        end
        _unsup("non-integer power of a matrix")
    end
    if length(x) == 1 && size(y, 1) == size(y, 2)           # scalar ^ square matrix = expm(log(s) * A)
        s0 = x[1]
        isempty(y) && return zeros(0, 0)
        (isnan(s0) || any(!isfinite, y)) && return ret(fill(NaN, size(y)))
        s0 < 0 && _cplx()
        s0 == 0 && _unsup("0 ^ matrix")
        return ret(exp(log(s0) .* y))
    end
    merr("operator ^: one argument must be a square matrix and the other a scalar; use .^ for element-wise power")
end

function uminus(a)
    a isa Real && return -Float64(a)
    ret(-farr(a))
end

function lnot(a)
    if a isa Real
        a != a && merr("NaN cannot be converted to a logical value")
        return a == 0
    end
    x = arr(a)
    (eltype(x) != Bool && any(isnan, x)) && merr("NaN cannot be converted to a logical value")
    ret(x .== 0)
end

function mtranspose(a)
    a isa AbstractMatrix && return ret(permutedims(a))
    if a isa AbstractString && length(a) > 1
        _unsup("transpose of a char array")
    end
    if a isa Cell
        (1 in a.shape || isempty(a.items)) || _unsup("transpose of a cell matrix")
        return Cell(a.items, (a.shape[2], a.shape[1]))
    end
    a
end

_chars(s::AbstractString) = Float64[Int(c) for c in s]

function _cmp(fn)
    function f(a, b)
        (a isa Real && b isa Real) && return fn(a, b)
        if a isa AbstractString || b isa AbstractString
            if a isa AbstractString && b isa AbstractString && (length(a) == length(b) || length(a) == 1 || length(b) == 1)
                x = reshape(_chars(a), 1, :)
                y = reshape(_chars(b), 1, :)
                return ret(fn.(x, y))
            end
            _unsup("comparison of char arrays with ==; use strcmp")
        end
        x, y = arr(a), arr(b)
        _bcast(x, y)
        ret(fn.(x, y))
    end
end
const mlt = _cmp(<)
const mgt = _cmp(>)
const mle = _cmp(<=)
const mge = _cmp(>=)
const meq = _cmp(==)
const mne = _cmp(!=)

function _logical(a)
    x = arr(a)
    (eltype(x) != Bool && any(isnan, x)) && merr("NaN cannot be converted to a logical value")
    x .!= 0
end
function el_and(a, b)
    x, y = _logical(a), _logical(b)
    _bcast(x, y)
    ret(x .& y)
end
function el_or(a, b)
    x, y = _logical(a), _logical(b)
    _bcast(x, y)
    ret(x .| y)
end

"`a:b` or `a:b:c` (the second argument is the step when three are given)."
function colon(a, b, c=nothing)
    start = scalar(a, "range bound")
    if c === nothing
        step, stop = 1.0, scalar(b, "range bound")
    else
        step, stop = scalar(b, "range step"), scalar(c, "range bound")
    end
    if step == 0 || (step > 0 && start > stop) || (step < 0 && start < stop) || !isfinite(step)
        return zeros(1, 0)
    end
    (!isfinite(stop - start) || (stop - start) / step > MAX_ELEMENTS) && _unsup("range with more than $MAX_ELEMENTS elements")
    n = Int(floor((stop - start) / step * (1 + 4 * eps(Float64)) + 1e-300)) + 1
    reshape(start .+ step .* (0:n-1), 1, n)
end

_isempty_val(v) = (v isa AbstractMatrix && isempty(v)) || (v isa AbstractString && v == "")

_is00(v) = (v isa AbstractMatrix && size(v) == (0, 0)) || (v isa AbstractString && v == "")

"""
Which operands of a numeric concatenation along `dim` (1 vertical, 2 horizontal) take part, and the dimension
error if they do not fit. As Octave's dim_vector::hvcat: a 0x0 operand is skipped; so is a 1x0 or 0x1 one that
does not fit (`[zeros(1,0); 1 2 3]`); anything else must agree in the other dimension (`[1 2 3, zeros(2,0)]` is an error).
"""
function _cat_parts(parts, dim, what)
    other = 3 - dim
    keep = Matrix[]
    cur = nothing                                 # running size of the result
    for p in parts
        d = size(p)
        d == (0, 0) && continue
        if cur === nothing
            cur = d; push!(keep, p); continue
        end
        if cur[other] == d[other]
            cur = dim == 1 ? (cur[1] + d[1], cur[2]) : (cur[1], cur[2] + d[2])
            push!(keep, p)
        elseif d[1] + d[2] == 1                     # 1x0 or 0x1: skipped
            continue
        elseif cur[1] + cur[2] == 1                 # the result so far is 1x0 or 0x1: replaced
            cur = d; empty!(keep); push!(keep, p)
        else
            merr("$what dimensions mismatch ($(cur[1])x$(cur[2]) vs $(d[1])x$(d[2]))")
        end
    end
    keep, cur === nothing ? (0, 0) : cur
end

function mhcat(vals)
    any(v -> v isa MObject, vals) && return obj_cat(vals, 2)
    nz = [v for v in vals if !_is00(v)]
    isempty(nz) && return zeros(0, 0)
    length(nz) == 1 && return nz[1]
    textual = any(v -> v isa AbstractString || v isa Cell, nz)
    if !textual
        parts = [arr(v) for v in nz]
        keep, dims = _cat_parts(parts, 2, "horizontal")
        isempty(keep) && return zeros(0, 0)
        res = all(p -> eltype(p) == Bool, keep) ? Matrix{Bool}(reduce(hcat, keep)) :
              Matrix{Float64}(reduce(hcat, [Matrix{Float64}(p) for p in keep]))
        size(res) == dims || (res = zeros(eltype(res), dims...))
        return ret(res)
    end
    vals = [v for v in vals if !_isempty_val(v)]
    isempty(vals) && return zeros(0, 0)
    if any(v -> v isa AbstractString, vals)
        all(v -> v isa AbstractString, vals) && return join(vals)
        _unsup("concatenation of char arrays with numbers")
    end
    items = Any[]
    for v in vals
        if v isa Cell
            v.shape[1] > 1 && _unsup("horizontal concatenation of cell columns")
            append!(items, v.items)
        else
            push!(items, v)
        end
    end
    Cell(items, (1, length(items)))
end

function mvcat(vals)
    any(v -> v isa MObject, vals) && return obj_cat(vals, 1)
    nz = [v for v in vals if !_is00(v)]
    isempty(nz) && return zeros(0, 0)
    length(nz) == 1 && return nz[1]
    textual = any(v -> v isa AbstractString || v isa Cell, nz)
    if !textual
        parts = [arr(v) for v in nz]
        keep, dims = _cat_parts(parts, 1, "vertical")
        isempty(keep) && return zeros(0, 0)
        res = all(p -> eltype(p) == Bool, keep) ? Matrix{Bool}(reduce(vcat, keep)) :
              Matrix{Float64}(reduce(vcat, [Matrix{Float64}(p) for p in keep]))
        size(res) == dims || (res = zeros(eltype(res), dims...))
        return ret(res)
    end
    vals = [v for v in vals if !_isempty_val(v)]
    isempty(vals) && return zeros(0, 0)
    length(vals) == 1 && return vals[1]
    any(v -> v isa AbstractString, vals) && _unsup("char matrices (vertical concatenation of char arrays)")
    items = Any[]
    for v in vals
        if v isa Cell
            v.shape[2] > 1 && _unsup("vertical concatenation of cell rows")
            append!(items, v.items)
        else
            push!(items, v)
        end
    end
    Cell(items, (length(items), 1))
end

# ---------------------------------------------------------------------- indexing
"One MATLAB subscript as 1-based positions along a dimension of length `n`."
function _lin_index(ix, n::Int; grow::Bool=false)::Vector{Int}
    ix === COLON && return collect(1:n)
    ix isa AbstractString && _unsup("char array used as an index")
    a = arr(ix)
    if eltype(a) == Bool
        f = vec(a)
        if length(f) > n && any(@view f[n+1:end])
            grow || merr("logical index out of bounds")
        end
        return findall(f)
    end
    f = vec(a)
    if !isempty(f) && (any(x -> x != floor(x), f) || any(<(1), f) || !all(isfinite, f))
        merr("subscript indices must be positive integers or logicals")
    end
    k = Int.(f)
    if !grow && !isempty(k) && maximum(k) > n
        merr("index ($(maximum(k))) out of bounds (dimension has $n element(s))")
    end
    k
end

"`v(args)` for arrays, strings and cells (returns a cell for cells)."
function mindex(v, args)
    v isa Tainted && arr(v)
    v isa Array{Float64,3} && return index3(v, args)         # 3-D arrays: only produced by src/pac.jl (AR matrices)
    if v isa Cell
        items, shp = v.items, v.shape
        a = isempty(items) ? zeros(Int, shp) : reshape(collect(1:length(items)), shp)
        sub = _index_arr(a, args; keep=true)
        return Cell(Any[items[i] for i in vec(sub)], size(sub))
    end
    if v isa AbstractString
        a = isempty(v) ? zeros(0, 0) : reshape(_chars(v), 1, :)
        sub = _index_arr(a, args; keep=true)
        size(sub, 1) > 1 && _unsup("char matrices")
        return String([Char(Int(c)) for c in vec(sub)])
    end
    if v isa MStruct || v isa Handle || v isa HostStruct
        all(x -> x isa Real && x == 1, args) && return v
        _unsup("indexing a struct or function handle")
    end
    ret(_index_arr(arr(v), args))
end

function _index_arr(a::AbstractMatrix, args; keep::Bool=false)
    n = length(args)
    n == 0 && return a
    if n == 1
        ix = args[1]
        if ix isa Float64 || (ix isa Real && !(ix isa Bool))
            (ix != round(ix) || ix < 1) && merr("subscript indices must be positive integers or logicals")
            k = Int(ix)
            k > length(a) && merr("index ($k) out of bounds (array has $(length(a)) element(s))")
            return reshape([a[k]], 1, 1)
        end
        ix === false && return zeros(eltype(a), 0, 0)         # A(false) is 0x0
        flat = vec(a)
        k = _lin_index(ix, length(flat))
        out = flat[k]
        ix === COLON && return reshape(out, :, 1)
        ia = arr(ix)
        if eltype(ia) == Bool
            # a logical subscript is find(mask): a vector source keeps its orientation, otherwise the result is a
            # row for a row mask and a column for a column or matrix mask
            if size(a, 1) == 1 && size(a, 2) != 1
                return reshape(out, 1, :)
            elseif size(a, 2) == 1 && size(a, 1) != 1
                return reshape(out, :, 1)
            end
            return (size(ia, 1) == 1 && size(ia, 2) != 1) ? reshape(out, 1, :) : reshape(out, :, 1)
        end
        if minimum(size(ia)) != 1 || (size(a, 1) != 1 && size(a, 2) != 1)
            return reshape(out, size(ia))
        end
        return size(a, 1) == 1 ? reshape(out, 1, :) : reshape(out, :, 1)
    end
    if n == 2
        r = _lin_index(args[1], size(a, 1))
        c = _lin_index(args[2], size(a, 2))
        return a[r, c]
    end
    all(x -> x isa Real && x == 1, args[3:end]) && return _index_arr(a, args[1:2]; keep=keep)
    _unsup("arrays with more than two dimensions")
end

function index_brace(v, args)
    v isa Tainted && arr(v)
    v isa Cell || merr("'{}' indexing applied to a $(mclass(v))")
    sub = mindex(v, args)
    length(sub.items) == 1 || _unsup("comma-separated list from a cell array (c{:})")
    sub.items[1]
end

"`v(args) = rhs`; `v` may be `nothing` (undefined). Returns the new value."
function assign_index(v, args, rhs)
    v isa Tainted && arr(v)
    if v isa Cell || rhs isa Cell
        if rhs isa Cell && (v === nothing || v isa Cell) && length(args) == 1 && length(rhs.items) == 1
            return assign_brace(v, args, rhs.items[1])
        end
        _unsup("'()' assignment with cell arrays")
    end
    if v isa AbstractString || rhs isa AbstractString || rhs isa MStruct || rhs isa Handle
        _unsup("indexed assignment with char arrays, structs or handles")
    end
    # an undefined variable takes the class of the right-hand side (`clear x; x(3) = true` is logical); `x = []`
    # is double
    a = v === nothing ? (rhs isa Bool || rhs isa AbstractMatrix{Bool} ? zeros(Bool, 0, 0) : zeros(0, 0)) : arr(v)
    r = arr(rhs)
    deleting = isempty(r) && rhs isa AbstractMatrix
    if !deleting
        if eltype(a) == Bool && eltype(r) != Bool
            # the right-hand side is converted to the class of the left one: logical stays logical (NaN is an error)
            any(isnan, r) && merr("invalid conversion from NaN to logical")
            r = Matrix{Bool}(r .!= 0)
        elseif eltype(a) != Bool && eltype(r) == Bool
            r = Matrix{Float64}(r)
        end
    end
    n = length(args)
    if n == 1
        k = _lin_index(args[1], length(a); grow=true)
        if deleting
            any(>(length(a)), k) && merr("matrix index is out of range for deletion")
            args[1] === COLON && return zeros(eltype(a), 0, 0)
            isempty(k) && return a
            keep = trues(length(a))
            keep[k] .= false
            out = vec(a)[keep]
            # a vector keeps its orientation; deleting elements of a matrix gives a row (Octave 8.4 and MATLAB)
            return ret((size(a, 1) == 1 || minimum(size(a)) > 1) ? reshape(out, 1, :) : reshape(out, :, 1))
        end
        need = isempty(k) ? 0 : maximum(k)
        if need > length(a)
            T = eltype(a)
            if length(a) == 0
                size(a, 1) > 1 && merr("A(I) = X: cannot resize an empty array along an ambiguous dimension")
                a = zeros(T, 1, need)
            elseif size(a, 1) == 1
                a = hcat(a, zeros(T, 1, need - size(a, 2)))
            elseif size(a, 2) == 1
                a = vcat(a, zeros(T, need - size(a, 1), 1))
            else
                merr("a matrix cannot be resized with a linear index")
            end
        else
            a = copy(a)
        end
        (length(r) != 1 && length(r) != length(k)) && merr("A(I) = X: X must have the same size as I ($(length(k)) element(s) on the left, $(length(r)) on the right)")
        if length(r) == 1
            a[k] .= r[1]
        else
            a[k] .= vec(r)
        end
        return ret(a)
    end
    if n == 2
        ri = args[1] === COLON ? nothing : _lin_index(args[1], size(a, 1); grow=true)
        ci = args[2] === COLON ? nothing : _lin_index(args[2], size(a, 2); grow=true)
        if deleting
            (ci !== nothing && any(>(size(a, 2)), ci)) && merr("matrix index is out of range for deletion")
            (ri !== nothing && any(>(size(a, 1)), ri)) && merr("matrix index is out of range for deletion")
            # an empty subscript deletes nothing; two colons delete every row (0xN, as Octave and MATLAB);
            # otherwise one subscript must span its whole dimension and the other says what goes
            ((ri !== nothing && isempty(ri)) || (ci !== nothing && isempty(ci))) && return a
            (ri === nothing && ci === nothing) && return a[Int[], :]
            # (a subscript that spans its dimension, such as 1:end, counts as a colon in MATLAB; Octave refuses it)
            rfull = ri === nothing || length(unique(ri)) == size(a, 1)
            cfull = ci === nothing || length(unique(ci)) == size(a, 2)
            if ri === nothing || (rfull && !cfull)
                keep = trues(size(a, 2)); keep[ci] .= false
                return ret(a[:, keep])
            end
            if ci === nothing || (cfull && !rfull)
                keep = trues(size(a, 1)); keep[ri] .= false
                return ret(a[keep, :])
            end
            merr("a null assignment can only have one non-colon index")
        end
        # colons take the extent of the array; only a 0x0 array lets them take it from the right-hand side
        # (Octave's zero_dims_inquire: the non-singleton extents of the right-hand side, in order)
        if ri === nothing || ci === nothing
            if size(a) == (0, 0)
                if ri === nothing && ci === nothing
                    ri = collect(1:size(r, 1)); ci = collect(1:size(r, 2))
                else
                    ext = [d for d in size(r) if d != 1]
                    e = isempty(ext) ? 1 : ext[1]
                    ri === nothing && (ri = collect(1:e))
                    ci === nothing && (ci = collect(1:e))
                end
            else
                ri === nothing && (ri = collect(1:size(a, 1)))
                ci === nothing && (ci = collect(1:size(a, 2)))
            end
        end
        nr = max(size(a, 1), isempty(ri) ? 0 : maximum(ri))
        nc = max(size(a, 2), isempty(ci) ? 0 : maximum(ci))
        if (nr, nc) != size(a)
            b = zeros(eltype(a), nr, nc)
            b[1:size(a, 1), 1:size(a, 2)] .= a
            a = b
        else
            a = copy(a)
        end
        if length(r) == 1
            a[ri, ci] .= r[1]
        elseif size(r) == (length(ri), length(ci))
            a[ri, ci] .= r
        elseif length(r) == length(ri) * length(ci) && (minimum(size(r)) == 1 && (length(ri) == 1 || length(ci) == 1))
            a[ri, ci] .= reshape(r, length(ri), length(ci))
        else
            merr("A(I,J) = X: dimensions mismatch ($(length(ri))x$(length(ci)) on the left, $(size(r,1))x$(size(r,2)) on the right)")
        end
        return ret(a)
    end
    _unsup("arrays with more than two dimensions")
end

function assign_brace(v, args, rhs)
    if v === nothing || (v isa AbstractMatrix && isempty(v))
        v = Cell(Any[], (0, 0))
    end
    v isa Cell || merr("'{}' assignment to a $(mclass(v))")
    empty = zeros(0, 0)
    if length(args) == 1
        k = as_index_int(args[1])
        k < 1 && merr("subscript indices must be positive integers")
        items = copy(v.items)
        shp = v.shape
        if k > length(items)
            (shp[1] > 1 && shp[2] > 1) && merr("a cell matrix cannot be resized with a linear index")
            append!(items, Any[empty for _ in 1:(k - length(items))])
            shp = (shp[2] == 1 && shp[1] > 1) ? (length(items), 1) : (1, length(items))
        end
        items[k] = rhs
        return Cell(items, shp)
    end
    if length(args) == 2
        i, j = as_index_int(args[1]), as_index_int(args[2])
        (i < 1 || j < 1) && merr("subscript indices must be positive integers")
        nr, nc = max(v.shape[1], i), max(v.shape[2], j)
        grid = Matrix{Any}(undef, nr, nc)
        fill!(grid, empty)
        for c in 1:v.shape[2], r_ in 1:v.shape[1]
            grid[r_, c] = v.items[(c - 1) * v.shape[1] + r_]
        end
        grid[i, j] = rhs
        return Cell(vec(grid), (nr, nc))
    end
    _unsup("cell arrays with more than two dimensions")
end
