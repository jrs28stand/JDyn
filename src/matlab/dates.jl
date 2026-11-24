# Dynare's `dates` class (matlab/dseries/src/@dates) for the interpreter.
#
# A dates object is a handle holding a frequency (1, 2, 4, 12, 52, 365; 0 = not set, MATLAB's `[]`) and a column
# of times: year*freq+subperiod for the frequencies 1/2/4/12, the count of ISO weeks since year 0 for weekly
# dates and the datenum for daily ones. Methods are registered in OBJM["dates"] as f(I, args, nout) with the
# object as args[1] (function-call form); the interpreter calls them for both `f(d, ...)` and `d.f(...)`.

"A MATLAB object with methods in OBJM (dates, dseries)."
abstract type MObject end

mutable struct MDates <: MObject
    freq::Int
    time::Vector{Float64}
end
MDates() = MDates(0, Float64[])
Base.copy(d::MDates) = MDates(d.freq, copy(d.time))

mclass(::MDates) = "dates"
shape_of(::MDates) = (1, 1)

"Methods by class: OBJM[class][name](I, args, nout)."
const OBJM = Dict{String,Dict{String,Function}}("dates" => Dict{String,Function}(), "dseries" => Dict{String,Function}())
"Names reachable with the dot syntax (`d.name`, `d.name(args)`) by class; the others are functions only."
const OBJDOT = Dict{String,Set{String}}("dates" => Set{String}(), "dseries" => Set{String}())
dm!(f, cls, names...) = (for n in names; OBJM[cls][n] = f; end; f)
dot!(cls, names...) = union!(OBJDOT[cls], names)

# ---------------------------------------------------------------------- calendar helpers
const DFREQS = (1, 2, 4, 12, 52, 365)

function freq2string(f)
    f == 1 && return "Y"; f == 2 && return "S"; f == 4 && return "Q"; f == 12 && return "M"
    f == 52 && return "W"; f == 365 && return "D"
    merr("dates::freq2string: Unknown frequency!")
end
function string2freq(s::AbstractString)
    u = uppercase(s)
    u in ("Y", "A") && return 1
    u in ("H", "S") && return 2
    u == "Q" && return 4
    u == "M" && return 12
    u == "W" && return 52
    u == "D" && return 365
    merr("dates::string2freq: Unknown frequency!")
end

_isint(x) = x isa Real && isfinite(x) && x == floor(x)
isleapyear(y) = mod(y, 4) == 0 && (mod(y, 100) != 0 || mod(y, 400) == 0)
_fix(x) = trunc(x)
islongyear(y) = mod(y + _fix(y / 4) - _fix(y / 100) + _fix(y / 400), 7) == 4 ||
                mod(y - 1 + _fix((y - 1) / 4) - _fix((y - 1) / 100) + _fix((y - 1) / 400), 7) == 3

"MATLAB's datenum(y, m, d) (proleptic Gregorian; datenum(0, 1, 1) = 1)."
function datenum(y::Integer, m::Integer, d::Integer)
    y -= m <= 2
    era = fld(y, 400)
    yoe = y - era * 400
    doy = fld(153 * (m + (m > 2 ? -3 : 9)) + 2, 5) + d - 1
    doe = yoe * 365 + fld(yoe, 4) - fld(yoe, 100) + doy
    era * 146097 + doe - 719468 + 719529
end
datenum(y, m, d) = Float64(datenum(Int(y), Int(m), Int(d)))
"(year, month, day) of a datenum."
function datevec_(n::Real)
    z = Int(n) - 719529 + 719468
    era = fld(z, 146097)
    doe = z - era * 146097
    yoe = fld(doe - fld(doe, 1460) + fld(doe, 36524) - fld(doe, 146096), 365)
    y = yoe + era * 400
    doy = doe - (365 * yoe + fld(yoe, 4) - fld(yoe, 100))
    mp = fld(5 * doy + 2, 153)
    d = doy - fld(153 * mp + 2, 5) + 1
    m = mp < 10 ? mp + 3 : mp - 9
    (m <= 2 ? y + 1 : y, m, d)
end

function isvalidday(y, m, d)
    (m > 12 || m < 1 || d < 1) && return false
    m in (1, 3, 5, 7, 8, 10, 12) && d > 31 && return false
    m in (4, 6, 9, 11) && d > 30 && return false
    m == 2 && d > (isleapyear(y) ? 29 : 28) && return false
    true
end

const _LONGYEARS = [Float64(y) for y in 1:400 if islongyear(y)]
function _longyears_since_zero(y::Real)
    y < 0 && return 0.0
    q = floor(y / 400); r = y - 400q
    length(_LONGYEARS) * q + count(<=(r), _LONGYEARS)
end
_longyearsince(y::Real, z::Real=0) = y >= z ? _longyears_since_zero(y) - _longyears_since_zero(z - 1) : 0.0
weeksince(y::Real, w::Real, z::Real=0) = 52 * (y - z) + (y - 1 >= z ? _longyearsince(y - 1, z) : 0.0) + w
const _WEEKSINCYCLE = vcat(0.0, cumsum([52.0 + islongyear(y) for y in 0:399]))
function isoyearweek(n::Real)
    (n < 1 || n != round(n)) && merr("isoyearweek: Input argument must only contain integers greater than zero!")
    b = floor((n - 1) / _WEEKSINCYCLE[end])
    m = n - b * _WEEKSINCYCLE[end]
    j = count(<(m), _WEEKSINCYCLE) - 1
    (400b + j, m - _WEEKSINCYCLE[j + 1])
end

_stringdate_re = (
    y = r"^-?[0-9]+[YyAa]$", s = r"^-?[0-9]+[HhSs][1-2]$", q = r"^-?[0-9]+[Qq][1-4]$",
    m = r"^-?[0-9]+[Mm]([1-9]|1[0-2])$", w = r"^[0-9]+[Ww]([1-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-3])$",
    d = r"^-?[0-9]+[-](0[1-9]|1[0-2])+[-](0[1-9]|[1-2][0-9]|3[0-1])$")

"(freq, time) of a string that is a date (`2000Q1`, `1990Y`, `2000-01-31`, ...), else nothing."
function string2date(s::AbstractString)
    r = _stringdate_re
    if occursin(r.y, s)
        return (1, parse(Float64, s[1:end-1]))
    elseif occursin(r.s, s)
        return (2, parse(Float64, s[1:end-2]) * 2 + parse(Float64, s[end:end]))
    elseif occursin(r.q, s)
        return (4, parse(Float64, s[1:end-2]) * 4 + parse(Float64, s[end:end]))
    elseif occursin(r.m, s)
        i = findfirst(c -> c in "Mm", s)
        return (12, parse(Float64, s[1:i-1]) * 12 + parse(Float64, s[i+1:end]))
    elseif occursin(r.w, s)
        i = findfirst(c -> c in "Ww", s)
        y, w = parse(Float64, s[1:i-1]), parse(Float64, s[i+1:end])
        (w == 0 || w > 52 + islongyear(y)) && return nothing
        return (52, weeksince(y, w))
    elseif occursin(r.d, s)
        neg = startswith(s, "-")
        p = split(neg ? s[2:end] : s, '-')
        y = parse(Int, p[1]) * (neg ? -1 : 1)
        isvalidday(y, parse(Int, p[2]), parse(Int, p[3])) || return nothing
        return (365, datenum(y, parse(Int, p[2]), parse(Int, p[3])))
    end
    nothing
end
isstringdate(x) = x isa AbstractString && string2date(x) !== nothing
isdate_(x) = (x isa Real && !(x isa Bool)) || isstringdate(x)
isfreq_(x) = (x isa AbstractString && length(x) == 1 && uppercase(x) in ("Y", "A", "H", "S", "Q", "M", "W", "D")) ||
             (x isa Real && !(x isa Bool) && x in DFREQS)
_tofreq(x) = x isa AbstractString ? string2freq(x) : Int(x)

_i2s(x::Real) = string(Int(x))
function date2string(time::Real, freq::Integer)
    freq == 365 && (v = datevec_(time); return string(lpad(v[1], 4, '0'), '-', lpad(v[2], 2, '0'), '-', lpad(v[3], 2, '0')))
    freq == 1 && return _i2s(time) * "Y"
    if freq == 52
        y, w = isoyearweek(time)
        return _i2s(y) * "W" * _i2s(w)
    end
    freq in DFREQS || merr("dates::freq2string: Unknown frequency!")
    year = floor((time - 1) / freq)
    _i2s(year) * freq2string(freq) * _i2s(time - year * freq)
end
date2string(d::MDates) = (length(d.time) == 1 || merr("dates::format: Input argument has to be a dates object with one element!"); date2string(d.time[1], d.freq))

dates_isempty(d::MDates) = isempty(d.time)
"A dates object of frequency `f` with the given times."
mkdates(f::Integer, t::AbstractVector) = MDates(Int(f), Float64.(collect(t)))

# ---------------------------------------------------------------------- construction
function _years_subperiods_time(freq, years, subs, who="dates")
    years = vec(years); subs = vec(subs)
    (length(years) != length(subs) && length(years) != 1 && length(subs) != 1) &&
        throw(MatlabError("$who::subsref: The years and the subperiods must have the same number of elements."))
    n = max(length(years), length(subs))
    years = length(years) == 1 ? fill(years[1], n) : years
    subs = length(subs) == 1 ? fill(subs[1], n) : subs
    if freq == 52
        any(<(0), years) && merr("dates::subsref: The weekly frequency does not extend to the years preceding the year zero.")
        any(i -> subs[i] < 1 || subs[i] > 52 + islongyear(years[i]), 1:n) &&
            merr("dates::subsref: Subperiods must be ISO weeks, between 1 and 52, or 53 in long years.")
        return [weeksince(years[i], subs[i]) for i in 1:n]
    end
    (all(_isint, years) && all(_isint, subs)) || merr("dates::subsref: The years and the subperiods must be integers.")
    any(s -> s < 1 || s > freq, subs) && merr("dates::subsref: Subperiods must be integers between 1 and $freq.")
    years .* freq .+ subs
end

_numvec(x, what) = (x isa Real || (x isa AbstractMatrix{<:Real} && min(size(x)...) == 1)) ? vec(farr(x)) : merr(what)
_isintvec(x) = (x isa Real || (x isa AbstractMatrix{<:Real} && !isempty(x) && min(size(x)...) == 1)) && all(_isint, vec(farr(x)))

function dates_horzcat(args)
    all(a -> a isa MDates, args) || merr("dates:horzcat:ArgCheck: All input arguments must be dates objects.")
    isempty(args) && return MDates()
    o = copy(args[1])
    for p in args[2:end]
        if o.freq == p.freq || o.freq == 0 || p.freq == 0
            if isempty(o.time)
                o = copy(p)
                continue
            end
            isempty(p.time) || append!(o.time, p.time)
        else
            throw(MatlabError("All input arguments must have the same frequency.", "dates:horzcat"))
        end
    end
    o
end

function dates_ctor(args::AbstractVector)
    n = length(args)
    n == 0 && return MDates()
    all(a -> a isa MDates, args) && return dates_horzcat(args)
    if all(isstringdate, args)
        tmp = [string2date(a) for a in args]
        all(t -> t[1] == tmp[1][1], tmp) || throw(MatlabError("All dates passed as inputs must have the same frequency!", "dates:ArgCheck"))
        return MDates(tmp[1][1], Float64[t[2] for t in tmp])
    end
    ac(msg) = throw(MatlabError(msg, "dates:ArgCheck"))
    if n == 1 && isfreq_(args[1])
        return MDates(_tofreq(args[1]), Float64[])
    end
    if n == 3 && isfreq_(args[1])
        f = _tofreq(args[1])
        f == 365 && ac("Requires four inputs for daily dates.")
        _isintvec(args[2]) || ac("Second input must be a vector of integers.")
        _isintvec(args[3]) || ac("Third input must be a vector of integers.")
        y, s = vec(farr(args[2])), vec(farr(args[3]))
        if f == 52
            any(<(0), y) && ac("Second input must contain nonnegative years (the weekly frequency does not extend to the years preceding the year zero).")
            all(>=(1), s) || ac("Third input must contain strictly positive integers.")
            any(i -> s[min(i, end)] > 52 && !islongyear(y[min(i, end)]), 1:max(length(y), length(s))) &&
                ac("Third input specifies non existing ISO weeks (argument cannot be greater than 52 except in long years).")
            any(>(53), s) && ac("Third input specifies non existing ISO weeks (argument cannot be greater than 53).")
            return MDates(f, _years_subperiods_time(f, y, s))
        end
        (all(>=(1), s) && all(<=(f), s)) || ac("Third input must contain integers between 1 and $f.")
        length(y) == length(s) || length(y) == 1 || length(s) == 1 || ac("dates: sizes do not match")
        return MDates(f, _years_subperiods_time(f, y, s))
    end
    if n == 2 && isfreq_(args[1])
        f = _tofreq(args[1])
        x = args[2]
        if f == 1
            _isintvec(x) || ac("Second input must be a vector of integers.")
            return MDates(1, vec(farr(x)))
        elseif f in (2, 4, 12)
            (x isa AbstractMatrix{<:Real} || x isa Real) && size(farr(x), 2) == 2 || ac("The second input must be a n*2 array of integers.")
            a = farr(x)
            all(_isint, a) || ac("Second input argument must be an array of integers.")
            (all(>=(1), a[:, 2]) && all(<=(f), a[:, 2])) || ac("Second column of the last input must contain integers between 1 and $f.")
            return MDates(f, a[:, 1] .* f .+ a[:, 2])
        elseif f == 365
            (x isa AbstractMatrix{<:Real} || x isa Real) && size(farr(x), 2) == 3 || ac("For daily dates the second input must be a n*3 array of integers.")
            a = farr(x)
            all(_isint, a) || ac("Second input argument must be an array of integers.")
            return MDates(f, [datenum(a[i, 1], a[i, 2], a[i, 3]) for i in 1:size(a, 1)])
        end
        _unsup("dates(52, [..]) (the MATLAB class returns an empty object)")
    end
    if n == 4 && isfreq_(args[1]) && (args[1] == 365 || (args[1] isa AbstractString && uppercase(args[1]) == "D"))
        all(_isintvec, args[2:4]) || ac("Second, third and fourth inputs must be vectors of integers.")
        y, m, d = (vec(farr(a)) for a in args[2:4])
        (all(>=(1), m) && all(<=(12), m)) || ac("Third input must contain integers between 1 and 12 (days).")
        length(y) == length(m) == length(d) || ac("Vectors passed as second, third, and fourth arguments must have the same number of elements.")
        return MDates(365, [datenum(y[i], m[i], d[i]) for i in 1:length(y)])
    end
    ac("The input cannot be interpreted as a date.")
end

"A dates object from a dates object or a date string (the argument of a method that accepts either)."
function as_dates(x, who="dates")
    x isa MDates && return x
    isstringdate(x) && return dates_ctor(Any[x])
    merr("$who: the input cannot be interpreted as a date")
end

# ---------------------------------------------------------------------- operators
_dcol(d::MDates) = reshape(d.time, :, 1)
_dret(a::AbstractMatrix) = ret(a)

function _dcheck(o, p, name)
    (o isa MDates && p isa MDates) || throw(MatlabError("Input arguments have to be dates objects!", "dates:$name:ArgCheck"))
    o.freq == p.freq || throw(MatlabError("Input arguments must have common frequency!", "dates:$name:ArgCheck"))
    if length(o.time) != length(p.time) && !(length(o.time) == 1 || length(p.time) == 1) && name != "eq"
        throw(MatlabError("Dimensions are not consistent!", "dates:$name:ArgCheck"))
    end
end
function _dcmp(fn, name)
    function f(I, a, nout)
        length(a) == 2 || throw(MatlabError("I need exactly two input arguments!", "dates:$name:ArgCheck"))
        o, p = a
        if name == "eq" && o isa MDates && p isa MDates && length(o.time) > 1 && length(p.time) > 1 && length(o.time) != length(p.time)
            return false
        end
        _dcheck(o, p, name)
        x, y = _dcol(o), _dcol(p)
        if length(x) == length(y)
            return ret(fn.(x, y))
        elseif length(x) == 1 || length(y) == 1
            return ret(fn.(x, y))
        elseif isempty(x) || isempty(y)
            return ret(zeros(Bool, 0, 1))
        end
        merr("nonconformant arguments")
    end
end
dm!(_dcmp(==, "eq"), "dates", "eq")
dm!(_dcmp(!=, "ne"), "dates", "ne")
dm!(_dcmp(<, "lt"), "dates", "lt")
dm!(_dcmp(>, "gt"), "dates", "gt")
dm!(_dcmp(<=, "le"), "dates", "le")
dm!(_dcmp(>=, "ge"), "dates", "ge")

_isscalar_int(x) = x isa Real && !(x isa Bool) && _isint(x)
function _isintvec_eq(p, n)
    p isa AbstractMatrix{<:Real} && !(p isa AbstractMatrix{Bool}) && min(size(p)...) == 1 && length(p) == n && all(_isint, p)
end

dm!("dates", "plus") do I, a, nout
    o, p = a[1], a[2]
    perr() = throw(MatlabError("Invalid input.", "dates:plus:ArgCheck"))
    if o isa MDates && p isa MDates
        o.freq == p.freq || throw(MatlabError("Input arguments must have common frequency.", "dates:plus:ArgCheck"))
        isempty(p.time) && return copy(o)
        isempty(o.time) && return copy(p)
        return MDates(o.freq, vcat(o.time, p.time))
    elseif o isa MDates || p isa MDates
        d, x = o isa MDates ? (o, p) : (p, o)
        (x isa Real || x isa AbstractMatrix{<:Real}) && !(x isa AbstractString) || perr()
        xv = vec(farr(x))
        ok = (length(xv) > 0 && min(size(farr(x))...) == 1 && length(xv) == length(d.time) && all(_isint, xv)) ||
             (length(d.time) == 1 && length(xv) > 0 && min(size(farr(x))...) == 1 && all(_isint, xv)) ||
             (length(xv) == 1 && _isint(xv[1]))
        ok || perr()
        return MDates(d.freq, d.time .+ xv)
    end
    perr()
end

dm!("dates", "minus") do I, a, nout
    o, p = a[1], a[2]
    if o isa MDates && p isa MDates
        o.freq == p.freq || throw(MatlabError("Input arguments must have common frequencies.", "dates:minus:ArgCheck"))
        (isempty(o.time) || isempty(p.time)) && throw(MatlabError("Input arguments must not be empty.", "dates:minus:ArgCheck"))
        u, v = o.time, p.time
        if length(u) != length(v)
            if length(u) == 1
                u = fill(u[1], length(v))
            elseif length(v) == 1
                v = fill(v[1], length(u))
            else
                throw(MatlabError("Input arguments lengths are not consistent.", "dates:minus:ArgCheck"))
            end
        end
        return ret(reshape(u .- v, :, 1))
    elseif o isa MDates && !(p isa MDates)
        xv = (p isa Real || p isa AbstractMatrix{<:Real}) ? vec(farr(p)) : Float64[]
        ok = !isempty(xv) && ((min(size(farr(p))...) == 1 && length(xv) == length(o.time) && all(_isint, xv)) ||
             (length(xv) == 1 && _isint(xv[1])) || (length(o.time) == 1 && min(size(farr(p))...) == 1 && all(_isint, xv)))
        ok || throw(MatlabError("Second argument has to be a vector of integers or scalar integer.", "dates:minus:ArgCheck"))
        return MDates(o.freq, o.time .- xv)
    end
    throw(MatlabError("Invalid input.", "dates:minus:ArgCheck"))
end

dm!("dates", "uminus") do I, a, nout
    o = a[1]; MDates(o.freq, o.time .- 1)
end
dm!("dates", "uplus") do I, a, nout
    o = a[1]; MDates(o.freq, o.time .+ 1)
end
dm!("dates", "mtimes") do I, a, nout
    o, n = a[1], a[2]
    (o isa MDates && _isscalar_int(n)) || throw(MatlabError("First and second input arguments have to be a dates object and a scalar integer!", "dates:mtimes:ArgCheck"))
    MDates(o.freq, repeat(o.time, Int(n)))
end

"`a:b` and `a:d:b` for dates objects."
function dates_colon(a, b, c=nothing)
    ce(msg) = throw(MatlabError(msg, "dates:colon:ArgCheck"))
    if c === nothing
        o, p, d = a, b, 1.0
        (o isa MDates && p isa MDates && length(o.time) == 1 && length(p.time) == 1) ||
            ce("In an expression like A:B, A and B must be one element dates objects.")
    else
        o, p, d = a, c, b
        (o isa MDates && p isa MDates && length(o.time) == 1 && d isa Real && _isint(d)) ||
            ce("In an expression like A:d:B, A and B must be one element dates objects and d a scalar integer.")
        d == 0 && ce("In an expression like A:d:B, d (the incremental number of periods) must nonzero.")
        d = Float64(d)
    end
    o.freq == p.freq || ce("Input arguments must have common frequency.")
    (o.time[1] > p.time[1] && d > 0) && ce("First date must preceed the second one.")
    (p.time[1] > o.time[1] && d < 0) && ce("Second date must preceed the first one.")
    MDates(o.freq, collect(o.time[1]:d:p.time[1]))
end
dm!("dates", "colon") do I, a, nout
    length(a) in (2, 3) || throw(MatlabError("The colon (:) operator for dates objects accepts only 2 or 3 input arguments.", "dates:colon:ArgCheck"))
    dates_colon(a...)
end

# ---------------------------------------------------------------------- methods
dm!("dates", "ndat", "length") do I, a, nout
    Float64(length(a[1].time))
end
dm!("dates", "isempty") do I, a, nout
    isempty(a[1].time)
end
dm!("dates", "copy") do I, a, nout
    copy(a[1])
end
dm!("dates", "char") do I, a, nout
    o = a[1]
    length(o.time) > 1 && throw(MatlabError("The input argument must be a dates object with one element!", "dates:char:ArgCheck"))
    isempty(o.time) && _unsup("char of an empty dates object")
    date2string(o.time[1], o.freq)
end
dm!("dates", "strings") do I, a, nout
    o = a[1]
    Cell(Any[date2string(t, o.freq) for t in o.time], (1, length(o.time)))
end
dm!("dates", "year") do I, a, nout
    d = a[1]
    d.freq == 365 && (isempty(d.time) ? merr("year of an empty dates") : return Float64(datevec_(d.time[1])[1]))
    d.freq == 52 && return ret(reshape(Float64[isoyearweek(t)[1] for t in d.time], :, 1))
    d.freq > 1 && return ret(reshape(floor.((d.time .- 1) ./ d.freq), :, 1))
    ret(reshape(copy(d.time), :, 1))
end
dm!("dates", "subperiod") do I, a, nout
    d = a[1]
    d.freq == 365 && merr("dates::subperiod: Method not available for daily frequency!")
    d.freq == 52 && return ret(reshape(Float64[isoyearweek(t)[2] for t in d.time], :, 1))
    ret(reshape(d.time .- floor.((d.time .- 1) ./ d.freq) .* d.freq, :, 1))
end
dm!("dates", "double") do I, a, nout
    o = a[1]
    (o.freq == 365 || o.freq == 52) && merr("This method is not implemented for daily frequency.")
    m = if o.freq == 1
        copy(o.time)
    else
        years = floor.((o.time .- 1) ./ o.freq)
        years .+ (o.time .- years .* o.freq .- 1) ./ o.freq
    end
    r = ret(reshape(m, :, 1))
    nout > 1 ? Any[r, Float64(o.freq)] : r
end
dm!("dates", "disp", "display") do I, a, nout
    Any[]
end

_sort_ip!(o::MDates) = (length(o.time) > 1 && sort!(o.time); o)
function _unique_ip!(o::MDates)
    length(o.time) <= 1 && return o
    # unique(..., 'legacy') keeps the last occurrence of each value, in the order of those occurrences
    last = Dict{Float64,Int}()
    for (i, t) in enumerate(o.time)
        last[t] = i
    end
    o.time = o.time[sort!(collect(values(last)))]
    o
end
dm!("dates", "sort") do I, a, nout; _sort_ip!(copy(a[1])); end
dm!("dates", "sort_") do I, a, nout; _sort_ip!(a[1]); end
dm!("dates", "unique") do I, a, nout; length(a[1].time) <= 1 ? a[1] : _unique_ip!(copy(a[1])); end
dm!("dates", "unique_") do I, a, nout; _unique_ip!(a[1]); end

function _append!(o::MDates, d, who)
    if d isa MDates
        length(d.time) == 1 || (isempty(d.time) && return o) ||
            throw(MatlabError("Input argument has to be a dates object with one element.", "dates:$who:ArgCheck"))
    elseif isstringdate(d)
        d = dates_ctor(Any[d])
    else
        throw(MatlabError("Invalid input.", "dates:$who:ArgCheck"))
    end
    o.freq == d.freq || throw(MatlabError("dates must have common frequency!", "dates:$who:ArgCheck"))
    o.time = vcat(o.time, d.time)
    o
end
dm!("dates", "append") do I, a, nout
    d = a[2]
    d isa MDates && isempty(d.time) && return a[1]
    length(a) == 2 || merr("Invalid call to append")
    o = a[1]
    d isa MDates && length(d.time) != 1 && throw(MatlabError("Input argument has to be a dates object with one element.", "dates:append:ArgCheck"))
    _append!(copy(o), d, "append")
end
dm!("dates", "append_") do I, a, nout
    length(a) == 2 || merr("Invalid call to append_")
    _append!(a[1], a[2], "append_")
end
function _pop!(o::MDates, a)
    isempty(o.time) && return o
    if isempty(a)
        o.time = o.time[1:end-1]
        return o
    end
    p = a[1]
    (p isa MDates || isdate_(p)) || throw(MatlabError("Input argument has to be a dates object with a single element, a string (which can be interpreted as a date) or an integer!", "dates:pop"))
    p isa AbstractString && (p = dates_ctor(Any[p]))
    if p isa Real
        o.time = o.time[[i for i in eachindex(o.time) if Float64(i) != p]]
    else
        o.freq == p.freq || throw(MatlabError("Inputs must have common frequency!", "dates:pop"))
        length(p.time) > 1 && throw(MatlabError("dates to be removed must have one element!", "dates:pop"))
        isempty(p.time) && return o
        idx = findall(==(p.time[1]), o.time)
        isempty(idx) && merr("Index exceeds matrix dimensions")
        o.time = o.time[[i for i in eachindex(o.time) if i != idx[end]]]
    end
    o
end
dm!("dates", "pop") do I, a, nout; _pop!(copy(a[1]), a[2:end]); end
dm!("dates", "pop_") do I, a, nout; _pop!(a[1], a[2:end]); end
function _remove!(o::MDates, a)
    isempty(a) && throw(MatlabError("Input argument is missing!", "dates:remove"))
    p = a[1]
    p isa MDates || throw(MatlabError("Input argument has to be a dates object", "dates:remove"))
    o.freq == p.freq || throw(MatlabError("Inputs must have common frequency!", "dates:remove"))
    o.time = _setdiff_legacy(o.time, p.time)
    o
end
dm!("dates", "remove") do I, a, nout; _remove!(copy(a[1]), a[2:end]); end
dm!("dates", "remove_") do I, a, nout; _remove!(a[1], a[2:end]); end

"Sorted values of `a` that are not in `b` (Octave's setdiff)."
_setdiff_legacy(a, b) = sort!(unique([x for x in a if !(x in b)]))
dot!("dates", "time", "freq", "sort", "sort_", "unique", "unique_", "double", "isempty", "length", "char", "ndat",
     "year", "subperiod", "strings", "append", "append_", "pop", "pop_", "remove", "remove_", "disp")

dm!("dates", "horzcat", "vertcat") do I, a, nout
    dates_horzcat(a)
end

function _dmaxmin(fn, name)
    function f(I, a, nout)
        all(x -> x isa MDates, a) || throw(MatlabError("All input arguments must be dates objects.", "dates:$name:ArgCheck"))
        o = length(a) == 1 ? a[1] : dates_horzcat(a)
        isempty(o.time) && return MDates(o.freq, Float64[])
        MDates(o.freq, [fn(o.time)])
    end
end
dm!(_dmaxmin(maximum, "max"), "dates", "max")
dm!(_dmaxmin(minimum, "min"), "dates", "min")

dm!("dates", "intersect") do I, a, nout
    o, p = a[1], a[2]
    (o isa MDates && p isa MDates) || throw(MatlabError("All input arguments must be dates objects!", "dates:intersect:ArgCheck"))
    (length(o.time) == length(p.time) && o.freq == p.freq && o.time == p.time) && return copy(o)
    o.freq == p.freq || return MDates()
    t = sort!(unique([x for x in o.time if x in p.time]))
    isempty(t) && return MDates()
    MDates(o.freq, t)
end
dm!("dates", "union") do I, a, nout
    all(x -> x isa MDates, a) || throw(MatlabError("All input arguments must be dates objects.", "dates:union:ArgCheck"))
    o = length(a) == 1 ? copy(a[1]) : dates_horzcat(a)
    _sort_ip!(_unique_ip!(o))
end
dm!("dates", "setdiff") do I, a, nout
    o, p = a[1], a[2]
    (o isa MDates && p isa MDates) || throw(MatlabError("All input arguments must be dates objects.", "dates:setdiff"))
    o.freq == p.freq || throw(MatlabError("All input arguments must have common frequency.", "dates:setdiff"))
    isempty(p.time) && return copy(o)
    (length(o.time) == length(p.time) && o.time == p.time) && return MDates(o.freq, Float64[])
    MDates(o.freq, _setdiff_legacy(o.time, p.time))
end
dm!("dates", "isequal") do I, a, nout
    length(a) >= 2 || merr("isequal: at least two arguments")
    all(x -> x isa MDates, a[1:2]) || throw(MatlabError("Both inputs must be dates objects!", "dates:isequal:ArgCheck"))
    a[1].freq == a[2].freq && a[1].time == a[2].time
end

# ---------------------------------------------------------------------- indexing and properties
function dates_property(o::MDates, name)
    name == "time" && return ret(reshape(copy(o.time), :, 1))
    name == "freq" && return o.freq == 0 ? zeros(0, 0) : Float64(o.freq)
    nothing
end

"`d(args)` for a dates object."
function dates_index(o::MDates, args)
    if isempty(o.time)
        _unsup("populating an empty dates object by calling it with (freq, years, subperiods)")
    end
    length(args) == 1 || merr("dates::subsref: indices has to be a vector of positive integers less than or equal to $(length(o.time))!")
    ix = args[1]
    n = length(o.time)
    if ix isa AbstractMatrix && isempty(ix)
        return MDates(o.freq, Float64[])
    end
    ix === COLON && return MDates(o.freq, copy(o.time))
    ix isa AbstractString && merr("dates::subsref: indices has to be a vector of positive integers less than or equal to $n!")
    a = arr(ix)
    isvec = min(size(a)...) == 1
    if isvec && length(a) == n && eltype(a) == Bool
        return MDates(o.freq, o.time[vec(a)])
    end
    if isvec && eltype(a) != Bool && all(_isint, a) && all(>(0), a) && all(<=(n), a)
        return MDates(o.freq, o.time[Int.(vec(a))])
    end
    if isvec && length(a) == n
        return MDates(o.freq, o.time[Int.(vec(a))])
    end
    merr("dates::subsref: indices has to be a vector of positive integers less than or equal to $n!")
end
