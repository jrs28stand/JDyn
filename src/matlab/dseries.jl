# Dynare's `dseries` class (matlab/dseries/src/@dseries) for the interpreter. Values are handles as in Dynare:
# the `name_` methods change the object in place, `name` returns a modified copy. Data matrices are never
# changed in place (they may be shared with MATLAB variables): a method replaces `o.data`.

mutable struct MDseries <: MObject
    data::Matrix{Float64}
    name::Vector{String}
    tex::Vector{String}
    dates::MDates
    ops::Vector{String}                      # "" = empty
    tags::Vector{Pair{String,Vector{Any}}}
end
MDseries() = MDseries(zeros(0, 0), String[], String[], MDates(), String[], Pair{String,Vector{Any}}[])
mclass(::MDseries) = "dseries"
shape_of(::MDseries) = (1, 1)

nobs(o::MDseries) = size(o.data, 1)
vobs(o::MDseries) = size(o.data, 2)
ds_isempty(o::MDseries) = isempty(o.data)
dsfreq(o::MDseries) = o.dates.freq
ds_firstdate(o) = (isempty(o.dates.time) ? merr("Index exceeds matrix dimensions") : MDates(o.dates.freq, [o.dates.time[1]]))
ds_lastdate(o) = (isempty(o.dates.time) ? merr("Index exceeds matrix dimensions") : MDates(o.dates.freq, [o.dates.time[end]]))

default_names(n) = ["Variable_$i" for i in 1:n]
name2tex(s::AbstractString) = replace(s, "_" => "\\_")

function copy_ds(o::MDseries)
    MDseries(o.data, copy(o.name), copy(o.tex), o.dates, copy(o.ops), [k => copy(v) for (k, v) in o.tags])
end

dserr(msg, id="") = throw(MatlabError(msg, id))
wrongin(msg) = dserr(msg, "dseries:WrongInputArguments")

_range(init::MDates, n::Integer) = n == 0 ? dates_colon(init, init + 0, nothing) : mkdates(init.freq, init.time[1] .+ (0:n-1))

# a MATLAB cell column of char arrays
_namecell(v::AbstractVector) = isempty(v) ? Cell(Any[], (0, 0)) : Cell(Any[String(x) for x in v], (length(v), 1))
_opscell(v::AbstractVector) = isempty(v) ? Cell(Any[], (0, 0)) : Cell(Any[isempty(x) ? zeros(0, 0) : String(x) for x in v], (length(v), 1))
function _cellstrs(x, what)
    x isa Cell && return String[(y isa AbstractString || merr("$what must be a cell array of char arrays"); String(y)) for y in x.items]
    x isa AbstractString && return String[String(x)]
    merr("$what must be a cell array of char arrays")
end

# ---------------------------------------------------------------------- file readers
function _read_csv(path)
    isfile(path) || merr("load_csv_file_data: I can't find file $path!")
    lines = readlines(path)
    isempty(lines) && merr("load_csv_file_data: $path is empty")
    varlist = String.(split(lines[1], ','))
    length(lines) < 2 && merr("load_csv_file_data: no data below column headers in $path")
    second = String.(split(lines[2], ','))
    withtime = isstringdate(second[1])
    init = MDates(1, [1.0])
    if withtime
        init = dates_ctor(Any[second[1]])
        length(varlist) < 2 && merr("load_csv_file_data: no data right to row headers in $path")
        varlist = varlist[2:end]
    end
    rows = [l for l in lines[2:end] if !isempty(strip(l))]
    ncol = length(split(rows[1], ',')) - withtime
    data = zeros(length(rows), ncol)
    for (i, l) in enumerate(rows)
        f = split(l, ',')
        for j in 1:ncol
            s = strip(f[j + withtime])
            data[i, j] = isempty(s) ? 0.0 : (v = tryparse(Float64, s); v === nothing ? (lowercase(s) == "nan" ? NaN : merr("load_csv_file_data: cannot read $(repr(s))")) : v)
        end
    end
    length(varlist) == size(data, 2) || merr("load_csv_file_data: number of row headers not equal to number of data columns")
    init, data, [replace(v, "\"" => "") for v in varlist]
end

"The variables of a MAT-file or of a struct as (init, data, names, tex, ops, tags)."
function _from_struct(s::MStruct)
    d = Dict{String,Any}(k => s[k] for k in s._keys)
    init = MDates(1, [1.0])
    if haskey(d, "INIT__")
        isstringdate(d["INIT__"]) || merr("load_mat_file_data: INIT__ cannot be interpreted as a date.")
        init = dates_ctor(Any[d["INIT__"]])
        delete!(d, "INIT__")
    end
    if haskey(d, "FREQ__")
        Float64(d["FREQ__"]) == init.freq || merr("load_mat_file_data: INIT__ and FREQ__ are not consistent!")
        delete!(d, "FREQ__")
    end
    names = nothing; tex = nothing; ops = nothing; tags = Pair{String,Vector{Any}}[]
    haskey(d, "NAMES__") && (names = _cellstrs(d["NAMES__"], "NAMES__"); delete!(d, "NAMES__"))
    haskey(d, "TEX__") && (tex = _cellstrs(d["TEX__"], "TEX__"); delete!(d, "TEX__"))
    if haskey(d, "OPS__")
        o = d["OPS__"]
        ops = o isa Cell ? String[x isa AbstractString ? String(x) : "" for x in o.items] : String[]
        delete!(d, "OPS__")
    end
    if haskey(d, "TAGS__")
        t = d["TAGS__"]
        t isa MStruct && for k in t._keys
            c = t[k]
            push!(tags, k => (c isa Cell ? Any[c.items...] : Any[]))
        end
        delete!(d, "TAGS__")
    end
    if haskey(d, "DATA__")
        data = farr(d["DATA__"])
        delete!(d, "DATA__")
    else
        names === nothing && (names = sort(collect(keys(d))))
        cols = Vector{Float64}[]
        for nm in names
            haskey(d, nm) || merr("load_mat_file: no variable $nm")
            v = d[nm]
            (v isa Real || v isa AbstractMatrix{<:Real}) || merr("load_mat_file:: All the variables must be vectors ($nm is not a vector)!")
            a = farr(v)
            min(size(a)...) == 1 || merr("load_mat_file:: All the variables must be vectors ($nm is not a vector)!")
            push!(cols, vec(a))
        end
        all(c -> length(c) == length(cols[1]), cols) || merr("load_mat_file:: All the vectors (variables) must have the same number of rows (observations)!")
        data = isempty(cols) ? zeros(0, 0) : reduce(hcat, cols)
    end
    names === nothing && (names = String[])
    ops === nothing && (ops = fill("", length(names)))
    init, data, names, tex, ops, tags
end

function _mat_struct(path)
    data = try
        MAT.matread(path)
    catch
        throw(Unsupported("dseries: $(basename(path)) is not a MAT-file that can be read", "file"))
    end
    s = MStruct()
    for k in sort(collect(keys(data)))
        s[String(k)] = _mat_value(data[k])
    end
    s
end

function _m_struct(I, path)
    text = read(path, String)
    body, funcs = mparse(text)
    sub = Interp(dirname(abspath(path)))
    sub.base.funcs = funcs
    try
        exec_block(sub, body, sub.base)
    catch e
        e isa ReturnSig || rethrow()
    end
    s = MStruct()
    for k in sort(collect(keys(sub.base.vars)))
        s[k] = sub.base.vars[k]
    end
    s
end

function _resolve_file(I, name)
    _need_exec("dseries(file)")
    cands = isabspath(name) ? [name] : [joinpath(I.dir === nothing ? pwd() : I.dir, name), name]
    for c in cands
        isfile(c) && return c
    end
    merr("dseries: cannot find file $name")
end

function load_data(I, filename)
    isempty(filename) && merr("dseries:load_data: Input argument cannot be an empty string!")
    path = _resolve_file(I, filename)
    ext = lowercase(splitext(path)[2])
    if ext == ".m"
        init, data, names, tex, ops, tags = _from_struct(_m_struct(I, path))
    elseif ext == ".mat"
        init, data, names, tex, ops, tags = _from_struct(_mat_struct(path))
    elseif ext == ".csv"
        init, data, names = _read_csv(path)
        tex = nothing; ops = fill("", length(names)); tags = Pair{String,Vector{Any}}[]
    elseif ext in (".xls", ".xlsx")
        throw(Unsupported("dseries: reading spreadsheets ($(basename(path))) is not implemented", "file"))
    else
        dserr("I'm not able to load data from $filename!", "dseries:WrongInputArguments")
    end
    tex === nothing && (tex = [name2tex(n) for n in names])
    init, data, names, tex, ops, tags
end

# ---------------------------------------------------------------------- constructor
function _names_arg(c, what)
    c === nothing && return nothing
    (c isa AbstractMatrix && isempty(c)) && return String[]
    c isa Cell && return _cellstrs(c, what)
    c isa AbstractString && return String[c]
    merr("$what must be a cell array of char arrays")
end

function ds_ctor(I, a::Vector{Any})
    n = length(a)
    n == 0 && return MDseries()
    o = MDseries()
    if n == 1
        x = a[1]
        if x isa MDates
            length(x.time) == 0 && wrongin("Input (identified as a dates object) must be non empty!")
            length(x.time) > 1 && wrongin("Input (identified as a dates object) must have a unique element!")
            o.dates = x
            return o
        elseif x isa AbstractString
            init, o.data, o.name, o.tex, o.ops, o.tags = load_data(I, x)
            o.dates = _range(init, nobs(o))
            return o
        elseif x isa MDseries
            return copy_ds(x)
        elseif (x isa Real || x isa AbstractMatrix{<:Real}) && !(x isa Bool || x isa AbstractMatrix{Bool})
            o.data = copy(farr(x))
            o.name = default_names(vobs(o))
            o.tex = [name2tex(s) for s in o.name]
            o.dates = _range(MDates(1, [1.0]), nobs(o))
            o.ops = fill("", vobs(o))
            return o
        elseif x isa MStruct
            return struct2dseries(I, x)
        end
        wrongin("Incorrect type for input argument")
    end
    n > 4 && wrongin("Wrong calling sequence!")
    if n == 2 && a[1] isa AbstractString && a[2] isa MDates
        init, o.data, o.name, o.tex, o.ops, o.tags = load_data(I, a[1])
        o.dates = _range(a[2], nobs(o))
        return o
    end
    if n == 2 && a[1] isa AbstractString && a[2] isa AbstractString && isstringdate(a[2])
        init, o.data, o.name, o.tex, o.ops, o.tags = load_data(I, a[1])
        o.dates = _range(dates_ctor(Any[a[2]]), nobs(o))
        return o
    end
    x, b = a[1], a[2]
    (x isa Real || x isa AbstractMatrix{<:Real}) || wrongin("Wrong calling sequence!")
    d = n < 4 ? nothing : _names_arg(a[4], "tex names")
    c = n < 3 ? nothing : _names_arg(a[3], "names")
    o.data = copy(farr(x))
    init = nothing
    if b isa AbstractMatrix && isempty(b)
        init = MDates(1, [1.0])
    elseif b isa MDates && length(b.time) == 1
        init = b
    elseif b isa AbstractString && isstringdate(b)
        init = dates_ctor(Any[b])
    elseif b isa Real && !(b isa Bool) && _isint(b)
        init = dates_ctor(Any[_i2s(b) * "Y"])
    elseif b isa MDates
        init = mkdates(b.freq, b.time[1:1])
        if nobs(o) > 1 && length(b.time) != nobs(o)
            wrongin("If second input is a range, its number of elements must match\nthe number of rows in the first input, unless the first input\nhas only one row.")
        elseif nobs(o) == 1
            o.data = repeat(o.data, length(b.time), 1)
        end
        o.dates = b
    elseif b isa AbstractMatrix{<:Real} && all(_isint, b)
        wrongin("Not implemented! If you need to define a range of years, you have to pass a dates object as the second input argument")
    else
        wrongin("Wrong calling sequence!")
    end
    if c !== nothing && !isempty(c)
        vobs(o) == length(c) || wrongin("The number of declared names does not match the number of variables!")
        o.name = copy(c)
    else
        o.name = default_names(vobs(o))
    end
    o.ops = fill("", length(o.name))
    if d !== nothing && !isempty(d)
        vobs(o) == length(d) || wrongin("The number of declared tex names does not match the number of variables!")
        o.tex = copy(d)
    else
        o.tex = [name2tex(s) for s in o.name]
    end
    if isempty(o.dates.time) && o.dates.freq == 0 || dates_isempty(o.dates)
        o.dates = _range(init, nobs(o))
    end
    o
end

function struct2dseries(I, s::MStruct)
    miss = [k for k in ("data", "dates", "name", "tex", "ops", "tags") if !haskey(s, k)]
    isempty(miss) || merr("dseries::struct2dseries: Missing field(s) in the structure: $(join(miss, ", "))!")
    _unsup("struct2dseries")
end

# ---------------------------------------------------------------------- helpers
_prev(o::MDseries, i) = isempty(o.ops[i]) ? o.name[i] : o.ops[i]
function _setops!(o::MDseries, f)
    for i in 1:vobs(o)
        o.ops[i] = f(_prev(o, i))
    end
    o
end
_num2str(I, x) = BUILTINS["num2str"](I, Any[x], 1)

"Dseries ensure: a dseries argument or an error."
_needds(x, who) = x isa MDseries ? x : throw(MatlabError("$who: Both input arguments must be dseries objects!"))

# ---------------------------------------------------------------------- align / arithmetic
function align!(o::MDseries, p::MDseries)
    dsfreq(o) == dsfreq(p) || merr("dseries::align: 'o' and 'p' dseries objects must have common frequencies!")
    isempty(intersect(o.dates.time, p.dates.time)) && merr("dseries::align: 'o' and 'p' dseries object must have at least one common date!")
    init = min(o.dates.time[1], p.dates.time[1])
    last = max(o.dates.time[end], p.dates.time[end])
    for q in (p, o)
        if q.dates.time[1] > init
            q.data = vcat(fill(NaN, Int(q.dates.time[1] - init), vobs(q)), q.data)
        end
    end
    for q in (p, o)
        if q.dates.time[end] < last
            q.data = vcat(q.data, fill(NaN, Int(last - q.dates.time[end]), vobs(q)))
        end
    end
    f = dsfreq(o)
    o.dates = mkdates(f, init .+ (0:nobs(o)-1))
    p.dates = mkdates(f, init .+ (0:nobs(p)-1))
    o, p
end
function align_copies(o, p)
    o2, p2 = copy_ds(o), copy_ds(p)
    align!(o2, p2)
end

_bsx(fn, a::AbstractMatrix, b::AbstractMatrix) = begin
    (size(a, 1) == size(b, 1) || size(a, 1) == 1 || size(b, 1) == 1) && (size(a, 2) == size(b, 2) || size(a, 2) == 1 || size(b, 2) == 1) ||
        merr("operator: nonconformant arguments (op1 is $(size(a,1))x$(size(a,2)), op2 is $(size(b,1))x$(size(b,2)))")
    fn.(a, b)
end

_isnumvec(x) = (x isa Real && !(x isa Bool)) || (x isa AbstractMatrix{Float64} && !isempty(x) && min(size(x)...) == 1)

function _numstr(I, x)
    x isa Real && return _num2str(I, x)
    string("[", BUILTINS["num2str"](I, Any[x, 12.0], 1), "]")
end

"plus, mtimes (names `plus`, `mtimes`; ops built from the operand's ops when there are some) and mrdivide."
function _ds_arith(I, name, fn, a, usesops_left=true)
    o, p = a[1], a[2]
    msg(w) = "dseries::$name: $w input argument must be a dseries object!"
    if _isnumvec(o)
        p isa MDseries || dserr(msg("Second"))
        ov = o isa Real ? fill(Float64(o), 1, 1) : o
        q = copy_ds(p)
        q.data = _bsx(fn, ov, p.data)
        for i in 1:vobs(q)
            s = o isa Real ? _num2str(I, o) : (size(o, 1) == 1 ? _num2str(I, o[i]) : _numstr(I, o))
            q.ops[i] = name == "mrdivide" ? "mrdivide($s, $(p.name[i]))" : "$name($s, $(isempty(p.ops[i]) ? p.name[i] : p.ops[i]))"
        end
        return q
    end
    if _isnumvec(p)
        o isa MDseries || dserr(msg("First"))
        pv = p isa Real ? fill(Float64(p), 1, 1) : p
        q = copy_ds(o)
        q.data = _bsx(fn, o.data, pv)
        for i in 1:vobs(q)
            s = p isa Real ? _num2str(I, p) : (size(p, 1) == 1 ? _num2str(I, p[i]) : _numstr(I, p))
            q.ops[i] = "$name($(isempty(q.ops[i]) ? q.name[i] : q.ops[i]), $s)"
        end
        return q
    end
    (o isa MDseries && p isa MDseries) || dserr("dseries::$name: wrong inputs")
    if name == "plus"
        ds_isempty(o) && return copy_ds(p)
        ds_isempty(p) && return copy_ds(o)
    end
    if vobs(o) != vobs(p) && !(vobs(o) == 1 || vobs(p) == 1)
        dserr("dseries::$(name == "plus" ? "plus" : "times"): Cannot $(name == "plus" ? "add" : name == "mtimes" ? "multiply" : "divide") inputname(1) and inputname(2) (wrong number of variables)!")
    end
    ido, idp = vobs(o) > vobs(p) ? (1:vobs(o), fill(1, vobs(o))) : vobs(o) < vobs(p) ? (fill(1, vobs(p)), 1:vobs(p)) : (1:vobs(o), 1:vobs(o))
    dsfreq(o) == dsfreq(p) || dserr("dseries::$(name == "plus" ? "plus" : "times"): Cannot $(name == "plus" ? "add" : name == "mtimes" ? "multiply" : "divide") inputname(1) and inputname(2) (frequencies are different)!")
    if nobs(o) != nobs(p) || o.dates.time[1] != p.dates.time[1]
        o, p = align_copies(o, p)
    end
    if vobs(o) >= vobs(p)
        q = copy_ds(o)
    else
        q = MDseries(zeros(size(p.data)), default_names(vobs(p)), [name2tex(s) for s in default_names(vobs(p))], p.dates, fill("", vobs(p)), Pair{String,Vector{Any}}[])
    end
    for i in 1:vobs(q)
        l = name == "mrdivide" || isempty(o.ops[ido[i]]) ? o.name[ido[i]] : o.ops[ido[i]]
        r = name == "mrdivide" || isempty(p.ops[idp[i]]) ? p.name[idp[i]] : p.ops[idp[i]]
        if name == "mrdivide"
            l = isempty(o.ops[ido[i]]) ? o.name[ido[i]] : o.ops[ido[i]]
            r = isempty(p.ops[idp[i]]) ? p.name[idp[i]] : p.ops[idp[i]]
        end
        q.ops[i] = "$name($l, $r)"
    end
    q.data = _bsx(fn, o.data, p.data)
    q
end
dm!("dseries", "plus") do I, a, nout; _ds_arith(I, "plus", +, a); end
dm!("dseries", "mtimes") do I, a, nout; _ds_arith(I, "mtimes", *, a); end
dm!("dseries", "mrdivide") do I, a, nout; _ds_arith(I, "mrdivide", /, a); end

dm!("dseries", "minus") do I, a, nout
    o, p = a[1], a[2]
    if _isnumvec(o)
        p isa MDseries || dserr("Second input argument must be a dseries object!", "dseries:WrongInputArguments")
        ov = o isa Real ? fill(Float64(o), 1, 1) : o
        q = copy_ds(p)
        q.data = _bsx(-, ov, p.data)
        for i in 1:vobs(q)
            s = o isa Real ? _numstr(I, o) : (size(o, 1) == 1 ? _numstr(I, o[i]) : _numstr(I, o))
            q.ops[i] = "minus($s, $(isempty(q.ops[i]) ? p.name[i] : p.ops[i]))"
        end
        return q
    end
    if _isnumvec(p)
        o isa MDseries || dserr("dseries::minus: First input argument must be a dseries object!")
        pv = p isa Real ? fill(Float64(p), 1, 1) : p
        q = copy_ds(o)
        q.data = _bsx(-, o.data, pv)
        for i in 1:vobs(q)
            s = p isa Real ? _numstr(I, p) : (size(p, 1) == 1 ? _numstr(I, p[i]) : _numstr(I, p))
            q.ops[i] = "minus($(isempty(q.ops[i]) ? o.name[i] : o.ops[i]), $s)"
        end
        return q
    end
    (o isa MDseries && p isa MDseries) || merr("dseries::minus: wrong inputs")
    if ds_isempty(o)
        q = copy_ds(p)
        q.data = -q.data
        for i in 1:vobs(q)
            q.ops[i] = "minus(dseries(), $(isempty(p.ops[i]) ? "" : "uminus"))"
        end
        _unsup("empty dseries minus a dseries")
    end
    ds_isempty(p) && return copy_ds(o)
    vobs(o) != vobs(p) && !(vobs(o) == 1 || vobs(p) == 1) &&
        dserr("Cannot substract inputname(1) and inputname(2) (wrong number of variables)!", "dseries:WrongInputArguments")
    ido, idp = vobs(o) > vobs(p) ? (1:vobs(o), fill(1, vobs(o))) : vobs(o) < vobs(p) ? (fill(1, vobs(p)), 1:vobs(p)) : (1:vobs(o), 1:vobs(o))
    dsfreq(o) == dsfreq(p) || dserr("Cannot substract inputname(1) and inputname(2) (frequencies are different)!", "dseries:WrongInputArguments")
    if nobs(o) != nobs(p) || o.dates.time[1] != p.dates.time[1]
        o, p = align_copies(o, p)
    end
    qn = max(vobs(o), vobs(p))
    q = MDseries(_bsx(-, o.data, p.data), String[o.name[ido[i]] for i in 1:qn], String[o.tex[ido[i]] for i in 1:qn], o.dates,
                 String["minus($(o.name[ido[i]]), $(p.name[idp[i]]))" for i in 1:qn], Pair{String,Vector{Any}}[])
    q
end

dm!("dseries", "uminus") do I, a, nout
    o = copy_ds(a[1])
    o.data = -o.data
    _setops!(o, s -> "uminus($s)")
end
dm!("dseries", "uplus") do I, a, nout; a[1]; end

dm!("dseries", "mpower") do I, a, nout
    o, p = a[1], a[2]
    _pw(x, y) = x .^ y
    if _isnumvec(o) && p isa MDseries && length(o) == nobs(p)
        q = MDseries(zeros(size(p.data)), default_names(vobs(p)), [name2tex(s) for s in default_names(vobs(p))], p.dates, fill("", vobs(p)), Pair{String,Vector{Any}}[])
        ov = o isa Real ? fill(Float64(o), 1, 1) : reshape(vec(o), :, 1)
        q.data = _bsx(_pow, ov, p.data)
        for i in 1:vobs(q)
            q.ops[i] = "power($(_numstr(I, o)), $(p.name[i]))"
        end
        return q
    end
    if _isnumvec(p) && o isa MDseries && length(p) == nobs(o)
        q = copy_ds(o)
        pv = p isa Real ? fill(Float64(p), 1, 1) : reshape(vec(p), :, 1)
        q.data = _bsx(_pow, o.data, pv)
        for i in 1:vobs(q)
            q.ops[i] = "power($(isempty(q.ops[i]) ? q.name[i] : q.ops[i]), $(_numstr(I, p)))"
        end
        return q
    end
    if o isa MDseries && p isa Real && !(p isa Bool)
        q = copy_ds(o)
        q.data = _pow.(o.data, Float64(p))
        for i in 1:vobs(q)
            q.ops[i] = "power($(isempty(o.ops[i]) ? q.name[i] : o.ops[i]), $(_num2str(I, p)))"
        end
        return q
    end
    if o isa MDseries && p isa MDseries
        (nobs(o) == nobs(p) && vobs(o) == vobs(p) && dsfreq(o) == dsfreq(p)) ||
            dserr("If both input arguments are dseries objects, they must have the same numbers of variables and observations and common frequency!", "dseries:WrongInputArguments")
        q = copy_ds(o)
        q.data = _pow.(o.data, p.data)
        for i in 1:vobs(q)
            l = isempty(o.ops[i]) ? q.name[i] : o.ops[i]
            r = isempty(p.ops[i]) ? p.name[i] : p.ops[i]
            q.ops[i] = "power($l, $r)"
        end
        return q
    end
    dserr("Wrong calling sequence!", "dseries:WrongInputArguments")
end

function _dscmp(fn, name)
    function f(I, a, nout)
        o, p = a[1], a[2]
        if p isa Real && o isa MDseries
            return ret(fn.(o.data, Float64(p)))
        elseif o isa Real && p isa MDseries
            return ret(fn.(Float64(o), p.data))
        elseif o isa MDseries && p isa MDseries
            return ret(_bsx(fn, o.data, p.data))
        end
        merr("dseries::$name: wrong types.")
    end
end
for (nm, fn) in (("lt", <), ("gt", >), ("le", <=), ("ge", >=))
    dm!(_dscmp(fn, nm), "dseries", nm)
end
function _ds_eqne(I, a, ne::Bool)
    nm = ne ? "ne" : "eq"
    length(a) == 2 || merr("dseries::$nm: I need exactly two input arguments!")
    o, p = a
    (o isa MDseries && p isa MDseries) || merr("dseries::$nm: Both input arguments must be dseries objects!")
    if nobs(o) != nobs(p) || vobs(o) != vobs(p) || dsfreq(o) != dsfreq(p) || o.dates.time[1] != p.dates.time[1]
        @warn "dseries::$nm: Both input arguments should have the same number of observations, variables, frequencies and initial period!"
        return ne
    end
    ret(ne ? o.data .!= p.data : o.data .== p.data)
end
dm!("dseries", "eq") do I, a, nout; _ds_eqne(I, a, false); end
dm!("dseries", "ne") do I, a, nout; _ds_eqne(I, a, true); end
dm!("dseries", "isequal") do I, a, nout
    length(a) == 2 || merr("dseries::isequal: I need exactly two input arguments!")
    o, p = a
    p isa MDseries || merr("dseries::isequal: Both input arguments must be dseries objects!")
    (nobs(o) == nobs(p) && vobs(o) == vobs(p) && dsfreq(o) == dsfreq(p) && o.dates.time == p.dates.time) || return false
    o.data == p.data || (all(((x, y),) -> x == y || (isnan(x) && isnan(y)) && false, zip(o.data, p.data)))
end

# ---------------------------------------------------------------------- properties and simple methods
function ds_tagstruct(o::MDseries)
    s = MStruct()
    for (k, v) in o.tags
        s[k] = isempty(v) ? Cell(Any[], (0, 0)) : Cell(Any[x === nothing ? zeros(0, 0) : x for x in v], (length(v), 1))
    end
    s
end

function ds_property(o::MDseries, name)
    name == "data" && return ret(o.data)
    name == "name" && return _namecell(o.name)
    name == "tex" && return _namecell(o.tex)
    name == "dates" && return o.dates
    name == "ops" && return _opscell(o.ops)
    name == "tags" && return ds_tagstruct(o)
    name == "nobs" && return Float64(nobs(o))
    name == "vobs" && return Float64(vobs(o))
    name == "init" && return dates_index(o.dates, Any[1.0])
    name == "last" && return dates_index(o.dates, Any[Float64(length(o.dates.time))])
    name == "freq" && return o.dates.freq == 0 ? zeros(0, 0) : Float64(o.dates.freq)
    nothing
end
const DS_PROPS = ("data", "name", "tex", "dates", "ops", "tags", "nobs", "vobs", "init", "last", "freq")

dm!("dseries", "nobs") do I, a, nout; Float64(nobs(a[1])); end
dm!("dseries", "vobs") do I, a, nout; Float64(vobs(a[1])); end
dm!("dseries", "isempty") do I, a, nout; ds_isempty(a[1]); end
dm!("dseries", "numel") do I, a, nout; 1.0; end
dm!("dseries", "copy") do I, a, nout; copy_ds(a[1]); end
dm!("dseries", "frequency") do I, a, nout; Float64(a[1].dates.freq); end
dm!("dseries", "firstdate") do I, a, nout; ds_firstdate(a[1]); end
dm!("dseries", "lastdate") do I, a, nout; ds_lastdate(a[1]); end
dm!("dseries", "length") do I, a, nout
    merr("dseries::length: we do not support the length operator on dseries. Please use 'nobs' or 'vobs'")
end
dm!("dseries", "size") do I, a, nout
    o = a[1]
    shp = size(o.data)
    if length(a) >= 2
        d = as_index_int(a[2], "dimension")
        return d <= 2 ? Float64(shp[d]) : 1.0
    end
    nout <= 1 ? ret(Float64[shp[1] shp[2]]) : Any[Float64(shp[1]), Float64(shp[2])]
end
dm!("dseries", "isnan") do I, a, nout; ret(isnan.(a[1].data)); end
dm!("dseries", "isinf") do I, a, nout; ret(isinf.(a[1].data)); end
dm!("dseries", "isreal") do I, a, nout; true; end
dm!("dseries", "disp", "display", "plot") do I, a, nout; Any[]; end

# ---- first and last observed periods
function _complete_rows(o)
    b = [all(!isnan, @view o.data[i, :]) for i in 1:nobs(o)]
    findall(b)
end
dm!("dseries", "firstobservedperiod") do I, a, nout
    o = a[1]
    c = _complete_rows(o)
    isempty(c) && merr("No overlapping non-NaN data points found in dseries.")
    MDates(dsfreq(o), [o.dates.time[1] + c[1] - 1])
end
dm!("dseries", "lastobservedperiod") do I, a, nout
    o = a[1]
    c = _complete_rows(o)
    isempty(c) && merr("No overlapping non-NaN data points found in dseries.")
    MDates(dsfreq(o), [o.dates.time[1] + c[end] - 1])
end
dm!("dseries", "lastobservedperiods") do I, a, nout
    o = a[1]
    s = MStruct()
    D = MDates(dsfreq(o), Float64[])
    for i in 1:vobs(o)
        last = 0
        for t in 1:nobs(o)
            isnan(o.data[t, i]) && break
            last = t
        end
        d = last == 0 ? MDates(dsfreq(o), Float64[]) : MDates(dsfreq(o), [o.dates.time[last]])
        s[o.name[i]] = d
        append!(D.time, d.time)
    end
    nout > 1 ? Any[s, D] : s
end

# ---- in-place transformations (`f_`) and copies (`f`)
function _inplace!(name, f!; nargs=nothing)
    dm!("dseries", name * "_") do I, a, nout
        f!(I, a[1], a[2:end]...)
        a[1]
    end
    dm!("dseries", name) do I, a, nout
        o = copy_ds(a[1])
        f!(I, o, a[2:end]...)
        o
    end
    dot!("dseries", name, name * "_")
end

_intarg(p, what="Second input argument must be an integer!") = (p isa Real && _isint(p)) ? Int(p) : wrongin(what)

_inplace!("lag", function (I, o, p=1.0)
    p isa Real && p < 0 && wrongin("Second input argument must be non negative! Use lead method instead.")
    k = _intarg(p)
    k == 0 && return
    n = nobs(o)
    o.data = vcat(fill(NaN, min(k, n), vobs(o)), o.data[1:max(n - k, 0), :])
    for i in 1:vobs(o)
        o.ops[i] = isempty(o.ops[i]) ? "lag($k)" : "lag($(o.ops[i]), $k)"
    end
end)
_inplace!("lead", function (I, o, p=1.0)
    p isa Real && p < 0 && wrongin("Second input argument must be non negative! Use lag method instead.")
    k = _intarg(p)
    k == 0 && return
    n = nobs(o)
    o.data = vcat(o.data[min(k + 1, n + 1):n, :], fill(NaN, min(k, n), vobs(o)))
    for i in 1:vobs(o)
        o.ops[i] = isempty(o.ops[i]) ? "lead($k)" : "lead($(o.ops[i]), $k)"
    end
end)
_inplace!("diff", function (I, o)
    d = copy(o.data)
    n = nobs(o)
    if n >= 2
        d[2:end, :] = o.data[2:end, :] .- o.data[1:end-1, :]
    end
    n >= 1 && (d[1, :] .= NaN)
    o.data = d
    _setops!(o, s -> "diff($s)")
end)
_inplace!("log", function (I, o)
    d = copy(o.data)
    d[d .< 0] .= NaN
    o.data = log.(d)
    _setops!(o, s -> "log($s)")
end)
_inplace!("exp", function (I, o)
    o.data = exp.(o.data)
    _setops!(o, s -> "exp($s)")
end)
_inplace!("abs", function (I, o)
    _setops!(o, s -> "abs($s)")
    o.data = abs.(o.data)
end)
_inplace!("flip", function (I, o)
    _setops!(o, s -> "flip($s)")
    o.data = o.data[end:-1:1, :]
end)
_inplace!("round", function (I, o, n=nothing)
    k = (n === nothing || (n isa AbstractMatrix && isempty(n))) ? 0 : Int(n)
    _setops!(o, s -> "round($s, $k)")
    o.data = _mround.(o.data .* 10.0^k) ./ 10.0^k
end)

function _shiftop!(o, name, steps, unit; growth=false)
    f = dsfreq(o)
    if !haskey(steps, f)
        if f in (1, 2, 4, 12)
            yn = Dict(1 => "yearly", 2 => "bi-annual", 4 => "quarterly", 12 => "monthly")[f]
            dserr("dseries::$name: I cannot compute $unit from $yn data!")
        elseif f in (52, 365)
            dserr("dseries::$name: I do not know yet how to compute $unit from $(f == 52 ? "weekly" : "daily") data!")
        end
        dserr("dseries::$name: object inputname(1) has unknown frequency")
    end
    k = steps[f]
    d = copy(o.data)
    n = nobs(o)
    if n > k
        d[k+1:end, :] = growth ? o.data[k+1:end, :] ./ o.data[1:end-k, :] .- 1 : o.data[k+1:end, :] .- o.data[1:end-k, :]
    end
    d[1:min(k, n), :] .= NaN
    o.data = d
    _setops!(o, s -> "$name($s)")
end
for (nm, steps, unit, growth) in (("hdiff", Dict(2 => 1, 4 => 2, 12 => 6), "bi-annual differences", false),
                                  ("qdiff", Dict(4 => 1, 12 => 3), "quaterly differences", false),
                                  ("mdiff", Dict(12 => 1), "monthly differences", false),
                                  ("hgrowth", Dict(2 => 1, 4 => 2, 12 => 6), "bi-annual growth rates", true),
                                  ("qgrowth", Dict(4 => 1, 12 => 3), "quaterly growth rates", true),
                                  ("mgrowth", Dict(12 => 1), "monthly growth rates", true),
                                  ("dgrowth", Dict(365 => 1), "daily growth rates", true))
    _inplace!(nm, let nm = nm, steps = steps, unit = unit, growth = growth
        (I, o) -> _shiftop!(o, nm, steps, unit; growth=growth)
    end)
end
for (nm, growth) in (("ydiff", false), ("ygrowth", true))
    _inplace!(nm, let nm = nm, growth = growth
        function (I, o)
            dsfreq(o) in (52, 365) && _unsup("$nm for weekly or daily data")
            _shiftop!(o, nm, Dict(1 => 1, 2 => 2, 4 => 4, 12 => 12), "yearly"; growth=growth)
        end
    end)
end

_cum_inplace(o, f, name) = begin
    d = o.data
    idx = findfirst(i -> all(!isnan, @view d[i, :]), 1:nobs(o))
    idx0 = idx === nothing ? 0 : idx
    common = true
    if idx0 == 0
        any(!isnan, d) && (common = false)
    elseif idx0 > 1
        any(!isnan, @view d[1:idx0-1, :]) && (common = false)
    end
    nd = copy(d)
    if idx0 == 1
        nd = f(d; dims=1)
    elseif common
        idx0 > 0 && (nd[idx0:end, :] = f(d[idx0:end, :]; dims=1))
    else
        # cumsumnan / cumprodnan: NaNs are skipped
        for j in 1:size(d, 2)
            acc = name == "cumsum" ? 0.0 : 1.0
            seen = false
            for t in 1:size(d, 1)
                if isnan(d[t, j])
                    nd[t, j] = NaN
                else
                    acc = name == "cumsum" ? acc + d[t, j] : acc * d[t, j]
                    nd[t, j] = acc
                end
            end
        end
    end
    o.data = nd
end
for (nm, f) in (("cumsum", cumsum), ("cumprod", cumprod))
    _inplace!(nm, let nm = nm, f = f
        function (I, o, rest...)
            isempty(rest) || _unsup("$nm with a base period or value")
            _cum_inplace(o, f, nm)
            _setops!(o, s -> "$nm($s)")
        end
    end)
end

dm!("dseries", "mean") do I, a, nout
    o = a[1]
    geo = length(a) > 1 && truth(a[2])
    ret(geo ? prod(o.data; dims=1) .^ (1 / nobs(o)) : mean(o.data; dims=1))
end
dm!("dseries", "std") do I, a, nout
    o = a[1]
    length(a) > 1 && truth(a[2]) && _unsup("geometric std")
    ret(std(o.data; dims=1))
end
dot!("dseries", "mean", "std")

function _center!(I, o, geometric=false)
    g = geometric isa Bool ? geometric : truth(geometric)
    m = mean(o.data; dims=1)
    o.data = g ? o.data ./ m : o.data .- m
    _setops!(o, s -> "center($s, $(g ? 1 : 0))")
end
_inplace!("center", _center!)

_inplace!("detrend", function (I, o, model=0.0)
    (model isa Real && _isint(model)) || dserr("dseries::detrend: Second argument must be a positive integer scalar!")
    m = Int(model)
    if m == 0
        o.data = o.data .- mean(o.data; dims=1)
    else
        n = nobs(o)
        x = ones(n, m + 1)
        x[:, 2] = 1:n
        for c in 3:m+1
            x[:, c] = x[:, c-1] .* x[:, 2]
        end
        o.data = o.data - x * (x \ o.data)
    end
    _setops!(o, s -> "detrend($s, $m)")
end)

# ---- filters
function hp_trend(y::AbstractMatrix, s::Real)
    T = size(y, 1)
    D = zeros(T, T)
    for i in 1:T
        i - 2 >= 1 && (D[i, i-2] = s)
        i - 1 >= 1 && (D[i, i-1] = -4s)
        D[i, i] = 1 + 6s
        i + 1 <= T && (D[i, i+1] = -4s)
        i + 2 <= T && (D[i, i+2] = s)
    end
    D[1, 1] = 1 + s; D[T, T] = D[1, 1]
    D[1, 2] = -2s; D[2, 1] = D[1, 2]; D[T-1, T] = D[1, 2]; D[T, T-1] = D[1, 2]
    D[2, 2] = 1 + 5s; D[T-1, T-1] = D[2, 2]
    D \ y
end
function _hp!(o, lambda, cycle, name)
    if lambda !== nothing && !(lambda isa AbstractMatrix && isempty(lambda))
        scalar(lambda) <= 0 && dserr("dseries::$name: Lambda must be a positive integer!")
    end
    lam = (lambda === nothing || (lambda isa AbstractMatrix && isempty(lambda))) ? nothing : scalar(lambda)
    nobs(o) >= 4 || _unsup("hp filter of fewer than four observations")
    tr = hp_trend(o.data, lam === nothing ? 1600.0 : lam)
    o.data = cycle ? o.data .- tr : tr
end
_inplace!("hpcycle", function (I, o, lambda=nothing)
    _hp!(o, lambda, true, "hpcycle")
    ls = lambda === nothing || (lambda isa AbstractMatrix && isempty(lambda)) ? "[]" : _num2str(I, lambda)
    _setops!(o, s -> "hpcycle($s, $ls)")
end)
_inplace!("hptrend", function (I, o, lambda=nothing)
    _hp!(o, lambda, false, "hptrend")
    ls = lambda === nothing || (lambda isa AbstractMatrix && isempty(lambda)) ? "[]" : _num2str(I, lambda)
    _setops!(o, s -> "hptrend($s, $ls)")
end)
_inplace!("baxter_king_filter", function (I, o, hf=nothing, lf=nothing, K=nothing)
    emp(x) = x === nothing || (x isa AbstractMatrix && isempty(x))
    Kk = emp(K) ? 12 : Int(scalar(K))
    low = emp(lf) ? 32.0 : scalar(lf)
    high = emp(hf) ? 6.0 : scalar(hf)
    if !emp(hf) && emp(lf)
        high < 2 && dserr("dseries::baxter_king_filter: Second argument must be greater than 2!")
        high > low && dserr("dseries::baxter_king_filter: Second argument must be smaller than the third argument!")
    end
    h = 2pi / high; l = 2pi / low
    w = zeros(Kk + 1)
    lp = collect(1.0:Kk)
    w[2:end] = (sin.(lp .* h) .- sin.(lp .* l)) ./ (lp .* pi)
    w[1] = (h - l) / pi
    swc = low > 1000 ? 1.0 : 0.0
    sw = w[1] + 2 * sum(w[2:end])
    w = w .+ (swc - sw) / (2Kk + 1)
    w = vcat(reverse(w[2:end]), w)
    n = nobs(o)
    tmp = zeros(size(o.data))
    for t in Kk+1:n-Kk
        tmp[t, :] = w' * o.data[t-Kk:t+Kk, :]
    end
    o.data = tmp[Kk+1:end-Kk, :]
    init = o.dates.time[1] + Kk
    o.dates = mkdates(dsfreq(o), init .+ (0:nobs(o)-1))
    s1, s2, s3 = _num2str(I, high), _num2str(I, low), _num2str(I, Float64(Kk))
    _setops!(o, s -> "baxter_king_filter($s, $s1, $s2, $s3)")
end)

# ---------------------------------------------------------------------- extraction
function _extract_names(o::MDseries, specs::Vector{String})
    names = String[]
    for spec in specs
        ats = findall('@', spec)
        isodd(length(ats)) && merr("dseries::extract: (Implicit loops) The number of @ symbols must be even!")
        isloop = !isempty(ats)
        hasbr = !isempty(findall('[', spec)) && !isempty(findall(']', spec)) && first(findall('[', spec)) == 1 && last(findall(']', spec)) == length(spec)
        wild = false
        if !hasbr && occursin('~', spec)
            wild = true
            hasbr = true
            spec = "[" * replace(spec, "~" => "\\w*") * "]"
        end
        if !hasbr && (occursin('[', spec) || occursin(']', spec))
            merr("dseries::extract: Square brackets are not allowed, unless regexp are used to select variables!")
        end
        hasbr && isloop && merr("dseries::extract: You cannot use implicit loops and regular expressions in the same rule!")
        if isloop
            append!(names, _loops(o.name, ats, spec))
        elseif hasbr
            re = Regex(spec[2:end-1])
            matched = String[]
            for nm in o.name
                m = match(re, nm)
                m === nothing || push!(matched, m.match)
            end
            lst = sort(unique(String[nm for nm in o.name if nm in matched]))
            if isempty(lst)
                @warn wild ? "dseries::extact: The wildcard expression '$(replace(spec[2:end-1], "\\w*" => "*"))' did not match any variable name!" :
                             "dseries::extact: The regular expression '[$(spec[2:end-1])]' did not match any variable name!"
            end
            append!(names, lst)
        else
            push!(names, spec)
        end
    end
    String[strip(n) for n in names]
end
function _loops(onames, ats, spec)
    nloops = length(ats) ÷ 2
    nloops > 2 && merr("dseries::loops: (Implicit loops) Cannot unroll more than two implicit loops!")
    exprs = [spec[ats[2i+1]+1:ats[2i+2]-1] for i in 0:nloops-1]
    any(e -> !occursin(',', e), exprs) && merr("dseries::loops: (Implicit loops) Wrong syntax!")
    toks(e) = String.(split(e, ','; keepempty=false))
    out = String[]
    if nloops == 1
        for t in toks(exprs[1])
            c = spec[1:ats[1]-1] * t * spec[ats[2]+1:end]
            c in onames || merr("dseries::loops: (Implicit loops) Variable '$c' does not exist in dseries object 'inputname(1)'!")
            push!(out, c)
        end
    else
        for t1 in toks(exprs[1]), t2 in toks(exprs[2])
            c = spec[1:ats[1]-1] * t1 * spec[ats[2]+1:ats[3]-1] * t2 * spec[ats[4]+1:end]
            c in onames || merr("dseries::loops: (Implicit loops) Variable '$c' does not exist in dseries object 'inputname(1)'!")
            push!(out, c)
        end
    end
    out
end

function _select_vars(o::MDseries, idx::Vector{Int})
    p = MDseries()
    p.data = o.data[:, idx]
    p.dates = o.dates
    p.name = o.name[idx]
    p.tex = o.tex[idx]
    p.ops = o.ops[idx]
    p.tags = [k => v[idx] for (k, v) in o.tags]
    p
end
function ds_extract(o::MDseries, specs::Vector{String})
    names = _extract_names(o, specs)
    idx = Int[]
    for nm in names
        i = findfirst(==(nm), o.name)
        i === nothing && merr("dseries::extract: Variable $nm is not a member of inputname(1)!")
        push!(idx, i)
    end
    _select_vars(o, idx)
end
dm!("dseries", "extract") do I, a, nout
    all(x -> x isa AbstractString, a[2:end]) || merr("dseries::extract: arguments must be char arrays")
    ds_extract(a[1], String[x for x in a[2:end]])
end

function _dates_bounds_check(o, D)
    mn, mx = minimum(D.time), maximum(D.time)
    D.freq == o.dates.freq || throw(MatlabError("Input arguments must have common frequency!", "dates:lt:ArgCheck"))
    mn < o.dates.time[1] && merr("dseries::subsref: Indices are out of bounds! Subsample cannot start before $(date2string(o.dates.time[1], o.dates.freq)).")
    mx > o.dates.time[end] && merr("dseries::subsref: Indices are out of bounds! Subsample cannot end after $(date2string(o.dates.time[end], o.dates.freq)).")
end

"`ds(args)`."
function ds_index(I, o::MDseries, args)
    isempty(args) && merr("dseries::subsref: Invalid syntax")
    x = args[1]
    if x isa Bool || (x isa AbstractMatrix{Bool})
        v = x isa Bool ? [x] : vec(x)
        isvec = x isa Bool || min(size(x)...) == 1
        isvec || merr("dseries::subsref: expected input is a vector of logicals.")
        n = length(v)
        (n == nobs(o) || n == vobs(o)) || merr("dseries::subsref: length of the vector of logicals must match the number of variables or periods.")
        sel_p = n == nobs(o)
        if nobs(o) == vobs(o)
            sel_p = !(x isa Bool) && size(x, 2) == 1 && size(x, 1) != 1
        end
        if sel_p
            r = copy_ds(o)
            r.data = o.data[v, :]
            r.dates = o.dates[v]
            return r
        end
        return _select_vars(o, findall(v))
    elseif x isa AbstractString && !isstringdate(x)
        ds_isempty(o) || merr("dseries::subsref: dseries object is not empty!")
        return ds_ctor(I, Any[x])
    elseif x isa Real && _isint(x)
        k = Int(x)
        return k > 0 ? OBJM["dseries"]["lead"](I, Any[o, Float64(k)], 1) : k < 0 ? OBJM["dseries"]["lag"](I, Any[o, Float64(-k)], 1) : o
    elseif x isa MDates || isstringdate(x)
        D = x isa MDates ? x : dates_ctor(Any[x])
        isempty(D.time) && merr("Index exceeds matrix dimensions")
        _dates_bounds_check(o, D)
        common = sort!(unique([t for t in o.dates.time if t in D.time]))
        pos = [Int(t - o.dates.time[1]) + 1 for t in common]
        r = copy_ds(o)
        r.data = o.data[pos, :]
        r.dates = MDates(o.dates.freq, o.dates.time[pos])
        return r
    elseif (x isa AbstractMatrix{Float64} || x isa Real) && ds_isempty(o)
        a = farr(x)
        r = copy_ds(o)
        r.dates = isempty(o.dates.time) ? _range(MDates(1, [1.0]), size(a, 1)) : _range(MDates(o.dates.freq, o.dates.time[1:1]), size(a, 1))
        r.data = copy(a)
        r.name = default_names(size(a, 2))
        r.tex = [name2tex(s) for s in r.name]
        r.ops = fill("", length(r.name))
        return r
    elseif x isa AbstractMatrix{Float64} || x isa Real
        merr("dseries::subsref: It is not possible to populate a non empty dseries object!")
    end
    merr("dseries::subsref: Invalid syntax")
end

"`ds{args}`."
function ds_brace(I, o::MDseries, args)
    if all(x -> x isa AbstractString, args)
        return ds_extract(o, String[x for x in args])
    elseif length(args) == 1 && args[1] isa Union{Real,AbstractMatrix{Float64}} && all(_isint, args[1])
        idx = Int.(vec(farr(args[1])))
        (isempty(idx) || maximum(idx) > vobs(o) || minimum(idx) < 1) && merr("dseries::subsref: Indices are out of bounds!")
        return _select_vars(o, idx)
    end
    merr("dseries::subsref: Invalid syntax")
end

"`ds.name` for a variable of the dseries."
function ds_variable(o::MDseries, name)
    i = findfirst(==(name), o.name)
    i === nothing && merr("dseries::subsref: Unknown public method, public member or variable!")
    _select_vars(o, [i])
end

# ---------------------------------------------------------------------- subsample, merge, concatenation
dm!("dseries", "subsample") do I, a, nout
    o = a[1]
    length(a) == 1 && return copy_ds(o)
    function todate(x)
        x isa MDates && return x
        isstringdate(x) && return dates_ctor(Any[x])
        merr("the arguments to subsample must be dates")
    end
    d1 = todate(a[2])
    d2 = length(a) >= 3 ? todate(a[3]) : d1
    length(a) >= 3 && d2.time[1] < d1.time[1] && merr("the second date must be greater than or equal to the first date")
    d1.time[1] < o.dates.time[1] && merr("the first argument must be greater than or equal to $(date2string(o.dates.time[1], o.dates.freq))")
    d2.time[1] > o.dates.time[end] && merr("the second argument must be less than or equal to $(date2string(o.dates.time[end], o.dates.freq))")
    i1 = Int(d1.time[1] - o.dates.time[1]) + 1
    i2 = Int(d2.time[1] - o.dates.time[1]) + 1
    p = MDseries()
    p.data = o.data[i1:i2, :]
    p.name = copy(o.name); p.tex = copy(o.tex); p.ops = copy(o.ops)
    p.dates = MDates(o.dates.freq, o.dates.time[i1:i2])
    p.tags = [k => copy(v) for (k, v) in o.tags]
    p
end
dot!("dseries", "subsample", "extract")

function ds_merge(o::MDseries, p::MDseries, legacy::Bool=false)
    ds_isempty(o) && return p
    ds_isempty(p) && return o
    dsfreq(o) == dsfreq(p) || merr("dseries::merge: Cannot merge inputname(1) and inputname(2) (frequencies are different)!")
    allnames = vcat(o.name, p.name)
    # unique(..., 'last'): sorted names, index of the last occurrence
    lastidx = Dict{String,Int}()
    for (i, nm) in enumerate(allnames)
        lastidx[nm] = i
    end
    qnames = sort!(collect(keys(lastidx)))
    IBC = [lastidx[nm] for nm in qnames]
    q = MDseries()
    q.name = qnames
    q.tex = vcat(o.tex, p.tex)[IBC]
    q.ops = vcat(o.ops, p.ops)[IBC]
    otn, ptn = first.(o.tags), first.(p.tags)
    for tn in sort!(unique(vcat(otn, ptn)))
        ov = get(Dict(o.tags), tn, fill(nothing, vobs(o)))
        pv = get(Dict(p.tags), tn, fill(nothing, vobs(p)))
        push!(q.tags, tn => Any[vcat(ov, pv)[IBC]...])
    end
    common = legacy ? String[] : sort!(intersect(o.name, p.name))
    first_o, first_p = o.dates.time[1], p.dates.time[1]
    if first_o >= first_p
        diff = Int(first_o - first_p)
        qn = max(nobs(o) + diff, nobs(p))
        Z1 = vcat(fill(NaN, diff, vobs(o)), o.data)
        qn > nobs(o) + diff && (Z1 = vcat(Z1, fill(NaN, qn - nobs(o) - diff, vobs(o))))
        Z2 = p.data
        qn > nobs(p) && (Z2 = vcat(Z2, fill(NaN, qn - nobs(p), vobs(p))))
        Z = hcat(Z1, Z2)
        q.data = Z[:, IBC]
        for nm in common
            jo = findfirst(==(nm), o.name); jp = findfirst(==(nm), p.name); jq = findfirst(==(nm), q.name)
            for t in 1:qn
                isnan(q.data[t, jq]) && !isnan(Z1[t, jo]) && isnan(Z2[t, jp]) && (q.data[t, jq] = Z1[t, jo])
            end
        end
        qinit = first_p
    else
        diff = Int(first_p - first_o)
        qn = max(nobs(p) + diff, nobs(o))
        Z1 = vcat(fill(NaN, diff, vobs(p)), p.data)
        qn > nobs(p) + diff && (Z1 = vcat(Z1, fill(NaN, qn - nobs(p) - diff, vobs(p))))
        Z2 = o.data
        qn > nobs(o) && (Z2 = vcat(Z2, fill(NaN, qn - nobs(o), vobs(o))))
        Z = hcat(Z2, Z1)
        q.data = Z[:, IBC]
        for nm in common
            jo = findfirst(==(nm), o.name); jp = findfirst(==(nm), p.name); jq = findfirst(==(nm), q.name)
            for t in 1:qn
                isnan(q.data[t, jq]) && isnan(Z1[t, jp]) && !isnan(Z2[t, jo]) && (q.data[t, jq] = Z2[t, jo])
            end
        end
        qinit = first_o
    end
    q.dates = mkdates(dsfreq(o), qinit .+ (0:size(q.data, 1)-1))
    q
end
dm!("dseries", "merge") do I, a, nout
    (length(a) >= 2 && a[1] isa MDseries && a[2] isa MDseries) || merr("dseries::merge: Both inputs must be dseries objects!")
    ds_merge(a[1], a[2], length(a) >= 3 && truth(a[3]))
end
dot!("dseries", "merge")

function _concat2(b::MDseries, c::MDseries)
    ds_isempty(b) && return c
    ds_isempty(c) && return b
    common = intersect(b.name, c.name)
    isempty(common) || merr("dseries::horzcat: I cannot concatenate dseries objects with common variable names ($(join(common, ", ")))!")
    dsfreq(b) == dsfreq(c) || merr("dseries::horzcat: All time series objects must have common frequency!")
    a = MDseries()
    a.ops = vcat(b.ops, c.ops)
    a.name = vcat(b.name, c.name)
    a.tex = vcat(b.tex, c.tex)
    btn, ctn = first.(b.tags), first.(c.tags)
    for tn in sort!(unique(vcat(btn, ctn)))
        if tn in btn && tn in ctn
            bv = Dict(b.tags)[tn]
            push!(a.tags, tn => Any[vcat(bv, bv)...])           # (sic: Dynare concatenates b's tags twice)
        elseif tn in btn
            push!(a.tags, tn => Any[vcat(Dict(b.tags)[tn], fill(nothing, vobs(c)))...])
        else
            push!(a.tags, tn => Any[vcat(fill(nothing, vobs(b)), Dict(c.tags)[tn])...])
        end
    end
    bd, cd = b.data, c.data
    if nobs(b) == nobs(c) && b.dates.time[1] == c.dates.time[1]
        a.data = hcat(bd, cd)
        a.dates = b.dates
    else
        fb, fc = b.dates.time[1], c.dates.time[1]
        nb, nc = nobs(b), nobs(c)
        if fb <= fc
            fb < fc && (cd = vcat(fill(NaN, Int(fc - fb), vobs(c)), cd))
        else
            bd = vcat(fill(NaN, Int(fb - fc), vobs(b)), bd)
        end
        blast, clast = fb + nb, fc + nc
        if blast < clast
            bd = vcat(bd, fill(NaN, Int(clast - blast), vobs(b)))
        elseif blast > clast
            cd = vcat(cd, fill(NaN, Int(blast - clast), vobs(c)))
        end
        a.data = hcat(bd, cd)
        t = sort!(unique(vcat(b.dates.time, c.dates.time,
                              (maximum(c.dates.time) < minimum(b.dates.time) ? collect(maximum(c.dates.time):minimum(b.dates.time)) :
                               maximum(b.dates.time) < minimum(c.dates.time) ? collect(maximum(b.dates.time):minimum(c.dates.time)) : Float64[]))))
        a.dates = MDates(dsfreq(b), t)
    end
    a
end
function ds_horzcat(args)
    all(x -> x isa MDseries, args) || merr("dseries::horzcat: all elements must be dseries objects")
    isempty(args) && return MDseries()
    length(args) == 1 && return args[1]
    o = _concat2(copy_ds(args[1]), copy_ds(args[2]))
    for x in args[3:end]
        o = _concat2(o, copy_ds(x))
    end
    o
end
dm!("dseries", "horzcat") do I, a, nout; ds_horzcat(a); end
function ds_vertcat(args)
    all(x -> x isa MDseries, args) || merr("dseries::vertcat: all elements must be dseries objects")
    isempty(args) && return MDseries()
    o = copy_ds(args[1])
    for c in args[2:end]
        b = o
        dsfreq(b) == dsfreq(c) || merr("dseries::vertcat: Frequencies must be common!")
        vobs(b) == vobs(c) || merr("dseries::vertcat: Number of variables must be common!")
        cd = c.data
        if b.name != c.name
            idx = [findfirst(==(nm), c.name) for nm in b.name]
            any(isnothing, idx) && merr("dseries::vertcat: Variables must be common!")
            cd = c.data[:, idx]
        end
        d = copy_ds(b)
        d.data = vcat(b.data, cd)
        d.dates = dates_horzcat([b.dates, c.dates])
        o = d
    end
    o
end
dm!("dseries", "vertcat") do I, a, nout; ds_vertcat(a); end

dm!("dseries", "align") do I, a, nout
    o, p = align_copies(a[1], a[2])
    Any[o, p]
end
dm!("dseries", "align_") do I, a, nout
    align!(a[1], a[2])
    Any[a[1], a[2]]
end

# ---------------------------------------------------------------------- insert, remove, rename, tags
function _remove_idx!(o::MDseries, io::Vector{Int})
    keep = trues(vobs(o)); keep[io] .= false
    o.data = o.data[:, keep]
    o.name = o.name[keep]; o.tex = o.tex[keep]; o.ops = o.ops[keep]
    o.tags = [k => v[keep] for (k, v) in o.tags]
    o
end
function _remove!(I, o::MDseries, a)
    if a isa AbstractString
        id = findfirst(==(a), o.name)
        id === nothing || _remove_idx!(o, [id])
    elseif a isa Cell && all(x -> x isa AbstractString, a.items)
        io = Int[]
        for nm in a.items
            i = findfirst(==(nm), o.name)
            i === nothing ? @warn("Variable $nm is not a member of the dseries object.") : push!(io, i)
        end
        _remove_idx!(o, io)
    else
        merr("Unexpected type.")
    end
    o
end
dm!("dseries", "remove_") do I, a, nout; _remove!(I, a[1], a[2]); end
dm!("dseries", "remove") do I, a, nout; _remove!(I, copy_ds(a[1]), a[2]); end
dm!("dseries", "pop_") do I, a, nout
    o = a[1]
    id = length(a) < 2 ? vobs(o) : findfirst(==(a[2]), o.name)
    (id === nothing || id == 0) && return nout > 1 ? Any[o, 0.0] : o
    _remove_idx!(o, [id])
    nout > 1 ? Any[o, Float64(id)] : o
end
dm!("dseries", "pop") do I, a, nout
    o = copy_ds(a[1])
    id = length(a) < 2 ? vobs(o) : findfirst(==(a[2]), o.name)
    (id === nothing || id == 0) && return nout > 1 ? Any[o, 0.0] : o
    _remove_idx!(o, [id])
    nout > 1 ? Any[o, Float64(id)] : o
end
dot!("dseries", "remove", "remove_", "pop", "pop_")

function _rename!(o::MDseries, a)
    ds_isempty(o) && merr("dseries::rename: Cannot rename variable(s) because the object is empty!")
    if length(a) == 1
        old = a[1]
        if old isa AbstractString && vobs(o) == 1
            o.name = [String(old)]
        elseif old isa Cell && all(x -> x isa AbstractString, old.items) && vobs(o) == length(old.items)
            o.name = String[x for x in old.items]
        else
            merr("dseries::rename: Cannot rename variables (wrong input types or wrong dimensions).")
        end
        return o
    end
    old, new = a[1], a[2]
    if old isa AbstractString && new isa AbstractString
        id = findfirst(==(old), o.name)
        id === nothing && merr("dseries::rename: Variable to be replaced ($old) is unknown")
        o.name[id] = new
    elseif old isa Cell && new isa Cell
        length(old.items) == length(new.items) || merr("dseries::rename: Input cell arrays must have the same number of elements.")
        unk = setdiff(String[x for x in old.items], o.name)
        isempty(unk) || merr("dseries::rename: Some variable ($(join(unk, " "))) to be replaced are unknown.")
        for (x, y) in zip(old.items, new.items)
            o.name[findfirst(==(x), o.name)] = y
        end
    else
        merr("dseries::rename: Input arguments must be a pair of char arrays or a pair of cell arrays.")
    end
    o
end
dm!("dseries", "rename_") do I, a, nout; _rename!(a[1], a[2:end]); end
dm!("dseries", "rename") do I, a, nout; _rename!(copy_ds(a[1]), a[2:end]); end
dot!("dseries", "rename", "rename_", "set_names", "tex_rename", "tex_rename_", "tag", "fill_")
dm!("dseries", "set_names") do I, a, nout
    o = a[1]
    vobs(o) == length(a) - 1 || merr("dseries::set_names: The number of variables in first input does not match the number of declared names!")
    for i in 1:vobs(o)
        x = a[i+1]
        (x isa AbstractMatrix && isempty(x)) || (o.name[i] = String(x))
    end
    o
end
function _tex_rename!(o, a)
    if length(a) == 1
        new = a[1]; idname = 1
    else
        name, new = a[1], a[2]
        idname = findfirst(==(name), o.name)
        idname === nothing && merr("dseries::tex_rename: Variable $name is unknown in dseries object inputname(1)!")
    end
    if new isa Cell
        length(new.items) == vobs(o) || merr("dseries::tex_rename: third input argument (newtexname) name must either be a string or a cell array of strings with the same number of entries as varibles in the dseries")
        o.tex = String[x for x in new.items]
    else
        o.tex[idname] = String(new)
    end
    o
end
dm!("dseries", "tex_rename_") do I, a, nout; _tex_rename!(a[1], a[2:end]); end
dm!("dseries", "tex_rename") do I, a, nout; _tex_rename!(copy_ds(a[1]), a[2:end]); end
dm!("dseries", "tag") do I, a, nout
    o = a[1]
    if length(a) < 3
        tn = String(a[2])
        any(p -> p.first == tn, o.tags) || push!(o.tags, tn => Any[nothing for _ in 1:vobs(o)])
    else
        tn, vn = String(a[2]), String(a[3])
        k = findfirst(p -> p.first == tn, o.tags)
        k === nothing && merr("dseries::tag: Tag name $tn is unknown!")
        id = findfirst(==(vn), o.name)
        id === nothing && merr("dseries::tag: Variable $vn is unknown!")
        o.tags[k].second[id] = a[4]
    end
    o
end
dm!("dseries", "resetops") do I, a, nout
    o = a[1]
    o.ops = a[2] isa Cell ? String[x isa AbstractString ? String(x) : "" for x in a[2].items] : String[]
    o
end

dm!("dseries", "insert") do I, a, nout
    o = copy_ds(a[1]); p = a[3-1]
    ids = vec(farr(a[3]))
    common = intersect(o.name, p.name)
    isempty(common) || merr("dseries::insert: Variable(s) $(join(common, " ")) already exist in dseries object!")
    dsfreq(o) == dsfreq(p) || merr("dseries::insert: dseries objects must have common frequencies!")
    o, p = align_copies(o, p)
    n = length(ids)
    if n > 1
        perm = sortperm(ids)
        ids = ids[perm]
        p.data = p.data[:, perm]; p.name = p.name[perm]; p.tex = p.tex[perm]; p.ops = p.ops[perm]
    end
    isempty(o.tags) && isempty(p.tags) || _unsup("insert with tags")
    for i in 1:n
        id = Int(ids[i]) + (i - 1)
        id = clamp(id, 1, vobs(o) + 1)
        o.data = hcat(o.data[:, 1:id-1], p.data[:, i], o.data[:, id:end])
        insert!(o.name, id, p.name[i]); insert!(o.tex, id, p.tex[i]); insert!(o.ops, id, p.ops[i])
    end
    o
end
dot!("dseries", "insert")

# ---------------------------------------------------------------------- save
dm!("dseries", "save") do I, a, nout
    o = a[1]
    base = length(a) >= 2 && !(a[2] isa AbstractMatrix) ? String(a[2]) : "dynare_series"
    fmt = length(a) >= 3 && !(a[3] isa AbstractMatrix) ? String(a[3]) : "mat"
    _need_exec("dseries save")
    dir = I.dir === nothing ? pwd() : I.dir
    path(ext) = isabspath(base) ? base * ext : joinpath(dir, base * ext)
    if fmt == "csv"
        p = path(".csv")
        isfile(p) && cp(p, replace(p, r"\.csv$" => ".old.csv"); force=true)
        open(p, "w") do io
            print(io, ",", join(o.name, ","), "\n")
            for t in 1:nobs(o)
                print(io, date2string(o.dates.time[t], o.dates.freq))
                for j in 1:vobs(o)
                    print(io, ", ", lpad(_fmt_g15(o.data[t, j]), 15))
                end
                print(io, "\n")
            end
        end
    elseif fmt == "mat"
        p = path(".mat")
        isfile(p) && cp(p, replace(p, r"\.mat$" => ".old.mat"); force=true)
        d = Dict{String,Any}("INIT__" => date2string(o.dates.time[1], o.dates.freq), "FREQ__" => Float64(dsfreq(o)),
                             "NAMES__" => Any[o.name...], "TEX__" => Any[o.tex...],
                             "OPS__" => Any[isempty(x) ? zeros(0, 0) : x for x in o.ops], "DATA__" => o.data)
        MAT.matwrite(p, d)
    else
        _unsup("dseries save in format $fmt")
    end
    Any[]
end
dot!("dseries", "save")
function _fmt_g15(x)
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    Printf.format(Printf.Format("%.8g"), x)
end
