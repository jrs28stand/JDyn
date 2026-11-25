# Interpreter parity with Octave 8.4 on pydyn's case sets pydyn_fn / pydyn_files / pydyn_mi (test/ref_gaps/cases):
# regexp (lo-regexp.cc's rules on top of PCRE), intersect, cell2mat, chol, sparse, char(), read-only file queries confined to
# the model's folder. Loaded after builtins_extra.jl: these definitions replace the earlier ones of the same name.
# Refusals (Unsupported) wherever Octave's result is not reproduced; nothing is guessed.

# ---------------------------------------------------------------------- regexp / regexpi
const _RX_OPTS = Set(["once", "matchcase", "ignorecase", "stringanchors", "lineanchors", "literalspacing", "freespacing",
                      "noemptymatch", "emptymatch", "dotall", "dotexceptnewline", "start", "end", "tokenextents", "match",
                      "tokens", "names", "split"])
const _RX_OUT = ("start", "end", "tokenextents", "match", "tokens", "names", "split")
const _RX_ESC_OK = Set("dDwWsSbBAntrfa")

"Refuses the PCRE constructs whose meaning could differ between Julia's PCRE2 and Octave's (ALT_BSUX escapes, \\< and \\>, `{,n}`)."
function _rx_check(pat::AbstractString)
    isascii(pat) || _unsup("regexp with non-ASCII characters", "function")
    cs = collect(pat)
    i = 1
    n = length(cs)
    in_cls = false
    while i <= n
        c = cs[i]
        if c == '\\'
            i == n && _unsup("regexp: pattern ends with a backslash", "function")
            d = cs[i+1]
            if d in _RX_ESC_OK
                i += 2
            elseif d == 'x'
                (i + 3 <= n && all(isxdigit, cs[i+2:i+3])) || _unsup("regexp: \\x escape that is not followed by two hexadecimal digits", "function")
                i += 4
            elseif d in "123456789" && !in_cls
                (i + 2 <= n && isdigit(cs[i+2])) && _unsup("regexp: back-reference with several digits", "function")
                i += 2
            elseif d in "<>"
                _unsup("regexp: \\< and \\> word boundaries", "function")
            elseif !isletter(d) && !isdigit(d)
                i += 2
            else
                _unsup("regexp: the escape \\$d (PCRE and Julia's PCRE2 differ or it is not supported)", "function")
            end
            continue
        end
        if in_cls
            in_cls = !(c == ']')
        elseif c == '['
            in_cls = true
            i += 1
            i <= n && cs[i] == '^' && (i += 1)
            i <= n && cs[i] == ']' && (i += 1)              # a leading ] is a literal
            continue
        elseif c == '{' && i < n && cs[i+1] == ','
            _unsup("regexp: {,n} (a literal in the PCRE of Octave 8.4, a quantifier in newer PCRE2)", "function")
        end
        i += 1
    end
    nothing
end

mutable struct _RxOpts
    once::Bool; dotall::Bool; icase::Bool; multiline::Bool; extended::Bool; emptymatch::Bool
end

function _rx_matches(rx::Regex, s::String, o::_RxOpts)
    out = RegexMatch[]
    n = ncodeunits(s)
    idx = 0                                  # 0-based, as lo-regexp.cc
    while true
        m = idx <= n ? match(rx, s, idx + 1) : nothing
        m === nothing && break
        st = m.offset - 1
        en = st + ncodeunits(m.match)
        if en <= st && !o.emptymatch         # zero-length match: skip to the next character
            idx = en + 1
            idx < n ? continue : break
        end
        push!(out, m)
        if en <= st
            idx = en + 1
            idx <= n && continue
        else
            idx = en
        end
        (o.once || idx >= n) && break
    end
    out
end

# tokens of one match: (strings, extents rows, named tokens), lo-regexp.cc: a group is dropped if it did not take part, ended at
# position 0, or has the span of the group before it; no groups at all: the whole match is the one token (without extents)
function _rx_tokens(m::RegexMatch, ng::Int, names::Dict{Int,String})
    toks = String[]; ext = Float64[]; named = Dict{String,String}()
    ps = pe = -1
    for g in 1:ng
        c = m.captures[g]
        if c === nothing
            s0 = e0 = -1
        else
            s0 = m.offsets[g] - 1; e0 = s0 + ncodeunits(c)
        end
        if s0 >= 0 && e0 > 0 && (g == 1 || (s0, e0) != (ps, pe))
            push!(toks, String(c))
            push!(ext, s0 + 1.0, Float64(e0))
            haskey(names, g) && (named[names[g]] = String(c))
        end
        ps, pe = s0, e0
    end
    ng == 0 && return (String[m.match], zeros(0, 2), named)
    (toks, reshape(ext, 2, :)', named)
end

function _rx_one(s::String, pat::String, o::_RxOpts, want::Set{String})
    _rx_check(pat)
    isascii(s) || _unsup("regexp on non-ASCII text", "function")
    flags = (o.dotall ? "s" : "") * (o.icase ? "i" : "") * (o.multiline ? "m" : "") * (o.extended ? "x" : "")
    rx = try
        Regex(pat, flags)
    catch e
        e isa ErrorException ? merr("regexp: " * replace(e.msg, "PCRE compilation error: " => "")) : rethrow()
    end
    names = Base.PCRE.capture_names(rx.regex)
    ng = Int(Base.PCRE.info(rx.regex, Base.PCRE.INFO_CAPTURECOUNT, UInt32))
    nmlist = [names[k] for k in sort(collect(keys(names)))]
    ms = _rx_matches(rx, s, o)
    if !isempty(nmlist) && "names" in want && length(ms) != 1
        _unsup("regexp: 'names' with named tokens and zero or several matches (struct arrays)", "function")
    end
    o.once && o.emptymatch && _unsup("regexp: 'once' together with 'emptymatch'", "function")
    o.once && ng == 0 && "tokens" in want && _unsup("regexp: 'once' tokens of a pattern without groups (Octave's result is not the usual one)", "function")
    res = Dict{String,Any}()
    if o.once
        if !isempty(ms)
            m = ms[1]
            toks, ext, named = _rx_tokens(m, ng, names)
            isempty(toks) && "tokenextents" in want && _unsup("regexp: 'once' token extents without tokens", "function")
            res["start"] = Float64(m.offset); res["end"] = Float64(m.offset - 1 + ncodeunits(m.match))
            res["match"] = String(m.match)
            res["tokens"] = Cell(Any[toks...], (1, length(toks)))
            res["tokenextents"] = ret(Matrix{Float64}(ext))
            st = MStruct(); for k in nmlist; st[k] = get(named, k, ""); end
            res["names"] = st
            res["split"] = Cell(Any[String(s[1:m.offset-1]), String(s[m.offset+ncodeunits(m.match):end])], (1, 2))
        else
            res["start"] = zeros(0, 0); res["end"] = zeros(0, 0); res["match"] = ""
            res["tokens"] = Cell(Any[], (0, 0)); res["tokenextents"] = zeros(0, 0)
            isempty(nmlist) || _unsup("regexp: 'once' with named tokens and no match", "function")
            res["names"] = MStruct()
            res["split"] = s
        end
        return res
    end
    res["start"] = ret(reshape(Float64[m.offset for m in ms], 1, :))
    res["end"] = ret(reshape(Float64[m.offset - 1 + ncodeunits(m.match) for m in ms], 1, :))
    res["match"] = Cell(Any[String(m.match) for m in ms], (1, length(ms)))
    tk = Any[]; te = Any[]
    for m in ms
        toks, ext, _ = _rx_tokens(m, ng, names)
        push!(tk, Cell(Any[toks...], (1, length(toks))))
        push!(te, ret(Matrix{Float64}(ext)))
    end
    res["tokens"] = Cell(tk, (1, length(tk)))
    res["tokenextents"] = Cell(te, (1, length(te)))
    st = MStruct()
    if !isempty(ms)
        _, _, named = _rx_tokens(ms[1], ng, names)
        for k in nmlist; st[k] = get(named, k, ""); end
    end
    res["names"] = st
    pieces = Any[]; last = 0
    for m in ms
        push!(pieces, String(s[last+1:m.offset-1])); last = m.offset - 1 + ncodeunits(m.match)
    end
    push!(pieces, String(s[last+1:end]))
    res["split"] = Cell(pieces, (1, length(pieces)))
    res
end

function _regexp_p4(I, a, nout, icase::Bool, who)
    length(a) >= 2 || merr("Invalid call to $who")
    s, pat = a[1], a[2]
    o = _RxOpts(false, true, icase, false, false, false)
    order = String[]
    for op in a[3:end]
        op isa AbstractString || _unsup("$who with a non-char option", "function")
        k = lowercase(String(op))
        k in _RX_OPTS || merr("$who: unrecognized option")
        if k == "once"; o.once = true
        elseif k == "matchcase"; o.icase = false
        elseif k == "ignorecase"; o.icase = true
        elseif k == "stringanchors"; o.multiline = false
        elseif k == "lineanchors"; o.multiline = true
        elseif k == "literalspacing"; o.extended = false
        elseif k == "freespacing"; o.extended = true
        elseif k == "noemptymatch"; o.emptymatch = false
        elseif k == "emptymatch"; o.emptymatch = true
        elseif k == "dotall"; o.dotall = true
        elseif k == "dotexceptnewline"; o.dotall = false
        else k in order || push!(order, k)
        end
    end
    outs = vcat(order, [k for k in _RX_OUT if !(k in order)])
    want = outs[1:max(nout, 1)]
    wset = Set(want)
    s isa Cell && pat isa Cell && _unsup("regexp with both a cell array of strings and a cell array of patterns", "function")
    if s isa Cell || pat isa Cell
        cell = s isa Cell ? s : pat
        for x in cell.items
            x isa AbstractString || merr("$who: all cells must be strings")
        end
        (s isa AbstractString || pat isa AbstractString) || _unsup("regexp: arguments that are not strings or cell arrays of strings", "function")
        results = [s isa Cell ? _rx_one(String(x), String(pat), o, wset) : _rx_one(String(s), String(x), o, wset) for x in cell.items]
        return Any[Cell(Any[r[k] for r in results], cell.shape) for k in want]
    end
    (s isa AbstractString && pat isa AbstractString) || _unsup("regexp: arguments that are not strings or cell arrays of strings (Octave converts numbers with a warning)", "function")
    r = _rx_one(String(s), String(pat), o, wset)
    Any[r[k] for k in want]
end
bi!("regexp") do I, a, nout; _regexp_p4(I, a, nout, false, "regexp"); end
bi!("regexpi") do I, a, nout; _regexp_p4(I, a, nout, true, "regexpi"); end

# ---------------------------------------------------------------------- intersect (Octave 8.4)
# (kind, items, is_row, is_empty) of an argument: a double array, a char row or a cellstr
function _setop_items(v, name)
    if v isa AbstractString
        isempty(v) && return (:char, Any[], false, true)
        return (:char, Any[c for c in v], true, false)
    elseif v isa Cell
        all(x -> x isa AbstractString, v.items) || _unsup("$name: a cell array with elements that are not strings", "function")
        isempty(v.items) && return (:cell, Any[], false, true)
        return (:cell, Any[String(x) for x in v.items], v.shape[1] == 1, false)
    elseif v isa Union{MStruct,Handle}
        _unsup("$name of a $(mclass(v)) value", "function")
    end
    mclass(v) == "double" || _unsup("$name of a $(mclass(v)) value (Octave keeps the class; only doubles, strings and cellstr are implemented)", "function")
    x = arr(v)
    isempty(x) && return (:num, Any[], false, true)
    (:num, Any[Float64(t) for t in vec(x)], size(x, 1) == 1, false)
end

bi!("intersect") do I, a, nout
    length(a) < 2 && merr("Invalid call to intersect")
    stable = false
    for o in a[3:end]
        o isa AbstractString || _unsup("intersect with a non-char option", "function")
        o == "stable" ? (stable = true) : o == "sorted" ? (stable = false) : _unsup("intersect option '$o'", "function")
    end
    ka, xa, ra, ea = _setop_items(a[1], "intersect")
    kb, xb, rb, eb = _setop_items(a[2], "intersect")
    if ka != kb
        if Set([ka, kb]) == Set([:char, :cell]) && !(ea || eb)
            ka == :char ? (xa = Any[a[1]]; ra = true; ka = :cell) : (xb = Any[a[2]]; rb = true; kb = :cell)
        else
            _unsup("intersect of values of different kinds (or an empty one of another kind)", "function")
        end
    end
    kind = ka
    if ea || eb
        kind == :char && _unsup("intersect with an empty char", "function")
        c = kind == :cell ? Cell(Any[], (0, 0)) : zeros(0, 0)
        return Any[c, zeros(0, 0), zeros(0, 0)][1:max(nout, 1)]
    end
    first_a = Dict{Any,Int}(); first_b = Dict{Any,Int}()
    for (i, v) in enumerate(xa); (v isa Float64 && isnan(v)) || haskey(first_a, v) || (first_a[v] = i); end
    for (i, v) in enumerate(xb); (v isa Float64 && isnan(v)) || haskey(first_b, v) || (first_b[v] = i); end
    common = [v for v in keys(first_a) if haskey(first_b, v)]
    common = stable ? sort(common; by=v -> first_a[v]) : sort(common)
    isrow = ra && rb
    ia = Float64[first_a[v] for v in common]; ib = Float64[first_b[v] for v in common]
    c = if kind == :char
        (!isrow && length(common) != 1) && _unsup("intersect of char arrays that are not rows", "function")
        String(join(common))
    elseif kind == :cell
        Cell(Any[common...], isrow ? (1, length(common)) : (length(common), 1))
    else
        v = Float64.(common)
        ret(isrow ? reshape(v, 1, :) : reshape(v, :, 1))
    end
    Any[c, ret(reshape(ia, :, 1)), ret(reshape(ib, :, 1))][1:max(nout, 1)]
end

# ---------------------------------------------------------------------- cell2mat: double and logical cells mix (the result is double)
let old = BUILTINS["cell2mat"]
    bi!("cell2mat") do I, a, nout
        if length(a) == 1 && a[1] isa Cell && !(0 in a[1].shape)
            c = a[1]
            allstr = all(x -> x isa AbstractString, c.items)
            if !allstr && !any(x -> x isa AbstractString, c.items) && !any(x -> x isa Union{Cell,MStruct,Handle}, c.items)
                kinds = Set(mclass(x) for x in c.items)
                if kinds == Set(["double", "logical"])
                    r, k = c.shape
                    rows = Any[mhcat(Any[(x = c.items[i + (j - 1) * r]; x isa Bool ? Float64(x) : x isa AbstractMatrix{Bool} ? Float64.(x) : x) for j in 1:k]) for i in 1:r]
                    return mvcat(rows)
                end
            end
            allstr && c.shape[1] != 1 && _unsup("cell2mat of strings in several rows (char matrix)", "function")
        end
        old(I, a, nout)
    end
end

# ---------------------------------------------------------------------- char() and sparse (a full matrix: values only)
let old = BUILTINS["char"]
    bi!("char") do I, a, nout
        isempty(a) && return ""
        old(I, a, nout)
    end
end
bi!("sparse") do I, a, nout
    n = length(a)
    n == 1 && return ret(Matrix{Float64}(farr(a[1])))
    if n == 2
        m, k = as_index_int(a[1]), as_index_int(a[2])
        (m < 0 || k < 0) && merr("sparse: dimensions M and N must be non-negative")
        return zeros(m, k)
    end
    n in (3, 5, 6) || _unsup("sparse with this number of arguments", "function")
    ii, jj, sv = vec(farr(a[1])), vec(farr(a[2])), vec(Float64.(farr(a[3])))
    for (v, pos) in ((ii, "(%s,_)"), (jj, "(_,%s)"))
        k = findfirst(x -> x != floor(x) || x < 1, v)
        k === nothing || merr("index " * replace(pos, "%s" => string(round(v[k]; digits=10) == floor(v[k]) ? string(Int(v[k])) : string(v[k]))) * ": subscripts must be either integers 1 to (2^63)-1 or logicals")
    end
    cnt = max(length(ii), length(jj), length(sv))
    for v in (ii, jj, sv)
        length(v) in (1, cnt) || merr("sparse: dimension mismatch")
    end
    bc(v) = length(v) == 1 ? fill(v[1], cnt) : v
    ii, jj, sv = bc(ii), bc(jj), bc(sv)
    m, k = n >= 5 ? (as_index_int(a[4]), as_index_int(a[5])) : (cnt > 0 ? Int(maximum(ii)) : 0, cnt > 0 ? Int(maximum(jj)) : 0)
    cnt > 0 && maximum(ii) > m && merr("sparse: row index $(Int(maximum(ii)))out of bound $m")
    cnt > 0 && maximum(jj) > k && merr("sparse: column index $(Int(maximum(jj)))out of bound $k")
    out = zeros(m, k)
    for t in 1:cnt
        out[Int(ii[t]), Int(jj[t])] += sv[t]
    end
    out
end

# ---------------------------------------------------------------------- chol (Octave: reads one triangle; [R, p] = chol(A))
bi!("chol") do I, a, nout
    (nout > 2 || !(1 <= length(a) <= 3)) && _unsup("chol with more than two outputs or more than two options", "function")
    lower = false
    for o in a[2:end]
        (o isa AbstractString && lowercase(o) in ("lower", "upper", "vector")) || merr("chol: optional argument must be one of \"vector\", \"lower\", or \"upper\"")
        lowercase(o) == "vector" && _unsup("chol with the 'vector' option (sparse matrices)", "function")
        lower = lowercase(o) == "lower"
    end
    x = farr(a[1])
    (size(x, 1) == size(x, 2)) || merr("chol: requires square matrix")
    all(isfinite, x) || _unsup("chol of a matrix with NaN or Inf", "function")
    n = size(x, 1)
    sym = lower ? LinearAlgebra.tril(x) + LinearAlgebra.tril(x, -1)' : LinearAlgebra.triu(x) + LinearAlgebra.triu(x, 1)'
    p = 0; k = n
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(sym); check=false)
    Lm = nothing
    if LinearAlgebra.issuccess(C)
        Lm = Matrix(C.L)
    else
        d = zeros(n, n)
        for j in 1:n
            s_ = sym[j, j] - sum(abs2, d[j, 1:j-1])
            if !(s_ > 0)
                p = j; k = j - 1; break
            end
            d[j, j] = sqrt(s_)
            for i in j+1:n
                d[i, j] = (sym[i, j] - sum(d[i, 1:j-1] .* d[j, 1:j-1])) / d[j, j]
            end
        end
        Lm = d[1:k, 1:k]
        nout < 2 && merr("chol: input matrix must be positive definite")
    end
    R = lower ? Lm : Matrix(Lm')
    Any[ret(R), Float64(p)][1:max(nout, 1)]
end

# ---------------------------------------------------------------------- read-only file queries inside the model's folder
function _folder_file(I, name::AbstractString, what)
    I.dir === nothing && _unsup("$what: no model folder to look in", "file")
    isempty(name) && return nothing
    (isascii(name) && !occursin('\0', name)) || _unsup("$what: unusual characters in the file name", "file")
    if startswith(name, "/") || startswith(name, "~") || (length(name) > 1 && name[2] == ':') || any(c -> c in "*?[]\\", name) || ".." in split(name, "/")
        _unsup("$what: $(repr(name)) is not a plain relative path inside the model's folder", "file")
    end
    base = realpath(I.dir)
    path = joinpath(base, name)
    ispath(path) || return nothing
    real = realpath(path)
    (real == base || startswith(real, base * "/")) || _unsup("$what: $(repr(name)) leaves the model's folder", "file")
    path
end
function _file_query(what, test)
    (I, a, nout) -> begin
        length(a) == 1 || merr("$what: function called with too many inputs")
        x = a[1]
        if x isa AbstractString
            p = _folder_file(I, x, what)
            return p !== nothing && test(p)
        elseif x isa Cell && all(t -> t isa AbstractString, x.items)
            isempty(x.items) && return Matrix{Bool}(undef, x.shape...)
            return ret(reshape(Bool[(p = _folder_file(I, t, what); p !== nothing && test(p)) for t in x.items], x.shape))
        end
        merr("$what: F must be a string or cell array of strings")
    end
end
BUILTINS["isfile"] = _file_query("isfile", isfile)
BUILTINS["isfolder"] = BUILTINS["isdir"] = _file_query("isfolder", isdir)
bi!("pwd") do I, a, nout
    (isempty(a) && I.dir !== nothing) || _unsup("pwd without a model folder", "file")
    realpath(I.dir)
end
let old = BUILTINS["exist"]
    bi!("exist") do I, a, nout
        name = _str(a[1], "exist")
        kind = length(a) > 1 ? a[2] : nothing
        sc = I.scope_stack[end]
        has_var(I, sc, name) && return 1.0
        (kind == "var" && !I.open_world) && return 0.0
        kind in (nothing, "file", "dir", "var") || _unsup("exist(name, '$kind') depends on the MATLAB installation", "environment")
        known_fn = haskey(BUILTINS, name) || haskey(LAZY, name) || haskey(dir_files(I), name)
        if kind in (nothing, "file", "dir") && I.dir !== nothing && !endswith(name, ".m")
            has_ext = occursin('/', name) || occursin('.', name)
            if has_ext || !known_fn
                p = _folder_file(I, name, "exist")
                p !== nothing && isdir(p) && return 7.0
                p !== nothing && isfile(p) && kind != "dir" && return 2.0
                (p !== nothing || kind == "dir" || has_ext) && return 0.0
            end
        end
        old(I, a, nout)
    end
end

# ---------------------------------------------------------------------- cellfun / arrayfun argument checks
let oldc = BUILTINS["cellfun"]
    bi!("cellfun") do I, a, nout
        length(a) >= 2 && !(a[2] isa Cell) && merr("cellfun: C must be a cell array")
        oldc(I, a, nout)
    end
end
