# MATLAB lexer and parser (port of pydyn/matlab.py). Source lines are strings, or
# `(:dyn, k)` placeholders standing for the k-th Dynare statement of a .mod file.

const KEYWORDS = Set(["if", "elseif", "else", "end", "for", "while", "switch", "case", "otherwise", "break",
    "continue", "return", "function", "try", "catch", "global", "persistent", "endfunction",
    "endif", "endfor", "endwhile", "endswitch", "end_try_catch", "parfor", "classdef"])
const _END_ALIASES = Set(["endfunction", "endif", "endfor", "endwhile", "endswitch", "end_try_catch"])
"Names accepted in command syntax (`hold on`, `close all`, `load file`) at the start of a line."
const COMMAND_WORDS = Set(["close", "clear", "clc", "hold", "format", "warning", "more", "figure", "load", "save",
    "addpath", "rmpath", "diary", "echo", "pause", "disp", "dbstop", "profile", "delete",
    "cd", "clearvars", "pkg", "graphics_toolkit", "drawnow", "grid", "box", "axis", "clf",
    "dynare", "type", "which", "mex", "import", "syms", "tic", "toc", "legend", "colormap",
    "shading", "rng", "beep", "commandwindow", "dbclear", "keyboard", "print", "run"])
const _CMD_RE = r"^\s*([A-Za-z_]\w*)[ \t]+(?![=(;,]|[-+*/^<>~&|.:]?=|[-+*/^<>&|:](?:\s|$)|\.[*/^\\'])(\S[^;,%]*)(.*)$"
const _NUM_RE = r"\G(?:\d+(?:\.(?![*/^\\'])\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
const _ID_RE = r"\G[A-Za-z]\w*"
const _OPS2 = Set(["==", "~=", "!=", "<=", ">=", "&&", "||", ".*", "./", ".\\", ".^", ".'"])
const _OPS1 = "+-*/\\^'<>&|~!=()[]{},;:.@"

mutable struct Tok
    k::Symbol          # :num :str :dstr :id :kw :op :nl :eof :err :cmd :dyn
    v::Any
    ws::Bool           # preceded by white space
    line::Int          # 1-based line number
    col::Int           # 1-based column (byte) in the line
end
Tok(k, v, ws, line) = Tok(k, v, ws, line, 1)

_asciify(s::AbstractString) = isascii(s) ? String(s) : String(map(c -> isascii(c) ? c : '?', collect(s)))

function tokenize(lines::Vector)
    toks = Tok[]
    depth = Char[]
    cont = false
    block_comment = 0
    for (ln, text0) in enumerate(lines)
        if !(text0 isa AbstractString)
            push!(toks, Tok(:dyn, text0[2], true, ln))
            push!(toks, Tok(:nl, "\n", true, ln))
            empty!(depth); cont = false
            continue
        end
        text = _asciify(text0)
        # non-ASCII characters are replaced by one '?' each for tokenizing (byte columns); string literals take the
        # original character at the same position
        origc = isascii(text0) ? nothing : collect(String(text0))
        st = strip(text)
        if block_comment > 0
            if st == "%}"
                block_comment -= 1
            elseif st == "%{"
                block_comment += 1
            end
            continue
        end
        if st == "%{"
            block_comment = 1
            continue
        end
        i, n = 1, ncodeunits(text)
        if isempty(depth) && !cont
            m = match(_CMD_RE, text)
            if m !== nothing && m.captures[1] in COMMAND_WORDS
                push!(toks, Tok(:cmd, (String(m.captures[1]), String(strip(m.captures[2]))), true, ln))
                i = m.offsets[2] + ncodeunits(m.captures[2])
            end
        end
        cont = false
        ws = true
        while i <= n
            c = text[i]
            if c in " \t\r\n\f\v"
                ws = true; i += 1; continue
            end
            c == '%' && break
            if startswith(SubString(text, i), "...")
                cont = true
                break
            end
            prev = (!isempty(toks) && toks[end].k != :nl) ? toks[end] : nothing
            if c == '\'' || c == '"'
                value_end = prev !== nothing && (prev.k in (:num, :id, :str) ||
                    (prev.k == :op && prev.v in (")", "]", "}", "'", ".'")) || (prev.k == :kw && prev.v == "end"))
                in_br = !isempty(depth) && depth[end] in "[{"
                if c == '\'' && value_end && (!ws || !in_br)
                    push!(toks, Tok(:op, "'", ws, ln, i)); i += 1; ws = false
                    continue
                end
                j = i + 1
                buf = IOBuffer()
                while true
                    if j > n
                        push!(toks, Tok(:err, "unterminated string", ws, ln, i))
                        j = n + 1
                        break
                    end
                    if text[j] == c
                        if j + 1 <= n && text[j+1] == c
                            write(buf, c); j += 2; continue
                        end
                        push!(toks, Tok(c == '\'' ? :str : :dstr, String(take!(buf)), ws, ln, i))
                        j += 1
                        break
                    end
                    write(buf, origc === nothing ? text[j] : origc[j]); j += 1
                end
                i = j; ws = false
                continue
            end
            if isdigit(c) || c == '.'
                m = match(_NUM_RE, text, i)
                if m !== nothing
                    s = m.match
                    j = i + ncodeunits(s)
                    if j <= n && (isletter(text[j]) || text[j] == '_')
                        push!(toks, Tok(:err, "bad number or date literal $(repr(text[i:min(n, j+2)]))", ws, ln, i))
                    else
                        push!(toks, Tok(:num, Float64(parse(BigFloat, replace(replace(s, 'd' => 'e'), 'D' => 'e'))), ws, ln, i))
                    end
                    i = j; ws = false
                    continue
                end
            end
            m = match(_ID_RE, text, i)
            if m !== nothing
                s = String(m.match)
                if s in KEYWORDS && !(prev !== nothing && prev.k == :op && prev.v == ".")
                    push!(toks, Tok(:kw, s in _END_ALIASES ? "end" : s, ws, ln, i))
                else
                    push!(toks, Tok(:id, s, ws, ln, i))
                end
                i += ncodeunits(s); ws = false
                continue
            end
            two = i + 1 <= n ? text[i:i+1] : ""
            if two in _OPS2
                push!(toks, Tok(:op, two, ws, ln, i)); i += 2; ws = false
                continue
            end
            if c in _OPS1
                if c in "([{"
                    push!(depth, c)
                elseif c in ")]}" && !isempty(depth)
                    pop!(depth)
                end
                push!(toks, Tok(:op, string(c), ws, ln, i)); i += 1; ws = false
                continue
            end
            push!(toks, Tok(:err, "unexpected character $(repr(c))", ws, ln, i))
            i += 1; ws = false
        end
        cont || push!(toks, Tok(:nl, "\n", true, ln))
    end
    push!(toks, Tok(:eof, nothing, true, length(lines) + 1))
    toks
end

# ====================================================================== syntax tree
abstract type MNode end

macro node(name, fields...)
    fs = [:($(f)::Any) for f in fields]
    quote
        mutable struct $(esc(name)) <: MNode
            $(map(esc, fs)...)
            line::Int
        end
        $(esc(name))($(map(esc, fields)...)) = $(esc(name))($(map(esc, fields)...), -1)
    end
end

@node Num v
@node Str v dq
@node Id name
@node ColonAll
@node EndKw
@node Unop op e
@node Binop op l r
@node RangeE a b c
@node IndexE base args brace
@node FieldE base name dyn
@node MatrixLit rows
@node CellLit rows
@node AnonFunc params body
@node FuncHandle name
@node BadExpr text why

@node ExprStmt e text
@node Assign targets rhs text
@node IfS clauses orelse text
@node ForS var e body text
@node WhileS cond body text
@node SwitchS e cases default text
@node BreakS text
@node ContinueS text
@node ReturnS text
@node TryS body ident handler text
@node GlobalS names text
@node CommandS name args text
@node DynS k text
@node Unparsed text why
@node FuncDef name ins outs body text

const _BLOCK_END = Set(["end", "else", "elseif", "case", "otherwise", "catch", "function"])

mutable struct Parser
    lines::Vector{Any}
    toks::Vector{Tok}
    i::Int
    idx_depth::Int
    in_mat::Vector{Bool}
    end_functions::Bool
end

function Parser(lines::Vector)
    p = Parser(collect(Any, lines), tokenize(lines), 1, 0, [false], false)
    p.end_functions = _functions_use_end(p)
    p
end

tk(p::Parser) = p.toks[p.i]
peek(p::Parser, k=1) = p.toks[min(p.i + k, length(p.toks))]
is_op(p::Parser, ops...) = (t = p.toks[p.i]; t.k == :op && t.v in ops)
is_kw(p::Parser, kws...) = (t = p.toks[p.i]; t.k == :kw && t.v in kws)
function expect_op(p::Parser, op)
    is_op(p, op) || throw(ParseError("expected $(repr(op)), found $(_show(tk(p)))"))
    p.i += 1
end
_show(t::Tok) = t.k == :nl ? "end of line" : (t.k == :eof ? "end of input" : repr(t.v))

"Source text of lines a..b (token line numbers)."
function src(p::Parser, a, b=a)
    a = max(a, 1); b = min(b, length(p.lines))
    strip(join([x for x in p.lines[a:b] if x isa AbstractString], "\n"))
end

function _functions_use_end(p::Parser)
    nfun = nopen = nend = 0
    br = 0
    for t in p.toks
        if t.k == :op && t.v in ("(", "[", "{")
            br += 1
        elseif t.k == :op && t.v in (")", "]", "}")
            br = max(0, br - 1)
        elseif t.k == :nl
            br = 0
        elseif t.k == :kw && br == 0
            if t.v == "function"
                nfun += 1
            elseif t.v in ("if", "for", "while", "switch", "try", "parfor")
                nopen += 1
            elseif t.v == "end"
                nend += 1
            end
        end
    end
    nfun > 0 && nend >= nopen + nfun
end

# ---- expressions
function pexpr(p::Parser)
    left = andand(p)
    while is_op(p, "||")
        p.i += 1
        left = Binop("||", left, andand(p))
    end
    left
end
function andand(p::Parser)
    left = elor(p)
    while is_op(p, "&&")
        p.i += 1
        left = Binop("&&", left, elor(p))
    end
    left
end
function elor(p::Parser)
    left = eland(p)
    while is_op(p, "|")
        p.i += 1
        left = Binop("|", left, eland(p))
    end
    left
end
function eland(p::Parser)
    left = pcmp(p)
    while is_op(p, "&")
        p.i += 1
        left = Binop("&", left, pcmp(p))
    end
    left
end
function pcmp(p::Parser)
    left = rng(p)
    while is_op(p, "==", "~=", "!=", "<", "<=", ">", ">=")
        _mat_break(p) && break
        op = tk(p).v
        p.i += 1
        left = Binop(op == "!=" ? "~=" : op, left, rng(p))
    end
    left
end
function rng(p::Parser)
    a = padd(p)
    if is_op(p, ":") && !_colon_ends(p)
        p.i += 1
        b = padd(p)
        if is_op(p, ":") && !_colon_ends(p)
            p.i += 1
            return RangeE(a, b, padd(p))
        end
        return RangeE(a, nothing, b)
    end
    a
end
function _colon_ends(p::Parser)
    nx = peek(p)
    (nx.k == :op && nx.v in (")", ",")) || nx.k in (:nl, :eof)
end
"Inside [ ] or { }, `a -b` separates two elements while `a - b` and `a-b` do not."
function _mat_break(p::Parser)
    p.in_mat[end] || return false
    t, nx = tk(p), peek(p)
    t.ws && !nx.ws
end
function padd(p::Parser)
    left = pmul(p)
    while is_op(p, "+", "-")
        _mat_break(p) && break
        op = tk(p).v
        p.i += 1
        left = Binop(op, left, pmul(p))
    end
    left
end
function pmul(p::Parser)
    left = punary(p)
    while is_op(p, "*", "/", "\\", ".*", "./", ".\\")
        op = tk(p).v
        p.i += 1
        left = Binop(op, left, punary(p))
    end
    left
end
function punary(p::Parser)
    if is_op(p, "-", "+", "~", "!")
        op = tk(p).v
        p.i += 1
        return Unop(op == "!" ? "~" : op, punary(p))
    end
    ppower(p)
end
function ppower(p::Parser)
    left = postfix(p)
    while true
        if is_op(p, "'", ".'")
            p.i += 1
            left = Unop("'", left)
        elseif is_op(p, "^", ".^")
            op = tk(p).v
            p.i += 1
            left = Binop(op, left, pow_operand(p))
        else
            return left
        end
    end
end
function pow_operand(p::Parser)
    if is_op(p, "-", "+", "~", "!")
        op = tk(p).v
        p.i += 1
        return Unop(op == "!" ? "~" : op, pow_operand(p))
    end
    postfix(p)
end
function pargs(p::Parser, close)
    p.idx_depth += 1
    push!(p.in_mat, false)
    out = Any[]
    try
        while tk(p).k == :nl && close == "}"
            p.i += 1
        end
        if is_op(p, close)
            p.i += 1
            return out
        end
        while true
            if is_op(p, ":") && peek(p).k == :op && peek(p).v in (",", close)
                p.i += 1
                push!(out, ColonAll())
            else
                push!(out, pexpr(p))
            end
            if is_op(p, ",")
                p.i += 1
                continue
            end
            expect_op(p, close)
            return out
        end
    finally
        p.idx_depth -= 1
        pop!(p.in_mat)
    end
end
function postfix(p::Parser)
    e = primary(p)
    while true
        t = tk(p)
        t.k != :op && return e
        if t.v == "("
            (p.in_mat[end] && t.ws) && return e
            p.i += 1
            e = IndexE(e, pargs(p, ")"), false)
        elseif t.v == "{"
            (p.in_mat[end] && t.ws) && return e
            p.i += 1
            e = IndexE(e, pargs(p, "}"), true)
        elseif t.v == "." && !t.ws && peek(p).k in (:id, :kw) && !peek(p).ws
            p.i += 1
            e = FieldE(e, tk(p).v, nothing)
            p.i += 1
        elseif t.v == "." && !t.ws && peek(p).k == :op && peek(p).v == "("
            p.i += 2
            push!(p.in_mat, false)
            d = pexpr(p)
            pop!(p.in_mat)
            expect_op(p, ")")
            e = FieldE(e, nothing, d)
        else
            return e
        end
    end
end
function primary(p::Parser)
    t = tk(p)
    if t.k == :num
        p.i += 1
        return Num(Float64(t.v))
    end
    if t.k == :str || t.k == :dstr
        p.i += 1
        return Str(t.v, t.k == :dstr)
    end
    if t.k == :id
        p.i += 1
        return Id(t.v)
    end
    if t.k == :kw && t.v == "end" && p.idx_depth > 0
        p.i += 1
        return EndKw()
    end
    if t.k == :op
        if t.v == "("
            p.i += 1
            push!(p.in_mat, false)
            e = pexpr(p)
            pop!(p.in_mat)
            expect_op(p, ")")
            return Unop("()", e)
        end
        if t.v == "["
            p.i += 1
            return MatrixLit(pmatrix(p, "]"))
        end
        if t.v == "{"
            p.i += 1
            return CellLit(pmatrix(p, "}"))
        end
        if t.v == "@"
            p.i += 1
            if is_op(p, "(")
                p.i += 1
                params = String[]
                while !is_op(p, ")")
                    if tk(p).k == :id
                        push!(params, tk(p).v)
                    elseif is_op(p, "~")
                        push!(params, "~")
                    else
                        throw(ParseError("bad parameter list of an anonymous function"))
                    end
                    p.i += 1
                    is_op(p, ",") && (p.i += 1)
                end
                p.i += 1
                push!(p.in_mat, false)
                save = p.idx_depth; p.idx_depth = 0
                body = pexpr(p)
                p.idx_depth = save
                pop!(p.in_mat)
                return AnonFunc(params, body)
            end
            if tk(p).k == :id
                name = tk(p).v
                p.i += 1
                while is_op(p, ".") && peek(p).k == :id
                    name *= "." * peek(p).v
                    p.i += 2
                end
                return FuncHandle(name)
            end
            throw(ParseError("bad function handle"))
        end
    end
    t.k == :err && throw(ParseError(t.v))
    throw(ParseError("unexpected $(_show(t))"))
end
function pmatrix(p::Parser, close)
    rows = Any[]; row = Any[]
    push!(p.in_mat, true)
    save = p.idx_depth
    try
        while true
            t = tk(p)
            if t.k == :op && t.v == close
                p.i += 1
                break
            end
            if t.k == :nl || (t.k == :op && t.v == ";")
                p.i += 1
                if !isempty(row)
                    push!(rows, row); row = Any[]
                end
                continue
            end
            if t.k == :op && t.v == ","
                p.i += 1
                continue
            end
            t.k in (:eof, :dyn, :cmd) && throw(ParseError("unbalanced brackets"))
            push!(row, pexpr(p))
        end
        isempty(row) || push!(rows, row)
        return rows
    finally
        pop!(p.in_mat)
        p.idx_depth = save
    end
end

# ---- statements
"Parse everything; returns (statements, functions)."
function program(p::Parser)
    body = Any[]; funcs = Dict{String,Any}()
    while tk(p).k != :eof
        if is_kw(p, "function")
            f = funcdef(p)
            haskey(funcs, f.name) || (funcs[f.name] = f)
            continue
        end
        if is_kw(p, "end", "else", "elseif", "case", "otherwise", "catch")
            throw(ParseError("'$(tk(p).v)' without a matching block opener (line: $(repr(src(p, tk(p).line))))"))
        end
        s = statement(p)
        s === nothing || push!(body, s)
    end
    body, funcs
end
function block(p::Parser, stop, opener_line=nothing)
    out = Any[]
    while true
        t = tk(p)
        t.k == :eof && throw(ParseError("unexpected end of input inside a block (missing 'end')"))
        (t.k == :kw && t.v in stop) && return out
        (t.k == :kw && t.v in _BLOCK_END) && throw(ParseError("unexpected '$(t.v)' (line: $(repr(src(p, t.line))))"))
        s = statement(p, opener_line)
        s === nothing || push!(out, s)
    end
end
function _skip_seps(p::Parser)
    while tk(p).k == :nl || is_op(p, ";", ",")
        p.i += 1
    end
end
function _end_stmt(p::Parser)
    t = tk(p)
    if t.k in (:nl, :eof) || (t.k == :op && t.v in (";", ","))
        while is_op(p, ";", ",")
            p.i += 1
        end
        tk(p).k == :nl && (p.i += 1)
        return
    end
    (t.k == :kw && t.v in _BLOCK_END) && return
    throw(ParseError("unexpected $(_show(t))"))
end
"Condition of if/while/switch/case: a failure here makes the block undecidable, not unparseable."
function _header_expr(p::Parser, line0)
    start = p.i
    push!(p.in_mat, false)
    try
        e = pexpr(p)
        if !(tk(p).k in (:nl, :eof, :id, :kw, :cmd, :dyn) || is_op(p, ",", ";", "[", "("))
            throw(ParseError("unexpected $(_show(tk(p)))"))
        end
        return e
    catch exc
        exc isa ParseError || rethrow()
        p.i = start
        while !(tk(p).k in (:nl, :eof))
            p.i += 1
        end
        return BadExpr(src(p, line0), exc.msg)
    finally
        pop!(p.in_mat)
    end
end
function statement(p::Parser, opener_line=nothing)
    t = tk(p)
    if t.k == :nl || (t.k == :op && t.v in (";", ","))
        p.i += 1
        return nothing
    end
    line0 = t.line
    if t.k == :dyn
        p.i += 1
        s = DynS(t.v, nothing); s.line = line0
        return s
    end
    if t.k == :cmd
        p.i += 1
        s = CommandS(t.v[1], t.v[2], src(p, line0)); s.line = line0
        return s
    end
    if t.k == :kw
        s = compound(p, t)
        if s !== nothing
            s.line = line0
            return s
        end
    end
    start = p.i
    idx_depth, in_mat = p.idx_depth, copy(p.in_mat)
    try
        s = simple(p)
        _end_stmt(p)
        s.text = src(p, line0, p.toks[p.i-1].line)
        s.line = line0
        return s
    catch exc
        exc isa ParseError || rethrow()
        p.idx_depth, p.in_mat = idx_depth, in_mat
        p.i = start
        return _recover(p, line0, opener_line, exc.msg)
    end
end
"Skip ONE statement that could not be parsed: up to the next ';' or ',' outside brackets, or the end of the line."
function _recover(p::Parser, line0, opener_line, why)
    depth, j = 0, p.i
    firsttok = p.toks[j]
    stop_j = nothing
    while true
        t = p.toks[j]
        t.k in (:eof, :dyn) && break
        if t.k == :op && t.v in ("(", "[", "{")
            depth += 1
        elseif t.k == :op && t.v in (")", "]", "}")
            depth -= 1
        elseif (t.k == :nl || (t.k == :op && t.v in (";", ","))) && depth <= 0
            stop_j = j
            break
        elseif t.k == :kw && depth <= 0 && t.v in ("end", "else", "elseif", "case", "otherwise", "catch") &&
               opener_line !== nothing && t.line == opener_line && j > p.i
            stop_j = j - 1
            break
        elseif t.k == :nl && t.line - line0 > 400
            break
        end
        j += 1
    end
    if stop_j === nothing || occursin("unterminated", why)
        j = p.i
        while !(p.toks[j].k in (:nl, :eof, :dyn))
            j += 1
        end
        stop_j = j
        firsttok = Tok(:op, "", true, line0, 1)
    end
    te = p.toks[stop_j]
    last_line = te.k != :eof ? te.line : p.toks[stop_j-1].line
    if last_line == line0 && p.lines[line0] isa AbstractString
        ltxt = _asciify(p.lines[line0])
        stopc = (te.k == :op && te.v in (";", ",")) ? te.col - 1 : ncodeunits(ltxt)
        text = te.k != :kw ? strip(ltxt[firsttok.col:max(firsttok.col - 1, stopc)]) : strip(ltxt[firsttok.col:end])
    else
        text = src(p, line0, last_line)
    end
    kws = [t.v for t in p.toks[p.i:stop_j] if t.k == :kw && t.v != "end"]
    p.i = (te.k == :nl || (te.k == :op && te.v in (";", ","))) ? stop_j + 1 : stop_j + (te.k in (:eof, :dyn) ? 0 : 1)
    if any(k -> k in ("if", "for", "while", "switch", "try", "function", "parfor", "classdef", "global", "persistent"), kws)
        throw(ParseError("cannot parse a statement that opens a block or declares globals: $(repr(first(text, 100))) ($why)"))
    end
    s = Unparsed(String(text), why)
    s.line = line0
    s
end
function compound(p::Parser, t)
    line0 = t.line
    kw = t.v
    if kw == "if"
        p.i += 1
        clauses = Any[]; orelse = Any[]
        cond = _header_expr(p, line0)
        while true
            body = block(p, ("elseif", "else", "end"), line0)
            push!(clauses, (cond, body))
            if is_kw(p, "elseif")
                l1 = tk(p).line
                p.i += 1
                cond = _header_expr(p, l1)
                continue
            end
            if is_kw(p, "else")
                p.i += 1
                orelse = block(p, ("end",), line0)
            end
            p.i += 1
            break
        end
        return IfS(clauses, orelse, src(p, line0))
    end
    if kw in ("for", "parfor")
        p.i += 1
        paren = is_op(p, "(")
        start = p.i
        var = nothing; e = nothing
        try
            paren && (p.i += 1)
            tk(p).k != :id && throw(ParseError("bad for statement"))
            var = tk(p).v
            p.i += 1
            expect_op(p, "=")
            push!(p.in_mat, false)
            try
                e = pexpr(p)
            finally
                pop!(p.in_mat)
            end
            paren && expect_op(p, ")")
        catch exc
            exc isa ParseError || rethrow()
            p.i = start
            m = match(r"^\s*\(?\s*([A-Za-z_]\w*)\s*=", String(src(p, line0))[4:end])
            var = m === nothing ? nothing : String(m.captures[1])
            while !(tk(p).k in (:nl, :eof))
                p.i += 1
            end
            e = BadExpr(src(p, line0), exc.msg)
        end
        body = block(p, ("end",), line0)
        p.i += 1
        return ForS(var, e, body, src(p, line0))
    end
    if kw == "while"
        p.i += 1
        cond = _header_expr(p, line0)
        body = block(p, ("end",), line0)
        p.i += 1
        return WhileS(cond, body, src(p, line0))
    end
    if kw == "switch"
        p.i += 1
        e = _header_expr(p, line0)
        _skip_seps(p)
        cases = Any[]; default = nothing
        while true
            if is_kw(p, "case")
                l1 = tk(p).line
                p.i += 1
                ce = _header_expr(p, l1)
                push!(cases, (ce, block(p, ("case", "otherwise", "end"), line0)))
            elseif is_kw(p, "otherwise")
                p.i += 1
                default = block(p, ("case", "otherwise", "end"), line0)
            elseif is_kw(p, "end")
                p.i += 1
                break
            else
                throw(ParseError("bad switch statement"))
            end
        end
        return SwitchS(e, cases, default, src(p, line0))
    end
    if kw == "try"
        p.i += 1
        is_op(p, ",") && (p.i += 1)
        body = block(p, ("catch", "end"), line0)
        ident = nothing; handler = Any[]
        if is_kw(p, "catch")
            cl = tk(p).line
            p.i += 1
            nx = peek(p)
            if tk(p).k == :id && tk(p).line == cl && (nx.k == :nl || (nx.k == :op && nx.v in (";", ",")))
                ident = tk(p).v
                p.i += 1
            end
            handler = block(p, ("end",), line0)
        end
        p.i += 1
        return TryS(body, ident, handler, src(p, line0))
    end
    if kw in ("break", "continue", "return")
        p.i += 1
        _end_stmt(p)
        txt = src(p, line0)
        return kw == "break" ? BreakS(txt) : kw == "continue" ? ContinueS(txt) : ReturnS(txt)
    end
    if kw in ("global", "persistent")
        p.i += 1
        names = String[]
        while tk(p).k == :id
            push!(names, tk(p).v)
            p.i += 1
        end
        _end_stmt(p)
        kw == "persistent" && return Unparsed(src(p, line0), "persistent variables are not supported")
        return GlobalS(names, src(p, line0))
    end
    kw == "function" && throw(ParseError("function definition inside a block"))
    kw == "classdef" && throw(ParseError("classdef is not supported"))
    nothing
end
_fmtg(x::Float64) = x == round(x) && abs(x) < 1e15 ? string(Int(x)) : string(x)
function simple(p::Parser)
    t = tk(p)
    if t.k == :id && t.v in COMMAND_WORDS
        nx = peek(p)
        if nx.line == t.line && nx.ws && (nx.k in (:id, :num, :str, :kw) || (nx.k == :op && nx.v == "-" && !peek(p, 2).ws))
            j = p.i + 1
            words = String[]
            ok = true
            while !(p.toks[j].k in (:nl, :eof, :dyn)) && !(p.toks[j].k == :op && p.toks[j].v in (";", ","))
                tj = p.toks[j]
                if !(tj.k in (:id, :num, :str, :kw)) && !(tj.k == :op && tj.v in ("-", ".", "*"))
                    ok = false
                    break
                end
                piece = tj.k == :num ? _fmtg(tj.v) : string(tj.v)
                if !isempty(words) && !tj.ws
                    words[end] *= piece
                else
                    push!(words, piece)
                end
                j += 1
            end
            if ok && !isempty(words)
                p.i = j
                return CommandS(t.v, join(words, " "), nothing)
            end
        end
    end
    if t.k == :op && t.v == "["
        depth, j = 0, p.i
        while true
            tj = p.toks[j]
            if tj.k in (:eof, :nl, :dyn)
                j = nothing
                break
            end
            if tj.k == :op && tj.v in ("[", "(", "{")
                depth += 1
            elseif tj.k == :op && tj.v in ("]", ")", "}")
                depth -= 1
                depth == 0 && break
            end
            j += 1
        end
        if j !== nothing && p.toks[j+1].k == :op && p.toks[j+1].v == "="
            p.i += 1
            targets = Any[]
            push!(p.in_mat, true)
            try
                while !is_op(p, "]")
                    if is_op(p, ",")
                        p.i += 1
                        continue
                    end
                    if is_op(p, "~", "!")
                        p.i += 1
                        push!(targets, nothing)
                        continue
                    end
                    e = postfix(p)
                    _check_lvalue(e)
                    push!(targets, e)
                end
            finally
                pop!(p.in_mat)
            end
            p.i += 2
            return Assign(targets, pexpr(p), nothing)
        end
    end
    e = pexpr(p)
    if is_op(p, "=")
        _check_lvalue(e)
        p.i += 1
        return Assign(Any[e], pexpr(p), nothing)
    end
    ExprStmt(e, nothing)
end
function _check_lvalue(e)
    while true
        e isa Id && return
        if e isa IndexE || e isa FieldE
            e = e.base
            continue
        end
        throw(ParseError("invalid target of an assignment"))
    end
end
function funcdef(p::Parser)
    line0 = tk(p).line
    p.i += 1
    outs = String[]; ins = String[]
    if is_op(p, "[")
        p.i += 1
        while !is_op(p, "]")
            if tk(p).k == :id
                push!(outs, tk(p).v)
            elseif !is_op(p, ",")
                throw(ParseError("bad function declaration"))
            end
            p.i += 1
        end
        p.i += 1
        expect_op(p, "=")
    elseif tk(p).k == :id && peek(p).k == :op && peek(p).v == "="
        push!(outs, tk(p).v)
        p.i += 2
    end
    tk(p).k != :id && throw(ParseError("bad function declaration"))
    name = tk(p).v
    p.i += 1
    if is_op(p, "(")
        p.i += 1
        while !is_op(p, ")")
            if tk(p).k == :id
                push!(ins, tk(p).v)
            elseif is_op(p, "~")
                push!(ins, "~")
            elseif !is_op(p, ",")
                throw(ParseError("bad function declaration"))
            end
            p.i += 1
        end
        p.i += 1
    end
    body = Any[]
    while true
        t = tk(p)
        (t.k == :eof || (t.k == :kw && t.v == "function")) && break
        if t.k == :kw && t.v == "end"
            p.end_functions || throw(ParseError("unexpected 'end'"))
            p.i += 1
            break
        end
        (t.k == :kw && t.v in _BLOCK_END) && throw(ParseError("unexpected '$(t.v)'"))
        s = statement(p)
        s === nothing || push!(body, s)
    end
    fd = FuncDef(name, ins, outs, body, src(p, line0))
    fd.line = line0
    fd
end

"Parse MATLAB source; returns (statements, functions)."
function mparse(text_or_lines)
    lines = text_or_lines isa AbstractString ? split(text_or_lines, "\n") : collect(Any, text_or_lines)
    program(Parser(collect(Any, lines)))
end
