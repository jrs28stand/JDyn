# Tree-walking evaluator for parsed MATLAB (port of pydyn/matlab.py's Interpreter).

struct BreakSig <: Exception end
struct ContinueSig <: Exception end
struct ReturnSig <: Exception end
struct MissingT end
const MISSING_ = MissingT()
const HOST_GLOBALS = ("M_", "options_", "oo_")

mutable struct Scope
    vars::Dict{String,Any}
    global_names::Set{String}
    funcs::Dict{String,Any}
    is_base::Bool
    nargin::Union{Nothing,Int}
    nargout::Union{Nothing,Int}
end
Scope(; vars=Dict{String,Any}(), funcs=Dict{String,Any}(), is_base=false) =
    Scope(vars, is_base ? Set(HOST_GLOBALS) : Set{String}(), funcs, is_base, nothing, nothing)
Scope(vars::Dict{String,Any}; funcs=Dict{String,Any}()) = Scope(; vars=vars, funcs=funcs)

"A parsed .m file."
struct Unit
    path::String
    body::Vector{Any}
    funcs::Dict{String,Any}
    main::Any
end
function Unit(path, body, funcs)
    main = (!isempty(funcs) && isempty(body)) ? _first_func(funcs) : nothing
    Unit(path, body, funcs, main)
end
is_script(u::Unit) = u.main === nothing

"Every syntax-tree node below (and including) `node`."
function walk(node)
    out = MNode[]
    stack = Any[node]
    while !isempty(stack)
        n = pop!(stack)
        if n isa MNode
            push!(out, n)
            for f in fieldnames(typeof(n))
                f === :line && continue
                push!(stack, getfield(n, f))
            end
        elseif n isa AbstractVector || n isa Tuple
            append!(stack, n)
        end
    end
    out
end

mutable struct Interp
    dir::Union{Nothing,String}
    host::Any
    globals::Dict{String,Any}
    base::Scope
    scope_stack::Vector{Scope}
    end_stack::Vector{Tuple{Any,Int,Int}}
    steps::Int
    depth::Int
    open_world::Bool
    units::Dict{String,Any}
    dir_files_::Union{Nothing,Dict{String,String}}
    empty_loop_value::Any
    load_hook::Any                 # function (I, args, sc) for `load`, or nothing
    func_order::Dict{String,Int}   # order in which functions appear in a file (first = main)
    jlfuncs::Dict{String,Any}      # Julia callables usable as MATLAB functions (numbers in, numbers out)
    session::Any                   # the ModSession running a .mod file, if any
end

function Interp(directory=nothing, host=nothing)
    base = Scope(; is_base=true)
    I = Interp(directory === nothing ? nothing : String(directory), host, Dict{String,Any}(), base, [base],
               Tuple{Any,Int,Int}[], 0, 0, false, Dict{String,Any}(), nothing, zeros(1, 0), nothing, Dict{String,Int}(),
               Dict{String,Any}(), nothing)
    if host !== nothing
        I.globals["M_"] = host_M(host)
        I.globals["options_"] = host_options(host)
        I.globals["oo_"] = host_oo(host)
    end
    I
end

# ---- files
function dir_files(I::Interp)
    if I.dir_files_ === nothing
        I.dir_files_ = Dict{String,String}()
        if I.dir !== nothing && isdir(I.dir)
            for f in readdir(I.dir)
                if endswith(f, ".m") && occursin(r"^[A-Za-z_]\w*\.m$", f)
                    I.dir_files_[f[1:end-2]] = joinpath(I.dir, f)
                end
            end
        end
    end
    I.dir_files_
end

"Functions of a parsed file in file order (Dict keeps no order: the parser records it)."
_first_func(funcs::Dict) = begin
    best = nothing; bl = typemax(Int)
    for f in values(funcs)
        if f.line < bl
            best, bl = f, f.line
        end
    end
    best
end

"Parsed `name.m` from the model directory, a ParseError, or nothing."
function unit(I::Interp, name)
    haskey(I.units, name) && return I.units[name]
    p = get(dir_files(I), name, nothing)
    u = nothing
    if p !== nothing
        try
            body, funcs = mparse(read(p, String))
            u = Unit(p, body, funcs)
        catch exc
            exc isa ParseError || rethrow()
            u = exc
        end
    end
    I.units[name] = u
    u
end

need_host(I::Interp, what) = I.host === nothing ? throw(Unsupported("$what needs a model", "function")) : I.host

function tick(I::Interp)
    I.steps += 1
    I.steps > MAX_STEPS[] && _unsup("step limit exceeded (endless loop?)")
end

# ---- variables
has_var(I::Interp, sc::Scope, name) = haskey(sc.vars, name) || (name in sc.global_names && haskey(I.globals, name))
function getvar(I::Interp, sc::Scope, name)
    v = get(sc.vars, name, MISSING_)
    if v === MISSING_ && name in sc.global_names
        v = get(I.globals, name, MISSING_)
    end
    v
end
function setvar(I::Interp, sc::Scope, name, value)
    if name in sc.global_names
        if name in HOST_GLOBALS && I.host !== nothing
            g = get(I.globals, name, nothing)
            value === g || restore!(g, value)       # M_ = M_saved: accepted for a copy made earlier
            return
        end
        I.globals[name] = value
    else
        sc.vars[name] = value
    end
    nothing
end

_snap(v) = v isa HostStruct ? snapshot(v) : v

# ---- expressions
ev(I::Interp, n::Num, sc) = n.v
ev(I::Interp, n::Str, sc) = n.v
function ev(I::Interp, n::Id, sc)
    v = get(sc.vars, n.name, MISSING_)
    if v === MISSING_
        if n.name in sc.global_names && haskey(I.globals, n.name)
            return _snap(I.globals[n.name])          # M_ used as a value is a copy
        end
        (n.name == "nargin" && sc.nargin !== nothing) && return Float64(sc.nargin)
        (n.name == "nargout" && sc.nargout !== nothing) && return Float64(sc.nargout)
        r = mcall(I, n.name, Any[], 1, sc)
        isempty(r) && merr("'$(n.name)' returns no value")
        return r[1]
    end
    v isa Tainted && throw(Unsupported("'$(n.name)' is unknown: $(v.why)", "tainted"))
    v
end
function ev(I::Interp, n::EndKw, sc)
    isempty(I.end_stack) && throw(Unsupported("'end' inside a call to an unknown function or variable", "function"))
    v, k, nargs = I.end_stack[end]
    shp = v === nothing ? (0, 0) : shape_of(v)
    nargs == 1 && return Float64(shp[1] * shp[2])
    k <= 2 ? Float64(shp[k]) : 1.0
end
function ev(I::Interp, n::Unop, sc)
    v = ev(I, n.e, sc)
    op = n.op
    op == "()" && return v
    op == "-" && return uminus(v)
    op == "+" && return v isa Float64 ? v : ret(farr(v))
    op == "~" && return lnot(v)
    mtranspose(v)
end
const _BIN = Dict{String,Function}("+" => plus, "-" => minus, "*" => mtimes, "/" => mrdivide, "\\" => mldivide,
    "^" => mpower, ".*" => times, "./" => rdivide, ".\\" => ldivide, ".^" => power, "<" => mlt, ">" => mgt,
    "<=" => mle, ">=" => mge, "==" => meq, "~=" => mne, "&" => el_and, "|" => el_or)
function ev(I::Interp, n::Binop, sc)
    op = n.op
    if op == "&&" || op == "||"
        a = _sc_bool(ev(I, n.l, sc))
        (op == "&&") != a && return a
        return _sc_bool(ev(I, n.r, sc))
    end
    _BIN[op](ev(I, n.l, sc), ev(I, n.r, sc))
end
function _sc_bool(v)
    v isa AbstractString && merr("operands of && and || must be logical scalars")
    a = arr(v)
    length(a) != 1 && merr("operands of && and || must be convertible to logical scalar values")
    truth(a[1])
end
function ev(I::Interp, n::RangeE, sc)
    a = ev(I, n.a, sc)
    c = ev(I, n.c, sc)
    n.b === nothing ? colon(a, c) : colon(a, ev(I, n.b, sc), c)
end
function ev(I::Interp, n::MatrixLit, sc)
    rows = Any[]
    for row in n.rows
        for e in row
            (e isa Str && e.dq) && _unsup("string objects (double quotes) in a concatenation")
        end
        push!(rows, mhcat(Any[ev(I, e, sc) for e in row]))
    end
    length(rows) != 1 ? mvcat(rows) : rows[1]
end
function ev(I::Interp, n::CellLit, sc)
    isempty(n.rows) && return Cell(Any[], (0, 0))
    rows = [Any[ev(I, e, sc) for e in row] for row in n.rows]
    w = length(rows[1])
    any(r -> length(r) != w, rows) && merr("cell array rows must have the same length")
    Cell(Any[rows[i][j] for j in 1:w for i in 1:length(rows)], (length(rows), w))
end
function ev(I::Interp, n::AnonFunc, sc)
    names = setdiff(Set(x.name for x in walk(n.body) if x isa Id), Set(n.params))
    env = Dict{String,Any}()
    for nm in names
        v = getvar(I, sc, nm)
        v === MISSING_ || (env[nm] = _snap(v))       # captured by value
    end
    Handle(I, n.params, n.body, env, nothing, sc.funcs)
end
function ev(I::Interp, n::FuncHandle, sc)
    occursin(r"^[A-Za-z_]\w*$", n.name) || throw(Unsupported("handle to $(n.name)", "function"))
    Handle(I, nothing, nothing, nothing, n.name, sc.funcs)
end
ev(I::Interp, n::BadExpr, sc) = throw(Unsupported("cannot parse $(repr(first(n.text, 80))): $(n.why)", "syntax"))
ev(I::Interp, n::ColonAll, sc) = merr("':' used outside an index")

function index_args(I::Interp, v, argnodes, sc)
    out = Any[]
    n = length(argnodes)
    for (k, a) in enumerate(argnodes)
        if a isa ColonAll
            push!(out, COLON); continue
        end
        if a isa Num
            push!(out, a.v); continue
        end
        push!(I.end_stack, (v, k, n))
        try
            push!(out, ev(I, a, sc))
        finally
            pop!(I.end_stack)
        end
    end
    out
end
function ev(I::Interp, n::IndexE, sc)
    base = n.base
    if base isa Id
        v = getvar(I, sc, base.name)
        if v === MISSING_
            n.brace && merr("'$(base.name)' undefined")
            r = call_node(I, base.name, n.args, 1, sc)
            isempty(r) && merr("'$(base.name)' returns no value")
            return r[1]
        end
        v isa Tainted && throw(Unsupported("'$(base.name)' is unknown: $(v.why)", "tainted"))
    else
        v = ev(I, base, sc)
    end
    if v isa Handle && !n.brace
        r = call_handle(I, v, Any[ev(I, a, sc) for a in n.args], 1)
        isempty(r) && merr("function handle returns no value")
        return r[1]
    end
    v = _snap(v)
    args = index_args(I, v, n.args, sc)
    n.brace ? index_brace(v, args) : mindex(v, args)
end
function getfield_(I::Interp, v, name)
    v isa HostStruct && return get_field(v, name)
    if v isa MStruct
        if !haskey(v, name)
            v._kind !== nothing && _has_field(v, name)
            merr("reference to non-existent field '$name'")
        end
        return v[name]
    end
    v isa Tainted && arr(v)
    throw(Unsupported("field access '.$name' on a $(mclass(v)) value (objects and packages are not supported)", "function"))
end
function ev(I::Interp, n::FieldE, sc)
    b = n.base
    if b isa Id
        v = getvar(I, sc, b.name)
        if v === MISSING_
            (I.host !== nothing && host_is_param(I.host, b.name) && !I.open_world) && merr("'$(b.name)' undefined")
            throw(Unsupported("unknown variable, package or object '$(b.name)'", b.name in HOST_GLOBALS ? "results" : "function"))
        end
        v isa Tainted && throw(Unsupported("'$(b.name)' is unknown: $(v.why)", "tainted"))
    else
        v = ev(I, b, sc)
    end
    name = n.name !== nothing ? n.name : ev(I, n.dyn, sc)
    name isa AbstractString || merr("dynamic field name must be a char array")
    getfield_(I, v, name)
end

"Evaluate an expression that may return several values (a call). Returns a Vector{Any}."
function ev_multi(I::Interp, n, sc, nargout)
    if n isa IndexE && !n.brace
        base = n.base
        if base isa Id && getvar(I, sc, base.name) === MISSING_
            return call_node(I, base.name, n.args, nargout, sc)
        end
        if !(base isa Id) || getvar(I, sc, base.name) isa Handle
            v = ev(I, base, sc)
            v isa Handle && return call_handle(I, v, Any[ev(I, a, sc) for a in n.args], nargout)
        end
    elseif n isa Id && getvar(I, sc, n.name) === MISSING_ && !(n.name in ("nargin", "nargout"))
        return mcall(I, n.name, Any[], nargout, sc)
    end
    Any[ev(I, n, sc)]
end

# ---- calls
_aslist(r) = r isa Vector{Any} ? r : Any[r]

function call_node(I::Interp, name, argnodes, nargout, sc)
    if haskey(LAZY, name) && !haskey(sc.funcs, name) && !haskey(dir_files(I), name)
        return _aslist(LAZY[name](I, argnodes, sc, nargout))
    end
    args = Any[a isa ColonAll ? ":" : ev(I, a, sc) for a in argnodes]
    mcall(I, name, args, nargout, sc)
end

function mcall(I::Interp, name, args, nargout, sc)
    f = get(sc.funcs, name, nothing)
    f !== nothing && return call_user(I, f, sc.funcs, args, nargout)
    jf = get(I.jlfuncs, name, nothing)
    jf !== nothing && return _call_julia(jf, args, nargout)
    if haskey(dir_files(I), name)
        u = unit(I, name)
        u isa ParseError && throw(Unsupported("$name.m cannot be parsed: $(u.msg)", "syntax"))
        if is_script(u)
            (!isempty(args) || nargout > 1) && merr("$name is a script")
            run_script(I, u, sc)
            return Any[]
        end
        return call_user(I, u.main, u.funcs, args, nargout)
    end
    f = get(BUILTINS, name, nothing)
    if f !== nothing
        push!(I.scope_stack, sc)
        r = try
            f(I, args, nargout)
        catch e
            e isa _JULIA_ERRORS ? throw(matlab_error(e)) : rethrow()
        finally
            pop!(I.scope_stack)
        end
        return _aslist(r)
    end
    haskey(LAZY, name) && return _aslist(LAZY[name](I, Any[], sc, nargout))
    (I.host !== nothing && host_is_param(I.host, name) && !I.open_world) && merr("'$name' undefined (the parameter has no value yet)")
    throw(Unsupported("unknown function or variable '$name'", "function"))
end

_to_julia(v::Real) = Float64(v)
_to_julia(v::AbstractMatrix) = Matrix{Float64}(v)
_to_julia(v) = merr("a $(mclass(v)) value was passed to a Julia function")
_to_matlab(v::Bool) = v
_to_matlab(v::Real) = Float64(v)
_to_matlab(v::AbstractString) = String(v)
_to_matlab(v::AbstractVector) = ret(reshape(Float64.(v), :, 1))
_to_matlab(v::AbstractMatrix) = ret(Matrix{Float64}(v))
"A Julia function called from MATLAB code: several outputs are returned as a tuple."
function _call_julia(f, args, nargout)
    out = f(map(_to_julia, args)...)
    out isa Tuple ? Any[_to_matlab(v) for v in out] : Any[_to_matlab(out)]
end

function call_user(I::Interp, f::FuncDef, funcs, args, nargout)
    varargin = !isempty(f.ins) && f.ins[end] == "varargin"
    nfixed = length(f.ins) - (varargin ? 1 : 0)
    (length(args) > nfixed && !varargin) && merr("$(f.name): called with too many inputs")
    (nargout > length(f.outs) && !(!isempty(f.outs) && f.outs[end] == "varargout")) && merr("$(f.name): called with too many outputs")
    (!isempty(f.outs) && f.outs[end] == "varargout") && _unsup("varargout")
    sc = Scope(; funcs=funcs)
    for (nm, v) in zip(f.ins[1:nfixed], args)
        nm != "~" && (sc.vars[nm] = v)
    end
    if varargin
        extra = args[nfixed+1:end]
        sc.vars["varargin"] = Cell(collect(Any, extra), (1, length(extra)))
    end
    sc.nargin, sc.nargout = length(args), nargout
    I.depth += 1
    if I.depth > 60
        I.depth -= 1
        merr("maximum recursion depth exceeded")
    end
    push!(I.scope_stack, sc)
    try
        exec_block(I, f.body, sc)
    catch e
        if e isa ReturnSig
        elseif e isa BreakSig || e isa ContinueSig
            merr("break/continue outside a loop")
        else
            rethrow()
        end
    finally
        I.depth -= 1
        pop!(I.scope_stack)
    end
    outs = Any[]
    for (i, o) in enumerate(f.outs[1:min(end, max(nargout, 1))])
        v = get(sc.vars, o, MISSING_)
        if v === MISSING_
            (i == 1 && nargout == 0) && break
            merr("$(f.name): output argument '$o' is not assigned")
        end
        v isa Tainted && arr(v)
        push!(outs, v)
    end
    outs
end

function call_handle(I::Interp, h::Handle, args, nargout)
    tick(I)
    h.name !== nothing && return mcall(I, h.name, args, nargout, Scope(; funcs=something(h.files, Dict{String,Any}())))
    length(args) > length(h.params) && merr("anonymous function called with too many inputs")
    sc = Scope(copy(h.env); funcs=something(h.files, Dict{String,Any}()))
    for (nm, v) in zip(h.params, args)
        nm != "~" && (sc.vars[nm] = v)
    end
    push!(I.scope_stack, sc)
    try
        return ev_multi(I, h.body, sc, nargout)
    finally
        pop!(I.scope_stack)
    end
end

"Julia callable for a function argument of fzero/fsolve (handle or function name)."
function mcallable(I::Interp, f)
    f isa Handle && return (a...) -> _one(call_handle(I, f, collect(Any, a), 1))
    if f isa AbstractString && occursin(r"^[A-Za-z_]\w*$", f)
        sc = I.scope_stack[end]
        return (a...) -> _one(mcall(I, f, collect(Any, a), 1, sc))
    end
    merr("a function handle or function name is required")
end
_one(r) = isempty(r) ? merr("function returns no value") : r[1]

# ---- statements
function run_script(I::Interp, u::Unit, sc::Scope)
    saved = sc.funcs
    sc.funcs = merge(saved, u.funcs)
    try
        exec_block(I, u.body, sc)
    catch e
        e isa ReturnSig || rethrow()
    finally
        sc.funcs = saved
    end
end
"Body of an `eval`."
function run_parsed(I::Interp, body, sc, nout)
    if nout > 0
        (length(body) != 1 || !(body[1] isa ExprStmt)) && merr("eval: the string must be an expression when outputs are requested")
        return ev_multi(I, body[1].e, sc, nout)
    end
    exec_block(I, body, sc)
    Any[]
end
function run_eval(I::Interp, code, sc, nout)
    code isa AbstractString || throw(Unsupported("eval of a non-string", "eval"))
    body, funcs = try
        mparse(String(code))
    catch e
        e isa ParseError || rethrow()
        throw(Unsupported("eval: cannot parse $(repr(first(code, 60))): $(e.msg)", "eval"))
    end
    run_parsed(I, body, sc, nout)
end

exec_block(I::Interp, stmts, sc) = (for s in stmts; mexec(I, s, sc); end)

function mexec(I::Interp, s, sc)
    tick(I)
    if s isa Assign || s isa ExprStmt || s isa CommandS || s isa GlobalS || s isa Unparsed
        return exec_simple(I, s, sc)
    end
    if s isa IfS
        for (cond, body) in s.clauses
            truth(ev(I, cond, sc)) && return exec_block(I, body, sc)
        end
        return exec_block(I, s.orelse, sc)
    end
    if s isa ForS
        values = for_values(I, s, sc)
        isempty(values) && setvar(I, sc, s.var, I.empty_loop_value)
        for v in values
            tick(I)
            setvar(I, sc, s.var, v)
            try
                exec_block(I, s.body, sc)
            catch e
                e isa BreakSig && break
                e isa ContinueSig && continue
                rethrow()
            end
        end
        return nothing
    end
    if s isa WhileS
        while truth(ev(I, s.cond, sc))
            tick(I)
            try
                exec_block(I, s.body, sc)
            catch e
                e isa BreakSig && break
                e isa ContinueSig && continue
                rethrow()
            end
        end
        return nothing
    end
    if s isa SwitchS
        body = switch_body(I, s, sc)
        return (body === nothing || isempty(body)) ? nothing : exec_block(I, body, sc)
    end
    if s isa TryS
        saved_scopes, saved_end, saved_depth = length(I.scope_stack), length(I.end_stack), I.depth
        try
            return exec_block(I, s.body, sc)
        catch e0
            e = matlab_error(e0)                 # a Julia exception from an operator is a MATLAB error too
            e isa MatlabError || rethrow()
            resize!(I.scope_stack, saved_scopes)
            resize!(I.end_stack, saved_end)
            I.depth = saved_depth
            if s.ident !== nothing
                setvar(I, sc, s.ident, MStruct(["message" => e.msg, "identifier" => e.id, "stack" => MStruct()]))
            end
            return exec_block(I, s.handler, sc)
        end
    end
    s isa BreakS && throw(BreakSig())
    s isa ContinueS && throw(ContinueSig())
    s isa ReturnS && throw(ReturnSig())
    s isa DynS && _unsup("Dynare statement inside a function")
    _unsup("statement $(typeof(s))")
end

function for_values(I::Interp, s, sc)
    v = ev(I, s.e, sc)
    I.empty_loop_value = (v isa AbstractMatrix && size(v) == (0, 0)) ? zeros(0, 0) : zeros(1, 0)
    if v isa Cell
        v.shape[1] <= 1 || _unsup("for over a cell matrix")
        return Any[Cell(Any[x], (1, 1)) for x in v.items]
    end
    v isa AbstractString && return Any[string(c) for c in v]
    a = arr(v)
    isempty(a) && return Any[]
    size(a, 1) == 1 && return Any[eltype(a) == Bool ? x : Float64(x) for x in a[1, :]]
    Any[ret(a[:, j:j]) for j in 1:size(a, 2)]
end

function switch_body(I::Interp, s, sc)
    v = ev(I, s.e, sc)
    function mmatch(c)
        c isa Cell && return any(mmatch(x) for x in c.items)
        if v isa AbstractString || c isa AbstractString
            return v isa AbstractString && c isa AbstractString && v == c
        end
        a, b = arr(v), arr(c)
        length(a) != 1 && merr("switch expression must be a scalar or a char array")
        length(b) == 1 && a[1] == b[1]
    end
    for (ce, body) in s.cases
        mmatch(ev(I, ce, sc)) && return body
    end
    s.default
end

function exec_simple(I::Interp, s, sc)
    if s isa Assign
        tg = s.targets
        if length(tg) == 1
            rhs = s.rhs
            if rhs isa MatrixLit && isempty(rhs.rows) && tg[1] isa IndexE && !tg[1].brace
                return assign_to(I, tg[1], zeros(0, 0), sc)        # x(idx) = []
            end
            vals = ev_multi(I, rhs, sc, 1)
            isempty(vals) && merr("the right-hand side of the assignment returns no value")
            return assign_to(I, tg[1], vals[1], sc)
        end
        vals = ev_multi(I, s.rhs, sc, length(tg))
        last = something(findlast(x -> x !== nothing, tg), 0)
        length(vals) < last && merr("the right-hand side returns $(length(vals)) value(s), $last requested")
        for (x, v) in zip(tg, vals)
            x === nothing || assign_to(I, x, v, sc)
        end
        return nothing
    end
    if s isa ExprStmt
        e = s.e
        vals = ev_multi(I, e, sc, 0)
        (!isempty(vals) && !(e isa Id)) && (sc.vars["ans"] = vals[1])
        return nothing
    end
    s isa CommandS && return command(I, s, sc)
    if s isa GlobalS
        for nm in s.names
            haskey(sc.vars, nm) && _unsup("global declaration of the existing local variable '$nm'")
            (nm in HOST_GLOBALS && I.host === nothing) && throw(Unsupported("global $nm without a model", "function"))
            push!(sc.global_names, nm)
            haskey(I.globals, nm) || (I.globals[nm] = zeros(0, 0))
        end
        return nothing
    end
    s isa Unparsed && throw(Unsupported("cannot parse: $(s.why)", "syntax"))
    _unsup("statement $(typeof(s))")
end

function command(I::Interp, s, sc)
    name, args = s.name, split(s.args)
    if name in ("clear", "clearvars")
        if any(a -> startswith(a, "-") || a in ("all", "global", "functions", "classes", "mex", "java", "import"), args)
            throw(Unsupported("'$(join(split(something(s.text, "clear")), " "))' (clears more than local variables)", "eval"))
        end
        if isempty(args) || args == ["variables"]
            empty!(sc.vars)
        else
            for a in args
                if occursin('*', a) || occursin('?', a)
                    pat = Regex("^" * replace(replace(replace(a, r"([.+^$()\[\]{}|\\])" => s"\\\1"), "*" => ".*"), "?" => ".") * "\$")
                    for k in [k for k in keys(sc.vars) if occursin(pat, k)]
                        delete!(sc.vars, k)
                    end
                elseif haskey(sc.vars, a)
                    delete!(sc.vars, a)
                end
            end
        end
        return nothing
    end
    name == "load" && return load_file(I, collect(String, args), sc)
    if name == "run"
        push!(I.scope_stack, sc)
        try
            return BUILTINS["run"](I, Any[strip(join(args, " "), ['\'', '"'])], 0)
        finally
            pop!(I.scope_stack)
        end
    end
    (name == "disp" || (haskey(LAZY, name) && LAZY[name] === _noop)) && return nothing
    haskey(LAZY, name) && return LAZY[name](I, Any[], sc, 0)
    throw(Unsupported("command '$name'", "function"))
end

load_file(I::Interp, args, sc) = I.load_hook === nothing ? throw(Unsupported("load", "file")) : I.load_hook(I, args, sc)

function assign_to(I::Interp, t, value, sc)
    value isa Tainted && arr(value)
    t isa Id && return setvar(I, sc, t.name, value)
    chain = Any[]
    e = t
    while !(e isa Id)
        push!(chain, e)
        e = e.base
    end
    reverse!(chain)
    cur = getvar(I, sc, e.name)
    cur === MISSING_ && (cur = nothing)
    new = _assign_chain(I, cur, chain, 1, value, sc)
    setvar(I, sc, e.name, new)
end

function _assign_chain(I::Interp, cur, chain, i, value, sc)
    cur isa Tainted && arr(cur)
    acc = chain[i]
    last = i == length(chain)
    if acc isa FieldE
        name = acc.name !== nothing ? acc.name : ev(I, acc.dyn, sc)
        name isa AbstractString || merr("dynamic field name must be a char array")
        if cur isa HostStruct
            if !last && i == length(chain) - 1 && chain[i+1] isa IndexE && !chain[i+1].brace && can_assign_sub(cur)
                sub = peek_field(cur, name; raw=true)
                args = index_args(I, sub, chain[i+1].args, sc)
                assign_sub!(cur, name, args, value)
                return cur
            end
            set_field!(cur, name, last ? value : _assign_chain(I, peek_field(cur, name), chain, i + 1, value, sc))
            return cur
        end
        (cur === nothing || (cur isa AbstractMatrix && isempty(cur))) && (cur = MStruct())
        cur isa MStruct || merr("field assignment to a $(mclass(cur)) value")
        s = clone(cur)
        s[name] = last ? value : _assign_chain(I, get(s, name, nothing), chain, i + 1, value, sc)
        return s
    end
    args = index_args(I, cur, acc.args, sc)
    if acc.brace
        last && return assign_brace(cur, args, value)
        sub = try
            cur isa Cell ? index_brace(cur, args) : nothing
        catch e
            e isa MatlabError || rethrow()
            nothing
        end
        return assign_brace(cur, args, _assign_chain(I, sub, chain, i + 1, value, sc))
    end
    last && return assign_index(cur, args, value)
    if (cur === nothing || cur isa MStruct) && all(x -> x isa Real && x == 1, args)
        return _assign_chain(I, cur, chain, i + 1, value, sc)
    end
    _unsup("struct arrays")
end

"Run MATLAB source in a fresh interpreter (no model); returns the workspace."
function mrun(text; variables=Dict{String,Any}(), directory=nothing)
    I = Interp(directory)
    merge!(I.base.vars, variables)
    body, funcs = mparse(text)
    I.base.funcs = funcs
    try
        exec_block(I, body, I.base)
    catch e
        e isa ReturnSig || rethrow()
    end
    I.base.vars
end

bi!("clear", "clearvars") do I, a, nout
    (nout > 0 || !all(x -> x isa AbstractString, a)) && merr("clear: arguments must be char arrays")
    c = CommandS("clear", join(a, " "), "clear " * join(a, " "))
    r = command(I, c, I.scope_stack[end])
    r === nothing ? Any[] : r
end
bi!("load") do I, a, nout
    (nout > 0 || isempty(a) || !all(x -> x isa AbstractString, a)) && throw(Unsupported("load with an output or computed arguments", "file"))
    r = load_file(I, collect(String, a), I.scope_stack[end])
    r === nothing ? Any[] : r
end

# host interface defaults (overridden for the Dynare host in session.jl)
function host_M end
function host_options end
function host_oo end
function host_is_param end
function host_set_param end
function host_get_param end
function host_set_shock_stderr end
function snapshot end
function restore! end
function get_field end
function has_field end
function field_names end
function peek_field end
function set_field! end
function assign_sub! end
can_assign_sub(::Any) = false
