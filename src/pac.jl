# Dynare's semi-structural toolbox: var_model, trend_component_model, var_expectation_model / var_expectation,
# pac_model / pac_expectation / pac_target_info, and the MATLAB helpers that work on them.
#
# What the preprocessor does (also with language=julia): substitutes auxiliary variables and parameters into the
# model block (h0/h1 parameters of pac_expectation, the expectation parameters of var_expectation, ...), so the model
# equations load as usual. What it writes only into its MATLAB output, and so is read here from a second preprocessor
# run in MATLAB mode (kept under <model>/model/semi): the layouts `M_.var.<name>`, `M_.trend_component.<name>`,
# `M_.var_expectation.<name>`, `M_.pac.<name>` (driver.m) and the coefficient files varmatrices.m,
# trend_component_ar_a0.m, pac_target_coefficients.m.
# What Dynare computes at run time (companion matrices, the h vectors of PAC, the expectation parameters,
# the MCE parameters) is ported from Dynare 7.2's matlab/+pac, matlab/+var_expectation, get_companion_matrix.m and
# matlab/pac-tools, and registered as MATLAB functions of the interpreter (`get_companion_matrix`, `pac.initialize`,
# `pac.update.expectation`, `var_expectation.update`, ...). They write M_.params of the running .mod session and
# M_.var / M_.pac / ... (kept in `model.cache[:semi_M]`, copied to M_ of the workspace), and oo_.var.<name>.CompanionMatrix.

const _SEMI_KINDS = ("var_model", "trend_component_model", "var_expectation_model", "pac_model", "pac_target_info")
const _SEMI_TOP = ("var", "trend_component", "var_expectation", "pac")
const _SEMI_FILES = ("driver.m", "varmatrices.m", "trend_component_ar_a0.m", "pac_target_coefficients.m")

# ---------------------------------------------------------------------- preprocessor, second run (MATLAB output)
"Does the JSON of the preprocessor hold one of the semi-structural statements?"
_has_semi_statements(json_text::AbstractString) =
    occursin(r"\"statementName\"\s*:\s*\"(var_model|trend_component_model|var_expectation_model|pac_model|pac_target_info)\"", json_text)

"""
Run the preprocessor in MATLAB mode (on the macro-expanded file, in a scratch folder) and keep driver.m and the coefficient
files in `<model>/model/semi`: they hold what the language=julia output leaves out (M_.var, M_.pac, ..., varmatrices.m).
"""
function _semi_prepare(exe, modfile, extra_args, jd)
    text = read(joinpath(jd, "modfile.json"), String)
    _has_semi_statements(text) || return
    name = splitext(basename(modfile))[1]
    dest = joinpath(dirname(jd), "semi")
    tmp = mktempdir()
    try
        expanded = joinpath(tmp, name * ".mod")
        out = IOBuffer(); err = IOBuffer()
        proc = run(pipeline(ignorestatus(Cmd(`$exe $(basename(modfile)) savemacro=$expanded onlymacro $extra_args`; dir=dirname(modfile)));
                            stdout=out, stderr=err))
        (proc.exitcode == 0 && isfile(expanded)) || error("preprocessor failed (macro expansion for the semi-structural objects):\n$(String(take!(out)))\n$(String(take!(err)))")
        src = read(expanded, String)
        src = replace(src, r"\bmodel\s*\(([^()]*)\)" => m -> begin
            inner = join(filter(o -> strip(o) != "use_dll", split(match(r"\(([^()]*)\)", m).captures[1], ",")), ",")
            isempty(strip(inner)) ? "model" : "model($inner)"
        end)
        src = replace(src, r"(--\+\s*options:[^\n]*?)\buse_dll\b" => s"\1")
        write(expanded, src)
        out = IOBuffer(); err = IOBuffer()
        proc = run(pipeline(ignorestatus(Cmd(`$exe $(name).mod nolog`; dir=tmp)); stdout=out, stderr=err))
        proc.exitcode == 0 || error("preprocessor failed (MATLAB output for the semi-structural objects):\n$(String(take!(out)))\n$(String(take!(err)))")
        rm(dest; recursive=true, force=true)
        mkpath(dest)
        for f in _SEMI_FILES
            p = joinpath(tmp, "+" * name, f)
            isfile(p) && cp(p, joinpath(dest, f))
        end
        isfile(joinpath(dest, "driver.m")) || error("the preprocessor wrote no driver.m for the semi-structural objects")
    finally
        rm(tmp; recursive=true, force=true)
    end
end

# ---------------------------------------------------------------------- reading driver.m
const _PATH_TOKEN = r"^([A-Za-z_]\w*)(?:\((\d+)\)|\{(\d+)\})?$"

function _split_top(s::AbstractString, seps)
    parts = String[]; cur = IOBuffer(); inq = false
    for c in s
        if c == '\''
            inq = !inq
            write(cur, c)
        elseif !inq && c in seps
            push!(parts, String(take!(cur)))
        else
            write(cur, c)
        end
    end
    push!(parts, String(take!(cur)))
    parts
end

function _driver_scalar(tok::AbstractString)
    tok == "true" && return true
    tok == "false" && return false
    tok == "NaN" && return NaN
    tok == "Inf" && return Inf
    tok == "-Inf" && return -Inf
    v = tryparse(Float64, tok)
    v === nothing && throw(ModelError("semi-structural objects: cannot read the value $(repr(tok)) of the preprocessor's driver.m"))
    v
end

function _driver_value(s::AbstractString, root::MStruct)
    s = strip(s)
    if startswith(s, "[") && endswith(s, "]")
        toks = filter(!isempty, split(strip(s[2:end-1]), r"[\s,]+"))
        isempty(toks) && return zeros(0, 0)
        vals = Any[_driver_scalar(t) for t in toks]
        v = all(x -> x isa Bool, vals) ? Matrix{Bool}(reshape(Bool[x for x in vals], 1, :)) :
            reshape(Float64[x for x in vals], 1, :)
        return ret(v)
    elseif startswith(s, "{") && endswith(s, "}")
        items = Any[]
        for p in _split_top(strip(s[2:end-1]), (';', ','))
            p = strip(p)
            isempty(p) && continue
            (startswith(p, "'") && endswith(p, "'")) || throw(ModelError("semi-structural objects: cannot read the cell item $(repr(p))"))
            push!(items, replace(p[2:end-1], "''" => "'"))
        end
        return Cell(items, (length(items), 1))
    elseif startswith(s, "'") && endswith(s, "'")
        return replace(s[2:end-1], "''" => "'")
    elseif startswith(s, "M_.")
        cur = root
        for tok in split(s[4:end], ".")
            cur = cur[String(tok)]
        end
        return deepcopy(cur)
    end
    _driver_scalar(s)
end

"Assign `value` at the path (`a.b(2).c{1}.d`) of a struct, creating structs, struct arrays (Vector{MStruct}) and cells."
function _driver_assign!(root::MStruct, toks, value)
    cur = root
    for (k, tok) in enumerate(toks)
        mt = match(_PATH_TOKEN, tok)
        mt === nothing && throw(ModelError("semi-structural objects: cannot read the field path $(join(toks, "."))"))
        nm = String(mt.captures[1])
        last = k == length(toks)
        if mt.captures[2] === nothing && mt.captures[3] === nothing
            if last
                cur[nm] = value
            else
                nxt = get(cur, nm, nothing)
                nxt isa MStruct || (nxt = MStruct(); cur[nm] = nxt)
                cur = nxt
            end
        elseif mt.captures[2] !== nothing                              # struct array element
            i = parse(Int, mt.captures[2])
            arr_ = get(cur, nm, nothing)
            arr_ isa Vector{MStruct} || (arr_ = MStruct[]; cur[nm] = arr_)
            while length(arr_) < i
                push!(arr_, MStruct())
            end
            last && throw(ModelError("semi-structural objects: unexpected struct array assignment"))
            cur = arr_[i]
        else                                                           # cell element
            i = parse(Int, mt.captures[3])
            c = get(cur, nm, nothing)
            c isa Cell || (c = Cell(Any[], (1, 0)); cur[nm] = c)
            while length(c.items) < i
                push!(c.items, MStruct())
            end
            c = Cell(c.items, (1, length(c.items)))
            cur[nm] = c
            last && throw(ModelError("semi-structural objects: unexpected cell assignment"))
            nxt = c.items[i]
            nxt isa MStruct || (nxt = MStruct(); c.items[i] = nxt)
            cur = nxt
        end
    end
end

"M_.var, M_.trend_component, M_.var_expectation and M_.pac of the preprocessor's driver.m, as MATLAB structs."
function _read_driver(text::AbstractString)
    S = MStruct()
    for line in split(text, '\n')
        mt = match(r"^M_\.(var|trend_component|var_expectation|pac)\.(.*?)\s*=\s*(.*?);?\s*$", line)
        mt === nothing && continue
        top = String(mt.captures[1])
        haskey(S, top) || (S[top] = MStruct())
        toks = split(String(mt.captures[2]), ".")
        _driver_assign!(S[top], toks, _driver_value(String(mt.captures[3]), S))
    end
    S
end

"Does M_ of this model have the semi-structural field `name` (var, pac, trend_component, var_expectation)?"
_semi_has(m, name) = name in _SEMI_TOP && haskey(m.cache, :semi_M) && haskey(m.cache[:semi_M], name)
_semi_oo_has(m, name) = name in _SEMI_TOP && haskey(m.cache, :semi_oo) && haskey(m.cache[:semi_oo], name)
semi_M(m::Model) = get!(() -> MStruct(), m.cache, :semi_M)::MStruct
"M_ fields of the semi-structural objects present in the model (for `build_M`)."
function _semi_into_M!(M::MStruct, m::Model)
    S = get(m.cache, :semi_M, nothing)
    S isa MStruct || return M
    for k in keys(S)
        M[k] = deepcopy(S[k])
    end
    M
end

"Load the semi-structural objects when the model has such statements (called by `Model` before the statements run)."
function _semi_init!(m::Model)
    kinds = [get(st, "statementName", "") for st in m.statements]
    any(k -> k in _SEMI_KINDS, kinds) || return m
    dir = joinpath(dirname(m.json_dir), "semi")
    isfile(joinpath(dir, "driver.m")) ||
        throw(ModelError("var_model / trend_component_model / var_expectation_model / pac_model need the preprocessor's MATLAB output (<model>/model/semi/driver.m): load the model with JDyn.load or dynare"))
    m.cache[:semi_M] = _read_driver(read(joinpath(dir, "driver.m"), String))
    files = Dict{String,String}()
    for f in _SEMI_FILES
        p = joinpath(dir, f)
        isfile(p) && (files[f] = read(p, String))
    end
    m.cache[:semi_files] = files
    m
end

# ---------------------------------------------------------------------- coefficient files (varmatrices.m, ...)
const _COEF_SAFE = r"^[\d\s.eE+\-*/^()]*(?:p\[\d+\][\d\s.eE+\-*/^()]*)*$"
"Compile the right-hand side of `x(i,j,k) = params(3)*(1-params(4));` into a function of the parameter vector."
function _coef_fun(expr::AbstractString)
    e = replace(strip(expr), r"params\((\d+)\)" => s"p[\1]")
    occursin(_COEF_SAFE, e) || throw(ModelError("semi-structural objects: unexpected coefficient expression $(repr(expr)) in the preprocessor's output"))
    e = replace(e, r"(?<![\w.])\.(\d)" => s"0.\1")          # MATLAB's .05
    f = Core.eval(@__MODULE__, Meta.parse("p -> " * e))
    p -> Base.invokelatest(f, p)
end

function _coef_blocks(text::AbstractString, what::AbstractString)
    blocks = Dict{String,Vector{String}}()
    cur = nothing
    for line in split(text, '\n')
        t = strip(line)
        mt = match(r"^if\s+strcmp\(\s*model_name\s*,\s*'([^']+)'\s*\)", t)
        if mt !== nothing
            cur = String(mt.captures[1]); blocks[cur] = String[]
        elseif cur !== nothing
            t == "return" ? (cur = nothing) : push!(blocks[cur], String(t))
        end
    end
    blocks
end

"`varmatrices(model_name, params, reducedform)` of the preprocessor's file: (ar 3-D, a0, constants)."
function _varmatrices(m::Model, name::AbstractString, params::Vector{Float64}, reducedform::Bool)
    txt = get(get(m.cache, :semi_files, Dict{String,String}()), "varmatrices.m", nothing)
    txt === nothing && merr("$name is not a valid var_model name")
    blocks = get!(() -> _coef_blocks(txt, "varmatrices"), m.cache, :semi_varmat_blocks)::Dict{String,Vector{String}}
    haskey(blocks, name) || merr("$name is not a valid var_model name")
    ar = nothing; a0 = nothing; cst = nothing
    for t in blocks[name]
        t = rstrip(t, ';')
        if (mt = match(r"^ar\s*=\s*zeros\((\d+),\s*(\d+),\s*(\d+)\)$", t)) !== nothing
            ar = zeros(parse.(Int, mt.captures)...)
        elseif (mt = match(r"^ar\((\d+),(\d+),(\d+)\)\s*=\s*(.*)$", t)) !== nothing
            i, j, k = parse.(Int, mt.captures[1:3])
            ar[i, j, k] = _coef_fun(mt.captures[4])(params)
        elseif (mt = match(r"^a0\s*=\s*eye\((\d+)\)$", t)) !== nothing
            a0 = Matrix{Float64}(LinearAlgebra.I, parse(Int, mt.captures[1]), parse(Int, mt.captures[1]))
        elseif (mt = match(r"^a0\((\d+),(\d+)\)\s*=\s*(.*)$", t)) !== nothing
            a0[parse(Int, mt.captures[1]), parse(Int, mt.captures[2])] = _coef_fun(mt.captures[3])(params)
        elseif (mt = match(r"^constants\s*=\s*zeros\((\d+),\s*1\)$", t)) !== nothing
            cst = zeros(parse(Int, mt.captures[1]))
        elseif (mt = match(r"^constants\((\d+)\)\s*=\s*(.*)$", t)) !== nothing
            cst[parse(Int, mt.captures[1])] = _coef_fun(mt.captures[2])(params)
        end
    end
    ar === nothing && merr("$name is not a valid var_model name")
    n = size(ar, 1)
    a0 === nothing && (a0 = Matrix{Float64}(LinearAlgebra.I, n, n))
    cst === nothing && (cst = zeros(n))
    if reducedform
        for i in axes(ar, 3)
            ar[:, :, i] = a0 \ ar[:, :, i]
        end
        cst = a0 \ cst
        a0 = Matrix{Float64}(LinearAlgebra.I, n, n)
    end
    ar, a0, cst
end

"`trend_component_ar_a0(model_name, params)`: (AR 3-D, A0, A0star)."
function _tc_matrices(m::Model, name::AbstractString, params::Vector{Float64})
    txt = get(get(m.cache, :semi_files, Dict{String,String}()), "trend_component_ar_a0.m", nothing)
    txt === nothing && merr("$name is not a valid trend_component_model name")
    blocks = get!(() -> _coef_blocks(txt, "tc"), m.cache, :semi_tc_blocks)::Dict{String,Vector{String}}
    haskey(blocks, name) || merr("$name is not a valid trend_component_model name")
    AR = nothing; A0 = nothing; A0s = nothing
    for t in blocks[name]
        t = rstrip(t, ';')
        (isempty(t) || startswith(t, "%")) && continue
        if (mt = match(r"^AR\s*=\s*zeros\((\d+),\s*(\d+),\s*(\d+)\)$", t)) !== nothing
            AR = zeros(parse.(Int, mt.captures)...)
        elseif (mt = match(r"^AR\((\d+),\s*(\d+),\s*(\d+)\)\s*=\s*(.*)$", t)) !== nothing
            i, j, k = parse.(Int, mt.captures[1:3])
            AR[i, j, k] = _coef_fun(mt.captures[4])(params)
        elseif (mt = match(r"^A0\s*=\s*zeros\((\d+),\s*(\d+)\)$", t)) !== nothing
            A0 = zeros(parse.(Int, mt.captures)...)
        elseif (mt = match(r"^A0\((\d+),\s*(\d+)\)\s*=\s*(.*)$", t)) !== nothing
            A0[parse(Int, mt.captures[1]), parse(Int, mt.captures[2])] = _coef_fun(mt.captures[3])(params)
        elseif (mt = match(r"^A0star\s*=\s*zeros\((\d+),\s*(\d+)\)$", t)) !== nothing
            A0s = zeros(parse.(Int, mt.captures)...)
        elseif (mt = match(r"^A0star\((\d+),\s*(\d+)\)\s*=\s*(.*)$", t)) !== nothing
            A0s[parse(Int, mt.captures[1]), parse(Int, mt.captures[2])] = _coef_fun(mt.captures[3])(params)
        end
    end
    (AR === nothing || A0 === nothing || A0s === nothing) && throw(ModelError("semi-structural objects: cannot read trend_component_ar_a0.m for $name"))
    AR, A0, A0s
end

"`pac_target_coefficients(model_name, params)` of the preprocessor's file."
function _pac_target_coefficients(m::Model, name::AbstractString, params::Vector{Float64})
    txt = get(get(m.cache, :semi_files, Dict{String,String}()), "pac_target_coefficients.m", nothing)
    txt === nothing && merr("Unknown PAC model: $name")
    blocks = get!(() -> _coef_blocks(txt, "pac"), m.cache, :semi_pactarget_blocks)::Dict{String,Vector{String}}
    haskey(blocks, name) || merr("Unknown PAC model: $name")
    c = nothing
    for t in blocks[name]
        t = rstrip(t, ';')
        if (mt = match(r"^coeffs\s*=\s*NaN\((\d+),\s*(\d+)\)$", t)) !== nothing
            c = fill(NaN, parse(Int, mt.captures[1]), parse(Int, mt.captures[2]))
        elseif (mt = match(r"^coeffs\((\d+)\)\s*=\s*(.*)$", t)) !== nothing
            c[parse(Int, mt.captures[1])] = _coef_fun(mt.captures[2])(params)
        end
    end
    c === nothing && merr("Unknown PAC model: $name")
    ret(c)
end

# ---------------------------------------------------------------------- access helpers
function _need_host(I)
    h = I.host
    (h isa Host) || throw(Unsupported("this function needs a model", "function"))
    h
end
function _semi_params(h::Host)
    isempty(h.tainted) || throw(Unsupported("the semi-structural functions need all parameters: the value of some is unknown ($(h.taint_why))", "tainted"))
    h.params
end
function _setp!(h::Host, idx, vals)
    idx = Int.(vec(idx)); vals = Float64.(vec(vals))
    length(vals) == 1 && length(idx) != 1 && (vals = fill(vals[1], length(idx)))
    length(idx) == length(vals) || merr("=: nonconformant arguments (op1 is 1x$(length(idx)), op2 is 1x$(length(vals)))")
    for (i, v) in zip(idx, vals)
        h.params[i] = v
        delete!(h.tainted, i)
    end
    h.nwrites += 1
    nothing
end

_f(s, name) = (s isa MStruct && haskey(s, name)) ? s[name] : merr("Reference to non-existent field '$name'.")
_has(s, name) = s isa MStruct && haskey(s, name)
_vec(x) = x isa Real ? Float64[x] : x isa AbstractMatrix ? vec(Float64.(x)) : x isa Cell ? x.items : merr("expected a numeric vector")
_ivec(x) = Int.(_vec(x))
_bvec(x) = x isa Bool ? Bool[x] : x isa AbstractMatrix{Bool} ? vec(x) : Bool.(_vec(x) .!= 0)
_names(c) = String[String(x) for x in (c isa Cell ? c.items : [c])]

"Auxiliary variable record (a Dict of the preprocessor's JSON) of endogenous variable `idx`, or nothing."
function _aux_info(m::Model, idx::Integer)
    idx <= orig_endo_nbr(m) && return nothing
    for a in get_or(m.raw, "aux_vars", [])
        Int(a["endo_index"]) == idx && return a
    end
    nothing
end
"get_aux_variable_id on an endogenous index: the auxiliary variable record or nothing."
_aux_of(m::Model, v) = v isa AbstractString ? (i = get(m.ie, v, 0); i == 0 ? nothing : _aux_info(m, i)) : _aux_info(m, Int(v))
"Matlab's get_difference_order."
function _diff_order(m::Model, var)
    n = 0
    a = _aux_of(m, var)
    while a !== nothing
        t = Int(a["type"])
        if t == 8
            n += 1
            v = Int(a["orig_index"])
            a = v <= orig_endo_nbr(m) ? nothing : _aux_info(m, v)
        elseif t == 9
            a = _aux_info(m, Int(a["orig_index"]))
        else
            break
        end
    end
    n
end
"Matlab's isauxiliary(name, types)."
function _isaux(m::Model, name::AbstractString, types=nothing)
    id = get(m.ie, name, 0)
    (id == 0 || id <= orig_endo_nbr(m)) && return false
    types === nothing && return true
    a = _aux_info(m, id)
    a === nothing && return false
    Int(a["type"]) in types
end

function _oo_root(h::Host)
    r = h.oo.results
    r isa MStruct && return r
    get!(() -> MStruct(), h.model.cache, :semi_oo)::MStruct
end
function _oo_get(h::Host, type, name)
    r = _oo_root(h)
    (_has(r, type) && _has(r[type], name)) || return nothing
    r[type][name]
end
function _oo_put!(h::Host, type, name, field, value)
    r = _oo_root(h)
    _has(r, type) && r[type] isa MStruct || (r[type] = MStruct())
    _has(r[type], name) && r[type][name] isa MStruct || (r[type][name] = MStruct())
    r[type][name][field] = value
end

# ---------------------------------------------------------------------- get_companion_matrix
function _aux_model_type(S, name)
    (_has(S, "var") && _has(S.var, name)) && return "var"
    (_has(S, "trend_component") && _has(S.trend_component, name)) && return "trend_component"
    merr("Unknown type of auxiliary model.")
end

"Dynare's get_companion_matrix: returns (A0, A0star, AR, B) and fills oo_.<type>.<name>.CompanionMatrix and M_.<type>.<name>.list_of_variables_in_companion_var."
function semi_companion!(h::Host, name::AbstractString, type=nothing)
    m = h.model; S = semi_M(m)
    type === nothing && (type = _aux_model_type(S, name))
    type in ("var", "trend_component") || merr("Unknown type of auxiliary model.")
    _has(S, type) && _has(S[type], name) || merr("Unknown type of auxiliary model.")
    obj = S[type][name]
    params = _semi_params(h)
    lhs = _ivec(_f(obj, "lhs"))
    n = length(lhs)
    if type == "var"
        structural = _f(obj, "structural") === true
        AR, _, Cst = _varmatrices(m, name, params, structural)
        isconstant = any(abs.(Cst) .> 0)
        obj["isconstant"] = isconstant
        p = size(AR, 3)
        nn = n * p + isconstant
        C = zeros(nn, nn)
        c = isconstant
        C[c+1:c+n, c+1:c+n] = AR[:, :, 1]
        for i in 2:p
            C[c+1:c+n, c+(i-1)*n+1:c+i*n] = AR[:, :, i]
            C[c+(i-1)*n+1:c+i*n, c+(i-2)*n+1:c+(i-1)*n] = Matrix{Float64}(LinearAlgebra.I, n, n)
        end
        if isconstant
            C[1, 1] = 1.0
            for i in 1:n
                C[1+i, 1] = Cst[i]
            end
        end
        _oo_put!(h, "var", name, "CompanionMatrix", C)
        obj["list_of_variables_in_companion_var"] = Cell(Any[m.endo_names[i] for i in lhs], (n, 1))
        return Any[zeros(0, 0), zeros(0, 0), AR, zeros(0, 0)]
    end
    AR, A0, A0star = _tc_matrices(m, name, params)
    p = size(AR, 3)
    targets = _bvec(_f(obj, "targets"))
    q = count(targets)
    mm = n - q
    target_eqn = _ivec(_f(obj, "target_eqn"))
    ecm = findall(!, targets)
    for i in 1:mm
        _diff_order(m, lhs[ecm[i]]) == 0 &&
            merr("$name is not a trend component model. The LHS variables should be in differences")
    end
    B = zeros(mm + q, mm + q, p + 1)
    B[ecm, ecm, 1] = Matrix{Float64}(LinearAlgebra.I, mm, mm) + A0 + AR[:, :, 1]
    B[ecm, target_eqn, 1] = -A0star
    B[target_eqn, target_eqn, 1] = Matrix{Float64}(LinearAlgebra.I, q, q)
    for i in 2:p
        B[ecm, ecm, i] = AR[:, :, i] - AR[:, :, i-1]
    end
    B[ecm, ecm, p+1] = -AR[:, :, p]
    nb = size(B, 1)
    C = zeros(nb * (p + 1), nb * (p + 1))
    for i in 1:p
        C[1:n, (i-1)*n+1:i*n] = B[:, :, i]
        C[i*n+1:(i+1)*n, (i-1)*n+1:i*n] = Matrix{Float64}(LinearAlgebra.I, n, n)
    end
    C[1:n, p*n+1:(p+1)*n] = B[:, :, p+1]
    _oo_put!(h, "trend_component", name, "CompanionMatrix", C)
    lst = Any[m.endo_names[i] for i in lhs]
    for i in 1:mm
        a = _aux_of(m, lst[ecm[i]])
        a === nothing && merr("This is a bug. Please contact the Dynare Team.")
        Int(a["type"]) == 8 || merr("This is a bug. Please contact the Dynare Team.")
        lst[ecm[i]] = m.endo_names[Int(a["orig_index"])]
    end
    obj["list_of_variables_in_companion_var"] = Cell(lst, (n, 1))
    Any[A0, A0star, AR, B]
end

# ---------------------------------------------------------------------- var_expectation
_isint(x) = isfinite(x) && x == round(x)

function var_expectation_initialize!(h::Host, name::AbstractString)
    S = semi_M(h.model)
    _has(S, "var_expectation") && _has(S.var_expectation, name) || merr("Reference to non-existent field '$name'.")
    ve = S.var_expectation[name]
    aux = String(_f(ve, "auxiliary_model_name"))
    type = _aux_model_type(S, aux)
    ve["auxiliary_model_type"] = type
    semi_companion!(h, aux, type)
    nothing
end

function var_expectation_update!(h::Host, name::AbstractString)
    m = h.model; S = semi_M(m)
    (_has(S, "var_expectation") && _has(S.var_expectation, name)) || merr("VAR_EXPECTATION_MODEL $name is not defined!")
    ve = S.var_expectation[name]
    type = String(_f(ve, "auxiliary_model_type"))
    auxname = String(_f(ve, "auxiliary_model_name"))
    (_has(S, type) && _has(S[type], auxname)) || merr("Unknown VAR ($auxname) in VAR_EXPECTATION_MODEL ($name)!")
    auxmodel = S[type][auxname]
    calib = _oo_get(h, type, auxname)
    calib === nothing && merr("Auxiliary model $auxname has to be estimated or calibrated first!")
    (_has(calib, "CompanionMatrix") && !any(isnan, calib.CompanionMatrix)) ||
        merr("Auxiliary model $auxname has to be estimated first.\nPlease use *get_companion_matrix command first.")
    params = _semi_params(h)
    discount = _has(ve, "discount_value") ? Float64(ve.discount_value) : _has(ve, "discount_index") ? params[Int(ve.discount_index)] :
               merr("This is most likely a bug. Pleasse conntact the Dynare Team.")
    discount <= 0 && merr("The discount factor must be positive.")
    discount > 1 && merr("The discount cannot be greater than one.")
    timeshift = _has(ve, "time_shift") ? Float64(ve.time_shift) : 0.0
    timeshift > 0 && merr("Option time_shift of the var_expectation command cannot be positive.")
    exprvars = _ivec(ve.expr.vars)
    lst = _names(_f(auxmodel, "list_of_variables_in_companion_var"))
    ids = Int[]
    for v in exprvars
        j = findfirst(==(m.endo_names[v]), lst)
        j === nothing && merr("Cannot find variable $(m.endo_names[v]) in the companion VAR")
        push!(ids, j)
    end
    if _has(auxmodel, "isconstant") && auxmodel.isconstant === true
        ids .+= 1
    end
    horizon = _vec(ve.horizon)
    wrong = true
    if length(horizon) == 1
        wrong = !(isfinite(horizon[1]) && _isint(horizon[1]) && horizon[1] > 0)
    elseif length(horizon) == 2
        wrong = !(isfinite(horizon[1]) && _isint(horizon[1]) && horizon[1] >= 0 &&
                  (isinf(horizon[2]) || (_isint(horizon[2]) && horizon[2] > horizon[1])))
    end
    wrong && merr("horizon must be an integer scalar or an integer vector with two elements.")
    C = Matrix{Float64}(calib.CompanionMatrix)
    n = size(C, 1)
    alpha = zeros(1, n)
    consts = _vec(ve.expr.constants)
    alpha[1, ids] = consts
    eparams = _vec(ve.expr.params)
    pid = findall(!isnan, eparams)
    alpha[1, ids[pid]] = eparams[pid]
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    shift(P) = timeshift < 0 ? P * _mpow(C, Int(-timeshift)) : P
    local par
    if length(horizon) == 1
        hz = Int(horizon[1])
        par = hz == 1 ? discount * (alpha * C) : alpha * _mpow(discount * C, hz)
        par = shift(par)
    else
        DC = discount * C
        if horizon[1] == 0 && isinf(horizon[2])
            par = shift(alpha / (I_n - DC))
        elseif horizon[1] > 0 && isinf(horizon[2])
            tmp1 = zeros(n, n)
            for hh in 0:Int(horizon[1])-1
                tmp1 += _mpow(DC, hh)
            end
            tmp1 = shift(alpha * tmp1)
            tmp2 = shift(alpha / (I_n - DC))
            par = tmp2 - tmp1
        else
            tmp = zeros(n, n)
            for hh in Int(horizon[1]):Int(horizon[2])
                tmp += _mpow(DC, hh)
            end
            par = shift(alpha * tmp)
        end
    end
    pidx = _ivec(ve.param_indices)
    if type == "var" && !(_f(auxmodel, "isconstant") === true)
        _setp!(h, pidx[1:1], [0.0])
        _setp!(h, pidx[2:end], vec(par))
    else
        _setp!(h, pidx, vec(par))
    end
    nothing
end
_mpow(A::Matrix{Float64}, k::Int) = k == 0 ? Matrix{Float64}(LinearAlgebra.I, size(A)...) : A^k

# ---------------------------------------------------------------------- PAC
_a2alpha(a) = begin
    m = length(a)
    al = zeros(m)
    al[m] = a[m]
    m > 2 && (al[2:m-1] = a[2:m-1] - a[3:m])
    al[1] = a[1] - a[2] - 1
    al
end
_polyval(p, x) = (r = 0.0; for c in p; r = r * x + c; end; r)
_iota(n, idx) = (v = zeros(n); for i in (idx isa Integer ? (idx,) : idx); v[i] = 1.0; end; v)

function _build_G(params)
    all(isreal, params) && length(params) >= 2 || merr("Input argument has to be a vector with at least two elements. The last element is the discount factor.")
    m = length(params) - 1
    alpha = reverse(_a2alpha(params[1:m]))
    beta = params[end]
    (beta < eps() || beta > 1 - eps()) && merr("beta has to be a discount factor!")
    G = zeros(m, m)
    m > 1 && (G[1:m-1, 2:m] = Matrix{Float64}(LinearAlgebra.I, m - 1, m - 1))
    G[m, :] = -alpha .* reverse(cumprod(fill(beta, m)))
    G, alpha, beta
end

"Dynare's hVectors: (h, lrcp)."
function _hvectors(params, H::Matrix{Float64}, auxtype, kind, id)
    G, alpha, beta = _build_G(params)
    A = vcat(alpha, 1.0)
    A1 = _polyval(A, 1.0); Ab = _polyval(A, beta)
    m = length(alpha); n = size(H, 1)
    tmp = Matrix{Float64}(LinearAlgebra.I, n * m, n * m) - kron(G, transpose(H))
    em = _iota(m, m)
    rhs = kron(em, _iota(n, id))
    Imm = Matrix{Float64}(LinearAlgebra.I, m, m)
    local h
    if kind == "ll"
        h = A1 * Ab * (kron(em, H)' * (tmp \ rhs))
    elseif kind == "dd"
        h = A1 * Ab * (kron(em' * inv(Imm - G), H') * (tmp \ rhs))
    elseif kind == "dl"
        h = A1 * Ab * (kron(em' * inv(Imm - G), (H' - Matrix{Float64}(LinearAlgebra.I, size(H, 1), size(H, 1)))) * (tmp \ rhs))
    else
        merr("Unknown kind value in PAC model.")
    end
    lrcp = NaN
    if kind != "ll"
        d = A1 * Ab * (em' * inv((Imm - G) * (Imm - G)) * em)
        lrcp = 1 - sum(params[2:end-1]) - d
    end
    vec(h), lrcp
end

function pac_initialize!(h::Host, pacmodel::AbstractString)
    S = semi_M(h.model)
    (_has(S, "pac") && _has(S.pac, pacmodel)) || merr("Reference to non-existent field '$pacmodel'.")
    pm = S.pac[pacmodel]
    aux = _f(pm, "auxiliary_model_name")
    if aux isa AbstractString && isempty(aux)
        pm["model_consistent_expectations"] = true
    else
        pm["model_consistent_expectations"] = false
        semi_companion!(h, String(aux), String(_f(pm, "auxiliary_model_type")))
    end
    for f in ("additive", "optim_additive", "non_optimizing_behaviour")
        _has(pm, f) || continue
        k = length(_vec(pm[f].params))
        pm[f]["bgp"] = Cell(Any[false for _ in 1:k], (1, k))
    end
    nothing
end

function _bgp_val(c, i)
    c isa Cell || merr("bgp is not a cell")
    c.items[i]
end

function pac_update_parameters!(h::Host, pacname::AbstractString)
    m = h.model; S = semi_M(m)
    (_has(S, "pac") && _has(S.pac, pacname)) || merr("PAC model $pacname is not defined in the model block!")
    pm = S.pac[pacname]
    _f(pm, "model_consistent_expectations") === true &&
        merr("This function cannot be used with Model Consistent Expectation. Try pac.mce.parameters instead.")
    type = String(_f(pm, "auxiliary_model_type")); auxname = String(_f(pm, "auxiliary_model_name"))
    (_has(S, type) && _has(S[type], auxname)) || merr("Unknown auxiliary model ($auxname) in PAC model ($pacname)!")
    varmodel = S[type][auxname]
    varcalib = _oo_get(h, type, auxname)
    varcalib === nothing && merr("Auxiliary model $auxname has to be estimated first!")
    (_has(varcalib, "CompanionMatrix") && !any(isnan, varcalib.CompanionMatrix)) || merr("Auxiliary model $auxname has to be estimated first.")
    Cmat = Matrix{Float64}(varcalib.CompanionMatrix)
    P = _semi_params(h)
    comps = _has(pm, "components") ? pm.components : MStruct[]
    ncomp = length(comps)
    ncomp > 0 && type == "trend_component" && merr("Composite PAC target not allowed with trend component model.")
    ec = pm.ec; ar = pm.ar
    maxlag = Int(pm.max_lag)
    pacvalues = vcat(P[_ivec(ec.params)], P[_ivec(ar.params)[1:maxlag]])
    lst = _names(_f(varmodel, "list_of_variables_in_companion_var"))
    endo = m.endo_names

    function find_trend(target::Int, what)
        id = findfirst(==(endo[target]), lst)
        id !== nothing && return id
        ad = [k for k in eachindex(lst) if _isaux(m, lst[k], (8, 10))]
        isempty(ad) && merr(what)
        for k in ad
            a = _aux_of(m, lst[k])
            (a !== nothing && haskey(a, "endo_index") && Int(a["endo_index"]) == target) && return k
            (a !== nothing && Int(a["type"]) == 8 && Int(a["orig_index"]) == target) && return k
        end
        merr(what)
    end

    ids = Int[]
    if ncomp > 0
        for c in comps
            push!(ids, find_trend(Int(c.endo_var), "Cannot find the trend variable in the Companion VAR model."))
        end
    else
        tg = _ivec(ec.vars)[_bvec(ec.istarget)]
        length(tg) == 1 || merr("Cannot find the trend variable in the auxiliary VAR / Trend component model.")
        push!(ids, find_trend(tg[1], "Cannot find the trend variable in the auxiliary VAR / Trend component model."))
    end
    isconst = type == "var" && (_f(varmodel, "isconstant") === true)
    kinds = String[]
    if ncomp == 0
        local k0
        if type == "var"
            k0 = _bvec(_f(varmodel, "nonstationary"))[ids[1]] ? "dd" : "ll"
            isconst && (ids[1] += 1)
        else
            k0 = "dd"
        end
        pk = _f(pm, "kind")
        push!(kinds, isempty(pk) ? k0 : String(pk))
    else
        isconst && (ids .+= 1)
        for c in comps
            (!_has(c, "kind") || isempty(c.kind)) && merr("kind declaration is mandatory for each component in pac_target_info.")
            push!(kinds, String(c.kind))
        end
    end
    beta = P[Int(pm.discount_index)]
    growth_flag = _has(pm, "growth_str") || any(c -> _has(c, "growth_str"), comps)
    gamma = _has(pm, "non_optimizing_behaviour") ? P[Int(pm.share_of_optimizing_agents_index)] : 1.0
    hs = Vector{Vector{Float64}}(); gn = Float64[]
    for i in eachindex(ids)
        hv, lr = _hvectors(vcat(pacvalues, beta), Cmat, type, kinds[i], ids[i])
        push!(hs, hv); push!(gn, lr)
    end
    if type == "var"
        if _has(pm, "h_param_indices")
            hi = _ivec(pm.h_param_indices)
            isconst ? _setp!(h, hi, hs[1]) : (_setp!(h, hi[1:1], [0.0]); _setp!(h, hi[2:end], hs[1]))
        else
            for i in 1:ncomp
                hi = _ivec(comps[i].h_param_indices)
                isconst ? _setp!(h, hi, hs[i]) : (_setp!(h, hi[1:1], [0.0]); _setp!(h, hi[2:end], hs[i]))
            end
        end
    else
        _setp!(h, _ivec(pm.h_param_indices), hs[1])
    end
    if growth_flag
        for j in eachindex(ids)
            isnan(gn[j]) && continue
            gg = -(gn[j] - 1)
            cc = 1.0 - gg * gamma
            ll = 0.0
            if _has(pm, "optim_additive")
                oa = pm.optim_additive
                tmp0 = 0.0
                oparams = _vec(oa.params); osf = _vec(oa.scaling_factor)
                for i in eachindex(oparams)
                    b = _bgp_val(oa.bgp, i)
                    if isnan(oparams[i]) && b === true
                        tmp0 += osf[i]
                    elseif !isnan(oparams[i]) && b === true
                        tmp0 += P[Int(oparams[i])] * osf[i]
                    elseif !(b isa Bool)
                        merr("It is not possible to provide a value for the mean of an exogenous variable appearing in the optimal part of the PAC equation.")
                    end
                end
                cc -= tmp0 * gamma
            end
            if gamma < 1 && _has(pm, "non_optimizing_behaviour") && _has(pm.non_optimizing_behaviour, "params")
                nb = pm.non_optimizing_behaviour
                tmp0 = 0.0; tmp1 = 0.0
                np_ = _vec(nb.params); nsf = _vec(nb.scaling_factor)
                for i in eachindex(np_)
                    b = _bgp_val(nb.bgp, i)
                    if isnan(np_[i]) && b === true
                        tmp0 += nsf[i]
                    elseif !isnan(np_[i]) && b === true
                        tmp0 += P[Int(np_[i])] * nsf[i]
                    elseif !(b isa Bool) && b isa Real && isnan(np_[i])
                        tmp1 += nsf[i] * b
                    elseif !(b isa Bool) && b isa Real && !isnan(np_[i])
                        tmp1 += nsf[i] * np_[i] * b
                    end
                end
                cc -= (1.0 - gamma) * tmp0
                ll = -(1.0 - gamma) * tmp1 / gamma
            end
            if _has(pm, "additive")
                ad = pm.additive
                tmp0 = 0.0; tmp1 = 0.0
                ap = _vec(ad.params); asf = _vec(ad.scaling_factor)
                for i in eachindex(ap)
                    b = _bgp_val(ad.bgp, i)
                    if isnan(ap[i]) && b === true
                        tmp0 += asf[i]
                    elseif !isnan(ap[i]) && b === true
                        tmp0 += P[Int(ap[i])] * asf[i]
                    elseif !(b isa Bool) && b isa Real && isnan(ap[i])
                        tmp1 += asf[i] * b
                    elseif !(b isa Bool) && b isa Real && !isnan(ap[i])
                        tmp1 += asf[i] * ap[i] * b
                    end
                end
                cc -= tmp0
                ll -= tmp1 / gamma
            end
            if _has(pm, "growth_neutrality_param_index")
                _setp!(h, [Int(pm.growth_neutrality_param_index)], [cc / gamma])
            else
                gi = _f(comps[j], "growth_neutrality_param_index")
                isempty(gi) || _setp!(h, _ivec(gi), [cc / gamma])
            end
        end
    end
    nothing
end

function pac_mce_parameters!(h::Host, pacname::AbstractString)
    m = h.model; S = semi_M(m)
    (_has(S, "pac") && _has(S.pac, pacname)) || merr("PAC model $pacname is not defined in the model block!")
    pm = S.pac[pacname]
    _f(pm, "model_consistent_expectations") === true || merr("This function is to be used only for PAC Model Consistent Expectations.")
    println("PAC model ", pacname, " is used in equation ", _f(pm, "eq_name"), ".")
    println()
    P = _semi_params(h)
    maxlag = Int(pm.max_lag)
    params = vcat(P[_ivec(pm.ec.params)], P[_ivec(pm.ar.params)], P[Int(pm.discount_index)])
    G, alpha, beta = _build_G(params)
    _setp!(h, _ivec(pm.mce.alpha), reverse(alpha))
    comps = _has(pm, "components") ? pm.components : MStruct[]
    has_gn = _has(pm, "growth_neutrality_param_index") ||
             (!isempty(comps) && any(c -> _has(c, "growth_neutrality_param_index") && !isempty(c.growth_neutrality_param_index), comps))
    if has_gn
        gamma = _has(pm, "non_optimizing_behaviour") ? P[Int(pm.share_of_optimizing_agents_index)] : 1.0
        A = vcat(alpha, 1.0)
        A1 = _polyval(A, 1.0)
        mm = length(alpha)
        Imm = Matrix{Float64}(LinearAlgebra.I, mm, mm)
        em = _iota(mm, mm)
        d = A1 * _polyval(A, beta) * (em' * inv((Imm - G) * (Imm - G)) * em)
        cc = 1 / gamma - (sum(params[2:end-1]) + d)
        if _has(pm, "optim_additive")
            oa = pm.optim_additive
            tmp0 = 0.0
            oparams = _vec(oa.params); osf = _vec(oa.scaling_factor)
            for i in eachindex(oparams)
                b = _bgp_val(oa.bgp, i)
                if isnan(oparams[i]) && b === true
                    tmp0 += osf[i]
                elseif !isnan(oparams[i]) && b === true
                    # Dynare 7.2 refers to an undefined variable (equations.(eqtag)) here
                    merr("Undefined variable 'equations' (Dynare 7.2 pac.mce.parameters defect)")
                elseif !(b isa Bool)
                    merr("It is not possible to provide a value for the mean of an exogenous variable appearing in the optimal part of the PAC equation.")
                end
            end
            cc -= tmp0
        end
        if gamma < 1 && _has(pm, "non_optimizing_behaviour") && _has(pm.non_optimizing_behaviour, "params")
            nb = pm.non_optimizing_behaviour
            tmp0 = 0.0
            np_ = _vec(nb.params); nsf = _vec(nb.scaling_factor)
            for i in eachindex(np_)
                b = _bgp_val(nb.bgp, i)
                if isnan(np_[i]) && b === true
                    tmp0 += nsf[i]
                elseif !isnan(np_[i]) && b === true
                    tmp0 += P[Int(np_[i])] * nsf[i]
                end
            end
            cc -= (1 - gamma) * tmp0 / gamma
        end
        if _has(pm, "additive")
            ad = pm.additive
            tmp0 = 0.0
            ap = _vec(ad.params); asf = _vec(ad.scaling_factor)
            for i in eachindex(ap)
                b = _bgp_val(ad.bgp, i)
                if isnan(ap[i]) && b === true
                    tmp0 += asf[i]
                elseif !isnan(ap[i]) && b === true
                    tmp0 += P[Int(ap[i])] * asf[i]
                end
            end
            cc -= tmp0 / gamma
        end
        if _has(pm, "growth_neutrality_param_index")
            _setp!(h, [Int(pm.growth_neutrality_param_index)], [cc])
        else
            for c in comps
                (_has(c, "growth_neutrality_param_index") && !isempty(c.growth_neutrality_param_index)) &&
                    _setp!(h, _ivec(c.growth_neutrality_param_index), [cc])
            end
        end
    end
    nothing
end

function pac_bgp_set!(h::Host, pacmodl::AbstractString, paceq, variable::AbstractString, nonzeromean)
    m = h.model; S = semi_M(m)
    (_has(S, "pac") && _has(S.pac, pacmodl)) || merr("Reference to non-existent field '$pacmodl'.")
    pm = S.pac[pacmodl]
    ide = get(m.ie, variable, 0); xflag = false
    if ide == 0
        ide = get(m.ix, variable, 0); xflag = true
    end
    ide == 0 && merr("Endogenous/Exogenous variable $variable is unknown.")
    for type in ("additive", "optim_additive", "non_optimizing_behaviour")
        _has(pm, type) || continue
        t = pm[type]
        if !_has(t, "bgp")
            k = length(_vec(t.params))
            t["bgp"] = Cell(Any[false for _ in 1:k], (1, k))
        end
        ie = findfirst(==(ide), _ivec(t.vars))
        ie === nothing && continue
        isend = _bvec(t.isendo)[ie]
        xflag ? (!isend || merr("Variable type issue.")) : (isend || merr("Variable type issue."))
        t.bgp.items[ie] = nonzeromean
        return zeros(0, 0)
    end
    @warn "$variable is not an exogenous variable in equation $paceq."
    zeros(0, 0)
end

function pac_bgp_get(h::Host, pacmodel::AbstractString, paceq, kind::AbstractString, id)
    S = semi_M(h.model)
    (_has(S, "pac") && _has(S.pac, pacmodel)) || merr("Reference to non-existent field '$pacmodel'.")
    t = _f(S.pac[pacmodel], kind)
    _f(t, "bgp").items[Int(id)]
end

# ---------------------------------------------------------------------- 3-D arrays (AR(:,:,i))
"Indexing of a 3-D array `A(i, j, k)` (also with two or one subscripts, as MATLAB's collapsed trailing dimensions)."
function index3(A::Array{Float64,3}, args)
    n1, n2, n3 = size(A)
    ix(a, n) = a === COLON ? collect(1:n) : (v = _lin_index(a, n); v)
    if length(args) == 3
        r = A[ix(args[1], n1), ix(args[2], n2), ix(args[3], n3)]
        return size(r, 3) == 1 ? ret(reshape(r, size(r, 1), size(r, 2))) : r
    elseif length(args) == 2
        B = reshape(A, n1, n2 * n3)
        return ret(B[ix(args[1], n1), ix(args[2], n2 * n3)])
    elseif length(args) == 1
        return ret(reshape(A[ix(args[1], n1 * n2 * n3)], :, 1))
    end
    merr("index3: wrong number of subscripts")
end

# ---------------------------------------------------------------------- registration with the interpreter
const _STR = (v, nm) -> (v isa AbstractString ? String(v) : merr("$nm: argument must be a character array"))

_arg(a, i, d=nothing) = length(a) >= i ? a[i] : d

function _register_semi!()
    bi!("get_companion_matrix") do I, a, nout
        _nargs(a, 1, 2, "get_companion_matrix")
        r = semi_companion!(_need_host(I), _STR(a[1], "get_companion_matrix"), length(a) > 1 ? _STR(a[2], "get_companion_matrix") : nothing)
        nout <= 1 ? Any[] : r[1:min(nout, 4)]
    end
    bi!("get_aux_variable_id") do I, a, nout
        h = _need_host(I); m = h.model
        _nargs(a, 1, 1, "get_aux_variable_id")
        v = a[1]
        v isa AbstractString && isempty(v) && return 0.0
        if v isa AbstractString
            id = get(m.ie, v, 0)
            id == 0 && return 0.0
        else
            id = Int(scalar(v))
            (id < 1 || id > m.n_endo) && merr("Input must be the name of an endogenous variable or an integer between 1 and $(m.n_endo)!")
        end
        id <= orig_endo_nbr(m) && return 0.0
        for (k, au) in enumerate(get_or(m.raw, "aux_vars", []))
            Int(au["endo_index"]) == id && return Float64(k)
        end
        0.0
    end
    bi!("get_difference_order") do I, a, nout
        _nargs(a, 1, 1, "get_difference_order")
        Float64(_diff_order(_need_host(I).model, a[1] isa AbstractString ? a[1] : Int(scalar(a[1]))))
    end
    bi!("isauxiliary") do I, a, nout
        _nargs(a, 1, 2, "isauxiliary")
        _isaux(_need_host(I).model, _STR(a[1], "isauxiliary"), length(a) > 1 ? Int.(vec(farr(a[2]))) : nothing)
    end
    reg(f, name) = (PKG_FUNCS[name] = f)
    reg("*.varmatrices") do I, a, nout
        _nargs(a, 2, 3, "varmatrices")
        ar, a0, c = _varmatrices(_need_host(I).model, _STR(a[1], "varmatrices"), Float64.(vec(farr(a[2]))), length(a) > 2 && truth(a[3]))
        Any[ar, ret(a0), ret(reshape(c, :, 1))][1:max(nout, 1)]
    end
    reg("*.trend_component_ar_a0") do I, a, nout
        _nargs(a, 2, 2, "trend_component_ar_a0")
        ar, a0, a0s = _tc_matrices(_need_host(I).model, _STR(a[1], "trend_component_ar_a0"), Float64.(vec(farr(a[2]))))
        Any[ar, ret(a0), ret(a0s)][1:max(nout, 1)]
    end
    reg("*.pac_target_coefficients") do I, a, nout
        _nargs(a, 2, 2, "pac_target_coefficients")
        Any[_pac_target_coefficients(_need_host(I).model, _STR(a[1], "pac_target_coefficients"), Float64.(vec(farr(a[2]))))]
    end
    for nm in ("get_companion_matrix_legacy", "get_ar_ec_matrices")
        bi!(nm) do I, a, nout
            throw(Unsupported("$nm (legacy VAR / trend component matrices from the model's Jacobian) is not implemented in JDyn", "function"))
        end
    end
    reg("var_expectation.initialize") do I, a, nout
        var_expectation_initialize!(_need_host(I), _STR(a[1], "var_expectation.initialize")); Any[]
    end
    reg("var_expectation.update") do I, a, nout
        var_expectation_update!(_need_host(I), _STR(a[1], "var_expectation.update")); Any[]
    end
    reg("pac.initialize") do I, a, nout
        pac_initialize!(_need_host(I), _STR(a[1], "pac.initialize")); Any[]
    end
    reg("pac.update.expectation") do I, a, nout
        pac_update_parameters!(_need_host(I), _STR(a[1], "pac.update.expectation")); Any[]
    end
    reg("pac.update.equation") do I, a, nout
        pac_update_parameters!(_need_host(I), _STR(a[1], "pac.update.equation")); Any[]
    end
    reg("pac.mce.parameters") do I, a, nout
        pac_mce_parameters!(_need_host(I), _STR(a[1], "pac.mce.parameters")); Any[]
    end
    reg("pac.bgp.set") do I, a, nout
        _nargs(a, 4, 4, "pac.bgp.set")
        pac_bgp_set!(_need_host(I), _STR(a[1], "pac.bgp.set"), a[2], _STR(a[3], "pac.bgp.set"), a[4])
        nout > 0 ? Any[zeros(0, 0)] : Any[]
    end
    reg("pac.bgp.get") do I, a, nout
        _nargs(a, 4, 4, "pac.bgp.get")
        Any[pac_bgp_get(_need_host(I), _STR(a[1], "pac.bgp.get"), a[2], _STR(a[3], "pac.bgp.get"), scalar(a[4]))]
    end
    for nm in ("pac.print", "var_expectation.print")
        reg(nm) do I, a, nout
            Any[]            # prints the expanded expectation term (output only)
        end
    end
    for nm in ("pac.estimate.nls", "pac.estimate.iterative_ols", "pac.estimate.init", "var_expectation.update_parameters",
               "pac.update.parameters", "pac.check")
        reg(nm) do I, a, nout
            throw(Unsupported("$nm is not implemented in JDyn", "function"))
        end
    end
    nothing
end
_register_semi!()
