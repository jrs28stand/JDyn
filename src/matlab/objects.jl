# Glue between the interpreter and the dates/dseries objects: operators, field and method syntax, indexing,
# assignment, concatenation, function-call dispatch and constructors.

const _OPNAME = Dict("+" => "plus", "-" => "minus", "*" => "mtimes", "/" => "mrdivide", "^" => "mpower",
    "==" => "eq", "~=" => "ne", "<" => "lt", ">" => "gt", "<=" => "le", ">=" => "ge")

objclass(args) = (for a in args; a isa MObject && return mclass(a); end; nothing)

function obj_binop(I, op, l, r)
    cls = l isa MObject ? mclass(l) : mclass(r)
    name = get(_OPNAME, op, nothing)
    name === nothing && throw(MatlabError("binary operator '$op' not implemented for '$(mclass(l))' by '$(mclass(r))' operations"))
    f = get(OBJM[cls], name, nothing)
    f === nothing && throw(MatlabError("binary operator '$op' not implemented for '$(mclass(l))' by '$(mclass(r))' operations"))
    _one_value(f(I, Any[l, r], 1))
end
function obj_unop(I, op, v)
    name = op == "-" ? "uminus" : op == "+" ? "uplus" : nothing
    f = name === nothing ? nothing : get(OBJM[mclass(v)], name, nothing)
    f === nothing && throw(MatlabError("unary operator '$op' not implemented for '$(mclass(v))' operations"))
    _one_value(f(I, Any[v], 1))
end
_one_value(r) = r isa Vector{Any} ? (isempty(r) ? merr("value on the right hand side of the assignment not defined") : r[1]) : r

function obj_cat(vals, dim)
    objs = [v for v in vals if !(_is00(v))]
    isempty(objs) && return zeros(0, 0)
    cls = objclass(objs)
    all(v -> v isa MObject && mclass(v) == cls, objs) ||
        throw(MatlabError("concatenation operator not implemented for '$(mclass(objs[1]))' by '$(mclass(objs[end]))' operations"))
    try
        cls == "dates" ? dates_horzcat(objs) : (dim == 2 ? ds_horzcat(objs) : ds_vertcat(objs))
    catch e
        e isa MatlabError || rethrow()
        throw(MatlabError("$cls/$(dim == 2 ? "horzcat" : "vertcat") method failed"))
    end
end

# ---- fields and methods
_isprop(o::MDates, n) = n in ("time", "freq")
_isprop(o::MDseries, n) = n in DS_PROPS
_property(o::MDates, n) = dates_property(o, n)
_property(o::MDseries, n) = ds_property(o, n)

function _method(o::MObject, name)
    cls = mclass(o)
    if cls == "dates"
        name in OBJDOT["dates"] || merr("dates::subsref: Unknown public member or method!")
    end
    f = get(OBJM[cls], name, nothing)
    f === nothing && throw(Unsupported("$cls method '$name' is not implemented", "function"))
    f
end

"`o.name` without arguments."
function obj_field(I, o::MObject, name)
    _isprop(o, name) && return _property(o, name)
    o isa MDates && name in ("append", "append_", "pop", "pop_", "remove", "remove_") && merr("S(2): out of bound 1 (dimensions are 1x1)")
    if o isa MDseries
        (name in OBJDOT["dseries"] || haskey(OBJM["dseries"], name) && name in ("size", "disp", "length", "isempty")) &&
            return _one_value(_method(o, name)(I, Any[o], 1))
        name == "length" && merr("dseries::subsref: we do not support the length operator on dseries. Please use 'nobs' or 'vobs'")
        return ds_variable(o, name)
    end
    r = _method(o, name)(I, Any[o], 1)
    _one_value(r)
end

"`o.name(args)`: a property followed by an index, a method call or (dseries) a variable followed by an index."
function obj_method_call(I, o::MObject, name::AbstractString, argnodes, sc, nargout)
    if _isprop(o, name)
        r = _property(o, name)
        isempty(argnodes) && return Any[r]
        args = index_args(I, r, argnodes, sc)
        return Any[obj_or_mindex(I, r, args)]
    end
    if o isa MDseries && !(name in OBJDOT["dseries"]) && !(name in ("size", "disp"))
        r = ds_variable(o, name)
        args = index_args(I, r, argnodes, sc)
        return Any[obj_or_mindex(I, r, args)]
    end
    f = _method(o, name)
    args = Any[a isa ColonAll ? ":" : ev(I, a, sc) for a in argnodes]
    _aslist(f(I, Any[o; args], nargout))
end

function obj_or_mindex(I, v, args)
    v isa MObject && return obj_index(I, v, args)
    mindex(v, args)
end
obj_index(I, v::MDates, args) = dates_index(v, args)
obj_index(I, v::MDseries, args) = ds_index(I, v, args)
function obj_brace(I, v, args)
    v isa MDseries && return ds_brace(I, v, args)
    merr("'{' undefined for arguments of type '$(mclass(v))'")
end

# ---- function-call dispatch
function obj_dispatch(I, name, args, nargout)
    cls = objclass(args)
    cls === nothing && return nothing
    f = get(OBJM[cls], name, nothing)
    f === nothing && return nothing
    _aslist(f(I, args, nargout))
end

# ---- assignment
function _ds_assign_values!(o::MDseries, Dates, B, who)
    D = Dates isa MDates ? Dates : dates_ctor(Any[Dates])
    tdx = [Int(t - o.dates.time[1]) + 1 for t in sort!(unique([t for t in o.dates.time if t in D.time]))]
    if B isa MDseries
        tdy = [Int(t - B.dates.time[1]) + 1 for t in sort!(unique([t for t in B.dates.time if t in D.time]))]
        isempty(tdy) && merr("dseries::subsasgn: Periods of the dseries objects on the left and right hand sides must intersect!")
        vobs(o) == vobs(B) || merr("dseries::subsasgn: Dimension error! The number of variables on the left and right hand side must match.")
        d = copy(o.data); d[tdx, :] = B.data[tdy, :]; o.data = d
    elseif B isa Real || B isa AbstractMatrix{Float64}
        b = farr(B)
        length(tdx) == size(b, 1) || merr("dseries::subsasgn: Dimension error! The number of periods on the left and right hand side must match.")
        size(o.data, 2) == size(b, 2) || merr("dseries::subsasgn: Dimension error! The number of variables on the left and right hand side must match.")
        d = copy(o.data); d[tdx, :] = b; o.data = d
    else
        merr("dseries::subsasgn: The object on the right hand side must be a dseries object or a numeric array!")
    end
    o
end

function _rename_for(A::MDseries, B::MDseries, names::Vector{String})
    B = copy_ds(B)
    for i in 1:vobs(B)
        if names[i] != B.name[i]
            id = findfirst(==(names[i]), A.name)
            if id === nothing
                B.name[i] = names[i]; B.tex[i] = name2tex(names[i])
            else
                B.name[i] = A.name[id]; B.tex[i] = A.tex[id]
            end
        end
    end
    B
end

function obj_assign(I, cur::MObject, chain, i, value, sc)
    cur isa MDseries || throw(Unsupported("assignment into a $(mclass(cur)) object", "feature"))
    acc = chain[i]
    last = i == length(chain)
    A = cur
    if last
        if acc isa FieldE
            name = acc.name !== nothing ? acc.name : ev(I, acc.dyn, sc)
            if name == "init" && value isa MDates && length(value.time) == 1
                A.dates = _range(value, nobs(A)); return A
            elseif name == "dates" && value isa MDates
                A.dates = value; return A
            elseif name in ("data", "name", "tex", "ops")
                merr("dseries::subsasgn: You cannot overwrite $name member!")
            end
            value isa MDseries || merr("scalar cannot be indexed with .")
            B = _rename_for(A, value, [name])
            ds_isempty(A) && return B
            return ds_merge(A, B, true)
        elseif acc.brace
            args = index_args(I, A, acc.args, sc)
            all(x -> x isa AbstractString, args) || merr("dseries::subsasgn: Wrong syntax!")
            any(x -> occursin('@', x) || occursin('[', x), args) && _unsup("implicit loops or regular expressions in ds{...} = ...")
            length(unique(args)) == length(args) || merr("dseries::subsasgn: Wrong syntax!")
            if value isa AbstractMatrix && isempty(value)
                for nm in args
                    _remove!(I, A, nm)
                end
                return A
            end
            value isa MDseries || merr("dseries::subsasgn: Can only assign dseries object.")
            length(args) == vobs(value) || merr("dseries::subsasgn: Wrong syntax!")
            B = _rename_for(A, value, String[args...])
            ds_isempty(A) && return B
            return ds_merge(A, B, true)
        else
            args = index_args(I, A, acc.args, sc)
            x = args[1]
            if x isa MDates || isstringdate(x)
                return _ds_assign_values!(A, x, value, "()")
            elseif x isa AbstractString && x == ":" && ds_isempty(A)
                value isa MDseries && (A.data = value.data; A.name = copy(value.name); A.tex = copy(value.tex); A.dates = value.dates; A.ops = copy(value.ops); return A)
                A.data = copy(farr(value)); A.name = default_names(vobs(A)); A.tex = [name2tex(s) for s in A.name]
                A.ops = fill("", vobs(A))
                init = isempty(A.dates.time) ? MDates(A.dates.freq == 0 ? 1 : A.dates.freq, [A.dates.freq == 0 ? 1.0 : A.dates.freq + 1.0]) : A.dates[1.0]
                A.dates = _range(init, nobs(A)); return A
            end
            merr("dseries::subsasgn: Wrong syntax!")
        end
    end
    i + 1 == length(chain) && chain[i+1] isa IndexE && !chain[i+1].brace || _unsup("nested assignment into a dseries")
    sA = if acc isa FieldE
        name = acc.name !== nothing ? acc.name : ev(I, acc.dyn, sc)
        ds_extract(A, String[name])
    elseif acc.brace
        ds_extract(A, String[x for x in index_args(I, A, acc.args, sc)])
    else
        _unsup("nested assignment into a dseries")
    end
    dargs = index_args(I, sA, chain[i+1].args, sc)
    D = dargs[1] isa MDates ? dargs[1] : dates_ctor(Any[dargs[1]])
    tdx = [Int(t - sA.dates.time[1]) + 1 for t in sort!(unique([t for t in sA.dates.time if t in D.time]))]
    d = copy(sA.data)
    if value isa MDseries && vobs(value) == vobs(sA)
        tdy = [Int(t - value.dates.time[1]) + 1 for t in sort!(unique([t for t in value.dates.time if t in D.time]))]
        isempty(tdy) && merr("dseries::subsasgn: Periods of the dseries objects on the left and right hand sides must intersect!")
        d[tdx, :] = value.data[tdy, :]
    elseif (value isa Real || value isa AbstractMatrix{Float64}) && (size(farr(value), 2) == vobs(sA) || size(farr(value), 2) == 1)
        b = farr(value)
        if length(tdx) == size(b, 1)
            d[tdx, :] = size(b, 2) == vobs(sA) ? b : repeat(b, 1, vobs(sA))
        elseif size(b, 1) == 1
            d[tdx, :] = repeat(b, length(tdx), size(b, 2) == vobs(sA) ? 1 : vobs(sA))
        else
            merr("dseries::subsasgn: Dimension error! The number of periods on the left and right hand side must match.")
        end
    else
        merr("dseries::subsasgn: Dimension error! The number of variables on the left and right hand side must match.")
    end
    sA.data = d
    ds_merge(A, sA, true)
end

# ---- constructors and predicates
bi!("dates") do I, a, nout; dates_ctor(a); end
bi!("dseries") do I, a, nout; ds_ctor(I, a); end
bi!("isdates") do I, a, nout; a[1] isa MDates; end
bi!("isdseries") do I, a, nout; a[1] isa MDseries; end
bi!("isdate") do I, a, nout; isdate_(a[1]); end
bi!("isstringdate") do I, a, nout; isstringdate(a[1]); end
bi!("isfreq") do I, a, nout; isfreq_(a[1]); end
bi!("isobject") do I, a, nout; a[1] isa MObject; end
bi!("date2string") do I, a, nout
    length(a) == 1 && a[1] isa MDates && return date2string(a[1])
    length(a) == 2 && return date2string(scalar(a[1]), Int(scalar(a[2])))
    merr("date2string: invalid call")
end
bi!("freq2string") do I, a, nout; freq2string(Int(scalar(a[1]))); end
bi!("string2freq") do I, a, nout; Float64(string2freq(a[1])); end
bi!("isleapyear") do I, a, nout; ret(isleapyear.(farr(a[1]))); end
bi!("islongyear") do I, a, nout; ret(islongyear.(farr(a[1]))); end
bi!("default_name") do I, a, nout; _namecell(default_names(Int(scalar(a[1])))); end
bi!("name2tex") do I, a, nout
    a[1] isa Cell ? Cell(Any[name2tex(x) for x in a[1].items], a[1].shape) : name2tex(a[1])
end
