# External functions (`external_function` statement) in the model block: the Julia code of the preprocessor (language=julia) contains MATLAB-style lines
#     TEF_0 = f(args);                       [TEF_0, TEFD_0, TEFDD_0] = f(args);        TEFD_def_1 = g(args);
#     TEFD_fdd_0_2 = jacob_element('f',2,{a,b});     TEFDD_fdd_0_2_2 = hess_element('f',2,2,{a,b});
# which are rewritten before the file is parsed (`_rewrite_external`) into calls of the table below. Functions come from `Model(functions=...)` (Julia
# callables taking the arguments as Float64; a function that returns its own derivatives returns a tuple `(value, gradient[, hessian])`) and from
# MATLAB files next to the .mod file (name.m, or +pkg/name.m for pkg.name), run by the MATLAB interpreter. Derivatives that are not provided are
# Dynare's jacob_element.m and hess_element.m (centred and five/seven-point differences, step 1e-6), reproduced operation by operation.
# The preprocessor cannot differentiate external functions three times: models with external functions are limited to order 2, as in Dynare.

mutable struct ExternalFunctions
    functions::Dict{String,Any}
    mod_dir::String
    caller::Any
end
ExternalFunctions(functions, mod_dir) = ExternalFunctions(Dict{String,Any}(String(k) => v for (k, v) in functions), String(mod_dir), nothing)

const _EXT_H = 1e-6

_ext_missing(name) = ModelError("external function $(repr(name)) is not available: pass a Julia callable with Model(...; functions=Dict($(repr(name)) => f)) " *
                                "or put $(last(split(name, "."))).m next to the .mod file")

function _ext_raw(t::ExternalFunctions, name::AbstractString, nout::Int, args)
    f = get(t.functions, name, nothing)
    if f !== nothing
        out = f(args...)
        if nout == 1
            return Any[out isa Tuple ? out[1] : out]
        end
        (out isa Tuple && length(out) >= nout) ||
            throw(ModelError("external function $(repr(name)) must return $nout outputs (value, gradient$(nout == 3 ? ", hessian" : "")) as the .mod file declares"))
        return Any[out[i] for i in 1:nout]
    end
    if t.caller === nothing
        isdir(t.mod_dir) || throw(_ext_missing(name))
        t.caller = matlab_function_caller(t.mod_dir)
    end
    parts = split(name, ".")
    isfile(joinpath(t.mod_dir, [("+" * p) for p in parts[1:end-1]]..., parts[end] * ".m")) || throw(_ext_missing(name))
    try
        out = t.caller(name, Any[args...], nout)
        length(out) >= nout || throw(ModelError("external function $(repr(name)) returned $(length(out)) outputs, $nout expected"))
        return Any[out[i] for i in 1:nout]
    catch e
        e isa MatlabError && throw(ModelError("external function $name: MATLAB error: $(e.msg)"))
        e isa Unsupported && throw(ModelError("external function $name cannot be interpreted ($(e.msg)); pass a Julia callable with Model(...; functions=...)"))
        rethrow()
    end
end

function _ext_float(v, name)
    a = v isa Real ? Float64(v) : farr(v)
    a isa Float64 && return a
    length(a) == 1 || throw(ModelError("external function $(repr(name)) returned $(length(a)) values instead of one"))
    Float64(a[1])
end

"`TEF_k = f(args)`: the value."
ext_value(t::ExternalFunctions, name, args...) = _ext_float(_ext_raw(t, name, 1, args)[1], name)

"`[TEF, TEFD, TEFDD] = f(args)`: the value and the gradient / Hessian as arrays."
function ext_multi(t::ExternalFunctions, name, nout, args...)
    out = _ext_raw(t, name, nout, args)
    ntuple(i -> i == 1 ? _ext_float(out[1], name) : _ext_array(out[i]), nout)
end
_ext_array(v) = v isa Real ? fill(Float64(v), 1, 1) : (a = farr(v); a isa AbstractArray ? Array{Float64}(a) : fill(Float64(a), 1, 1))

"`TEFD_def_k = g(args)`: the whole gradient (or Hessian) of a separate derivative function."
ext_array(t::ExternalFunctions, name, args...) = _ext_array(_ext_raw(t, name, 1, args)[1])

"`jacob_element.m`: centred difference with step 1e-6 in argument `element`."
function ext_jacob(t::ExternalFunctions, name, element, args...)
    h = _EXT_H
    p = Any[args...]; m = Any[args...]
    element <= length(args) || throw(ModelError("jacob_element: element $element is beyond the $(length(args)) arguments"))
    p[element] = p[element] + h
    m[element] = m[element] - h
    (ext_value(t, name, p...) - ext_value(t, name, m...)) / (2 * h)
end

"`hess_element.m`: five-point (diagonal) and seven-point (off-diagonal) second differences, step 1e-6."
function ext_hess(t::ExternalFunctions, name, e1, e2, args...)
    h = _EXT_H
    n = length(args)
    (e1 <= n && e2 <= n) || throw(ModelError("hess_element: elements ($e1, $e2) beyond the $n arguments"))
    a = Any[args...]
    p10 = copy(a); p01 = copy(a); m10 = copy(a); m01 = copy(a); p11 = copy(a); m11 = copy(a)
    p10[e1] += h; m10[e1] -= h
    p11[e1] += h; m11[e1] -= h
    p01[e2] += h; m01[e2] -= h
    p11[e2] += h; m11[e2] -= h
    f(x) = ext_value(t, name, x...)
    if e1 == e2
        return (16 * f(p10) + 16 * f(m10) - 30 * f(a) - f(p11) - f(m11)) / (12 * h^2)
    end
    (f(p10) + f(m10) + f(p01) + f(m01) - 2 * f(a) - f(p11) - f(m11)) / (-2 * h^2)
end

const _EXT_LINE_MULTI = r"^([ \t]*)\[([^\]\n]+)\]\s*=\s*([A-Za-z_][\w.]*)\((.*)\);[ \t]*$"m
const _EXT_LINE_VALUE = r"^([ \t]*)(TEF_\w+)\s*=\s*([A-Za-z_][\w.]*)\((.*)\);[ \t]*$"m
const _EXT_LINE_DEF = r"^([ \t]*)(TEFD_def_\w+|TEFDD_def_\w+)\s*=\s*([A-Za-z_][\w.]*)\((.*)\);[ \t]*$"m
const _EXT_JACOB = r"jacob_element\('([\w.]+)',\s*(\d+),\s*\{(.*)\}\)"
const _EXT_HESS = r"hess_element\('([\w.]+)',\s*(\d+),\s*(\d+),\s*\{(.*)\}\)"

"Does the generated code call external functions?"
_has_external(text::AbstractString) = occursin(r"^\s*(\[[^\]\n]*TEF\w*[^\]\n]*\]|TEF\w*)\s*=\s*[A-Za-z_]"m, text)

"The names of the external functions called (and of their derivative functions) in generated code."
function external_function_names(text::AbstractString)
    out = String[]
    for rx in (_EXT_LINE_MULTI, _EXT_LINE_VALUE, _EXT_LINE_DEF)
        for mt in eachmatch(rx, text)
            (rx === _EXT_LINE_MULTI && !occursin("TEF", mt.captures[2])) && continue
            push!(out, String(mt.captures[3]))
        end
    end
    for mt in eachmatch(_EXT_JACOB, text); push!(out, String(mt.captures[1])); end
    for mt in eachmatch(_EXT_HESS, text); push!(out, String(mt.captures[1])); end
    unique(out)
end

"Rewrite the MATLAB-style external function lines of generated Julia code into calls of `_ext_*` (defined in the model module)."
function _rewrite_external(text::AbstractString)
    text = replace(text, _EXT_LINE_MULTI => function (s)
        mt = match(_EXT_LINE_MULTI, s)
        occursin("TEF", mt.captures[2]) || return s
        n = count(==(','), mt.captures[2]) + 1
        "$(mt.captures[1])$(mt.captures[2]) = _ext_multi(\"$(mt.captures[3])\", $n$(isempty(strip(mt.captures[4])) ? "" : ", " * mt.captures[4]));"
    end)
    text = replace(text, _EXT_LINE_DEF => function (s)
        mt = match(_EXT_LINE_DEF, s)
        "$(mt.captures[1])$(mt.captures[2]) = _ext_array(\"$(mt.captures[3])\"$(isempty(strip(mt.captures[4])) ? "" : ", " * mt.captures[4]));"
    end)
    text = replace(text, _EXT_LINE_VALUE => function (s)
        mt = match(_EXT_LINE_VALUE, s)
        "$(mt.captures[1])$(mt.captures[2]) = _ext_value(\"$(mt.captures[3])\"$(isempty(strip(mt.captures[4])) ? "" : ", " * mt.captures[4]));"
    end)
    text = replace(text, _EXT_JACOB => s -> (mt = match(_EXT_JACOB, s); "_ext_jacob(\"$(mt.captures[1])\", $(mt.captures[2]), $(mt.captures[3]))"))
    replace(text, _EXT_HESS => s -> (mt = match(_EXT_HESS, s); "_ext_hess(\"$(mt.captures[1])\", $(mt.captures[2]), $(mt.captures[3]), $(mt.captures[4]))"))
end

"Define the `_ext_*` entry points of an `ExternalFunctions` table in the model module."
function _install_external!(mod::Module, t::ExternalFunctions)
    Core.eval(mod, :(const _EXT = $t))
    for (nm, f) in (("_ext_value", ext_value), ("_ext_multi", ext_multi), ("_ext_array", ext_array), ("_ext_jacob", ext_jacob), ("_ext_hess", ext_hess))
        Core.eval(mod, :($(Symbol(nm))(args...) = $f(_EXT, args...)))
    end
    nothing
end
