# A minimal entry point for calling a MATLAB function file (name.m, or +pkg/name.m for the dotted name pkg.name) of a directory from Julia
# with numeric arguments (used for the external functions of a model block, src/external.jl).

"""
    matlab_function_caller(dir) -> f

`f(name, args, nargout) -> Vector{Any}` runs `dir/name.m` (or `dir/+a/+b/c.m` for `"a.b.c"`) in the MATLAB interpreter on the arguments `args` and
returns its first `nargout` outputs (MATLAB values: numbers or arrays, see `farr`). The interpreter has no model attached.
"""
function matlab_function_caller(dir::AbstractString)
    I = Interp(dir, nothing)
    function (name::AbstractString, args::AbstractVector, nargout::Integer)
        I.steps = 0
        sc = I.scope_stack[end]
        parts = String.(split(name, "."))
        if length(parts) > 1
            u = pkg_unit(I, sc, parts)
            u === nothing && merr("the package function $name was not found under $dir")
            is_script(u) && merr("$name is a script, not a function")
            return call_user(I, u.main, u.funcs, Any[args...], nargout)
        end
        haskey(dir_files(I), name) || merr("$name.m was not found in $dir")
        u = unit(I, name)
        u isa ParseError && throw(Unsupported("$name.m cannot be parsed: $(u.msg)", "syntax"))
        is_script(u) && merr("$name.m is a script, not a function")
        call_user(I, u.main, u.funcs, Any[args...], nargout)
    end
end
