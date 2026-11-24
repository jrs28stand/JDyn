# Runs the unit tests embedded in Dynare 7.2's dseries toolbox (the `%@test:N ... %@eof:N` blocks of
# matlab/dseries/src/@dates, @dseries and utilities) through the interpreter.
#   julia --project=. test/dseries_unit_tests.jl [pattern]      (prints failures and a summary)
using JDyn
const SRC = "/root/ref/dynare-7.2/matlab/dseries/src"

function _isequaln(a, b)
    if a isa JDyn.Cell || b isa JDyn.Cell
        return a isa JDyn.Cell && b isa JDyn.Cell && a.shape == b.shape && all(_isequaln(x, y) for (x, y) in zip(a.items, b.items))
    elseif a isa AbstractString || b isa AbstractString
        return a isa AbstractString && b isa AbstractString && a == b
    elseif a isa JDyn.MDates && b isa JDyn.MDates
        return a.freq == b.freq && a.time == b.time
    elseif a isa JDyn.MDseries && b isa JDyn.MDseries
        return _isequaln(a.data, b.data) && a.name == b.name && a.dates.time == b.dates.time
    elseif (a isa Real || a isa AbstractMatrix{<:Real}) && (b isa Real || b isa AbstractMatrix{<:Real})
        x = a isa Real ? fill(Float64(a), 1, 1) : Float64.(a); y = b isa Real ? fill(Float64(b), 1, 1) : Float64.(b)
        return size(x) == size(y) && all(((p, q),) -> p == q || (isnan(p) && isnan(q)), zip(x, y))
    end
    false
end
JDyn.BUILTINS["dassert"] = (I, a, nout) -> begin
    JDyn.mclass(a[1]) == JDyn.mclass(a[2]) || JDyn.merr("dassert:: Compared objects are not of the same type!")
    if length(a) >= 3
        d = abs.(Float64.(vec(JDyn.farr(a[1]))) .- Float64.(vec(JDyn.farr(a[2])))); return !any(x -> x > JDyn.scalar(a[3]), d)
    end
    _isequaln(a[1], a[2])
end
JDyn.BUILTINS["isoctave"] = (I, a, nout) -> false

function extract_tests(path)
    out = Tuple{String,String}[]
    lines = readlines(path)
    i = 1
    while i <= length(lines)
        m = match(r"^%@test:(\d+)", lines[i])
        if m !== nothing
            body = String[]
            i += 1
            while i <= length(lines) && !startswith(lines[i], "%@eof")
                l = lines[i]
                if startswith(l, "%\$")
                    push!(body, replace(l, r"^%\$ ?" => ""))
                elseif !startswith(strip(l), "%")
                    push!(body, l)
                end
                i += 1
            end
            push!(out, ("$(relpath(path, SRC)):$(m.captures[1])", join(body, "\n")))
        end
        i += 1
    end
    out
end

function main(pat)
    files = String[]
    for (root, _, fs) in walkdir(SRC), f in fs
        endswith(f, ".m") && (occursin("@dates", root) || occursin("@dseries", root) || occursin("utilities", root)) && push!(files, joinpath(root, f))
    end
    sort!(files)
    npass = nfail = nerr = 0
    errs = Dict{String,Int}()
    for f in files, (id, code) in extract_tests(f)
        occursin(pat, id) || continue
        try
            v = JDyn.mrun("T = false;\n" * code; variables=Dict{String,Any}("dseries_src_root" => SRC * "/"))
            t = get(v, "T", false)
            if t === true || t == 1.0 || (t isa AbstractMatrix && all(x -> x != 0, t))
                npass += 1
            else
                nfail += 1; println("FAIL  ", id)
            end
        catch e
            msg = e isa JDyn.MatlabError ? e.msg : e isa JDyn.Unsupported ? "Unsupported: " * e.msg : sprint(showerror, e)
            nerr += 1
            k = first(msg, 90)
            errs[k] = get(errs, k, 0) + 1
            println("ERROR ", id, " : ", first(msg, 150))
        end
    end
    println("\npass $npass, fail $nfail, error $nerr")
    for (k, n) in sort(collect(errs); by=x -> -x[2])[1:min(15, end)]
        println(n, "  ", k)
    end
end
JDyn.ALLOW_RANDOM[] = true
main(isempty(ARGS) ? "" : ARGS[1])
