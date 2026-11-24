# dates / dseries objects of the MATLAB interpreter against Octave 8.4 running Dynare 7.2's dseries toolbox
# (test/dseries_cases.txt, reference test/dseries_ref.mat from test/r_dseries.m).
using Test, MAT
import JDyn

function _same(a, b)
    if b isa AbstractString
        return a isa AbstractString && String(a) == String(b)
    elseif b isa AbstractArray{Any}
        a isa JDyn.Cell || return false
        length(a.items) == length(b) || return false
        return all(_same(x, y) for (x, y) in zip(a.items, b))
    elseif b isa Number || b isa AbstractArray{<:Real}
        (a isa Real || a isa AbstractMatrix{<:Real}) || return false
        x = Float64.(a isa Real ? fill(a, 1, 1) : a); y = Float64.(b isa Number ? fill(b, 1, 1) : b)
        size(x) == size(y) || (length(x) == length(y) == 0) || (length(x) == length(y) && min(size(x)...) <= 1 && min(size(y)...) <= 1 && false) || return false
        return all(((p, q),) -> (isnan(p) && isnan(q)) || isapprox(p, q; rtol=1e-9, atol=1e-11), zip(x, y))
    end
    false
end

function dseries_mismatches()
    cases = split(read(joinpath(@__DIR__, "dseries_cases.txt"), String), "\n%%\n")
    ref = vec(matread(joinpath(@__DIR__, "dseries_ref.mat"))["results"])
    bad = Tuple{Int,String,Any,Any}[]
    for (k, c) in enumerate(cases)
        exp = ref[k]
        got = try
            v = JDyn.mrun("r = [];\n" * c; variables=Dict{String,Any}())
            v["r"]
        catch e
            e isa JDyn.MatlabError ? "ERROR: " * e.msg : "JULIA: " * sprint(showerror, e)
        end
        if exp isa AbstractString && startswith(exp, "ERROR: ")
            ok = got isa AbstractString && startswith(got, "ERROR: ") && (got == exp)
        else
            ok = _same(got, exp)
        end
        ok || push!(bad, (k, strip(c), got, exp))
    end
    bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    bad = dseries_mismatches()
    for (k, c, g, e) in bad
        println("CASE $k: $c\n  got: $(repr(g)[1:min(end,300)])\n  exp: $(repr(e)[1:min(end,300)])")
    end
    println(length(bad), " mismatches")
else
    @testset "dates and dseries against Octave" begin
        bad = dseries_mismatches()
        for (k, c, g, e) in bad
            @info "dseries case $k: $c" got = g expected = e
        end
        @test isempty(bad)
    end
end
