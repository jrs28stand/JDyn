# dseries field/brace/paren assignment with Dynare's merge rules against Octave 8.4 running Dynare 7.2's dseries class
# (test/ref_parity2/dseries/dseries_assign.mat from r_dseries_assign.m; the statements are pydyn's tests/test_dseries_unified.py::SCRIPT).
using Test, MAT
import JDyn

const SCRIPT = """
a = dseries([1 2 3; 4 5 6; 7 8 9], dates('2000Q1'), {'b','a','c'});
x = dseries([10;11;12], dates('2000Q2'), 'z');
a.z = x;
d1 = a.data; n1 = a.name;
a{'b'} = dseries([0;0;0;0],dates('2000Q1'),'q');
d2 = a.data;
a.init = dates('2001Q1'); i3 = a.init.char;
l = a.lag(1); ld = l.data;
a{'a'}(dates('2001Q2'):dates('2001Q3')) = [100;200]; d4 = a.data;
b = dseries([1 2; 3 4; 5 6], dates('2000Q1'), {'u','v'});
b.u(dates('2000Q2')) = 50; d5 = b.data;
b{'v'} = b{'u'}.lag(1); d6 = b.data; n6 = b.name;
"""

@testset "dseries assignment against Dynare's class" begin
    r = matread(joinpath(@__DIR__, "ref_parity2", "dseries", "dseries_assign.mat"))
    w = JDyn.mrun(SCRIPT)
    for k in ("d1", "d2", "ld", "d4", "d5", "d6")
        a = Float64.(r[k]); b = Float64.(w[k])
        @test size(a) == size(b)
        @test all(isnan.(a) .== isnan.(b)) && all(a[.!isnan.(a)] .== b[.!isnan.(b)])
    end
    @test String(w["i3"]) == String(r["i3"])
    names(c) = String[String(x) for x in (c isa JDyn.Cell ? c.items : vec(c))]
    @test names(w["n6"]) == names(r["n6"])
    @test names(w["n1"]) == names(r["n1"])
end

@testset "dseries assignment errors (Dynare's messages)" begin
    for (stmt, msg) in (("a.data = 1;", "You cannot overwrite data member"), ("a.z = 3;", "scalar cannot be indexed with ."),
                        ("a{'x', 'x'} = a;", "Wrong syntax"))
        err = try
            JDyn.mrun("a = dseries([1;2;3], dates('2000Q1'), 'x');\n" * stmt); nothing
        catch e
            e
        end
        @test err isa JDyn.MatlabError && occursin(msg, err.msg)
    end
end
