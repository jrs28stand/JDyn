using JDyn, MAT, Logging
include("/home/claude/work/jdyn-ho/test/common.jl")
m = load_copy("/home/claude/work/jdyn-ho/test/ref_occbin/occest_nkm.mod")
ob = JDyn.Occbin(m)
cn = ob.cons[1]
zr = copy(ob.ys)
f() = JDyn._call_now(cn.bind, zr, ob.params, ob.ys)
println(typeof(cn.bind), " ", cn.relax === nothing)
g() = Base.invokelatest(JDyn._call_now, cn.bind, zr, ob.params, ob.ys)
g(); println("call: ", minimum(@elapsed(for _ in 1:1000; g(); end) for _ in 1:20)/1000*1e6, " us, alloc ", @allocated g(), " B")
