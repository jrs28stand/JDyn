using JDyn, MAT, Logging, Random
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
oref(name) = matread(joinpath(OREF, name * ".mat"))
odata(r) = (d = r["data"]; d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d))
IVF = Dict("likelihood_inversion_filter" => true)
mk(m, data; kw...) = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, kw...); end
println("uptime ", strip(read(`uptime`, String)))
for (case, ks) in (("occest_nkm", 1:4), ("occest_nkms", 1:3))
    r = oref(case * "_lik"); m = load_copy(joinpath(OREF, case * ".mod")); e = mk(m, odata(r); occbin=IVF)
    X = r["X"]; X = size(X,1) == e.n ? X : Matrix(X')
    for k in ks
        JDyn.log_posterior(e, X[:,k])
        t = minimum(@elapsed(JDyn.log_posterior(e, X[:,k])) for _ in 1:5)
        println(case, " vector ", k, " min of 5: ", round(t*1000, digits=1), " ms")
    end
end
println("uptime ", strip(read(`uptime`, String)))
