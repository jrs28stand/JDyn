using JDyn, MAT, Logging, Test
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
oref(name) = matread(joinpath(OREF, name * ".mat"))
odata(r) = (d = r["data"]; d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d))
NKM = load_copy(joinpath(OREF, "occest_nkm.mod"))
r = oref("occest_nkm_lik")
data = odata(r)
IVF = Dict("likelihood_inversion_filter" => true)
e = with_logger(NullLogger()) do
    JDyn.Estimation(NKM; data=data, nobs=size(data,1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=IVF)
end
e.options["qz_criterium"] = 1 + 1e-6   # pydyn builds Occbin with its default qz (1+1e-6), not the estimation one
th = [109.65434996515319, 2.2349256172471215, 0.6453732314401828, 0.6609797328917931, 0.9999999229210729, 0.7881900975597803, 0.004289142824646319, 0.010112247559243531, 6.0572589762884507e-05]
println("uptime: ", read(`uptime`, String))
t = @elapsed lp = JDyn.log_posterior(e, th)
println("lp ", lp, " code ", e.last_info_code, " ", e.last_info, " time ", t)
tr = values(JDyn._RULE_TRIES)
for x in tr; println("nodes ", length(x.rules), " summarysize MB ", Base.summarysize(x)/1e6); end
println("peak RSS MB ", Sys.maxrss()/1e6)
println(propertynames(e))
println(e.x0)
