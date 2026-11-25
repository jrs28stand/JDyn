using JDyn, MAT, Logging, Random
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
odata(r) = Matrix{Float64}(r["data"])
mk(m, data; kw...) = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true), kw...); end
rows = [split(l, ",") for l in readlines("/home/claude/work/jdyn-ho/test/obc/est/nkm_knife_edge.csv") if !startswith(l, "#")]
ests = Dict()
for c in ("nkm", "nkms")
    r = matread(joinpath(OREF, "occest_$(c)_lik.mat")); m = load_copy(joinpath(OREF, "occest_$c.mod")); d = odata(r)
    ests[c] = (mk(m, d), mk(m, d; obc_method="news", obc_formulation="lcp"), mk(m, d; obc_method="news"), mk(m, d; obc_method="news", obc_formulation="regime"))
end
st = Dict(:n=>0, :lcpdiff=>0, :regdiff=>0, :bothfail=>0, :ofail=>0, :maxreg=>0.0, :maxlcp=>0.0, :pydiff=>0.0)
for rw in rows
    c = rw[1]; th = parse.(Float64, rw[5:end]); pyo = parse(Float64, rw[2]); pyr = parse(Float64, rw[4])
    eo, el, ea, er = ests[c]
    o, l, a = JDyn.log_posterior(eo, th), JDyn.log_posterior(el, th), JDyn.log_posterior(ea, th)
    st[:n] += 1
    if !isfinite(o); st[:ofail] += 1; println("occbin fails: ", c, " ", o, " lcp ", l, " auto ", a); continue; end
    isfinite(l) && abs(l-o) > 1e-8*abs(o) && (st[:lcpdiff] += 1; st[:maxlcp] = max(st[:maxlcp], abs(l-o)))
    if isfinite(a) && abs(a-o) > 1e-8*abs(o); st[:regdiff] += 1; st[:maxreg] = max(st[:maxreg], abs(a-o)); println("auto differs ", c, " ", o, " ", a); end
    isfinite(a) || (println("auto fails ", c, " ", o); st[:bothfail] += 1)
    isfinite(pyo) && (st[:pydiff] = max(st[:pydiff], abs(o-pyo)))
end
println(st)
