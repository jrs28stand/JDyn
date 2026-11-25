using JDyn, MAT, Logging, Profile
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
r = matread(joinpath(OREF, "occest_nkm_lik.mat")); m = load_copy(joinpath(OREF, "occest_nkm.mod")); d = Matrix{Float64}(r["data"])
e = with_logger(NullLogger()) do; JDyn.Estimation(m; data=d, nobs=size(d,1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true), obc_method="news", obc_formulation="regime"); end
X = r["X"]; X = size(X,1)==9 ? X : Matrix(X'); th = X[:,2]
JDyn.log_posterior(e, th)
Profile.clear(); Profile.@profile for _ in 1:150; JDyn.log_posterior(e, th); end
Profile.print(format=:flat, sortedby=:count, mincount=60, maxdepth=40)
println(); Profile.print(mincount=100, maxdepth=14)
