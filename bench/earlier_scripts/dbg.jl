using JDyn, MAT, Logging, Random, LinearAlgebra
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
r = matread(joinpath(OREF, "occest_nkms_lik.mat")); md = matread(joinpath(OREF, "occest_nkms_mode.mat"))
data = Matrix{Float64}(r["data"]); m = load_copy(joinpath(OREF, "occest_nkms.mod"))
e = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=4, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true)); end
x, cov = vec(md["xparam1"]), inv(md["hh"])
println("lp at mode ", JDyn.log_posterior(e, x), " ", e.last_info_code, " ", e.last_info)
mc = with_logger(NullLogger()) do; JDyn.sample_posterior(e; replic=20, nblocks=2, mode=x, cov=cov, jscale=1.3, seed=20261124); end
println(mc.acceptance_rate, " ", mc.accepted)
println(mc.draws[1,1:4,:]); println(propertynames(mc))
lb, ub = JDyn.prior_bounds(e.prior, e.options["prior_trunc"]); println(lb, ub)
for i in 1:3
  th = x .+ 0.0005*randn(3); println(th, " ", JDyn.log_posterior(e, th), " ", e.last_info_code, " ", e.last_info)
end
s = mc.draws[1,1,:]
println("start lp ", JDyn.log_posterior(e, s), " ", e.last_info_code, " ", e.last_info, " ; recorded ", mc.log_posterior[1,1:3])
println(mc.neval, " ", mc.info)
println("est.lb ", e.lb, " ub ", e.ub, " names ", e.names)
