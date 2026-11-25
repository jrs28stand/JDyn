using JDyn, MAT, Logging
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
r = matread(joinpath(OREF, "occest_nkm_lik.mat")); data = Matrix{Float64}(r["data"])
m = load_copy(joinpath(OREF, "occest_nkm.mod"))
e = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true)); end
e.options["qz_criterium"] = 1 + 1e-6
th = [109.65434996515319, 2.2349256172471215, 0.6453732314401828, 0.6609797328917931, 0.9999999229210729, 0.7881900975597803, 0.004289142824646319, 0.010112247559243531, 6.0572589762884507e-05]
isdefined(JDyn, :OCCBIN_RULE_MEMO_BYTES) && length(ARGS) > 0 && (JDyn.OCCBIN_RULE_MEMO_BYTES[] = parse(Int, ARGS[1]))
JDyn.log_posterior(e, data[1,:] |> x -> e.x0)   # warm up
t = @elapsed lp = JDyn.log_posterior(e, th)
println("budget ", ARGS, " lp ", lp, " code ", e.last_info_code, " time ", round(t,digits=1), " s, peak RSS MB ", round(Sys.maxrss()/2^20, digits=0), " uptime ", strip(read(`uptime`, String)))
