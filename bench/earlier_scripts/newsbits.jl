using JDyn, MAT, Logging, Random
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
for case in ("occest_nkm", "occest_nkms")
    r = matread(joinpath(OREF, case * "_lik.mat")); data = Matrix{Float64}(r["data"]); m = load_copy(joinpath(OREF, case * ".mod"))
    e = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true), obc_method="news"); end
    X = r["X"]; X = size(X,1)==e.n ? X : Matrix(X')
    for k in 1:size(X,2)
        lp = JDyn.log_posterior(e, X[:,k]); println(case, " ", k, " ", string(reinterpret(UInt64, lp), base=16), " ", e.last_info_code)
        res = JDyn.occbin_filter(e, X[:,k]; rng=Xoshiro(3)); println(case, " f", k, " ", hash((res.filtered_errs, res.stateval, res.resids, res.error_code)))
    end
    th = [114.66637644899114, 1.0892489266590764, 0.19766855658573093, 0.6874612608464392, 0.6985799052891387, 0.46604253593794787, 0.005397842157745668, 0.0016956255944929521, 0.0021002683787342213]
    case == "occest_nkm" && (lp = JDyn.log_posterior(e, th); println("extreme ", lp, " ", e.last_info_code))
end
