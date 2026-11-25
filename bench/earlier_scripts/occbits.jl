using JDyn, MAT, Logging, Random
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
function run1(name, m, data, X)
    e = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true)); end
    X === nothing && (X = hcat(e.x0, [e.x0 .* exp.(0.05 .* randn(MersenneTwister(k), e.n)) for k in 1:5]...))
    X = size(X,1)==e.n ? X : Matrix(X')
    for k in 1:size(X,2)
        lp = JDyn.log_posterior(e, X[:,k]); println(name, " ", k, " ", string(reinterpret(UInt64, lp), base=16), " ", e.last_info_code)
        res = JDyn.occbin_filter(e, X[:,k]; rng=Xoshiro(3)); println(name, " f", k, " ", hash((res.filtered_errs, res.stateval, res.resids, res.error_code, res.regime_history)))
    end
end
for case in ("occest_nkm", "occest_nkms")
    r = matread(joinpath(OREF, case * "_lik.mat")); run1(case, load_copy(joinpath(OREF, case * ".mod")), Matrix{Float64}(r["data"]), r["X"])
end
dd = matread("/home/claude/work/jdyn-ho/test/obc/est/occest_bf_data.mat")
run1("bf", load_copy("/home/claude/work/jdyn-ho/test/obc/est/bf.mod"), hcat(Float64.(vec(dd["q"])), Float64.(vec(dd["r"]))), nothing)
