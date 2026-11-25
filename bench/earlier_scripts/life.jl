using JDyn, MAT, Logging, Random
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
r = matread(joinpath(OREF, "occest_nkm_lik.mat")); data = Matrix{Float64}(r["data"]); X = r["X"]; X = size(X,1)==9 ? X : Matrix(X')
m = load_copy(joinpath(OREF, "occest_nkm.mod"))
e = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true)); end
rng = MersenneTwister(1); x = X[:,1]; global nok = 0
println("uptime ", strip(read(`uptime`, String)))
for i in 1:300
    th = x .* exp.(0.02 .* randn(rng, 9))
    lp = JDyn.log_posterior(e, th); isfinite(lp) && (global nok += 1)
    if i % 50 == 0
        GC.gc(); GC.gc()
        println("eval ", i, " finite ", nok, " live tries ", length(JDyn._RULE_TRIES), " gc live MB ", round(Base.gc_live_bytes()/2^20, digits=1), " maxrss MB ", round(Sys.maxrss()/2^20, digits=0), " peak trie ", JDyn._RULE_MEMO_PEAK)
    end
end
