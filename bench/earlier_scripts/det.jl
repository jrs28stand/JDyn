include("/home/claude/work/jdyn-ho/test/common.jl")
using JDyn, Logging, Random
EDIR="/home/claude/work/jdyn-ho/test/obc/est"
dd = matread(joinpath(EDIR, "occest_bf_data.mat")); data = hcat(Float64.(vec(dd["q"])), Float64.(vec(dd["r"])))
m = load_copy(joinpath(EDIR, "bf.mod"))
mk(; kw...) = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true), kw...) end
eo, en = mk(), mk(obc_method="news")
cov = Matrix(Diagonal(eo.prior.p2 .^ 2))
mc = JDyn.sample_posterior(eo; replic=150, nblocks=2, mode=eo.x0, cov=cov, jscale=0.15, seed=20261124)
for k in 1:3:150, c in 1:2
    th = mc.draws[c,k,:]
    r = Int[]
    for (e) in (eo, en, eo, en, eo, en)
        Random.seed!(k+c); push!(r, JDyn._dsge_likelihood(e, th)[2])
    end
    length(unique(r)) > 1 && println(k, " ", c, " ", r)
end
