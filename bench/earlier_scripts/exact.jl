using JDyn, Logging, Random
include("/home/claude/work/jdyn-ho/test/common.jl")
E = "/home/claude/work/jdyn-ho/test/obc"
m = load_copy(joinpath(E, "est", "zlb2m_0.mod")); ob = JDyn.Occbin(m); obs = [m.ie[v] for v in ("pi", "r")]
res = simulate(ob, randn(MersenneTwister(7), 100, m.n_exo); periods=100, check_ahead_periods=60)
e = with_logger(NullLogger()) do; JDyn.Estimation(m; data=res.piecewise[:, obs], nobs=100, first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true), obc_method="news"); end
for k in 0:5
    th = k == 0 ? copy(e.x0) : e.x0 .* exp.(0.03 .* randn(MersenneTwister(k), length(e.x0)))
    lp = JDyn.log_posterior(e, th); println("zlb2m_0 ", k, " ", string(reinterpret(UInt64, lp), base=16), " ", e.last_info_code)
end
for f in ("obc_zlb.mod", "nk_zlb2.mod", "obc_zlb2.mod", "nk_zlb.mod")
    mm = load_copy(joinpath(E, f)); o = JDyn.Occbin(mm)
    sh = 0.03 .* randn(MersenneTwister(3), 40, o.n_exo)
    r = simulate(o, sh; periods=80, obc_method=:news)
    println(f, " ", hash(r.piecewise), " ", hash(r.regime_history), " ", r.news.formulation)
end
