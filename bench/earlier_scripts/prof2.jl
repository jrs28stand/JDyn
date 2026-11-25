using JDyn, MAT, Logging, Profile, Random
include("/home/claude/work/jdyn-ho/test/common.jl")
m = load_copy("/home/claude/work/jdyn-ho/test/obc/est/zlb2m_1e_2.mod"); ob = JDyn.Occbin(m); obs = [m.ie[v] for v in ("pi", "r")]
res = simulate(ob, randn(MersenneTwister(7), 100, m.n_exo); periods=100, check_ahead_periods=60)
e = with_logger(NullLogger()) do; JDyn.Estimation(m; data=res.piecewise[:, obs], nobs=100, first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true), obc_method="news", obc_formulation="regime"); end
th = e.x0; JDyn.log_posterior(e, th)
Profile.clear(); Profile.@profile for _ in 1:100; JDyn.log_posterior(e, th); end
Profile.print(format=:flat, sortedby=:count, mincount=40, maxdepth=40)
