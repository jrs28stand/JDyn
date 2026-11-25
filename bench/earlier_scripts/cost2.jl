using JDyn, MAT, Logging, Random
include("/home/claude/work/jdyn-ho/test/common.jl")
const EDIR = "/home/claude/work/jdyn-ho/test/obc/est"; const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
mk(m, data; kw...) = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=4, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true), kw...); end
println("uptime ", strip(read(`uptime`, String)))
function tm(e, th); JDyn.log_posterior(e, th); minimum(@elapsed(JDyn.log_posterior(e, th)) for _ in 1:9); end
for name in ("zlb2m_0", "zlb2m_1e_2")
    m = load_copy(joinpath(EDIR, name * ".mod")); ob = JDyn.Occbin(m); obs = [m.ie[v] for v in ("pi", "r")]
    res = nothing
    for sd in 7:60
        try res = simulate(ob, randn(MersenneTwister(sd), 100, m.n_exo); periods=100, check_ahead_periods=60); break; catch; end
    end
    d = res.piecewise[:, obs]
    es = ["occbin" => mk(m, d), "lcp" => mk(m, d; obc_method="news", obc_formulation="lcp"), "regime" => mk(m, d; obc_method="news", obc_formulation="regime")]
    th = es[1][2].x0
    println(name, ": ", join(["$nm $(round(1000tm(e, th), digits=1)) ms" for (nm, e) in es], "; "))
    ths = [th .* exp.(0.02 .* randn(MersenneTwister(k), length(th))) for k in 1:4]
    println("   perturbed: ", join(["$nm " * join([round(1000tm(e, t), digits=1) for t in ths], "/") for (nm, e) in es], "; "))
end
# simulation per period, NKM
m = load_copy(joinpath(OREF, "occest_nkm.mod")); ob = JDyn.Occbin(m)
sh = 0.01 .* randn(MersenneTwister(1), 300, ob.n_exo)
for (nm, kw) in ("occbin" => (;), "lcp" => (obc_method=:news, obc_formulation=:lcp), "regime" => (obc_method=:news, obc_formulation=:regime))
    simulate(ob, sh[1:20, :]; periods=40, check_ahead_periods=60, kw...)
    t = minimum(@elapsed(simulate(ob, sh; periods=300, check_ahead_periods=60, kw...)) for _ in 1:5)
    println("NKM simulation $nm: ", round(1e6t/300, digits=1), " us per period (300 periods, 3 shocks)")
end
println("uptime ", strip(read(`uptime`, String)))
