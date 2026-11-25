include("/home/claude/work/jdyn-ho/test/common.jl")
using JDyn, Logging, Random, Printf
m = load_copy("/home/claude/work/jdyn-ho/test/obc/obc_zlb2.mod"); ob = JDyn.Occbin(m)
S = reshape(randn(MersenneTwister(1), 100) .* 0.05, :, 1)
kw = (obc_method=:news, obc_horizon=24, check_ahead_periods=24, periods=100)
r0 = JDyn.simulate(ob, S; kw...)
h1 = JDyn.obc_ho(ob; order=1, horizon=24, check_ahead=24)
println(h1)
o1 = JDyn.obc_ho_simulate(h1, S)
println("order 1: max diff vs news loop ", maximum(abs, o1.with_bound .- r0.piecewise), "; wedges ", maximum(abs, o1.wedges .- r0.news.wedges), " binding periods ", count(>(1e-9), vec(maximum(o1.wedges; dims=2))))
println("both constraints binding in ", count(t -> maximum(o1.wedges[t, 1:24]) > 1e-9 && maximum(o1.wedges[t, 25:48]) > 1e-9, 1:100))
res = Dict()
for md in (:scalar, :cache, :group)
    hh = JDyn.obc_ho(ob; order=2, horizon=24, check_ahead=24, cubature=true, qmc_level=7, regions=0, m_diagnostic=false, cubature_failure_tolerance=0.5)
    hh.cub.mode = md
    t = @elapsed r = with_logger(NullLogger()) do; JDyn.obc_ho_simulate(hh, S[1:40, :]); end
    res[md] = r
    @printf("%s: %.1f ms/period, relevant %d, dropped %d, nrows %d dim %d\n", md, 1000t/40, sum(r.relevant), sum(r.dropped), hh.NR, size(hh.R,2))
end
println(maximum(abs, res[:cache].wedges .- res[:scalar].wedges), " ", maximum(abs, res[:group].wedges .- res[:scalar].wedges))
