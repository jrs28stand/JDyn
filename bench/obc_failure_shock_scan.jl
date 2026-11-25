# Scan of shock seeds and scales for the failure tests of test/test_obc_regime_ho.jl: the strict rule (undecided tolerance 0) at level 7 on zlb2, LCP and regime formulation side by side (period, counts a/b/c, lower-bound flag).
# nice julia --project=. bench/obc_failure_shock_scan.jl
using Test; include(joinpath(@__DIR__, "..", "test", "common.jl")); using Random
include(joinpath(@__DIR__, "..", "test", "rho_common.jl"))
for sc in (0.02, 0.04, 0.06)
  for seed in 1:12
    S = randn(MersenneTwister(seed), 60, rho_model("zlb2").n_exo) .* sc
    kw = (order=2, cubature=true, qmc_level=7, regions="all", cubature_undecided_tolerance=0.0)
    e = try; quiet() do; JDyn.obc_ho_simulate(mklcp("zlb2"; kw...), S); end; nothing; catch e; e; end
    e2 = try; quiet() do; JDyn.obc_hor_simulate(mkreg("zlb2"; kw...), S); end; nothing; catch e; e; end
    f(e) = e isa JDyn.ObcError ? (e.period, e.detail.counts, e.detail.lower_bound) : e
    println(sc, " ", seed, " lcp ", f(e), " reg ", f(e2))
  end
end
