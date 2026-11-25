# Performance regressions (FFT in the filtered theoretical moments) and reproducible random draws (seeds).
using Test, Random
include("common.jl")

@testset "FFT against the direct sum" begin
    for n in (1, 2, 3, 8, 17, 100, 128, 1000, 512)
        x = randn(ComplexF64, n)
        for sgn in (+1, -1)
            ref = [sum(x[k+1] * cis(sgn * 2pi * k * t / n) for k in 0:n-1) for t in 0:n-1]
            @test maximum(abs, JDyn._dft(x, sgn) .- ref) < 1e-11 * max(1, n)
        end
    end
end

# ------------------------------------------------------------------------------------------------ seeds
const _SEED_MOD = """
var c k z;
varexo e;
parameters alpha beta rho delta;
alpha = 0.36; beta = 0.99; rho = 0.9; delta = 0.025;
model;
  c^(-1) = beta * c(+1)^(-1) * (alpha * exp(z(+1)) * k^(alpha-1) + 1 - delta);
  k = exp(z) * k(-1)^alpha - c + (1 - delta) * k(-1);
  z = rho * z(-1) + e;
end;
initval; c = 2.75; k = 38; z = 0; e = 0; end;
steady;
shocks; var e; stderr 0.01; end;
%PRE%
stoch_simul(order=2, irf=5, periods=200, nograph, nomoments);
%MID%
stoch_simul(order=2, irf=0, periods=200, nograph, nomoments);
"""

function _seed_run(pre="", mid=""; kw...)
    d = mktempdir()
    write(joinpath(d, "s.mod"), replace(_SEED_MOD, "%PRE%" => pre, "%MID%" => mid))
    with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "s.mod"); kw...)
    end
end

@testset "random draws are reproducible and follow set_dynare_seed / seed=" begin
    a = _seed_run()
    b = _seed_run()
    @test a.oo_.exo_simul == b.oo_.exo_simul                         # default seed: identical draws
    @test isapprox(a.oo_.endo_simul, b.oo_.endo_simul; atol=1e-12)
    @test a.oo_.irfs.z_e == b.oo_.irfs.z_e                           # order-2 GIRFs use the stream too
    c = _seed_run(; seed=5)
    d = _seed_run("set_dynare_seed(5);")
    e = _seed_run("set_dynare_seed(6);")
    @test c.oo_.exo_simul == d.oo_.exo_simul                         # seed= keyword == set_dynare_seed in the file: same draws
    @test isapprox(c.oo_.endo_simul, d.oo_.endo_simul; atol=1e-12)   # paths: BLAS last-bit noise under load
    @test c.oo_.irfs.z_e == d.oo_.irfs.z_e
    @test c.oo_.endo_simul != e.oo_.endo_simul && c.oo_.endo_simul != a.oo_.endo_simul
    @test a.options_.DynareRandomStreams.seed == 0 && a.options_.DynareRandomStreams.algo == "mt19937ar"
    @test c.options_.DynareRandomStreams.seed == 5 && e.options_.DynareRandomStreams.seed == 6
    # 'default' and 'reset' go back to seed 0 / the last seed; two stoch_simul in a row continue the stream
    f = _seed_run("set_dynare_seed(9);", "set_dynare_seed('reset');")
    g = _seed_run("set_dynare_seed(9);", "")
    @test f.oo_.endo_simul != g.oo_.endo_simul
    h = _seed_run("set_dynare_seed(9);")
    @test h.options_.DynareRandomStreams.seed == 9
    k = _seed_run("set_dynare_seed(9);", "set_dynare_seed('default');")      # the 2nd draw is the default stream's 1st
    @test k.oo_.endo_simul != g.oo_.endo_simul
    # a two-argument call
    l = _seed_run("set_dynare_seed('mt19937ar', 5);")
    @test l.oo_.irfs.z_e == c.oo_.irfs.z_e
    # 'clock' stores the seed it chose (so the run can be repeated)
    cl = _seed_run("set_dynare_seed('clock');")
    sd = cl.options_.DynareRandomStreams.seed
    @test sd > 0 && _seed_run("set_dynare_seed($(Int(sd)));").oo_.irfs.z_e == cl.oo_.irfs.z_e
    @test_throws Exception _seed_run("set_dynare_seed('nonsense');")
end

@testset "re-running commands on the workspace uses the same stream" begin
    ws = _seed_run(; seed=11)
    # re-seeding makes the command draw what a fresh dynare(...; seed=11) draws in its first stoch_simul
    ref = _seed_run(; seed=11)
    JDyn.set_dynare_seed!(ws, 11)
    JDyn.stoch_simul!(ws; order=2, irf=5, periods=200, nomoments=true)
    # the draws repeat exactly; the order-2 solution itself may differ by a few ulps between solves (threaded BLAS
    # summation order once other tests have warmed it up: ghxx differed by 4e-16 with identical inputs), so the
    # simulated paths are compared to 1e-12
    @test isapprox(ws.oo_.irfs.z_e, ref.oo_.irfs.z_e; atol=1e-12)
    JDyn.set_dynare_seed!(ws, 11)
    JDyn.stoch_simul!(ws; order=2, irf=5, periods=200, nomoments=true)
    again = copy(ws.oo_.endo_simul); again_exo = copy(ws.oo_.exo_simul)
    JDyn.set_dynare_seed!(ws, 11)
    JDyn.stoch_simul!(ws; order=2, irf=5, periods=200, nomoments=true)
    @test ws.oo_.exo_simul == again_exo
    @test isapprox(ws.oo_.endo_simul, again; atol=1e-12)
    # continuing without re-seeding gives new draws
    JDyn.stoch_simul!(ws; order=2, irf=5, periods=200, nomoments=true)
    @test ws.oo_.endo_simul != again
    # options_.DynareRandomStreams.seed edited from Julia re-seeds the stream
    seeded(n) = (ws.options_.DynareRandomStreams.seed = Float64(n);
                 JDyn.stoch_simul!(ws; order=2, irf=5, periods=200, nomoments=true); copy(ws.oo_.endo_simul))
    A = seeded(12); B = seeded(13); C = seeded(12)
    @test A == C && A != B
    # an explicit rng still wins
    r1 = (JDyn.stoch_simul!(ws; order=2, irf=0, periods=50, nomoments=true, rng=Random.Xoshiro(1)); copy(ws.oo_.endo_simul))
    r2 = (JDyn.stoch_simul!(ws; order=2, irf=0, periods=50, nomoments=true, rng=Random.Xoshiro(1)); copy(ws.oo_.endo_simul))
    @test r1 == r2
end

@testset "order-1 simulate: allocation-free loop equals the textbook recursion" begin
    ws = _seed_run(; seed=1)
    dr = JDyn.as_dr(JDyn.first_order(ws.model))
    sh = randn(Random.Xoshiro(1), 500, 1) .* 0.01
    ys, ov, sv = dr["ys"], dr["order_var"], dr["state_var"]
    ref = zeros(500, length(ys)); x = zeros(length(sv))
    for t in 1:500
        yh = dr["ghx"] * x + dr["ghu"] * sh[t, :]
        ref[t, ov] .= ys[ov] .+ yh
        x = ref[t, sv] - ys[sv]
    end
    @test JDyn.simulate(dr, sh) == ref
end

@testset "filtered theoretical moments with a large grid are fast" begin
    ws = _seed_run()
    t = @elapsed with_logger(NullLogger()) do
        JDyn.stoch_simul!(ws; order=1, irf=0, periods=0, hp_filter=1600, filtered_theoretical_moments_grid=8192)
    end
    @test t < 30
end
