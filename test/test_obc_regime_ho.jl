# Order 2 and cubature for the regime formulation (obc_formulation = regime; src/obc_regime_ho.jl, docs/OBC_HO_PROGRESS.md "Regime formulation at higher order").
# New design with NO external reference for the classes that are not exact complementary pairs (approximate class, multi-equation, several constraints): the checks are
# internal consistency, as in pydyn's tests/test_obc_regime_ho.py, which this file mirrors:
#  (a) order 1 is the regime formulation's own simulation; (b) on exact-class models order 2 equals the LCP (`obc_ho`) and, for `nk`, DynareOBC itself; (c) never binding: the pruned
#  simulation; (d) the affine maps equal the direct recursion; (e) every accepted period passes OccBin's bind/relax test in levels, recomputed independently; (f) zero second-order terms
#  give order 1; (g) approximate class: regime = OccBin where the LCP may not; (h) cubature: exact class equals the LCP point by point, DynareOBC's own root and points, batched search
#  equals the point-by-point one; routing by obc_formulation; the failure rule for both formulations.
using Test; include("common.jl")
using Random
include("rho_common.jl")                 # models, shocks and constructors shared with bench/obc_regime_ho_*.jl


@testset "regime formulation at higher order" begin

@testset "(a) order 1 is the regime formulation's own simulation: $name" for name in ("nkm", "zlb", "zlb2", "nk", "bf", "sw")
    m, S, hz = rho_model(name), rho_shocks(name), RHO_CFG[name][2]
    h = mkreg(name; order=1)
    r = JDyn.obc_hor_simulate(h, S)
    ob = JDyn.Occbin(m)
    ref = JDyn.simulate(ob, S; obc_method=:news, obc_formulation=:regime, obc_horizon=hz, check_ahead_periods=hz, periods=size(S, 1)).piecewise
    @test maximum(abs, r.with_bound .- ref) <= 1e-11 * max(1.0, maximum(abs, ref))
    @test rho_binds(r) >= 5
end

@testset "(b) exact class, order 2: the regime formulation equals the LCP: $name" for name in ("zlb", "zlb2", "nk", "borrcon", "nkm")
    S, hz = rho_shocks(name), RHO_CFG[name][2]
    a = JDyn.obc_hor_simulate(mkreg(name; order=2), S)
    b = JDyn.obc_ho_simulate(mklcp(name; order=2), S)
    @test maximum(abs, a.with_bound .- b.with_bound) <= 1e-12
    @test maximum(abs, a.without_bound .- b.without_bound) <= 1e-13
    @test rho_binds(a) >= 5
    if name != "borrcon"                                  # the wedges of the (single) switched equation are the LCP's up to the unit sign
        ra = reshape(a.wedges, size(S, 1), hz, :)
        @test maximum(abs, dropdims(sum(abs.(ra); dims=3); dims=3) .- reshape(sum(abs.(reshape(b.wedges, size(S, 1), hz, :)); dims=3), size(S, 1), hz)) <= 1e-10
    end
end

@testset "(b') exact class, order 2: nk equals DynareOBC (ref_o2.mat)" begin
    m = rho_model("nk")
    r = rho_ref("ref_o2.mat")
    nm = [strip(String(x)) for x in vec(r["names"])]
    E = permutedims(r["endo"][[findfirst(==(v), nm) for v in String.(m.endo_names)], :])
    S = reshape(vec(rho_ref("shocks.mat")["ShockSequence"]), :, 1)
    h = JDyn.obc_ho_regime(JDyn.Occbin(m); order=2, horizon=32, check_ahead=32)
    @test maximum(abs, JDyn.obc_hor_simulate(h, S).with_bound .- E) <= 1e-12
end

@testset "(c) never binding is the pruned simulation: $name" for name in ("nkm", "zlb2", "bf", "sw")
    S = rho_shocks(name) .* 0.01
    h = mkreg(name; order=2)
    r = JDyn.obc_hor_simulate(h, S)
    @test !any(!iszero, r.wedges)
    @test r.with_bound == r.without_bound
    if name in ("nkm", "zlb2")                                  # the pruned simulation of the LCP solver (same decision rules); sw and bf are not complementary pairs
        @test maximum(abs, r.without_bound .- JDyn.obc_ho_simulate(mklcp(name; order=2), S).without_bound) <= 1e-12
    end
end

@testset "(d) affine maps equal the direct recursion: $name, order $order" for name in ("nkm", "zlb2", "bf", "sw"), order in (1, 2)
    m = rho_model(name)
    h = mkreg(name; order=order)
    rng = MersenneTwister(1)
    n = m.n_endo
    for _ in 1:4
        y1, y2, o, prev = (0.01 .* randn(rng, n) for _ in 1:4)
        order == 1 && (y2 = zeros(n))
        u = 0.01 .* randn(rng, m.n_exo)
        a, Y, A = JDyn.hor_inputs_from_state(h, y1, y2, o, prev, u)
        a2, Y2, A2 = JDyn.hor_inputs_direct(h, y1, y2, o, prev, u)
        sc = max(1.0, maximum(abs, A2))
        @test maximum(abs, a .- a2) <= 1e-12 * sc && maximum(abs, Y .- Y2) <= 1e-12 * sc && maximum(abs, A .- A2) <= 1e-12 * sc
    end
end

@testset "(e) accepted periods pass OccBin's test in levels, recomputed independently: $name" for name in ("nkm", "zlb2", "bf", "sw", "borrcon")
    m = rho_model(name)
    h = mkreg(name; order=2)
    h.trace = Any[]
    r = JDyn.obc_hor_simulate(h, rho_shocks(name))
    ob, rg, N, T = h.ob, h.rg, h.N, h.T
    Yw = JDyn._Yw(rg)
    nbind = 0
    for tr in h.trace
        E = JDyn.hor_expected_path(h, tr.y1, tr.y2, tr.o_pre)                     # (N+1) x n deviations, declaration order
        path = any(!iszero, tr.w) ? E .+ reduce(hcat, [(Yw[t] * tr.w) for t in 1:N+1])' : E
        for (j, cn) in enumerate(ob.cons)
            bind_bad, relax_bad = false, false
            for t in 1:N
                zl = path[t, :] .+ ob.ys
                b = Base.invokelatest(cn.bind, zl, ob.params, ob.ys)
                bnd = b === true || (b isa Number && !isnan(b) && b != 0)
                rx = cn.relax === nothing ? !bnd : (x = Base.invokelatest(cn.relax, zl, ob.params, ob.ys); x === true || (x isa Number && !isnan(x) && x != 0))
                zt = t <= T && tr.z[t, j]
                bind_bad |= bnd && !zt
                relax_bad |= rx && zt
            end
            @test !bind_bad && !relax_bad
        end
        nbind += any(tr.z) ? 1 : 0
    end
    @test nbind >= 3
end

@testset "(f) zero second-order terms give order 1" begin
    S = rho_shocks("nkm")
    h1, h2 = mkreg("nkm"; order=1), mkreg("nkm"; order=2)
    h2.ghxx .*= 0; h2.ghxu .*= 0; h2.ghuu .*= 0; h2.cst2 .*= 0; h2.c2 .*= 0
    JDyn._hor_build_maps!(h2)
    @test maximum(abs, JDyn.obc_hor_simulate(h2, S).with_bound .- JDyn.obc_hor_simulate(h1, S).with_bound) <= 1e-11
end

@testset "(g) approximate class (NKM): the regime formulation is OccBin's solution, the LCP can differ at knife-edge periods" begin
    m = rho_model("nkm")
    S = rho_shocks("nkm"; T=400, seed=11)
    ob = JDyn.Occbin(m)
    ref = JDyn.simulate(ob, S; periods=size(S, 1))
    reg = JDyn.obc_hor_simulate(JDyn.obc_ho_regime(ob; order=1), S).with_bound
    @test maximum(abs, reg .- ref.piecewise) <= 1e-8
    lcp = JDyn.obc_ho_simulate(JDyn.obc_ho(ob; order=1, m_diagnostic=false), S).with_bound
    @test all(isfinite, lcp)
end

# ------------------------------------------------------------------------------------------------ cubature
@testset "(h1) cubature, exact class: same root, same points, same relevant points, same wedges as the LCP: $name regions=$regions level=$level" for (name, regions, level) in
        (("nk", "all", 7), ("zlb", "all", 7), ("nk", 1, 9), ("zlb2", 1, 9), ("zlb", 4, 9), ("zlb2", 4, 9))
    S = rho_shocks(name; T=60)
    kw = (order=2, cubature=true, qmc_level=level, regions=regions)
    a, b = mkreg(name; kw...), mklcp(name; kw...)
    @test maximum(abs, a.R .- b.R) <= 1e-13
    ra, rb = JDyn.obc_hor_simulate(a, S), JDyn.obc_ho_simulate(b, S)
    @test ra.relevant == rb.relevant
    @test maximum(abs, ra.with_bound .- rb.with_bound) <= 1e-12
    @test sum(ra.relevant) > 1000
end

@testset "(h2) cubature, exact class: failures and the tolerance equal the LCP's" begin
    S = rho_fail_shocks()
    kw = (order=2, cubature=true, qmc_level=7, regions="all", cubature_undecided_tolerance=0.0)
    errs = [try; quiet() do; mk === mkreg ? JDyn.obc_hor_simulate(mk("zlb2"; kw...), S) : JDyn.obc_ho_simulate(mk("zlb2"; kw...), S); end; nothing; catch e; e; end for mk in (mkreg, mklcp)]
    for e in errs
        @test e isa JDyn.ObcError && e.detail isa JDyn.CubatureFailure && e.period == 2 && e.detail.counts.b == 1 && e.status == :horizon_too_short
    end
    ra = quiet() do; JDyn.obc_hor_simulate(mkreg("zlb2"; cubature_failure_tolerance=0.05, kw...), S); end
    rb = quiet() do; JDyn.obc_ho_simulate(mklcp("zlb2"; cubature_failure_tolerance=0.05, kw...), S); end
    @test ra.dropped == rb.dropped && sum(ra.dropped) >= 1
    @test maximum(abs, ra.with_bound .- rb.with_bound) <= 1e-12
end

@testset "(h3) cubature equals DynareOBC (nk): one region to 1e-12, four regions as in the LCP test" begin
    m = rho_model("nk")
    r = rho_ref("ref_o2cub.mat")
    nm = [strip(String(x)) for x in vec(r["names"])]
    E = permutedims(r["endo"][[findfirst(==(v), nm) for v in String.(m.endo_names)], :])
    S = reshape(vec(rho_ref("shocks.mat")["ShockSequence"]), :, 1)
    ob = JDyn.Occbin(m)
    h = JDyn.obc_ho_regime(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=9, regions=1)
    Rd = real.(r["Rcc"])
    @test maximum(abs, h.R * h.R' .- Rd * Rd') <= 1e-16 * 10
    h.R, h.xi = Rd, rho_ref("sob_o2cub.mat")["x"]
    @test maximum(abs, JDyn.obc_hor_simulate(h, S).with_bound .- E) <= 1e-12
    r4 = rho_ref("ref_o2cub4.mat")
    nm4 = [strip(String(x)) for x in vec(r4["names"])]
    E4 = permutedims(r4["endo"][[findfirst(==(v), nm4) for v in String.(m.endo_names)], :])
    h4 = JDyn.obc_ho_regime(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=9, regions=4)
    h4.R, h4.xi = Rd, rho_ref("sob_o2cub.mat")["x"]
    d = vec(maximum(abs.(JDyn.obc_hor_simulate(h4, S).with_bound .- E4); dims=2))
    @test count(<(1e-12), d) >= 290 && maximum(d) < 0.01
end

@testset "(h4) vanishing variance is no cubature: $name" for name in ("nkm", "bf", "zlb")
    m, S = rho_model(name), rho_shocks(name; T=60)
    tiny = Matrix(m.Sigma_e) .* 1e-12
    c = JDyn.obc_hor_simulate(mkreg(name; order=2, cubature=true, Sigma_e=tiny, qmc_level=5), S)
    b = JDyn.obc_hor_simulate(mkreg(name; order=2, Sigma_e=tiny), S)
    @test maximum(abs, c.with_bound .- b.with_bound) <= 1e-9
end

@testset "(h5) cubature makes agents react before the bound regime" begin
    S = rho_shocks("nkm")
    a = JDyn.obc_hor_simulate(mkreg("nkm"; order=2), S)
    c = JDyn.obc_hor_simulate(mkreg("nkm"; order=2, cubature=true, qmc_level=7, regions=1), S)
    @test rho_binds(c) > 3 * rho_binds(a)
end

@testset "(h6) the batched guess-and-verify returns, point by point, what the complete search returns" begin
    S = rho_shocks("nkm"; T=80)
    h = mkreg("nkm"; order=2, cubature=true, qmc_level=6, regions="all", cubature_failure_tolerance=0.5)
    h.trace = Any[]
    quiet() do; JDyn.obc_hor_simulate(h, S); end
    rg, nb, N, T, k = h.rg, h.nb, h.N, h.T, h.nk
    Ro = JDyn._hor_other_roots(h)
    checked = 0
    for tr in h.trace[1:60]
        a, Y, A = JDyn.hor_inputs_from_state(h, tr.y1, tr.y2, tr.o_pre, tr.prev, tr.u)
        pY = vec(Y) .+ Ro[nb*N+1:end, :] * h.xi
        pA = vec(A) .+ Ro[1:nb*N, :] * h.xi
        rel, vbs = JDyn._hor_relevant(h, pY)
        isempty(rel) && continue
        sel = 1:min(40, length(rel))
        rl = rel[sel]
        pa = reduce(hcat, [vcat([pA[(b-1)*N+1:(b-1)*N+T, p] for b in 1:nb]...) for p in rl])
        W, ok = JDyn._hor_gv_many(h, pa, pY[:, rl], vbs[sel])
        for (j, p) in enumerate(rl)
            Ap, Yp = reshape(pA[:, p], N, nb), reshape(pY[:, p], k, N)
            st, z, w, _ = JDyn._hor_search(h, Ap, Yp, 0.1)
            if st != :ok
                @test !ok[j]
                continue
            end
            if ok[j]
                @test maximum(abs, W[:, j] .- w) <= 1e-11 * max(1.0, maximum(abs, w))
                checked += 1
            end
        end
    end
    @test checked > 100
end

# ------------------------------------------------------------------------------------------------ routing and approximate class
@testset "(h7) routing by obc_formulation at higher order" begin
    nkm, zlb, bf = rho_model("nkm"), rho_model("zlb"), rho_model("bf")
    form(m, S; kw...) = quiet() do; JDyn.simulate(JDyn.Occbin(m), S; obc_method=:news, obc_order=2, kw...).news.formulation; end
    @test form(nkm, rho_shocks("nkm")[1:5, :]; obc_horizon=40, check_ahead_periods=40) == :regime                      # approximate class: auto is the regime formulation
    @test form(nkm, rho_shocks("nkm")[1:5, :]; obc_horizon=40, check_ahead_periods=40, obc_formulation=:lcp) == :lcp
    @test form(zlb, rho_shocks("zlb")[1:5, :]; obc_horizon=20, check_ahead_periods=20) == :lcp                          # exact class: unchanged
    @test form(zlb, rho_shocks("zlb")[1:5, :]; obc_horizon=20, check_ahead_periods=20, obc_formulation=:regime) == :regime
    @test form(bf, rho_shocks("bf")[1:5, :]; obc_horizon=32, check_ahead_periods=32) == :regime                         # multi-equation: was refused
    @test_throws ArgumentError form(bf, rho_shocks("bf")[1:5, :]; obc_horizon=32, check_ahead_periods=32, obc_formulation=:lcp)
    @test_throws ArgumentError form(zlb, rho_shocks("zlb")[1:5, :]; obc_horizon=20, check_ahead_periods=20, obc_formulation="hybrid")
    # cubature too
    @test form(nkm, rho_shocks("nkm")[1:5, :]; obc_horizon=40, check_ahead_periods=40, obc_cubature=true, obc_qmc_level=4) == :regime
end

@testset "(h8) exact class: auto is bit for bit the LCP" begin
    m, S = rho_model("nk"), rho_shocks("nk")
    ob = JDyn.Occbin(m)
    kw = (obc_method=:news, obc_order=2, obc_horizon=32, check_ahead_periods=32, periods=size(S, 1))
    a = quiet() do; JDyn.simulate(ob, S; kw...); end
    b = quiet() do; JDyn.simulate(ob, S; obc_formulation=:lcp, kw...); end
    @test a.piecewise == b.piecewise && a.news.formulation == :lcp
    c = quiet() do; JDyn.simulate(ob, S; obc_cubature=true, obc_qmc_level=5, kw...); end
    d = quiet() do; JDyn.simulate(ob, S; obc_cubature=true, obc_qmc_level=5, obc_formulation=:lcp, kw...); end
    @test c.piecewise == d.piecewise
end

@testset "(h9) sw (multi-equation): cubature raises with class (a) points; the default time limit and tolerances" begin
    S = rho_shocks("sw"; T=10)
    h = mkreg("sw"; order=2, cubature=true, qmc_level=5)
    @test h.milp_time_limit == 0.1 && h.undecided_tolerance == 0.005 && h.undecided_min_points == 3 && h.tolerance == 0.0
    err = try; quiet() do; JDyn.obc_hor_simulate(h, S); end; nothing; catch e; e; end
    @test err isa JDyn.ObcError && err.detail isa JDyn.CubatureFailure && err.detail.counts.a >= 1 && err.status == :no_solution
    @test occursin("PROVEN no solution", err.msg) && occursin("docs/OBC_CUBATURE_FAILURES.md", err.msg)
end

@testset "(h10) approximate-class family (two constraints): order-2 regime vs order-2 LCP differ at knife-edge periods only; regime accepted in every period: $name" for (name, maxp) in (("zlb2m_0", 0), ("zlb2m_1e_2", 5))
    m = rho_model(name)
    S = randn(MersenneTwister(4), 300, m.n_exo) .* sqrt.(diag(Matrix(m.Sigma_e)))' .* 1.6
    ob = JDyn.Occbin(m)
    h = JDyn.obc_ho_regime(ob; order=2)
    h.trace = Any[]
    a = JDyn.obc_hor_simulate(h, S)
    b = JDyn.obc_ho_simulate(JDyn.obc_ho(ob; order=2, m_diagnostic=false), S)
    dd = vec(maximum(abs.(a.with_bound .- b.with_bound); dims=2))
    @test rho_binds(a) >= 10
    @test count(>(1e-8), dd) <= maxp && maximum(dd) < 1e-3
    @test length(h.trace) == size(S, 1)
end

# ------------------------------------------------------------------------------------------------ the failure rule for both formulations
@testset "(h11) failure rule for the regime formulation (docs/OBC_CUBATURE_FAILURES.md)" begin
    @test JDyn._hor_class(:horizon_too_short, false) == :b
    @test JDyn._hor_class(:no_solution, true) == :a
    @test JDyn._hor_class(:no_solution, false) == :c                  # an infeasible integer programme is not a proof
    @test JDyn._hor_class(:not_converged, true) == :c
    # default NKM at level 11: completes, one warning, counts by class, the rule holds in every period
    S = rho_shocks("nkm"; T=100)
    h = mkreg("nkm"; order=2, cubature=true, qmc_level=11)
    r = @test_logs (:warn, r"NOT proven to have no solution") JDyn.obc_hor_simulate(h, S)
    @test sum(r.dropped) > 0 && sum(r.dropped_a) == 0
    @test r.dropped == r.dropped_a .+ r.dropped_b .+ r.dropped_c
    @test sum(r.dropped_c) == r.stats[:failed_c] && sum(r.dropped) == r.stats[:dropped_points]
    @test all(r.dropped[t] <= JDyn.cubature_limits(r.relevant[t], 0.0, 0.005)[2] for t in eachindex(r.dropped))
    # undecided tolerance 0 is the strict rule
    e = try; quiet() do; JDyn.obc_hor_simulate(mkreg("nkm"; order=2, cubature=true, qmc_level=11, cubature_undecided_tolerance=0), S); end; nothing; catch e; e; end
    @test e isa JDyn.ObcError && e.detail.counts.c >= 1 && e.status == :not_converged && e.detail.undecided_tolerance == 0 && e.period >= 1
    @test occursin("docs/OBC_CUBATURE_FAILURES.md", e.msg) && any(occursin("larger obc_cubature_milp_time_limit", a) for a in e.detail.advice)
    # where nothing fails the two settings are the same simulation
    S2 = rho_shocks("nk"; T=60)
    a = JDyn.obc_hor_simulate(mkreg("nk"; order=2, cubature=true, qmc_level=7), S2)
    b = JDyn.obc_hor_simulate(mkreg("nk"; order=2, cubature=true, qmc_level=7, cubature_undecided_tolerance=0), S2)
    @test maximum(abs, a.with_bound .- b.with_bound) <= 1e-14 && sum(a.dropped) == 0            # (two builds of the model differ by ~1e-20 in ghxx: a property of k_order)
end

@testset "(h12) the LCP class follows the same rule and class (a) still raises" begin
    S = rho_fail_shocks()
    kw = (order=2, cubature=true, qmc_level=7, regions="all")
    r = @test_logs (:warn, r"by class") JDyn.obc_ho_simulate(mklcp("zlb2"; kw...), S)       # class (b) points (horizon): dropped within the tolerance
    @test sum(r.dropped_b) > 0 && sum(r.dropped_a) == 0 && sum(r.dropped_c) == 0
    @test_throws JDyn.ObcError quiet() do; JDyn.obc_ho_simulate(mklcp("zlb2"; cubature_undecided_tolerance=0.0, kw...), S); end
    ra = quiet() do; JDyn.obc_hor_simulate(mkreg("zlb2"; kw...), S); end
    @test ra.dropped_b == r.dropped_b
    @test maximum(abs, ra.with_bound .- r.with_bound) <= 1e-12
end

@testset "(h13) interaction with the failure tolerance, refusals" begin
    S = rho_fail_shocks()
    kw = (order=2, cubature=true, qmc_level=7, regions="all", cubature_undecided_tolerance=0.0)
    r = quiet() do; JDyn.obc_hor_simulate(mkreg("zlb2"; cubature_failure_tolerance=0.02, kw...), S); end
    @test all(r.dropped .<= 0.02 .* r.relevant) && sum(r.dropped) > 0
    @test_throws JDyn.ObcError quiet() do; JDyn.obc_hor_simulate(mkreg("zlb2"; cubature_failure_tolerance=0.0001, kw...), S); end
    @test_throws ArgumentError mkreg("zlb2"; kw..., cubature_undecided_tolerance=1.0)
    @test_throws ArgumentError mkreg("zlb2"; kw..., cubature_failure_tolerance=1.0)
    @test_throws ArgumentError mkreg("zlb2"; kw..., cubature_milp_time_limit=-1)
    @test_throws ArgumentError mkreg("zlb2"; kw..., cubature_undecided_min_points=-1)
    # through simulate: the options reach the regime solver
    ob = JDyn.Occbin(rho_model("nkm"))
    ks = (obc_method=:news, obc_order=2, obc_cubature=true, obc_qmc_level=5, obc_horizon=40, check_ahead_periods=40, periods=30)
    rs = quiet() do; JDyn.simulate(ob, rho_shocks("nkm"; T=30); ks...); end
    @test rs.news.formulation == :regime && all(i -> hasproperty(i, :dropped_a) && i.formulation == :regime, rs.news.info)
    @test size(rs.news.wedges, 2) == length(JDyn.regime_obc(ob, 40, 80).cols) * 40
end

end
