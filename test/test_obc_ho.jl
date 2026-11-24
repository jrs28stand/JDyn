# obc_method = news at order 2 and with cubature (src/obc_ho.jl, src/obc_qmc.jl; docs/OBC_HO_PROGRESS.md), against the real DynareOBC
# (run under Octave 8.4 with Dynare 4.5.4; recipe and reference files in pydyn's reference/dynareobc_octave) and against pydyn's prototype
# (pydyn/obc_ho.py): a small NK model with a lower bound on the interest rate, 300 shock draws, 83 periods at the bound.
using Test; include("common.jl")
using Random

const HO_DIR = joinpath(@__DIR__, "obc", "ho")
const JDyn_SRC_ROOT = DYNARE_SRC
ho_ref(name) = matread(joinpath(HO_DIR, name))
"Load a copy of the file's whole folder (include files)."
function load_folder(path)
    d = mktempdir()
    for f in readdir(dirname(path))
        p = joinpath(dirname(path), f)
        isfile(p) && cp(p, joinpath(d, f))
    end
    with_logger(NullLogger()) do
        JDyn.load(joinpath(d, basename(path)))
    end
end

@testset "obc_order / obc_cubature (src/obc_ho.jl)" begin
    m = load_copy(joinpath(HO_DIR, "nk_occ.mod"))
    ob = JDyn.Occbin(m)
    S = reshape(vec(ho_ref("shocks.mat")["ShockSequence"]), :, 1)
    names = String.(m.endo_names)
    function ref_endo(file)
        r = ho_ref(file)
        nm = [strip(String(x)) for x in vec(r["names"])]
        r, permutedims(r["endo"][[findfirst(==(v), nm) for v in names], :])
    end
    iR = findfirst(==("R"), names)

    @testset "Sobol points reproduce Dynare's qmc_sequence" begin
        x = ho_ref("sob_o2cub.mat")["x"]
        p = JDyn.qmc_normal_points(12, 9)
        @test size(p) == size(x)
        @test maximum(abs, p .- x) < 1e-14                      # DynareOBC's points (Dynare 4.5.4), icdf in Dynare 7.2's arithmetic
        @test maximum(abs, sum(p; dims=2)) < 1e-12              # mean zero: the set is symmetric
        @test JDyn.sobol_sequence(2, 3) == [0.5 0.75 0.25; 0.5 0.25 0.75]
        @test JDyn.sobol_sequence(3, 2; first=0)[:, 1] == zeros(3)
        u = [1e-9, 0.01, 0.02425, 0.1, 0.5, 0.9, 0.97575, 0.99, 1 - 1e-9]
        @test maximum(abs.(JDyn.qmc_icdf.(u) .- (-sqrt(2.0) .* JDyn.SpecialFunctions.erfcinv.(2 .* u)))) < 1e-8
        @test JDyn.qmc_icdf(0.0) == -Inf && JDyn.qmc_icdf(1.0) == Inf
        @test size(JDyn.sobol_direction_numbers(1111)) == (1111, 62)
        @test_throws ArgumentError JDyn.sobol_sequence(1112, 2)
    end

    @testset "order $order: M, q map and simulation against DynareOBC" for order in (1, 2)
        r, E = ref_endo("ref_o$(order).mat")
        h = JDyn.obc_ho(ob; order=order, horizon=32, check_ahead=32)
        @test size(h.Mf) == (64, 32)
        @test maximum(abs, h.Mf .- r["M"][1:64, 1:32]) < 1e-13
        # the affine map of q is the recursion of the expected path followed by the gap definition
        rng = MersenneTwister(order)
        for _ in 1:3
            y1, y2, o, prev, u = 0.01 .* randn(rng, 5), order == 2 ? 1e-3 .* randn(rng, 5) : zeros(5), 1e-3 .* randn(rng, 5), 1e-2 .* randn(rng, 5), randn(rng, 1)
            qd = JDyn.ho_gap_path(h, JDyn.ho_expected_path(h, y1, y2, o), prev, u)
            q = h.qc .+ h.Q1 * y1 .+ h.Qo * o
            order == 2 && (q .+= h.Q2 * y2 .+ h.Qw * kron(y1[h.sv], y1[h.sv]))
            q[1] += h.aC' * prev + h.aJ' * u
            @test maximum(abs, q .- qd) < 1e-14
        end
        sim = JDyn.obc_ho_simulate(h, S)
        @test maximum(abs, sim.with_bound .- E) < 1e-12
        @test count(<(1 + 1e-9), E[:, iR]) > 50
        @test count(w -> maximum(w) > 1e-10, eachrow(sim.wedges)) >= 50
    end

    @testset "the two orders differ; the base simulation is the pruned one" begin
        _, E1 = ref_endo("ref_o1.mat")
        _, E2 = ref_endo("ref_o2.mat")
        @test maximum(abs, E1 .- E2) > 1e-4
        h = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32)
        dr = JDyn.k_order(m; order=2, ys=ob.ys, params=ob.params, pruning=true)
        sim = JDyn.obc_ho_simulate(h, S)
        @test maximum(abs, sim.without_bound .- JDyn.simulate(dr, S; order=2, pruning=true)) < 1e-12
        small = JDyn.obc_ho_simulate(h, 0.05 .* S)             # the bound is never reached
        @test !any(!=(0), small.wedges)
        @test small.with_bound == small.without_bound
    end

    @testset "expected path = Monte Carlo mean of pruned simulations" begin
        h = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32)
        n, sv = h.n, h.sv
        function step(y1, y2, u)
            x1, x2 = y1[sv], y2[sv]
            y2n = h.ghx * x2 .+ 0.5 .* (h.ghxx * kron(x1, x1) .+ h.ghuu * kron(u, u) .+ 2 .* h.ghxu * kron(x1, u)) .+ h.cst2
            h.ghx * x1 .+ h.ghu * u, y2n
        end
        y1, y2 = zeros(n), zeros(n)
        for t in 1:151
            y1, y2 = step(y1, y2, S[t, :])
        end
        path = JDyn.ho_expected_path(h, y1, y2, zeros(n))
        K, nrep = 6, 20000
        rg = MersenneTwister(9)
        acc, acc2 = zeros(K, n), zeros(K, n)
        for _ in 1:nrep
            z1, z2 = y1, y2
            for k in 1:K
                z1, z2 = step(z1, z2, randn(rg, 1))
                tot = z1 .+ z2
                acc[k, :] .+= tot; acc2[k, :] .+= tot .^ 2
            end
        end
        mean_ = acc ./ nrep
        se = sqrt.(max.(acc2 ./ nrep .- mean_ .^ 2, 0) ./ nrep)
        @test maximum(abs.(mean_ .- path[2:K+1, :]) ./ max.(se, 1e-12)) < 4.5
    end

    @testset "cubature" begin
        r, E = ref_endo("ref_o2cub.mat")
        _, E2 = ref_endo("ref_o2.mat")
        @test maximum(abs, E .- E2) > 1e-5                      # cubature changes the answer
        h = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=9)
        Rd = real.(r["Rcc"])
        @test size(h.R) == size(Rd) == (64, 12)
        @test maximum(abs, h.R * h.R' .- Rd * Rd') < 1e-15      # the conditional covariance, tapered and pruned
        @test size(h.xi) == (12, 1023)
        # eigenvector signs are arbitrary; with the signs of DynareOBC's root and the Sobol points of Dynare, the result is the same to rounding
        Ra = copy(h.R)
        h.R = h.R .* sign.([dot(h.R[:, j], Rd[:, j]) for j in 1:12])'
        @test maximum(abs, JDyn.obc_ho_simulate(h, S).with_bound .- E) < 1e-12
        # DynareOBC's own root and points injected
        h.R, h.xi = Rd, ho_ref("sob_o2cub.mat")["x"]
        sim = JDyn.obc_ho_simulate(h, S)
        @test maximum(abs, sim.with_bound .- E) < 1e-12
        @test sim.stats[:cubature_points] == 300 * 1023
        # the result of a one-region cubature with an arbitrary eigenvector sign pattern is a different quasi-Monte-Carlo estimate (1e-5 to 1e-4
        # here, a third of the cubature effect and more); with pydyn's signs and SciPy's Joe-Kuo points it is pydyn's number, 1.49e-6
        sp = [-1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1.0]
        h.R = Ra .* sign.([dot(Ra[:, j], Rd[:, j]) for j in 1:12])' .* sp'
        h.xi = ho_ref("scipy_x.mat")["x"]
        err = maximum(abs, JDyn.obc_ho_simulate(h, S).with_bound .- E)
        @test err < 5e-6
        @test isapprox(err, 1.4878498172388177e-6; rtol=1e-6)
        h.R = Ra
        @test maximum(abs, JDyn.obc_ho_simulate(h, S).with_bound .- E) < 2e-4
    end

    @testset "cubature limits" begin
        tiny = m.Sigma_e .* 1e-12
        c = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=5, Sigma_e=tiny)
        b = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, Sigma_e=tiny)
        @test maximum(abs, JDyn.obc_ho_simulate(c, S[1:100, :]).with_bound .- JDyn.obc_ho_simulate(b, S[1:100, :]).with_bound) < 1e-9
        # without taper and pruning R R' is the exact covariance of the gap path from the shocks of the next 16 periods
        w = JDyn.obc_ho(ob; order=1, horizon=32, check_ahead=32, cubature=true, taper=false, pruning_cutoff=nothing, qmc_level=3)
        cov = zeros(64, 64)
        for i in 1:16
            B = zeros(64, 1); B[i+1:64, :] = w.Gq[1:64-i, :]
            cov .+= B * m.Sigma_e * B'
        end
        @test maximum(abs, w.R * w.R' .- cov) < 1e-12 * max(1.0, maximum(abs, cov))
        t = JDyn.obc_ho(ob; order=1, horizon=32, check_ahead=32, cubature=true, qmc_level=3)
        @test tr(t.R * t.R') < tr(cov)                          # the cosine taper shrinks it
        # cubature (one region): wedges in at least as many periods; it acts where the bound binds in expectation later (on this model the
        # bound binds in the same 83 periods with and without)
        a = JDyn.obc_ho_simulate(JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32), S)
        cb = JDyn.obc_ho_simulate(JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=7), S)
        @test count(w -> maximum(w) > 1e-10, eachrow(cb.wedges)) >= count(w -> maximum(w) > 1e-10, eachrow(a.wedges))
    end

    @testset "entry point: simulate(ob, shocks; obc_method = :news, obc_order, obc_cubature)" begin
        _, E2 = ref_endo("ref_o2.mat")
        _, Ec = ref_endo("ref_o2cub.mat")
        kw = (obc_method=:news, obc_horizon=32, check_ahead_periods=32, periods=300)
        res = JDyn.simulate(ob, S; obc_order=2, kw...)
        @test res isa JDyn.OccbinResult && size(res.piecewise) == (300, 5)
        @test maximum(abs, res.piecewise .- E2) < 1e-12
        @test size(res.news.wedges) == (300, 32) && all(==(:ok), res.news.status)
        @test maximum(res.linear[:, iR]) > 1.0                      # `linear`: the simulation that ignores the bound
        # order 1 with cubature, and order 1 alone through the new loop equals the order-1 OccBin-style loop (a shock in every period)
        r1 = JDyn.simulate(ob, S; obc_order=1, obc_cubature=true, obc_qmc_level=4, kw...)
        r0 = JDyn.simulate(ob, S; kw...)
        @test maximum(abs, JDyn.simulate(ob, S; obc_order=2, obc_cubature=false, kw...).piecewise .- res.piecewise) == 0
        ho1 = JDyn.obc_ho_simulate(JDyn.obc_ho(ob; order=1, horizon=32, check_ahead=32), S)
        @test maximum(abs, ho1.with_bound .- r0.piecewise) < 1e-9
        @test maximum(abs, r1.piecewise .- r0.piecewise) > 0
        @test_throws ArgumentError JDyn.simulate(ob, S; obc_order=3, kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S; obc_order=2, obc_cubature_regions=2, kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S; obc_order=2, obc_method=:minnorm, obc_horizon=32, periods=300)
        @test_throws ArgumentError JDyn.simulate(ob, S; obc_order=2, obc_periods_of_uncertainty=0, kw...)
        # shocks shorter than `periods` are padded with zero draws
        short = JDyn.simulate(ob, S[1:50, :]; obc_order=2, obc_method=:news, obc_horizon=32, check_ahead_periods=32, periods=60)
        @test size(short.piecewise) == (60, 5) && maximum(abs, short.piecewise[1:50, :] .- E2[1:50, :]) < 1e-12
    end

    @testset "numerically singular active-set blocks are rejected (Simple1, shock -0.02)" begin
        ms = load_copy(joinpath(HO_DIR, "simple1_occ.mod"))
        obs = JDyn.Occbin(ms)
        sys = JDyn.obc_system(obs; horizon=40, method=:news)
        q = JDyn.obc_q(sys, zeros(3), [-0.02])
        y, st, _ = JDyn.solve_active_set(sys.M, q)
        @test !(st == :ok && maximum(abs, y) > 1e6)                  # was :ok with wedges of 1.8e14
        err = try; JDyn.simulate(obs, reshape([-0.02], 1, 1); obc_method=:news, obc_horizon=40, periods=20, check_ahead_periods=40); nothing; catch e; e; end
        @test err isa JDyn.ObcError && err.status == :no_solution
    end

    @testset "mod-file, workspace and keyword routes of the higher-order options" begin
        S300 = S
        _, E2 = ref_endo("ref_o2.mat")
        d = mktempdir()
        txt = read(joinpath(HO_DIR, "nk_occ.mod"), String)
        lines = "options_.occbin.obc_method = 'news';\noptions_.occbin.obc_order = 2;\noptions_.occbin.obc_horizon = 32;\noptions_.occbin.obc_check_ahead = 32;\n"
        write(joinpath(d, "nk_o2.mod"), replace(txt, "steady; check;" => "shocks(surprise); var e; periods 1; values 10; end;\nsteady; check;") |>
                                        t -> replace(t, "occbin_setup;" => lines * "occbin_setup;"))
        m2 = with_logger(NullLogger()) do; JDyn.load(joinpath(d, "nk_o2.mod")); end
        ob2 = JDyn.Occbin(m2)
        mo = JDyn.mod_options(ob2)
        @test mo.obc_method == :news && mo.obc_order == 2 && mo.obc_horizon == 32 && mo.obc_check_ahead == 32
        r = JDyn.simulate(ob2, S300; mo..., periods=300)
        @test maximum(abs, r.piecewise .- E2) < 1e-12                       # the native lines reproduce the ref_o2 check
        @test r.news.method == :news && size(r.news.wedges) == (300, 32)
        # cubature from native lines: the same as the keyword route, and a cubature answer (not the no-cubature one)
        write(joinpath(d, "nk_o2c.mod"), replace(read(joinpath(d, "nk_o2.mod"), String), "occbin_setup;" =>
              "options_.occbin.obc_cubature = true;\noptions_.occbin.obc_qmc_level = 9;\noptions_.occbin.obc_periods_of_uncertainty = 16;\noptions_.occbin.obc_cubature_regions = 1;\noccbin_setup;"))
        mc = with_logger(NullLogger()) do; JDyn.load(joinpath(d, "nk_o2c.mod")); end
        obc_ = JDyn.Occbin(mc)
        moc = JDyn.mod_options(obc_)
        @test moc.obc_cubature === true && moc.obc_qmc_level == 9 && moc.obc_periods_of_uncertainty == 16 && moc.obc_cubature_regions == 1
        rc = JDyn.simulate(obc_, S300; moc..., periods=300)
        rk = JDyn.simulate(ob2, S300; obc_method=:news, obc_order=2, obc_cubature=true, obc_qmc_level=9, obc_horizon=32, obc_check_ahead=32, periods=300)
        @test maximum(abs, rc.piecewise .- rk.piecewise) < 1e-15
        _, Ec = ref_endo("ref_o2cub.mat")
        @test 1e-9 < maximum(abs, rc.piecewise .- E2) && maximum(abs, rc.piecewise .- Ec) < 2e-4   # own root signs: see the cubature testset
        # live workspace: dynare() runs the file, occbin_solver! uses options_.occbin; keywords change the options
        ws = with_logger(NullLogger()) do; dynare(joinpath(d, "nk_o2.mod")); end
        @test ws.options_.occbin.obc_order == 2
        occbin_solver!(ws)
        @test ws.oo_.occbin.news.method == "news" && size(ws.oo_.occbin.news.wedges, 2) == 32
        w2 = Matrix(ws.oo_.occbin.simul.piecewise)
        rs = JDyn.simulate(ob2; mo...)
        @test maximum(abs, w2 .- rs.piecewise) < 1e-12
        @test any(!iszero, ws.oo_.occbin.news.wedges)
        occbin_solver!(ws; obc_order=1)
        @test ws.options_.occbin.obc_order == 1
        occbin_solver!(ws; obc_cubature=true, obc_qmc_level=4)
        @test ws.options_.occbin.obc_cubature == true || ws.options_.occbin.obc_cubature == 1
    end

    @testset "unsupported combinations are refused, not ignored" begin
        kw = (obc_method=:news, obc_horizon=32, check_ahead_periods=32, periods=50)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_order=3, kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_order=2, obc_cubature_regions=4, kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_order=1, obc_cubature_regions=4, kw...)       # also when cubature is off
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_order=2, obc_raise=false, kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_order=2, obc_fast_path=false, kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_order=2, obc_solver_options=(formulation=:regime,), kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_order=2, obc_selection=:longest, kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_cubature=true, obc_method=:occbin)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_order=2, obc_method=:minnorm, obc_horizon=32, periods=50)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_order=2, obc_qmc_level=-1, kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_selection=:median, kw...)
        @test_throws ArgumentError JDyn.simulate(ob, S[1:50, :]; obc_selection=:longest, obc_method=:occbin)
        # constraint classes that are not one complementary pair, and several constraints
        for (rel, what) in (("examples/occbin/rbc_occbin.mod", "inconsistent"), ("tests/occbin/ppf/bf_ppf.mod", "multi_equation"))
            mm = load_folder(joinpath(JDyn_SRC_ROOT, rel))
            obm = JDyn.Occbin(mm)
            err = try JDyn.simulate(obm, zeros(5, obm.n_exo); obc_method=:news, obc_order=2); nothing catch e; e end
            @test err isa JDyn.ModelError && occursin(what, err.msg)
        end
        # an unknown option value in the file is an error when the file is loaded into the options
        d = mktempdir()
        write(joinpath(d, "bad.mod"), replace(read(joinpath(HO_DIR, "nk_occ.mod"), String), "occbin_setup;" => "options_.occbin.obc_order = 3;\noccbin_setup;"))
        mb = with_logger(NullLogger()) do; JDyn.load(joinpath(d, "bad.mod")); end
        @test_throws ArgumentError JDyn.mod_options(JDyn.Occbin(mb))
    end
end

@testset "full average over the cubature points (obc_cubature_regions = all, src/obc_cubature.jl)" begin
    m = load_copy(joinpath(HO_DIR, "nk_occ.mod"))
    ob = JDyn.Occbin(m)
    S = reshape(vec(ho_ref("shocks.mat")["ShockSequence"]), :, 1)
    @test JDyn._obc_regions("all") == 0 && JDyn._obc_regions(:all) == 0 && JDyn._obc_regions(0) == 0 && JDyn._obc_regions(1) == 1 && JDyn._obc_regions(["all"]) == 0
    @test_throws ArgumentError JDyn._obc_regions(2)
    @test_throws ArgumentError JDyn._obc_regions("some")
    @test_throws ArgumentError JDyn._obc_regions(-1)
    h = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=9, regions=0)
    T, N = h.T, h.N
    @testset "point solver and lockstep solver equal solve_active_set, point by point" begin
        rng = MersenneTwister(11)
        st = h.cub
        # gap paths that start below zero and recover (a decaying dip, some noise, an occasional second dip): plausible shapes with solutions and failures
        P = map(1:1500) do _
            s, ρ = 10.0^(-3 + 1.5 * rand(rng)), 0.7 + 0.25 * rand(rng)
            p = s .* (0.1 .- 1.1 .* ρ .^ (0:N-1)) .+ 0.05 * s .* randn(rng, N)
            rand(rng) < 0.3 && (p .+= -0.5 * s .* ρ .^ abs.((0:N-1) .- rand(rng, 5:20)))
            p
        end
        # scalar reference: the iteration of solve_active_set on the head rows and the check of the rows beyond the horizon
        ref = map(P) do p
            y, s, _ = JDyn.solve_active_set(view(h.Mf, 1:T, :), p[1:T]; tol=h.tol)
            ok = s == :ok && (N == T || minimum(p[T+1:N] .+ h.Mb * y) >= -h.tol)
            (ok, y)
        end
        # cached scalar iteration
        for (p, (ok, y)) in zip(P, ref)
            ok1 = JDyn._cub_point!(st, h.sys.M, st.Mscale, p, T, h.tol)
            ok1 && N > T && minimum(p[T+1:N] .+ h.Mb * st.y[1:T]) < -h.tol && (ok1 = false)
            @test ok1 == ok
            ok && @test maximum(abs, st.y[1:T] .- y) < 1e-12
        end
        @test length(st.cache.d) > 10 && st.cache.hits > 0
        # lockstep
        Pm = reduce(hcat, P)
        Y = zeros(T, length(P)); okv = zeros(Bool, length(P))
        JDyn._cub_group!(st, h.sys.M, st.Mscale, view(Pm, 1:T, :), T, h.tol, Y, okv)
        nsolved = 0
        for (j, (ok, y)) in enumerate(ref)
            gok = okv[j] && (N == T || minimum(Pm[T+1:N, j] .+ h.Mb * Y[:, j]) >= -h.tol)
            @test gok == ok
            ok && (nsolved += 1; @test maximum(abs, Y[:, j] .- y) < 1e-12)
        end
        @test nsolved > 300
        # a tiny cache: the answers do not change
        small = JDyn.CubCache(20_000)
        st2 = JDyn.CubState(h.sys.M, N); st2.cache = small
        for (p, (ok, y)) in zip(P[1:300], ref[1:300])
            ok1 = JDyn._cub_point!(st2, h.sys.M, st2.Mscale, p, T, h.tol)
            ok1 && N > T && minimum(p[T+1:N] .+ h.Mb * st2.y[1:T]) < -h.tol && (ok1 = false)
            @test ok1 == ok
        end
        @test small.clears > 0 && small.bytes <= small.cap
    end
    @testset "the three modes give the same simulation; the average equals the hand-written loop" begin
        sims = Dict{Symbol,Any}()
        for md in (:scalar, :cache, :group)
            hh = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=9, regions=0)
            hh.cub.mode = md
            sims[md] = JDyn.obc_ho_simulate(hh, S[1:60, :])
        end
        @test count(>(1e-9), vec(maximum(sims[:cache].wedges; dims=2))) > 20
        @test maximum(abs, sims[:cache].wedges .- sims[:scalar].wedges) < 1e-12 && maximum(abs, sims[:group].wedges .- sims[:scalar].wedges) < 1e-12
        @test maximum(abs, sims[:cache].with_bound .- sims[:scalar].with_bound) < 1e-12
        # hand-written: relevant points, a solve each, the sum over all points
        rng = MersenneTwister(5)
        ncmp = 0
        for _ in 1:60
            ncmp >= 4 && break
            y1 = 0.01 .* randn(rng, h.n)                    # a gap path of the model near the bound (some draws have no solution at some points: skipped)
            q = h.qc .+ h.Q1 * (y1 .* (0.5 + 4 * rand(rng)))
            minimum(q) < 0 || continue
            h.cub.mode = :cache
            y = try; JDyn.ho_news_shocks(h, q); catch e; e isa JDyn.ObcError || rethrow(); continue; end
            acc = zeros(T); np = size(h.xi, 2); nrel = 0
            for j in 1:np
                p = q .+ h.R * h.xi[:, j]
                pmax = max(1.0, maximum(abs, p))
                minimum(p) < -1e-8 * pmax || continue
                nrel += 1
                acc .+= JDyn._ho_lcp(h, p, 0)
            end
            nrel > 20 || continue
            @test maximum(abs, y .- acc ./ np) < 1e-12
            ncmp += 1
        end
        @test ncmp >= 3
        # vanishing variance: the average over identical points is the solution at q
        hv = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=5, regions=0)
        hv.R = 1e-9 .* hv.R
        q = 0.004 .* (0.1 .- 1.1 .* 0.8 .^ (0:N-1))
        qv = hv.qc .+ hv.Q1 * (0.03 .* ones(hv.n))
        minimum(qv) < 0 || (qv = hv.qc .+ hv.Q1 * (-0.03 .* ones(hv.n)))
        @test maximum(abs, JDyn.ho_news_shocks(hv, qv) .- JDyn._ho_lcp(hv, qv, 0)) < 1e-6
    end
end
