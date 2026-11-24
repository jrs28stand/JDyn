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
            q[1] += (h.aC * prev + h.aJ * u)[1]
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
        h = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=9, regions=1)
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
        c = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=5, regions=1, Sigma_e=tiny)
        b = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, Sigma_e=tiny)
        @test maximum(abs, JDyn.obc_ho_simulate(c, S[1:100, :]).with_bound .- JDyn.obc_ho_simulate(b, S[1:100, :]).with_bound) < 1e-9
        # without taper and pruning R R' is the exact covariance of the gap path from the shocks of the next 16 periods
        w = JDyn.obc_ho(ob; order=1, horizon=32, check_ahead=32, cubature=true, taper=false, pruning_cutoff=nothing, qmc_level=3, regions=1)
        cov = zeros(64, 64)
        for i in 1:16
            B = zeros(64, 1); B[i+1:64, :] = w.Gq[1:64-i, :]
            cov .+= B * m.Sigma_e * B'
        end
        @test maximum(abs, w.R * w.R' .- cov) < 1e-12 * max(1.0, maximum(abs, cov))
        t = JDyn.obc_ho(ob; order=1, horizon=32, check_ahead=32, cubature=true, qmc_level=3, regions=1)
        @test tr(t.R * t.R') < tr(cov)                          # the cosine taper shrinks it
        # cubature (one region): wedges in at least as many periods; it acts where the bound binds in expectation later (on this model the
        # bound binds in the same 83 periods with and without)
        a = JDyn.obc_ho_simulate(JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32), S)
        cb = JDyn.obc_ho_simulate(JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=7, regions=1), S)
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
        r1 = JDyn.simulate(ob, S; obc_order=1, obc_cubature=true, obc_qmc_level=4, obc_cubature_regions=1, kw...)
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
        rk = JDyn.simulate(ob2, S300; obc_method=:news, obc_order=2, obc_cubature=true, obc_qmc_level=9, obc_cubature_regions=1, obc_horizon=32, obc_check_ahead=32, periods=300)
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

@testset "cubature defaults: the full average over 4,095 points" begin
    m = load_copy(joinpath(HO_DIR, "nk_occ.mod"))
    ob = JDyn.Occbin(m)
    h = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true)
    @test h.regions == 0 && h.qmc_level == 11 && size(h.xi, 2) == 4095
    @test JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, regions=1).regions == 1
    @test JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, regions="all").regions == 0
    S = reshape(vec(ho_ref("shocks.mat")["ShockSequence"]), :, 1)[1:40, :]
    kw = (obc_method=:news, obc_order=2, obc_cubature=true, obc_horizon=32, check_ahead_periods=32, periods=40)
    rd = JDyn.simulate(ob, S; kw...)                                             # defaults: all, level 11
    ra = JDyn.simulate(ob, S; obc_cubature_regions="all", obc_qmc_level=11, kw...)
    r1 = JDyn.simulate(ob, S; obc_cubature_regions=1, obc_qmc_level=11, kw...)
    @test rd.piecewise == ra.piecewise
    @test maximum(abs, rd.piecewise .- r1.piecewise) > 1e-6                      # the full average prices what one region does not
end

# ---------------------------------------------------------------------------------------------- models for the failure tests
"The 32-variable SW-type stand-in of pydyn's bench (obc_sw_zlb.mod with `rlag = rnot`: only the policy-rate equation switches); M is not an S-matrix."
function sw_cp_model()
    src = read(joinpath(@__DIR__, "obc", "obc_sw_zlb.mod"), String)
    a = first(findfirst("[name = 'Auxiliary notional interest rate',bind='zlb']", src))
    b = first(findfirst("    [name='Output gap check']", src))
    d = mktempdir()
    for f in readdir(joinpath(@__DIR__, "obc"))
        isfile(joinpath(@__DIR__, "obc", f)) && cp(joinpath(@__DIR__, "obc", f), joinpath(d, f); force=true)
    end
    write(joinpath(d, "sw_cp.mod"), src[1:a-1] * "[name = 'Auxiliary notional interest rate']\nrlag = rnot;\n" * src[b:end])
    with_logger(NullLogger()) do; JDyn.load(joinpath(d, "sw_cp.mod")); end
end

@testset "up-front diagnostic of M when cubature is on" begin
    m = load_copy(joinpath(HO_DIR, "nk_occ.mod"))
    ob = JDyn.Occbin(m)
    h = @test_logs JDyn.obc_ho(ob; order=1, horizon=32, check_ahead=32, cubature=true, qmc_level=3)       # S-matrix: no warning
    d = h.cub.m_diag
    @test d.is_s === true && d.level == :s && d.seconds >= 0 && d.is_p === nothing
    @test JDyn.obc_ho(ob; order=1, horizon=32, check_ahead=32, cubature=true, qmc_level=3).cub.m_diag === d           # cached per parameter set
    @test JDyn.obc_ho(ob; order=1, horizon=32, check_ahead=32, cubature=true, qmc_level=3, m_diagnostic=false).cub.m_diag === nothing
    @test JDyn.obc_ho(ob; order=1, horizon=32, check_ahead=32, qmc_level=3).cub.m_diag === nothing                   # no cubature: not computed
    @test JDyn.obc_ho(ob; order=1, horizon=12, check_ahead=12, cubature=true, qmc_level=3, m_diagnostic="full").cub.m_diag.level == :full
    @test_throws ArgumentError JDyn.obc_ho(ob; order=1, cubature=true, qmc_level=3, m_diagnostic="maybe")
    sw = sw_cp_model()
    h2 = @test_logs (:warn, r"not an S-matrix") JDyn.obc_ho(JDyn.Occbin(sw); order=1, horizon=32, check_ahead=32, cubature=true, qmc_level=3)
    @test h2.cub.m_diag.is_s === false
    # the options reach the solver through simulate and the mod-file route
    S = reshape(vec(ho_ref("shocks.mat")["ShockSequence"]), :, 1)[1:20, :]
    kw = (obc_method=:news, obc_order=1, obc_cubature=true, obc_qmc_level=3, obc_horizon=32, check_ahead_periods=32, periods=20)
    @test JDyn.simulate(ob, S; obc_m_diagnostic=false, kw...).piecewise == JDyn.simulate(ob, S; kw...).piecewise
    @test_throws ArgumentError JDyn.simulate(ob, S; obc_m_diagnostic="maybe", kw...)
    @test_throws ArgumentError JDyn.simulate(ob, S; obc_cubature_failure_tolerance=1.0, kw...)
    @test_throws ArgumentError JDyn.simulate(ob, S; obc_cubature_milp_time_limit=-1, kw...)
    @test_throws ArgumentError JDyn.simulate(ob, S; obc_method=:news, obc_order=2, obc_cubature=false, obc_cubature_failure_tolerance=0.1, obc_horizon=32, check_ahead_periods=32)
end

@testset "failed cubature points: classes, raise by default, tolerance mode" begin
    mb = load_copy(joinpath(HO_DIR, "occ_borrcon.mod"))
    obb = JDyn.Occbin(mb)
    Sb = randn(MersenneTwister(4), 30, 2) .* sqrt.(diag(Matrix(mb.Sigma_e)))' .* 0.1
    mk(; kw...) = JDyn.obc_ho(obb; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=5, m_diagnostic=false, kw...)
    # default: the first failed point raises, with the payload; class (b): the check-ahead rows
    err = try; JDyn.obc_ho_simulate(mk(), Sb); nothing; catch e; e; end
    @test err isa JDyn.ObcError && err.detail isa JDyn.CubatureFailure
    d = err.detail
    @test d.n_points == 63 && 0 < d.n_relevant <= 63 && d.lower_bound && d.n_failed >= 1 && d.period >= 1 && err.period == d.period
    @test d.counts.a + d.counts.b + d.counts.c == d.n_failed && d.counts.b >= 1 && d.status == :horizon_too_short == err.status
    @test d.n_active_set_failed >= d.n_failed && any(occursin("longer obc_horizon", a) for a in d.advice) && d.tolerance == 0.0
    # the advice works here: with a longer horizon no point fails
    @test size(JDyn.obc_ho_simulate(JDyn.obc_ho(obb; order=2, horizon=96, check_ahead=96, cubature=true, qmc_level=5, m_diagnostic=false), Sb[1:10, :]).wedges) == (10, 96)
    # a model with points that have no solution at all (the SW-type stand-in): class (a)
    sw = sw_cp_model()
    Ssw = randn(MersenneTwister(4), 30, size(sw.Sigma_e, 1)) .* sqrt.(diag(Matrix(sw.Sigma_e)))' .* 0.25
    h2 = JDyn.obc_ho(JDyn.Occbin(sw); order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=5, m_diagnostic=false)
    e2 = try; JDyn.obc_ho_simulate(h2, Ssw); nothing; catch e; e; end
    @test e2 isa JDyn.ObcError && e2.detail.counts.a >= 1 && e2.status == :no_solution && e2.detail.period >= 1
    # tolerance mode: failed points leave the sample; one warning per simulation; weights 1/(points - failed)
    h = mk(cubature_failure_tolerance=0.9)
    h.cub.qlog = Vector{Float64}[]
    r = @test_logs (:warn, r"were dropped") JDyn.obc_ho_simulate(h, Sb[1:12, :])
    @test sum(r.dropped) > 0 && sum(r.dropped) == r.stats[:dropped_points] && all(r.dropped .<= 0.9 .* r.relevant) && maximum(r.relevant) <= 63
    qs = h.cub.qlog; h.cub.qlog = nothing
    checked = 0
    for q in qs
        h.cub.tolerance = 0.9
        y = with_logger(NullLogger()) do; JDyn.ho_news_shocks(h, q, 0); end
        nf = h.cub.dropped_last
        nf > 0 || continue
        # by hand: the relevant points, each classified or solved
        N, T = h.N, h.T
        _ho_prepare = JDyn._ho_prepare_points!(h)
        acc = zeros(T); nfh = 0
        for j in 1:size(h.xs, 2)
            p = q .+ h.R * h.xs[:, j]
            minimum(p) < -1e-8 * max(1.0, maximum(abs, p)) || continue
            yy, s, _ = JDyn.solve_active_set(view(h.Mf, 1:T, :), p[1:T]; tol=h.tol)
            if s == :ok && minimum(p[T+1:N] .+ h.Mb * yy) >= -h.tol
                acc .+= yy
            else
                y2, cls = JDyn._cub_classify(h, p)
                cls === nothing ? (acc .+= y2) : (nfh += 1)
            end
        end
        @test nfh == nf
        @test maximum(abs, y .- acc ./ (63 - nf)) < 1e-11
        checked += 1
    end
    @test checked >= 2
    # more failures than the tolerance: raise, with the exceeded count as a lower bound
    e4 = try; with_logger(NullLogger()) do; JDyn.obc_ho_simulate(mk(cubature_failure_tolerance=0.02), Sb); end; nothing; catch e; e; end
    @test e4 isa JDyn.ObcError && e4.detail.tolerance == 0.02 && e4.detail.n_failed > 0.02 * e4.detail.n_relevant
    @test_throws ArgumentError mk(cubature_failure_tolerance=1.0)
    @test_throws ArgumentError mk(cubature_milp_time_limit=-1)
    # the MILP step: with LPs that cannot prove anything the integer programme decides; a time limit gives the undecided class (c)
    hc = mk()
    h.cub.qlog = nothing
    hc.cub.qlog = Vector{Float64}[]
    try; with_logger(NullLogger()) do; JDyn.obc_ho_simulate(hc, Sb[1:3, :]); end; catch; end
    q = hc.cub.qlog[1]; hc.cub.qlog = nothing
    JDyn._ho_prepare_points!(hc)
    T, N = hc.T, hc.N
    pbad = nothing
    for j in 1:size(hc.xs, 2)
        p = q .+ hc.R * hc.xs[:, j]
        minimum(p) < -1e-8 * max(1.0, maximum(abs, p)) || continue
        yy, s, _ = JDyn.solve_active_set(view(hc.Mf, 1:T, :), p[1:T]; tol=hc.tol)
        (s == :ok && minimum(p[T+1:N] .+ hc.Mb * yy) >= -hc.tol) || (pbad = p; break)
    end
    @test pbad !== nothing
    @test JDyn._cub_classify(hc, pbad) == (nothing, :b)                      # the LP proof
    hc.cub.lp_head = JDyn.CubLP(fill(1e3, T, T)); hc.cub.lp_full = JDyn.CubLP(fill(1e3, N, T))     # LPs that are always feasible: reach the MILP
    hc.cub.milp_time_limit = 1e-6
    t = @elapsed cls = JDyn._cub_classify(hc, pbad)
    @test cls == (nothing, :c) && t < 0.5
    hc.cub.milp_time_limit = 30.0
    @test JDyn._cub_classify(hc, pbad) == (nothing, :b)                      # the MILP on the head rows alone finds a solution: (b)
    hc.cub.milp_time_limit = 1e-6
    @test_throws JDyn.ObcError JDyn.ho_news_shocks(hc, q, 1)
    e5 = try; JDyn.ho_news_shocks(hc, q, 1); nothing; catch e; e; end
    @test e5.detail.counts.c >= 1 && e5.status == :not_converged
    # one region: the unit is the region's mean point; a failed region counts all its points (a failing mean point is rare, so call the unit solver on a failing point)
    h1 = mk(regions=1)
    e6 = try; JDyn._cub_one!(h1, pbad, 10, 63, 3); nothing; catch e; e; end
    @test e6 isa JDyn.ObcError && e6.detail.n_failed == 10 == e6.detail.n_relevant && !e6.detail.lower_bound && e6.detail.n_active_set_failed === nothing && e6.detail.period == 3
end

@testset "two constraints at higher order (obc_zlb2.mod: the bound on the rate and a floor on the second variable)" begin
    m = load_copy(joinpath(@__DIR__, "obc", "obc_zlb2.mod"))
    ob = JDyn.Occbin(m)
    S = reshape(randn(MersenneTwister(1), 100) .* 0.05, :, 1)
    kw = (obc_method=:news, obc_horizon=24, check_ahead_periods=24, periods=100)
    r0 = JDyn.simulate(ob, S; kw...)                                              # JDyn's own order-1 news simulation (the loop of src/obc.jl)
    h1 = JDyn.obc_ho(ob; order=1, horizon=24, check_ahead=24)
    @test h1.nc == 2 && h1.nT == 48 && h1.NR == 96 && size(h1.Mf) == (96, 48) && size(h1.Gq) == (96, 1)
    o1 = JDyn.obc_ho_simulate(h1, S)
    @test maximum(abs, o1.with_bound .- r0.piecewise) < 1e-12
    @test maximum(abs, o1.wedges .- r0.news.wedges) < 1e-12
    @test count(t -> maximum(o1.wedges[t, 1:24]) > 1e-9 && maximum(o1.wedges[t, 25:48]) > 1e-9, 1:100) > 20      # both bind together in many periods
    # the same through simulate(): history per constraint, wedges 2 T per period
    ro = JDyn.simulate(ob, S; obc_order=1, obc_cubature=false, kw...)
    @test size(ro.news.wedges) == (100, 48) && all(length(rh) == 2 for rh in ro.regime_history) && maximum(abs, ro.piecewise .- r0.piecewise) < 1e-12
    # the affine map of q equals the recursion followed by the gap definition, for both constraints
    h2 = JDyn.obc_ho(ob; order=2, horizon=24, check_ahead=24)
    rng = MersenneTwister(3)
    for _ in 1:3
        y1, y2, o, prev, u = 0.01 .* randn(rng, h2.n), 1e-3 .* randn(rng, h2.n), 1e-3 .* randn(rng, h2.n), 1e-2 .* randn(rng, h2.n), randn(rng, 1)
        qd = JDyn.ho_gap_path(h2, JDyn.ho_expected_path(h2, y1, y2, o), prev, u)
        q = h2.qc .+ h2.Q1 * y1 .+ h2.Qo * o .+ h2.Q2 * y2 .+ h2.Qw * kron(y1[h2.sv], y1[h2.sv])
        for c in 1:2
            q[h2.rows[1, c]] += (h2.aC[c, :]' * prev + h2.aJ[c, :]' * u)
        end
        @test maximum(abs, q .- qd) < 1e-12
    end
    # cubature, batch against scalar: the three modes give the same simulation, no failed point
    sims = Dict{Symbol,Any}()
    for md in (:scalar, :cache, :group)
        hh = JDyn.obc_ho(ob; order=2, horizon=24, check_ahead=24, cubature=true, qmc_level=6, regions=0, m_diagnostic=false)
        hh.cub.mode = md
        sims[md] = JDyn.obc_ho_simulate(hh, S[1:30, :])
    end
    @test sum(sims[:scalar].relevant) > 500 && sum(sims[:scalar].dropped) == 0
    @test maximum(abs, sims[:cache].wedges .- sims[:scalar].wedges) < 1e-12 && maximum(abs, sims[:group].wedges .- sims[:scalar].wedges) < 1e-12
    # and the full average equals the loop over the relevant points of a gap path with both constraints in play
    hh = JDyn.obc_ho(ob; order=2, horizon=24, check_ahead=24, cubature=true, qmc_level=6, regions=0, m_diagnostic=false)
    qs = Vector{Float64}[]; hh.cub.qlog = qs
    JDyn.obc_ho_simulate(hh, S[1:30, :]); hh.cub.qlog = nothing
    JDyn._ho_prepare_points!(hh)
    nchk = 0
    for q in qs[10:20]
        y = JDyn.ho_news_shocks(hh, q, 0)
        acc = zeros(hh.nT); np = size(hh.xi, 2)
        for j in 1:np
            p = q .+ hh.R * hh.xi[:, j]
            minimum(p) < -1e-8 * max(1.0, maximum(abs, p)) || continue
            acc .+= JDyn._ho_lcp(hh, p, 0)
        end
        @test maximum(abs, y .- acc ./ np) < 1e-12
        nchk += 1
    end
    @test nchk == 11
    # one region with two constraints
    h1r = JDyn.obc_ho(ob; order=2, horizon=24, check_ahead=24, cubature=true, qmc_level=6, regions=1, m_diagnostic=false)
    @test size(JDyn.obc_ho_simulate(h1r, S[1:30, :]).wedges) == (30, 48)
end
