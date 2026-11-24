# obc_method = news at order 2 and with cubature (src/obc_ho.jl, src/obc_qmc.jl; docs/OBC_HO_PROGRESS.md), against the real DynareOBC
# (run under Octave 8.4 with Dynare 4.5.4; recipe and reference files in pydyn's reference/dynareobc_octave) and against pydyn's prototype
# (pydyn/obc_ho.py): a small NK model with a lower bound on the interest rate, 300 shock draws, 83 periods at the bound.
using Test; include("common.jl")
using Random

const HO_DIR = joinpath(@__DIR__, "obc", "ho")
ho_ref(name) = matread(joinpath(HO_DIR, name))

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
end
