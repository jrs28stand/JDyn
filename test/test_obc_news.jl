# obc_method = news (src/highs.jl, src/obc_solvers.jl `solve_news`, `m_diagnostics`, src/obc_regime.jl): the HiGHS wrapper, the solvers
# and diagnostics on small problems with known answers, `news` against JDyn's OccBin on every complementarity model (with the active-set
# fast path and forced through the integer programme), the reference file test/obc/obc_zlb.mat, the general regime formulation on
# Dynare's multi-equation and two-constraint files, and the Smets-Wouters benchmark (tied wedges, multiple equilibria, no solution).
# Mirrors pydyn's tests/test_obc.py for what is ported (docs/OBC_PROGRESS.md, "State (news port)").
using Test; include("common.jl")
using Random

const OBC_DIR = joinpath(@__DIR__, "obc")
const NEWS_TOL = 1e-11            # news vs OccBin paths (levels)

"Load a copy of the file's whole folder (include files)."
function load_dir(path; kw...)
    d = mktempdir()
    for f in readdir(dirname(path))
        p = joinpath(dirname(path), f)
        isfile(p) && cp(p, joinpath(d, f))
    end
    with_logger(NullLogger()) do
        JDyn.load(joinpath(d, basename(path)); kw...)
    end
end

nbinding(r) = count(per -> any(x -> any(!=(0), x[1]), per), r.regime_history)
shock_sd(m, scale) = scale == 0 ? fill(0.02, m.n_exo) : scale .* [s < 1e-9 ? 0.01 : s for s in sqrt.(diag(m.Sigma_e))]

# ------------------------------------------------------------------ 1. the HiGHS wrapper
@testset "HiGHS wrapper (src/highs.jl)" begin
    # max x + y s.t. x + 2y <= 4, 0 <= x, y <= 3: LP optimum (3, 0.5)
    st, x = JDyn.highs_lp([-1.0, -1.0], [1.0 2.0], [-Inf], [4.0], [0.0, 0.0], [3.0, 3.0])
    @test st == :optimal && x ≈ [3.0, 0.5]
    # infeasible: x >= 5 with x <= 1
    st, x = JDyn.highs_lp([1.0], [1.0;;], [5.0], [Inf], [0.0], [1.0])
    @test st == :infeasible && isempty(x)
    # unbounded: min -x, x >= 0
    st, x = JDyn.highs_lp([-1.0], [1.0;;], [0.0], [Inf], [0.0], [Inf])
    @test st in (:unbounded, :infeasible_or_unbounded)
    # equality rows and free variables: min x s.t. x - y = 2, y = 3, x, y free
    st, x = JDyn.highs_lp([1.0, 0.0], [1.0 -1.0; 0.0 1.0], [2.0, 3.0], [2.0, 3.0], [-Inf, -Inf], [Inf, Inf])
    @test st == :optimal && x ≈ [5.0, 3.0]
    # integer: max x + y s.t. 2x + 2y <= 5 (LP optimum 2.5, integer optimum 2)
    st, x = JDyn.highs_milp([-1.0, -1.0], [2.0 2.0], [-Inf], [5.0], [0.0, 0.0], [10.0, 10.0], [true, true])
    @test st == :optimal && x[1] + x[2] ≈ 2 && all(v -> abs(v - round(v)) < 1e-9, x)
    # mixed: x integer, y continuous: max x + y s.t. x + y <= 2.5, y <= 0.7
    st, x = JDyn.highs_milp([-1.0, -1.0], [1.0 1.0], [-Inf], [2.5], [0.0, 0.0], [10.0, 0.7], [true, false]; time_limit=20.0, options=(mip_rel_gap=0.0,))
    @test st == :optimal && x[1] + x[2] ≈ 2.5 && abs(x[1] - round(x[1])) < 1e-9
    # integer infeasible: 2x = 1 with x integer in [0, 1]
    st, x = JDyn.highs_milp([0.0], [2.0;;], [1.0], [1.0], [0.0], [1.0], [true])
    @test st == :infeasible && isempty(x)
end

# ------------------------------------------------------------------ 2. solvers and diagnostics on small problems
@testset "news solver and diagnostics on small problems" begin
    # M = -1, q = -1: y >= 0 makes the gap more negative, no solution (and the linear programme proves it)
    y, st, info = solve_news([-1.0;;], [-1.0])
    @test st == :no_solution && info.proof
    # M = [1 2; 2 1], q = (-1, -1): three solutions, one is returned and verified; not a P-matrix, but an S-matrix
    M, q = [1.0 2.0; 2.0 1.0], [-1.0, -1.0]
    y, st, info = solve_news(M, q)
    @test st == :ok && obc_verify(M, q, y)[1] && info.polished && info.max_violation <= 1e-10
    d = m_diagnostics(M)
    @test d.is_p === false && d.is_s === true
    # the identity is P (and S); [-1] is neither; [1 -3; 0 1] is P without a positive definite symmetric part
    @test m_diagnostics(Matrix(1.0I, 3, 3)).is_p === true && m_diagnostics(Matrix(1.0I, 3, 3)).is_s === true
    d = m_diagnostics([-1.0;;])
    @test d.is_p === false && d.is_s === false
    d = m_diagnostics([1.0 -3.0; 0.0 1.0])
    @test d.is_p === true && d.symmetric_part_positive_definite === false && d.largest_p == 2
    # the Tsatsomeros-Li recursion agrees with brute-force principal minors on random small matrices (both outcomes occur)
    rng = MersenneTwister(11)
    seen = Set{Bool}()
    for _ in 1:300
        n = rand(rng, 1:6)
        A = Matrix(1.0I, n, n) .* (0.2 + 1.8 * rand(rng)) .+ 0.45 .* randn(rng, n, n)
        r = JDyn._is_p_exact(A)
        @test r == JDyn._is_p_minors(A)
        push!(seen, r)
    end
    @test seen == Set([true, false])
    # a larger matrix: the diagnostics' exactness limit
    A = Matrix(1.0I, 20, 20) .+ 0.01 .* randn(rng, 20, 20)
    d = m_diagnostics(A; max_exact=8)
    @test d.is_p === true && d.symmetric_part_positive_definite        # decided by the sufficient condition above max_exact
    # shortcut, and the pure integer programme against the active set and minnorm on P-matrix problems
    @test solve_news([1.0;;], [0.5])[1] == [0.0] && solve_news([1.0;;], [0.5])[2] == :ok
    rng = MersenneTwister(3)
    for k in 1:25
        n = rand(rng, 2:12)
        A = randn(rng, n, n); S = randn(rng, n, n)
        M = A * A' + 0.1I + (S - S')
        q = randn(rng, n)
        y1, s1, i1 = solve_news(M, q)
        y2, s2, _ = solve_active_set(M, q)
        y3, s3, _ = solve_minnorm(M, q)
        @test s1 == :ok && s3 == :ok && obc_verify(M, q, y1)[1] && maximum(abs.(y1 - y3)) < 1e-9
        s2 == :ok && @test maximum(abs.(y1 - y2)) < 1e-9
    end
    # rows beyond the horizon: M = [1], q = -1 gives y = 1; the gap beyond the horizon -0.5 + ... fails: horizon too short, not no solution
    y, st, info = solve_news([1.0;;], [-1.0]; Mb=[-1.0;;], qb=[0.5])
    @test st == :horizon_too_short && y ≈ [1.0]
    y, st, _ = solve_news([1.0;;], [-1.0]; Mb=[1.0;;], qb=[0.5])
    @test st == :ok && y ≈ [1.0]
    # a time limit is accepted
    @test solve_news(M, q; time_limit=30.0)[2] == :ok
end

# ------------------------------------------------------------------ 3. news = OccBin on every complementarity model
const NEWS_MODELS = [
    (joinpath(OBC_DIR, "nk_zlb.mod"), 1.0),
    (joinpath(OBC_DIR, "nk_zlb2.mod"), 1.0),
    (joinpath(OBC_DIR, "obc_zlb.mod"), 2.0),
    (joinpath(OBC_DIR, "obc_zlb2.mod"), 2.0),
    (joinpath(DYNARE_SRC, "tests", "occbin", "model_borrcon", "borrcon.mod"), 1.0),
    (joinpath(DYNARE_SRC, "tests", "occbin", "model_borrcon", "borrcon_0_std_shocks.mod"), 0.0),
    (joinpath(DYNARE_SRC, "tests", "occbin", "filter", "NKM_0_std_shocks.mod"), 3.0),
    (joinpath(REF, "borrcon_sim.mod"), 1.0),
    (joinpath(REF, "borrcon_curb.mod"), 1.0),
    (joinpath(REF, "occ_borrcon.mod"), 1.0),
]

@testset "news = OccBin ($(basename(path)))" for (path, scale) in NEWS_MODELS
    m = load_dir(path); ob = Occbin(m)
    has_surprise = any(st -> get(st, "surprise", false) == true, get(m.options, "shock_paths", Any[]))
    @test all(c -> c.class == :complementarity, obc_constraints(ob))
    is_nkm = occursin("NKM", basename(path))             # the approximate class: a knife-edge regime difference is a known property
    for fast in (true, false)                            # with the active-set fast path, and forced through the integer programme
        if has_surprise
            a = simulate(ob)
            b = simulate(ob; obc_method=:news, obc_fast_path=fast)
            @test isapprox(b.piecewise, a.piecewise; atol=NEWS_TOL, rtol=0) && b.regime_history == a.regime_history
            @test b.news.method == :news && b.news.formulation == :lcp && all(==(:ok), b.news.status)
            @test b.news.diagnostics.is_s isa Bool
            fast && println(rpad(basename(path), 26), "file shocks: max |news - occbin| = ", maximum(abs.(b.piecewise - a.piecewise)), ", periods binding: ", nbinding(a))
        end
        sh = randn(MersenneTwister(20261124), 200, ob.n_exo) .* shock_sd(m, scale)'
        a = simulate(ob, sh; periods=200)
        b = simulate(ob, sh; periods=200, obc_method=:news, obc_fast_path=fast)
        @test nbinding(a) > 0
        d = maximum(abs.(b.piecewise - a.piecewise))
        diffp = [t for t in eachindex(a.regime_history) if a.regime_history[t] != b.regime_history[t]]
        fast && println(rpad(basename(path), 26), "random 200: max |news - occbin| = ", d, ", periods binding: ", nbinding(a),
                        isempty(diffp) ? "" : ", periods with a different regime history: $diffp")
        @test d <= NEWS_TOL && isempty(diffp)
        fast || @test all(i -> get(i, :polished, true) && all(==(:ok), b.news.status), b.news.info)      # every integer-programme point was polished and verified
    end
end

# ------------------------------------------------------------------ 4. the reference file (Dynare 7.2 + GLPK)
@testset "news against test/obc/obc_zlb.mat" begin
    ref = matread(joinpath(OBC_DIR, "obc_zlb.mat"))
    T = Int(ref["T"])
    m = load_dir(joinpath(OBC_DIR, "obc_zlb.mod")); ob = Occbin(m)
    sys = obc_system(ob; horizon=T)
    ext = JDyn._obc_ext(sys, Int(ref["N"]))
    Mfull = vcat(sys.M, ext.Mb)
    @test maximum(abs.(Mfull - ref["M"])) < 1e-12
    println("obc_zlb M (horizon $T, ", size(Mfull, 1), " rows) vs obc_zlb.mat: max abs difference ", maximum(abs.(Mfull - ref["M"])))
    xin = vcat(zeros(ob.n), [ref["shock"]])
    q = obc_q(sys, zeros(ob.n), [ref["shock"]])
    qb = ext.q0b .+ ext.Qxb * xin
    @test maximum(abs.(vcat(q, qb) - vec(ref["q"]))) < 1e-12
    y, st, _ = solve_news(sys.M, q; Mb=ext.Mb, qb=qb)
    @test st == :ok && maximum(abs.(y - vec(ref["y"]))) < 1e-9
    println("obc_zlb wedges (news, integer programme) vs GLPK: ", maximum(abs.(y - vec(ref["y"]))))
    for fast in (true, false)
        r = simulate(ob; obc_method=:news, obc_horizon=T, obc_fast_path=fast, check_ahead_periods=Int(ref["N"]) - T)
        @test maximum(abs.(r.piecewise[:, m.ie["r"]] - vec(ref["r_occbin"]))) < 1e-12
        @test maximum(abs.(r.news.wedges[1, :] - vec(ref["y"]))) < 1e-9
        @test r.regime_history[1][1][2] == Int.(vec(ref["regimestart"]))
        fast || println("obc_zlb path (news, pure integer programme) vs Dynare's OccBin: ", maximum(abs.(r.piecewise[:, m.ie["r"]] - vec(ref["r_occbin"]))))
    end
end

# ------------------------------------------------------------------ 5. the regime formulation
@testset "regime formulation: R equals M, equals LCP and OccBin" begin
    for (path, T) in ((joinpath(OBC_DIR, "nk_zlb.mod"), 30), (joinpath(OBC_DIR, "nk_zlb2.mod"), 25), (joinpath(OBC_DIR, "obc_zlb2.mod"), 20),
                      (joinpath(DYNARE_SRC, "tests", "occbin", "model_borrcon", "borrcon.mod"), 30))
        m = load_dir(path); ob = Occbin(m)
        sys = obc_system(ob; horizon=T)
        rg = RegimeOBC(ob; horizon=T, check_ahead=T)
        nc = length(sys.cons)
        sgy = kron(Diagonal([c.s_g for c in sys.cons]), I(T)) * rg.R * kron(Diagonal([c.s_y for c in sys.cons]), I(T))
        @test maximum(abs.(sgy - sys.M)) < 1e-10               # R is the response of the alternative residuals to unit wedges: M up to the signs
    end
    # equal to the LCP formulation and to OccBin on the one- and two-constraint complementarity models
    for (path, scale) in ((joinpath(OBC_DIR, "obc_zlb.mod"), 2.0), (joinpath(OBC_DIR, "obc_zlb2.mod"), 2.0), (joinpath(OBC_DIR, "nk_zlb2.mod"), 1.0))
        m = load_dir(path); ob = Occbin(m)
        sh = randn(MersenneTwister(5), 150, ob.n_exo) .* shock_sd(m, scale)'
        a = simulate(ob, sh; periods=150)
        b = simulate(ob, sh; periods=150, obc_method=:news, obc_solver_options=(formulation=:regime,))
        c = simulate(ob, sh; periods=150, obc_method=:news)
        @test b.news.formulation == :regime && c.news.formulation == :lcp
        @test maximum(abs.(b.piecewise - a.piecewise)) < NEWS_TOL && b.regime_history == a.regime_history
        @test maximum(abs.(b.piecewise - c.piecewise)) < NEWS_TOL
        @test nbinding(a) > 10
    end
    # the integer programme over regime indicators on its own equals the guess-and-verify answer (one constraint)
    m = load_dir(joinpath(OBC_DIR, "obc_zlb.mod")); ob = Occbin(m)
    rg = RegimeOBC(ob; horizon=20, check_ahead=40)
    init, u = zeros(ob.n), [-0.05]
    st1, z1, w1, path1, i1 = JDyn.regime_solve_period(rg, init, u; method=:milp)
    st2, z2, w2, path2, i2 = JDyn.regime_solve_period(rg, init, u)
    @test st1 == :ok && st2 == :ok && i1.used == :milp && i2.used == :guess_verify
    @test z1 == z2 && maximum(abs.(path1 - path2)) < 1e-12
end

@testset "regime formulation on Dynare's multi-equation files" begin
    for rel in ("tests/occbin/ppf/bf_ppf.mod", "tests/estimation/slice/bf.mod")
        m = load_dir(joinpath(DYNARE_SRC, rel)); ob = Occbin(m)
        @test [c.class for c in obc_constraints(ob)] == [:multi_equation]
        sh = randn(MersenneTwister(7), 200, ob.n_exo) .* shock_sd(m, 2.0)'
        a = simulate(ob, sh; periods=200)
        b = simulate(ob, sh; periods=200, obc_method=:news, obc_horizon=40)
        d = maximum(abs.(b.piecewise - a.piecewise))
        @test d < NEWS_TOL && b.regime_history == a.regime_history
        @test b.news.formulation == :regime && count(i -> i.used == :guess_verify, b.news.info) > 20
        @test nbinding(a) > 20
        println(rpad(basename(rel), 12), "regime formulation, random 200: max |news - occbin| = ", d, ", periods binding: ", nbinding(a))
        # the forced integer programme over regime indicators (obc_fast_path = false) on a few periods
        c = simulate(ob, sh[1:8, :]; periods=8, obc_method=:news, obc_horizon=40, obc_fast_path=false)
        a8 = simulate(ob, sh[1:8, :]; periods=8)
        @test maximum(abs.(c.piecewise - a8.piecewise)) < 1e-9 && c.regime_history == a8.regime_history
        # minnorm keeps refusing them: its objective is defined on the LCP
        @test_throws ModelError simulate(ob, sh[1:2, :]; obc_method=:minnorm)
        # the tied-wedge diagnostics are written to oo_.occbin.news when the tie is found
        s = JDyn._news_struct(b.news)
        @test s.formulation == "regime" && s.method == "news"
    end
end

@testset "regime formulation: Dynare's rbc_occbin (two constraints, 169 shock periods)" begin
    m = load_dir(joinpath(DYNARE_SRC, "examples", "occbin", "rbc_occbin.mod")); ob = Occbin(m)
    Sh, pos = surprise_shocks(ob)
    @test size(Sh, 1) == 169
    a = simulate(ob, Sh; periods=200, check_ahead_periods=200, exo_pos=pos)
    b = simulate(ob, Sh; periods=200, check_ahead_periods=70, obc_method=:news, obc_horizon=130, exo_pos=pos)
    d = maximum(abs.(b.piecewise - a.piecewise))
    @test d < NEWS_TOL && b.regime_history == a.regime_history && b.news.formulation == :regime
    println("rbc_occbin regime formulation (horizon 130): max |news - occbin| = ", d, ", periods binding: ", nbinding(a))
    err = try simulate(ob, Sh; periods=200, obc_method=:news, obc_horizon=40, exo_pos=pos); nothing catch e; e end
    @test err isa ObcError && err.status == :horizon_too_short && err.period == 1
    r = simulate(ob, Sh; periods=200, obc_method=:news, obc_horizon=40, exo_pos=pos, obc_raise=false)
    @test r.news.status[1] == :horizon_too_short
end

# ------------------------------------------------------------------ 6. Smets-Wouters with a lower bound whose rule also switches its lag term
@testset "Smets-Wouters benchmark: tied wedges, multiple equilibria, no solution" begin
    m = load_dir(joinpath(OBC_DIR, "obc_sw_zlb.mod")); ob = Occbin(m)
    rg = RegimeOBC(ob; horizon=40, check_ahead=60)
    tie = JDyn._tie(rg)
    @test tie !== nothing && tie.mirror && length(tie.secondary) == 1
    @test isapprox(tie.d, [1.0, -1.0]; atol=1e-10)
    d = obc_diagnostics(rg; max_exact=12)
    @test d.is_s === false && d.is_p === false && d.largest_p == 10
    Mt = (tie.s_g * tie.s_y) .* tie.Mraw[1:40, :]
    println("sw LCP M (horizon 40): diagonal range ", extrema(diag(Mt)), ", is_s ", d.is_s, ", is_p ", d.is_p, ", largest_p ", d.largest_p)
    # -4 on eb from the steady state: exactly two single-spell solutions; OccBin and news return the 9-period one
    S = zeros(m.n_exo); S[m.ix["eb"]] = -4.0
    sols = single_spell_solutions(rg, zeros(ob.n), S)
    @test sols == [(0, 9), (0, 13)]
    x = vcat(zeros(ob.n), S)
    a0 = rg.a0 .+ rg.Am * x
    paths = []
    for (l, k) in sols
        z = falses(rg.N, 1); z[l+1:l+k, 1] .= true
        w = JDyn._regime_wedges(rg, z, a0)
        pcv = JDyn._regime_pcv(rg, x, w)
        vb, vr = JDyn._regime_violations(rg, z, pcv)
        @test !any(vb) && !any(vr)                          # a solution by OccBin's own test
        push!(paths, JDyn._wedge_path(ob.P, rg.G, rg.FE, ob.P * zeros(ob.n) .+ ob.Q * S, w, rg.T, 29))
    end
    @test maximum(abs.(paths[1] - paths[2])) > 1.0          # far apart
    aocc = simulate(ob, reshape(S, 1, :); periods=60, check_ahead_periods=100)
    bnews = simulate(ob, reshape(S, 1, :); periods=60, check_ahead_periods=60, obc_method=:news)
    @test aocc.regime_history[1] == [([1, 0], [1, 10])] == bnews.regime_history[1]
    @test maximum(abs.(bnews.piecewise - aocc.piecewise)) < 1e-9
    println("sw eb = -4: single-spell solutions ", sols, "; news vs OccBin max path difference ", maximum(abs.(bnews.piecewise - aocc.piecewise)))
    # -6: OccBin fails (error 313); news proves that no solution exists (infeasible linear programme)
    S6 = zeros(1, m.n_exo); S6[1, m.ix["eb"]] = -6.0
    e1 = try simulate(ob, S6; periods=60, check_ahead_periods=100); nothing catch e; e end
    @test e1 isa OccbinError && JDyn.error_flag(e1) == 313
    e2 = try simulate(ob, S6; periods=60, check_ahead_periods=60, obc_method=:news); nothing catch e; e end
    @test e2 isa ObcError && e2.status == :no_solution && occursin("linear programme infeasible", e2.msg)
    @test isempty(single_spell_solutions(rg, zeros(ob.n), S6[1, :]))
    # eb = -3 and em = 0.5: the offsets G are non-zero; the solution found passes OccBin's test
    S3 = zeros(m.n_exo); S3[m.ix["eb"]] = -3.0; S3[m.ix["em"]] = 0.5
    Mq, q, G, info = obc_lcp(rg, zeros(ob.n), S3)
    @test G !== nothing && maximum(abs, G) > 1e-3
    @test length(single_spell_solutions(rg, zeros(ob.n), S3)) >= 1
    z, w, pcv = JDyn._solve_lcp(rg, zeros(ob.n), S3)
    vb, vr = JDyn._regime_violations(rg, z, pcv)
    @test !any(vb) && !any(vr)
    r3 = simulate(ob, reshape(S3, 1, :); periods=60, check_ahead_periods=60, obc_method=:news)
    @test r3.news.status == [:ok]
end

# ------------------------------------------------------------------ 7. interface: oo_.occbin.news, hook, defaults
@testset "news: results struct, timing hook, refusals, defaults" begin
    m = load_dir(joinpath(OBC_DIR, "obc_zlb.mod")); ob = Occbin(m)
    r = simulate(ob; obc_method=:news)
    s = JDyn._news_struct(r.news)
    @test s.method == "news" && s.formulation == "lcp" && s.diagnostics.is_s === true && s.diagnostics.largest_p >= 1
    @test s.diagnostics.is_p isa Union{Bool,String}
    d = mktempdir(); cp(joinpath(OBC_DIR, "obc_zlb.mod"), joinpath(d, "obc_zlb.mod"))
    ws = with_logger(NullLogger()) do; dynare(joinpath(d, "obc_zlb.mod")); end
    occbin_solver!(ws; obc_method="news")
    @test ws.oo_.occbin.news.method == "news" && ws.oo_.occbin.news.formulation == "lcp" && haskey(ws.oo_.occbin.news.diagnostics, "is_s")
    # the diagnostics as a function of the model
    dg = obc_diagnostics(ob; horizon=20)
    @test dg.is_s === true && dg.largest_p >= 1
    # the debug timing hook still works (bench/obc_bench.jl)
    ticks = UInt64[]
    JDyn.OBC_TICKS[] = ticks
    try
        simulate(ob, randn(MersenneTwister(1), 5, ob.n_exo) .* 0.02; periods=5, obc_method=:news)
    finally
        JDyn.OBC_TICKS[] = nothing
    end
    @test length(ticks) == 6
    # unknown formulation is not an error for the LCP; unknown method is
    @test_throws ArgumentError simulate(ob; obc_method=:lcp)
    # horizon too short: the bound binds for 7 periods, the horizon is 3 (both with the fast path and forced)
    for fast in (true, false)
        err = try simulate(ob; obc_method=:news, obc_horizon=3, obc_fast_path=fast); nothing catch e; e end
        @test err isa ObcError && err.status == :horizon_too_short && err.period == 1
    end
    # default behaviour: no news results, OccBin untouched
    @test simulate(ob).news === nothing
end
