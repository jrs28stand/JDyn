# obc_method (src/obc.jl, src/obc_solvers.jl): the post-processor's classification of Dynare 7.2's OccBin files, M and q
# against a stacked-system computation and against Dynare 7.2 (test/obc/obc_zlb.mat, made by test/obc/r_obc_zlb.m),
# the solvers on small LCPs, and `minnorm` against JDyn's own OccBin on every complementarity model (the file's shocks
# and a seeded 200-period random simulation), refusals and the interface (keywords, native lines, oo_.occbin.news).
using Test; include("common.jl")
include(joinpath(@__DIR__, "obc", "classify_dynare.jl"))
using Random

const OBC_DIR = joinpath(@__DIR__, "obc")
const PATH_TOL = 1e-12            # minnorm vs OccBin paths (levels): solver tolerance; both solve the same linear systems

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

"Gaps over T periods from the stacked perfect-foresight system (independent of src/obc.jl's recursion)."
function stacked_gaps(ob, cons, T, w, init, u)
    n, nx = ob.n, ob.n_exo
    C, B, A, J, _ = JDyn.regime_matrices(ob, ntuple(_ -> false, ob.k))
    N = n * T
    K = zeros(N, N); rhs = zeros(N)
    for t in 1:T
        r = (t-1)*n+1:t*n
        K[r, r] = B
        t > 1 && (K[r, (t-2)*n+1:(t-1)*n] = C)
        t < T ? (K[r, t*n+1:(t+1)*n] = A) : (K[r, r] += A * ob.P)      # y_{T+1} = P y_T: no wedges after T
        t == 1 && (rhs[r] -= C * init + J * u)
        for (j, c) in enumerate(cons)
            rhs[r[c.equations[1]]] += c.s_y * w[(j-1)*T+t]                # ref_resid = s_y * y
        end
    end
    Y = reshape(K \ rhs, n, T)
    g = zeros(length(cons) * T)
    for (j, c) in enumerate(cons), t in 1:T
        prev = t == 1 ? init : Y[:, t-1]
        nxt = t < T ? Y[:, t+1] : ob.P * Y[:, T]
        a = c.alt_resid[1]
        g[(j-1)*T+t] = c.s_g * (a.c + dot(a.g, vcat(prev, Y[:, t], nxt, t == 1 ? u : zeros(nx))))
    end
    g
end

"Binding regime in some period of a simulation."
nbinding(r) = count(per -> any(x -> any(!=(0), x[1]), per), r.regime_history)

# ------------------------------------------------------------------ 1. classification of Dynare's files
# file => [(constraint, class, s_g, s_y, approx > 0)]; regenerate docs/OBC_CLASSIFICATION.md with
# `julia --project=. test/obc/classify_dynare.jl` when this changes.
const EXPECTED_CLASSES = Dict(
    "examples/occbin/rbc_occbin.mod" => [("IRR", :complementarity, 1, 1, false), ("INEG", :inconsistent, 0, 0, false)],
    "tests/estimation/slice/bf.mod" => [("zlb", :multi_equation, 0, 0, false)],
    "tests/occbin/filter/NKM_0_std_shocks.mod" => [("zlb", :complementarity, 1, 1, true)],
    "tests/occbin/filter/NKM_calib_smoother.mod" => [("zlb", :complementarity, 1, 1, true)],
    "tests/occbin/model_borrcon/borrcon.mod" => [("borrcon", :complementarity, 1, -1, false)],
    "tests/occbin/model_borrcon/borrcon_0_std_shocks.mod" => [("borrcon", :complementarity, 1, -1, false)],
    "tests/occbin/model_irrcap_twoconstraints/dynrbc.mod" => [("IRR", :complementarity, 1, 1, false), ("INEG", :inconsistent, 0, 0, false)],
    "tests/occbin/model_irrcap_twoconstraints/dynrbc_0_std_shocks.mod" => [("IRR", :complementarity, 1, 1, false), ("INEG", :inconsistent, 0, 0, false)],
    "tests/occbin/model_irrcap_twoconstraints/dynrbc_token_xfail.mod" => nothing,      # the preprocessor rejects it (by design)
    "tests/occbin/ppf/bf_ppf.mod" => [("zlb", :multi_equation, 0, 0, false)],
    "tests/occbin/ppf/swpkfppf.mod" => [("zlb", :multi_equation, 0, 0, false)],
)

@testset "OBC: classification of Dynare 7.2's OccBin files" begin
    files = occbin_mod_files(DYNARE_SRC)
    @test Set(files) == Set(keys(EXPECTED_CLASSES))
    for rel in files
        r = classify_file(DYNARE_SRC, rel)
        exp = EXPECTED_CLASSES[rel]
        if exp === nothing
            @test r.rows === nothing
            continue
        end
        @test r.rows !== nothing
        r.rows === nothing && (println(rel, ": ", r.error); continue)
        got = [(x.constraint, x.class, x.s_g, x.s_y, x.approx > 0) for x in r.rows]
        got == exp || println("classification of $rel: ", got)
        @test got == exp
    end
    # the doc is the generated table
    md, _, _ = classification_markdown(DYNARE_SRC)
    @test strip(md) == strip(read(joinpath(@__DIR__, "..", "docs", "OBC_CLASSIFICATION.md"), String))
end

# ------------------------------------------------------------------ 2. worked cases of the contract
@testset "OBC: worked cases (gap and wedge)" begin
    # lower bound without a relax condition: g = r - rbar, y = r - rn
    m = load_dir(joinpath(OBC_DIR, "nk_zlb.mod"))
    c = only(obc_constraints(m))
    @test c.class == :complementarity && c.approx == 0
    cur(v) = m.n_endo + m.ie[v]                  # position of the current-period variable v in [constant; y(-1); y; y(+1); u]
    g = c.s_g .* JDyn._vec(c.alt_resid[1])
    y = c.s_y .* JDyn._vec(c.ref_resid[1])
    rss, rbar = m.params[m.ip["rss"]], m.params[m.ip["rbar"]]
    @test g[1] ≈ rss - rbar && g[1+cur("r")] ≈ 1 && count(!iszero, g[2:end]) == 1
    @test y[1] == 0 && y[1+cur("r")] ≈ 1 && y[1+cur("rn")] ≈ -1 && count(!iszero, y[2:end]) == 2
    # Dynare's borrowing constraint (the constraint binds in the reference regime): g = lb (level), y = M*y - b
    m = load_dir(joinpath(DYNARE_SRC, "tests", "occbin", "model_borrcon", "borrcon.mod"))
    c = only(obc_constraints(m))
    @test c.class == :complementarity && c.s_g == 1 && c.s_y == -1
    g = c.s_g .* JDyn._vec(c.alt_resid[1])
    y = c.s_y .* JDyn._vec(c.ref_resid[1])
    ys, _ = steady_state(m)
    @test g[1] ≈ ys[m.ie["lb"]] && g[1] > 0 && g[1+cur("lb")] ≈ 1 && count(!iszero, g[2:end]) == 1
    Mp = m.params[m.ip["M"]]
    @test y[1+cur("b")] ≈ -1 && y[1+cur("y")] ≈ Mp && count(!iszero, y[2:end]) == 2
end

# ------------------------------------------------------------------ 3. M and q
@testset "OBC: M and q against a stacked system (1e-10) and Dynare 7.2 (1e-12)" begin
    for (path, T) in ((joinpath(OBC_DIR, "nk_zlb.mod"), 40), (joinpath(OBC_DIR, "nk_zlb2.mod"), 30), (joinpath(OBC_DIR, "obc_zlb2.mod"), 25),
                      (joinpath(DYNARE_SRC, "tests", "occbin", "model_borrcon", "borrcon.mod"), 40))
        m = load_dir(path); ob = Occbin(m)
        sys = obc_system(ob; horizon=T)
        N = length(sys.cons) * T
        @test size(sys.M) == (N, N)
        z0, u0 = zeros(ob.n), zeros(ob.n_exo)
        base = stacked_gaps(ob, sys.cons, T, zeros(N), z0, u0)
        Ms = hcat([stacked_gaps(ob, sys.cons, T, (e = zeros(N); e[k] = 1.0; e), z0, u0) - base for k in 1:N]...)
        @test maximum(abs.(Ms - sys.M)) < 1e-10
        rng = MersenneTwister(7)
        init, u, w = 0.01 .* randn(rng, ob.n), 0.01 .* randn(rng, ob.n_exo), abs.(0.01 .* randn(rng, N))
        @test maximum(abs.(stacked_gaps(ob, sys.cons, T, w, init, u) - (obc_q(sys, init, u) + sys.M * w))) < 1e-10
        @test obc_system(ob; horizon=T) === sys                      # cached per parameter set (Occbin object) and horizon
    end
    # Dynare 7.2: the model with news shocks written into the .mod file by hand (obc_zlb_news.mod), T = 20, 60 rows
    ref = matread(joinpath(OBC_DIR, "obc_zlb.mat"))
    m = load_dir(joinpath(OBC_DIR, "obc_zlb.mod")); ob = Occbin(m)
    sys = obc_system(ob; horizon=20)
    M60 = hcat([obc_gaps(sys, (e = zeros(20); e[k] = 1.0; e), zeros(ob.n), zeros(ob.n_exo), 60; homogeneous=true)[2]' for k in 1:20]...)
    @test maximum(abs.(M60 - ref["M"])) < 1e-12
    q60 = vec(obc_gaps(sys, zeros(20), zeros(ob.n), [ref["shock"]], 60)[2])
    @test maximum(abs.(q60 - vec(ref["q"]))) < 1e-12
    y, st, _ = solve_minnorm(sys.M, obc_q(sys, zeros(ob.n), [ref["shock"]]))
    @test st == :ok && maximum(abs.(y - vec(ref["y"]))) < 1e-9          # GLPK's solution (its tolerance)
    r = simulate(ob; obc_method=:minnorm)
    @test maximum(abs.(r.piecewise[:, m.ie["r"]] - vec(ref["r_occbin"]))) < 1e-12     # Dynare's OccBin path
    @test r.regime_history[1][1][2] == Int.(vec(ref["regimestart"]))
end

# ------------------------------------------------------------------ 4. solvers
@testset "OBC: solvers on small problems" begin
    # verification and shortcut
    @test solve_minnorm([1.0;;], [0.5])[2] == :ok && solve_minnorm([1.0;;], [0.5])[1] == [0.0]
    @test JDyn.obc_verify([1.0;;], [-1.0], [1.0])[1] && !JDyn.obc_verify([1.0;;], [-1.0], [0.5])[1]
    # random P-matrices (positive definite plus skew part): unique solution, all solvers agree
    rng = MersenneTwister(3)
    for k in 1:50
        n = rand(rng, 2:30)
        A = randn(rng, n, n); S = randn(rng, n, n)
        M = A * A' + 0.1I + (S - S')
        q = randn(rng, n)
        y1, s1, _ = solve_minnorm(M, q)
        y2, s2, _ = solve_minnorm(M, q; descent=false)
        @test s1 == :ok && s2 == :ok && JDyn.obc_verify(M, q, y1)[1] && maximum(abs.(y1 - y2)) < 1e-9
        y3, s3, _ = solve_active_set(M, q)
        s3 == :ok && @test maximum(abs.(y1 - y3)) < 1e-9
    end
    # several solutions: (1,0), (0,1) and (1/3,1/3); minnorm returns the smallest
    M, q = [1.0 2.0; 2.0 1.0], [-1.0, -1.0]
    y, st, info = solve_minnorm(M, q)
    @test st == :ok && y ≈ [1/3, 1/3] && info.alternatives >= 1
    # no solution: y >= 0, -1 - y >= 0 is empty
    @test solve_minnorm([-1.0;;], [-1.0])[2] == :no_solution
    @test solve_active_set([-1.0;;], [-1.0])[2] == :not_converged
    # news: implemented (src/obc_solvers.jl; tested in test_obc_news.jl); registered as the solver of obc_method = news
    @test solve_news(M, q)[2] == :ok
    @test JDyn.OBC_SOLVERS[:news] === solve_news
end

# ------------------------------------------------------------------ 5. minnorm against OccBin on every complementarity model
# (path, scale of the random shocks relative to the model's standard deviations; 0: the file has zero standard
# deviations, use shocks of standard deviation 0.02)
const AGREEMENT_MODELS = [
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

@testset "OBC: minnorm = OccBin ($(basename(path)))" for (path, scale) in AGREEMENT_MODELS
    m = load_dir(path); ob = Occbin(m)
    has_surprise = any(st -> get(st, "surprise", false) == true, get(m.options, "shock_paths", Any[]))
    @test all(c -> c.class == :complementarity, obc_constraints(ob))
    for fast in (true, false)                     # with the shared active-set fast path, and the pure minnorm solver
        if has_surprise
            a = simulate(ob)
            b = simulate(ob; obc_method=:minnorm, obc_fast_path=fast)
            @test isapprox(b.piecewise, a.piecewise; atol=PATH_TOL, rtol=0) && b.regime_history == a.regime_history
            @test b.linear == a.linear && a.news === nothing && b.news.method == :minnorm && all(==(:ok), b.news.status)
            fast && println(rpad(basename(path), 26), "file shocks: max |minnorm - occbin| = ", maximum(abs.(b.piecewise - a.piecewise)),
                            ", periods binding: ", nbinding(a))
        end
        # seeded 200-period random simulation
        sd = scale == 0 ? fill(0.02, ob.n_exo) : scale .* sqrt.(diag(m.Sigma_e))
        sh = randn(MersenneTwister(20261124), 200, ob.n_exo) .* sd'
        a = simulate(ob, sh; periods=200)
        b = simulate(ob, sh; periods=200, obc_method=:minnorm, obc_fast_path=fast)
        @test nbinding(a) > 0                     # the comparison means something
        @test isapprox(b.piecewise, a.piecewise; atol=PATH_TOL, rtol=0) && b.regime_history == a.regime_history
        fast && println(rpad(basename(path), 26), "random 200: max |minnorm - occbin| = ", maximum(abs.(b.piecewise - a.piecewise)),
                        ", periods binding: ", nbinding(a), ", fast path used in ",
                        count(i -> get(i, :solver, :none) == :active_set, [x isa NamedTuple ? Dict(pairs(x)) : Dict() for x in b.news.info]), " periods")
    end
end

@testset "OBC: IRFs with obc_method" begin
    m = load_dir(joinpath(OBC_DIR, "nk_zlb2.mod")); ob = Occbin(m)
    a = irf_occbin(ob; shocksize=[0.03, 0.01, 0.003])
    b = irf_occbin(ob; shocksize=[0.03, 0.01, 0.003], obc_method="minnorm")
    @test keys(a) == keys(b)
    @test all(isapprox(b[k].piecewise, a[k].piecewise; atol=PATH_TOL, rtol=0) for k in keys(a))
    @test maximum(abs(b[k].piecewise[t, i] - b[k].linear[t, i]) for k in keys(b) for t in 1:41 for i in 1:ob.n) > 1e-3
    @test isapprox(irf(ob, [-0.03, 0, 0]; obc_method=:minnorm), irf(ob, [-0.03, 0, 0]); atol=PATH_TOL)
end

# ------------------------------------------------------------------ 6. refusals and failures
@testset "OBC: refusals" begin
    # Dynare's rbc_occbin: INEG is inconsistent; bf_ppf: multi_equation
    for (rel, cls) in (("examples/occbin/rbc_occbin.mod", "inconsistent"), ("tests/occbin/ppf/bf_ppf.mod", "multi_equation"))
        m = load_dir(joinpath(DYNARE_SRC, rel)); ob = Occbin(m)
        err = try simulate(ob, zeros(1, ob.n_exo); obc_method=:minnorm); nothing catch e; e end
        @test err isa ModelError && occursin(cls, err.msg) && occursin("obc_method = occbin", err.msg)
    end
    # a floor whose relax condition contradicts its bind condition (the orientation test)
    txt = replace(read(joinpath(OBC_DIR, "nk_zlb2.mod"), String), "relax r + sp > rlbar;" => "relax rl > r + sp;")
    m = build(txt; name="wrongway")
    c = obc_constraints(m)
    @test c[1].class == :complementarity && c[2].class == :inconsistent && occursin("orientation", c[2].why)
    @test_throws ModelError simulate(Occbin(m); obc_method=:minnorm)
    # unknown method, bad horizon; news is implemented (formerly a stub that threw)
    ob = Occbin(load_dir(joinpath(OBC_DIR, "obc_zlb.mod")))
    @test_throws ArgumentError simulate(ob; obc_method=:lcp)
    @test_throws ArgumentError simulate(ob; obc_method=:minnorm, obc_horizon=0)
    @test simulate(ob; obc_method=:news).news.method == :news
    # horizon too short: the bound binds for 7 periods, the horizon is 3
    err = try simulate(ob; obc_method=:minnorm, obc_horizon=3); nothing catch e; e end
    @test err isa ObcError && err.status == :horizon_too_short && err.period == 1
    r = simulate(ob; obc_method=:minnorm, obc_horizon=3, obc_raise=false)
    @test r.news.status[1] == :horizon_too_short
    # OccBin's regime-iteration options are recognised and not used: one warning
    @test_logs (:warn, r"not used") match_mode=:any simulate(ob; obc_method=:minnorm, maxit=5, curb_retrench=true)
end

# ------------------------------------------------------------------ 7. interface: native lines, workspace, oo_
@testset "OBC: native lines and the live workspace" begin
    d = mktempdir()
    cp(joinpath(OBC_DIR, "obc_zlb.mod"), joinpath(d, "obc_zlb.mod"))
    write(joinpath(d, "zlb_minnorm.mod"), "@#include \"obc_zlb.mod\"\noptions_.occbin.obc_method = 'minnorm';\noptions_.occbin.obc_horizon = 30;\n")
    m = with_logger(NullLogger()) do; JDyn.load(joinpath(d, "zlb_minnorm.mod")); end
    ob = Occbin(m)
    @test mod_options(ob).obc_method == :minnorm && mod_options(ob).obc_horizon == 30
    r = simulate(ob)
    @test r.news isa ObcNews && r.news.horizon == 30 && size(r.news.wedges) == (1, 30)
    r0 = simulate(ob; obc_method=:occbin)                 # keyword overrides the file
    @test r0.news === nothing && isapprox(r.piecewise, r0.piecewise; atol=PATH_TOL) && r.regime_history == r0.regime_history
    # dynare(): the native lines come before occbin_solver in the file? Here after it (in the include): the method in
    # force for the file's own occbin_solver statement is OccBin's; occbin_solver! afterwards uses options_.occbin
    ws = with_logger(NullLogger()) do; dynare(joinpath(d, "zlb_minnorm.mod")); end
    @test ws.options_.occbin.obc_method == "minnorm"
    occbin_solver!(ws)
    @test ws.oo_.occbin.news.method == "minnorm" && ws.oo_.occbin.news.horizon == 30
    @test isapprox(ws.oo_.occbin.simul.piecewise, r0.piecewise; atol=PATH_TOL)
    occbin_solver!(ws; obc_method="occbin")
    @test ws.options_.occbin.obc_method == "occbin"
    # native line before occbin_solver, keyword on the workspace
    txt = replace(read(joinpath(OBC_DIR, "obc_zlb.mod"), String), "occbin_setup;" => "options_.occbin.obc_method = 'minnorm';\noccbin_setup;")
    write(joinpath(d, "zlb2.mod"), txt)
    ws = with_logger(NullLogger()) do; dynare(joinpath(d, "zlb2.mod")); end
    @test haskey(ws.oo_.occbin, "news") && ws.oo_.occbin.news.status[1] == "ok"
    @test isapprox(ws.oo_.occbin.simul.piecewise, r0.piecewise; atol=PATH_TOL)
    ws2 = with_logger(NullLogger()) do; dynare(joinpath(d, "obc_zlb.mod")); end
    @test !haskey(ws2.oo_.occbin, "news")                  # default: OccBin, results as before
    @test ws2.oo_.occbin.simul.piecewise == r0.piecewise
end
