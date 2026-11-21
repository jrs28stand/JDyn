# Regression tests for the issues found in pydyn's independent review (first-order parts of pydyn
# tests/test_review.py; values marked "Dynare" were obtained from Dynare 7.2 under Octave), plus the
# handling of qz_criterium and of .mod options that the first-order code does not act on.
using Test
include("common.jl")

function fd_jac(m, y; h=1e-6)
    n = m.n_endo
    J = zeros(n, n)
    for j in 1:n
        d = zeros(n); d[j] = h
        J[:, j] .= (JDyn.static_resid(m, y .+ d) .- JDyn.static_resid(m, y .- d)) ./ (2h)
    end
    J
end

const TWO = "var c k; varexo e; parameters a; a=0.5;\n"

@testset "comparisons are numeric" begin
    m = build(TWO * "model; c = (c > a) + (k > a); k = 0.5*k(-1)+0.5*k(+1) + e; end;")
    @test JDyn.static_resid(m, [1.0, 1.0]) ≈ [-1.0, 0.0]                      # Dynare
end

@testset "chained comparisons" begin
    m = build("var c k; varexo e; parameters a b; a=3; b=1;\nmodel; c = (b < a < 2) + e; k = (c == 1 == 0); end;")
    @test JDyn.static_resid(m, [2.0, 0.0]) ≈ [1.0, -1.0]                      # Dynare
end

@testset "max/min derivatives: $expr" for expr in ["max(k, 0)", "min(k, 1)", "max(0, k)"]
    m = build(TWO * "model; c = $expr + a*c(-1); k = 0.5*k(-1) + e; end;")
    y = [0.3, 0.4]
    @test maximum(abs, JDyn.static_jac(m, y) .- fd_jac(m, y)) < 1e-7
end

const SH = "var x1 x2; varexo e1 e2; model; x1 = 0.5*x1(-1)+e1; x2 = 0.5*x2(-1)+e2; end;\n" *
           "shocks; var e1; stderr 1; var e2; stderr 1; corr e1,e2=0.5; end;\n"
@testset "shock blocks follow Dynare's order: $second" for (second, expected) in [
        ("shocks; var e1; stderr 2; end;", [4 0.5; 0.5 1]),
        ("shocks(overwrite); var e1; stderr 2; var e2; stderr 3; end;", [4 0; 0 9]),
        ("shocks; var e1,e2 = 0.1; end;", [1 0.1; 0.1 1])]
    m = build(SH * second)
    @test m.Sigma_e ≈ expected                                                 # Dynare
end

@testset "native parameter updates" begin
    m = build("var c; varexo e; parameters rho s; rho=0.5; s=2;\n" *
              "lambda = 0.25;\nscale = lambda*2;\nset_param_value('rho', 0.9);\nM_.params(2) = scale;\n" *
              "model; c = rho*c(-1) + s*e; end;")
    @test get_param(m, "rho") == 0.9 && get_param(m, "s") == 0.5
end

@testset "purely backward and forward models: $eqs" for (eqs, G11) in [
        ("c = a*c(-1) + e; k = 0.9*k(-1) + c;", 0.5),          # no leads
        ("c = a*c(+1) + e; k = 0.9*k(+1) + c;", 0.0)]          # no lags
    m = build(TWO * "model; $eqs end;")
    sol = JDyn.first_order(m)
    @test abs(sol.G[1, 1] - G11) < 1e-12
    @test abs(sol.H[1, 1] - 1.0) < 1e-12
end

@testset "numerical steady state with auxiliary variables (bkk_1992)" begin
    m = load_copy(joinpath(EXAMPLES, "macroprocessor", "bkk_1992.mod"))
    @test !isempty(m.raw["aux_vars"])
    ys, params = JDyn.steady_state(m)
    @test maximum(abs, JDyn.static_resid(m, ys, nothing, params)) < 1e-10
    @test JDyn.first_order(m) isa JDyn.FirstOrderSolution
end

@testset "initval may refer to earlier values" begin
    m = build("var a b; varexo e; parameters rho; rho=0.5;\n" *
              "model; a = 1 + rho*(a(-1)-1) + e; b = 2*a + rho; end;\n" *
              "initval; a=1; b=2*a+rho; end;")
    @test m.initval == Dict("a" => 1.0, "b" => 2.5)
end

@testset "model without statements" begin
    m = build("var a b; varexo e1; model; a=0.5*a(-1)+e1; b=0.5*b(+1)+a; end;")
    @test JDyn.first_order(m) isa JDyn.FirstOrderSolution
end

@testset "variable named like a temporary term" begin
    m = build("var c k T34; varexo e; parameters alpha; alpha=0.5;\n" *
              "model; c = 0.3*c(-1) + e; k = alpha*k(-1) + exp(c)*exp(c)+exp(c)^2;\n" *
              "T34 = 0.5*T34(-1) + exp(c)*exp(c) + e; end;")
    y = [0.1, 0.2, 0.3]
    @test maximum(abs, JDyn.static_jac(m, y) .- fd_jac(m, y)) < 1e-6
end

# ---------------------------------------------------------------------- qz_criterium (pydyn GAPS.md hazard 9)
# Dynare 7.2 (verified under Octave, GAPS.md h9): x = 1.02 x(-1) + e with qz_criterium=1.05 is solved,
# ghx = 1.02 (the root 1.02 counts as stable); with the default 1+1e-6 Blanchard-Kahn fails.
const EXPL = "var x y; varexo e; model; x = 1.02*x(-1) + e; y = 0.5*y(+1) + x; end;\nshocks; var e; stderr 1; end;\n"
@testset "qz_criterium honoured by first_order" begin
    m = build(EXPL * "stoch_simul(order=1, qz_criterium=1.05, irf=0, nograph);")
    sol = JDyn.first_order(m)
    @test JDyn.ghx(sol)[findfirst(==(m.ie["x"]), sol.order_var), 1] ≈ 1.02
    m = build(EXPL * "options_.qz_criterium = 1.05;\nstoch_simul(order=1, irf=0, nograph);")   # MATLAB line, as Dynare
    @test JDyn.ghx(JDyn.first_order(m))[findfirst(==(m.ie["x"]), JDyn.first_order(m).order_var), 1] ≈ 1.02
    m0 = build(EXPL * "stoch_simul(order=1, irf=0, nograph);")
    @test_throws JDyn.BlanchardKahnError JDyn.first_order(m0)
    @test JDyn.first_order(m0; qz_criterium=1.05) isa JDyn.FirstOrderSolution     # keyword overrides
    # Dynare's default excludes a root within 1e-6 of one: a unit root is "stable" (x = x(-1) + e solves)
    mu = build("var x; varexo e; model; x = x(-1) + e; end;")
    @test JDyn.ghx(JDyn.first_order(mu))[1, 1] ≈ 1.0
end

# ---------------------------------------------------------------------- options that are not acted on
function warnings_of(f)
    logger = Test.TestLogger(min_level=Logging.Warn)
    out = with_logger(f, logger)
    out, [string(r.message) for r in logger.logs]
end

@testset "unsupported solver options warn" begin
    m = build(TWO * "model; c = a*c(-1) + 0.1*c(+1) + e; k = 0.5*k(-1) + c; end;\nshocks; var e; stderr 1; end;\n" *
              "stoch_simul(order=1, dr=cycle_reduction, irf=0, nograph);")
    sol, w = warnings_of(() -> JDyn.first_order(m))
    @test any(s -> occursin("dr", s) && occursin("QZ", s), w)
    ref = JDyn.first_order(build(TWO * "model; c = a*c(-1) + 0.1*c(+1) + e; k = 0.5*k(-1) + c; end;\nshocks; var e; stderr 1; end;"))
    @test JDyn.ghx(sol) ≈ JDyn.ghx(ref)                        # same solution whatever the algorithm
    m = build(TWO * "model; c = a*c(-1) + e; k = 0.5*k(-1) + c; end;\nsteady(solve_algo=0, maxit=7);")
    _, w = warnings_of(() -> JDyn.steady_state(m))
    @test any(s -> occursin("solve_algo", s) && occursin("maxit", s), w)
end

@testset "qz_zero_threshold honoured" begin
    m = build(TWO * "model; c = a*c(-1) + e; k = 0.5*k(-1) + c; end;\nstoch_simul(order=1, qz_zero_threshold=1e-8, irf=0, nograph);")
    _, w = warnings_of(() -> JDyn.first_order(m))
    @test !any(s -> occursin("qz_zero_threshold", s), w)
end

@testset "stoch_simul options the moments do not act on warn" begin
    m = build(TWO * "model; c = a*c(-1) + e; k = 0.5*k(-1) + c; end;\nshocks; var e; stderr 1; end;\n" *
              "stoch_simul(order=1, periods=200, simul_replic=3, irf=5, nograph);")
    mo, w = warnings_of(() -> JDyn.stoch_simul_moments(m; rng=JDyn.Random.MersenneTwister(1)))
    @test any(s -> occursin("simul_replic", s), w)
    @test !any(s -> occursin("irf", s) || occursin("periods", s), w)
    @test mo.kind == "simulated"
end
