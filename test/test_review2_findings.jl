# Second-pass review (docs/REVIEW2.md): one @test_broken per confirmed open finding (the suite stays green; when
# a finding is fixed its @test_broken reports an "unexpected pass": change it to @test), plus cheap @test for
# things checked clean. Ground truth (Dynare 7.2 under Octave 8.4, or the MATLAB/Octave semantics) is written
# into the tests; REVIEW2.md says how each value was obtained.
#
# Runs on its own:   julia --project=. test/test_review2_findings.jl      (about 2 minutes; R13 starts a child Julia)
# The R26-R30 tests need src/results.jl (branch `results`); on a branch without it they are skipped.
using Test
include(joinpath(@__DIR__, "common.jl"))
using JDyn: mrun, MatlabError, Unsupported, ModelError

val(code) = try
    mrun(code)["r"]
catch e
    e
end
clean_error(code) = try
    mrun(code); false
catch e
    e isa MatlabError || e isa Unsupported
end
quietly(f) = with_logger(f, NullLogger())

const RBC = """
var c k a; varexo e; parameters alpha beta rho;
alpha = 0.35; beta = 0.95; rho = 0.8;
model;
1/c = beta*alpha*exp(a(+1))*k^(alpha-1)/c(+1);
c + k = exp(a)*k(-1)^alpha;
a = rho*a(-1) + e;
end;
initval; k = 0.2; c = 0.5; a = 0; end;
shocks; var e = 0.01; end;
steady;
"""

"Write `text` as name.mod in a fresh folder; return the path (for dynare())."
function modfile(text; name="m", files=Dict{String,String}())
    d = mktempdir()
    for (f, c) in files
        write(joinpath(d, f), c)
    end
    p = joinpath(d, name * ".mod"); write(p, text); p
end

# data for the smoother/estimation tests (y and w = y with missing values; z = y + a random walk), 40 periods: the
# file given to Dynare 7.2 for the reference values below
const DAT1 = """
y = [NaN; NaN; -1.02068155763; 0.981111876598; 0.840076435146; 1.88292275402; 2.28262422379; 1.68757133143; 0.1865310337; -1.45903479963; NaN; NaN; -1.22055113274; -0.347453612573; -0.712579058291; -1.14497071634; -2.4498596656; -4.40650265848; -3.63795574071; -2.81481672887; -1.48791374368; -1.93134221025; -2.72464959138; -3.2379431784; -4.78357248439; NaN; -2.83027199618; -2.8305336481; -0.891560376583; 0.269429009455; 0.087261427812; 0.719010502212; 0.568210952789; 2.35164802159; 3.64160681753; 2.79385798614; 2.04846863704; 2.05321217797; 1.61982902842; 1.28122775283];
z = [0.719720228829; 0.275059697539; -0.0302362188185; 2.05032575123; 2.27403955299; 3.30146423434; 3.21430352539; 2.40018811583; 0.144890754513; -1.62000150654; NaN; NaN; -2.09043019174; -1.64044399421; -3.01315826789; -4.54389543676; -5.29802863984; -6.85539075108; -6.24301338307; -5.15614372421; -3.94442582083; -4.6901619513; -5.50593083118; -5.48492716931; -7.5774288262; -9.09867767297; -7.87472969285; -8.13759746764; -7.19133242074; -6.39605007911; NaN; -6.85077426865; -6.82486618461; -6.24406837503; -4.48007478981; -5.17535882784; -5.09738392017; -5.20041821915; -5.73373946416; -6.51183229559];
w = [NaN; NaN; -1.02068155763; 0.981111876598; 0.840076435146; 1.88292275402; 2.28262422379; 1.68757133143; 0.1865310337; -1.45903479963; NaN; NaN; -1.22055113274; -0.347453612573; -0.712579058291; -1.14497071634; -2.4498596656; -4.40650265848; -3.63795574071; -2.81481672887; -1.48791374368; -1.93134221025; -2.72464959138; -3.2379431784; -4.78357248439; NaN; -2.83027199618; -2.8305336481; -0.891560376583; 0.269429009455; 0.087261427812; 0.719010502212; 0.568210952789; 2.35164802159; 3.64160681753; 2.79385798614; 2.04846863704; 2.05321217797; 1.61982902842; 1.28122775283];
"""

@testset "second-pass review findings" begin

# ====================================================================== pass-1 findings, re-verified
@testset "R01 unit-root steady state: identity variable keeps its initval (Dynare: Dulmage-Mendelsohn blocks)" begin
    m = build("""
        var y c; varexo e; parameters gam; gam = 0.3;
        model; y = y(-1) + e; c = 0.5*c(+1) + gam*exp(y); end;
        initval; y = 0.2; c = 1; end;
        shocks; var e = 0.01; end;
        steady;
        """)
    ys, _ = steady_state(m)
    @test maximum(abs, JDyn.static_resid(m, ys)) < 1e-12          # a valid steady state (the problem has a continuum)
    @test_broken isapprox(ys, [0.2, 0.7328416548961019]; rtol=1e-10)   # Dynare 7.2: y stays at initval; JDyn y = 0.324
end

@testset "R02 R03 R05 R19 R20 interpreter: silently wrong values (Octave 8.4 ground truth)" begin
    @test val("l = [true false true]; l(2) = 5; r = l;") isa AbstractArray{Bool}
    @test val("l = false(1,3); l(3) = 1; a = [10 20 30]; r = a(l);") == 30.0
    @test size(val("A = [1 2 3; 4 5 6]; r = A(logical([1 0 1]));")) == (1, 2)
    @test size(val("A = [1 2 3; 4 5 6]; r = A(logical([1;0;1]));")) == (2, 1)     # column mask: right
    @test size(val("r = zeros(0,3); r(:,10) = 5;")) == (0, 10)
    @test val("r = zeros(0,3); r(2,:) = 5;") == [0.0 0 0; 5 5 5]
    @test val("r = []; r(end+1,:) = [1 2 3]; r(end+1,:) = 4;") == [1.0 2 3; 4 4 4]  # growth from 0x0: right
    @test length(val("r = unique([3 NaN 1 NaN]);")) == 4
end

@testset "R04 R06 R10 interpreter: refusals of valid MATLAB" begin
    @test val("if any([NaN 1]), r = 1; else, r = 0; end") == 1.0
    @test val("r = all(NaN);") === true
    @test (x = val("r = [1;2;3]; r(:,:) = [];"); x isa AbstractMatrix && size(x) == (0, 1))
    @test val("r = 2 ^ [1 2; 3 4];") isa AbstractMatrix
end

@testset "R07 (S4) [1 2 3; zeros(0,2)] is a dimension error in Octave" begin
    @test val("r = [1 2 3; zeros(0,2)];") isa MatlabError
end

@testset "R08 Julia exceptions escape MATLAB try/catch" begin
    @test val("try, x = trace(ones(2,3)); r = 1; catch, r = 2; end") in (1.0, 2.0)
    @test !(val("r = sprintf('%c', 65.5);") isa Exception)
end

@testset "R09 sprintf numeric conversions of char data" begin
    @test val("r = sprintf('%d|', 'ab');") == "97|98|"
end

@testset "R11 set_shock_stderr! keeps the covariance (Dynare: M_.Sigma_e(i,i) = value^2 only)" begin
    m = build("""
        var y x; varexo e u; parameters a; a = 0.5;
        model; y = a*y(-1) + e; x = 0.3*x(-1) + u; end;
        shocks; var e = 1; var u = 4; corr e,u = 0.5; end;
        """)
    set_shock_stderr!(m, "u", 3.0)
    @test_broken m.Sigma_e == [1.0 1.0; 1.0 9.0]
end

@testset "R12 set_params! is not atomic" begin
    m = build("var y; varexo e; parameters a b; a = 0.5; b = 1;\nmodel; y = a*y(-1) + b*e; end;\n")
    try; set_params!(m, ["a" => 0.9, "nonexistent" => 1.0]); catch; end
    @test_broken get_param(m, "a") == 0.5
end

@testset "R13 steady_state_model can call Julia's exit()" begin
    d = mktempdir()
    write(joinpath(d, "ex.mod"), "var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; end;\nsteady_state_model; y = exit(7); end;\n")
    script = """
        using JDyn, Logging
        ENV["DYNARE_PREPROCESSOR"] = "$(ENV["DYNARE_PREPROCESSOR"])"
        m = with_logger(NullLogger()) do; JDyn.load("$(joinpath(d, "ex.mod"))"); end
        try; steady_state(m); catch; end
        println("ALIVE")
        """
    out = IOBuffer()
    p = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) -e $script`); stdout=out, stderr=devnull))
    @test occursin("ALIVE", String(take!(out)))       # today: the child exits with code 7, nothing printed
end

@testset "R14 negative variance inside the QZ criterion: raw DomainError" begin
    m = build("var y; varexo e; parameters rho; rho = 1.0000005;\nmodel; y = rho*y(-1) + e; end;\nshocks; var e = 1; end;\nsteady;\nstoch_simul(order=1, irf=0);\n")
    r = try; stoch_simul_moments(m); nothing; catch e; e; end
    @test_broken !(r isa DomainError)
end

@testset "R15 (S4, fixed) irf_plot_threshold makes stoch_simul_moments warn" begin
    lg = Test.TestLogger(; min_level=Logging.Warn)
    with_logger(lg) do
        m = build("var y; varexo e; parameters rho; rho = 0.9;\nmodel; y = rho*y(-1) + e; end;\nshocks; var e = 1; end;\nsteady;\nstoch_simul(order=1, irf=5, irf_plot_threshold=0.1);\n"; quiet=false)
        stoch_simul_moments(m)
    end
    @test !any(l -> occursin("plot_threshold", string(l.message)), lg.logs)
end

@testset "R16 irf_figures at order 2 returns order-1 IRFs without a warning" begin
    lg = Test.TestLogger(; min_level=Logging.Warn)
    res = with_logger(lg) do
        m = build(RBC * "stoch_simul(irf=4);\n"; quiet=false)
        JDyn.irf_figures(m)
    end
    @test res.order == 1
    @test_broken any(l -> occursin("order", string(l.message)), lg.logs)
end

@testset "R17 load() does not forward extra_args" begin
    d = mktempdir(); write(joinpath(d, "mac.mod"), "@#ifndef A\n@#define A = 1\n@#endif\nvar y; varexo e; parameters a; a = 0.1*@{A};\nmodel; y = a*y(-1) + e; end;\n")
    r = try; quietly(() -> JDyn.load(joinpath(d, "mac.mod"); extra_args=["-DA=3"])); catch e; e; end
    @test_broken r isa JDyn.Model && r.params[1] ≈ 0.3
end

@testset "R18 refusals as raw Julia exceptions" begin
    f = joinpath(DYNARE_SRC, "tests", "deterministic_simulations", "ramst_dseries.mod")
    if isfile(f)
        r = try; load_copy(f); nothing; catch e; e; end
        @test_broken r === nothing || r isa ModelError
    end
end

@testset "R21 k_order refuses an unused NaN parameter (Dynare solves it)" begin
    m = build(RBC * "parameters unused;\nstoch_simul(order=2, irf=0);\n")
    @test isnan(get_param(m, "unused"))
    @test first_order(m) isa JDyn.FirstOrderSolution
    r = try; k_order(m; order=2); catch e; e; end
    @test_broken r isa AbstractDict
end

# ====================================================================== new findings (load API)
@testset "R32 (S4) non-ASCII characters become '?'" begin
    @test val("r = double('é');") == 233.0          # MATLAB: 233 (UTF-16 code unit); Octave: [195 169]; JDyn: 63
end

@testset "R22 steady_state_model is evaluated with Julia's Base, not MATLAB semantics" begin
    m = build("var y; varexo e; parameters a; a = 2.5;\nmodel; y = 0.5*y(-1) + e; end;\nsteady_state_model; y = round(a); end;\n")
    @test val("r = round(2.5);") == 3.0                                  # JDyn's own interpreter: right
    @test steady_state(m; check=false)[1][1] == 3.0                # MATLAB round(2.5) = 3; compiled Julia round: 2
    m2 = build("var y; varexo e; parameters a; a = 2.5;\nmodel; y = 0.5*y(-1) + e; end;\nsteady_state_model; y = mod(a, 0); end;\n")
    r = try; steady_state(m2; check=false)[1][1]; catch e; e; end
    @test r == 2.5                                                 # MATLAB mod(x, 0) = x; JDyn: SteadyStateError (NaN)
end

@testset "R23 options of earlier stoch_simul statements are dropped by the load API" begin
    m = build(RBC * "stoch_simul(order=1, irf=0, hp_filter=1600);\nstoch_simul(order=1, irf=3) c;\n")
    mo = stoch_simul_moments(m)
    # Dynare 7.2 (Octave): the second stoch_simul keeps options_.hp_filter = 1600; oo_.var = 0.00347420017853539
    @test_broken isapprox(mo.var[1, 1], 0.00347420017853539; rtol=1e-6)    # JDyn: 0.007660 (unfiltered)
end

@testset "R24 homotopy_setup(from_initval_to_endval) ignored: steady state at the initval exogenous value" begin
    f = joinpath(DYNARE_SRC, "tests", "steady_state", "homotopy", "homotopy_from_initval_to_endval.mod")
    if isfile(f)
        m = load_copy(f; extra=[joinpath(dirname(f), "common.mod")])
        ys, _ = steady_state(m)
        # Dynare 7.2 (Octave): oo_.steady_state = [6.12244879311, 51.0204019802, 2], exo_steady_state = 2
        @test_broken isapprox(ys[1:2], [6.12244879311, 51.0204019802]; rtol=1e-6)   # JDyn: [1.5306, 12.755] (x = 1)
    end
end

@testset "R25 stochastic singularity accepted silently (Dynare refuses)" begin
    base = """
        var y x w; varexo e u; parameters rho;
        rho = 0.9;
        model; y = rho*y(-1) + e; x = 0.5*x(-1) + u; w = y; end;
        shocks; var e = 1; var u = 0.25; end;
        steady;
        varobs y w;
        """
    m = build(base * "calib_smoother(datafile=dat1);\n"; files=Dict("dat1.m" => DAT1))
    # Dynare 7.2: DsgeSmoother error 'F is singular in stationary period' (use_univariate_smoother_if_singularity_is_detected is false)
    r = try; quietly(() -> smooth(m)); catch e; e; end
    @test_broken r isa Exception
end

@testset "R25 checked clean: diffuse smoother and likelihood with missing data on a unit-root model" begin
    m = build("""
        var y x z; varexo e u; parameters rho;
        rho = 0.9;
        model; y = rho*y(-1) + e; x = x(-1) + u; z = y + x; end;
        shocks; var e = 1; var u = 0.25; end;
        steady_state_model; y = 0; x = 0; z = 0; end;
        varobs z y;
        estimated_params; rho, 0.8; stderr e, 0.9; end;
        estimation(datafile=dat1, mode_compute=0, diffuse_filter);
        """; files=Dict("dat1.m" => DAT1))
    est = quietly(() -> JDyn.Estimation(m))
    ll = log_likelihood(est, est.x0)
    @test isapprox(-ll, 90.1848565996967; rtol=1e-12)   # Dynare 7.2: oo_.likelihood_at_initial_parameters (options_.debug = 1)
end

@testset "checked clean: order 3 at and near a unit root matches Dynare's k-order solver structure" begin
    for rho in (1.0, 0.9999999)
        m = build("""
            var a c k; varexo e u; parameters rho gam bet;
            rho = $rho; gam = 0.3; bet = 0.95;
            model;
            a = rho*a(-1) + e;
            c = bet*c(+1)*exp(0.1*(a(+1)-a)) + gam*exp(0.5*a) - 0.2*k(-1)^2 + u;
            k = 0.5*k(-1) + 0.1*c + 0.01*a;
            end;
            steady_state_model; a = 0; c = (-(1-bet) + sqrt((1-bet)^2 + 4*0.008*gam))/(2*0.008); k = 0.2*c; end;
            shocks; var e = 0.0001; var u = 0.0004; end;
            stoch_simul(order=3, irf=0);
            """)
        d = k_order(m; order=3)
        # Dynare 7.2 (Octave) oo_.dr.ghx for rho = 1 (order_var c a k... as Dynare): first entries
        rho == 1.0 && @test isapprox(vec(d["ghx"])[1:3], [0.447789743503374, 0.0, -0.522102564966261]; atol=1e-13)
        @test all(isfinite, d["ghxxx"])
    end
end

# ====================================================================== results branch (dynare(), re-run commands)
if isdefined(JDyn, :dynare)
    dyn(text; kw...) = quietly(() -> JDyn.dynare(modfile(text); kw...))

    @testset "R26 (fixed) options_ is the source of truth for the re-run commands" begin
        ws = dyn(RBC * "stoch_simul(order=1, irf=4);\n")
        # loglinear is acted on: this RBC has a = 0 in steady state, so Dynare refuses to loglinearise too
        @test_throws JDyn.LoglinearError quietly(() -> JDyn.stoch_simul!(ws; loglinear=true))
        @test ws.options_.loglinear == true                    # persisted, as in Dynare
        r = try; quietly(() -> JDyn.stoch_simul!(ws; loglinear=false, qz_criterium=0.5)); nothing; catch e; e; end
        @test r isa JDyn.BlanchardKahnError                    # root 0.8 > qz_criterium 0.5: Dynare's BK error
        ws.options_.qz_criterium = zeros(0, 0)
        quietly(() -> JDyn.stoch_simul!(ws; conditional_variance_decomposition=[1, 4]))
        @test haskey(ws.oo_, "conditional_variance_decomposition")
        f = joinpath(EXAMPLES, "occbin", "rbc_occbin.mod")
        if isfile(f)
            d = mktempdir(); cp(f, joinpath(d, "rbc_occbin.mod"))
            wo = quietly(() -> JDyn.dynare(joinpath(d, "rbc_occbin.mod")))
            quietly(() -> JDyn.occbin_solver!(wo; simul_periods=7))
            @test wo.options_.occbin.simul.periods == 7
            # Dynare simulates max(simul.periods, shock periods) periods: 169 here (200 with the file's value)
            @test size(wo.oo_.occbin.simul.piecewise, 1) == max(7, size(wo.oo_.occbin.simul.shocks_sequence, 1)) == 169
        end
    end

    @testset "R27 commands run at the final calibration: oo_ and M_ differ from Dynare" begin
        ws = dyn("""
            var y; varexo e; parameters a b;
            a = 0.5; b = 3;
            model; y = a*y(-1) + b + e; end;
            steady_state_model; b = 2*a; y = b/(1-a); end;
            shocks; var e = 1; end;
            steady;
            a2 = M_.params(2) * 2;
            set_param_value('a', a2/10);
            """)
        # Dynare 7.2 (Octave): M_.params = [0.2; 1], oo_.steady_state = 2 (JDyn.load gets this right too)
        @test vec(ws.M_.params) ≈ [0.2, 1.0]           # JDyn dynare(): [0.2, 0.4]
        @test vec(ws.oo_.steady_state) ≈ [2.0]         # JDyn dynare(): 0.5
        ws2 = dyn("""
            var y x z; varexo e u; parameters rho sig;
            rho = 0.5; sig = 1;
            model; y = rho*y(-1) + sig*e; x = 0.5*x(+1) + y + u + z; z = 1.2*z(-1) - 0.5*z(-2) + u; end;
            shocks; var e = 1; var u = 0.25; corr e, u = 0.3; end;
            steady;
            stoch_simul(order=1, irf=5) y x;
            rho = 0.9;
            stoch_simul(order=1, irf=5, relative_irf) y;
            """)
        # Dynare 7.2: oo_.irfs.x_e stays from the first stoch_simul (rho = 0.5)
        @test isapprox(vec(ws2.oo_.irfs.x_e), [1.769047619, 0.9380952381, 0.5161904762, 0.2503809524, 0.09236190476]; rtol=1e-8)
        @test isapprox(vec(ws2.oo_.irfs.y_e), [100, 90, 81, 72.9, 65.61]; rtol=1e-10)   # the last run: right
    end

    @testset "R28 stoch_simul! IRF rules differ from Dynare's stoch_simul.m (relative_irf at order 2, skip threshold)" begin
        f = joinpath(DYNARE_SRC, "tests", "irfs", "example1_unit_std.mod")
        if isfile(f)
            d = mktempdir(); cp(f, joinpath(d, "example1_unit_std.mod"))
            ws = quietly(() -> JDyn.dynare(joinpath(d, "example1_unit_std.mod")))
            # Dynare 7.2 (Octave): final oo_.irfs.a_e(1) = 0.01 (relative_irf at order 2 normalises the shock to 0.01)
            @test ws.oo_.irfs.a_e[1] ≈ 0.01                 # JDyn: 0.0095 (the shock's stderr)
        end
        ws = dyn("var y; varexo e; parameters rho; rho = 0.5;\nmodel; y = rho*y(-1) + e; end;\nshocks; var e; stderr 8e-7; end;\nsteady;\nstoch_simul(order=1, irf=3);\n")
        @test haskey(ws.oo_.irfs, "y_e")                   # Dynare: cs(i,i) > 5e-7 -> IRF stored; JDyn skips below 1e-6
    end

    @testset "R29 check!: refuses BK failures and stores |eigenvalues|; stoch_simul! drops oo_.dr.eigval" begin
        r = try
            dyn("var y x; varexo e; parameters rho a;\nrho = 0.5; a = 2;\nmodel; y = rho*y(-1) + e; x = a*x(+1) + y; end;\nshocks; var e = 1; end;\nsteady;\ncheck;\n")
        catch e
            e
        end
        @test r isa JDyn.DynareWorkspace                   # Dynare 7.2: prints 'order condition NOT verified', continues
        ws = dyn("""
            var y x z; varexo e u; parameters rho;
            rho = 0.9;
            model; y = rho*y(-1) + e; x = 0.5*x(+1) + y + u + z; z = 1.2*z(-1) - 0.5*z(-2) + u; end;
            shocks; var e = 1; var u = 0.25; end;
            steady;
            check;
            """)
        ev = vec(ws.oo_.dr.eigval)
        @test any(x -> imag(x) != 0, ev)                   # Dynare: 0.6 +/- 0.3742i, 0.9, 2; JDyn: moduli only
        JDyn.stoch_simul!(ws; order=1, irf=0)
        @test haskey(ws.oo_.dr, "eigval")                  # Dynare's stoch_simul keeps dr.eigval
        r = try
            dyn(replace(RBC, "steady;" => "steady;\nshocks; var e; periods 1; values 0.1; end;\nperfect_foresight_setup(periods=50);\nperfect_foresight_solver(maxit=1, no_homotopy);\n"))
        catch e
            e
        end
        @test r isa JDyn.DynareWorkspace                   # Dynare: status 0 stored, file continues
    end

    @testset "R30 stoch_simul(periods) ignores histval" begin
        text = replace(RBC, "steady;" => "steady;\nhistval; k(0) = 0.15; a(0) = 0.02; end;\nstoch_simul(order=1, irf=0, periods=150, nomoments);\n")
        ex = zeros(150, 1)
        ws = dyn(text; shocks=Dict(1 => ex))
        # Dynare: the simulation starts from histval, so with zero shocks a(1) = 0.8*0.02 = 0.016
        @test ws.oo_.endo_simul[3, 1] ≈ 0.016              # JDyn: 0 (starts at the steady state)
    end

    @testset "R31 the no-argument commands act on the last workspace made, not on the globals" begin
        wsA = dyn(RBC * "stoch_simul(order=1, irf=4);\n")
        wsB = dyn(RBC * "stoch_simul(order=1, irf=4);\n")
        before = copy(wsA.oo_.irfs.c_e)
        wsA.M_.params[3] = 0.5
        JDyn.CURRENT[] = nothing
        @test_throws ArgumentError JDyn.stoch_simul!()                # dynare() no longer sets the current workspace
        quietly(() -> JDyn.stoch_simul!(wsA))
        @test wsA.oo_.irfs.c_e != before
        # and a parameter set through the documented Model API is overwritten by M_ on the next command
        JDyn.set_param!(wsB.model, "rho", 0.3)
        quietly(() -> JDyn.stoch_simul!(wsB))
        @test get_param(wsB.model, "rho") == 0.3
    end

    @testset "checked clean (results): perfect foresight through dynare() against Dynare" begin
        ws = dyn(replace(RBC, "steady;" => "steady;\nshocks; var e; periods 1; values 0.1; end;\nperfect_foresight_setup(periods=50);\nperfect_foresight_solver(linear_approximation);\n"))
        @test ws.oo_.deterministic_simulation.status == true
        @test size(ws.oo_.endo_simul) == (3, 52)
    end
else
    @info "src/results.jl not on this branch: R26-R31 skipped"
end

end
