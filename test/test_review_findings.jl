# Findings of the hostile review (docs/REVIEW.md), each as a test that currently FAILS and is therefore
# recorded with @test_broken (the suite stays green). When a finding is fixed its @test_broken starts to
# "pass", which Test reports as an error: change it to @test then.
#
# Runs on its own:   julia --project=. test/test_review_findings.jl
# (not part of test/runtests.jl). Dynare/Octave reference values are written into the tests, with the
# way they were obtained in REVIEW.md.
using Test
include(joinpath(@__DIR__, "common.jl"))
using JDyn: mrun, MatlabError, Unsupported, ModelError

run_m(code) = mrun(code)
"The workspace value r of a MATLAB snippet, or the exception it raised."
val(code) = try
    run_m(code)["r"]
catch e
    e
end
"Does `code` raise a MATLAB-level error (MatlabError) or a refusal (Unsupported), not a Julia exception?"
clean_error(code) = try
    run_m(code); false
catch e
    e isa MatlabError || e isa Unsupported
end

@testset "review findings (all expected to fail today)" begin

    # ------------------------------------------------------------------ R01 steady state of unit-root models
    # y = y(-1) + e: the static equation y = y is satisfied by every y. Dynare 7.2 (Octave) keeps y at its
    # initval (0.2) and moves only c; JDyn moves y as well (min-norm Newton steps on a singular Jacobian).
    @testset "R01 unit-root model keeps initval where the static equation is an identity" begin
        m = build("""
            var y c; varexo e; parameters gam; gam = 0.3;
            model; y = y(-1) + e; c = 0.5*c(+1) + gam*exp(y); end;
            initval; y = 0.2; c = 1; end;
            shocks; var e = 0.01; end;
            steady;
            """)
        ys, _ = steady_state(m)
        @test isapprox(ys, [0.2, 0.7328416548961019]; rtol=1e-8)        # oo_.steady_state of Dynare 7.2
        sol = first_order(m)
        @test isapprox(vec(ghx(sol)), [1.0, 0.7328416548961019]; rtol=1e-8)   # oo_.dr.ghx of Dynare 7.2
    end

    # ------------------------------------------------------------------ R02 class of logical arrays after indexed assignment
    @testset "R02 logical arrays stay logical under indexed assignment and deletion" begin
        # MATLAB and Octave 8.4: l(2) = 5 gives the logical array [1 1 1]
        @test_broken val("l = [true false true]; l(2) = 5; r = l;") isa Matrix{Bool}
        @test_broken val("l = [true false true]; l(end) = []; r = l;") isa Matrix{Bool}
        # so that it still is a mask: a(l) selects the second element (JDyn: 'Index ... 0' error)
        @test_broken val("l = false(1,3); l(2) = 1; a = [10 20 30]; r = a(l);") == 20.0
        @test_broken val("l = [true false true]; l(end) = []; a = [10 20 30]; r = a(l);") == 10.0
    end

    # ------------------------------------------------------------------ R03 shape of A(mask) for a row mask on a matrix
    @testset "R03 A(logical row) on a matrix is a row" begin
        r = val("A = [1 2 3; 4 5 6]; r = A(logical([1 0 1]));")
        @test_broken r isa AbstractMatrix && size(r) == (1, 2)      # Octave 8.4 and MATLAB: 1x2 [1 2]; JDyn: 2x1
    end

    # ------------------------------------------------------------------ R04 any/all with NaN
    @testset "R04 any/all accept NaN" begin
        @test_broken val("r = any([1 NaN 3]);") === true       # MATLAB/Octave: 1 (JDyn: MatlabError)
        @test_broken val("r = any(NaN);") === false            # any ignores NaN
        @test_broken val("r = all([1 NaN]);") === true
    end

    # ------------------------------------------------------------------ R19 scalar broadcast into a colon over an array with no rows
    @testset "R19 A(i,:) = scalar on an array with zero rows fills the whole row" begin
        # Octave 8.4 (and MATLAB): zeros(0,3); r(2,:) = 5  ->  [0 0 0; 5 5 5]; JDyn: [0 0 0; 5 0 0]
        @test_broken val("r = zeros(0,3); r(2,:) = 5;") == [0.0 0 0; 5 5 5]
        @test_broken val("r = zeros(0,3); r(end+1,:) = 7;") == [7.0 7 7]
        @test_broken val("a = [3 1 4]; r = zeros(0,3); r(a>2,:) = true;") == [1.0 1 1; 0 0 0; 1 1 1]
        @test_broken val("r = zeros(2,0); r(:,2) = 5;") == [0.0 5; 0 5]
    end

    # ------------------------------------------------------------------ R05 shape rules for growing and assigning to empty arrays
    @testset "R05 growing an array with an empty dimension keeps that dimension" begin
        @test_broken size(val("r = zeros(0,3); r(:,10) = 5;")) == (0, 10)    # Octave: 0x10 (JDyn: 1x10 with a 5)
        @test_broken size(val("r = zeros(3,0); r(2,:) = 1;")) == (3, 0)      # Octave: 3x0 (JDyn: 3x1)
    end

    # ------------------------------------------------------------------ R06 deletion with colon subscripts
    @testset "R06 A(:,:) = [] and A(2:end,[]) = [] are legal" begin
        @test_broken val("r = [1 2 3]; r(:,:) = [];") isa AbstractMatrix && size(val("r = [1 2 3]; r(:,:) = [];")) == (0, 3)
        @test_broken size(val("r = [1;2;3]; r(:,:) = [];")) == (0, 1)
        @test_broken val("r = [1 2 3]; r(2:end,[]) = [];") == [1.0 2.0 3.0]
    end

    # ------------------------------------------------------------------ R07 horizontal concatenation of incompatible empties
    @testset "R07 [1 2 3, zeros(2,0)] is an error (Octave: horizontal dimensions mismatch (1x3 vs 2x0))" begin
        @test_broken val("r = [1 2 3, zeros(2,0)];") isa MatlabError
    end

    # ------------------------------------------------------------------ R08 Julia exceptions leak out of the interpreter
    @testset "R08 builtins raise MatlabError, not Julia exceptions" begin
        @test_broken clean_error("x = trace(ones(2,3));")            # DimensionMismatch (MATLAB: 'Input matrix must be square')
        @test_broken !(val("r = sprintf('%c', 65.5);") isa Exception)  # InexactError (Octave: '65.5')
        # and a try/catch around such a call must catch what MATLAB would catch
        @test_broken val("try, x = trace(ones(2,3)); r = -1; catch, r = -2; end") in (-1.0, -2.0)
    end

    # ------------------------------------------------------------------ R09 sprintf with numeric conversions applied to char data
    @testset "R09 sprintf('%d', 'abc') prints the character codes" begin
        @test_broken val("r = sprintf('%d', 'abc');") == "979899"        # MATLAB and Octave
        @test_broken val("r = sprintf('%-4.3g|', 'ab');") == "97  |98  |"
        @test_broken val("r = sprintf('%ld', 5);") == "5"                # '%ld' is left as text by JDyn
    end

    # ------------------------------------------------------------------ R10 scalar ^ matrix
    @testset "R10 scalar^matrix" begin
        # Octave 8.4: 2^[1 2;3 4]
        want = [10.4827393896286 14.1518788283206; 21.227818242481 31.7105576321096]
        got = val("r = 2 ^ [1 2; 3 4];")
        @test_broken got isa AbstractMatrix && isapprox(got, want; rtol=1e-10)
    end

    # ------------------------------------------------------------------ R11 set_shock_stderr! drops the covariances
    @testset "R11 set_shock_stderr! changes the variance only, as set_shock_stderr_value" begin
        m = build("""
            var y x; varexo e u; parameters a; a = 0.5;
            model; y = a*y(-1) + e; x = 0.3*x(-1) + u; end;
            shocks; var e = 1; var u = 4; corr e,u = 0.5; end;
            """)
        set_shock_stderr!(m, "u", 3.0)
        # Dynare 7.2 (Octave) after set_shock_stderr_value('u', 3): Sigma_e = [1 1; 1 9]
        @test m.Sigma_e[1, 2] == 1.0 && m.Sigma_e[2, 1] == 1.0 && m.Sigma_e[2, 2] == 9.0
        # the same call from MATLAB code in the .mod file does keep the covariance (this part is right):
        m2 = build("""
            var y x; varexo e u; parameters a; a = 0.5;
            model; y = a*y(-1) + e; x = 0.3*x(-1) + u; end;
            shocks; var e = 1; var u = 4; corr e,u = 0.5; end;
            set_shock_stderr_value('u', 3);
            """)
        @test m2.Sigma_e[1, 2] == 1.0
    end

    # ------------------------------------------------------------------ R12 set_params! is not atomic
    @testset "R12 set_params! with an unknown name leaves the parameters unchanged" begin
        m = build("""
            var y; varexo e; parameters alpha; alpha = 0.5;
            model; y = alpha*y(-1) + e; end;
            """)
        try
            set_params!(m, ["alpha" => 0.9, "nonexistent" => 1.0])
        catch
        end
        @test get_param(m, "alpha") == 0.5
    end

    # ------------------------------------------------------------------ R13 steady_state_model calling a Julia Base function
    @testset "R13 steady_state_model: y = exit(0) must not end the Julia process" begin
        modtext = "var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; end;\nsteady_state_model; y = exit(3); end;\nsteady;\n"
        d = mktempdir(); write(joinpath(d, "ex.mod"), modtext)
        script = """
            using JDyn
            ENV["DYNARE_PREPROCESSOR"] = "$(ENV["DYNARE_PREPROCESSOR"])"
            m = JDyn.load("$(joinpath(d, "ex.mod"))")
            try; steady_state(m); catch; end
            println("SURVIVED")
            """
        out = IOBuffer()
        proc = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) -e $script`); stdout=out, stderr=devnull))
        @test_broken occursin("SURVIVED", String(take!(out)))
    end

    # ------------------------------------------------------------------ R14 raw Julia exception from stoch_simul_moments
    @testset "R14 negative variance (root inside the QZ criterium) is not a raw DomainError" begin
        m = build("""
            var y; varexo e; parameters rho; rho = 1.0000005;
            model; y = rho*y(-1) + e; end;
            shocks; var e = 1; end;
            steady;
            stoch_simul(order=1, irf=0, nograph);
            """)
        # Dynare 7.2 solves it (root 1.0000005 < 1 + 1e-6) and prints variance -1e6; JDyn: DomainError from sqrt
        r = try
            stoch_simul_moments(m); nothing
        catch e
            e
        end
        @test !(r isa DomainError)
    end

    # ------------------------------------------------------------------ R15 spurious warning
    @testset "R15 irf_plot_threshold does not make stoch_simul_moments warn" begin
        logger = Test.TestLogger(; min_level=Logging.Warn)
        with_logger(logger) do
            m = build("""
                var y; varexo e; parameters rho; rho = 0.9;
                model; y = rho*y(-1) + e; end;
                shocks; var e = 1; end;
                steady;
                stoch_simul(order=1, irf=5, irf_plot_threshold=0.1, nograph);
                """; quiet=false)
            stoch_simul_moments(m)
        end
        @test_broken !any(l -> occursin("plot_threshold", string(l.message)), logger.logs)
    end

    # ------------------------------------------------------------------ R16 IRFs at the default order 2 are first-order without a word
    @testset "R16 irf_figures warns when stoch_simul asks for order >= 2" begin
        logger = Test.TestLogger(; min_level=Logging.Warn)
        res = with_logger(logger) do
            m = build("""
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
                stoch_simul(irf=4, nograph);
                """; quiet=false)           # no order: Dynare 7's default is 2
            JDyn.irf_figures(m)
        end
        @test res.order == 1                                  # (what is documented in the docstring)
        @test any(l -> occursin(r"order", string(l.message)), logger.logs)
    end

    # ------------------------------------------------------------------ R17 load() cannot pass macro variables
    @testset "R17 JDyn.load forwards extra_args (-D...) to the preprocessor" begin
        text = "@#ifndef A\n@#define A = 1\n@#endif\nvar y; varexo e; parameters a; a = 0.1*@{A};\nmodel; y = a*y(-1) + e; end;\n"
        d = mktempdir(); write(joinpath(d, "mac.mod"), text)
        m = try
            with_logger(NullLogger()) do
                JDyn.load(joinpath(d, "mac.mod"); extra_args=["-DA=2"])
            end
        catch e
            e
        end
        @test m isa JDyn.Model && isapprox(m.params[1], 0.2)
    end

    # ------------------------------------------------------------------ R18 refusals that are raw Julia exceptions
    @testset "R18 unsupported models are refused with a ModelError" begin
        ef = joinpath(DYNARE_SRC, "tests", "external_function", "first_and_2nd_deriv_given.mod")
        if isfile(ef)
            r = try
                load_copy(ef; extra=filter(f -> endswith(f, ".m"), readdir(dirname(ef); join=true)))
                nothing
            catch e
                e
            end
            @test r isa ModelError         # JDyn: LoadError: syntax: invalid assignment location "[TEF_0, TEFD_0, TEFDD_0]"
        end
        dts = joinpath(DYNARE_SRC, "tests", "deterministic_simulations", "ramst_dseries.mod")
        if isfile(dts)
            r = try
                load_copy(dts)
                nothing
            catch e
                e
            end
            @test r === nothing || r isa ModelError    # JDyn: ArgumentError: cannot parse "dates('2000Q2')" as Float64
        end
    end

    # ------------------------------------------------------------------ R20 unique treats NaNs as distinct
    @testset "R20 unique([NaN NaN]) has two elements" begin
        r = val("r = unique([NaN NaN]);")       # MATLAB (R2013a and later) and Octave 8.4: [NaN NaN]
        @test_broken r isa AbstractMatrix && size(r) == (1, 2)
    end

    # ------------------------------------------------------------------ R21 k_order refuses unused uninitialised parameters
    @testset "R21 k_order on a model with an unused, uninitialised parameter" begin
        text = """
            var c k a; varexo e; parameters alpha beta rho unused;
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
        m = build(text; output="second")
        @test first_order(m) isa JDyn.FirstOrderSolution           # first order is fine, as in Dynare 7.2
        r = try
            k_order(m; order=2)
        catch e
            e
        end
        @test r isa AbstractDict                             # JDyn: KOrderError "steady state, parameters or Sigma_e contain NaN or Inf"
    end
end
