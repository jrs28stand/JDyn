# JDyn's MATLAB subset: MATLAB code in .mod files, steady-state files and helper functions
# (port of pydyn's tests/test_matlab.py).
#
# Expected values come from Octave 8.4 and Dynare 7.2 (pydyn's reference/r_matlab.m, tools/sweep_matlab.py):
# - mat_semantics.mat: value of `r` after each snippet of mat_cases.txt;
# - mat_flow.mat: M_.params, M_.Sigma_e and workspace variables after `dynare mat_flow`;
# - mat_corpus.mat: M_.params, M_.Sigma_e and oo_.steady_state at the end of Dynare's run of .mod files of
#   the Dynare test suite (the files themselves are read from the Dynare source tree).
#
# JDYN_MATLAB_ONLY=semantics,mod,ss,review,corpus runs a subset (comma-separated).
using Test
include("common.jl")
using JDyn: MatlabError, Unsupported, ParseError, ModelError, SteadyStateError, mrun

const ONLY = split(get(ENV, "JDYN_MATLAB_ONLY", "semantics,mod,ss,review,corpus"), ",")
section(name) = name in ONLY

"Run MATLAB text without a model; returns the workspace."
run_m(code; variables=Dict{String,Any}(), directory=nothing) = mrun(code; variables=variables, directory=directory)

as_matrix(v::AbstractString) = mat(v)
as_matrix(v::Real) = fill(Float64(v), 1, 1)
as_matrix(v::AbstractMatrix) = Matrix{Float64}(v)

"Same shape, |a-b| <= rtol |b| elementwise, NaN equal to NaN, equal infinities."
function close_nan(a, b; rtol=1e-13, atol=0.0)
    a, b = as_matrix(a), as_matrix(b)
    size(a) == size(b) || return false
    all(zip(a, b)) do (x, y)
        (isnan(x) && isnan(y)) || x == y || (isfinite(x) && isfinite(y) && abs(x - y) <= atol + rtol * abs(y))
    end
end

"Throws an exception of type T whose message matches `pat` (if given)."
function throws_match(f, T, pat=nothing)
    try
        f()
    catch e
        e isa T || (@info "unexpected exception" e; return false)
        pat === nothing && return true
        msg = sprint(showerror, e)
        occursin(pat isa Regex ? pat : Regex(pat), msg) || (@info "message does not match" pat msg; return false)
        return true
    end
    @info "no exception"
    false
end

"Build with all warnings captured: returns (model, list of warning messages)."
function build_logged(text; kw...)
    logger = Test.TestLogger(; min_level=Logging.Warn)
    m = with_logger(logger) do
        build(text; quiet=false, kw...)
    end
    m, [string(r.message) for r in logger.logs if r.level >= Logging.Warn]
end

workspace(m) = JDyn.workspace(m)
skipped_kinds(m) = [k for (_, k, _) in m.matlab_skipped]

# ====================================================================== (a) semantics against Octave
if section("semantics")
@testset "semantics" begin
    CASES = split(read(joinpath(REF, "mat_cases.txt"), String), "\n%%\n")
    EXPECTED = vec(matread(joinpath(REF, "mat_semantics.mat"))["results"])

    @testset "case file and reference agree" begin
        @test length(CASES) == length(EXPECTED) >= 60
        @test all(e -> !(e isa AbstractString), EXPECTED)
    end

    @testset "semantics against Octave" begin
        nfail = 0
        for k in eachindex(CASES)
            ok = try
                got = as_matrix(run_m(String(CASES[k]))["r"])
                want = as_matrix(EXPECTED[k] isa Bool ? Float64(EXPECTED[k]) : EXPECTED[k])
                close_nan(got, want)
            catch e
                @info "case $k raised" CASES[k] e
                false
            end
            ok || (nfail += 1; @info "case $k differs" CASES[k])
            @test ok
        end
    end

    @testset "MATLAB-specific behaviour" begin
        @test run_m("r = sprintf('%d', 1.5);")["r"] == "1.500000e+00"       # Octave prints 1.5
        @test run_m("r = isoctave;")["r"] == false
        @test_throws Unsupported run_m("r = [\"a\" \"b\"];")             # "..." strings are string objects in MATLAB
    end

    @testset "complex results are refused, not MATLAB errors" begin
        for code in ["r = sqrt(-1);", "r = (-8)^(1/3);", "r = log(-1);", "r = acos(2);", "r = [4 -4].^0.5;"]
            @test throws_match(() -> run_m(code), Unsupported, "complex")
            @test throws_match(() -> run_m("try, " * code * " catch, r = 5; end"), Unsupported, "complex")
        end
    end

    @testset "indeterminate results are NaN as in MATLAB" begin
        ws = run_m("a = 0/0; b = Inf - Inf; c = 0*Inf; d = sin(Inf); e1 = 1/0; f = 10^400; g = -1/0; h = mod(5, 0);")
        @test all(k -> isnan(ws[k]), ["a", "b", "c", "d"])
        @test ws["e1"] == Inf && ws["f"] == Inf && ws["g"] == -Inf
        @test run_m("try, x = 0/0; k = 5; catch, k = 6; end")["k"] == 5
    end

    @testset "errors and refusals" begin
        for (code, exc) in [
                ("a = [1 2 3]; r = a(4);", MatlabError), ("a = [1 2 3]; r = a(0);", MatlabError),
                ("a = [1 2 3]; r = a(1.5);", MatlabError), ("r = [1 2] + [1 2 3];", MatlabError),
                ("r = [1 2; 3];", MatlabError), ("r = [1 2 3] * [1 2 3];", MatlabError),
                ("r = undefined_thing + 1;", Unsupported), ("r = rand(2);", Unsupported),
                ("r = 1i;", Unsupported), ("x = dseries(1);", Unsupported), ("r = evalin('base', '1');", Unsupported),
                ("r = system('ls');", Unsupported), ("fid = fopen('x.txt', 'w');", Unsupported),
                ("s = struct('a', {1, 2});", Unsupported), ("a = zeros(2, 2, 2);", Unsupported),
                ("if [1 2] && 1, r = 1; end", MatlabError), ("error('my:id', 'value %d', 4)", MatlabError)]
            @test throws_match(() -> run_m(code), exc) || (@info code; false)
        end
    end

    @testset "no access to Julia from MATLAB text" begin
        # names resolve only against MATLAB variables and the built-in table; MATLAB text never reaches Julia's eval
        for code in ["r = __import__('os');", "r = open('/etc/passwd');", "r = eval('__import__(''os'').getcwd()');",
                     "x.__class__ = 1;", "r = getattr(1, 'real');", "r = exec('1');", "r = globals();",
                     "r = Base.run(1);", "r = Core.eval(1);", "r = eval('Main.x');", "r = include('x.jl');",
                     "r = ccall(1);", "r = getfield(Main, 'x');", "r = read('/etc/passwd');"]
            @test (try
                run_m(code); false
            catch e
                e isa Unsupported || e isa MatlabError || e isa ParseError
            end) || (@info code; false)
        end
    end

    @testset "endless loop is stopped" begin
        old = JDyn.MAX_STEPS[]
        JDyn.MAX_STEPS[] = 5000
        try
            @test throws_match(() -> run_m("k = 0; while true, k = k + 1; end"), Unsupported, "step limit")
        finally
            JDyn.MAX_STEPS[] = old
        end
    end

    @testset "function files and scripts" begin
        d = mktempdir()
        write(joinpath(d, "addmul.m"), "function [s, p] = addmul(a, b)\n% help text\ns = a + b;\nif nargout > 1\n  p = helper(a) * b;\nend\nend\n" *
                                       "function y = helper(x)\n  y = x;\nend\n")
        write(joinpath(d, "setup_script.m"), "zz = 4;\nww = zz * 2;\n")
        ws = run_m("[s, p] = addmul(2, 3); t = addmul(1, 1); setup_script; q = ww + zz;"; directory=d)
        @test (ws["s"], ws["p"], ws["t"], ws["q"]) == (5.0, 6.0, 2.0, 12.0)
    end

    @testset "review A10 builtins" begin
        ws = run_m("a = str2double({'0.5','0.7'}); b = str2double('1,000'); c = round(0.49999999999999994); " *
                   "d = num2str(2^53); e1 = sprintf('%s', 65); f = round(-2.5); g = mod(10, 0.1) + mod(0.3, 0.1) + rem(1, 0.1);")
        @test ws["a"] == [0.5 0.7]
        @test (ws["b"], ws["c"], ws["d"], ws["e1"], ws["f"], ws["g"]) == (1000.0, 0.0, "9007199254740992", "A", -3.0, 0.0)
    end
end
end

# ====================================================================== (b) .mod files
const FLOW = "var c; varexo e; parameters rho s; rho=0.5; s=2;\ncalib = CALIB;\nif calib == 1\n rho = 0.9; s = 5;\nend\n" *
             "model; c = rho*c(-1) + s*e; end;"
flow(calib) = replace(FLOW, "CALIB" => string(calib))

"Build, ignoring warnings, and expect a ModelError whose message matches `pat`."
refused(text; pat=nothing, kw...) = throws_match(() -> build(text; kw...), ModelError, pat)

if section("mod")
@testset ".mod files" begin
    @testset "control flow with Dynare statements against Dynare" begin
        # if/for/while/switch/try around parameter assignments, set_param_value, M_.params(i), M_.Sigma_e,
        # function handles, fzero: same calibration as Dynare (mat_flow.mod, run under Octave)
        m, warns = build_logged(read(joinpath(REF, "mat_flow.mod"), String); name="mat_flow")
        @test isempty(warns)                                   # nothing is skipped in this file
        r = refmat("mat_flow.mat")
        @test m.param_names == split(r["param_names"], ",")
        @test isapprox(m.params, vec(r["params"]); rtol=1e-12)
        @test isapprox(m.Sigma_e, r["Sigma_e"]; rtol=1e-12)
        ws = workspace(m)
        @test M.get_param(m, "s") == 5.0 && m.session.sc.vars["s"] == 99.0     # one-line if: workspace only
        for name in ("x1", "x2", "x3", "k0")
            @test isapprox(ws[name], r[name]; rtol=1e-12)
        end
        @test isapprox(vec(m.session.sc.vars["w"]), vec(r["w"]); rtol=1e-12)
        @test m.matlab_options["irf"] == 7 && isempty(m.matlab_skipped)
        # the stale workspace copy: b1 was changed through M_.params(6), the MATLAB variable b1 was not
        @test ws["x1"] == ws["x2"] != M.get_param(m, "b1")
    end

    @testset "MATLAB control flow is executed" begin
        for (calib, rho, s) in [(0, 0.5, 2.0), (1, 0.9, 5.0)]
            m = build(flow(calib))
            @test (M.get_param(m, "rho"), M.get_param(m, "s")) == (rho, s)
        end
    end

    @testset "block with unknown condition around parameters is an error" begin
        @test refused(flow("some_function_jdyn_does_not_have(3)"); pat="param_init")
    end

    @testset "unknown condition without calibration is skipped with one warning" begin
        text = "var c; varexo e; parameters rho; rho=0.5;\nmodel; c = rho*c(-1) + e; end;\nshocks; var e; stderr 1; end;\n" *
               "stoch_simul(order=1, irf=5);\nfigure; plot(c_e); title('irf');\n" *
               "if max(abs(oo_.dr.ghx)) > 1\n  error('unstable');\nend\nx = oo_.dr.ys(1) + 1;\ny = 2*x;\ndisp(y)\n"
        m, warns = build_logged(text)
        @test any(w -> occursin("statement", w), warns)
        @test count(w -> occursin("MATLAB code", w), warns) == 1
        kinds = skipped_kinds(m)
        @test "plot" in kinds && "block" in kinds && "results" in kinds && "tainted" in kinds
        @test M.get_param(m, "rho") == 0.5 && !haskey(workspace(m), "y")
    end

    @testset "code that changes the model and cannot be followed is an error" begin
        for (line, pat) in [
                ("M_.params(1) = oo_.dr.ghx(1);", "assigns to M_"),
                ("set_param_value('rho', some_unknown_function(1));", "set_param_value"),
                ("M_.Sigma_e = cov(randn(10, 1));", "assigns to M_"),
                ("M_.H = 1;", "assigns to M_"),
                ("oo_.steady_state(1) = 3;", "oo_.steady_state"),
                ("eval(some_unknown_function(1));", "eval"),
                ("x = some_unknown_function(1);\nrho = x;", "could not compute"),
                ("x = rand;\nshocks; var e; stderr x; end;", "could not compute"),
                ("error('calibration is inconsistent');", "calibration is inconsistent"),
                ("M_.params(5) = 1;", "must not change size|out of bound"),
                ("load_params_from_somewhere;\nif exist('q', 'var')\n rho = 0.1;\nend", "param_init")]
            text = "var c; varexo e; parameters rho; rho=0.5;\nmodel; c = rho*c(-1) + e; end;\n" * line * "\n"
            @test refused(text; pat=pat) || (@info line; false)
        end
    end

    @testset "parameters set by the steady state are seen by later statements" begin
        # after `steady`, Dynare's M_.params holds the values set by the steady_state_model block
        base = "var c; varexo e; parameters rho cbar; rho=0.5;\nmodel; c = cbar + rho*(c(-1)-cbar) + e; end;\n" *
               "steady_state_model; cbar = 2*rho; c = cbar; end;\nsteady;\n"
        m = build(base * "set_param_value('rho', get_param_by_name('cbar')/4);\n")
        @test (M.get_param(m, "rho"), M.get_param(m, "cbar")) == (0.25, 1.0)
        m = build(base * "set_param_value('rho', get_param_by_name('rho')/2);\n"; name="m2")
        @test M.get_param(m, "rho") == 0.25
    end

    @testset "try block with unknown statement is not guessed" begin
        # a statement JDyn cannot run inside try might have raised in MATLAB: the block is undecided
        text = "var c; varexo e; parameters rho; rho=0.5;\nmodel; c = rho*c(-1) + e; end;\nflag = 0;\n" *
               "try\n  some_unknown_function(1);\n  flag = 1;\ncatch\n  flag = 2;\nend\n"
        m, warns = build_logged(text)
        @test !isempty(warns)
        @test !haskey(workspace(m), "flag")
        @test refused(text * "rho = flag;\n"; name="m2")
    end

    @testset "load of a .mat file next to the .mod file" begin
        d = mktempdir()
        matwrite(joinpath(d, "calib.mat"), Dict("rho_data" => 0.7, "vec" => [1.0 2.0 3.0]))
        m = with_logger(NullLogger()) do
            write(joinpath(d, "m.mod"), "var c; varexo e; parameters rho s;\nload calib\nrho = rho_data; s = vec(3);\n" *
                                        "model; c = rho*c(-1) + s*e; end;")
            M.load(joinpath(d, "m.mod"))
        end
        @test (M.get_param(m, "rho"), M.get_param(m, "s")) == (0.7, 3.0)
        # a file that does not exist yet: the variables are unknown
        @test refused("var c; varexo e; parameters rho;\nload not_there\nrho = rho_data;\nmodel; c = rho*c(-1) + e; end;"; name="m2")
    end

    @testset "script and function next to the .mod file" begin
        files = Dict("calibration.m" => "rho_s = 0.3;\nset_param_value('s', 4);\n", "twice.m" => "function y = twice(x)\ny = 2*x;\n")
        m = build("var c; varexo e; parameters rho s; s = 1;\ncalibration;\nrho = twice(rho_s);\nmodel; c = rho*c(-1) + s*e; end;";
                  files=files)
        @test (M.get_param(m, "rho"), M.get_param(m, "s")) == (0.6, 4.0)
    end
end
end
