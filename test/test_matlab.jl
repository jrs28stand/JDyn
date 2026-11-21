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
as_matrix(v::AbstractVector) = reshape(Float64.(v), :, 1)

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

    @testset "extra semantics against Octave (JDyn's own cases)" begin
        # test/matlab_extra_cases.txt, reference test/matlab_extra.mat from test/r_matlab_extra.m (Octave 8.4).
        # A case must give Octave's value, or be refused (Unsupported); a case that fails in Octave must fail.
        xcases = split(read(joinpath(@__DIR__, "matlab_extra_cases.txt"), String), "\n%%\n")
        xexp = vec(matread(joinpath(@__DIR__, "matlab_extra.mat"))["results"])
        @test length(xcases) == length(xexp)
        nref = 0
        for k in eachindex(xcases)
            want = xexp[k]
            got = try
                run_m(String(xcases[k]))["r"]
            catch e
                e
            end
            ok = if want isa AbstractString && startswith(want, "ERROR")
                got isa MatlabError || got isa Unsupported
            elseif got isa Unsupported
                nref += 1
                true
            elseif got isa Exception || want isa Complex || (want isa AbstractArray && eltype(want) <: Complex)
                false
            else
                close_nan(as_matrix(got), as_matrix(want isa Bool ? Float64(want) : want))
            end
            ok || @info "extra case $k" xcases[k] got want
            @test ok
        end
        println("  extra semantics: $(length(xcases)) cases, $nref refused (Unsupported)")
    end

    @testset "MATLAB-specific behaviour" begin
        @test run_m("r = sprintf('%d', 1.5);")["r"] == "1.500000e+00"       # Octave prints 1.5
        @test run_m("r = sprintf('%s', 3.5);")["r"] == "3.500000e+00"       # MATLAB: %e for a non-integer with %s
        @test run_m("r = sprintf('%d\\n');")["r"] == "\n"                    # Octave returns ''
        @test_throws MatlabError run_m("r = 3 && [];")                       # Octave: false
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

# ====================================================================== steady-state files
nk(; kw...) = load_copy(joinpath(SS, "nk_baseline.mod"); extra=[joinpath(SS, "nk_baseline_steadystate.m")], kw...)

const SSMODEL = "var c k; varexo e; parameters rho cbar kbar;\nrho = 0.5; cbar = 2;\n" *
                "model; c = cbar + rho*(c(-1)-cbar) + e; k = kbar + rho*(k(-1)-kbar); end;\n"

const OLD_STYLE = """function [ys, check] = old_steadystate(ys, exo)
% steady-state file in the old two-output form: parameters come from the global M_
global M_
check = 0;
rho = M_.params(1);
cbar = M_.params(2);
if rho >= 1
    check = 1;
    return
end
M_.params(3) = 3*cbar;
ys = [cbar; M_.params(3)];
"""

if section("ss")
@testset "steady-state files and helpers" begin
    @testset "nk_baseline with its MATLAB steady-state file" begin
        m = nk()
        r = refmat("clean_nk_baseline.mat")
        sol = M.first_order(m)
        @test isapprox(sol.ys, vec(r["ys"]); rtol=1e-8, atol=1e-10)
        @test isapprox(sol.params, vec(r["params"]); rtol=1e-10)
        @test isapprox(M.ghx(sol), r["ghx"]; rtol=1e-6, atol=1e-8)
        @test isapprox(M.ghu(sol), r["ghu"]; rtol=1e-6, atol=1e-8)
        M.set_param!(m, "betta", 0.995)                       # the file updates Rbar
        sol_b = M.first_order(m)
        @test isapprox(sol_b.params, vec(r["params_b"]); rtol=1e-10)
        @test isapprox(sol_b.ys, vec(r["ys_b"]); rtol=1e-8, atol=1e-10)
        @test isapprox(M.ghx(sol_b), r["ghx_b"]; rtol=1e-6, atol=1e-8)
    end

    @testset "steady-state file: parameters it sets are loaded" begin
        m = nk()
        ys, params = M.steady_state(m)
        @test maximum(abs, M.static_resid(m, ys, nothing, params)) < 1e-10
        # like Dynare's M_.params after `steady`, the loaded parameters include those the file sets
        @test M.get_param(m, "Rbar") == params[m.ip["Rbar"]] && isfinite(M.get_param(m, "Rbar"))
    end

    @testset "steady-state file failure flag" begin
        m = nk()
        M.set_param!(m, "betta", 50.0)                        # gammma1 < 0: the file sets check = 1 and returns
        @test throws_match(() -> M.steady_state(m), SteadyStateError, "signalled failure")
    end

    @testset "steady-state file timing (interpreted)" begin
        m = nk()
        fn, x = M.steady_state_file_fn(m), zeros(m.n_exo)
        fn(m, copy(m.params), x)
        t = @elapsed for _ in 1:20
            fn(m, copy(m.params), x)
        end
        println("  nk_baseline steady state per call (interpreted): $(round(Int, t / 20 * 1e6)) us")
        @test t / 20 < 0.05
    end

    @testset "user functions take precedence" begin
        calls = Ref(0)
        inner = M.steady_state_file_fn(nk())
        fn(model, params, exo) = (calls[] += 1; inner(model, params, exo))
        m = nk(; steady_state_fn=fn)
        M.steady_state(m)
        @test calls[] > 0
        m2 = load_copy(joinpath(SS, "collard_2001_analytical_steady_state.mod");
                       extra=[joinpath(SS, "collard_2001_analytical_steady_state_helper.m")],
                       functions=Dict("collard_2001_analytical_steady_state_helper" => (a...) -> 0.25))
        @test isempty(m2.matlab_functions)
        ys, _ = M.steady_state(m2; check=false)
        @test ys[m2.ie["h"]] == 0.25
    end

    @testset "helper function file in steady_state_model" begin
        # collard_2001_analytical_steady_state with its original .m helper (fsolve inside)
        m = load_copy(joinpath(SS, "collard_2001_analytical_steady_state.mod");
                      extra=[joinpath(SS, "collard_2001_analytical_steady_state_helper.m")])
        @test collect(keys(m.matlab_functions)) == ["collard_2001_analytical_steady_state_helper"]
        r = refmat("clean_collard_2001_analytical_steady_state.mat")
        sol = M.first_order(m)
        @test isapprox(m.Sigma_e, r["Sigma_e"]; rtol=1e-12)
        # Dynare's fsolve stops at TolFun = 1e-10; JDyn solves the helper's equation to machine precision
        @test isapprox(sol.ys, vec(r["ys"]); rtol=1e-7, atol=1e-9)
        at_ref = M.first_order(m; ys=vec(r["ys"]), params=vec(r["params"]))
        @test isapprox(M.ghx(at_ref), r["ghx"]; rtol=1e-9, atol=1e-11)
        @test isapprox(M.ghu(at_ref), r["ghu"]; rtol=1e-9, atol=1e-11)
    end

    @testset "helper with several outputs" begin
        files = Dict("two_out.m" => "function [a, b, info] = two_out(x)\ninfo = 0;\na = x;\nif x > 1\n  b = 3*x;\nelse\n  b = -1;\nend\n")
        m = build(SSMODEL * "steady_state_model;\n[c, kk, info] = two_out(cbar);\nkbar = kk;\nk = kbar;\nend;\n"; files=files)
        ys, params = M.steady_state(m)
        @test ys == [2.0, 6.0]
        @test params[m.ip["kbar"]] == 6.0 && haskey(m.matlab_functions, "two_out")
    end

    @testset "old two-output steady-state file is an error as in Dynare" begin
        # Dynare 7.2 calls [ys, params, info] = f(ys, exo, M_, options_); the old [ys, check] = f(ys, exo)
        # fails there ("called with too many inputs"), and here too
        m = build(SSMODEL; name="old", files=Dict("old_steadystate.m" => OLD_STYLE))
        @test throws_match(() -> M.steady_state(m), SteadyStateError, "too many")
    end

    @testset "steady-state file that cannot be interpreted is reported" begin
        bad = "function [ys, params, check] = bad_steadystate(ys, exo, M_, options_)\nparams = M_.params;\ncheck = 0;\nys = some_toolbox_function(ys);\n"
        m = build(SSMODEL; name="bad", files=Dict("bad_steadystate.m" => bad))
        @test throws_match(() -> M.steady_state(m), SteadyStateError, "bad_steadystate.m cannot be interpreted")
    end
end
end

# ====================================================================== independent review (findings A1-A11, B1-B6)
# Inputs are the reviewer's cases; expected values are those Dynare 7.2 printed under Octave 8.4
# (M_.params as [p q r] unless stated). REFUSED means JDyn must raise ModelError.
const RV_HEAD = "var y c; varexo e u; parameters p q r;\n" *
                "model; y = 0.5*y(-1)+e+0*(p+q+r); c = 0.3*c(-1) + u; end;\n"
const RV_P = "p=1;q=2;r=3;\n"
const RV_RBC = "var y c k; varexo e; parameters alpha beta delta A;\n" *
               "model; 1/c = beta/c(+1)*(alpha*A*k^(alpha-1)+1-delta); y = A*exp(e)*k(-1)^alpha; k = y - c + (1-delta)*k(-1); end;\n" *
               "alpha = 0.33; beta = 0.99; delta = 0.025; A = 1;\n"
const RV_SSM_A = "steady_state_model;\nk = 10;\nA = (1/beta - 1 + delta)/(alpha*k^(alpha-1));\ny = A*k^alpha;\nc = y - delta*k;\nend;\n"
const RV_SSM = "steady_state_model;\nk = ((1/beta - 1 + delta)/(alpha*A))^(1/(alpha-1));\ny = A*k^alpha;\nc = y - delta*k;\nend;\n"
rv_ssfile(name, body) = """function [ys,params,check] = $(name)_steadystate(ys,exo,M_,options_)
NumberOfParameters = M_.param_nbr;
for ii = 1:NumberOfParameters
  paramname = M_.param_names{ii};
  eval([ paramname ' = M_.params(' int2str(ii) ');']);
end
check = 0;
$body
params=NaN(NumberOfParameters,1);
for iter = 1:length(M_.params)
  eval([ 'params(' num2str(iter) ') = ' M_.param_names{iter} ';' ])
end
NumberOfEndogenousVariables = M_.orig_endo_nbr;
for ii = 1:NumberOfEndogenousVariables
  varname = M_.endo_names{ii};
  eval(['ys(' int2str(ii) ') = ' varname ';']);
end
end
"""
const SETGET = Dict("setget.m" => "function y = setget(x)\nglobal M_\nM_.params(1) = x;\ny = x;\nend\n")
const REFUSED = :refused

rv(body; files=Dict{String,String}(), head=RV_HEAD, name="m") = build(head * body; name=name, files=files)

const RV_CASES = Dict{String,Any}(
    # A1: one unparsable statement must not hide the other statements of its line
    "A1_u01_complex_literal" => (RV_P * "z = abs(3+4i); M_.params(1) = z;", nothing, REFUSED),          # Dynare [5 2 3]
    "A1_u10_format_long" => (RV_P * "x = 1; format long; M_.params(1) = 5;", nothing, [5, 2, 3]),
    "A1_u11_close_all" => (RV_P * "x = 1; close all; M_.params(1) = 5;", nothing, [5, 2, 3]),
    "A1_u16_hex" => (RV_P * "x = 0x10; M_.params(1) = x;", nothing, REFUSED),                            # Dynare [16 2 3]
    "A1_u05_plus_equal" => (RV_P * "x = 3; x += 1; M_.params(1) = x;", nothing, REFUSED),                # Octave only: [4 2 3]
    "A1_u14_increment" => (RV_P * "x = 3; x++; M_.params(1) = x;", nothing, REFUSED),
    "A1_u06_inside_if" => (RV_P * "if 1, x = 2i; M_.params(1) = 3; end", nothing, [3, 2, 3]),
    "A1_u02_stale_workspace" => (RV_P * "w = 1;\nx = 1; clear x; w = 2;\nM_.params(1) = w;", nothing, [2, 2, 3]),
    "A1_u07_param_copy" => (RV_P * "x = 2i; p = 3;\nM_.params(2) = p;", nothing, [1, 3, 3]),
    # A4: M_ and options_ are values
    "A4_m01_copy" => (RV_P * "M2 = M_;\nM2.params(1) = 100;\nx = M_.params(1);\nq = x;", nothing, [1, 1, 3]),
    "A4_k16_restore" => (RV_P * "M_bak = M_;\np = 9;\nM_ = M_bak;", nothing, [1, 2, 3]),
    "A4_m05_restore_after_loop" => (RV_P * "M_bak = M_;\nfor k = 1:2\n set_param_value('p', k*10);\nend\nM_ = M_bak;", nothing, [1, 2, 3]),
    "A4_m06_struct_field" => (RV_P * "s.M = M_;\np = 9;\nzz = s.M.params(1);\nM_.params(3) = zz;", nothing, [9, 2, 1]),
    "A4_m08_handle_capture" => (RV_P * "f = @() M_.params(1);\np = 9;\nzz = f();\nM_.params(3) = zz;", nothing, [9, 2, 1]),
    "A4_m16_options" => (RV_P * "opt_bak = options_;\noptions_.order = 3;\noptions_ = opt_bak;\nx = options_.order;\nq = x;", nothing, [1, 2, 3]),
    # A5: arguments of display functions are evaluated
    "A5_k30_disp" => (RV_P * "disp(setget(3));", SETGET, [3, 2, 3]),
    "A5_k31_fprintf" => (RV_P * "fprintf('%d\\n', setget(3));", SETGET, [3, 2, 3]),
    "A5_y03_warning" => (RV_P * "warning('value %d', setget(3));", SETGET, [3, 2, 3]),
    "A5_y07_display" => (RV_P * "x = 1;\ndisplay(setget(3));", SETGET, [3, 2, 3]),
    "A5_y04_evalc" => (RV_P * "x = 5;\ndisp(evalc('x = 7;'));\nM_.params(2) = x;", nothing, REFUSED),   # Dynare [1 7 3]
    # A6 / B4: NaN-producing operations are not errors
    "A6_t01_try" => (RV_P * "try\n x = 0/0;\n M_.params(1) = 5;\ncatch\n M_.params(1) = 6;\nend", nothing, [5, 2, 3]),
    "B4_t02_nan" => (RV_P * "x = 0/0;\nM_.params(1) = isnan(x);", nothing, [1, 2, 3]),
    "B4_t03_inf_minus_inf" => (RV_P * "x = Inf - Inf;\nif isnan(x)\n p = 7;\nend", nothing, [7, 2, 3]),
    "B4_u48_roundoff" => (RV_P * "x = 0.1*3; y1 = 0.3; M_.params(1) = (x == y1) + 10*(abs(x-y1) < eps) + 100*(x >= y1); " *
                          "M_.params(2) = 1e-320*1e10; M_.params(3) = realmax*10 + (-realmax*10);", nothing,
                          [110, 9.9998886718268301e-311, NaN]),
    # A7: for over an empty range empties the loop variable
    "A7_t06" => (RV_P * "k = 5;\nfor k = 1:0\nend\nif isempty(k)\n p = 7;\nelse\n p = 8;\nend", nothing, [7, 2, 3]),
    "A7_c29" => (RV_P * "k = 10;\nfor k = 1:0\n p = 5;\nend\nq = isempty(k);\nr = numel(k);", nothing, [1, 1, 0]),
    "A7_y12" => (RV_P * "k = 3;\nfor k = zeros(1,0)\nend\nM_.params(2) = numel(k);", nothing, [1, 0, 3]),
    # A8: mod with a non-integer divisor
    "A8_t04" => (RV_P * "if mod(1, 0.1) == 0\n p = 7;\nelse\n p = 8;\nend", nothing, [7, 2, 3]),
    "A8_w04" => ("x = 0.1; z = 1;\np = mod(z, x); q = 1; r = 1;", nothing, [0, 1, 1]),
    # A9: max/min with NaN in Dynare statements
    "A9_y01" => ("x = NaN;\np = max(x, 1); q = min(1, x); r = 1;", nothing, [1, 1, 1]),
    # B3: Inf in Dynare statements
    "B3_v19" => ("flag = true; x = -1; z = 0;\np = flag*2 + flag; q = (x < z) + (x == -1) + (x >= z)*10; r = 1/z;", nothing, [3, 2, Inf]),
    "B3_w02" => ("x = 10; z = -1;\np = x^400; q = 1; r = 1;", nothing, [Inf, 1, 1]),
    # B6: a parameter without a value is not a MATLAB variable
    "B6_d19" => ("p = 1;\nzz = q;\n", nothing, REFUSED),                                                 # Dynare: 'q' undefined
    # section C (relaxed refusals)
    "C_m18_row_params" => (RV_P * "M_.params = [7 8 9];", nothing, [7, 8, 9]),
    "C_k12_run" => (RV_P * "run('myscript.m');", Dict("myscript.m" => "set_param_value('p', 33);\nzz = 17;\n"), [33, 2, 3]),
    "C_a29_beta_function" => ("p = beta(2,3); q = 1; r = lgamma(5);", nothing, [1 / 12, 1, 3.1780538303479458]),
    "C_k45_options_default" => (RV_P * "x = options_.order;\nq = x;", nothing, [1, 2, 3]),
    "C_n07_resid_keeps_params" => (RV_P * "resid;\nq = p/2;", nothing, [1, 0.5, 3]),
    # second pass by the author (values from Dynare 7.2 under Octave, cases p03-p40 of the pass)
    "P2_p03_condition_side_effect" => (RV_P * "if setget(3) > 2, zz = 9; end\nM_.params(2) = zz;", SETGET, [3, 9, 3]),
    "P2_p05_cell_copy_of_M" => (RV_P * "cc = {M_};\np = 9;\nM_.params(3) = cc{1}.params(1);", nothing, [9, 2, 1]),
    "P2_p06_M_passed_by_value" => (RV_P * "r0 = getp(M_);\nM_.params(2) = r0;",
                                   Dict("getp.m" => "function v = getp(M)\nM.params(1) = 50;\nv = M.params(1);\nend\n"), [1, 50, 3]),
    "P2_p07_global_variable" => (RV_P * "global gx\ngx = 4;\nsetg();\nM_.params(1) = gx;",
                                 Dict("setg.m" => "function setg()\nglobal gx\ngx = gx + 1;\nend\n"), [5, 2, 3]),
    "P2_p10_oo_steady_state_before_steady" => (RV_P * "stoch_simul(order=1,irf=0,noprint);\nx = oo_.steady_state(1) + 5;\nq = x;", nothing, [1, 5, 3]),
    "P2_p20_option_changed_by_command" => (RV_P * "options_.order = 1;\nstoch_simul(order=2,irf=0,noprint);\nx = options_.order;\nq = x;",
                                           nothing, REFUSED),                                           # Dynare q = 2
    "P2_p21_option_not_touched_by_command" => (RV_P * "options_.irf = 7;\nstoch_simul(order=1,noprint,nograph);\nx = options_.irf;\nq = x;",
                                               nothing, [1, 7, 3]),
    "P2_p40_oo_from_initval" => (RV_P * "initval; y = 2; end;\nx = oo_.steady_state(1) + oo_.exo_steady_state(2);\nq = x;", nothing, [1, 2, 3]),
    "P2_unterminated_string" => (RV_P * "x = 'abc; M_.params(1) = 5;", nothing, REFUSED),               # a syntax error in MATLAB
)

if section("review")
@testset "review" begin
    @testset "review cases" begin
        for case in sort(collect(keys(RV_CASES)))
            body, files, expected = RV_CASES[case]
            files = something(files, Dict{String,String}())
            if expected === REFUSED
                @test throws_match(() -> rv(body; files=files), ModelError) || (@info case; false)
                continue
            end
            ok = try
                m = rv(body; files=files)
                close_nan(m.params, Float64.(expected); rtol=1e-14) || (@info case m.params; false)
            catch e
                @info case e
                false
            end
            @test ok
        end
    end

    @testset "A1 warning does not overclaim" begin
        m, warns = build_logged(RV_HEAD * RV_P * "x = 2i;\n")
        msg = [w for w in warns if occursin("MATLAB code", w)]
        @test length(msg) == 1 && !occursin("None of them writes", msg[1]) && occursin("found no write", msg[1])
        @test m.matlab_skipped[1][1] == "x = 2i" && m.params == [1, 2, 3]
    end

    @testset "A2 parameters set by the steady state" begin
        # files WITHOUT any MATLAB line: after steady/stoch_simul Dynare's M_.params holds A set by the steady state
        A, delta = 0.49751442178543509, 0.24875721089271755
        m = rv(RV_SSM_A * "steady;\ndelta = A/2;\n"; head=RV_RBC, name="s32")
        @test isapprox(m.params, [0.33, 0.99, delta, A]; rtol=1e-14)
        @test isempty(m.matlab_skipped)
        m = rv(RV_SSM_A * "stoch_simul(order=1,irf=0,noprint);\ndelta = A/2;\n"; head=RV_RBC, name="n03")
        @test isapprox(m.params, [0.33, 0.99, delta, A]; rtol=1e-14)
        m = rv(RV_SSM_A * "steady;\nshocks; var e = A; end;\n"; head=RV_RBC, name="n04")
        @test isapprox(m.Sigma_e, fill(A, 1, 1); rtol=1e-14)
        @test isapprox(m.params, [0.33, 0.99, 0.025, A]; rtol=1e-14)
        body = "k = 10;\nA = (1/beta - 1 + delta)/(alpha*k^(alpha-1));\ny = A*k^alpha;\nc = y - delta*k;\n"
        m = rv("steady;\ndelta = A/2;\n"; head=RV_RBC, name="s27", files=Dict("s27_steadystate.m" => rv_ssfile("s27", body)))
        @test isapprox(m.params, [0.33, 0.99, delta, A]; rtol=1e-14)
        # s25: the MATLAB variable A is not updated by steady (x = 1), M_.params(A) is
        m = rv(RV_SSM_A * "steady;\nx = A;\nM_.params(3) = x/2;\n"; head=RV_RBC, name="s25")
        @test isapprox(m.params, [0.33, 0.99, 0.5, A]; rtol=1e-14)
    end

    @testset "A3/B1 initval and endval read the current steady-state vector" begin
        m = rv(RV_P * "initval; y = 1; c = 2; end;\nendval; y = 3; c = y*2; end;\n"; name="a47")
        @test m.initval == Dict("y" => 1.0, "c" => 2.0) && m.endval == Dict("y" => 3.0, "c" => 6.0)
        m = rv(RV_P * "initval; c = y*2 + 1; y = 4; end;\n"; name="a49")          # unset variables are 0
        @test m.initval == Dict("c" => 1.0, "y" => 4.0)
        m = rv(RV_P * "initval; y = 1; e = 0.5; end;\nendval; e = 1; c = e*3; y = c + 1; end;\n"; name="v12")
        @test m.endval == Dict("e" => 1.0, "c" => 3.0, "y" => 4.0) && m.initval == Dict("y" => 1.0, "e" => 0.5)
        m = rv(RV_P * "endval; e = 1; c = e*3; y = c + 1; end;\n"; name="v13")
        @test m.endval == Dict("e" => 1.0, "c" => 3.0, "y" => 4.0)
        # v02: analytical steady state, then endval refers to it: Dynare k = 56.696838122096885
        m = rv(RV_SSM * "initval; k = 20; c = 1.5; y = 2.5; end;\nsteady;\nendval; k = k*2; c = 1; y = 1; end;\n"; head=RV_RBC, name="v02")
        @test isapprox(m.endval["k"], 56.696838122096885; rtol=1e-14)
        @test m.initval == Dict("k" => 20.0, "c" => 1.5, "y" => 2.5)
        # v01: numerical steady state. Dynare's own solver stops at k = 28.348419047 (residual 1e-10),
        # the exact value is 28.348419061: agreement only to the solver's tolerance (rtol 1e-8 here)
        m = rv("initval; k = 20; c = 1.5; y = 2.5; end;\nsteady;\ninitval; c = k/10; end;\n"; head=RV_RBC, name="v01")
        @test isapprox(m.initval["c"], 2.8348419047165483; rtol=1e-8)
        @test m.initval["k"] == 20.0 && m.initval["y"] == 2.5
        @test isapprox(m.initval["c"], 2.8348419061048443; rtol=1e-12)
    end

    @testset "A6/A7 in steady-state files" begin
        # x70-x72: Dynare's oo_.steady_state [y c k] = [50 3 1], [2 3 5], [2 3 5]
        head = "var y c k; varexo e; parameters alpha beta delta A lambda is np;\n" *
               "model; 1/c = beta/c(+1)*(alpha*A*k^(alpha-1)+1-delta) + 0*(lambda+is+np); y = A*exp(e)*k(-1)^alpha; " *
               "k = y - c + (1-delta)*k(-1); end;\nalpha = 0.33; beta = 0.99; delta = 0.025; A = 1; lambda = 2; is = 3; np = 4;\n"
        cases = Dict("x70" => ("k = 1; y = 2; c = 3;\nfor ii = 1:0\n k = 100;\nend\nif isempty(ii), y = 50; end\n", [50, 3, 1]),
                     "x71" => ("k = 1; y = 2; c = 3; t = 0;\ntry\n t = 1/t;\n k = 5;\ncatch\n k = 6;\nend\n", [2, 3, 5]),
                     "x72" => ("k = 1; y = 2; c = 3; t = 10;\ntry\n t = t^400;\n k = 5;\ncatch\n k = 6;\nend\n", [2, 3, 5]))
        for (name, (body, expected)) in cases
            m = rv("steady(nocheck);\n"; head=head, name=name, files=Dict("$(name)_steadystate.m" => rv_ssfile(name, body)))
            fn = M.steady_state_file_fn(m)
            ys, _ = fn(m, copy(m.params), zeros(1))
            @test ys[1:3] == expected
        end
    end

    @testset "A9 initval with NaN" begin
        m = rv(RV_P * "x = NaN;\ninitval; y = max(x, 2); c = min(x, 2); end;\n")          # w15
        @test m.initval == Dict("y" => 2.0, "c" => 2.0)
    end

    @testset "A11 two-output steady-state file" begin
        f = "function [ys,check] = s12_steadystate(ys,exo)\nglobal M_\ncheck = 0;\nalpha = M_.params(1); beta = M_.params(2); " *
            "delta = M_.params(3); A = M_.params(4);\nk = ((1/beta - 1 + delta)/(alpha*A))^(1/(alpha-1));\ny = A*k^alpha;\n" *
            "c = y - delta*k;\nys = [y; c; k];\nend\n"
        m = rv("steady;\n"; head=RV_RBC, name="s12", files=Dict("s12_steadystate.m" => f))
        @test throws_match(() -> M.steady_state(m), SteadyStateError, "too many inputs")   # Dynare: "called with too many inputs"
    end

    @testset "B2 block-local named x" begin
        # v04/v05: a steady_state_model local called x next to an exogenous variable
        expected = [3.0153277085137282, 2.3066172319875173, 28.348419061048443]
        for (name, block) in (("v04", "x = (1/beta - 1 + delta)/(alpha*A*exp(e));\nk = x^(1/(alpha-1));\n"),
                              ("v05", "x = (1/beta - 1 + delta)/(alpha*A);\nk = x^(1/(alpha-1));\n"))
            m = rv("steady_state_model;\n" * block * "y = A*k^alpha*exp(e);\nc = y - delta*k;\nend;\nsteady(nocheck);\n";
                   head=RV_RBC, name=name)
            @test isapprox(M.steady_state(m)[1], expected; rtol=1e-14)
            @test isapprox(M._from_steady_state_model_interpreted(m, copy(m.params), zeros(1), zeros(3))[1], expected; rtol=1e-14)
        end
    end

    @testset "B3 Inf in shocks" begin
        m = rv(RV_P * "x = 0;\nshocks; var e; stderr 1/x; end;\n")                # w12: Sigma_e(1,1) = Inf
        @test m.Sigma_e[1, 1] == Inf
    end

    @testset "B5 JSON repairs" begin
        # the preprocessor writes invalid JSON for an equation tag containing a backslash
        m = rv("var y; varexo e; parameters p; p = 0.5;\nmodel; [eq='\\#1', name='growth: \$\\Delta A\$'] y = p*y(-1) + e; end;\n"; head="")
        @test M.get_param(m, "p") == 0.5
    end

    @testset "C vector elements and M_ fields in Dynare statements" begin
        m = rv(RV_P * "v = [0.1 0.2 0.3];\nshocks; var e; stderr v(2); var u = v(3); end;\n" *
               "initval; y = v(1)*10; end;\nq = M_.Sigma_e(2,2);\n")
        @test isapprox(diag(m.Sigma_e), [0.04, 0.3])
        @test m.initval == Dict("y" => 1.0) && M.get_param(m, "q") == 0.3
    end

    @testset "C helper reading global M_" begin
        # s39: Dynare oo_.steady_state = [3.0153277085137282 2.3066172319875173 28.348419061048443]
        kglob = "function k = kglob(dummy)\nglobal M_\nalpha = M_.params(1); beta = M_.params(2); delta = M_.params(3); " *
                "A = M_.params(4);\nk = ((1/beta - 1 + delta)/(alpha*A))^(1/(alpha-1));\nend\n"
        m = rv("steady_state_model;\nk = kglob(1);\ny = A*k^alpha;\nc = y - delta*k;\nend;\nsteady;\n";
               head=RV_RBC, name="s39", files=Dict("kglob.m" => kglob))
        @test isapprox(M.steady_state(m)[1], [3.0153277085137282, 2.3066172319875173, 28.348419061048443]; rtol=1e-14)
        M.set_param!(m, "A", 1.2)                               # the helper sees the parameters of the call
        @test M.steady_state(m)[1][3] > 30
    end

    @testset "E Dynare functions writing M_ through global" begin
        # v24 and the scan of Dynare's matlab/ tree: a call to such a function is refused, not skipped
        for (k, line) in enumerate(("set_parameters(xparam1);", "x = 1;\nolsgibbs(ds, 'eq', 1, 2, 3);", "dynare_estimation(var_list_);",
                                    "pac.update.parameters('pacman', M_, oo_);"))
            @test throws_match(() -> rv(RV_P * "xparam1 = 0.7;\n" * line * "\n"; name="e$k"), ModelError) || (@info line; false)
        end
    end

    @testset "second pass: steady-state parameters and copies" begin
        # p25, p28, p29, p36, p37 (Dynare's M_.params [alpha beta delta A])
        A = 0.49751442178543509
        cases = Dict("p25" => ("steady;\nM_bak = M_;\nA = 3;\nM_ = M_bak;\ndelta = A/2;", [0.33, 0.99, A / 2, A]),
                     "p28" => ("steady;\nA = 2;\nsteady;\ndelta = A/2;", [0.33, 0.99, A / 2, A]),
                     "p29" => ("steady;\nf = @() get_param_by_name('A');\nM_.params(3) = f()/2;", [0.33, 0.99, A / 2, A]),
                     "p36" => ("check;\ndelta = A/2;", [0.33, 0.99, 0.5, 1.0]),
                     "p37" => ("steady;\nset_param_value('A', 4);\nstoch_simul(order=1,irf=0,noprint);\ndelta = A/2;", [0.33, 0.99, A / 2, A]))
        for (name, (body, expected)) in cases
            m = rv(RV_SSM_A * body * "\n"; head=RV_RBC, name=name)
            @test isapprox(m.params, expected; rtol=1e-14) || (@info name m.params; false)
        end
    end
end
end

"""
Port of pydyn's tools/sweep_matlab.py `mutators`: scan Dynare's matlab/ tree for functions that change the
calibration behind the caller's back (declare `global M_` or return M_, and assign M_.params, M_.Sigma_e,
M_.H, M_.Correlation_matrix(_ME), M_.Skew_e or M_ as a whole), then the functions calling one of the global
writers (transitively, matched as `name(`). Returns the names missing from JDyn's refusal lists.
"""
function mutators(src)
    codes = Dict{String,String}()
    for (root, _, fs) in walkdir(joinpath(src, "matlab"))
        for f in fs
            endswith(f, ".m") || continue
            rel = relpath(root, joinpath(src, "matlab"))
            pk = [x[2:end] for x in splitpath(rel) if startswith(x, "+")]
            codes[join(vcat(pk, [f[1:end-2]]), ".")] = replace(read(joinpath(root, f), String), r"%[^\n]*" => "")
        end
    end
    direct = Dict{String,String}()
    global_writers = Set{String}()
    for (name, code) in codes
        writes = occursin(r"\bM_\.(params|Sigma_e|H|Correlation_matrix|Correlation_matrix_ME|Skew_e)\b\s*(\([^=\n]*\))?\s*=(?!=)", code)
        whole = occursin(r"(?m)^\s*M_\s*=(?!=)|\[[^\]\n]*\bM_\b(?!\.)[^\]\n]*\]\s*=(?!=)", code)
        head = match(r"(?m)^\s*function\s+(?:(\[[^\]]*\]|\w+)\s*=\s*)?([\w.]+)", code)
        ((writes || whole) && head !== nothing) || continue
        has_global = occursin(r"(?m)^\s*global\b[^\n]*\bM_\b", code)
        returns = occursin(r"\bM_\b", something(head.captures[1], ""))
        if has_global || returns
            direct[name] = has_global ? "global" : "returns M_"
            has_global && push!(global_writers, name)
        end
    end
    closure = setdiff(global_writers, Set(["steady", "prior", "global_initialization", "gui.stochastic-simulation.read"]))
    pats = Dict{String,Regex}()
    changed = true
    while changed
        changed = false
        for (name, code) in codes
            name in closure && continue
            if any(w -> occursin(get!(() -> Regex("(?<![\\w.])" * replace(w, "." => "\\.", "-" => "\\-") * "\\s*\\("), pats, w), code), closure)
                push!(closure, name)
                changed = true
            end
        end
    end
    names = sort(collect(union(Set(keys(direct)), closure)))
    known = union(JDyn.MUTATORS, JDyn.MUTATOR_DOTTED, JDyn.SOLVE_FUNCS,
                  Set(["set_param_value", "set_shock_stderr_value", "steady", "gui.stochastic-simulation.read"]))
    [n for n in names if !(n in known)]
end

if section("review")
@testset "interpreter fixes of the second review (REVIEW2 R02-R10, R19, R20, R22, R32)" begin
    # classes and shapes that the double-valued Octave reference of matlab_extra_cases.txt does not show
    @test run_m("l = [true false true]; l(2) = 5; r = l;")["r"] isa Matrix{Bool}
    @test run_m("l = [true false true]; l(end) = []; r = l;")["r"] == Bool[1 0]
    @test run_m("l = false(1,3); l(2) = 1; r = ([10 20 30])(l);")["r"] == 20.0
    @test run_m("clear x; x(3) = true; r = x;")["r"] isa Matrix{Bool}
    @test run_m("x = []; x(3) = true; r = x;")["r"] isa Matrix{Float64}
    @test_throws MatlabError run_m("l = [true false]; l(1) = NaN;")
    @test run_m("r = [1 2 3; 4 5 6](logical([1 0 1]));")["r"] == [1.0 2.0]
    @test run_m("r = unique([true false true]);")["r"] isa Matrix{Bool}
    @test run_m("r = any([NaN 0]);")["r"] === false && run_m("r = all(NaN);")["r"] === true
    # R08: Julia exceptions are MATLAB errors (they can be caught), outside a try they are MatlabErrors
    @test run_m("try, x = trace(ones(2,3)); r = 1; catch, r = 2; end")["r"] == 2.0
    @test_throws MatlabError run_m("x = trace(ones(2,3));")
    @test run_m("try, x = trace(ones(2,3)); catch err, r = ischar(err.message); end")["r"] === true
    # R09
    @test run_m("r = sprintf('%d|', 'ab');")["r"] == "97|98|"
    @test run_m("r = sprintf('%ld', 5);")["r"] == "5"
    @test run_m("r = sprintf('%c', 65.5);")["r"] isa AbstractString
    @test run_m("r = sprintf('%s=%d;', 'ab', 5);")["r"] == "ab=5;"
    # R32: non-ASCII characters in string literals (MATLAB: UTF-16 code units; the comment is skipped)
    @test run_m("r = double('é');")["r"] == 233.0
    @test run_m("r = double('aé日'); % é")["r"] == [97.0 233.0 26085.0]
    @test run_m("r = sprintf('é%d', 1);")["r"] == "é1"
    @test run_m("r = strrep('aéa', 'é', 'x');")["r"] == "axa"
end

@testset "built-ins added for the Dynare test suite (values checked against Octave 8.4)" begin
    @test run_m("r = cell2mat({1 2; 3 4});")["r"] == [1.0 2.0; 3.0 4.0]
    @test run_m("r = cell2mat({[1 2], 3; [4 5], 6});")["r"] == [1.0 2.0 3.0; 4.0 5.0 6.0]
    @test run_m("r = cell2mat({'ab','cd'});")["r"] == "abcd"
    @test_throws MatlabError run_m("r = cell2mat({1, 'a'});")
    @test run_m("s.a = 1; s.b = 'x'; z = struct2cell(s); r = size(z); q = z{2};")["r"] == [2.0 1.0]
    @test run_m("s.a = 1; s.b = [2 3; 4 5]; r = struct2array(s)';")["r"] == [1.0 2.0 4.0 3.0 5.0]
    @test run_m("r = num2cell([1 2; 3 4]); q = r{2,1};")["q"] == 3.0
    @test run_m("r = class(cellstr('abc  '));")["r"] == "cell"
    @test run_m("r = isequaln([1 NaN], [1 NaN]);")["r"] === true
    @test run_m("r = isequaln([1 NaN], [1 2]);")["r"] === false
    @test run_m("r = union([3 1 2], [2 5]);")["r"] == [1.0 2.0 3.0 5.0]
    @test size(run_m("r = union([3;1], [2;5]);")["r"]) == (4, 1)
    w = run_m("[c, ia, ib] = intersect([5 1 3 3], [3 5 7]);")
    @test w["c"] == [3.0 5.0] && vec(w["ia"]) == [3.0, 1.0] && vec(w["ib"]) == [1.0, 2.0]
    w = run_m("[d, ia] = setdiff([5 1 3 3 9], 3);")
    @test w["d"] == [1.0 5.0 9.0] && vec(w["ia"]) == [2.0, 1.0, 5.0]
    @test size(run_m("r = setdiff([5;1;3], 3);")["r"]) == (2, 1)
    @test run_m("r = setdiff({'z','b','a'}, {'b'});")["r"].items == Any["a", "z"]
    @test size(run_m("r = intersect([1 2], [3 4]);")["r"]) == (1, 0)
    @test_throws Unsupported run_m("r = union([1 2], [3; 4]);")        # MATLAB and Octave disagree
    @test run_m("t = tic; r = toc(t) >= 0;")["r"] === true
    # varlist_indices (Dynare's matlab/varlist_indices.m)
    @test run_m("r = varlist_indices('b', {'a';'b';'c'});")["r"] == 2.0
    @test run_m("r = varlist_indices({'c','a'}, {'a';'b';'c'});")["r"] == [3.0 1.0]
    @test_throws MatlabError run_m("r = varlist_indices({'z'}, {'a';'b'});")
    @test run_m("[r, n] = varlist_indices({'z','a'}, {'a';'b'}, 1);")["n"] == 1.0
    # random numbers: refused by the loader, available (dimensions as MATLAB) when ALLOW_RANDOM is set
    @test_throws Unsupported run_m("r = randn(2);")
    JDyn.ALLOW_RANDOM[] = true
    try
        @test size(run_m("r = rand(2,3);")["r"]) == (2, 3)
        @test size(run_m("r = randn(3);")["r"]) == (3, 3)
        @test size(run_m("r = randn([3 2]);")["r"]) == (3, 2)
        @test run_m("r = randi([3 5], 1, 50); q = all(r >= 3 & r <= 5);")["q"] === true
        @test sort(vec(run_m("r = randperm(4);")["r"])) == [1.0, 2.0, 3.0, 4.0]
        @test run_m("randn('state', 3); a = randn(); randn('state', 3); b = randn(); r = a == b;")["r"] === true
    finally
        JDyn.ALLOW_RANDOM[] = false
    end
    # error()/assert() are flagged as the file's own
    e = try; run_m("error('mine');"); catch ex; ex; end
    @test e isa MatlabError && e.user
    e = try; run_m("x = [1 2](5);"); catch ex; ex; end
    @test e isa MatlabError && !e.user
end

@testset "steady_state_model runs with MATLAB semantics in a bare module (REVIEW2 R22, R13)" begin
    function ssm(expr; a=2.5, extra="", files=Dict{String,String}())
        m = build("var y; varexo e; parameters a b; a = $a; b = 0;\nmodel; y = 0.5*y(-1) + e; end;\nsteady_state_model; y = $expr; end;\n" * extra;
                  files=files)
        m
    end
    ys(m) = steady_state(m; check=false)[1][1]
    @test ys(ssm("round(a)")) == 3.0 && ys(ssm("round(a)"; a=-2.5)) == -3.0 && ys(ssm("round(a)"; a=0.49999999999999994)) == 0.0
    @test ys(ssm("mod(a, b)")) == 2.5 && ys(ssm("mod(-7.5, 4)")) == 0.5 && ys(ssm("mod(7.5, -4)")) == -0.5
    @test ys(ssm("rem(-7.5, 4)")) == -3.5
    @test throws_match(() -> ys(ssm("rem(a, b)")), SteadyStateError, "non-finite")      # rem(x, 0) is NaN
    @test ys(ssm("sign(-a) + 10*sign(b)")) == -1.0 && ys(ssm("fix(-a)")) == -2.0 && ys(ssm("floor(-a)")) == -3.0
    @test ys(ssm("max(a, b)")) == 2.5 && ys(ssm("min(a, -a)")) == -2.5
    @test ys(ssm("a^b + b^b")) == 2.0                                  # 0^0 = 1
    # the compiled path is kept for functions both languages have, and used for these
    m = ssm("round(a) + mod(a, 2) + sqrt(a)")
    @test m.fns.steady_state !== nothing && ys(m) ≈ 3 + 0.5 + sqrt(2.5)
    @test m.ssm_compiled_ok
    # names Julia has and MATLAB (or the interpreter) does not: nothing of Base is visible
    @test throws_match(() -> ys(ssm("exit(3)")), Union{SteadyStateError,Unsupported,MatlabError})
    @test throws_match(() -> ys(ssm("sleep(0)")), Union{SteadyStateError,Unsupported,MatlabError})
    # a complex result is refused as before (the compiled path raises DomainError, the interpreter refuses)
    @test throws_match(() -> ys(ssm("(-a)^0.5")), Union{SteadyStateError,Unsupported,MatlabError})
    # a Julia function given by the user takes precedence over the compiled module's own names
    m = build("var y; varexo e; parameters a; a = 2.5;\nmodel; y = 0.5*y(-1) + e; end;\nsteady_state_model; y = round(a); end;\n";
              functions=Dict("round" => (x...) -> 42.0))
    @test ys(m) == 42.0
end
end

if section("review")
@testset "mutator list is complete for the Dynare tree" begin
    # review E: every Dynare function that writes the calibration through `global M_` (or returns a changed M_)
    # is in JDyn's refusal lists
    if isdir(joinpath(DYNARE_SRC, "matlab"))
        missing_ = mutators(DYNARE_SRC)
        isempty(missing_) || @info "missing from JDyn.MUTATORS" missing_
        @test isempty(missing_)
    else
        @info "Dynare source tree not available: mutator scan skipped"
    end
end
end

# ====================================================================== (b) corpus files against Dynare
"""
Load one file of the corpus (`rel` under the Dynare source tree) and compare with Dynare's results.
Returns (outcome, detail): "identical", "refused" (ModelError while running the statements),
"out of scope" (model features JDyn does not support, preprocessor failure), "DIFFERENT" (loaded
with a different calibration, or a steady state that differs), "steady state failed", "crash".
"""
function corpus_case(rel, ref)
    src = joinpath(DYNARE_SRC, rel)
    isfile(src) || return ("skipped", "not in the Dynare source tree")
    d = mktempdir()
    for f in readdir(dirname(src))
        p = joinpath(dirname(src), f)
        (isfile(p) && filesize(p) < 3_000_000) && cp(p, joinpath(d, f); force=true)
    end
    M.ISOCTAVE[] = true                                     # the reference run used Octave
    try
        m = try
            with_logger(NullLogger()) do
                M.load(joinpath(d, basename(src)))
            end
        catch e
            msg = sprint(showerror, e)
            if e isa ModelError
                scope = occursin(r"deterministic exogenous|external functions in the model|expected the preprocessor|generated function", msg)
                return (scope ? "out of scope" : "refused", first(msg, 300))
            end
            occursin("preprocessor failed", msg) && return ("out of scope", first(msg, 300))
            return ("crash", first(msg, 300))
        end
        params = m.params
        if Int(ref["after_steady"]) != 0
            params = with_logger(NullLogger()) do
                M.steady_state(m; check=false)[2]
            end
        end
        rp = vec(Float64.(ref["params"] isa Real ? [ref["params"]] : ref["params"]))
        close_nan(params, rp; rtol=1e-10) || return ("DIFFERENT", "params differ: " *
            join([string(n, " ", a, " vs ", b) for (n, a, b) in zip(m.param_names, params, rp) if !close_nan(a, b; rtol=1e-10)][1:min(end, 5)], "; "))
        rs = ref["Sigma_e"] isa Real ? fill(Float64(ref["Sigma_e"]), 1, 1) : Float64.(ref["Sigma_e"])
        length(rs) == length(m.Sigma_e) && close_nan(m.Sigma_e, reshape(rs, size(m.Sigma_e)); rtol=1e-10) ||
            return ("DIFFERENT", "Sigma_e differs")
        if Int(ref["check_ys"]) != 0
            ys = try
                with_logger(NullLogger()) do
                    M.steady_state(m)[1]
                end
            catch e
                return ("steady state failed", first(sprint(showerror, e), 300))
            end
            # rtol: Dynare's own steady-state solver stops at 1e-6 on the residuals for numerical steady states
            rt = Float64(ref["ys_rtol"])
            ry = vec(Float64.(ref["ys"] isa Real ? [ref["ys"]] : ref["ys"]))
            close_nan(ys, ry; rtol=rt, atol=rt) || return ("DIFFERENT", "steady state differs: max |diff| = $(maximum(abs, ys .- ry; init=0.0))")
        end
        return ("identical", "")
    catch e
        e isa InterruptException && rethrow()
        return ("crash", first(sprint(showerror, e), 300))
    finally
        M.ISOCTAVE[] = false
        rm(d; recursive=true, force=true)
    end
end

if section("corpus")
@testset "corpus" begin
    path = joinpath(REF, "mat_corpus.mat")
    if !isfile(path) || !isdir(DYNARE_SRC)
        @info "corpus reference or Dynare source tree missing: corpus test skipped"
    else
        c = matread(path)
        files = String.(vec(c["files"]))
        @test length(files) >= 40
        sel = get(ENV, "JDYN_CORPUS", "")                   # e.g. "1:20" or a substring of the file names
        idx = isempty(sel) ? eachindex(files) : occursin(r"^\d+:\d+$", sel) ? (eval(Meta.parse(sel))) :
              [k for k in eachindex(files) if occursin(Regex(sel), files[k])]
        log = get(ENV, "JDYN_CORPUS_LOG", "")
        counts = Dict{String,Int}()
        for k in idx
            t = @elapsed (outcome, detail) = corpus_case(files[k], c["refs"]["f$(k-1)"])
            counts[outcome] = get(counts, outcome, 0) + 1
            line = "$k | $(files[k]) | $outcome | $(round(t; digits=1)) s | $detail"
            println(line)
            isempty(log) || open(io -> println(io, line), log, "a")
            # loading with a different calibration without an error is the worst outcome: there must be none
            @test outcome != "DIFFERENT"
            @test outcome in ("identical", "refused", "out of scope", "skipped")
        end
        println("corpus: ", join(["$k $v" for (k, v) in sort(collect(counts))], ", "))
    end
end
end
