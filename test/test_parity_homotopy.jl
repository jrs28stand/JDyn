# Homotopy stepping (src/homotopy.jl) against Dynare 7.2 / Octave 8.4 (test/ref_parity/gen_homotopy.sh), and save_params_and_steady_state
include(joinpath(@__DIR__, "common.jl"))

const _HOM_REF = Dict{String,NamedTuple}()
for l in eachline(joinpath(@__DIR__, "ref_parity", "homotopy_ref.txt"))
    f = split(l)
    mode, steps, ns, nf = parse.(Int, f[2:5])
    nums = parse.(Float64, f[6:end])
    if isempty(nums)
        _HOM_REF[f[1]] = (mode=mode, steps=steps, ns=ns, nf=nf, ys=Float64[], exo=Float64[], params=Float64[])
    else
        i = findall(==(-999.0), nums)
        _HOM_REF[f[1]] = (mode=mode, steps=steps, ns=ns, nf=nf, ys=nums[1:i[1]-1], exo=nums[i[1]+1:i[2]-1], params=nums[i[2]+1:end])
    end
end

const _HOM_CASES = Dict("a1" => (0.1, 2.0, :par), "a2" => (0.1, 2.0, :par), "a3" => (0.1, 2.0, :par), "b1" => (0.1, 2.0, :endval),
    "b2" => (0.1, 2.0, :endval), "b3" => (0.1, 2.0, :endval), "c3" => (0.9, 20.0, :par), "c1" => (0.3, 4.0, :par),
    "d3" => (3.0, 500.0, :par), "e3" => (-0.95, 0.01, :par), "f3" => (200.0, 0.0001, :par), "g3" => (-0.99, 30.0, :par))

function _hom_mod(name)
    bet1, x1, variant = _HOM_CASES[name]
    r = _HOM_REF[name]
    setup = variant == :endval ? "steady(solve_algo=9);\nendval; x = $x1; end;\nhomotopy_setup(from_initval_to_endval); end;" :
                                 "homotopy_setup; bet, 0.05, $bet1; x, $x1; end;"
    """
var c k; varexo x; parameters alph gam delt bet aa;
alph=0.5; gam=0.5; delt=0.02; aa=0.5; bet=0.05;
model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;
initval;
x = 1;
k = ((delt+bet)/(aa*x*alph))^(1/(alph-1));
c = aa*x*k^alph-delt*k;
end;
$setup
steady(homotopy_mode=$(r.mode), homotopy_steps=$(r.steps), solve_algo=9);
"""
end

@testset "homotopy modes 1, 2, 3 step as Dynare's homotopy1/2/3.m" begin
    for name in sort(collect(keys(_HOM_REF)))
        r = _HOM_REF[name]
        if isempty(r.ys)
            @test_throws JDyn.SteadyStateError build(_hom_mod(name))
            continue
        end
        m = build(_hom_mod(name))
        ys = [m.oo[v] for v in m.endo_names]
        @test maximum(abs.(ys .- r.ys) ./ abs.(r.ys)) < 1e-12
        @test m.oo["x"] == r.exo[1]
        @test m.params == r.params
        log = m.cache[:homotopy_log]
        @test count(e -> e.ok, log) == r.ns + (r.mode == 3)        # Dynare prints no line for the final successful solve of mode 3
        @test count(e -> !e.ok, log) == r.nf
    end
end

@testset "stepping differs from solving at the final values" begin
    # the same final calibration solved directly from the initial steady state stops at another point within Dynare's tolerance:
    # the stepping is observable (and is what Dynare's homotopy1.m gives, to 1e-15)
    txt = replace(_hom_mod("a1"), "homotopy_setup; bet, 0.05, 0.1; x, 2.0; end;\nsteady(homotopy_mode=1, homotopy_steps=50, solve_algo=9);" =>
                  "steady(solve_algo=9);\nbet = 0.1;\ninitval; x = 2; c = 1.5306122448979593; k = 12.755102040816327; end;\nsteady(solve_algo=9);")
    m = build(txt)
    @test isempty(get(m.cache, :homotopy_log, []))
    ys, _ = JDyn.steady_state(m)
    @test 1e-8 < abs(ys[2] - _HOM_REF["a1"].ys[2]) / _HOM_REF["a1"].ys[2] < 1e-3
end

@testset "homotopy errors as Dynare's steady.m" begin
    @test_throws JDyn.ModelError build(replace(_hom_mod("a1"), "homotopy_setup; bet, 0.05, 0.1; x, 2.0; end;" => ""))          # no homotopy_setup
    @test_throws JDyn.ModelError build(replace(_hom_mod("a2"), "x, 2.0;" => "x, 1.0;"))                                         # mode 2: initial = final
    @test_throws JDyn.ModelError build(replace(_hom_mod("b1"), "endval; x = 2.0; end;\n" => ""))                                 # from_initval_to_endval without endval
end

@testset "save_params_and_steady_state" begin
    d = mktempdir()
    write(joinpath(d, "s.mod"), """
var c k; varexo x; parameters alph;
alph=0.5;
model; c + k - x*k(-1)^alph - 0.98*k(-1); c^(-0.5) - (1/1.05)*(0.5*alph*x(+1)*k^(alph-1) + 0.98)*c(+1)^(-0.5); end;
initval; x=1; k=10; c=2; end;
steady(solve_algo=9);
save_params_and_steady_state('pp.txt');
""")
    with_logger(NullLogger()) do
        JDyn.load(joinpath(d, "s.mod"))
    end
    lines = readlines(joinpath(d, "pp.txt"))
    ref = readlines(joinpath(@__DIR__, "ref_parity", "save_params_ref.txt"))
    @test length(lines) == length(ref)
    @test [split(l)[1] for l in lines] == [split(l)[1] for l in ref]
    @test maximum(abs.(parse.(Float64, [split(l)[2] for l in lines]) .- parse.(Float64, [split(l)[2] for l in ref]))) < 1e-12
    # the file loads back: load_params_and_steady_state (the preprocessor reads it)
    write(joinpath(d, "l.mod"), replace(read(joinpath(d, "s.mod"), String), "steady(solve_algo=9);\nsave_params_and_steady_state('pp.txt');" => "load_params_and_steady_state('pp.txt');"))
    m = with_logger(NullLogger()) do
        JDyn.load(joinpath(d, "l.mod"))
    end
    @test m.oo["k"] == parse(Float64, split(lines[3])[2])
end
