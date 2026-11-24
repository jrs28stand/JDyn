# Semi-structural toolbox (var_model, trend_component_model, var_expectation_model, pac_model, pac_target_info, MCE)
# against Dynare 7.2 under Octave. Fixtures: Dynare's own tests of tests/pac, tests/var-expectations and
# tests/trend-component-and-var-models, cut before their first dseries/simulation line (test/ref_pac/make_fixtures.py);
# references (M_.params after pac.initialize / pac.update.expectation / var_expectation.update / pac.mce.parameters,
# the companion matrices, the moduli of the eigenvalues of a stoch_simul) from test/ref_pac/gen_refs.sh.
using Test, JDyn, MAT, LinearAlgebra, Logging

haskey(ENV, "DYNARE_PREPROCESSOR") || (ENV["DYNARE_PREPROCESSOR"] = "/root/ref/pp-standalone/src/dynare-preprocessor")
const PACREF = joinpath(@__DIR__, "ref_pac")

function run_fixture(name)
    d = mktempdir()
    cp(joinpath(PACREF, name * ".mod"), joinpath(d, name * ".mod"))
    with_logger(NullLogger()) do
        redirect_stdout(devnull) do
            JDyn.dynare(joinpath(d, name * ".mod"))
        end
    end
end

fixtures = [first(split(l)) for l in readlines(joinpath(PACREF, "fixtures.txt")) if !isempty(strip(l))]
worst = Dict{String,Float64}()
@testset "semi-structural toolbox against Dynare 7.2" begin
    for name in fixtures
        reffile = joinpath(PACREF, name * ".ref.mat")
        isfile(reffile) || continue
        @testset "$name" begin
            ws = run_fixture(name)
            r = matread(reffile)
            p = vec(Float64.(ws.M_.params))
            rp = vec(Float64.(r["params"]))
            @test length(p) == length(rp)
            @test isnan.(p) == isnan.(rp)                    # parameters nobody sets stay NaN in both
            ok = .!isnan.(rp)
            e = maximum(abs.(p[ok] .- rp[ok]) ./ max.(1.0, abs.(rp[ok])))
            @test e < 1e-9
            worst[name] = e
            for (k, v) in r
                startswith(k, "C_") || continue
                mt = match(r"^C_(var|trend_component)_(.*)$", k)
                type, nm = mt.captures
                C = ws.oo_[type][nm].CompanionMatrix
                @test size(C) == size(v)
                @test maximum(abs.(C .- v)) < 1e-10
            end
            if haskey(r, "eigabs") && haskey(ws.oo_, "dr") && haskey(ws.oo_.dr, "eigval")
                big(x) = filter(y -> y < 1e8, x)                  # infinite eigenvalues (their number depends on the QZ)
                ev = big(sort(abs.(filter(isfinite, vec(ws.oo_.dr.eigval)))))
                re = big(vec(r["eigabs"]))
                @test length(ev) == length(re)
                @test maximum(abs.(ev .- re)) < 1e-6 * max(1, maximum(re))
            end
        end
    end
end

# Dynare's own checks inside the .mod files (assert/error on the companion matrices, AR matrices, expectation weights ...): the files run to the end
@testset "tests of Dynare's suite that check themselves" begin
    for name in filter(n -> startswith(n, "tcv_") || n in ("ve_1", "ve_11", "ve_1ts_a", "ve_3"), fixtures)
        @testset "$name" begin
            @test run_fixture(name) isa JDyn.DynareWorkspace
        end
    end
end

# what is refused stays refused, clearly
@testset "refusals" begin
    d = mktempdir()
    src = read(joinpath(PACREF, "p_v1.mod"), String)
    write(joinpath(d, "r1.mod"), src * "\npac.estimate.nls('pacman', struct(), 1, 1);\n")
    @test_throws JDyn.ModelError with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "r1.mod"))
    end
end
