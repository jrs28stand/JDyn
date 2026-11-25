# external_function in the model block against Dynare 7.2 (pydyn's references test/ref_parity/ext_fn: Dynare's tests/external_function models,
# ext_<case>.mat from r_external.m): decision rules at Dynare's steady state, JDyn's own solution, Julia callables
using Test
isdefined(@__MODULE__, :REF) || include("common.jl")

@testset "external_function against Dynare" begin
    ED = joinpath(@__DIR__, "ref_parity", "ext_fn")
    folder = mktempdir()
    for f in readdir(ED)
        (endswith(f, ".mod") || endswith(f, ".m")) && cp(joinpath(ED, f), joinpath(folder, f))
    end
    cp(joinpath(ED, "+matlab"), joinpath(folder, "+matlab"))
    function eload(case; kw...)
        with_logger(NullLogger()) do
            JDyn.load(joinpath(folder, case * ".mod"); kw...)
        end
    end
    keys_ = ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2")
    md(a, b) = maximum(abs.(vec(Float64.(a)) .- vec(Float64.(b))))
    analytic = ["first_and_2nd_deriv_given", "first_and_2nd_deriv_given_namespace"]
    numerical = ["no_deriv_given", "first_deriv_given", "extfun_in_mlv"]
    worst = 0.0
    for case in vcat(analytic, numerical)
        m, r = eload(case), matread(joinpath(ED, "ext_$case.mat"))
        dr = JDyn.k_order(m; order=2, ys=vec(Float64.(r["ys"])))
        for k in keys_
            d = md(dr[k], r[k])
            worst = max(worst, d / max(1.0, maximum(abs, r[k])))
            @test d < 1e-12 * max(1.0, maximum(abs, r[k]))
        end
    end
    @info "external functions: worst scaled difference of the decision rules at Dynare's steady state: $worst"
    # JDyn's own steady state and solution: analytic cases to 1e-9; numerical Hessians amplify the 1e-11 steady-state difference (hess_element, step 1e-6)
    for case in vcat(["benchmark"], analytic, numerical)
        m, r = eload(case), matread(joinpath(ED, "ext_$case.mat"))
        dr = JDyn.k_order(m; order=2)
        @test md(dr["ys"], r["ys"]) < 1e-9
        tol = Dict(k => 1e-9 for k in keys_)
        case in numerical && merge!(tol, Dict("ghxx" => 1e-2, "ghxu" => 1e-4, "ghuu" => 1e-5))
        for k in keys_
            @test md(dr[k], r[k]) < tol[k]
        end
    end
    # Julia callables take precedence over the .m files; a function that returns its own derivatives returns a tuple
    ncalls = Ref(0)
    f(a, b) = (ncalls[] += 1; a * b^2)
    f12(a, b) = (a * b^2, [b^2 2 * a * b], [0.0 2b; 2b 2a])
    r = matread(joinpath(ED, "ext_no_deriv_given.mat"))
    m = eload("no_deriv_given"; functions=Dict("extFunNoDerivs" => f))
    dr = JDyn.k_order(m; order=2, ys=vec(Float64.(r["ys"])))
    @test ncalls[] > 0
    @test all(md(dr[k], r[k]) < 1e-12 * max(1.0, maximum(abs, r[k])) for k in keys_)
    r = matread(joinpath(ED, "ext_first_and_2nd_deriv_given.mat"))
    m = eload("first_and_2nd_deriv_given"; functions=Dict("extFunWithFirstAndSecondDerivs" => f12))
    dr = JDyn.k_order(m; order=2, ys=vec(Float64.(r["ys"])))
    @test all(md(dr[k], r[k]) < 1e-12 * max(1.0, maximum(abs, r[k])) for k in keys_)
    m = eload("first_and_2nd_deriv_given"; functions=Dict("extFunWithFirstAndSecondDerivs" => f))
    @test_throws JDyn.ModelError JDyn.k_order(m; order=2, ys=vec(Float64.(r["ys"])))
    # a missing function is reported
    d = mktempdir(); cp(joinpath(folder, "no_deriv_given.mod"), joinpath(d, "no_deriv_given.mod"))
    m = with_logger(NullLogger()) do
        JDyn.load(joinpath(d, "no_deriv_given.mod"))
    end
    @test_throws Exception JDyn.steady_state(m)
end
