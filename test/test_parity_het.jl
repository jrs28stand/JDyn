# heteroskedastic_shocks / heteroskedastic_filter against Dynare 7.2 (pydyn's references test/ref_parity/het, made by r_heteroskedastic.m):
# dsge_likelihood at 5 parameter vectors, the smoother at the initial values; variants 0 (complete data), 1 (first_obs=11, nobs=80, presample=4 and an
# extra block), 2 (missing observations), 4 (kalman_algo=2: univariate filter), 5 (lik_init=2), 6/7 (smoother)
using Test
isdefined(@__MODULE__, :REF) || include("common.jl")

@testset "heteroskedastic_shocks / heteroskedastic_filter against Dynare" begin
    HD = joinpath(@__DIR__, "ref_parity", "het")
    function hload(v)
        d = mktempdir()
        for f in ("est_het.mod", "est_nk_data.m", "est_nk_miss.m")
            cp(joinpath(HD, f), joinpath(d, f))
        end
        with_logger(NullLogger()) do
            JDyn.Model(JDyn.run_preprocessor(joinpath(d, "est_het.mod"); extra_args=["-DVARIANT=$v"]))
        end
    end
    for v in (0, 1, 2, 4, 5)
        m = hload(v)
        r = matread(joinpath(HD, "est_het_v$v.mat"))
        est = Estimation(m)
        @test est.options["heteroskedastic_filter"] == true || est.options["heteroskedastic_filter"] == 1
        Qv, Qs = JDyn.heteroskedastic_arrays(m, est.nobs, Int(est.options["first_obs"]))
        rQv, rQs = Float64.(r["Qvalue"]), Float64.(r["Qscale"])
        @test isnan.(Qv) == isnan.(rQv) && isnan.(Qs) == isnan.(rQs)
        @test Qv[.!isnan.(Qv)] ≈ rQv[.!isnan.(rQv)] rtol = 1e-15
        @test Qs[.!isnan.(Qs)] ≈ rQs[.!isnan.(rQs)] rtol = 1e-15
        X = Float64.(r["X"]); info = vec(Int.(r["info"]))
        worst = 0.0
        for k in 1:size(X, 2)
            info[k] == 0 || (@test info[k] == 41; continue)
            ll = JDyn.log_likelihood(est, X[:, k])
            worst = max(worst, abs(ll - r["loglik"][k]) / abs(r["loglik"][k]))
            @test ll ≈ r["loglik"][k] rtol = 1e-10
        end
        v == 0 && @info "heteroskedastic likelihood: worst relative difference $worst"
        if v == 0
            plain = JDyn.log_likelihood(Estimation(m; heteroskedastic_filter=false), X[:, 1])
            @test plain ≈ -r["fval_plain"] - r["lnprior"][1] rtol = 1e-10
            @test abs(plain - r["loglik"][1]) > 100
        end
    end
    for v in (6, 7)
        m = hload(v)
        r = matread(joinpath(HD, "est_het_v$v.mat"))
        est = Estimation(m)
        res = JDyn.smooth(m; estimation=est, theta=est.x0)
        n = size(r["smoothed"], 2)
        for (i, name) in enumerate(m.endo_names[1:n])
            @test res.SmoothedVariables[name] ≈ Float64.(r["smoothed"][:, i]) rtol = 1e-8 atol = 1e-9
            @test res.UpdatedVariables[name] ≈ Float64.(r["updated"][:, i]) rtol = 1e-8 atol = 1e-9
        end
        @test res.shocks ≈ Float64.(r["shocks"]) rtol = 1e-8 atol = 1e-9
        plain = JDyn.smooth(m; estimation=est, theta=est.x0, heteroskedastic_filter=false)
        @test maximum(abs.(plain.shocks .- Float64.(r["shocks"]))) > 0.1
        @test_throws JDyn.SmootherError JDyn.smooth(m; estimation=est, theta=est.x0, kalman_algo=2)
    end
end
