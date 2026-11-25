using JDyn, MAT, Logging, Random, LinearAlgebra
include("/home/claude/work/jdyn-ho/test/common.jl")
const OREF = "/home/claude/work/jdyn-ho/test/ref_occbin"
odata(r) = Matrix{Float64}(r["data"])
mk(m, data; kw...) = with_logger(NullLogger()) do; JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=4, mh_replic=0, allow_unsupported=true, occbin=Dict("likelihood_inversion_filter"=>true), kw...); end
println("uptime ", strip(read(`uptime`, String)))
for c in ("nkm", "nkms")
    r = matread(joinpath(OREF, "occest_$(c)_lik.mat")); m = load_copy(joinpath(OREF, "occest_$c.mod")); d = odata(r)
    X = r["X"]; X = size(X,1)==(c=="nkm" ? 9 : 3) ? X : Matrix(X')
    es = ["occbin" => mk(m, d), "lcp" => mk(m, d; obc_method="news", obc_formulation="lcp"), "regime" => mk(m, d; obc_method="news", obc_formulation="regime")]
    for k in 1:min(3, size(X,2))
        s = "$c vec $k: "
        for (nm, e) in es
            JDyn.log_posterior(e, X[:,k]); t = minimum(@elapsed(JDyn.log_posterior(e, X[:,k])) for _ in 1:7); s *= "$nm $(round(1000t, digits=1)) ms; "
        end
        println(s)
    end
    if c == "nkms"
        md = matread(joinpath(OREF, "occest_nkms_mode.mat")); x, cov = vec(md["xparam1"]), inv(md["hh"])
        res = Dict()
        for (nm, e) in es
            tm = @elapsed mo = with_logger(NullLogger()) do; JDyn.posterior_mode(e; compute_hessian=false); end
            tc = @elapsed mc = with_logger(NullLogger()) do; JDyn.sample_posterior(e; replic=300, nblocks=2, mode=x, cov=cov, jscale=1.3, seed=20261124); end
            res[nm] = (mo, mc)
            println("nkms ", nm, ": mode ", round(tm, digits=1), " s (", mo.nfev, " evals, f=", mo.log_posterior, "), chains 2x300 ", round(tc, digits=1), " s, acc ", mc.acceptance_rate)
        end
        for nm in ("lcp", "regime")
            println(nm, " vs occbin: draws identical ", res[nm][2].draws == res["occbin"][2].draws, ", accepted identical ", res[nm][2].accepted == res["occbin"][2].accepted, ", mode diff ", maximum(abs.(res[nm][1].x .- res["occbin"][1].x)))
        end
    end
end
println("uptime ", strip(read(`uptime`, String)))
