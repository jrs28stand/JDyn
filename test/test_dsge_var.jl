# DSGE-VAR (dsge_var, dsge_varlag; src/dsge_var.jl) and endogenous_prior against Dynare 7.2 under Octave 8.4. References:
# test/ref_dsge_var: pydyn's est_dvar_v<V>.mat / est_endo_v<V>.mat (likelihood at several parameter vectors, modes; generators
# r_dsge_var.m and r_endogenous_prior.m) and Octave runs made here (oct_dvar1_results.mat, oct_dvar1_irf.mat: the DSGE-VAR IRF block of
# PosteriorIRF_core1.m with randn('state', 5); oct11/: est_dvar.mod -DVARIANT=11 with MCMC and bayesian_irf).
include("common.jl")

const DV = joinpath(@__DIR__, "ref_dsge_var")
dvref(name) = matread(joinpath(DV, name))
function dvdir()
    d = mktempdir()
    for f in readdir(DV)
        (endswith(f, ".m") || endswith(f, ".mod") || endswith(f, ".mat")) && cp(joinpath(DV, f), joinpath(d, f); force=true)
    end
    d
end
function dvload(mod, v; kw...)
    d = dvdir()
    m = with_logger(NullLogger()) do
        JDyn.load(joinpath(d, mod * ".mod"); extra_args=["-DVARIANT=$v"])
    end
    est = with_logger(NullLogger()) do
        JDyn.Estimation(m; allow_unsupported=true, kw...)
    end
    m, est
end

@testset "DSGE-VAR posterior kernel (calibrated, estimated and infinite weight)" begin
    for v in (0, 1, 2)
        r = dvref("est_dvar_v$v.mat")
        _, est = dvload("est_dvar", v)
        @test est.names == split(r["names"], ";")
        X = r["X"]
        for k in 1:size(X, 2)
            lp = JDyn.log_posterior(est, X[:, k])
            if r["info"][1, k] != 0
                @test lp == -Inf && est.last_info_code == r["info"][1, k]        # 51: the weight is below (k + n) / T
            else
                @test -lp ≈ r["fval"][k] rtol = (v == 2 ? 5e-12 : 1e-11)
            end
        end
    end
end

@testset "endogenous_prior kernel" begin
    for v in (0, 1, 2, 9)                              # complete data, first_obs/nobs/presample, missing observations, correlated measurement errors
        r = dvref("est_endo_v$v.mat")
        _, est = dvload("est_endo", v; endogenous_prior=true)
        _, plain = dvload("est_endo", v)
        for k in 1:size(r["X"], 2)
            @test -JDyn.log_posterior(est, r["X"][:, k]) ≈ r["fval"][k] rtol = 1e-11
            @test -JDyn.log_posterior(plain, r["X"][:, k]) ≈ r["fval_plain"][k] rtol = 1e-11
            @test abs(r["fval"][k] - r["fval_plain"][k]) > 1
        end
    end
    est = dvload("est_endo", 0; endogenous_prior=true, lik_init=2)[2]
    @test_throws JDyn.EstimationError JDyn.log_posterior(est, est.x0)
end

@testset "DSGE-VAR and endogenous prior posterior modes (csminwel from the .mod file)" begin
    d = dvdir()
    r1 = JDyn.dynare(joinpath(d, "est_dvar.mod"); extra_args=["-DVARIANT=1"])
    ref = dvref("est_dvar_v1.mat")
    opt = r1.oo_["posterior"]["optimization"]
    @test maximum(abs.(vec(opt["mode"]) .- vec(ref["xmode"]))) < 1e-4
    @test abs(opt["log_density"] - ref["logpost_mode"]) < 1e-7
    # oo_.dsge_var.posterior_mode at Octave's own mode
    oref = dvref("oct_dvar1_results.mat")["oo_"]
    est = r1.model |> m -> JDyn.Estimation(m; allow_unsupported=true, statement=1)
    xm = vec(oref["posterior"]["optimization"]["mode"])
    t = JDyn.dsge_var_posterior(est, xm)
    b = oref["dsge_var"]["posterior_mode"]
    @test maximum(abs.(t.PHI_tilde .- b["PHI_tilde"])) < 1e-10
    @test maximum(abs.(t.SIGMA_u_tilde .- b["SIGMA_u_tilde"])) < 1e-10
    @test maximum(abs.(t.iXX .- b["iXX"])) < 1e-10
    @test t.prior.DF == b["prior"]["DF"] && t.prior.ArtificialSampleSize == b["prior"]["ArtificialSampleSize"]
    @test maximum(abs.(t.prior.PHI_star .- b["prior"]["PHI_star"])) < 1e-10
    pm = r1.oo_["dsge_var"]["posterior_mode"]
    @test maximum(abs.(pm["PHI_tilde"] .- b["PHI_tilde"])) < 1e-3                    # at JDyn's own mode (within csminwel's accuracy)
    @test r1.oo_["MarginalDensity"]["LaplaceApproximation"] ≈ oref["MarginalDensity"]["LaplaceApproximation"] atol = 1e-4
    r2 = JDyn.dynare(joinpath(d, "est_endo.mod"); extra_args=["-DVARIANT=20"])
    ref = dvref("est_endo_v20.mat")
    @test maximum(abs.(vec(r2.oo_["posterior"]["optimization"]["mode"]) .- vec(ref["xmode"]))) < 1e-4
    @test abs(r2.oo_["posterior"]["optimization"]["log_density"] - ref["logpost_mode"]) < 1e-7
end

@testset "DSGE-VAR impulse responses replay PosteriorIRF_core1.m" begin
    _, est = dvload("est_dvar", 1)
    r = dvref("oct_dvar1_irf.mat")
    rng = JDyn.OctaveRNG(5)
    for k in 1:3
        dr = JDyn.dsge_var_irf_draw(est, vec(r["deep"]), rng, 8)
        @test maximum(abs.(dr .- r["stock"][:, :, :, k])) < 1e-11
    end
end

@testset "dynare(): DSGE-VAR with MCMC and bayesian_irf" begin
    d = dvdir()
    r = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "est_dvar.mod"); extra_args=["-DVARIANT=11"], mcmc_rng=:octave)
    end
    for j in 1:2        # the chains follow Dynare's (the modes agree to csminwel's accuracy, 5e-5)
        f = matread(joinpath(d, "est_dvar", "metropolis", "est_dvar_mh1_blck$j.mat")); g = matread(joinpath(DV, "oct11", "est_dvar_mh1_blck$j.mat"))
        @test size(f["x2"]) == size(g["x2"]) == (300, 11)
        @test maximum(abs.(f["x2"] .- g["x2"])) < 1e-3
    end
    ref = dvref("oct_dvar11_results.mat")["oo_"]
    @test r.oo_["MarginalDensity"]["ModifiedHarmonicMean"] ≈ ref["MarginalDensity"]["ModifiedHarmonicMean"] atol = 1e-3
    P = r.oo_["PosteriorIRF"]
    @test haskey(P, "dsge") && haskey(P, "bvardsge")
    @test sort(collect(keys(P["bvardsge"]["Mean"]))) == sort(collect(keys(ref["PosteriorIRF"]["bvardsge"]["Mean"])))
    @test size(P["bvardsge"]["deciles"]["yobs_ea"]) == (8, 9)
    # Dynare's own 40 sub-draws replayed: the DSGE responses are exact
    prm = matread(joinpath(DV, "oct11", "est_dvar_param_irf1.mat"))["stock"]
    est = JDyn.Estimation(r.model; allow_unsupported=true, statement=1)
    ir = JDyn.posterior_irf(est; params=prm, periods=8)
    b = ref["PosteriorIRF"]["dsge"]["Mean"]
    @test maximum(maximum(abs.(ir[k].Mean .- b[k])) for k in keys(b)) < 1e-12
    ib = JDyn.posterior_irf_dsgevar(est, prm; periods=8, rng=JDyn.OctaveRNG(5))
    @test all(isfinite, ib.mean) && size(ib.mean, 1) == 8
end

@testset "DSGE-VAR refusals" begin
    m, _ = dvload("est_dvar", 0)
    est = with_logger(NullLogger()) do
        JDyn.Estimation(m; allow_unsupported=true, presample=2)
    end
    @test_throws JDyn.EstimationError JDyn.log_posterior(est, est.x0)
    est = with_logger(NullLogger()) do
        JDyn.Estimation(m; allow_unsupported=true, first_obs=2)                   # first_obs - dsge_varlag < 1
    end
    @test_throws JDyn.EstimationError JDyn.log_posterior(est, est.x0)
    raw = JDyn.load_data("est_nk_data", ["yobs", "piobs", "robs"]; base_dir=m.mod_dir)
    raw[10, 2] = NaN
    est = with_logger(NullLogger()) do
        JDyn.Estimation(m; allow_unsupported=true, data=raw)
    end
    @test_throws JDyn.EstimationError JDyn.log_posterior(est, est.x0)
end
