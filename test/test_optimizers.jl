# Mode-finding algorithms (src/optimizers.jl) against Dynare 7.2 under Octave 8.4: test/optim/r_optim.m stores
# oo_.posterior.optimization (mode, log density, iterations, function count) and the Hessian of each run,
# r_newrat_step.m the first steps of newrat.
include("common.jl")
using Random

const OPTDIR = joinpath(@__DIR__, "optim")
optref(name) = matread(joinpath(OPTDIR, name))

function optload(mod, data)
    d = mktempdir()
    cp(joinpath(OPTDIR, "$mod.mod"), joinpath(d, "$mod.mod"))
    cp(joinpath(REF, data), joinpath(d, data))
    with_logger(NullLogger()) do
        JDyn.load(joinpath(d, "$mod.mod"); extra_args=["-DMC=0"])
    end
end

"Distance between two modes in posterior standard deviations (Dynare's inverse Hessian at its mode)."
function in_std(x, ref)
    all(isfinite, ref["hh"]) || return NaN               # Dynare's Hessian failed at its mode
    sd = sqrt.(diag(inv(ref["hh"])))
    maximum(abs.(x - vec(ref["opt"]["mode"])) ./ sd)
end

@testset "newrat (mode_compute=5)" begin
    m = optload("est_opt_sch", "est_schorfheide_data.m")
    est = JDyn.Estimation(m)
    @testset "first step: outer-product-gradient Hessian, step sizes, csminit" begin
        r = optref("newrat_step.mat")
        @test est.x0 ≈ vec(r["xparam1"]) atol = 1e-15
        fcn = th -> (q = JDyn._dsge_likelihood(est, th; with_lik=true); (q[1], q[2], q[3], q[2] == 0 ? vcat(-q[5], q[6]) : Float64[]))
        f0, _, dlik = JDyn._penalty_full(fcn, est.x0, 1e8)
        @test abs(f0 - r["f0"]) < 1e-10
        @test maximum(abs.(dlik - vec(r["dlik"]))) < 1e-10         # dsge_likelihood's contributions
        hi = JDyn._HessInfo(fill(1e-6, est.n), 1e-5, NaN, false)
        _, gg, _, igg, _, h1, fc = JDyn._mr_hessian(fcn, est.x0, f0, 0, 1e-5, hi, est.lb, est.ub)
        # the steps make f change by about 1e-5: the likelihoods' rounding (1e-11) moves the gradient by about 1e-6 relative
        @test fc == r["fc"]
        @test maximum(abs.(h1 - vec(r["h1"])) ./ vec(r["h1"])) < 1e-6
        @test maximum(abs.(gg - vec(r["gg"])) ./ abs.(vec(r["gg"]))) < 1e-5
        @test maximum(abs.(igg - r["igg"]) ./ abs.(r["igg"])) < 1e-3
        f1, x1, fc1, _ = JDyn._csminit(fcn, est.x0, f0, f0, gg, false, igg)
        @test fc1 == r["fc1"]
        d1 = maximum(abs.(x1 - vec(r["x1"])) ./ abs.(vec(r["x1"])))
        println("  newrat first iterate: f diff $(abs(f1 - r["fval1"])), x relative diff $d1")
        @test abs(f1 - r["fval1"]) < 1e-3 && d1 < 1e-4
    end
    @testset "mode of the NK model" begin
        ref = optref("opt_nk_5.mat")
        m2 = optload("est_opt_nk", "est_nk_data.m")
        est2 = JDyn.Estimation(m2)
        @test join(est2.names, ";") == ref["names"]
        mode = JDyn.posterior_mode(est2; mode_compute=5, compute_hessian=false)
        @test mode.success && occursin("No further improvement", mode.message)
        # same optimum to newrat's tolerance (TolFun 1e-5); the iterates are not Dynare's because newrat's steps are so small that
        # the likelihoods' rounding differences change the univariate steps (see docs/STATUS.md)
        println("  newrat on est_opt_nk: log posterior diff $(abs(mode.log_posterior - ref["opt"]["log_density"])), mode diff $(in_std(mode.x, ref)) std")
        @test abs(mode.log_posterior - ref["opt"]["log_density"]) < 1e-4
        @test in_std(mode.x, ref) < 0.05
    end
end

@testset "simplex (mode_compute=8): Dynare's iterates" begin
    ref = optref("opt_nk_8.mat")
    m = optload("est_opt_nk", "est_nk_data.m")
    est = JDyn.Estimation(m)
    mode = JDyn.posterior_mode(est; mode_compute=8, compute_hessian=false)
    @test maximum(abs.(mode.x - vec(ref["opt"]["mode"]))) < 1e-10
    @test abs(mode.log_posterior - ref["opt"]["log_density"]) < 1e-9
    @test mode.iterations == ref["opt"]["optimization_info"]["iterations"]
    @test mode.nfev == ref["opt"]["optimization_info"]["funcCount"]
    # est_opt_sch (opt_sch_8.mat) is not compared: from the prior means the simplex stalls far from the mode in both Dynare
    # (log posterior 1308.9) and JDyn (1261.5), and a near-tie between two vertices, decided by the likelihoods' rounding,
    # separates the two runs early (docs/STATUS.md)
end

@testset "CMA-ES (mode_compute=9) and fminsearch (mode_compute=7): same optimum" begin
    ref = optref("opt_nk_9.mat")
    m = optload("est_opt_nk", "est_nk_data.m")
    est = JDyn.Estimation(m)
    mode = JDyn.posterior_mode(est; mode_compute=9, compute_hessian=false, rng=Random.Xoshiro(42))
    println("  cmaes on est_opt_nk: log posterior diff $(abs(mode.log_posterior - ref["opt"]["log_density"])), mode diff $(in_std(mode.x, ref)) std")
    @test abs(mode.log_posterior - ref["opt"]["log_density"]) < 1e-4
    @test in_std(mode.x, ref) < 0.05
    if isfile(joinpath(OPTDIR, "opt_sch_7.mat"))
        r7 = optref("opt_sch_7.mat")
        est7 = JDyn.Estimation(optload("est_opt_sch", "est_schorfheide_data.m"))
        mode7 = JDyn.posterior_mode(est7; mode_compute=7, compute_hessian=false)
        dlp, ds = abs(mode7.log_posterior - r7["opt"]["log_density"]), in_std(mode7.x, r7)
        println("  fminsearch on est_opt_sch: log posterior diff $dlp, mode diff $ds std")
        @test dlp < 1e-3 && ds < 0.05
    end
    # options and refusals
    @test_throws JDyn.EstimationError JDyn.posterior_mode(est; mode_compute=1)
    @test_throws JDyn.EstimationError JDyn.posterior_mode(est; mode_compute=6)
    # an optim key the optimiser does not use is warned about, not refused (Dynare ignores it)
    @test JDyn.posterior_mode(JDyn.Estimation(m; optim_opt="'NumgradEpsilon',1e-6"); mode_compute=8, maxiter=1) isa Any
    @test_throws JDyn.EstimationError JDyn.posterior_mode(JDyn.Estimation(m; optim_opt="'Hessian',2"); mode_compute=5)
    short = JDyn.posterior_mode(JDyn.Estimation(m; optim_opt="'MaxIter',2"); mode_compute=5, compute_hessian=false)
    @test short.iterations == 2 && !short.success
end
