# Posterior samplers (src/samplers.jl, mcmc_driver.jl, smc.jl, mcmc_stage.jl, mh_files.jl) against Dynare 7.2 under Octave 8.4.
# References: test/ref_samplers (pydyn's single-sampler runs, smp_case*.mat, generator in gen/r_samplers.m; and complete Dynare
# runs oct20..oct25 and mh20/mh21 made with smp_nk.mod -DCASE=<n>). Dynare's chains are replayed from the seed with OctaveRNG.
include("common.jl")
using Statistics

const SD = joinpath(@__DIR__, "ref_samplers")
V(x) = x isa AbstractArray ? vec(x) : [x]
sref(name) = matread(joinpath(SD, name))

function sload(case; kw...)
    mod = case in (1, 3, 5, 6, 8, 13, 20) ? "smp_nk" : "smp_schorfheide"
    d = mktempdir()
    for f in readdir(SD)
        (endswith(f, ".m") || endswith(f, ".mod") || endswith(f, ".mat")) && cp(joinpath(SD, f), joinpath(d, f); force=true)
    end
    m = with_logger(NullLogger()) do
        JDyn.load(joinpath(d, mod * ".mod"); extra_args=["-DCASE=$case"])
    end
    est = with_logger(NullLogger()) do
        JDyn.Estimation(m; allow_unsupported=true, kw...)
    end
    est, mod == "smp_nk" ? "post_nk_mode0" : "post_schorfheide_mode0"
end

function schains(r)
    x2 = r["x2"]; lp = r["logpo2"]; xi = r["x_init"]
    ndims(x2) == 2 && (x2 = reshape(x2, 1, size(x2)...))
    lp = ndims(lp) == 1 ? reshape(lp, 1, :) : lp
    xi = ndims(xi) == 1 ? reshape(xi, 1, :) : xi
    x2, lp, xi
end
dbounds(r) = (sampler_bounds=(V(r["sampler_lb"]), V(r["sampler_ub"])), kernel_bounds=(V(r["mh_lb"]), V(r["mh_ub"])))

function replay(case; kw...)
    r = sref("smp_case$case.mat")
    est, mf = sload(case)
    x2, lp, xi = schains(r)
    mode = JDyn.mode_from_file(est, mf)
    mc = JDyn.sample_mcmc(est; replic=size(x2, 2), mode=mode, rng=:octave, dbounds(r)..., kw...)
    mc, x2, lp, xi, r
end

@testset "mode_file: names, order and Hessian as check_mode_file.m" begin
    est, mf = sload(8)
    mode = JDyn.mode_from_file(est, mf)
    mfile = matread(joinpath(SD, "post_nk_mode0.mat"))
    @test mode.x == vec(mfile["xparam1"])
    @test mode.inv_hessian ≈ inv(mfile["hh"]) rtol = 1e-9
    @test_throws JDyn.EstimationError JDyn.mode_from_file(est, "no_such_mode_file")
end

@testset "RWMH replays Dynare's chains draw for draw" begin
    for case in (8, 7)                                  # estimated_params bounds on rhoa; Student proposal (4 degrees of freedom)
        mc, x2, lp, xi = replay(case)
        @test maximum(abs.(mc.draws .- x2)) < 1e-12
        @test maximum(abs.(mc.log_posterior .- lp)) < 1e-8
        @test maximum(abs.(mc.x0 .- xi)) < 1e-14
    end
    mc8, = replay(8)
    est8, _ = sload(8)
    @test est8.ub[findfirst(==("rhoa"), est8.names)] == 0.86 && maximum(mc8.draws[:, :, findfirst(==("rhoa"), est8.names)]) < 0.86   # estimated_params bounds
end

@testset "independent MH (normal and Student proposals)" begin
    for case in (6, 10)
        mc, x2, lp, xi, r = replay(case)
        @test maximum(abs.(mc.draws .- x2)) < 1e-12
        @test maximum(abs.(mc.log_posterior .- lp)) < 1e-8
        @test maximum(abs.(mc.x0 .- xi)) < 1e-14
        @test vec(mc.acceptance_rate) ≈ V(r["acceptance"]) rtol = 1e-12
    end
end

@testset "slice sampler: prior draws, prior mode, rotated, multimodal" begin
    r = sref("smp_case1.mat"); est, _ = sload(1)
    x2, lp, xi = schains(r)
    mc = JDyn.sample_mcmc(est; replic=size(x2, 2), nblocks=1, x0=xi[1:1, :], rng=:octave, dbounds(r)...)
    @test maximum(abs.(mc.draws[1, :, :] .- x2[1, :, :])) < 1e-11          # chain 2 is not compared (see test_samplers.py in pydyn)
    r = sref("smp_case2.mat"); est, _ = sload(2)
    x2, lp, xi = schains(r)
    mc = JDyn.sample_mcmc(est; replic=size(x2, 2), rng=:octave, dbounds(r)...)
    @test maximum(abs.(mc.draws .- x2)) < 1e-12
    @test maximum(abs.(mc.x0 .- xi)) < 1e-14
    @test vec(mean(mc.neval; dims=2)) ≈ V(r["feval"]) rtol = 1e-14
    for case in (3, 4)                                                    # rotated at the mode; two modes
        mc, x2, lp, xi, r = replay(case)
        @test maximum(abs.(mc.draws .- x2)) < 1e-11
        @test maximum(abs.(mc.x0 .- xi)) < 1e-14
        @test vec(mean(mc.neval; dims=2)) ≈ V(r["feval"]) rtol = 1e-14
    end
end

@testset "TaRB blocks replayed from Dynare's random-number states" begin
    for (case, minagree) in ((13, 12), (14, 8))
        r = sref("smp_case$case.mat")
        est, _ = sload(case)
        opts = JDyn.sampler_options(JDyn.TARB, get(est.options, "posterior_sampler_options", nothing))
        js = JDyn.jscale(est, nothing)[1]

        nb = Int(r["nblocks_logged"]); agree = 0
        for k in 1:nb
            g(f) = r["b$(k)_$f"]
            o = JDyn.OctaveRNG(0)
            JDyn.octave_set_state!(o.u, V(g("su"))); JDyn.octave_set_state!(o.n, V(g("sn")))
            cur, = JDyn.tarb_block(est, V(g("current")), Float64(V(g("last_posterior"))[1]), Int.(V(g("idx"))), V(r["sampler_lb"]), V(r["sampler_ub"]),
                                   V(r["mh_lb"]), V(r["mh_ub"]), js, opts, o, [0])
            agree += maximum(abs.(cur .- V(g("after"))) ./ max.(abs.(V(g("after"))), 1e-9)) < 2e-5       # csminwel's path differs in the tails
        end
        @test agree >= minagree
    end
end

@testset "SMC: hssmc and dsmh from Dynare's initial particles" begin
    for (case, meth) in ((11, "hssmc"), (12, "dsmh"))
        r = sref("smp_case$case.mat"); est, _ = sload(case)
        res = JDyn.sample_smc(est; rng=:octave, initial=r["p_first"])
        P = size(res.particles) == size(r["p_last"]) ? res.particles : res.particles'
        @test res.method == meth
        @test maximum(abs.(P .- r["p_last"]) ./ max.(abs.(r["p_last"]), 1e-12)) < 1e-8
        @test maximum(abs.(res.log_posterior .- V(r["lp_last"]))) < 1e-7
        @test res.log_marginal_density ≈ only(r["mdd"]) rtol = 1e-11
    end
end

# complete runs through dynare(): mode_file, MH files, load_mh_file, posterior statistics against Dynare's own output
function srun(case, d)
    with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "smp_nk.mod"); mcmc_rng=:octave, extra_args=["-DCASE=$case"])
    end
end
function sdir()
    d = mktempdir()
    for f in readdir(SD)
        (endswith(f, ".m") || endswith(f, ".mod") || endswith(f, ".mat")) && cp(joinpath(SD, f), joinpath(d, f); force=true)
    end
    d
end
mhfile(d, k, j) = matread(joinpath(d, "smp_nk", "metropolis", "smp_nk_mh$(k)_blck$(j).mat"))

@testset "dynare(): mode_file + mh_replic, then load_mh_file continues the chains" begin
    d = sdir()
    r1 = srun(20, d)
    ref = matread(joinpath(SD, "oct20", "smp_nk_results.mat"))["oo_"]
    for j in 1:2
        f = mhfile(d, 1, j); g = matread(joinpath(SD, "mh20", "smp_nk_mh1_blck$j.mat"))
        @test size(f["x2"]) == (300, 12)
        @test maximum(abs.(f["x2"] .- g["x2"])) < 1e-12
        @test maximum(abs.(f["logpo2"] .- g["logpo2"])) < 1e-8
        @test vec(f["LastSeeds"]["file1"]["Unifor"]) == vec(g["LastSeeds"]["file1"]["Unifor"])      # the generator states are Dynare's
        @test vec(f["LastSeeds"]["file1"]["Normal"]) == vec(g["LastSeeds"]["file1"]["Normal"])
    end
    pj = r1.oo_["posterior_mean"]["parameters"]; po = ref["posterior_mean"]["parameters"]
    @test maximum(abs(pj[k] - po[k]) for k in keys(po)) < 1e-12
    @test r1.oo_["MarginalDensity"]["ModifiedHarmonicMean"] ≈ ref["MarginalDensity"]["ModifiedHarmonicMean"] rtol = 1e-11
    h = matread(joinpath(d, "smp_nk", "metropolis", "smp_nk_mh_history_0.mat"))["record"]
    ho = matread(joinpath(SD, "mh20", "smp_nk_mh_history_0.mat"))["record"]
    @test h["AcceptanceRatio"] == ho["AcceptanceRatio"] && h["MhDraws"] == ho["MhDraws"] && h["MAX_nruns"] == ho["MAX_nruns"]
    # second session: the chains continue from the files of the first (as written by JDyn, and as written by Octave)
    for src in (:jdyn, :dynare)
        d2 = src == :jdyn ? d : sdir()
        if src == :dynare
            mkpath(joinpath(d2, "smp_nk", "metropolis"))
            for f in readdir(joinpath(SD, "mh20")); cp(joinpath(SD, "mh20", f), joinpath(d2, "smp_nk", "metropolis", f); force=true); end
        end
        srun(21, d2)
        for j in 1:2
            f = mhfile(d2, 1, j); g = matread(joinpath(SD, "mh21", "smp_nk_mh1_blck$j.mat"))
            @test size(f["x2"]) == (500, 12)
            @test maximum(abs.(f["x2"] .- g["x2"])) < 1e-12
        end
        @test isfile(joinpath(d2, "smp_nk", "metropolis", "smp_nk_mh_history_1.mat"))
        h = matread(joinpath(d2, "smp_nk", "metropolis", "smp_nk_mh_history_1.mat"))["record"]
        ho = matread(joinpath(SD, "mh21", "smp_nk_mh_history_1.mat"))["record"]
        @test h["MhDraws"] == ho["MhDraws"]
        @test h["InitialLogPost"] ≈ ho["InitialLogPost"] rtol = 1e-12
        @test maximum(abs.(h["LastParameters"] .- ho["LastParameters"])) < 1e-12
    end
end

@testset "dynare(): mh_tune_jscale, jumping covariances, slice sampler" begin
    for (case, tol) in ((22, 1e-12), (23, 1e-12), (24, 1e-12), (25, 1e-8))
        d = sdir()
        r = srun(case, d)
        ref = matread(joinpath(SD, "oct$case", "smp_nk_results.mat"))["oo_"]
        for j in 1:2
            f = mhfile(d, 1, j); g = matread(joinpath(SD, "oct$case", "smp_nk_mh1_blck$j.mat"))
            @test size(f["x2"]) == size(g["x2"])
            @test maximum(abs.(f["x2"] .- g["x2"])) < tol
        end
        pj = r.oo_["posterior_mean"]["parameters"]; po = ref["posterior_mean"]["parameters"]
        @test maximum(abs(pj[k] - po[k]) for k in keys(po)) < max(tol, 1e-10)
        if case == 22
            h = matread(joinpath(d, "smp_nk", "metropolis", "smp_nk_mh_history_0.mat"))["record"]
            ho = matread(joinpath(SD, "oct22", "smp_nk_mh_history_0.mat"))["record"]
            @test vec(h["ProposalScaleVec"]) ≈ vec(ho["ProposalScaleVec"]) rtol = 1e-14          # the tuned scale 0.56172897...
            @test h["AcceptanceRatio"] == ho["AcceptanceRatio"]
        end
    end
end

@testset "options: parsing and refusals" begin
    @test JDyn.sampler_options(JDyn.SLICE, "'rotated',1,'mode_files','a'")["rotated"] == 1
    @test_logs (:warn, r"Unknown option") JDyn.sampler_options(JDyn.SLICE, "'no_such_option',1")          # a warning, as Dynare
    @test_throws JDyn.EstimationError JDyn.sampler_options(JDyn.RWMH, "'proposal_distribution','rand_multivariate_laplace'")
    @test_throws JDyn.EstimationError JDyn.sampler_options(JDyn.SLICE, "'maximize',1")
    est, _ = sload(8)
    @test_throws JDyn.EstimationError JDyn.sample_mcmc(est; method="no_such_sampler", replic=2, nblocks=1)
    @test_throws JDyn.EstimationError JDyn.sample_mcmc(est; method="hssmc", replic=2, nblocks=1)
end

@testset "dynare(): SMC and TaRB samplers run end to end and fill oo_" begin
    for (case, key) in ((11, "hssmc"), (12, "dsmh"))
        d = sdir()
        r = with_logger(NullLogger()) do
            JDyn.dynare(joinpath(d, "smp_schorfheide.mod"); mcmc_rng=:octave, extra_args=["-DCASE=$case"])
        end
        md = r.oo_["MarginalDensity"]
        @test haskey(md, key) && isfinite(md[key]) && !haskey(md, "ModifiedHarmonicMean")
        pm = r.oo_["posterior_mean"]["parameters"]
        @test all(isfinite, [pm[k] for k in keys(pm)])
    end
    d = sdir()
    r = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "smp_schorfheide.mod"); mcmc_rng=:octave, extra_args=["-DCASE=14"])     # TaRB, Student proposal, 2 chains
    end
    @test haskey(r.oo_, "posterior_mean") && haskey(r.oo_["MarginalDensity"], "ModifiedHarmonicMean")
    @test isfile(joinpath(d, "smp_schorfheide", "metropolis", "smp_schorfheide_mh_history_0.mat"))
end

@testset "estimate(m): mode_file and the new samplers through the library function" begin
    d = sdir()
    m = with_logger(NullLogger()) do
        JDyn.load(joinpath(d, "smp_nk.mod"); extra_args=["-DCASE=23"])          # identity jumping covariance, mode file, 2 chains x 150
    end
    res = with_logger(NullLogger()) do
        JDyn.estimate(m; allow_unsupported=true)
    end
    @test res.mode.x ≈ vec(matread(joinpath(SD, "post_nk_mode0.mat"))["xparam1"])
    @test size(res.mcmc.draws) == (2, 150, 12)
    ref = matread(joinpath(SD, "oct23", "smp_nk_mh1_blck1.mat"))
    @test maximum(abs.(res.mcmc.draws[1, :, :] .- ref["x2"])) > 1e-6          # Julia's own generators unless mcmc_rng is :octave (set_mcmc_rng!)
    JDyn.set_mcmc_rng!(m, :octave)
    res = with_logger(NullLogger()) do
        JDyn.estimate(m; allow_unsupported=true)
    end
    @test maximum(abs.(res.mcmc.draws[1, :, :] .- ref["x2"])) < 1e-12
    mh = with_logger(NullLogger()) do
        JDyn.load(joinpath(d, "smp_schorfheide.mod"); extra_args=["-DCASE=12"])
    end
    r2 = with_logger(NullLogger()) do
        JDyn.estimate(mh; allow_unsupported=true)
    end
    @test isfinite(r2.marginal_density_mhm) && size(r2.mcmc.draws, 1) == 1
end
