# Posterior analyses over MCMC draws (src/mcmc_diagnostics.jl, posterior.jl, posterior_forecast.jl, posterior_moments.jl,
# posterior_stage.jl) against Dynare 7.2 under Octave 8.4. References: test/ref_posterior (generators in test/ref_posterior/gen). Dynare's own
# chains and sub-draws are replayed (MCMCResult built from the saved chains; sub-draws and random numbers passed in), so the
# statistics agree to rounding.
include("common.jl")
using Random

const PD = joinpath(@__DIR__, "ref_posterior")
pref(name) = matread(joinpath(PD, name))

function ploadmod(mod, defs)
    d = mktempdir()
    for f in readdir(PD)
        (endswith(f, ".m") || endswith(f, ".mod") || endswith(f, ".mat")) && cp(joinpath(PD, f), joinpath(d, f); force=true)
    end
    m = with_logger(NullLogger()) do
        JDyn.load(joinpath(d, mod * ".mod"); extra_args=defs)
    end
    m, d
end

function chains(mod)
    c = pref("$(mod)_chains.mat")
    idx = Int.(c["index"])
    nb = Int(c["nblck"]); T = size(idx, 1); n = size(c["points"], 2)
    draws = zeros(nb, T, n); lp = zeros(nb, T)
    for b in 1:nb, t in 1:T
        draws[b, t, :] = c["points"][idx[t, b], :]
        lp[b, t] = c["points_lp"][idx[t, b]]
    end
    JDyn.MCMCResult(draws, lp, vec(c["acceptance"]), String.(split(c["names"], ";")), Matrix(c["x_init"]), Float64(c["mh_drop"]), falses(nb, T)), c
end

"Largest |a - b| / (|b| + max |b|) over entries; NaNs must sit in the same places."
function rdiff(a, b)
    a, b = Float64.(a), reshape(Float64.(b), size(a))
    isnan.(a) == isnan.(b) || return Inf
    k = .!isnan.(b)
    any(k) || return 0.0
    maximum(abs.(a[k] .- b[k]) ./ (abs.(b[k]) .+ maximum(abs.(b[k]))))
end

@testset "MCMC convergence diagnostics" begin
    for mod in ("post_nk", "post_schorfheide")
        mc, c = chains(mod)
        r = pref("$(mod)_stats.mat")
        nb, T, n = size(mc.draws)
        qrs = haskey(r, "raftery_qrs") ? vec(r["raftery_qrs"]) : [0.025, 0.005, 0.95]
        cd = JDyn.mcmc_diagnostics(String.(split(c["names"], ";")), mc; geweke_interval=vec(r["geweke_interval"]),
                                   taper_steps=Int.(vec(r["taper_steps"])), raftery_lewis_qrs=qrs, raftery_lewis=haskey(r, "raftery"))
        @test maximum(abs.(cd.inefficiency_factors .- r["ifac"]) ./ abs.(r["ifac"])) < 1e-9
        fields = split(r["geweke_fields"], ";")
        nm = String.(split(c["names"], ";"))
        worst = 0.0
        for b in 1:nb, j in 1:n, (k, f) in enumerate(fields)
            worst = max(worst, abs(cd.geweke[b][nm[j]][f] - r["geweke"][b, j, k]) / abs(r["geweke"][b, j, k]))
        end
        @test worst < 1e-12
        wp = 0.0
        for b in 1:nb, j in 1:n
            g = cd.geweke[b][nm[j]]
            wp = max(wp, rdiff(vec(g["pooled_mean"]), r["geweke_pooled_mean"][b, j, :]), rdiff(vec(g["pooled_nse"]), r["geweke_pooled_nse"][b, j, :]))
            wp = max(wp, maximum(abs.(vec(g["prob_chi2_test"]) .- r["geweke_prob"][b, j, :]) ./ (abs.(r["geweke_prob"][b, j, :]) .+ 1e-12)))
        end
        @test wp < 1e-9
        bg = cd.brooks_gelman
        @test bg.x == vec(r["bg_x"])
        @test maximum(abs.(hcat([bg.univariate[:, :, j] for j in 1:n]..., bg.multivariate) .- r["bg"]) ./ abs.(r["bg"])) < 1e-9
        if cd.raftery_lewis !== nothing
            for b in 1:nb, (k, f) in enumerate(split(r["raftery_fields"], ";"))
                @test cd.raftery_lewis[b][f] == r["raftery"][b, :, k]
            end
            @test cd.raftery_lewis[1]["N_min"] == r["raftery_N_min"]
        end
    end
    # oo_.convergence
    mc, c = chains("post_nk")
    oo = JDyn.MStruct()
    JDyn.store_convergence!(oo, JDyn.mcmc_diagnostics(String.(split(c["names"], ";")), mc))
    @test haskey(oo, "convergence")
end

@testset "posterior IRFs replay Dynare's sub-draws" begin
    for mod in ("post_nk", "post_schorfheide")
        m, d = ploadmod(mod, ["-DPOST=2"])
        est = with_logger(NullLogger()) do
            JDyn.Estimation(m; allow_unsupported=true)
        end
        r = pref("$(mod)_stats.mat")
        nk = mod == "post_nk"
        ir = JDyn.posterior_irf(est; params=Matrix(r["irf_params"]), keep_draws=true, periods=nk ? 12 : 20, relative_irf=nk,
                                irf_shocks=nk ? ["ea", "er"] : nothing, mh_conf_sig=Float64(pref("$(mod)_chains.mat")["mh_conf_sig"]),
                                variables=String.(split(r["irf_varlist"], ";")))
        @test ir.names == String.(split(r["irf_names"], ";"))
        sc = maximum(abs.(r["irf_mean"]))
        for (a, nm) in ((ir.mean, "irf_mean"), (ir.median, "irf_median"), (ir.hpd_inf, "irf_hpdinf"), (ir.hpd_sup, "irf_hpdsup"), (ir.var, "irf_var"))
            @test maximum(abs.(a .- r[nm])) / sc < 1e-11
        end
        @test maximum(abs.(ir.deciles .- r["irf_deciles"])) / sc < 1e-11
    end
end

@testset "posterior forecasts replay Dynare's sub-draws and random numbers" begin
    for (tag, post, variant) in (("p7", 7, 0), ("p8_9", 8, 9))
        m, d = ploadmod("fc_post_nk", ["-DPOST=$post", "-DVARIANT=$variant"])
        r = pref("fc_post_$tag.mat")
        est = with_logger(NullLogger()) do
            JDyn.Estimation(m; datafile=joinpath(d, "est_nk_data.m"), allow_unsupported=true)
        end
        h = Int(r["horizon"])
        pf = with_logger(NullLogger()) do
            JDyn.posterior_forecast(est; params=Matrix(r["params"]), rng=JDyn.OctaveRNG(77), horizon=h, mh_conf_sig=Float64(r["mh_conf_sig"]))
        end
        for (nm, dd) in (("MeanForecast", pf.MeanForecast), ("PointForecast", pf.PointForecast), ("PointForecastME", pf.PointForecastME))
            ref = r[nm]
            @test Set(keys(dd)) == Set(keys(ref["Mean"]))
            for v in keys(dd), (f, a) in (("Mean", dd[v].mean), ("Median", dd[v].median), ("Var", dd[v].var), ("HPDinf", dd[v].hpd_inf), ("HPDsup", dd[v].hpd_sup))
                @test maximum(abs.(vec(a) .- vec(ref[f][v])) ./ (1 .+ abs.(vec(ref[f][v])))) < 1e-10
            end
            for v in keys(dd)
                @test maximum(abs.(dd[v].deciles .- ref["deciles"][v])) < 1e-9
            end
        end
        @test all(pf.PointForecast["y"].var .> pf.MeanForecast["y"].var .- 1e-12)
        oo = JDyn.MStruct()
        JDyn.store_posterior_forecast!(oo, pf)
        @test haskey(oo, "MeanForecast") && haskey(oo, "PointForecast") && haskey(oo, "PointForecastME")
    end
end

@testset "posterior theoretical moments replay Dynare's sub-draws" begin
    tri(n) = [(i, j) for i in 1:n for j in i:n]
    for (mod, case) in (("nk", 1), ("nk", 2), ("nk", 4), ("nk", 5), ("schorfheide", 1))
        m, d = ploadmod("postmom_$mod", ["-DPM=$case"])
        r = pref("postmom_$(mod)_$case.mat")
        est = with_logger(NullLogger()) do
            JDyn.Estimation(m; datafile=joinpath(d, "est_$(mod)_data.m"), allow_unsupported=true, mh_conf_sig=Float64(r["mh_conf_sig"]))
        end
        vl = String.(split(r["vartan"], ";"))
        steps = Int.(vec(r["steps"])); steps = all(==(0), steps) ? Int[] : steps
        res = with_logger(NullLogger()) do
            JDyn.posterior_theoretical_moments(est; params=Matrix(r["pdraws"]), var_list=vl, ar=Int(r["ar"]), steps=steps,
                nodecomposition=r["nodecomposition"] == true, contemporaneous_correlation=r["contemporaneous_correlation"] == true, keep_draws=true)
        end
        n, DT, t = length(vl), Int(r["dt"]), tri(length(vl))
        pairs(a) = (c = [a[i, j, ntuple(_ -> :, ndims(a) - 2)...] for (i, j) in t]; ndims(a) == 2 ? reshape(Float64.(c), :, 1) : reduce(vcat, [reshape(x, 1, size(x)...) for x in c]))
        function cmpf(fam, p, take, dt)
            mx = 0.0
            for (f, k) in ((:mean, "mean"), (:median, "median"), (:variance, "var"), (:hpd_inf, "hpdinf"), (:hpd_sup, "hpdsup"), (:deciles, "deciles"))
                mx = max(mx, rdiff(take(getfield(fam, f)), r["$(p)_$k"]))
            end
            dd = take(fam.density); nd = ndims(dd)
            dd = dd[ntuple(i -> i == nd - 1 ? (1:dt:size(dd, nd - 1)) : Colon(), nd)...]
            max(mx, rdiff(dd, r["$(p)_density"]))
        end
        @test cmpf(res.covariance, "cov", pairs, 8) < 1e-9
        r["contemporaneous_correlation"] == true ? (@test cmpf(res.contemporaneous_correlation, "ccorr", pairs, 8) < 1e-9) : (@test res.contemporaneous_correlation === nothing)
        @test cmpf(res.correlation, "corr", identity, DT) < 1e-9
        haskey(r, "vd_mean") ? (@test cmpf(res.variance_decomposition, "vd", identity, DT) < 1e-9) : (@test res.variance_decomposition === nothing)
        haskey(r, "vdme_mean") ? (@test cmpf(res.variance_decomposition_me, "vdme", identity, DT) < 1e-9) : (@test res.variance_decomposition_me === nothing)
        haskey(r, "cvd_mean") ? (@test cmpf(res.conditional_variance_decomposition, "cvd", identity, DT) < 1e-9) : (@test res.conditional_variance_decomposition === nothing)
        haskey(r, "cvdme_mean") ? (@test cmpf(res.conditional_variance_decomposition_me, "cvdme", identity, DT) < 1e-9) : (@test res.conditional_variance_decomposition_me === nothing)
        k = size(r["draw_cov"], 1)
        tol = mod == "nk" ? 1e-11 : 1e-8
        @test rdiff(permutedims(reduce(hcat, [[res.draws.variance[b, i, j] for (i, j) in t] for b in 1:k])), r["draw_cov"]) < tol
        oo = JDyn.MStruct()
        JDyn.store_moments!(oo, res)
        @test haskey(oo["PosteriorTheoreticalMoments"]["dsge"], "covariance")
    end
end

@testset "posterior smoothed and filtered variables replay Dynare's sub-draws" begin
    for (mod, case) in (("nk", 3), ("schorfheide", 2), ("nk", 6))
        m, d = ploadmod("postmom_$mod", ["-DPM=$case"])
        r = pref("postmom_$(mod)_$case.mat")
        est = with_logger(NullLogger()) do
            JDyn.Estimation(m; datafile=joinpath(d, "est_$(mod)_data.m"), allow_unsupported=true)
        end
        P = Matrix(r["sm_params"]); size(P, 2) == length(est.names) || (P = reshape(P, :, length(est.names)))
        res = with_logger(NullLogger()) do
            JDyn.posterior_smoother_statistics(est; params=P, keep_draws=true)
        end
        @test res.var_names == String.(split(r["sm_varlist"], ";"))
        pidx = Int.(vec(r["pidx"]))
        dts = haskey(r, "dts") ? Int(r["dts"]) : 32
        prs = [("smv", res.smoothed_variables), ("upd", res.updated_variables), ("inn", res.smoothed_shocks),
               ("sme", res.smoothed_measurement_errors), ("cst", res.constant), ("tco", res.trend_coeff),
               ("ini", res.init_state), ("trd", res.trend), ("flt", res.filtered_variables)]
        append!(prs, [("fk$k", sd) for (k, sd) in sort(collect(res.filtered_k_step_ahead))])
        nchecked = 0
        for (pre, sd) in prs
            if !haskey(r, "$(pre)_mean")
                @test sd === nothing
                continue
            end
            nchecked += 1
            ix = size(sd.mean, 1) == 1 ? [1] : pidx
            for (f, k) in ((:mean, "mean"), (:median, "median"), (:var, "var"), (:hpd_inf, "hpdinf"), (:hpd_sup, "hpdsup"))
                @test rdiff(getfield(sd, f)[ix, :], r["$(pre)_$k"]) < 1e-8
            end
            @test rdiff(sd.deciles[:, ix, :], r["$(pre)_deciles"]) < 1e-8
            @test rdiff(sd.density[1:dts:end, :, ix, :], r["$(pre)_density"]) < 1e-7
        end
        @test nchecked >= 8
        haskey(r, "kstep_mean") && @test rdiff(res.filtered_k_step_ahead_mean, r["kstep_mean"]) < 1e-9
        dr = res.draws
        st(a) = permutedims(a, (3, 2, 1))
        nd = ndims(r["stock_smooth"]) == 3 ? size(r["stock_smooth"], 3) : 1
        for (k, rk) in (("smooth", "stock_smooth"), ("update", "stock_update"), ("innov", "stock_inno"), ("constant", "stock_const"), ("trend", "stock_trend"))
            @test rdiff(st(dr[k][1:nd, :, :]), r[rk]) < 1e-10
        end
        @test rdiff(permutedims(dr["init_state"][:, 1, :]), r["stock_init"]) < 1e-10
        @test rdiff(permutedims(dr["trend_coeff"][:, 1, :]), r["stock_tcoef"]) < 1e-10
        oo = JDyn.MStruct()
        JDyn.store_smoother_statistics!(oo, res)
        @test haskey(oo, "SmoothedVariables")
    end
end
