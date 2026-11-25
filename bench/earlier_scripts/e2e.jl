include("/home/claude/work/jdyn-ho/test/common.jl")
using JDyn, Logging, Random, Printf, LinearAlgebra
const OREF="/home/claude/work/jdyn-ho/test/ref_occbin"
odata(r) = (d = r["data"]; d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d))
function mkest(m, data; kw...)
    with_logger(NullLogger()) do
        JDyn.Estimation(m; data=data, nobs=size(data,1), first_obs=1, mode_compute=4, mh_replic=0, allow_unsupported=true,
                        occbin=Dict("likelihood_inversion_filter"=>true), kw...)
    end
end
mutable struct Rec; th::Vector{Vector{Float64}}; f::Vector{Float64}; code::Vector{Int}; t::Float64; fb::Int; pi::Int; sk::Int; err313::Int; last::Any; end
Rec() = Rec(Vector{Float64}[], Float64[], Int[], 0.0, 0, 0, 0, 0, nothing)
function record!(rc::Rec, est, th, f, code, dt)
    push!(rc.th, copy(th)); push!(rc.f, f); push!(rc.code, code); rc.t += dt
    s = JDyn.obc_settings(est)
    if s.method != :occbin && s.last_solver !== nothing && s.last_solver !== rc.last
        st = s.last_solver.stats; rc.fb += st.fallbacks; rc.pi += st.proved_infeasible; rc.sk += st.skipped; rc.err313 += get(st.errors, 313, 0)
        rc.last = s.last_solver
    end
end
function run_mode(est, x0)
    rc = Rec()
    fcn = th -> (t0=time(); r = JDyn._dsge_likelihood(est, th); record!(rc, est, th, r[1], r[2], time()-t0); (r[1], r[2], r[3]))
    t = @elapsed (fh, xh, _, _, iters, nfev, retcode, msg) = JDyn.csminwel(fcn, copy(x0); crit=1e-7, nit=1000, method=2, epsilon=1e-6, rng=MersenneTwister(1))
    (x=xh, f=fh, iters=iters, nfev=nfev, nevals=length(rc.f), time=t, rec=rc, msg=msg, retcode=retcode)
end
function firstdiff(a::Rec, b::Rec)
    for i in 1:min(length(a.th), length(b.th))
        maximum(abs, a.th[i] .- b.th[i]) > 1e-9 && return i
    end
    0
end
function replay(est, rc::Rec)
    f = zeros(length(rc.th)); code = zeros(Int, length(rc.th)); rr = Rec()
    for (i, th) in enumerate(rc.th)
        t0=time(); r = JDyn._dsge_likelihood(est, th); f[i] = r[1]; code[i] = r[2]; record!(rr, est, th, r[1], r[2], time()-t0)
    end
    f, code, rr
end
function compare(label, rc_o::Rec, f_n, code_n)
    # f is minus log posterior (penalty when code != 0)
    ok_o, ok_n = rc_o.code .== 0, code_n .== 0
    both = ok_o .& ok_n
    d = abs.(rc_o.f[both] .- f_n[both])
    @printf("  %s: %d vectors; both finite %d; max |dlp| %.3g; >1e-9: %d; >1e-6: %d; >0.01: %d; news fails only %d; OccBin fails only %d; both fail %d\n",
            label, length(ok_o), count(both), isempty(d) ? 0.0 : maximum(d), count(>(1e-9), d), count(>(1e-6), d), count(>(0.01), d),
            count(ok_o .& .!ok_n), count(.!ok_o .& ok_n), count(.!ok_o .& .!ok_n))
    println("    codes OccBin ", Dict(c=>count(==(c), rc_o.code) for c in unique(rc_o.code)), " news ", Dict(c=>count(==(c), code_n) for c in unique(code_n)))
end
function modes(name, mfile, x0fun)
    m = load_copy(joinpath(OREF, mfile)); r = matread(joinpath(OREF, "occest_$(name)_lik.mat")); data = odata(r)
    eo, en = mkest(m, data), mkest(m, data; obc_method="news")
    X = r["X"]; X = X isa AbstractVector ? reshape(X, eo.n, :) : (size(X,1)==eo.n ? X : Matrix(X'))
    x0 = x0fun(eo, X)
    println("== mode, $name, start $(x0fun === nothing ? "" : "")"); flush(stdout)
    ro = run_mode(eo, x0); rn = run_mode(en, x0)
    @printf("  OccBin: f=%.7f nfev=%d iters=%d time=%.1fs per eval %.1f ms retcode %s\n  news  : f=%.7f nfev=%d iters=%d time=%.1fs per eval %.1f ms retcode %s\n",
            -ro.f, ro.nfev, ro.iters, ro.time, 1000ro.rec.t/ro.nevals, ro.retcode, -rn.f, rn.nfev, rn.iters, rn.time, 1000rn.rec.t/rn.nevals, rn.retcode)
    println("  OccBin mode ", round.(ro.x, sigdigits=6)); println("  news   mode ", round.(rn.x, sigdigits=6))
    println("  max |mode diff| ", maximum(abs, ro.x .- rn.x), "; first evaluation with different vector: ", firstdiff(ro.rec, rn.rec))
    println("  news solver in its own search: fallbacks $(rn.rec.fb), proved infeasible $(rn.rec.pi), skipped $(rn.rec.sk), 313 $(rn.rec.err313)")
    fn, cn, rr = replay(en, ro.rec); compare("news at OccBin's $(length(ro.rec.th)) vectors", ro.rec, fn, cn)
    println("    solver counts there: fallbacks $(rr.fb), proved infeasible $(rr.pi), skipped $(rr.sk)")
    fo, co, _ = replay(eo, rn.rec); compare2 = Rec(); compare2.f = rn.rec.f; compare2.code = rn.rec.code
    compare("OccBin at news's $(length(rn.rec.th)) vectors (columns: 'OccBin' = news run, 'news' = OccBin replay)", compare2, fo, co)
    flush(stdout)
    (eo, en, ro, rn, data)
end
function mcmc(label, eo, en, mode_x, cov, js, nch, ndraw, seed)
    lb, ub = JDyn.prior_bounds(eo.prior, eo.options["prior_trunc"])
    d = Matrix(cholesky(Symmetric(cov)).U); chol = d * js
    res = []
    for ch in 1:nch
        rng = MersenneTwister(seed + ch)
        start = nothing; 
        for _ in 1:200
            cand = mode_x + vec(randn(rng, 1, eo.n) * (d * (2*js)))
            all(cand .> lb) && all(cand .< ub) && isfinite(JDyn.log_posterior(eo, cand)) && (start = cand; break)
        end
        inn = randn(rng, ndraw, eo.n); u = rand(rng, ndraw)
        out = []
        for e in (eo, en)
            rc = Rec()
            lp = th -> (t0=time(); r = JDyn._dsge_likelihood(e, th; ); v = r[2]==0 ? -r[1] : -Inf; record!(rc, e, th, r[1], r[2], time()-t0); v)
            lp0 = JDyn.log_posterior(e, start)
            t = @elapsed (dr, lpo, acc) = JDyn.metropolis_hastings(lp, start, lp0, chol, lb, ub, inn, u)
            push!(out, (dr=dr, lp=lpo, acc=acc, rec=rc, time=t))
        end
        o, n = out
        same = o.dr == n.dr
        fd = findfirst(i -> o.dr[i,:] != n.dr[i,:], 1:ndraw)
        @printf("  chain %d: acceptance OccBin %.3f news %.3f; wall OccBin %.1fs news %.1fs (%.1fx); identical draws: %s; max draw diff %.3g; first differing draw: %s\n",
                ch, mean(o.acc), mean(n.acc), o.time, n.time, o.time/n.time, same, maximum(abs, o.dr .- n.dr), fd)
        k = min(length(o.rec.f), length(n.rec.f))
        pair = [(i, o.rec.th[i] == n.rec.th[i]) for i in 1:k]
        npair = count(last, pair)
        both = [i for (i, s) in pair if s && o.rec.code[i]==0 && n.rec.code[i]==0]
        dd = [abs(o.rec.f[i]-n.rec.f[i]) for i in both]
        bad = [i for (i,s) in pair if s && (o.rec.code[i]==0) != (n.rec.code[i]==0)]
        @printf("    evaluations %d (OccBin) / %d (news); at identical vectors %d; likelihood differences >1e-9: %d (max %.4g) ; finite in one only: %d; codes OccBin %s news %s\n",
                length(o.rec.f), length(n.rec.f), npair, count(>(1e-9), dd), isempty(dd) ? 0.0 : maximum(dd), length(bad),
                Dict(c=>count(==(c), o.rec.code) for c in unique(o.rec.code)), Dict(c=>count(==(c), n.rec.code) for c in unique(n.rec.code)))
        for i in both
            abs(o.rec.f[i]-n.rec.f[i]) > 1e-9 && (@printf("      first: evaluation %d: %.6f (OccBin) vs %.6f (news), diff %.4g\n", i, -o.rec.f[i], -n.rec.f[i], abs(o.rec.f[i]-n.rec.f[i])); break)
        end
        println("    news solver over the chain: fallbacks $(n.rec.fb), proved infeasible $(n.rec.pi), skipped $(n.rec.sk), 313 $(n.rec.err313)")
        flush(stdout)
        push!(res, (o, n))
    end
    res
end
using Statistics
which = ARGS
if "nkms" in which
    eo, en, ro, rn, data = modes("nkms", "occest_nkms.mod", (e, X) -> copy(e.x0))
    mo = JDyn.ModeResult(copy(ro.x), eo.names, -ro.f); JDyn.add_hessian!(eo, mo)
    println("  Hessian PD (OccBin mode): ", mo.inv_hessian !== nothing)
    if mo.inv_hessian !== nothing
        for js in (0.4, 0.9, 1.3, 1.8)   # pilot acceptance with news
            lb, ub = JDyn.prior_bounds(en.prior, en.options["prior_trunc"])
            d = Matrix(cholesky(Symmetric(mo.inv_hessian)).U) * js
            rng = MersenneTwister(5)
            lp0 = JDyn.log_posterior(en, mo.x)
            dr, lpo, acc = JDyn.metropolis_hastings(th -> JDyn.log_posterior(en, th), mo.x, lp0, d, lb, ub, randn(rng, 300, en.n), rand(rng, 300))
            @printf("  pilot jscale %.2f: acceptance %.3f\n", js, mean(acc)); flush(stdout)
        end
    end
end
if "nkms_mcmc" in which
    eo, en, ro, rn, data = modes("nkms", "occest_nkms.mod", (e, X) -> copy(e.x0))
    mo = JDyn.ModeResult(copy(ro.x), eo.names, -ro.f); JDyn.add_hessian!(eo, mo)
    println("== MCMC nkms, 2 chains x 2000, jscale 1.3, proposal inverse Hessian at OccBin's mode"); flush(stdout)
    mcmc("nkms", eo, en, mo.x, mo.inv_hessian, 1.3, 2, 2000, 20261124)
end
if "nkm" in which
    modes("nkm", "occest_nkm.mod", (e, X) -> copy(e.x0))
    modes("nkm", "occest_nkm.mod", (e, X) -> copy(X[:, 1]))
end
if "nkm_mcmc" in which
    eo, en, ro, rn, data = modes("nkm", "occest_nkm.mod", (e, X) -> copy(X[:, 1]))
    println("== MCMC nkm, 2 chains x 1500, jscale 0.07, proposal prior variance, start around OccBin's mode"); flush(stdout)
    mcmc("nkm", eo, en, ro.x, Matrix(Diagonal(eo.prior.p2 .^ 2)), 0.07, 2, 1500, 20261124)
end
if "first30" in which
    m = load_copy(joinpath(OREF, "occest_nkm.mod")); r = matread(joinpath(OREF, "occest_nkm_lik.mat")); data = odata(r)
    eo, en = mkest(m, data), mkest(m, data; obc_method="news")
    fcn(e, rc) = th -> (t0=time(); rr = JDyn._dsge_likelihood(e, th); record!(rc, e, th, rr[1], rr[2], time()-t0); (rr[1], rr[2], rr[3]))
    ro, rn = Rec(), Rec()
    try; JDyn.csminwel(fcn(eo, ro), copy(eo.x0); crit=1e-7, nit=12, method=2, epsilon=1e-6, rng=MersenneTwister(1)); catch; end
    fn, cn, _ = replay(en, ro)
    for i in 1:length(ro.f)
        (isfinite(ro.f[i]) && isfinite(fn[i]) && abs(ro.f[i]-fn[i]) > 1e-9) && (println("first evaluation with different likelihood among OccBin's path: ", i, " (", -ro.f[i], " vs ", -fn[i], ") of ", length(ro.f)); break)
    end
    println("OccBin path length ", length(ro.f), "; max |dlp| over path ", maximum(abs(ro.f[i]-fn[i]) for i in 1:length(fn) if isfinite(ro.f[i]) && isfinite(fn[i])))
end
if "bf" in which
    EDIR = "/home/claude/work/jdyn-ho/test/obc/est"
    dd = matread(joinpath(EDIR, "occest_bf_data.mat")); data = hcat(Float64.(vec(dd["q"])), Float64.(vec(dd["r"])))
    m = load_copy(joinpath(EDIR, "bf.mod"))
    eo, en = mkest(m, data), mkest(m, data; obc_method="news")
    x0 = copy(eo.x0); println("bf x0 ", x0, " lp ", JDyn.log_posterior(eo, x0))
    cov = Matrix(Diagonal(eo.prior.p2 .^ 2))
    lb, ub = JDyn.prior_bounds(en.prior, en.options["prior_trunc"])
    for js in (0.05, 0.1, 0.2, 0.4)
        rng = MersenneTwister(5); d = Matrix(cholesky(Symmetric(cov)).U) * js
        dr, lpo, acc = JDyn.metropolis_hastings(th -> JDyn.log_posterior(en, th), x0, JDyn.log_posterior(en, x0), d, lb, ub, randn(rng, 200, en.n), rand(rng, 200))
        @printf("  pilot jscale %.2f: acceptance %.3f\n", js, mean(acc)); flush(stdout)
    end
end
if "bf_mcmc" in which
    EDIR = "/home/claude/work/jdyn-ho/test/obc/est"
    dd = matread(joinpath(EDIR, "occest_bf_data.mat")); data = hcat(Float64.(vec(dd["q"])), Float64.(vec(dd["r"])))
    m = load_copy(joinpath(EDIR, "bf.mod"))
    eo, en = mkest(m, data), mkest(m, data; obc_method="news")
    println("== MCMC bf, 2 chains x 1000, prior variance, jscale ", ARGS[2]); flush(stdout)
    mcmc("bf", eo, en, copy(eo.x0), Matrix(Diagonal(eo.prior.p2 .^ 2)), parse(Float64, ARGS[2]), 2, 1000, 20261124)
end
