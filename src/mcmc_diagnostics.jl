# MCMC convergence diagnostics of Dynare 7.2 (`mcmc_diagnostics.m` and friends), computed on the draws of an `MCMCResult`:
#
#     cd = mcmc_diagnostics(est, mc)            # options (geweke_interval, taper_steps, raftery_lewis_*) from the estimation command
#     cd.inefficiency_factors                   # (chains x parameters), Dynare's `Ifac`
#     cd.geweke[b]["SE_ea"]                     # Dict: posteriormean, posteriorstd, nse_iid, rne_iid, nse_taper_k, ..., pooled_mean, ...
#     cd.raftery_lewis[b]                       # Dict of vectors (M_burn, N_prec, k_thin, k_ind, I_stat, N_total) and N_min
#     cd.brooks_gelman.univariate               # (points x 6 x parameters); multivariate (points x 6): the posterior kernel
#
# Derived from Dynare 7.2's matlab/convergence_diagnostics/{mcmc_diagnostics, mcmc_diagnostics_core, geweke_moments,
# geweke_chi2_test, mcmc_ifac, raftery_lewis}.m (GPL-3.0-or-later, Dynare Team) through pydyn's port. Conventions of Dynare that
# are reproduced on purpose:
#   * the inefficiency factor is 1 + 2 sum_{i=0}^{Nc} w((i+1)/Nc) rho_i: Parzen window shifted by one lag, lag 0 included
#     (about 3 for independent draws);
#   * the Geweke statistics of the whole chain include the burn-in; the two sub-samples that are compared exclude it;
#   * Raftery-Lewis is computed on the last file of draws of each chain (the whole chain unless it exceeds
#     ceil(MaxNumberOfBytes / (npar + 2) / 8) draws), burn-in included, with `quantile` method 5.
# The Brooks-Gelman series are what Dynare plots; it does not store them in oo_ and neither does `store_convergence!`.

"Dynare's `get_the_name.m`: SE_<exo>, SE_EOBS_<obs>, CC_<a>_<b>, CC_EOBS_<a>_<b> or the parameter name, in theta order."
function dynare_parameter_names(est::Estimation)
    sl = _slots(est)
    out = Vector{String}(undef, est.n)
    for (grp, nm, i) in sl
        out[i] = grp == "parameters" ? nm :
                 grp == "shocks_std" ? "SE_" * nm :
                 grp == "measurement_errors_std" ? "SE_EOBS_" * nm :
                 grp == "shocks_corr" ? "CC_" * nm : "CC_EOBS_" * nm
    end
    out
end

"MATLAB's `round`: halves away from zero."
_mround(x::Real) = x >= 0 ? floor(x + 0.5) : -floor(-x + 0.5)
_iround(x::Real) = Int(_mround(x))

"Draws kept after the burn-in of every chain: (chains, kept, parameters) (`CutSample.m`: the first floor(mh_drop * replic) are dropped)."
function kept_array(mc::MCMCResult; drop=mc.drop)
    first = max(0, floor(Int, drop * size(mc.draws, 2)))
    mc.draws[:, first+1:end, :]
end

"""
    mcmc_ifac(x, Nc) -> Float64

Inefficiency factor of one series with a Parzen window of `Nc` lags, exactly as `mcmc_ifac.m` (see the file header).
"""
function mcmc_ifac(x::AbstractVector, Nc)
    n = length(x)
    Nc = floor(Int, min(Nc, n / 2))
    isodd(Nc) && (Nc -= 1)
    m = sum(x) / n
    xc = Float64.(x) .- m
    sd2 = sum(abs2, xc) / n                         # std(y, 1)^2
    s = 1.0                                         # lag 0: parzen(1) * 1 added below
    tot = 0.0
    for i in 0:Nc
        a = i == 0 ? 1.0 : dot(view(xc, i+1:n), view(xc, 1:n-i)) / (n * sd2)
        k = i + 1
        w = k <= Nc / 2 ? 1 - 6 * (k / Nc)^2 + 6 * (k / Nc)^3 : 2 * (1 - k / Nc)^3
        tot += w * a
    end
    1 + 2 * tot
end

"Inefficiency factors of every parameter in every chain, (chains x parameters), from the kept draws with `min(1000, kept/2)` lags."
function inefficiency_factors(mc::MCMCResult; drop=mc.drop)
    k = kept_array(mc; drop=drop)
    nb, T, n = size(k)
    nc = min(1000, T / 2)
    [mcmc_ifac(view(k, b, :, j), nc) for b in 1:nb, j in 1:n]
end

"Sum from left to right (MATLAB's `sum` is not pairwise; Dynare computes a variance as E[x^2] - E[x]^2, which keeps few digits for a tight parameter)."
function _seqsum(a)
    s = 0.0
    @inbounds for v in a
        s += v
    end
    s
end

"`geweke_moments.m` of one series: a Dict with posteriormean, posteriorstd, nse_iid, rne_iid and nse/rne_taper_k per taper step."
function geweke_moments(x::AbstractVector, taper_steps)
    ng = 100
    ns = length(x) ÷ ng
    used = ns * ng
    wm = zeros(ng)
    wv = zeros(ng)
    for ig in 1:ng
        s1 = 0.0
        s2 = 0.0
        for i in (ig-1)*ns+1:ig*ns
            s1 += x[i]
            s2 += x[i]^2
        end
        wm[ig] = s1 / ns
        wv[ig] = s2 / ns
    end
    mean_ = _seqsum(wm) / ng
    var_ = _seqsum(wv) / ng - mean_^2
    xs = view(x, 1:used)
    mu = _seqsum(xs) / used
    nse = sqrt(_seqsum(abs2.(xs .- mu)) / used) / sqrt(used)       # std(x, 1) / sqrt(n)
    out = Dict{String,Any}("posteriormean" => mean_, "posteriorstd" => sqrt(var_), "nse_iid" => nse,
                               "rne_iid" => var_ / (used * nse^2))
    c = wm .- mean_
    acov = [dot(view(c, lag+1:ng), view(c, 1:ng-lag)) / ng for lag in 0:ng-1]
    for t in taper_steps
        t = Int(t)
        s = acov[1] + sum(2 * (1 - l / t) * acov[1+l] for l in 1:t-1; init=0.0)
        nse_t = sqrt(s / ng)
        out["nse_taper_$t"] = nse_t
        out["rne_taper_$t"] = var_ / (used * nse_t^2)
    end
    out
end

"`geweke_chi2_test.m`: pooled mean, pooled nse and the p-value of the chi2 test of equal means, one column per taper (the first is no taper)."
function _geweke_chi2(r1::Dict, r2::Dict, taper_steps)
    keys_ = ["nse_iid"; ["nse_taper_$(Int(t))" for t in taper_steps]]
    K = length(keys_)
    pm, pn, pp = zeros(K), zeros(K), zeros(K)
    for (k, nk) in enumerate(keys_)
        nse = (r1[nk], r2[nk])
        means = (r1["posteriormean"], r2["posteriormean"])
        w = 1 / nse[1]^2 + 1 / nse[2]^2
        pm[k] = (means[1] / nse[1]^2 + means[2] / nse[2]^2) / w
        pn[k] = 1 / sqrt(w)
        stat = (means[1] - means[2])^2 / (nse[1]^2 + nse[2]^2)
        pp[k] = 1 - _chi2cdf1(stat)
    end
    pm, pn, pp
end
# chi2 cdf with one degree of freedom; 1 - cdf is computed through erfc so that small p-values keep their digits
_chi2cdf1(x) = x <= 0 ? 0.0 : 1 - SpecialFunctions.erfc(sqrt(x / 2))

function _first_vs_second_order(d::AbstractVector{Int})
    n = length(d)
    tran = zeros(2, 2, 2)
    for t in 3:n
        tran[d[t-2]+1, d[t-1]+1, d[t]+1] += 1
    end
    g2 = 0.0
    for a in 1:2, b in 1:2, c in 1:2
        if tran[a, b, c] != 0
            fitted = (tran[a, b, 1] + tran[a, b, 2]) * (tran[1, b, c] + tran[2, b, c]) /
                     (tran[1, b, 1] + tran[1, b, 2] + tran[2, b, 1] + tran[2, b, 2])
            g2 += log(tran[a, b, c] / fitted) * tran[a, b, c]
        end
    end
    g2 *= 2
    g2, g2 - log(n - 2) * 2
end

function _independence_test(d::AbstractVector{Int})
    n = length(d)
    trans = zeros(2, 2)
    for i in 2:n
        trans[d[i-1]+1, d[i]+1] += 1
    end
    dcm1 = n - 1
    g2 = 0.0
    for a in 1:2, b in 1:2
        if trans[a, b] != 0
            fitted = (trans[a, 1] + trans[a, 2]) * (trans[1, b] + trans[2, b]) / dcm1
            g2 += log(trans[a, b] / fitted) * trans[a, b]
        end
    end
    g2 *= 2
    g2, g2 - log(dcm1)
end

"""
    raftery_lewis(runs, q=0.025, r=0.005, s=0.95) -> Dict

Raftery and Lewis (1992), `raftery_lewis.m`, for every column of `runs`: M_burn, N_prec, k_thin, k_ind, I_stat, N_total (vectors)
and N_min (scalar).
"""
function raftery_lewis(runs::AbstractMatrix, q=0.025, r=0.005, s=0.95)
    (0 < q < 1) || throw(ArgumentError("Quantile must be between 0 and 1"))
    n_runs, n_vars = size(runs)
    Phi = _norminv((s + 1) / 2)
    out = Dict{String,Any}(k => fill(NaN, n_vars) for k in ("M_burn", "N_prec", "k_thin", "k_ind", "I_stat", "N_total"))
    N_min = trunc(Phi^2 * (1 - q) * q / r^2 + 1)
    out["N_min"] = N_min
    epss = 0.001
    for nv in 1:n_vars
        qv = quantile_r5(runs[:, nv], q)
        work = Int.(runs[:, nv] .<= qv)
        k_thin, bic = 1, 1.0
        thinned = work
        while bic > 0
            thinned = work[1:k_thin:n_runs]
            _, bic = _first_vs_second_order(thinned)
            k_thin += 1
        end
        k_thin -= 1
        tm = zeros(2, 2)
        for i in 2:length(thinned)
            tm[thinned[i-1]+1, thinned[i]+1] += 1
        end
        alpha = tm[1, 2] / (tm[1, 1] + tm[1, 2])
        beta = tm[2, 1] / (tm[2, 1] + tm[2, 2])
        kmind = k_thin
        _, bic = _independence_test(thinned)
        while bic > 0
            thinned = work[1:kmind:n_runs]
            _, bic = _independence_test(thinned)
            kmind += 1
        end
        m_star = log((alpha + beta) * epss / max(alpha, beta)) / log(abs(1 - alpha - beta))
        out["M_burn"][nv] = trunc((m_star + 1) * k_thin)
        n_star = (2 - (alpha + beta)) * alpha * beta * Phi^2 / ((alpha + beta)^3 * r^2)
        out["N_prec"][nv] = trunc(n_star + 1) * k_thin
        out["I_stat"][nv] = (out["M_burn"][nv] + out["N_prec"][nv]) / N_min
        out["k_ind"][nv] = max(trunc(out["I_stat"][nv] + 1), kmind)
        out["k_thin"][nv] = k_thin
        out["N_total"][nv] = out["M_burn"][nv] + out["N_prec"][nv]
    end
    out
end

"Inverse of the standard normal cdf (via the inverse complementary error function)."
_norminv(p) = -sqrt(2) * SpecialFunctions.erfcinv(2p)

"""
Six Brooks-Gelman series of one scalar series `tmp` (chains x draws), `mcmc_diagnostics_core.m`: at the draws `origin:step:T`,
over the draws after the first `drop` share up to that point: interval length, second and third central absolute moments, each
pooled and then (averaged over chains) within.
"""
function _bg_series(tmp::AbstractMatrix, origin::Int, step::Int, drop::Float64)
    nb, T = size(tmp)
    alpha = 0.2
    iters = collect(origin:step:T)
    out = zeros(length(iters), 6)
    for (w, it) in enumerate(iters)
        linea = ceil(Int, drop * it)
        n = it - linea + 1
        cinf, csup = max(1, _iround(n * alpha / 2)), _iround(n * (1 - alpha / 2))
        CINF, CSUP = max(1, _iround(nb * n * alpha / 2)), _iround(nb * n * (1 - alpha / 2))
        lo = max(linea, 1)
        pooled = sort(vec(@view tmp[:, lo:it]))
        out[w, 1] = pooled[CSUP] - pooled[CINF]
        pm = sum(pooled) / length(pooled)
        out[w, 3] = sum(abs2, pooled .- pm) / (nb * n - 1)
        out[w, 5] = sum(x -> abs(x)^3, pooled .- pm) / (nb * n - 1)
        for b in 1:nb
            pmet = sort(@view tmp[b, lo:it])
            out[w, 2] += pmet[csup] - pmet[cinf]
            wm = sum(pmet) / length(pmet)
            out[w, 4] += sum(abs2, pmet .- wm) / (n - 1)
            out[w, 6] += sum(x -> abs(x)^3, pmet .- wm) / (n - 1)
        end
    end
    out[:, [2, 4, 6]] ./= nb
    iters, out
end

"Brooks-Gelman diagnostics: `x` (evaluation points), `univariate` (points x 6 x parameters) and `multivariate` (points x 6, the posterior kernel)."
struct BrooksGelman
    x::Vector{Int}
    univariate::Union{Nothing,Array{Float64,3}}
    multivariate::Union{Nothing,Matrix{Float64}}
end

function brooks_gelman(mc::MCMCResult; drop=mc.drop)
    nb, T, n = size(mc.draws)
    (nb < 2 || T <= 2000) && return BrooksGelman(Int[], nothing, nothing)
    origin = 1000
    step = ceil(Int, (T - origin) / 100)
    uni = nothing
    xs = Int[]
    for j in 1:n
        x, s = _bg_series(mc.draws[:, :, j], origin, step, drop)
        uni === nothing && (uni = zeros(length(x), 6, n); xs = x)
        uni[:, :, j] = s
    end
    _, multi = _bg_series(mc.log_posterior, origin, step, drop)
    BrooksGelman(xs, uni, multi)
end

"Result of `mcmc_diagnostics`."
struct ConvergenceDiagnostics
    names::Vector{String}
    inefficiency_factors::Matrix{Float64}
    geweke::Union{Nothing,Vector{Dict{String,Any}}}              # one Dict per chain, keyed by Dynare's parameter names
    geweke_samples::Union{Nothing,NTuple{4,Int}}                 # first draw of the begin sample, its last, first of the end sample, last
    raftery_lewis::Union{Nothing,Vector{Dict{String,Any}}}
    brooks_gelman::Union{Nothing,BrooksGelman}
    taper_steps::Vector{Int}
end

"""
    mcmc_diagnostics(est, mc; geweke_interval, taper_steps, raftery_lewis_qrs, raftery_lewis=false, drop) -> ConvergenceDiagnostics

What Dynare's `mcmc_diagnostics` computes after a Metropolis-Hastings run. Options default to those of the estimation command
(`est.options`: `convergence.geweke.geweke_interval` [0.2 0.5], `convergence.geweke.taper_steps` [4 8 15],
`convergence.rafterylewis.indicator` / `.qrs` [0.025 0.005 0.95], `mh_drop`). With 2000 draws per chain or fewer only the
inefficiency factors are computed (Dynare returns); Brooks-Gelman needs two chains.
"""
function mcmc_diagnostics(est::Estimation, mc::MCMCResult; geweke_interval=nothing, taper_steps=nothing, raftery_lewis_qrs=nothing,
                          raftery_lewis=nothing, drop=nothing)
    o = est.options
    geweke_interval = geweke_interval === nothing ? get(o, "convergence.geweke.geweke_interval", [0.2, 0.5]) : geweke_interval
    taper_steps = taper_steps === nothing ? get(o, "convergence.geweke.taper_steps", [4, 8, 15]) : taper_steps
    do_rl = raftery_lewis === nothing ? _est_flag(get(o, "convergence.rafterylewis.indicator", false)) : raftery_lewis
    qrs = raftery_lewis_qrs === nothing ? get(o, "convergence.rafterylewis.qrs", [0.025, 0.005, 0.95]) : raftery_lewis_qrs
    mcmc_diagnostics(dynare_parameter_names(est), mc; geweke_interval, taper_steps, raftery_lewis_qrs=qrs, raftery_lewis=do_rl, drop)
end

function mcmc_diagnostics(names::Vector{String}, mc::MCMCResult; geweke_interval=[0.2, 0.5], taper_steps=[4, 8, 15],
                          raftery_lewis_qrs=[0.025, 0.005, 0.95], raftery_lewis::Bool=false, drop=nothing)
    drop = drop === nothing ? mc.drop : Float64(drop)
    nb, T, n = size(mc.draws)
    ifac = inefficiency_factors(mc; drop=drop)
    taper = Int.(collect(taper_steps isa Real ? [taper_steps] : taper_steps))
    T <= 2000 && return ConvergenceDiagnostics(names, ifac, nothing, nothing, nothing, nothing, taper)
    gi = Float64.(collect(geweke_interval))
    if length(gi) != 2 || any(<(0), gi) || any(>(1), gi) || gi[2] - gi[1] < 0
        @info "CONVERGENCE DIAGNOSTICS: Invalid option for geweke_interval. Using the default of [0.2 0.5]."
        gi = [0.2, 0.5]
    end
    first_obs = max(1, ceil(Int, drop * T))
    last_begin = first_obs + _iround(gi[1] * T * (1 - drop))
    first_end = first_obs + _iround(gi[2] * T * (1 - drop))
    geweke = Dict{String,Any}[]
    for b in 1:nb
        G = Dict{String,Any}()
        for j in 1:n
            x = mc.draws[b, :, j]
            res = geweke_moments(x, taper)
            r1 = geweke_moments(x[first_obs:last_begin], taper)
            r2 = geweke_moments(x[first_end:end], taper)
            pm, pn, pp = _geweke_chi2(r1, r2, taper)
            res["pooled_mean"] = _row(pm)
            res["pooled_nse"] = _row(pn)
            res["prob_chi2_test"] = _row(pp)
            G[names[j]] = res
        end
        push!(geweke, G)
    end
    rl = nothing
    if raftery_lewis
        q = Float64.(collect(raftery_lewis_qrs))
        if length(q) != 3 || any(<(0), q) || any(>(1), q) || q[1] - q[2] <= 0
            @info "Invalid option for raftery_lewis_qrs. Using the default of [0.025 0.005 0.95]."
            q = [0.025, 0.005, 0.95]
        end
        maxruns = ceil(Int, 1e7 / (n + 2) / 8)          # draws per file of the chain (options_.MaxNumberOfBytes = 1e7)
        lastfile = T > maxruns ? T - ((T - 1) ÷ maxruns) * maxruns : T
        rl = [begin
                  R = raftery_lewis_core(mc.draws[b, T-lastfile+1:T, :], q[1], q[2], q[3])
                  R["parameter_names"] = names
                  R
              end for b in 1:nb]
    end
    bg = nb >= 2 ? brooks_gelman(mc; drop=drop) : nothing
    ConvergenceDiagnostics(names, ifac, geweke, (first_obs, last_begin, first_end, T), rl, bg, taper)
end
raftery_lewis_core(args...) = raftery_lewis(args...)

"""
    store_convergence!(oo, cd)

`oo_.convergence.geweke(b).<name>` and `oo_.convergence.raftery_lewis(b)` as `mcmc_diagnostics.m` writes them (struct arrays are
1 x nblck Cells of structs). A previous `oo_.convergence.raftery_lewis` is removed when Raftery-Lewis is asked for again.
"""
function store_convergence!(oo::MStruct, cd::ConvergenceDiagnostics)
    cd.geweke === nothing && return oo
    C = get(oo, "convergence", nothing)
    C isa MStruct || (C = MStruct(); oo.convergence = C)
    if cd.raftery_lewis !== nothing
        delete!(C, "raftery_lewis")
    end
    gw = Any[]
    for G in cd.geweke
        S = MStruct()
        for nm in cd.names
            r = MStruct()
            for k in ("posteriormean", "posteriorstd", "nse_iid", "rne_iid")
                r[k] = G[nm][k]
            end
            for t in cd.taper_steps
                r["nse_taper_$t"] = G[nm]["nse_taper_$t"]
                r["rne_taper_$t"] = G[nm]["rne_taper_$t"]
            end
            for k in ("pooled_mean", "pooled_nse", "prob_chi2_test")
                r[k] = G[nm][k]
            end
            S[nm] = r
        end
        push!(gw, S)
    end
    C.geweke = Cell(gw)
    if cd.raftery_lewis !== nothing
        rl = Any[]
        for R in cd.raftery_lewis
            S = MStruct()
            for k in ("M_burn", "N_prec", "k_thin", "k_ind", "I_stat", "N_total")
                S[k] = _col(R[k])
            end
            S["N_min"] = R["N_min"]
            S["parameter_names"] = _cellcol(R["parameter_names"])
            push!(rl, S)
        end
        C.raftery_lewis = Cell(rl)
    end
    oo
end
