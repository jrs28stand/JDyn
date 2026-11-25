# Sequential Monte Carlo samplers of Dynare 7.2: Herbst and Schorfheide's tempered SMC (`hssmc`) and the dynamic striated
# Metropolis-Hastings sampler (`dsmh`) of matlab/estimation/smc/{hssmc, dsmh, smc_samplers_initialization, tempered_likelihood}.m and
# matlab/estimation/resampler/kitagawa.m (port of pydyn/samplers.py). Random numbers as in src/samplers.jl: with `mcmc_rng=:octave` the
# particle j of the initialisation is drawn from OctaveRNG(seed + j) and everything after it from OctaveRNG(seed + particles), as Dynare
# leaves its global stream (`smc_samplers_initialization.m`); `initial` supplies the initial particles (Dynare's gamma/beta/Weibull
# prior draws come from generators it does not seed, so only runs started from given particles can be replayed with such priors).

"""
Result of a sequential Monte Carlo sampler: the final equally weighted `particles` (n x parameters), their tempered log posterior
kernel (`log_posterior`, the posterior kernel at the end) and `log_likelihood`, the log marginal data density estimate
(`log_marginal_density`: Dynare's `oo_.MarginalDensity.hssmc` / `.dsmh`), the tempering schedule and, per stage, the effective sample
size, acceptance rate and scale. `as_mcmc(res)` presents the particles as one chain without burn-in.
"""
struct SMCResult
    method::String
    particles::Matrix{Float64}
    log_posterior::Vector{Float64}
    log_likelihood::Vector{Float64}
    log_marginal_density::Float64
    names::Vector{String}
    tempering::Vector{Float64}
    ess::Vector{Float64}
    acceptance::Vector{Float64}
    scale::Vector{Float64}
    weights::Union{Nothing,Vector{Float64}}
    stages::Vector{Dict{Symbol,Any}}
end

"The particles as one chain without burn-in (`MCMCResult`), for the posterior statistics."
function as_mcmc(r::SMCResult)
    n = size(r.particles, 1)
    MCMCResult(reshape(copy(r.particles), 1, n, :), reshape(copy(r.log_posterior), 1, n), [NaN], r.names, copy(r.particles[1:1, :]), 0.0,
               trues(1, n), nothing, Dict{Symbol,Any}(:method => r.method, :smc => true))
end

"Systematic resampling (`kitagawa.m`): 1-based indices of `m` (default: as many as weights) particles."
function kitagawa(weights::AbstractVector, noise::Real, m::Union{Nothing,Integer}=nothing)
    n = m === nothing ? length(weights) : Int(m)
    cw = cumsum(weights)
    out = Vector{Int}(undef, n)
    j = 1
    for i in 1:n
        w = ((i - 1) + noise) * (1.0 / n)
        while w > cw[j]
            j += 1
        end
        out[i] = j
    end
    out
end

function _smc_setup(est::Estimation, kind::Symbol, seed::Integer, n::Int)
    lb, ub = prior_bounds(est.prior, est.options["prior_trunc"])
    function tempered(x, lam)
        lpk = try
            Float64(log_posterior(est, x; bounds=(lb, ub)))
        catch
            -Inf
        end
        lprior = log_prior(est, x)
        ll = lpk - lprior
        lam * ll + lprior, ll
    end
    r = kind == :octave ? OctaveRNG(seed + n) : Random.Xoshiro(seed * 7919 + n)
    lb, ub, tempered, r
end

function _smc_init(est::Estimation, n::Int, lb, ub, tempered, kind::Symbol, seed::Integer, initial)
    P = zeros(n, est.n); tlp = zeros(n); ll = zeros(n)
    for j in 1:n
        if initial !== nothing
            P[j, :] = initial[j, :]
            tlp[j], ll[j] = tempered(P[j, :], 0.0)
            isfinite(ll[j]) || throw(EstimationError("initial particle $j: the likelihood is not finite"))
            continue
        end
        rj = kind == :octave ? OctaveRNG(seed + j) : Random.Xoshiro(seed * 7919 + 31 * j)
        while true
            cand = prior_draw(est.prior, lb, ub, rj)
            if all(cand .>= lb) && all(cand .<= ub)
                P[j, :] = cand
                tlp[j], ll[j] = tempered(cand, 0.0)
                isfinite(ll[j]) && break
            end
        end
    end
    P, tlp, ll
end

_sexp(x) = x > 700 ? Inf : exp(x)

"""
    sample_smc(est; method, sampler_options, seed, rng, initial, verbose) -> SMCResult

`hssmc` or `dsmh` (default: the `posterior_sampling_method` option). `initial` (particles x parameters) replaces the prior draws of
the first stage.
"""
function sample_smc(est::Estimation; method=nothing, sampler_options=nothing, seed=0, rng=nothing, initial=nothing, verbose=false)
    method = String(method === nothing ? get(est.options, "posterior_sampling_method", HSSMC) : method)
    method in SMC_METHODS || throw(EstimationError("$method is not a sequential Monte Carlo sampler"))
    spec = sampler_options === nothing ? get(est.options, "posterior_sampler_options", nothing) : sampler_options
    opts = JDyn.sampler_options(method, spec)
    kind = _mcmc_kind(est, rng)
    method == HSSMC ? _hssmc(est, opts, Int(seed), kind, initial, verbose) : _dsmh(est, opts, Int(seed), kind, initial, verbose)
end

function _hssmc(est::Estimation, opts, seed, kind, initial, verbose)
    N, steps = Int(opts["particles"]), Int(opts["steps"])
    lb, ub, tempered, r = _smc_setup(est, kind, seed, N)
    npar = est.n
    phi = ((0:steps-1) ./ (steps - 1)) .^ opts["lambda"]
    P, tlp, ll = _smc_init(est, N, lb, ub, tempered, kind, seed, initial)
    mdd = 0.0
    scl = zeros(steps); ess = zeros(steps); acpt_hist = zeros(steps)
    weights = fill(1.0 / N, N)
    scale, acpt, target = Float64(opts["scale"]), Float64(opts["acpt"]), Float64(opts["target"])
    stages = Dict{Symbol,Any}[Dict(:particles => copy(P), :log_posterior => copy(tlp), :log_likelihood => copy(ll))]
    resampled = false
    for i in 2:steps
        weights = weights .* exp.((phi[i] - phi[i-1]) .* ll)
        sw = sum(weights)
        weights = weights ./ sw
        mdd += log(sw)
        ess[i] = 1.0 / sum(abs2, weights)
        if 2 * ess[i] < N
            resampled = true
            k = kitagawa(weights, rand(r))
            P = P[k, :]; ll = ll[k]; tlp = tlp[k]
            weights = fill(1.0 / N, N)
        end
        scale *= 0.95 + 0.1 / (1 + exp(-16 * (acpt - target)))
        scl[i] = scale
        mu = P' * weights
        Z = P' .- mu
        R = Z * (Z' .* weights)
        R = 0.5 .* (R .+ R')
        RR = Matrix(cholesky(Symmetric(R)).L)
        tlp = tlp .+ (phi[i] - phi[i-1]) .* ll
        accepted = falses(N)
        for j in 1:N
            cand = P[j, :] .+ scale .* (RR * _nrand(r, npar))
            if all(cand .>= lb) && all(cand .<= ub)
                tl, lk = tempered(cand, phi[i])
                if isfinite(lk) && rand(r) < _sexp(tl - tlp[j])
                    accepted[j] = true
                    P[j, :] = cand; ll[j] = lk; tlp[j] = tl
                end
            end
        end
        acpt = count(accepted) / N
        acpt_hist[i] = acpt
        verbose && @info @sprintf("%3d  %5.4f  %9.5E  %5.4f  %4.3f  %s", i, phi[i], ess[i], acpt, scale, resampled ? "yes" : "no")
        if i == steps && !resampled
            k = kitagawa(weights, rand(r))
            P = P[k, :]
        end
        st = Dict{Symbol,Any}(:particles => copy(P), :log_posterior => copy(tlp), :log_likelihood => copy(ll))
        resampled || (st[:weights] = copy(weights))
        push!(stages, st)
        resampled = false
    end
    SMCResult(HSSMC, P, tlp, ll, mdd, est.names, collect(phi), ess, acpt_hist, scl, weights, stages)
end

function _dsmh(est::Estimation, opts, seed, kind, initial, verbose)
    H, M, N, G, K, tau = (Int(opts[k]) for k in ("H", "M", "N", "G", "K", "tau"))
    a0, a1 = Float64(opts["alpha0"]), Float64(opts["alpha1"])
    npar = est.n
    lb, ub, tempered, r = _smc_setup(est, kind, seed, N * G)
    T = est.nobs
    lambda1 = 1 / (10 * npar * T)
    lam = [exp((H - i) / (H - 1) * log(lambda1)) for i in 1:H]
    c = 1.0
    param, tlp_prev, ll = _smc_init(est, N * G, lb, ub, tempered, kind, seed, initial)
    ess = zeros(H); acc_hist = zeros(H); c_hist = zeros(H)
    zhat = 0.0
    stages = Dict{Symbol,Any}[]
    for i in 1:H
        order = sortperm(tlp_prev)                       # sortrows (stable)
        tlp_prev, ll, param = tlp_prev[order], ll[order], param[order, :]
        dl = i == 1 ? lam[i] : lam[i] - lam[i-1]
        tlp_i = tlp_prev .+ ll .* dl
        weights = exp.(tlp_i .- tlp_prev)
        zhat += log(sum(weights) / length(weights))
        weights = weights ./ sum(weights)
        ess[i] = 1 / sum(abs2, weights)
        mu = param' * weights
        Z = param' .- mu
        Om = (Z .* weights') * Z'
        Oc = Matrix(cholesky(Symmetric((Om + Om') ./ 2)).L)
        # tune_c
        lower, upper = (0.5 * (a0 + a1))^5, (0.5 * (a0 + a1))^(1 / 5)
        acpt = 0.0
        while true
            acpt = 0.0
            k = kitagawa(weights, rand(r), G)
            p0 = param[k, :]; t0 = tlp_i[k]
            for j in 1:G, _ in 1:K
                cand = p0[j, :] .+ sqrt(c) .* (Oc * _nrand(r, npar))
                if all(cand .>= lb) && all(cand .<= ub)
                    tx, lx = tempered(cand, lam[i])
                    if isfinite(lx) && rand(r) < _sexp(tx - t0[j])
                        acpt += 1 / (G * K)
                        p0[j, :] = cand; t0[j] = tx
                    end
                end
            end
            if a0 <= acpt <= a1
                break
            elseif acpt < lower
                c = c / 5
            elseif lower <= acpt <= upper
                c = c * log(0.5 * (a0 + a1)) / log(acpt)
            else
                c = 5c
            end
        end
        # mutation_DSMH
        idx_levels = [m * N * G ÷ M for m in 1:M-1]
        p = 0.1 / tau
        levels = vcat(-Inf, tlp_prev[idx_levels])
        out_param = copy(param); out_tlp = copy(tlp_i); out_ll = copy(ll)
        k = kitagawa(weights, rand(r), G)
        p0 = param[k, :]; tm0 = tlp_prev[k]; ti0 = tlp_i[k]; l0 = ll[k]
        for l in 1:N*tau
            for j in 1:G
                u1 = rand(r); u2 = rand(r)
                if u1 < p
                    kk = M
                    for m in 1:M-1
                        if tm0[j] < levels[m+1]
                            kk = m
                            break
                        end
                    end
                    indx = Int(floor((kk - 1) * N * G / M + 1 + u2 * (N * G / M - 1)))
                    alp = (ll[indx] - l0[j]) * dl
                    if u2 < _sexp(alp)
                        p0[j, :] = param[indx, :]; ti0[j] = tlp_i[indx]; l0[j] = ll[indx]; tm0[j] = tlp_prev[indx]
                    end
                else
                    cand = p0[j, :] .+ sqrt(c) .* (Oc * _nrand(r, npar))
                    if all(cand .>= lb) && all(cand .<= ub)
                        tx, lx = tempered(cand, lam[i])
                        if isfinite(lx) && u2 < _sexp(tx - ti0[j])
                            p0[j, :] = cand; ti0[j] = tx; l0[j] = lx
                            tm0[j] = tx - lx * dl
                        end
                    end
                end
            end
            if l % tau == 0
                rg = ((l ÷ tau - 1) * G + 1):(l * G ÷ tau)
                out_param[rg, :] = p0; out_tlp[rg] = ti0; out_ll[rg] = l0
            end
        end
        param, tlp_prev, ll = out_param, out_tlp, out_ll
        acc_hist[i], c_hist[i] = acpt, c
        push!(stages, Dict{Symbol,Any}(:particles => copy(param), :log_posterior => copy(tlp_prev), :log_likelihood => copy(ll)))
        verbose && @info @sprintf("%4d  %5.4f  %9.5E  %5.4f  %5.4f  %+5.4f", i, lam[i], ess[i], c, acpt, zhat)
    end
    SMCResult(DSMH, param, tlp_prev, ll, zhat, est.names, lam, ess, acc_hist, c_hist, nothing, stages)
end
