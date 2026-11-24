# Posterior analysis over the draws of a Metropolis-Hastings run: Dynare's sub-sample of draws (`get_posterior_subsample.m`,
# `set_number_of_subdraws.m`), posterior impulse responses (`bayesian_irf`, `PosteriorIRF.m`) and the shared summary of a set of
# series over draws (`posterior_moments.m`, `pm3.m`).
#
#     mc = sample_posterior(est; mode=mode)               # or the chains of a Dynare run
#     ir = posterior_irf(est, mc)                         # sub-draws from the model's stream, as Dynare
#     ir = posterior_irf(est; params=draws)               # Dynare's own sub-draws, replayed
#     ir["y_ea"].Mean, ir.mean, ir.hpd_inf, ...           # (periods x series), series ordered shock by shock
#
# Derived from Dynare 7.2's matlab/estimation/{PosteriorIRF, PosteriorIRF_core1, get_posterior_subsample,
# set_number_of_subdraws, posterior_moments}.m and matlab/stochastic_solver/{irf, getIrfShocksIndx}.m (GPL-3.0-or-later,
# Dynare Team), through pydyn's port (pydyn/posterior.py). Conventions kept on purpose: the impulse of shock i is column i of
# the lower Cholesky factor of Sigma_e (one standard deviation; the contemporaneous effect on later correlated shocks is
# included); with `relative_irf` the responses are multiplied by 100 and divided by that standard deviation; shocks with a
# standard deviation of at most 5e-7 and responses that move by less than 1e-12 over the horizon are left at zero; a draw at
# which the model cannot be solved is an error. Dynare 7.2 stores no kernel density for IRFs (only Mean, Median, Var, deciles,
# HPDinf, HPDsup). Impulse responses are first order only: `order > 1` (generalised IRFs over draws) raises.

"Number of draws kept per chain after the burn-in (`set_number_of_subdraws.m`'s NumberOfDrawsPerChain)."
kept_per_chain(mc::MCMCResult; drop=mc.drop) = size(mc.draws, 2) - floor(Int, drop * size(mc.draws, 2))

"""
    number_of_subdraws(mc, est=nothing; sub_draws=nothing, drop) -> Int

`set_number_of_subdraws.m`: `sub_draws` if given (Dynare compares it with draws x chains x chains and raises when it is larger),
else `min(posterior_max_subsample_draws, kept draws of all chains)`.
"""
function number_of_subdraws(mc::MCMCResult, est=nothing; sub_draws=nothing, drop=mc.drop)
    per_chain = kept_per_chain(mc; drop=drop)
    nb = size(mc.draws, 1)
    total = per_chain * nb
    o = est === nothing ? Dict{String,Any}() : est.options
    sd = sub_draws === nothing ? get(o, "sub_draws", nothing) : sub_draws
    if sd === nothing || (sd isa AbstractVector && isempty(sd))
        return Int(min(get(o, "posterior_max_subsample_draws", 1200), total))
    end
    sd = Int(sd)
    sd > total * nb && throw(EstimationError("The value of option sub_draws ($sd) is greater than the number of available draws in the MCMC ($(total * nb))!"))
    sd
end

"""
    posterior_subsample(mc, size; drop, rng) -> (draws (size x n), log_posterior (size))

`get_posterior_subsample.m`: `size` pairs (chain, draw) picked uniformly with replacement among the kept draws, then ordered by
chain (stable). The picks are `ceil(rand * nblck)` and `ceil(rand * kept)`, two uniforms per sample, as Dynare; `rng` is the
generator (Julia's, so the draws are not those of MATLAB or Octave).
"""
function posterior_subsample(mc::MCMCResult, size::Integer; drop=mc.drop, rng=Random.default_rng())
    nb, T, n = Base.size(mc.draws)
    first = max(0, floor(Int, drop * T))
    kept = T - first
    chain = zeros(Int, size)
    draw = zeros(Int, size)
    for i in 1:size
        chain[i] = clamp(ceil(Int, rand(rng) * nb), 1, nb)
        draw[i] = clamp(ceil(Int, rand(rng) * kept), 1, kept)
    end
    ord = sortperm(chain; alg=MergeSort)
    out = zeros(size, n)
    lp = zeros(size)
    for (k, i) in enumerate(ord)
        out[k, :] = mc.draws[chain[i], first+draw[i], :]
        lp[k] = mc.log_posterior[chain[i], first+draw[i]]
    end
    out, lp
end

"Pooled kept draws and their log posterior, chain after chain (`GetAllPosteriorDraws.m`): (nblck * kept, n), (nblck * kept)."
function pooled_draws(mc::MCMCResult; drop=mc.drop)
    first = max(0, floor(Int, drop * size(mc.draws, 2)))
    d = mc.draws[:, first+1:end, :]
    lp = mc.log_posterior[:, first+1:end]
    reshape(permutedims(d, (2, 1, 3)), :, size(d, 3)), vec(lp)
end

"""
    posterior_draws(est, mc; sub_draws=nothing, drop, rng) -> (draws, log_posterior)

The parameter vectors the posterior statistics run over (`prior_posterior_statistics.m`): with `B` equal to the kept draws of one
chain Dynare reads all pooled draws and uses the first B rows of them (the first chain); otherwise the random sub-sample.
"""
function posterior_draws(est::Estimation, mc::MCMCResult; sub_draws=nothing, drop=mc.drop, rng=dynare_rng(est.model))
    B = number_of_subdraws(mc, est; sub_draws=sub_draws, drop=drop)
    if B == kept_per_chain(mc; drop=drop)
        x, lp = pooled_draws(mc; drop=drop)
        return x[1:B, :], lp[1:B]
    end
    posterior_subsample(mc, B; drop=drop, rng=rng)
end

# ====================================================================== summary of series over draws
"""
    SeriesStats

`posterior_moments.m` of every column of a (draws x series) array: mean, median, variance, the HPD limits (NaN when
`round((1 - mh_conf_sig) * draws) <= 2`), the nine deciles (NaN with fewer than 10 draws) and optionally the kernel density
(512 x 2 per series; NaN for a variance below 1e-12).
"""
struct SeriesStats
    mean::Vector{Float64}
    median::Vector{Float64}
    var::Vector{Float64}
    hpd_inf::Vector{Float64}
    hpd_sup::Vector{Float64}
    deciles::Matrix{Float64}               # (9, series)
    density::Union{Nothing,Array{Float64,3}}   # (grid, 2, series)
end

"""
    series_stats(x::AbstractMatrix, mh_conf_sig; density=false, gridpoints=512) -> SeriesStats

`x` is (draws x series).
"""
function series_stats(x::AbstractMatrix, mh_conf_sig::Real; density::Bool=false, gridpoints::Int=512)
    ns = size(x, 2)
    mu, med, v, hi, hs = (fill(NaN, ns) for _ in 1:5)
    dec = fill(NaN, 9, ns)
    dens = density ? fill(NaN, gridpoints, 2, ns) : nothing
    for j in 1:ns
        a, b, c, hpd, d = posterior_moments(view(x, :, j), mh_conf_sig)
        mu[j], med[j], v[j], hi[j], hs[j] = a, b, c, hpd[1], hpd[2]
        dec[:, j] = d
        density && (dens[:, :, j] = _posterior_density(view(x, :, j); ngrid=gridpoints))
    end
    SeriesStats(mu, med, v, hi, hs, dec, dens)
end

"Dynare's `options_.estimation.moments_posterior_density`: (indicator, gridpoints); other kernels and bandwidths are refused."
function _density_options(est::Estimation)
    o = est.options
    bw = get(o, "estimation.moments_posterior_density.bandwidth", 0)
    kf = get(o, "estimation.moments_posterior_density.kernel_function", "gaussian")
    (bw == 0 || bw === nothing) && (kf == "gaussian" || kf === nothing) ||
        throw(EstimationError("moments_posterior_density with bandwidth=$bw, kernel_function=$kf is not implemented (only the rule-of-thumb bandwidth with the gaussian kernel)"))
    gp = Int(get(o, "estimation.moments_posterior_density.gridpoints", 512))
    gp == 512 || throw(EstimationError("moments_posterior_density.gridpoints=$gp is not implemented (512)"))
    ind = get(o, "estimation.moments_posterior_density.indicator", true)
    _est_flag(ind), gp
end

# ====================================================================== posterior IRFs
"Shocks for which IRFs are computed (`getIrfShocksIndx.m`): all, or the unique indices of `irf_shocks` in declaration order."
function irf_shock_indices(m::Model, names)
    (names === nothing || isempty(names)) && return collect(1:length(m.exo_names))
    names = names isa AbstractString ? [names] : names
    idx = Int[]
    for s in names
        k = findfirst(==(String(s)), m.exo_names)
        k === nothing && throw(EstimationError("irf_shocks: $s is not an exogenous variable"))
        push!(idx, k)
    end
    sort!(unique!(idx))
end

_irf_shock_names(v) = v isa AbstractDict ? String.(get(v, "symbol_list", String[])) : v === nothing ? nothing : String.(collect(v))

"""
    irf_at(est, theta; periods=40, relative_irf=false, shocks=nothing, variables=nothing) -> Array (periods, variables, shocks)

First-order impulse responses at `theta` as `PosteriorIRF_core1.m` computes them (shocks in declaration order; `shocks` are the
indices to compute, the others stay zero; `variables` names, default all). Throws `SteadyStateError` / `BlanchardKahnError`
when the model cannot be solved.
"""
function irf_at(est::Estimation, theta::AbstractVector; periods::Integer=40, relative_irf::Bool=false, shocks=nothing, variables=nothing)
    m = est.model
    Q, _ = covariances(est, theta)
    _, G, H, _ = solve(est, theta)
    SS = get_lower_cholesky_covariance(Q)
    idx = variables === nothing ? collect(1:length(m.endo_names)) : Int[m.ie[v] for v in variables]
    shocks = shocks === nothing ? collect(1:length(m.exo_names)) : shocks
    out = zeros(periods, length(idx), length(m.exo_names))
    y = zeros(periods, size(G, 1))
    for i in shocks
        SS[i, i] > 5e-7 || continue
        cur = H * SS[:, i]
        for t in 1:periods
            y[t, :] = cur
            cur = G * cur
        end
        relative_irf && (y .= 100 .* y ./ SS[i, i])
        for (jj, j) in enumerate(idx)
            c = view(y, :, j)
            maximum(c) - minimum(c) > 1e-12 && (out[:, jj, i] = c)
        end
    end
    out
end

"""
    PosteriorIRF

Posterior distribution of impulse responses (`oo_.PosteriorIRF.dsge`). `names` are `variable_shock`, shock by shock then variable;
the arrays have one column per name: `mean`, `median`, `var`, `hpd_inf`, `hpd_sup` (periods x series), `deciles` (periods x 9 x series).
`ir[name]` gives a NamedTuple with Dynare's field names.
"""
struct PosteriorIRF
    names::Vector{String}
    variables::Vector{String}
    shocks::Vector{String}
    mean::Matrix{Float64}
    median::Matrix{Float64}
    var::Matrix{Float64}
    hpd_inf::Matrix{Float64}
    hpd_sup::Matrix{Float64}
    deciles::Array{Float64,3}
    mh_conf_sig::Float64
    params::Matrix{Float64}
    draws::Union{Nothing,Array{Float64,4}}        # (draws, periods, variables, shocks in declaration order)
end

function Base.getindex(r::PosteriorIRF, name::AbstractString)
    k = findfirst(==(name), r.names)
    k === nothing && throw(KeyError(name))
    (Mean=r.mean[:, k], Median=r.median[:, k], Var=r.var[:, k], HPDinf=r.hpd_inf[:, k], HPDsup=r.hpd_sup[:, k], deciles=r.deciles[:, :, k])
end

"""
    posterior_irf(est, mc=nothing; params=nothing, periods, relative_irf, irf_shocks, variables, sub_draws, mh_conf_sig, drop, rng, keep_draws=false)

Posterior mean, median, variance, deciles and HPD bands of first-order impulse responses (`bayesian_irf`, `PosteriorIRF.m`). The model is
solved at `sub_draws` parameter vectors drawn from `mc` (`posterior_subsample`; the model's stream by default) or at the rows of
`params` (Dynare's own draws replayed), and `irf_at` evaluated at each. Defaults come from the estimation command: `irf` (40),
`relative_irf`, `irf_shocks`, `mh_conf_sig`, `sub_draws`; `variables` default to the list after the command, else the observed variables.
"""
function posterior_irf(est::Estimation, mc::Union{Nothing,MCMCResult}=nothing; params=nothing, periods=nothing, relative_irf=nothing,
                       irf_shocks=nothing, variables=nothing, sub_draws=nothing, mh_conf_sig=nothing, drop=nothing, rng=nothing,
                       keep_draws::Bool=false, var_list=nothing)
    o = est.options
    m = est.model
    haskey(o, "nl") && throw(EstimationError("bayesian_irf: posterior IRFs are first order; order > 1 (generalised IRFs over posterior draws) is not implemented"))
    Int(get(o, "order", 1)) == 1 || throw(EstimationError("bayesian_irf with order > 1 is not implemented"))
    p = periods === nothing ? get(o, "irf", 0) : periods
    periods = (p === nothing || p == 0 || (p isa AbstractVector && isempty(p))) ? 40 : Int(p)
    relative_irf = relative_irf === nothing ? _est_flag(get(o, "relative_irf", false)) : relative_irf
    mh_conf_sig = mh_conf_sig === nothing ? Float64(o["mh_conf_sig"]) : Float64(mh_conf_sig)
    vl = variables !== nothing ? String.(variables) : var_list !== nothing ? String.(var_list) : String[]
    isempty(vl) && (vl = String.(_estimation_var_list_est(est)))
    isempty(vl) && (vl = copy(est.varobs))
    for v in vl
        haskey(m.ie, v) || throw(EstimationError("PosteriorIRF: $v is not a declared endogenous variable"))
    end
    shocks = irf_shock_indices(m, irf_shocks === nothing ? _irf_shock_names(get(o, "irf_shocks", nothing)) : irf_shocks)
    if params === nothing
        mc === nothing && throw(ArgumentError("posterior_irf needs the chains (mc) or parameter draws (params=...)"))
        drop = drop === nothing ? mc.drop : drop
        B = number_of_subdraws(mc, est; sub_draws=sub_draws, drop=drop)
        params, _ = posterior_subsample(mc, B; drop=drop, rng=rng === nothing ? dynare_rng(m) : rng)
    end
    params = Matrix{Float64}(params isa AbstractVector ? reshape(params, 1, :) : params)
    B = size(params, 1)
    _iround((1 - mh_conf_sig) * B) < 2 &&
        @info "PosteriorIRF:: options_.mh_conf_sig times options_.sub_draws is too small to generate HPDIs. I am omitting them."
    nexo = length(m.exo_names)
    stock = zeros(B, periods, length(vl), nexo)
    for k in 1:B
        stock[k, :, :, :] = try
            irf_at(est, view(params, k, :); periods=periods, relative_irf=relative_irf, shocks=shocks, variables=vl)
        catch e
            e isa Union{SteadyStateError,BlanchardKahnError,LoglinearError} || rethrow()
            throw(EstimationError("PosteriorIRF :: Dynare is unable to solve the model ($(sprint(showerror, e))) with sample posterior"))
        end
    end
    names = String[]
    cols = Tuple{Int,Int}[]
    for i in shocks, (j, v) in enumerate(vl)
        push!(names, "$(v)_$(m.exo_names[i])")
        push!(cols, (j, i))
    end
    ns = length(cols)
    M, Md, V, HI, HS = (zeros(periods, ns) for _ in 1:5)
    D = zeros(periods, 9, ns)
    for (s, (j, i)) in enumerate(cols), t in 1:periods
        a, b, c, hpd, d = posterior_moments(view(stock, :, t, j, i), mh_conf_sig)
        M[t, s], Md[t, s], V[t, s], HI[t, s], HS[t, s] = a, b, c, hpd[1], hpd[2]
        D[t, :, s] = d
    end
    PosteriorIRF(names, vl, [m.exo_names[i] for i in shocks], M, Md, V, HI, HS, D, mh_conf_sig, params, keep_draws ? stock : nothing)
end

"The variable list after the estimation command of the statement `est` was built from (empty if none)."
function _estimation_var_list_est(est::Estimation)
    try
        _estimation_var_list(est.model, (;))
    catch
        String[]
    end
end

"`oo_.PosteriorIRF.dsge.{Mean, Median, Var, deciles, HPDinf, HPDsup}.<variable>_<shock>` (columns; deciles periods x 9)."
function store_posterior_irf!(oo::MStruct, r::PosteriorIRF)
    P = get(oo, "PosteriorIRF", nothing)
    P isa MStruct || (P = MStruct(); oo.PosteriorIRF = P)
    D = MStruct()
    fields = (("Mean", r.mean), ("Median", r.median), ("Var", r.var), ("deciles", nothing), ("HPDinf", r.hpd_inf), ("HPDsup", r.hpd_sup))
    for (f, a) in fields
        S = MStruct()
        for (k, nm) in enumerate(r.names)
            S[nm] = a === nothing ? Matrix{Float64}(r.deciles[:, :, k]) : _col(a[:, k])
        end
        D[f] = S
    end
    P.dsge = D
    oo
end
