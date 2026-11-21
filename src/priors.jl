# Prior distributions with Dynare's parameterisations (port of pydyn/priors.py, which follows
# Dynare 7.2's matlab/estimation/{set_prior, priordens, prior_bounds}.m and matlab/distributions/*.m).
#
# Shape codes are those of the preprocessor and `bayestopt_.pshape`:
# 1 beta, 2 gamma, 3 normal, 4 inverse gamma type 1, 5 uniform, 6 inverse gamma type 2, 8 Weibull;
# 0 means "no prior" (maximum likelihood).

struct PriorError <: Exception
    msg::String
end
Base.showerror(io::IO, e::PriorError) = print(io, "PriorError: ", e.msg)

const SHAPE_NAMES = Dict(0 => "none", 1 => "beta", 2 => "gamma", 3 => "normal", 4 => "inv_gamma1",
                         5 => "uniform", 6 => "inv_gamma2", 8 => "weibull")
const SHAPE_CODES = Dict("none" => 0, "beta" => 1, "gamma" => 2, "normal" => 3, "inv_gamma1" => 4, "uniform" => 5,
    "inv_gamma2" => 6, "weibull" => 8, "inv_gamma" => 4, "invgamma" => 4, "invgamma1" => 4, "invgamma2" => 6,
    "beta_pdf" => 1, "gamma_pdf" => 2, "normal_pdf" => 3, "inv_gamma_pdf" => 4, "inv_gamma1_pdf" => 4,
    "uniform_pdf" => 5, "inv_gamma2_pdf" => 6, "weibull_pdf" => 8)

# ---------------------------------------------------------------------- (mean, variance) -> hyper-parameters
"Shape parameters (a, b) of a beta distribution on [lb, ub] (beta_specification.m)."
function beta_specification(mu, sigma2, lb=0.0, ub=1.0)
    (mu < lb || mu > ub) && throw(PriorError("beta prior: mean $mu outside [$lb, $ub]"))
    len = ub - lb
    mu = (mu - lb) / len
    sigma2 = sigma2 / (len * len)
    sigma2 > (1 - mu) * mu && throw(PriorError("beta prior: the std has to be smaller than $(sqrt((1 - mu) * mu) * len)"))
    a = (1 - mu) * mu * mu / sigma2 - mu
    b = a * (1 / mu - 1)
    a, b
end

"Shape a and scale b of a gamma distribution shifted by lb (gamma_specification.m)."
function gamma_specification(mu, sigma2, lb=0.0)
    mu < lb && throw(PriorError("gamma prior: mean $mu below the lower bound $lb"))
    isinf(sigma2) && throw(PriorError("gamma prior: the variance has to be finite"))
    mu = mu - lb
    b = sigma2 / mu
    mu / b, b
end

_ig1fun(nu, mu2, sigma2) = log(2 * mu2) - log((sigma2 + mu2) * (nu - 2)) +
                           2 * (SpecialFunctions.loggamma(nu / 2) - SpecialFunctions.loggamma((nu - 1) / 2))

"(s, nu) of an inverse gamma distribution of type 1 or 2, shifted by lb (inverse_gamma_specification.m)."
function inverse_gamma_specification(mu, sigma2, lb=0.0, type=1)
    sigma2 <= 0 && throw(PriorError("inverse gamma prior: the variance must be positive"))
    mu <= lb && throw(PriorError("inverse gamma prior: mean $mu must be above the lower bound $lb"))
    sigma = sqrt(sigma2)
    mu2 = (mu - lb) * (mu - lb)
    if type == 2
        nu = 2 * (2 + mu2 / sigma2)
        s = 2 * mu * (1 + mu2 / sigma2)          # sic: Dynare uses mu here, not mu-lb
        return s, nu
    end
    type == 1 || throw(PriorError("inverse gamma prior: type must be 1 or 2"))
    isfinite(sigma2) || return 2 * mu2 / pi, 2.0
    nu = sqrt(2 * (2 + mu2 / sigma2))
    nu2, nu1 = 2 * nu, 2.0
    err, err2 = _ig1fun(nu, mu2, sigma2), _ig1fun(nu2, mu2, sigma2)
    if err2 > 0
        while nu2 < 1e12
            nu1, nu2 = nu2, nu2 * 2
            err2 = _ig1fun(nu2, mu2, sigma2)
            err2 < 0 && break
        end
        err2 > 0 && throw(PriorError("inverse gamma prior: no interval with a sign change; the prior variance is probably too small compared with the prior mean"))
    end
    while abs(nu2 / nu1 - 1) > 1e-14
        if err > 0
            nu1 = nu
            if nu < nu2
                nu = nu2
            else
                nu = 2 * nu
                nu2 = nu
            end
        else
            nu2 = nu
        end
        nu = (nu1 + nu2) / 2
        err = _ig1fun(nu, mu2, sigma2)
    end
    s = (sigma2 + mu2) * (nu - 2)
    if abs(log(mu - lb) - log(sqrt(s / 2)) - SpecialFunctions.loggamma((nu - 1) / 2) + SpecialFunctions.loggamma(nu / 2)) > 1e-7 ||
       abs(sigma - sqrt(s / (nu - 2) - (mu - lb) * (mu - lb))) > 1e-7
        throw(PriorError("inverse gamma prior: failed in solving for the hyper-parameters"))
    end
    s, nu
end

"(shape, scale) of a Weibull distribution shifted by lb (weibull_specification.m)."
function weibull_specification(mu, sigma2, lb=0.0)
    mu < lb && throw(PriorError("Weibull prior: mean $mu below the lower bound $lb"))
    isinf(sigma2) && throw(PriorError("Weibull prior: the variance has to be finite"))
    mu = mu - lb
    target = log1p(sigma2 / (mu * mu))
    lg = SpecialFunctions.loggamma
    eqn(k) = lg(1 + 2 / k) - 2 * lg(1 + 1 / k) - target       # decreasing in k
    lo, hi = 1e-2, 1e2
    while eqn(hi) > 0 && hi < 1e8
        hi *= 10
    end
    while eqn(lo) < 0 && lo > 1e-8
        lo /= 10
    end
    (eqn(lo) >= 0 && eqn(hi) <= 0) || throw(PriorError("Weibull prior: no solution for the shape parameter"))
    for _ in 1:300                                       # bisection to rounding error
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        eqn(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    shape = (lo + hi) / 2
    shape, mu / exp(lg(1 + 1 / shape))
end

"(mean, std, lower, upper) of a uniform prior (uniform_specification.m)."
function uniform_specification(m, s, p3, p4)
    if !(isnan(p3) || isnan(p4))
        (isnan(m) && isnan(s)) || throw(PriorError("uniform prior: with lower and upper bounds given, leave mean and std empty"))
        return (p3 + p4) / 2, (p4 - p3) / sqrt(12), p3, p4
    end
    (isfinite(m) && isfinite(s)) || throw(PriorError("uniform prior: mean or standard deviation is not finite"))
    m, s, m - s * sqrt(3), m + s * sqrt(3)
end

# ---------------------------------------------------------------------- log densities (lpdf*.m)
lpdf_beta(x, a, b, aa=0.0, bb=1.0) = !(x - aa > 0 && x - bb < 0) ? -Inf :
    -SpecialFunctions.logbeta(a, b) + (a - 1) * log(x - aa) + (b - 1) * log(bb - x) - (a + b - 1) * log(bb - aa)
lpdf_gamma(x, a, b) = !(x > 0) ? -Inf : -SpecialFunctions.loggamma(a) - a * log(b) + (a - 1) * log(x) - x / b
lpdf_normal(x, a, b) = -log(b) - 0.5 * log(2 * pi) - 0.5 * ((x - a) / b) * ((x - a) / b)
lpdf_ig1(x, s, nu) = !(x > 0) ? -Inf : log(2) - SpecialFunctions.loggamma(0.5 * nu) - 0.5 * nu * (log(2) - log(s)) -
                                       (nu + 1) * log(x) - 0.5 * s / (x * x)
lpdf_ig2(x, s, nu) = !(x > 0) ? -Inf : -SpecialFunctions.loggamma(0.5 * nu) - (0.5 * nu) * (log(2) - log(s)) -
                                       0.5 * (nu + 2) * log(x) - 0.5 * s / x

"Weibull log density (shape a, scale b, location c); reproduces lpdfgweibull.m, including its value 1.0 at x == c when the shape is one."
function lpdf_weibull(x, a, b, c=0.0)
    !(x - c >= 0) && return -Inf
    (abs(a - 1) < 1e-12 && x - c < 1e-12) && return 1.0
    x0 = x - c
    log(a) - a * log(b) + (a - 1) * log(x0) - (x0 / b)^a
end

# ---------------------------------------------------------------------- a set of independent priors
"""
    Prior(shape, p1, p2, p3=nothing, p4=nothing; names=nothing)

Independent priors for a parameter vector, as Dynare's `bayestopt_`: `shape` integer codes (or names),
`p1`/`p2` prior mean and standard deviation, `p3`/`p4` bounds of the generalised distributions (NaN for the
defaults). After construction `p6`/`p7` hold the hyper-parameters as `set_prior.m` computes them: beta (a, b),
gamma (shape, scale), normal (mean, std), inverse gamma 1 and 2 (s, nu), uniform (lower, upper), Weibull (shape, scale).
"""
struct Prior
    shape::Vector{Int}
    p1::Vector{Float64}
    p2::Vector{Float64}
    p3::Vector{Float64}
    p4::Vector{Float64}
    p6::Vector{Float64}
    p7::Vector{Float64}
    names::Vector{String}
end

function Prior(shape, p1, p2, p3=nothing, p4=nothing; names=nothing)
    n = length(shape)
    sh = Int[s isa AbstractString ? SHAPE_CODES[s] : Int(s) for s in shape]
    p1 = Float64.(collect(p1)); p2 = Float64.(collect(p2))
    p3 = p3 === nothing ? fill(NaN, n) : Float64.(collect(p3))
    p4 = p4 === nothing ? fill(NaN, n) : Float64.(collect(p4))
    p6 = fill(NaN, n); p7 = fill(NaN, n)
    nm = names === nothing ? ["p$i" for i in 1:n] : String.(collect(names))
    any((sh .!= 0) .& (p2 .== 0)) && throw(PriorError("a prior standard deviation of zero is not allowed"))
    for i in 1:n
        try
            _specify!(sh[i], p1, p2, p3, p4, p6, p7, i)
        catch e
            e isa PriorError && throw(PriorError("$(nm[i]): $(e.msg)"))
            rethrow()
        end
    end
    Prior(sh, p1, p2, p3, p4, p6, p7, nm)
end

function _specify!(sh, p1, p2, p3, p4, p6, p7, i)
    sh == 0 && return
    a1, a2, a3, a4 = p1[i], p2[i], p3[i], p4[i]
    if sh == 1
        a3 = isnan(a3) ? 0.0 : a3
        a4 = isnan(a4) ? 1.0 : a4
        p6[i], p7[i] = beta_specification(a1, a2^2, a3, a4)
    elseif sh == 2
        a3 = isnan(a3) ? 0.0 : a3
        a4 = isnan(a4) ? Inf : a4
        p6[i], p7[i] = gamma_specification(a1, a2^2, a3)
    elseif sh == 3
        a3 = isnan(a3) ? -Inf : a3
        a4 = isnan(a4) ? Inf : a4
        p6[i], p7[i] = a1, a2
    elseif sh == 4 || sh == 6
        a3 = isnan(a3) ? 0.0 : a3
        a4 = isnan(a4) ? Inf : a4
        p6[i], p7[i] = inverse_gamma_specification(a1, a2^2, a3, sh == 4 ? 1 : 2)
    elseif sh == 5
        p1[i], p2[i], p6[i], p7[i] = uniform_specification(a1, a2, a3, a4)
        a3, a4 = p6[i], p7[i]
    elseif sh == 8
        a3 = isnan(a3) ? 0.0 : a3
        a4 = isnan(a4) ? Inf : a4
        p6[i], p7[i] = weibull_specification(a1, a2^2, a3)
    else
        throw(PriorError("unknown prior shape $sh"))
    end
    p3[i], p4[i] = a3, a4
end

Base.length(p::Prior) = length(p.shape)
is_bayesian(p::Prior) = any(>(0), p.shape)

"Log density of every element (`-Inf` outside the support)."
function logpdf_each(p::Prior, x::AbstractVector)
    out = zeros(length(p))
    for i in eachindex(p.shape)
        sh, xi, p3, p4, p6, p7 = p.shape[i], x[i], p.p3[i], p.p4[i], p.p6[i], p.p7[i]
        out[i] = sh == 1 ? lpdf_beta(xi, p6, p7, p3, p4) :
                 sh == 2 ? lpdf_gamma(xi - p3, p6, p7) :
                 sh == 3 ? lpdf_normal(xi, p6, p7) :
                 sh == 4 ? lpdf_ig1(xi - p3, p6, p7) :
                 sh == 5 ? ((xi - p3 < 0 || xi - p4 > 0) ? -Inf : log(1.0 / (p4 - p3))) :
                 sh == 6 ? lpdf_ig2(xi - p3, p6, p7) :
                 sh == 8 ? lpdf_weibull(xi, p6, p7, p3) : 0.0
    end
    out
end

"Joint log prior density as `priordens.m` (0 for a model without priors)."
logpdf(p::Prior, x::AbstractVector) = sum(logpdf_each(p, x))

_norminv(q, mu, sigma) = mu - sigma * sqrt(2) * SpecialFunctions.erfcinv(2q)
_gaminv(q, a, scale) = SpecialFunctions.gamma_inc_inv(a, q, 1 - q) * scale
_betainv(q, a, b) = SpecialFunctions.beta_inc_inv(a, b, q)[1]
_wblinv(q, scale, shape) = scale * (-log1p(-q))^(1 / shape)

"""
    prior_bounds(p, trunc=0.0) -> (lb, ub)

Support of each prior (`prior_bounds.m`); with `trunc` > 0 the quantiles `trunc` and `1 - trunc`.
"""
function prior_bounds(p::Prior, trunc::Real=0.0)
    n = length(p)
    lb, ub = fill(-Inf, n), fill(Inf, n)
    q = trunc
    for i in 1:n
        sh, p3, p4, p6, p7 = p.shape[i], p.p3[i], p.p4[i], p.p6[i], p.p7[i]
        if sh == 1
            lb[i], ub[i] = q == 0 ? (p3, p4) : (_betainv(q, p6, p7) * (p4 - p3) + p3, _betainv(1 - q, p6, p7) * (p4 - p3) + p3)
        elseif sh == 2
            lb[i], ub[i] = q == 0 ? (p3, Inf) : (_gaminv(q, p6, p7) + p3, _gaminv(1 - q, p6, p7) + p3)
        elseif sh == 3
            lb[i], ub[i] = q == 0 ? (p3, p4) : (max(_norminv(q, p6, p7), p3), min(_norminv(1 - q, p6, p7), p4))
        elseif sh == 4
            lb[i], ub[i] = q == 0 ? (p3, Inf) : (1 / sqrt(_gaminv(1 - q, p7 / 2, 2 / p6)) + p3, 1 / sqrt(_gaminv(q, p7 / 2, 2 / p6)) + p3)
        elseif sh == 5
            lb[i], ub[i] = q == 0 ? (p6, p7) : (p6 + (p7 - p6) * q, p7 - (p7 - p6) * q)
        elseif sh == 6
            lb[i], ub[i] = q == 0 ? (p3, Inf) : (1 / _gaminv(1 - q, p7 / 2, 2 / p6) + p3, 1 / _gaminv(q, p7 / 2, 2 / p6) + p3)
        elseif sh == 8
            lb[i], ub[i] = q == 0 ? (p3, Inf) : (p3 + _wblinv(q, p7, p6), p3 + _wblinv(1 - q, p7, p6))
        end
    end
    lb, ub
end
