# OccBin forecasts: Dynare 7.2's `occbin.forecast` (`+occbin/forecast.m`). Port of pydyn's occbin_forecast.py.
#
# The forecast starts from `M_.endo_histval(:, 1)` (levels, declaration order), uses the base-case shocks
# `options_.occbin.forecast.SHOCKS0` (zero by default) and, with `replic > 0`, adds Gaussian draws with the covariance
# matrix of the shocks (plain `randn`, or a Sobol sequence with `qmc`). The statistics of the simulated paths (mean, median,
# variance, HPD interval, deciles, min, max) are Dynare's `posterior_moments`. The random numbers come from `rng`
# (an `OctaveRNG` reproduces Octave's `randn`; `dynare()` passes the workspace's stream, which is Octave's with `random=:octave`).

const OCCBIN_FORECAST_STATISTICS = ("Mean", "Median", "Var", "HPDinf", "HPDsup", "Deciles", "Min", "Max")

"""
    OccbinForecast

Dynare's `forecast` and `yf`. `linear[stat][variable]` and `piecewise[stat][variable]` are the statistics (`"Mean"`, and with
`replic > 0` also `"Median"`, `"Var"`, `"HPDinf"`, `"HPDsup"`, `"Deciles"` (horizon x 9), `"Min"`, `"Max"`) over the horizon.
`regimes` is `regime_history` (`replic = 0`) or one per replication (`nothing` for the ones that failed); `error_flag` the solver's
flag (`replic = 0`) or the number of failed replications; `failed` the per-replication flags (Dynare's logical vector `error_flag`);
`yf` is `(horizon + 1) x n_endo` in levels (the initial point, then the piecewise forecast); `paths` (`replic > 0`) the
`horizon x n_endo x n_successful` arrays of both simulations, `shocks` the shocks of the successful replications.
"""
mutable struct OccbinForecast
    linear::Dict{String,Dict{String,Matrix{Float64}}}
    piecewise::Dict{String,Dict{String,Matrix{Float64}}}
    regimes::Any
    error_flag::Int
    failed::Vector{Bool}
    yf::Union{Nothing,Matrix{Float64}}
    replic::Int
    successful::Vector{Int}
    paths::Union{Nothing,NamedTuple{(:linear, :piecewise),Tuple{Array{Float64,3},Array{Float64,3}}}}
    shocks::Union{Nothing,Array{Float64,3}}
end
OccbinForecast() = OccbinForecast(Dict(), Dict(), nothing, 0, Bool[], nothing, 0, Int[], nothing, nothing)
Base.getproperty(f::OccbinForecast, s::Symbol) = s === :ok ? getfield(f, :yf) !== nothing : getfield(f, s)

"`shocks` (`options_.occbin.forecast.SHOCKS0`) as a `horizon x n_exo` matrix: a `(n_shocks x horizon)` matrix whose rows are the first shocks (declaration order), or a vector of `(name, values)`."
function _forecast_base_shocks(ob::Occbin, shocks, horizon::Int)
    m = ob.model
    base = zeros(horizon, ob.n_exo)
    (shocks === nothing || (shocks isa Union{AbstractArray,Tuple,AbstractDict} && isempty(shocks))) && return base
    if shocks isa AbstractDict
        shocks = collect(pairs(shocks))
    end
    if shocks isa Union{AbstractVector,Tuple} && !isempty(shocks) && first(shocks) isa Union{Tuple,Pair} && first(first(shocks)) isa AbstractString
        periods = length(last(first(shocks)))
        for s in shocks
            name, values = first(s), last(s)
            haskey(m.ix, name) || throw(ArgumentError("occbin.forecast: unknown exogenous shock $name"))
            length(values) == periods || throw(ArgumentError("occbin.forecast: unknown exogenous shock $name"))
            periods == horizon || throw(ArgumentError("occbin.forecast: the shock paths have $periods periods, the forecast horizon is $horizon (Dynare's assignment fails)"))
            base[:, m.ix[name]] = Float64.(values)
        end
        return base
    end
    mat = shocks isa AbstractVector ? reshape(Float64.(shocks), 1, :) : Float64.(shocks)
    size(mat, 1) > ob.n_exo && throw(ArgumentError("occbin.forecast: SHOCKS0 has more rows than the model has shocks"))
    size(mat, 2) == horizon || throw(ArgumentError("occbin.forecast: SHOCKS0 has $(size(mat, 2)) columns, the forecast horizon is $horizon (Dynare's assignment fails)"))
    base[:, 1:size(mat, 1)] = permutedims(mat)
    base
end

_round_half_away(x) = x >= 0 ? Int(floor(x + 0.5)) : -Int(floor(-x + 0.5))

"""
    occbin_forecast(ob::Occbin, horizon; endo_histval=nothing, shocks=nothing, frcst_regimes=nothing, replic=0, qmc=false,
                    maxit=30, check_ahead_periods=30, conf_sig=0.9, sigma_e=nothing, rng=nothing, simul_options=nothing) -> OccbinForecast

`[forecast, error_flag, yf] = occbin.forecast(options_, M_, dr, ys, ..., forecast_horizon)`. `endo_histval`: the initial state in
levels, declaration order (default: the steady state); `shocks`: `options_.occbin.forecast.SHOCKS0`; `frcst_regimes` the initial regime
guess (`init_regime` format of `simulate`); `conf_sig` is `options_.forecasts.conf_sig`; `sigma_e` defaults to the model's covariance
matrix of the shocks; `simul_options` are the other `options_.occbin.simul` entries as `simulate` takes them (`mod_options(ob)` by
default). A failed forecast (`replic = 0`) is returned with `ok == false` and the solver's `error_flag`.
"""
function occbin_forecast(ob::Occbin, horizon::Integer; endo_histval=nothing, shocks=nothing, frcst_regimes=nothing, replic::Integer=0,
                         qmc::Bool=false, maxit::Integer=30, check_ahead_periods::Integer=30, conf_sig::Real=0.9, sigma_e=nothing,
                         rng=nothing, simul_options=nothing)
    horizon = Int(horizon)
    m = ob.model
    ys = ob.ys
    y0 = endo_histval === nothing ? copy(ys) : Float64.(vec(endo_histval))
    length(y0) == ob.n || throw(ArgumentError("occbin.forecast: the initial state has $(length(y0)) values, the model has $(ob.n) endogenous variables"))
    sim = Dict{Symbol,Any}(pairs(simul_options === nothing ? mod_options(ob) : simul_options))
    sim[:maxit] = Int(maxit); sim[:check_ahead_periods] = Int(check_ahead_periods); sim[:periods] = horizon
    init = y0 .- ys
    base = _forecast_base_shocks(ob, shocks, horizon)
    init_regime = frcst_regimes === nothing || isempty(frcst_regimes) ? nothing : frcst_regimes
    names = m.endo_names
    res = OccbinForecast()
    if replic > 0
        S = Matrix{Float64}(sigma_e === nothing ? m.Sigma_e : sigma_e)
        ishock = [i for i in 1:ob.n_exo if sqrt(S[i, i]) != 0]
        neff = length(ishock)
        U, s, _ = svd(S[ishock, ishock]; alg=LinearAlgebra.QRIteration())           # Octave's svd uses LAPACK's gesvd
        nrep = Int(replic)
        if qmc
            nrep = 2^_round_half_away(log2(nrep + 1)) - 1
            add, _ = qmc_sequence(horizon * neff, 1, 1, nrep)
        else
            rng === nothing && throw(ArgumentError("occbin.forecast: a random number generator (rng) is needed for replic > 0"))
            add = Matrix{Float64}(undef, horizon * neff, nrep)
            randn!(rng, add)
        end
        add = reshape(add, neff, horizon, nrep)
        lin = fill(NaN, horizon, ob.n, nrep)
        pw = fill(NaN, horizon, ob.n, nrep)
        failed = trues(nrep)
        regs = Vector{Any}(nothing, nrep)
        sh_all = fill(NaN, horizon, neff, nrep)
        L = U * Diagonal(sqrt.(s))
        last_ok = false
        for it in 1:nrep
            sh = base[:, ishock] .+ permutedims(L * add[:, :, it])
            local out
            try
                out = simulate(ob, sh; init=init, exo_pos=ishock, init_regime=init_regime, sim...)
            catch e
                e isa OccbinError || rethrow()
                last_ok = false
                continue
            end
            last_ok = true
            failed[it] = false
            lin[:, :, it] = out.linear; pw[:, :, it] = out.piecewise; regs[it] = out.regime_history
            sh_all[:, :, it] = sh
        end
        ok = findall(.!failed)
        res.replic = nrep; res.successful = ok; res.failed = collect(failed); res.error_flag = count(failed)
        lin, pw = lin[:, :, ok], pw[:, :, ok]
        res.paths = (linear=lin, piecewise=pw)
        res.shocks = sh_all[:, :, ok]
        res.regimes = regs
        isempty(ok) && throw(OccbinForecastError("occbin.forecast: no replication succeeded"))
        for (fld, z) in ((:linear, lin), (:piecewise, pw))
            d = Dict{String,Dict{String,Matrix{Float64}}}(k => Dict{String,Matrix{Float64}}() for k in OCCBIN_FORECAST_STATISTICS)
            for (i, nm) in enumerate(names)
                mean_ = zeros(horizon, 1); med = zeros(horizon, 1); var_ = zeros(horizon, 1)
                hl = zeros(horizon, 1); hu = zeros(horizon, 1); dec = zeros(horizon, 9)
                for j in 1:horizon
                    a, b, c, hpd, dd = posterior_moments(z[j, i, :], conf_sig)
                    mean_[j] = a; med[j] = b; var_[j] = c; hl[j] = hpd[1]; hu[j] = hpd[2]; dec[j, :] = dd
                end
                d["Mean"][nm] = mean_; d["Median"][nm] = med; d["Var"][nm] = var_
                d["HPDinf"][nm] = hl; d["HPDsup"][nm] = hu; d["Deciles"][nm] = dec
                d["Min"][nm] = reshape(minimum(z[:, i, :]; dims=2), horizon, 1)
                d["Max"][nm] = reshape(maximum(z[:, i, :]; dims=2), horizon, 1)
            end
            setfield!(res, fld, d)
        end
        # Dynare's `out` is the last simulation run: its piecewise path gives yf, whether or not it converged (a failed one has none here)
        res.yf = vcat(reshape(y0, 1, :), last_ok ? pw[:, :, end] : fill(NaN, horizon, ob.n))
        return res
    end
    out = try
        simulate(ob, base; init=init, init_regime=init_regime, sim...)
    catch e
        e isa OccbinError || rethrow()
        res.error_flag = error_flag(e)
        return res
    end
    res.error_flag = 0
    res.regimes = out.regime_history
    res.linear = Dict("Mean" => Dict(nm => reshape(out.linear[:, i], :, 1) for (i, nm) in enumerate(names)))
    res.piecewise = Dict("Mean" => Dict(nm => reshape(out.piecewise[:, i], :, 1) for (i, nm) in enumerate(names)))
    res.yf = vcat(reshape(y0, 1, :), out.piecewise)
    res
end

struct OccbinForecastError <: Exception
    msg::String
end
Base.showerror(io::IO, e::OccbinForecastError) = print(io, "OccbinForecastError: ", e.msg)

"""
    occbin_forecast(est::Estimation, theta, horizon; endo_histval=nothing, kw...)

`occbin.forecast` at the parameters `theta` of an OccBin estimation: the covariance matrix of the shocks is the one at `theta`;
`endo_histval` is the initial state in levels (e.g. the smoothed state of the last period plus the steady state). The other
keywords are those of `occbin_forecast(ob, horizon; ...)`.
"""
function occbin_forecast(est::Estimation, theta::AbstractVector, horizon::Integer; sigma_e=nothing, kw...)
    ob = occbin_at(est, theta)
    occbin_forecast(ob, horizon; sigma_e=sigma_e === nothing ? covariances(est, theta)[1] : sigma_e, kw...)
end
