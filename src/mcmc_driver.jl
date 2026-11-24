# The MCMC driver for all of Dynare's chain samplers (random walk, Student random walk, independent, slice, TaRB): starting points,
# proposal covariances, random numbers per chain, continuation of earlier chains (`load_mh_file`), `mh_tune_jscale`.
# See src/samplers.jl for the sampler iterations and the conventions.

"The generator kind for the chains: the keyword, else the model's setting (`set_rng_kind!`, `dynare(...; particle_rng=)`)."
_mcmc_kind(est::Estimation, rng) = rng === nothing ? get(_MCMC_RNG_KIND, est.model, rng_kind(est.model)) : rng
const _MCMC_RNG_KIND = WeakKeyDict{Any,Symbol}()

"""
    set_mcmc_rng!(m::Model, kind)      kind = :julia (default) or :octave

Which generator the MCMC samplers of `m` draw from. `:octave` seeds them as Dynare does (the DynareRandomStreams seed, chain j with
seed + 1 + ... + j), so chains equal Dynare's draw for draw (src/octave_rng.jl). `set_rng_kind!` sets this together with the particle filters'.
"""
function set_mcmc_rng!(m, kind::Symbol)
    kind in RNG_KINDS || throw(ArgumentError("unknown random generator kind $kind (use :julia or :octave)"))
    _MCMC_RNG_KIND[m] = kind
    m
end

"Dynare's chain seeds: chain j of a run seeded with s is seeded with s + 1 + ... + j."
dynare_chain_seeds(seed::Integer, nblocks::Integer) = Int[seed + div(j * (j + 1), 2) for j in 1:nblocks]

function _chain_sources(kind::Symbol, seed::Integer, nblocks::Int; previous_seeds=nothing, offset::Int=0)
    if kind == :octave
        init = OctaveRNG(seed)
        chains = if previous_seeds !== nothing && length(previous_seeds) >= nblocks
            AbstractRNG[rng_from_seeds(previous_seeds[j]) for j in 1:nblocks]
        else
            AbstractRNG[OctaveRNG(s) for s in dynare_chain_seeds(seed + offset, nblocks)]
        end
        return chains, init
    elseif kind == :julia
        chains = AbstractRNG[Random.Xoshiro(seed * 7919 + j + 1_000_003 * offset) for j in 1:nblocks]
        return chains, Random.Xoshiro(seed * 7919 + 999_983 + offset)
    end
    throw(ArgumentError("unknown random generator kind $(repr(kind)) (use :julia or :octave)"))
end

"MCMCResult with sampler information in `info` (see the keys set by `sample_mcmc`)."
function _mc_result(draws, logpo, acc, names, x0, drop, accepted; neval=nothing, info=Dict{Symbol,Any}())
    MCMCResult(draws, logpo, acc, names, x0, drop, accepted, neval, info)
end

"Mean and covariance of the kept draws of an earlier run (`compute_mh_covariance_matrix.m`)."
function _mh_covariance(prev::MCMCResult)
    X = kept_draws(prev)
    N = size(X, 1)
    mu = vec(sum(X; dims=1)) ./ N
    Xc = X .- mu'
    mu, (Xc' * Xc) ./ N
end

"""
    sample_mcmc(est; method, replic, nblocks, mode, cov, jscale, seed, x0, drop, init_scale_factor, jumping_covariance, previous,
                rng, sampler_options, tune_jscale, tune_guess, sampler_bounds, kernel_bounds, rotation, modes) -> MCMCResult

Dynare's chain samplers (`posterior_sampler.m`): `method` (default the `posterior_sampling_method` option) is
`random_walk_metropolis_hastings` (with `proposal_distribution='rand_multivariate_student'` in `sampler_options`, as the
`posterior_sampler_options` of the estimation command), `independent_metropolis_hastings`, `slice` (plain, `rotated`, multimodal with
`mode` or `mode_files`) or `tailored_random_block_metropolis_hastings`. `rng` is `:julia` or `:octave` (default: the model's
setting): with `:octave` the chains are Dynare's draw for draw. `mode` is a ModeResult (or a vector; `cov` gives the proposal
covariance then), `previous` an MCMCResult whose chains are continued (`load_mh_file`; with `:octave` the stored generator states of
`previous.info[:last_seeds]` are used). `sampler_bounds` / `kernel_bounds` replace the bounds proposals are checked against
(default `est.lb/est.ub`) and the bounds of the posterior kernel (the prior support with `prior_trunc`); with `prior_trunc > 0`
Dynare's bounds come from Octave's inverse CDFs, which differ from JDyn's by up to about 1e-8 (relative), so replays of Dynare's chains
may pass Dynare's. `tune_jscale` is the target acceptance rate of `mh_tune_jscale`.
"""
function sample_mcmc(est::Estimation; method=nothing, replic=nothing, nblocks=nothing, mode=nothing, cov=nothing, jscale=nothing,
                     seed=0, x0=nothing, drop=nothing, init_scale_factor=nothing, jumping_covariance=nothing, previous=nothing,
                     rng=nothing, sampler_options=nothing, tune_jscale=nothing, tune_guess=nothing, tune_rng=nothing,
                     sampler_bounds=nothing, kernel_bounds=nothing, rotation=nothing, modes=nothing, mode_compute_ran=false)
    o = est.options
    is_bayesian(est.prior) || throw(EstimationError("the sampler needs priors"))
    method = String(method === nothing ? get(o, "posterior_sampling_method", RWMH) : method)
    method in SMC_METHODS && throw(EstimationError("$method is a sequential Monte Carlo sampler: use sample_smc"))
    method in MCMC_METHODS || (JDyn.sampler_options(method, nothing); throw(EstimationError("unknown posterior_sampling_method $method")))
    spec = sampler_options === nothing ? get(o, "posterior_sampler_options", nothing) : sampler_options
    opts = JDyn.sampler_options(method, spec)
    kind = _mcmc_kind(est, rng)
    replic = Int(replic === nothing ? o["mh_replic"] : replic)
    drop = drop === nothing ? o["mh_drop"] : Float64(drop)
    trunc = o["prior_trunc"]
    mlb, mub = kernel_bounds === nothing ? prior_bounds(est.prior, trunc) : (Float64.(kernel_bounds[1]), Float64.(kernel_bounds[2]))
    slb, sub = sampler_bounds === nothing ? (copy(est.lb), copy(est.ub)) : (Float64.(sampler_bounds[1]), Float64.(sampler_bounds[2]))
    method == SLICE && (any(isinf, slb) || any(isinf, sub)) && throw(EstimationError("The option \"slice\" is inconsistent with prior_trunc=0."))
    logpost(x) = try
        Float64(log_posterior(est, x; bounds=(mlb, mub)))
    catch
        -Inf
    end
    npar = est.n
    nprev_draws = previous === nothing ? 0 : size(previous.draws, 2)
    if previous !== nothing
        nblocks = size(previous.draws, 1)
        replic == 0 && return previous
    end
    nblocks = Int(nblocks === nothing ? o["mh_nblck"] : nblocks)
    # random sources
    pseeds = previous === nothing ? nothing : get(previous.info, :last_seeds, nothing)
    chain_rng, init_rng = _chain_sources(kind, Int(seed), nblocks; previous_seeds=kind == :octave ? pseeds : nothing, offset=nprev_draws)
    initial_seeds = kind == :octave ? ChainSeeds[seeds_of(r) for r in chain_rng] : nothing
    # jscale (set_prior.m), scale_file, mh_tune_jscale
    js, base = JDyn.jscale(est, jscale)
    if method in (RWMH, TARB) && haskey(opts, "scale_file")
        sc = _scale_file(est, String(opts["scale_file"]))
        js = fill(sc, npar); base = sc
    end
    # proposal covariance
    xmode = mode isa ModeResult ? Float64.(mode.x) : mode === nothing ? Float64.(est.x0) : Float64.(mode)
    inv_h = cov !== nothing ? Matrix{Float64}(cov) : (mode isa ModeResult && mode.inv_hessian !== nothing ? Matrix{Float64}(mode.inv_hessian) : nothing)
    jcov = jumping_covariance === nothing ? String(get(o, "mcmc_jumping_covariance", "hessian")) : String(jumping_covariance)
    use_mh_cov = _est_flag(get(opts, "use_mh_covariance_matrix", 0))
    function mode_covariance()
        cov !== nothing && return Matrix{Float64}(cov)
        (inv_h === nothing && jcov == "hessian") && throw(EstimationError("check_posterior_sampler_options:: I cannot start the MCMC because the Hessian of the posterior kernel at the mode was not computed or there is no previous MCMC to load"))
        set_mcmc_jumping_covariance(inv_h, npar, jcov, est)
    end
    V = nothing
    prior_start = false
    V1 = nothing; WR = nothing
    modes_list = nothing
    if method == SLICE
        if opts["mode_files"] !== nothing
            modes_list = _load_slice_modes(est, String(opts["mode_files"]))
            opts["rotated"] = 1.0; opts["WR"] = nothing
        elseif modes !== nothing
            modes_list = [Float64.(m) for m in modes]
        elseif opts["mode"] !== nothing && !isempty(opts["mode"])
            modes_list = opts["mode"]
        end
        multimodal = modes_list !== nothing
        multimodal && (opts["rotated"] = 1.0)
        with_mode = _est_flag(opts["slice_initialize_with_mode"]) && previous === nothing
        if with_mode && mode === nothing && cov === nothing
            throw(EstimationError("check_posterior_sampler_options:: The option \"slice_initialize_with_mode\" is inconsistent with mode_compute=0 or empty mode_file."))
        end
        if !with_mode && previous === nothing
            prior_start = true
            xmode = prior_mode(est.prior)
            k = isnan.(xmode); xmode[k] = est.prior.p1[k]
            outb = (xmode .< slb) .| (xmode .> sub)
            xmode[outb] = est.prior.p1[outb]
        end
        if previous !== nothing && use_mh_cov
            V = _mh_covariance(previous)[2]
        elseif with_mode
            V = mode_covariance()
        elseif multimodal
            var = est.prior.p2 .^ 2
            var[isinf.(var)] .= 1.0
            V = Matrix(Diagonal(var))
        end
        if _est_flag(opts["rotated"]) && !multimodal
            if rotation !== nothing
                V1 = Matrix{Float64}(rotation[1]); WR = rotation[2] === nothing ? nothing : Float64.(rotation[2])
            elseif V === nothing
                throw(EstimationError("check_posterior_sampler_options:: Rotated slice cannot start: there is no previous MCMC to load (with use_mh_covariance_matrix) or the Hessian at the mode is not computed (use slice_initialize_with_mode)"))
            else
                # Octave's eig of a symmetric matrix is LAPACK dsyev (upper triangle): the eigenvector signs are dsyev's (eigen(Symmetric) uses dsyevr)
                w, V1 = LinearAlgebra.LAPACK.syev!('V', 'U', Matrix{Float64}((V .+ V') ./ 2))
                WR = 3 .* sqrt.(w)
            end
        end
        (!_est_flag(opts["rotated"]) && !with_mode && previous === nothing) && (V = nothing)
    elseif previous !== nothing && use_mh_cov
        V = _mh_covariance(previous)[2]
    elseif previous === nothing
        V = mode_covariance()
    end
    d = nothing
    if method == IMH && previous !== nothing && mode === nothing && haskey(previous.info, :proposal_center)
        xmode = Float64.(previous.info[:proposal_center])
    end
    if previous !== nothing && method in (RWMH, IMH) && !use_mh_cov
        haskey(previous.info, :proposal_chol) || throw(EstimationError("load_mh_file: the stored chains carry no proposal density"))
        d = Matrix{Float64}(previous.info[:proposal_chol])
        js = Float64.(previous.info[:jscale_vector]); base = Float64(previous.info[:jscale])
    elseif V !== nothing
        d = Matrix(cholesky(Symmetric(V)).U)
    end
    # mh_tune_jscale (dynare_estimation_1.m: before the sampler; always with the default normal rwmh proposal and the global streams)
    target = tune_jscale === nothing ? _tune_target(o) : tune_jscale
    if target !== nothing && target !== false
        if method != RWMH
            @warn "mh_tune_jscale is only available with Random Walk Metropolis Hastings!"
        else
            V0 = mode_covariance()
            tr = tune_rng === nothing ? (kind == :octave ? OctaveRNG(Int(seed)) : Random.Xoshiro(Int(seed) + 17)) : tune_rng
            guess = tune_guess === nothing ? get(o, "mh_tune_guess", nothing) : tune_guess
            tuned = tune_mh_jscale(logpost, V0, xmode, mlb, mub; target=Float64(target), guess=guess, r=tr)
            js = fill(tuned, npar); base = tuned
            @info "mh_jscale has been set equal to $tuned"
        end
    end
    W1 = method == SLICE ? Float64(opts["initial_step_size"]) .* (sub .- slb) : nothing
    # starting points
    starts = zeros(nblocks, npar); lp0 = zeros(nblocks)
    if previous !== nothing
        starts = Matrix{Float64}(previous.draws[:, end, :]); lp0 = Float64.(previous.log_posterior[:, end])
    elseif x0 !== nothing
        X0 = reshape(Float64.(x0), nblocks, npar)
        for j in 1:nblocks
            starts[j, :] = X0[j, :]; lp0[j] = logpost(X0[j, :])
            isfinite(lp0[j]) || throw(EstimationError("chain $j: the log posterior is not finite at the starting point"))
        end
    elseif nblocks == 1
        (all(xmode .>= mlb) && all(xmode .<= mub)) ||
            throw(EstimationError("Initialization failed: the $(prior_start ? "prior" : "posterior") mode lies outside the prior bounds"))
        starts[1, :] = xmode; lp0[1] = logpost(xmode)
    else
        factor = init_scale_factor === nothing ? o["mh_init_scale_factor"] : Float64(init_scale_factor)
        for j in 1:nblocks
            valid = false; init_iter = 0; trial = 1
            while !valid && trial <= 10
                cand = d === nothing ? prior_draw(est.prior, mlb, mub, init_rng) :
                       rand_multivariate_normal(init_rng, xmode, d .* (factor * base))
                if all(cand .>= mlb) && all(cand .<= mub)
                    lp = logpost(cand)
                    if isfinite(lp)
                        starts[j, :] = cand; lp0[j] = lp; valid = true
                    end
                end
                init_iter += 1
                if !valid && init_iter > 100
                    factor *= 0.9
                    trial += 1
                end
            end
            valid || throw(EstimationError("I'm unable to find a starting value for block $j."))
        end
    end
    student = get(opts, "proposal_distribution", "") == "rand_multivariate_student"
    if method == IMH && student && !haskey(opts, "student_degrees_of_freedom")
        throw(EstimationError("independent_metropolis_hastings with rand_multivariate_student: set student_degrees_of_freedom (Dynare 7.2 has no default for this sampler and stops)"))
    end
    dof = get(opts, "student_degrees_of_freedom", nothing)
    chol_prop = d === nothing ? nothing : d * Diagonal(js)
    draws = zeros(nblocks, replic, npar); logpo = zeros(nblocks, replic); acc = zeros(nblocks, replic); neval = zeros(nblocks, replic)
    prior_flags = fill(_est_flag(get(opts, "use_prior_draws", false)), nblocks)
    for j in 1:nblocks
        r = chain_rng[j]
        x = copy(starts[j, :]); lp = lp0[j]
        pdraw = () -> prior_draw(est.prior, mlb, mub, r)
        for t in 1:replic
            a = 1.0; ne = 1.0
            if method == SLICE
                if _est_flag(opts["rotated"])
                    if modes_list !== nothing
                        Vm = multimodal_directions(x, modes_list, r)
                        x, lp, nev = rotated_slice_iteration(logpost, x, slb, sub, Vm, nothing, r)
                        x, lp, nev2, prior_flags[j] = slice_iteration(logpost, x, slb, sub, W1, r, pdraw, prior_flags[j], lp)
                        ne = sum(nev) + sum(nev2)
                    else
                        x, lp, nev = rotated_slice_iteration(logpost, x, slb, sub, V1, WR, r)
                        ne = sum(nev)
                    end
                else
                    x, lp, nev, prior_flags[j] = slice_iteration(logpost, x, slb, sub, W1, r, pdraw, prior_flags[j], lp)
                    ne = sum(nev)
                end
                a = 1.0
            elseif method == TARB
                x, lp, a, nfe = tarb_iteration(est, x, lp, slb, sub, mlb, mub, js, opts, r)
                ne = Float64(nfe)
            else
                center = method == RWMH ? x : xmode
                par = student ? rand_multivariate_student(r, center, chol_prop, dof) : rand_multivariate_normal(r, center, chol_prop)
                lpp = (all(par .> slb) && all(par .< sub)) ? logpost(par) : -Inf
                ratio = if method == RWMH
                    lpp - lp
                else
                    qo, qn = student ? (multivariate_student_pdf(x, xmode, chol_prop, dof), multivariate_student_pdf(par, xmode, chol_prop, dof)) :
                                       (multivariate_normal_pdf(x, xmode, chol_prop, npar), multivariate_normal_pdf(par, xmode, chol_prop, npar))
                    lpp - lp + log(qo) - log(qn)
                end
                if lpp > -Inf && log(rand(r)) < ratio
                    x = par; lp = lpp; a = 1.0
                else
                    a = 0.0
                end
            end
            draws[j, t, :] = x; logpo[j, t] = lp; acc[j, t] = a; neval[j, t] = ne
        end
    end
    info = Dict{Symbol,Any}(:method => method, :jscale => base, :jscale_vector => Float64.(js), :sampler_options => opts,
                            :feval => vec(sum(neval; dims=2)) ./ max(replic, 1), :new_draws => replic,
                            :initial_logpost => lp0, :kind => kind)
    d === nothing || (info[:proposal_chol] = d)
    method == IMH && (info[:proposal_center] = xmode)
    V1 === nothing || (info[:rotation] = V1)
    WR === nothing || (info[:rotation_widths] = WR)
    if kind == :octave
        info[:last_seeds] = ChainSeeds[seeds_of(r) for r in chain_rng]
        info[:initial_seeds] = initial_seeds
    end
    X0 = starts
    if previous !== nothing
        draws = cat(previous.draws, draws; dims=2)
        logpo = hcat(previous.log_posterior, logpo)
        acc = hcat(Float64.(previous.accepted), acc)
        neval = hcat(previous.neval === nothing ? ones(size(previous.log_posterior)) : previous.neval, neval)
        X0 = previous.x0
        haskey(previous.info, :proposal_chol) && !haskey(info, :proposal_chol) && (info[:proposal_chol] = previous.info[:proposal_chol])
    end
    nnew = replic
    rate = vec(sum(acc[:, end-nnew+1:end]; dims=2)) ./ max(nnew, 1)
    _mc_result(draws, logpo, rate, est.names, X0, drop, acc .> 0; neval=neval, info=info)
end

"Tuning target of `mh_tune_jscale` from the estimation options (`nothing`: off)."
function _tune_target(o)
    v = get(o, "mh_tune_jscale", nothing)
    (v === nothing || v === false) && return nothing
    v === true ? 0.33 : Float64(v)
end

function _scale_file(est::Estimation, name::String)
    for b in (isabspath(name) ? [""] : [est.model.mod_dir, pwd()]), ext in ("", ".mat")
        p = isempty(b) ? name * ext : joinpath(b, name * ext)
        if isfile(p)
            d = MAT.matread(p)
            haskey(d, "Scale") || throw(EstimationError("scale_file $p: no variable Scale"))
            return Float64(only(d["Scale"]))
        end
    end
    throw(EstimationError("initial_estimation_checks:: The specified scale_file does not exist."))
end

function _load_slice_modes(est::Estimation, name::String)
    p = resolve_mat_file(est.model, name)
    d = MAT.matread(p)
    haskey(d, "xparams") || throw(EstimationError("mode_files $name: no variable xparams"))
    xp = Matrix{Float64}(d["xparams"])
    size(xp, 2) < 2 && throw(EstimationError("check_posterior_sampler_options:: Variable xparams loaded in file <$name> has size [$(size(xp, 1))x$(size(xp, 2))]: it must contain at least two columns, to allow multi-modal sampling."))
    [xp[:, j] for j in 1:size(xp, 2)]
end
