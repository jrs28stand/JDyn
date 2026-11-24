# Posterior samplers of Dynare 7.2 beyond the plain random-walk Metropolis-Hastings of src/estimation.jl, and what they share:
# `posterior_sampler_options`, Student proposals, jumping covariances, `mh_tune_jscale`, the slice sampler (plain, rotated and
# multimodal), independent MH, tailored random block MH (TaRB), and Dynare's random numbers for all of them (`mcmc_rng=:octave`,
# src/octave_rng.jl). Port of pydyn/samplers.py (itself a port of matlab/estimation/{posterior_sampler_iteration, slice_sampler,
# rotated_slice_sampler, check_posterior_sampler_options, posterior_sampler_initialization, TaRB_optimizer_wrapper, chol_SE,
# calibrate_mh_scale_parameter}.m, matlab/utilities/estimation/set_mcmc_jumping_covariance.m, matlab/distributions/*.m and
# matlab/@dprior/draw.m of Dynare 7.2).
#
# Random numbers. With `mcmc_rng=:octave` chain j draws from OctaveRNG seeded with s + 1 + ... + j (posterior_sampler_initialization.m
# adds j to the seed cumulatively), the starting points of several chains from OctaveRNG(s) (s = options_.DynareRandomStreams.seed),
# so a chain is Dynare's chain draw for draw given the same mode and starting point. Draws from priors other than normal and
# uniform use the chain's uniforms through the inverse CDF, not Octave's gamrnd/betarnd/wblrnd (whose state Dynare does not seed):
# those starting points differ from Dynare's. With `mcmc_rng=:julia` (default) each chain has its own Xoshiro stream.
#
# Bounds as Dynare 7.2 does it (verified on Dynare's chains in pydyn): the posterior kernel is evaluated with the prior support
# (`prior_trunc` applied), but the samplers check their proposals against `est.lb/est.ub`, the prior support intersected with the
# bounds of the estimated_params block (`sampler_options.bounds`, set in check_posterior_sampler_options.m before the bounds are
# reset for the MCMC); the slice sampler steps out within those.

const RWMH = "random_walk_metropolis_hastings"
const SLICE = "slice"
const TARB = "tailored_random_block_metropolis_hastings"
const IMH = "independent_metropolis_hastings"
const HSSMC = "hssmc"
const DSMH = "dsmh"
const SMC_METHODS = (HSSMC, DSMH)
const MCMC_METHODS = (RWMH, SLICE, TARB, IMH)
const _NOT_PORTED = Dict("dime_mcmc" => "the DIME ensemble sampler (dime.m) is not implemented",
                         "online" => "the online auxiliary particle filter (online_auxiliary_filter.m) is not implemented")

# default_option_values.m (posterior_sampler_options.*)
const _SAMPLER_DEFAULTS = Dict{String,Dict{String,Any}}(
    RWMH => Dict{String,Any}("proposal_distribution" => "rand_multivariate_normal", "student_degrees_of_freedom" => 3.0,
                             "use_mh_covariance_matrix" => 0.0, "save_tmp_file" => 0.0),
    TARB => Dict{String,Any}("proposal_distribution" => "rand_multivariate_normal", "student_degrees_of_freedom" => 3.0,
                             "mode_compute" => 4.0, "new_block_probability" => 0.25, "optim_opt" => "", "save_tmp_file" => 1.0),
    SLICE => Dict{String,Any}("rotated" => 0.0, "slice_initialize_with_mode" => 0.0, "use_mh_covariance_matrix" => 0.0,
                              "mode" => nothing, "mode_files" => nothing, "maximize" => false, "maximize_using_mh_bounds" => false,
                              "mode_compute" => 5.0, "initial_step_size" => 0.8, "use_prior_draws" => false, "save_tmp_file" => 1.0,
                              "save_iter_info_file" => true, "fast_likelihood_evaluation_for_rejection" => false,
                              "fast_likelihood_evaluation_for_rejection_penalty" => 10.0,
                              "draw_init_state_from_smoother" => false, "draw_init_state_with_rotated_slice" => false),
    IMH => Dict{String,Any}("proposal_distribution" => "rand_multivariate_normal", "use_mh_covariance_matrix" => 0.0,
                            "save_tmp_file" => 0.0),
    HSSMC => Dict{String,Any}("steps" => 25.0, "lambda" => 2.0, "particles" => 20000.0, "scale" => 0.5, "acpt" => 1.0, "target" => 0.25),
    DSMH => Dict{String,Any}("H" => 25.0, "M" => 20.0, "N" => 200.0, "G" => 10.0, "K" => 10.0, "alpha0" => 0.2, "alpha1" => 0.3, "tau" => 10.0))
# options each sampler accepts (check_posterior_sampler_options.m); the others give Dynare's "Unknown option" warning
const _SAMPLER_ACCEPTED = Dict{String,Set{String}}(
    RWMH => Set(["proposal_distribution", "student_degrees_of_freedom", "use_mh_covariance_matrix", "scale_file", "save_tmp_file"]),
    TARB => Set(["proposal_distribution", "student_degrees_of_freedom", "mode_compute", "optim", "new_block_probability", "scale_file", "save_tmp_file"]),
    IMH => Set(["proposal_distribution", "student_degrees_of_freedom", "use_mh_covariance_matrix", "save_tmp_file"]),
    SLICE => Set(["rotated", "mode", "mode_files", "slice_initialize_with_mode", "initial_step_size", "use_mh_covariance_matrix",
                  "save_tmp_file", "save_iter_info_file", "fast_likelihood_evaluation_for_rejection",
                  "fast_likelihood_evaluation_for_rejection_penalty", "use_prior_draws", "draw_init_state_from_smoother",
                  "draw_init_state_with_rotated_slice", "maximize", "maximize_using_mh_bounds", "mode_compute"]),
    HSSMC => Set(["target", "steps", "scale", "particles", "lambda"]),
    DSMH => Set(["H", "M", "N", "G", "K", "tau", "alpha0", "alpha1"]))
const _SAMPLER_WARN = Dict(RWMH => "rwmh_sampler", TARB => "tarb_sampler", IMH => "imh_sampler", SLICE => "slice_sampler", HSSMC => "hssmc", DSMH => "dsmh")

# ---------------------------------------------------------------------------------------------------- option strings
"MATLAB's `str2num` for the values of option strings: numbers, `true`/`false`, vectors and matrices; `nothing` if it is not numeric."
function _str2num(s::AbstractString)
    t = strip(s)
    t == "true" && return true
    t == "false" && return false
    body = startswith(t, "[") && endswith(t, "]") ? t[2:end-1] : t
    rows = Vector{Float64}[]
    for r in split(body, ";")
        vals = Float64[]
        for tok in split(strip(r), r"[\s,]+"; keepempty=false)
            v = tryparse(Float64, tok)
            v === nothing && return nothing
            push!(vals, v)
        end
        isempty(vals) || push!(rows, vals)
    end
    (isempty(rows) || length(unique(length.(rows))) != 1) && return nothing
    A = permutedims(reduce(hcat, rows))
    length(A) == 1 ? A[1, 1] : size(A, 1) == 1 ? vec(A) : A
end

"""
    read_key_value_string(s) -> Vector{Pair{String,Any}}

`read_key_value_string.m`: `'key1',value1,'key2',value2` as the preprocessor writes the `posterior_sampler_options` list (the
JSON string has doubled quotes). Values `str2num` reads are numbers (or matrices), the others strings without their quotes.
"""
function read_key_value_string(s::AbstractString)
    s = replace(String(s), "''" => "'")
    s = replace(s, r",\s*" => ",")
    isempty(strip(s)) && return Pair{String,Any}[]
    count(==('('), s) == count(==(')'), s) || throw(EstimationError("read_key_value_string: The number of opening and closing brackets does not match!"))
    begin_ = [m.offset for m in eachmatch(r"'\(", s)]
    end_ = [m.offset + 1 for m in eachmatch(r"\)'", s)]
    length(begin_) == length(end_) || throw(EstimationError("read_key_value_string: Sublists could not be identified!"))
    commas = [i for (i, ch) in enumerate(s) if ch == ',' && !any(b <= i <= e for (b, e) in zip(begin_, end_))]
    iseven(length(commas)) && throw(EstimationError("read_key_value_string: Wrong number of Key-Value pairs!"))
    bounds = vcat(0, commas, length(s) + 1)
    pieces = [s[bounds[k]+1:bounds[k+1]-1] for k in 1:length(bounds)-1]
    out = Pair{String,Any}[]
    for k in 1:2:length(pieces)
        key = strip(pieces[k]); key = length(key) >= 2 ? key[2:end-1] : key
        val = pieces[k+1]
        num = _str2num(val)
        if num === nothing
            v = strip(val)
            if startswith(v, "'(") && endswith(v, ")'")
                v = v[3:end-2]
            elseif length(v) >= 2 && v[1] == '\'' && v[end] == '\''
                v = v[2:end-1]
            end
            push!(out, key => String(v))
        else
            push!(out, key => num)
        end
    end
    out
end

_kv_pairs(spec) = spec === nothing ? Pair{String,Any}[] : spec isa AbstractString ? read_key_value_string(spec) :
                  spec isa AbstractDict ? Pair{String,Any}[String(k) => v for (k, v) in spec] : Pair{String,Any}[String(k) => v for (k, v) in spec]
_has_key(spec, key) = any(p -> first(p) == key, _kv_pairs(spec))

"""
    sampler_options(method, spec) -> Dict

Options of `method` (`check_posterior_sampler_options.m`, first call): Dynare's defaults overridden by `spec` (the
`posterior_sampler_options` string `'key',value,...`, a Dict or a list of pairs). Unknown keys give Dynare's warning; invalid
values raise an `EstimationError` with Dynare's message.
"""
function sampler_options(method::AbstractString, spec=nothing; warn::Bool=true)
    haskey(_NOT_PORTED, method) && throw(EstimationError("posterior_sampling_method=$method: $(_NOT_PORTED[method])"))
    haskey(_SAMPLER_DEFAULTS, method) || throw(EstimationError("check_posterior_sampler_options:: Unknown posterior_sampling_method option $method"))
    opts = Dict{String,Any}(k => deepcopy(v) for (k, v) in _SAMPLER_DEFAULTS[method])
    for (key, val) in _kv_pairs(spec)
        if !(key in _SAMPLER_ACCEPTED[method])
            warn && @warn "$(_SAMPLER_WARN[method]): Unknown option ($key)!"
            continue
        end
        if key == "proposal_distribution"
            lowercase(String(val)) in ("rand_multivariate_student", "rand_multivariate_normal") ||
                throw(EstimationError("initial_estimation_checks:: the proposal_distribution option to estimation takes either rand_multivariate_student or rand_multivariate_normal as options"))
            val = lowercase(String(val))
        elseif key == "student_degrees_of_freedom"
            (val isa Real && val > 0) || throw(EstimationError("initial_estimation_checks:: the student_degrees_of_freedom takes a positive integer argument"))
        elseif key == "new_block_probability"
            (0 <= val <= 1) || throw(EstimationError("check_posterior_sampler_options:: The tarb new_block_probability must be between 0 and 1!"))
        elseif key == "initial_step_size"
            (0 < val < 1) || throw(EstimationError("check_posterior_sampler_options:: slice initial_step_size must be between 0 and 1"))
        elseif key == "optim"
            key = "optim_opt"
        elseif key == "mode"
            M = val isa AbstractMatrix ? val : reshape(Float64.(collect(val)), :, 1)
            val = [Float64.(M[:, j]) for j in 1:size(M, 2)]
        end
        opts[key] = val
    end
    if method == SLICE
        opts["mode"] !== nothing && !isempty(opts["mode"]) && (opts["rotated"] = 1.0; opts["WR"] = nothing)
        (_est_flag(opts["draw_init_state_from_smoother"]) || _est_flag(opts["draw_init_state_with_rotated_slice"])) &&
            throw(EstimationError("slice: draw_init_state_from_smoother / draw_init_state_with_rotated_slice act on estimated initial states, which JDyn does not support"))
        (_est_flag(opts["maximize"]) || _est_flag(opts["maximize_using_mh_bounds"])) &&
            throw(EstimationError("slice: the maximize options (an optimiser in place of the slice draws) are not implemented"))
        _est_flag(opts["fast_likelihood_evaluation_for_rejection"]) &&
            throw(EstimationError("slice: fast_likelihood_evaluation_for_rejection is not implemented"))
    end
    if method == DSMH
        mod(opts["N"] * opts["G"], opts["M"]) == 0 ||
            throw(EstimationError("check_posterior_sampler_options:: DSMH requires M to divide N*G = $(Int(opts["N"] * opts["G"])) evenly."))
    end
    opts
end

# ---------------------------------------------------------------------------------------------------- proposals
_nrand(r, n) = n == 1 ? [randn(r)] : vec(randn(r, n))
_urand(r) = rand(r)
_urand(r, n) = n == 1 ? [rand(r)] : vec(rand(r, n))

"`rand_multivariate_normal.m`: `mean + randn(1,n) * chol_upper`."
rand_multivariate_normal(r, mean::AbstractVector, chol_upper::AbstractMatrix) = mean .+ vec(transpose(_nrand(r, length(mean))) * chol_upper)

"`rand_multivariate_student.m` (one draw): the normals of the scale are drawn after those of the direction."
function rand_multivariate_student(r, mean::AbstractVector, chol_upper::AbstractMatrix, df)
    isinteger(df) || throw(EstimationError("rand_multivariate_student: student_degrees_of_freedom must be an integer (randn(1, df))"))
    z = vec(transpose(_nrand(r, length(mean))) * chol_upper)
    w = _nrand(r, Int(df))
    mean .+ z .* sqrt(df / sum(abs2, w))
end

"`multivariate_normal_pdf.m` (the density, not its log; it underflows to zero where Dynare's does)."
function multivariate_normal_pdf(x::AbstractVector, mean::AbstractVector, chol_upper::AbstractMatrix, n)
    d = x .- mean
    v = UpperTriangular(chol_upper) \ (transpose(UpperTriangular(chol_upper)) \ d)
    (2pi)^(-0.5n) * prod(diag(chol_upper))^(-1) * exp(-0.5 * dot(d, v))
end

"`multivariate_student_pdf.m`."
function multivariate_student_pdf(x::AbstractVector, mean::AbstractVector, chol_upper::AbstractMatrix, df)
    df > 0 || throw(EstimationError("Degrees of freedom 'df' must be positive"))
    d = x .- mean
    nn = length(d)
    t1 = SpecialFunctions.gamma(0.5 * (nn + df)) / (SpecialFunctions.gamma(0.5 * df) * (df * pi)^(0.5nn))
    t2 = t1 / prod(diag(chol_upper))
    q = transpose(UpperTriangular(chol_upper)) \ d
    t2 / (1 + sum(abs2, q) / df)^(0.5 * (nn + df))
end

"""
    chol_SE(A) -> (R, indefinite, E)

Schnabel and Eskow's modified Cholesky factorisation (`chol_SE.m`, no pivoting): upper triangular `R` with `R'R = A + diag(E)`.
"""
function chol_SE(A0::AbstractMatrix)
    A = Matrix{Float64}(A0)
    sum(abs, A .- A') > 1e-8 && throw(EstimationError("A is not symmetric"))
    sum(abs, A .- A') > 0 && (A = (A .+ A') ./ 2)
    n = size(A, 1)
    tau1 = eps()^(1 / 3); tau2 = eps()^(1 / 3)
    phase1 = true; delta = 0.0
    g = zeros(n); E = zeros(n)
    gammma = maximum(diag(A))
    any(<(0), diag(A)) && (phase1 = false)
    taugam = tau1 * gammma
    function gersh(A, j)
        gg = zeros(n)
        for ii in j:n
            up = ii == 1 ? 0.0 : sum(abs, A[ii, j:ii-1])
            after = ii == n ? 0.0 : sum(abs, A[ii+1:n, ii])
            gg[ii] = up + after - A[ii, ii]
        end
        gg
    end
    phase1 || (g = gersh(A, 1))
    if n == 1
        delta = tau2 * abs(A[1, 1]) - A[1, 1]
        delta > 0 && (E[1] = delta)
        A[1, 1] == 0 && (E[1] = tau2)
        A[1, 1] = sqrt(A[1, 1] + E[1])
    end
    for j in 1:n-1
        if phase1
            if A[j, j] > 0
                minimum(diag(A[j+1:n, j+1:n]) .- A[j+1:n, j] .^ 2 ./ A[j, j]) < taugam && (phase1 = false)
            else
                phase1 = false
            end
            if phase1
                A[j, j] = sqrt(A[j, j])
                tempjj = A[j, j]
                for ii in j+1:n
                    A[ii, j] = A[ii, j] / tempjj
                end
                for ii in j+1:n
                    temp = A[ii, j]
                    for k in j+1:ii
                        A[ii, k] = A[ii, k] - temp * A[k, j]
                    end
                end
                j == n - 1 && (A[n, n] = A[n, n] >= 0 ? sqrt(A[n, n]) : NaN)
            else
                g = gersh(A, j)
            end
        end
        if !phase1
            if j != n - 1
                normj = sum(abs, A[j+1:n, j])
                delta = max(0.0, delta, -A[j, j] + normj, -A[j, j] + taugam)
                E[j] = delta
                A[j, j] = A[j, j] + E[j]
                if A[j, j] != normj
                    temp = normj / A[j, j] - 1
                    for ii in j+1:n
                        g[ii] = g[ii] + abs(A[ii, j]) * temp
                    end
                end
                A[j, j] = sqrt(A[j, j])
                tempjj = A[j, j]
                for ii in j+1:n
                    A[ii, j] = A[ii, j] / tempjj
                end
                for ii in j+1:n
                    temp = A[ii, j]
                    for k in j+1:ii
                        A[ii, k] = A[ii, k] - temp * A[k, j]
                    end
                end
            else
                A[n-1, n] = A[n, n-1]
                ev = eigvals(Symmetric(A[n-1:n, n-1:n]))
                delta = max(0.0, delta, -minimum(ev) + tau2 * max((maximum(ev) - minimum(ev)) / (1 - tau1), gammma))
                if delta > 0
                    A[n-1, n-1] += delta; A[n, n] += delta
                    E[n-1] = delta; E[n] = delta
                end
                A[n-1, n-1] = sqrt(A[n-1, n-1])
                A[n, n-1] = A[n, n-1] / A[n-1, n-1]
                A[n, n] = A[n, n] - A[n, n-1]^2
                A[n, n] = sqrt(A[n, n])
            end
        end
    end
    Matrix(transpose(LowerTriangular(A))), !phase1, E
end

# ---------------------------------------------------------------------------------------------------- prior mode and draws
"`bayestopt_.p5` (set_prior.m / compute_prior_mode.m): NaN for uniform priors; the first mode of a beta prior with two modes."
function prior_mode(p::Prior)
    out = fill(NaN, length(p.shape))
    for i in eachindex(p.shape)
        sh = p.shape[i]; a, b = p.p6[i], p.p7[i]; p3, p4 = p.p3[i], p.p4[i]
        m = NaN
        if sh == 1
            if a > 1 && b > 1
                m = (a - 1) / (a + b - 2)
            elseif a < 1 && b < 1
                m = 0.0
            elseif (a < 1 && b > 1 - eps()) || (abs(a - 1) < 2eps() && b > 1)
                m = 0.0
            elseif (a > 1 && b < 1 + eps()) || (abs(a - 1) < 2eps() && b < 1)
                m = 1.0
            elseif abs(a - 1) < 2eps() && abs(b - 1) < 2eps()
                m = 0.5
            end
            m = m * (p4 - p3) + p3
        elseif sh == 2
            m = (a < 1 ? 0.0 : (a - 1) * b) + p3
        elseif sh == 3
            m = p.p1[i]
        elseif sh == 4
            m = 1 / sqrt((b + 1) / a) + p3
        elseif sh == 6
            m = a / (b + 2) + p3
        elseif sh == 8
            m = (a <= 1 ? 0.0 : b * ((a - 1) / a)^(1 / a)) + p3
        end
        out[i] = m
    end
    out
end

"""
    prior_draw(prior, lb, ub, r) -> Vector

One draw from the priors truncated to [lb, ub] (`@dprior/draw.m`): uniform priors from `rand`, normal priors from `randn` (Dynare's
numbers with an OctaveRNG), the others by the inverse CDF of one uniform each (Dynare uses `gamrnd` etc., which cannot be replayed).
"""
function prior_draw(p::Prior, lb::AbstractVector, ub::AbstractVector, r)
    n = length(p.shape)
    x = fill(NaN, n)
    oob(idx) = [i for i in idx if x[i] > ub[i] || x[i] < lb[i]]
    function fill_group!(idx, draw)
        isempty(idx) && return
        for i in idx
            x[i] = draw(i)
        end
        bad = oob(idx)
        while !isempty(bad)
            for i in bad
                x[i] = draw(i)
            end
            bad = oob(idx)
        end
    end
    # uniform and normal groups draw their whole vector at once (as draw.m), in this order
    idx = findall(==(5), p.shape)
    if !isempty(idx)
        x[idx] = _urand(r, length(idx)) .* (p.p4[idx] .- p.p3[idx]) .+ p.p3[idx]
        bad = oob(idx)
        while !isempty(idx) && !isempty(bad)
            x[idx] = _urand(r, length(idx)) .* (p.p4[idx] .- p.p3[idx]) .+ p.p3[idx]
            bad = oob(idx)
        end
    end
    idx = findall(==(3), p.shape)
    if !isempty(idx)
        x[idx] = _nrand(r, length(idx)) .* p.p7[idx] .+ p.p6[idx]
        bad = oob(idx)
        while !isempty(bad)
            x[bad] = _nrand(r, length(bad)) .* p.p7[bad] .+ p.p6[bad]
            bad = oob(idx)
        end
    end
    fill_group!(findall(==(2), p.shape), i -> _gaminv(rand(r), p.p6[i], p.p7[i]) + p.p3[i])
    fill_group!(findall(==(1), p.shape), i -> (p.p4[i] - p.p3[i]) * _betainv(rand(r), p.p6[i], p.p7[i]) + p.p3[i])
    fill_group!(findall(==(4), p.shape), i -> sqrt(1 / _gaminv(rand(r), p.p7[i] / 2, 2 / p.p6[i])) + p.p3[i])
    fill_group!(findall(==(6), p.shape), i -> 1 / _gaminv(rand(r), p.p7[i] / 2, 2 / p.p6[i]) + p.p3[i])
    fill_group!(findall(==(8), p.shape), i -> _wblinv(rand(r), p.p7[i], p.p6[i]) + p.p3[i])
    x
end

# ---------------------------------------------------------------------------------------------------- jumping covariance
"""
    set_mcmc_jumping_covariance(invhess, n, kind, est; base_dir) -> Matrix

`set_mcmc_jumping_covariance.m`: `hessian` keeps the inverse Hessian at the mode, `prior_variance` the inverse of the matrix of
inverse prior variances, `identity_matrix` the identity, anything else names a .mat file with a matrix `jumping_covariance`.
"""
function set_mcmc_jumping_covariance(invhess, n::Int, kind::AbstractString, est::Estimation; base_dir=est.model.mod_dir)
    kind == "hessian" && return invhess
    if kind == "prior_variance"
        any(isinf, est.prior.p2) && throw(EstimationError("dynare_estimation_1: Infinite prior variances detected. You cannot use the prior variances as the proposal density, if some variances are Inf."))
        hess = Diagonal(1 ./ est.prior.p2 .^ 2)
        hsd = sqrt.(diag(hess))
        return inv(Matrix(hess) ./ (hsd * hsd')) ./ (hsd * hsd')
    elseif kind == "identity_matrix"
        return Matrix{Float64}(I, n, n)
    end
    path = nothing
    for b in (isabspath(kind) ? [""] : [base_dir, pwd()]), ext in ("", ".mat")
        p = isempty(b) ? kind * ext : joinpath(b, kind * ext)
        isfile(p) && (path = p; break)
    end
    path === nothing && throw(EstimationError("dynare_estimation_1: No matrix named 'jumping_covariance' could be found in $kind.mat!"))
    d = MAT.matread(path)
    haskey(d, "jumping_covariance") || throw(EstimationError("dynare_estimation_1: No matrix named 'jumping_covariance' could be found in $kind.mat!"))
    hess = Matrix{Float64}(d["jumping_covariance"])
    (size(hess, 1) == size(hess, 2) == n) ||
        throw(EstimationError("dynare_estimation_1: 'jumping_covariance' matrix (loaded from $kind.mat) must be square and have $n rows and columns!"))
    _ispd(hess) || throw(EstimationError("dynare_estimation_1: Specified 'MCMC_jumping_covariance' is not positive definite!"))
    hsd = sqrt.(diag(hess))
    inv(hess ./ (hsd * hsd')) ./ (hsd * hsd')
end

# ---------------------------------------------------------------------------------------------------- tuning the scale
"""
    tune_mh_jscale(log_post, invhess, x0, lb, ub; target, guess, maxiter, rho, stepsize, c1, c2, c3, r) -> Float64

`calibrate_mh_scale_parameter.m`: a random-walk Metropolis-Hastings run from the mode that adapts the scale until the overall and
local acceptance rates stay within c2 and c1 of `target` for more than c3 checks. Dynare's tuning always uses the default
`rwmh` proposal (normal) and Dynare's global random streams: `r` is the stream (OctaveRNG(seed) replays Dynare when the global
stream is untouched, as with `mode_compute=0` and a mode file).
"""
function tune_mh_jscale(log_post, invhess::AbstractMatrix, x0::AbstractVector, lb, ub; target=0.33, guess=nothing, maxiter=200000,
                        rho=0.7, stepsize=1000, c1=0.02, c2=0.05, c3=4, r=Random.default_rng())
    n = length(x0)
    j = 1; jj = 1; isux = 0; jsux = 0; i = 0
    logpo0 = log_post(x0)
    logpo1 = logpo0
    correction = 1.0
    scale = guess === nothing ? 2.38 / sqrt(n) : Float64(guess)
    par = Float64.(x0)
    dd = try
        Matrix(cholesky(Symmetric(Matrix{Float64}(invhess))).U)
    catch
        throw(EstimationError("The covariance matrix has to be a symmetric positive definite matrix!"))
    end
    while j <= maxiter
        prop = rand_multivariate_normal(r, par, scale .* dd)
        logpo0 = (all(prop .> lb) && all(prop .< ub)) ? log_post(prop) : -Inf
        if logpo0 > -Inf && log(rand(r)) < logpo0 - logpo1
            par = prop; logpo1 = logpo0
            isux += 1; jsux += 1
        end
        if mod(j, stepsize) == 0
            r1 = jsux / jj
            r2 = isux / j
            cc = r1 / target
            correction = abs(cc - 1) > 0.05 ? correction^rho * cc^(1 - rho) : cc
            cc > 0 ? (scale *= correction) : (scale /= 10)
            jsux = 0; jj = 0
            (abs(r2 - target) < c2 && abs(r1 - target) < c1) ? (i += 1) : (i = 0)
            i > c3 && break
        end
        j += 1; jj += 1
    end
    scale
end

# ---------------------------------------------------------------------------------------------------- slice iterations
"""
One iteration of Dynare's univariate slice sampler (`slice_sampler.m`) through every parameter in turn. `logpost(x)` is the log
posterior kernel (-Inf where it is not defined), [lb, ub] the sampling box, `W1` the initial interval widths, `r` the random
source. Returns `(theta, fxsim, neval, use_prior_draws)`.
"""
function slice_iteration(logpost, theta0::AbstractVector, lb, ub, W1, r, prior_draw_fn, use_prior_draws::Bool, last_posterior)
    theta = Float64.(theta0)
    npar = length(theta)
    neval = zeros(npar)
    fxsim = NaN
    if use_prior_draws
        fxsim = last_posterior
        Z1 = fxsim + log(rand(r))
        ilogpo2 = -Inf; nattempts = 0; best = nothing
        while ilogpo2 < Z1 && nattempts < 10
            nattempts += 1
            local cand, itest
            while true
                cand = prior_draw_fn()
                if all(cand .>= lb) && all(cand .<= ub)
                    itest = logpost(cand)
                    isfinite(itest) && break
                end
            end
            if itest > ilogpo2
                ilogpo2 = itest; best = cand
            end
        end
        if ilogpo2 >= Z1
            theta = copy(best); fxsim = ilogpo2
        else
            use_prior_draws = false
        end
    end
    islow = falses(npar)
    it = 0
    while it < npar
        it += 1
        neval[it] = 0
        W = W1[it]
        xold = theta[it]
        XLB, XUB = lb[it], ub[it]
        XLB == XUB && continue
        fxold = logpost(theta)
        if !isfinite(fxold)
            @info "slice_sampler:: Iteration $it started with bad parameter set (fval is inf or nan)"
            icount = 0
            while !isfinite(fxold) && icount < 1000
                icount += 1
                theta = prior_draw_fn()
                (all(theta .>= lb) && all(theta .<= ub)) && (fxold = logpost(theta))
            end
            it = 1
            neval[it] = 0
            W = W1[it]; xold = theta[it]; XLB, XUB = lb[it], ub[it]
        end
        neval[it] += 1
        Z = fxold + log(rand(r))
        u = rand(r)
        L = max(XLB, xold - W * u)
        R = min(XUB, L + W)
        while L > XLB
            theta[it] = L
            fxl = logpost(theta)
            neval[it] += 1
            fxl <= Z && break
            L = max(XLB, L - W)
            if neval[it] > 30
                L = XLB; theta[it] = L
                fxl = logpost(theta)
                icount = 0
                while (isinf(fxl) || isnan(fxl)) && icount < 300
                    icount += 1
                    L += sqrt(eps()); theta[it] = L
                    fxl = logpost(theta)
                end
            end
        end
        neval1 = neval[it]
        while R < XUB
            theta[it] = R
            fxr = logpost(theta)
            neval[it] += 1
            fxr <= Z && break
            R = min(XUB, R + W)
            if neval[it] > neval1 + 30
                R = XUB; theta[it] = R
                fxr = logpost(theta)
                icount = 0
                while (isinf(fxr) || isnan(fxr)) && icount < 300
                    icount += 1
                    R -= sqrt(eps()); theta[it] = R
                    fxr = logpost(theta)
                end
            end
        end
        fxsim = Z - 1
        while fxsim < Z
            u = rand(r)
            xsim = L + u * (R - L)
            theta[it] = xsim
            fxsim = logpost(theta)
            neval[it] += 1
            xsim > xold ? (R = xsim) : (L = xsim)
            if (R - L) < 1e-6 && fxsim < Z
                @info "The sampling for parameter number $it is taking too long as the sampling set is too tight. Check the prior."
                islow[it] = true
                theta[it] = xold
                fxsim = fxold
                break
            end
        end
        if isinf(fxsim) || isnan(fxsim)
            theta[it] = xold
            fxsim = fxold
            @info "SLICE: posterior density is infinite. Reset values at initial ones."
        end
    end
    if any(islow)
        fxsim = logpost(theta)
        Z1 = fxsim + log(rand(r))
        ilogpo2 = -Inf; nattempts = 0; best = nothing
        while ilogpo2 < Z1 && nattempts < 10
            nattempts += 1
            local cand, itest
            while true
                cand = copy(theta)
                mine = prior_draw_fn()
                cand[islow] = mine[islow]
                if all(cand[islow] .>= lb[islow]) && all(cand[islow] .<= ub[islow])
                    itest = logpost(cand)
                    isfinite(itest) && break
                end
            end
            if itest > ilogpo2
                ilogpo2 = itest; best = cand
            end
        end
        if ilogpo2 > Z1
            theta = best; fxsim = ilogpo2
        end
    end
    theta, fxsim, neval, use_prior_draws
end

"""
`rotated_slice_sampler.m` (no initial-state estimation): a slice step along every column of `V1` in turn, within the part of the
line that stays in the box [lb, ub]; `W1` the initial widths (`nothing`: the whole feasible segment).
"""
function rotated_slice_iteration(logpost, theta0::AbstractVector, lb, ub, V1::AbstractMatrix, W1, r)
    theta = Float64.(theta0)
    npar = size(V1, 2)
    neval = zeros(npar)
    fxsim = NaN
    for it in 1:npar
        base = copy(theta)
        v = V1[:, it]
        xold = 0.0
        # tb = sort([(lb - theta)./v (ub - theta)./v], 2); MATLAB's max/min ignore NaN
        lo = Float64[]; hi = Float64[]
        for k in eachindex(v)
            a = (lb[k] - theta[k]) / v[k]; b = (ub[k] - theta[k]) / v[k]
            push!(lo, min(a, b)); push!(hi, max(a, b))
        end
        XLB = maximum(filter(!isnan, lo)); XUB = minimum(filter(!isnan, hi))
        W = W1 === nothing ? XUB - XLB : W1[it]
        fxold = logpost(theta)
        neval[it] += 1
        Z = fxold + log(rand(r))
        u = rand(r)
        L = max(XLB, xold - W * u)
        R = min(XUB, L + W)
        while L > XLB
            theta = base .+ L .* v
            fxl = logpost(theta)
            neval[it] += 1
            fxl <= Z && break
            L = max(XLB, L - W)
        end
        while R < XUB
            theta = base .+ R .* v
            fxr = logpost(theta)
            neval[it] += 1
            fxr <= Z && break
            R = min(XUB, R + W)
        end
        fxsim = Z - 1
        while fxsim < Z
            u = rand(r)
            xsim = L + u * (R - L)
            theta = base .+ xsim .* v
            fxsim = logpost(theta)
            neval[it] += 1
            xsim > xold ? (R = xsim) : (L = xsim)
        end
    end
    theta, fxsim, neval
end

"Directions of the multimodal rotated slice: unit vectors from the mode nearest `theta` to the other modes, in the order of `randperm(n-1)`."
function multimodal_directions(theta, modes, r)
    n = length(modes)
    dist = [sqrt(sum(abs2, theta .- m)) for m in modes]
    im = argmin(dist)
    cols = [(modes[j] .- modes[im]) ./ norm(modes[j] .- modes[im]) for j in 1:n if j != im]
    V1 = reduce(hcat, cols)
    perm = octave_randperm(r, n - 1)
    V1[:, perm]
end

"Octave's `randperm(n)` for an OctaveRNG; Julia's for other generators."
octave_randperm(r::OctaveRNG, n::Int) = Random.randperm(r, n)
octave_randperm(r, n::Int) = Random.randperm(r, n)

# ---------------------------------------------------------------------------------------------------- TaRB
"Random blocks of a TaRB iteration: `randperm(npar)` and a new block after each parameter with probability `new_block_probability`."
function tarb_blocks(npar::Int, p_new, r)
    indices = octave_randperm(r, npar)
    flags = npar > 1 ? _urand(r, npar - 1) .> (1 - p_new) : Bool[]
    blocks = vcat(1, 1 .+ cumsum(flags))
    [indices[blocks .== b] for b in 1:blocks[end]]
end

"""
One block of a TaRB-MH iteration (`posterior_sampler_iteration.m`): the block's conditional mode from `current` by csminwel, the
proposal from the inverse Hessian there, Chib and Ramamurthy's acceptance. `kernel(x)` is the log posterior kernel,
`objective(x)` its negative with Dynare's penalty convention `(fval, info, penalty)`.
"""
function tarb_block(est::Estimation, current0, last_posterior, idx, lb, ub, mlb, mub, jscale, opts, r, counter)
    current = Float64.(current0)
    nx = length(idx)
    start = current[idx]
    wrap(v) = (x = copy(current); x[idx] = v; x)
    opt = _parse_optim_opt(get(opts, "optim_opt", "") == "" ? est.options["optim_opt"] : opts["optim_opt"])
    Int(opts["mode_compute"]) == 4 || throw(EstimationError("TaRB with mode_compute=$(Int(opts["mode_compute"])): only 4 (csminwel, Dynare's default for TaRB) is available for the block mode search"))
    fcn = v -> (counter[1] += 1; rr = _dsge_likelihood(est, wrap(v); bounds=(mlb, mub)); (rr[1], rr[2], rr[3]))
    H0 = nothing
    fh, xh, _, _, iters, nfev, retcode, msg = csminwel(fcn, start; crit=Float64(get(opt, "TolFun", 1e-7)), nit=Int(get(opt, "MaxIter", 1000)),
        method=Int(get(opt, "NumgradAlgorithm", 2)), epsilon=Float64(get(opt, "NumgradEpsilon", 1e-6)), rng=r)
    xh === nothing && throw(EstimationError("TaRB: csminwel: bad initial parameter"))
    xopt = xh
    fneg(v) = (counter[1] += 1; -log_posterior(est, wrap(v); bounds=(mlb, mub)))
    hmat = reshape(hessian(fneg, xopt, est.options["gstep"]), nx, nx)
    if any(!isfinite, hmat)
        ih = Matrix{Float64}(I, nx, nx) .* 1e-4
    else
        ih = try
            inv(hmat)
        catch
            fill(Inf, nx, nx)
        end
        any(!isfinite, ih) && (ih = Matrix{Float64}(I, nx, nx) .* 1e-4)
    end
    ihs = (ih .+ ih') ./ 2 == ih ? ih : ih
    cu = try
        Matrix(cholesky(Hermitian(ihs, :U)).U)
    catch
        chol_SE(ihs)[1]
    end
    cu = cu * Diagonal(jscale[idx])
    student = opts["proposal_distribution"] == "rand_multivariate_student"
    n = student ? opts["student_degrees_of_freedom"] : nx
    prop = student ? rand_multivariate_student(r, xopt, cu, n) : rand_multivariate_normal(r, xopt, cu)
    lp = -Inf
    if all(prop .> lb[idx]) && all(prop .< ub[idx])
        counter[1] += 1
        lp = try
            log_posterior(est, wrap(prop); bounds=(mlb, mub))
        catch
            -Inf
        end
    end
    accepted = false
    if lp > -Inf
        fwd = student ? multivariate_student_pdf(prop, xopt, cu, n) : multivariate_normal_pdf(prop, xopt, cu, n)
        bwd = student ? multivariate_student_pdf(start, xopt, cu, n) : multivariate_normal_pdf(start, xopt, cu, n)
        accprob = lp - last_posterior + log(bwd) - log(fwd)
        if log(rand(r)) < accprob
            current[idx] = prop
            last_posterior = lp
            accepted = true
        end
    end
    current, last_posterior, accepted
end

function tarb_iteration(est::Estimation, theta, last_posterior, lb, ub, mlb, mub, jscale, opts, r)
    blocks = tarb_blocks(length(theta), opts["new_block_probability"], r)
    current = Float64.(theta)
    nacc = 0
    counter = [0]
    for idx in blocks
        current, last_posterior, a = tarb_block(est, current, last_posterior, idx, lb, ub, mlb, mub, jscale, opts, r, counter)
        nacc += a
    end
    current, last_posterior, nacc / length(blocks), counter[1]
end
