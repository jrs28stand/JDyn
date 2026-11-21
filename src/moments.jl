# Moments of a solved model at first order, with Dynare's definitions and defaults (port of the
# first-order parts of pydyn/moments.py, after Dynare's th_autocovariances.m, lyapunov_symm.m,
# compute_variance_decomposition.m, conditional_variance_decomposition.m and disp_moments.m).
#
# Conventions (as Dynare): shocks with a non-diagonal Sigma_e are orthogonalised with the lower Cholesky
# factor in declaration order; gamma_y[1] is the variance matrix, gamma_y[1+i] the autocorrelation of
# order i (entry (a, b) is corr(y_a,t, y_b,t-i)), gamma_y[ar+2] the variance decomposition (fractions).

"""
    lyapunov_symm(a, b; qz_criterium) -> (x, u)

Solve `x = a x a' + b` as Dynare's `lyapunov_symm.m` (Schur method). Eigenvalues of `a` with modulus above
`2 - qz_criterium` are projected out; `u` holds the Schur vectors of the removed roots.
"""
function lyapunov_symm(a::AbstractMatrix, b::AbstractMatrix; qz_criterium=QZ_CRITERIUM, complex_threshold=1e-15)
    n0 = size(a, 1)
    n0 == 0 && return zeros(0, 0), zeros(0, 0)
    n0 == 1 && return b ./ (1.0 .- a .* a), zeros(1, 0)
    thr = 2.0 - qz_criterium
    F = schur(Matrix{Float64}(a))
    sel = abs.(F.values) .> thr
    ordschur!(F, sel)
    k = count(sel)
    n = n0 - k
    U = F.Z
    u = U[:, 1:k]
    n == 0 && return zeros(n0, n0), u
    T = F.T[k+1:end, k+1:end]
    U2 = U[:, k+1:end]
    B = U2' * b * U2
    x = zeros(n, n)
    i = n
    while i >= 2
        if abs(T[i, i-1]) < complex_threshold
            if i == n
                c = zeros(i)
            else
                c = T[1:i, :] * (x[:, i+1:end] * T[i, i+1:end]) + T[i, i] * (T[1:i, i+1:end] * x[i+1:end, i])
            end
            q = Matrix{Float64}(I, i, i) - T[1:i, 1:i] * T[i, i]
            x[1:i, i] .= q \ (B[1:i, i] + c)
            x[i, 1:i-1] .= x[1:i-1, i]
            i -= 1
        else
            if i == n
                c = zeros(i); c1 = zeros(i)
            else
                c = T[1:i, :] * (x[:, i+1:end] * T[i, i+1:end]) + T[i, i] * (T[1:i, i+1:end] * x[i+1:end, i]) +
                    T[i, i-1] * (T[1:i, i+1:end] * x[i+1:end, i-1])
                c1 = T[1:i, :] * (x[:, i+1:end] * T[i-1, i+1:end]) + T[i-1, i-1] * (T[1:i, i+1:end] * x[i+1:end, i-1]) +
                     T[i-1, i] * (T[1:i, i+1:end] * x[i+1:end, i])
            end
            Ti = T[1:i, 1:i]
            Id = Matrix{Float64}(I, i, i)
            q = [Id-Ti*T[i, i] -Ti*T[i, i-1]; -Ti*T[i-1, i] Id-Ti*T[i-1, i-1]]
            z = q \ vcat(B[1:i, i] + c, B[1:i, i-1] + c1)
            x[1:i, i] .= z[1:i]
            x[1:i, i-1] .= z[i+1:end]
            x[i, 1:i-1] .= x[1:i-1, i]
            x[i-1, 1:i-2] .= x[1:i-2, i-1]
            i -= 2
        end
    end
    if i == 1
        c = dot(T[1, :], x[:, 2:end] * T[1, 2:end]) + T[1, 1] * dot(T[1, 2:end], x[2:end, 1])
        x[1, 1] = (B[1, 1] + c) / (1.0 - T[1, 1] * T[1, 1])
    end
    U2 * x * U2', u
end

"Lower Cholesky factor that ignores zero-variance shocks (get_lower_cholesky_covariance.m)."
function get_lower_cholesky_covariance(S::AbstractMatrix; add_tiny=1e-14)
    nz = findall(!=(0), sqrt.(diag(S)))
    out = zeros(size(S))
    sub = Symmetric(Matrix(S[nz, nz]))
    C = cholesky(sub; check=false)
    if !issuccess(C)
        C = cholesky(Symmetric(Matrix(sub) + add_tiny * I); check=true)
    end
    out[nz, nz] .= Matrix(C.L)
    out
end

# ------------------------------------------------------------------------------------ filters
"Dynare's sample_hp_filter.m: returns (trend, cycle) for a (T, n) array."
function hp_filter(y::AbstractMatrix, lamb=1600.0)
    T = size(y, 1); s = Float64(lamb)
    D = zeros(T, T)
    for t in 1:T
        D[t, t] = 1 + 6s
        t > 1 && (D[t, t-1] = -4s); t < T && (D[t, t+1] = -4s)
        t > 2 && (D[t, t-2] = s); t < T - 1 && (D[t, t+2] = s)
    end
    D[1, 1] = 1 + s; D[T, T] = 1 + s
    D[1, 2] = D[2, 1] = -2s; D[T-1, T] = D[T, T-1] = -2s
    D[2, 2] = 1 + 5s; D[T-1, T-1] = 1 + 5s
    trend = D \ y
    trend, y - trend
end

"Dynare's one_sided_hp_filter.m (Kalman filter of the local linear trend), default initialisation."
function one_sided_hp_filter(y::AbstractMatrix, lamb=1600.0)
    T, n = size(y)
    F = [2.0 -1.0; 1.0 0.0]; H = [1.0, 0.0]; Q = [1/lamb 0.0; 0.0 0.0]; R = 1.0
    trend = fill(NaN, T, n)
    for k in 1:n
        x = [2y[1, k] - y[2, k], 3y[1, k] - 2y[2, k]]
        P = 1e5 * Matrix{Float64}(I, 2, 2)
        for j in 1:T
            S = dot(H, P * H) + R
            K = (F * P * H) / S
            x = F * x + K * (y[j, k] - dot(H, x))
            Tm = F - K * H'
            P = Tm * P * Tm' + Q + R * (K * K')
            trend[j, k] = x[2]
        end
    end
    trend, y - trend
end

"Baxter-King filter as dseries' baxter_king_filter: returns the T - 2K filtered rows."
function baxter_king_filter(y::AbstractMatrix, high=6, low=32, K=12)
    hf, lf = 2pi / high, 2pi / low
    lp = 1:K
    w = zeros(K + 1)
    w[2:end] .= (sin.(lp .* hf) .- sin.(lp .* lf)) ./ (lp .* pi)
    w[1] = (hf - lf) / pi
    target = low > 1000 ? 1.0 : 0.0
    w .+= (target - (w[1] + 2 * sum(w[2:end]))) / (2K + 1)
    w = vcat(reverse(w[2:end]), w)
    T = size(y, 1)
    out = zeros(T - 2K, size(y, 2))
    for t in K+1:T-K
        out[t-K, :] .= vec(w' * y[t-K:t+K, :])
    end
    out
end

"""
Square root of variances: a negative variance (a root inside the QZ criterion but outside the unit circle makes the
Lyapunov solution indefinite; MATLAB would continue with complex numbers) gives NaN and one warning.
"""
function _sqrt_nan(v::AbstractVector)
    any(<(0), v) && @warn "negative variance in the theoretical moments (a root of the model lies outside the unit circle but inside qz_criterium): standard deviations and correlations of those variables are NaN"
    [x >= 0 || isnan(x) ? sqrt(x) : NaN for x in v]
end

# ------------------------------------------------------------------------------------ options
const _MOM_DEFAULTS = Dict{String,Any}("order" => 1, "ar" => 5, "hp_filter" => 0, "one_sided_hp_filter" => 0,
    "bandpass" => false, "passband" => (6, 32), "bandpass_K" => 12, "filtered_theoretical_moments_grid" => 512,
    "nodecomposition" => false, "nocorr" => false, "pruning" => false, "periods" => 0, "drop" => 100,
    "conditional_variance_decomposition" => Int[], "contemporaneous_correlation" => false,
    "qz_criterium" => QZ_CRITERIUM, "schur_vec_tol" => 1e-11, "lyapunov_complex_threshold" => 1e-15,
    "lyapunov_doubling_tol" => 1e-16,
    "add_tiny_number_to_cholesky" => 1e-14)
const _MOM_FROM_MOD = Dict("bandpass.indicator" => "bandpass", "bandpass.passband" => "passband")

"Dynare defaults, then the stoch_simul options of the .mod file, then `overrides`."
function _mom_options(m, overrides)
    o = copy(_MOM_DEFAULTS)
    if m !== nothing
        o["qz_criterium"] = qz_criterium(m)          # also from check / estimation / options_.qz_criterium
        for (k, v) in get(m.options, "stoch_simul", Dict{String,Any}())
            k = get(_MOM_FROM_MOD, k, k)
            haskey(o, k) && (o[k] = v)
        end
    end
    for (k, v) in overrides
        k = string(k)
        if k == "bandpass_filter"
            if v !== nothing && v !== false
                o["bandpass"] = true
                v !== true && (o["passband"] = Tuple(v))
            end
            continue
        end
        k == "steps" && (k = "conditional_variance_decomposition")
        haskey(o, k) || throw(ArgumentError("unknown option $k"))
        v === nothing || (o[k] = v)
    end
    cv = o["conditional_variance_decomposition"]
    o["conditional_variance_decomposition"] = cv isa Real ? [Int(cv)] : Int.(collect(cv))
    o["passband"] = Tuple(Float64.(collect(o["passband"])))
    o["hp_filter"] = Float64(o["hp_filter"])
    o["one_sided_hp_filter"] = Float64(o["one_sided_hp_filter"])
    o["ar"] = Int(o["ar"])
    o["order"] = Int(o["order"])
    o["bandpass"] = o["bandpass"] == true || o["bandpass"] == 1
    count([o["hp_filter"] != 0, o["one_sided_hp_filter"] != 0, o["bandpass"]]) > 1 &&
        throw(ArgumentError("can only use one of HP, one-sided HP, and bandpass filters"))
    o
end
_filter_name(o) = o["hp_filter"] != 0 ? "hp" : o["one_sided_hp_filter"] != 0 ? "one-sided hp" : o["bandpass"] ? "bandpass" : ""

function _default_var_list(m)
    for st in m.statements
        if get(st, "statementName", "") == "stoch_simul" && !isempty(get_or(st, "symbol_list", []))
            return String.(st["symbol_list"])
        end
    end
    orig_endo_names(m)
end
function _var_indices(m, var_list)
    var_list === nothing && (var_list = _default_var_list(m))
    idx = Int[]
    for v in var_list
        haskey(m.ie, v) || throw(ArgumentError("The following symbols are not endogenous variables: $v"))
        m.ie[v] in idx && (@warn "$v is specified twice in the variable list and is considered only once"; continue)
        push!(idx, m.ie[v])
    end
    idx
end

"Decision-rule arrays in Dynare's layout (rows in DR order)."
struct Rules
    ys::Vector{Float64}
    ghx::Matrix{Float64}
    ghu::Matrix{Float64}
    order_var::Vector{Int}
    state_var::Vector{Int}
    n::Int
    nspred::Int
    nexo::Int
    nstatic::Int
    inv_order_var::Vector{Int}
    index_states::Vector{Int}
    has2::Bool                       # second-order terms present
    ghxx::Matrix{Float64}
    ghxu::Matrix{Float64}
    ghuu::Matrix{Float64}
    ghs2::Vector{Float64}
end
function Rules(m, dr)
    dr = _as_dr(dr)
    ghx_ = Matrix{Float64}(dr["ghx"]); ghu_ = Matrix{Float64}(dr["ghu"])
    ov = Int.(dr["order_var"])
    _, _, cnt = dr_ordering(m.lead_lag_incidence)
    cnt.npred + cnt.nboth == size(ghx_, 2) || throw(ArgumentError("decision rule does not match the model"))
    inv = zeros(Int, length(ov)); inv[ov] .= 1:length(ov)
    has2 = all(k -> haskey(dr, k), ("ghxx", "ghxu", "ghuu", "ghs2"))
    z = zeros(0, 0)
    Rules(Float64.(dr["ys"]), ghx_, ghu_, ov, Int.(dr["state_var"]), length(ov), size(ghx_, 2), size(ghu_, 2),
          cnt.nstatic, inv, collect(cnt.nstatic+1:cnt.nstatic+size(ghx_, 2)), has2,
          has2 ? Matrix{Float64}(dr["ghxx"]) : z, has2 ? Matrix{Float64}(dr["ghxu"]) : z,
          has2 ? Matrix{Float64}(dr["ghuu"]) : z, has2 ? vec(Float64.(dr["ghs2"])) : Float64[])
end

function _measurement_errors(m)
    H = Float64[get(m.measurement_errors, v, 0.0) for v in m.varobs]
    vid = Int[m.ie[v] for v in m.varobs]
    H, vid
end

"Moments of a variable list, named as Dynare's oo_ fields where there is one."
Base.@kwdef mutable struct Moments
    kind::String                               # "theoretical" or "simulated"
    order::Int
    filter::String
    filter_param::Any
    var_names::Vector{String}
    var_index::Vector{Int}
    exo_names::Vector{String}
    stationary::Vector{Bool}
    ar::Int
    mean::Vector{Float64} = Float64[]
    std::Vector{Float64} = Float64[]
    variance::Vector{Float64} = Float64[]
    var::Matrix{Float64} = zeros(0, 0)
    corr::Union{Nothing,Matrix{Float64}} = nothing
    autocorr::Vector{Matrix{Float64}} = Matrix{Float64}[]
    gamma_y::Vector{Any} = Any[]
    skewness::Union{Nothing,Vector{Float64}} = nothing
    kurtosis::Union{Nothing,Vector{Float64}} = nothing
    variance_decomposition::Union{Nothing,Matrix{Float64}} = nothing        # percent, (nvar, nexo)
    variance_decomposition_me::Union{Nothing,Matrix{Float64}} = nothing     # percent, (nobs, nexo + 1)
    conditional_steps::Vector{Int} = Int[]
    conditional_variance_decomposition::Union{Nothing,Array{Float64,3}} = nothing
    conditional_variance_decomposition_me::Union{Nothing,Array{Float64,3}} = nothing
    me_var_names::Vector{String} = String[]
    pruning::Bool = false                                                   # pruned state space moments (order 2)
    mean_correction::Union{Nothing,Vector{Float64}} = nothing               # order 2 without pruning: mean minus steady state
end

function _intersect_stable(a, b)
    c = Int[]; ia = Int[]; ib = Int[]; seen = Set{Int}()
    for (i, v) in enumerate(a)
        v in seen && continue
        push!(seen, v)
        j = findfirst(==(v), b)
        if j !== nothing
            push!(c, v); push!(ia, i); push!(ib, j)
        end
    end
    c, ia, ib
end

function _compute_variance_decomposition(R::Rules, Sigma, o, A, aa, ghu_s, stationary, idx_stat, nvars, var_stationary)
    R.nexo == 1 && return ones(nvars, 1)
    cs = get_lower_cholesky_covariance(Sigma; add_tiny=o["add_tiny_number_to_cholesky"])
    b1 = ghu_s * cs
    b2 = R.ghu[idx_stat, :] * cs
    out = fill(NaN, nvars, R.nexo)
    total = zeros(length(stationary))
    for i in 1:R.nexo
        Vx, _ = lyapunov_symm(A, b1[:, i] * b1[:, i]'; qz_criterium=o["qz_criterium"], complex_threshold=o["lyapunov_complex_threshold"])
        vx2 = diag(aa * Vx * aa' + b2[:, i] * b2[:, i]')
        out[stationary, i] .= vx2
        total .+= vx2
    end
    rel = abs.(total .- var_stationary) ./ var_stationary
    if any(isfinite, rel) && maximum(filter(isfinite, rel)) > 1e-4 && maximum(abs.(total .- var_stationary)) > 1e-7
        @warn "Aggregate variance and sum of variances by shocks differ by more than 0.01 %"
    end
    out[stationary, :] ./= total
    out
end

"""
    conditional_variance_decomposition(m, dr, steps; var_list, Sigma_e, H, varobs_id) -> (cvd, cvd_me)

`cvd` has size (nvar, nsteps, nexo) in fractions; `cvd_me` adds the measurement error (last) for observed
variables, or is `nothing`.
"""
function conditional_variance_decomposition(m, dr, steps; var_list=nothing, Sigma_e=nothing, H=nothing, varobs_id=nothing)
    R = dr isa Rules ? dr : Rules(m, dr)
    ivar = (var_list !== nothing && !isempty(var_list) && var_list[1] isa Integer) ? Int.(var_list) : _var_indices(m, var_list)
    Sigma = Sigma_e === nothing ? m.Sigma_e : Sigma_e
    steps = Int.(collect(steps))
    any(<=(0), steps) && throw(ArgumentError("Conditional variance decomposition: All periods must be strictly positive"))
    any(i -> steps[i+1] <= steps[i], 1:length(steps)-1) && throw(ArgumentError("conditional variance decomposition: periods must be strictly increasing"))
    n, nexo = R.n, R.nexo
    trans = zeros(n, n)
    trans[:, R.index_states] .= R.ghx
    B = isdiag(Sigma) ? R.ghu .* sqrt.(diag(Sigma))' : R.ghu * Matrix(cholesky(Symmetric(Sigma)).L)
    cv = zeros(n, length(steps), nexo)
    for i in 1:nexo
        BB = B[:, i] * B[:, i]'
        V = zeros(n, n)
        j = 1
        for h in 1:steps[end]
            V = trans * V * trans' + BB
            if h == steps[j]
                cv[R.order_var, j, i] .= diag(V)
                j += 1
            end
        end
    end
    cv = cv[ivar, :, :]
    tot = dropdims(sum(cv; dims=3); dims=3)
    cvd = cv ./ tot
    cvd_me = nothing
    if H === nothing || varobs_id === nothing
        Hd, vid = _measurement_errors(m)
        H = H === nothing ? Hd : H
        varobs_id = varobs_id === nothing ? vid : varobs_id
    end
    if any(!=(0), H)
        obs_pos, isub, iobs = _intersect_stable(ivar, varobs_id)
        cvd_me = zeros(length(obs_pos), length(steps), nexo + 1)
        for i in 1:nexo
            cvd_me[:, :, i] .= cv[isub, :, i] ./ (tot[isub, :] .+ H[iobs])
        end
        cvd_me[:, :, nexo+1] .= 1.0 .- dropdims(sum(cvd_me[:, :, 1:nexo]; dims=3); dims=3)
    end
    cvd, cvd_me
end

"th_autocovariances.m at first order. Returns (gamma, stationary positions)."
function _th_autocovariances(R::Rules, Sigma, ivar, o)
    nar = o["ar"]
    filtered = o["hp_filter"] != 0 || o["bandpass"]
    local_order = o["order"]
    local_order >= 3 && throw(ArgumentError("Theoretical moments not implemented above 2nd order"))
    local_order > 1 && filtered && throw(ArgumentError("Theoretical filtered moments not implemented above 1st order"))
    nvar = length(ivar)
    ns = R.nspred
    ghx_, ghu_ = R.ghx, R.ghu
    inv = R.inv_order_var
    ghu_s = ghu_[R.index_states, :]
    A = ghx_[R.index_states, :]
    gamma = Vector{Any}(nothing, nar + 2)
    stationary = collect(1:nvar)
    idx_stat = inv[ivar]
    Vx = nothing
    if !filtered
        Vx, u = lyapunov_symm(A, ghu_s * Sigma * ghu_s'; qz_criterium=o["qz_criterium"], complex_threshold=o["lyapunov_complex_threshold"])
        if size(u, 2) > 0
            x = abs.(ghx_ * u)
            stationary = findall(vec(all(x[inv[ivar], :] .< o["schur_vec_tol"]; dims=2)))
            idx_stat = inv[ivar[stationary]]
        end
    end
    mean_corr = nothing
    if local_order == 2
        R.has2 || throw(ArgumentError("second-order decision rules (ghxx, ghxu, ghuu, ghs2) are needed"))
        sig = vec(Sigma)
        mean_corr = fill(NaN, nvar)
        if ns > 0
            ist = R.index_states
            Ex = (R.ghs2[ist] .+ R.ghxx[ist, :] * vec(Vx) .+ R.ghuu[ist, :] * sig) ./ 2.0
            Ex = try
                (Matrix{Float64}(I, ns, ns) - ghx_[ist, :]) \ Ex
            catch e
                e isa LinearAlgebra.SingularException ? fill(NaN, ns) : rethrow()   # unit root among the states
            end
            mean_corr[stationary] .= ghx_[idx_stat, :] * Ex .+ (R.ghs2[idx_stat] .+ R.ghxx[idx_stat, :] * vec(Vx) .+
                                                                R.ghuu[idx_stat, :] * sig) ./ 2.0
        else
            mean_corr[stationary] .= (R.ghs2[idx_stat] .+ R.ghuu[idx_stat, :] * sig) ./ 2.0
        end
    end
    aa = ghx_[idx_stat, :]
    bb = ghu_[idx_stat, :]
    if !filtered
        v = fill(NaN, nvar, nvar)
        v[stationary, stationary] .= aa * Vx * aa' + bb * Sigma * bb'
        v[abs.(v) .< 1e-12] .= 0.0
        gamma[1] = v
        if nar > 0
            vxy = A * Vx * aa' + ghu_s * Sigma * bb'
            sy = _sqrt_nan(diag(v))[stationary]
            sy = sy * sy'
            for i in 1:nar
                g = fill(NaN, nvar, nvar)
                g[stationary, stationary] .= (aa * vxy) ./ sy
                gamma[i+1] = g
                vxy = A * vxy
            end
        end
        if !(o["nodecomposition"] == true) && R.nexo > 0 && !isempty(stationary)
            gamma[nar+2] = _compute_variance_decomposition(R, Sigma, o, A, aa, ghu_s, stationary, idx_stat, nvar,
                                                           diag(v[stationary, stationary]))
        end
    else
        ngrid = Int(o["filtered_theoretical_moments_grid"])
        freqs = (0:ngrid-1) .* (2pi / ngrid)
        tneg = exp.(-im .* freqs)
        if o["bandpass"]
            gain = zeros(ngrid)
            low, high = o["passband"][2], max(2, o["passband"][1])
            gain[(freqs .>= 2pi / low) .& (freqs .<= 2pi / high)] .= 1.0
            gain[(freqs .<= -2pi / low + 2pi) .& (freqs .>= -2pi / high + 2pi)] .= 1.0
        else
            lam = o["hp_filter"]
            gain = 4lam .* (1 .- cos.(freqs)) .^ 2 ./ (1 .+ 4lam .* (1 .- cos.(freqs)) .^ 2)
        end
        ga = findall(!=(0), gain)
        IA = Matrix{Float64}(I, ns, ns); IE = Matrix{Float64}(I, R.nexo, R.nexo)
        function filtered_acov(Sig, b1, b2)
            f_hp = zeros(ComplexF64, ngrid, nvar, nvar)
            for g in ga
                X = ns > 0 ? (IA - A .* tneg[g]) \ b1 : zeros(ComplexF64, 0, R.nexo)
                top = vcat(X, IE)
                f_om = top * Sig * top' / (2pi)
                W = hcat(aa .* tneg[g], b2 === nothing ? bb : b2)
                f_hp[g, :, :] .= gain[g]^2 .* (W * f_om * W')
            end
            # real(ifft over frequencies) * 2 pi
            out = zeros(ngrid, nvar, nvar)
            for a in 1:nvar, b in 1:nvar
                col = f_hp[:, a, b]
                for t in 0:ngrid-1
                    s = zero(ComplexF64)
                    for k in 0:ngrid-1
                        s += col[k+1] * cis(2pi * k * t / ngrid)
                    end
                    out[t+1, a, b] = real(s) / ngrid * 2pi
                end
            end
            out
        end
        imv = filtered_acov(Sigma, ghu_s, nothing)
        gamma[1] = imv[1, :, :]
        if nar > 0
            sy = _sqrt_nan(diag(gamma[1])); sy = sy * sy'
            for i in 1:nar
                gamma[i+1] = imv[i+1, :, :] ./ sy
            end
        end
        if !(o["nodecomposition"] == true) && R.nexo > 0
            if R.nexo == 1
                gamma[nar+2] = ones(nvar, 1)
            else
                cs = get_lower_cholesky_covariance(Sigma; add_tiny=o["add_tiny_number_to_cholesky"])
                vv = diag(filtered_acov(cs * cs', ghu_s, ghu_[idx_stat, :])[1, :, :])
                vd = zeros(nvar, R.nexo)
                for i in 1:R.nexo
                    ii = filtered_acov(cs[:, i] * cs[:, i]', ghu_s, ghu_[idx_stat, :])
                    vd[:, i] .= abs.(diag(ii[1, :, :])) ./ vv
                end
                gamma[nar+2] = vd
            end
        end
    end
    mean_corr === nothing || push!(gamma, mean_corr)          # gamma[nar+3], as Dynare's Gamma_y{nar+3}
    gamma, stationary
end

"Doubling algorithm of Dynare's disclyap_fast MEX file: (x, errorflag)."
function disclyap_fast(g, v; tol=1e-16, max_iter=2000)
    P0, A0 = Matrix{Float64}(v), Matrix{Float64}(g)
    matd, it = 1.0, 1
    while matd > tol && it < max_iter
        P1 = P0 + A0 * P0 * A0'
        A1 = A0 * A0
        matd = maximum(abs, P1 - P0)
        P0, A0 = P1, A1
        it += 1
    end
    (it == max_iter || isnan(matd)) && return fill(NaN, size(P0)), true
    (P0 + P0') ./ 2, false
end

"K with K kron(a, b) = kron(b, a) for a, b in R^n."
function _commutation(n)
    K = zeros(n * n, n * n)
    for i in 0:n-1, j in 0:n-1
        K[j*n+i+1, i*n+j+1] = 1.0
    end
    K
end

"""
Order-2 pruned-state-space moments (Dynare's pruned_state_space_system.m with order 2; port of pydyn's
`_pruned_order2`). Returns `(E_y, Var_y, Corr_y, Corr_yi)` for the variables `ivar`.
"""
function _pruned_order2(R::Rules, Sigma, ivar, o)
    nar = o["ar"]
    R.has2 || throw(ArgumentError("second-order decision rules (ghxx, ghxu, ghuu, ghs2) are needed"))
    x, u = R.nspred, R.nexo
    indx = R.index_states
    indy = R.inv_order_var[ivar]
    Yss = R.ys[R.order_var]
    hx, gx = R.ghx[indx, :], R.ghx[indy, :]
    hu, gu = R.ghu[indx, :], R.ghu[indy, :]
    hxx, gxx = R.ghxx[indx, :], R.ghxx[indy, :]
    hxu, gxu = R.ghxu[indx, :], R.ghxu[indy, :]
    huu, guu = R.ghuu[indx, :], R.ghuu[indy, :]
    hss, gss = R.ghs2[indx], R.ghs2[indy]
    E_uu = Matrix{Float64}(Sigma)
    Var_z1, Schur_u = lyapunov_symm(hx, hu * E_uu * hu'; qz_criterium=o["qz_criterium"], complex_threshold=o["lyapunov_complex_threshold"])
    size(Schur_u, 2) > 0 && error("pruned_state_space_system: the pruned state space currently does not support unit roots at higher order.")
    E_xfxf = Var_z1
    vS = vec(E_uu)
    hx_hx, hx_hu, hu_hu = kron(hx, hx), kron(hx, hu), kron(hu, hu)
    I_xx = Matrix{Float64}(I, x * x, x * x)
    K_xx = _commutation(x)
    invIx_hx = inv(Matrix{Float64}(I, x, x) - hx)
    # fourth moments of the Gaussian shocks (Isserlis); rows/columns indexed like kron(u, u)
    V22 = zeros(u * u, u * u)
    for i in 1:u, j in 1:u, k in 1:u, l in 1:u
        V22[(i-1)*u+j, (k-1)*u+l] = E_uu[i, j] * E_uu[k, l] + E_uu[i, k] * E_uu[j, l] + E_uu[i, l] * E_uu[j, k]
    end
    V22 .-= vS * vS'
    E_xfxf_uu = kron(E_xfxf, E_uu')
    nz = 2x + x * x
    ni = u + u * u + x * u
    id1, id2, id3 = 1:x, x+1:2x, 2x+1:nz
    j1, j2, j3 = 1:u, u+1:u+u*u, u+u*u+1:ni
    A = zeros(nz, nz)
    A[id1, id1] = hx
    A[id2, id2] = hx
    A[id2, id3] = 0.5 .* hxx
    A[id3, id3] = hx_hx
    B = zeros(nz, ni)
    B[id1, j1] = hu
    B[id2, j2] = 0.5 .* huu
    B[id2, j3] = hxu
    B[id3, j2] = hu_hu
    B[id3, j3] = (I_xx + K_xx) * hx_hu
    c2 = 0.5 .* hss .+ 0.5 .* (huu * vS)
    C = zeros(length(indy), nz)
    C[:, id1] = gx
    C[:, id2] = gx
    C[:, id3] = 0.5 .* gxx
    D = zeros(length(indy), ni)
    D[:, j1] = gu
    D[:, j2] = 0.5 .* guu
    D[:, j3] = gxu
    d = 0.5 .* gss .+ 0.5 .* (guu * vS)
    Varinov = zeros(ni, ni)
    Varinov[j1, j1] = E_uu
    Varinov[j2, j2] = V22
    Varinov[j3, j3] = E_xfxf_uu
    E_xs = invIx_hx * (0.5 .* (hxx * vec(E_xfxf)) .+ c2)
    Om_z = B * Varinov * B'
    Var_z, err = disclyap_fast(A, Om_z; tol=o["lyapunov_doubling_tol"])
    if err
        println("PRUNED_STATE_SPACE_SYSTEM: error flag in disclyap_fast at order=2, use lyapunov_symm")
        Var_z, _ = lyapunov_symm(A, Om_z; qz_criterium=1.0, complex_threshold=o["lyapunov_complex_threshold"])
    end
    Var_z[id1, id2] .= 0; Var_z[id1, id3] .= 0; Var_z[id2, id1] .= 0; Var_z[id3, id1] .= 0
    Var_y = C * Var_z * C' + D * Varinov * D'
    Var_y[abs.(Var_y) .< 1e-12] .= 0.0
    sd = _sqrt_nan(diag(Var_y))
    sy = sd * sd'
    Corr_y = Var_y ./ sy
    tmp = A * Var_z * C' + B * Varinov * D'
    Ai = Matrix{Float64}(I, nz, nz)
    Corr_yi = Matrix{Float64}[]
    for _ in 1:nar
        push!(Corr_yi, (C * Ai * tmp) ./ sy)
        Ai = Ai * A
    end
    E_z = vcat(zeros(x), E_xs, vec(E_xfxf))
    E_y = Yss[indy] .+ C * E_z .+ d
    E_y, Var_y, Corr_y, Corr_yi
end

function _me_vd(vd_percent, var, isub, iobs, Hd, nexo)
    v = diag(var)[isub]
    out = vd_percent[isub, :] .* (v ./ (v .+ Hd[iobs]))
    hcat(out, 100.0 .- sum(out; dims=2))
end

"""
    theoretical_moments(m, dr=nothing; var_list=nothing, Sigma_e=nothing, H=nothing, options...) -> Moments

Theoretical moments (order 1; order 2 plain or pruned) as Dynare's `disp_th_moments` computes them, with the options of the
`.mod` file's `stoch_simul` (`ar`, `hp_filter`, `bandpass_filter`, `nodecomposition`, `nocorr`,
`conditional_variance_decomposition`, ...) unless overridden by keywords.
"""
function theoretical_moments(m::Model, dr=nothing; var_list=nothing, Sigma_e=nothing, H=nothing, options...)
    o = _mom_options(m, Dict(string(k) => v for (k, v) in options))
    o["one_sided_hp_filter"] != 0 && throw(ArgumentError("theoretical moments incompatible with one-sided HP filter. Use simulated moments instead"))
    order = o["order"]
    order >= 3 && throw(ArgumentError("Theoretical moments are implemented up to second order"))
    if order == 2 && m.max_deriv_order < 2
        @warn "using order = 1 because Hessian is equal to zero"
        order = o["order"] = 1
    end
    dr === nothing && (dr = solve_dr(m, order))
    R = Rules(m, dr)
    Sigma = Sigma_e === nothing ? m.Sigma_e : Sigma_e
    ivar = _var_indices(m, var_list)
    nvar = length(ivar)
    nar = o["ar"]
    steps = o["conditional_variance_decomposition"]
    Hd, vid = _measurement_errors(m)
    H === nothing || (Hd = Float64.(H))
    me_present, obs_pos, isub, iobs = false, Int[], Int[], Int[]
    if any(!=(0), Hd)
        obs_pos, isub, iobs = _intersect_stable(ivar, vid)
        me_present = !isempty(obs_pos)
    end
    pruned = order == 2 && o["pruning"] == true
    pruned && (o["hp_filter"] != 0 || o["bandpass"]) &&
        throw(ArgumentError("theoretical moments incompatible with filtering at order 2. Use simulated moments instead"))
    res = Moments(kind=pruned ? "pruned" : "theoretical", order=order, pruning=pruned, filter=_filter_name(o),
                  filter_param=o["hp_filter"] != 0 ? o["hp_filter"] : o["bandpass"] ? o["passband"] : nothing,
                  var_names=[m.endo_names[i] for i in ivar], var_index=ivar, exo_names=copy(m.exo_names),
                  stationary=trues(nvar), ar=nar)
    if pruned
        E_y, Var_y, Corr_y, Corr_yi = _pruned_order2(R, Sigma, ivar, o)
        s2 = diag(Var_y)
        sd = _sqrt_nan(s2)
        res.mean, res.var, res.variance, res.std = E_y, Var_y, s2, sd
        res.gamma_y = Any[Var_y, Corr_yi...]
        i1 = findall(x -> abs(x) > 1e-12, s2)
        if !isempty(i1)
            o["nocorr"] == true || (res.corr = Corr_y)
            res.autocorr = Corr_yi
        end
        if !(o["nodecomposition"] == true) && R.nexo > 1
            ghu_s = R.ghu[R.index_states, :]
            A = R.ghx[R.index_states, :]
            idx = R.inv_order_var[ivar]
            vd = _compute_variance_decomposition(R, Sigma, o, A, R.ghx[idx, :], ghu_s, collect(1:nvar), idx, nvar, s2)
            push!(res.gamma_y, vd)
            res.variance_decomposition = 100 .* vd
            me_present && (res.variance_decomposition_me = _me_vd(res.variance_decomposition, res.var, isub, iobs, Hd, R.nexo))
            if !isempty(steps)
                res.conditional_steps = steps
                res.conditional_variance_decomposition, res.conditional_variance_decomposition_me =
                    conditional_variance_decomposition(m, R, steps; var_list=ivar, Sigma_e=Sigma, H=Hd, varobs_id=vid)
            end
        end
    else
        gamma, stat = _th_autocovariances(R, Sigma, ivar, o)
        res.stationary = [i in stat for i in 1:nvar]
        res.gamma_y = gamma
        mean = R.ys[ivar]
        mean[.!res.stationary] .= NaN
        var = gamma[1]
        s2 = diag(var)
        sd = [x >= 0 ? sqrt(x) : NaN for x in s2]
        if order == 2
            res.mean_correction = gamma[nar+3]
            mean = mean .+ gamma[nar+3]
        end
        res.mean, res.var, res.variance, res.std = mean, var, s2, sd
        if !isempty(stat) && !(o["nodecomposition"] == true)
            res.variance_decomposition = 100 .* gamma[nar+2]
            me_present && (res.variance_decomposition_me = _me_vd(res.variance_decomposition, var, isub, iobs, Hd, R.nexo))
        end
        if !isempty(steps)
            res.conditional_steps = steps
            res.conditional_variance_decomposition, res.conditional_variance_decomposition_me =
                conditional_variance_decomposition(m, R, steps; var_list=ivar, Sigma_e=Sigma, H=Hd, varobs_id=vid)
        end
        i1 = findall(x -> abs(x) > 1e-12, s2)
        if !isempty(i1)
            if !(o["nocorr"] == true) && !isempty(stat)
                corr = fill(NaN, size(var))
                corr[i1, i1] .= var[i1, i1] ./ (sd[i1] * sd[i1]')
                res.corr = corr
            end
            (nar > 0 && !isempty(stat)) && (res.autocorr = [gamma[i+1] for i in 1:nar])
        end
    end
    me_present && (res.me_var_names = [m.endo_names[i] for i in obs_pos])
    res
end

"Decision rules at `order` (1 to 3) as a Dict: `first_order` at order 1, Dynare's k-order library above."
solve_dr(m::Model, order::Integer) = order == 1 ? as_dr(first_order(m)) : k_order(m; order=order)

"get_filtered_time_series.m, plus the removal of the band-pass edge rows."
function get_filtered_time_series(y, mean, o)
    o["hp_filter"] != 0 && return hp_filter(y, o["hp_filter"])[2]
    o["one_sided_hp_filter"] != 0 && return one_sided_hp_filter(y, o["one_sided_hp_filter"])[2]
    o["bandpass"] && return baxter_king_filter(y, o["passband"][1], o["passband"][2], o["bandpass_K"])
    y .- mean'
end

"""
    simulated_moments(y; ivar, model, drop, var_names, options...) -> Moments

Moments of a simulated path as Dynare's `disp_moments.m` (`drop` rows discarded first).
"""
function simulated_moments(y::AbstractMatrix; ivar=nothing, model=nothing, drop=nothing, var_names=nothing, shocks=nothing, dr=nothing, options...)
    o = _mom_options(model, Dict(string(k) => v for (k, v) in options))
    drop = drop === nothing ? Int(o["drop"]) : drop
    ivar = ivar === nothing ? collect(1:size(y, 2)) : ivar
    names = var_names !== nothing ? collect(var_names) : model !== nothing ? [model.endo_names[i] for i in ivar] : ["v$i" for i in ivar]
    ys = y[drop+1:end, ivar]
    nvar = length(ivar)
    mu = vec(sum(ys; dims=1)) ./ size(ys, 1)
    yf = get_filtered_time_series(ys, mu, o)
    T = size(yf, 1)
    s2 = vec(sum(yf .* yf; dims=1)) ./ T
    s = sqrt.(s2)
    var = yf' * yf / T
    skew = vec(sum(yf .^ 3; dims=1)) ./ T ./ s2 .^ 1.5
    kurt = vec(sum(yf .^ 4; dims=1)) ./ T ./ (s2 .* s2) .- 3.0
    zero_ = findall(<(1e-10), abs.(s))
    skew[zero_] .= NaN; kurt[zero_] .= NaN; s[zero_] .= 0; s2[zero_] .= 0
    var[zero_, :] .= 0; var[:, zero_] .= 0
    corr = nothing
    if !(o["nocorr"] == true)
        corr = (yf' * yf / T) ./ (s * s')
        corr[zero_, :] .= NaN; corr[:, zero_] .= NaN
    end
    ar = o["ar"]
    ac = Matrix{Float64}[]
    _std1(x) = sqrt.(vec(sum((x .- sum(x; dims=1) ./ size(x, 1)) .^ 2; dims=1)) ./ (size(x, 1) - 1))
    for i in 1:ar
        a1, a0 = yf[ar+1:end, :], yf[ar-i+1:T-i, :]
        c = a1' * a0 ./ ((T - ar) .* (_std1(a1) * _std1(a0)'))
        c[zero_, :] .= NaN; c[:, zero_] .= NaN
        push!(ac, c)
    end
    res = Moments(kind="simulated", order=o["order"], filter=_filter_name(o),
                  filter_param=o["hp_filter"] != 0 ? o["hp_filter"] : o["one_sided_hp_filter"] != 0 ? o["one_sided_hp_filter"] : o["bandpass"] ? o["passband"] : nothing,
                  var_names=names, var_index=ivar, exo_names=model !== nothing ? copy(model.exo_names) : String[],
                  stationary=trues(nvar), ar=ar, mean=mu, std=s, variance=s2, var=var, corr=corr, autocorr=ac,
                  skewness=skew, kurtosis=kurt)
    if !(o["nodecomposition"] == true) && shocks !== nothing && model !== nothing
        res.variance_decomposition = _simulated_vd(model, dr, shocks, ivar, drop, o, s2, zero_)
    end
    res
end

function _simulated_vd(m, dr, shocks, ivar, drop, o, s2, zero_idx)
    Sigma = m.Sigma_e
    nexo = size(Sigma, 1)
    nvar = length(ivar)
    nexo == 1 && return 100.0 .* ones(nvar, 1)
    iv = findall(!=(0), diag(Sigma))
    chol_S = Matrix(cholesky(Symmetric(Sigma[iv, iv])).U)
    shock_mat = zeros(size(shocks))
    shock_mat[:, iv] .= shocks[:, iv] / chol_S
    out = zeros(nvar, nexo)
    for j in iv
        tmp = zeros(size(shocks))
        tmp[:, j] .= shock_mat[:, j]
        tmp[:, iv] .= tmp[:, iv] * chol_S
        yo = simulate(_as_dr(dr), tmp; order=o["order"], pruning=o["pruning"] == true)[drop+1:end, ivar]
        yo = get_filtered_time_series(yo, vec(sum(yo; dims=1)) ./ size(yo, 1), o)
        mu = sum(yo; dims=1) ./ size(yo, 1)
        out[:, j] .= vec(sum((yo .- mu) .^ 2; dims=1)) ./ (size(yo, 1) - 1) ./ s2 .* 100.0
    end
    out[zero_idx, :] .= NaN
    out
end

"""
    stoch_simul_moments(m; var_list=nothing, dr=nothing, shocks=nothing, rng=nothing, options...) -> Moments

What `stoch_simul` reports at first order for `var_list`: theoretical moments if `periods == 0` (the default),
otherwise moments of a simulation of `periods` periods (shocks drawn as `randn(periods, n) * chol(Sigma_e)'`
unless `shocks` is given).
"""
# stoch_simul options that do not change the moments (IRFs, display, solver choices checked by first_order)
const _MOM_NEUTRAL = Set(["irf", "irf_shocks", "relative_irf", "irf_in_percent", "analytical_girf", "emas_girf",
    "emas_drop", "emas_tolf", "emas_max_iter", "stderr_multiples", "diagonal_only", "replic", "nomodelsummary", "print",
    "dr_display_tol", "loglinear", "k_order_solver", "partial_information", "aim_solver", "dr_cycle_reduction",
    "dr_logarithmic_reduction", "dr_cycle_reduction_tol", "dr_cycle_reduction_maxiter", "dr_logarithmic_reduction_tol",
    "dr_logarithmic_reduction_maxiter", "qz_zero_threshold", "solve_algo", "logdata"])

function stoch_simul_moments(m::Model; var_list=nothing, dr=nothing, shocks=nothing, rng=nothing, H=nothing, options...)
    warn_unused(m, "stoch_simul", union(Set(keys(_MOM_DEFAULTS)), Set(keys(_MOM_FROM_MOD)), _MOM_NEUTRAL), "stoch_simul_moments")
    o = _mom_options(m, Dict(string(k) => v for (k, v) in options))
    o["periods"] == 0 && return theoretical_moments(m, dr; var_list=var_list, H=H, options...)
    order = o["order"]
    dr === nothing && (dr = solve_dr(m, order))
    ivar = _var_indices(m, var_list)
    periods = Int(o["periods"])
    periods <= o["drop"] && throw(ArgumentError("STOCH_SIMUL error: The horizon of simulation is shorter than the number of observations to be dropped. Either increase periods or decrease drop."))
    Sigma = m.Sigma_e
    if shocks === nothing
        rng = rng === nothing ? Random.default_rng() : rng
        iv = findall(!=(0), diag(Sigma))
        shocks = zeros(periods, size(Sigma, 1))
        shocks[:, iv] .= randn(rng, periods, length(iv)) * Matrix(cholesky(Symmetric(Sigma[iv, iv])).L)'
    end
    y = simulate(_as_dr(dr), shocks; order=order, pruning=o["pruning"] == true)
    keys_ = Dict(k => v for (k, v) in options if haskey(_MOM_DEFAULTS, string(k)) || string(k) in ("bandpass_filter", "steps"))
    simulated_moments(y; ivar=ivar, shocks=shocks, dr=dr, model=m, keys_...)
end
