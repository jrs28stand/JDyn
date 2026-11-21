# First-order perturbation solution (port of pydyn/perturbation.py, after Dynare's dyn_first_order_solver.m).

struct BlanchardKahnError <: Exception
    msg::String
end
Base.showerror(io::IO, e::BlanchardKahnError) = print(io, "BlanchardKahnError: ", e.msg)
struct LoglinearError <: Exception
    msg::String
end
Base.showerror(io::IO, e::LoglinearError) = print(io, "LoglinearError: ", e.msg)

"Dynare's default: generalised eigenvalues below this in modulus count as stable."
const QZ_CRITERIUM = 1.0 + 1e-6
"Dynare's options_.qz_zero_threshold."
const QZ_ZERO_THRESHOLD = 1e-6

"""
    FirstOrderSolution

`y_t - ys = G (y_{t-1} - ys) + H u_t`, variables in declaration order. `ghx(sol)` / `ghu(sol)` reproduce
Dynare's `oo_.dr.ghx` / `oo_.dr.ghu` (rows in decision-rule order `order_var`, `ghx` columns restricted
to the state variables `state_var`); all indices are 1-based.
"""
struct FirstOrderSolution
    ys::Vector{Float64}
    G::Matrix{Float64}
    H::Matrix{Float64}
    order_var::Vector{Int}
    state_var::Vector{Int}
    eigenvalues::Vector{ComplexF64}
    jacobian::Matrix{Float64}
    params::Vector{Float64}
    loglinear::Bool
    steady_state::Vector{Float64}       # the steady state in levels
end
ghx(s::FirstOrderSolution) = s.G[s.order_var, s.state_var]
ghu(s::FirstOrderSolution) = s.H[s.order_var, :]

"Decision rule as a Dict in the form `simulate` / `irf` take (Dynare's oo_.dr fields)."
as_dr(s::FirstOrderSolution) = Dict{String,Any}("ys" => s.ys, "order_var" => s.order_var, "state_var" => s.state_var,
                                                "ghx" => ghx(s), "ghu" => ghu(s), "order" => 1)

"Deviations from the steady state after a one-off shock vector at t=1; size (periods, n)."
function irf(s::FirstOrderSolution, shock::AbstractVector; periods::Int=40)
    out = zeros(periods, size(s.G, 1))
    y = s.H * Float64.(shock)
    for t in 1:periods
        out[t, :] .= y
        y = s.G * y
    end
    out
end

"""
    simulate(sol::FirstOrderSolution, shocks; y0=nothing) -> (T, n) levels

Simulate for a (T, n_exo) shock array from `y0` (levels, default the steady state). Like Dynare's
`simult_` (and `simulate(dr, ...)` for a decision-rule Dict), the result is in levels (in logs for a
`loglinear` solution); subtract `sol.ys'` for deviations.
"""
simulate(s::FirstOrderSolution, shocks::AbstractMatrix; y0=nothing) = simulate(as_dr(s), shocks; order=1, y0=y0)

"Dynare's decision-rule ordering: static, purely backward, mixed, purely forward."
function dr_ordering(lli::AbstractMatrix)
    lag = lli[:, 1] .> 0
    lead = lli[:, 3] .> 0
    idx = collect(1:length(lag))
    static, pred, both, fwrd = idx[.!lag .& .!lead], idx[lag .& .!lead], idx[lag .& lead], idx[.!lag .& lead]
    vcat(static, pred, both, fwrd), vcat(pred, both),
        (nstatic=length(static), npred=length(pred), nboth=length(both), nfwrd=length(fwrd))
end

# ---------------------------------------------------------------------- ordered QZ
# Dynare's mjdgges: LAPACK dgges with a selection function, so that the decomposition and the ordering
# are one call (Julia's schur + ordschur! is dgges3 followed by dtgsen, 20-40% slower on these sizes).
# The criterion is read from a per-thread slot set just before the ccall (no task switch can happen
# between the two).
const _QZ_CRIT = fill(QZ_CRITERIUM, 1)
const _BlasInt = LinearAlgebra.BlasInt
function _qz_select(ar::Ptr{Float64}, ai::Ptr{Float64}, b::Ptr{Float64})::LinearAlgebra.BlasInt
    c = @inbounds _QZ_CRIT[Threads.threadid()]
    re, im, bb = unsafe_load(ar), unsafe_load(ai), unsafe_load(b)
    LinearAlgebra.BlasInt(re * re + im * im < c * c * bb * bb)
end

"""
    _ordered_qz(E, D, qz_criterium) -> (S, T, Z, alpha, beta, sdim)

Real generalised Schur form of the pencil (E, D) with the generalised eigenvalues alpha/beta of modulus
below `qz_criterium` first. Falls back to `schur` + `ordschur!` when LAPACK reports that rounding made
the reordering ambiguous (info n+2 / n+3).
"""
function _ordered_qz(E::Matrix{Float64}, D::Matrix{Float64}, qz_criterium::Real)
    n = size(E, 1)
    A, B = copy(E), copy(D)
    sdim = Ref{_BlasInt}(0); info = Ref{_BlasInt}(0)
    alphar, alphai, beta = zeros(n), zeros(n), zeros(n)
    vsl = zeros(1, 1); vsr = zeros(n, n)
    lwork = 8n + 16 + 32n
    work = zeros(lwork); bwork = zeros(_BlasInt, n)
    sel = @cfunction(_qz_select, _BlasInt, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}))
    tid = Threads.threadid()
    length(_QZ_CRIT) < tid && append!(_QZ_CRIT, fill(QZ_CRITERIUM, tid - length(_QZ_CRIT)))
    _QZ_CRIT[tid] = Float64(qz_criterium)
    ccall((LinearAlgebra.BLAS.@blasfunc(dgges_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{_BlasInt}, Ptr{Float64}, Ref{_BlasInt},
           Ptr{Float64}, Ref{_BlasInt}, Ref{_BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
           Ref{_BlasInt}, Ptr{Float64}, Ref{_BlasInt}, Ptr{Float64}, Ref{_BlasInt}, Ptr{_BlasInt}, Ref{_BlasInt},
           Clong, Clong, Clong),
          'N', 'V', 'S', sel, n, A, max(1, n), B, max(1, n), sdim, alphar, alphai, beta, vsl, 1, vsr, max(1, n),
          work, lwork, bwork, info, 1, 1, 1)
    if info[] == 0
        return A, B, vsr, complex.(alphar, alphai), beta, Int(sdim[])
    elseif info[] < 0 || info[] <= n
        throw(BlanchardKahnError("generalised Schur decomposition failed (LAPACK dgges info=$(info[]))"))
    end
    GS = schur(E, D)
    select = abs.(GS.α) .< qz_criterium .* abs.(GS.β)
    ordschur!(GS, select)
    GS.S, GS.T, GS.Z, Vector{ComplexF64}(GS.α), Vector{Float64}(real.(GS.β)), count(select)
end

"""
    solve_linear_re(fm, f0, fp, fu; qz_criterium, lag, lead) -> (G, H, eigenvalues)

Solve `fp E y(+1) + f0 y + fm y(-1) + fu u = 0` for `y = G y(-1) + H u` (Dynare's algorithm: static
variables removed by QR, ordered QZ on the dynamic block).
"""
function solve_linear_re(fm, f0, fp, fu; qz_criterium=QZ_CRITERIUM, lag=nothing, lead=nothing,
                         qz_zero_threshold=QZ_ZERO_THRESHOLD)
    lag = lag === nothing ? vec(any(fm .!= 0; dims=1)) : lag
    lead = lead === nothing ? vec(any(fp .!= 0; dims=1)) : lead
    n = size(f0, 1)
    idx = collect(1:n)
    static, pred, both, fwrd = idx[.!lag .& .!lead], idx[lag .& .!lead], idx[lag .& lead], idx[.!lag .& lead]
    ov = vcat(static, pred, both, fwrd)
    ns, npred, nboth, nfwrd = length(static), length(pred), length(both), length(fwrd)
    nspred, nsfwrd = npred + nboth, nboth + nfwrd
    ndyn = n - ns
    nd = nspred + nsfwrd
    sv = ov[ns+1:ns+nspred]
    jv = ov[ns+npred+1:end]
    k = size(fu, 2)
    # [A | B | C | fu]: lagged states, all current variables, led jumpers, shocks (DR order)
    aa = hcat(fm[:, sv], f0[:, ov], fp[:, jv], fu)
    all(isfinite, aa) || throw(BlanchardKahnError("the Jacobian of the dynamic model contains NaN or Inf"))
    if ns > 0
        F = qr(aa[:, nspred+1:nspred+ns])
        aa = F.Q' * aa
    end
    A = aa[:, 1:nspred]
    B = aa[:, nspred+1:nspred+n]
    C = aa[:, nspred+n+1:nspred+n+nsfwrd]
    FU = aa[:, nspred+n+nsfwrd+1:end]
    local S, T, w, eig, sdim
    if nd > 0
        D = zeros(nd, nd); E = zeros(nd, nd)
        D[1:ndyn, 1:npred] .= B[ns+1:end, ns+1:ns+npred]
        D[1:ndyn, nspred+1:end] .= C[ns+1:end, :]
        E[1:ndyn, 1:nspred] .= -A[ns+1:end, :]
        E[1:ndyn, nspred+1:end] .= -B[ns+1:end, ns+npred+1:end]
        for r in 1:nboth
            D[ndyn+r, npred+r] = 1.0
            E[ndyn+r, nspred+r] = 1.0
        end
        S, T, w, alpha, beta, sdim = _ordered_qz(E, D, qz_criterium)
        if any((abs.(alpha) .<= qz_zero_threshold) .& (abs.(beta) .<= qz_zero_threshold))
            throw(BlanchardKahnError("one of the generalised eigenvalues is close to 0/0 (the model has a singularity: Dynare's error 7)"))
        end
        eig = alpha ./ beta
    else
        S = T = w = zeros(0, 0)
        eig = ComplexF64[]
        sdim = 0
    end
    if sdim != nspred
        kind = sdim > nspred ? "indeterminacy" : "no stable solution"
        throw(BlanchardKahnError("Blanchard-Kahn conditions not satisfied ($kind): $(nd - sdim) unstable roots for $nsfwrd forward-looking variables"))
    end
    w11 = w[1:nspred, 1:nspred]; w12 = w[1:nspred, nspred+1:end]; w22 = w[nspred+1:end, nspred+1:end]
    if nsfwrd > 0
        Fw = lu(permutedims(w22); check=false)
        rc = issuccess(Fw) ? LinearAlgebra.LAPACK.gecon!('1', copy(Fw.factors), opnorm(permutedims(w22), 1)) : 0.0
        rc < 1e-9 && throw(BlanchardKahnError("rank condition not satisfied"))
        gx = -(Fw \ permutedims(w12))                 # jumpers on states
    else
        gx = zeros(0, nspred)
    end
    if nspred > 0
        # hx = w11 T11^-1 S11 w11^-1: transition matrix of the states
        hx = (w11 / UpperTriangular(T[1:nspred, 1:nspred])) * (S[1:nspred, 1:nspred] / w11)
    else
        hx = zeros(0, 0)
    end
    ghx_ = zeros(n, nspred)
    ghx_[ns+1:ns+nspred, :] .= hx
    ghx_[ns+nspred+1:end, :] .= gx[nboth+1:end, :]
    Cgx = C * gx
    if ns > 0
        rhs = -(Cgx[1:ns, :] * hx) - A[1:ns, :] - B[1:ns, ns+1:end] * ghx_[ns+1:end, :]
        ghx_[1:ns, :] .= UpperTriangular(B[1:ns, 1:ns]) \ rhs
    end
    A_ = copy(B)
    A_[:, ns+1:ns+nspred] .+= Cgx
    G = zeros(n, n)
    H = zeros(n, k)
    G[ov, sv] .= ghx_
    k > 0 && (H[ov, :] .= -(A_ \ FU))
    G, H, Vector{ComplexF64}(eig)
end

"""Variables Dynare takes logs of under `loglinear`: every endogenous variable except the auxiliary
variables for leads and lags of exogenous variables, which stay in levels."""
function log_mask(m::Model)
    mask = trues(m.n_endo)
    for a in get_or(m.raw, "aux_vars", [])
        get(a, "type", -1) in (2, 3) && (mask[Int(a["endo_index"])] = false)
    end
    mask
end

"Rules in log deviations for the variables in `mask` (as Dynare's stochastic_solvers.m)."
function loglinearize(ys, G, H, mask)
    if any(ys[mask] .<= 0) || !all(isfinite, ys[mask])
        bad = findall(mask .& .!(ys .> 0))
        throw(LoglinearError("The loglinearization of the model cannot be performed, because the steady state is not strictly positive (variables at positions $bad)"))
    end
    d = [mask[i] ? ys[i] : 1.0 for i in eachindex(ys)]
    log_ys = [mask[i] ? log(ys[i]) : ys[i] for i in eachindex(ys)]
    log_ys, G .* d' ./ d, H ./ d
end

"""
    first_order(m; ys=nothing, params=nothing, qz_criterium=nothing, loglinear=nothing) -> FirstOrderSolution

First-order solution at the steady state (computed unless `ys` is given). `qz_criterium` and
`loglinear` default to the values set in the .mod file (`stoch_simul`, `check`, `estimation`).
"""
function first_order(m::Model; ys=nothing, params=nothing, qz_criterium=nothing, loglinear=nothing)
    so = stoch_options(m)
    loglinear = loglinear === nothing ? get(so, "loglinear", false) == true : loglinear
    # solver choices: the preprocessor writes dr=cycle_reduction as dr_cycle_reduction=true, etc. (pydyn looked
    # for a "dr" key that the JSON never has, so it missed them). The solution is the same up to the
    # solver's tolerance, except for partial_information, which solves a different problem.
    for key in ("partial_information", "aim_solver", "dr", "dr_cycle_reduction", "dr_logarithmic_reduction",
                "dr_cycle_reduction_tol", "dr_cycle_reduction_maxiter", "dr_logarithmic_reduction_tol",
                "dr_logarithmic_reduction_maxiter", "sylvester", "sylvester_fixed_point_tol")
        (haskey(so, key) && so[key] !== false) && warn_once(m, "the .mod file sets $key: JDyn always uses its QZ-based solver")
    end
    if ys === nothing
        ys, params = steady_state(m; params=params)
    end
    params = params === nothing ? copy(m.params) : Vector{Float64}(params)
    ys = Vector{Float64}(ys)
    if loglinear
        mask = log_mask(m)
        if any(.!(ys[mask] .> 0))
            bad = [m.endo_names[i] for i in findall(mask .& .!(ys .> 0))]
            throw(LoglinearError("The loglinearization of the model cannot be performed, because the steady state is not strictly positive: $(join(bad, ", "))"))
        end
    end
    n = m.n_endo
    J = dynamic_jac(m, vcat(ys, ys, ys), exo_steady(m), params, ys)
    fm, f0, fp, fu = J[:, 1:n], J[:, n+1:2n], J[:, 2n+1:3n], J[:, 3n+1:end]
    lli = m.lead_lag_incidence
    G, H, eig = solve_linear_re(fm, f0, fp, fu; qz_criterium=JDyn.qz_criterium(m, qz_criterium),
                                qz_zero_threshold=JDyn.qz_zero_threshold(m), lag=lli[:, 1] .> 0, lead=lli[:, 3] .> 0)
    order_var, state_var, _ = dr_ordering(lli)
    nonstate = trues(n); nonstate[state_var] .= false
    G[:, nonstate] .= 0.0                              # clean numerical noise in columns of non-states
    levels = copy(ys)
    if loglinear
        ys, G, H = loglinearize(levels, G, H, log_mask(m))
    end
    FirstOrderSolution(ys, G, H, order_var, state_var, eig, J, params, loglinear, levels)
end
