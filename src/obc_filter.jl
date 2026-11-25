# The `obc_*` options in estimation and the news-shock solvers for the inversion filter (port of pydyn/obc_filter.py and of the obc_* handling of
# pydyn/occbin_estimation.py, GPL-3.0-or-later; docs/OBC_ESTIMATION.md).
#
#   * option parsing shared by the mod-file route (`options_.occbin.obc_* = ...;` native lines), the live workspace (`occbin_solver!`,
#     `Estimation(...; occbin=Dict("obc_method" => ...))`) and the keyword route (`simulate(...; obc_order=2)`, `Estimation(...; obc_method="news")`);
#   * equilibrium selection (`obc_selection`): the single-spell solutions of a one-constraint LCP and the choice among them;
#   * `LCPFilterSolver` / `RegimeFilterSolver`: the one-period solver of the inversion filter with the news method. For a given binding set the news-shock
#     solution is affine in [state; shock], so the first period's rule (T, R, C) comes from the same maps as the path and is cached per binding set.

# ====================================================================================================== options
const OBC_SELECTIONS = (:occbin, :shortest, :longest, :min_wedge)

"The `obc_selection` value as a Symbol (`occbin`, `shortest`, `longest`, `min_wedge`); anything else is an error."
function _obc_selection(x)
    s = Symbol(lowercase(strip(string(x isa AbstractArray && length(x) == 1 ? x[1] : x))))
    s in OBC_SELECTIONS || throw(ArgumentError("unknown obc_selection '$x': use one of $(join(OBC_SELECTIONS, ", "))"))
    s
end

_obc_flag(x) = (v = x isa AbstractArray && length(x) == 1 ? x[1] : x; v isa Union{Bool,Real,AbstractString} ? _asbool(v) :
                throw(ArgumentError("obc_cubature must be true or false (got $x)")))

"""
`obc_*` options of a flat option dictionary (`options_.occbin` of the .mod file or of the workspace) as `simulate` keywords:
`obc_method`, `obc_horizon`, `obc_selection`, `obc_order`, `obc_cubature`, `obc_periods_of_uncertainty`, `obc_qmc_level`,
`obc_cubature_regions`, `obc_check_ahead`. Invalid values are errors.
"""
function _obc_flat_options(flat::AbstractDict)
    out = Dict{Symbol,Any}()
    haskey(flat, "obc_method") && (out[:obc_method] = _obc_method(flat["obc_method"]))
    haskey(flat, "obc_horizon") && (out[:obc_horizon] = _obc_horizon(flat["obc_horizon"]))
    haskey(flat, "obc_selection") && (out[:obc_selection] = _obc_selection(flat["obc_selection"]))
    haskey(flat, "obc_formulation") && (out[:obc_formulation] = _obc_formulation(flat["obc_formulation"]))
    haskey(flat, "obc_order") && (out[:obc_order] = _obc_order(flat["obc_order"]))
    haskey(flat, "obc_cubature") && (out[:obc_cubature] = _obc_flag(flat["obc_cubature"]))
    haskey(flat, "obc_periods_of_uncertainty") &&
        (out[:obc_periods_of_uncertainty] = _obc_int_option(flat["obc_periods_of_uncertainty"], "obc_periods_of_uncertainty"))
    haskey(flat, "obc_qmc_level") && (out[:obc_qmc_level] = _obc_int_option(flat["obc_qmc_level"], "obc_qmc_level"; min=0))
    haskey(flat, "obc_cubature_regions") && (out[:obc_cubature_regions] = _obc_regions(flat["obc_cubature_regions"]))
    haskey(flat, "obc_m_diagnostic") && (out[:obc_m_diagnostic] = _obc_mdiag(flat["obc_m_diagnostic"]))
    haskey(flat, "obc_cubature_milp_time_limit") &&
        (out[:obc_cubature_milp_time_limit] = _obc_float_option(flat["obc_cubature_milp_time_limit"], "obc_cubature_milp_time_limit"; lo=0.0))
    haskey(flat, "obc_cubature_failure_tolerance") &&
        (out[:obc_cubature_failure_tolerance] = _obc_float_option(flat["obc_cubature_failure_tolerance"], "obc_cubature_failure_tolerance"; lo=0.0, hi=1.0, open_hi=true))
    haskey(flat, "obc_check_ahead") && (out[:obc_check_ahead] = _obc_int_option(flat["obc_check_ahead"], "obc_check_ahead"; min=0))
    out
end

"`(method, selection)` with the estimation alias resolved: `minnorm` is `news` with `obc_selection = min_wedge` (as in pydyn)."
function _obc_canonical(method::Symbol, selection::Symbol)
    if method == :minnorm
        selection in (:occbin, :min_wedge) ||
            throw(ArgumentError("obc_method = minnorm is news with obc_selection = min_wedge; it cannot be combined with obc_selection = $selection"))
        return :news, :min_wedge
    end
    method, selection
end

# ====================================================================================================== equilibrium selection
"""
    _single_spells(Mf, q, T; l_max=8, tol=1e-8) -> Vector{(l, k, y)}

All solutions of the complementarity problem with one binding spell `l+1 .. l+k` (1-based periods, `l <= l_max`; one constraint). `Mf` is `N x T`
(complementarity rows first, then the check-ahead rows), `q` its constant. On the spell the wedges solve `M[S, S] y = -q[S]` and must be positive;
the gap must be non-negative on every row. More than one entry means the state has several solutions. A block with a pivot of at most
`OBC_PIVOT_RTOL` times the largest entry of `M` is skipped (a rounding-noise "solution").
"""
function _single_spells(Mf::AbstractMatrix, q::AbstractVector, T::Int; l_max::Int=8, tol::Real=1e-8)
    scale = max(1.0, maximum(abs, q))
    mscale = maximum(abs, Mf)
    out = Tuple{Int,Int,Vector{Float64}}[]
    for l in 0:min(l_max, T - 1), k in 1:T-l
        sl = l+1:l+k
        F = lu(Mf[sl, sl]; check=false)
        issuccess(F) || continue
        minimum(abs, diag(F.U)) <= OBC_PIVOT_RTOL * mscale && continue
        yk = -(F \ q[sl])
        (all(isfinite, yk) && minimum(yk) > tol * scale) || continue            # a spell of exactly k periods needs positive wedges
        minimum(q .+ view(Mf, :, sl) * yk) < -tol * scale && continue
        y = zeros(T); y[sl] = yk
        push!(out, (l, k, y))
    end
    out
end

"The candidate `(l, k, y)` chosen by `selection`: fewest binding periods (`shortest`), most (`longest`) or smallest sum of squared wedges (`min_wedge`); ties go to the earliest start."
function _pick_spell(cands, selection::Symbol)
    key = selection == :shortest ? (c -> (c[2], c[1])) : selection == :longest ? (c -> (-c[2], c[1])) :
          (c -> (sum(abs2, c[3]), c[1], c[2]))
    argmin(map(key, cands)) |> i -> cands[i]
end

"Replace the default solution `y` (which binds somewhere) by the single-spell solution chosen by `selection`; the default stands when there is none. Returns `(y, ncandidates)`."
function _select_spell(Mf::AbstractMatrix, q::AbstractVector, T::Int, y::AbstractVector, selection::Symbol, tol::Real)
    cands = _single_spells(Mf, q, T; tol=tol)
    isempty(cands) && return y, 0
    _pick_spell(cands, selection)[3], length(cands)
end

# ====================================================================================================== solvers
"The one-period result the inversion filter reads: `piecewise` (1 x n, levels), `ss[1] = (T, R, C)`, `regime_history[1]`."
struct _OnePeriod
    piecewise::Matrix{Float64}
    ss::Vector{_TRC}
    regime_history::Vector{OccRegime}
end

abstract type NewsFilterSolver end

"State of `obc_horizon_max` of a solver: the larger solvers (horizon doubled up to `horizon_max`) are built once and kept with it, i.e. per parameter vector."
mutable struct _HorizonRetry
    horizon::Int
    check_ahead::Int
    horizon_max::Int                 # 0: off
    big::Dict{Int,NewsFilterSolver}
end
function _horizon_retry(horizon::Integer, check_ahead::Integer, horizon_max::Integer)
    horizon_max == 0 || horizon_max >= horizon ||
        throw(ArgumentError("obc_horizon_max = $horizon_max is below the horizon $horizon"))
    _HorizonRetry(Int(horizon), Int(check_ahead), Int(horizon_max), Dict{Int,NewsFilterSolver}())
end

# Why a trial point of the news inversion filter failed. Dynare's numeric code stays 313 (311 for linear algebra); the cause is extra information:
#   :horizon_too_short  the wedges that meet the complementarity rows inside the news horizon cannot meet the check-ahead rows (a linear programme on the
#                       head rows is feasible, the one on all rows is not), or the regime solver / integer programme says so
#   :no_solution        proved: not even the complementarity rows can be met (or the regime solver: no solution)
#   :undecided          the cheap checks did not decide and no integer programme was run (the default); counted as a failure. Not proved to be a horizon
#                       problem, but it is the usual form it takes (pydyn soak/repro/finite_vs_inf_news_304.py: horizon 40 fails with 168 undecided trial
#                       points, horizon 80 reproduces OccBin to 1e-11); `obc_horizon_max` retries these too
#   :milp_timeout       the integer programme reached `obc_milp_time_limit`
#   :numerical          a linear-algebra failure (code 311)
const OBC_FAILURE_CAUSES = (:horizon_too_short, :no_solution, :undecided, :milp_timeout, :numerical)

mutable struct _SolverStats
    calls::Int
    fallbacks::Int
    proved_infeasible::Int
    timeouts::Int
    skipped::Int
    rescued::Int
    errors::Dict{Int,Int}
    causes::Dict{Symbol,Int}          # cause -> failed trial points (cumulative; a point solved by a retry is not counted)
    period_causes::Dict{Symbol,Int}   # the same since `begin_period!`
    retries::Int                      # trial points solved again with a larger horizon (`obc_horizon_max`)
    retried_ok::Int                   # ... and solved
    retry_seconds::Float64            # time spent in retries, including building the larger solvers
    period::Int
    last_cause::Symbol                # cause of the last failed trial point (`:none` after a success)
    formulation::Symbol               # `:lcp` or `:regime` (set by `news_filter_solver`)
    note::String                      # why `auto` chose it (approximate-class constraint), "" otherwise
end
_SolverStats() = _SolverStats(0, 0, 0, 0, 0, 0, Dict{Int,Int}(), Dict{Symbol,Int}(), Dict{Symbol,Int}(), 0, 0, 0.0, 0, :none, :lcp, "")
function _fail!(st::_SolverStats, code::Int, cause::Symbol=:undecided)
    st.errors[code] = get(st.errors, code, 0) + 1
    st.causes[cause] = get(st.causes, cause, 0) + 1
    st.period_causes[cause] = get(st.period_causes, cause, 0) + 1
    st.last_cause = cause
    (code, nothing)
end

"""
    ObcFailure

Why an evaluation of the news inversion filter failed (code 304): `period` (1-based), `primary` (`:horizon_too_short` if any trial point of the period failed
for that reason, the actionable one, else the most frequent cause, or `:root_search` when no trial point failed in the news solver and the root search did
not converge), `counts` (cause => trial points of the period), `horizon`, `horizon_max` (0: off) and `advice` (the message added to `est.last_info`).
Retrieve it with `obc_failure(est)` or `ivf.failure`.
"""
struct ObcFailure
    period::Int
    primary::Symbol
    counts::Dict{Symbol,Int}
    horizon::Int
    horizon_max::Int
    advice::String
end
Base.show(io::IO, f::ObcFailure) = print(io, "ObcFailure(period ", f.period, ", ", f.primary, ": ", f.advice, ")")

function _filter_input(ob::Occbin, shocks, init, exo_pos)
    u = zeros(ob.n_exo)
    sh = vec(Float64.(shocks))
    exo_pos === nothing ? (u[1:length(sh)] = sh) : (u[exo_pos] = sh)
    u, vcat(init, u)
end

function _finish(ob::Occbin, S0::Matrix{Float64}, S::Matrix{Float64}, s0::Vector{Float64}, regime::OccRegime, x::AbstractVector)
    n = ob.n
    state = S * x .+ s0
    (0, _OnePeriod(reshape(state .+ ob.ys, 1, :), [(S[:, 1:n], S[:, n+1:end], s0)], [regime]))
end

"""
    LCPFilterSolver(ob; horizon=40, check_ahead=40, selection=:occbin, milp_time_limit=0, horizon_max=0)

One complementary pair per constraint (`obc_system` / Holden's LCP). `milp_time_limit` is the time the integer programme may take at a trial point
on which the cheap checks (active-set iteration, proof of non-existence by a linear programme) do not decide; 0 (the default): none, the trial point
has failed (error 313, as OccBin's own failures there). `horizon_max > 0` (`obc_horizon_max`): a trial point that fails with `:horizon_too_short` or `:undecided` is solved again
with the horizon doubled, up to `horizon_max` (the larger solvers are built once and kept with this one); see `OBC_FAILURE_CAUSES`.
"""
mutable struct LCPFilterSolver <: NewsFilterSolver
    ob::Occbin
    sys::ObcSystem
    T::Int                       # periods with wedges (per constraint)
    nT::Int                      # wedges: constraints x T
    Nr::Int                      # gap rows: constraints x (T + check ahead)
    Mf::Matrix{Float64}          # Nr x nT: complementarity rows first, then the rows beyond the horizon
    MT::Matrix{Float64}          # nT x nT
    Mb::Matrix{Float64}
    q0f::Vector{Float64}
    Qxf::Matrix{Float64}
    S0::Matrix{Float64}
    V0::Matrix{Float64}
    selection::Symbol
    milp_time_limit::Float64
    tol::Float64
    maps::Dict{BitVector,Tuple{Matrix{Float64},Vector{Float64},OccRegime}}
    stats::_SolverStats
    base::OccRegime
    retry::_HorizonRetry
end

function LCPFilterSolver(ob::Occbin; horizon::Integer=40, check_ahead::Integer=40, selection::Symbol=:occbin, milp_time_limit::Real=0.0, horizon_max::Integer=0)
    selection != :occbin && length(obc_constraints(ob)) != 1 &&
        throw(ArgumentError("obc_selection = $selection is implemented for models with one constraint only"))
    T, N = Int(horizon), Int(horizon) + Int(check_ahead)
    sys = obc_system(ob; horizon=T, method=:news)
    ext = _obc_ext(sys, N)
    Mf = vcat(sys.M, ext.Mb)
    nc = length(sys.cons)
    LCPFilterSolver(ob, sys, T, nc * T, size(Mf, 1), Mf, Matrix(sys.M), Matrix(ext.Mb), vcat(sys.q0, ext.q0b), vcat(sys.Qx, ext.Qxb), hcat(ob.P, ob.Q), Matrix(sys.Wn),
                    selection, Float64(milp_time_limit), 1e-8, Dict{BitVector,Tuple{Matrix{Float64},Vector{Float64},OccRegime}}(), _SolverStats(),
                    [_map_regime(falses(T)) for _ in 1:nc], _horizon_retry(T, check_ahead, horizon_max))
end

function _lcp_map(sv::LCPFilterSolver, on::BitVector)
    get!(sv.maps, on) do
        T, nc = sv.T, length(sv.sys.cons)
        idx = findall(on)
        sol = -(sv.MT[idx, idx] \ hcat(sv.Qxf[idx, :], sv.q0f[idx]))
        W, c = sol[:, 1:end-1], sol[:, end]
        S = sv.S0 .+ sv.V0[:, idx] * W
        s0 = sv.V0[:, idx] * c
        (S, s0, [_map_regime(view(on, (j-1)*T+1:j*T)) for j in 1:nc])
    end
end

"""
    filter_solve(solver, shocks, init, exo_pos) -> (error_flag, result)

One period of the news inversion filter: the news solution from state `init` (deviations) with `shocks` (on the exogenous variables `exo_pos`). A failure keeps
Dynare's code (313, 311) and records its cause (`OBC_FAILURE_CAUSES`) in `solver.stats`; with `obc_horizon_max` a point that failed for lack of horizon is solved again.
"""
function filter_solve(sv::NewsFilterSolver, shocks, init, exo_pos)
    flag, res = _solve_once(sv, shocks, init, exo_pos)
    flag == 0 && return flag, res
    st, hr = sv.stats, sv.retry
    (hr.horizon_max > hr.horizon && st.last_cause in (:horizon_too_short, :undecided)) || return flag, res
    cause = st.last_cause
    got = _retry_larger(sv, shocks, init, exo_pos)
    got === nothing && return flag, res
    st.errors[flag] -= 1                           # solved with the larger horizon: not a failure
    st.causes[cause] -= 1
    st.period_causes[cause] -= 1
    st.last_cause = :none
    got
end

"`obc_horizon_max`: solve a failed trial point again with the horizon doubled up to the maximum. `nothing` if no larger horizon solves it."
function _retry_larger(sv::NewsFilterSolver, shocks, init, exo_pos)
    hr, st = sv.retry, sv.stats
    t0 = time_ns()
    h = hr.horizon
    out = nothing
    while h < hr.horizon_max
        h = min(2h, hr.horizon_max)
        big = get!(hr.big, h) do
            _like_solver(sv; horizon=h, check_ahead=max(h, hr.check_ahead))
        end
        st.retries += 1
        flag, res = _solve_once(big, shocks, init, exo_pos)
        if flag == 0
            st.retried_ok += 1
            out = (flag, res)
            break
        end
        big.stats.last_cause in (:horizon_too_short, :undecided) || break
    end
    st.retry_seconds += (time_ns() - t0) * 1e-9
    out
end

function _solve_once(sv::LCPFilterSolver, shocks, init, exo_pos)
    ob, T, nT, N, tol, st = sv.ob, sv.T, sv.nT, sv.Nr, sv.tol, sv.stats
    st.calls += 1
    u, x = _filter_input(ob, shocks, init, exo_pos)
    q = sv.q0f .+ sv.Qxf * x
    scale = max(1.0, maximum(abs, q))
    on = falses(nT)
    if !all(>=(-tol * scale), q)
        f0 = st.fallbacks
        qT = view(q, 1:nT)
        y, stat, _ = solve_active_set(sv.MT, qT; tol=tol)
        ok = stat == :ok && (N == nT || minimum(view(q, nT+1:N) .+ sv.Mb * y) >= -tol * scale)
        if !ok
            st.fallbacks += 1
            # cheap proof of non-existence first (the filter's root finder also tries implausible shocks)
            stl, _ = highs_lp(zeros(nT), -sv.Mf, fill(-Inf, N), q, zeros(nT), fill(Inf, nT))
            if stl == :infeasible
                st.proved_infeasible += 1
                # (b) the head rows alone can be met: the check-ahead rows are what fail, i.e. the horizon is too short; (a) not even they can
                sth, _ = highs_lp(zeros(nT), -view(sv.Mf, 1:nT, :), fill(-Inf, nT), qT, zeros(nT), fill(Inf, nT))
                return _fail!(st, 313, sth == :infeasible ? :no_solution : :horizon_too_short)
            end
            y = nothing
            cause = :undecided
            if sv.milp_time_limit > 0
                y2, st2, i2 = solve_news(sv.MT, Vector(qT); tol=tol, Mb=sv.Mb, qb=Vector(view(q, nT+1:N)), horizons=[T], n_c=length(sv.sys.cons),
                                         time_limit=sv.milp_time_limit)
                if st2 == :ok
                    y = y2
                elseif st2 == :not_converged
                    st.timeouts += 1; cause = :milp_timeout
                elseif st2 == :horizon_too_short
                    cause = :horizon_too_short
                elseif st2 == :no_solution && get(i2, :proof, false) === true
                    cause = :no_solution
                end
            else
                st.skipped += 1
            end
            y === nothing && return _fail!(st, 313, cause)
        end
        st.fallbacks > f0 && (st.rescued += 1)
        if sv.selection != :occbin && any(!iszero, y)
            y, _ = _select_spell(sv.Mf, q, nT, y, sv.selection, tol)
        end
        on = y .> 1e-9 * max(1.0, maximum(abs, y))
    end
    if !any(on)
        return _finish(ob, sv.S0, sv.S0, zeros(ob.n), sv.base, x)
    end
    S, s0, regime = _lcp_map(sv, on)
    _finish(ob, sv.S0, S, s0, regime, x)
end

_like_solver(sv::LCPFilterSolver; horizon, check_ahead) =
    LCPFilterSolver(sv.ob; horizon=horizon, check_ahead=check_ahead, selection=sv.selection, milp_time_limit=sv.milp_time_limit)

"""
    RegimeFilterSolver(ob; horizon=40, check_ahead=40, selection=:occbin, milp_time_limit=0, horizon_max=0)

The general formulation (`RegimeOBC`) for every model OccBin accepts: constraints that switch several equations, conditions that are not the mirror
image of the switched equation. For a regime sequence the wedges on the binding positions `S` are `w = -R[S,S]^-1 (a0 + Am [state; shock])[S]`, so the next
state is `[P Q] x + V0[:, S] w` and the first period's rule is read off and cached per `S`. `obc_selection` (one constraint only) picks among the single-spell solutions.
`horizon_max` as for `LCPFilterSolver`.
"""
mutable struct RegimeFilterSolver <: NewsFilterSolver
    ob::Occbin
    rg::RegimeOBC
    S0::Matrix{Float64}
    milp_time_limit::Float64
    selection::Symbol
    maps::Dict{Vector{Int},Tuple{Matrix{Float64},Vector{Float64}}}
    stats::_SolverStats
    retry::_HorizonRetry
end

function RegimeFilterSolver(ob::Occbin; horizon::Integer=40, check_ahead::Integer=40, selection::Symbol=:occbin, milp_time_limit::Real=0.0, horizon_max::Integer=0)
    ob.k == 1 || selection == :occbin || throw(ArgumentError("obc_selection = $selection is implemented for models with one constraint only (this model has $(ob.k)); use obc_selection = occbin"))
    T = Int(horizon)
    RegimeFilterSolver(ob, regime_obc(ob, T, T + Int(check_ahead)), hcat(ob.P, ob.Q), Float64(milp_time_limit), selection,
                       Dict{Vector{Int},Tuple{Matrix{Float64},Vector{Float64}}}(), _SolverStats(), _horizon_retry(T, check_ahead, horizon_max))
end

_like_solver(sv::RegimeFilterSolver; horizon, check_ahead) =
    RegimeFilterSolver(sv.ob; horizon=horizon, check_ahead=check_ahead, selection=sv.selection, milp_time_limit=sv.milp_time_limit)

function _solve_once(sv::RegimeFilterSolver, shocks, init, exo_pos)
    ob, rg, st = sv.ob, sv.rg, sv.stats
    st.calls += 1
    u, x = _filter_input(ob, shocks, init, exo_pos)
    res = try
        old = REGIME_MILP_TIME_LIMIT[]
        sv.milp_time_limit > 0 && (REGIME_MILP_TIME_LIMIT[] = sv.milp_time_limit)
        try
            _regime_step(rg, init, u; milp=sv.milp_time_limit > 0, selection=sv.selection)
        finally
            REGIME_MILP_TIME_LIMIT[] = old
        end
    catch e
        e isa Union{LinearAlgebra.SingularException,LinearAlgebra.LAPACKException,DomainError} || rethrow()
        return _fail!(st, 311, :numerical)
    end
    status, z, w, info = res
    if status != :ok
        st.fallbacks += 1
        status == :not_converged && (st.timeouts += 1)
        status == :no_solution && occursin("linear programme infeasible", info.message) && (st.proved_infeasible += 1)
        attempted = !occursin("not attempted", info.message)
        attempted || (st.skipped += 1)
        cause = status == :horizon_too_short ? :horizon_too_short : status == :not_converged ? :milp_timeout :
                (status == :no_solution && attempted) ? :no_solution : :undecided
        return _fail!(st, 313, cause)
    end
    if !(info.used in (:none, :guess_verify))
        st.fallbacks += 1
        st.rescued += 1
    end
    S = _regime_index(rg, z)
    M, s0 = get!(sv.maps, S) do
        if isempty(S)
            (sv.S0, zeros(ob.n))
        else
            sol = -(rg.R[S, S] \ hcat(rg.Am[S, :], rg.a0[S]))
            (sv.S0 .+ rg.V0[:, S] * sol[:, 1:end-1], rg.V0[:, S] * sol[:, end])
        end
    end
    regime = [_map_regime(view(z, 1:rg.T, j)) for j in 1:ob.k]
    _finish(ob, sv.S0, M, s0, regime, x)
end

"""
    news_filter_solver(ob; formulation=:auto, horizon=40, check_ahead=40, selection=:occbin, milp_time_limit=0) -> LCPFilterSolver | RegimeFilterSolver

The solver of the inversion filter for `obc_method = news`. `formulation = :auto` (default): the LCP one when every constraint is exactly one complementary pair, the regime
one otherwise, and also when a constraint is of the approximate complementarity class (the LCP can pick another regime sequence than OccBin at knife-edge periods; the regime
solver follows OccBin's own test; `solver.stats.formulation`, `solver.stats.note`); `:lcp` (an error unless every constraint is one complementary pair) and `:regime`
force one (the choice `simulate` makes).
"""
function news_filter_solver(ob::Occbin; formulation::Symbol=:auto, kw...)
    sel = get(kw, :selection, :occbin)
    form, note = _resolve_formulation(ob, formulation; selection=sel)
    sv = form == :lcp ? LCPFilterSolver(ob; kw...) : RegimeFilterSolver(ob; kw...)
    sv.stats.formulation, sv.stats.note = form, note
    isempty(note) || (form == :regime ? info_once(ob.model, note) : warn_once(ob.model, note))
    sv
end

"Start of period `t` of the inversion filter (the per-period failure causes restart)."
begin_period!(sv::NewsFilterSolver, t::Integer) = (sv.stats.period = Int(t); empty!(sv.stats.period_causes); nothing)

"""
    failure_info(solver) -> ObcFailure

The cause of the inversion filter's failure in the current period (see `ObcFailure`). The horizon of the message is the solver's own (the base horizon, also
when `obc_horizon_max` tried larger ones).
"""
function failure_info(sv::NewsFilterSolver)
    st, hr = sv.stats, sv.retry
    c = Dict(k => v for (k, v) in st.period_causes if v > 0)
    t, h = st.period, hr.horizon
    if isempty(c)
        return ObcFailure(t, :root_search, c, h, hr.horizon_max,
                          "the root search of the period did not converge although no trial point failed in the news solver")
    end
    primary = haskey(c, :horizon_too_short) ? :horizon_too_short : first(sort(collect(c); by=p -> (-p[2], string(p[1])))).first
    tried = hr.horizon_max > h ? " (obc_horizon_max = $(hr.horizon_max) was tried)" : " (or set obc_horizon_max to retry with a doubled horizon)"
    advice = if primary == :horizon_too_short
        "news horizon $h too short at period $t: try obc_horizon >= $(2h)$tried"
    elseif primary == :no_solution
        "the trial shocks have no equilibrium at this state (proved by a linear programme)"
    elseif primary == :undecided
        "active-set iteration did not decide and no integer programme was run (obc_milp_time_limit = 0); a truncated horizon ($h) can cause this: try obc_horizon >= $(2h)$tried"
    elseif primary == :milp_timeout
        "the integer programme reached obc_milp_time_limit"
    else
        "linear-algebra failure"
    end
    ObcFailure(t, primary, c, h, hr.horizon_max, advice)
end

# ====================================================================================================== settings of an estimation
"""
    ObcEstSettings

`obc_method` (`:occbin` or `:news`; `minnorm` is resolved to `news` with `min_wedge`), `obc_horizon`, `obc_selection` and `obc_milp_time_limit` of an
`Estimation` (in `est.options["obc"]`); `last_solver` is the news solver of the last likelihood evaluation (its counters: `stats`).
"""
mutable struct ObcEstSettings
    method::Symbol
    horizon::Int
    selection::Symbol
    milp_time_limit::Float64
    last_solver::Any
    horizon_max::Int            # 0: off
    last_failure::Any           # `ObcFailure` of the last failed news filter run, else `nothing`
    formulation::Symbol         # `obc_formulation`: :auto, :lcp, :regime
end
Base.show(io::IO, s::ObcEstSettings) = print(io, "ObcEstSettings(", s.method, ", horizon = ", s.horizon, ", selection = ", s.selection, ", milp_time_limit = ", s.milp_time_limit,
                                              ", horizon_max = ", s.horizon_max, ", formulation = ", s.formulation, ")")

"""
`obc_horizon_max` as an Int: a positive integer, or 0 (also `nothing`, `false`) for off. `max < horizon` is checked where the horizon is known.
"""
function _obc_horizon_max(x)
    v = x isa AbstractArray ? (length(x) == 1 ? x[1] : isempty(x) ? nothing : throw(ArgumentError("obc_horizon_max must be a scalar"))) : x
    (v === nothing || v === false) && return 0
    v isa AbstractString && (v = tryparse(Float64, v); v === nothing && throw(ArgumentError("obc_horizon_max must be a positive integer (got $x)")))
    (v isa Real && isfinite(v) && v >= 0 && v == round(v)) || throw(ArgumentError("obc_horizon_max must be a positive integer, or 0 for off (got $x)"))
    Int(v)
end

function _check_horizon_max(hmax::Int, horizon::Int)
    hmax == 0 || hmax >= horizon || throw(ArgumentError("obc_horizon_max = $hmax is below obc_horizon = $horizon"))
    hmax
end

const _PKF_NEWS_MSG = "obc_method = news is implemented for the inversion filter only (occbin_setup(likelihood_inversion_filter)); " *
                      "the piecewise Kalman filter needs OccBin's regime matrices. Use obc_method = occbin or the inversion filter."

"Split the `obc_*` entries off an `occbin=` override collection: `(rest, obc::Dict{String,Any})`."
function _split_obc_overrides(overrides)
    overrides === nothing && return nothing, Dict{String,Any}()
    rest, obc = Dict{String,Any}(), Dict{String,Any}()
    for (k, v) in overrides
        ks = string(k)
        startswith(ks, "obc_") ? (obc[ks] = v) : (rest[ks] = v)
    end
    isempty(rest) ? nothing : rest, obc
end

"Settings from keywords, the live overrides and the native `options_.occbin.obc_*` lines of the .mod file (in that order of precedence)."
function _obc_est_settings(m::Model, live::Dict{String,Any}; method=nothing, horizon=nothing, selection=nothing, milp_time_limit=nothing, horizon_max=nothing, formulation=nothing)
    native = Dict{String,Any}()
    o = get(m.matlab_options, "occbin", nothing)
    if o isa MStruct
        for k in ("obc_method", "obc_horizon", "obc_selection", "obc_horizon_max", "obc_formulation")
            haskey(o, k) && (native[k] = o[k])
        end
    end
    pick(k, kw) = kw !== nothing ? kw : haskey(live, k) ? live[k] : get(native, k, nothing)
    meth = pick("obc_method", method)
    sel = pick("obc_selection", selection)
    hor = pick("obc_horizon", horizon)
    meth = meth === nothing ? :occbin : _obc_method(meth)
    sel = sel === nothing ? :occbin : _obc_selection(sel)
    meth in (:occbin, :news, :minnorm) || throw(ArgumentError("unknown obc_method $meth"))
    meth, sel = _obc_canonical(meth, sel)
    hor = hor === nothing ? 40 : _obc_horizon(hor)
    hmax = _check_horizon_max(_obc_horizon_max(pick("obc_horizon_max", horizon_max)), hor)
    form = pick("obc_formulation", formulation)
    ObcEstSettings(meth, hor, sel, milp_time_limit === nothing ? 0.0 : Float64(milp_time_limit), nothing, hmax, nothing, form === nothing ? :auto : _obc_formulation(form))
end

"`obc_formulation = lcp` is an error for a model with a constraint that is not one complementary pair (checked at the model's own parameters when they solve)."
function _check_lcp_model(m::Model)
    try
        _resolve_formulation(Occbin(m), :lcp)
    catch e
        e isa ArgumentError && rethrow()
    end
    nothing
end

"The settings of an OccBin estimation (`nothing` for a model without constraints)."
obc_settings(est::Estimation) = get(est.options, "obc", nothing)

"""
    set_obc!(est; method=nothing, horizon=nothing, selection=nothing, milp_time_limit=nothing, horizon_max=nothing) -> est

Change the `obc_*` options of an OccBin estimation (pydyn: `est.obc_method = 'news'`). Unknown values are errors; `news` with the piecewise Kalman
filter is refused; `minnorm` is `news` with `obc_selection = min_wedge` (a conflicting `selection` is an error). `horizon_max` (`obc_horizon_max`, 0 = off):
a trial point of the news filter that fails for lack of horizon is solved again with the horizon doubled up to this maximum (must be at least the horizon).
"""
function set_obc!(est::Estimation; method=nothing, horizon=nothing, selection=nothing, milp_time_limit=nothing, horizon_max=nothing, formulation=nothing)
    s = obc_settings(est)
    s === nothing && throw(EstimationError("the model has no occbin_constraints block"))
    meth = method === nothing ? s.method : _obc_method(method)
    sel = selection === nothing ? s.selection : _obc_selection(selection)
    meth, sel = meth == :minnorm ? _obc_canonical(meth, selection === nothing ? :occbin : sel) : (meth, sel)
    hor = horizon === nothing ? s.horizon : _obc_horizon(horizon)
    hmax = horizon_max === nothing ? s.horizon_max : _obc_horizon_max(horizon_max)
    _check_horizon_max(hmax, hor)
    meth == :news && !Bool(occbin_options(est)["likelihood.inversion_filter"]) && throw(EstimationError(_PKF_NEWS_MSG))
    form = formulation === nothing ? s.formulation : _obc_formulation(formulation)
    form == :lcp && _check_lcp_model(est.model)
    s.method, s.selection, s.horizon, s.horizon_max, s.formulation = meth, sel, hor, hmax, form
    milp_time_limit === nothing || (s.milp_time_limit = Float64(milp_time_limit))
    est
end

"The news solver of the filter at `ob`: `nothing` (OccBin's own solver) for `obc_method = occbin`."
function _obc_solver(est::Estimation, ob::Occbin)
    s = obc_settings(est)
    if s.method == :occbin
        s.selection != :occbin && throw(EstimationError("obc_selection other than occbin needs obc_method = news"))
        return nothing
    end
    s.last_solver = news_filter_solver(ob; horizon=s.horizon, selection=s.selection, milp_time_limit=s.milp_time_limit, horizon_max=s.horizon_max, formulation=s.formulation)
end

"""
    obc_failure(est) -> ObcFailure or nothing

Why the last likelihood evaluation (or `occbin_filter`) of an estimation with `obc_method = news` failed: the cause of the trial-point failures of the failing period
(`horizon_too_short`, `no_solution`, `undecided`, `milp_timeout`, `numerical`, or `root_search`) and the advice. Dynare's code stays 304; `est.last_info` carries the
same text. `nothing` after a successful evaluation and for `obc_method = occbin`. Counters of the whole evaluation: `obc_settings(est).last_solver.stats`
(`causes`, `retries`, `retried_ok`, `retry_seconds`).
"""
obc_failure(est::Estimation) = (s = obc_settings(est); s === nothing ? nothing : s.last_failure)
