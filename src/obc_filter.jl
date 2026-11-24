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
    haskey(flat, "obc_order") && (out[:obc_order] = _obc_order(flat["obc_order"]))
    haskey(flat, "obc_cubature") && (out[:obc_cubature] = _obc_flag(flat["obc_cubature"]))
    haskey(flat, "obc_periods_of_uncertainty") &&
        (out[:obc_periods_of_uncertainty] = _obc_int_option(flat["obc_periods_of_uncertainty"], "obc_periods_of_uncertainty"))
    haskey(flat, "obc_qmc_level") && (out[:obc_qmc_level] = _obc_int_option(flat["obc_qmc_level"], "obc_qmc_level"; min=0))
    haskey(flat, "obc_cubature_regions") && (out[:obc_cubature_regions] = _obc_int_option(flat["obc_cubature_regions"], "obc_cubature_regions"))
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

mutable struct _SolverStats
    calls::Int
    fallbacks::Int
    proved_infeasible::Int
    timeouts::Int
    skipped::Int
    rescued::Int
    errors::Dict{Int,Int}
end
_SolverStats() = _SolverStats(0, 0, 0, 0, 0, 0, Dict{Int,Int}())
_fail!(st::_SolverStats, code::Int) = (st.errors[code] = get(st.errors, code, 0) + 1; (code, nothing))

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
    LCPFilterSolver(ob; horizon=40, check_ahead=40, selection=:occbin, milp_time_limit=0)

One complementary pair per constraint (`obc_system` / Holden's LCP). `milp_time_limit` is the time the integer programme may take at a trial point
on which the cheap checks (active-set iteration, proof of non-existence by a linear programme) do not decide; 0 (the default): none, the trial point
has failed (error 313, as OccBin's own failures there).
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
end

function LCPFilterSolver(ob::Occbin; horizon::Integer=40, check_ahead::Integer=40, selection::Symbol=:occbin, milp_time_limit::Real=0.0)
    selection != :occbin && length(obc_constraints(ob)) != 1 &&
        throw(ArgumentError("obc_selection = $selection is implemented for models with one constraint only"))
    T, N = Int(horizon), Int(horizon) + Int(check_ahead)
    sys = obc_system(ob; horizon=T, method=:news)
    ext = _obc_ext(sys, N)
    Mf = vcat(sys.M, ext.Mb)
    nc = length(sys.cons)
    LCPFilterSolver(ob, sys, T, nc * T, size(Mf, 1), Mf, Matrix(sys.M), Matrix(ext.Mb), vcat(sys.q0, ext.q0b), vcat(sys.Qx, ext.Qxb), hcat(ob.P, ob.Q), Matrix(sys.Wn),
                    selection, Float64(milp_time_limit), 1e-8, Dict{BitVector,Tuple{Matrix{Float64},Vector{Float64},OccRegime}}(), _SolverStats(),
                    [_map_regime(falses(T)) for _ in 1:nc])
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

"`(error_flag, result)` for one period: the news solution from state `init` (deviations) with `shocks` (on the exogenous variables `exo_pos`)."
function filter_solve(sv::LCPFilterSolver, shocks, init, exo_pos)
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
                return _fail!(st, 313)
            end
            y = nothing
            if sv.milp_time_limit > 0
                y2, st2, _ = solve_news(sv.MT, Vector(qT); tol=tol, Mb=sv.Mb, qb=Vector(view(q, nT+1:N)), horizons=[T], n_c=length(sv.sys.cons),
                                        time_limit=sv.milp_time_limit)
                st2 == :ok ? (y = y2) : (st2 == :not_converged && (st.timeouts += 1))
            else
                st.skipped += 1
            end
            y === nothing && return _fail!(st, 313)
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

"""
    RegimeFilterSolver(ob; horizon=40, check_ahead=40, milp_time_limit=0)

The general formulation (`RegimeOBC`) for every model OccBin accepts: constraints that switch several equations, conditions that are not the mirror
image of the switched equation. For a regime sequence the wedges on the binding positions `S` are `w = -R[S,S]^-1 (a0 + Am [state; shock])[S]`, so the next
state is `[P Q] x + V0[:, S] w` and the first period's rule is read off and cached per `S`. `obc_selection` (one constraint only) picks among the single-spell solutions.
"""
mutable struct RegimeFilterSolver <: NewsFilterSolver
    ob::Occbin
    rg::RegimeOBC
    S0::Matrix{Float64}
    milp_time_limit::Float64
    selection::Symbol
    maps::Dict{Vector{Int},Tuple{Matrix{Float64},Vector{Float64}}}
    stats::_SolverStats
end

function RegimeFilterSolver(ob::Occbin; horizon::Integer=40, check_ahead::Integer=40, selection::Symbol=:occbin, milp_time_limit::Real=0.0)
    ob.k == 1 || selection == :occbin || throw(ArgumentError("obc_selection = $selection is implemented for models with one constraint only (this model has $(ob.k)); use obc_selection = occbin"))
    T = Int(horizon)
    RegimeFilterSolver(ob, regime_obc(ob, T, T + Int(check_ahead)), hcat(ob.P, ob.Q), Float64(milp_time_limit), selection,
                       Dict{Vector{Int},Tuple{Matrix{Float64},Vector{Float64}}}(), _SolverStats())
end

function filter_solve(sv::RegimeFilterSolver, shocks, init, exo_pos)
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
        return _fail!(st, 311)
    end
    status, z, w, info = res
    if status != :ok
        st.fallbacks += 1
        status == :not_converged && (st.timeouts += 1)
        status == :no_solution && occursin("linear programme infeasible", info.message) && (st.proved_infeasible += 1)
        occursin("not attempted", info.message) && (st.skipped += 1)
        return _fail!(st, 313)
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
    news_filter_solver(ob; horizon=40, check_ahead=40, selection=:occbin, milp_time_limit=0) -> LCPFilterSolver | RegimeFilterSolver

The solver of the inversion filter for `obc_method = news`: the LCP one when every constraint is one complementary pair, the regime one otherwise
(the choice `simulate` makes).
"""
function news_filter_solver(ob::Occbin; kw...)
    cons = get!(() -> obc_constraints(ob), _OBC_CONS_CACHE, ob)
    all(c -> c.class == :complementarity, cons) ? LCPFilterSolver(ob; kw...) : RegimeFilterSolver(ob; kw...)
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
end
Base.show(io::IO, s::ObcEstSettings) = print(io, "ObcEstSettings(", s.method, ", horizon = ", s.horizon, ", selection = ", s.selection, ", milp_time_limit = ", s.milp_time_limit, ")")

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
function _obc_est_settings(m::Model, live::Dict{String,Any}; method=nothing, horizon=nothing, selection=nothing, milp_time_limit=nothing)
    native = Dict{String,Any}()
    o = get(m.matlab_options, "occbin", nothing)
    if o isa MStruct
        for k in ("obc_method", "obc_horizon", "obc_selection")
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
    ObcEstSettings(meth, hor === nothing ? 40 : _obc_horizon(hor), sel,
                   milp_time_limit === nothing ? 0.0 : Float64(milp_time_limit), nothing)
end

"The settings of an OccBin estimation (`nothing` for a model without constraints)."
obc_settings(est::Estimation) = get(est.options, "obc", nothing)

"""
    set_obc!(est; method=nothing, horizon=nothing, selection=nothing, milp_time_limit=nothing) -> est

Change the `obc_*` options of an OccBin estimation (pydyn: `est.obc_method = 'news'`). Unknown values are errors; `news` with the piecewise Kalman
filter is refused; `minnorm` is `news` with `obc_selection = min_wedge` (a conflicting `selection` is an error).
"""
function set_obc!(est::Estimation; method=nothing, horizon=nothing, selection=nothing, milp_time_limit=nothing)
    s = obc_settings(est)
    s === nothing && throw(EstimationError("the model has no occbin_constraints block"))
    meth = method === nothing ? s.method : _obc_method(method)
    sel = selection === nothing ? s.selection : _obc_selection(selection)
    meth, sel = meth == :minnorm ? _obc_canonical(meth, selection === nothing ? :occbin : sel) : (meth, sel)
    hor = horizon === nothing ? s.horizon : _obc_horizon(horizon)
    meth == :news && !Bool(occbin_options(est)["likelihood.inversion_filter"]) && throw(EstimationError(_PKF_NEWS_MSG))
    s.method, s.selection, s.horizon = meth, sel, hor
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
    s.last_solver = news_filter_solver(ob; horizon=s.horizon, selection=s.selection, milp_time_limit=s.milp_time_limit)
end
