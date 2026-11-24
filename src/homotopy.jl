# Homotopy for the steady state (Dynare's `steady(homotopy_mode=1|2|3, homotopy_steps=n)` with a `homotopy_setup` block): ports of
# homotopy1/2/3 of matlab/steady.m, which step the parameters and exogenous variables from their initial to their final values and
# solve the static model at every step from the previous steady state. Also `save_params_and_steady_state`.

"Octave's range `base:inc:limit` (the number of elements has a tolerance of 3 eps, the last element is clipped to `limit`)."
function _matlab_range(base::Float64, inc::Float64, limit::Float64)
    inc == 0 && return Float64[]
    n = floor(Int, ((limit - base) / inc) * (1 + 3 * eps()) + 1e-300) + 1
    n <= 0 && return Float64[]
    r = [base + (i - 1) * inc for i in 1:n]
    inc > 0 ? (r[end] > limit && (r[end] = limit)) : (r[end] < limit && (r[end] = limit))
    r
end

struct _HomRow
    name::String
    isparam::Bool
    idx::Int
    old::Float64            # NaN: the current value
    final::Float64
end

"The rows of Dynare's `options_.homotopy_values` (plus those `from_initval_to_endval` adds, one per exogenous variable not listed)."
function _homotopy_rows(m::Model)
    rows = _HomRow[]
    for d in something(m.homotopy, Any[])
        nm = String(d["name"])
        old = _value(m, get(d, "initial_value", "NaN"))
        fin = _value(m, d["final_value"])
        haskey(m.ip, nm) ? push!(rows, _HomRow(nm, true, m.ip[nm], Float64(old), Float64(fin))) :
                           push!(rows, _HomRow(nm, false, m.ix[nm], Float64(old), Float64(fin)))
    end
    if get(m.options, "homotopy_from_initval_to_endval", false) == true
        isempty(m.endval) && throw(ModelError("HOMOTOPY_SETUP: the from_initval_to_endval option cannot be used without an endval block"))
        listed = Set(r.name for r in rows if !r.isparam)
        for nm in m.exo_names
            nm in listed && continue
            ini = get(m.initval, nm, 0.0)
            push!(rows, _HomRow(nm, false, m.ix[nm], ini, get(m.endval, nm, ini)))
        end
    end
    rows
end

mutable struct _HomState
    m::Model
    params::Vector{Float64}
    exo::Vector{Float64}
    ys::Vector{Float64}
    log::Vector{NamedTuple}
end

"One `evaluate_steady_state` at the state's parameters and exogenous values; false when it fails (Dynare's `info(1) ~= 0`)."
function _hom_solve!(S::_HomState; first::Bool=false)
    m = S.m
    saved = m.initval
    try
        if first
            m.initval = merge(copy(m.initval), m.oo)
        end
        ys, p = with_logger(NullLogger()) do
            steady_state(m; params=copy(S.params), exo=copy(S.exo), guess=first ? nothing : fill_auxiliary(m, S.ys, S.exo, S.params))
        end
        if !all(isfinite, ys)
            push!(S.log, (ok=false, params=copy(S.params), exo=copy(S.exo), ys=ys))
            return false
        end
        S.ys, S.params = ys, p
        push!(S.log, (ok=true, params=copy(S.params), exo=copy(S.exo), ys=copy(S.ys)))
        return true
    catch e
        (e isa SteadyStateError || e isa LinearAlgebra.SingularException || e isa DomainError || e isa LinearAlgebra.LAPACKException) || rethrow()
        push!(S.log, (ok=false, params=copy(S.params), exo=copy(S.exo), ys=copy(S.ys)))
        return false
    finally
        m.initval = saved
    end
end

function _hom_set!(S::_HomState, rows, vals)
    for (r, v) in zip(rows, vals)
        r.isparam ? (S.params[r.idx] = v) : (S.exo[r.idx] = v)
    end
end
_hom_get(S::_HomState, r::_HomRow) = r.isparam ? S.params[r.idx] : S.exo[r.idx]

function _homotopy1!(S::_HomState, rows, steps)
    nv = length(rows)
    old = [isnan(r.old) ? _hom_get(S, r) : r.old for r in rows]
    points = zeros(nv, steps + 1)
    for (i, r) in enumerate(rows)
        if old[i] != r.final
            rng = _matlab_range(old[i], (r.final - old[i]) / steps, r.final)
            length(rng) == steps + 1 || throw(ModelError("homotopy mode 1: the range of $(r.name) has $(length(rng)) points, not $(steps + 1) (Octave/MATLAB fail on the assignment as well)"))
            points[i, :] = rng
        else
            points[i, :] .= r.final
        end
    end
    last_p = copy(S.params); last_x = copy(S.exo)
    for i in 1:steps+1
        _hom_set!(S, rows, points[:, i])
        if !_hom_solve!(S; first=i == 1)
            if i == 1
                return 2
            end
            S.params .= last_p; S.exo .= last_x
            return 1
        end
        last_p = copy(S.params); last_x = copy(S.exo)
    end
    0
end

function _homotopy2!(S::_HomState, rows, steps)
    nv = length(rows)
    old = [r.old for r in rows]
    for (i, r) in enumerate(rows)
        if isnan(old[i])
            old[i] = _hom_get(S, r)
        else
            _hom_set!(S, [r], [old[i]])
        end
    end
    any(old .== [r.final for r in rows]) && throw(ModelError("HOMOTOPY mode 2: initial and final values should be different"))
    last_p = copy(S.params); last_x = copy(S.exo)
    for (i, r) in enumerate(rows)
        for v in _matlab_range(old[i], (r.final - old[i]) / steps, r.final)
            _hom_set!(S, [r], [v])
            if !_hom_solve!(S; first=i == 1 && v == old[1])
                if i == 1 && v == old[1]
                    return 2
                end
                S.params .= last_p; S.exo .= last_x
                return 1
            end
            last_p = copy(S.params); last_x = copy(S.exo)
        end
    end
    0
end

function _homotopy3!(S::_HomState, rows, steps)
    tol = 1e-8
    nv = length(rows)
    old = [isnan(r.old) ? _hom_get(S, r) : r.old for r in rows]
    target = [r.final for r in rows]
    iplus = findall(target .> old); iminus = findall(target .< old)
    cur = copy(old)
    inc = (target .- old) ./ 2
    kplus = Int[]; kminus = Int[]
    last_values = Float64[]; last_p = copy(S.params); last_x = copy(S.exo)
    iter = 1
    while iter <= steps
        _hom_set!(S, rows, cur)
        old_ys = copy(S.ys)
        if _hom_solve!(S; first=iter == 1)
            if length(kplus) + length(kminus) == nv
                return 0
            end
            last_values = copy(cur); last_p = copy(S.params); last_x = copy(S.exo)
            inc = 2 .* inc
        elseif iter == 1
            return 2
        else
            inc = inc ./ 2
            S.ys = old_ys
        end
        cur = last_values .+ inc
        kplus = findall(cur[iplus] .>= target[iplus])
        cur[iplus[kplus]] = target[iplus[kplus]]
        kminus = findall(cur[iminus] .<= target[iminus])
        cur[iminus[kminus]] = target[iminus[kminus]]
        if maximum(abs, inc) < tol
            S.params .= last_p; S.exo .= last_x
            return 1
        end
        iter += 1
    end
    S.params .= last_p; S.exo .= last_x
    1
end

"""
`steady(homotopy_mode=..., homotopy_steps=...)` of Dynare's `steady.m` for a model with a `homotopy_setup` block. On success the model's
parameters, exogenous steady-state values and `oo` hold the final values and the last steady state (the log of the solved points is in
`m.cache[:homotopy_log]`). Errors as Dynare does unless `homotopy_force_continue`.
"""
function _steady_homotopy!(m::Model, opts)
    mode = Int(get(opts, "homotopy_mode", 0))
    mode in (0, 1, 2, 3) || throw(ModelError("STEADY: invalid value for homotopy_mode option"))
    if mode == 0
        (m.homotopy !== nothing || get(m.options, "homotopy_from_initval_to_endval", false) == true) &&
            warn_once(m, "STEADY: a homotopy_setup block is present but homotopy will not be performed because homotopy_mode option is equal to 0")
        return
    end
    (m.homotopy === nothing && get(m.options, "homotopy_from_initval_to_endval", false) != true) &&
        throw(ModelError("STEADY: a homotopy_setup block must be present when the homotopy_mode option is specified"))
    (m.steady_state_fn !== nothing || m.steady_state_model !== nothing || m.steady_state_file !== nothing) &&
        throw(ModelError("STEADY: Can't use homotopy when providing a steady state external file"))
    steps = Int(get(opts, "homotopy_steps", 1))
    rows = _homotopy_rows(m)
    m.session !== nothing && m.session.pending !== nothing && resolve!(m.session)     # a previous steady statement: its result is the starting point
    ys0 = Float64[get(m.oo, v, 0.0) for v in m.endo_names]
    S = _HomState(m, copy(m.params), exo_steady(m), ys0, NamedTuple[])
    code = mode == 1 ? _homotopy1!(S, rows, steps) : mode == 2 ? _homotopy2!(S, rows, steps) : _homotopy3!(S, rows, steps)
    m.cache[:homotopy_log] = S.log
    if code != 0
        msg = code == 2 ? "homotopy failed at the first iteration (for the starting values)" :
              "homotopy step was not completed; the last values for which a solution was found are: " *
              join(["$(r.name) = $(round(_hom_get(S, r); sigdigits=8))" for r in rows], ", ")
        if get(opts, "homotopy_force_continue", 0) in (1, true)
            @warn "STEADY: $msg; homotopy_force_continue is set, so the steady state goes on"
        else
            throw(SteadyStateError("Homotopy step failed: $msg"))
        end
    end
    m.params .= S.params
    for (i, v) in enumerate(m.exo_names)
        m.initval[v] = S.exo[i]; m.oo[v] = S.exo[i]
    end
    for (v, y) in zip(m.endo_names, S.ys)
        isfinite(y) && (m.oo[v] = y)
    end
    nothing
end

"`save_params_and_steady_state(file)`: parameters, steady state and exogenous steady state, one `name value` line each (`%.16g`)."
function _save_params_and_steady_state(m::Model, filename::AbstractString)
    path = isabspath(filename) ? filename : joinpath(m.mod_dir, filename)
    sess = m.session
    sess !== nothing && sess.pending !== nothing && resolve!(sess)
    open(path, "w") do io
        for (i, p) in enumerate(m.param_names)
            println(io, p, " ", Printf.@sprintf("%.16g", m.params[i]))
        end
        for v in m.endo_names
            println(io, v, " ", Printf.@sprintf("%.16g", get(m.oo, v, 0.0)))
        end
        for v in m.exo_names
            println(io, v, " ", Printf.@sprintf("%.16g", get(m.oo, v, get(m.initval, v, 0.0))))
        end
    end
    nothing
end
