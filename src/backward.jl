# Purely backward models: simul_backward_model (matlab/backward/simul_backward_model.m, simul_backward_nonlinear_model.m,
# simul_backward_linear_model.m and +backward_model/{initialize,simul_nonlinear_model,simul_linear_model}.m of Dynare 7.2).
#
# In every period the dynamic equations are solved for the current endogenous variables from the previous period's values
# (Newton's method of dynare_solve, algorithm 4 as `_dynare_solve`); a linear model is inverted once. Without innovations the
# structural shocks are drawn as Dynare does: the seed is reset to its default (options_.bnlms.set_dynare_seed_to_default),
# then `randn(T, k) * chol(Sigma_e(pos, pos))`. Use `dynare(...; random=:octave)` for the very draws of Octave.
#
# Julia API:  sim = simul_backward_model(ws, periods; innovations=nothing, initialconditions=nothing)
# returns the dseries (endogenous variables of the model file, then the exogenous ones) and fills `oo_.endo_simul`,
# `oo_.exo_simul` and `oo_.bnlms.shocks`. The same function is the MATLAB-callable `simul_backward_model(init, T [, innov])`.
# Only models whose maximum lag is one are supported (auxiliary variables for longer lags are refused).

struct BackwardError <: Exception
    msg::String
end
Base.showerror(io::IO, e::BackwardError) = print(io, "BackwardError: ", e.msg)

"The innovations Dynare draws when none are given: a `T x n_exo` matrix."
function backward_innovations(ws, T::Int)
    m = ws.model
    S = m.Sigma_e
    v = diag(S)
    pos = findall(>(0), v)
    X = zeros(T, size(S, 1))
    isempty(pos) && return X
    U = try
        cholesky(Symmetric(S[pos, pos])).U
    catch
        throw(BackwardError("chol: the covariance matrix of the structural innovations is not positive definite"))
    end
    set_dynare_seed!(ws, "default")                     # options_.bnlms.set_dynare_seed_to_default = true
    R = dynare_rng(ws)
    Z = Matrix{Float64}(undef, T, length(pos))
    randn!(R, Z)
    X[:, pos] = Z * U
    X
end

function _backward_initial(ws, initialconditions)
    m = ws.model
    n = m.n_endo
    if initialconditions === nothing || (initialconditions isa AbstractArray && isempty(initialconditions))
        h = ws.M_.endo_histval
        (h isa AbstractArray && length(h) == n) || throw(BackwardError("no initial conditions: use histval or pass a dseries"))
        return reshape(Float64.(vec(h)), n, 1), zeros(1, m.n_exo)
    end
    d = initialconditions
    d isa MDseries || throw(BackwardError("First input argument must be a dseries object or an empty array!"))
    nobs = size(d.data, 1)
    nobs >= 1 || throw(BackwardError("The dseries object provided as first input argument should at least have 1 periods!"))
    Y = fill(NaN, n, nobs)
    for (i, nm) in enumerate(m.endo_names)
        j = findfirst(==(nm), d.name)
        j === nothing || (Y[i, :] = d.data[:, j])
    end
    X = fill(NaN, nobs, m.n_exo)
    for (i, nm) in enumerate(m.exo_names)
        j = findfirst(==(nm), d.name)
        j === nothing || (X[:, i] = d.data[:, j])
    end
    any(isnan, Y[:, end]) && throw(BackwardError("Please fix the dseries object used for setting the initial conditions!"))
    Y, X
end

"""
    backward_simulate(ws, T; innovations=nothing, initialconditions=nothing, linear=false) -> (endo, exo, errorflag, message)

`endo` is `n_endo x (n0 + T)`, `exo` is `(n0 + T) x n_exo` (the columns of `oo_.endo_simul` and rows of `oo_.exo_simul`).
"""
function backward_simulate(ws, T::Int; innovations=nothing, initialconditions=nothing, linear::Bool=false)
    m = ws.model
    get(m.info, "maximum_lead", 0) == 0 || throw(BackwardError("Model defined in $(m.fname).mod is not backward or static."))
    ml = Int(get(m.info, "maximum_lag", 0))
    ml >= 1 || throw(BackwardError("Model defined in $(m.fname).mod is static. Use simul_static_model instead."))
    ml == 1 || throw(BackwardError("models with a maximum lag above 1 are not supported"))
    nx = m.n_exo
    Y0, X0 = _backward_initial(ws, initialconditions)
    X = innovations === nothing ? backward_innovations(ws, T) : Matrix{Float64}(innovations)
    size(X) == (T, nx) || throw(BackwardError("innovations must be a $T x $nx matrix"))
    _bwd_core(ws, Y0, X0, X, linear)
end

"The simulation proper: `Y0` (n x n0) and `X0` (n0 x n_exo) are the initial periods, `X` (T x n_exo) the innovations."
function _bwd_core(ws, Y0, X0, X, linear::Bool)
    m = ws.model
    n, nx = m.n_endo, m.n_exo
    n0 = size(Y0, 2); T = size(X, 1)
    Y = fill(NaN, n, n0 + T); Y[:, 1:n0] = Y0
    Xall = vcat(X0, X)
    ss = let s = get(ws.oo_, "steady_state", nothing); s isa AbstractArray && length(s) == n ? Float64.(vec(s)) : zeros(n) end
    p = m.params
    if linear
        y = zeros(3n); x = zeros(nx)
        c = dynamic_resid(m, y, x, p, ss)
        J = dynamic_jac(m, y, x, p, ss)
        A0 = J[:, n+1:2n]; A1 = J[:, 1:n]; B = J[:, 3n+1:3n+nx]
        F = try lu(A0) catch; return Y, Xall, true, "the Jacobian of the current variables is singular" end
        for it in n0+1:n0+T
            Y[:, it] = -(F \ (c + A1 * Y[:, it-1] + B * Xall[it, :]))
        end
        return Y, Xall, false, ""
    end
    o = _pfo_options(m, ws.options_, false)
    tolf, tolx, maxit = o["dynatol.f"], o["dynatol.x"], o["simul.maxit"]
    z0 = zeros(n)
    for it in n0+1:n0+T
        ylag = Y[:, it-1]; x = Xall[it, :]
        fun = z -> dynamic_resid(m, vcat(ylag, z, z0), x, p, ss)
        jac = z -> dynamic_jac(m, vcat(ylag, z, z0), x, p, ss)[:, n+1:2n]
        z, ok = try
            r = _dynare_solve(fun, jac, copy(ylag); tolf=tolf, tolx=tolx, maxit=maxit, rng=dynare_rng(ws))
            r[1], r[2]
        catch e
            e isa Union{LinearAlgebra.SingularException,DomainError} || rethrow()
            nothing, false
        end
        if !ok
            return Y[:, 1:it-1], Xall, true, "Nonlinear solver routine failed in period $(it - n0)."
        end
        Y[:, it] = z
    end
    Y, Xall, false, ""
end

function _backward_dseries(ws, Y, X)
    m = ws.model
    no = orig_endo_nbr(m)
    names = vcat(m.endo_names[1:no], m.exo_names)
    N = size(Y, 2)
    data = hcat(Matrix(Y[1:no, :]'), X[1:N, :])
    MDseries(data, names, copy(names), MDates(1, collect(1.0:N)), fill("", length(names)), Pair{String,Vector{Any}}[])
end

"""
    simul_backward_model(ws, periods; innovations=nothing, initialconditions=nothing) -> (dseries, errorflag)

Simulate a backward model `periods` periods; the dseries holds the initial period(s) and the simulation. Fills
`ws.oo_.endo_simul`, `ws.oo_.exo_simul` and (when drawn) `ws.oo_.bnlms.shocks`.
"""
function simul_backward_model(ws::DynareWorkspace, periods::Integer; innovations=nothing, initialconditions=nothing)
    m = ws.model
    linear = _optb(ws, "linear", false)
    Y, X, err, msg = backward_simulate(ws, Int(periods); innovations=innovations, initialconditions=initialconditions, linear=linear)
    err && @warn "simul_backward_model: $msg"
    oo = ws.oo_
    oo.endo_simul = Y
    oo.exo_simul = X
    if innovations === nothing
        b = MStruct(); b.shocks = X[end-Int(periods)+1:end, :]; oo.bnlms = b
    end
    _backward_dseries(ws, Y, X), err
end

"""
    simul_static_model(ws, periods; innovations=nothing) -> dseries

`backward_model.simul_static_model`: in each period the static model is solved from zeros (Newton, algorithm 4);
a failed period is NaN. Fills `oo_.endo_simul`, `oo_.exo_simul`.
"""
function simul_static_model(ws::DynareWorkspace, periods::Integer; innovations=nothing)
    m = ws.model
    get(m.info, "maximum_lag", 0) == 0 || throw(BackwardError("$(m.fname).mod has lagged variables, but it should be a static model."))
    get(m.info, "maximum_lead", 0) == 0 || throw(BackwardError("$(m.fname).mod has leaded variables, but it should be a static model."))
    T = Int(periods); n = m.n_endo
    X = innovations === nothing ? backward_innovations(ws, T) : Matrix{Float64}(innovations)
    o = _pfo_options(m, ws.options_, false)
    Y = fill(NaN, n, T)
    for t in 1:T
        x = X[t, :]
        fun = z -> static_resid(m, z, x, m.params)
        jac = z -> static_jac(m, z, x, m.params)
        r = try _dynare_solve(fun, jac, zeros(n); tolf=o["dynatol.f"], tolx=o["dynatol.x"], maxit=o["simul.maxit"], rng=dynare_rng(ws)) catch e
            e isa Union{LinearAlgebra.SingularException,DomainError} || rethrow(); (nothing, false) end
        r[2] ? (Y[:, t] = r[1]) : @warn "simul_static_model: nonlinear solver failed in period $t"
    end
    ws.oo_.endo_simul = Y; ws.oo_.exo_simul = X
    no = orig_endo_nbr(m); names = vcat(m.endo_names[1:no], m.exo_names)
    MDseries(hcat(Matrix(Y[1:no, :]'), X), names, copy(names), MDates(1, collect(1.0:T)), fill("", length(names)), Pair{String,Vector{Any}}[])
end

"""
    backward_model_irf(ws, shocks; periods=40, initialconditions=nothing, innovationbaseline=nothing, variables=nothing)
        -> (deviations::Dict, baseline::MDseries)

`backward_model.irf` for stochastic shocks given by name: the unit-standard-deviation impulse (a column of the lower
Cholesky factor of `Sigma_e`) is added to the first-period innovations; `deviations[name]` is the dseries of the
differences to the baseline of the chosen variables (default: all the model's endogenous variables), followed by the innovations.
"""
function backward_model_irf(ws::DynareWorkspace, shocks::AbstractVector{<:AbstractString}; periods::Integer=40,
                            initialconditions=nothing, innovationbaseline=nothing, variables=nothing)
    m = ws.model
    get(m.info, "maximum_lead", 0) == 0 || throw(BackwardError("backward_model_irf:: The specified model is not backward looking!"))
    linear = _optb(ws, "linear", false)
    T = Int(periods)
    Y0, X0 = _backward_initial(ws, initialconditions)
    X0 = replace(X0, NaN => 0.0)
    Inn = zeros(T, m.n_exo)
    if innovationbaseline !== nothing
        d = innovationbaseline
        size(d.data, 1) >= T || throw(BackwardError("The second input argument must at least have $T observations or lower the number of periods."))
        for (i, nm) in enumerate(m.exo_names)
            j = findfirst(==(nm), d.name); j === nothing || (Inn[:, i] = d.data[1:T, j])
        end
    end
    any(!iszero, m.Sigma_e) || throw(BackwardError("You did not specify the size of the shocks!"))
    C = get_lower_cholesky_covariance(m.Sigma_e; add_tiny=1e-14)
    no = orig_endo_nbr(m)
    vars = variables === nothing ? m.endo_names[1:no] : String.(variables)
    idv = [something(findfirst(==(v), m.endo_names), 0) for v in vars]
    any(==(0), idv) && throw(BackwardError("backward_model_irf: unknown variable $(vars[findfirst(==(0), idv)])"))
    Yb, _, err, _ = _bwd_core(ws, Y0, X0, Inn, linear)
    err && throw(BackwardError("Simulation failed. Cannot simulate baseline."))
    N = size(Yb, 2); n0 = N - T
    mk(data, names) = MDseries(data, names, copy(names), MDates(1, collect(1.0:size(data, 1))), fill("", length(names)), Pair{String,Vector{Any}}[])
    out = Dict{String,MDseries}()
    for sh in shocks
        j = findfirst(==(sh), m.exo_names)
        j === nothing && throw(BackwardError("backward_model_irf: Exogenous variable $sh is unknown!"))
        inn = copy(Inn); inn[1, :] .+= C[:, j]
        Y1, _, err1, _ = _bwd_core(ws, Y0, X0, inn, linear)
        err1 && (@warn "Simulation failed. Cannot compute IRF for $sh."; continue)
        dev = Matrix((Y1[idv, :] .- Yb[idv, :])')
        # as Dynare: the innovations dseries starts after the initial conditions, the union with the deviations is a merge
        innfull = vcat(fill(NaN, n0, m.n_exo), inn)
        out[sh] = mk(hcat(dev, innfull), vcat(vars, m.exo_names))
    end
    out, mk(hcat(Matrix(Yb[1:no, :]'), vcat(fill(NaN, n0, m.n_exo), Inn)), vcat(m.endo_names[1:no], m.exo_names))
end

"""
    backward_model_forecast(ws, variables; periods=8, initialconditions=nothing, withuncertainty=false, B=1000) -> NamedTuple

`backward_model.forecast`: `pointforecast` (matrix periods+n0 x variables), and with `withuncertainty` the `mean`, `median`,
`std`, `lb` and `ub` (2.5% and 97.5%) over `B` simulations with innovations `(C * randn(n_exo, periods))'`.
"""
function backward_model_forecast(ws::DynareWorkspace, variables; periods::Integer=8, initialconditions=nothing,
                                 withuncertainty::Bool=false, B::Integer=1000)
    m = ws.model
    get(m.info, "maximum_lead", 0) == 0 || throw(BackwardError("backward_model_forecast:: The specified model is not backward looking!"))
    linear = _optb(ws, "linear", false)
    vars = String.(variables); T = Int(periods)
    idy = [something(findfirst(==(v), m.endo_names), 0) for v in vars]
    any(==(0), idy) && throw(BackwardError("backward_model_forecast:: Variable $(vars[findfirst(==(0), idy)]) is unknown!"))
    Y0, X0 = _backward_initial(ws, initialconditions)
    X0 = replace(X0, NaN => 0.0)
    Y, _, err, _ = _bwd_core(ws, Y0, X0, zeros(T, m.n_exo), linear)
    err && throw(BackwardError("Simulation failed."))
    point = Matrix(Y[idy, :]')
    withuncertainty || return (; pointforecast=point, variables=vars)
    C = get_lower_cholesky_covariance(m.Sigma_e; add_tiny=1e-14)
    R = dynare_rng(ws)
    N = size(Y, 2); A = zeros(length(idy), N, B)
    for i in 1:B
        inn = Matrix((C * randn(R, m.n_exo, T))')
        Yi, _, e1, _ = _bwd_core(ws, Y0, X0, inn, linear)
        e1 && throw(BackwardError("Simulation failed."))
        A[:, :, i] = Yi[idy, :]
    end
    sorted = sort(A; dims=3)
    st = dropdims(sqrt.(max.(sum(abs2, A .- mean(A; dims=3); dims=3) ./ B, 0)); dims=3)
    med = dropdims(mapslices(median, A; dims=3); dims=3)
    (; pointforecast=point, variables=vars, mean=Matrix(dropdims(mean(A; dims=3); dims=3)'), median=Matrix(med'), std=Matrix(st'),
       lb=Matrix(sorted[:, :, round(Int, 0.025B)]'), ub=Matrix(sorted[:, :, round(Int, 0.975B)]'))
end

# ---------------------------------------------------------------------------------------------- MATLAB entry point
bi!("simul_backward_model") do I, a, nout
    _nargs(a, 2, 3, "simul_backward_model")
    ws = _pfo_ws(I)
    m = ws.model
    T = Int(round(scalar(a[2], "simul_backward_model")))
    init = a[1]
    (init isa MDseries || (init isa AbstractMatrix && isempty(init))) || merr("First input argument must be a dseries object or an empty array!")
    innov = nothing
    if length(a) == 3
        d = a[3]
        d isa MDseries || merr("Third argument must be a dseries object!")
        size(d.data, 1) >= T || merr("Time span in third argument is too short (should not be less than $T, the value of the second argument)")
        innov = zeros(T, m.n_exo)
        for (i, nm) in enumerate(m.exo_names)
            j = findfirst(==(nm), d.name)
            j === nothing || (innov[:, i] = d.data[1:T, j])
        end
    end
    if Int(get(m.info, "maximum_lag", 0)) == 0 && get(m.info, "maximum_lead", 0) == 0
        return BUILTINS["simul_static_model"](I, length(a) == 3 ? Any[a[2], a[3]] : Any[a[2]], nout)
    end
    sim, err = try
        simul_backward_model(ws, T; innovations=innov, initialconditions=init isa MDseries ? init : nothing)
    catch e
        e isa BackwardError && merr(e.msg)
        rethrow()
    end
    Any[sim, err][1:max(nout, 1)]
end

bi!("simul_backward_nonlinear_model", "simul_backward_linear_model") do I, a, nout
    length(a) in (5, 6) || merr("simul_backward_*_model: expected (initialconditions, samplesize, options_, M_, oo_[, innovations])")
    BUILTINS["simul_backward_model"](I, Any[a[1], a[2]], nout)
end
bi!("simul_static_model") do I, a, nout
    ws = _pfo_ws(I)
    isempty(a) && merr("simul_static_model: samplesize is missing")
    T = Int(round(scalar(a[1], "simul_static_model")))
    innov = nothing
    if length(a) >= 2
        d = a[2]; d isa MDseries || merr("Third argument must be a dseries object!")
        innov = zeros(T, ws.model.n_exo)
        for (i, nm) in enumerate(ws.model.exo_names)
            j = findfirst(==(nm), d.name); j === nothing || (innov[:, i] = d.data[1:T, j])
        end
    end
    try
        simul_static_model(ws, T; innovations=innov)
    catch e
        e isa BackwardError && merr(e.msg)
        rethrow()
    end
end

# ---------------------------------------------------------------------------------------------- set_historical_values, model_inversion
_date_time(d::MDates, i) = d.time[i]

function _date_arg(x, freq)
    x isa MDates && return x.freq, x.time[1]
    x isa AbstractString || throw(BackwardError("the period must be a date"))
    mt = _date_of(String(x))
    mt === nothing && throw(BackwardError("the period must be a date"))
    mt
end

"""
    set_historical_values(ws, ds, initialperiod)

Set `M_.endo_histval` (and `M_.exo_histval`) from the dseries `ds` at `initialperiod` (last initial period); models with
a maximum lag of one and without auxiliary variables for lags.
"""
function set_historical_values(ws::DynareWorkspace, ds::MDseries, initialperiod)
    m = ws.model; M = ws.M_
    ml = Int(get(m.info, "maximum_lag", 0))
    no = orig_endo_nbr(m)
    (no == m.n_endo || all(a -> get(a, "type", 0) != 1, get_or(m.raw, "aux_vars", []))) ||
        throw(BackwardError("set_historical_values: auxiliary variables for lags are not supported"))
    fr, t0 = _date_arg(initialperiod, ds.dates.freq)
    fr == ds.dates.freq || throw(BackwardError("the period and the dseries have different frequencies"))
    row(t) = (k = findfirst(==(t), ds.dates.time); k === nothing ? throw(BackwardError("period outside the dseries")) : k)
    H = zeros(m.n_endo, ml)
    for i in 1:no
        m.lead_lag_incidence[i, 1] > 0 || continue
        j = findfirst(==(m.endo_names[i]), ds.name)
        j === nothing && throw(BackwardError("Can't find $(m.endo_names[i]) in dseries"))
        H[i, ml] = ds.data[row(t0), j]
    end
    M.endo_histval = H
    mex = Int(get(m.info, "maximum_exo_lag", 0))
    XH = zeros(m.n_exo, ml)
    for t in 1:mex, (i, nm) in enumerate(m.exo_names)
        j = findfirst(==(nm), ds.name)
        j === nothing || (XH[i, ml+1-t] = ds.data[row(t0 - t), j])
    end
    M.exo_histval = XH
    nothing
end

"""
    model_inversion(ws, constraints, exogenousvariables, initialconditions) -> (endogenous dseries, exogenous dseries)

`model_inversion.m` for backward and static models: the free innovations (the exogenous variables that are NaN over the
constraint range) are found, period by period, so that the constrained endogenous variables take the values in `constraints`
(`backward_model.inversion`, `static_model_inversion`). Forward-looking models (the deterministic extended-path
inversion) are refused.
"""
function model_inversion(ws::DynareWorkspace, constraints::MDseries, exogenousvariables::MDseries, initialconditions)
    m = ws.model
    n, nx = m.n_endo, m.n_exo
    get(m.info, "maximum_lead", 0) == 0 || throw(BackwardError("model_inversion: forward-looking models are not supported (deterministic inversion with perfect_foresight_solver_core)"))
    static = Int(get(m.info, "maximum_lag", 0)) == 0
    crange = constraints.dates; T = size(constraints.data, 1)
    ex = exogenousvariables
    ex.dates.freq == crange.freq || throw(BackwardError("model_inversion: frequencies differ"))
    rows = [findfirst(==(t), ex.dates.time) for t in crange.time]
    any(isnothing, rows) && throw(BackwardError("model_inversion: the exogenous dseries does not cover the range of the constraints"))
    free = [ex.name[j] for j in 1:length(ex.name) if all(isnan, ex.data[rows, j])]
    length(free) == size(constraints.data, 2) || throw(BackwardError("The number of instruments must be equal to the number of constrained variables!"))
    xcols = [something(findfirst(==(nm), ex.name), 0) for nm in m.exo_names]
    any(==(0), xcols) && throw(BackwardError("model_inversion: exogenous variable missing in the second argument"))
    X = ex.data[:, xcols]
    xfree = length(free) < nx ? [findfirst(==(f), m.exo_names) for f in free] : collect(1:nx)
    if length(free) < n
        yctrl = [findfirst(==(c), m.endo_names) for c in constraints.name]
        any(isnothing, yctrl) && throw(BackwardError("model_inversion: unknown constrained variable"))
        yfree = setdiff(1:n, yctrl)
    else
        yctrl, yfree = collect(1:n), Int[]
    end
    nyfree, nxfree = length(yfree), length(xfree)
    Jid = vcat(n .+ yfree, 3n .+ xfree)
    o = _pfo_options(m, ws.options_, false)
    ss = let s = get(ws.oo_, "steady_state", nothing); s isa AbstractArray && length(s) == n ? Float64.(vec(s)) : zeros(n) end
    off = static ? 0 : 1
    Y = fill(NaN, n, T + off)
    if !static
        (initialconditions isa MDseries) || throw(BackwardError("model_inversion: Third input argument must be a dseries object!"))
        k0 = findfirst(==(crange.time[1] - 1), initialconditions.dates.time)
        k0 === nothing && throw(BackwardError("model_inversion: the initial conditions do not cover the period before the constraints"))
        for (i, nm) in enumerate(m.endo_names)
            j = findfirst(==(nm), initialconditions.name)
            j === nothing || (Y[i, 1] = initialconditions.data[k0, j])
        end
    end
    for (i, c) in enumerate(yctrl)
        Y[c, off+1:end] = constraints.data[:, i]
    end
    algo_note = static ? 4 : _optf(ws, "solve_algo", 4)
    for t in 1:T
        ity = t + off
        ycur = Y[yctrl, ity]
        ylag = static ? zeros(n) : Y[:, ity-1]
        xrow = X[rows[t], :]
        z0 = vcat(static ? Y[yfree, ity] : Y[yfree, ity-1], zeros(nxfree))
        build(z) = begin
            y = vcat(ylag, zeros(2n))
            y[n .+ yctrl] = ycur
            nyfree > 0 && (y[n .+ yfree] = z[1:nyfree])
            x = copy(xrow); x[xfree] = z[nyfree+1:end]
            y, x
        end
        fun = z -> (let (y, x) = build(z); dynamic_resid(m, y, x, m.params, ss) end)
        jac = z -> (let (y, x) = build(z); dynamic_jac(m, y, x, m.params, ss)[:, Jid] end)
        r = try
            _dynare_solve(fun, jac, z0; tolf=o["dynatol.f"], tolx=o["dynatol.x"], maxit=o["simul.maxit"], rng=dynare_rng(ws))
        catch e
            e isa Union{LinearAlgebra.SingularException,DomainError} || rethrow()
            (nothing, false)
        end
        r[2] || throw(BackwardError(static ? "Enable to solve the system of equations" : "Nonlinear solver failed"))
        z = r[1]
        X[rows[t], xfree] = z[nyfree+1:end]
        nyfree > 0 && (Y[yfree, ity] = z[1:nyfree])
    end
    mk(data, names) = MDseries(data, names, copy(names), MDates(crange.freq, copy(crange.time)), fill("", length(names)), Pair{String,Vector{Any}}[])
    mk(Matrix(Y[:, off+1:end]'), copy(m.endo_names)), mk(X[rows, :], copy(m.exo_names))
end

bi!("model_inversion") do I, a, nout
    length(a) == 6 || merr("model_inversion: This routine require six input arguments!")
    a[1] isa MDseries || merr("model_inversion: First input argument must be a dseries object!")
    a[2] isa MDseries || merr("model_inversion: Second input argument must be a dseries object!")
    ws = _pfo_ws(I)
    try
        r = model_inversion(ws, a[1], a[2], a[3])
        Any[r[1], r[2]][1:max(nout, 1)]
    catch e
        e isa BackwardError && merr(e.msg)
        rethrow()
    end
end
bi!("set_historical_values") do I, a, nout
    length(a) == 2 || merr("set_historical_values: two arguments expected")
    ws = _pfo_ws(I)
    ds = a[1]
    ds isa MDseries || merr("set_historical_values: the first argument must be a dseries")
    try
        set_historical_values(ws, ds, a[2])
    catch e
        e isa BackwardError && merr(e.msg)
        rethrow()
    end
    Any[]
end

# dates of an N-row simulation that starts at `init`
_bwd_dates(init::Union{Nothing,MDseries}, N) = init === nothing || isempty(init.dates.time) ? MDates(1, collect(1.0:N)) :
    MDates(init.dates.freq, init.dates.time[1] .+ collect(0.0:N-1))
_with_dates(d::MDseries, dts::MDates) = (d.dates = dts; d)
_bwd_cellstrs(x, what) = x isa Cell ? String[x.items[i] isa AbstractString ? String(x.items[i]) : merr("$what: cell of strings expected") for i in eachindex(x.items)] :
                     merr("$what has to be a cell of strings")

bi!("backward_model_irf") do I, a, nout
    length(a) >= 3 || merr("backward_model_irf: not enough input arguments")
    ws = _pfo_ws(I)
    init = a[1] isa MDseries && !isempty(a[1].data) ? a[1] : nothing
    (init === nothing && !(a[1] isa AbstractMatrix && isempty(a[1])) && !(a[1] isa MDseries)) && merr("backward_model_irf: first argument must be a dseries or []")
    base = a[2] isa MDseries && !isempty(a[2].data) ? a[2] : nothing
    shocks = a[3]
    shocks isa Cell || merr("Third input argument has to be a cell of string or dseries objects!")
    all(x -> x isa AbstractString, shocks.items) || throw(Unsupported("backward_model_irf with deterministic shocks given as dseries", "feature"))
    nout > 2 && throw(Unsupported("backward_model_irf with a third output (irfs)", "feature"))
    vars = length(a) >= 4 && a[4] isa Cell ? _bwd_cellstrs(a[4], "Fourth input argument") : nothing
    periods = length(a) >= 5 && !(a[5] isa AbstractMatrix && isempty(a[5])) ? Int(round(scalar(a[5], "backward_model_irf"))) : 40
    length(a) >= 6 && !(a[6] isa AbstractMatrix && isempty(a[6])) && throw(Unsupported("backward_model_irf with a transformation", "feature"))
    dev, basel = try
        backward_model_irf(ws, _bwd_cellstrs(shocks, "Third input argument"); periods=periods, initialconditions=init, innovationbaseline=base, variables=vars)
    catch e
        e isa BackwardError && merr(e.msg)
        rethrow()
    end
    S = MStruct()
    for sh in _bwd_cellstrs(shocks, "shocks")
        haskey(dev, sh) && (S[sh] = _with_dates(dev[sh], _bwd_dates(init, size(dev[sh].data, 1))))
    end
    _with_dates(basel, _bwd_dates(init, size(basel.data, 1)))
    Any[S, basel][1:max(nout, 1)]
end

bi!("backward_model_forecast") do I, a, nout
    length(a) >= 1 || merr("backward_model_forecast: not enough input arguments")
    ws = _pfo_ws(I)
    a[1] isa MDseries || merr("backward_model_forecast: the first argument must be a dseries")
    init = a[1]
    vars = length(a) >= 2 ? _bwd_cellstrs(a[2], "Second argument") : ws.model.endo_names
    periods = length(a) >= 3 ? Int(round(scalar(a[3], "backward_model_forecast"))) : 8
    unc = length(a) >= 4 ? truth(a[4]) : false
    r = try
        backward_model_forecast(ws, vars; periods=periods, initialconditions=init, withuncertainty=unc)
    catch e
        e isa BackwardError && merr(e.msg)
        rethrow()
    end
    mk(M) = MDseries(M, copy(vars), copy(vars), _bwd_dates(init, size(M, 1)), fill("", length(vars)), Pair{String,Vector{Any}}[])
    S = MStruct()
    S.pointforecast = mk(r.pointforecast)
    S.start = MDates(init.dates.freq, [init.dates.time[end] + 1])
    if unc
        S.meanforecast = mk(r.mean); S.medianforecast = mk(r.median); S.stdforecast = mk(r.std); S.lb = mk(r.lb); S.ub = mk(r.ub)
    end
    S
end
