# Optimal-policy commands on a workspace (M_, options_, oo_): ramsey_model, ramsey_policy, discretionary_policy,
# evaluate_planner_objective, osr, as Dynare's driver runs them (src/optimal_policy.jl holds the numerics).

"`ramsey_model` writes options_.ramsey_policy and options_.instruments; nothing is computed."
function _ramsey_model_statement!(ws, st)
    ws.options_.ramsey_policy = true
    ws.options_.discretionary_policy = false
    names = _instrument_names(ws.model)
    ws.options_.instruments = Cell(Any[n for n in names], (length(names), 1))
    nothing
end

"oo_.dr as the Dict the numerical code takes."
function _dr_from_oo(ws)
    m = ws.model
    haskey(ws.oo_, "dr") && haskey(ws.oo_.dr, "ghx") && haskey(ws.oo_.dr, "ys") ||
        throw(OptimalPolicyError("evaluate_planner_objective requires decision rules to have previously been computed (e.g. by stoch_simul or discretionary_policy)"))
    D = ws.oo_.dr
    d = Dict{String,Any}("ys" => vec(Float64.(D.ys)), "order_var" => Int.(vec(D.order_var)), "state_var" => Int.(vec(D.state_var)),
                         "params" => copy(m.params))
    for k in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss")
        haskey(D, k) && (d[k] = Matrix{Float64}(D[k]))
    end
    haskey(D, "ghs2") && (d["ghs2"] = vec(Float64.(D.ghs2)))
    d
end

function _pov_struct(v::PlannerObjectiveValue)
    s = MStruct()
    s.unconditional = v.unconditional
    c = getfield(v, :conditional)
    if c isa Float64
        s.conditional = c
    else
        cs = MStruct()
        cs.zero_initial_multiplier = c.zero_initial_multiplier
        cs.steady_initial_multiplier = c.steady_initial_multiplier
        s.conditional = cs
    end
    s
end

"""
    evaluate_planner_objective!(ws=current workspace; options...)

Dynare's `evaluate_planner_objective` on the decision rules in `oo_.dr`: fills `oo_.planner_objective_value`. The shocks of the first
period are `oo_.exo_simul(1,:)` (a `stoch_simul` with periods > 0), the deterministic shocks of the file, or the exogenous steady
state; the initial state `M_.endo_histval` when set. After a perfect-foresight simulation of a Ramsey model the welfare of the path.
`replicate_dynare_bugs` (options_) reproduces Dynare 7.2's order-2 `Wyu` ordering defect (see src/optimal_policy.jl).
"""
function evaluate_planner_objective!(ws::DynareWorkspace; options...)
    m = _sync_model!(ws)
    _take_options!(ws, options)
    oo = ws.oo_
    if is_ramsey(m) && get(ws.state, :ran_pf, false) == true
        v = evaluate_planner_objective_perfect_foresight(m, Matrix{Float64}(oo.endo_simul); params=m.params)
        oo.planner_objective_value = _pov_struct(v)
        return ws
    end
    dr = _dr_from_oo(ws)
    disc = is_discretionary(m)
    order = disc ? 1 : _opti(ws, "order", 2)
    u = nothing
    if isempty(m.det_shocks) && haskey(oo, "exo_simul") && !isempty(oo.exo_simul)
        u = vec(Float64.(oo.exo_simul[1, :]))
    end
    hv = ws.M_.endo_histval
    bugs = _optb(ws, "replicate_dynare_bugs", false)
    v = _with_options(ws, _SOLVE_KINDS) do
        evaluate_planner_objective(m, dr; order=order, u=u, histval=isempty(hv) ? nothing : vec(Float64.(hv[:, end])),
                                   replicate_dynare_bugs=bugs, print_results=!_optb(ws, "noprint", false))
    end
    oo.planner_objective_value = _pov_struct(v)
    ws
end

"""
    ramsey_policy!(ws=current workspace, var_list=[]; options...)

`ramsey_policy` (= `ramsey_model` + `stoch_simul` + `evaluate_planner_objective`) at the current calibration.
"""
function ramsey_policy!(ws::DynareWorkspace, var_list=String[]; rng=nothing, options...)
    is_ramsey(ws.model) || throw(OptimalPolicyError("the model has no ramsey_model / ramsey_policy statement"))
    ws.options_.ramsey_policy = true
    stoch_simul!(ws, var_list; rng=rng, options...)
    ws.oo_.steady_state = ws.oo_.dr.ys
    evaluate_planner_objective!(ws)
end

"""
    discretionary_policy!(ws=current workspace, var_list=[]; options...)

`discretionary_policy`: the discretionary rules (order 1) into `oo_.dr`, then `stoch_simul`'s moments, IRFs and simulation on them, and
the welfare in `oo_.planner_objective_value`.
"""
function discretionary_policy!(ws::DynareWorkspace, var_list=String[]; rng=nothing, options...)
    m = _sync_model!(ws)
    is_discretionary(m) || throw(OptimalPolicyError("the .mod file has no discretionary_policy statement"))
    _take_options!(ws, options)
    ws.options_.discretionary_policy = true
    res = _with_options(() -> discretionary_policy(m; welfare=false), ws, _SOLVE_KINDS)
    dr = res.dr
    _, _, cnt = dr_ordering(m.lead_lag_incidence)
    nsp = cnt.npred + cnt.nboth
    Gy = Matrix{Float64}(dr["ghx"])[cnt.nstatic+1:cnt.nstatic+nsp, :]
    eig = nsp == 0 ? ComplexF64[] : eigvals(Gy)
    dr["params"] = copy(m.params)
    ws.state[:solved_dr] = (dr, copy(m.params), eig)
    order0 = get(ws.options_, "order", nothing)
    ws.options_.order = 1.0
    try
        _with_options(() -> _stoch_simul(ws, m, ws.oo_, var_list, nothing, rng), ws, _SOLVE_KINDS)
    finally
        order0 === nothing || (ws.options_.order = order0)
    end
    ws.oo_.steady_state = ws.oo_.dr.ys
    evaluate_planner_objective!(ws)
end

"""
    osr!(ws=current workspace, var_list=[]; options...)

`osr`: finds the optimal parameters of `osr_params`, sets them in `M_.params`, runs `stoch_simul` at the optimum and stores
`oo_.osr` (`objective_function`, `optim_params`, `optimization_info`, `error_indicator`).
"""
function osr!(ws::DynareWorkspace, var_list=String[]; rng=nothing, options...)
    m = _sync_model!(ws)
    _take_options!(ws, options)
    flat = _flat_options(ws.options_)
    opts = Dict{String,Any}(k => v for (k, v) in flat if k == "order" || k == "optim_opt" || k == "analytic_derivation" || startswith(k, "osr."))
    haskey(opts, "order") || (opts["order"] = 1)
    r = osr(m; options=opts, rng=rng, apply=true, replicate_dynare_bugs=_optb(ws, "replicate_dynare_bugs", false))
    _store_params!(ws, m.params)
    names = _osr_param_names(m)
    S = MStruct()
    S.param_names = _cellcol(names)
    S.param_indices = _col([m.ip[n] for n in names])
    ws.M_.osr = S
    stoch_simul!(ws, var_list; rng=rng)
    o = MStruct()
    o.error_indicator = Float64(r.error_indicator)
    o.objective_function = r.objective_function
    op = MStruct()
    for (k, v) in r.optim_params
        op[k] = v
    end
    o.optim_params = op
    info = MStruct()
    info.exitflag = Float64(r.exitflag)
    info.iterations = Float64(r.iterations)
    info.funcCount = Float64(r.func_count)
    info.message = r.message
    o.optimization_info = info
    ws.oo_.osr = o
    ws
end

for f in (:ramsey_policy!, :discretionary_policy!, :osr!, :evaluate_planner_objective!)
    @eval $f(; kw...) = $f(_current(); kw...)
end
ramsey_policy!(var_list::AbstractVector; kw...) = ramsey_policy!(_current(), var_list; kw...)
discretionary_policy!(var_list::AbstractVector; kw...) = discretionary_policy!(_current(), var_list; kw...)
osr!(var_list::AbstractVector; kw...) = osr!(_current(), var_list; kw...)
