# Dynare's perfect-foresight functions as MATLAB code in a .mod file calls them:
#
#     oo_ = perfect_foresight_setup(M_, options_, oo_);
#     oo_ = make_ex_(M_, options_, oo_);   oo_ = make_y_(M_, options_, oo_);
#     oo_ = perfect_foresight_solver(M_, options_, oo_);
#     oo_ = perfect_foresight_with_expectation_errors_setup(M_, options_, oo_);
#     oo_ = perfect_foresight_with_expectation_errors_solver(M_, options_, oo_);
#     ys  = evaluate_steady_state(ys_init, exo_ss, M_, options_ [, check_flag]);
#
# (ports of matlab/perfect-foresight-models/*.m of Dynare 7.2, built on the Julia solver of perfect_foresight.jl).
# Unlike the commands run in place (results.jl), these read their inputs from the `oo_` they are given, as
# Dynare does: oo_.endo_simul (initial and terminal conditions and the guess), oo_.exo_simul, oo_.steady_state,
# oo_.exo_steady_state, oo_.initial_steady_state, and, for expectation errors, oo_.pfwee.{terminal_info, shocks_info}.
# The functions are registered as MATLAB builtins (BUILTINS, as pac.jl does); this file is self-contained.
#
# oo_.pfwee.shocks_info is a three-dimensional Julia array (exo x real time x information time): MATLAB lines of the
# file do not index it, only the solver reads it.

# ---------------------------------------------------------------------------------------------- the workspace behind a host
const _PF_HOST_WS = WeakKeyDict{Any,Any}()
"Remember which workspace a .mod-file session (its Host) runs for: dynare() calls this when it builds the workspace."
_pf_register_host!(ws) = (ws.model.session === nothing || (_PF_HOST_WS[ws.model.session.host] = ws); nothing)

function _pfo_ws(I)
    if I.host === nothing && _DRIVER_KW[] !== nothing && CURRENT[] !== nothing      # a run_matlab driver script (src/matlab_driver.jl)
        ws = CURRENT[]
        _driver_sync_in!(I, ws)
        _driver_sync_out!(I, ws)
        return ws
    end
    h = _need_host(I)
    ws = get(_PF_HOST_WS, h, nothing)
    ws === nothing && throw(Unsupported("perfect-foresight functions need the workspace of a running dynare() session", "function"))
    ws
end

_pfo_has(oo::MStruct, k) = haskey(oo, k) && !(oo[k] isa AbstractArray && isempty(oo[k]))
function _pfo_vec(oo::MStruct, k)
    _pfo_has(oo, k) || return nothing
    Float64.(vec(farr(oo[k])))
end
_pfo_need(oo::MStruct, k, what) = (v = _pfo_vec(oo, k); v === nothing ? merr("$what: oo_.$k is empty") : v)

"The options the perfect foresight functions read: options_ of the workspace (MATLAB lines applied), then the fields the caller's copy changed."
function _pfo_options_struct(ws, optarg)
    _apply_matlab_options!(ws)
    O = deepcopy(ws.options_)
    if optarg isa MStruct
        cur = try
            snapshot(ws.model.session.host.options)
        catch
            nothing
        end
        for k in keys(optarg)
            (cur isa MStruct && haskey(cur, k) && isequal(cur[k], optarg[k])) && continue
            v = optarg[k]
            v isa MStruct && get(O, k, nothing) isa MStruct ? _merge_struct!(O[k], v) : (O[k] = deepcopy(v))
        end
    end
    O
end

"Normalised options (as `perfect_foresight` builds them) from the flattened options_ struct."
function _pfo_options(m::Model, O::MStruct, pfwee::Bool)
    flat = _flat_options(O)
    o = copy(PF_DEFAULTS)
    given = Set{String}()
    for k in keys(PF_DEFAULTS)
        haskey(flat, k) && (o[k] = flat[k]; push!(given, k))
    end
    _pf_normalize_options!(m, o, given)
    o["linear"] = _pf_model_is_linear(m)
    try
        o["lmmcp.tolfun"] = Float64(scalar(m.matlab_options["lmmcp"]["TolFun"], "options_.lmmcp.TolFun"))
    catch
    end
    haskey(flat, "lmmcp.maxiter") && (o["lmmcp.maxiter"] = Int(flat["lmmcp.maxiter"]))
    has_complementarity(m) && !o["lmmcp.status"] &&
        warn_once(m, "the model has complementarity conditions but perfect_foresight_solver is used without the lmmcp option: " *
                     "as in Dynare the conditions are ignored and the unconstrained path is computed (pass lmmcp=true to impose them)")
    o
end

"get_simulation_periods.m (without dates)."
function _pfo_periods(O::MStruct)
    haskey(O, "periods") || merr("Options first_simulation_period and last_simulation_period must be set when periods is not provided")
    p = scalar(O["periods"], "options_.periods")
    p > 0 || throw(Unsupported("options_.periods = 0: first_simulation_period / last_simulation_period (dates) are not supported by JDyn's perfect foresight", "feature"))
    Int(p)
end

"A PFWorkspace (as `perfect_foresight_setup` finds Dynare's state) whose steady states are those of `oo`."
function _pfo_workspace(m::Model, oo::MStruct, params)
    pws = _pf_replay(m, params, nothing, (), ())
    ss, xss = _pfo_vec(oo, "steady_state"), _pfo_vec(oo, "exo_steady_state")
    ss === nothing || (length(ss) == m.n_endo || merr("oo_.steady_state has $(length(ss)) elements, the model has $(m.n_endo) endogenous variables"); pws.ss = ss)
    xss === nothing || (length(xss) == m.n_exo || merr("oo_.exo_steady_state has $(length(xss)) elements, the model has $(m.n_exo) exogenous variables"); pws.xss = xss)
    pws.ss0, pws.xss0 = _pfo_vec(oo, "initial_steady_state"), _pfo_vec(oo, "initial_exo_steady_state")
    (pws.ss0 === nothing) == (pws.xss0 === nothing) || (pws.ss0 === nothing ? (pws.xss0 = nothing) : (pws.xss0 = copy(pws.xss)))
    pws
end

_pfo_params(m::Model, Marg) = (Marg isa MStruct && haskey(Marg, "params")) ? Float64.(vec(farr(Marg["params"]))) : copy(m.params)

# ---------------------------------------------------------------------------------------------- make_ex_, make_y_, setup
function _pfo_make_paths!(m::Model, O::MStruct, oo::MStruct, params; ex::Bool=true, y::Bool=true)
    T = _pfo_periods(O)
    pws = _pfo_workspace(m, oo, params)
    saved = pws.det_shocks
    y && !ex && (pws.det_shocks = [])
    ys, xs = _pf_make_paths(m, pws, T)
    pws.det_shocks = saved
    ex && (oo["exo_simul"] = Matrix{Float64}(permutedims(xs)))
    y && (oo["endo_simul"] = Matrix{Float64}(ys))
    oo
end

function _pfo_setup!(m::Model, O::MStruct, oo::MStruct, params)
    T = _pfo_periods(O)
    o = _pfo_options(m, O, false)
    o["simul.endval_steady"] && m.info["maximum_lead"] == 0 &&
        merr("PERFECT_FORESIGHT_SETUP: Option endval_steady cannot be used on a purely backward or static model.")
    ds = get(oo, "deterministic_simulation", MStruct())
    oo["deterministic_simulation"] = ds
    haskey(ds, "controlled_paths_by_period") || (ds["controlled_paths_by_period"] = zeros(0, 0))
    _pfo_make_paths!(m, O, oo, params)
end

# ---------------------------------------------------------------------------------------------- perfect_foresight_solver
"Stores a solver outcome in oo_ as perfect_foresight_solver.m does."
function _pfo_store!(oo::MStruct, out, m::Model, exo_t::Bool=true)
    ds = get(oo, "deterministic_simulation", MStruct())
    ds["status"] = out["status"] === true
    ds["error"] = Float64(out["maxerror"])
    ds["homotopy_completion_share"] = Float64(out["share"])
    ds["homotopy_iterations"] = Float64(out["homotopy_iterations"])
    out["iterations"] === nothing || (ds["iterations"] = Float64(out["iterations"]))
    oo["deterministic_simulation"] = ds
    oo["endo_simul"] = Matrix{Float64}(out["endo"])
    if out["status"] === true
        out["recompute"] === true && (oo["steady_state"] = _col(out["ss"]))
        get(out, "cp", false) === true && (oo["exo_simul"] = Matrix{Float64}(permutedims(out["exo"])))
    else
        oo["exo_simul"] = Matrix{Float64}(permutedims(out["exo"]))
        oo["steady_state"] = _col(out["ss"])
        oo["exo_steady_state"] = _col(out["xss"])
    end
    nothing
end

function _pfo_solve!(m::Model, O::MStruct, oo::MStruct, params)
    T = _pfo_periods(O)
    o = _pfo_options(m, O, false)
    lag, lead = m.info["maximum_lag"], m.info["maximum_lead"]
    rows = lag + T + lead
    (haskey(oo, "endo_simul") && haskey(oo, "exo_simul")) || merr("perfect_foresight_solver: oo_.endo_simul and oo_.exo_simul are not set (run perfect_foresight_setup first)")
    endo, exo = Matrix{Float64}(farr(oo["endo_simul"])), Matrix{Float64}(permutedims(farr(oo["exo_simul"])))
    size(endo) == (m.n_endo, rows) ||
        merr("perfect_foresight_solver: oo_.endo_simul has size $(size(endo)), expected ($(m.n_endo), $rows) for options_.periods = $T")
    size(exo) == (m.n_exo, rows) ||
        merr("perfect_foresight_solver: oo_.exo_simul has size $(reverse(size(exo))), expected ($rows, $(m.n_exo)) for options_.periods = $T")
    ss, xss = _pfo_need(oo, "steady_state", "perfect_foresight_solver"), _pfo_vec(oo, "exo_steady_state")
    xss === nothing && (xss = zeros(m.n_exo))
    ss0, xss0 = _pfo_vec(oo, "initial_steady_state"), _pfo_vec(oo, "initial_exo_steady_state")
    (ss0 === nothing) == (xss0 === nothing) || (ss0 === nothing ? (xss0 = nothing) : (xss0 = copy(xss)))
    sim = PFSim(m, params, o, endo, exo, ss, xss, ss0, xss0, false)
    sim.cp = _pf_cp_by_period(m, _pf_replay(m, params, nothing, (), ()), T)
    out = _pf_solve(sim, T)
    _pfo_store!(oo, out, m)
    out["solver"] = sim.solver
    out
end

# ---------------------------------------------------------------------------------------------- expectation errors
"perfect_foresight_with_expectation_errors_setup.m: oo_.pfwee and the initial guess."
function _pfo_pfwee_setup!(m::Model, O::MStruct, oo::MStruct, params)
    T = _pfo_periods(O)
    o = _pfo_options(m, O, true)
    pws = _pfo_workspace(m, oo, params)
    datafile = o["datafile"]
    datafile isa AbstractString && isempty(datafile) && (datafile = nothing)
    shocks, terminal = _pf_pfwee_information(m, pws, T, datafile)
    info = MStruct()
    info["terminal_info"] = terminal
    info["shocks_info"] = shocks
    info["controlled_paths_by_period"] = isempty(pws.cpaths) ? nothing : [_pf_cp_by_period(m, pws, T, ip) for ip in 1:T]
    oo["pfwee"] = info
    pws.det_shocks = []                       # the setup command builds the paths without the shocks
    endo, x = _pf_make_paths(m, pws, T)
    oo["endo_simul"] = Matrix{Float64}(endo)
    oo["exo_simul"] = Matrix{Float64}(permutedims(x))
    ds = get(oo, "deterministic_simulation", MStruct())
    haskey(ds, "controlled_paths_by_period") || (ds["controlled_paths_by_period"] = zeros(0, 0))
    oo["deterministic_simulation"] = ds
    oo
end

"perfect_foresight_with_expectation_errors_solver.m on `oo`."
function _pfo_pfwee_solve!(m::Model, O::MStruct, oo::MStruct, params; verbose::Bool=false)
    T = _pfo_periods(O)
    o = _pfo_options(m, O, true)
    lag, lead = m.info["maximum_lag"], m.info["maximum_lead"]
    (haskey(oo, "pfwee") && oo["pfwee"] isa MStruct && haskey(oo["pfwee"], "shocks_info")) ||
        merr("perfect_foresight_with_expectation_errors_solver: oo_.pfwee is not set (run perfect_foresight_with_expectation_errors_setup first)")
    shocks, terminal = oo["pfwee"]["shocks_info"], oo["pfwee"]["terminal_info"]
    size(terminal) == (m.n_exo, T) ||
        merr("perfect_foresight_with_expectation_errors_solver: oo_.pfwee was built for another number of periods than options_.periods = $T")
    const_len = o["pfwee.constant_simulation_length"]
    endo_simul, exo_simul = Matrix{Float64}(farr(oo["endo_simul"])), Matrix{Float64}(permutedims(farr(oo["exo_simul"])))
    size(endo_simul, 2) == lag + T + lead && size(exo_simul, 2) == lag + T + lead ||
        merr("perfect_foresight_with_expectation_errors_solver: oo_.endo_simul or oo_.exo_simul does not have maximum_lag + periods + maximum_lead rows")
    lead > 0 && (o["simul.endval_steady"] = true)
    ss = _pfo_need(oo, "steady_state", "perfect_foresight_with_expectation_errors_solver")
    xss = _pfo_vec(oo, "exo_steady_state")
    xss === nothing && (xss = zeros(m.n_exo))
    ss0, xss0 = _pfo_vec(oo, "initial_steady_state"), _pfo_vec(oo, "initial_exo_steady_state")
    if ss0 === nothing || xss0 === nothing
        ss0, xss0 = copy(ss), copy(xss)
        oo["initial_steady_state"] = _col(ss0)
        oo["initial_exo_steady_state"] = _col(xss0)
    end
    cps = oo["pfwee"]["controlled_paths_by_period"]
    cps === nothing || (const_len && merr("Options constant_simulation_length is incompatible with perfect_foresight_controlled_paths or shock_paths+exogenize block"))
    info_period, increment, share0 = 1, 0, nothing
    out = nothing
    while info_period <= T
        p = info_period - 1
        xss = terminal[:, info_period]
        if const_len && increment > 0
            endo_simul = hcat(endo_simul, repeat(ss, 1, increment))
            exo_simul = hcat(exo_simul, fill(NaN, m.n_exo, increment))
        end
        endo = endo_simul[:, p+1:end]
        exo = exo_simul[:, p+1:end]
        sim_length = const_len ? T : T - p
        known = T - p
        exo[:, lag+1:lag+known] = shocks[:, info_period:T, info_period]
        exo[:, lag+known+1:end] .= xss
        sim = PFSim(m, params, o, endo, exo, ss, xss, ss0, xss0, verbose)
        cps === nothing || (sim.cp = cps[info_period][info_period:end])
        out = _pf_solve(sim, sim_length)
        out["solver"] = sim.solver
        if !out["status"]
            throw(PerfectForesightError("perfect_foresight_with_expectation_errors_solver: failed to compute the solution " *
                                        "for information available at period $info_period", out))
        end
        out["marginal"] && throw(ModelError("marginal linearization together with expectation errors is not implemented"))
        if share0 === nothing
            share0 = out["share"]
            o["simul.homotopy_max_completion_share"] = share0
        elseif out["share"] != share0
            throw(PerfectForesightError("could not find a solution for information available at period $info_period " *
                                        "with the same homotopy completion share as period 1"))
        end
        ss = out["ss"]
        endo_simul[:, p+1:end] = out["endo"]
        exo_simul[:, p+1:end] = out["exo"]
        increment = 1
        while info_period + increment <= T && terminal[:, info_period] == terminal[:, info_period+increment] &&
              isequal(shocks[:, info_period+increment:T, info_period], shocks[:, info_period+increment:T, info_period+increment]) &&
              (cps === nothing || cps[info_period][info_period+increment:end] == cps[info_period+increment][info_period+increment:end])
            increment += 1
        end
        info_period += increment
    end
    out["endo"], out["exo"] = endo_simul, exo_simul
    _pfo_store!(oo, out, m)
    oo["endo_simul"] = Matrix{Float64}(endo_simul)
    oo["exo_simul"] = Matrix{Float64}(permutedims(exo_simul))
    oo["steady_state"] = _col(ss)
    oo["exo_steady_state"] = _col(xss)
    out
end

# ---------------------------------------------------------------------------------------------- MATLAB entry points
function _pfo_args(I, a, name)
    _nargs(a, 3, 4, name)
    ws = _pfo_ws(I)
    oo = a[3]
    oo isa MStruct || merr("$name: the third argument must be oo_")
    ws, deepcopy(oo), _pfo_options_struct(ws, a[2]), _pfo_params(ws.model, a[1])
end

function _register_pf_functions!()
    bi!("make_ex_") do I, a, nout
        ws, oo, O, params = _pfo_args(I, a, "make_ex_")
        Any[_pfo_make_paths!(ws.model, O, oo, params; ex=true, y=false)]
    end
    bi!("make_y_") do I, a, nout
        ws, oo, O, params = _pfo_args(I, a, "make_y_")
        Any[_pfo_make_paths!(ws.model, O, oo, params; ex=false, y=true)]
    end
    bi!("perfect_foresight_setup") do I, a, nout
        ws, oo, O, params = _pfo_args(I, a, "perfect_foresight_setup")
        Any[_pfo_setup!(ws.model, O, oo, params)]
    end
    bi!("perfect_foresight_solver") do I, a, nout
        length(a) > 3 && throw(Unsupported("perfect_foresight_solver with the marginal_linearization_previous_raw_sims argument", "feature"))
        nout > 1 && throw(Unsupported("perfect_foresight_solver with a second output (the dseries of the simulation)", "feature"))
        ws, oo, O, params = _pfo_args(I, a, "perfect_foresight_solver")
        _pfo_solve!(ws.model, O, oo, params)
        Any[oo]
    end
    bi!("perfect_foresight_with_expectation_errors_setup") do I, a, nout
        ws, oo, O, params = _pfo_args(I, a, "perfect_foresight_with_expectation_errors_setup")
        Any[_pfo_pfwee_setup!(ws.model, O, oo, params)]
    end
    bi!("perfect_foresight_with_expectation_errors_solver") do I, a, nout
        nout > 1 && throw(Unsupported("perfect_foresight_with_expectation_errors_solver with a second output (the dseries of the simulation)", "feature"))
        ws, oo, O, params = _pfo_args(I, a, "perfect_foresight_with_expectation_errors_solver")
        try
            _pfo_pfwee_solve!(ws.model, O, oo, params)
        catch e
            e isa PerfectForesightError && merr(e.msg)
            rethrow()
        end
        Any[oo]
    end
    bi!("evaluate_steady_state") do I, a, nout
        _nargs(a, 4, 5, "evaluate_steady_state")
        ws = _pfo_ws(I)
        m = ws.model
        size(farr(a[1]), 2) > 1 && merr("ys_init must be a column-vector")
        size(farr(a[2]), 2) > 1 && merr("exo_ss must be a column-vector")
        guess = Float64.(vec(farr(a[1])))
        length(guess) == m.n_endo || merr("evaluate_steady_state: ys_init has $(length(guess)) elements, the model has $(m.n_endo) endogenous variables")
        x = Float64.(vec(farr(a[2])))
        params = _pfo_params(m, a[3])
        ys, p, info = try
            ys, p = _pf_steady(m, params, guess, x; tol=DYNARE_TOLF, check=true)
            ys, p, 0.0
        catch e
            e isa Union{SteadyStateError,ModelError} || rethrow()
            guess, params, 20.0
        end
        Any[ret(_col(ys)), ret(_col(p)), info][1:max(nout, 1)]
    end
    nothing
end
_register_pf_functions!()
