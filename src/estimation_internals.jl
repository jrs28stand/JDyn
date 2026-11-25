# Dynare's estimation internals that .mod files call from MATLAB lines (kalman/lik_init/*.mod):
#   [dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] =
#       dynare_estimation_init(var_list, fname, [], M_, options_, oo_, estim_params_, bayestopt_);
#   oo_ = initial_estimation_checks(str2func('dsge_likelihood'), xparam1, dataset_, dataset_info, M_, estim_params_, options_,
#                                   bayestopt_, bounds, oo_);
# JDyn does not build Dynare's layout of estim_params_, bayestopt_, dataset_, dataset_info and bounds: they are opaque values that can
# only be passed on to these two routines; bayestopt_ has the index fields of the restricted state space (mf0, mf1, mf2, mfys,
# smoother_var_list, smoother_mf, smoother_restrict_columns, smoother_saved_var_list), every other use is refused.

"A Dynare global JDyn does not build in Dynare's layout; `fields` are computed on access."
struct DynareOpaque <: HostStruct
    name::String
    why::String
    fields::Dict{String,Function}
end
DynareOpaque(name, why) = DynareOpaque(name, why, Dict{String,Function}())
has_field(o::DynareOpaque, name) = haskey(o.fields, name)
field_names(o::DynareOpaque) = sort!(collect(keys(o.fields)))
snapshot(o::DynareOpaque) = o
function get_field(o::DynareOpaque, name)
    haskey(o.fields, name) && return o.fields[name]()
    throw(Unsupported("$(o.name).$name: JDyn does not build Dynare's $(o.name) ($(o.why))", "results"))
end

const _EST_WHY = "it can only be passed to dynare_estimation_init and initial_estimation_checks"
const _BT_FIELDS = ("mf0", "mf1", "mf2", "mfys", "smoother_var_list", "smoother_mf", "smoother_restrict_columns", "smoother_saved_var_list")

"The restricted state space of dynare_estimation_init.m (decision-rule order, 1-based): observed and predetermined variables."
function _est_layout(est::Estimation)
    m = est.model
    isempty(m.varobs) && throw(Unsupported("bayestopt_: the model has no varobs", "results"))
    _est_flag(get(est.options, "selected_variables_only", false)) && throw(Unsupported("bayestopt_ with selected_variables_only", "results"))
    order_var, _, cnt = dr_ordering(m.lead_lag_incidence)
    inv = invperm(order_var)
    obs_decl = Int[m.ie[v] for v in m.varobs]
    obs_dr = inv[obs_decl]
    state_dr = collect(cnt.nstatic .+ (1:(cnt.npred + cnt.nboth)))
    k2 = sort(union(obs_dr, state_dr))
    (; k2, obs_decl, obs_dr, state_dr, n=m.n_endo)
end

function _bayestopt(est::Estimation)
    L = Ref{Any}(nothing)
    layout() = (L[] === nothing && (L[] = _est_layout(est)); L[])
    col(v) = ret(reshape(Float64.(v), :, 1))
    pos(k2, v) = [findfirst(==(x), k2) for x in v]
    f = Dict{String,Function}(
        "mf0" => () -> (l = layout(); col(pos(l.k2, l.state_dr))),
        "mf1" => () -> (l = layout(); col(pos(l.k2, l.obs_dr))),
        "mf2" => () -> col(layout().obs_dr),
        "mfys" => () -> col(layout().obs_decl),
        "smoother_var_list" => () -> col(1:layout().n),
        "smoother_mf" => () -> col(layout().obs_dr),
        "smoother_restrict_columns" => () -> col(layout().state_dr),
        "smoother_saved_var_list" => () -> col(1:layout().n))
    DynareOpaque("bayestopt_", "JDyn builds only the fields " * join(_BT_FIELDS, ", "), f)
end

"Puts the opaque globals `estim_params_` and `bayestopt_` into the .mod file's workspace (Dynare declares them as globals)."
function _install_estimation_globals!(ws)
    sess = ws.model.session
    sess === nothing && return
    sess.sc.vars["estim_params_"] = DynareOpaque("estim_params_", _EST_WHY)
    sess.sc.vars["bayestopt_"] = DynareOpaque("bayestopt_", "JDyn builds only the fields " * join(_BT_FIELDS, ", "))
end

function _register_estimation_internals!()
    bi!("dynare_estimation_init") do I, a, nout
        length(a) == 8 && a[7] isa DynareOpaque && a[8] isa DynareOpaque && a[7].name == "estim_params_" && a[8].name == "bayestopt_" &&
            a[4] isa MStruct && a[5] isa MStruct && a[6] isa MStruct ||
            _unsup("dynare_estimation_init(var_list, fname, [], M_, options_, oo_, estim_params_, bayestopt_) with the Dynare globals as arguments is the only form implemented", "function")
        ws = _pfo_ws(I)
        _apply_matlab_options!(ws)
        _model_to_M!(ws)
        m = ws.model
        est = _with_options(ws, _SOLVE_KINDS) do
            Estimation(m; statement=0, _estimation_kwargs(ws, ())...)
        end
        ws.state[:est_init] = est
        # dynare_estimation_init.m: the parameters at their initial values, the steady state and decision rules there, oo_.prior
        oo = a[6]               # the copy of oo_ the call received: the assignment of the outputs puts it back into oo_
        # (the steady state stored is the one at the calibrated parameters of M_: Dynare's reference for fs2000 agrees with that)
        ys = try
            first(steady_state(m))
        catch e
            e isa SteadyStateError || rethrow()
            throw(ModelError("dynare_estimation_init: the steady state of the model at its calibrated parameters could not be computed ($(e.msg)); JDyn refuses rather than leave oo_.steady_state unset"))
        end
        oo.steady_state = _col(ys)
        ws.options_.noconstant = 0.0; a[5].noconstant = 0.0
        _store_prior!(oo, est)
        _store_dr_estimation!(oo, m)
        Any[DynareOpaque("dataset_", _EST_WHY), DynareOpaque("dataset_info", _EST_WHY), ret(reshape(Float64.(est.x0), :, 1)), zeros(0, 0),
            a[4], a[5], a[6], a[7], _bayestopt(est), DynareOpaque("bounds", _EST_WHY)][1:max(nout, 1)]
    end
    bi!("initial_estimation_checks") do I, a, nout
        length(a) == 10 && a[3] isa DynareOpaque && a[4] isa DynareOpaque && a[6] isa DynareOpaque && a[8] isa DynareOpaque && a[9] isa DynareOpaque ||
            _unsup("initial_estimation_checks(fun, xparam1, dataset_, dataset_info, M_, estim_params_, options_, bayestopt_, bounds, oo_) with the values dynare_estimation_init returned is the only form implemented", "function")
        ws = _pfo_ws(I)
        est = get(ws.state, :est_init, nothing)
        est === nothing && merr("initial_estimation_checks: dynare_estimation_init has not been called")
        initial_estimation_checks(est)
        a[10]
    end
    nothing
end
_register_estimation_internals!()
