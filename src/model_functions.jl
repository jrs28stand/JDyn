# The model's own generated functions, called from MATLAB lines of the .mod file as `<fname>.dynamic_resid(...)`,
# `<fname>.dynamic_g1(...)`, `<fname>.static_resid(...)`, `<fname>.static_g1(...)` and `<fname>.set_auxiliary_variables(...)`
# (Dynare 7.2's +<fname> package). They run the Julia functions the preprocessor generated for the model (src/model.jl), with the
# MATLAB calling convention: `[residual, T_order, T] = f.dynamic_resid(y, x, params, steady_state)` where `y` stacks the lagged,
# current and led endogenous variables (3 * endo_nbr) and `x` is the (one-period) row of exogenous variables; `dynamic_g1` returns
# the Jacobian in Dynare's compressed column layout (the dynamic variables present in the lead/lag incidence matrix, then the
# exogenous variables) as a full matrix (the interpreter has no sparse type). T_order and T are empty: they only carry the
# temporary terms from `dynamic_resid` to `dynamic_g1`, which recomputes them.

function _mf_vec(v, n, what)
    x = Float64.(vec(farr(v)))
    length(x) == n || merr("$what: the argument must have $n elements (has $(length(x)))")
    x
end

function _mf_x(m, v, what)
    x = farr(v)
    size(x, 1) <= 1 || (size(x, 2) == 1 && size(x, 1) == m.n_exo) || _unsup("$what: exogenous variables over several periods", "function")
    _mf_vec(v, m.n_exo + n_exo_det(m), what)
end

function _mf_columns(m)
    lli = m.lead_lag_incidence
    cols = Int[]
    for t in 1:3, i in 1:m.n_endo
        lli[i, t] != 0 && push!(cols, (t - 1) * m.n_endo + i)
    end
    append!(cols, 3m.n_endo .+ (1:(m.n_exo + n_exo_det(m))))
    cols
end

function _register_model_functions!()
    reg(f, name) = (PKG_FUNCS[name] = f)
    reg("*.dynamic_resid") do I, a, nout
        m = _need_host(I).model
        length(a) == 4 || merr("Invalid call to dynamic_resid")
        r = dynamic_resid(m, _mf_vec(a[1], 3m.n_endo, "dynamic_resid"), _mf_x(m, a[2], "dynamic_resid"), _mf_vec(a[3], length(m.params), "dynamic_resid"),
                          _mf_vec(a[4], m.n_endo, "dynamic_resid"))
        Any[ret(reshape(r, :, 1)), zeros(0, 1), zeros(0, 1)][1:max(nout, 1)]
    end
    reg("*.dynamic_g1") do I, a, nout
        m = _need_host(I).model
        length(a) in (4, 9) || merr("Invalid call to dynamic_g1")
        n = m.n_endo
        J = dynamic_jac(m, _mf_vec(a[1], 3n, "dynamic_g1"), _mf_x(m, a[2], "dynamic_g1"), _mf_vec(a[3], length(m.params), "dynamic_g1"),
                        _mf_vec(a[4], n, "dynamic_g1"))
        ret(Matrix{Float64}(J[:, _mf_columns(m)]))
    end
    reg("*.static_resid") do I, a, nout
        m = _need_host(I).model
        length(a) == 3 || merr("Invalid call to static_resid")
        r = static_resid(m, _mf_vec(a[1], m.n_endo, "static_resid"), _mf_x(m, a[2], "static_resid"), _mf_vec(a[3], length(m.params), "static_resid"))
        Any[ret(reshape(r, :, 1)), zeros(0, 1), zeros(0, 1)][1:max(nout, 1)]
    end
    reg("*.static_g1") do I, a, nout
        m = _need_host(I).model
        length(a) in (3, 5) || merr("Invalid call to static_g1")
        ret(Matrix{Float64}(static_jac(m, _mf_vec(a[1], m.n_endo, "static_g1"), _mf_x(m, a[2], "static_g1"), _mf_vec(a[3], length(m.params), "static_g1"))))
    end
    reg("*.set_auxiliary_variables") do I, a, nout
        m = _need_host(I).model
        length(a) == 3 || merr("Invalid call to set_auxiliary_variables")
        y = fill_auxiliary(m, _mf_vec(a[1], m.n_endo, "set_auxiliary_variables"), _mf_x(m, a[2], "set_auxiliary_variables"),
                           _mf_vec(a[3], length(m.params), "set_auxiliary_variables"))
        ret(reshape(y, :, 1))
    end
    nothing
end
_register_model_functions!()
