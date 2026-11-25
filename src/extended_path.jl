# Extended path (matlab/ep/extended_path.m, extended_path_initialization.m, extended_path_shocks.m, extended_path_core.m,
# extended_path_homotopy.m, setup_stochastic_perfect_foresight_model_solver.m, setup_integration_nodes.m,
# solve_stochastic_perfect_foresight_model_0.m, ep_problem_0.m, gauss_hermite_weights_and_nodes.m of Dynare 7.2, GPL-3.0-or-later,
# Dynare Team), as a Julia port after pydyn's extended_path.py / extended_path_stoch.py.
#
# * order 0 (Fair-Taylor): in every period the model is solved as a perfect foresight problem over `ep.periods` periods with
#   the realised shock first and the steady state of the exogenous variables afterwards. With Dynare's defaults
#   (`ep.stack_solve_algo = 7`, `ep.solve_algo = 9`) the stacked system is solved by the trust-region algorithm of
#   `dynare_solve` (here on the sparse stacked Jacobian: same iterates); with `ep.stack_solve_algo = 0` and for linear, purely
#   backward and mixed-complementarity problems (`ep.solve_algo = 10`, lmmcp) JDyn's perfect foresight core is used. When it
#   fails, the homotopy of extended_path_homotopy.m (method 2) is tried unless `options_.no_homotopy`.
# * order > 0 with the full tree of future innovations (`ep.stochastic.algo = 0`), integration by Tensor-Gaussian-Quadrature or
#   Unscented, models with maximum lag and lead one.
#
# Refused: the sparse tree (`ep.stochastic.algo = 1`), hybrid (`ep.stochastic.hybrid_order > 1`), Stroud cubatures, calibrated
# innovations, `ep.use_first_order_solution_as_initial_guess`, extended_path_mc.
#
# Julia API: ts, status = extended_path(ws, samplesize; initialconditions=nothing, exogenousvariables=nothing)
# fills `oo_.endo_simul` (`n_endo x (maximum_lag + samplesize)`), `oo_.exo_simul` and `oo_.extended_path.status`.

struct EPError <: Exception
    msg::String
end
Base.showerror(io::IO, e::EPError) = print(io, "EPError: ", e.msg)

# ------------------------------------------------------------------------------------------------ options
"`options_.ep` with Dynare's defaults (default_option_values.m), read from the workspace."
function _ep_options(ws, O::MStruct=ws.options_)
    g(k, d) = (v = _getpath(O, "ep." * k, d); v isa AbstractMatrix && isempty(v) ? d : v)
    b(k, d) = (v = g(k, d); v isa Bool ? v : v != 0)
    (; periods=Int(g("periods", 200)), maxit=Int(g("maxit", 500)), stack_solve_algo=Int(g("stack_solve_algo", 7)),
       solve_algo=Int(g("solve_algo", 9)), innovation_distribution=string(g("innovation_distribution", "gaussian")),
       set_dynare_seed_to_default=b("set_dynare_seed_to_default", true),
       first_order_guess=b("use_first_order_solution_as_initial_guess", false),
       order=Int(g("stochastic.order", 0)), algo=Int(g("stochastic.algo", 0)),
       integration=string(g("stochastic.IntegrationAlgorithm", "Tensor-Gaussian-Quadrature")),
       nodes=Int(g("stochastic.quadrature.nodes", 5)), hybrid_order=Int(g("stochastic.hybrid_order", 0)),
       verbosity=Int(g("verbosity", 0)), use_bytecode=b("use_bytecode", false))
end

# ------------------------------------------------------------------------------------------------ trust region
_ep_finite(x::AbstractArray) = all(isfinite, x)
_ep_finite(x::SparseMatrixCSC) = all(isfinite, nonzeros(x))

function _ep_solve(A::SparseMatrixCSC, b::AbstractVector)
    try
        F = lu(A)
        return F \ b
    catch e
        e isa Union{LinearAlgebra.SingularException,ZeroPivotException,ArgumentError} || rethrow()
        return fill(NaN, length(b))
    end
end

"The dogleg of trust_region.m (sparse Jacobian `r`)."
function _ep_dogleg(r::SparseMatrixCSC, b::Vector{Float64}, d::Vector{Float64}, delta::Float64)
    x = _ep_solve(r, b)
    qnorm = norm(d .* x)
    qnorm <= delta && return x
    s = (r' * b) ./ d
    gnorm = norm(s)
    local alpha, sgnorm
    if gnorm > 0
        s = (s ./ gnorm) ./ d
        temp0 = norm(r * s)
        sgnorm = gnorm / (temp0 * temp0)
        if sgnorm < delta
            bnorm = norm(b)
            temp1 = delta / qnorm
            temp2 = sgnorm / delta
            temp0 = bnorm * bnorm * temp2 / (gnorm * qnorm)
            temp0 = temp0 - temp1 * temp2 * temp2 + sqrt((temp0 - temp1)^2 + (1.0 - temp1 * temp1) * (1.0 - temp2 * temp2))
            alpha = temp1 * (1.0 - temp2 * temp2) / temp0
        else
            alpha = 0.0
        end
    else
        alpha = delta / qnorm
        sgnorm = 0.0
    end
    alpha > 0 ? alpha .* x .+ (1.0 - alpha) * min(sgnorm, delta) .* s : min(sgnorm, delta) .* s
end

_ep_safe(f, x) = try f(x) catch e; e isa Union{DomainError,OverflowError,LinearAlgebra.SingularException} ? nothing : rethrow() end

"""
matlab/optimization/trust_region.m (`solve_algo = 9`) on the whole system; `fun(x)` returns `(residuals, sparse Jacobian)` (fresh
arrays). Returns `(x, errorflag)`.
"""
function _ep_trust_region(fun, x0::Vector{Float64}, tolf, tolx, maxiter, factor=1.0)
    radiusfactor, actualreductionthreshold, relativereductionthreshold = 0.5, 0.001, 0.1
    EPS = eps(Float64)
    x = copy(x0)
    errorflag, it, info, delta, ncsucc, ncslow = true, 1, 0, 0.0, 0, 0
    xinit = copy(x)
    r0 = _ep_safe(fun, x)
    r0 === nothing && return x, true
    fval, fjac = r0
    fnorm = norm(fval)
    (_ep_finite(fval) && _ep_finite(fjac)) || return x, true
    fnorm < tolf && return x, false
    fjacnorm_ = Float64[]
    xnorm = 0.0
    while it <= maxiter && info == 0
        fjacnorm = sqrt.(vec(sum(abs2, fjac; dims=1)))
        if it == 1
            fjacnorm_ = copy(fjacnorm)
            fjacnorm_[fjacnorm .< EPS] .= 1.0
            xnorm = norm(fjacnorm_ .* x)
            delta = xnorm > 0 ? xnorm * factor : factor
        else
            fjacnorm_ = max.(0.1 .* fjacnorm_, fjacnorm)
            xnorm = norm(fjacnorm_ .* x)
        end
        p = _ep_dogleg(fjac, fval, fjacnorm_, delta)
        pnorm = norm(fjacnorm_ .* p)
        xx = x .- p
        xold = copy(x)
        ww = fval .- fjac * p
        r1 = _ep_safe(fun, xx)
        if r1 === nothing || !_ep_finite(r1[1])
            delta *= radiusfactor
            it += 1
            continue
        end
        fval1 = r1[1]
        fnorm1 = norm(fval1)
        if fnorm1 < tolf
            x = xx
            errorflag = false
            info = 1
            continue
        end
        actualreduction = fnorm1 < fnorm ? 1.0 - (fnorm1 / fnorm)^2 : -1.0
        tt = norm(ww)
        predictedreduction = tt < fnorm ? 1.0 - (tt / fnorm)^2 : 0.0
        ratio = predictedreduction > 0 ? actualreduction / predictedreduction : 0.0
        it == 1 && (delta = min(delta, pnorm))
        if ratio < relativereductionthreshold
            ncsucc = 0
            delta *= radiusfactor
        else
            ncsucc += 1
            if abs(ratio - 1.0) < relativereductionthreshold
                delta = pnorm / radiusfactor
            elseif ratio >= radiusfactor || ncsucc > 1
                delta = max(delta, pnorm / radiusfactor)
            end
        end
        if ratio > 1e-4
            x = xx
            fval = fval1
            xnorm = norm(fjacnorm_ .* x)
            fnorm = fnorm1
        end
        ncslow += 1
        actualreduction >= actualreductionthreshold && (ncslow = 0)
        it += 1
        if it == maxiter
            info = 2; x = copy(xinit)
            continue
        end
        if delta < tolx * xnorm
            info = 3; x = copy(xinit); errorflag = true
            continue
        end
        if tolx > 0 && max(0.1 * delta, pnorm) <= 10 * eps(xnorm) * xnorm
            info = 5; x = copy(xinit); errorflag = true
            continue
        end
        if ncslow == 15
            info = 4; x = copy(xinit); errorflag = true
            continue
        end
        fjac0 = fjac
        r2 = _ep_safe(fun, x)
        if r2 === nothing || !_ep_finite(r2[2])
            x = xold; fjac = fjac0
            delta *= radiusfactor
            continue
        end
        fjac = r2[2]
    end
    x, errorflag
end

# ------------------------------------------------------------------------------------------------ state of a run
mutable struct EPState
    ws::Any
    m::Model
    ep::Any
    o::Dict{String,Any}
    params::Vector{Float64}
    ss::Vector{Float64}
    xss::Vector{Float64}
    P::PFStacked
    sim::PFSim
    periods::Int
    no_homotopy::Bool
end

function _ep_state(ws, ep, ss, xss; periods=ep.periods)
    m = ws.model
    O = deepcopy(ws.options_)
    o = _pfo_options(m, O, false)
    o["simul.maxit"] = ep.maxit
    o["stack_solve_algo"] = ep.stack_solve_algo
    ep.solve_algo == 10 && (o["lmmcp.status"] = true)
    ep.solve_algo == 11 && throw(Unsupported("ep.solve_algo = 11 (the PATH solver)", "feature"))
    params = Vector{Float64}(m.params)
    sim = PFSim(m, params, o, zeros(m.n_endo, 0), zeros(m.n_exo, 0), ss, xss, nothing, nothing, false)
    nh = _optb(ws, "no_homotopy", false)
    EPState(ws, m, ep, o, params, ss, xss, PFStacked(m, params, ss), sim, periods, nh)
end

# perfect_foresight_solver_core as extended_path_core.m calls it: endo n x rows, exo nx x rows. Returns (endo, success).
function _ep_core0(E::EPState, endo::Matrix{Float64}, exo::Matrix{Float64}, periods::Int)
    m, o = E.m, E.o
    n = m.n_endo
    lag = Int(m.info["maximum_lag"])
    two_sided = m.info["maximum_endo_lag"] > 0 && m.info["maximum_endo_lead"] > 0
    linear = get(o, "linear", false) === true || o["linear_approximation"]
    if E.ep.stack_solve_algo == 7 && E.ep.solve_algo == 9 && two_sided && !linear && !o["lmmcp.status"]
        y0, yT = _pf_boundaries(m, endo, periods)
        X = exo[:, lag+1:lag+periods]
        fun = function (z)
            res, A = evaluate!(E.P, reshape(z, n, periods), y0, yT, X, true)
            copy(res), SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), copy(A.nzval))
        end
        z = vec(endo[:, lag+1:lag+periods])
        tolf, tolx = o["dynatol.f"], o["dynatol.x"]
        f0 = _ep_safe(zz -> evaluate!(E.P, reshape(zz, n, periods), y0, yT, X, false)[1], z)
        err = true
        if f0 !== nothing && all(isfinite, f0) && maximum(abs, f0) < tolf
            err = false
        else
            z, err = _ep_trust_region(fun, copy(z), tolf, tolx, E.ep.maxit, get(o, "trust_region_initial_step_bound_factor", 1.0))
        end
        out = copy(endo)
        out[:, lag+1:lag+periods] = reshape(z, n, periods)
        return out, (!err) && all(isfinite, z)
    end
    stack = E.ep.stack_solve_algo
    if !(stack in (0, 7)) && !o["lmmcp.status"]
        # Dynare's other stacked solvers (bytecode, LBJ, ...) are not ported: Newton on the stacked system
    end
    y, success, _, _ = _pf_core(E.sim, endo, exo, E.ss, E.xss, periods)
    y, success === true
end

# extended_path_homotopy.m for order 0
function _ep_homotopy0(E::EPState, endo::Matrix{Float64}, exo::Matrix{Float64}, periods::Int, method::Int=2)
    ss, xss = E.ss, E.xss
    endo0 = copy(endo)
    noconv = true
    info = false
    if method in (1, 2)
        iteration, weight, maxiter = 0, 0.1, 100
        increase_flag, increase_factor, decrease_factor = false, 1.2, 1.1
        state = fill(false, 5)
        oldweight = weight
        while noconv
            iteration += 1
            e = copy(endo)
            e[:, 1] = ss .+ weight .* (endo0[:, 1] .- ss)
            x = weight .* exo .+ (1 - weight) .* xss
            new, success = _ep_core0(E, e, x, periods)
            flag = success
            state = vcat(flag, state[1:end-1])
            if flag
                if weight == 1
                    noconv = false
                    break
                end
                if all(state)
                    increase_factor = 1 + (increase_factor - 1) * 1.1
                    state = fill(false, 5)
                end
                oldweight = weight
                weight = min(weight * increase_factor, 1.0)
                increase_flag = true
                endo = new
            else
                weight = increase_flag ? oldweight + (weight - oldweight) / 100 : min(weight / decrease_factor, 1.0)
            end
            iteration > maxiter && break
            weight < 1e-9 && break
        end
        info = !noconv
    end
    if method == 3 || (method == 2 && noconv)
        method == 2 && (endo = endo0)
        weights = collect(0:1000) ./ 1000
        noconv, index = true, 1
        while noconv
            weight = weights[index]
            x = weight .* exo .+ (1 - weight) .* xss
            new, success = _ep_core0(E, endo, x, periods)
            if success
                if weight == 1
                    noconv = false
                    continue
                end
                index += 1
                endo = new
            else
                break
            end
        end
        info = !noconv
    end
    info, endo
end

# ------------------------------------------------------------------------------------------------ integration nodes
function _ep_gauss_hermite(n::Int)
    n == 1 && return [0.0], [1.0]
    b = sqrt.((1:n-1) ./ 2)
    E = eigen(SymTridiagonal(zeros(n), collect(Float64, b)))
    idx = sortperm(E.values)
    sqrt(2.0) .* E.values[idx], (E.vectors[1, idx]) .^ 2
end

"setup_integration_nodes.m: `(nodes, weights)` for the `n` shocks with a positive variance; `Omega` is the upper Cholesky factor."
function _ep_integration_nodes(algorithm::AbstractString, nodes_per_shock::Int, Omega::AbstractMatrix)
    n = size(Omega, 1)
    if algorithm == "Tensor-Gaussian-Quadrature"
        n == 0 && return zeros(1, 0), [1.0]
        nodes0, weights0 = _ep_gauss_hermite(nodes_per_shock)
        N0 = repeat(nodes0, 1, n) * Omega
        q = length(weights0)
        nn = q^n
        nodes = Matrix{Float64}(undef, nn, n)
        w = ones(nn)
        for i in 1:n
            rep = q^(i - 1)
            for r in 1:nn
                k = (div(r - 1, rep) % q) + 1
                nodes[r, i] = N0[k, i]
                w[r] *= weights0[k]
            end
        end
        return nodes, w
    elseif algorithm == "Unscented"
        k = 3.0
        C = sqrt(n + k) .* permutedims(Omega)
        nodes = n == 0 ? zeros(1, 0) : vcat(zeros(1, n), -C, C)
        weights = vcat(k / (n + k), fill(1.0 / (2 * (n + k)), 2n))
        return nodes, weights
    elseif algorithm in ("Stroud-Cubature-3", "Stroud-Cubature-5")
        throw(Unsupported("the integration algorithm $algorithm of the stochastic extended path (implemented: Tensor-Gaussian-Quadrature, Unscented)", "feature"))
    end
    throw(EPError("Stochastic extended path:: Unknown integration algorithm $algorithm!"))
end

# ------------------------------------------------------------------------------------------------ the full tree
mutable struct EPTree
    m::Model
    params::Vector{Float64}
    ss::Vector{Float64}
    n::Int
    order::Int
    periods::Int
    pos::Vector{Int}
    nodes::Matrix{Float64}
    weights::Vector{Float64}
    nnodes::Int
    W::Int
    S::Int
    dimension::Int
    i_cols_j::Vector{Int}
    i_cols_1::Vector{Int}
    i_cols_T::Vector{Int}
    Ap0::Vector{Int}
    As0::Vector{Int}
    Af0::Vector{Int}
    icA::Vector{Int}
    upd_r::Vector{Int}
    upd_y::Vector{Int}
    nblocks1::Int
end

function EPTree(m::Model, params, ss, Sigma_e, order::Int, periods::Int, nodes::Matrix{Float64}, weights::Vector{Float64})
    n = m.n_endo
    (m.info["maximum_lag"] == 1 && m.info["maximum_lead"] == 1) ||
        throw(Unsupported("the stochastic extended path for models whose maximum lag or lead differs from one", "feature"))
    L = m.lead_lag_incidence .> 0                                       # n x 3
    all(L[:, 2]) || throw(Unsupported("the stochastic extended path when a variable does not appear at the current period", "feature"))
    pos = findall(>(0), diag(Sigma_e))
    k0 = findfirst(i -> sum(abs, nodes[i, :]) < 1e-12, 1:size(nodes, 1))
    k0 === nothing && throw(EPError("solve_stochastic_perfect_foresight_model_0: there is no node equal to zero"))
    perm = vcat(k0, setdiff(1:length(weights), k0))
    nodes, weights = nodes[perm, :], weights[perm]
    nn = length(weights)
    W = nn^order
    periods > order || throw(EPError("ep.periods ($periods) must exceed the order ($order) of the stochastic extended path"))
    flat = findall(vec(L))
    nyp, nyf = count(L[:, 1]), count(L[:, 3])
    S = sum(nn^i for i in 0:order-1)
    dimension = n * (S + (periods - order) * W)
    dimension > 3_000_000 && throw(Unsupported("the tree of future innovations has $dimension unknowns ($nn nodes per period, order $order, $periods periods): too large", "feature"))
    icA = vcat(findall(L[:, 1]), findall(L[:, 2]) .+ W * n, findall(L[:, 3]) .+ 2 * W * n)
    upd_r = zeros(Int, dimension); upd_y = zeros(Int, dimension)
    upd_r[1:n] = 1:n
    upd_y[1:n] = n .+ (1:n)
    i1, i2, n1, n2 = n + 1, 2n, n + 1, 2n
    for i in 2:periods
        for j in 1:nn^min(i - 1, order)
            upd_r[i1:i2] = (n1:n2) .+ (j - 1) * n * periods
            upd_y[i1:i2] = (n1:n2) .+ n .+ (j - 1) * n * (periods + 2)
            i1 = i2 + 1
            i2 += n
        end
        n1 = n2 + 1
        n2 += n
    end
    EPTree(m, Vector{Float64}(params), Vector{Float64}(ss), n, order, periods, pos, nodes, weights, nn, W, S, dimension, flat,
           findall(vec(L[:, 2:3])) .+ n, findall(vec(L[:, 1:2])), flat[1:nyp], flat[nyp+1:nyp+n], flat[nyp+n+1:nyp+n+nyf] .- n,
           icA, upd_r, upd_y, sum(nn^i for i in 0:order))
end

# ep_problem_0.m: residuals and Jacobian at the unknowns `y`. `endo_simul` is n x (periods+2), `x` is (periods+2) x n_exo.
function _ep_problem0(T::EPTree, y::Vector{Float64}, endo_simul::Matrix{Float64}, x::Matrix{Float64})
    m, n, order, P, nn, W = T.m, T.n, T.order, T.periods, T.nnodes, T.W
    params, ss, nodes, weights = T.params, T.ss, T.nodes, T.weights
    Y = repeat(vec(endo_simul), 1, W)
    Y[T.upd_y] .= y
    a1r, a1c, a1v = Int[], Int[], Float64[]
    function put!(rows_, cols_, blk)
        for ic in axes(blk, 2), ir in axes(blk, 1)
            v = blk[ir, ic]
            v == 0 && continue
            push!(a1r, rows_[ir]); push!(a1c, cols_[ic]); push!(a1v, v)
        end
    end
    res = zeros(n, P, W)
    i_cols_p = collect(1:n); i_cols_s = n .+ collect(1:n); i_cols_f = 2n .+ collect(1:n)
    i_rows = collect(1:n)
    Ap, As, Af = copy(T.Ap0), copy(T.As0), copy(T.Af0)
    n3 = 3n
    evalz(z, xi) = (dynamic_resid(m, z, xi, params, ss), dynamic_jac(m, z, xi, params, ss)[:, 1:n3])
    for i in 1:order+1
        i_w_p = 1
        for j in 1:nn^(i - 1)
            innovation = copy(x)
            i > 1 && (innovation[i+1, T.pos] = nodes[mod(j - 1, nn)+1, :])
            if i <= order
                for k in 1:nn
                    z = vcat(Y[i_cols_p, i_w_p], Y[i_cols_s, j], Y[i_cols_f, (j-1)*nn+k])
                    d1, jac = evalz(z, innovation[i+1, :])
                    if i == 1
                        put!(i_rows, vcat(As .- n, Af), weights[k] .* jac[:, T.i_cols_1])
                    else
                        put!(i_rows, vcat(Ap, As, Af), weights[k] .* jac[:, T.i_cols_j])
                    end
                    res[:, i, j] .+= weights[k] .* d1
                    Af = Af .+ n
                end
            else
                z = vcat(Y[i_cols_p, i_w_p], Y[i_cols_s, j], Y[i_cols_f, j])
                d1, jac = evalz(z, innovation[i+1, :])
                if i == 1
                    put!(i_rows, vcat(As .- n, Af), jac[:, T.i_cols_1])
                else
                    put!(i_rows, vcat(Ap, As, Af), jac[:, T.i_cols_j])
                end
                res[:, i, j] = d1
                Af = Af .+ n
            end
            i_rows = i_rows .+ n
            mod(j, nn) == 0 && (i_w_p += 1)
            if i > 1
                mod(j, nn) == 0 && (Ap = Ap .+ n)
                As = As .+ n
            end
        end
        i_cols_p = i_cols_p .+ n; i_cols_s = i_cols_s .+ n; i_cols_f = i_cols_f .+ n
    end
    a2r, a2c, a2v = Int[], Int[], Float64[]
    for j in 1:W
        i_rows_y = collect(1:n3) .+ (order + 1) * n
        offset_c = n * (T.S + j - 1)
        offset_r = (j - 1) * n
        for i in order+2:P
            d1, jac = evalz(Y[i_rows_y, j], x[i+1, :])
            sub = i == P ? jac[:, T.i_cols_T] : jac[:, T.i_cols_j]
            for ic in axes(sub, 2), ir in axes(sub, 1)
                v = sub[ir, ic]
                v == 0 && continue
                push!(a2r, offset_r + ir); push!(a2c, offset_c + T.icA[ic]); push!(a2v, v)
            end
            res[:, i, j] = d1
            i_rows_y = i_rows_y .+ n
            offset_c += W * n
            offset_r += W * n
        end
    end
    A1 = sparse(a1r, a1c, a1v, n * T.nblocks1, T.dimension)
    A2 = sparse(a2r, a2c, a2v, n * (P - order - 1) * W, T.dimension)
    vec(res)[T.upd_r], vcat(A1, A2)
end

"solve_stochastic_perfect_foresight_model_0.m: `(errorflag, endo_simul, y)`; only the second column of `endo_simul` is updated."
function _ep_solve_tree(T::EPTree, endo_simul::Matrix{Float64}, exo::Matrix{Float64}, y::Union{Nothing,Vector{Float64}}, o, maxit::Int)
    n = T.n
    y === nothing && (y = repeat(T.ss, T.dimension ÷ n))
    tolf, tolx = o["dynatol.f"], o["dynatol.x"]
    fun = z -> _ep_problem0(T, z, endo_simul, exo)
    f0 = _ep_safe(z -> fun(z)[1], y)
    ynew, err = y, true
    if f0 !== nothing && all(isfinite, f0) && maximum(abs, f0) < tolf
        err = false
    else
        ynew, err = _ep_trust_region(fun, copy(y), tolf, tolx, maxit, get(o, "trust_region_initial_step_bound_factor", 1.0))
    end
    out = copy(endo_simul)
    out[:, 2] = ynew[1:n]
    err, out, ynew
end

# ------------------------------------------------------------------------------------------------ the loop
"""
    extended_path(ws, samplesize; initialconditions=nothing, exogenousvariables=nothing, options=ws.options_) -> (endo, exo, status)

extended_path.m. `endo` is `n_endo x (maximum_lag + samplesize)` (`ts.data'`), `exo` the rows of `oo_.exo_simul`
(`(maximum_lag + samplesize) x n_exo`). `initialconditions` is a vector of the endogenous variables, `exogenousvariables`
a `samplesize x n_exo` matrix of innovations.
"""
function extended_path(ws::DynareWorkspace, samplesize::Integer; initialconditions=nothing, exogenousvariables=nothing,
                       options::MStruct=ws.options_, oo::MStruct=ws.oo_)
    m = ws.model
    _apply_matlab_options!(ws)
    ep0 = _ep_options(ws, options)
    n, nx = m.n_endo, m.n_exo
    lag, lead = Int(m.info["maximum_lag"]), Int(m.info["maximum_lead"])
    lag == 0 && throw(EPError("(Stochastic) Extended path is not supported for purely forward-looking models."))
    ep = ep0
    if lead == 0
        ep.order > 0 && throw(EPError("Stochastic extended path does not support purely backward-looking models."))
        ep = merge(ep, (; periods=1))
    end
    n_exo_det(m) == 0 || throw(EPError("Extended path does not support varexo_det."))
    ep.first_order_guess && throw(Unsupported("ep.use_first_order_solution_as_initial_guess", "feature"))
    ep.hybrid_order > 1 && throw(Unsupported("the hybrid stochastic extended path (ep.stochastic.hybrid_order > 1)", "feature"))
    ep.hybrid_order == 1 && @warn "extended_path:: hybrid=1 is equivalent to hybrid=0 (option value must be an integer greater than 1 to be effective)."
    ep.order > 0 && ep.algo == 1 && throw(Unsupported("the stochastic extended path with the sparse tree (ep.stochastic.algo = 1, tree=sparse)", "feature"))
    ep.algo in (0, 1) || throw(EPError("extended_path_core: unknown ep.stochastic.algo = $(ep.algo)"))
    (ep.innovation_distribution in ("gaussian", "calibrated")) ||
        throw(EPError("extended_path:: $(ep.innovation_distribution) distribution for the structural innovations is not (yet) implemented!"))
    ep.innovation_distribution == "calibrated" && throw(Unsupported("ep.innovation_distribution = calibrated", "feature"))
    ss = let s = get(oo, "steady_state", nothing); s isa AbstractArray && length(s) == n ? Float64.(vec(s)) : throw(EPError("oo_.steady_state is empty: run steady first")) end
    xss = let s = get(oo, "exo_steady_state", nothing); s isa AbstractArray && length(s) == nx ? Float64.(vec(s)) : zeros(nx) end
    # extended_path_initialization.m
    y0 = if initialconditions === nothing || (initialconditions isa AbstractArray && isempty(initialconditions))
        h = ws.M_.endo_histval
        (h isa AbstractArray && length(h) == n) ? Float64.(vec(h)) : copy(ss)
    else
        Float64.(vec(initialconditions))
    end
    ep.set_dynare_seed_to_default && set_dynare_seed!(ws, "default")
    pos = findall(>(0), diag(m.Sigma_e))
    Omega = isempty(pos) ? zeros(0, 0) : Matrix(cholesky(Symmetric(m.Sigma_e[pos, pos])).U)
    if exogenousvariables !== nothing && !(exogenousvariables isa AbstractArray && isempty(exogenousvariables)) &&
       ep.innovation_distribution == "gaussian" && any(!iszero, xss)
        throw(EPError("extended_path: a Gaussian innovation_distribution is incompatible with non-mean 0 exogenous variables."))
    end
    # extended_path_shocks.m
    T = Int(samplesize)
    shocks = zeros(T, nx)
    if exogenousvariables === nothing || (exogenousvariables isa AbstractArray && isempty(exogenousvariables))
        if !isempty(pos)
            R = dynare_rng(ws)
            Z = Matrix{Float64}(undef, length(pos), T)
            randn!(R, Z)
            shocks[:, pos] = permutedims(permutedims(Omega) * Z)
        end
    else
        shocks = Matrix{Float64}(exogenousvariables)
        size(shocks, 2) == nx || throw(EPError("extended_path: the exogenous variables must have $nx columns"))
        size(shocks, 1) >= T || throw(EPError("extended_path: the exogenous variables must have at least $T rows"))
    end
    rows = lag + ep.periods + lead
    exo_full = repeat(xss', rows, 1)                       # rows x nx
    paths = fill(NaN, n, T + lag)
    paths[:, 1] = y0
    status = true
    E = ep.order == 0 ? _ep_state(ws, ep, ss, xss) : nothing
    tree = nothing; o_tree = nothing
    if ep.order > 0
        o_tree = _pfo_options(m, deepcopy(ws.options_), false)
        o_tree["simul.maxit"] = ep.maxit
        ep.solve_algo == 9 || throw(Unsupported("ep.solve_algo = $(ep.solve_algo) in the stochastic extended path (implemented: 9)", "feature"))
        nodes, weights = _ep_integration_nodes(ep.integration, ep.nodes, Omega)
        tree = EPTree(m, m.params, ss, m.Sigma_e, ep.order, ep.periods, nodes, weights)
    end
    ytree = nothing
    endopath = zeros(n, 0)
    t = 1
    while t <= T
        t += 1
        exo = copy(exo_full)
        exo[2, :] = shocks[t-1, :]
        init = t > 2 ? hcat(endopath[:, 2:end], ss) : hcat(paths[:, t-1], repeat(ss, 1, ep.periods + lead))
        success = false
        if ep.order == 0
            endo, success = _ep_core0(E, init, Matrix(exo'), ep.periods)
            if !success && !E.no_homotopy
                success, endo = _ep_homotopy0(E, endo, Matrix(exo'), ep.periods, 2)
            end
        else
            err, endo, ytree = _ep_solve_tree(tree, init, exo, ytree, o_tree, ep.maxit)
            success = !err
        end
        if !success
            @warn "extended_path: No convergence of the (stochastic) perfect foresight solver (in period $t)!"
            status = false
            break
        end
        paths[:, t] = endo[:, 2]
        endopath = endo
    end
    exo_sim = vcat(repeat(xss', lag, 1), shocks[1:T, :])
    paths, exo_sim, status
end

"The dseries of an extended path: all the endogenous variables, from `initial_period` (default 1Y)."
function _ep_dseries(m::Model, paths::Matrix{Float64})
    names = copy(m.endo_names)
    MDseries(Matrix(paths'), names, copy(names), MDates(1, collect(1.0:size(paths, 2))), fill("", length(names)), Pair{String,Vector{Any}}[])
end

function _ep_store!(ws, paths, exo_sim, status)
    oo = ws.oo_
    oo.endo_simul = paths
    oo.exo_simul = exo_sim
    s = MStruct(); s.status = status
    oo.extended_path = s
    nothing
end

"The `extended_path` statement: options already in options_, `periods` is the sample size."
function _extended_path_statement!(ws, st)
    o = get_or(st, "options", Dict{String,Any}())
    haskey(o, "periods") || throw(ModelError("the 'periods' option of 'extended_path' is mandatory"))
    T = Int(round(_pf_float(o["periods"])))
    paths, exo_sim, status = try
        extended_path(ws, T)
    catch e
        e isa EPError && throw(ModelError(e.msg))
        rethrow()
    end
    _ep_store!(ws, paths, exo_sim, status)
    ts = _ep_dseries(ws.model, paths)
    ws.model.session.sc.vars["Simulated_time_series"] = ts
    nothing
end

bi!("extended_path") do I, a, nout
    length(a) == 6 || merr("extended_path: six input arguments expected")
    ws = _pfo_ws(I)
    m = ws.model
    init = a[1]
    exo = a[3]
    oo = a[6] isa MStruct ? deepcopy(a[6]) : merr("extended_path: the last argument must be oo_")
    O = _pfo_options_struct(ws, a[4])
    n = m.n_endo
    ics = init isa AbstractArray && !isempty(init) ? Float64.(vec(farr(init))) : nothing
    T = Int(round(scalar(a[2], "extended_path")))
    exov = exo isa AbstractArray && !isempty(exo) ? Matrix{Float64}(farr(exo)) : nothing
    paths, exo_sim, status = try
        extended_path(ws, T; initialconditions=ics, exogenousvariables=exov, options=O, oo=oo)
    catch e
        e isa EPError && merr(e.msg)
        rethrow()
    end
    oo.endo_simul = paths
    oo.exo_simul = exo_sim
    st = MStruct(); st.status = status; oo.extended_path = st
    Any[_ep_dseries(m, paths), oo][1:max(nout, 1)]
end
