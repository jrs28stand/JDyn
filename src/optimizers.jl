# Mode-finding algorithms of Dynare's `estimation(mode_compute=...)` besides csminwel (src/estimation.jl): ports of
# Dynare 7.2's matlab/optimization/{newrat, mr_hessian, mr_gstep, simplex_optimization_routine, cmaes}.m (GPL-3.0-or-later,
# Dynare Team) and MATLAB's documented Nelder-Mead `fminsearch` algorithm (Lagarias et al. 1998).
#
#   mode_compute = 5   newrat (Dynare's own): deterministic; same iterates as Dynare (outer-product-gradient Hessian
#                      from the likelihood contributions, csminit line searches, univariate gradient steps)
#   mode_compute = 7   fminsearch (MATLAB's Nelder-Mead; Dynare under Octave calls Octave's different fminsearch)
#   mode_compute = 8   Dynare's simplex_optimization_routine (Nelder-Mead with restarts)
#   mode_compute = 9   CMA-ES (Dynare's cmaes.m defaults; random)
# Objective functions follow `_dsge_likelihood`: `fcn(x) -> (fval, info, penalty[, contributions])`.

# ====================================================================== newrat (mode_compute=5)
mutable struct _HessInfo
    h1::Vector{Float64}
    htol::Float64
    htol_rel::Float64
    robust::Bool
end

"`penalty_objective_function` with the third output (the likelihood contributions, empty for a rejected point)."
function _penalty_full(fcn, x, penalty)
    r = fcn(x)
    fval, info, pen = r[1], r[2], r[3]
    info != 0 ? (penalty + pen, 0, Float64[]) : (fval, 1, length(r) >= 4 ? r[4] : Float64[])
end

_sel(v, mask) = v[findall(mask)]                  # MATLAB v(mask) with a mask that may be shorter than v

"""
`mr_hessian.m` with `hflag = 0`, as newrat uses it: two-sided gradient with steps adapted so that the function changes by
about `htol`, and the outer product of the gradients of the likelihood contributions. Returns `(hessian_mat (vec), gg, htol1,
ihh, hh_mat0, h1, fcount)`; `hessian_mat` is empty without contributions.
"""
function _mr_hessian(fcn, x, penalty, hflag, htol0, hi::_HessInfo, lb, ub)
    hflag == 0 || throw(EstimationError("newrat: only the outer-product-gradient Hessian (optim 'Hessian', 1) is implemented"))
    fcount = 0
    n = length(x)
    f0, _, ff0 = _penalty_full(fcn, x, penalty)
    fcount += 1
    h2 = ub - lb
    hmax = min.(ub - x, x - lb)
    opg = !isempty(ff0)
    hi.h1 = min.(hi.h1, 0.9 .* hmax)
    htol0 < hi.htol && (hi.htol = htol0)
    htol1 = nothing
    if !isnan(hi.htol_rel)
        htol1 = hi.htol
        hi.htol = abs(hi.htol_rel * f0)
    end
    xh1 = copy(x)
    f1, f_1 = zeros(n), zeros(n)
    ff1 = opg ? zeros(length(ff0)) : Float64[]
    ff_1 = copy(ff1)
    ggh = opg ? zeros(length(ff0), n) : zeros(0, 0)
    i = 0
    hhtol = fill(hi.htol, n)
    while i < n
        i += 1
        hi.htol = hhtol[i]
        h10 = hi.h1[i]
        hcheck = false
        xh1[i] = x[i] + hi.h1[i]
        fx, _, ffx = _penalty_full(fcn, xh1, penalty); fcount += 1
        it = 1
        dx = [fx - f0]
        ic = false
        icount = 0
        h0 = [hi.h1[i]]
        istoobig = [false]
        istoosmall = [false]
        give_it_up = false
        while (abs(dx[it]) < 0.5 * hi.htol || abs(dx[it]) > 3 * hi.htol) && icount < 10 && !ic && hmax[i] > 2e-10 && !give_it_up
            icount += 1
            _setgrow!(istoobig, it, false)
            _setgrow!(istoosmall, it, false)
            if abs(dx[it]) < 0.5 * hi.htol
                istoosmall[it] = true
                hi.h1[i] == 0.9 * hmax[i] && (give_it_up = true)
                htmp = abs(dx[it]) != 0 ? 0.9 * hi.htol / abs(dx[it]) * hi.h1[i] : 2.1 * hi.h1[i]
                htmp = min(htmp, 0.9 * hmax[i])
                if any(!=(0), _sel(h0, istoobig[1:it]))
                    htmp = 0.5 * (minimum(_sel(h0, istoobig)) + maximum(_sel(h0, istoosmall)))
                end
                hi.h1[i] = max(htmp, 1e-10)
                xh1[i] = x[i] + hi.h1[i]
                fx, _, ffx = _penalty_full(fcn, xh1, penalty); fcount += 1
            end
            if abs(dx[it]) > 3 * hi.htol
                istoobig[it] = true
                htmp = hi.htol / abs(dx[it]) * hi.h1[i]
                if any(!=(0), _sel(h0, istoosmall[1:it]))
                    htmp = 0.5 * (minimum(_sel(h0, istoobig)) + maximum(_sel(h0, istoosmall)))
                end
                hi.h1[i] = max(htmp, 1e-10)
                xh1[i] = x[i] + hi.h1[i]
                fx, _, ffx = _penalty_full(fcn, xh1, penalty); fcount += 1
                iter = 0
                while fx - f0 == 0 && iter < 50
                    hi.h1[i] *= 2
                    xh1[i] = x[i] + hi.h1[i]
                    fx, _, ffx = _penalty_full(fcn, xh1, penalty); fcount += 1
                    ic = true
                    iter += 1
                end
            end
            it += 1
            _setgrow!(dx, it, fx - f0)
            _setgrow!(h0, it, hi.h1[i])
            if hi.h1[i] < 1e-12 * min(1, h2[i]) && hi.h1[i] < 0.9 * hmax[i]
                ic = true
                hcheck = true
            end
        end
        if icount == 10 || hi.h1[i] == 1e-10
            _setgrow!(istoobig, it, false)
            _setgrow!(istoosmall, it, false)
            abs(dx[it]) < 0.5 * hi.htol && (istoosmall[it] = true)
            abs(dx[it]) > 3 * hi.htol && (istoobig[it] = true)
            if any(istoobig) && (istoosmall[it] || istoobig[it])
                ddx, ij = findmin(_sel(dx, istoobig))
                fx = f0 + ddx
                hi.h1[i] = h0[ij]                  # as Dynare: ij is a position among the 'too big' steps
            end
        end
        f1[i] = fx
        if opg
            ff1 = (isempty(ffx) || any(isnan, ffx)) ? fill(fx / length(ff0), length(ff0)) : ffx
        end
        xh1[i] = x[i] - hi.h1[i]
        fx, _, ffx = _penalty_full(fcn, xh1, penalty); fcount += 1
        f_1[i] = fx
        if opg
            ff_1 = (isempty(ffx) || any(isnan, ffx)) ? fill(fx / length(ff0), length(ff0)) : ffx
            ggh[:, i] = (ff1 - ff_1) ./ (2 * hi.h1[i])
        end
        xh1[i] = x[i]
        if hcheck && hi.htol < 1
            hi.htol = min(1, max(minimum(abs.(dx)) * 2, hi.htol * 10))
            hi.h1[i] = h10
            hhtol[i] = hi.htol
            i -= 1
        end
    end
    gg = (f1 - f_1) ./ (2 .* hi.h1)
    if opg
        gga = ggh .* (2 .* hi.h1')
        hh_mat = gga' * gga
        hh_mat0 = ggh' * ggh
        A = Matrix(Diagonal(2 .* hi.h1))
        ihh = A' * (hh_mat \ A)
        hessian_mat = vec(hh_mat0)
        if any(isnan, hessian_mat)
            hh_mat0 = Matrix{Float64}(I, n, n)
            ihh = copy(hh_mat0)
            hessian_mat = vec(hh_mat0)
        end
        hh1 = copy(hi.h1)
    else
        hessian_mat, ihh, hh_mat0, hh1 = Float64[], zeros(0, 0), zeros(0, 0), Float64[]
    end
    isnan(hi.htol_rel) && (htol1 = hhtol)
    hessian_mat, gg, htol1, ihh, hh_mat0, hh1, fcount
end
_setgrow!(v, i, x) = (i > length(v) ? (append!(v, fill(zero(eltype(v)), i - length(v) - 1)); push!(v, x)) : (v[i] = x); v)

_check_bounds(x, lb, ub, tol) = (y = copy(x); y[y .>= ub] = ub[y .>= ub] .- tol; y[y .<= lb] = lb[y .<= lb] .+ tol; y)

"`mr_gstep.m`: a sequence of univariate (gradient) steps, one parameter at a time."
function _mr_gstep(h1, x, lb, ub, fcn, penalty, htol0, gradient_epsilon, robust)
    fcount = 0
    n = length(x)
    h1 = isempty(h1) ? fill(gradient_epsilon, n) : copy(h1)
    htol = htol0 === nothing ? fill(1e-6, n) : (length(htol0) == 1 ? fill(Float64(htol0[1]), n) : copy(htol0))
    x = copy(x)
    f0 = _penalty_full(fcn, x, penalty)[1]; fcount += 1
    xh1 = copy(x)
    f1, f_1 = zeros(n), zeros(n)
    ig = zeros(Int, n)
    cb(z) = _check_bounds(z, lb, ub, 1e-10)
    for i in 1:n
        xh1[i] = x[i] + h1[i]
        f1[i] = _penalty_full(fcn, xh1, penalty)[1]; fcount += 1
        xh1[i] = x[i] - h1[i]
        f_1[i] = _penalty_full(fcn, xh1, penalty)[1]; fcount += 1
        gg = zeros(n)
        hh = zeros(n)
        gg[i] = (f1[i] - f_1[i]) / (2 * h1[i])
        hh[i] = 1 / max(1e-9, abs((f1[i] + f_1[i] - 2 * f0) / (h1[i] * h1[i])))
        if gg[i] * (hh[i] * gg[i]) / 2 > htol[i]
            ff, xx, fc, retcode = _csminit(fcn, x, penalty, f0, gg, false, Matrix(Diagonal(hh)))
            fcount += fc
            if retcode != 0 && robust
                xa = abs(x[i]) < 1e-6 ? collect(range(x[i] / 2, sign(x[i]) * 1e-6 * 3 / 2; length=7)) :
                                        collect(range(x[i] / 2, x[i] * 3 / 2; length=7))
                fa = fill(NaN, 7)
                for k in 1:7
                    xh1[i] = xa[k]
                    fa[k] = _penalty_full(fcn, xh1, penalty)[1]; fcount += 1
                end
                b = [ones(7) xa xa .* xa ./ 2] \ fa
                gg[i] = x[i] * b[3] + b[2]
                hh[i] = 1 / b[3]
                ff2, xx2, fc, _ = _csminit(fcn, x, penalty, f0, gg, false, Matrix(Diagonal(hh)))
                fcount += fc
                if ff2 < ff
                    ff, xx = ff2, xx2
                end
                if minimum(fa) < ff
                    ff, im = findmin(fa)
                    xx = copy(xx); xx[i] = xa[im]
                end
            end
            ig[i] = 1
            if robust
                if xx != cb(xx)
                    xx = cb(xx)
                    xx[i] = xx[i] < x[i] ? min(xx[i] + h1[i], 0.5 * (xx[i] + x[i])) : max(xx[i] - h1[i], 0.5 * (xx[i] + x[i]))
                    ff, ef, _ = _penalty_full(fcn, xx, penalty); fcount += 1
                    if ef == 1 && ff < f0
                        f0, x = ff, xx
                    end
                else
                    if abs(f0 - ff) < abs(gg[i] * (hh[i] * gg[i]) / 2 / 100) || abs(x[i] - xx[i]) < 1e-10
                        ff1, xx1, fc, _ = _csminit(fcn, x, penalty, f0, -gg, false, Matrix(Diagonal(hh)))
                        fcount += fc
                        if xx1 != cb(xx1)
                            xx1 = cb(xx1)
                            xx1[i] = xx1[i] < x[i] ? min(xx1[i] + h1[i], 0.5 * (xx1[i] + x[i])) : max(xx1[i] - h1[i], 0.5 * (xx1[i] + x[i]))
                            ff1, _, _ = _penalty_full(fcn, xx1, penalty); fcount += 1
                        end
                        if ff1 < ff
                            ff, xx = ff1, xx1
                        end
                    end
                    f0, x = ff, xx
                end
            else
                f0 = ff
                x = cb(xx)
            end
        end
        xh1 = copy(x)
    end
    f0, x, ig, fcount
end

"""
    newrat(fcn, x, lb, ub; ftol=1e-5, nit=1000, htol=NaN, htol_rel=NaN, robust=false, gradient_epsilon=1e-6)
        -> (x, hh, gg, fval, igg, iterations, fcount, exitflag, message)

Dynare's `newrat.m` (mode_compute=5) as `dynare_minimize_objective` calls it (`flagg = 0`, outer-product-gradient Hessian
in each iteration). `fcn(x) -> (fval, info, penalty, contributions)` where `contributions` are the terms Dynare's
`dsge_likelihood` returns as its fourth output (`[-log prior; minus log-likelihood of each period]`). `htol` is Dynare's
`TolGstep` (NaN: `ftol`).
"""
function newrat(fcn, x::AbstractVector, lb::AbstractVector, ub::AbstractVector; ftol=1e-5, nit=1000, htol=NaN, htol_rel=NaN,
                robust=false, gradient_epsilon=1e-6)
    penalty = 1e8
    fcount = 0
    nx = length(x)
    x = Float64.(x)
    xparam1 = copy(x)
    gstep_crit = isnan(htol) ? Float64(ftol) : Float64(htol)
    hi = _HessInfo(fill(Float64(gradient_epsilon), nx), gstep_crit, Float64(htol_rel), robust)
    htol_base = max(1e-7, gstep_crit)
    gtol = 1e-3
    fval0, exit_flag, _ = _penalty_full(fcn, x, penalty)
    fcount += 1
    exit_flag == 0 && return x, nothing, nothing, fval0, nothing, 0, fcount, -2, "Bad initial parameter."
    opg = true
    penalty = fval0
    dum, gg, htol0, igg, hhg, h1, fc = _mr_hessian(fcn, x, penalty, 0, htol_base, hi, lb, ub)
    fcount += fc
    hh = hhg
    if isempty(dum)
        opg = false
        igg = 1e-4 * Matrix{Float64}(I, nx, nx)
    end
    H = igg
    fvals = [fval0]
    xs = [copy(x)]
    gs = [copy(gg)]
    check = false
    igrad = true
    inx = Matrix{Float64}(I, nx, nx)
    jit = 0
    ig = ones(Int, nx)
    ggx = zeros(nx)
    message = ""
    exitflag = -3
    cbn(z) = _check_bounds(z, lb, ub, eps())
    fval = fval0
    while norm(gg) > gtol && !check && jit < nit
        jit += 1
        icount = jit
        penalty = fvals[icount]
        fval, x0, fc, _ = _csminit(fcn, xparam1, penalty, fvals[icount], gg, false, H)
        fcount += fc
        if igrad
            fval1, x01, fc, _ = _csminit(fcn, x0, penalty, fval, gg, false, inx)
            fcount += fc
            fval - fval1 > 1 || (igrad = false)
            fval, x0 = fval1, x01
        end
        ig_pos = findall(!=(0), ig)
        if length(ig_pos) < nx
            ggx .= 0.0
            ggx[ig_pos] = gg[ig_pos]
            hhx = opg ? reshape(dum, nx, nx) : hh
            iggx = Matrix{Float64}(I, nx, nx)
            iggx[ig_pos, ig_pos] = inv(hhx[ig_pos, ig_pos])
            _, x0, fc, _ = _csminit(fcn, x0, penalty, fval, ggx, false, iggx)     # as Dynare: fval is not updated
            fcount += fc
        end
        if x0 != cbn(x0)
            x0 = cbn(x0)
            fvala, ef, _ = _penalty_full(fcn, x0, penalty); fcount += 1
            ef == 1 && (penalty = fvala)
        end
        fvala, x0, ig, fc = _mr_gstep(h1, x0, lb, ub, fcn, penalty, htol0, gradient_epsilon, robust)
        fcount += fc
        if x0 != cbn(x0)
            x0 = cbn(x0)
            fvala, ef, _ = _penalty_full(fcn, x0, penalty); fcount += 1
            ef == 1 && (penalty = fvala)
        end
        fval = fvala
        if fvals[icount] - fval < ftol
            ihh = Matrix(Diagonal(1 ./ diag(hhg)))
            fval2, x0, fc, _ = _csminit(fcn, x0, penalty, fval, gg, false, ihh)
            fcount += fc
            x0 = cbn(x0)
            fval = fval2
        end
        if fvals[icount] - fval < ftol
            fval3, x0, fc, _ = _csminit(fcn, x0, penalty, fval, gg, false, inx .* 1e-4)
            fcount += fc
            x0 = cbn(x0)
            fval = fval3
        end
        xparam1 = x0
        push!(xs, copy(xparam1))
        push!(fvals, fval)
        if fvals[icount] - fval < ftol
            message = "No further improvement is possible!"
            check = true
        else
            if norm(xs[icount] - xparam1) > 1e-12
                dum, gg, htol0, igg, hhg, h1, fc = _mr_hessian(fcn, xparam1, penalty, 0, htol_base, hi, lb, ub)
                fcount += fc
                isempty(dum) && (opg = false)
                if !opg
                    H = _bfgsi(H, gg - gs[icount], xparam1 - xs[icount])
                    hh = inv(H)
                    hhg = hh
                else
                    hh = hhg
                    H = igg
                end
            end
        end
        push!(gs, copy(gg))
    end
    if jit == nit
        message = strip(message * " Maximum number of iterations reached.")
        exitflag = 0
    end
    if norm(gg) <= gtol
        message = strip(message * " Estimation ended: Gradient norm < $gtol")
        exitflag = -1
    end
    if check
        message = strip(message * " Estimation successful.")
        exitflag = 1
    end
    xparam1, hh, gg, fval, igg, jit, fcount, exitflag, message
end

# ====================================================================== simplex (mode_compute=8)
_stable_sortperm(v) = sortperm(v; alg=Base.Sort.DEFAULT_STABLE)

function _simplex_init(fcn, point, point_score, delta, zero_delta, check_delta)
    fc = 0
    n = length(point)
    v = zeros(n, n + 1)
    v[:, 1] = point
    fv = zeros(n + 1)
    fv[1] = point_score
    delta = length(delta) == 1 ? fill(Float64(delta[1]), n) : copy(delta)
    for j in 1:n
        mk(j) = (y = copy(point); y[j] = y[j] != 0 ? (1 + delta[j]) * y[j] : zero_delta; y)
        y = mk(j)
        v[:, j+1] = y
        fv[j+1], ok = fcn(y)
        fc += 1
        if check_delta
            while !ok
                point[j] != 0 ? (delta[j] /= 1.1) : (zero_delta /= 1.1)
                y = mk(j)
                v[:, j+1] = y
                fv[j+1], ok = fcn(y)
                fc += 1
            end
        end
    end
    p = _stable_sortperm(fv)
    v[:, p], fv[p], delta, fc
end

"""
    simplex_optimization(fcn, x; maxiter=10000, tolx=1e-4, tolf=1e-4, maxfcall=nothing, maxfcallfactor=500, delta_factor=0.05)
        -> (x, fval, exitflag, iterations, fcount, message)

Dynare's `simplex_optimization_routine.m` (mode_compute=8): Nelder-Mead with an optimised expansion step and up to three
restarts from the best point. `fcn(x) -> (fval, ok)`, `fval = Inf` at a rejected point. Deterministic.
"""
function simplex_optimization(fcn, x0::AbstractVector; maxiter=10000, tolx=1e-4, tolf=1e-4, maxfcall=nothing, maxfcallfactor=500,
                              delta_factor=0.05)
    x = Float64.(x0)
    iter_total = 0
    fc_total = 0
    n = length(x)
    max_func_calls = maxfcall === nothing ? maxfcallfactor * n : maxfcall
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    optimize_expansion = true
    delta = delta_factor
    DELTA = delta
    zero_delta = delta / 200
    max_no_improvements = n * 10
    tv1, tv2 = 1:n, 2:n+1
    score0, ok = fcn(x)
    fc_total += 1
    ok || throw(EstimationError("simplex_optimization_routine:: Initial condition is wrong!"))
    v, fv, delta, fci = _simplex_init(fcn, x, score0, delta, zero_delta, true)
    fc_total += fci
    func_count = n + 1
    iter_count = 1
    no_improvements = 0
    simplex_iterations = 1
    max_algo = 3
    algo_iter = 1
    best_point, best_score = v[:, 1], fv[1]
    convergence = false
    tooslow = false
    nib = 0
    max_nib = 1
    critF = critX = 1.0
    f(z) = (fc_total += 1; fcn(z)[1])
    while func_count < max_func_calls && iter_count < maxiter && algo_iter <= max_algo
        if critF <= max(tolf, 10 * eps(fv[1])) && critX <= max(tolx, 10 * eps(maximum(v[:, 1])))
            convergence = true
        end
        if critX <= tolx^2 && critF > 1
            tooslow = true
        end
        xbar = vec(sum(v[:, tv1]; dims=2)) ./ n
        xr = xbar + rho * (xbar - v[:, end])
        fxr = f(xr); func_count += 1
        if fxr < fv[1]
            xe = xbar + rho * chi * (xbar - v[:, end])
            fxe = f(xe); func_count += 1
            if fxe < fxr
                if optimize_expansion
                    xee = xbar + rho * chi * 1.01 * (xbar - v[:, end])
                    fxee = f(xee); func_count += 1
                    if fxee < fxe
                        weight = rho * chi * 1.02
                        fold, xold = fxee, xee
                        while true
                            weight *= 1.02
                            xeee = xbar + weight * (xbar - v[:, end])
                            fxeee = f(xeee); func_count += 1
                            if fxeee < fold && -(fxeee - fold) > tolf * 10 * fold
                                fold, xold = fxeee, xeee
                            else
                                break
                            end
                        end
                        xe, fxe = xold, fold
                    else
                        weight = rho * chi
                        fold, xold = fxee, xee
                        while true
                            weight /= 1.02
                            xeee = xbar + weight * (xbar - v[:, end])
                            fxeee = f(xeee); func_count += 1
                            if fxeee < fold && -(fxeee - fold) > tolf * 10 * fold
                                fold, xold = fxeee, xeee
                            else
                                break
                            end
                        end
                        xe, fxe = xold, fold
                    end
                end
                v[:, end] = xe; fv[end] = fxe
            else
                v[:, end] = xr; fv[end] = fxr
            end
        else
            if fxr < fv[n]
                v[:, end] = xr; fv[end] = fxr
            else
                shrink = false
                if fxr < fv[end]
                    xc = (1 + psi * rho) * xbar - psi * rho * v[:, end]
                    fxc = f(xc); func_count += 1
                    fxc <= fxr ? (v[:, end] = xc; fv[end] = fxc) : (shrink = true)
                else
                    xcc = (1 - psi) * xbar + psi * v[:, end]
                    fxcc = f(xcc); func_count += 1
                    fxcc < fv[end] ? (v[:, end] = xcc; fv[end] = fxcc) : (shrink = true)
                end
                if shrink
                    for j in tv2
                        v[:, j] = v[:, 1] + sigma * (v[:, j] - v[:, 1])
                        fv[j] = f(v[:, j])
                    end
                    func_count += n
                end
            end
        end
        p = _stable_sortperm(fv)
        fv, v = fv[p], v[:, p]
        critF = maximum(abs.(fv[1] .- fv[tv2]))
        critX = maximum(abs.(v[:, tv2] .- v[:, 1]))
        iter_count += 1
        iter_total += 1
        simplex_iterations += 1
        no_improvements = abs(best_score - fv[1]) < tolf ? no_improvements + 1 : 0
        best_point, best_score = v[:, 1], fv[1]
        if no_improvements > max_no_improvements
            delta = delta * 1.05
            v, fv, delta, fci = _simplex_init(fcn, best_point, best_score, delta, zero_delta, true)
            fc_total += fci
            no_improvements = 0
            func_count += n
            iter_count += 1
            iter_total += 1
            nib += 1
            simplex_iterations += 1
        end
        if func_count == max_func_calls || iter_count == maxiter || nib == max_nib || convergence || tooslow
            v, fv, delta, fci = _simplex_init(fcn, best_point, best_score, DELTA, zero_delta, true)
            fc_total += fci
            if !(func_count == max_func_calls) && !(iter_count == maxiter) && nib == max_nib
                nib = 0
                if algo_iter == max_algo
                    max_no_improvements = Inf
                    continue
                end
            end
            if algo_iter < max_algo
                delta = delta * 1.05
                v, fv, delta, fci = _simplex_init(fcn, best_point, best_score, delta, zero_delta, true)
                fc_total += fci
                func_count = n
                iter_count = 1
                convergence = false
                no_improvements = 0
                simplex_iterations += 1
                algo_iter += 1
            else
                break
            end
        end
    end
    exitflag = 1
    msg = "The function converged to a solution x."
    func_count >= max_func_calls && (msg = "Maximum number of objective function calls has been exceeded!"; exitflag = 0)
    iter_count >= maxiter && (msg = "Maximum number of iterations has been exceeded!"; exitflag = 0)
    v[:, 1], fv[1], exitflag, iter_total, fc_total, msg
end

# ====================================================================== fminsearch (mode_compute=7)
"""
    fminsearch(fcn, x; maxiter=6000, maxfunevals=1e6, tolx=1e-6, tolf=1e-8) -> (x, fval, exitflag, iterations, fcount)

MATLAB's `fminsearch` (Nelder-Mead of Lagarias et al. 1998: reflection 1, expansion 2, contraction 0.5, shrink 0.5; initial
simplex 5% steps, 0.00025 for zero coordinates) with Dynare's options for mode_compute=7. `fcn(x) -> fval`.
"""
function fminsearch(fcn, x0::AbstractVector; maxiter=6000, maxfunevals=1_000_000, tolx=1e-6, tolf=1e-8)
    n = length(x0)
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    v = zeros(n, n + 1)
    fv = zeros(n + 1)
    x = Float64.(x0)
    v[:, 1] = x
    fv[1] = fcn(x)
    for j in 1:n
        y = copy(x)
        y[j] = y[j] != 0 ? (1 + 0.05) * y[j] : 0.00025
        v[:, j+1] = y
        fv[j+1] = fcn(y)
    end
    p = _stable_sortperm(fv); fv, v = fv[p], v[:, p]
    itercount = 1
    func_evals = n + 1
    converged = false
    while func_evals < maxfunevals && itercount < maxiter
        if maximum(abs.(fv[1] .- fv[2:end])) <= max(tolf, 10 * eps(fv[1])) &&
           maximum(abs.(v[:, 2:end] .- v[:, 1])) <= max(tolx, 10 * eps(maximum(v[:, 1])))
            converged = true
            break
        end
        xbar = vec(sum(v[:, 1:n]; dims=2)) ./ n
        xr = (1 + rho) * xbar - rho * v[:, end]
        fxr = fcn(xr); func_evals += 1
        if fxr < fv[1]
            xe = (1 + rho * chi) * xbar - rho * chi * v[:, end]
            fxe = fcn(xe); func_evals += 1
            fxe < fxr ? (v[:, end] = xe; fv[end] = fxe) : (v[:, end] = xr; fv[end] = fxr)
        elseif fxr < fv[n]
            v[:, end] = xr; fv[end] = fxr
        else
            shrink = false
            if fxr < fv[end]
                xc = (1 + psi * rho) * xbar - psi * rho * v[:, end]
                fxc = fcn(xc); func_evals += 1
                fxc <= fxr ? (v[:, end] = xc; fv[end] = fxc) : (shrink = true)
            else
                xcc = (1 - psi) * xbar + psi * v[:, end]
                fxcc = fcn(xcc); func_evals += 1
                fxcc < fv[end] ? (v[:, end] = xcc; fv[end] = fxcc) : (shrink = true)
            end
            if shrink
                for j in 2:n+1
                    v[:, j] = v[:, 1] + sigma * (v[:, j] - v[:, 1])
                    fv[j] = fcn(v[:, j])
                end
                func_evals += n
            end
        end
        p = _stable_sortperm(fv); fv, v = fv[p], v[:, p]
        itercount += 1
    end
    v[:, 1], fv[1], converged ? 1 : 0, itercount, func_evals
end

# ====================================================================== CMA-ES (mode_compute=9)
"""
    cmaes(fcn, x0, lb, ub; sigma0=nothing, rng, maxiter=nothing, maxfunevals=nothing, tolfun=1e-7, tolx=1e-7)
        -> (xbest, fbest, iterations, fcount, message, success)

CMA-ES with the default strategy parameters of Dynare's `cmaes.m` (Hansen, version 3.61: population 4 + floor(3 log N),
superlinear recombination weights, rank-one and rank-mu updates, cumulative step-size adaptation) and its boundary
handling (candidates evaluated at their projection on the bounds, quadratic penalty with adaptive weights), started as
Dynare's mode_compute=9 starts it (`sigma0` = 0.2 x the width of the bounds, 0.01 for unbounded parameters). Returns the
best point ever evaluated. Random: the optimum, not the path, is comparable with Dynare's.
"""
function cmaes(fcn, x0::AbstractVector, lb::AbstractVector, ub::AbstractVector; sigma0=nothing, rng=Random.default_rng(), maxiter=nothing,
               maxfunevals=nothing, tolfun=1e-7, tolx=1e-7)
    N = length(x0)
    H0 = sigma0 === nothing ? (ub - lb) .* 0.2 : Float64.(sigma0)
    H0[.!isfinite.(H0)] .= 0.01
    while maximum(H0) / minimum(H0) > 1e6
        H0[H0 .== maximum(H0)] .*= 0.9
    end
    sigma = maximum(H0)
    diagD = H0 ./ sigma
    lambda = 4 + floor(Int, 3 * log(N))
    mu = floor(Int, lambda / 2)
    w = log(max(mu, lambda / 2) + 0.5) .- log.(1:mu)
    w ./= sum(w)
    mueff = sum(w)^2 / sum(abs2, w)
    cs = (mueff + 2) / (N + mueff + 3)
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    c1 = 2 / ((N + 1.3)^2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)^2 + mueff))
    chiN = sqrt(N) * (1 - 1 / (4N) + 1 / (21N^2))
    maxiter = maxiter === nothing ? 1e3 * (N + 5)^2 / sqrt(lambda) : Float64(maxiter)
    maxfunevals = maxfunevals === nothing ? Inf : Float64(maxfunevals)
    clip(x) = clamp.(x, lb, ub)
    xmean = clip(Float64.(x0))
    B = Matrix{Float64}(I, N, N)
    C = Diagonal(diagD .^ 2) |> Matrix
    D = copy(diagD)
    pc, ps = zeros(N), zeros(N)
    # boundary handling (cmaes.m, bnd.*)
    bounded = (lb .> -Inf) .| (ub .< Inf)
    active = any(bounded)
    maxdx = (ub - lb) ./ 2
    if active && any(sigma .* sqrt.(diag(C)) .> maxdx)
        sigma = minimum(maxdx ./ sqrt.(diag(C)))
    end
    bweights = zeros(N)
    dfithist = [1.0]
    validfitval = false
    iniphase = true
    histlen = 10 + ceil(Int, 30 * N / lambda)
    fhist = Float64[]
    fbest, xbest = Inf, copy(xmean)
    counteval = 0
    countiter = 0
    message = "maxiter"
    xold = copy(xmean)
    while true
        countiter += 1
        arz = randn(rng, N, lambda)
        arx = xmean .+ sigma .* (B * (D .* arz))
        arxvalid = similar(arx)
        fraw = zeros(lambda)
        for k in 1:lambda
            xv = clip(arx[:, k])
            f = fcn(xv); counteval += 1
            tries = 0
            while isnan(f) && tries < 100                        # cmaes.m resamples NaN values
                arz[:, k] = randn(rng, N)
                arx[:, k] = xmean .+ sigma .* (B * (D .* arz[:, k]))
                xv = clip(arx[:, k])
                f = fcn(xv); counteval += 1; tries += 1
            end
            arxvalid[:, k] = xv
            fraw[k] = f
            if f < fbest
                fbest, xbest = f, copy(xv)
            end
        end
        fsel = copy(fraw)
        diagC = diag(C)
        if active
            q = [quantile_r5(fraw, 0.25), quantile_r5(fraw, 0.75)]
            val = (q[2] - q[1]) / N / (sum(diagC) / N) / sigma^2
            if !isfinite(val)
                val = maximum(dfithist)
            elseif val == 0
                pos = filter(>(0), dfithist)
                val = isempty(pos) ? 1.0 : minimum(pos)
            elseif !validfitval
                empty!(dfithist)
                validfitval = true
            end
            length(dfithist) < 20 + 3N / lambda ? push!(dfithist, val) : (popfirst!(dfithist); push!(dfithist, val))
            tx = clip(xmean)
            ti = tx .!= xmean
            if iniphase && any(ti)
                sd = sort(dfithist); md = isodd(length(sd)) ? sd[(end+1)÷2] : (sd[end÷2] + sd[end÷2+1]) / 2
                bweights[bounded] .= 2.0002 * md
                dd = diagC[bounded] ./ (sum(diagC) / N)
                bweights[bounded] ./= dd
                validfitval && countiter > 2 && (iniphase = false)
            end
            if any(ti)
                d = xmean - tx
                idx = ti .& (abs.(d) .> 3 * max(1, sqrt(N) / mueff) .* sigma .* sqrt.(diagC)) .& (sign.(d) .== sign.(xmean - xold))
                bweights[idx] .*= 1.2^(min(1, mueff / 10 / N))
            end
            fsel = fraw .+ vec(bweights' * (arxvalid - arx) .^ 2)
        end
        idx = _stable_sortperm(fsel)
        xold = copy(xmean)
        sel = idx[1:mu]
        xmean = arx[:, sel] * w
        zmean = arz[:, sel] * w
        ps = (1 - cs) .* ps .+ sqrt(cs * (2 - cs) * mueff) .* (B * zmean)
        hsig = norm(ps) / sqrt(1 - (1 - cs)^(2 * counteval / lambda)) / chiN < 1.4 + 2 / (N + 1)
        pc = (1 - cc) .* pc .+ hsig * sqrt(cc * (2 - cc) * mueff) .* (xmean - xold) ./ sigma
        artmp = (arx[:, sel] .- xold) ./ sigma
        C = (1 - c1 - cmu) .* C .+ c1 .* (pc * pc' .+ (1 - hsig) * cc * (2 - cc) .* C) .+ cmu .* (artmp * Diagonal(w) * artmp')
        sigma *= exp(min(1, (cs / damps) * (norm(ps) / chiN - 1)))
        C = (C + C') ./ 2
        ev = eigen(Symmetric(C))
        D = sqrt.(max.(ev.values, 0.0))
        B = ev.vectors
        # stopping rules of cmaes.m
        push!(fhist, fraw[idx[1]])
        length(fhist) > histlen && popfirst!(fhist)
        if countiter > 2 && maximum(vcat(fraw, fhist)) - minimum(vcat(fraw, fhist)) <= tolfun
            message = "tolfun"; break
        end
        if all(sigma .* max.(abs.(pc), sqrt.(diag(C))) .< tolx)
            message = "tolx"; break
        end
        if maximum(D)^2 / max(minimum(D)^2, 1e-300) > 1e14
            message = "warnconditioncov"; break
        end
        countiter >= maxiter && (message = "maxiter"; break)
        counteval >= maxfunevals && (message = "maxfunevals"; break)
    end
    xbest, fbest, countiter, counteval + 1, message, message in ("tolfun", "tolx")
end
