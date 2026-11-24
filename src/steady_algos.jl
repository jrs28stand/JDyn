# Dynare's other steady-state solvers: `solve_algo = 9` (trust_region.m on the whole system) and `solve_algo = 13`, `14` (the
# `block_trust_region` MEX, mex/sources/block_trust_region/trust_region.f08, with and without the Dulmage-Mendelsohn blocks).
# Ports of pydyn/steady.py (`_trust_region_f08`, `dynare_solve`); the iterates are Dynare's to rounding.

"""
`trust_region_solve` of `trust_region.f08` (`solve_algo` 13 and 14) on `fv(x)` (residuals) and `jac(x)`; `x` is updated in
place. Returns Dynare's exit code: -1 the guess is a solution, 0 ill-behaved at the guess, 1 success, 2 maxiter, 3 spurious
convergence, 4 slow progress, 5 `tolx` too small.
"""
function _trust_region_f08!(x::Vector{Float64}, fv, jac; tolx, tolf, maxiter, factor=1.0)
    maxslowiter = 15
    niter = 1; ncsucc = 0; ncslow = 0; info = 0
    fvec = fv(x); fjac = jac(x)
    fn = norm(fvec)
    jcn = vec(sqrt.(sum(abs2, fjac; dims=1)))
    recompute_gn = true
    gn = zeros(length(x))
    (any(isnan, fvec) || any(isnan, fjac)) && return 0
    fn < tolf && return -1
    dg = ones(length(x))
    delta = 0.0
    xnorm = 0.0
    while info == 0
        if niter == 1
            dg = [c != 0 ? c : 1.0 for c in jcn]
        else
            dg = max.(0.1 .* dg, jcn)
        end
        xnorm = norm(dg .* x)
        niter == 1 && (delta = (xnorm > 0 ? xnorm : 1.0) * factor)
        if recompute_gn
            gn = _f08_gauss_newton(fjac, fvec)
        end
        recompute_gn = false
        p = -_dogleg_f08(fjac, fvec, dg, delta, gn)
        pnorm = norm(dg .* p)
        x2 = x .+ p
        fvec2 = fv(x2)
        fn2 = norm(fvec2)
        if fn2 < tolf
            x .= x2
            info = 1
            continue
        end
        actred = fn2 < fn ? 1 - (fn2 / fn)^2 : -1.0
        t = norm(fvec .+ fjac * p)
        prered = t < fn ? 1 - (t / fn)^2 : 0.0
        ratio = prered > 0 ? actred / prered : 0.0
        niter == 1 && (delta = min(delta, pnorm))
        if ratio < 0.1
            delta /= 2
            ncsucc = 0
        else
            ncsucc += 1
            if abs(1 - ratio) <= 0.1
                delta = 2 * pnorm
            elseif ratio >= 0.5 || ncsucc > 1
                delta = max(delta, 2 * pnorm)
            end
        end
        if ratio >= 1e-4
            x0 = copy(x)
            x .= x2
            fvec = fv(x); fjac = jac(x)
            recompute_gn = true
            fn = norm(fvec)
            jcn = vec(sqrt.(sum(abs2, fjac; dims=1)))
            if any(isnan, fvec) || any(isnan, fjac)
                x .= x0
                fvec = fv(x); fjac = jac(x)
                recompute_gn = true
                fn = norm(fvec)
                jcn = vec(sqrt.(sum(abs2, fjac; dims=1)))
                delta /= 2
                ncslow += 1
                niter += 1
                continue
            end
            xnorm = norm(dg .* x)
        end
        ncslow += 1
        actred >= 0.001 && (ncslow = 0)
        niter += 1
        if niter == maxiter
            info = 2
            continue
        end
        if delta <= tolx * xnorm
            info = 3
            continue
        end
        if tolx > 0 && max(0.1 * delta, pnorm) <= 10 * eps(1.0) * xnorm
            info = 5
            continue
        end
        ncslow == maxslowiter && (info = 4)
    end
    info
end

"Gauss-Newton direction `r \\ b` by LU (LAPACK dgesv); the minimum-norm least-squares solution (dgelsd, rcond = -1) when `r` is singular."
function _f08_gauss_newton(r::Matrix{Float64}, b::Vector{Float64})
    all(isfinite, r) && all(isfinite, b) || return fill(NaN, length(b))
    F = lu(r; check=false)
    issuccess(F) && return F \ b
    sol, _ = LinearAlgebra.LAPACK.gelsd!(copy(r), copy(b), -1.0)
    sol[1:length(b)]
end

"The double dogleg of `trust_region.f08` given the Gauss-Newton direction `gn`."
function _dogleg_f08(r, b, d, delta, gn)
    xn = norm(d .* gn)
    xn <= delta && return copy(gn)
    s = (r' * b) ./ d
    sn = norm(s)
    if sn > 0
        s = s ./ sn ./ d
        tn = norm(r * s)
        snm = sn / tn^2
        if snm < delta
            bn = norm(b); dxn = delta / xn; snmd = snm / delta
            tmp = (bn / sn) * (bn / xn) * snmd
            alpha = dxn * (1 - snmd^2) / (tmp - dxn * snmd^2 + sqrt((tmp - dxn)^2 + (1 - dxn^2) * (1 - snmd^2)))
        else
            alpha = 0.0
        end
    else
        alpha = delta / xn
        snm = 0.0
    end
    alpha .* gn .+ ((1 - alpha) * min(snm, delta)) .* s
end

"""
`block_trust_region` (`solve_algo` 13: with the Dulmage-Mendelsohn blocks, solved from the last to the first; 14: on the whole system).
Returns `(x, errorflag, info)`. A non-square block whose equations hold at the guess is skipped, otherwise Dynare's MEX stops with an error
(here: `SteadyStateError`).
"""
function _block_trust_region(fun, jac, x0::Vector{Float64}, fvec0::Vector{Float64}, fjac0::Matrix{Float64}; tolf, tolx, maxit, factor, decompose::Bool)
    x = copy(x0)
    n = length(x)
    info = 0
    if decompose
        j1, j2, r, s = dmperm(fjac0)
        for i in length(r)-1:-1:1
            rows = j1[r[i]:r[i+1]-1]
            cols = j2[s[i]:s[i+1]-1]
            if length(rows) != length(cols)
                norm(fvec0[rows]) < tolf && continue
                throw(SteadyStateError("DYNARE_SOLVE (solve_algo=13): the Dulmage-Mendelsohn decomposition returned a non-square block. This means that the Jacobian is singular. You may want to try another value for solve_algo."))
            end
            function fsub(z)
                y = copy(x); y[cols] .= z
                fun(y)[rows]
            end
            function jsub(z)
                y = copy(x); y[cols] .= z
                jac(y)[rows, cols]
            end
            z = x[cols]
            info = _trust_region_f08!(z, fsub, jsub; tolx=tolx, tolf=tolf, maxiter=maxit, factor=factor)
            x[cols] .= z
        end
        fv = fun(x)
        if maximum(abs, fv; init=0.0) > tolf || any(isnan, fv)
            info = _trust_region_f08!(x, fun, jac; tolx=tolx, tolf=tolf, maxiter=maxit, factor=factor)
        elseif length(r) - 1 > 1
            info = 1
        end
    else
        info = _trust_region_f08!(x, fun, jac; tolx=tolx, tolf=tolf, maxiter=maxit, factor=factor)
    end
    x, !(info == 1 || info == -1), info
end
