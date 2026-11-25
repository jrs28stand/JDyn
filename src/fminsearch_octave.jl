# Port of Octave 8.4's fminsearch.m (scripts/optimization/fminsearch.m, Copyright (C) 2003-2023 The Octave Project Developers,
# GPL-3.0-or-later), whose inner routine nmsmax is the Nelder-Mead code of N. J. Higham's Matrix Toolbox (Copyright (C) 2002, 2013
# N. J. Higham; modified for Octave by A. Adler, 2003), distributed with Octave under the GPL. Modified in 2026 for Julia (the
# algorithm, stopping rules and counts of the original are kept). Port of pydyn's fminsearch.py.
#
# Dynare's `occbin.findmin` calls `fminsearch` with default options, and under Octave that is this routine (MATLAB's own
# fminsearch is a different Nelder-Mead and takes other steps).

"""
    fminsearch_octave(fcn, x0; tol_x=1e-4, tol_f=1e-4, max_fun_evals=nothing, max_iter=nothing) -> (x, fval, exitflag, (iterations, funcCount, message))

Octave's `fminsearch` (regular initial simplex, direction -1): `fval` is one more call of `fcn` at `x`, as `fminsearch` makes.
"""
function fminsearch_octave(fcn, x0::AbstractVector; tol_x::Real=1e-4, tol_f::Real=1e-4, max_fun_evals=nothing, max_iter=nothing)
    x0 = Float64.(vec(collect(x0)))
    n = length(x0)
    max_fun_evals === nothing && (max_fun_evals = 200 * n)
    max_iter === nothing && (max_iter = 200 * n)
    F(v) = -Float64(real(fcn(copy(v))))
    V = zeros(n, n + 1)
    f = zeros(n + 1)
    V[:, 1] = x0
    f[1] = F(x0)
    nf = 1
    scale = max(maximum(abs, x0), 1.0)
    alpha = scale / (n * sqrt(2)) .* [sqrt(n + 1) - 1 + n, sqrt(n + 1) - 1]
    for j in 1:n
        V[:, j+1] = x0 .+ alpha[2]
    end
    for j in 1:n
        V[j, j+1] = x0[j] + alpha[1]
        f[j+1] = F(V[:, j+1])
    end
    nf += n
    function reorder!()
        j = reverse(sortperm(f; alg=MergeSort))
        V .= V[:, j]; f .= f[j]
    end
    reorder!()
    exitflag = 0
    msg = ""
    k = 0
    a, b, g = 1.0, 0.5, 2.0
    while exitflag != -1
        k += 1
        if k > max_iter
            msg = "Exceeded maximum iterations\n"
            break
        end
        if nf >= max_fun_evals
            msg = "Exceeded maximum number of function evaluations\n"
            exitflag = 0
            break
        end
        v1 = V[:, 1]
        size_simplex = _norm1_cols(V[:, 2:end] .- v1) / max(1.0, sum(abs, v1))
        step_f = maximum(abs.(f[1] .- f[2:end]))
        if size_simplex <= tol_x && step_f <= tol_f
            msg = "Algorithm converged.\n"
            exitflag = 1
            break
        end
        vbar = vec(sum(V[:, 1:n]; dims=2)) ./ n
        vr = (1 + a) .* vbar .- a .* V[:, n+1]
        fr = F(vr)
        nf += 1
        vk, fk = vr, fr
        if fr > f[n]
            if fr > f[1]
                ve = g .* vr .+ (1 - g) .* vbar
                fe = F(ve)
                nf += 1
                if fe > f[1]
                    vk, fk = ve, fe
                end
            end
        else
            vt, ft = V[:, n+1], f[n+1]
            if fr > ft
                vt, ft = vr, fr
            end
            vc = b .* vt .+ (1 - b) .* vbar
            fc = F(vc)
            nf += 1
            if fc > f[n]
                vk, fk = vc, fc
            else
                for j in 2:n
                    V[:, j] = (V[:, 1] .+ V[:, j]) ./ 2
                    f[j] = F(V[:, j])
                end
                nf += n - 1
                vk = (V[:, 1] .+ V[:, n+1]) ./ 2
                fk = F(vk)
                nf += 1
            end
        end
        V[:, n+1] = vk
        f[n+1] = fk
        reorder!()
    end
    x = V[:, 1]
    fval = Float64(real(fcn(copy(x))))
    x, fval, exitflag, (iterations=k, funcCount=nf, message=msg)
end

"The matrix 1-norm (largest absolute column sum), as `norm(A, 1)`."
_norm1_cols(A) = isempty(A) ? 0.0 : maximum(sum(abs, A; dims=1))
