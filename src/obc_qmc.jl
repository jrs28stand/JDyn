# Unscrambled Sobol sequence as in Dynare's `qmc_sequence` (mex/sources/sobol: sobol.f08, I8_SOBOL by J. Burkardt after Bratley and
# Fox, Joe and Kuo; GNU LGPL) with its inverse normal CDF (gaussian.hh: Acklam's rational approximation and one Halley step), used by
# the cubature of src/obc_ho.jl. The direction-number tables (src/obc_sobol_data.jl) were generated from the Fortran source; the
# generator is written afresh for Julia (docs/OBC_HO_PROGRESS.md). Reproduces DynareOBC's points exactly (test/test_obc_ho.jl).

include("obc_sobol_data.jl")

const _SOBOL_MAXCOL = 62

"""
    sobol_direction_numbers(dim) -> Matrix{Int64}

The `dim x 62` table `v` of I8_SOBOL (rows are dimensions), already multiplied by the powers of two it applies; `recipd = 2^-62`.
"""
function sobol_direction_numbers(dim::Integer)
    1 <= dim <= 1111 || throw(ArgumentError("the Sobol generator supports 1 to 1111 dimensions (got $dim)"))
    mc = _SOBOL_MAXCOL
    v = zeros(Int64, dim, mc)
    v[1, :] .= 1
    ninit = size(SOBOL_VINIT, 1)
    for i in 2:dim
        p = SOBOL_POLY[i]
        m = 0
        j = p
        while (j ÷= 2) > 0
            m += 1
        end
        v[i, 1:m] .= SOBOL_VINIT[1:m, i]
        for jj in m+1:mc
            newv = v[i, jj-m]
            l = Int64(1)
            for k in 1:m
                l *= 2
                (p >> (m - k)) & 1 == 1 && (newv = xor(newv, l * v[i, jj-k]))
            end
            v[i, jj] = newv
        end
    end
    l = Int64(1)
    for j in mc-1:-1:1
        l *= 2
        v[:, j] .*= l
    end
    v
end

_bit_lo0(n::Integer) = (k = 1; while (n >> (k - 1)) & 1 == 1; k += 1; end; k)     # position (1-based) of the lowest zero bit

"""
    sobol_sequence(dim, n; first=1) -> Matrix{Float64}

Points `first` .. `first + n - 1` of the unscrambled Sobol sequence (point 0 is the origin, `first = 1` the first one after it,
(0.5, ..., 0.5)), as a `dim x n` matrix of uniforms, in Dynare's arithmetic.
"""
function sobol_sequence(dim::Integer, n::Integer; first::Integer=1)
    v = sobol_direction_numbers(dim)
    recipd = 2.0^-_SOBOL_MAXCOL
    lastq = zeros(Int64, dim)
    out = Matrix{Float64}(undef, dim, n)
    for s in 0:first+n-1
        s >= first && (@views out[:, s-first+1] .= lastq .* recipd)
        l = _bit_lo0(s)
        @views lastq .= xor.(lastq, v[:, l])
    end
    out
end

const _ICDF_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
const _ICDF_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
const _ICDF_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
const _ICDF_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)

"Dynare's `icdf` (gaussian.hh): the inverse of the standard normal CDF, Acklam's approximation plus one Halley refinement."
function qmc_icdf(u::Float64)
    u == 0 && return -Inf
    u == 1 && return Inf
    (0 < u < 1) || return 0.0
    A, B, C, D = _ICDF_A, _ICDF_B, _ICDF_C, _ICDF_D
    g = if u < 0.02425
        t = sqrt(-2 * log(u))
        (((((C[1] * t + C[2]) * t + C[3]) * t + C[4]) * t + C[5]) * t + C[6]) / ((((D[1] * t + D[2]) * t + D[3]) * t + D[4]) * t + 1)
    elseif u <= 0.97575
        t = u - 0.5
        T2 = t * t
        (((((A[1] * T2 + A[2]) * T2 + A[3]) * T2 + A[4]) * T2 + A[5]) * T2 + A[6]) * t /
        (((((B[1] * T2 + B[2]) * T2 + B[3]) * T2 + B[4]) * T2 + B[5]) * T2 + 1)
    else
        t = sqrt(-2 * log(1 - u))
        -(((((C[1] * t + C[2]) * t + C[3]) * t + C[4]) * t + C[5]) * t + C[6]) / ((((D[1] * t + D[2]) * t + D[3]) * t + D[4]) * t + 1)
    end
    e = 0.5 * SpecialFunctions.erfc(-g / sqrt(2.0)) - u
    ee = e * sqrt(2 * pi) * exp(0.5 * g * g)
    g - ee / (1 + 0.5 * g * ee)
end

"""
    qmc_normal_points(dim, level) -> Matrix{Float64}

DynareOBC's cubature points: the first `2^(level+1) - 1` points of the unscrambled Sobol sequence after the origin, mapped to
standard normals (`dim x (2^(level+1) - 1)`); cached per `(dim, level)`, so the matrix is shared and must not be modified. The set is symmetric, so its mean is exactly zero.
"""
function qmc_normal_points(dim::Integer, level::Integer)
    lock(_QMC_LOCK) do
        get!(() -> qmc_icdf.(sobol_sequence(dim, 2^(Int(level) + 1) - 1)), _QMC_CACHE, (Int(dim), Int(level)))
    end
end
const _QMC_CACHE = Dict{Tuple{Int,Int},Matrix{Float64}}()
const _QMC_LOCK = ReentrantLock()
