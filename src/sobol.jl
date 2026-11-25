# Dynare's `qmc_sequence` mex file (Sobol sequences): Burkardt's `i8_sobol` (Bratley and Fox, with the Joe-Kuo direction
# numbers for up to 1111 dimensions) and the Gaussian transformation of `mex/sources/sobol/gaussian.hh`, reproduced bit for
# bit. The direction numbers (src/data/sobol_tables.json) were extracted from Dynare 7.2's `sobol.f08` (GPL-3.0-or-later,
# Dynare Team; i8_sobol is GNU LGPL, John Burkardt). Port of pydyn's sobol.py.

const SOBOL_DIM_MAX = 1111
const _SOBOL_LOG_MAX = 62

const _SOBOL_TABLES = Ref{Any}(nothing)
const _SOBOL_DIRS = Dict{Int,Tuple{Matrix{Int64},Float64}}()
const _SOBOL_LOCK = ReentrantLock()

function _sobol_tables()
    lock(_SOBOL_LOCK) do
        if _SOBOL_TABLES[] === nothing
            d = JSON.parsefile(joinpath(@__DIR__, "data", "sobol_tables.json"))
            _SOBOL_TABLES[] = (Vector{Vector{Int}}([Int.(r) for r in d["v"]]), Int.(d["poly"]))
        end
        _SOBOL_TABLES[]
    end
end

"V (dim x maxcol) of i8_sobol after its initialisation (columns already scaled by powers of two) and `recipd`."
function _sobol_directions(dim::Int)
    lock(_SOBOL_LOCK) do
        haskey(_SOBOL_DIRS, dim) && return _SOBOL_DIRS[dim]
        vi, poly = _sobol_tables()
        maxcol = _SOBOL_LOG_MAX
        V = zeros(Int64, dim, maxcol)
        V[1, :] .= 1
        for i in 2:dim, j in 1:13
            V[i, j] = vi[j][i]
        end
        for i in 2:dim
            j = poly[i]
            m = 0
            while true
                j ÷= 2
                j <= 0 && break
                m += 1
            end
            j = poly[i]
            includ = falses(m)
            for k in m:-1:1
                j2 = j ÷ 2
                includ[k] = j != 2 * j2
                j = j2
            end
            for jj in m+1:maxcol
                newv = V[i, jj-m]
                lval = Int64(1)
                for k in 1:m
                    lval *= 2
                    includ[k] && (newv = xor(newv, lval * V[i, jj-k]))
                end
                V[i, jj] = newv
            end
        end
        lval = Int64(1)
        for j in maxcol-1:-1:1
            lval *= 2
            for i in 1:dim
                V[i, j] *= lval
            end
        end
        _SOBOL_DIRS[dim] = (V, 0.5 / Float64(lval))
    end
end

function _sobol_bit_lo0(n::Int)
    bit = 0
    while true
        bit += 1
        i2 = n ÷ 2
        n == 2 * i2 && return bit
        n = i2
    end
end

"`n` successive points of the sequence from index `seed` (`i8_sobol`): a `dim x n` matrix and the new seed."
function sobol_points(dim::Integer, seed::Integer, n::Integer)
    1 <= dim <= SOBOL_DIM_MAX || throw(ArgumentError("qmc_sequence: the dimension must be between 1 and $SOBOL_DIM_MAX"))
    seed = max(Int(seed), 0)
    V, recipd = _sobol_directions(Int(dim))
    lastq = zeros(Int64, dim)
    for s in 0:seed-1
        lastq .= xor.(lastq, @view V[:, _sobol_bit_lo0(s)])
    end
    out = Matrix{Float64}(undef, dim, n)
    for k in 0:n-1
        out[:, k+1] .= Float64.(lastq) .* recipd
        lastq .= xor.(lastq, @view V[:, _sobol_bit_lo0(seed + k)])
    end
    out, seed + n
end

const _SB_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
const _SB_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
const _SB_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
const _SB_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)

"`icdf` of gaussian.hh: Acklam's rational approximation and one step of Halley's refinement."
function sobol_icdf(u::Float64)
    lb, ub = .02425, .97575
    g = 0.0
    A, B, C, D = _SB_A, _SB_B, _SB_C, _SB_D
    if 0 < u < lb
        t = sqrt(-2 * log(u))
        g = (((((C[1] * t + C[2]) * t + C[3]) * t + C[4]) * t + C[5]) * t + C[6]) /
            ((((D[1] * t + D[2]) * t + D[3]) * t + D[4]) * t + 1)
    elseif lb <= u <= ub
        t = u - .5
        T = t * t
        g = (((((A[1] * T + A[2]) * T + A[3]) * T + A[4]) * T + A[5]) * T + A[6]) * t /
            (((((B[1] * T + B[2]) * T + B[3]) * T + B[4]) * T + B[5]) * T + 1)
    elseif ub < u < 1
        t = sqrt(-2 * log(1 - u))
        g = -(((((C[1] * t + C[2]) * t + C[3]) * t + C[4]) * t + C[5]) * t + C[6]) /
            ((((D[1] * t + D[2]) * t + D[3]) * t + D[4]) * t + 1)
    end
    if 0 < u < 1
        t = .5 * SpecialFunctions.erfc(-g / sqrt(2.0)) - u
        t2 = t * sqrt(2 * pi) * exp(.5 * g * g)
        g = g - t2 / (1 + .5 * g * t2)
    end
    u == 0 && (g = -Inf)
    u == 1 && (g = Inf)
    g
end

"""
    qmc_sequence(d, seed, kind, n, lu=nothing) -> (a, new_seed)

Dynare's `[a, seed] = qmc_sequence(d, int64(seed), kind, n, lu)`: a `d x n` matrix of Sobol numbers. `kind` 0 uniform (`lu`: `d x 2`
bounds), 1 Gaussian (`lu`: the `d x d` lower Cholesky factor of the covariance matrix), 2 uniform on the hypersphere (`lu`: the radius).
"""
function qmc_sequence(d::Integer, seed::Integer, kind::Integer, n::Integer, lu=nothing)
    kind in (0, 1, 2) || throw(ArgumentError("qmc_sequence: the type must be 0, 1 or 2"))
    (kind == 2 && d < 2) && throw(ArgumentError("qmc_sequence: the dimension must be greater than 1 for a uniform QMC on an hypersphere"))
    a, seed_out = sobol_points(d, seed, n)
    if kind == 0 && lu !== nothing
        l = Float64.(lu)
        a = l[:, 1] .+ (l[:, 2] .- l[:, 1]) .* a
    elseif kind in (1, 2)
        a = reshape(sobol_icdf.(vec(a)), size(a))
        if kind == 1 && lu !== nothing
            a = Float64.(lu) * a
        elseif kind == 2
            nrm = sqrt.(sum(abs2, a; dims=1))
            a = a ./ nrm .* (lu === nothing ? 1.0 : Float64(lu))
        end
    end
    a, seed_out
end
