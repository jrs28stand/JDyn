# Octave's random number generators (rand, randn) bit for bit, so that JDyn reproduces Dynare's random draws when
# Dynare runs under GNU Octave (set_dynare_seed / rand('state', s); randn('state', s)).
#
# Port of pydyn/octave_rng.py, which reproduces liboctave/numeric/randmtzig.cc and the state handling of oct-rand.cc
# (Copyright (C) 2006-2024 The Octave Project Developers, GPL-3.0-or-later; the Mersenne Twister MT19937 and its
# init_by_array seeding are by Makoto Matsumoto and Takuji Nishimura, 1997-2002, BSD-3-Clause; the ziggurat method is
# that of Marsaglia and Tsang, 2000, as implemented in Octave by David Bateman). Checked against Octave 8.4
# (test/ref_nonlinear/octave_rng.mat, written by r_octave_rng.m).
#
# Octave keeps one Mersenne Twister state per distribution: the uniform and the normal stream are two independent
# MT19937 generators that start from the same state, `init_by_array([seed])`.
#   uniforms: 53 bits, (a 2^26 + b) / 2^53 with a, b the top 27 and 26 bits of two MT outputs (never 0);
#   normals:  256-level ziggurat, 64-bit variant (54 random bits per try; tail and wedge tests draw uniforms from the
#             normal generator itself).
# Matrices are filled in column-major order, as MATLAB's randn(r, c).
#
# Where the generators are used: the particle filters (src/nonlinear_filters.jl), which Dynare reseeds with the
# 'default' seed (0) before every likelihood evaluation. `OctaveRNG` is also an `AbstractRNG` (rand -> the uniform
# stream, randn -> the normal stream) for code that wants Octave's numbers elsewhere.

const _MT_N = 624
const _MT_M = 397

"One MT19937 generator with Octave's conversions (one Octave distribution state)."
mutable struct OctaveMT
    mt::Vector{UInt32}
    pos::Int                 # index of the next word; 625 - pos is Octave's `left`
end

function _init_by_array(key::Vector{UInt32})
    mt = Vector{UInt32}(undef, _MT_N)
    mt[1] = 0x012bd6aa               # 19650218
    @inbounds for i in 2:_MT_N
        p = mt[i-1]
        mt[i] = UInt32((UInt64(1812433253) * (p ⊻ (p >> 30)) + (i - 1)) & 0xffffffff)
    end
    i, j = 2, 1
    klen = length(key)
    @inbounds for _ in 1:max(_MT_N, klen)
        p = mt[i-1]
        mt[i] = UInt32(((UInt64(mt[i]) ⊻ ((UInt64(p ⊻ (p >> 30)) * 1664525) & 0xffffffff)) + key[j] + (j - 1)) & 0xffffffff)
        i += 1; j += 1
        if i > _MT_N
            mt[1] = mt[_MT_N]; i = 2
        end
        j > klen && (j = 1)
    end
    @inbounds for _ in 1:(_MT_N - 1)
        p = mt[i-1]
        mt[i] = UInt32(((UInt64(mt[i]) ⊻ ((UInt64(p ⊻ (p >> 30)) * 1566083941) & 0xffffffff)) - (i - 1)) & 0xffffffff)
        i += 1
        if i > _MT_N
            mt[1] = mt[_MT_N]; i = 2
        end
    end
    mt[1] = 0x80000000
    mt
end

OctaveMT(word::Integer) = OctaveMT(_init_by_array(UInt32[UInt32(word & 0xffffffff)]), _MT_N + 1)

function _mt_twist!(mt::Vector{UInt32})
    @inbounds begin
        for k in 1:(_MT_N - _MT_M)
            y = (mt[k] & 0x80000000) | (mt[k+1] & 0x7fffffff)
            mt[k] = mt[k+_MT_M] ⊻ (y >> 1) ⊻ ((y & 0x1) * 0x9908b0df)
        end
        for k in (_MT_N - _MT_M + 1):(_MT_N - 1)
            y = (mt[k] & 0x80000000) | (mt[k+1] & 0x7fffffff)
            mt[k] = mt[k+_MT_M-_MT_N] ⊻ (y >> 1) ⊻ ((y & 0x1) * 0x9908b0df)
        end
        y = (mt[_MT_N] & 0x80000000) | (mt[1] & 0x7fffffff)
        mt[_MT_N] = mt[_MT_M] ⊻ (y >> 1) ⊻ ((y & 0x1) * 0x9908b0df)
    end
    nothing
end

@inline function _r32(g::OctaveMT)
    if g.pos > _MT_N
        _mt_twist!(g.mt)
        g.pos = 1
    end
    y = @inbounds g.mt[g.pos]
    g.pos += 1
    y ⊻= (y >> 11)
    y ⊻= (y << 7) & 0x9d2c5680
    y ⊻= (y << 15) & 0xefc60000
    y ⊻= (y >> 18)
    y
end

const _NMANTISSA = 9007199254740992.0           # 2^53

@inline function _u53(g::OctaveMT)
    while true
        a = _r32(g) >> 5
        b = _r32(g) >> 6
        (a == 0 && b == 0) || return (Float64(a) * 67108864.0 + Float64(b)) / _NMANTISSA
    end
end

# the C library's log and exp (Octave uses them; Julia's own differ in the last bit now and then)
@inline _clog(x::Float64) = ccall(:log, Float64, (Float64,), x)
@inline _cexp(x::Float64) = ccall(:exp, Float64, (Float64,), x)

# ziggurat tables of randmtzig.cc (normal distribution, 256 levels, 64-bit integers)
const _NOR_R = 3.6541528853610088
const _NOR_INV_R = 0.27366123732975828
const _NOR_AREA = 0.00492867323399

function _ziggurat_tables()
    ki = zeros(Int64, 256); wi = zeros(256); fi = zeros(256)
    x1 = _NOR_R
    wi[256] = x1 / _NMANTISSA
    fi[256] = _cexp(-0.5 * x1 * x1)
    ki[1] = trunc(Int64, x1 * fi[256] / _NOR_AREA * _NMANTISSA)
    wi[1] = _NOR_AREA / fi[256] / _NMANTISSA
    fi[1] = 1.0
    for i in 255:-1:2
        x = sqrt(-2.0 * _clog(_NOR_AREA / x1 + fi[i+1]))
        ki[i+1] = trunc(Int64, x / x1 * _NMANTISSA)
        wi[i] = x / _NMANTISSA
        fi[i] = _cexp(-0.5 * x * x)
        x1 = x
    end
    ki[2] = 0
    ki, wi, fi
end
const _ZKI, _ZWI, _ZFI = _ziggurat_tables()

@inline function _normal(g::OctaveMT)
    @inbounds while true
        lo = UInt64(_r32(g))
        hi = UInt64(_r32(g)) & 0x3fffff
        r = (hi << 32) | lo
        rabs = Int64(r >> 1)
        idx = Int(rabs & 0xff) + 1
        x = ((r & 1) == 1 ? -Float64(rabs) : Float64(rabs)) * _ZWI[idx]
        rabs < _ZKI[idx] && return x
        if idx == 1
            local xx, yy
            while true
                xx = -_NOR_INV_R * _clog(_u53(g))
                yy = -_clog(_u53(g))
                yy + yy > xx * xx && break
            end
            return (rabs & 0x100) != 0 ? -_NOR_R - xx : _NOR_R + xx
        end
        (_ZFI[idx-1] - _ZFI[idx]) * _u53(g) + _ZFI[idx] < _cexp(-0.5 * x * x) && return x
    end
end

"Octave's conversion of a seed value to a 32-bit word (`double2uint32`)."
function _double2uint32(d::Real)
    isfinite(d) || return UInt32(0)
    top = 4294967295.0
    d = rem(Float64(d), top)
    d < 0 && (d += top)
    UInt32(trunc(Int64, d))
end

"""
    OctaveRNG(seed=0)

Octave's `rand('state', seed); randn('state', seed)`: two independent Mersenne Twisters (uniform and normal streams).
`rand(r)`/`randn(r)` give one number; `rand(r, dims...)`, `rand!(r, A)`, `randn!(r, A)` fill column-major as MATLAB's
`rand(m, n)`/`randn(m, n)` (note that Julia's `rand(r, 3)` is a vector of 3, not Octave's 3x3). Any finite seed is
reduced to 32 bits as Octave does.
"""
mutable struct OctaveRNG <: AbstractRNG
    u::OctaveMT
    n::OctaveMT
end
function OctaveRNG(seed::Real=0)
    w = _double2uint32(seed)
    OctaveRNG(OctaveMT(w), OctaveMT(w))
end

"Re-seed as `rand('state', seed); randn('state', seed)`."
function octave_seed!(r::OctaveRNG, seed::Real=0)
    w = _double2uint32(seed)
    r.u = OctaveMT(w); r.n = OctaveMT(w)
    r
end

Base.copy(r::OctaveRNG) = OctaveRNG(OctaveMT(copy(r.u.mt), r.u.pos), OctaveMT(copy(r.n.mt), r.n.pos))

"The 625-element state vector Octave returns for `rand('state')` (uniform stream) or `randn('state')` (normal stream)."
function octave_state(g::OctaveMT)
    s = Vector{Float64}(undef, _MT_N + 1)
    # `left` is 1 for a generator that has used all its words (pos 625), as after seeding
    @inbounds for i in 1:_MT_N
        s[i] = Float64(g.mt[i])
    end
    s[end] = Float64(_MT_N + 2 - g.pos)
    s
end

"Set a generator to a 625-element Octave state vector. Only states with `left` < 624 are exact for later draws of the twist."
function octave_set_state!(g::OctaveMT, s::AbstractVector)
    length(s) == _MT_N + 1 && 0 < s[end] <= _MT_N || throw(ArgumentError("an Octave generator state has 625 elements, the last in 1..624"))
    g.mt .= UInt32.(s[1:_MT_N])
    g.pos = _MT_N + 2 - Int(s[end])
    g
end

Random.rand(r::OctaveRNG, ::Type{Float64}) = _u53(r.u)
Random.rand(r::OctaveRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float64}}) = _u53(r.u)
Random.randn(r::OctaveRNG) = _normal(r.n)
Random.randn(r::OctaveRNG, ::Type{Float64}) = _normal(r.n)

function Random.rand!(r::OctaveRNG, A::AbstractArray{Float64}, ::Random.SamplerTrivial{Random.CloseOpen01{Float64}})
    g = r.u
    @inbounds for i in eachindex(A)
        A[i] = _u53(g)
    end
    A
end
Random.rand!(r::OctaveRNG, A::AbstractArray{Float64}) = Random.rand!(r, A, Random.SamplerTrivial(Random.CloseOpen01{Float64}()))
function Random.randn!(r::OctaveRNG, A::AbstractArray{Float64})
    g = r.n
    @inbounds for i in eachindex(A)
        A[i] = _normal(g)
    end
    A
end
Random.rand(r::OctaveRNG, d1::Integer, dims::Integer...) = Random.rand!(r, Array{Float64}(undef, d1, dims...))
Random.randn(r::OctaveRNG, d1::Integer, dims::Integer...) = Random.randn!(r, Array{Float64}(undef, d1, dims...))

"""
    randperm(r::OctaveRNG, n[, m])

Octave's `randperm(n, m)`: `m` uniforms of the uniform stream and a Knuth shuffle (1-based result).
"""
function Random.randperm(r::OctaveRNG, n::Integer, m::Integer=n)
    iv = collect(1:n)
    for i in 1:m
        k = i + floor(Int, _u53(r.u) * (n - i + 1))
        iv[i], iv[k] = iv[k], iv[i]
    end
    iv[1:m]
end

# ---------------------------------------------------------------------------------------------------- streams
# The particle filters ask a stream for column-major matrices of normals and uniforms (MATLAB's randn(r, c)).
abstract type ParticleStream end

"Octave's generators (see `OctaveRNG`); with seed 0 a filter reproduces Dynare run under Octave."
struct OctaveStream <: ParticleStream
    rng::OctaveRNG
end
OctaveStream(seed::Real=0) = OctaveStream(OctaveRNG(seed))

"Julia's generators: one Xoshiro for normals and one for uniforms."
struct JuliaStream <: ParticleStream
    n::Random.Xoshiro
    u::Random.Xoshiro
end
JuliaStream(seed::Integer=0) = JuliaStream(Random.Xoshiro(seed), Random.Xoshiro(seed + 0x5bd1e995))

"Replays recorded normals and uniforms (in order; errors when they run out)."
mutable struct ReplayStream <: ParticleStream
    normals::Vector{Float64}
    uniforms::Vector{Float64}
    i_n::Int
    i_u::Int
end
ReplayStream(; normals=Float64[], uniforms=Float64[]) = ReplayStream(Float64.(vec(normals)), Float64.(vec(uniforms)), 0, 0)

stream_randn!(s::OctaveStream, A::AbstractArray{Float64}) = Random.randn!(s.rng, A)
stream_rand!(s::OctaveStream, A::AbstractArray{Float64}) = Random.rand!(s.rng, A)
stream_randn!(s::JuliaStream, A::AbstractArray{Float64}) = Random.randn!(s.n, A)
stream_rand!(s::JuliaStream, A::AbstractArray{Float64}) = Random.rand!(s.u, A)
function stream_randn!(s::ReplayStream, A::AbstractArray{Float64})
    k = length(A)
    s.i_n + k <= length(s.normals) || throw(ErrorException("ReplayStream: not enough recorded normal numbers"))
    copyto!(A, 1, s.normals, s.i_n + 1, k); s.i_n += k
    A
end
function stream_rand!(s::ReplayStream, A::AbstractArray{Float64})
    k = length(A)
    s.i_u + k <= length(s.uniforms) || throw(ErrorException("ReplayStream: not enough recorded uniform numbers"))
    copyto!(A, 1, s.uniforms, s.i_u + 1, k); s.i_u += k
    A
end
stream_randn(s::ParticleStream, r::Integer, c::Integer=1) = stream_randn!(s, Matrix{Float64}(undef, r, c))
stream_rand(s::ParticleStream, r::Integer, c::Integer=1) = stream_rand!(s, Matrix{Float64}(undef, r, c))
stream_rand1(s::ParticleStream) = stream_rand!(s, Vector{Float64}(undef, 1))[1]

# ---------------------------------------------------------------------------------------------------- per-model kind
const _RNG_KIND = WeakKeyDict{Any,Symbol}()
const RNG_KINDS = (:julia, :octave)

"""
    set_rng_kind!(m::Model, kind)      kind = :julia (default) or :octave
    rng_kind(m::Model)

Which generator the particle filters of `m` draw from (Dynare reseeds them with its 'default' seed, 0, at every likelihood
evaluation, so the likelihood is a deterministic function of the parameters). `:octave` gives Octave's own numbers, so
the likelihood equals Dynare's under Octave (up to rounding); `:julia` uses Xoshiro256++ streams seeded the same way. The
`rng=` keyword of `Estimation`/`estimate` overrides it.
"""
function set_rng_kind!(m, kind::Symbol)
    kind in RNG_KINDS || throw(ArgumentError("unknown random generator kind $kind (use :julia or :octave)"))
    _RNG_KIND[m] = kind
    m
end
rng_kind(m) = get(_RNG_KIND, m, :julia)

"A fresh filter stream of the given kind and seed (Dynare's filters seed with 0)."
function make_stream(kind, seed::Real=0)
    kind isa ParticleStream && return kind
    kind isa Function && return kind()
    kind == :octave && return OctaveStream(seed)
    kind == :julia && return JuliaStream(Int(seed))
    throw(ArgumentError("unknown random stream $(repr(kind)): use :octave, :julia, a ParticleStream or a function returning one"))
end

# Random.seed! of an Octave pair (the interpreter's `_oct_mt`, src/matlab/rng.jl): both states from `seed` (rand('state', seed); randn('state', seed))
function Random.seed!(r::OctaveRNG, seed::Integer)
    r.u = _oct_mt(Float64(seed)); r.n = _oct_mt(Float64(seed))
    r
end
Random.seed!(r::OctaveRNG) = Random.seed!(r, 0)
