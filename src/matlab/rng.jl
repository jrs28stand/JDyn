# Octave's random numbers in the interpreter: rand, randn, randi, randperm, rng, set_dynare_seed (port of the
# random-number section of pydyn/matlab.py; Octave 8.4's generators are `OctaveRNG`, src/octave_rng.jl).
#
# Two modes. JDyn's default is a seeded Julia generator (src/matlab/builtins_extra.jl): reproducible, not Octave's numbers.
# The Octave mode gives Octave's numbers bit for bit. It is selected
#   * for `mrun(text; rng=:octave)` (or an OctaveRNG), and
#   * for `dynare(file; random=:octave)`, which also makes the workspace's stream (simulated shocks, forecasts, ...) an
#     OctaveRNG, so that the .mod file's own rand/randn calls and Dynare's commands consume ONE stream in file order, as
#     Dynare does under Octave. The stream starts from Dynare's `set_dynare_seed('default')` state.
# In Octave mode `rand`/`randn` are two separate Mersenne-Twister states (Octave's), `rand('state', s)` reseeds one of them.

"Octave 8's conversion of a seed element to a 32-bit word: round half away from zero, saturate to [0, 2^32-1], NaN -> 0."
function _oct_word(v::Real)
    isnan(v) && return UInt32(0)
    x = Float64(v)
    r = x >= 0 ? floor(x + 0.5) : -floor(-x + 0.5)
    UInt32(clamp(r, 0.0, 4294967295.0))
end

"An Octave generator state after `rand('state', seed)` (oct-rand.cc set_internal_state); `seed` is a number or a vector."
function _oct_mt(seed)
    if seed isa AbstractString
        lowercase(seed) == "reset" && throw(Unsupported("rand('state', 'reset'): Octave then seeds from the clock and system entropy", "random"))
        merr("rand: unrecognized string argument")
    end
    (seed isa Real || is_num(seed)) || merr("rand: STATE must be a numeric vector")
    x = Float64[isinf(v) ? 0.0 : Float64(v) for v in vec(ndfarr(seed))]
    isempty(x) && throw(Unsupported("rand('state', []): an empty state (undefined in Octave)", "random"))
    if length(x) == 625 && 0 < x[625] <= 624
        mt = UInt32[_oct_word(v) for v in x[1:624]]
        return OctaveMT(mt, 626 - Int(_oct_word(x[625])))
    end
    OctaveMT(_init_by_array(UInt32[_oct_word(v) for v in x]), _MT_N + 1)
end

"The Octave generator pair the interpreter draws from, or `nothing` in JDyn's default mode."
function _octave_rng(I)
    I.rng isa OctaveRNG && return I.rng
    s = I.session
    if s !== nothing
        r = dynare_rng(s.model)
        r isa OctaveRNG && return r
    end
    nothing
end

function _oct_rand_dims(name, a)
    if !isempty(a) && a[end] isa AbstractString
        lowercase(a[end]) == "double" || throw(Unsupported("$name: class argument $(repr(a[end]))", "random"))
        a = a[1:end-1]
    end
    for x in a
        (x isa Real || is_num(x)) || merr("$name: dimensions must be numeric")
        (x isa Real && x != round(x)) && merr("conversion of $(x) to int64_t value failed")
    end
    isempty(a) && return (1, 1)
    _nd_dims(Any[x for x in a], name)
end

function _oct_randfun(I, name, a, which::Symbol)
    R = _octave_rng(I)
    if !isempty(a) && a[1] isa AbstractString && lowercase(a[1]) ∉ ("double", "single")
        key = lowercase(a[1])
        if key in ("state", "twister")
            g = which === :u ? R.u : R.n
            length(a) == 1 && return reshape(octave_state(g), :, 1)
            length(a) > 2 && merr("$name: wrong number of arguments")
            m = _oct_mt(a[2])
            which === :u ? (R.u = m) : (R.n = m)
            return Any[]
        end
        key == "seed" && throw(Unsupported("$name('seed'): Octave's old generator is not reproduced", "random"))
        merr("$name: unrecognized string argument")
    end
    dims = _oct_rand_dims(name, a)
    n = prod(dims)
    n == 0 && return ndret(zeros(dims...))
    v = Vector{Float64}(undef, n)
    which === :u ? rand!(R, v) : randn!(R, v)
    ndret(reshape(v, dims))
end

function _oct_randi(I, a)
    isempty(a) && merr("Invalid call to randi")
    R = _octave_rng(I)
    b = a[1]
    ((b isa Real && !(b isa Bool)) || (b isa AbstractMatrix{Float64})) || merr("randi: IMIN and IMAX must be integer bounds")
    bnd = vec(farr(b))
    isempty(bnd) && throw(Unsupported("randi with empty bounds"))
    any(x -> x != trunc(x), bnd) && merr("randi: IMIN and IMAX must be integer bounds")
    local imin, imax
    if length(bnd) == 1
        imin, imax = 1.0, bnd[1]
        imax < 1 && merr("randi: require IMAX >= 1")
    else
        imin, imax = bnd[1], bnd[2]
        imax < imin && merr("randi: require IMIN <= IMAX")
    end
    fm = 2.0^53
    (abs(imax) >= fm || abs(imin) >= fm) && merr("randi: IMIN and IMAX must be smaller than flintmax()")
    imax - imin >= fm - 1 && merr("randi: integer range must be smaller than flintmax()-1")
    rest = collect(Any, a[2:end])
    if !isempty(rest) && rest[end] isa AbstractString
        rest[end] == "double" || throw(Unsupported("randi: class argument $(repr(rest[end]))", "random"))
        pop!(rest)
    end
    local dims
    if isempty(rest)
        dims = [1, 1]
    else
        all(x -> x isa Real || is_num(x), rest) || merr("randi: dimensions must be numeric")
        dl = Float64[v for x in rest for v in vec(ndfarr(x))]
        length(rest) == 1 && length(dl) == 1 && (dl = [dl[1], dl[1]])
        (length(dl) < 2 || any(v -> v != floor(v) || v < 0, dl)) &&
            throw(Unsupported("randi with dimensions that are negative, not integers or a single element", "random"))
        dims = Int.(dl)
    end
    N = Float64(prod(dims))
    N > MAX_ELEMENTS && _unsup("array with more than $MAX_ELEMENTS elements")
    rng_ = (imax - imin) + 1
    K = floor((fm + 1) / rng_)
    p = (K * rng_) / (fm + 1)
    local rp
    while true
        M = Int(ceil(N / p + 10 * sqrt(N / p - N)))
        u = Vector{Float64}(undef, max(M, 0))
        isempty(u) || rand!(R, u)
        rp = filter(<(K * rng_), floor.(u .* (fm + 1)))
        length(rp) >= N && break
    end
    out = imin .+ floor.(rp[1:Int(N)] ./ K)
    ndret(reshape(out, canon_dims(dims)))
end

function _oct_randperm(I, a)
    _nargs(a, 1, 2, "randperm")
    R = _octave_rng(I)
    n = scalar(a[1], "randperm: N")
    m = length(a) == 2 ? scalar(a[2], "randperm: M") : n
    (n != floor(n) || m != floor(m) || !isfinite(n) || !isfinite(m)) && merr("randperm: N and M must be integers")
    n, m = Int(n), Int(m)
    (m < 0 || n < 0) && merr("randperm: M and N must be non-negative")
    m > n && merr("randperm: M must be less than or equal to N")
    n > MAX_ELEMENTS && _unsup("array with more than $MAX_ELEMENTS elements")
    m == 0 && return zeros(1, 0)
    ret(reshape(Float64.(Random.randperm(R, n, m)), 1, :))
end

"`set_dynare_seed(...)` of a stand-alone interpreter in Octave mode (set_dynare_seed_local_options.m, the Octave branch)."
function _oct_set_dynare_seed(I, a)
    isempty(a) && merr("set_dynare_seed:: I need at least one input argument!")
    length(a) > 1 && throw(Unsupported("set_dynare_seed with a RandStream algorithm (MATLAB's generators are not reproduced)", "random"))
    R = I.rng
    x = a[1]
    seed = 0.0
    if x isa AbstractString
        k = lowercase(x)
        if k == "default"
            seed = 0.0
        elseif k == "clock"
            seed = Float64(floor((time() / 86400.0 + 719529.0) * 86400.0))
        elseif k != "reset"
            merr("set_dynare_seed:: Something is wrong in the calling sequence!")
        else
            seed = get(I.globals, "__dynare_seed__", 0.0)
        end
    else
        v = scalar(x, "set_dynare_seed")
        (isfinite(v) && v == floor(v)) || merr("set_dynare_seed:: Something is wrong in the calling sequence!")
        seed = v
    end
    I.globals["__dynare_seed__"] = seed
    R.u = _oct_mt(seed); R.n = _oct_mt(seed)
    Any[]
end

function _oct_rng_fn(I, a, nout)
    length(a) > 2 && merr("Invalid call to rng")
    R = _octave_rng(I)
    current() = begin
        s = MStruct()
        s.Type = "twister"
        s.Seed = "Not applicable"
        s.State = Cell(Any[reshape(octave_state(R.u), :, 1), reshape(octave_state(R.n), :, 1)], (1, 2))
        s
    end
    isempty(a) && return current()
    x = a[1]
    gen = nothing
    if length(a) == 2
        a[2] isa AbstractString || merr("rng: GENERATOR must be a string")
        gen = lowercase(a[2])
        gen in ("simdtwister", "combrecursive", "philox", "threefry", "multfibonacci", "v4") &&
            merr("rng: random number generator \"$gen\" is not available in Octave")
        gen in ("twister", "v5uniform", "v5normal") || merr("rng: unknown random number generator \"$gen\"")
    end
    local su, sn
    if (x isa Real || x isa AbstractMatrix{Float64}) && length(x) == 1 && scalar(x) >= 0
        su = sn = x
    elseif x isa AbstractString && lowercase(x) == "shuffle"
        throw(Unsupported("rng('shuffle'): Octave then seeds from the clock and system entropy", "random"))
    elseif x isa AbstractString && lowercase(x) == "default" && length(a) == 1
        gen, su, sn = "twister", 0.0, 0.0
    elseif x isa MStruct && length(a) == 1
        Set(keys(x)) == Set(["Type", "Seed", "State"]) || merr("rng: input structure requires \"Type\", \"Seed\", \"State\" fields\"")
        x["Type"] isa AbstractString || _unsup("rng: a structure whose Type is not a char array")
        gen = x["Type"]
        st = x["State"]
        if st isa Cell
            length(st.items) == 2 || merr("rng: State must hold the states of rand and randn")
            su, sn = st.items
        else
            su = sn = st
        end
    else
        merr("Invalid call to rng")
    end
    gen = something(gen, "twister")
    gen == "twister" || throw(Unsupported("rng with generator $(repr(gen)) (Octave's old generators are not reproduced)", "random"))
    cur = nout > 0 ? current() : nothing
    u, n = _oct_mt(su), _oct_mt(sn)
    R.u, R.n = u, n
    cur === nothing ? Any[] : cur
end

"Octave-mode builtins; `nothing` when the call is not one or the interpreter is in JDyn's default mode."
function rng_hook(I, name::String, a::Vector{Any}, nout::Int)
    name in ("rand", "randn", "randi", "randperm", "rng", "set_dynare_seed") || return nothing
    _octave_rng(I) === nothing && return nothing
    name == "rand" && return _oct_randfun(I, "rand", a, :u)
    name == "randn" && return _oct_randfun(I, "randn", a, :n)
    name == "randi" && return _oct_randi(I, a)
    name == "randperm" && return _oct_randperm(I, a)
    name == "rng" && return _oct_rng_fn(I, a, nout)
    # set_dynare_seed: a model's own stream is seeded by seeds.jl; a stand-alone interpreter here
    (I.rng isa OctaveRNG && I.session === nothing) && return _oct_set_dynare_seed(I, a)
    nothing
end

# Random.seed! of an Octave pair: both states from `seed` (rand('state', seed); randn('state', seed))
function Random.seed!(r::OctaveRNG, seed::Integer)
    r.u = _oct_mt(Float64(seed)); r.n = _oct_mt(Float64(seed))
    r
end
Random.seed!(r::OctaveRNG) = Random.seed!(r, 0)
