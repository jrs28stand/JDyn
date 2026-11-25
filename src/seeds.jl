# Reproducible random draws: Dynare's set_dynare_seed / options_.DynareRandomStreams for JDyn.
#
# Every model (hence every workspace made by dynare()) owns one stream, `dynare_rng(m)`, a Random.Xoshiro that is
# re-seeded by set_dynare_seed. Like Dynare's global stream it starts from the 'default' seed 0, so a run is
# reproducible unless set_dynare_seed('clock') asks otherwise, and consecutive commands continue the same stream.
# JDyn cannot reproduce MATLAB's Mersenne-twister (or Octave's) draw for draw: the draws are Julia's, the same for
# the same seed on every run, but not MATLAB's.
#
# Every random draw of JDyn uses this stream unless an explicit `rng=` is given: simulated shocks of stoch_simul
# (periods > 0), the GIRFs of order > 1, MCMC chains and csminwel's random Hessian perturbation in estimation, the
# random initial guesses of the steady-state solver, forecast draws.

mutable struct RandomStreams
    algo::String          # as Dynare reports it; JDyn's generator is always Xoshiro256++
    seed::Int
    rng::Random.AbstractRNG
end

const _STREAMS = WeakKeyDict{Any,RandomStreams}()
const _STREAMS_LOCK = ReentrantLock()
const DEFAULT_SEED = 0
const DYNARE_RNG_ALGOS = ("mcg16807", "mlfg6331_64", "mrg32k3a", "mt19937ar", "shr3cong", "swb2712")

_streams(m) = lock(_STREAMS_LOCK) do
    get!(() -> RandomStreams("mt19937ar", DEFAULT_SEED, Random.Xoshiro(DEFAULT_SEED)), _STREAMS, m)
end

"""
    use_octave_random!(m::Model)

Make the model's stream (and the interpreter's `rand`/`randn`/`randi`/`randperm`/`rng`) Octave's generators, seeded with the
current seed: draws are then Octave's numbers bit for bit (`dynare(file; random=:octave)`; src/matlab/rng.jl).
"""
function use_octave_random!(m::Model)
    st = _streams(m)
    st.rng isa OctaveRNG || (st.rng = OctaveRNG(st.seed))
    set_rng_kind!(m, :octave)
    set_mcmc_rng!(m, :octave)
    m
end

"The model's random stream (Dynare's global stream for this workspace)."
dynare_rng(m::Model) = _streams(m).rng

"Seed as Dynare's: `rem(floor(now*24*60*60), 2^32)` for 'clock', derived from the wall clock."
_clock_seed() = Int(rem(floor(Int, time()), 2^32))

function _store_streams!(m::Model, st::RandomStreams)
    s = MStruct()
    s.algo = st.algo
    s.seed = Float64(st.seed)
    m.matlab_options["DynareRandomStreams"] = s
    st
end

"""
    set_dynare_seed!(m::Model, seed)            seed: an integer, "default" (0), "clock" or "reset" (same seed again)
    set_dynare_seed!(m::Model, algo, seed)      algo as in Dynare (accepted, recorded; the generator is JDyn's)

Dynare's `set_dynare_seed` for one model: re-seeds its stream and records `options_.DynareRandomStreams.{algo,seed}`.
"""
function set_dynare_seed!(m::Model, a, b=nothing)
    st = _streams(m)
    if b !== nothing
        a isa AbstractString && lowercase(a) in DYNARE_RNG_ALGOS ||
            throw(ArgumentError("set_dynare_seed: the first argument must name a uniform random number algorithm ($(join(DYNARE_RNG_ALGOS, ", ")))"))
        b isa Real && isinteger(b) || throw(ArgumentError("set_dynare_seed: the seed must be an integer"))
        st.algo = lowercase(a); st.seed = Int(b)
    elseif a isa AbstractString
        la = lowercase(a)
        if la == "default"
            st.algo = "mt19937ar"; st.seed = DEFAULT_SEED
        elseif la == "reset"
        elseif la == "clock"
            st.algo = "mt19937ar"; st.seed = _clock_seed()
        else
            throw(ArgumentError("set_dynare_seed: something is wrong in the calling sequence ('$a')"))
        end
    elseif a isa Real && isinteger(a)
        st.algo = "mt19937ar"; st.seed = Int(a)
    else
        throw(ArgumentError("set_dynare_seed: the seed must be an integer, 'default', 'clock' or 'reset'"))
    end
    Random.seed!(st.rng, st.seed)
    _store_streams!(m, st)
end

# the MATLAB builtin: called by the interpreter for `set_dynare_seed(...)` lines of a .mod file
function _matlab_set_dynare_seed(I, a)
    isempty(a) && merr("set_dynare_seed:: I need at least one input argument!")
    length(a) > 2 && throw(Unsupported("set_dynare_seed with $(length(a)) arguments", "feature"))
    s = I.session
    s === nothing && return                                  # no model to seed
    conv(x) = x isa AbstractString ? String(x) : (x isa AbstractArray && length(x) == 1 ? x[1] : x)
    args = Any[conv(x) for x in a]
    try
        set_dynare_seed!(s.model, args...)
    catch e
        e isa ArgumentError && merr(e.msg)
        rethrow()
    end
    nothing
end

"""
    dynare_rng(ws::DynareWorkspace)

The workspace's stream. If `options_.DynareRandomStreams.seed` was changed (from Julia or by MATLAB lines of the
.mod file) since the stream was last seeded, the stream is re-seeded from it first.
"""
function dynare_rng(ws)
    m = ws.model
    st = _streams(m)
    o = get(ws.options_, "DynareRandomStreams", nothing)
    if o isa MStruct && haskey(o, "seed")
        sd = o["seed"]
        if sd isa Real && isfinite(sd) && isinteger(sd) && Int(sd) != st.seed
            a = haskey(o, "algo") && o["algo"] isa AbstractString ? String(o["algo"]) : st.algo
            set_dynare_seed!(m, a, Int(sd))
        end
    end
    st.rng
end

"""
    set_dynare_seed!(ws, seed) / set_dynare_seed!(ws, algo, seed) / set_dynare_seed!(seed)

Re-seed the workspace's stream (the last `dynare` workspace without `ws`) and update `ws.options_.DynareRandomStreams`.
After `set_dynare_seed!(ws, n)` a command draws what the first command of a `dynare(file; seed=n)` run draws.
"""
function set_dynare_seed!(ws::DynareWorkspace, a, b=nothing)
    st = set_dynare_seed!(ws.model, a, b)
    ws.options_.DynareRandomStreams = MStruct(["algo" => st.algo, "seed" => Float64(st.seed)])
    ws.state[:matlab_options]["DynareRandomStreams"] = deepcopy(ws.model.matlab_options["DynareRandomStreams"])
    ws
end
set_dynare_seed!(a, b=nothing) = set_dynare_seed!(_current(), a, b)
