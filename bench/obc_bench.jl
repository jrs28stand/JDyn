# Benchmark harness for JDyn's OBC methods: P1-P4 and R1 of /home/claude/work/bench/BENCH_PROTOCOL.md.
#
#   nice julia --project=<env> bench/obc_bench.jl [--tag NAME] [--models zlb,zlb2,nkm,sw] [--parts p1,p2,p3,p4,r1]
#                                                 [--reps 5] [--out DIR]
#
# <env> must have JDyn (this repository). If it also has a package `JDynBase` (a copy of the baseline source with the module
# renamed, see docs/OBC_PERF.md), every measurement is made on both in the same process, interleaved (protocol: "time before
# and after in the same process"); JDynBase is then the BEFORE column and JDyn the AFTER column. Without JDynBase only
# JDyn is measured.
#
# Methods: occbin (the default), minnorm (with the shared active-set fast path) and minnorm_pure (obc_fast_path=false).
# `news` is a stub in JDyn and `sw` is a multi_equation constraint that the news methods refuse: sw is measured with OccBin only.
#
# Conventions (all recorded in the output): horizon H: OccBin check_ahead_periods = H; minnorm obc_horizon = H and
# check_ahead_periods = H. Shocks: standard normal draws (Random.MersenneTwister(SEED)) times the model's shock standard
# deviations (0.01 where zero) times the scale s. Timings: one warm-up call, GC.gc() before every timed call, at least 5
# repeats interleaved over the cells (the cell order rotates each repeat), minimum and median reported.
# BLAS runs on one thread.

using Random, LinearAlgebra, Printf, Logging, Statistics
using JDyn
const BACKENDS = Pair{Symbol,Module}[:after=>JDyn]
if Base.find_package("JDynBase") !== nothing
    @eval using JDynBase
    pushfirst!(BACKENDS, :before => getfield(Main, :JDynBase))
end
LinearAlgebra.BLAS.set_num_threads(1)

const SEED = 20261124
const REPO = normpath(joinpath(@__DIR__, ".."))
const DYNARE_SRC = get(ENV, "JDYN_DYNARE_SRC", "/home/claude/ref/dynare-7.2")

# ------------------------------------------------------------------ command line
function parse_args(args)
    o = Dict{String,String}("tag" => "run", "models" => "zlb,zlb2,nkm,sw", "parts" => "p1,p2,p3,p4,r1", "reps" => "5",
                            "out" => joinpath(REPO, "bench", "results"))
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument $a")
        o[a[3:end]] = args[i+1]
        i += 2
    end
    o
end
const OPTS = parse_args(ARGS)
const REPS = parse(Int, OPTS["reps"])
const PARTS = Set(split(OPTS["parts"], ","))
mkpath(OPTS["out"])
const OUTIO = open(joinpath(OPTS["out"], "obc_bench_$(OPTS["tag"]).md"), "w")
function say(args...)
    println(stdout, args...); println(OUTIO, args...); flush(stdout); flush(OUTIO)
end

# ------------------------------------------------------------------ models
struct ModelSpec
    id::String
    path::String
    params::NTuple{2,String}        # the two parameters perturbed in P3 (they do not move the steady state)
    news::Bool                      # news methods apply (complementarity)
    fixed_scale::Union{Nothing,Float64}   # skip the scale search (see the sw note in docs/OBC_PERF.md)
end
const MODELS = Dict(
    "zlb" => ModelSpec("zlb", joinpath(REPO, "test", "obc", "obc_zlb.mod"), ("phi", "rho"), true, nothing),
    "zlb2" => ModelSpec("zlb2", joinpath(REPO, "test", "obc", "obc_zlb2.mod"), ("phi", "rho"), true, nothing),
    "nkm" => ModelSpec("nkm", joinpath(DYNARE_SRC, "tests", "occbin", "filter", "NKM_0_std_shocks.mod"), ("phipi", "phiy"), true, nothing),
    "sw" => ModelSpec("sw", joinpath(REPO, "test", "obc", "obc_sw_zlb.mod"), ("crpi", "cry"), false, 1.0),
)
const MODEL_ORDER = ["zlb", "zlb2", "nkm", "sw"]

# method name => simulate keywords (H is added by `simkw`)
const METHOD_KW = Dict(:occbin => NamedTuple(), :minnorm => (obc_method=:minnorm,),
                       :minnorm_pure => (obc_method=:minnorm, obc_fast_path=false))
methods_of(spec::ModelSpec) = spec.news ? [:occbin, :minnorm, :minnorm_pure] : [:occbin]
simkw(method::Symbol, H::Int) = method == :occbin ? (check_ahead_periods=H,) :
                                merge(METHOD_KW[method], (obc_horizon=H, check_ahead_periods=H))

"Load a copy of the file's whole folder (include files); the preprocessor run is not part of any timing."
function load_dir(J::Module, path)
    d = mktempdir()
    for f in readdir(dirname(path))
        p = joinpath(dirname(path), f)
        isfile(p) && cp(p, joinpath(d, f))
    end
    with_logger(NullLogger()) do
        J.load(joinpath(d, basename(path)))
    end
end

struct Loaded
    spec::ModelSpec
    m::Any
    ob::Any
    sd::Vector{Float64}
end
function load_model(J::Module, spec::ModelSpec)
    m = load_dir(J, spec.path)
    ob = J.Occbin(m)
    sd = [s == 0 ? 0.01 : s for s in sqrt.(diag(m.Sigma_e))]
    Loaded(spec, m, ob, sd)
end

"Standard normal draws times sd times s: (S x n_exo)."
unit_shocks(S, sd) = randn(MersenneTwister(SEED), S, length(sd)) .* sd'

# ------------------------------------------------------------------ timing helpers
tmin(v) = minimum(v)
tmed(v) = median(v)

"""
Interleaved timing. `cells` is a vector of `label => f`; `f()` returns anything. One warm-up call of each, then `reps` rounds in which
every cell is timed once (GC.gc() before each call, order rotated every round). Returns label => (times, last result).
"""
function interleave(cells, reps::Int)
    for (_, f) in cells
        f()
    end
    times = Dict(l => Float64[] for (l, _) in cells)
    last = Dict{Any,Any}()
    n = length(cells)
    for r in 1:reps
        for i in circshift(1:n, -(r - 1))
            l, f = cells[i]
            GC.gc()
            local res
            t = @elapsed res = f()
            push!(times[l], t)
            last[l] = res
        end
    end
    Dict(l => (times[l], last[l]) for (l, _) in cells)
end

fmt(x; d=1) = @sprintf("%.*f", d, x)
mm(v; scale=1.0, d=1) = string(fmt(tmin(v) * scale; d=d), " / ", fmt(tmed(v) * scale; d=d))

# ------------------------------------------------------------------ simulation helpers
"The 'binding in the current period' and 'binding somewhere on the path' indicators of a result."
function binding_flags(res)
    now_, any_ = Bool[], Bool[]
    for per in res.regime_history
        push!(now_, any(x -> x[1][1] != 0, per))
        push!(any_, any(x -> any(!=(0), x[1]), per))
    end
    now_, any_
end

function run_sim(J::Module, L::Loaded, shocks, method, H; init=nothing, periods=size(shocks, 1), ticks=nothing)
    kw = simkw(method, H)
    if ticks !== nothing
        empty!(ticks); sizehint!(ticks, size(shocks, 1) + 2)
        J.OBC_TICKS[] = ticks
    end
    try
        init === nothing ? J.simulate(L.ob, shocks; periods=periods, kw...) :
                           J.simulate(L.ob, shocks; periods=periods, init=init, kw...)
    finally
        ticks !== nothing && (J.OBC_TICKS[] = nothing)
    end
end

"Failure of a call as a short label (OccBin's error_flag, ObcError's status, or the exception type)."
function failure_label(J::Module, e)
    if e isa J.OccbinError
        "occbin $(J.error_flag(e))"
    elseif e isa J.ObcError
        "obc $(e.status)"
    else
        "error " * string(typeof(e).name.name)
    end
end

# ------------------------------------------------------------------ scale of the shocks
const SCALE_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 100.0]

"""
Shock matrix `unit .* s`, repaired where a method fails: whenever a method in `meths` raises an error at shock period k of the full
simulation, row k is replaced by a fresh draw (from a separate seeded stream, same distribution) and the simulation is run again.
Returns (shocks, number of redraws, true if the cap on re-runs was hit). Models where nothing fails need no redraws.
"""
function make_shocks(J::Module, L::Loaded, unit, s, H, meths; maxruns=4000)
    shocks = unit .* s
    rng = MersenneTwister(SEED + 4)
    redraws = 0
    for run in 1:maxruns
        bad = 0
        for meth in meths
            try
                run_sim(J, L, shocks, meth, H)
            catch e
                (e isa J.OccbinError || e isa J.ObcError) || rethrow()
                bad = e.period
                break
            end
        end
        bad == 0 && return shocks, redraws, false
        shocks[bad, :] = randn(rng, size(shocks, 2)) .* L.sd .* s
        redraws += 1
    end
    shocks, redraws, true
end

"""
Smallest s in the grid for which the constraint binds in 30-60% of the 1000 shock periods (current period in a binding regime) and no
method fails. If no such s exists without redraws (sw: OccBin fails in about a tenth of the periods at every scale), the smallest s in
the window with repaired shocks (`make_shocks`) is used and the number of redraws is reported.
Returns (s, share, shocks, redraws, log lines).
"""
function choose_scale(J::Module, L::Loaded, unit, H)
    log = String[]
    cand = nothing
    meths = methods_of(L.spec)
    for s in SCALE_GRID
        shocks, nred, capped = make_shocks(J, L, unit, s, H, meths)
        capped && (push!(log, "s=$s: still failing after the cap on re-runs"); continue)
        res = run_sim(J, L, shocks, :occbin, H)
        share = mean(binding_flags(res)[1])
        push!(log, @sprintf("s=%g share=%.0f%% redraws=%d", s, 100share, nred))
        if 0.30 <= share <= 0.60
            nred == 0 && return s, share, shocks, nred, log
            cand === nothing && (cand = (s, share, shocks, nred))
        end
        share > 0.60 && break
    end
    cand === nothing ? (NaN, NaN, nothing, 0, log) : (cand..., log)
end

# ------------------------------------------------------------------ P1
struct P1Row
    method::Symbol
    backend::Symbol
    total_us::Vector{Float64}         # per repeat, microseconds per shock period (whole call)
    quiet_us::Vector{Float64}         # periods where nothing binds anywhere on the path
    bind_us::Vector{Float64}          # periods where something binds
    kib_per_period::Float64
    res::Any
end

function p1(backs, loaded::Dict{Symbol,Loaded}, shocks, H; reps=REPS, meths=methods_of(first(values(loaded)).spec))
    S = size(shocks, 1)
    cells = Pair{Any,Function}[]
    stats = Dict{Any,Tuple{Vector{Float64},Vector{Float64}}}()
    for (b, J) in backs, meth in meths
        key = (meth, b)
        stats[key] = (Float64[], Float64[])
        L = loaded[b]
        ticks = UInt64[]
        push!(cells, key => function ()
            res = run_sim(J, L, shocks, meth, H; ticks=ticks)
            nowb, anyb = binding_flags(res)
            d = diff(Int64.(ticks))[1:S] ./ 1e3                    # microseconds per period
            push!(stats[key][1], isempty(d[.!anyb]) ? NaN : mean(d[.!anyb]))
            push!(stats[key][2], isempty(d[anyb]) ? NaN : mean(d[anyb]))
            res
        end)
    end
    out = interleave(cells, reps)
    for v in values(stats)                                      # the first call of each cell is the warm-up
        popfirst!(v[1]); popfirst!(v[2])
    end
    rows = P1Row[]
    for (b, J) in backs, meth in meths
        key = (meth, b)
        times, res = out[key]
        L = loaded[b]
        alloc = @allocated run_sim(J, L, shocks, meth, H)
        push!(rows, P1Row(meth, b, times ./ S .* 1e6, stats[key][1], stats[key][2], alloc / S / 1024, res))
    end
    rows
end

"Maximum path difference and regime agreement of a result against a reference result."
function compare_results(res, ref)
    d = maximum(abs.(res.piecewise .- ref.piecewise))
    d, res.regime_history == ref.regime_history
end

function report_p1(rows, H, s, share, S)
    ref = first(r for r in rows if r.method == :occbin && r.backend == :after).res
    nowb, anyb = binding_flags(ref)
    say("| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |")
    say("|---|---|---|---|---|---|---|---|")
    for r in rows
        d, same = compare_results(r.res, ref)
        say("| ", r.method, " | ", r.backend, " | ", mm(r.total_us), " | ", mm(filter(!isnan, r.quiet_us)), " | ", mm(filter(!isnan, r.bind_us)),
            " | ", fmt(r.kib_per_period; d=1), " | ", @sprintf("%.1e", d), " | ", same, " |")
    end
    say("")
    say("Shock scale s = $s, horizon $H, $S shock periods. Binding in the current period in ", fmt(100mean(nowb); d=1),
        "% of periods; something binds somewhere on the path in ", fmt(100mean(anyb); d=1), "% (the split above is by that second indicator). ",
        "Timings: $(length(first(rows).total_us)) repeats, interleaved, min / median.")
    say("")
end

# ------------------------------------------------------------------ P2
"Parameter vectors for P2/P3: the two named parameters moved by a random walk of relative steps of 1e-3 (deterministic)."
function perturbed_params(L::Loaded, n)
    m = L.m
    base = copy(m.params)
    rng = MersenneTwister(SEED + 1)
    out = Vector{Float64}[]
    p = copy(base)
    for i in 1:n
        for nm in L.spec.params
            k = m.ip[nm]
            p[k] *= 1 + 1e-3 * randn(rng)
        end
        push!(out, copy(p))
    end
    out
end

const P2_PARTS = [:steady_state, :first_order, :Occbin, :regimes, :obc_constraints, :obc_system]

function p2(backs, loaded::Dict{Symbol,Loaded}, H; reps=REPS)
    L1 = first(values(loaded))
    ps = perturbed_params(L1, reps + 1)                 # +1: the warm-up call uses the first
    parts = Dict((b, k) => Float64[] for (b, _) in backs for k in P2_PARTS)
    counter = Dict(b => Ref(0) for (b, _) in backs)
    cells = Pair{Any,Function}[]
    for (b, J) in backs
        L = loaded[b]
        push!(cells, b => function ()
            counter[b][] += 1
            p = ps[min(counter[b][], length(ps))]
            tick() = time_ns()
            m = L.m
            pp = copy(p)
            for ip in [m.ip["occbin_$(c["name"])_bind"] for c in m.occbin_constraints]
                pp[ip] = 0.0
            end
            wr = counter[b][] > 1
            t0 = tick(); ys, p2_ = J.steady_state(m; params=pp); t1 = tick()
            fo = J.first_order(m; ys=ys, params=p2_, loglinear=false); t2 = tick()
            ob = J.Occbin(m; params=p); t3 = tick()
            for reg in Iterators.product(ntuple(_ -> (false, true), ob.k)...)
                J.regime_matrices(ob, reg)
            end
            t4 = tick()
            nc = L.spec.news ? J.obc_constraints(ob) : nothing; t5 = tick()
            ob2 = J.Occbin(m; params=p); t6 = tick()
            L.spec.news && J.obc_system(ob2; horizon=H); t7 = tick()
            if wr
                push!(parts[(b, :steady_state)], (t1 - t0) / 1e6); push!(parts[(b, :first_order)], (t2 - t1) / 1e6)
                push!(parts[(b, :Occbin)], (t3 - t2) / 1e6); push!(parts[(b, :regimes)], (t4 - t3) / 1e6)
                push!(parts[(b, :obc_constraints)], (t5 - t4) / 1e6); push!(parts[(b, :obc_system)], (t7 - t6) / 1e6)
            end
            nothing
        end)
    end
    # the warm-up call of `interleave` is the first call of each cell; `wr` skips it
    interleave(cells, reps)
    parts
end

function report_p2(parts, backs, spec::ModelSpec, H)
    say("| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |")
    say("|---|---|---|---|---|---|---|---|---|")
    for (b, _) in backs
        g(k) = parts[(b, k)]
        rd_o = g(:Occbin) .+ g(:regimes)
        rd_m = g(:Occbin) .+ g(:obc_system)
        say("| ", b, " | ", mm(g(:steady_state); d=2), " | ", mm(g(:first_order); d=2), " | ", mm(g(:Occbin); d=2), " | ", mm(g(:regimes); d=2), " | ",
            spec.news ? mm(g(:obc_constraints); d=2) : "-", " | ", spec.news ? mm(g(:obc_system); d=2) : "-", " | ", mm(rd_o; d=2), " | ",
            spec.news ? mm(rd_m; d=2) : "-", " |")
    end
    say("")
    say("Milliseconds, min / median over $(length(parts[(first(backs)[1], :Occbin)])) interleaved repeats, a different perturbed parameter vector in each (two parameters: ",
        join(spec.params, ", "), "); horizon $H.")
    say("")
end

# ------------------------------------------------------------------ P3
function p3(backs, loaded::Dict{Symbol,Loaded}, shocks200, H; iters=50, reps=REPS, meths=methods_of(first(values(loaded)).spec))
    L1 = first(values(loaded))
    ps = perturbed_params(L1, iters)
    fails = Dict{Any,Dict{String,Int}}()
    cells = Pair{Any,Function}[]
    for (b, J) in backs, meth in meths
        L = loaded[b]
        key = (meth, b)
        fails[key] = Dict{String,Int}()
        push!(cells, key => function ()
            empty!(fails[key])
            for i in 1:iters
                ob = J.Occbin(L.m; params=ps[i])
                Li = Loaded(L.spec, L.m, ob, L.sd)
                try
                    run_sim(J, Li, shocks200, meth, H; periods=size(shocks200, 1))
                catch e
                    lab = failure_label(J, e)
                    fails[key][lab] = get(fails[key], lab, 0) + 1
                end
            end
            nothing
        end)
    end
    out = interleave(cells, reps)
    out, fails, ps
end

"Path differences of every method against OccBin (backend :after) over the P3 iterations, and OccBin before vs after (bit for bit)."
function p3_check(backs, loaded, shocks200, H, ps, meths)
    Jn = backs[findfirst(x -> x[1] == :after, backs)].second
    L = loaded[:after]
    worst = Dict(m => 0.0 for m in meths)
    regs = Dict(m => 0 for m in meths)
    bit = true
    for (i, p) in enumerate(ps)
        ob = Jn.Occbin(L.m; params=p)
        Li = Loaded(L.spec, L.m, ob, L.sd)
        ref = try run_sim(Jn, Li, shocks200, :occbin, H) catch; nothing end
        ref === nothing && continue
        for meth in meths
            meth == :occbin && continue
            r = try run_sim(Jn, Li, shocks200, meth, H) catch; nothing end
            r === nothing && continue
            d, same = compare_results(r, ref)
            worst[meth] = max(worst[meth], d); regs[meth] += !same
        end
        if length(backs) > 1
            Jb = backs[findfirst(x -> x[1] == :before, backs)].second
            Lb = loaded[:before]
            obb = Jb.Occbin(Lb.m; params=p)
            rb = try run_sim(Jb, Loaded(Lb.spec, Lb.m, obb, Lb.sd), shocks200, :occbin, H) catch; nothing end
            rb === nothing || (bit &= isequal(rb.piecewise, ref.piecewise) && rb.regime_history == ref.regime_history)
        end
    end
    worst, regs, bit
end

# ------------------------------------------------------------------ R1
const R1_SCALES = [0.5, 1.0, 1.5, 2.0]

function r1(backs, loaded::Dict{Symbol,Loaded}, H; nstates=2000)
    J = backs[findfirst(x -> x[1] == :after, backs)].second          # reliability is a property of the current code
    L = loaded[:after]
    ob = L.ob
    n, nx = ob.n, ob.n_exo
    # 2000 states from a long unconstrained simulation at s = 1: 10200 periods, drop the first 200, keep every 5th
    rng = MersenneTwister(SEED + 2)
    x = zeros(n)
    states = Vector{Float64}[]
    for t in 1:10200
        x = ob.P * x + ob.Q * (randn(rng, nx) .* L.sd)
        t > 200 && (t - 200) % 5 == 0 && push!(states, copy(x))
    end
    length(states) == nstates || error("expected $nstates states")
    draws = randn(MersenneTwister(SEED + 3), nstates, nx) .* L.sd'
    meths = methods_of(L.spec)
    results = Dict{Tuple{Float64,Symbol},Vector{Any}}()           # (s, method) => per state: (:ok, regime_history, y1) or (:fail, label)
    for s in R1_SCALES, meth in meths
        v = Any[]
        for i in 1:nstates
            sh = reshape(draws[i, :] .* s, 1, nx)
            try
                r = J.simulate(ob, sh; periods=1, init=states[i], simkw(meth, H)...)
                push!(v, (:ok, r.regime_history[1], r.piecewise[1, :]))
            catch e
                push!(v, (:fail, failure_label(J, e)))
            end
        end
        results[(s, meth)] = v
    end
    results, meths, nstates
end

function report_r1(results, meths, nstates)
    say("| s | method | solved | failed | failure types | solved by both it and OccBin | different regime sequences vs OccBin | max |path diff| vs OccBin (solved by both) |")
    say("|---|---|---|---|---|---|---|---|")
    for s in R1_SCALES, meth in meths
        v = results[(s, meth)]
        ok = count(x -> x[1] == :ok, v)
        types = Dict{String,Int}()
        for x in v
            x[1] == :fail && (types[x[2]] = get(types, x[2], 0) + 1)
        end
        ref = results[(s, :occbin)]
        both = [i for i in eachindex(v) if v[i][1] == :ok && ref[i][1] == :ok]
        diffreg = count(i -> v[i][2] != ref[i][2], both)
        dmax = isempty(both) ? NaN : maximum(maximum(abs.(v[i][3] .- ref[i][3])) for i in both)
        say("| ", s, " | ", meth, " | ", ok, " | ", nstates - ok, " | ", isempty(types) ? "-" : join(["$k: $c" for (k, c) in sort(collect(types))], ", "),
            " | ", meth == :occbin ? "-" : length(both), " | ", meth == :occbin ? "-" : diffreg, " | ", meth == :occbin ? "-" : @sprintf("%.1e", dmax), " |")
    end
    # failures of one method where another solves (for the record)
    say("")
    for s in R1_SCALES
        o = results[(s, :occbin)]
        for meth in meths
            meth == :occbin && continue
            v = results[(s, meth)]
            a = count(i -> o[i][1] == :fail && v[i][1] == :ok, eachindex(v))
            b = count(i -> o[i][1] == :ok && v[i][1] == :fail, eachindex(v))
            say("s = $s: $meth solves $a states where OccBin fails; OccBin solves $b states where $meth fails.")
        end
    end
    say("")
end

# ------------------------------------------------------------------ main
function main()
    say("# OBC benchmark run `", OPTS["tag"], "`")
    say("")
    say("- date: ", Libc.strftime("%Y-%m-%d %H:%M:%S", time()), "; julia ", VERSION, "; seed ", SEED, "; backends: ", join([string(b) for (b, _) in BACKENDS], ", "),
        " (before = baseline source commit ", isfile("/home/claude/work/jdynbase/BASE_COMMIT") ? strip(read("/home/claude/work/jdynbase/BASE_COMMIT", String)) : "n/a",
        "; after = ", strip(read(`git -C $REPO rev-parse --short HEAD`, String)), " + working tree)")
    say("- load average at start: ", strip(read("/proc/loadavg", String)), "; nproc ", Sys.CPU_THREADS, "; BLAS threads 1; repeats ", REPS)
    say("")
    for id in MODEL_ORDER
        id in split(OPTS["models"], ",") || continue
        spec = MODELS[id]
        say("## Model `", id, "` (", spec.news ? "OccBin, minnorm, minnorm without the fast path" : "OccBin only: multi_equation constraint, the news methods refuse it", ")")
        say("")
        loaded = Dict{Symbol,Loaded}(b => load_model(J, spec) for (b, J) in BACKENDS)
        L = loaded[:after]
        say("n_endo = ", L.ob.n, ", n_exo = ", L.ob.n_exo, ", constraints = ", L.ob.k, ".")
        say("")
        H = 40
        unit = unit_shocks(1000, L.sd)
        Jn = BACKENDS[findfirst(x -> x[1] == :after, BACKENDS)].second
        if spec.fixed_scale === nothing
            s, share, shocks1000, nred, tried = choose_scale(Jn, L, unit, H)
        else
            s = spec.fixed_scale
            shocks1000, nred, _ = make_shocks(Jn, L, unit, s, H, methods_of(spec))
            share = mean(binding_flags(run_sim(Jn, L, shocks1000, :occbin, H))[1])
            tried = ["s=$s fixed (no scale gives 30-60% binding without absurd shocks: see docs/OBC_PERF.md); share=$(round(100share; digits=1))% redraws=$nred"]
        end
        say("Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): ", join(tried, " | "))
        say("")
        if isnan(s)
            say("**No scale in the grid gives 30-60% binding; skipping this model's timings.**")
            say("")
            continue
        end
        nred > 0 && (say("**Failing draws were replaced: $nred of the 1000 shock periods were redrawn (fresh normal draw, same scale) until the simulation ran through with every method; the sequence is therefore not a pure iid normal sample.**"); say(""))
        if "p1" in PARTS
            say("### P1 per period (1000 shock periods, H = $H)")
            say("")
            rows = p1(BACKENDS, loaded, shocks1000, H)
            report_p1(rows, H, s, share, 1000)
        end
        if "p2" in PARTS
            say("### P2 per parameter set (H = $H)")
            say("")
            report_p2(p2(BACKENDS, loaded, H), BACKENDS, spec, H)
        end
        if "p3" in PARTS
            say("### P3 likelihood-like loop (50 iterations: perturb two parameters, redo P2, simulate 200 periods; H = $H)")
            say("")
            shocks200 = shocks1000[1:200, :]
            meths = methods_of(spec)
            out, fails, ps = p3(BACKENDS, loaded, shocks200, H; meths=meths)
            say("| method | backend | seconds per loop min / median | ms per iteration (min) | failed iterations |")
            say("|---|---|---|---|---|")
            for (b, _) in BACKENDS, meth in meths
                t = out[(meth, b)][1]
                f = fails[(meth, b)]
                say("| ", meth, " | ", b, " | ", mm(t; d=3), " | ", fmt(tmin(t) / 50 * 1e3; d=1), " | ", isempty(f) ? "0" : join(["$k: $c" for (k, c) in f], ", "), " |")
            end
            worst, regs, bit = p3_check(BACKENDS, loaded, shocks200, H, ps, meths)
            say("")
            say("Correctness over the 50 parameter sets: ", join(["$m: max path diff vs OccBin $(@sprintf("%.1e", worst[m])), $(regs[m]) iterations with different regime histories" for m in meths if m != :occbin], "; "),
                length(BACKENDS) > 1 ? "; OccBin before vs after identical (paths bit for bit, regime histories): $bit" : "", ".")
            say("")
        end
        if "p4" in PARTS && id in ("nkm", "sw")
            say("### P4 scaling with the horizon (s = $s)")
            say("")
            for Hh in (20, 40, 80)
                say("#### H = $Hh")
                say("")
                try
                    shH, nredH, _ = make_shocks(Jn, L, unit, s, Hh, methods_of(spec))
                    nredH > 0 && say("Failing draws replaced at this horizon: $nredH of 1000.")
                    rows = p1(BACKENDS, loaded, shH, Hh)
                    shareH = mean(binding_flags(first(r for r in rows if r.method == :occbin && r.backend == :after).res)[1])
                    report_p1(rows, Hh, s, shareH, 1000)
                catch e
                    say("P1 failed at H = $Hh: ", first(sprint(showerror, e), 300))
                end
                report_p2(p2(BACKENDS, loaded, Hh), BACKENDS, spec, Hh)
            end
        end
        if "r1" in PARTS
            say("### R1 reliability (2000 states from a long unconstrained simulation, one period solved from each, H = $H)")
            say("")
            say("No method has a time limit; OccBin's own limit is maxit = 30 iterations (counted as its error 311). Different regime sequences are counted among the states both methods solve.")
            say("")
            results, meths, nst = r1(BACKENDS, loaded, H)
            report_r1(results, meths, nst)
        end
        say("load average at end of model: ", strip(read("/proc/loadavg", String)))
        say("")
    end
    close(OUTIO)
end

main()
