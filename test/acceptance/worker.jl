# One Julia session that runs tests (started and supervised by run_dynare_tests.jl).
using JDyn, Logging
include(joinpath(@__DIR__, "common.jl"))
# ---------------------------------------------------------------------------------------------- worker
function worker_main(o)
    ENV["DYNARE_PREPROCESSOR"] = get(ENV, "DYNARE_PREPROCESSOR", "/root/ref/pp-standalone/src/dynare-preprocessor")
    entries = read_manifest(; shard=o["shard"], pat=o["pat"])
    resfile = o["worker"]
    RESFILE[] = resfile
    done = Set{String}()
    isfile(resfile) && for l in eachline(resfile); p = split(l, TAB); push!(done, "$(p[1])|$(p[2])"); end
    scratch_root = get(ENV, "JDYN_ACCEPT_SCRATCH", "/var/tmp/jdyn-accept-$(getpid())")
    warm_up(scratch_root)
    first = true
    for e in entries
        todo = [t for t in e.tests if !(("$(e.id)|$t") in done)]
        isempty(todo) && continue
        base = joinpath(scratch_root, replace(e.id, '/' => '_'), "tests")
        mkpath(base)
        isfile(joinpath(base, ".copied")) || begin
            touch(joinpath(base, ".copied"))
            for f in vcat(e.tests, e.extras)
                src = normpath(joinpath(o["tests"], f)); dst = normpath(joinpath(base, f))
                isfile(src) || continue
                mkpath(dirname(dst)); cp(src, dst; force=true)
                # a `+package` directory or a Dynare test reading siblings is covered by `extra`, as in the driver
            end
        end
        for t in todo
            key = "$(e.id)|$t"
            # the first test of a fresh worker gets extra time: Julia compiles the code paths it uses
            open(joinpath(dirname(resfile), "current"), "w") do io; println(io, key, TAB, time() + (first ? 60 : 0)); end
            first = false
            r = run_one(e, t, base)
            open(resfile, "a") do io
                println(io, join([e.id, t, r.cat, round(r.secs; digits=1), r.warn, r.etype, r.msg, r.warns], TAB))
            end
        end
        rm(joinpath(scratch_root, replace(e.id, '/' => '_')); recursive=true, force=true)
    end
    open(joinpath(dirname(resfile), "finished"), "w") do io; println(io, "done"); end
end

"Run a small model once (not recorded) so that the common code paths are compiled before the first test."
function warm_up(scratch)
    d = joinpath(scratch, "warmup"); mkpath(d)
    write(joinpath(d, "w.mod"), """
var y c k z; varexo e; parameters alpha beta delta rho;
alpha = 0.36; beta = 0.99; delta = 0.025; rho = 0.9;
model;
c^(-1) = beta*c(+1)^(-1)*(alpha*exp(z(+1))*k^(alpha-1)+1-delta);
k = exp(z)*k(-1)^alpha - c + (1-delta)*k(-1);
z = rho*z(-1) + e;
y = exp(z)*k(-1)^alpha;
end;
initval; k = 10; c = 1.5; z = 0; y = 2; e = 0; end;
shocks; var e; stderr 0.01; end;
steady; check;
stoch_simul(order=1, irf=5, nograph, noprint);
""")

    cd(d)
    try
        Logging.with_logger(Logging.NullLogger()) do; JDyn.dynare(joinpath(d, "w.mod")); end
    catch e
        @warn "warm-up failed" e
    end
end

mutable struct CollectLogger <: Base.CoreLogging.AbstractLogger
    msgs::Vector{String}
end
Base.CoreLogging.min_enabled_level(::CollectLogger) = Base.CoreLogging.Warn
Base.CoreLogging.shouldlog(::CollectLogger, args...) = true
Base.CoreLogging.catch_exceptions(::CollectLogger) = true
function Base.CoreLogging.handle_message(l::CollectLogger, level, message, args...; kwargs...)
    level >= Base.CoreLogging.Warn && push!(l.msgs, string(message))
end

const RESFILE = Ref("")
clean(s) = replace(first(replace(String(s), r"\s+" => " "), 400), TAB => " ")

function classify(e, JD)
    if e isa JD.MatlabError
        return e.user ? ("CHECK_FAILED", "MatlabError", e.msg) : ("ERROR", "MatlabError", e.msg)
    elseif e isa JD.ModelError && hasfield(JD.ModelError, :cause) && e.cause isa JD.MatlabError && e.cause.user
        # the session wraps the MatlabError of the file's own error()/assert() in a ModelError
        return ("CHECK_FAILED", "MatlabError", e.cause.msg)
    elseif e isa JD.ModelError || e isa JD.Unsupported
        return ("REFUSED", string(nameof(typeof(e))), e.msg)
    elseif e isa JD.EstimationError
        return ("REFUSED", "EstimationError", sprint(showerror, e))
    end
    msg = sprint(showerror, e)
    if e isa InterruptException
        rethrow()
    end
    unsup = occursin(r"not (yet )?(supported|implemented)|unsupported|cannot yet|does not support"i, msg)
    nm = string(nameof(typeof(e)))
    (unsup ? "REFUSED" : "ERROR", nm, msg)
end

function run_one(e::Entry, t::String, base::String)
    JD = JDyn
    if endswith(t, "_data.m") && isfile(joinpath(base, t))
        # a data-generating script that later tests of the chain read (Dynare's test order runs it first, in the same directory)
        cd(dirname(joinpath(base, t)))
        t0 = time()
        try
            body, funcs = JDyn.mparse(read(joinpath(base, t), String))
            I = JDyn.Interp(dirname(joinpath(base, t)), nothing)
            I.jlfuncs["ispc"] = () -> 0.0           # the xlswrite part of the script is for Windows only
            sc = JDyn.Scope(; vars=Dict{String,Any}())
            push!(I.scope_stack, sc)
            allow0 = JDyn.ALLOW_RANDOM[]; JDyn.ALLOW_RANDOM[] = true
            try; JDyn.run_script(I, JDyn.Unit(joinpath(base, t), body, funcs), sc); finally; JDyn.ALLOW_RANDOM[] = allow0; end
            return (cat="OK", secs=time() - t0, warn=0, etype="", msg="", warns="")
        catch ex
            ex isa InterruptException && rethrow()
            return (cat="ERROR", secs=time() - t0, warn=0, etype=string(nameof(typeof(ex))), msg=clean(sprint(showerror, ex)), warns="")
        end
    end
    if !endswith(t, ".mod")
        return (cat="SKIPPED", secs=0.0, warn=0, etype="", msg=".m script: not a model run", warns="")
    end
    path = joinpath(base, t)
    isfile(path) || return (cat="ERROR", secs=0.0, warn=0, etype="Missing", msg="test file not found in the tests tree", warns="")
    cd(dirname(path))
    lg = CollectLogger(String[])
    t0 = time()
    skipped = Any[]
    cat, et, msg = try
        ws = Logging.with_logger(lg) do
            JDyn.dynare(path; extra_args=e.args)
        end
        skipped = try; collect(ws.model.matlab_skipped); catch; Any[]; end
        ("OK", "", "")
    catch ex
        # the master sends SIGINT when the test overruns: Julia raises it as an InterruptException
        ex isa InterruptException ? ("TIMEOUT", "", "interrupted: over the time limit") : classify(ex, JD)
    end
    secs = time() - t0
    GC.gc()
    if e.should_fail
        cat = cat == "OK" ? "XPASS" : (cat == "TIMEOUT" ? cat : "XFAIL")
    end
    if !isempty(skipped)       # every MATLAB statement JDyn did not execute: test, kind, reason, text
        open(joinpath(dirname(RESFILE[]), "skipped.tsv"), "a") do io
            for (txt, kind, why) in skipped
                println(io, join([t, kind, clean(sprint(showerror, why isa Exception ? why : ErrorException(string(why)))), clean(txt)], TAB))
            end
        end
    end
    ws = unique(lg.msgs)
    cat == "OK" && !isempty(ws) && (cat = "OK_WARN")
    (cat=cat, secs=secs, warn=length(lg.msgs), etype=et, msg=clean(msg), warns=clean(join(first.(ws, 300), " || ")))
end


Base.exit_on_sigint(false)       # SIGINT from the master (time limit) becomes an InterruptException
worker_main(parse_args(ARGS))
