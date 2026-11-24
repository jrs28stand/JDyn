# C.1: cost of the timing hook (`_obc_tick()`, one Ref load and a branch per shock period) left in src/occbin.jl.
#   nice julia --project=. bench/obc_hook_cost.jl [scratch-dir] [reps]
# The scratch copy (default /tmp/jdyn-nohook, OUTSIDE the repository) is built here from src/: module renamed to JDynNoHook and the two
# `_obc_tick()` call sites of src/occbin.jl deleted. Both modules are loaded in this process; OccBin P1 (1000 shock periods, horizon 40) is
# timed on both, interleaved, `reps` repeats, order rotated, GC.gc() before each call, and the results are compared bit for bit.
using Random, LinearAlgebra, Printf, Logging, Statistics
using JDyn
LinearAlgebra.BLAS.set_num_threads(1)
const REPO = normpath(joinpath(@__DIR__, ".."))
const SCR = length(ARGS) >= 1 ? ARGS[1] : "/tmp/jdyn-nohook"
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 11
const SEED = 20261124
const DYNARE_SRC = get(ENV, "JDYN_DYNARE_SRC", "/home/claude/ref/dynare-7.2")

function build_scratch()
    rm(SCR; recursive=true, force=true)
    mkpath(SCR)
    cp(joinpath(REPO, "src"), joinpath(SCR, "src"))
    mkpath(joinpath(SCR, "deps", "korder"))
    cp(joinpath(REPO, "deps", "korder", "libjdyn_korder.so"), joinpath(SCR, "deps", "korder", "libjdyn_korder.so"))
    for f in readdir(joinpath(SCR, "src"); join=true)
        (isfile(f) && endswith(f, ".jl")) || continue
        t = read(f, String)
        t2 = replace(t, "module JDyn\n" => "module JDynNoHook\n", "JDyn.qz_criterium" => "JDynNoHook.qz_criterium", "JDyn.jscale" => "JDynNoHook.jscale")
        if basename(f) == "occbin.jl"
            n0 = count("_obc_tick()", t2)
            t2 = replace(t2, r"^[ \t]*_obc_tick\(\)\n"m => "")
            n1 = count("_obc_tick()", t2)
            println("occbin.jl: $n0 occurrences of _obc_tick() before, $n1 after removal")
            n0 == 2 && n1 == 0 || error("expected exactly the two call sites")
        end
        t2 == t || write(f, t2)
    end
    mv(joinpath(SCR, "src", "JDyn.jl"), joinpath(SCR, "src", "JDynNoHook.jl"))
end
build_scratch()
include(joinpath(SCR, "src", "JDynNoHook.jl"))
const NH = JDynNoHook
println("hook sites in the scratch copy's src/occbin.jl: ", count("_obc_tick()", read(joinpath(SCR, "src", "occbin.jl"), String)),
        "; in the repository's: ", count("_obc_tick()", read(joinpath(REPO, "src", "occbin.jl"), String)))

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
const MODELS = [("zlb", joinpath(REPO, "test", "obc", "obc_zlb.mod"), 1.0), ("zlb2", joinpath(REPO, "test", "obc", "obc_zlb2.mod"), 1.0),
                ("nkm", joinpath(DYNARE_SRC, "tests", "occbin", "filter", "NKM_0_std_shocks.mod"), 3.0),
                ("sw", joinpath(REPO, "test", "obc", "obc_sw_zlb.mod"), 0.25)]
println("| model | s | with hook us/period min / median | without hook min / median | median ratio (with / without) | min ratio | identical results |")
println("|---|---|---|---|---|---|---|")
for (id, path, s) in MODELS
    mA = load_dir(JDyn, path); mB = load_dir(NH, path)
    obA, obB = JDyn.Occbin(mA), NH.Occbin(mB)
    sd = [x == 0 ? 0.01 : x for x in sqrt.(diag(mA.Sigma_e))]
    shocks = randn(MersenneTwister(SEED), 1000, length(sd)) .* sd' .* s
    H = 40
    runA() = JDyn.simulate(obA, shocks; periods=1000, check_ahead_periods=H)
    runB() = NH.simulate(obB, shocks; periods=1000, check_ahead_periods=H)
    ra = rb = nothing
    try
        ra = runA(); rb = runB()
    catch e
        println("| ", id, " | ", s, " | failed on these shocks (", first(sprint(showerror, e), 80), ") | | | | |")
        continue
    end
    ta, tb = Float64[], Float64[]
    for r in 1:REPS
        for which in (iseven(r) ? (:A, :B) : (:B, :A))
            GC.gc()
            if which == :A
                push!(ta, @elapsed(ra = runA()))
            else
                push!(tb, @elapsed(rb = runB()))
            end
        end
    end
    same = isequal(ra.piecewise, rb.piecewise) && ra.regime_history == rb.regime_history
    f(v) = @sprintf("%.1f / %.1f", minimum(v) * 1e3, median(v) * 1e3)
    @printf("| %s | %g | %s | %s | %.3f | %.3f | %s |\n", id, s, f(ta), f(tb), median(ta) / median(tb), minimum(ta) / minimum(tb), same)
end
println("(times in microseconds per shock period: total milliseconds for 1000 periods)")
