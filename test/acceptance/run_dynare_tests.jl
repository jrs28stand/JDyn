# Acceptance sweep: run Dynare 7.2's own test suite (the .mod tests of meson.build) through JDyn.dynare.
#
#   nice julia --project=. test/acceptance/run_dynare_tests.jl [options] [pattern ...]
#     --timeout=SECS    per-test limit (default 120); a test over it is killed and recorded TIMEOUT
#     --tests=DIR       Dynare's tests directory (default /root/ref/dynare-7.2/tests)
#     --out=FILE        results csv (default test/acceptance/results.csv)
#     --shard=K/N       run entries K, K+N, ... of the manifest (to run N sweeps in parallel; merge with --merge)
#     --merge           only rebuild results.csv from the partial files, and docs/ACCEPTANCE.md
#     pattern           run only the manifest entries whose id contains the pattern
#
# The master process starts a long-lived worker (one Julia session, no recompilation between files) and kills
# it when a test overruns; the worker is restarted and resumes after the test. As Dynare's test driver
# (scripts/test-driver) does, the files of an entry (test + `extra` files) are copied into a scratch
# directory with the same relative layout and the tests of the entry run there in order, so a test can
# read the output of an earlier one. Categories:
#   OK            ran to the end without warnings (own checks, if any, passed)
#   OK_WARN       ran to the end but JDyn warned that statements were not run (MATLAB lines skipped, commands that do not
#                 fill oo_): its own checks may not have been evaluated
#   CHECK_FAILED  the file's own error()/assert() fired: a discrepancy to Dynare (or a test-environment problem)
#   REFUSED       ModelError/Unsupported/EstimationError naming an unsupported feature
#   ERROR         anything else: a raw Julia exception, an interpreter error, a failure of the preprocessor
#   TIMEOUT       over the time limit
#   XFAIL         the entry is `should_fail` in meson.build and JDyn failed too (as expected)
#   XPASS         `should_fail` entry that ran without error
#   SKIPPED       a .m script (unit test of Dynare's MATLAB library, data script), not a model run
include(joinpath(@__DIR__, "common.jl"))
# ---------------------------------------------------------------------------------------------- master
function master_main(o)
    entries = read_manifest(; shard=o["shard"], pat=o["pat"])
    dir = joinpath(HERE, "partial" * (o["shard"][2] > 1 ? "_$(o["shard"][1] + 1)of$(o["shard"][2])" : ""))
    mkpath(dir)
    resfile = joinpath(dir, "results.tsv")
    cur = joinpath(dir, "current"); fin = joinpath(dir, "finished")
    rm(fin; force=true)
    wargs = vcat(filter(a -> a != "--merge", ARGS[1:end]), ["--worker=$resfile"])
    exe = Base.julia_cmd()
    limit = o["timeout"]
    restarts = 0
    while !isfile(fin) && restarts < 400
        rm(cur; force=true)             # a stale marker would be taken for a test that has run too long
        logf = open(joinpath(dir, "worker.log"), "a")
        proc = run(pipeline(`$exe --project=$ROOT $(joinpath(HERE, "worker.jl")) $wargs`; stdout=logf, stderr=logf); wait=false)
        interrupted = ""; t_int = 0.0; t_start = time()
        while process_running(proc) && !isfile(fin)
            sleep(2)
            if !isfile(cur) && time() - t_start > 900      # stuck before the first test (warm-up)
                kill(proc, Base.SIGKILL); wait(proc); break
            end
            if isfile(cur)
                p = split(readchomp(cur), TAB)
                if length(p) == 2
                    key = p[1]; ts = parse(Float64, p[2])
                    # a result already written for this key means the worker is between tests
                    if time() - ts > limit && !(key in readkeys(resfile))
                        if interrupted != key
                            # ask politely first: SIGINT becomes an InterruptException in the worker, which records TIMEOUT
                            interrupted = key; t_int = time()
                            kill(proc, Base.SIGINT)
                        elseif time() - t_int > 30      # no reaction (a loop without safepoints): kill and restart
                            kill(proc, Base.SIGKILL); wait(proc)
                            id, t = split(key, '|')
                            open(resfile, "a") do io
                                println(io, join([id, t, "TIMEOUT", limit, 0, "", "no result after $(Int(limit)) s (killed)", ""], TAB))
                            end
                            break
                        end
                    end
                end
            end
        end
        if process_running(proc); wait(proc); end
        close(logf)
        isfile(fin) && break
        # the worker died without finishing and without a timeout: the current test crashed it
        if !process_running(proc) && !isfile(fin) && isfile(cur)
            key = split(readchomp(cur), TAB)[1]
            if !(key in readkeys(resfile))
                id, t = split(key, '|')
                open(resfile, "a") do io
                    println(io, join([id, t, "ERROR", 0, 0, "Crash", "the Julia process died (segfault/OOM) in this test", ""], TAB))
                end
            end
        end
        restarts += 1
    end
end
readkeys(f) = isfile(f) ? Set(begin p = split(l, TAB); "$(p[1])|$(p[2])" end for l in eachline(f)) : Set{String}()

# ---------------------------------------------------------------------------------------------- report
include(joinpath(HERE, "report.jl"))

function main()
    o = parse_args(ARGS)
    if o["merge"]
        write_reports(o["out"])
    else
        master_main(o)
        write_reports(o["out"])
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
