include("/home/claude/work/jdyn-ho/test/common.jl")
using JDyn, Logging, Random, Printf
function sw_cp_model()
    src = read("/home/claude/work/jdyn-ho/test/obc/obc_sw_zlb.mod", String)
    a = first(findfirst("[name = 'Auxiliary notional interest rate',bind='zlb']", src)); b = first(findfirst("    [name='Output gap check']", src))
    d = mktempdir()
    for f in readdir("/home/claude/work/jdyn-ho/test/obc"); isfile(joinpath("/home/claude/work/jdyn-ho/test/obc", f)) && cp(joinpath("/home/claude/work/jdyn-ho/test/obc", f), joinpath(d, f); force=true); end
    write(joinpath(d, "sw_cp.mod"), src[1:a-1] * "[name = 'Auxiliary notional interest rate']\nrlag = rnot;\n" * src[b:end])
    with_logger(NullLogger()) do; JDyn.load(joinpath(d, "sw_cp.mod")); end
end
for (name, mf, hor, sc, tols) in (("borrcon", () -> load_copy("/home/claude/work/jdyn-ho/test/obc/ho/occ_borrcon.mod"), 32, 0.1, (0.0, 0.2, 0.7)),
                                  ("sw_cp", sw_cp_model, 32, 0.25, (0.0, 0.2, 0.7)))
    m = mf(); ob = JDyn.Occbin(m); sd = sqrt.(diag(Matrix(m.Sigma_e)))
    S = randn(MersenneTwister(4), 30, length(sd)) .* sd' .* sc
    for tol in tols
        h = JDyn.obc_ho(ob; order=2, horizon=hor, check_ahead=hor, cubature=true, qmc_level=11, regions=0, m_diagnostic=false, cubature_failure_tolerance=tol)
        nper = 8
        res = ""
        t = Inf; r = nothing; err = nothing
        for rep in 1:5
            t = min(t, @elapsed begin
                try
                    r = with_logger(NullLogger()) do; JDyn.obc_ho_simulate(h, S[1:nper, :]); end
                    err = nothing
                catch e
                    e isa JDyn.ObcError || rethrow(); err = e
                end
            end)
        end
        if err !== nothing
            d = err.detail
            @printf("%s L11 tol %.1f: RAISED in period %d after %.0f ms (n_failed %s%d, relevant %d, active-set failures %s, counts %s)\n", name, tol, d.period, 1000t, d.lower_bound ? ">=" : "", d.n_failed, d.n_relevant, d.n_active_set_failed, d.counts)
        else
            cl = r.stats[:classified]
            @printf("%s L11 tol %.1f: %d periods %.0f ms = %.0f ms/period; dropped %d of %d relevant; classified %d points: ~%.2f ms per classified point (incl. everything else)\n", name, tol, nper, 1000t, 1000t / nper, sum(r.dropped), sum(r.relevant), cl, 1000t / max(cl, 1))
        end
        flush(stdout)
    end
end
