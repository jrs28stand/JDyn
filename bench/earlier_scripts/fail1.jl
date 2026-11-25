include("/home/claude/work/jdyn-ho/test/common.jl")
using JDyn, Logging, Random, Printf
function sw_cp_model()
    src = read("/home/claude/work/jdyn-ho/test/obc/obc_sw_zlb.mod", String)
    a = first(findfirst("[name = 'Auxiliary notional interest rate',bind='zlb']", src))
    b = first(findfirst("    [name='Output gap check']", src))
    d = mktempdir()
    for f in readdir("/home/claude/work/jdyn-ho/test/obc"); isfile(joinpath("/home/claude/work/jdyn-ho/test/obc", f)) && cp(joinpath("/home/claude/work/jdyn-ho/test/obc", f), joinpath(d, f); force=true); end
    write(joinpath(d, "sw_cp.mod"), src[1:a-1] * "[name = 'Auxiliary notional interest rate']\nrlag = rnot;\n" * src[b:end])
    with_logger(NullLogger()) do; JDyn.load(joinpath(d, "sw_cp.mod")); end
end
mb = load_copy("/home/claude/work/jdyn-ho/test/obc/ho/occ_borrcon.mod")
ob = JDyn.Occbin(mb)
sd = sqrt.(diag(Matrix(mb.Sigma_e)))
S = randn(MersenneTwister(4), 30, length(sd)) .* sd' .* 0.1
h = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=5, m_diagnostic=false)
println("borrcon M diag: ", JDyn.cub_m_diagnostic(h.sys.M, 1))
try; JDyn.obc_ho_simulate(h, S); println("no failure"); catch e; println(e isa JDyn.ObcError ? (e.status, e.period, e.msg, e.detail) : e); end
hl = JDyn.obc_ho(ob; order=2, horizon=96, check_ahead=96, cubature=true, qmc_level=5, m_diagnostic=false)
try; JDyn.obc_ho_simulate(hl, S[1:10,:]); println("h96 ok"); catch e; println(e isa JDyn.ObcError ? (e.status, e.msg) : e); end
ht = JDyn.obc_ho(ob; order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=5, m_diagnostic=false, cubature_failure_tolerance=0.9)
r = JDyn.obc_ho_simulate(ht, S[1:12,:]); println("tol: dropped ", sum(r.dropped), " relevant ", sum(r.relevant), " stats ", r.stats)
sw = sw_cp_model(); sdw = sqrt.(diag(Matrix(sw.Sigma_e)))
Sw = randn(MersenneTwister(4), 30, length(sdw)) .* sdw' .* 0.25
h2 = JDyn.obc_ho(JDyn.Occbin(sw); order=2, horizon=32, check_ahead=32, cubature=true, qmc_level=5)
try; JDyn.obc_ho_simulate(h2, Sw); println("no failure"); catch e; println(e isa JDyn.ObcError ? (e.status, e.period, e.msg) : e); e isa JDyn.ObcError || Base.show_backtrace(stdout, catch_backtrace()); end
