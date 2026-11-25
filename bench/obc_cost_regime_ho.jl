# Cost per period of the higher-order solvers, regime formulation vs LCP, at order 2 and with cubature (levels 9 and 11), on NKM (approximate class, one constraint), the
# two-constraint approximate model zlb2m_1e_2 and bf (multi-equation: the LCP refuses it, regime only).
#
#   nice julia --project=. bench/obc_cost_regime_ho.jl [model ...]
#
# One warm-up call (compilation) per cell, then the minimum of REPS timed runs of the whole simulation divided by the number of periods; the machine load (1-minute load
# average at the start of the cell) is recorded next to every timing. Appends a markdown table to bench/results/obc_cost_regime_ho.md.
using Test; include(joinpath(@__DIR__, "..", "test", "common.jl"))
using Random
include(joinpath(@__DIR__, "..", "test", "rho_common.jl"))
using Printf

const OUT = joinpath(@__DIR__, "results", "obc_cost_regime_ho.md")
const REPS = 2
load1() = parse(Float64, split(read("/proc/loadavg", String))[1])
const HZ = Dict("nkm" => 40, "zlb2m_1e_2" => 40, "bf" => 32)
const SC = Dict("nkm" => 1.5, "zlb2m_1e_2" => 1.6, "bf" => 1.0)

function cell(model, form, order, level; T)
    m = rho_model(model)
    sd = sqrt.(diag(Matrix(m.Sigma_e)))
    S = randn(MersenneTwister(4), T, m.n_exo) .* sd' .* SC[model]
    ob = JDyn.Occbin(m)
    hz = HZ[model]
    kw = (order=order, horizon=hz, check_ahead=hz, cubature=level > 0, qmc_level=level)
    form == :lcp && model == "bf" && return nothing
    h = form == :regime ? JDyn.obc_ho_regime(ob; kw...) : JDyn.obc_ho(ob; kw..., m_diagnostic=false)
    sim = form == :regime ? JDyn.obc_hor_simulate : JDyn.obc_ho_simulate
    run1() = with_logger(NullLogger()) do; sim(h, S); end
    run1()
    ld = load1()
    best, r = Inf, nothing
    for _ in 1:REPS
        t = @elapsed r = run1()
        best = min(best, t)
    end
    (ms=1000 * best / T, load=ld, rel=sum(r.relevant), drop=sum(r.dropped), binds=count(w -> maximum(abs, w) > 1e-10, eachrow(r.wedges)), T=T)
end

function main(args)
    models = isempty(args) ? ["nkm", "zlb2m_1e_2", "bf"] : args
    mkpath(dirname(OUT))
    new = !isfile(OUT)
    open(OUT, "a") do io
        new && println(io, "| model | solver | ms per period (min of $REPS runs) | periods | periods with a wedge | relevant points | dropped | load |\n|---|---|---|---|---|---|---|---|")
        for model in models, (order, level, T) in ((2, 0, 150), (2, 9, 40), (2, 11, 40)), form in (:regime, :lcp)
            c = try; cell(model, form, order, level; T=T); catch e; (ms=NaN, load=load1(), rel=0, drop=0, binds=0, T=T); end
            c === nothing && (println(io, "| $model | $(form) $(level == 0 ? "order 2" : "cubature level $level") | refused (not a complementary pair) | $T | | | | |"); continue)
            line = @sprintf("| %s | %s %s | %.2f | %d | %d | %d | %d | %.2f |", model, form, level == 0 ? "order 2" : "cubature level $level", c.ms, c.T, c.binds, c.rel, c.drop, c.load)
            println(line); println(io, line); flush(io)
        end
    end
end
main(ARGS)
