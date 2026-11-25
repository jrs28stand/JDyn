# Cross-port check of the higher-order regime formulation (src/obc_regime_ho.jl) and of the split cubature failure rule against pydyn, run DIRECTLY (not a stored file):
#
#   DYNARE_PREPROCESSOR=... PYDYN_EXAMPLES=... PYTHONPATH=/home/claude/pydyn nice julia --project=. bench/obc_cross_regime_ho.jl [case ...]
#
# For each case JDyn writes the shocks (and, with cubature, its root `R` and points `xi`) to a .mat file, bench/obc_cross_regime_ho_pydyn.py runs pydyn's
# `HigherOrderRegime` / `HigherOrderNews` (lcp) on the same model file with the same inputs, and the two answers are compared: paths, wedges, relevant points and the dropped
# points by class. With cubature the same root and the same points are used by both (injected), so what is compared is the solver, not the quasi-Monte-Carlo set.
# Results are appended to bench/results/obc_cross_regime_ho.md with the machine load.
using Test; include(joinpath(@__DIR__, "..", "test", "common.jl"))
using Random
include(joinpath(@__DIR__, "..", "test", "rho_common.jl"))
using MAT, Printf, Dates

const PY = joinpath(@__DIR__, "obc_cross_regime_ho_pydyn.py")
const OUT = joinpath(@__DIR__, "results", "obc_cross_regime_ho.md")
load1() = parse(Float64, split(read("/proc/loadavg", String))[1])

# name, model, formulation, order, cubature, level, regions, periods, scale, seed
const CASES = Dict(
    "nkm_o2"        => (model="nkm",        form=:regime, order=2, cub=false, level=0,  regions="all", T=150, sc=1.5, seed=4),
    "nkm_o2_c9"     => (model="nkm",        form=:regime, order=2, cub=true,  level=9,  regions="all", T=60,  sc=1.5, seed=4),
    "zlb2m_o2"      => (model="zlb2m_1e_2", form=:regime, order=2, cub=false, level=0,  regions="all", T=150, sc=1.6, seed=4),
    "zlb2m_o2_c9"   => (model="zlb2m_1e_2", form=:regime, order=2, cub=true,  level=9,  regions="all", T=60,  sc=1.6, seed=4),
    "nk_o2_c9"      => (model="nk",         form=:regime, order=2, cub=true,  level=9,  regions="all", T=60,  sc=1.0, seed=4),
    "nkm_lcp_c9"    => (model="nkm",        form=:lcp,    order=2, cub=true,  level=9,  regions="all", T=60,  sc=1.5, seed=4),
    "zlb2_lcp_c7"   => (model="zlb2",       form=:lcp,    order=2, cub=true,  level=7,  regions="all", T=40,  sc=0.06, seed=9))

function shocks_of(c)
    m = rho_model(c.model)
    sd = c.model in ("zlb", "zlb2") ? ones(m.n_exo) : sqrt.(diag(Matrix(m.Sigma_e)))
    randn(MersenneTwister(c.seed), c.T, m.n_exo) .* sd' .* c.sc
end

function run_case(name)
    c = CASES[name]
    m = rho_model(c.model)
    hz = c.model == "nkm" ? 40 : (c.model == "nk" ? 32 : (startswith(c.model, "zlb2m") ? 40 : 20))
    S = shocks_of(c)
    ob = JDyn.Occbin(m)
    kw = (order=c.order, horizon=hz, check_ahead=hz, cubature=c.cub, qmc_level=c.level, regions=c.regions)
    h = c.form == :regime ? JDyn.obc_ho_regime(ob; kw...) : JDyn.obc_ho(ob; kw..., m_diagnostic=false)
    sim = c.form == :regime ? JDyn.obc_hor_simulate : JDyn.obc_ho_simulate
    with_logger(NullLogger()) do; sim(h, S[1:2, :]); end                                   # compile
    load_j = load1()
    tj = @elapsed r = with_logger(NullLogger()) do; sim(h, S); end
    d = mktempdir()
    inp, out = joinpath(d, "in.mat"), joinpath(d, "out.mat")
    cfg = Dict{String,Any}("shocks" => S, "order" => c.order, "horizon" => hz, "check_ahead" => hz, "cubature" => c.cub, "qmc_level" => c.level, "regions" => string(c.regions),
                           "formulation" => string(c.form), "failure_tolerance" => 0.0, "undecided_tolerance" => 0.005, "milp_time_limit" => 0.1)
    c.cub && (cfg["R"] = h.R; cfg["xi"] = h.xi)
    matwrite(inp, cfg)
    env = Dict("PYTHONPATH" => get(ENV, "PYTHONPATH", "/home/claude/pydyn"), "DYNARE_PREPROCESSOR" => get(ENV, "DYNARE_PREPROCESSOR", "/home/claude/ref/pp-standalone/src/dynare-preprocessor"),
               "PYDYN_EXAMPLES" => get(ENV, "PYDYN_EXAMPLES", "/home/claude/ref/dynare-7.2/examples"))
    load_p = load1()
    run(setenv(`nice python3 $PY $(RHO_SRC[c.model]) $inp $out`, merge(Dict(ENV), env)))
    p = matread(out)
    dw = maximum(abs, r.with_bound .- p["with_bound"])
    dwe = maximum(abs, r.wedges .- p["wedges"])
    per = vec(maximum(abs.(r.with_bound .- p["with_bound"]); dims=2))
    rel_eq = vec(r.relevant) == vec(Int.(p["relevant"]))
    cls_eq = all(vec(getfield(r, k)) == vec(Int.(p[String(k)])) for k in (:dropped_a, :dropped_b, :dropped_c))
    row = (name=name, binds=count(w -> maximum(abs, w) > 1e-10, eachrow(r.wedges)), rel=sum(r.relevant), dropped=Tuple(sum(getfield(r, k)) for k in (:dropped_a, :dropped_b, :dropped_c)),
           pdropped=Tuple(sum(Int.(p[String(k)])) for k in (:dropped_a, :dropped_b, :dropped_c)), dw=dw, dwe=dwe, nbad=count(>(1e-8), per), rel_eq=rel_eq, cls_eq=cls_eq,
           tj=tj, tp=p["seconds"], T=c.T, load_j=load_j, load_p=load_p)
    row
end

function main(args)
    names = isempty(args) ? sort(collect(keys(CASES))) : args
    mkpath(dirname(OUT))
    new = !isfile(OUT)
    open(OUT, "a") do io
        new && println(io, "| case | periods with a wedge | relevant pts | dropped a/b/c JDyn | dropped a/b/c pydyn | max abs path diff | max abs wedge diff | periods with diff > 1e-8 | relevant equal | classes equal | JDyn s | pydyn s | load (1 min) JDyn / pydyn |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for nm in names
            r = run_case(nm)
            line = @sprintf("| %s | %d/%d | %d | %s | %s | %.2e | %.2e | %d | %s | %s | %.1f | %.1f | %.2f / %.2f |", r.name, r.binds, r.T, r.rel, join(r.dropped, "/"), join(r.pdropped, "/"),
                            r.dw, r.dwe, r.nbad, r.rel_eq, r.cls_eq, r.tj, r.tp, r.load_j, r.load_p)
            println(line); println(io, line); flush(io)
        end
    end
end
main(ARGS)
