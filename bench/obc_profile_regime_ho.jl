# Where the time of one cubature period of the regime formulation goes (src/obc_regime_ho.jl): the perturbation of the condition paths (matrix product), the relevance test
# (OccBin's bind condition on every point, N periods), the batched guess-and-verify and the complete searches of the points it does not settle.
#
#   nice julia --project=. bench/obc_profile_regime_ho.jl [model] [level]
using Test; include(joinpath(@__DIR__, "..", "test", "common.jl"))
using Random, Printf
include(joinpath(@__DIR__, "..", "test", "rho_common.jl"))
model = length(ARGS) >= 1 ? ARGS[1] : "nkm"
level = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 9
m = rho_model(model)
sd = sqrt.(diag(Matrix(m.Sigma_e)))
S = randn(MersenneTwister(4), 40, m.n_exo) .* sd' .* (model == "bf" ? 1.0 : model == "nkm" ? 1.5 : 1.6)
hz = model == "bf" ? 32 : 40
h = JDyn.obc_ho_regime(JDyn.Occbin(m); order=2, horizon=hz, check_ahead=hz, cubature=true, qmc_level=level, cubature_undecided_tolerance=0.02)
h.trace = Any[]
with_logger(NullLogger()) do; JDyn.obc_hor_simulate(h, S); end          # compile, and the inputs of every period
function profile(h, model, level)
    tr = h.trace
    nb, N, T, nk = h.nb, h.N, h.T, h.nk
    Ro = JDyn._hor_other_roots(h)
    tp = trel = tgv = 0.0
    nrel = nbad = 0
    for t in tr
        a, Y, A = JDyn.hor_inputs_from_state(h, t.y1, t.y2, t.o_pre, t.prev, t.u)
        t0 = time_ns()
        pY = Ro[nb*N+1:end, :] * h.xi; pY .+= vec(Y); pA = Ro[1:nb*N, :] * h.xi; pA .+= vec(A)
        t1 = time_ns()
        rel, vbs = JDyn._hor_relevant(h, pY)
        t2 = time_ns()
        tp += (t1 - t0) / 1e9; trel += (t2 - t1) / 1e9
        isempty(rel) && continue
        pa = reduce(hcat, [vcat([pA[(b-1)*N+1:(b-1)*N+T, p] for b in 1:nb]...) for p in rel])
        t3 = time_ns()
        W, ok = JDyn._hor_gv_many(h, pa, pY[:, rel], vbs)
        tgv += (time_ns() - t3) / 1e9
        nrel += length(rel); nbad += count(!, ok)
    end
    np = length(tr)
    @printf("%s level %d (%d points), %d periods; per period: perturbation products %.2f ms, relevance test %.2f ms, batched guess-and-verify %.2f ms; relevant points %.0f, not settled by the batch %.1f\n",
            model, level, size(h.xi, 2), np, 1000tp / np, 1000trel / np, 1000tgv / np, nrel / np, nbad / np)
end
profile(h, model, level)
