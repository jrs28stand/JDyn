# C.2: does the difference between the verification tolerances of JDyn (absolute 1e-10 on max(-y, -g, |y g|), `obc_verify`) and pydyn
# (`verify_lcp`: tol 1e-8 times max(1, max|q|), on min(|y|, |g|)) change an accept/reject decision on the R1 states?
#   nice julia --project=. bench/obc_verify_tol.jl [models] [nstates]
# For every R1 state (same states and draws as bench/obc_bench.jl: seed, 2000 states, scales 0.5, 1, 1.5, 2) the complementarity problem (M, q) is built
# (zlb, zlb2, nkm: `obc_system` + `obc_q`; sw: the tied LCP of `obc_lcp`, G = nothing states only) and the shared fast path `solve_active_set` run under
# four acceptance rules (the rule is the `tol` it gets, so it moves both the start of the guess and the final verification):
#   abs1e-10 (JDyn today), rel1e-10 (1e-10 * max(1, max|q|)), abs1e-8 (what `_solve_lcp` passes for sw), rel1e-8 (pydyn's).
# Decisions (accepted or not, and accepted point) are compared; then the accepted point of the loosest rule is judged by both measures.
using Random, LinearAlgebra, Printf, Logging, Statistics
using JDyn
LinearAlgebra.BLAS.set_num_threads(1)
const REPO = normpath(joinpath(@__DIR__, ".."))
const SEED = 20261124
const DYNARE_SRC = get(ENV, "JDYN_DYNARE_SRC", "/home/claude/ref/dynare-7.2")
const SPECS = Dict("zlb" => joinpath(REPO, "test", "obc", "obc_zlb.mod"), "zlb2" => joinpath(REPO, "test", "obc", "obc_zlb2.mod"),
                   "nkm" => joinpath(DYNARE_SRC, "tests", "occbin", "filter", "NKM_0_std_shocks.mod"), "sw" => joinpath(REPO, "test", "obc", "obc_sw_zlb.mod"))
function load_dir(path)
    d = mktempdir()
    for f in readdir(dirname(path))
        p = joinpath(dirname(path), f)
        isfile(p) && cp(p, joinpath(d, f))
    end
    with_logger(NullLogger()) do
        JDyn.load(joinpath(d, basename(path)))
    end
end
jmeasure(M, q, y) = (g = q .+ M * y; max(maximum(-y), maximum(-g), maximum(abs.(y .* g))))            # obc_verify's violation
pmeasure(M, q, y, n_s) = (g = q .+ M * y; max(maximum(-y), maximum(-g), maximum(min.(abs.(y), abs.(g[1:n_s])))))   # verify_lcp's, before scaling by tol * scale

function run_model(id, nstates)
    m = load_dir(SPECS[id]); ob = JDyn.Occbin(m)
    sd = [x == 0 ? 0.01 : x for x in sqrt.(diag(m.Sigma_e))]
    n, nx = ob.n, ob.n_exo
    rng = MersenneTwister(SEED + 2)
    x = zeros(n); states = Vector{Float64}[]
    for t in 1:10200
        x = ob.P * x + ob.Q * (randn(rng, nx) .* sd)
        t > 200 && (t - 200) % 5 == 0 && push!(states, copy(x))
    end
    states = states[1:nstates]
    draws = randn(MersenneTwister(SEED + 3), length(states), nx) .* sd'
    H = 40
    sys = id == "sw" ? nothing : JDyn.obc_system(ob; horizon=H)
    rg = id == "sw" ? JDyn.regime_obc(ob, H, 2H) : nothing
    rules = [("abs1e-10", (q -> 1e-10)), ("rel1e-10", (q -> 1e-10 * max(1.0, maximum(abs, q)))), ("abs1e-8", (q -> 1e-8)), ("rel1e-8", (q -> 1e-8 * max(1.0, maximum(abs, q))))]
    println("### ", id)
    println()
    println("| s | problems (M, q) used | binding (q < 0 somewhere) | max |q| over problems: max / share with max|q| > 1 | decisions differ: ", join([r[1] for r in rules[2:end]], " / "), " vs abs1e-10 | points differ >1e-9 | accepted by rel1e-8 point: JDyn measure > 1e-10 | pydyn measure > 1e-8 scale | rejected by abs1e-10 but accepted by rel1e-8 |")
    println("|---|---|---|---|---|---|---|---|---|")
    for s in (0.5, 1.0, 1.5, 2.0)
        used = binding = 0; qmax = 0.0; big = 0
        dec = zeros(Int, 3); pdiff = 0; jbad = 0; pbad = 0; onlyrel = 0
        for i in eachindex(states)
            u = draws[i, :] .* s
            if id == "sw"
                got = JDyn.obc_lcp(rg, states[i], u)
                got === nothing && continue
                M, q, G, _ = got
                G === nothing || continue
                T = rg.T
                M1, q1 = M[1:T, :], q[1:T]
                qfull = q
            else
                M1 = sys.M; xin = vcat(states[i], u); q1 = sys.q0 .+ sys.Qx * xin; qfull = q1; T = length(q1)
            end
            used += 1
            minimum(q1) < 0 && (binding += 1)
            mq = maximum(abs, qfull); qmax = max(qmax, mq); mq > 1 && (big += 1)
            res = [JDyn.solve_active_set(M1, q1; tol=r[2](qfull)) for r in rules]
            for k in 2:4
                dec[k-1] += (res[k][2] == :ok) != (res[1][2] == :ok)
            end
            if res[1][2] == :ok && res[4][2] == :ok && maximum(abs, res[1][1] .- res[4][1]) > 1e-9
                pdiff += 1
            end
            if res[4][2] == :ok
                y = res[4][1]
                sc = max(1.0, maximum(abs, qfull))
                jm = jmeasure(M1, q1, y); pm = pmeasure(M1, q1, y, T)
                jm > 1e-10 && (jbad += 1)
                pm > 1e-8 * sc && (pbad += 1)
                res[1][2] != :ok && (onlyrel += 1)
            end
        end
        @printf("| %g | %d | %d | %.3g / %.1f%% | %s | %d | %d | %d | %d |\n", s, used, binding, qmax, 100big / max(used, 1), join(dec, " / "), pdiff, jbad, pbad, onlyrel)
    end
    println()
end
ms = length(ARGS) >= 1 ? split(ARGS[1], ",") : ["zlb", "zlb2", "nkm", "sw"]
ns = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2000
for id in ms
    run_model(id, ns)
end
