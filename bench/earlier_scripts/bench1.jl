include("/home/claude/work/jdyn-ho/test/common.jl")
using JDyn, Logging, Random, Printf, Statistics
modes = Symbol.(split(get(ENV, "MODES", "scalar,cache"), ","))
levels = parse.(Int, split(get(ENV, "LEVELS", "9,11,13"), ","))
function setup(name)
    if name == "zlb2"
        m = load_copy("/home/claude/work/jdyn-ho/test/obc/obc_zlb2.mod")
        S = reshape(randn(MersenneTwister(1), 100) .* 0.05, :, 1)
    elseif name == "nk"
        m = load_copy("/home/claude/work/jdyn-ho/test/obc/ho/nk_occ.mod")
        S = reshape(vec(matread("/home/claude/work/jdyn-ho/test/obc/ho/shocks.mat")["ShockSequence"]), :, 1)[1:100, :]
    else
        m = load_copy("/home/claude/work/jdyn-ho/test/ref_occbin/occest_nkm.mod")
        sd = sqrt.(diag(Matrix(m.Sigma_e)))
        S = randn(MersenneTwister(3), 100, length(sd)) .* sd' .* parse(Float64, get(ENV, "NKMSCALE", "2.0"))
    end
    JDyn.Occbin(m), S
end
for name in split(get(ENV, "MODELS", "nk,nkm"), ",")
    ob, S = setup(name)
    for L in levels
        res = Dict{Symbol,Any}()
        h = nothing
        for md in modes
            hh = JDyn.obc_ho(ob; order=2, horizon=40, check_ahead=40, cubature=true, qmc_level=L, regions=0)
            hh.cub.mode = md
            JDyn.obc_ho_simulate(hh, S[1:5, :])               # warm up
            hh.cub.mode = md
            t = Inf; r = nothing
            for _ in 1:5
                empty!(hh.cub.cache.d); hh.cub.cache.bytes = 0
                t = min(t, @elapsed(r = JDyn.obc_ho_simulate(hh, S)))
            end
            # steady (cache filled) time
            t2 = minimum(@elapsed(JDyn.obc_ho_simulate(hh, S)) for _ in 1:5)
            res[md] = (t, t2, r)
            h = hh
        end
        r0 = res[modes[1]][3]
        nb = count(>(1e-9), vec(maximum(r0.wedges; dims=2)))
        rel = r0.stats[:relevant_points] / size(S, 1)
        @printf("%s L%d: points %d, dim %d, binding periods %d/100, relevant/period %.0f; ", name, L, size(h.xi, 2), size(h.R, 2), nb, rel)
        for md in modes
            t, t2, r = res[md]
            @printf("%s: %.0f us/period (warm cache %.0f; max diff vs %s %.1e)  ", md, 1e6 * t / size(S, 1), 1e6 * t2 / size(S, 1), modes[1], maximum(abs, r.wedges .- r0.wedges))
        end
        println(" cache sets ", length(h.cub.cache.d)); flush(stdout)
    end
end
