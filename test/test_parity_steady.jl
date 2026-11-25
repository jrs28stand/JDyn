# solve_algo 9, 13, 14 (steady.jl / steady_algos.jl): Dynare 7.2 (Octave 8.4) iterates, test/ref_parity/gen_solve_algo.sh
include(joinpath(@__DIR__, "common.jl"))

const _SA_REF = Dict{Tuple{String,Int},Vector{Float64}}()
for l in eachline(joinpath(@__DIR__, "ref_parity", "solve_algo_ref.txt"))
    f = split(l)
    _SA_REF[(f[1], parse(Int, f[2]))] = parse.(Float64, f[3:end])
end

_sa_model(name, algo) = name == "blocks" ? """
var a b c d f; varexo e;
model;
log(a) = 0.5 + 0.3*log(a(-1));
b*exp(b(-1)) = a(-1);
c = 0.3*d(-1)^0.5 + b(-1);
d = c(-1)^2/(1+c(-1)) + 0.1;
f = a*c + d + e;
end;
initval; a = 3; b = 0.5; c = 1; d = 1; f = 1; end;
steady(solve_algo=$algo);
""" : """
var y c k; varexo e; parameters alpha beta delta A;
model; 1/c = beta/c(+1)*(alpha*A*k^(alpha-1)+1-delta); y = A*exp(e)*k(-1)^alpha; k = y - c + (1-delta)*k(-1); end;
alpha = 0.33; beta = 0.99; delta = 0.025; A = 1;
initval; $(name == "rbc" ? "k = 20; c = 1.5; y = 2.5;" : "k = 300; c = 0.2; y = 0.3;") end;
steady(solve_algo=$algo);
"""

@testset "solve_algo 9, 13, 14 reproduce Dynare's iterates" begin
    for name in ("rbc", "blocks"), algo in (4, 9, 13, 14)
        m = build(_sa_model(name, algo))
        ys, _ = JDyn.steady_state(m)
        ref = _SA_REF[(name, algo)]
        algo == 4 || @test maximum(abs.(ys .- ref) ./ max.(abs.(ref), 1)) < 1e-13
        algo == 4 && @test maximum(abs.(ys .- ref)) < 1e-5            # default solver: polished to machine precision
    end
    # 9 and 14 are the whole-system trust region, 4 and 13 the block-wise one: Dynare's iterates differ, and so do JDyn's
    @test maximum(abs.(_SA_REF[("blocks", 9)] .- _SA_REF[("blocks", 13)])) > 1e-6
    # a start far from the root: Dynare's 13 reaches the root, 4, 9 and 14 report "Impossible to find the steady state"
    ys, _ = JDyn.steady_state(build(_sa_model("rbcfar", 13)))
    @test maximum(abs.(ys .- _SA_REF[("rbcfar", 13)])) < 1e-12
    for algo in (9, 14)
        @test_throws JDyn.SteadyStateError JDyn.steady_state(build(_sa_model("rbcfar", algo)))
    end
end

@testset "solve_algo options: maxit, tolx, workspace" begin
    # maxit=2 is too few from this guess: an error, as Dynare's solvers
    for algo in (9, 13, 14)
        m = build(_sa_model("rbc", algo)[1:end-length("steady(solve_algo=$algo);\n")] * "steady(solve_algo=$algo, maxit=2);")
        @test_throws JDyn.SteadyStateError JDyn.steady_state(m)
    end
    # the dynare() workspace route reads the same option
    d = mktempdir()
    write(joinpath(d, "w.mod"), _sa_model("blocks", 13) * "\n")
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "w.mod"))
    end
    @test maximum(abs.(vec(ws.oo_.steady_state) .- _SA_REF[("blocks", 13)])) < 1e-12
end
