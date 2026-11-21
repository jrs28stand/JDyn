# One likelihood evaluation (steady state, order-2 solution, filter) at the sizes of benchmark B23.
#   DYNARE_PREPROCESSOR=... julia --project=. test/ref_nonlinear/bench_b23.jl
using JDyn, MAT, Logging
const J = JDyn
d = mktempdir(); R = @__DIR__
for f in ("nl_rbc.mod", "nl_rbc_data.m"); cp(joinpath(R, f), joinpath(d, f)); end
m = with_logger(NullLogger()) do; J.Model(J.run_preprocessor(joinpath(d, "nl_rbc.mod"))); end
x0 = vec(matread(joinpath(R, "nl_rbc_seeds.mat"))["x0"])
for kind in (:octave, :julia), (alg, N) in (("sis", 20000), ("apf", 20000), ("gf", 1), ("nlkf", 1), ("cpf", 1000))
    est = Estimation(m; filter_algorithm=alg, number_of_particles=N, particle_rng=kind)
    J._nonlinear_dsge_likelihood(est, x0)
    t = minimum(@elapsed(J._nonlinear_dsge_likelihood(est, x0)) for _ in 1:3)
    println(rpad("$alg N=$N $kind", 24), round(t, digits=3), " s")
end
