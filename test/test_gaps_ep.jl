# Extended path (order 0 with the stacked trust region; stochastic full tree with Gaussian quadrature and unscented nodes)
# against Octave 8.4 / Dynare 7.2 (ep/rbc2, ep/ar_backward; rbc2 with order 2 and 3 integration rules).
using Test, MAT
import JDyn
@testset "extended path" begin
    dir = joinpath(@__DIR__, "ref_gaps", "ep"); ref = matread(joinpath(dir, "ref.mat"))
    function run(txt, name="m")
        d = mktempdir(); write(joinpath(d, name * ".mod"), txt)
        cd(() -> JDyn.dynare(joinpath(d, name * ".mod"); random=:octave), d)
    end
    ws = run(read(joinpath(dir, "rbc2.mod"), String), "rbc2")
    @test ws.oo_.extended_path.status == 1
    @test maximum(abs.(ws.oo_.endo_simul .- ref["rbc2"])) < 1e-9         # order 0 then order 1; measured 3e-16
    ws = run(read(joinpath(dir, "ar_backward.mod"), String), "ar")
    @test maximum(abs.(ws.oo_.endo_simul .- ref["ar_backward"])) < 1e-12
    base = read(joinpath(dir, "rbc2o.mod"), String); base = base[1:first(findfirst("extended_path(", base))-1]
    for (k, call) in (("e_u2", "extended_path(periods=10, order=2, integration=unscented);"),
                      ("e_t2", "extended_path(periods=10, order=2, number_of_quadrature_nodes=3, integration=tensor_gaussian_quadrature);"),
                      ("e_t3", "extended_path(periods=12, order=3, number_of_quadrature_nodes=3, integration=tensor_gaussian_quadrature);"))
        ws = run(base * call * "\n", "rbc2o")
        @test maximum(abs.(ws.oo_.endo_simul .- ref[k])) < 1e-9            # measured 1e-15
    end
    # refused rather than guessed
    @test_throws JDyn.Unsupported run(replace(base, "steady;" => "steady;") * "extended_path(periods=5, order=2, integration=stroud_cubature);\n", "rbc2o")
end
