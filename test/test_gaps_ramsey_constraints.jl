# ramsey_constraints (i > 0) with solve_algo=10 (lmmcp) against Octave 8.4 / Dynare 7.2: tests/optimal_policy/RamseyConstraints/test1.mod
using Test, MAT
import JDyn
@testset "ramsey_constraints" begin
    d = mktempdir(); dir = joinpath(@__DIR__, "ref_gaps", "ramsey_constraints")
    cp(joinpath(dir, "test1.mod"), joinpath(d, "test1.mod"))
    ws = cd(() -> JDyn.dynare(joinpath(d, "test1.mod")), d)
    ref = matread(joinpath(dir, "ref.mat"))["endo_simul"]
    @test maximum(abs.(ws.oo_.endo_simul .- ref)) < 1e-8          # measured 3.6e-10
    @test minimum(ws.oo_.endo_simul[1, :]) >= -1e-9                # the constraint binds in the first periods
end
