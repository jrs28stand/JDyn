# M_.Correlation_matrix: set by `corr` statements of shocks blocks (not by covariances), reset by shocks(overwrite) -- Dynare 7.2 output (corr8.mod)
using Test
isdefined(@__MODULE__, :REF) || include("common.jl")
@testset "M_.Correlation_matrix follows Dynare" begin
    d = mktempdir()
    write(joinpath(d, "corr8.mod"), """
var y a b;
varexo e u w;
parameters rho;
rho = 0.5;
model;
y = rho*y(-1) + e;
a = 0.3*a(-1) + u;
b = 0.2*b(-1) + w;
end;
shocks;
var e = 1;
var u = 4;
var w = 9;
corr e, u = 0.5;
var u, w = 0.6;
end;
steady;
""")
    m = JDyn.load(joinpath(d, "corr8.mod"))
    @test JDyn.declared_correlation_matrix(m) == [1 0.5 0; 0.5 1 0; 0 0 1]
    @test m.Sigma_e == [1 1 0; 1 4 0.6; 0 0.6 9]
    ws = JDyn.dynare(joinpath(d, "corr8.mod"))
    @test ws.M_.Correlation_matrix == [1 0.5 0; 0.5 1 0; 0 0 1]
end
