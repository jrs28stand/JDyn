# The loglinear option of stoch_simul at order 1, against Dynare 7.2 (first-order parts of pydyn
# tests/test_loglinear.py; reference reference/clean_loglinear.mat from reference/r_loglinear.m).
# The model has an endogenous auxiliary variable (a lag of two on capital: logged by Dynare) and
# auxiliary variables for leads and lags of exogenous variables (kept in levels by Dynare).
using Test
include("common.jl")

_v(x) = x isa Number ? [Float64(x)] : vec(Float64.(x))
allclose(a, b; rtol=1e-9, atol=0.0) = all(abs.(a .- b) .<= atol .+ rtol .* abs.(b))

function load_ll(; order=1, edit=identity)
    d = mktempdir()
    write(joinpath(d, "ll_rbc.mod"), edit(read(joinpath(REF, "ll_rbc.mod"), String)))
    with_logger(NullLogger()) do
        JDyn.Model(JDyn.run_preprocessor(joinpath(d, "ll_rbc.mod"); extra_args=["-DORDER=$order"]))
    end
end

const M = load_ll()
const SOL = JDyn.first_order(M)
const R = refmat("clean_loglinear.mat")

@testset "log mask follows Dynare" begin
    @test M.endo_names == split(R["endo_names"], ",")
    types = _v(R["aux_types"])
    expected = vcat(trues(length(JDyn.orig_endo_names(M))), [!(t in (2, 3)) for t in types])
    @test JDyn.log_mask(M) == expected
end

@testset "rules and steady state in logs" begin
    @test SOL.loglinear
    @test allclose(SOL.ys, _v(R["ys"]); rtol=1e-12, atol=1e-14)
    @test SOL.order_var == vec(Int.(R["order_var"]))
    @test allclose(JDyn.ghx(SOL), Float64.(R["ghx"]); rtol=1e-9, atol=1e-12)
    @test allclose(JDyn.ghu(SOL), Float64.(R["ghu"]); rtol=1e-9, atol=1e-12)
    @test all(SOL.steady_state[1:5] .> 0) && !isapprox(SOL.steady_state, SOL.ys)
    println("loglinear: max |d ghx| = $(maximum(abs, JDyn.ghx(SOL) .- R["ghx"])), |d ghu| = $(maximum(abs, JDyn.ghu(SOL) .- R["ghu"]))")
end

@testset "IRFs and simulation" begin
    chol = Matrix(cholesky(Symmetric(M.Sigma_e)).L)
    irfs = R["irfs"]
    for (k, label) in enumerate(split(R["irf_names"], ","))
        shock = M.exo_names[findfirst(x -> endswith(label, "_" * x), M.exo_names)]
        series = vec(irfs[k, :])
        path = JDyn.irf(SOL, chol[:, M.ix[shock]]; periods=length(series))
        @test allclose(path[:, M.ie[label[1:end-length(shock)-1]]], series; rtol=1e-9, atol=1e-13)
    end
    sim = JDyn.simulate(JDyn.as_dr(SOL), Float64.(R["ex"]))          # levels (here logs), as Dynare's simult_
    @test allclose(sim, R["sim"][2:end, :]; rtol=1e-10, atol=1e-12)
    println("loglinear: max |d sim| = $(maximum(abs, sim .- R["sim"][2:end, :]))")
end

@testset "moments" begin
    mo = JDyn.stoch_simul_moments(M; var_list=JDyn.orig_endo_names(M), dr=JDyn.as_dr(SOL), order=1, ar=3,
                                  conditional_variance_decomposition=[1, 4])
    @test allclose(mo.mean, _v(R["mean"]); rtol=1e-12, atol=1e-14)
    @test allclose(mo.var, Float64.(R["var"]); rtol=1e-9, atol=1e-15)
    @test allclose(hcat(mo.autocorr...), Float64.(R["autocorr"]); rtol=1e-9, atol=1e-12)
    @test allclose(mo.variance_decomposition, Float64.(R["vd"]); rtol=1e-9, atol=1e-10)
    @test allclose(mo.conditional_variance_decomposition, Float64.(R["cvd"]); rtol=1e-9, atol=1e-10)
end

@testset "levels on request" begin
    lev = JDyn.first_order(M; loglinear=false)
    @test !lev.loglinear
    @test isapprox(lev.ys, SOL.steady_state)
    # k_order (orders 2-3) is outside this stream
end

@testset "non-positive steady state is refused" begin
    function add_negative_variable(t)
        t = replace(t, "var c k y z inv;" => "var c k y z inv nw;")
        t = replace(t, "z = 1 - rho" => "nw = -c;\nz = 1 - rho")
        replace(t, "c = y - inv;\nend;" => "c = y - inv;\nnw = -c;\nend;")
    end
    m = load_ll(; edit=add_negative_variable)
    err = try
        JDyn.first_order(m); nothing
    catch e
        e
    end
    @test err isa JDyn.LoglinearError && occursin("not strictly positive: nw", err.msg)
    @test JDyn.first_order(m; loglinear=false) isa JDyn.FirstOrderSolution
end
