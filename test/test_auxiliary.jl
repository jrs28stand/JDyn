# Model-local variables (#) and auxiliary variables for leads/lags beyond one period (first-order parts
# of pydyn tests/test_auxiliary.py; reference reference/clean_aux_a.mat from reference/r_aux.m).
using Test
include("common.jl")

_v(x) = x isa Number ? [Float64(x)] : vec(Float64.(x))
allclose(a, b; rtol=1e-9, atol=0.0) = all(abs.(a .- b) .<= atol .+ rtol .* abs.(b))

const MA = load_copy(joinpath(REF, "aux_a.mod"))
const RA = refmat("clean_aux_a.mat")

@testset "auxiliary variables created as in Dynare" begin
    @test MA.endo_names == split(RA["endo_names"], ",")
    @test JDyn.orig_endo_names(MA) == ["c", "k", "a", "r", "z"]
    kinds = Set(occursin("LAG", n) ? rsplit(n, "_"; limit=3)[1] : rsplit(n, "_"; limit=2)[1] for n in JDyn.aux_names(MA))
    @test kinds == Set(["AUX_ENDO_LEAD", "AUX_EXO_LEAD", "AUX_ENDO_LAG", "AUX_EXO_LAG"])
end

@testset "steady state fills auxiliary variables" begin
    ys, _ = JDyn.steady_state(MA)
    @test allclose(ys, _v(RA["ys"]); rtol=1e-10, atol=1e-12)
    println("aux_a: max |d ys| = $(maximum(abs, ys .- _v(RA["ys"])))")
end

@testset "first-order rules and IRFs" begin
    sol = JDyn.first_order(MA)
    @test sol.order_var == vec(Int.(RA["order_var"]))
    @test allclose(JDyn.ghx(sol), Float64.(RA["ghx"]); rtol=1e-9, atol=1e-11)
    @test allclose(JDyn.ghu(sol), Float64.(RA["ghu"]); rtol=1e-9, atol=1e-11)
    chol = Matrix(cholesky(Symmetric(MA.Sigma_e)).L)
    worst = 0.0
    for i in 1:MA.n_exo                       # first-order IRFs from Dynare's irf.m
        path = JDyn.irf(sol, chol[:, i]; periods=12)
        @test allclose(path, RA["irf1"][:, :, i]; rtol=1e-8, atol=1e-12)
        worst = max(worst, maximum(abs, path .- RA["irf1"][:, :, i]))
    end
    # first-order simulation with Dynare's shocks (levels; row 1 of Dynare's output is y0)
    y1 = JDyn.simulate(JDyn.as_dr(sol), Float64.(RA["ex"]); order=1)
    @test allclose(y1, RA["y1"][2:end, :]; rtol=1e-9, atol=1e-11)
    println("aux_a: max |d ghx| = $(maximum(abs, JDyn.ghx(sol) .- RA["ghx"])), |d ghu| = $(maximum(abs, JDyn.ghu(sol) .- RA["ghu"])), |d irf| = $worst, |d sim| = $(maximum(abs, y1 .- RA["y1"][2:end, :]))")
end

@testset "local variables equal hand expansion" begin
    b = load_copy(joinpath(REF, "aux_b.mod"))
    @test b.endo_names[1:5] == MA.endo_names[1:5]
    sa, sb = JDyn.first_order(MA), JDyn.first_order(b)
    @test allclose(sa.ys, sb.ys; rtol=1e-13)
    @test allclose(sa.G, sb.G; rtol=1e-10, atol=1e-12)
    @test allclose(sa.H, sb.H; rtol=1e-10, atol=1e-12)
end

@testset "steady-state function need not set auxiliary variables" begin
    function fn(model, params, exo)
        p = Dict(zip(model.param_names, params))
        rr = 1 / p["beta"] - 1 + p["delta"]
        k = (rr / p["alpha"])^(1 / (p["alpha"] - 1))
        c = k^p["alpha"] - p["delta"] * k
        [c, k, 0.0, rr, 0.1 * k]                  # the five declared variables only
    end
    b = load_copy(joinpath(REF, "aux_b.mod"); steady_state_fn=fn)
    ys, _ = JDyn.steady_state(b)
    @test allclose(ys, _v(RA["ys"]); rtol=1e-10, atol=1e-12)
end

@testset "numerical steady state with auxiliary variables" begin
    saved = MA.steady_state_model
    MA.steady_state_model = nothing
    try
        r = _v(RA["ys"])
        guess = [i <= 5 ? r[i] * 1.05 : NaN for i in 1:MA.n_endo]
        guess = JDyn.fill_auxiliary(MA, guess, JDyn.exo_steady(MA), MA.params)
        ys, _ = JDyn.steady_state(MA; guess=guess)
        @test allclose(ys, r; rtol=1e-8, atol=1e-10)
    finally
        MA.steady_state_model = saved
    end
end

@testset "perfect foresight example" begin
    @test_skip "perfect foresight is outside the first-order stream"
end
