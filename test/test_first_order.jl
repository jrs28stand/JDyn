# First-order solutions against Dynare 7.2 (port of pydyn tests/test_first_order.py).
using Test
include("common.jl")

const FO_CASES = Dict(
    "collard_2001_simulated_moments" => SS,
    "rbc_occbin" => joinpath(EXAMPLES, "occbin"),
    # the estimation example with the estimation commands replaced by stoch_simul(order=1)
    "schorfheide_2000_sim" => REF,
)

@testset "first order: $name" for name in sort(collect(keys(FO_CASES)))
    m = load_copy(joinpath(FO_CASES[name], name * ".mod"))
    r = refmat("clean_$name.mat")
    rys, rparams = vec(Float64.(r["ys"])), vec(Float64.(r["params"]))

    @testset "symbols and calibration" begin
        @test m.endo_names == split(r["endo_names"], ",")
        @test m.exo_names == split(r["exo_names"], ",")
        @test isapprox(m.params, rparams; rtol=1e-13)
        rS = r["Sigma_e"] isa Number ? fill(Float64(r["Sigma_e"]), 1, 1) : Matrix{Float64}(r["Sigma_e"])
        @test maximum(abs.(m.Sigma_e .- rS) .- 1e-12 .* abs.(rS)) <= 0
    end

    @testset "steady state" begin
        # Dynare stops its nonlinear solver at a looser tolerance: compare loosely, and require that
        # our residual is no worse than Dynare's
        ys, params = JDyn.steady_state(m; params=rparams)
        @test all(abs.(ys .- rys) .<= 1e-6 .+ 1e-4 .* abs.(rys))
        ours = maximum(abs, JDyn.static_resid(m, ys, nothing, params))
        theirs = maximum(abs, JDyn.static_resid(m, rys, nothing, rparams))
        println("$name: max |ys - Dynare| = $(maximum(abs, ys .- rys)), residual $ours (Dynare's $theirs)")
        @test ours <= max(theirs, 1e-12)
    end

    sol = JDyn.first_order(m; ys=rys, params=rparams)
    @testset "decision rules" begin
        @test sol.order_var == vec(Int.(r["order_var"]))
        gx, gu = JDyn.ghx(sol), JDyn.ghu(sol)
        rx = reshape(Float64.(r["ghx"]), size(gx)); ru = reshape(Float64.(r["ghu"]), size(gu))
        dx, du = maximum(abs, gx .- rx), maximum(abs, gu .- ru)
        println("$name: max |d ghx| = $dx, max |d ghu| = $du")
        @test all(abs.(gx .- rx) .<= 1e-11 .+ 1e-9 .* abs.(rx))
        @test all(abs.(gu .- ru) .<= 1e-11 .+ 1e-9 .* abs.(ru))
    end

    @testset "IRFs" begin
        if !haskey(r, "irfs")
            @test_skip "no IRFs in reference"
        else
            chol = Matrix(cholesky(Symmetric(m.Sigma_e)).L)
            names = split(r["irf_names"], ","); irfs = r["irfs"]
            byLen = sort(m.exo_names; by=length, rev=true)
            worst = 0.0
            for (k, label) in enumerate(names)
                series = vec(irfs[k, :])
                shock = byLen[findfirst(x -> endswith(label, "_" * x), byLen)]
                var = label[1:end-length(shock)-1]
                path = JDyn.irf(sol, chol[:, m.ix[shock]]; periods=length(series))[:, m.ie[var]]
                worst = max(worst, maximum(abs, path .- series))
                @test all(abs.(path .- series) .<= 1e-12 .+ 1e-9 .* abs.(series))
            end
            println("$name: $(length(names)) IRFs, max |diff| = $worst")
        end
    end
end
