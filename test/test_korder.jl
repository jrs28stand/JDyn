# Orders 2 and 3 through Dynare's k-order library: decision rules, simulation (with and without pruning) and
# generalised IRFs against Dynare 7.2 (port of pydyn tests/test_korder.py, test_irf_higher_order.py and the
# higher-order parts of test_stage3.py, test_auxiliary.py). Needs the library: deps/korder/build.sh.
using Test
isdefined(@__MODULE__, :REF) || include("common.jl")

const _HAVE_LIB = JDyn.korder_available()
_HAVE_LIB || @warn "libjdyn_korder not built (deps/korder/build.sh): higher-order tests skipped"
const MAXDIFF = Dict{String,Float64}()
function record!(key, d)
    MAXDIFF[key] = max(get(MAXDIFF, key, 0.0), d)
    d
end
maxabs(a, b) = maximum(abs, a .- b; init=0.0)
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-12); init=0.0)
# allclose as numpy: |a-b| <= atol + rtol |b|
close(a, b; rtol, atol) = all(abs.(a .- b) .<= atol .+ rtol .* abs.(b))
const COLLARD2 = joinpath(SS, "collard_2001_simulated_moments.mod")
const COLLARD3 = joinpath(REF, "collard_o3.mod")

function _load(modpath; output=nothing)
    load_copy(modpath; output=output)
end

if _HAVE_LIB
@testset "k_order decision rules" begin
    @testset "order 2 (collard_2001_simulated_moments)" begin
        m = _load(COLLARD2)
        r = refmat("clean_collard_order2.mat")
        dr = JDyn.k_order(m; order=2, ys=vec(r["ys"]), params=vec(r["params"]))
        for nm in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2")
            want = reshape(Float64.(r[nm]), size(dr[nm]))
            record!("rules order 2", maxabs(dr[nm], want))
            @test close(dr[nm], want; rtol=1e-9, atol=1e-12)
        end
    end
    @testset "order 3 (collard_o3)" begin
        m = _load(COLLARD3)
        r = refmat("clean_collard_order3.mat")
        dr = JDyn.k_order(m; order=3, ys=vec(r["ys"]), params=vec(r["params"]))
        for nm in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss",
                   "g_0", "g_1", "g_2", "g_3")
            want = reshape(Float64.(r[nm]), size(dr[nm]))
            record!("rules order 3", maxabs(dr[nm], want))
            @test close(dr[nm], want; rtol=1e-9, atol=1e-12)
        end
    end
    @testset "order 1 agrees with the Julia solver" begin
        m = _load(COLLARD2)
        sol = JDyn.first_order(m)
        res = JDyn.k_order(m; order=1, ys=sol.ys)
        record!("order 1 vs first_order", max(maxabs(res["ghx"], ghx(sol)), maxabs(res["ghu"], ghu(sol))))
        @test close(res["ghx"], ghx(sol); rtol=1e-9, atol=1e-12)
        @test close(res["ghu"], ghu(sol); rtol=1e-9, atol=1e-12)
    end
    @testset "refusals" begin
        m = _load(COLLARD2)           # the .mod file asks for order 2 only
        @test_throws JDyn.KOrderError JDyn.k_order(m; order=3)
        @test_throws JDyn.KOrderError JDyn.k_order(m; order=2, loglinear=true)
        @test_throws JDyn.KOrderError JDyn.k_order(m; order=2, library="/nonexistent/lib.so")
    end
    @testset "order 2 with auxiliary variables (aux_a)" begin
        m = _load(joinpath(REF, "aux_a.mod"); output="second")
        r = refmat("clean_aux_a.mat")
        dr = JDyn.k_order(m; order=2)
        for nm in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2")
            want = reshape(Float64.(r[nm]), size(dr[nm]))
            record!("rules order 2, aux_a", maxabs(dr[nm], want))
            @test close(dr[nm], want; rtol=1e-8, atol=1e-10)
        end
        y1 = simulate(dr, Float64.(r["ex"]); order=1)
        y2 = simulate(dr, Float64.(r["ex"]); order=2, pruning=true)
        record!("simulation aux_a order 1", maxabs(y1, Float64.(r["y1"])[2:end, :]))
        record!("simulation aux_a order 2 pruned", maxabs(y2, Float64.(r["y2_pruned"])[2:end, :]))
        @test close(y1, Float64.(r["y1"])[2:end, :]; rtol=1e-9, atol=1e-11)
        @test close(y2, Float64.(r["y2_pruned"])[2:end, :]; rtol=1e-9, atol=1e-11)
    end
    @testset "linear model: no second-order terms" begin
        m = build("var c k; varexo e; parameters a; a=0.5;\nmodel; c = a*c(-1) + 0.1*c(+1) + e + k; k = 0.5*k(-1)+e; end;\n" *
                  "shocks; var e; stderr 1; end;\nstoch_simul(order=2, irf=0);")
        dr = JDyn.k_order(m; order=2)
        @test maximum(abs, dr["ghxx"]) < 1e-12
        @test close(dr["ghx"], ghx(JDyn.first_order(m)); rtol=0, atol=1e-12)
    end
end

@testset "simulation and IRFs, orders 2 and 3" begin
    @testset "simulation order 2" begin
        m = _load(COLLARD2)
        r = refmat("clean_sim_order2.mat")
        dr = JDyn.k_order(m; order=2, ys=vec(r["ys"]), params=vec(r["params"]))
        for (key, order, pruning) in (("y1", 1, false), ("y2_unpruned", 2, false), ("y2_pruned", 2, true))
            y = simulate(dr, Float64.(r["ex"]); order=order, pruning=pruning)
            want = Float64.(r[key])[2:end, :]       # row 1 of Dynare's output is y0
            record!("simulation $key", maxabs(y, want))
            @test close(y, want; rtol=1e-10, atol=1e-12)
        end
        chol = Matrix(cholesky(Symmetric(m.Sigma_e)).L)
        for (key, pruning) in (("irf2_unpruned", false), ("irf2_pruned", true))
            for i in 1:m.n_exo
                y = irf(dr, chol[:, i]; Sigma_e=m.Sigma_e, periods=Int(r["long"]), order=2, pruning=pruning,
                        drop=Int(r["drop"]), draws=Float64.(r["draws"]))
                want = permutedims(Float64.(r[key])[:, :, i])
                record!("GIRF $key", maxabs(y, want))
                @test close(y, want; rtol=1e-9, atol=1e-13)
            end
        end
    end
    @testset "simulation order 3" begin
        m = _load(COLLARD3)
        r = refmat("clean_sim_order3.mat")
        dr = JDyn.k_order(m; order=3, ys=vec(r["ys"]), params=vec(r["params"]))
        for (key, pruning) in (("y3_pruned", true), ("y3_unpruned", false))
            y = simulate(dr, Float64.(r["ex"]); order=3, pruning=pruning)
            want = Float64.(r[key])[2:end, :]
            record!("simulation $key", maxabs(y, want))
            @test close(y, want; rtol=1e-10, atol=1e-12)
        end
    end
    @testset "generalised IRFs, order 3 and Dynare's own irf.m" begin
        m = _load(COLLARD3)
        r = refmat("clean_irf_order3.mat")
        dr = JDyn.k_order(m; order=3, ys=vec(r["ys"]), params=vec(r["params"]))
        chol = Matrix(cholesky(Symmetric(m.Sigma_e)).L)
        for (key, pruning) in (("irf3_unpruned", false), ("irf3_pruned", true)), i in 1:m.n_exo
            y = irf(dr, chol[:, i]; Sigma_e=m.Sigma_e, periods=Int(r["long"]), order=3, pruning=pruning,
                    drop=Int(r["drop"]), draws=Float64.(r["draws"]))
            want = permutedims(Float64.(r[key])[:, :, i])
            record!("GIRF $key", maxabs(y, want))
            @test close(y, want; rtol=1e-9, atol=1e-13)
        end
        for (key, order, pruning, shock) in (("irf_dynare", 3, true, 1), ("irf_dynare_order2", 2, true, 1),
                                             ("irf_dynare_unpruned", 3, false, 2))
            want = permutedims(Float64.(r[key]))
            @test maximum(abs, want) > 1e-4
            y = irf(dr, chol[:, shock]; Sigma_e=m.Sigma_e, periods=Int(r["long"]), order=order, pruning=pruning,
                    drop=30, draws=Float64.(r["dynare_draws"]))
            record!("GIRF $key (Dynare's irf.m)", maxabs(y, want))
            @test close(y, want; rtol=1e-9, atol=1e-13)
        end
    end
end

@testset "max differences" begin
    for k in sort(collect(keys(MAXDIFF)))
        @printf("  %-40s %.2e\n", k, MAXDIFF[k])
    end
end
end   # _HAVE_LIB
