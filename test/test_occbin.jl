# OccBin: the piecewise-linear simulation of the rbc_occbin example against Dynare 7.2 (port of pydyn's
# tests/test_occbin.py), plus the plotted content (tidy tables).
using Test; include("common.jl")

@testset "OccBin: rbc_occbin simulation" begin
    m = load_copy(joinpath(EXAMPLES, "occbin", "rbc_occbin.mod"))
    ob = Occbin(m)
    ref = refmat("clean_rbc_occbin.mat")
    opts = m.options["occbin_solver"]
    sh, pos = JDyn.surprise_shocks(ob)
    @test pos == [1]
    res = simulate(ob, sh; periods=opts["simul.periods"], check_ahead_periods=opts["simul.check_ahead_periods"], exo_pos=pos)

    @testset "inputs" begin
        @test vec(res.shocks) == vec(ref["occ_shocks"])
        @test ob.k == 2 && ob.names == ["IRR", "INEG"]
        @test maximum(abs.(res.ys .- ref["ys"])) < 1e-10
    end
    @testset "paths" begin
        @test isapprox(res.linear, ref["occ_linear"]; rtol=1e-10, atol=1e-12)
        @test maximum(abs.(ref["occ_piecewise"] .- ref["occ_linear"])) > 1e-3      # the constraints do bind
        dpw = maximum(abs.(res.piecewise .- ref["occ_piecewise"]))
        dlin = maximum(abs.(res.linear .- ref["occ_linear"]))
        println("rbc_occbin: max |piecewise - Dynare| = ", dpw, ", max |linear - Dynare| = ", dlin)
        @test isapprox(res.piecewise, ref["occ_piecewise"]; rtol=1e-10, atol=1e-12)
    end
    @testset "regime history" begin
        rh = ref["occ_regime"]          # rows: regime1, regimestart1, regime2, regimestart2
        @test size(rh, 2) == length(res.regime_history)
        ok = true
        for (t, per) in enumerate(res.regime_history), (j, (regime, start)) in enumerate(per)
            ok &= Int.(vec(collect(rh[2j-1, t]))) == regime && Int.(vec(collect(rh[2j, t]))) == start
        end
        @test ok
    end
    @testset "the .mod file's own options and shocks (simulate(ob))" begin
        r2 = simulate(ob)
        @test r2.piecewise == res.piecewise
        @test Dict(pairs(mod_options(ob))) == Dict(:periods => 200, :check_ahead_periods => 200)
    end
    @testset "IRF is state dependent" begin
        # a positive TFP shock raises investment: neither constraint binds, the response is linear
        up = irf(ob, [0.01]; periods=30)
        @test isapprox(up, irf(ob.ref, [0.01]; periods=30); atol=1e-12)
        # a large negative shock hits the investment floor: the response departs from the linear one
        down = irf(ob, [-0.05]; periods=30)
        @test maximum(abs.(down .- irf(ob.ref, [-0.05]; periods=30))) > 1e-4
    end
    @testset "plotted content" begin
        tab = JDyn.occbin_graph_table(ob, res)
        vars = ["log_C", "epsilon", "Lambda", "log_K", "log_Invest", "A"]
        @test unique(r.variable for r in tab) == vars
        @test Set(r.solution for r in tab) == Set(["piecewise", "linear", "shock"])
        c = findfirst(==("log_C"), m.endo_names)
        pw = [r.value for r in tab if r.variable == "log_C" && r.solution == "piecewise"]
        @test pw == res.piecewise[:, c]
        shk = [r.value for r in tab if r.variable == "epsilon"]
        @test length(shk) == 200 && shk[1:169] == vec(ref["occ_shocks"]) && all(iszero, shk[170:end])
        @test length(tab) == 200 * (5 * 2 + 1)                   # 5 endogenous variables twice, plus the shock once
        dev = JDyn.irf_table(res; variables=["log_C"], noconstant=true)
        @test [r.value for r in dev if r.solution == "linear"] ≈ res.linear[:, c] .- res.ys[c]
        path = mktempdir()
        f = JDyn.write_csv(joinpath(path, "occbin_simul.csv"), tab)
        lines = readlines(f)
        @test lines[1] == "variable,period,solution,value" && length(lines) == length(tab) + 1
        @test_throws ArgumentError JDyn.irf_table(res; variables=["nonexistent"])
    end
end

# one constraint, Dynare's borrowing-constraint model (pydyn's tests/test_stage3.py::test_occbin_one_constraint):
# simulate with the .mod file's own shocks(surprise) and occbin_setup / occbin_solver options
@testset "OccBin: borrowing constraint ($name)" for (name, curb) in (("borrcon_sim", false), ("borrcon_curb", true))
    m = load_copy(joinpath(REF, name * ".mod"))
    r = refmat("clean_$name.mat")
    ob = Occbin(m)
    @test ob.k == 1
    @test maximum(abs.(r["piecewise"] .- r["linear"])) > 1e-3
    opts = m.options["occbin_solver"]
    explicit = simulate(ob, JDyn.surprise_shocks(ob)[1]; periods=m.options["occbin_setup"]["simul.periods"],
                        maxit=get(opts, "simul.maxit", 30), curb_retrench=curb)
    from_mod = simulate(ob)                                      # the same, from the options of the file
    for res in (explicit, from_mod)
        @test isapprox(res.linear, r["linear"]; rtol=1e-10, atol=1e-12)
        @test isapprox(res.piecewise, r["piecewise"]; rtol=1e-10, atol=1e-12)
        rh = r["regime"]                                         # rows: regime, regimestart
        @test all(Int.(vec(collect(rh[1, t]))) == per[1][1] && Int.(vec(collect(rh[2, t]))) == per[1][2]
                  for (t, per) in enumerate(res.regime_history))
    end
    println(name, ": max |piecewise - Dynare| = ", maximum(abs.(from_mod.piecewise .- r["piecewise"])))
end
