# dynare(): M_, options_ and oo_ as MATLAB-style structs, against Dynare 7.2's own results
include("common.jl")

function _copy_dir(src)
    d = mktempdir()
    for f in readdir(src; join=true)
        isfile(f) && cp(f, joinpath(d, basename(f)); force=true)
    end
    d
end

@testset "MStruct and Cell read like MATLAB from Julia" begin
    s = JDyn.MStruct()
    s.a = 1.0
    s.b = JDyn.MStruct()
    s.b.c = [1.0 2.0]
    @test s.a == 1.0 && s.b.c == [1.0 2.0]
    @test propertynames(s) == (:a, :b)
    @test hasproperty(s, :a) && !hasproperty(s, :z)
    @test_throws ArgumentError s.z
    c = JDyn.Cell(Any["x", "y"], (2, 1))
    @test c[2] == "y" && collect(c) == ["x", "y"] && length(c) == 2 && size(c) == (2, 1)
end

@testset "dynare(): perfect_foresight_rbc (oo_.endo_simul)" begin
    d = _copy_dir(joinpath(EXAMPLES, "perfect_foresight"))
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "perfect_foresight_rbc.mod"))
    end
    r = matread(joinpath(REF, "examples", "perfect_foresight__perfect_foresight_rbc.mat"))
    oo_, M_ = ws.oo_, ws.M_
    @test size(oo_.endo_simul) == size(r["endo_simul"])                    # n_endo x (maxlag + periods + maxlead)
    @test maximum(abs, oo_.endo_simul .- r["endo_simul"]) < 1e-9
    @test size(oo_.exo_simul) == size(r["exo_simul"])                      # (maxlag + periods + maxlead) x n_exo
    @test maximum(abs, vec(oo_.exo_simul) .- vec(r["exo_simul"])) < 1e-14
    @test oo_.deterministic_simulation.status == true
    @test vec(M_.params) == vec(r["params"])
    @test maximum(abs, vec(oo_.steady_state) .- vec(r["ys"])) < 1e-12
    @test size(M_.lead_lag_incidence) == (3, 3) && M_.endo_nbr == 3 && M_.orig_endo_nbr == 2
    @test M_.endo_names[1] == "c" && size(M_.endo_names) == (3, 1)
    @test ws.options_.periods == 200
end

@testset "dynare(): collard_2001 stoch_simul (oo_.dr, oo_.endo_simul, oo_.irfs)" begin
    d = _copy_dir(SS)
    r = matread(joinpath(REF, "examples", "stochastic_simulations__collard_2001_simulated_moments.mat"))
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "collard_2001_simulated_moments.mod"); shocks=Dict(1 => r["exo_simul"]))
    end
    oo_ = ws.oo_
    # Dynare's steady state stops at a residual of 1e-6 (tolf = 1e-5); JDyn's is exact, so the
    # results agree to ~1e-5 here (test_korder.jl checks the decision rules at Dynare's ys to 1e-12)
    @test vec(oo_.dr.order_var) == vec(r["order_var"])
    for k in ("ghx", "ghu", "ghxx", "ghxu", "ghuu")
        @test maximum(abs, oo_.dr[k] .- r[k]) < 1e-4
    end
    @test size(oo_.endo_simul) == size(r["endo_simul"])                    # n_endo x periods
    @test maximum(abs, oo_.endo_simul .- r["endo_simul"]) < 1e-4
    @test oo_.exo_simul == r["exo_simul"]
    @test hasproperty(oo_.irfs, :y_e) && size(oo_.irfs.y_e) == (1, 40)
    @test size(oo_.mean) == (length(ws.oo_.var_list), 1)
end
