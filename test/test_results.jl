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

@testset "re-running commands on the workspace (no preprocessing)" begin
    d = _copy_dir(joinpath(EXAMPLES, "perfect_foresight"))
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "perfect_foresight_rbc.mod"))
    end
    oo_ = ws.oo_
    base, x0 = copy(oo_.endo_simul), copy(oo_.exo_simul)
    oo_.exo_simul .= x0[1] .+ 2 .* (x0 .- x0[1])                       # a bigger shock path
    JDyn.perfect_foresight_solver!(ws; tolf=1e-12)
    big = copy(oo_.endo_simul)
    r = JDyn.perfect_foresight(ws.model; exo=Matrix(oo_.exo_simul), tolf=1e-12)
    @test maximum(abs, big .- r.endo_simul) < 1e-9
    @test maximum(abs, big[1, 2:5] .- base[1, 2:5]) > 1e-3
    oo_.exo_simul .= x0
    JDyn.perfect_foresight_solver!(ws)
    @test maximum(abs, oo_.endo_simul .- base) < 1e-6                     # Dynare's tolf = 1e-5

    d = _copy_dir(SS)
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "collard_2001_theoretical_moments.mod"))
    end
    oo_, M_, m = ws.oo_, ws.M_, ws.model
    for rho in (0.5, 0.9)
        JDyn.set_param_value!(ws, "rho", rho)
        JDyn.stoch_simul!(ws; order=1, irf=12, nomoments=true)
        @test ws.options_.irf == 12 && ws.options_.order == 1                 # options persist, as in Dynare
        sol = JDyn.first_order(m; params=vec(M_.params))
        y = JDyn.irf(sol, JDyn.get_lower_cholesky_covariance(m.Sigma_e)[:, m.ix["e"]]; periods=12)
        @test size(oo_.irfs.y_e) == (1, 12)
        @test maximum(abs, vec(oo_.irfs.y_e) .- y[:, m.ie["y"]]) < 1e-14
        @test maximum(abs, oo_.dr.ghx .- JDyn.ghx(sol)) < 1e-14
    end
    M_.params[m.ip["rho"]] = 0.7                                          # writing M_ directly also works
    JDyn.stoch_simul!(ws, ["y"])
    @test JDyn.get_param(m, "rho") == 0.7 && collect(oo_.var_list) == ["y"]
    M_.Sigma_e .= 2 .* M_.Sigma_e
    JDyn.stoch_simul!(ws, ["y"])
    @test m.Sigma_e == M_.Sigma_e
end

@testset "steady-state file sets parameters again on every re-run (nk_baseline)" begin
    d = _copy_dir(SS)
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "nk_baseline.mod"))
    end
    M_, oo_, m = ws.M_, ws.oo_, ws.model
    JDyn.set_param_value!(ws, "betta", 0.99)
    JDyn.stoch_simul!(ws; order=1, irf=10, nomoments=true)
    P(n) = M_.params[m.ip[n]]
    # nk_baseline_steadystate.m sets gammma1 = mu_z*mu_I/betta - (1-delta): it follows the new betta
    @test P("gammma1") ≈ P("Lambdax") * exp(P("Lambdamu")) / 0.99 - (1 - P("delta")) rtol = 1e-14
    @test maximum(abs, JDyn.static_resid(m, vec(oo_.steady_state), nothing, vec(M_.params))) < 1e-10
end
