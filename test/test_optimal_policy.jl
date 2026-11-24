# Optimal policy (src/optimal_policy.jl, src/results_policy.jl) against Dynare 7.2 under Octave 8.4: Ramsey steady state, rules and welfare,
# discretionary policy, optimal simple rules. References: test/ref_optimal_policy/clean_pol_*.mat, written by r_policy.sh / pol_export.m
# from Dynare's tests/optimal_policy, tests/discretionary_policy and examples/optimal_policy models (pol_*.mod).
include("common.jl")

const OPDIR = joinpath(@__DIR__, "ref_optimal_policy")
opref(case) = matread(joinpath(OPDIR, "clean_$case.mat"))
const MAXDIFF = Dict{String,Float64}()
note!(case, d) = (MAXDIFF[case] = max(get(MAXDIFF, case, 0.0), d); d)

# find_c.m of Dynare's tests/optimal_policy/Ramsey, with JDyn's csolve
_nk_ss(c, pai, beta, epsilon, phi, gamma, omega) = pai * (pai - 1) / c - beta * pai * (pai - 1) / c - epsilon * phi * (c + (omega / 2) * (pai - 1)^2)^(gamma + 1) / omega +
    (c + (omega / 2) * (pai - 1)^2) * (epsilon - 1) / (omega * c)
function find_c(c0, pai, beta, epsilon, phi, gamma, omega)
    f = c -> [_nk_ss(c[1], pai, beta, epsilon, phi, gamma, omega)]
    grad = c -> (f0 = f(c); (f0, reshape((f(c .+ 1e-6) .- f0) ./ 1e-6, 1, 1)))
    first(JDyn.csolve(f, [Float64(c0)], grad, 1e-8, 100))[1]
end

function load_op(mod; defs=String[], files=String[], output="second")
    d = mktempdir()
    cp(joinpath(OPDIR, mod * ".mod"), joinpath(d, mod * ".mod"))
    for f in files
        cp(joinpath(OPDIR, f), joinpath(d, f))
    end
    with_logger(NullLogger()) do
        JDyn.load(joinpath(d, mod * ".mod"); output=output, extra_args=defs, functions=Dict{String,Any}("find_c" => find_c))
    end
end
mat(x) = x isa Number ? fill(Float64(x), 1, 1) : Float64.(x)
rd(a, b) = (A = mat(a); B = mat(b); size(A) != size(B) && (B = reshape(B, size(A))); maximum(abs.(A .- B); init=0.0))

function compare_rules(case, dr, r, order; rtol=1e-9)
    @test Int.(vec(dr["order_var"])) == Int.(vec(r["order_var"]))
    note!(case, rd(dr["ys"], r["ys"]))
    @test isapprox(vec(dr["ys"]), vec(r["ys"]); rtol=1e-12, atol=1e-11)
    for k in ("ghx", "ghu", (order == 2 ? ("ghxx", "ghxu", "ghuu", "ghs2") : ())...)
        a = dr[k]; b = r[k]
        note!(case, rd(a, b))
        @test isapprox(vec(Float64.(a)), vec(Float64.(b)); rtol=rtol, atol=1e-11)
    end
end

function compare_welfare(case, v, r; ramsey=true, rtol=1e-11)
    note!(case, abs(v.unconditional - r["W_unc"]))
    @test isapprox(v.unconditional, r["W_unc"]; rtol=rtol, atol=1e-11)
    if ramsey
        note!(case, abs(v.zero_initial_multiplier - r["W_cond_L0"]))
        note!(case, abs(v.steady_initial_multiplier - r["W_cond_LSS"]))
        @test isapprox(v.zero_initial_multiplier, r["W_cond_L0"]; rtol=rtol, atol=1e-11)
        @test isapprox(v.steady_initial_multiplier, r["W_cond_LSS"]; rtol=rtol, atol=1e-11)
    else
        note!(case, abs(v.conditional - r["W_cond"]))
        @test isapprox(v.conditional, r["W_cond"]; rtol=rtol, atol=1e-11)
    end
end

# (case, .mod file, macro definitions, extra files)
const RAMSEY = [
    ("pol_nk_ramsey", "pol_nk_ramsey", String[], String[]),
    ("pol_nk_ramsey_o2", "pol_nk_ramsey", ["-DORDER=2"], String[]),
    ("pol_neo_ramsey", "pol_neo_ramsey", String[], String[]),
    ("pol_neo_ramsey_o1", "pol_neo_ramsey", ["-DORDER=1"], String[]),
    ("pol_ramsey_ex", "pol_ramsey_ex", String[], ["find_c.m"]),
    ("pol_ramsey_ex_o2", "pol_ramsey_ex", ["-DORDER=2"], ["find_c.m"]),
    ("pol_ramsey_ex2", "pol_ramsey_ex2", String[], ["find_c.m"]),
    ("pol_ramsey_aux", "pol_ramsey_aux", String[], ["find_c.m"]),
    ("pol_nk_ramsey_exp", "pol_nk_ramsey_exp", String[], String[]),
    ("pol_ramsey_lin", "pol_ramsey_lin", String[], String[]),
    ("pol_nk_ramsey_sf", "pol_nk_ramsey_sf", String[], ["pol_nk_ramsey_sf_steadystate.m"]),
    ("pol_gali_commit", "pol_gali_commit", String[], String[]),
    ("pol_mult_elim", "pol_mult_elim", String[], String[]),
    ("pol_ramsey_histval", "pol_ramsey_histval", String[], ["find_c.m"]),
    ("pol_ramsey_histval_o2", "pol_ramsey_histval", ["-DORDER=2"], ["find_c.m"]),
    ("pol_ramsey_2inst", "pol_ramsey_2inst", String[], String[]),
    ("pol_ramsey_policy", "pol_ramsey_policy", String[], String[]),
    ("pol_ramsey_policy_o2", "pol_ramsey_policy", ["-DORDER=2"], String[]),
]

function run_op(mod; defs=String[], files=String[], output="second")
    d = mktempdir()
    cp(joinpath(OPDIR, mod * ".mod"), joinpath(d, mod * ".mod"))
    for f in files
        cp(joinpath(OPDIR, f), joinpath(d, f))
    end
    with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, mod * ".mod"); output=output, extra_args=defs, functions=Dict{String,Any}("find_c" => find_c))
    end
end
oo_dr(ws) = (D = ws.oo_.dr; Dict{String,Any}(k => (k == "ghs2" ? vec(D[k]) : D[k]) for k in ("ys", "order_var", "ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2") if haskey(D, k)))
function oo_welfare(ws)
    p = ws.oo_.planner_objective_value
    c = p.conditional
    c isa Number ? (unconditional=p.unconditional, conditional=c) :
        (unconditional=p.unconditional, zero_initial_multiplier=c.zero_initial_multiplier, steady_initial_multiplier=c.steady_initial_multiplier)
end

@testset "Ramsey steady state, rules and welfare (dynare())" begin
    for (case, mod, defs, files) in RAMSEY
        @testset "$case" begin
            r = opref(case)
            ws = run_op(mod; defs=defs, files=files)
            m = ws.model
            @test is_ramsey(m)
            @test (ws.M_.ramsey_orig_endo_nbr, ws.M_.ramsey_orig_eq_nbr) == (r["ramsey_orig_endo_nbr"], r["ramsey_orig_eq_nbr"])
            order = Int(r["order"])
            compare_rules(case, oo_dr(ws), r, order)
            compare_welfare(case, oo_welfare(ws), r)
        end
    end
end

@testset "Ramsey steady state errors" begin
    d = mktempdir()
    text = replace(read(joinpath(OPDIR, "pol_nk_ramsey_sf.mod"), String), "R=1/beta-1;" => "R=0.03;")
    write(joinpath(d, "pol_nk_ramsey_sf.mod"), text)
    cp(joinpath(OPDIR, "pol_nk_ramsey_sf_steadystate.m"), joinpath(d, "pol_nk_ramsey_sf_steadystate.m"))
    m = with_logger(NullLogger()) do
        JDyn.load(joinpath(d, "pol_nk_ramsey_sf.mod"))
    end
    err = try; steady_state(m); nothing; catch e; e; end
    @test err isa OptimalPolicyError && err.info == 85
end

# tests/discretionary_policy models
@testset "discretionary_policy (dynare())" begin
    for case in ["pol_dennis", "pol_gali_disc", "pol_nk_disc_fwd", "pol_gali15", "pol_gali15_nl"]
        @testset "$case" begin
            r = opref(case)
            ws = run_op(case)
            m = ws.model
            @test is_discretionary(m) && !is_ramsey(m)
            dr = oo_dr(ws)
            @test Int.(vec(dr["order_var"])) == Int.(vec(r["order_var"]))
            for k in ("ys", "ghx", "ghu")
                note!(case, rd(dr[k], r[k]))
                @test isapprox(vec(Float64.(dr[k])), vec(Float64.(r[k])); rtol=1e-11, atol=1e-12)
            end
            v = oo_welfare(ws)
            if haskey(r, "exo_simul1")
                # Dynare's conditional welfare starts from its own first simulated shock: recompute with it
                @test isapprox(v.unconditional, r["W_unc"]; rtol=1e-12)
                v = with_logger(NullLogger()) do
                    evaluate_planner_objective(m, discretionary_policy(m; welfare=false).dr; order=1, u=vec(mat(r["exo_simul1"])), ramsey=false, discretionary=true)
                end
            end
            compare_welfare(case, v, r; ramsey=false, rtol=1e-12)
            if haskey(r, "var") && !haskey(r, "exo_simul1")
                note!(case, rd(ws.oo_.var, r["var"]))
                @test isapprox(vec(ws.oo_.var), vec(Float64.(r["var"])); rtol=1e-10, atol=1e-13)
            end
        end
    end
end

# tests/optimal_policy/OSR models
@testset "osr" begin
    for case in ["pol_osr", "pol_osr_corr", "pol_osr_cov", "pol_osr_planner", "pol_osr_ns"]
        @testset "$case objective at Dynare's optimum" begin
            r = opref(case)
            m = load_op(case)
            names = String.(split(r["osr_names"], ","))
            @test names == JDyn._osr_param_names(m)
            f = JDyn.osr_objective(m, vec(mat(r["osr_x"])), names)
            note!(case * " objective", abs(f - r["osr_f"]))
            @test isapprox(f, r["osr_f"]; rtol=1e-12, atol=1e-13)
        end
    end
    @testset "pol_osr_planner: same csminwel path (welfare at order 2)" begin
        r = opref("pol_osr_planner")
        ws = run_op("pol_osr_planner")
        o = ws.oo_.osr
        info = r["osr_info"]
        @test (o.optimization_info.iterations, o.optimization_info.funcCount, o.optimization_info.exitflag) ==
              (Float64(info["iterations"]), Float64(info["funcCount"]), Float64(info["exitflag"]))
        @test isapprox(o.optim_params.alpha, r["osr_x"]; rtol=1e-10)
        @test isapprox(o.objective_function, r["osr_f"]; rtol=1e-13)
    end
    for (case, rel) in [("pol_osr", 2e-6), ("pol_osr_corr", 1e-6), ("pol_osr_cov", 1e-10), ("pol_osr_ns", 0.02)]
        @testset "$case csminwel optimum" begin
            r = opref(case)
            ws = run_op(case)
            o = ws.oo_.osr
            names = String.(split(r["osr_names"], ","))
            x = [o.optim_params[n] for n in names]
            note!(case * " optimum", maximum(abs.(x .- vec(mat(r["osr_x"])))))
            @test isapprox(o.objective_function, r["osr_f"]; rtol=rel)
            @test o.error_indicator == 0
        end
    end
end

# Dynare 7.2 defect in evaluate_planner_objective at order 2 (Wyu built with kron(gu,gy) but used with kron(yhat,u)): matters with
# several shocks and several predetermined variables. Reference: Dynare 7.2 / Octave 8.4 on pol_nk_ramsey_2shocks.mod (two shocks, order 2;
# pol_nk_ramsey_2shocks_extra.m prints the values): unconditional, conditional with multipliers 0, with multipliers at steady state.
@testset "order-2 welfare: Dynare's Wyu ordering defect" begin
    DYN = (-25.21481898801387, -25.21462904698012, -25.21479747970518)
    ws = run_op("pol_nk_ramsey_2shocks")
    evaluate_planner_objective!(ws; replicate_dynare_bugs=true)
    p = ws.oo_.planner_objective_value
    got = (p.unconditional, p.conditional.zero_initial_multiplier, p.conditional.steady_initial_multiplier)
    note!("2shocks replicate", maximum(abs.(collect(got) .- collect(DYN))))
    @test all(isapprox.(got, DYN; rtol=1e-11))
    evaluate_planner_objective!(ws; replicate_dynare_bugs=false)
    p = ws.oo_.planner_objective_value
    @test p.unconditional == got[1]
    @test abs(p.conditional.zero_initial_multiplier - DYN[2]) > 1e-9          # corrected: differs from Dynare's
    # the consistent layout: Uyy*kron(gy,gu)*kron(yhat,u) is the cross term of 0.5*(gy yhat + gu u)' Uyy (gy yhat + gu u)
    n, ny, nu = 5, 3, 2
    A = randn(n, n); Uyy = A + A'; gy = randn(n, ny); gu = randn(n, nu); yh = randn(ny); u = randn(nu)
    @test isapprox((reshape(Uyy, 1, :) * kron(gy, gu) * kron(yh, u))[1], (gy * yh)' * Uyy * (gu * u); rtol=1e-12)
end
