# Deterministic simulations against Dynare 7.2 (Octave 8.4); port of pydyn's tests/test_perfect_foresight.py.
#
# References: pydyn's reference/pf_*.mod run by reference/r_perfect_foresight.m (pf_*.mat).
# Dynare was run with tolf=1e-8 on the stacked problems (1e-6 for the homotopy model, whose residuals have a
# round-off floor of 1.5e-8; 1e-12 for the period-by-period models; LMMCP TolFun 1e-10 and 1e-8) and JDyn
# reads the same tolerances from the .mod files, so both stop at the same Newton iterate. The tests assert
# rtol=1e-10, atol=1e-11 on the paths; the measured differences are printed at the end.
using Test
isdefined(@__MODULE__, :REF) || include("common.jl")

const PF_RTOL, PF_ATOL = 1e-10, 1e-11
const PF_DIFFS = Dict{String,Float64}()
const PF_RELDIFFS = Dict{String,Float64}()

function pf_build(name; text=nothing)
    d = mktempdir()
    if text === nothing
        cp(joinpath(REF, name * ".mod"), joinpath(d, name * ".mod"))
    else
        write(joinpath(d, name * ".mod"), text)
    end
    with_logger(NullLogger()) do
        JDyn.load(joinpath(d, name * ".mod"))
    end
end

pf_ref(name) = matread(joinpath(REF, name * ".mat"))
pf_solve(m; kw...) = with_logger(NullLogger()) do
    perfect_foresight(m; kw...)
end
_vec(x) = x isa Number ? [Float64(x)] : vec(Float64.(x))
_names(r) = String.(split(r["endo_names"], ","))

"Largest absolute difference between two paths (NaN-aware: a NaN where the other is finite counts as Inf)."
function pf_maxdiff(a, b)
    size(a) == size(b) || return Inf
    d = 0.0
    for (x, y) in zip(a, b)
        (isnan(x) && isnan(y)) && continue
        d = max(d, abs(x - y))
    end
    d
end

"Largest difference relative to the value in the reference, |a - b| / max(|b|, 1)."
pf_reldiff(a, b) = size(a) == size(b) ? maximum(abs(x - y) / max(abs(y), 1) for (x, y) in zip(a, b) if !(isnan(x) && isnan(y)); init=0.0) : Inf

function pf_check(res, r, label; rtol=PF_RTOL, atol=PF_ATOL)
    @test res.status && Int(r["status"]) == 1
    endo, exo = Matrix{Float64}(r["endo_simul"]), Matrix{Float64}(r["exo_simul"])
    @test size(res.endo_simul) == size(endo)
    @test size(res.exo_simul) == size(exo)
    @test maximum(abs, res.exo_simul .- exo) <= 1e-14 * max(1, maximum(abs, exo))
    @test all(abs.(res.endo_simul .- endo) .<= atol .+ rtol .* abs.(endo))
    @test all(abs.(res.steady_state .- _vec(r["steady_state"])) .<= atol .+ rtol .* abs.(_vec(r["steady_state"])))
    @test res.homotopy_iterations == Int(r["homotopy_iterations"])
    haskey(r, "iterations") && @test res.iterations == Int(r["iterations"])
    @test res.endo_names == _names(r)
    PF_DIFFS[label] = pf_maxdiff(res.endo_simul, endo)
    PF_RELDIFFS[label] = pf_reldiff(res.endo_simul, endo)
end

# name -> what it covers
const PF_STACKED = Dict(
    "pf_temp" => "temporary shocks on two variables over ranges, from the steady state; long lags/leads, exo lag and lead",
    "pf_endval" => "permanent shock: endval, steady after each block, temporary shocks on the way",
    "pf_endval_nosteady" => "endval without steady: terminal condition = the values of the block",
    "pf_initval_nosteady" => "initval without steady: start and end at the values of the block",
    "pf_histval" => "histval with a lag beyond one and a lagged exogenous variable",
    "pf_mshocks" => "mshocks relative to the terminal and to the initial steady state",
    "pf_mshocks0" => "mshocks without endval; shocks(overwrite)",
    "pf_etp" => "endogenous_terminal_period")

@testset "Perfect foresight" begin
    @testset "stacked Newton: $name" for name in sort(collect(keys(PF_STACKED)))
        m, r = pf_build(name), pf_ref(name)
        res = pf_solve(m)
        pf_check(res, r, name)
        @test res.solver == "stacked Newton" && res.homotopy_completion_share == 1
        @test size(res.endo_simul) == (m.n_endo, res.periods + 2)        # oo_.endo_simul: initial column, periods, terminal column
        @test res.max_residual < Float64(r["tolf"]) && Float64(r["error"]) < Float64(r["tolf"])
        if haskey(r, "initial_steady_state")
            @test res.initial_steady_state !== nothing
            @test maximum(abs, res.initial_steady_state .- _vec(r["initial_steady_state"])) <= 1e-11 + 1e-10 * maximum(abs, _vec(r["initial_steady_state"]))
            @test maximum(abs, res.endo_simul[:, 1] .- _vec(r["initial_steady_state"])) <= 1e-11 + 1e-10 * maximum(abs, _vec(r["initial_steady_state"]))
        else
            @test res.initial_steady_state === nothing
        end
    end

    @testset "initval without steady is not a steady state" begin
        # Dynare's semantics: without `steady` the values of initval are y_0, the guess and the terminal condition
        m, r = pf_build("pf_initval_nosteady"), pf_ref("pf_initval_nosteady")
        res = pf_solve(m)
        k = findfirst(==("k"), m.endo_names)
        @test res.endo_simul[k, 1] == 6.0 && res.endo_simul[k, end] == 6.0 == r["endo_simul"][k, end]
        aux = findfirst(==("AUX_ENDO_LAG_1_1"), m.endo_names)        # k(-1): set to k by the initval block
        @test res.endo_simul[aux, 1] == 6.0
        res2 = pf_solve(m; steady=true)
        ys, _ = with_logger(NullLogger()) do
            steady_state(m)
        end
        @test maximum(abs, res2.endo_simul[:, 1] .- ys) < 1e-10
        @test maximum(abs, res2.endo_simul[:, end] .- ys) < 1e-10
    end

    @testset "endval without steady keeps the values" begin
        m = pf_build("pf_endval_nosteady")
        res = pf_solve(m)
        @test res["k"][end] == 7.5 && res["z"][end] == 1.05 && res["z"][1] == 1.0
        @test res.info["recompute"] == false                    # the endval values are not a steady state: kept as they are
        @test abs(res["k"][1] - 7.5) > 1                        # starts from the initial steady state
    end

    @testset "histval conventions" begin
        m, r = pf_build("pf_histval"), pf_ref("pf_histval")
        res = pf_solve(m)
        @test res["k"][1] == 60 && res["x"][1] == 0.9
        @test res["c"][1] == 0 && res["z"][1] == 0              # not in the histval block: zero, not the steady state
        @test res["AUX_ENDO_LAG_2_1"][1] == 0.1                 # z(-1)
        @test res["AUX_EXO_LAG_3_0"][1] == 0.9                  # x(0)
        @test all(abs.(res.endo_simul .- r["endo_simul"]) .<= PF_ATOL .+ PF_RTOL .* abs.(r["endo_simul"]))
    end

    @testset "period-by-period models: $name" for (name, extra) in (("pf_backward", 1), ("pf_forward", 1), ("pf_static", 0))
        # purely backward (with histval and a second lag), purely forward and static models
        m, r = pf_build(name), pf_ref(name)
        res = pf_solve(m)
        pf_check(res, r, name)
        @test res.solver == "period by period"
        @test size(res.endo_simul, 2) == res.periods + extra
        @test (res.maximum_lag, res.maximum_lead) == (Int(r["maximum_lag"]), Int(r["maximum_lead"]))
    end

    @testset "model(linear) and linear_approximation" begin
        local m, res
        for name in ("pf_linear", "pf_linapprox")
            m, r = pf_build(name), pf_ref(name)
            res = pf_solve(m)
            pf_check(res, r, name)
            @test res.solver == "linear"
        end
        # the nonlinear solution of the same model differs from its linear approximation
        exact = pf_solve(m; linear_approximation=false)
        @test exact.solver == "stacked Newton" && maximum(abs, exact.endo_simul .- res.endo_simul) > 1e-5
    end

    @testset "homotopy is needed" begin
        # Newton fails on the full permanent shock; Dynare 7's homotopy on the shock size solves it
        m, r0 = pf_build("pf_nohomotopy"), pf_ref("pf_nohomotopy")
        @test Int(r0["status"]) == 0
        err = try; pf_solve(m); nothing; catch e; e; end
        @test err isa PerfectForesightError
        @test !err.result.status && err.result.homotopy_completion_share == 0
        @test err.result.iterations == Int(r0["iterations"])
        @test !pf_solve(m; raise_on_failure=false).status
        r = pf_ref("pf_homotopy")
        res = pf_solve(m; no_homotopy=false)                    # keyword overrides the .mod file
        pf_check(res, r, "pf_nohomotopy (no_homotopy=false)")
        @test res.homotopy_iterations == 15 && res.homotopy_completion_share == 1
        for (name, n_it) in (("pf_homotopy", 15), ("pf_homotopy_opts", 12))
            res = pf_solve(pf_build(name))
            pf_check(res, pf_ref(name), name)
            @test res.homotopy_iterations == n_it
        end
    end

    @testset "homotopy linearization fallbacks" begin
        local m
        for (name, flag) in (("pf_homotopy_lin", :homotopy_linearization), ("pf_homotopy_marg", :homotopy_marginal_linearization))
            m, r = pf_build(name), pf_ref(name)
            res = pf_solve(m)
            pf_check(res, r, name)
            @test getfield(res, flag) && Int(r[String(flag)]) == 1
            @test res.homotopy_completion_share ≈ Float64(r["homotopy_completion_share"]) == 0.8
            @test isapprox(res.max_residual, Float64(r["error"]); rtol=1e-6)
        end
        @test_throws ArgumentError pf_solve(m; homotopy_marginal_linearization_fallback=0.0)
        @test_throws ArgumentError pf_solve(m; no_homotopy=true)
    end

    @testset "zero lower bound with lmmcp" begin
        # [mcp = 'i > -0.01'] + perfect_foresight_solver(lmmcp): the bound binds for 11 periods
        m, r = pf_build("pf_zlb"), pf_ref("pf_zlb")
        res = pf_solve(m)
        pf_check(res, r, "pf_zlb")
        @test res.solver == "lmmcp"
        i = res["i"][2:end-1]
        @test count(<=(-0.01 + 1e-12), i) == 11 && minimum(i) >= -0.01 - 1e-12 && i[21] > -0.01
        lb, ub, order = JDyn.complementarity_conditions(m)
        @test lb[findfirst(==("i"), m.endo_names)] == -0.01 && all(isinf, ub) && order == [1, 2, 4, 3]
        # without the lmmcp option Dynare ignores the condition; JDyn does the same and says so
        text = replace(read(joinpath(REF, "pf_zlb.mod"), String), "perfect_foresight_solver(lmmcp, no_homotopy)" => "perfect_foresight_solver")
        m2 = pf_build("pf_zlb_off"; text=text)
        free = @test_logs (:warn, r"complementarity") match_mode = :any perfect_foresight(m2)
        @test minimum(free["i"]) < -0.02
        again = pf_solve(m2; lmmcp=true, lmmcp_tolfun=1e-10)
        @test all(abs.(again.endo_simul .- r["endo_simul"]) .<= PF_ATOL .+ PF_RTOL .* abs.(r["endo_simul"]))
    end

    @testset "irreversible investment with lmmcp (rbcii)" begin
        # Dynare's tests/lmmcp/rbcii.mod (the ⟂ syntax, lower and upper bound): investment is at its lower
        # bound for 51 periods; LMMCP TolFun is 1e-8 in both
        m, r = pf_build("pf_rbcii"), pf_ref("pf_rbcii")
        res = pf_solve(m)
        pf_check(res, r, "pf_rbcii"; rtol=1e-9, atol=1e-10)
        inv = res["i"][2:end-1]
        @test count(<=(1e-7), inv) == 51 && minimum(inv) >= -1e-7        # the bound holds to the LMMCP tolerance
        @test isapprox(res.max_residual, Float64(r["error"]); rtol=1e-3)
    end

    @testset "expectation errors" begin
        # perfect_foresight_with_expectation_errors_setup/_solver with shocks(learnt_in=...), mshocks(learnt_in=...),
        # endval(learnt_in=...) (level, add, multiply, overwrite)
        m, r = pf_build("pf_pfwee"), pf_ref("pf_pfwee")
        res = pf_solve(m)
        pf_check(res, r, "pf_pfwee")
        si, ti = res.info["shocks_info"], res.info["terminal_info"]
        @test isnan.(si) == isnan.(r["shocks_info"])
        @test pf_maxdiff(si, r["shocks_info"]) <= 1e-14
        @test pf_maxdiff(ti, r["terminal_info"]) <= 1e-14
        @test [p for (p, _, _) in res.info["solves"]] == [1, 2, 3, 6]          # one solve per period with news
    end

    @testset "unanticipated shocks" begin
        m, r = pf_build("pf_surprise"), pf_ref("pf_surprise")
        pf_check(pf_solve(m), r, "pf_surprise")
        text = read(joinpath(REF, "pf_surprise.mod"), String)
        text = replace(text, r"^shocks\(learnt_in[^;]*;.*?end;\s*"ms => "")
        text = replace(text, r"perfect_foresight_with_expectation_errors_setup\(periods = 60\);.*"s => "perfect_foresight_setup(periods=60);\n")
        m2 = pf_build("pf_surprise_jl"; text=text)
        res = with_logger(NullLogger()) do
            JDyn.unanticipated_shocks(m2, [(1, Dict("x" => 1.2)), (3, Dict("x" => 0.9)), (4, Dict("x" => 1.1))]; tolf=1e-8, constant_simulation_length=true)
        end
        @test all(abs.(res.endo_simul .- r["endo_simul"]) .<= PF_ATOL .+ PF_RTOL .* abs.(r["endo_simul"]))
        @test pf_maxdiff(res.exo_simul, r["exo_simul"]) <= 1e-14
        @test size(res.endo_simul, 2) == 62 + 3        # constant_simulation_length: the horizon moves with each surprise
        # the same shocks announced in period 1 give another path
        anticipated = pf_solve(m2; exo=res.exo_simul[1:62, :], tolf=1e-8)
        @test maximum(abs, anticipated.endo_simul[:, 1:4] .- res.endo_simul[:, 1:4]) > 1e-4
    end

    @testset "options from the file and keywords" begin
        m, r = pf_build("pf_temp"), pf_ref("pf_temp")
        base = pf_solve(m)
        @test base.info["tolf"] == 1e-8 == Float64(r["tolf"])           # tolf read from the .mod file
        tight = pf_solve(m; tolf=1e-13)
        @test tight.iterations == base.iterations + 1 && tight.max_residual < 1e-13
        @test pf_solve(m; tol=1e-13).iterations == tight.iterations     # the older keyword
        @test_throws PerfectForesightError pf_solve(m; maxit=2, no_homotopy=true)
        short = pf_solve(m; periods=40)
        @test size(short.endo_simul) == (m.n_endo, 42)
        @test pf_maxdiff(JDyn.pf_exo_path(m, 150), r["exo_simul"]) <= 1e-14
        err = try; pf_solve(m; periods=8); nothing; catch e; e; end
        @test err isa PerfectForesightError && occursin("after the terminal period", err.msg)
        @test_throws ArgumentError pf_solve(m; not_an_option=1)
        err = try; pf_solve(pf_build("pf_backward"); endogenous_terminal_period=true); nothing; catch e; e; end
        @test err isa ArgumentError && occursin("purely backward", err.msg)
        text = replace(read(joinpath(REF, "pf_temp.mod"), String), "tolf=1e-8, tolx=1e-8" => "tolf=1e-8, stack_solve_algo=2, markowitz=0.3, robust_lin_solve")
        m2 = pf_build("pf_temp_opts"; text=text)
        res = @test_logs (:warn, r"stack_solve_algo=2") (:warn, r"markowitz.*robust_lin_solve|robust_lin_solve.*markowitz") match_mode = :any perfect_foresight(m2)
        @test all(abs.(res.endo_simul .- r["endo_simul"]) .<= PF_ATOL .+ PF_RTOL .* abs.(r["endo_simul"]))
    end

    @testset "result conveniences" begin
        m = pf_build("pf_endval")
        res = pf_solve(m)
        p = JDyn.paths(res)
        @test sort(collect(keys(p))) == sort(m.endo_names) && size(p["k"]) == (152,)
        @test p["k"] == res["k"]
        d0, dT = JDyn.paths(res; deviation=:initial), JDyn.paths(res; deviation=:terminal, exo=true, simulation_only=true)
        @test abs(d0["k"][1]) < 1e-12 && abs(dT["k"][end]) < 0.05 && size(dT["k"]) == (150,)
        @test abs(dT["z"][end]) < 1e-15 && abs(JDyn.paths(res; deviation=:initial, exo=true)["z"][end] - 0.1) < 1e-15
        @test res.exo_names == ["z", "g"] && res.periods == 150
    end

    @testset "stacked evaluation: Jacobian by finite differences" begin
        m = pf_build("pf_temp")
        n, T = m.n_endo, 12
        ys, _ = with_logger(NullLogger()) do
            steady_state(m)
        end
        P = JDyn.PFStacked(m, m.params, ys)
        Y = repeat(ys, 1, T) .* (1 .+ 0.01 .* sin.(reshape(1:n*T, n, T)))
        X = repeat(JDyn.exo_steady(m), 1, T)
        res, A = JDyn.evaluate!(P, Y, ys, ys, X)
        res, A = copy(res), copy(A)
        J = zeros(T * n, T * n)
        h = 1e-6
        for j in 1:T*n
            Yp = copy(Y); Yp[j] += h
            Ym = copy(Y); Ym[j] -= h
            rp, _ = JDyn.evaluate!(P, Yp, ys, ys, X, false); rp = copy(rp)
            rm, _ = JDyn.evaluate!(P, Ym, ys, ys, X, false)
            J[:, j] = (rp .- rm) ./ (2h)
        end
        @test maximum(abs, Matrix(A) .- J) < 1e-6
    end
end

@testset "perfect_foresight_rbc and expectation errors (Dynare's shipped examples)" begin
    # unedited .mod files of Dynare 7.2's examples/perfect_foresight against Dynare's own results
    # (pydyn's reference/examples/perfect_foresight__*.mat: oo_.endo_simul at Dynare's default tolf = 1e-5)
    for name in ("perfect_foresight_rbc", "perfect_foresight_expectation_errors")
        d = mktempdir()
        for f in readdir(joinpath(EXAMPLES, "perfect_foresight"); join=true)
            isfile(f) && cp(f, joinpath(d, basename(f)); force=true)
        end
        m = with_logger(NullLogger()) do
            JDyn.load(joinpath(d, name * ".mod"))
        end
        r = matread(joinpath(REF, "examples", "perfect_foresight__$name.mat"))
        res = pf_solve(m)
        @test res.status
        endo = Matrix{Float64}(r["endo_simul"])
        @test size(res.endo_simul) == size(endo)
        PF_DIFFS["example $name (Dynare's tolf)"] = pf_maxdiff(res.endo_simul, endo)
        PF_RELDIFFS["example $name (Dynare's tolf)"] = pf_reldiff(res.endo_simul, endo)
        @test PF_DIFFS["example $name (Dynare's tolf)"] < 1e-9
        tight = pf_solve(m; tolf=1e-12)
        PF_DIFFS["example $name (tolf=1e-12)"] = pf_maxdiff(tight.endo_simul, endo)
        @test PF_DIFFS["example $name (tolf=1e-12)"] < 1e-6
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__) || get(ENV, "JDYN_PF_PRINT", "") == "1"
    println("\nMaximum differences of the paths against Dynare 7.2 (absolute; relative to max(|reference|, 1)):")
    for k in sort(collect(keys(PF_DIFFS)))
        println("  ", rpad(k, 58), @sprintf("%.2e   %.2e", PF_DIFFS[k], get(PF_RELDIFFS, k, NaN)))
    end
end
