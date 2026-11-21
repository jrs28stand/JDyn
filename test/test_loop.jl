# Parameter loops and steady-state hooks against Dynare 7.2 (first-order parts of pydyn
# tests/test_stage3.py; references from pydyn's reference/r_stage3.m), plus the cost of one
# set_param! + steady state + first-order solve.
using Test
include("common.jl")

_v(x) = x isa Number ? [Float64(x)] : vec(Float64.(x))
_m(x, shape) = reshape(Float64.(x isa Number ? [x] : x), shape)
allclose(a, b; rtol=1e-9, atol=0.0) = all(abs.(a .- b) .<= atol .+ rtol .* abs.(b))

@testset "parameter loop without reloading" begin
    # load once, then change parameters and re-solve, as set_param_value + stoch_simul in Dynare
    m = load_copy(joinpath(REF, "schorfheide_2000_sim.mod"))
    r = refmat("clean_param_loop.mat")
    alp, rho = _v(r["alp"]), _v(r["rho"])
    worst = zeros(4)
    for k in eachindex(alp)
        set_params!(m; alp=alp[k], rho=rho[k])
        sol = JDyn.first_order(m)                 # steady state recomputed from the new parameters
        rys = _v(r["ys"][k])
        gx, gu = JDyn.ghx(sol), JDyn.ghu(sol)
        rx, ru = _m(r["ghx"][k], size(gx)), _m(r["ghu"][k], size(gu))
        @test allclose(sol.ys, rys; rtol=1e-10)
        @test allclose(gx, rx; rtol=1e-8, atol=1e-10)
        @test allclose(gu, ru; rtol=1e-8, atol=1e-10)
        chol = Matrix(cholesky(Symmetric(m.Sigma_e)).L)
        path = JDyn.irf(sol, chol[:, 1]; periods=20)
        rirf = _m(r["irf"][k], size(path))
        @test allclose(path, rirf; rtol=1e-8, atol=1e-11)
        worst .= max.(worst, [maximum(abs, sol.ys .- rys), maximum(abs, gx .- rx), maximum(abs, gu .- ru), maximum(abs, path .- rirf)])
    end
    println("parameter loop: max |d ys|, |d ghx|, |d ghu|, |d irf| = $worst")
    @test length(unique(round.(_v(y); digits=8) for y in r["ys"])) > 1      # the loop really changes the steady state
end

# ---------------------------------------------------------------------- steady-state hooks
"Julia counterpart of examples/stochastic_simulations/nk_baseline_steadystate.m (as pydyn's Python port)."
function nk_steady_state(model, params, exo)
    p = Dict(zip(model.param_names, params))
    betta, h, alppha = p["betta"], p["h"], p["alppha"]
    thetap, thetaw, chi, chiw = p["thetap"], p["thetaw"], p["chi"], p["chiw"]
    epsilon, eta, gammma, varpsi, Phi = p["epsilon"], p["eta"], p["gammma"], p["varpsi"], p["Phi"]
    LambdaA, Lambdamu, delta = p["LambdaA"], p["Lambdamu"], p["delta"]
    PI = p["PIbar"]
    u = q = d = phi = 1.0
    LambdaYd = (LambdaA + alppha * Lambdamu) / (1 - alppha)
    mu_z, mu_I, mu_A = exp(LambdaYd), exp(Lambdamu), exp(LambdaA)
    Lambdax = mu_z
    gammma1 = mu_z * mu_I / betta - (1 - delta)
    gammma1 < 0 && throw(JDyn.SteadyStateError("parameter restriction violated: gammma1 < 0"))
    r = gammma1
    R = 1 + (PI * mu_z / betta - 1)
    Rbar = R
    PIstar = ((1 - thetap * PI^(-(1 - epsilon) * (1 - chi))) / (1 - thetap))^(1 / (1 - epsilon))
    PIstarw = ((1 - thetaw * PI^(-(1 - chiw) * (1 - eta)) * mu_z^(-(1 - eta))) / (1 - thetaw))^(1 / (1 - eta))
    mc = ((epsilon - 1) / epsilon * (1 - betta * thetap * PI^((1 - chi) * epsilon)) /
          (1 - betta * thetap * PI^(-(1 - epsilon) * (1 - chi))) * PIstar)
    w = (1 - alppha) * (mc * (alppha / r)^alppha)^(1 / (1 - alppha))
    wstar = w * PIstarw
    vp = (1 - thetap) / (1 - thetap * PI^((1 - chi) * epsilon)) * PIstar^(-epsilon)
    vw = (1 - thetaw) / (1 - thetaw * PI^((1 - chiw) * eta) * mu_z^eta) * PIstarw^(-eta)
    tempvaromega = alppha / (1 - alppha) * w / r * mu_z * mu_I
    cons_per_ld = mu_A / mu_z / vp * tempvaromega^alppha - tempvaromega * (1 - (1 - delta) / (mu_z * mu_I))
    labour_eq(ld) = ((1 - betta * thetaw * mu_z^(eta - 1) * PI^(-(1 - chiw) * (1 - eta))) /
                     (1 - betta * thetaw * mu_z^(eta * (1 + gammma)) * PI^(eta * (1 - chiw) * (1 + gammma))) -
                     (eta - 1) / eta * wstar / (varpsi * PIstarw^(-eta * gammma) * ld^gammma) *
                     ((1 - h / mu_z)^(-1) - betta * h / (mu_z - h)) / (cons_per_ld * ld - Phi / vp))
    # secant iterations from 0.25 (scipy.optimize.newton without a derivative)
    x0, x1 = 0.25, 0.25 * (1 + 1e-4) + 1e-4
    f0, f1 = labour_eq(x0), labour_eq(x1)
    for _ in 1:200
        (f1 == f0) && break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0, x1, f1 = x1, f1, x2, labour_eq(x2)
        abs(x1 - x0) < 1e-15 && break
    end
    ld = x1
    (isfinite(ld) && abs(labour_eq(ld)) <= 1e-10) || throw(JDyn.SteadyStateError("no solution for steady-state labour"))
    l = vw * ld
    k = tempvaromega * ld
    x = (1 - (1 - delta) / (mu_z * mu_I)) * k
    yd = (mu_A / mu_z * k^alppha * ld^(1 - alppha) - Phi) / vp
    c = cons_per_ld * ld - Phi / vp
    lambda = (1 - h * betta / mu_z) / (1 - h / mu_z) / c
    F = yd - 1 / (1 - alppha) * w * ld
    f = ((eta - 1) / eta * wstar * PIstarw^(-eta) * lambda * ld /
         (1 - betta * thetaw * mu_z^(eta - 1) * PI^(-(1 - chiw) * (1 - eta))))
    g1 = lambda * mc * yd / (1 - betta * thetap * PI^((1 - chi) * epsilon))
    g2 = epsilon / (epsilon - 1) * g1
    vals = Dict("d" => d, "c" => c, "mu_z" => mu_z, "mu_I" => mu_I, "mu_A" => mu_A, "lambda" => lambda, "R" => R,
                "PI" => PI, "r" => r, "x" => x, "u" => u, "q" => q, "f" => f, "ld" => ld, "w" => w, "wstar" => wstar,
                "PIstarw" => PIstarw, "PIstar" => PIstar, "g1" => g1, "g2" => g2, "yd" => yd, "mc" => mc, "k" => k,
                "vp" => vp, "vw" => vw, "l" => l, "phi" => phi, "F" => F)
    # parameters set in the .m file (Rbar, gammma1, Lambdax, LambdaYd) are written back
    newp = Dict("Rbar" => Rbar, "gammma1" => gammma1, "Lambdax" => Lambdax, "LambdaYd" => LambdaYd)
    ys = Float64[vals[v] for v in model.endo_names[1:Int(model.raw["orig_endo_nbr"])]]
    ys, Float64[get(newp, nm, p[nm]) for nm in model.param_names]
end

function check_nk(m, r)
    @test m.param_names == split(r["param_names"], ",")
    sol = JDyn.first_order(m)
    @test allclose(sol.ys, _v(r["ys"]); rtol=1e-8, atol=1e-10)
    @test allclose(sol.params, _v(r["params"]); rtol=1e-10)
    @test allclose(JDyn.ghx(sol), Float64.(r["ghx"]); rtol=1e-6, atol=1e-8)
    @test allclose(JDyn.ghu(sol), Float64.(r["ghu"]); rtol=1e-6, atol=1e-8)
    d1 = maximum(abs, JDyn.ghx(sol) .- r["ghx"])
    # change a deep parameter: the function must update the dependent parameter (Rbar)
    set_param!(m, "betta", 0.995)
    sb = JDyn.first_order(m)
    @test abs(sb.params[m.ip["Rbar"]] - sol.params[m.ip["Rbar"]]) > 1e-4
    @test allclose(sb.params, _v(r["params_b"]); rtol=1e-10)
    @test allclose(sb.ys, _v(r["ys_b"]); rtol=1e-8, atol=1e-10)
    @test allclose(JDyn.ghx(sb), Float64.(r["ghx_b"]); rtol=1e-6, atol=1e-8)
    @test allclose(JDyn.ghu(sb), Float64.(r["ghu_b"]); rtol=1e-6, atol=1e-8)
    max(d1, maximum(abs, JDyn.ghx(sb) .- r["ghx_b"]), maximum(abs, sb.ys .- _v(r["ys_b"])))
end

@testset "steady-state function updates parameters (Julia function)" begin
    m = load_copy(joinpath(SS, "nk_baseline.mod"); steady_state_fn=nk_steady_state)
    d = check_nk(m, refmat("clean_nk_baseline.mat"))
    println("nk_baseline with a Julia steady-state function: max diff = $d")
end

@testset "steady-state file updates parameters (_steadystate.m, interpreted)" begin
    m = load_copy(joinpath(SS, "nk_baseline.mod"); extra=[joinpath(SS, "nk_baseline_steadystate.m")])
    d = check_nk(m, refmat("clean_nk_baseline.mat"))
    println("nk_baseline with its _steadystate.m: max diff = $d")
end

function collard_helper(alpha, beta, delta, psi, theta)
    kh = ((1 / beta - (1 - delta)) / alpha)^(1 / (alpha - 1))
    f(h) = 1 - (((kh * h)^(alpha - 1)) * h^(1 - alpha) - ((1 - alpha) * kh^alpha / (theta * h^psi)) / (kh * h) + (1 - delta))
    h = 0.2
    for _ in 1:100                                   # Newton with a central-difference derivative
        fh = f(h)
        abs(fh) < 1e-15 && break
        dh = 1e-7 * max(abs(h), 1.0)
        h -= fh / ((f(h + dh) - f(h - dh)) / (2dh))
    end
    h
end

@testset "helper function in steady_state_model ($how)" for how in ("Julia function", ".m helper")
    r = refmat("clean_collard_2001_analytical_steady_state.mat")
    mod = joinpath(SS, "collard_2001_analytical_steady_state.mod")
    m = how == "Julia function" ?
        load_copy(mod; functions=Dict("collard_2001_analytical_steady_state_helper" => collard_helper)) :
        load_copy(mod; extra=[joinpath(SS, "collard_2001_analytical_steady_state_helper.m")])
    sol = JDyn.first_order(m)
    @test allclose(m.Sigma_e, Float64.(r["Sigma_e"]); rtol=1e-12)
    @test allclose(sol.ys, _v(r["ys"]); rtol=1e-7, atol=1e-9)
    at = JDyn.first_order(m; ys=_v(r["ys"]), params=_v(r["params"]))
    @test allclose(JDyn.ghx(at), Float64.(r["ghx"]); rtol=1e-9, atol=1e-11)
    @test allclose(JDyn.ghu(at), Float64.(r["ghu"]); rtol=1e-9, atol=1e-11)
    println("collard helper ($how): max |d ys| = $(maximum(abs, sol.ys .- _v(r["ys"]))), max |d ghx| = $(maximum(abs, JDyn.ghx(at) .- r["ghx"]))")
end

# ---------------------------------------------------------------------- timing
"Median time (s) of one set_param! + steady state + first-order solve."
function time_loop(m, pname; n=200)
    p0 = get_param(m, pname)
    for _ in 1:5
        set_param!(m, pname, p0); JDyn.first_order(m)
    end
    ts = Float64[]
    for k in 1:n
        set_param!(m, pname, p0 * (1 + 1e-4 * (k % 7)))
        push!(ts, @elapsed JDyn.first_order(m))
    end
    set_param!(m, pname, p0)
    sort(ts)[div(n, 2)]
end

if get(ENV, "JDYN_TIMING", "1") == "1"
    @testset "timing" begin
        m15 = load_copy(joinpath(REF, "schorfheide_2000_sim.mod"))               # 15 variables, steady_state_model
        m28 = load_copy(joinpath(SS, "nk_baseline.mod"); steady_state_fn=nk_steady_state)   # 28 variables
        m42 = load_copy(joinpath(EXAMPLES, "macroprocessor", "bkk_1992.mod"))     # 42 variables, numerical steady state
        t15, t28, t42 = time_loop(m15, "rho"), time_loop(m28, "betta"), time_loop(m42, "beta_H")
        @printf("set_param! + steady state + first order: %d vars %.3f ms; %d vars %.3f ms; %d vars %.3f ms\n",
                m15.n_endo, 1e3t15, m28.n_endo, 1e3t28, m42.n_endo, 1e3t42)
        @test t15 < 0.05
    end
end
