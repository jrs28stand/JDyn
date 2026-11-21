# Theoretical and simulated moments at first order against Dynare 7.2 (first-order parts of pydyn
# tests/test_moments.py; references reference/mom_*.mat from reference/r_moments.m), plus independent
# checks of the building blocks. Order-2 / pruned moments and the printed tables are not covered here.
#
# Elementwise rtol=1e-9 and an absolute tolerance of 1e-11 times the largest entry of the array; simulated
# moments of HP-filtered paths use 1e-8 / 1e-9 (conditioning of the HP system).
using Test, Random
include("common.jl")

const MO = JDyn
const MODELS = ["mom_collard", "mom_unitroot", "mom_aux", "mom_schorfheide"]
const SUBSET = Dict("mom_collard" => ["c", "y", "a"], "mom_unitroot" => ["z", "d", "q", "w"],
                    "mom_aux" => ["z", "c", "AUX_ENDO_LAG_2_1", "AUX_EXO_LAG_6_1"],
                    "mom_schorfheide" => ["y", "R", "gy_obs", "AUX_ENDO_LEAD_65"])
const CVD = [1, 2, 4, 10, 40]
# the option sets of reference/r_moments.m at order 1: name => (options, uses the variable subset)
const TH_CASES = Dict(
    "th1" => ((order=1, steps=CVD), false),
    "th1_hp" => ((order=1, hp_filter=1600, ar=4, steps=[1, 5]), false),
    "th1_bp" => ((order=1, bandpass_filter=(6, 32), ar=4), false),
    "th1_hp128" => ((order=1, hp_filter=100, ar=2, filtered_theoretical_moments_grid=128), false),
    "th1_sub" => ((order=1, ar=2, steps=[1, 3, 8]), true),
    "th1_nodec" => ((order=1, ar=2, nodecomposition=true, steps=[2, 6]), false),
)
const SIM_CASES = Dict(
    "s1" => ((order=1, ar=4), false),
    "s1_hp" => ((order=1, ar=4, hp_filter=1600), false),
    "s1_os" => ((order=1, ar=3, one_sided_hp_filter=1600), false),
    "s1_bp" => ((order=1, ar=3, bandpass_filter=(6, 32)), false),
    "s1_sub" => ((order=1, ar=2), true),
)

struct Case
    name::String
    m::Any
    r::Dict{String,Any}
    dr::Any
end
function Case(name)
    m = load_copy(joinpath(REF, name * ".mod"))
    Case(name, m, refmat(name * ".mat"), JDyn.first_order(m))
end
"Dynare's own decision rules, to test the moment code in isolation."
dynare_dr(c::Case) = Dict{String,Any}("ys" => vec(Float64.(c.r["dr_ys"])), "ghx" => Float64.(c.r["dr_ghx"]),
                                      "ghu" => Float64.(c.r["dr_ghu"]), "order_var" => vec(Int.(c.r["dr_order_var"])),
                                      "state_var" => c.dr.state_var, "order" => 1)

const WORST = Dict{String,Float64}()
"pydyn's close(): same NaN pattern, rtol and atol relative to the largest entry of the reference."
function close(a, b; rtol=1e-9, atol=1e-11, what="")
    a = a isa Number ? fill(Float64(a), 1) : Float64.(a)
    b = b isa Number ? fill(Float64(b), 1) : Float64.(b)
    size(a) != size(b) && length(a) == length(b) && (b = reshape(b, size(a)))
    size(a) == size(b) || (println("$what: shape $(size(a)) vs $(size(b))"); return false)
    isnan.(a) == isnan.(b) || (println("$what: NaN pattern differs"); return false)
    ok = .!isnan.(b)
    any(ok) || return true
    scale = maximum(abs, b[ok])
    d = abs.(a[ok] .- b[ok])
    WORST[what] = max(get(WORST, what, 0.0), maximum(d) / max(scale, eps()))
    pass = all(d .<= atol * scale .+ rtol .* abs.(b[ok]))
    pass || println("$what: max diff $(maximum(d)) (scale $scale)")
    pass
end
stack3(v) = cat(v...; dims=3)

function compare_theoretical(mo, r, c; rtol=1e-9, atol=1e-11)
    kw = TH_CASES[c][1]
    @test mo.var_index == vec(Int.(r["$(c)_ivar"]))
    @test close(mo.mean, r["$(c)_mean"]; rtol, atol, what="mean")
    @test close(mo.var, r["$(c)_var"]; rtol, atol, what="var")
    @test close(mo.variance, diag(r["$(c)_var"]); rtol, atol, what="variance")
    haskey(r, "$(c)_contemporaneous_correlation") &&
        @test close(mo.corr, r["$(c)_contemporaneous_correlation"]; rtol, atol, what="corr")
    @test close(stack3(mo.autocorr), r["$(c)_autocorr"]; rtol, atol, what="autocorr")
    ar = mo.ar
    if haskey(r, "$(c)_variance_decomposition")
        @test close(mo.variance_decomposition, r["$(c)_variance_decomposition"]; rtol, atol, what="variance decomposition")
        @test close(mo.gamma_y[ar+2] .* 100, r["$(c)_variance_decomposition"]; rtol, atol, what="gamma_y vd")
    else
        @test mo.variance_decomposition === nothing
    end
    if haskey(r, "$(c)_variance_decomposition_ME")
        @test close(mo.variance_decomposition_me, r["$(c)_variance_decomposition_ME"]; rtol, atol, what="vd with ME")
    else
        @test mo.variance_decomposition_me === nothing
    end
    for (f, attr) in (("conditional_variance_decomposition", :conditional_variance_decomposition),
                      ("conditional_variance_decomposition_ME", :conditional_variance_decomposition_me))
        if haskey(r, "$(c)_$f")
            @test close(getfield(mo, attr), r["$(c)_$f"]; rtol, atol, what=f)
        else
            @test getfield(mo, attr) === nothing
        end
    end
end

const CASES = Dict(n => Case(n) for n in MODELS)

@testset "theoretical moments against Dynare" begin
    for n in MODELS, c in sort(collect(keys(TH_CASES)))
        @testset "$n-$c" begin
            cs = CASES[n]; kw, sub = TH_CASES[c]
            mo = with_logger(NullLogger()) do
                MO.theoretical_moments(cs.m, cs.dr; var_list=sub ? SUBSET[n] : nothing, kw...)
            end
            @test !haskey(cs.r, "$(c)_error")
            compare_theoretical(mo, cs.r, c)
        end
    end
end

@testset "theoretical moments with Dynare's decision rules" begin
    for (n, c) in [("mom_collard", "th1"), ("mom_aux", "th1"), ("mom_schorfheide", "th1"), ("mom_schorfheide", "th1_hp"),
                   ("mom_unitroot", "th1"), ("mom_unitroot", "th1_bp")]
        @testset "$n-$c" begin
            cs = CASES[n]; kw, sub = TH_CASES[c]
            mo = with_logger(NullLogger()) do
                MO.theoretical_moments(cs.m, dynare_dr(cs); var_list=sub ? SUBSET[n] : nothing, kw...)
            end
            compare_theoretical(mo, cs.r, c; rtol=1e-11, atol=1e-12)
        end
    end
end

@testset "default variable list, options and auxiliary variables" begin
    cs = CASES["mom_aux"]
    @test cs.m.options["stoch_simul"]["order"] == 2
    mo = MO.theoretical_moments(cs.m, cs.dr; order=1, ar=2)
    @test mo.var_names == JDyn.orig_endo_names(cs.m) && length(mo.var_names) == 5 < cs.m.n_endo
    @test length(mo.autocorr) == 2
    cc = CASES["mom_collard"]
    @test cc.m.options["stoch_simul"]["ar"] == 3
    mo = MO.stoch_simul_moments(cc.m; dr=cc.dr, order=1)            # ar=3 from the .mod file
    @test (mo.order, mo.ar, mo.kind) == (1, 3, "theoretical")
    @test close(mo.var, MO.theoretical_moments(cc.m, cc.dr; order=1).var; rtol=1e-14, atol=0.0)
    @test_throws ArgumentError MO.theoretical_moments(cc.m, cc.dr; var_list=["nonexistent"], order=1)
    mo = @test_logs (:warn, r"specified twice") MO.theoretical_moments(cc.m, cc.dr; var_list=["c", "y", "c"], order=1)
    @test mo.var_names == ["c", "y"]
end

@testset "unit root gives NaN for non-stationary variables" begin
    cs = CASES["mom_unitroot"]
    names = cs.m.endo_names
    @test names == ["x", "w", "q", "d", "z", "p", "s"]
    mo = MO.theoretical_moments(cs.m, cs.dr; ar=3, steps=[1, 8])
    nonstat = ["x", "q", "s"]          # x random walk; q = x + w; s loads on x(-1)
    stat = [v for v in names if !(v in nonstat)]
    ist = [findfirst(==(v), names) for v in stat]; ins = [findfirst(==(v), names) for v in nonstat]
    @test mo.stationary == [v in stat for v in names]
    for arr in (mo.mean, mo.std, mo.variance)
        @test all(isnan, arr[ins]) && all(isfinite, arr[ist])
    end
    @test all(isnan, mo.var[ins, :]) && all(isnan, mo.var[:, ins])
    @test all(isfinite, mo.var[ist, ist])
    @test all(isnan, mo.variance_decomposition[ins, :]) && all(isfinite, mo.variance_decomposition[ist, :])
    for a in mo.autocorr
        @test all(isnan, a[ins, :]) && all(isnan, a[:, ins])
    end
    se2, su2, rho = 0.01^2, 0.02^2, 0.8
    w, d, z, p = (findfirst(==(v), names) for v in ("w", "d", "z", "p"))
    var_w = su2 / (1 - rho^2)
    phi1, phi2 = 0.5 + rho, -0.5 * rho                    # z is AR(2) with innovation u
    var_z = su2 * (1 - phi2) / ((1 + phi2) * ((1 - phi2)^2 - phi1^2))
    @test isapprox(mo.variance[[w, d, z, p]], [var_w, se2, var_z, var_w / 0.6^2]; rtol=1e-12)
    @test all(abs.(mo.mean[ist]) .<= 1e-15)
    @test isapprox([mo.autocorr[k][w, w] for k in 1:3], rho .^ (1:3); rtol=1e-12)
    r1 = phi1 / (1 - phi2); r2 = phi1 * r1 + phi2; r3 = phi1 * r2 + phi2 * r1
    @test isapprox([mo.autocorr[k][z, z] for k in 1:3], [r1, r2, r3]; rtol=1e-12)
    @test abs(mo.corr[w, p] - 1) < 1e-12 && abs(mo.corr[w, d]) < 1e-12
    vd = mo.variance_decomposition                       # shocks e, u, v (v has zero variance)
    @test all(abs.(vd[[w, d, z, p], :] .- [0 100 0; 100 0 0; 0 100 0; 0 100 0]) .<= 1e-10)
    cv = mo.conditional_variance_decomposition
    @test all(isfinite, cv)
    @test all(abs.(cv[findfirst(==("x"), names), :, :] .- [1 0 0; 1 0 0]) .<= 1e-14)
    @test isapprox(cv[findfirst(==("q"), names), 1, :], [se2 / (se2 + su2), su2 / (se2 + su2), 0]; rtol=1e-12)
    hp = MO.theoretical_moments(cs.m, cs.dr; order=1, hp_filter=1600, ar=2)
    @test all(hp.stationary) && all(isfinite, hp.var) && all(isfinite, hp.mean)
end

@testset "simulated moments against Dynare" begin
    for n in MODELS, c in sort(collect(keys(SIM_CASES)))
        @testset "$n-$c" begin
            cs = CASES[n]; kw, sub = SIM_CASES[c]
            r = cs.r
            key = r["$(c)_simkey"]
            endo, exo = permutedims(Float64.(r["sim_$(key)_endo"])), Float64.(r["sim_$(key)_exo"])
            ivar = vec(Int.(r["$(c)_ivar"]))
            hp = haskey(kw, :hp_filter)
            tol = hp ? (rtol=1e-8, atol=1e-9) : (rtol=1e-9, atol=1e-11)
            mo = MO.simulated_moments(endo; ivar=ivar, model=cs.m, drop=30, shocks=exo, dr=cs.dr, kw...)
            @test mo.var_names == [cs.m.endo_names[i] for i in ivar]
            for f in (:mean, :var, :skewness, :kurtosis)
                @test close(getfield(mo, f), r["$(c)_$f"]; tol..., what="sim $f")
            end
            @test close(mo.corr, r["$(c)_contemporaneous_correlation"]; tol..., what="sim corr")
            @test close(stack3(mo.autocorr), r["$(c)_autocorr"]; tol..., what="sim autocorr")
            @test close(mo.variance_decomposition, r["$(c)_variance_decomposition"]; tol..., what="sim vd")
            @test close(mo.std .^ 2, diag(r["$(c)_var"]); tol..., what="sim std")
        end
    end
end

@testset "stoch_simul_moments reproduces Dynare's simulation ($n-$c)" for (n, c) in [("mom_collard", "s1"), ("mom_schorfheide", "s1_hp")]
    cs = CASES[n]; kw, _ = SIM_CASES[c]
    r = cs.r
    key = r["$(c)_simkey"]
    mo = MO.stoch_simul_moments(cs.m; dr=cs.dr, shocks=Float64.(r["sim_$(key)_exo"]), periods=160, drop=30, kw...)
    @test mo.kind == "simulated" && mo.var_names == JDyn.orig_endo_names(cs.m)
    tol = haskey(kw, :hp_filter) ? (rtol=1e-8, atol=1e-9) : (rtol=1e-9, atol=1e-11)
    @test close(mo.var, r["$(c)_var"]; tol..., what="stoch_simul var")
    @test close(mo.variance_decomposition, r["$(c)_variance_decomposition"]; tol..., what="stoch_simul vd")
    a = MO.stoch_simul_moments(cs.m; dr=cs.dr, order=1, periods=300, drop=50, rng=MersenneTwister(1))
    b = MO.stoch_simul_moments(cs.m; dr=cs.dr, order=1, periods=300, drop=50, rng=MersenneTwister(1))
    @test a.var == b.var
    @test_throws ArgumentError MO.stoch_simul_moments(cs.m; dr=cs.dr, order=1, periods=40, drop=50)
end

# ------------------------------------------------------------------------------------ building blocks
"Reference solution of x = A x A' + Q by vectorisation."
dlyap_ref(A, Q) = reshape((I - kron(A, A)) \ vec(Q), size(A))

@testset "lyapunov_symm (independent checks) and unit-root projection" begin
    rng = MersenneTwister(3)
    n = 9
    A = randn(rng, n, n)
    A .*= 0.9 / maximum(abs, eigvals(A))               # stable, complex eigenvalues
    Bm = randn(rng, n, n); Q = Bm * Bm'
    X, u = MO.lyapunov_symm(A, Q)
    @test size(u) == (n, 0)
    @test isapprox(X, dlyap_ref(A, Q); rtol=1e-11)
    @test isapprox(X, A * X * A' + Q; rtol=1e-12)
    A2 = Diagonal([0.5, -0.3, 0.9]) |> Matrix
    @test isapprox(MO.lyapunov_symm(A2, Matrix(1.0I, 3, 3))[1], Diagonal(1 ./ (1 .- diag(A2) .^ 2)))
    @test isapprox(MO.lyapunov_symm(fill(0.5, 1, 1), fill(2.0, 1, 1))[1], fill(2 / 0.75, 1, 1))
    S = randn(rng, 4, 4)
    blk = zeros(4, 4); blk[1, 1] = 1.0; blk[2:3, 2:3] = [0.5 0.3; -0.3 0.5]; blk[4, 4] = 0.4
    Au = S * blk / S
    Xu, uu = MO.lyapunov_symm(Au, Matrix(1.0I, 4, 4))
    @test size(uu) == (4, 1)
    P = I - uu * uu'
    @test maximum(abs, Xu - P * Xu * P) < 1e-12
    @test maximum(abs, Xu - (P * Au * Xu * Au' * P + P * P)) < 1e-10
    @test size(MO.lyapunov_symm(Matrix(Diagonal([1.2, 0.5])), Matrix(1.0I, 2, 2))[2]) == (2, 1)
end

@testset "filters against direct formulas" begin
    rng = MersenneTwister(5)
    y = cumsum(randn(rng, 60, 2); dims=1)
    T, lam = size(y, 1), 1600.0
    Dm = diff(diff(Matrix(1.0I, T, T); dims=1); dims=1)
    trend_ref = (I + lam * Dm' * Dm) \ y
    tr, cy = MO.hp_filter(y, lam)
    @test isapprox(tr, trend_ref; rtol=1e-9)
    @test all(abs.(cy .- (y - trend_ref)) .<= 1e-9 .+ 1e-8 .* abs.(y - trend_ref))
    t = collect(0.0:79.0)
    out = MO.baxter_king_filter(hcat(ones(80), t, sin.(2pi .* t ./ 12)), 6, 32, 12)
    @test size(out) == (80 - 24, 3)
    @test maximum(abs, out[:, 1:2]) < 1e-12
    @test maximum(abs, out[:, 3]) > 0.9
    t1, _ = MO.one_sided_hp_filter(y, 1600.0)
    y_alt = copy(y); y_alt[41:end, :] .+= 10.0
    t2, _ = MO.one_sided_hp_filter(y_alt, 1600.0)
    @test isapprox(t1[1:40, :], t2[1:40, :]; rtol=1e-12)
end

@testset "variance decomposition with correlated shocks" begin
    cs = CASES["mom_collard"]; m = cs.m
    sol = JDyn.first_order(m)
    mo = MO.theoretical_moments(m, sol; order=1, ar=1)
    @test isapprox(vec(sum(mo.variance_decomposition; dims=2)), fill(100.0, size(mo.variance_decomposition, 1)); rtol=1e-12)
    L = Matrix(cholesky(Symmetric(m.Sigma_e)).L)
    @test abs(m.Sigma_e[1, 2]) > 0
    contrib = zeros(m.n_endo, 2)
    for i in 1:2
        v = zeros(m.n_endo); Gk = Matrix(1.0I, m.n_endo, m.n_endo)
        for _ in 1:4000
            f = Gk * sol.H * L[:, i]
            v .+= f .* f
            Gk = sol.G * Gk
        end
        contrib[:, i] .= v
    end
    ref = 100 .* contrib ./ sum(contrib; dims=2)
    @test all(abs.(mo.variance_decomposition .- ref[mo.var_index, :]) .<= 1e-8 .+ 1e-8 .* abs.(ref[mo.var_index, :]))
    P = [0.0 1.0; 1.0 0.0]
    dr_sw = JDyn.as_dr(sol)
    dr_sw["ghu"] = JDyn.ghu(sol) * P
    sw = MO.theoretical_moments(m, dr_sw; Sigma_e=P * m.Sigma_e * P, order=1, ar=1)
    @test isapprox(sw.var, mo.var; rtol=1e-12)
    @test maximum(abs, sw.variance_decomposition[:, end:-1:1] .- mo.variance_decomposition) > 0.1
end

@testset "conditional variance decomposition with measurement error" begin
    cs = CASES["mom_aux"]
    @test cs.m.varobs == ["c", "z", "k"]
    @test cs.m.measurement_errors["c"] ≈ 0.0025 && cs.m.measurement_errors["z"] ≈ 9e-6 && length(cs.m.measurement_errors) == 2
    cvd, cvd_me = MO.conditional_variance_decomposition(cs.m, cs.dr, [1, 3]; var_list=["z", "c", "r"])
    @test isapprox(dropdims(sum(cvd; dims=3); dims=3), ones(3, 2); rtol=1e-12)
    @test size(cvd_me) == (2, 2, 3)                        # observed among the requested: z, c (in this order)
    @test isapprox(dropdims(sum(cvd_me; dims=3); dims=3), ones(2, 2); rtol=1e-12)
    ratio = cvd_me[:, :, 1:2] ./ cvd[1:2, :, :]
    @test isapprox(ratio[:, :, 1], ratio[:, :, 2]; rtol=1e-10)
    @test all(ratio[:, :, 1] .< 1) && all(cvd_me[:, :, 3] .> 0)
    @test_throws ArgumentError MO.conditional_variance_decomposition(cs.m, cs.dr, [0, 1]; var_list=["z"])
    @test_throws ArgumentError MO.conditional_variance_decomposition(cs.m, cs.dr, [4, 2]; var_list=["z"])
    @test MO.conditional_variance_decomposition(cs.m, cs.dr, [1]; var_list=["z"], H=zeros(3))[2] === nothing
end

@testset "unsupported requests raise" begin
    cs = CASES["mom_collard"]
    @test_throws ArgumentError MO.theoretical_moments(cs.m, cs.dr; order=1, one_sided_hp_filter=1600)
    @test_throws ArgumentError MO.theoretical_moments(cs.m, cs.dr; order=1, hp_filter=1600, bandpass_filter=(6, 32))
    @test_throws ArgumentError MO.theoretical_moments(cs.m, cs.dr; order=3)
    @test_throws ArgumentError MO.theoretical_moments(cs.m, cs.dr; order=1, no_such_option=1)
end

println("largest relative differences (|diff| / max |reference|) by quantity:")
for k in sort(collect(keys(WORST)))
    @printf("  %-40s %.2e\n", k, WORST[k])
end
