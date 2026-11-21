# Theoretical moments at order 2 (plain: first-order covariances plus the ghs2 mean correction; pruned state
# space) and simulated moments at order 2 against Dynare 7.2 (order-2 parts of pydyn tests/test_moments.py;
# references reference/mom_*.mat from reference/r_moments.m). Needs the k-order library (deps/korder/build.sh).
using Test, Random
isdefined(@__MODULE__, :REF) || include("common.jl")

const MO = JDyn
if !JDyn.korder_available()
    @warn "libjdyn_korder not built: order-2 moment tests skipped"
else
const MODELS2 = ["mom_collard", "mom_aux", "mom_schorfheide"]          # mom_unitroot is not run at order 2 (as pydyn)
const SUBSET = Dict("mom_collard" => ["c", "y", "a"], "mom_unitroot" => ["z", "d", "q", "w"],
                    "mom_aux" => ["z", "c", "AUX_ENDO_LAG_2_1", "AUX_EXO_LAG_6_1"],
                    "mom_schorfheide" => ["y", "R", "gy_obs", "AUX_ENDO_LEAD_65"])
const TH2 = Dict(
    "th2" => ((order=2, pruning=false, ar=3, steps=[1, 4]), false),
    "th2p" => ((order=2, pruning=true, ar=3, steps=[1, 4, 10]), false),
    "th2p_sub" => ((order=2, pruning=true, ar=2, steps=[2, 5]), true),
    "th2p_nodec" => ((order=2, pruning=true, ar=2, nodecomposition=true, steps=[2, 5]), false),
    "th2_sub" => ((order=2, pruning=false, ar=2), true),
)
const SIM2 = Dict(
    "s2p" => ((order=2, pruning=true, ar=4), false),
    "s2p_hp" => ((order=2, pruning=true, ar=3, hp_filter=1600), false),
)
const WORST2 = Dict{String,Float64}()
"pydyn's close(): same NaN pattern, rtol and atol relative to the largest entry of the reference."
function close2(a, b; rtol=1e-9, atol=1e-11, what="")
    a = a isa Number ? fill(Float64(a), 1) : Float64.(a)
    b = b isa Number ? fill(Float64(b), 1) : Float64.(b)
    size(a) != size(b) && length(a) == length(b) && (b = reshape(b, size(a)))
    size(a) == size(b) || (println("$what: shape $(size(a)) vs $(size(b))"); return false)
    isnan.(a) == isnan.(b) || (println("$what: NaN pattern differs"); return false)
    ok = .!isnan.(b)
    any(ok) || return true
    scale = maximum(abs, b[ok])
    d = abs.(a[ok] .- b[ok])
    WORST2[what] = max(get(WORST2, what, 0.0), maximum(d) / max(scale, eps()))
    pass = all(d .<= atol * scale .+ rtol .* abs.(b[ok]))
    pass || println("$what: max diff $(maximum(d)) (scale $scale)")
    pass
end
stack3(v) = cat(v...; dims=3)

struct Case2
    name::String
    m::Any
    r::Dict{String,Any}
    dr::Dict{Int,Any}
end
function Case2(name)
    m = load_copy(joinpath(REF, name * ".mod"))
    Case2(name, m, refmat(name * ".mat"), Dict{Int,Any}())
end
function drule(c::Case2, order)
    get!(c.dr, order) do
        order == 1 ? JDyn.as_dr(JDyn.first_order(c.m)) : JDyn.k_order(c.m; order=order)
    end
end
"Dynare's own decision rules, to test the moment code in isolation."
function dynare_dr(c::Case2, order)
    base = drule(c, 1)
    d = Dict{String,Any}("ys" => vec(Float64.(c.r["dr_ys"])), "ghx" => Float64.(c.r["dr_ghx"]), "ghu" => Float64.(c.r["dr_ghu"]),
                         "order_var" => vec(Int.(c.r["dr_order_var"])), "state_var" => base["state_var"], "order" => order)
    if order == 2
        d["ghxx"] = Float64.(c.r["dr_ghxx"]); d["ghxu"] = Float64.(c.r["dr_ghxu"]); d["ghuu"] = Float64.(c.r["dr_ghuu"])
        d["ghs2"] = vec(Float64.(c.r["dr_ghs2"]))
    end
    d
end

function compare_theoretical2(mo, r, c, kw; rtol=1e-9, atol=1e-11)
    @test mo.var_index == vec(Int.(r["$(c)_ivar"]))
    @test close2(mo.mean, r["$(c)_mean"]; rtol, atol, what="mean")
    @test close2(mo.var, r["$(c)_var"]; rtol, atol, what="var")
    @test close2(mo.variance, diag(r["$(c)_var"]); rtol, atol, what="variance")
    haskey(r, "$(c)_contemporaneous_correlation") &&
        @test close2(mo.corr, r["$(c)_contemporaneous_correlation"]; rtol, atol, what="corr")
    @test close2(stack3(mo.autocorr), r["$(c)_autocorr"]; rtol, atol, what="autocorr")
    ar = mo.ar
    if haskey(r, "$(c)_variance_decomposition")
        @test close2(mo.variance_decomposition, r["$(c)_variance_decomposition"]; rtol, atol, what="variance decomposition")
        @test close2(mo.gamma_y[ar+2] .* 100, r["$(c)_variance_decomposition"]; rtol, atol, what="gamma_y vd")
    else
        @test mo.variance_decomposition === nothing
    end
    haskey(r, "$(c)_variance_decomposition_ME") ?
        @test(close2(mo.variance_decomposition_me, r["$(c)_variance_decomposition_ME"]; rtol, atol, what="vd with ME")) :
        @test(mo.variance_decomposition_me === nothing)
    for (f, attr) in (("conditional_variance_decomposition", :conditional_variance_decomposition),
                      ("conditional_variance_decomposition_ME", :conditional_variance_decomposition_me))
        haskey(r, "$(c)_$f") ? @test(close2(getfield(mo, attr), r["$(c)_$f"]; rtol, atol, what=f)) :
                               @test(getfield(mo, attr) === nothing)
    end
    kw.order == 2 && !kw.pruning &&
        @test close2(mo.mean_correction, r["$(c)_gy$(ar + 3)"]; rtol, atol, what="mean correction")
end

const CASES2 = Dict(n => Case2(n) for n in MODELS2)

@testset "order-2 theoretical moments against Dynare (own decision rules)" begin
    for n in MODELS2, c in sort(collect(keys(TH2)))
        @testset "$n-$c" begin
            cs = CASES2[n]; kw, sub = TH2[c]
            mo = with_logger(NullLogger()) do
                MO.theoretical_moments(cs.m, drule(cs, 2); var_list=sub ? SUBSET[n] : nothing, kw...)
            end
            @test !haskey(cs.r, "$(c)_error")
            @test mo.kind == (kw.pruning ? "pruned" : "theoretical") && mo.order == 2
            compare_theoretical2(mo, cs.r, c, kw)
        end
    end
end

@testset "order-2 theoretical moments with Dynare's decision rules" begin
    for (n, c) in [("mom_collard", "th2"), ("mom_collard", "th2p"), ("mom_aux", "th2p"), ("mom_schorfheide", "th2p")]
        @testset "$n-$c" begin
            cs = CASES2[n]; kw, sub = TH2[c]
            mo = MO.theoretical_moments(cs.m, dynare_dr(cs, 2); var_list=sub ? SUBSET[n] : nothing, kw...)
            compare_theoretical2(mo, cs.r, c, kw; rtol=1e-11, atol=1e-12)
        end
    end
end

@testset "simulated moments at order 2 (Dynare's path)" begin
    for n in MODELS2, c in sort(collect(keys(SIM2)))
        @testset "$n-$c" begin
            cs = CASES2[n]; kw, _ = SIM2[c]
            r = cs.r
            key = r["$(c)_simkey"]
            endo, exo = permutedims(Float64.(r["sim_$(key)_endo"])), Float64.(r["sim_$(key)_exo"])
            ivar = vec(Int.(r["$(c)_ivar"]))
            tol = haskey(kw, :hp_filter) ? (rtol=1e-8, atol=1e-9) : (rtol=1e-9, atol=1e-11)
            mo = MO.simulated_moments(endo; ivar=ivar, model=cs.m, drop=30, shocks=exo, dr=drule(cs, 2), kw...)
            for f in (:mean, :var, :skewness, :kurtosis)
                @test close2(getfield(mo, f), r["$(c)_$f"]; tol..., what="sim $f")
            end
            @test close2(mo.corr, r["$(c)_contemporaneous_correlation"]; tol..., what="sim corr")
            @test close2(stack3(mo.autocorr), r["$(c)_autocorr"]; tol..., what="sim autocorr")
            @test close2(mo.variance_decomposition, r["$(c)_variance_decomposition"]; tol..., what="sim vd")
            # the whole chain: our own simulation of Dynare's shocks reproduces Dynare's path
            y = simulate(drule(cs, 2), exo; order=2, pruning=true)
            @test close2(y, endo[1:size(y, 1), :]; rtol=1e-9, atol=1e-11, what="sim path")
        end
    end
    cs = CASES2["mom_aux"]; kw, _ = SIM2["s2p"]; r = cs.r
    key = r["s2p_simkey"]
    mo = MO.stoch_simul_moments(cs.m; dr=drule(cs, 2), shocks=Float64.(r["sim_$(key)_exo"]), periods=160, drop=30, kw...)
    @test mo.kind == "simulated" && mo.order == 2 && mo.var_names == JDyn.orig_endo_names(cs.m)
    @test close2(mo.var, r["s2p_var"]; rtol=1e-9, atol=1e-11, what="stoch_simul var")
    @test close2(mo.variance_decomposition, r["s2p_variance_decomposition"]; rtol=1e-9, atol=1e-11, what="stoch_simul vd")
end

@testset "stoch_simul_moments at the .mod file's order" begin
    cc = CASES2["mom_collard"]
    @test cc.m.options["stoch_simul"]["ar"] == 3
    mo = MO.stoch_simul_moments(cc.m; dr=drule(cc, 2))               # order 2 and ar=3 from the .mod file
    @test (mo.order, mo.ar, mo.kind) == (2, 3, "theoretical")
    @test close2(mo.mean, cc.r["th2_mean"]; what="mean")
    @test close2(mo.var, cc.r["th2_var"]; what="var")
    mo2 = MO.stoch_simul_moments(cc.m)                                  # solved by JDyn with the k-order library
    @test mo2.order == 2 && isapprox(mo2.var, mo.var; rtol=1e-9)
end

@testset "pruned second order: closed form" begin
    # x is AR(1), y = x + x^2/2 (static): E y = s2/2, Var y = s2 + s2^2/2, Cov(y_t, y_t-k) = rho^k s2 + rho^2k s2^2/2
    m = load_copy(joinpath(REF, "mom_quad.mod"))
    rho, s2 = 0.8, 0.1^2 / (1 - 0.8^2)
    dr = JDyn.k_order(m; order=2)
    pr = MO.theoretical_moments(m, dr; order=2, pruning=true, ar=3)
    var_y = s2 + 0.5 * s2^2
    acov = [rho^k * s2 + 0.5 * rho^(2k) * s2^2 for k in 1:3]
    @test isapprox(pr.mean, [0.0, 0.5 * s2]; atol=1e-15)
    @test isapprox(pr.variance, [s2, var_y]; rtol=1e-12)
    @test isapprox([a[2, 2] for a in pr.autocorr], acov ./ var_y; rtol=1e-11)
    @test isapprox([a[1, 1] for a in pr.autocorr], rho .^ (1:3); rtol=1e-11)
    @test isapprox(pr.corr[1, 2], s2 / sqrt(s2 * var_y); rtol=1e-11)
    un = MO.theoretical_moments(m, dr; order=2, pruning=false, ar=1)
    @test isapprox(un.mean, pr.mean; atol=1e-15)
    @test isapprox(un.variance, [s2, s2]; rtol=1e-12)
    @test isapprox(un.mean_correction, [0.0, 0.5 * s2]; atol=1e-15)
end

@testset "unsupported order-2 requests" begin
    cs = CASES2["mom_collard"]
    dr = dynare_dr(cs, 2)
    dr_ur = copy(dr)                            # a unit root among the states: first state loads on itself
    g = zeros(size(dr["ghx"])); g[JDyn.Rules(cs.m, dr).index_states[1], 1] = 1.0; dr_ur["ghx"] = g
    @test_throws ErrorException MO.theoretical_moments(cs.m, dr_ur; order=2, pruning=true)
    @test_throws ArgumentError MO.theoretical_moments(cs.m, dr; order=2, hp_filter=1600)
    @test_throws ArgumentError MO.theoretical_moments(cs.m, dr; order=2, pruning=true, hp_filter=1600)
    @test_throws ArgumentError MO.theoretical_moments(cs.m, dr; order=3)
end

println("order 2 moments: largest relative differences (|diff| / max |reference|):")
for k in sort(collect(keys(WORST2)))
    @printf("  %-40s %.2e\n", k, WORST2[k])
end
end   # korder_available
