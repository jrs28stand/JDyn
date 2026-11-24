# Third-order pruned theoretical moments against Dynare 7.2 (pydyn's references test/ref_parity/mom3_*.mat, made by r_moments3.m)
using Test
isdefined(@__MODULE__, :REF) || include("common.jl")

if !JDyn.korder_available()
    @warn "libjdyn_korder not built: order-3 moment tests skipped"
else
const R3 = joinpath(@__DIR__, "ref_parity")
const MODELS3 = ["mom3_collard", "mom3_aux", "mom3_schorfheide"]
const SUBSET3 = Dict("mom3_collard" => ["c", "y", "a"], "mom3_aux" => ["z", "c", "AUX_ENDO_LAG_2_1", "AUX_EXO_LAG_6_1"],
                     "mom3_schorfheide" => ["y", "R", "gy_obs", "AUX_ENDO_LEAD_65"])
const CASES3 = Dict("th3p" => ((order=3, pruning=true, ar=3), false), "th3p_sub" => ((order=3, pruning=true, ar=2, steps=[2, 5]), true))

function close3(a, b; rtol=1e-9, atol=1e-11)
    a = a isa Number ? fill(Float64(a), 1) : Float64.(a)
    b = b isa Number ? fill(Float64(b), 1) : Float64.(b)
    size(a) != size(b) && length(a) == length(b) && (b = reshape(b, size(a)))
    size(a) == size(b) || return (Inf, false)
    isnan.(a) == isnan.(b) || return (Inf, false)
    ok = .!isnan.(b)
    any(ok) || return (0.0, true)
    scale = maximum(abs, b[ok])
    d = abs.(a[ok] .- b[ok])
    (maximum(d) / max(scale, eps()), all(d .<= atol * scale .+ rtol .* abs.(b[ok])))
end

worst3 = Ref(0.0)
@testset "order-3 pruned theoretical moments against Dynare" begin
    for n in MODELS3
        m = load_copy(joinpath(R3, n * ".mod"))
        r = matread(joinpath(R3, n * ".mat"))
        dr_own = JDyn.k_order(m; order=3)
        # Dynare's own decision rules (the moment code in isolation) and JDyn's
        dr_dyn = Dict{String,Any}(k[4:end] => r[k] for k in keys(r) if startswith(k, "dr_"))
        for k in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss")
            dr_dyn[k] = Matrix{Float64}(reshape(Float64.(dr_dyn[k]), size(dr_dyn[k], 1), :))
        end
        dr_dyn["ghs2"] = vec(Float64.(dr_dyn["ghs2"]))
        dr_dyn["ys"] = vec(Float64.(dr_dyn["ys"]))
        dr_dyn["order_var"] = vec(Int.(dr_dyn["order_var"]))
        dr_dyn["state_var"] = dr_own["state_var"]
        for (label, dr) in (("jdyn dr", dr_own), ("dynare dr", dr_dyn)), (c, (kw, sub)) in CASES3
            mo = with_logger(NullLogger()) do
                JDyn.theoretical_moments(m, dr; var_list=sub ? SUBSET3[n] : nothing, kw...)
            end
            tag = "$n $label $c"
            @test mo.kind == "pruned" && mo.order == 3
            @test mo.var_index == vec(Int.(r["$(c)_ivar"]))
            for (what, a, b) in (("mean", mo.mean, r["$(c)_mean"]), ("var", mo.var, r["$(c)_var"]),
                                 ("corr", mo.corr, r["$(c)_contemporaneous_correlation"]),
                                 ("autocorr", cat(mo.autocorr...; dims=3), r["$(c)_autocorr"]))
                dd, ok = close3(a, b; rtol=label == "dynare dr" ? 1e-9 : 1e-7, atol=label == "dynare dr" ? 1e-11 : 1e-9)
                label == "dynare dr" && (worst3[] = max(worst3[], dd))
                ok || println("$tag $what: max relative-to-scale diff $dd")
                @test ok
            end
            @test mo.variance_decomposition === nothing && mo.conditional_variance_decomposition === nothing
            @test !haskey(r, "$(c)_variance_decomposition")
        end
    end
    println("order-3 moments with Dynare's decision rules: worst scaled difference $(worst3[])")
end

@testset "order-3 moments refusals" begin
    m = load_copy(joinpath(R3, "mom3_collard.mod"))
    dr = JDyn.k_order(m; order=3)
    @test_throws ArgumentError JDyn.theoretical_moments(m, dr; order=3, pruning=false)         # Dynare computes nothing here
    @test_throws ArgumentError JDyn.theoretical_moments(m, dr; order=3, pruning=true, hp_filter=1600)
    @test_throws ArgumentError JDyn.theoretical_moments(m, dr; order=4, pruning=true)
end

@testset "stoch_simul order 3 pruning fills oo_ moments" begin
    d = mktempdir()
    cp(joinpath(R3, "mom3_collard.mod"), joinpath(d, "mom3_collard.mod"))
    txt = replace(read(joinpath(d, "mom3_collard.mod"), String), "stoch_simul(order=3, nograph, ar=3, irf=0);" => "stoch_simul(order=3, pruning, nograph, ar=3, irf=0);")
    write(joinpath(d, "mom3_collard.mod"), txt)
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "mom3_collard.mod"))
    end
    r = matread(joinpath(R3, "mom3_collard.mat"))
    @test close3(ws.oo_.mean, r["th3p_mean"])[2]
    @test close3(ws.oo_.var, r["th3p_var"])[2]
end
end
