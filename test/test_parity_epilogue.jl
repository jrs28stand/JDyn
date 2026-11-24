# epilogue block against Dynare 7.2 (pydyn's reference epilogue/epi_example.mat: generated epilogue_dynamic.m / epilogue_static.m)
using Test
isdefined(@__MODULE__, :REF) || include("common.jl")

@testset "epilogue block against Dynare" begin
    d = joinpath(@__DIR__, "ref_parity", "epilogue")
    m = load_copy(joinpath(d, "epi_example.mod"))
    r = matread(joinpath(d, "epi_example.mat"))
    ep = JDyn.epilogue(m)
    @test epilogue_names(ep) == split(String(r["names"]), ",")
    endo_in, exo_in = Matrix{Float64}(r["endo_in"]), Matrix{Float64}(r["exo_in"])
    S = Dict{String,Vector{Float64}}()
    for (i, v) in enumerate(m.endo_names); S[v] = endo_in[:, i]; end
    for (i, v) in enumerate(m.exo_names); S[v] = exo_in[:, i]; end
    x = fill(NaN, size(endo_in, 1)); x[1:7] = vec(Float64.(r["x_in"])); S["x"] = x
    prm = vec(Float64.(r["params"]))
    for rounding in (true, false)
        out = epilogue_dynamic(ep, S; params=prm, replicate_dynare_bugs=rounding)
        for n in epilogue_names(ep)
            a, b = out[n], vec(Float64.(r["dyn_" * n]))
            @test isnan.(a) == isnan.(b)
            ok = .!isnan.(b)
            any(ok) || continue
            rounding ? (@test a[ok] == b[ok]) : (@test maximum(abs.(a[ok] .- b[ok])) < 1e-6)
        end
        @test all(isnan, out["z"])
        @test all(isnan, out["v"])
    end
    ys = vec(Float64.(r["steady_state"]))
    ss = Dict{String,Vector{Float64}}(v => fill(ys[i], 3) for (i, v) in enumerate(m.endo_names))
    for v in m.exo_names; ss[v] = zeros(3); end
    ss["x"] = [1.0, 2, 3]; ss["z"] = [0.1, 0.2, 0.3]
    st = epilogue_static(ep, ss; params=prm)
    for n in epilogue_names(ep)
        @test st[n] ≈ vec(Float64.(r["sta_" * n])) rtol=1e-14 atol=1e-14
    end
    @test JDyn.parse_epilogue("var y; model; y=1; end;\nepilogue; // comment\n a = y(-1)/2; /* block */ b = a + 1;\nend;\n") == [("a", "y(-1)/2"), ("b", "a + 1")]
end
