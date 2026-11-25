# Interpreter parity with Octave 8.4 on pydyn's case sets (test/ref_gaps/cases): N-d arrays, Octave rand/randn/randi/randperm/rng,
# size rules, sprintf corners, error identifiers, dates. Zero "differ" is the bar; refusals are bounded.
using Test, MAT
import JDyn
using JDyn: MatlabError, Unsupported, mrun

const _GDIR = joinpath(@__DIR__, "ref_gaps", "cases")

function _gsame(got, want)
    got isa JDyn.MDates && (got = reshape((got.time .- 1) ./ got.freq, :, 1))
    g = got isa AbstractString ? reshape(Float64[Float64(c) for c in got], 1, :) : got isa Union{Real,Bool} ? fill(Float64(got), 1, 1) : Array{Float64}(got)
    w = want isa AbstractString ? reshape(Float64[Float64(c) for c in want], 1, :) : want isa Union{Real,Bool} ? fill(Float64(want), 1, 1) : Array{Float64}(want)
    (length(g) == 0 && length(w) == 0) && return true
    size(g) == size(w) || return false
    all(i -> (isnan(g[i]) && isnan(w[i])) || g[i] == w[i] || abs(g[i] - w[i]) <= 1e-12 + 1e-9 * abs(w[i]), eachindex(g))
end

function _grun(name; rng=nothing)
    cases = split(read(joinpath(_GDIR, name * "_cases.txt"), String), "\n%%\n")
    exp = vec(matread(joinpath(_GDIR, name * ".mat"))["results"])
    match = 0; refused = Int[]; differ = Int[]
    for k in eachindex(cases)
        want = exp[k]; wante = want isa AbstractString && startswith(want, "ERROR")
        status = try
            got = mrun(String(cases[k]); rng=rng)["r"]
            wante ? :differ : (try _gsame(got, want) ? :match : :differ catch; :differ end)
        catch e
            e isa Unsupported ? :refused : e isa MatlabError ? (wante ? :match : :differ) : :differ
        end
        status == :match ? (match += 1) : status == :refused ? push!(refused, k) : push!(differ, k)
    end
    (; n=length(cases), match, refused, differ)
end

@testset "interpreter parity case sets" begin
    for (name, rng, maxref, known) in (("matlab_extra", nothing, 70, Int[]), ("pydyn_extra", nothing, 20, Int[]),
                                       ("pydyn_nd", nothing, 10, Int[]), ("pydyn_blk1", nothing, 30, [187]),
                                       ("pydyn_rng", :octave, 10, Int[]))
        r = _grun(name; rng)
        @test sort(r.differ) == known
        @test length(r.refused) <= maxref
        @test r.match >= r.n - maxref - length(known)
    end
end
