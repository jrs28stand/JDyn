# Parity batch 2, results layer: pac.estimate.nls / iterative_ols, shock_decomposition with forecasts and the loglinear smoother
# auxiliaries, run through dynare() and compared with Dynare 7.2 under Octave 8.4 (cases and references of pydyn's reference/oo
# store, copied to test/ref_parity2/oo/). Only the fields of the feature are compared (oo_ paths below), at the case's tolerance.
using Test; include("common.jl")
using Random

const OOP2 = joinpath(@__DIR__, "ref_parity2", "oo")

"Largest relative difference (floor 1) between a reference value and ours, with the path of the worst field; `nothing` where a field is absent."
function oo_diff(ref, ours, path, out::Vector{Tuple{String,Float64}})
    if ref isa AbstractDict
        ours isa JDyn.MStruct || (push!(out, (path * " (not a struct)", Inf)); return)
        for (k, v) in ref
            haskey(ours, k) || (isempty(v) || push!(out, (path * "." * k * " (missing)", Inf)); continue)
            oo_diff(v, ours[k], path * "." * k, out)
        end
    elseif ours isa JDyn.MDseries
        rd = ref isa AbstractDict ? ref["data"] : ref
        oo_diff(rd, ours.data, path * ".data", out)
    elseif ref isa AbstractString || (ref isa AbstractArray && eltype(ref) <: AbstractString)
        # names (parnames cell): compare the strings
        a = ref isa AbstractString ? [ref] : vec(String.(ref))
        b = ours isa JDyn.Cell ? String.(vec(ours.items)) : ours isa AbstractString ? [ours] : String[]
        a == b || push!(out, (path * " (strings)", Inf))
    elseif ref isa AbstractArray && eltype(ref) <: AbstractArray   # a cell of arrays
        ours isa JDyn.Cell || (push!(out, (path * " (not a cell)", Inf)); return)
        for (i, (r, o)) in enumerate(zip(vec(ref), vec(ours.items)))
            oo_diff(r, o, "$path{$i}", out)
        end
    else
        a = ref isa Real ? [Float64(ref)] : vec(Float64.(ref)); b = try
            ours isa Real ? [Float64(ours)] : vec(Float64.(ours))
        catch
            push!(out, (path * " (not numeric)", Inf)); return
        end
        length(a) == length(b) || (push!(out, (path * " (size $(length(a)) vs $(length(b)))", Inf)); return)
        isempty(a) && return
        both = .!(isnan.(a) .& isnan.(b))
        push!(out, (path, any(isnan.(a) .!= isnan.(b)) ? Inf : (any(both) ? maximum(abs.(a[both] .- b[both])) / max(1.0, maximum(abs.(a[both]))) : 0.0)))
    end
    nothing
end

function run_case(case; random=:octave)
    d = mktempdir()
    for f in readdir(OOP2)
        endswith(f, ".mod") || continue
    end
    cp(joinpath(OOP2, case * ".mod"), joinpath(d, case * ".mod"))
    for extra in ("fsdat_simul.m", "fsdat_simul.mat")
        isfile(joinpath(OOP2, extra)) && cp(joinpath(OOP2, extra), joinpath(d, extra))
    end
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, case * ".mod"); random=random)
    end
    ws, matread(joinpath(OOP2, case * ".mat"))
end

function check_case(case, paths, tol; skip=String[])
    ws, ref = run_case(case)
    out = Tuple{String,Float64}[]
    roo = ref["oo_"]
    for p in paths
        parts = split(p, ".")
        r = roo; o = ws.oo_
        for q in parts
            haskey(r, q) || (push!(out, ("oo_." * p * " (not in the reference)", Inf)); r = nothing; break)
            r = r[q]
            o = o isa JDyn.MStruct && haskey(o, q) ? o[q] : nothing
            o === nothing && (push!(out, ("oo_." * p * " (missing)", Inf)); r = nothing; break)
        end
        r === nothing || oo_diff(r, o, "oo_." * p, out)
    end
    bad = [(p, v) for (p, v) in out if v > tol && !any(s -> occursin(s, p), skip)]
    worst = isempty(out) ? 0.0 : maximum(v for (_, v) in out if !isinf(v) || true)
    println(rpad(case, 12), "worst relative difference ", @sprintf("%.2e", maximum([v for (p, v) in out if !any(s -> occursin(s, p), skip)]; init=0.0)),
            " over ", length(out), " fields")
    isempty(bad) || foreach(b -> println("   ", b), bad)
    @test isempty(bad)
    ws
end

# pac.estimate: the parameters are written to M_, the results to oo_.pac.<model>
const PE_TOL = Dict("pe_t6" => 1e-6, "pe_t7" => 1e-6, "pe_t9" => 1e-9, "pe_t12" => 1e-9, "pe_t15" => 5e-4, "pe_t17" => 1e-6, "pe_t18" => 1e-6,
                    "pe_t19g" => 1e-9, "pe_t25" => 1e-9, "pe_v9" => 1e-6, "pe_t8a" => 1e-9)
for case in sort(collect(keys(PE_TOL)))
    @testset "pac.estimate: $case" begin
        ws = check_case(case, ["pac"], PE_TOL[case])
    end
end
