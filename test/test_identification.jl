# Identification at order 1 (src/identification.jl) against Dynare 7.2 under Octave 8.4: ref_identification/<case>_identif.mat is
# the file Dynare writes into <case>/identification/ (pydyn's references).  Fields that are rounding noise of exactly singular
# problems are not compared: `cond` of rank-deficient matrices, the multicorrelations Mco/Pco of collinear columns, the arbitrary
# null-space basis V and the null-space based indno of the Hessian.
using Test; include("common.jl")
using Random, Logging, MAT

haskey(ENV, "ID_DEV") && Base.include(JDyn, joinpath(@__DIR__, "..", "src", "identification.jl"))   # development: no precompile
const RI = joinpath(@__DIR__, "ref_identification")
const ID_CASES0 = ["BrockMirman", "fl1", "cgg_criteria_differ", "as2007_QT", "forward_looking_varobs_x", "au5", "au6", "kim2",
                  "fs2000_corr_pt", "as2007_kron_m1", "as2007_kron_p1", "as2007_negF"]
const ID_CASES = haskey(ENV, "ID_ONLY") ? split(ENV["ID_ONLY"], ",") : ID_CASES0
const ID_BASIS = Dict("fs2000_corr_pt" => ("ide_minimal_point.dMINIMAL", "ide_minimal_point.norm_dMINIMAL"))
const ID_KRON = ("si_dMOMENTSnorm", "si_dREDUCEDFORMnorm", "si_dDYNAMICnorm", "jweak", "jweak_pair", "indno")
const ID_SINGULAR = Dict("kim2" => ("si_dMOMENTSnorm", "si_dREDUCEDFORMnorm", "si_dDYNAMICnorm"),
                         "as2007_kron_m1" => ID_KRON, "as2007_kron_p1" => ID_KRON,
                         "as2007_negF" => (ID_KRON..., "ind_dREDUCEDFORM", "norm_dREDUCEDFORM", "si_dREDUCEDFORM", "REDUCEDFORM"))
const ID_TOL = Dict("kim2" => 1e-7, "au5" => 1e-7, "au6" => 1e-7, "as2007_kron_m1" => 1e-5, "as2007_kron_p1" => 1e-5, "as2007_negF" => 1e-5)
const ID_NOISE = ("cond", "Mco", "Pco", "V")

function id_cmp(a, b, path, bad, tol, skip)
    last = String(last_component(path))
    if a isa AbstractDict
        b isa AbstractDict || (push!(bad, "$path: not a struct"); return)
        Set(keys(a)) == Set(keys(b)) || (push!(bad, "$path: fields $(sort(collect(keys(a)))) vs $(sort(collect(keys(b))))"); return)
        for k in keys(a)
            id_cmp(a[k], b[k], "$path.$k", bad, tol, skip)
        end
        return
    end
    (last in ID_NOISE || path == "ide_hess_point.indno" || path in skip || last in skip) && return
    a isa AbstractString && return
    if a isa AbstractArray && eltype(a) === Any
        size(a) == size(b) || (push!(bad, "$path: cell size $(size(a)) vs $(size(b))"); return)
        for i in eachindex(a)
            id_cmp(a[i], b[i], "$path{$i}", bad, tol, skip)
        end
        return
    end
    a = Array{Float64}(a isa Number ? fill(a, 1, 1) : a); b = Array{Float64}(b isa Number ? fill(b, 1, 1) : b)
    if size(a) != size(b) && length(a) == length(b) && 1 in size(a) && 1 in size(b)
        b = reshape(b, size(a))
    end
    size(a) == size(b) || (push!(bad, "$path: shape $(size(a)) vs $(size(b))"); return)
    na, nb = isnan.(a), isnan.(b)
    na == nb || (push!(bad, "$path: NaN pattern"); return)
    a = ifelse.(na, 0.0, a); b = ifelse.(nb, 0.0, b)
    sc = max(1.0, isempty(a) ? 1.0 : maximum(abs, a))
    isempty(a) || maximum(abs, a .- b) <= tol * sc || push!(bad, "$path: maxdiff $(maximum(abs, a .- b)) (scale $sc)")
end
last_component(p) = split(p, '.')[end]

function id_run_case(case)
    d = mktempdir()
    for f in readdir(RI)
        (endswith(f, ".inc") || endswith(f, "_QT.mat")) && cp(joinpath(RI, f), joinpath(d, f))
    end
    cp(joinpath(RI, case * ".mod"), joinpath(d, case * ".mod"))
    with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, case * ".mod"); random=:octave)
    end
    joinpath(d, case, "identification")
end

@testset "identification (order 1) vs Dynare 7.2" begin
    @testset "$case" for case in ID_CASES
        out = id_run_case(case)
        skip = (get(ID_BASIS, case, ())..., get(ID_SINGULAR, case, ())...)
        stems = [case * "_identif"; [splitext(f)[1] for f in sort(readdir(RI)) if occursin(Regex("^" * case * "_identif_\\d+\\.mat\$"), f)]]
        for stem in stems
            ref = matread(joinpath(RI, stem * ".mat"))
            got = matread(joinpath(out, stem * ".mat"))
            bad = String[]
            for k in keys(ref)
                k == "store_options_ident" && continue
                @test haskey(got, k)
                haskey(got, k) || continue
                if startswith(k, "STO_")
                    a, b = ref[k], got[k]
                    @test size(a) == size(b)
                    if eltype(a) !== Any && size(a) == size(b)
                        @test maximum(abs, a .- b) <= 1e-7 * max(1.0, maximum(abs, a))
                    end
                    continue
                end
                id_cmp(ref[k], got[k], k, bad, get(ID_TOL, case, 1e-8), skip)
            end
            @test isempty(bad)
            isempty(bad) || println(stem, ": ", join(bad, "; "))
        end
        @test any(f -> occursin(Regex("^" * case * "_.*_identif\\.mat\$"), f), readdir(out))
    end
end
