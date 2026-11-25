# Parity batch 2, results layer: pac.estimate.nls / iterative_ols, shock_decomposition with forecasts and the loglinear smoother
# auxiliaries, run through dynare() and compared with Dynare 7.2 under Octave 8.4 (cases and references of pydyn's reference/oo
# store, copied to test/ref_parity2/oo/). Only the fields of the feature are compared (oo_ paths below), at the case's tolerance.
using Test; include("common.jl")
using Random

# JDYN_TEST_ONLY=pe_t12,sd_fc_pm runs only the named cases
selected(case) = (o = get(ENV, "JDYN_TEST_ONLY", ""); isempty(o) || case in split(o, ","))
const OOP2 = joinpath(@__DIR__, "ref_parity2", "oo")

"Largest relative difference (floor 1) between a reference value and ours, with the path of the worst field; `nothing` where a field is absent."
function oo_diff(ref, ours, path, out::Vector{Tuple{String,Float64}})
    if ours isa JDyn.MDseries
        rd = ref isa AbstractDict ? get(ref, "data", nothing) : ref
        rd === nothing ? push!(out, (path * " (dseries without data in the reference)", Inf)) : oo_diff(rd, ours.data, path * ".data", out)
    elseif ref isa AbstractDict
        ours isa JDyn.MStruct || (push!(out, (path * " (not a struct)", Inf)); return)
        for (k, v) in ref
            haskey(ours, k) || (isempty(v) || push!(out, (path * "." * k * " (missing)", Inf)); continue)
            oo_diff(v, ours[k], path * "." * k, out)
        end
    elseif ours isa JDyn.MDseries
        rd = ref isa AbstractDict ? ref["data"] : ref
        oo_diff(rd, ours.data, path * ".data", out)
    elseif ref isa AbstractString || (ref isa AbstractArray && !isempty(ref) && all(x -> x isa AbstractString, ref))
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
        cplx(x) = x isa Number ? ComplexF64[x] : vec(ComplexF64.(x))
        a = cplx(ref); b = try
            cplx(ours)
        catch
            push!(out, (path * " (not numeric)", Inf)); return
        end
        length(a) == length(b) || (push!(out, (path * " (size $(length(a)) vs $(length(b)))", Inf)); return)
        isempty(a) && return
        if any(!iszero(imag(x)) for x in a) || any(!iszero(imag(x)) for x in b)       # eigenvalues: compare the moduli, as the comparator
            a, b = sort(abs.(a)), sort(abs.(b))
        else
            a, b = real.(a), real.(b)
        end
        # NaN and infinities must sit in the same places (and be the same infinity)
        nonfin = .!(isfinite.(a)) .| .!(isfinite.(b))
        same_nonfin = all((isnan(a[i]) && isnan(b[i])) || a[i] == b[i] for i in findall(nonfin))
        fin = .!nonfin
        push!(out, (path, !same_nonfin ? Inf : (any(fin) ? maximum(abs.(a[fin] .- b[fin])) / max(1.0, maximum(abs.(a[fin]))) : 0.0)))
    end
    nothing
end

function run_case(case; random=:octave, edit=identity)
    d = mktempdir()
    write(joinpath(d, case * ".mod"), edit(read(joinpath(OOP2, case * ".mod"), String)))
    for extra in ("fsdat_simul.m", "fsdat_simul.mat")
        isfile(joinpath(OOP2, extra)) && cp(joinpath(OOP2, extra), joinpath(d, extra))
    end
    for f in readdir(OOP2)       # a mode file of the case
        startswith(f, case * "_") && endswith(f, ".mat") && cp(joinpath(OOP2, f), joinpath(d, f))
    end
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, case * ".mod"); random=random)
    end
    ws, matread(joinpath(OOP2, case * ".mat"))
end

function check_case(case, paths, tol; skip=String[], edit=identity)
    ws, ref = run_case(case; edit=edit)
    out = Tuple{String,Float64}[]
    roo = ref["oo_"]
    paths == ["*"] && (paths = [k for k in SD_PATHS if haskey(roo, k)])
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
const PE_TOL = Dict("pe_t6" => 1e-6, "pe_t7" => 1e-6, "pe_t9" => 1e-9, "pe_t12" => 1e-9, "pe_t15" => 1e-4, "pe_t17" => 1e-7, "pe_t18" => 1e-8,
                    "pe_t19g" => 1e-9, "pe_t25" => 5e-7, "pe_v9" => 1e-6, "pe_t8a" => 5e-8)
for case in sort(collect(keys(PE_TOL)))
    selected(case) || continue
    @testset "pac.estimate: $case" begin
        ws = check_case(case, ["pac"], PE_TOL[case])
    end
end

# shock_decomposition with forecast_type, from the calibration, the mode (maximum likelihood), the posterior mean / median / mode: the
# whole oo_ is compared (forecast_shock_decomposition, shock_decomposition_info, the estimation and conditional forecast fields)
const SD_PATHS = ["forecast_shock_decomposition", "shock_decomposition_info", "plot_shock_decomposition_info", "forecast", "conditional_forecast",
    "mle_mode", "posterior_mode", "posterior_mean", "posterior_median", "posterior_std", "SmoothedShocks", "SmoothedVariables", "UpdatedVariables", "MeanForecast", "PointForecast", "steady_state"]
const SD_TOL = Dict("sd_fc_cal" => 1e-10, "sd_fc_ml" => 1e-10, "sd_fc_pm" => 1e-9, "sd_fc_pmm" => 1e-9)
for case in sort(collect(keys(SD_TOL)))
    selected(case) || continue
    @testset "shock_decomposition forecast_type: $case" begin
        check_case(case, ["*"], SD_TOL[case])
    end
end

# posterior smoother of a loglinear model with auxiliary variables (consider_all_endogenous_and_auxiliary), smoother2histval from it
# (Smoother.loglinear), then forecast from those initial values at order 1 levels. Dynare's posterior smoother statistics are -Inf for
# the auxiliary variable of a lagged exogenous variable (log of its steady state 0): reproduced under replicate_dynare_bugs.
if selected("sm_ll_aux")
    @testset "posterior smoother, loglinear auxiliaries: sm_ll_aux" begin
        check_case("sm_ll_aux", ["steady_state", "dr", "Smoother", "SmoothedVariables", "SmoothedShocks", "UpdatedVariables", "Constant",
                                 "posterior_mean", "posterior_mode", "forecast"], 1e-6;
                   skip = ["oo_.dr.eigval"],      # the eigenvalues of the pencil that are infinite: rounding noise (1.3e16 vs 1.3e16)
                   edit = s -> replace(s, "options_.solve_tolf = 1e-12;" => "options_.solve_tolf = 1e-12;\noptions_.replicate_dynare_bugs = 1;"))
    end
end
