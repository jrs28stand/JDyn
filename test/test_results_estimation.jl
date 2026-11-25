# estimation / calib_smoother: oo_, M_ and options_ against Dynare 7.2 (Octave 8.4) on small models.
# References: test/ref_oo_estimation (generators in its gen/ folder).
include("common.jl")
include("oo_compare.jl")

const RE = joinpath(@__DIR__, "ref_oo_estimation")
const _SKIP = Set(["time", "runtime", "initval_series", "initial_", "endo_simul", "exo_simul", "exo_det_simul", "gui",
                   "dr.eigval", "dr.Gy", "dr.gx"])   # eigval, Gy, gx of dr are not written (QZ ordering)

function _re_copy()
    d = mktempdir()
    for f in readdir(RE; join=true)
        isfile(f) && cp(f, joinpath(d, basename(f)))
    end
    d
end

function _re_run(mod, defs)
    d = _re_copy()
    with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, mod); extra_args=defs)
    end
end

function _keys_equal(j, r, p, probs)
    if r isa AbstractDict
        j isa JDyn.MStruct || (push!(probs, "$p: not a struct"); return)
        for k in keys(r)
            haskey(j, k) ? _keys_equal(j[k], r[k], p * "." * k, probs) : push!(probs, "$p.$k: missing")
        end
        for k in keys(j)
            haskey(r, k) || push!(probs, "$p.$k: extra")
        end
    else
        ja = j isa Number ? fill(j, 1, 1) : j
        ra = r isa Number ? fill(r, 1, 1) : r
        (ja isa AbstractArray && ra isa AbstractArray && size(ja) != size(ra)) && push!(probs, "$p: size $(size(ja)) vs $(size(ra))")
    end
end

function _cmp_case(ws, name; tol=1e-9, skip=_SKIP, mkeys=("params", "Sigma_e", "H", "Correlation_matrix"))
    ref = matread(joinpath(RE, name * "_oo.mat"))
    probs, worst, at = oo_compare(ws.oo_, ref; skip=skip)
    big = filter(p -> !occursin("max diff", p) || parse(Float64, split(p, "max diff ")[2]) > tol, probs)
    # fields that only the reference has (stoch_simul leftovers: endo_simul etc.) are skipped above
    Mr = matread(joinpath(RE, name * "_M.mat"))
    dM = Dict(k => (size(ws.M_[k]) == size(Mr[k]) ? maximum(abs.(ws.M_[k] .- Mr[k])) : Inf) for k in mkeys)
    big, worst, dM
end

@testset "estimation: mode_compute=0, smoother only (est_small variant 1)" begin
    ws = _re_run("est_small.mod", ["-DVARIANT=1"])
    big, worst, dM = _cmp_case(ws, "est_small_v1")
    @test isempty(big)
    @test worst < 1e-12
    @test all(v -> v < 1e-12, values(dM))
    O = ws.options_
    @test O.first_obs == 1.0 && O.nobs == 120.0     # the early return of dynare_estimation_1 keeps first_obs = 1
    @test length(O.varobs_id) == 3
    @test ws.oo_.dr.inv_order_var isa AbstractMatrix && size(ws.oo_.dr.inv_order_var, 1) == 1
    @test size(ws.oo_.SmoothedVariables.y) == (120, 1)
    @test ndims(ws.oo_.FilteredVariablesKStepAhead) == 3
end

@testset "estimation without a smoother (est_small variant 2)" begin
    ws = _re_run("est_small.mod", ["-DVARIANT=2"])
    big, worst, dM = _cmp_case(ws, "est_small_v2")
    @test isempty(big)
    @test all(v -> v < 1e-12, values(dM))
    @test !haskey(ws.oo_, "SmoothedVariables")
    @test haskey(ws.oo_, "prior") && !haskey(ws.oo_, "posterior_mode")
end

@testset "calib_smoother (est_small variant 4)" begin
    ws = _re_run("est_small.mod", ["-DVARIANT=4"])
    big, worst, dM = _cmp_case(ws, "est_small_v4")
    @test isempty(big)
    @test worst < 1e-12
    @test all(v -> v < 1e-12, values(dM))
    @test ws.options_.parameter_set == "calibration"
end

@testset "estimation with mode_compute=4 (variants 0, 5, 6): optimiser-limited agreement" begin
    for v in (0, 5, 6)
        ws = _re_run("est_small.mod", ["-DVARIANT=$v"])
        # csminwel stops at slightly different points than Dynare's: values agree to 1e-3, the layout is exact
        big, worst, dM = _cmp_case(ws, "est_small_v$v"; tol=1e-3, skip=union(_SKIP, Set(["optimization_info"])))
        @test isempty(big)
        @test all(v -> v < 1e-3, values(dM))
        ref = matread(joinpath(RE, "est_small_v$(v)_oo.mat"))
        probs = String[]
        _keys_equal(ws.oo_.posterior.optimization, ref["posterior"]["optimization"], "oo_.posterior.optimization", probs)
        @test isempty(filter(p -> !occursin("optimization_info", p), probs))
        @test ws.oo_.posterior.optimization.optimization_info isa JDyn.MStruct
    end
end

@testset "estimation with a short Metropolis-Hastings run (variant 3): field structure" begin
    ws = _re_run("est_small.mod", ["-DVARIANT=3"])
    ref = matread(joinpath(RE, "est_small_v3_oo.mat"))
    probs = String[]
    for k in ("posterior_mean", "posterior_median", "posterior_var", "posterior_hpdinf", "posterior_hpdsup", "posterior_deciles",
              "posterior_density", "posterior_std_at_mode", "posterior_mode", "MarginalDensity", "posterior", "prior",
              "SmoothedVariables", "SmoothedShocks")
        haskey(ref, k) || continue
        haskey(ws.oo_, k) ? _keys_equal(ws.oo_[k], ref[k], "oo_." * k, probs) : push!(probs, "oo_.$k missing")
    end
    filter!(p -> !occursin("optimization_info", p), probs)
    @test isempty(probs)
    @test haskey(ws.oo_.MarginalDensity, "ModifiedHarmonicMean") == haskey(ref["MarginalDensity"], "ModifiedHarmonicMean")
    @test ws.options_.mh_replic == 300.0
end

@testset "smoother with measurement errors and loglinear-free model (smo_me, CASE=2)" begin
    ws = _re_run("smo_me.mod", ["-DCASE=2"])
    # M_.Correlation_matrix: the identity (a `var a, b = cov` covariance does not enter it; only `corr` statements and estimated correlations do)
    big, worst, dM = _cmp_case(ws, "smo_me_case2"; skip=union(_SKIP, Set(["var_list"])), mkeys=("params", "Sigma_e", "H", "Correlation_matrix"))
    @test isempty(big)
    @test worst < 1e-10
    @test all(v -> v < 1e-10, values(dM))
end

@testset "posterior density: Dynare's kernel estimate on 512 points" begin
    r = matread(joinpath(RE, "dens_ref.mat"))
    d = JDyn._posterior_density(vec(r["x"]))
    @test size(d) == size(r["density"])
    @test maximum(abs.(d .- r["density"])) < 1e-12
end

@testset "estimation! / calib_smoother! re-run in place" begin
    d = _re_copy()
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "est_small.mod"); extra_args=["-DVARIANT=2"])
    end
    vl = ["y", "pi", "r", "a", "u", "yobs", "piobs", "robs"]
    cd(d) do
        with_logger(NullLogger()) do
            JDyn.estimation!(ws, vl; datafile="est_nk_data", mode_compute=0, smoother=true, filtered_vars=true,
                             filter_step_ahead=[1 3], filter_covariance=true, smoothed_state_uncertainty=true, nodisplay=true)
        end
    end
    ref = matread(joinpath(RE, "est_small_v1_oo.mat"))
    probs, worst, _ = oo_compare(ws.oo_, ref; skip=_SKIP)
    @test worst < 1e-12
    @test isempty(filter(p -> !occursin("max diff", p), probs))
    # a posterior mode and then a calib_smoother run at that parameter set (read back from oo_)
    cd(d) do
        with_logger(NullLogger()) do
            JDyn.estimation!(ws; datafile="est_nk_data", mode_compute=4, nodisplay=true)
            JDyn.calib_smoother!(ws, ["y", "pi"]; datafile="est_nk_data", parameter_set="posterior_mode")
        end
    end
    @test haskey(ws.oo_, "posterior_mode") && ws.options_.parameter_set == "posterior_mode"
    @test size(ws.oo_.SmoothedVariables.y) == (120, 1) || size(ws.oo_.SmoothedVariables.y, 2) == 1
    @test ws.oo_.dr.restrict_var_list isa AbstractMatrix
end

@testset "estimation!: unsupported options are refused" begin
    d = _re_copy()
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d, "est_small.mod"); extra_args=["-DVARIANT=2"])
    end
    cd(d) do
        @test_throws Exception with_logger(NullLogger()) do
            JDyn.estimation!(ws; datafile="est_nk_data", mode_compute=0, no_init_estimation_check_first_obs=true)
        end
    end
    d2 = _re_copy()
    ws2 = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(d2, "est_small.mod"); extra_args=["-DVARIANT=2"])
    end
    cd(d2) do
        @test_throws Exception with_logger(NullLogger()) do
            JDyn.estimation!(ws2; datafile="est_nk_data", mode_compute=0, definitely_not_an_option=1)
        end
    end
end
