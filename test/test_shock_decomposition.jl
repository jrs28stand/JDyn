# Shock decompositions (shock_decomposition, initial_condition_decomposition, realtime, squeeze, shock groups, init2shocks, the numbers
# plot_shock_decomposition draws) against Dynare 7.2 under Octave 8.4. References are pydyn's: smo_*_case*.mat (decompositions of the
# smoother cases), sd_nk_case*.mat, sd_ur_case*.mat and the *_plot.mat files made by reference/r_shock_decomposition.m. Dynare's own
# parameter vectors (posterior mode ...) are fed through `theta=`, so the comparisons do not depend on an optimiser or a sampler.
# The figures themselves are not drawn: their content comes out as tables (decomposition_table); the references of the drawn
# bars and lines (drawn_*) are not compared. Run alone: `using Test; include("common.jl")` then this file.
using Test, JDyn, MAT, LinearAlgebra, Random, Logging, Printf
isdefined(@__MODULE__, :REF) || include(joinpath(@__DIR__, "common.jl"))
using JDyn: Estimation

const SDD_DIR = mktempdir()
for f in readdir(REF; join=true)
    b = basename(f)
    ((startswith(b, "smo_") && (endswith(b, ".mod") || endswith(b, "data.m"))) || b in ("sd_nk.mod", "sd_ur.mod", "est_nk_data.m", "est_ur_data.m", "est_ur_miss.m")) &&
        cp(f, joinpath(SDD_DIR, b))
end
const _SDD_MODELS = Dict{Tuple{String,Int},JDyn.Model}()
function sddload(name, case)
    get!(_SDD_MODELS, (name, case)) do
        with_logger(NullLogger()) do
            JDyn.Model(JDyn.run_preprocessor(joinpath(SDD_DIR, name * ".mod"); extra_args=["-DCASE=$case"]))
        end
    end
end
quiet(f) = with_logger(f, NullLogger())
sref(name) = matread(joinpath(REF, name * ".mat"))
asvec(x) = x isa Number ? [x] : vec(x)
maxdiffs = Dict{String,Float64}()
track!(key, v) = (maxdiffs[key] = max(get(maxdiffs, key, 0.0), v))
function md(a, b)
    b = reshape(b, size(a))
    @assert isnan.(a) == isnan.(b) "NaN pattern differs"
    ok = .!isnan.(b)
    any(ok) ? maximum(abs.(a[ok] .- b[ok])) : 0.0
end
const VL = ["y", "pi", "robs"]

@testset "shock decompositions" begin

@testset "shock_decomposition of the smoother cases: $name case $case" for (name, case, init_state) in
        (("smo_fs", 1, 0), ("smo_fs", 2, 1), ("smo_fs", 4, 0), ("smo_me", 1, 0), ("smo_me", 3, 0), ("smo_ur", 1, 0), ("smo_ur", 3, 0))
    m, r = sddload(name, case), sref("$(name)_case$case")
    res = quiet(() -> JDyn.smooth(m; replicate_dynare_bugs=true))
    dec = quiet(() -> shock_decomposition(m, res))               # init_state from the .mod file
    @test dec.init_state == init_state
    @test size(dec.z) == (m.n_endo, m.n_exo + 2, JDyn.nobs(res)) == size(r["shock_decomposition"])
    @test dec.columns == vcat(m.exo_names, ["initial_conditions", "smoothed"])
    e = md(dec.z, r["shock_decomposition"])
    track!("shock_decomposition (smo_*)", e)
    @test e < 1e-12
    @test maximum(abs.(dropdims(sum(dec.z[:, 1:end-1, :]; dims=2); dims=2) .- dec.z[:, end, :])) < 1e-13
    v = m.endo_names[4]
    @test dec[v]["smoothed"] == res.smoothed[:, 4]
    chk = JDyn.check_smoother(quiet(() -> JDyn.smooth(m)), shock_decomposition(m, quiet(() -> JDyn.smooth(m)); init_state=0))
    @test Set(keys(chk)) == Set(["transition", "simulation", "measurement", "decomposition", "decomposition_initial"])
    @test maximum(values(chk)) < 1e-10
end

@testset "shock_decomposition runs the smoother itself" begin
    m = sddload("smo_me", 1)
    dec = quiet(() -> shock_decomposition(m))               # data and options from the .mod file
    @test md(dec.z, sref("smo_me_case1")["shock_decomposition"]) < 1e-12
    res = quiet(() -> JDyn.smooth(m))
    @test_throws MethodError shock_decomposition(m, res; nobs=10)    # a result and smoother options cannot be combined
end

@testset "initial_condition_decomposition: $name" for (name, case) in (("smo_fs", 1), ("smo_me", 1), ("smo_ur", 1))
    m, r = sddload(name, case), sref("$(name)_case$case")
    res = quiet(() -> JDyn.smooth(m; replicate_dynare_bugs=true))
    dec = initial_condition_decomposition(m, res)
    @test size(dec.z) == (m.n_endo, m.n_endo + 2, JDyn.nobs(res))
    e = md(dec.z[:, :, Int.(asvec(r["icd_idx"]))], r["initval_decomposition"])
    track!("initial_condition_decomposition (smo_*)", e)
    @test e < 1e-12
end

@testset "group_shocks" begin
    m = sddload("smo_me", 1)
    dec = shock_decomposition(m, quiet(() -> JDyn.smooth(m)))
    g = group_shocks(dec; groups=["supply" => ["e_a"], "demand" => ["e_g", "e_v"]])
    @test g.columns == ["supply", "demand", "Initial values", "smoothed"]    # Dynare's label: no shock left over
    @test g.z[:, 1, :] == dec.z[:, 1, :] && g.z[:, 2, :] ≈ dec.z[:, 2, :] .+ dec.z[:, 3, :] && g.z[:, 3, :] == dec.z[:, 4, :]
    g = group_shocks(dec; groups=["policy" => ["e_v"]])             # shocks in no group join the initial values
    @test g.columns == ["policy", "Others + Initial Values", "smoothed"]
    @test g.z[:, 2, :] ≈ dec.z[:, 1, :] .+ dec.z[:, 3, :] .+ dec.z[:, 4, :]
    @test maximum(abs.(dropdims(sum(g.z[:, 1:end-1, :]; dims=2); dims=2) .- g.z[:, end, :])) < 1e-13
    @test_throws JDyn.SmootherError group_shocks(dec; groups=["x" => ["nope"]])
    @test_throws JDyn.SmootherError group_shocks(g; groups=["x" => ["e_a"]])         # grouped already
end

# ------------------------------------------------------------------ estimated parameters: groups, plotted numbers
mode_run = Ref{Any}(nothing)
function mode_state()
    mode_run[] === nothing || return mode_run[]
    m = sddload("sd_nk", 1); est = Estimation(m); r = sref("sd_nk_case1")
    res = quiet(() -> JDyn.smooth(m; estimation=est, theta=vec(r["theta_mode"])))
    mode_run[] = (m=m, est=est, r=r, res=res, dec=shock_decomposition(m, res), icd=initial_condition_decomposition(m, res))
end

@testset "shock_decomposition statement at the posterior mode" begin
    (; m, est, r, res, dec) = mode_state()
    # shock_decomposition(parameter_set=posterior_mode) of the .mod file: Dynare's stored oo_.shock_decomposition
    d = quiet(() -> shock_decomposition(m; estimation=est, mode=vec(r["theta_mode"])))
    @test d.parameter_set == "posterior_mode"
    e = md(d.z, r["sd_in_mod"])
    track!("shock_decomposition at the posterior mode (sd_nk)", e)
    @test e < 1e-12
    @test_throws JDyn.SmootherError quiet(() -> shock_decomposition(m))        # posterior_mode without an estimation
    cal = quiet(() -> shock_decomposition(m; parameter_set="calibration"))
    @test cal.parameter_set == "calibration" && maximum(abs.(cal.z .- d.z)) > 1e-3
    # posterior mean: Dynare's stored decomposition
    rm_ = quiet(() -> shock_decomposition(m, quiet(() -> JDyn.smooth(m; estimation=est, theta=vec(r["theta_mean"])))))
    @test md(rm_.z, r["posterior_mean"]["shock_decomposition"]) < 1e-12
    icd = initial_condition_decomposition(m, res)
    @test md(icd.z[:, :, Int.(asvec(r["posterior_mode"]["icd_idx"]))], r["posterior_mode"]["initval_decomposition"]) < 1e-12
    @test md(dec.z, r["posterior_mode"]["shock_decomposition"]) < 1e-12
    ml = sddload("sd_nk", 2); em = Estimation(ml); rml = sref("sd_nk_case2")
    dml = quiet(() -> shock_decomposition(ml; estimation=em, theta=vec(rml["theta_mode"])))
    @test md(dml.z, rml["mle_mode"]["shock_decomposition"]) < 1e-12 && md(dml.z, rml["sd_in_mod"]) < 1e-12
end

@testset "groups of the .mod file match Dynare's make_the_groups" begin
    (; m, dec) = mode_state()
    p = sref("sd_nk_case1_plot")
    iv = [findfirst(==(v), m.endo_names) for v in VL]
    @test shock_groups(m, "grp") == ["supply" => ["eu"], "demand and policy" => ["ea", "er"]]
    g = group_shocks(dec; model=m, name="grp")
    @test g.columns == ["supply", "demand and policy", "Initial values", "smoothed"]
    @test md(g.z[iv, :, :], p["p_grp"]) < 1e-12
    g = group_shocks(dec; model=m, name="partial")
    @test g.columns == ["policy", "Others + Initial Values", "smoothed"]
    @test md(g.z[iv, :, :], p["p_partial"]) < 1e-12
    @test maximum(abs.(dropdims(sum(g.z[:, 1:end-1, :]; dims=2); dims=2) .- g.z[:, end, :])) < 1e-12
    @test shock_groups(m) == ["policy" => ["er"]]              # the last block when no name is given
    @test_throws JDyn.SmootherError shock_groups(m, "nope")
    @test_throws JDyn.SmootherError group_shocks(g; model=m, name="grp")
end

@testset "numbers of plot_shock_decomposition" begin
    # plot_shock_decomposition(..., get_decomp_only): groups, type = yoy / aoa, diff, flip, init2shocks, and the steady state Dynare attaches
    (; m, est, r, res, dec, icd) = mode_state()
    p = sref("sd_nk_case1_plot")
    f(; kw...) = decomposition_plot_data(dec, VL; model=m, kw...)
    last = vec(r["ys_last_solved"])              # Dynare's "steady state" here: oo_.dr.ys, the posterior-mean one
    T = JDyn.nobs(res)
    function chk(label, a, b)
        e = md(a, b)
        track!("plot_shock_decomposition numbers", e)
        @test e < 1e-12
    end
    d = f(use_shock_groups="grp")
    chk("grp", d.z, p["p_grp"])
    @test d.labels == ["supply", "demand and policy", "Initial values"] && d.periods == collect(1:T)
    # our default steady state is that of the parameter set (posterior mode); Dynare's is stale
    @test d.steady_state ≈ res.SteadyState[[findfirst(==(v), m.endo_names) for v in VL]]
    @test abs(d.steady_state[3] - vec(p["p_grp_ss"])[3]) > 0.05
    chk("grp ss", f(use_shock_groups="grp", steady_state=last).steady_state, p["p_grp_ss"])
    d = f(type="yoy", steadystate=true, steady_state=last)
    chk("yoy", d.z, p["p_yoy"]); chk("yoy ss", d.steady_state, p["p_yoy_ss"])
    @test size(d.z, 3) == T - 3 && d.periods[1] == 4 && d.steadystate
    d = f(type="qoq", diff=true, flip=true, use_shock_groups="grp", steady_state=last)
    chk("diff flip", d.z, p["p_diff_flip"])                    # first period NaN in both
    @test all(isnan, d.z[:, :, 1]) && !any(!iszero, d.steady_state)
    chk("yoy diff flip", f(type="yoy", diff=true, flip=true).z, p["p_yoy_diff_flip"])
    d = f(type="aoa", steady_state=last)                       # every fourth period, sample starting in Q1
    chk("aoa", d.z, p["p_aoa"]); chk("aoa ss", d.steady_state, p["p_aoa_ss"])
    @test d.periods == collect(1:T÷4)
    iv = [findfirst(==(v), m.endo_names) for v in VL]
    @test f(type="aoa", first_quarter=3).z == dec.z[iv, :, 2:4:end]                            # Q3, Q4, ...
    d = f(init2shocks="i2s", icd=icd)
    chk("init2shocks", d.z, p["p_i2s"])
    @test d.labels == ["ea + a", "eu + u", "er", "Initial values"]
    @test init2shocks(m, "i2s") == [("a", "ea"), ("u", "eu")]
    d = f(init2shocks="i2s", icd=icd, use_shock_groups="grp")
    chk("init2shocks groups", d.z, p["p_i2s_grp"])
    @test d.labels == ["supply", "demand and policy", "Initial values"]
    @test_throws JDyn.SmootherError f(init2shocks="i2s")        # needs icd
    @test_throws JDyn.SmootherError f(type="mom")
end

@testset "cropping, panels and tables" begin
    (; m, res, dec) = mode_state()
    d = decomposition_plot_data(dec, VL; model=m, plot_init_date=11, plot_end_date=40)
    @test [p.name for p in d.panels] == VL
    pan = d.panels[2]
    @test pan.periods == collect(11:40) && size(pan.contributions) == (4, 30)
    @test pan.contributions == dec.z[findfirst(==("pi"), m.endo_names), 1:4, 11:40]
    @test maximum(abs.(vec(sum(pan.contributions; dims=1)) .- pan.total)) < 1e-12
    @test size(d.z, 3) == JDyn.nobs(res)                       # z itself is not cropped (as get_decomp_only)
    @test_throws JDyn.SmootherError decomposition_plot_data(dec, VL; model=m, plot_init_date=500)
    d = quiet(() -> decomposition_plot_data(dec, VL; model=m, plot_end_date=500))     # warns, uses the last period
    @test d.panels[1].periods[end] == JDyn.nobs(res)
    # a variable that never moves gives no figure, as in Dynare
    flat = ShockDecomposition(z=zeros(1, 3, 5), endo_names=["v"], columns=["e", "initial_conditions", "smoothed"])
    @test isempty(decomposition_plot_data(flat).panels)
    # default variable list: the model's original variables
    @test decomposition_plot_data(dec; model=m).var_list == JDyn.orig_endo_names(m)
    bottom, top = stack_bars([1.0 -1.0 NaN; 2.0 3.0 1.0; -1.0 -2.0 0.0])
    @test bottom == [0 0 0; 1 0 0; 0 -1 0] && top[:, 1:2] == [1 -1; 3 3; -1 -3]
    # tables: content of the figures, written as CSV
    tab = decomposition_table(decomposition_plot_data(dec, VL; model=m, plot_init_date=11, plot_end_date=40))
    @test length(tab) == 3 * (4 + 1) * 30 && keys(first(tab)) == (:variable, :component, :period, :value)
    file = joinpath(mktempdir(), "sd.csv")
    JDyn.write_csv(file, tab)
    lines = readlines(file)
    @test lines[1] == "variable,component,period,value" && length(lines) == length(tab) + 1
    full = decomposition_table(dec; vars=["y"])
    @test length(full) == size(dec.z, 2) * size(dec.z, 3) && full[1].component == "ea" && full[end].component == "smoothed"
    @test decomposition_table(decomposition_plot_data(dec, VL; model=m); panels=false)[1].variable == "y"
end

# ------------------------------------------------------------------ calibration: realtime, expand, squeeze
cal_run = Ref{Any}(nothing)
function cal_state()
    cal_run[] === nothing || return cal_run[]
    m = sddload("sd_nk", 3)
    res = quiet(() -> JDyn.smooth(m))
    cal_run[] = (m=m, res=res, dec=shock_decomposition(m, res), rt=quiet(() -> realtime_shock_decomposition(m)))
end

@testset "calibrated case and expand" begin
    (; m, res, dec) = cal_state()
    r, p = sref("sd_nk_case3"), sref("sd_nk_case3_plot")
    @test md(dec.z, r["shock_decomposition"]) < 1e-12
    d = decomposition_plot_data(dec, VL; model=m, use_shock_groups="grp", steadystate=true)
    @test md(d.z, p["p_grp"]) < 1e-12
    @test md(d.steady_state, p["p_grp_ss"]) < 1e-12              # calibration: no stale steady state
    # Dynare numbers the sample from 1 whatever first_obs is (first_obs=5 here)
    @test res.first_obs == 5 && d.periods[1] == 1 == Int(vec(p["drawn_first_obs"]["f1_p0_line_x"])[1])
    # "expand group": the shocks of one group, the rest with the initial values
    d = decomposition_plot_data(dec, ["pi"]; model=m, use_shock_groups="grp", expand="demand and policy")
    @test md(d.z[1, :, :], p["p_expand"]) < 1e-12
    @test d.labels == ["ea", "er", "Others + Initial Values"]
    @test_throws JDyn.SmootherError group_shocks(dec; model=m, name="grp", expand="nope")
end

@testset "realtime_shock_decomposition: the four families of results" begin
    # realtime_shock_decomposition(forecast=4, presample=20, save_realtime=[30,45,60]) of the file
    (; m, dec, res, rt) = cal_state()
    r = sref("sd_nk_case3")
    @test (rt.nobs, rt.forecast) == (60, 4)
    stores = ["rt_" => rt.realtime, "cond_" => rt.conditional, "fcst_" => rt.realtime_forecast, "rtcond_" => rt.realtime_conditional]
    n = 0
    for (prefix, store) in stores
        keys_ = Set(k[length(prefix)+1:end] for k in keys(r) if startswith(k, prefix))
        prefix == "rt_" && filter!(k -> !startswith(k, "cond_") && !startswith(k, "fcst_"), keys_)
        @test keys_ == Set(keys(store))
        for k in keys_
            @test size(store[k]) == size(r[prefix*k])
            e = md(store[k], r[prefix*k])
            track!("realtime_shock_decomposition", e)
            @test e < 1e-12
            n += 1
        end
    end
    @test n == 13
    @test size(rt.realtime["pool"]) == (m.n_endo, m.n_exo + 2, 64) && size(rt.realtime["time_30"], 3) == 34
    # Dynare's presample rule: max(1, estimation presample - 1); the option can only lower it
    @test rt.presample == 1
    # Dynare's realtime recursion gives the shocks of period 1 to the initial conditions
    last = rt.realtime["time_60"][:, :, 1:60]
    @test all(last[:, 1:m.n_exo, 1] .== 0)
    @test last[:, end, :] == dec.z[:, end, :]
    alt = shock_decomposition(m, res; init_state=1)
    @test maximum(abs.(last .- alt.z)) < 1e-12
end

@testset "realtime variants" begin
    (; m, rt) = cal_state()
    r = sref("sd_nk_case3")
    rt0 = quiet(() -> realtime_shock_decomposition(m; forecast=0, presample=0, save_realtime=[10, 60]))
    @test md(rt0.realtime["pool"], r["rt0_pool"]) < 1e-12 && md(rt0.realtime["time_10"], r["rt0_time_10"]) < 1e-12
    @test md(rt0.conditional["pool"], r["rt0_cond_pool"]) < 1e-12
    @test isempty(rt0.realtime_forecast) && isempty(rt0.realtime_conditional) && Set(keys(rt0.conditional)) == Set(["pool"])
    fast = quiet(() -> realtime_shock_decomposition(m; fast_realtime=[40, 50], save_realtime=[45]))
    @test md(fast.realtime["pool"], r["fast_pool"]) < 1e-12 && md(fast.conditional["pool"], r["fast_cond_pool"]) < 1e-12
    @test md(fast.realtime_forecast["time_45"], r["fast_fcst_time_45"]) < 1e-12
    @test maximum(abs.(fast.realtime["pool"] .- rt.realtime["pool"])) > 1e-4
end

@testset "selecting a realtime result for plots: $name" for (name, realtime, vintage, first) in
        (("rt_pool", 1, 0, 1), ("rt_vintage", 1, 45, 1), ("cond_pool", 2, 0, 1), ("cond_vintage", 2, 45, 45), ("fcst_pool", 3, 0, 1), ("fcst_vintage", 3, 45, 45))
    (; m, rt) = cal_state()
    p = sref("sd_nk_case3_plot")
    d = decomposition_plot_data(select_vintage(rt, realtime, vintage), VL; model=m, use_shock_groups="grp")
    e = md(d.z, p["p_" * name])
    track!("plot numbers of realtime results", e)
    @test e < 1e-12
    @test d.periods[1] == first
    @test_throws JDyn.SmootherError select_vintage(rt, 1, 31)
    @test_throws JDyn.SmootherError select_vintage(rt, 4)
end

@testset "squeeze_shock_decomposition" begin
    (; m, dec, rt) = cal_state()
    p = sref("sd_nk_case3_plot")
    @test Int.(asvec(p["squeezed_i_var"])) == [findfirst(==(v), m.endo_names) for v in ("pi", "y")]
    sq = squeeze_shock_decomposition(dec, ["pi", "y"])
    @test sq.endo_names == ["pi", "y"] && md(sq.z, p["squeezed"]) < 1e-12
    sq_rt = squeeze_shock_decomposition(rt, ["pi", "y"])
    @test md(sq_rt.realtime["pool"], p["squeezed_rt_pool"]) < 1e-12
    @test size(sq_rt.realtime_forecast["time_45"], 1) == 2 && length(sq_rt.steady_state) == 2
    d = decomposition_plot_data(sq; model=m, use_shock_groups="grp")
    @test d.var_list == ["pi", "y"] || d.var_list == ["y", "pi"]
    @test_throws JDyn.SmootherError squeeze_shock_decomposition(dec, ["nope"])
end

@testset "screening of shocks" begin
    # more than 18 components: the 16 largest shocks (mean absolute contribution) for each variable, the others summed; nothing happens with groups
    # or 18 components or fewer
    p = sref("sd_nk_case3_plot")
    z = p["screen_z"]
    cols = vcat(["s$j" for j in 1:size(z, 2)-2], ["initial_conditions", "smoothed"])
    dec = ShockDecomposition(z=z, endo_names=["v1", "v2"], columns=cols)
    d = decomposition_plot_data(dec; screen_shocks=true)
    want = reverse(split(p["drawn_screen"]["f1_p0_labels"], "|")[2:end])           # Dynare lists them bottom-up
    @test d.panels[1].labels == want && want[end-1:end] == ["Others", "Initial values"]
    @test d.panels[2].labels != d.panels[1].labels                                  # chosen variable by variable
    @test size(d.panels[1].contributions) == (18, 12) && occursin("SCREEN", d.title)
    @test maximum(abs.(vec(sum(d.panels[1].contributions; dims=1)) .- dec.z[1, end, :])) < 1e-12
    plain = decomposition_plot_data(dec)
    @test size(plain.panels[1].contributions) == (22, 12)
    few = ShockDecomposition(z=dec.z[:, 5:end, :], endo_names=dec.endo_names, columns=dec.columns[5:end])
    @test size(decomposition_plot_data(few; screen_shocks=true).panels[1].contributions) == (18, 12)
    grouped = decomposition_plot_data(dec; screen_shocks=true, groups=["a" => ["s1", "s2"]])
    @test grouped.panels[1].labels == ["a", "Others + Initial Values"]
end

@testset "options that are not acted on are reported" begin
    m = quiet(() -> JDyn.Model(JDyn.run_preprocessor(joinpath(SDD_DIR, "sd_nk.mod"); extra_args=["-DCASE=3"])))
    extra = Any[
        Dict("statementName" => "shock_decomposition", "options" => Dict{String,Any}("shock_decomp.with_epilogue" => true, "parameter_set" => "calibration")),
        Dict("statementName" => "realtime_shock_decomposition", "options" => Dict{String,Any}("shock_decomp.forecast_params" => [1], "shock_decomp.forecast" => 2)),
        Dict("statementName" => "initial_condition_decomposition", "options" => Dict{String,Any}("initial_condition_decomp.write_xls" => true)),
        Dict("statementName" => "calib_smoother", "options" => Dict{String,Any}("no_init_estimation_check_first_obs" => true, "datafile" => "est_nk_data", "first_obs" => 5, "nobs" => 60)),
    ]
    m.statements = vcat(m.statements, extra)
    buf = IOBuffer()
    res = with_logger(Logging.SimpleLogger(buf)) do
        JDyn.smooth(m)
    end
    @test occursin("no_init_estimation_check_first_obs", String(take!(buf)))
    with_logger(Logging.SimpleLogger(buf)) do
        shock_decomposition(m, res)
    end
    s = String(take!(buf))
    @test occursin("with_epilogue", s) && occursin("shock_decomposition options", s) && !occursin("init_state", s)
    with_logger(Logging.SimpleLogger(buf)) do
        initial_condition_decomposition(m, res)
    end
    @test occursin("write_xls", String(take!(buf)))
    rt = with_logger(Logging.SimpleLogger(buf)) do
        realtime_shock_decomposition(m)
    end
    s = String(take!(buf))
    @test occursin("forecast_params", s) && rt.forecast == 2 && !occursin("save_realtime", s)
end

end # shock decompositions testset

println("\nMaximum differences against Dynare (shock decompositions):")
for k in sort(collect(keys(maxdiffs)))
    println("  ", rpad(k, 58), @sprintf("%.3g", maxdiffs[k]))
end
