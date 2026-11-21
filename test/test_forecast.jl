# forecast and conditional_forecast (src/forecast.jl) against Dynare 7.2 under Octave 8.4. The references in
# test/forecast/*.mat come from test/forecast/r_forecast.m and r_forecast_est.m: the .mod files store oo_.forecast and
# oo_.conditional_forecast after each statement (F*, CF*) and, under -DOCTAVE, the standard normals the stochastic
# statements drew (*_draws, recorded with the same seed), which JDyn replays.
include("common.jl")
using Random

const FCDIR = joinpath(@__DIR__, "forecast")
fcref(name) = matread(joinpath(FCDIR, name))

"Largest absolute difference between a JDyn MStruct (or Dict) and a reference struct read by MAT.jl; fields must match."
function maxdiff(a, b; path="")
    if b isa AbstractDict
        ka = Set(String.(collect(keys(a))))
        kb = Set(String.(collect(keys(b))))
        ka == kb || error("$path: fields differ: JDyn $(sort(collect(setdiff(ka, kb)))) / Dynare $(sort(collect(setdiff(kb, ka))))")
        return maximum((maxdiff(a[k], b[k]; path=path * "." * k) for k in kb); init=0.0)
    end
    b isa AbstractString && return (a == b ? 0.0 : error("$path: $(repr(a)) vs $(repr(b))"))
    (a isa JDyn.Cell || b isa AbstractArray{<:AbstractString} || (b isa AbstractArray && eltype(b) == Any)) &&
        return (vec(String.(collect(a))) == vec(String.(collect(b))) ? 0.0 : error("$path: $(collect(a)) vs $(collect(b))"))
    A = a isa Number ? fill(Float64(a), 1, 1) : Float64.(a)
    B = b isa Number ? fill(Float64(b), 1, 1) : Float64.(b)
    size(A) == size(B) || error("$path: size $(size(A)) vs $(size(B))")
    maximum(abs.(A .- B); init=0.0)
end

function run_fc_cal(v)
    d = mktempdir()
    cp(joinpath(FCDIR, "fc_cal.mod"), joinpath(d, "fc_cal.mod"))
    with_logger(NullLogger()) do
        dynare(joinpath(d, "fc_cal.mod"); extra_args=["-DVARIANT=$v"])
    end
end

@testset "forecast after stoch_simul and conditional_forecast with calibration (fc_cal.mod)" begin
    for v in 0:3
        ref = fcref("fc_cal_v$v.mat")
        ws = run_fc_cal(v)
        @testset "variant $v: statements run in place" begin
            # deterministic statements: point forecasts and intervals at order 1, conditional forecasts with every shock controlled
            for k in (v == 3 ? String[] : ["F1", "F2", "F3", "CF1"])
                d = maxdiff(ws.vars[k], ref[k]; path=k)
                @test d < 1e-11
                d < 1e-11 || println("  fc_cal v$v $k: $d")
            end
            v == 1 && @test haskey(ws.vars["F1"], "HPDinf_ME") && haskey(ws.vars["F1"].HPDinf_ME, "gy_obs") && !haskey(ws.vars["F1"].HPDinf_ME, "k")
            v == 3 && @test maxdiff(ws.vars["F1"].Mean, ref["F1"]["Mean"]) < 1e-11      # order 2: the mean is deterministic
            v == 3 && @test maxdiff(ws.vars["F2"].Mean, ref["F2"]["Mean"]) < 1e-11
        end
        @testset "variant $v: Dynare's random numbers replayed" begin
            if v == 3
                # order 2: intervals from forecast_replic simulations (quantiles, MATLAB's method)
                h = ws.M_.endo_histval
                ws.M_.endo_histval = zeros(0, 0)
                ws.options_.forecasts.conf_sig = 0.9
                forecast!(ws; periods=6, draws=ref["F1_draws"])
                @test maxdiff(ws.oo_.forecast, ref["F1"]) < 1e-11
                ws.M_.endo_histval = h
                ws.options_.forecasts.conf_sig = 0.8
                forecast!(ws, ["k", "gy_obs"]; periods=5, draws=ref["F2_draws"])
                @test maxdiff(ws.oo_.forecast, ref["F2"]) < 1e-11
            else
                h = ws.M_.endo_histval
                ws.M_.endo_histval = zeros(0, 0)          # CF2 runs before the histval block
                conditional_forecast!(ws; controlled_varexo=["e_m"], parameter_set="calibration", replic=50, periods=10, conf_sig=0.7,
                                      draws=ref["CF2_draws"])
                d2 = maxdiff(ws.oo_.conditional_forecast, ref["CF2"])
                @test d2 < 1e-11
                ws.M_.endo_histval = h
                conditional_forecast!(ws; controlled_varexo=["e_m"], parameter_set="calibration", replic=30, periods=8, draws=ref["CF3_draws"])
                d3 = maxdiff(ws.oo_.conditional_forecast, ref["CF3"])
                @test d3 < 1e-11
                (d2 < 1e-11 && d3 < 1e-11) || println("  fc_cal v$v CF2 $d2 CF3 $d3")
                @test ws.vars["options_cond_fcst_"].replic == 30.0
            end
        end
    end
    @testset "Julia API on the loaded model" begin
        m = load_copy(joinpath(FCDIR, "fc_cal.mod"); output="second")
        ref = fcref("fc_cal_v0.mat")
        fc = forecast(m; periods=8, conf_sig=0.95, var_list=["k", "gy_obs"])   # the file's histval block (as F3)
        @test maximum(abs.(fc.Mean["k"] - vec(ref["F3"]["Mean"]["k"]))) < 1e-11
        @test maximum(abs.(fc.HPDsup["gy_obs"] - vec(ref["F3"]["HPDsup"]["gy_obs"]))) < 1e-11
        cf = conditional_forecast(m; paths=["gp_obs" => [1.009, 1.009, NaN, 1.012]], controlled_varexo=["e_m"], replic=30, periods=8,
                                  draws=ref["CF3_draws"])
        @test maxdiff(conditional_forecast_struct(cf), ref["CF3"]) < 1e-11
        tbl = conditional_forecast_table(cf, ["gp_obs", "k"]; periods=4)
        @test length(tbl) == 10 && tbl[1].variable == "gp_obs" && tbl[2].cond_mean == cf.cond_mean["gp_obs"][2]
        # the conditional mean meets the target; draws do not matter when every shock is controlled
        cf1 = conditional_forecast(m; paths=["gy_obs" => [1.01, 0.996], "gp_obs" => [1.012, 1.012]], controlled_varexo=["e_a", "e_m"],
                                   replic=3, periods=4, histval=false, rng=Random.Xoshiro(1))
        @test abs(cf1.cond_mean["gy_obs"][2] - 1.01) < 1e-12 && abs(cf1.cond_mean["gp_obs"][3] - 1.012) < 1e-12
        @test cf1.cond_ci["k"][1, :] ≈ cf1.cond_ci["k"][2, :]
        @test_throws ModelError conditional_forecast(m; paths=["gy_obs" => [1.0]], controlled_varexo=["e_a", "e_m"], replic=2, periods=3)
        @test_throws ModelError conditional_forecast(m; paths=["gy_obs" => [1.0]], controlled_varexo=["nope"], replic=2, periods=3)
        @test_throws ModelError forecast(m; var_list=["nope"])
        # order 2 without draws: Julia's random numbers, a mean that does not depend on them
        f2 = forecast(m; periods=3, order=2, replic=50, histval=false, rng=Random.Xoshiro(3))
        @test maximum(abs.(f2.Mean["k"] - vec(fcref("fc_cal_v3.mat")["F1"]["Mean"]["k"])[1:3])) < 1e-11
    end
end

@testset "forecast and conditional_forecast after estimation" begin
    @testset "Schorfheide (loglinear): forecast at the posterior mode, conditional forecasts" begin
        ref = fcref("est_fc_sch.mat")
        m = load_copy(joinpath(FCDIR, "est_fc_sch.mod"); extra=[joinpath(REF, "est_schorfheide_data.m")])
        est = JDyn.Estimation(m)
        # the mode: shock standard deviations (M_.Sigma_e at the mode), then the parameters (oo_.posterior_mode.parameters)
        xmode = vcat(sqrt.(diag(ref["Sigma_e"])), vec(ref["XMODE"]))
        @test est.names[1:2] == ["stderr e_a", "stderr e_m"] && length(xmode) == est.n
        sm = smooth(m; estimation=est, theta=xmode)
        fc = forecast(sm, m; periods=8, var_list=["gy_obs", "gp_obs", "k"])
        d = maxdiff(forecast_struct(fc), ref["FE"])
        @test d < 1e-10
        d < 1e-10 || println("  est_fc_sch FE: $d")
        cf1 = conditional_forecast(m; paths=["gy_obs" => [0.006, 0.002, 0.004, 0.004], "gp_obs" => fill(0.01, 5)], controlled_varexo=["e_m", "e_a"],
                                   parameter_set="posterior_mode", estimation=est, mode=xmode, replic=10, periods=10)
        d = maxdiff(conditional_forecast_struct(cf1), ref["CF1"])
        @test d < 1e-10
        d < 1e-10 || println("  est_fc_sch CF1: $d")
        cf2 = conditional_forecast(m; paths=["gp_obs" => fill(0.012, 3)], controlled_varexo=["e_m"], parameter_set="posterior_mode",
                                   estimation=est, mode=xmode, replic=40, periods=7, conf_sig=0.6, draws=ref["CF2_draws"])
        d = maxdiff(conditional_forecast_struct(cf2), ref["CF2"])
        @test d < 1e-10
        d < 1e-10 || println("  est_fc_sch CF2: $d")
        # estimate() fills res.forecast from the smoother at the mode (here the mode is given: mode_compute=0 with smoother)
        res = with_logger(NullLogger()) do
            estimate(m; mode_compute=0, smoother=true, forecast=8)
        end
        fc0 = forecast(smooth(m; estimation=est, theta=est.x0), m; periods=8, var_list=["gy_obs", "gp_obs", "k"])
        @test res.forecast isa ForecastResult && isapprox(res.forecast.Mean["k"], fc0.Mean["k"]; atol=1e-12)
    end
    @testset "small NK model with prefilter and a measurement error (mode_compute=0, smoother)" begin
        ref = fcref("est_fc_nk.mat")
        m = load_copy(joinpath(FCDIR, "est_fc_nk.mod"); extra=[joinpath(REF, "est_nk_data.m")])
        res = with_logger(NullLogger()) do
            estimate(m)
        end
        fc = res.forecast
        @test haskey(forecast_struct(fc), "HPDinf_ME") && haskey(fc.HPDinf_ME, "yobs")
        # Dynare's intervals of the observed variables stay centred on the model's steady state (not on the data mean)
        dyn = res.smoother === nothing ? nothing : forecast(res.smoother, m; periods=6, var_list=["robs", "yobs", "piobs", "y"], replicate_dynare_bugs=true)
        d = maxdiff(forecast_struct(dyn), ref["FE"])
        @test d < 1e-10
        d < 1e-10 || println("  est_fc_nk FE: $d")
        @test maximum(abs.(fc.Mean["robs"] - vec(ref["FE"]["Mean"]["robs"]))) < 1e-10
        shift = fc.Mean["piobs"] - dyn.Mean["piobs"]
        @test maximum(abs.(shift)) == 0.0
        mid = (fc.HPDinf["piobs"] + fc.HPDsup["piobs"]) / 2
        @test maximum(abs.(mid - fc.Mean["piobs"])) < 1e-12              # corrected: the interval is centred on the mean
        @test maximum(abs.((dyn.HPDinf["piobs"] + dyn.HPDsup["piobs"]) / 2 - fc.Mean["piobs"])) > 1e-3
        @test_throws ArgumentError forecast(res.smoother, m; periods=6, var_list=["robs", "y"], replicate_dynare_bugs=true)
    end
    if isfile(joinpath(FCDIR, "est_fc_ur.mat"))
        @testset "unit root with observation trends (diffuse filter): forecast, conditional forecast at the prior mean" begin
            ref = fcref("est_fc_ur.mat")
            m = load_copy(joinpath(FCDIR, "est_fc_ur.mod"); extra=[joinpath(REF, "est_ur_data.m")])
            res = with_logger(NullLogger()) do
                estimate(m)
            end
            dyn = forecast(res.smoother, m; periods=6, var_list=["y", "x", "yobs", "g"], replicate_dynare_bugs=true)
            d = maxdiff(forecast_struct(dyn), ref["FE"])
            @test d < 1e-9
            d < 1e-9 || println("  est_fc_ur FE: $d")
            fc = res.forecast
            @test fc.Mean["yobs"] == dyn.Mean["yobs"]
            @test maximum(abs.((fc.HPDinf["yobs"] + fc.HPDsup["yobs"]) / 2 - fc.Mean["yobs"])) < 1e-10
            cf = conditional_forecast(m; paths=["yobs" => fill(2.5, 3)], controlled_varexo=["eg"], parameter_set="prior_mean",
                                      estimation=res.est, replic=25, periods=6, draws=ref["CF1_draws"])
            d = maxdiff(conditional_forecast_struct(cf), ref["CF1"])
            @test d < 1e-9
            d < 1e-9 || println("  est_fc_ur CF1: $d")
        end
    end
end
