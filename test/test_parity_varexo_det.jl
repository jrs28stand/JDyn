# varexo_det in stochastic models (src/varexo_det.jl): decision rules ghud/ghxud/ghuud/ghudud and simult_varexo_det.m paths at orders
# 1 and 2, and the forecast command, against Dynare 7.2 / Octave 8.4 (pydyn's references test/ref_parity/xdet/).
include(joinpath(@__DIR__, "common.jl"))

const XD = joinpath(@__DIR__, "ref_parity", "xdet")
xd_load(name) = load_copy(joinpath(XD, name * ".mod"); output="second")

function xd_close(a, b; tol=1e-12)
    A = a isa Number ? fill(Float64(a), 1, 1) : Float64.(a)
    B = b isa Number ? fill(Float64(b), 1, 1) : Float64.(b)
    A isa Vector && (A = reshape(A, :, 1)); B isa Vector && (B = reshape(B, :, 1))
    if size(A) != size(B)
        (ndims(A) <= 2 && min(size(A)...) == 1 && min(size(B)...) == 1 && length(A) == length(B)) || return Inf
        B = reshape(B, size(A))
    end
    maximum(abs.(A .- B)) / max(1.0, maximum(abs.(B)))
end

@testset "varexo_det decision rules and simult_varexo_det against Dynare" begin
    worst = 0.0
    for order in (1, 2)
        order == 2 && !JDyn.korder_available() && continue
        m = xd_load("xdet_o$order")
        r = matread(joinpath(XD, "xdet_o$order.mat"))
        @test JDyn.n_exo_det(m) == 2 && JDyn.exo_det_names(m) == ["ahat", "bhat"]
        @test JDyn.exo_det_length(m) == Int(r["exo_det_length"]) == 3
        @test vec(JDyn.exo_det_steady(m)) == vec(Float64.(r["exo_det_steady_state"]))
        ys = vec(Float64.(r["ys"]))
        dr = order == 1 ? JDyn.as_dr(JDyn.first_order(m; ys=ys)) : JDyn.k_order(m; order=2, ys=ys)
        d = JDyn.exo_det_decision_rules(m, dr)
        e = xd_close(d["ghx"], r["ghx"]); @test e < 1e-12; worst = max(worst, e)
        for j in 1:3
            e = xd_close(d["ghud"][j], r["ghud_$j"]); @test e < 1e-12; worst = max(worst, e)
            if order == 2
                for (k, ref) in (("ghxud", "ghxud_$j"), ("ghuud", "ghuud_$j"))
                    e = xd_close(d[k][j], r[ref]); @test e < 1e-12; worst = max(worst, e)
                end
                for k in 1:j
                    e = xd_close(d["ghudud"][(k, j)], r["ghudud_$(k)_$j"]); @test e < 1e-12; worst = max(worst, e)
                end
            end
        end
        Y = JDyn.simult_varexo_det(m, d, Matrix{Float64}(r["sim_ex"]), Matrix{Float64}(r["sim_ex_det"]); y0=vec(Float64.(r["sim_y0"])))
        e = xd_close(permutedims(Y), r["sim"]); @test e < 1e-12; worst = max(worst, e)
    end
    println("varexo_det rules and paths: worst scaled difference $worst")
end

@testset "varexo_det: the stochastic view is unchanged, refusals" begin
    m = xd_load("xdet_o1")
    sol = JDyn.first_order(m)
    @test size(JDyn.ghu(sol)) == (m.n_endo, m.n_exo)
    mo = with_logger(NullLogger()) do
        JDyn.theoretical_moments(m)
    end
    @test all(isfinite, mo.var)
    ws = with_logger(NullLogger()) do
        JDyn.dynare(joinpath(XD, "xdet_o1.mod"))
    end
    @test ws.M_.exo_det_nbr == 2 && vec(ws.oo_.exo_det_steady_state) == [0.0, 0.0]
    dr = JDyn.as_dr(sol)
    dr["order"] = 3
    @test_throws JDyn.ModelError JDyn.exo_det_decision_rules(m, dr)
    @test_throws JDyn.ModelError JDyn.forecast(m; periods=3, order=3, dr=dr)
    @test_throws JDyn.ModelError JDyn.forecast(m; periods=3, order=2, pruning=true, dr=dr)
    @test_throws JDyn.PerfectForesightError JDyn.perfect_foresight(m; periods=20)
    # a model without varexo_det has none
    m0 = build("var x; varexo e; parameters r; r = 0.5; model; x = r*x(-1) + e; end; shocks; var e; stderr 1; end;")
    @test JDyn.n_exo_det(m0) == 0
    @test_throws JDyn.ModelError JDyn.exo_det_decision_rules(m0, JDyn.as_dr(JDyn.first_order(m0)))
end

fc_fields(r, name) = (haskey(r, name) ? r[name] : Dict{String,Any}())
@testset "forecast with varexo_det against Dynare (fc_xdet_o1)" begin
    m = xd_load("fc_xdet_o1")
    r = matread(joinpath(XD, "fc_xdet_o1.mat"))
    ys = vec(Float64.(r["ys_"]))
    dr = JDyn.as_dr(JDyn.first_order(m; ys=ys))
    worst = 0.0
    cases = [("fc1", ["y", "c", "k"], 2, 0.9, false), ("fc2", String[], 12, 0.7, false), ("fc3", String[], 0, 0.7, false),
             ("fc4", ["y", "c", "k"], 6, 0.95, true)]
    for (name, vl, periods, conf, me) in cases
        out = JDyn.forecast(m; var_list=vl, periods=periods, conf_sig=conf, dr=dr, y0=ys)
        ref = r[name]
        for f in ("Mean", "HPDinf", "HPDsup", "Exogenous")
            got = f == "Exogenous" ? out.Exogenous : getfield(out, Symbol(f))
            @test Set(keys(got)) == Set(keys(ref[f]))
            for (v, x) in got
                e = xd_close(x, ref[f][v]); worst = max(worst, e)
                @test e < 1e-11
            end
        end
        me && for f in ("HPDinf_ME", "HPDsup_ME")
            got = getfield(out, Symbol(f))
            @test Set(keys(got)) == Set(keys(ref[f]))
            for (v, x) in got
                e = xd_close(x, ref[f][v]); worst = max(worst, e)
                @test e < 1e-11
            end
        end
    end
    println("forecast with varexo_det (order 1): worst scaled difference $worst")
end

@testset "forecast with varexo_det at order 2, Dynare's draws replayed (fc_xdet_o2)" begin
    if !JDyn.korder_available()
        @warn "k-order library missing: skipped"
    else
        m = xd_load("fc_xdet_o2")
        r = matread(joinpath(XD, "fc_xdet_o2.mat"))
        ys = vec(Float64.(r["ys_"]))
        dr = JDyn.k_order(m; order=2, ys=ys)
        worst = 0.0
        for (name, vl, periods, conf, seed, me) in (("fc1", ["y", "c", "k", "a"], 5, 0.8, 5, false), ("fc2", ["y", "c", "k"], 7, 0.9, 6, true))
            out = JDyn.forecast(m; var_list=vl, periods=periods, conf_sig=conf, order=2, replic=150, rng=JDyn.OctaveRNG(seed), dr=dr, y0=ys)
            ref = r[name]
            for f in ("Mean", "HPDinf", "HPDsup", "Exogenous", (me ? ("HPDinf_ME", "HPDsup_ME") : ())...)
                got = f == "Exogenous" ? out.Exogenous : getfield(out, Symbol(f))
                @test Set(keys(got)) == Set(keys(ref[f]))
                for (v, x) in got
                    e = xd_close(x, ref[f][v]); worst = max(worst, e)
                    @test e < 1e-11
                end
            end
        end
        println("forecast with varexo_det (order 2, replayed draws): worst scaled difference $worst")
    end
end
