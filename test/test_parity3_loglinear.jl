# Parity batch 3, loglinear results layer: after stoch_simul(loglinear) oo_.steady_state and oo_.dr.ys are in logs,
# options_.logged_steady_state is 1, and steady / a further stoch_simul undo it. References: Dynare 7.2 under Octave 8.4
# (loglinear/example4_loglinear, final state), in test/ref_parity3/.
using Test; include("common.jl")
using Random, Logging, MAT

const R3 = joinpath(@__DIR__, "ref_parity3")
const EX4 = joinpath(DYNARE_SRC, "tests", "loglinear", "example4_loglinear.mod")

@testset "parity3 loglinear results layer" begin
    @testset "example4_loglinear vs Dynare" begin
        if isfile(EX4)
            d = mktempdir(); cp(EX4, joinpath(d, "ex4.mod"))
            ws = with_logger(NullLogger()) do
                JDyn.dynare(joinpath(d, "ex4.mod"); random=:octave)
            end
            ref = matread(joinpath(R3, "example4_loglinear_ref.mat"))
            @test vec(ws.oo_["steady_state"]) ≈ vec(ref["steady_state"]) rtol=1e-9 atol=1e-11
            @test vec(ws.oo_["dr"]["ys"]) ≈ vec(ref["drys"]) rtol=1e-9 atol=1e-11
            @test ws.oo_["dr"]["Gy"] ≈ ref["Gy"] rtol=1e-8 atol=1e-10
            @test Float64(ref["flag"]) == 1.0
        else
            @info "Dynare tests not present; example4_loglinear skipped"
        end
    end
    @testset "flag sequence stoch_simul(loglinear) / steady" begin
        mod = """
        var z c k;
        varexo e;
        parameters beta delta alpha rho;
        beta=0.99; delta=0.025; alpha=0.36; rho=0.9;
        model;
        c + k = z*k(-1)^alpha + (1-delta)*k(-1);
        1/c = beta*(alpha*z(+1)*k^(alpha-1) + 1 - delta)/c(+1);
        z = z(-1)^rho*exp(e);
        end;
        initval; z=1; k=10; c=2; e=0; end;
        steady;
        ss0 = oo_.steady_state;
        shocks; var e; stderr 0.01; end;
        stoch_simul(order=1, loglinear, irf=0, nograph, nomoments);
        f1 = options_.logged_steady_state;
        ss1 = oo_.steady_state;
        drys1 = oo_.dr.ys;
        stoch_simul(order=1, loglinear, irf=0, nograph, nomoments);
        f2 = options_.logged_steady_state;
        ss2 = oo_.steady_state;
        steady;
        f3 = options_.logged_steady_state;
        ss3 = oo_.steady_state;
        """
        d = mktempdir(); write(joinpath(d, "ll.mod"), mod)
        ws = with_logger(NullLogger()) do
            JDyn.dynare(joinpath(d, "ll.mod"))
        end
        v = ws.vars
        # steady.m leaves the flag as it was (1) but oo_.steady_state is in levels again
        @test v["f1"] == 1 && v["f2"] == 1 && v["f3"] == 1
        ss0 = vec(v["ss0"])
        @test vec(v["ss1"]) ≈ log.(ss0) rtol=1e-9 atol=1e-12
        @test vec(v["ss2"]) ≈ vec(v["ss1"]) rtol=1e-12
        @test vec(v["drys1"]) ≈ vec(v["ss1"]) rtol=1e-12
        @test vec(v["ss3"]) ≈ ss0 rtol=1e-12
    end
    @testset "estimation of a loglinear model (ll_est) vs Dynare" begin
        d = mktempdir()
        for f in ("ll_est.mod", "fsdat_simul.m"); cp(joinpath(R3, f), joinpath(d, f)); end
        ws = with_logger(NullLogger()) do
            JDyn.dynare(joinpath(d, "ll_est.mod"); random=:octave)
        end
        ref = matread(joinpath(R3, "ll_est_ref.mat"))
        # flags 1 / 0 / 1: logged after stoch_simul, levels (undone) in estimation, logged again at the next stoch_simul
        for k in ("flag1", "flag2", "flag3"); @test Float64(ws.vars[k]) == Float64(ref[k]); end
        for k in ("ss1", "drys1", "Gy1", "ss2", "drys2", "Gy2", "mode2", "ss3", "drys3", "Gy3")
            a = ref[k]; b = ws.vars[k]
            a = a isa Number ? [a] : vec(a); b = b isa Number ? [b] : vec(Float64.(b))
            @test maximum(abs.(a .- b)) / max(1, maximum(abs.(a))) < 1e-7
        end
    end
end
