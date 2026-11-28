# run_matlab / @run_matlab: MATLAB driver scripts that call Dynare
include("common.jl")

function _driver_dir()
    d = mktempdir()
    write(joinpath(d, "ar1.mod"), """
    @#ifndef horizon
    @#define horizon = 5
    @#endif
    var y; varexo e; parameters rho;
    rho = 0.5;
    set_param_value('rho', rho_val);
    model; y = rho*y(-1) + e; end;
    shocks; var e; stderr 1; end;
    stoch_simul(order=1, irf=@{horizon}, nograph);
    """)
    write(joinpath(d, "driver.m"), """
    rho_val = 0.8;
    dynare ar1 noclearall -Dhorizon=7
    g1 = oo_.dr.ghx;
    n1 = length(oo_.irfs.y_e);
    rho_val = 0.4;
    dynare('ar1', 'noclearall');
    g2 = oo_.dr.ghx;
    n2 = length(oo_.irfs.y_e);
    ratio = g1 / g2;
    """)
    write(joinpath(d, "loop.m"), """
    rho_val = 0.8;
    dynare ar1 noclearall
    vals = [0.2 0.6];
    g = zeros(1, 2);
    for i = 1:2
        set_param_value('rho', vals(i));
        [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, {'y'});
        g(i) = oo_.dr.ghx;
    end
    M_.params(1) = 0.3;
    stoch_simul('y');
    g3 = oo_.dr.ghx;
    """)
    write(joinpath(d, "clears.m"), """
    rho_val = 0.8;
    dynare ar1
    """)
    d
end

@testset "run_matlab: driver script with noclearall, -D and post-processing" begin
    d = _driver_dir()
    r = with_logger(NullLogger()) do
        run_matlab(joinpath(d, "driver.m"))
    end
    v = r.vars
    @test v["g1"][1] ≈ 0.8 && v["g2"][1] ≈ 0.4 && v["ratio"][1] ≈ 2.0
    @test v["n1"] == 7 && v["n2"] == 5
    @test length(r.workspaces) == 2 && r.workspaces[2].oo_.dr.ghx[1] ≈ 0.4
    @test v["oo_"] === r.workspaces[2].oo_ && JDyn.CURRENT[] === r.workspaces[2]
    # without noclearall the script's variables are cleared first, as Dynare does: rho_val is gone
    @test_throws Exception with_logger(NullLogger()) do
        run_matlab(joinpath(d, "clears.m"))
    end
    # @run_matlab defines the script's variables in the caller's module
    m = Module(:RMTest)
    Core.eval(m, :(using JDyn))
    with_logger(NullLogger()) do
        Core.eval(m, :(@run_matlab $(joinpath(d, "driver.m"))))
    end
    @test Core.eval(m, :(ratio))[1] ≈ 2.0 && Core.eval(m, :(oo_.dr.ghx))[1] ≈ 0.4
end

@testset "run_matlab: Dynare commands from the script (set_param_value, stoch_simul, M_ edits)" begin
    d = _driver_dir()
    r = with_logger(NullLogger()) do
        run_matlab(joinpath(d, "loop.m"))
    end
    @test r.vars["g"] ≈ [0.2 0.6]
    @test r.vars["g3"][1] ≈ 0.3
    @test r.workspaces[1].M_.params[1] ≈ 0.3
end
