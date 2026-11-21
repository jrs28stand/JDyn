# Precompilation workload. Without it the first dynare() call of a Julia session spends minutes compiling JDyn
# (the model loader, the MATLAB-subset interpreter, the steady-state solver, perturbation, moments, ...); code
# compiled while the package is precompiled is stored in the package image. No PrecompileTools dependency: the
# statements run under `jl_generating_output`, i.e. only while the image is built. The model is a small RBC whose
# preprocessor output (Dynare 7.2) is kept in src/precompile_fixture/, so no preprocessor is needed at that time;
# its generated functions are loaded into the package's own modules (_WorkloadModel, see model.jl).
function _precompile_workload()
    fx = joinpath(@__DIR__, "precompile_fixture")      # w.mod preprocessed once (Dynare 7.2, output=second): no preprocessor needed
    try
        Logging.with_logger(Logging.NullLogger()) do
            ws = dynare(joinpath(fx, "w.mod"); output="second", _json_dir=joinpath(fx, "w", "model", "json"))
            set_param_value!(ws, "rho", 0.8)
            stoch_simul!(ws; order=1, irf=10, periods=0)
            stoch_simul!(ws; order=2, irf=5, periods=300)
            steady!(ws)
            first_order(ws.model)
        end
    catch e
        @warn "JDyn precompile workload failed" exception=(e, catch_backtrace())
    end
    nothing
end

ccall(:jl_generating_output, Cint, ()) == 1 && _precompile_workload()
