# No Dynare option is silently ignored: src/option_policy.jl classifies every option of the statements JDyn runs.
using Test
import JDyn
using JDyn: OPTION_SPECS, GRAMMAR_OPTIONS, UNIMPLEMENTED_STATEMENTS, classify_option, refused_options, option_policy_message

@testset "option policy" begin
    @testset "the tables are disjoint and cover the grammar of Dynare 7.2" begin
        for (st, s) in OPTION_SPECS
            @test isempty(s.honoured ∩ s.harmless) && isempty(s.honoured ∩ s.refused) && isempty(s.harmless ∩ s.refused)
            @test s.honoured ∪ s.harmless ∪ s.refused == Set(GRAMMAR_OPTIONS[st])
        end
        # every statement of the grammar is either classified option by option or declared unimplemented
        @test Set(keys(GRAMMAR_OPTIONS)) ⊆ Set(keys(OPTION_SPECS)) ∪ UNIMPLEMENTED_STATEMENTS
        @test isempty(Set(keys(OPTION_SPECS)) ∩ UNIMPLEMENTED_STATEMENTS)
        @test sum(length(v) for v in values(GRAMMAR_OPTIONS)) >= 1170          # 1,172 option keys in the grammar of Dynare 7.2
    end
    @testset "classification" begin
        @test classify_option("stoch_simul", "order") === :honoured
        @test classify_option("stoch_simul", "simul_replic") === :honoured
        @test classify_option("stoch_simul", "SpectralDensity.trigger") === :honoured
        @test classify_option("stoch_simul", "nograph") === :harmless
        @test classify_option("stoch_simul", "partial_information") === :refused
        @test classify_option("stoch_simul", "an_option_of_a_later_dynare") === :refused
        @test classify_option("plot_shock_decomposition", "plot_shock_decomp.colormap") === :harmless
        @test classify_option("stoch_simul", "irf_opt.EM.tolf") === :harmless                 # no consumer in Dynare 7.2
        @test classify_option("sensitivity", "rmse") === nothing
        msg = option_policy_message("estimation", Dict{String,Any}("skewed_kalman.mvnlogcdf" => 1, "order" => 1))
        @test msg !== nothing && occursin("skewed_kalman.mvnlogcdf", msg) && !occursin("order (", msg)
    end
    @testset "value-aware options" begin
        ok = Dict{String,Any}("dr_cycle_reduction" => "true", "dr_cycle_reduction_tol" => "1e-7", "dr_cycle_reduction_maxiter" => 300,
                              "dr_logarithmic_reduction" => "true", "dr_logarithmic_reduction_tol" => "1e-12")
        @test isempty(refused_options("stoch_simul", ok))
        for (k, v) in ("dr_cycle_reduction_tol" => "1e-6", "dr_cycle_reduction_maxiter" => 50, "dr_logarithmic_reduction_tol" => "1e-8")
            @test [x[1] for x in refused_options("stoch_simul", Dict{String,Any}(k => v))] == [k]
        end
        @test isempty(refused_options("estimation", Dict{String,Any}("lyapunov_db" => "true")))
        @test isempty(refused_options("estimation", Dict{String,Any}("lyapunov_srs" => "true")))
        @test !isempty(refused_options("estimation", Dict{String,Any}("lyapunov_fp" => "true")))      # 1.6e-8 off at the default tolerance
        @test isempty(refused_options("steady", Dict{String,Any}("fsolve_options" => "''Display'',''iter'',''AutoScaling'',''on''")))
        @test !isempty(refused_options("steady", Dict{String,Any}("fsolve_options" => "''TolFun'',1e-3")))
        @test !isempty(refused_options("check", Dict{String,Any}("fsolve_options" => "''Display'',''iter'',''MaxIter'',3")))
        @test !isempty(refused_options("estimation", Dict{String,Any}("particle.number_of_particles" => 100, "order" => 1)))     # order 2 only
        @test isempty(refused_options("estimation", Dict{String,Any}("particle.number_of_particles" => 100, "order" => 2)))
    end
    @testset "dynare() acts on the table" begin
        d = mktempdir()
        mod(stmt) = """var y c k a; varexo e;
        parameters alpha beta delta rho;
        alpha=0.36;beta=0.99;delta=0.025;rho=0.9;
        model;
        c^(-1)=beta*c(+1)^(-1)*(alpha*a(+1)*k^(alpha-1)+1-delta);
        k=a*k(-1)^alpha+(1-delta)*k(-1)-c;
        y=a*k(-1)^alpha;
        log(a)=rho*log(a(-1))+e;
        end;
        initval; k=30;c=2.5;a=1;y=3;e=0;end;
        steady;
        shocks; var e; stderr 0.01; end;
        $stmt
        """
        write(joinpath(d, "m.mod"), mod("stoch_simul(order=1,irf=0,periods=50,partial_information,nograph);"))
        err = try JDyn.dynare(joinpath(d, "m.mod")); nothing catch e; e end
        @test err isa JDyn.ModelError && occursin("partial_information", err.msg) && occursin("stoch_simul", err.msg)
        write(joinpath(d, "m.mod"), mod("stoch_simul(order=1,irf=0,nograph,nomodelsummary,dr=cycle_reduction,dr_cycle_reduction_tol=1e-9);"))
        ws = JDyn.dynare(joinpath(d, "m.mod"))
        @test size(ws.oo_.dr.ghx, 1) == 4
        # simul_replic: oo_.endo_simul is the first replication, oo_.exo_simul the last; the raw data go to <fname>/Output/<fname>_simul
        write(joinpath(d, "m.mod"), mod("stoch_simul(order=1,irf=0,periods=30,simul_replic=3,nograph);"))
        ws = JDyn.dynare(joinpath(d, "m.mod"))
        raw = reinterpret(Float64, read(joinpath(d, "m", "Output", "m_simul")))
        @test length(raw) == 3 * 4 * 30
        @test reshape(raw, 4, 30, 3)[:, :, 1] ≈ ws.oo_.endo_simul
        write(joinpath(d, "m.mod"), mod("stoch_simul(order=1,irf=0,periods=30,nograph);"))
        ws1 = JDyn.dynare(joinpath(d, "m.mod"))
        @test ws1.oo_.endo_simul ≈ ws.oo_.endo_simul                                  # the first replication is the single run
        # SpectralDensity.trigger: filled at order 1, silent at order 2 (UnivariateSpectralDensity.m)
        write(joinpath(d, "m.mod"), mod("stoch_simul(order=1,irf=0,spectral_density,filtered_theoretical_moments_grid=64,nograph);"))
        ws = JDyn.dynare(joinpath(d, "m.mod"))
        @test size(ws.oo_.SpectralDensity.density) == (4, 64) && length(ws.oo_.SpectralDensity.freqs) == 64
        write(joinpath(d, "m.mod"), mod("stoch_simul(order=2,irf=0,spectral_density,nograph);"))
        @test !haskey(JDyn.dynare(joinpath(d, "m.mod")).oo_, "SpectralDensity")
        # contemporaneous_correlation fills oo_ (nocorr suppresses it)
        write(joinpath(d, "m.mod"), mod("stoch_simul(order=1,irf=0,contemporaneous_correlation,nograph);"))
        C = JDyn.dynare(joinpath(d, "m.mod")).oo_.contemporaneous_correlation
        @test size(C) == (4, 4) && all(isnan(C[i, i]) || C[i, i] ≈ 1 for i in 1:4)
    end
end
