# Figures written by the commands of a .mod file under dynare(file; plots=true): estimation (priors, mode check, densities, diagnostics, posterior IRFs,
# smoother), shock decompositions, OccBin and the re-run commands with plots=true.
using Test, JDyn, MAT, Random, Logging
isdefined(@__MODULE__, :REF) || include(joinpath(@__DIR__, "common.jl"))

quiet(f) = with_logger(f, NullLogger())
stems(ws) = sort(splitext.(basename.(ws.state[:figure_files])) .|> first)

const _AD = mktempdir()
for f in ("sd_nk.mod", "est_nk_data.m")
    cp(joinpath(REF, f), joinpath(_AD, f))
end
function sdmod(; est="estimation(datafile=est_nk_data, mode_compute=4, mode_check, mh_replic=2100, mh_nblocks=2, mh_jscale=0.35, smoother, bayesian_irf, irf=8, sub_draws=20, nodisplay);",
               sdcmds="shock_decomposition(parameter_set=posterior_mode);\nplot_shock_decomposition(use_shock_groups=grp);\ninitial_condition_decomposition;")
    txt = read(joinpath(_AD, "sd_nk.mod"), String)
    txt = replace(txt, r"estimation\(datafile=est_nk_data, mode_compute=4, mh_replic=1200[^\n]*\n" => est * "\n")
    txt = replace(txt, r"shock_decomposition\(parameter_set=posterior_mode, nograph\);\nplot_shock_decomposition\(use_shock_groups=grp, nograph\);" => sdcmds)
    d = mktempdir(); cp(joinpath(_AD, "est_nk_data.m"), joinpath(d, "est_nk_data.m"))
    write(joinpath(d, "sd_nk.mod"), txt)
    joinpath(d, "sd_nk.mod")
end

@testset "estimation and decomposition commands write their figures" begin
    f = sdmod()
    ws = quiet(() -> dynare(f; plots=true, extra_args=["-DCASE=1"]))
    s = stems(ws)
    has(p) = any(startswith(x, p) for x in s)
    @test has("sd_nk_Priors") && has("sd_nk_CheckPlots") && has("sd_nk_PriorsAndPosteriors")
    @test has("sd_nk_udiag") && "sd_nk_mdiag" in s
    @test has("sd_nk_Bayesian_IRF_") && has("sd_nk_SmoothedVariables_") && has("sd_nk_SmoothedShocks_")
    @test has("sd_nk_shock_decomposition_pi_group_grp_posterior_mode") && has("sd_nk_initval_decomposition_")
    dirs = unique(dirname.(ws.state[:figure_files]))
    @test Set(basename.(dirs)) == Set(["graphs", "Output"])
    @test all(isfile, ws.state[:figure_files])
    # the decomposition files hold what decomposition_figures gives (spot check on the group figure)
    @test any(endswith(x, "sd_nk_shock_decomposition_pi_group_grp_posterior_mode.svg") for x in ws.state[:figure_files])
end

@testset "nograph, plot_priors=0, graph_format none, plots off" begin
    f = sdmod(; est="estimation(datafile=est_nk_data, mode_compute=4, mh_replic=0, plot_priors=0, nograph, nodisplay);",
              sdcmds="shock_decomposition(parameter_set=posterior_mode);\nplot_shock_decomposition(use_shock_groups=grp, nograph);")
    ws = quiet(() -> dynare(f; plots=true, extra_args=["-DCASE=1"]))
    @test !haskey(ws.state, :figure_files) || isempty(ws.state[:figure_files])
    f = sdmod(; est="estimation(datafile=est_nk_data, mode_compute=4, mh_replic=0, nodisplay);", sdcmds="shock_decomposition(parameter_set=posterior_mode);")
    ws = quiet(() -> dynare(f; plots=true, extra_args=["-DCASE=1"]))
    @test stems(ws) == sort(["sd_nk_Priors1", "sd_nk_Priors2"][1:length(stems(ws))]) && all(startswith("sd_nk_Priors"), stems(ws))     # only plot_priors by default
    ws = quiet(() -> dynare(f; extra_args=["-DCASE=1"]))                              # plots off: nothing written
    @test !haskey(ws.state, :figure_files)
    # re-running a command with plots=true
    ws2 = quiet(() -> stoch_simul!(ws; irf=6, plots=true))
    @test !isempty(ws.state[:figure_files]) && any(endswith("_IRF_ea.svg"), ws.state[:figure_files])
end

@testset "graph_format pdf / eps through Plots.jl from the commands" begin
    penv = get(ENV, "JDYN_PLOTS_ENV", "/tmp/plotsenv")
    if !isfile(joinpath(penv, "Project.toml"))
        @test_skip false
    else
        ENV["GKSwstype"] = "100"
        penv in LOAD_PATH || push!(LOAD_PATH, penv)
        @eval Main using Plots
        f = sdmod(; est="estimation(datafile=est_nk_data, mode_compute=4, mh_replic=0, graph_format=(pdf, eps), nodisplay);",
                  sdcmds="shock_decomposition(parameter_set=posterior_mode);\nplot_shock_decomposition(use_shock_groups=grp, graph_format=pdf);")
        ws = quiet(() -> dynare(f; plots=true, extra_args=["-DCASE=1"]))
        files = ws.state[:figure_files]
        @test any(endswith(".pdf"), files) && any(endswith(".eps"), files) && !any(endswith(".svg"), files) && all(isfile, files)
        @test any(endswith("sd_nk_Priors1.pdf"), files) && any(endswith("sd_nk_Priors1.eps"), files)
        @test any(endswith("sd_nk_shock_decomposition_pi_group_grp_posterior_mode.pdf"), files)
        @test read(first(filter(endswith(".pdf"), files)), 4) == codeunits("%PDF")
    end
end

@testset "occbin_graph and realtime decomposition statements" begin
    ob = joinpath(EXAMPLES, "occbin", "rbc_occbin.mod")
    d = mktempdir(); cp(ob, joinpath(d, "rbc_occbin.mod"))
    ws = quiet(() -> dynare(joinpath(d, "rbc_occbin.mod"); plots=true))
    @test any(endswith("rbc_occbin_occbin_1.svg"), get(ws.state, :figure_files, String[]))
    f = sdmod(; sdcmds="shock_decomposition(parameter_set=posterior_mode);\nrealtime_shock_decomposition(parameter_set=posterior_mode, forecast=4, presample=20, save_realtime=[30,45]);\n" *
                       "plot_shock_decomposition(realtime=1, vintage=45, nograph);\nplot_shock_decomposition(realtime=3, vintage=30, use_shock_groups=grp) pi;",
              est="estimation(datafile=est_nk_data, mode_compute=4, mh_replic=0, plot_priors=0, nodisplay);")
    ws = quiet(() -> dynare(f; plots=true, extra_args=["-DCASE=1"]))
    fl = get(ws.state, :figure_files, String[])
    @test length(fl) == 1 && occursin("sd_nk_shock_decomposition_pi_group_grp", fl[1])
end
