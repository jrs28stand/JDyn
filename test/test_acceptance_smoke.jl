# Smoke subset of the acceptance sweep (docs/ACCEPTANCE.md): 20 Dynare 7.2 test files that ran cleanly (category OK)
# through JDyn.dynare must keep doing so. Needs the Dynare 7.2 source tree (DYNARE_TESTS, default
# /root/ref/dynare-7.2/tests) and the preprocessor; skipped otherwise.
using Test, Logging
using JDyn

const _TESTS = get(ENV, "DYNARE_TESTS", "/root/ref/dynare-7.2/tests")
const _SMOKE = [
    "lmmcp/purely_forward.mod",
    "on-the-fly/ex3.mod",
    "equation_tags/ramst_static_tag_block.mod",
    "stochastic_simulations/example1_irf_shocks.mod",
    "macro_processor/example1_macro.mod",
    "stat_functions/ramst_normcdf_and_friends_mfile.mod",
    "preprocessor_checks/comments.mod",
    "expectations/expectation.mod",
    "steady_state/walsh1_initval.mod",
    "steady_state_operator/standard.mod",
    "fs2000/fs2000c.mod",
    "deterministic_simulations/initval_endval_blocks/initval_endval_test.mod",
    "trend_var/fs2000_log_nonstationary.mod",
    "stochastic_purely_forward/stochastic_purely_forward_with_static.mod",
    "dynare-command-options/ramst.mod",
    "../examples/stochastic_simulations/collard_2001_theoretical_moments.mod",
    "../examples/stochastic_simulations/collard_2001_analytical_steady_state.mod",
    "../examples/stochastic_simulations/nk_baseline.mod",
    "../examples/perfect_foresight/perfect_foresight_rbc.mod",
    "../examples/occbin/rbc_occbin.mod"]

@testset "acceptance smoke (Dynare's own test files)" begin
    if !isdir(_TESTS) || (get(ENV, "DYNARE_PREPROCESSOR", "") == "" && Sys.which("dynare-preprocessor") === nothing)
        @info "Dynare test tree or preprocessor missing: acceptance smoke test skipped"
    else
        manifest = Dict{String,Vector{String}}()
        for l in eachline(joinpath(@__DIR__, "acceptance", "manifest.tsv"))
            (isempty(l) || l[1] == '#') && continue
            f = split(l, '\t'; keepempty=true)
            manifest[f[1]] = vcat(split(f[2], ';'), isempty(f[3]) ? String[] : split(f[3], ';'))
        end
        for id in _SMOKE
            scratch = mktempdir(); base = joinpath(scratch, "tests"); mkpath(base)
            for f in manifest[id]
                src = normpath(joinpath(_TESTS, f)); dst = normpath(joinpath(base, f))
                isfile(src) && (mkpath(dirname(dst)); cp(src, dst))
            end
            path = joinpath(base, id)
            cd(dirname(path)) do
                @test (with_logger(NullLogger()) do; JDyn.dynare(path); end) isa JDyn.DynareWorkspace
            end
            rm(scratch; recursive=true, force=true)
        end
    end
end
