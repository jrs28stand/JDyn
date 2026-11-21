# Dynare 7.2's shipped examples, unedited, against Dynare's results at first order
# (port of pydyn tests/test_examples.py; the comparison is in sweep_examples.jl).
using Test
include("common.jl")
include("sweep_examples.jl")

const EXPECTED = Dict(
    "estimation/gali_2015" => MATCH,
    "estimation/rbc_irf_matching" => "solved, no reference",      # Dynare's run stops before stoch_simul here
    "estimation/schorfheide_2000" => MATCH,
    "macroprocessor/bkk_1992" => MATCH,
    "occbin/rbc_occbin" => MATCH,
    "reporting/collard_2001_reporting" => MATCH1,                  # order=2 in the .mod file
    "stochastic_simulations/aguiar_gopinath_2007_trend" => MATCH,
    "stochastic_simulations/collard_2001_analytical_steady_state" => MATCH1,   # uses its .m helper function
    "stochastic_simulations/collard_2001_simulated_moments" => MATCH1,
    "stochastic_simulations/collard_2001_theoretical_moments" => MATCH1,
    "stochastic_simulations/nk_baseline" => MATCH,                              # uses its _steadystate.m file
    "stochastic_simulations/schorfheide_2000_nonstationary" => MATCH1,
    "heterogeneity/hank_one_asset" => "not supported",
    "optimal_policy/nk_ramsey_osr" => "not supported",
    "semistructural/pac_model" => "not supported",
    "perfect_foresight/perfect_foresight_rbc" => MATCH,                         # perfect_foresight_solver (src/perfect_foresight.jl)
    "perfect_foresight/perfect_foresight_expectation_errors" => MATCH,
)

@testset "Dynare examples" begin
    for ex in sort(collect(keys(EXPECTED)))
        group, name = split(ex, "/")
        out = check_shipped_example(group, name)
        println(rpad(ex, 60), " | ", rpad(out["status"], 38), " | ", out["detail"])
        @test out["status"] == EXPECTED[ex]
        if EXPECTED[ex] in (MATCH, MATCH1)
            @test isfile(joinpath(EXAMPLE_REFS, "$(group)__$(name).mat"))
        end
    end
end
