# Dynare 7.2's shipped examples, unedited, against Dynare's results at first order (and at order 2 where the .mod file asks for it)
# (port of pydyn tests/test_examples.py; the comparison is in sweep_examples.jl).
using Test
include("common.jl")
include("sweep_examples.jl")

# examples run at order 2 are compared at order 2 as well when the k-order library is built
const MATCH2 = JDyn.korder_available() ? MATCH : MATCH1

const EXPECTED = Dict(
    "estimation/gali_2015" => MATCH,
    "estimation/rbc_irf_matching" => "solved, no reference",      # Dynare's run stops before stoch_simul here
    "estimation/schorfheide_2000" => MATCH,
    "macroprocessor/bkk_1992" => MATCH,
    "occbin/rbc_occbin" => MATCH,
    "reporting/collard_2001_reporting" => MATCH2,                  # order=2 in the .mod file
    "stochastic_simulations/aguiar_gopinath_2007_trend" => MATCH,
    "stochastic_simulations/collard_2001_analytical_steady_state" => MATCH2,   # uses its .m helper function
    "stochastic_simulations/collard_2001_simulated_moments" => MATCH2,
    "stochastic_simulations/collard_2001_theoretical_moments" => MATCH2,
    "stochastic_simulations/nk_baseline" => MATCH,                              # uses its _steadystate.m file
    "stochastic_simulations/schorfheide_2000_nonstationary" => MATCH2,
    "heterogeneity/hank_one_asset" => "not supported",
    "optimal_policy/nk_ramsey_osr" => "not supported",                          # loads and runs now (test_optimal_policy.jl checks optimal policy against Dynare); this sweep compares first-order rules only
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
        EXPECTED[ex] == "load failed" && @test occursin("not supported by JDyn", out["detail"])   # a clear refusal, not a crash
        if EXPECTED[ex] in (MATCH, MATCH1)
            @test isfile(joinpath(EXAMPLE_REFS, "$(group)__$(name).mat"))
        end
    end
end
