# Run all test files in one Julia session (each in its own module, as when run alone):
#   julia --project=. test/runtests.jl [file ...]      (or Pkg.test())
const ALL_TESTS = ["test_matlab.jl", "test_first_order.jl", "test_examples.jl", "test_loop.jl", "test_moments.jl",
                        "test_auxiliary.jl", "test_loglinear.jl", "test_review.jl", "test_plotting.jl", "test_korder.jl", "test_moments_order2.jl",
                   "test_occbin.jl", "test_occbin_options.jl", "test_estimation.jl",
                   "test_perfect_foresight.jl", "test_smoother.jl", "test_shock_decomposition.jl"]
failed = String[]
for f in (isempty(ARGS) ? ALL_TESTS : ARGS)
    println("=== ", f)
    mod = Module(Symbol(splitext(f)[1]))
    Core.eval(mod, :(include(x) = Base.include($mod, x)))
    try
        Base.include(mod, joinpath(@__DIR__, f))
    catch e
        push!(failed, f)
        println("FAILED: ", first(sprint(showerror, e), 500))
    end
end
isempty(failed) || (println("failed: ", join(failed, ", ")); exit(1))
println("all passed")
