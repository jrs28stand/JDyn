# Run the first-order test files in one Julia session (each in its own module, as when run alone):
#   julia --project=. test/runtests_numerics.jl [file ...]
const NUMERICS_TESTS = ["test_first_order.jl", "test_examples.jl", "test_loop.jl", "test_moments.jl",
                        "test_auxiliary.jl", "test_loglinear.jl", "test_review.jl"]
failed = String[]
for f in (isempty(ARGS) ? NUMERICS_TESTS : ARGS)
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
