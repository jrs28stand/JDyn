# Bit-for-bit check of the exact-class results across a code change: prints a hash of the bit patterns of the simulated paths (order 1 news, order 2, order 2 with cubature,
# LCP formulation, several models). Run it in two checkouts and compare the output:
#   julia --project=<checkout> <checkout>/bench/obc_exact_bits.jl
# (the shocks and models come from the checkout's own test folder).
using Test; include(joinpath(@__DIR__, "..", "test", "common.jl")); using Random, SHA
function h(x::AbstractMatrix)
    bytes2hex(sha1(reinterpret(UInt8, vec(Float64.(x)))))
end
# the strict rule (undecided tolerance 0) is what the code before the split rule always did: older checkouts do not have the keyword (OLD=1)
const UT = get(ENV, "OLD", "") == "1" ? NamedTuple() : (obc_cubature_undecided_tolerance=0.0,)
dir = joinpath(@__DIR__, "..", "test")
models = ("nk" => joinpath(dir, "obc", "ho", "nk_occ.mod"), "zlb" => joinpath(dir, "obc", "obc_zlb.mod"), "zlb2" => joinpath(dir, "obc", "obc_zlb2.mod"),
          "borrcon" => joinpath(dir, "obc", "ho", "occ_borrcon.mod"))
hz = Dict("nk" => 32, "zlb" => 20, "zlb2" => 20, "borrcon" => 40)
sc = Dict("nk" => 1.0, "zlb" => 0.02, "zlb2" => 0.02, "borrcon" => 0.5)
for (name, path) in models
    m = load_copy(path; output="second")
    ob = JDyn.Occbin(m)
    sd = name in ("zlb", "zlb2") ? ones(m.n_exo) : sqrt.(diag(Matrix(m.Sigma_e)))
    S = randn(MersenneTwister(4), 80, m.n_exo) .* sd' .* sc[name]
    kw = (obc_method=:news, obc_horizon=hz[name], check_ahead_periods=hz[name], periods=80, obc_formulation=:lcp)
    for (label, extra) in (("order1", NamedTuple()), ("order2", (obc_order=2,)), ("order2_cub7", (; obc_order=2, obc_cubature=true, obc_qmc_level=7, UT...)),
                           ("order2_cub5_regions1", (; obc_order=2, obc_cubature=true, obc_qmc_level=5, obc_cubature_regions=1, UT...)))
        r = try; with_logger(NullLogger()) do; JDyn.simulate(ob, S; kw..., extra...); end; catch e; e; end
        println(rpad(name, 8), rpad(label, 22), r isa Exception ? "error: " * first(sprint(showerror, r), 80) : h(r.piecewise) * " " * h(r.linear))
    end
end
