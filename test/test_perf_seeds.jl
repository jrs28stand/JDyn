# Performance regressions (FFT in the filtered theoretical moments) and reproducible random draws (seeds).
using Test
include("common.jl")

@testset "FFT against the direct sum" begin
    for n in (1, 2, 3, 8, 17, 100, 128, 1000, 512)
        x = randn(ComplexF64, n)
        for sgn in (+1, -1)
            ref = [sum(x[k+1] * cis(sgn * 2pi * k * t / n) for k in 0:n-1) for t in 0:n-1]
            @test maximum(abs, JDyn._dft(x, sgn) .- ref) < 1e-11 * max(1, n)
        end
    end
end
