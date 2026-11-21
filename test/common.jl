# Shared test set-up. References are pydyn's (Dynare 7.2 under Octave 8.4): set JDYN_REF to the pydyn
# repository's reference/ folder, JDYN_EXAMPLES to Dynare 7.2's examples/ folder, JDYN_DYNARE_SRC to the
# Dynare 7.2 source tree (for the MATLAB corpus) and DYNARE_PREPROCESSOR to the 7.2 preprocessor.
using Test, JDyn, MAT, LinearAlgebra, Logging

const REF = get(ENV, "JDYN_REF", joinpath(@__DIR__, "..", "..", "jrs28stand", "pydyn", "reference"))
const DYNARE_SRC = get(ENV, "JDYN_DYNARE_SRC", "/root/ref/dynare-7.2")
const EXAMPLES = get(ENV, "JDYN_EXAMPLES", joinpath(DYNARE_SRC, "examples"))
const SS = joinpath(EXAMPLES, "stochastic_simulations")
haskey(ENV, "DYNARE_PREPROCESSOR") || (ENV["DYNARE_PREPROCESSOR"] = "/root/ref/pp-standalone/src/dynare-preprocessor")

const M = JDyn

"Copy `files` (paths) into a fresh folder and load `modname`.mod from there."
function load_copy(modpath; extra=String[], quiet=true, kw...)
    d = mktempdir()
    cp(modpath, joinpath(d, basename(modpath)))
    for f in extra
        cp(f, joinpath(d, basename(f)); force=true)
    end
    quiet ? with_logger(NullLogger()) do
        JDyn.load(joinpath(d, basename(modpath)); kw...)
    end : JDyn.load(joinpath(d, basename(modpath)); kw...)
end

"Write `text` as name.mod (plus other files) in a fresh folder and load it."
function build(text; name="m", files=Dict{String,String}(), quiet=true, kw...)
    d = mktempdir()
    for (f, c) in files
        write(joinpath(d, f), c)
    end
    write(joinpath(d, name * ".mod"), text)
    quiet ? with_logger(NullLogger()) do
        JDyn.load(joinpath(d, name * ".mod"); kw...)
    end : JDyn.load(joinpath(d, name * ".mod"); kw...)
end

mat(x::AbstractString) = isempty(x) ? zeros(0, 0) : reshape(Float64[Int(c) for c in x], 1, :)
mat(x::Real) = fill(Float64(x), 1, 1)
mat(x::AbstractVector) = reshape(Float64.(x), :, 1)
mat(x::AbstractMatrix) = Matrix{Float64}(x)
refmat(name) = matread(joinpath(REF, name))
