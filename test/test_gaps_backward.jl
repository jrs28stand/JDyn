# Backward models (simul_backward_model, set_historical_values, auxiliary variables) against Octave 8.4 / Dynare 7.2 under random=:octave
# (the very Octave draws): model-inversion/bk-3a, bk-7, bk-diff-2.
using Test, MAT
import JDyn
@testset "backward models" begin
    dir = joinpath(@__DIR__, "ref_gaps", "backward"); ref = matread(joinpath(dir, "ref.mat"))
    function run(mod; bugs=false)
        d = mktempdir(); txt = read(joinpath(dir, mod), String)
        bugs && (txt = replace(txt, "TrueData = " => "options_.replicate_dynare_bugs = true;\nTrueData = "; count=1))
        write(joinpath(d, "varmodel.mod"), txt)
        cd(() -> JDyn.dynare(joinpath(d, "varmodel.mod"); random=:octave), d)
    end
    function cmp(ws, k)
        e = ws.oo_.endo_simul; x = ws.oo_.exo_simul
        r = ref[k*"_endo"]; rx = ref[k*"_exo"]
        (size(e) == size(r) && length(x) == length(rx)) || return Inf
        max(maximum(abs.(e .- r)), maximum(abs.(vec(x) .- vec(rx))))
    end
    @test cmp(run("bk3a_varmodel.mod"), "bk3a") < 1e-10                     # measured 3e-15
    @test cmp(run("bkdiff2_varmodel.mod"), "bkdiff2") < 1e-10               # diff auxiliary variables
    # bk-7: Dynare 7.2 reads the initial value of a lagged exogenous auxiliary variable from another variable's history; reproduced on request,
    # corrected by default (0.1 away in the initial period of the auxiliary).
    @test cmp(run("bk7_varmodel.mod"; bugs=true), "bk7") < 1e-10
    @test cmp(run("bk7_varmodel.mod"), "bk7") > 1e-3
end
