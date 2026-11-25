# The `data` command with dates (first_obs/last_obs/nobs), start date from set_time or the file; Octave 8.4 on fs2000_data used nobs=40
# with first_obs=1971Q1 and set_time(1970Q3): the mode (alp 0.3570877) was reproduced to 2e-5 (docs/GAPS_PROGRESS.md).
using Test
import JDyn
ws(state) = (state=state, model=(mod_dir=joinpath(@__DIR__, "ref_gaps", "data"),))
@testset "data command dates" begin
    kw = Dict{Symbol,Any}(); st = Dict{Symbol,Any}(:dataset => Dict{String,Any}("file" => "fsdat_simul.m", "first_obs" => "dates('1971Q1')", "nobs" => 40),
                                                    :set_time => "dates('1970Q3')")
    JDyn._apply_dataset!(ws(st), kw)
    @test kw[:first_obs] == 3 && kw[:nobs] == 40 && kw[:datafile] == "fsdat_simul.m"
    st[:dataset] = Dict{String,Any}("file" => "fsdat_simul.m", "first_obs" => "dates('1971Q1')", "last_obs" => "dates('1972Q4')")
    kw = Dict{Symbol,Any}(:nobs => 192); JDyn._apply_dataset!(ws(st), kw)
    @test kw[:first_obs] == 3 && kw[:nobs] == 8
    delete!(st, :set_time)
    @test_throws JDyn.EstimationError JDyn._apply_dataset!(ws(st), Dict{Symbol,Any}())
    st[:dataset] = Dict{String,Any}("file" => "fsdat_simul.m", "first_obs" => 5)
    kw = Dict{Symbol,Any}(); JDyn._apply_dataset!(ws(st), kw); @test kw[:first_obs] == 5 && !haskey(kw, :nobs)
end
