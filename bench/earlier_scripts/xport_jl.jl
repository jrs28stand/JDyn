include("/home/claude/work/jdyn-ho/test/common.jl")
using JDyn, Logging, Random, Printf, MAT
function sw_cp_model()
    src = read("/home/claude/work/jdyn-ho/test/obc/obc_sw_zlb.mod", String)
    a = first(findfirst("[name = 'Auxiliary notional interest rate',bind='zlb']", src))
    b = first(findfirst("    [name='Output gap check']", src))
    d = mktempdir()
    for f in readdir("/home/claude/work/jdyn-ho/test/obc"); isfile(joinpath("/home/claude/work/jdyn-ho/test/obc", f)) && cp(joinpath("/home/claude/work/jdyn-ho/test/obc", f), joinpath(d, f); force=true); end
    write(joinpath(d, "sw_cp.mod"), src[1:a-1] * "[name = 'Auxiliary notional interest rate']\nrlag = rnot;\n" * src[b:end])
    write("/tmp/ho/sw_cp.mod", read(joinpath(d, "sw_cp.mod")))
    with_logger(NullLogger()) do; JDyn.load(joinpath(d, "sw_cp.mod")); end
end
cases = [("nk", "/home/claude/work/jdyn-ho/test/obc/ho/nk_occ.mod", 40, 9, 0.0), ("nk", "/home/claude/work/jdyn-ho/test/obc/ho/nk_occ.mod", 40, 11, 0.0),
         ("nkm", "/home/claude/work/jdyn-ho/test/ref_occbin/occest_nkm.mod", 40, 9, 0.0), ("nkm", "/home/claude/work/jdyn-ho/test/ref_occbin/occest_nkm.mod", 40, 11, 0.0),
         ("zlb2", "/home/claude/work/jdyn-ho/test/obc/obc_zlb2.mod", 24, 7, 5.0), ("borrcon", "/home/claude/work/jdyn-ho/test/obc/ho/occ_borrcon.mod", 32, 5, 0.1), ("sw_cp", nothing, 32, 5, 0.25)]
for (name, file, hor, L, sc) in cases
    m = file === nothing ? sw_cp_model() : load_copy(file)
    ob = JDyn.Occbin(m)
    sd = sqrt.(diag(Matrix(m.Sigma_e)))
    S = name == "nk" ? reshape(vec(matread("/home/claude/work/jdyn-ho/test/obc/ho/shocks.mat")["ShockSequence"]), :, 1)[1:20, :] :
        randn(MersenneTwister(4), 20, length(sd)) .* sd' .* (name in ("nkm", "zlb2") ? 5.0 : sc)
    h = JDyn.obc_ho(ob; order=2, horizon=hor, check_ahead=hor, cubature=true, qmc_level=L, regions=0, m_diagnostic=false, cubature_failure_tolerance=0.95)
    h.cub.qlog = Vector{Float64}[]
    with_logger(NullLogger()) do; JDyn.obc_ho_simulate(h, S); end
    qs = h.cub.qlog; h.cub.qlog = nothing
    out = Dict{String,Any}("R" => h.R, "xi" => h.xi, "Q" => reduce(hcat, qs), "T" => hor, "N" => 2hor)
    Y0 = fill(NaN, h.nT, length(qs)); Y9 = zeros(h.nT, length(qs)); C9 = zeros(5, length(qs)); F = fill(-1.0, 7, length(qs))   # failure info: a b c n_failed lower_bound n_as relevant
    for (k, q) in enumerate(qs)
        h.cub.tolerance = 0.0
        try; Y0[:, k] = JDyn.ho_news_shocks(h, q, k)
        catch e
            e isa JDyn.ObcError && e.detail isa JDyn.CubatureFailure || rethrow()
            d = e.detail; F[:, k] = [d.counts.a, d.counts.b, d.counts.c, d.n_failed, d.lower_bound, something(d.n_active_set_failed, -1), d.n_relevant]
        end
        h.cub.tolerance = 0.95; foreach(kk -> h.stats[kk] = 0, collect(keys(h.stats)))
        try; Y9[:, k] = with_logger(NullLogger()) do; JDyn.ho_news_shocks(h, q, k); end; catch e; Y9[:, k] .= NaN; end
        C9[:, k] = [get(h.stats, :failed_a, 0), get(h.stats, :failed_b, 0), get(h.stats, :failed_c, 0), get(h.stats, :dropped_points, 0), get(h.stats, :batch_fail, 0)]
    end
    out["Y0"] = Y0; out["Y9"] = Y9; out["F"] = F; out["C9"] = C9
    dims = Dict{String,Any}()
    for (kk, v) in out
        v isa AbstractArray || continue
        dims[kk] = collect(size(v)); open("/tmp/ho/xp_$(name)_$(L)_$kk.bin", "w") do io; write(io, Float64.(v)); end
    end
    open("/tmp/ho/xp_$(name)_$(L).json", "w") do io; JDyn.JSON.print(io, dims); end
    println(name, " L", L, ": ", length(qs), " q, failing under tol 0: ", count(>=(0), F[1, :]), " ; points ", size(h.xi, 2), " dim ", size(h.R, 2))
end
