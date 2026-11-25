# The rule memo of Occbin (`_RuleTrie`, src/occbin.jl) is bounded by `OCCBIN_RULE_MEMO_BYTES` per Occbin object and is only a memo: with any budget
# (here the default, none at all, and a few nodes) every result of the OccBin solver, the inversion filter, the piecewise Kalman filter and the smoother is
# bit for bit the same. The Python port found that at an extreme parameter vector one inversion-filter evaluation memoised ~230,000 rules (1.3 GB);
# the last testset is that vector (pydyn's soak/repro/occbin_rule_trie_memory.py). docs/OBC_HO_PROGRESS.md, "Rule memo budget".
using Test; include("common.jl")
using Random

const OREF = joinpath(@__DIR__, "ref_occbin")
const OBCD = joinpath(@__DIR__, "obc")
oref(name) = matread(joinpath(OREF, name * ".mat"))
odata(r) = (d = r["data"]; d isa AbstractVector ? reshape(Float64.(d), :, 1) : Matrix{Float64}(d))
opoints(r, n) = (X = r["X"]; X isa AbstractVector ? reshape(X, n, :) : (size(X, 1) == n ? X : Matrix(X')))
IVF = Dict("likelihood_inversion_filter" => true)
mkest(m, data; kw...) = with_logger(NullLogger()) do
    JDyn.Estimation(m; data=data, nobs=size(data, 1), first_obs=1, mode_compute=0, mh_replic=0, allow_unsupported=true, kw...)
end
function load_dir2(path)
    d = mktempdir()
    for f in readdir(dirname(path))
        p = joinpath(dirname(path), f); isfile(p) && cp(p, joinpath(d, f))
    end
    with_logger(NullLogger()) do; JDyn.load(joinpath(d, basename(path))); end
end

"Bit-exact equality of arbitrary results (floats compare with `isequal`, so 0.0 and -0.0 differ and NaN equals NaN; structs field by field)."
same(a::AbstractFloat, b::AbstractFloat) = isequal(a, b) && reinterpret(UInt64, Float64(a)) == reinterpret(UInt64, Float64(b))
same(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(same(x, y) for (x, y) in zip(a, b))
same(a::Tuple, b::Tuple) = length(a) == length(b) && all(same(x, y) for (x, y) in zip(a, b))
same(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && same(values(a), values(b))
same(a::AbstractDict, b::AbstractDict) = keys(a) == keys(b) && all(same(a[k], b[k]) for k in keys(a))
same(a::T, b::T) where {T<:Union{Nothing,Missing,Symbol,AbstractString,Integer,Bool}} = a === b || a == b
same(a::JDyn.Occbin, b::JDyn.Occbin) = true                    # the model object itself is not a result
function same(a::T, b::T) where T
    isstructtype(T) || return isequal(a, b)
    all(same(getfield(a, f), getfield(b, f)) for f in fieldnames(T))
end
same(a, b) = false

fields(r) = (; (f => getfield(r, f) for f in fieldnames(typeof(r)) if f != :occbin)...)

"Everything the OccBin code produces at the test examples, with a fixed random stream for the filter's root search."
function battery()
    out = Pair{String,Any}[]
    # NKM (3 observables, ZLB, 24 variables), small NKM, rbc with two constraints (PKF): the stored Dynare vectors
    for (case, ks, kw) in (("occest_nkm", 1:4, (; occbin=IVF)), ("occest_nkms", 1:3, (; occbin=IVF)), ("occest_rbc", 1:3, (;)))
        r = oref(case * "_lik")
        m = load_copy(joinpath(OREF, case * ".mod"))
        e = mkest(m, odata(r); kw...)
        X = opoints(r, e.n)
        for k in ks
            lp = JDyn.log_posterior(e, X[:, k]); info = (e.last_info_code, e.last_info)
            push!(out, "$case lp $k" => (lp, info))
            push!(out, "$case filter $k" => fields(JDyn.occbin_filter(e, X[:, k]; rng=Xoshiro(100 + k))))
        end
        if case == "occest_rbc"
            push!(out, "rbc smoother" => (sm = JDyn.smooth(m, odata(r); estimation=e, theta=X[:, 1]).occbin.smoother; fields(sm)))
        end
    end
    # bf (the regime-switching constraint): the initial values and perturbed vectors, some of which fail
    dd = matread(joinpath(OBCD, "est", "occest_bf_data.mat"))
    data = hcat(Float64.(vec(dd["q"])), Float64.(vec(dd["r"])))
    mb = load_copy(joinpath(OBCD, "est", "bf.mod"))
    eb = mkest(mb, data; occbin=IVF)
    rng = MersenneTwister(1)
    for k in 0:5
        th = k == 0 ? copy(eb.x0) : eb.x0 .* exp.(0.05 .* randn(rng, 5))
        push!(out, "bf lp $k" => (JDyn.log_posterior(eb, th), eb.last_info_code))
        push!(out, "bf filter $k" => fields(JDyn.occbin_filter(eb, th; rng=Xoshiro(200 + k))))
    end
    # simulations, one Occbin object per model and several shock sequences (the trie persists across calls)
    for (name, path) in (("rbc_occbin", joinpath(EXAMPLES, "occbin", "rbc_occbin.mod")), ("obc_zlb", joinpath(OBCD, "obc_zlb.mod")),
                         ("nk_zlb2", joinpath(OBCD, "nk_zlb2.mod")), ("obc_zlb2", joinpath(OBCD, "obc_zlb2.mod")))
        ob = Occbin(load_dir2(path))
        rs = MersenneTwister(5)
        for j in 1:6
            sh = (j <= 2 ? 0.02 : 0.05) .* randn(rs, 12, 1)
            res = try fields(simulate(ob, sh; periods=60, check_ahead_periods=60)) catch err; sprint(showerror, err) end
            push!(out, "$name sim $j" => res)
        end
    end
    out
end

with_budget(f, b) = (old = JDyn.OCCBIN_RULE_MEMO_BYTES[]; JDyn.OCCBIN_RULE_MEMO_BYTES[] = b; try f() finally JDyn.OCCBIN_RULE_MEMO_BYTES[] = old end)

@testset "rule memo budget: results are bit for bit independent of it" begin
    JDyn.reset_rule_memo_peak!()
    t0 = @elapsed ref = battery()
    peak = copy(JDyn._RULE_MEMO_PEAK)
    println("  default budget ($(JDyn.OCCBIN_RULE_MEMO_BYTES[] / 2^20) MiB): battery $(round(t0, digits=1)) s, ", length(ref), " results; largest trie of any Occbin object: ",
            peak[1], " nodes, ", round(peak[2] / 2^20, digits=2), " MiB")
    @test length(ref) > 50 && peak[1] > 20                      # the examples do use the memo
    @test peak[2] < JDyn.OCCBIN_RULE_MEMO_BYTES[] / 20          # ordinary use stays far below the budget (>= 20x headroom)
    # the results themselves are not trivial: some filters fail, some succeed, some simulations bind
    @test any(p -> p.first == "bf lp 1" || p.first == "bf lp 2", ref)
    for budget in (0, 40_000, 400_000)
        JDyn.reset_rule_memo_peak!()
        alt = with_budget(budget) do; battery(); end
        @test [p.first for p in alt] == [p.first for p in ref]
        bad = [p.first for (p, q) in zip(alt, ref) if !same(p.second, q.second)]
        @test isempty(bad)
        isempty(bad) || println("  differs at budget $budget: ", bad)
        @test JDyn._RULE_MEMO_PEAK[2] <= budget
    end
end

@testset "rule memo budget: refused rules are counted, stored ones bounded, and unmemoised use stays exact" begin
    m = load_dir2(joinpath(OBCD, "obc_zlb.mod"))
    ob = Occbin(m)
    sh = 0.05 .* randn(MersenneTwister(3), 12, 1)
    r0 = with_budget(0) do; simulate(ob, sh; periods=60, check_ahead_periods=60); end
    st = JDyn.rule_memo_stats(ob)
    @test st.nodes == 0 && st.bytes == 0 && st.refused > 0
    ob2 = Occbin(m)
    r1 = simulate(ob2, sh; periods=60, check_ahead_periods=60)
    st2 = JDyn.rule_memo_stats(ob2)
    @test st2.nodes > 0 && st2.refused == 0 && st2.bytes == st2.nodes * 8 * (ob2.n^2 + ob2.n * ob2.n_exo + ob2.n)
    @test same(fields(r0), fields(r1))
end

@testset "rule memo budget: the extreme vector of the Python soak test (231,000 rules, 1.3 GB before the budget)" begin
    r = oref("occest_nkm_lik")
    e = mkest(load_copy(joinpath(OREF, "occest_nkm.mod")), odata(r); occbin=IVF)
    e.options["qz_criterium"] = 1 + 1e-6           # pydyn builds Occbin with its own qz_criterium (1 + 1e-6), not the estimation's
    th = [109.65434996515319, 2.2349256172471215, 0.6453732314401828, 0.6609797328917931, 0.9999999229210729, 0.7881900975597803,
          0.004289142824646319, 0.010112247559243531, 6.0572589762884507e-05]
    JDyn.reset_rule_memo_peak!()
    budget = 16 * 2^20
    t = @elapsed (lp, code) = with_budget(budget) do
        lp = JDyn.log_posterior(e, th); (lp, e.last_info_code)
    end
    println("  extreme vector, budget 16 MiB: ", round(t, digits=1), " s, code ", code, ", peak trie ", JDyn._RULE_MEMO_PEAK[1], " nodes / ", round(JDyn._RULE_MEMO_PEAK[2] / 2^20, digits=1), " MiB")
    @test code == 304
    @test JDyn._RULE_MEMO_PEAK[2] <= budget
end
