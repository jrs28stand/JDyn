# Shared by run_dynare_tests.jl (master) and worker.jl: arguments and the manifest.
const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", ".."))
const TAB = "\t"

function parse_args(args)
    o = Dict{String,Any}("timeout" => 120.0, "tests" => "/root/ref/dynare-7.2/tests",
                         "out" => joinpath(HERE, "results.csv"), "shard" => (0, 1), "merge" => false, "pat" => String[],
                         "worker" => nothing)
    for a in args
        if startswith(a, "--timeout=");  o["timeout"] = parse(Float64, a[11:end])
        elseif startswith(a, "--tests="); o["tests"] = a[9:end]
        elseif startswith(a, "--out=");   o["out"] = abspath(a[7:end])
        elseif startswith(a, "--shard="); k, n = parse.(Int, split(a[9:end], '/')); o["shard"] = (k - 1, n)
        elseif a == "--merge"; o["merge"] = true
        elseif startswith(a, "--worker="); o["worker"] = a[10:end]
        else push!(o["pat"], a)
        end
    end
    o
end

struct Entry
    id::String; tests::Vector{String}; extras::Vector{String}; should_fail::Bool; args::Vector{String}; env::Vector{String}
end
function read_manifest(; shard=(0, 1), pat=String[])
    es = Entry[]
    for l in eachline(joinpath(HERE, "manifest.tsv"))
        (isempty(l) || l[1] == '#') && continue
        f = split(l, TAB; keepempty=true)
        sp(s) = isempty(s) ? String[] : String.(split(s, ';'))
        push!(es, Entry(f[1], sp(f[2]), sp(f[3]), f[4] == "1", sp(f[5]), sp(f[6])))
    end
    es = [e for (i, e) in enumerate(es) if (i - 1) % shard[2] == shard[1]]
    isempty(pat) ? es : filter(e -> any(p -> occursin(p, join(e.tests, ";")), pat), es)
end

