# Dynare's files of the posterior stage: the mode file (`<fname>_mode.mat`, option `mode_file`), the Metropolis-Hastings draws
# (`<fname>/metropolis/<fname>_mh<k>_blck<j>.mat`: x2, logpo2, LastSeeds) and the MH history (`<fname>_mh_history_<id>.mat`: the
# struct `record`), read and written as Dynare 7.2 does, so that `load_mh_file` continues chains made by Dynare or by JDyn.
# (matlab/estimation/{posterior_sampler_initialization, posterior_sampler_core, write_mh_history_file, load_last_mh_history_file,
# set_proposal_density_to_previous_value}.m, matlab/utilities/estimation/check_mode_file.m.)

"""
    resolve_mat_file(m::Model, name) -> path

`name` as given to `mode_file`: with or without `.mat`, relative to the .mod file's directory (Dynare's current directory).
"""
function resolve_mat_file(m::Model, name::AbstractString)
    name = String(name)
    for base in (isabspath(name) ? [""] : [m.mod_dir, pwd()]), ext in ("", ".mat")
        p = isempty(base) ? name * ext : joinpath(base, name * ext)
        isfile(p) && return p
    end
    throw(EstimationError("the file $name (.mat) does not exist"))
end

_matvec(x) = x isa AbstractArray ? vec(Float64.(x)) : Float64[x]
_matstr(x) = x isa AbstractString ? String[String(x)] : x isa AbstractArray ? String[strip(String(s)) for s in x] : String[]

"""
    read_mode_file(m, name) -> (; xparam1, hh, parameter_names, fval)

Reads Dynare's mode file (variables `xparam1`, `hh`, `parameter_names`, `fval`; `hh` and `fval` may be absent).
"""
function read_mode_file(m::Model, name::AbstractString)
    d = MAT.matread(resolve_mat_file(m, name))
    haskey(d, "xparam1") || throw(EstimationError("mode_file $name: no variable xparam1"))
    hh = haskey(d, "hh") ? Matrix{Float64}(d["hh"]) : nothing
    (xparam1=_matvec(d["xparam1"]), hh=hh, parameter_names=haskey(d, "parameter_names") ? _matstr(d["parameter_names"]) : nothing,
     fval=haskey(d, "fval") ? Float64(only(d["fval"])) : nothing)
end

"""
    match_mode_file(est::Estimation, mf; mode_compute) -> (xparam1, hh)

`check_mode_file.m`: the file's mode and Hessian reordered to the estimated parameters of `est` (names matched exactly, or
`stderr x` against `x`); with `mode_compute > 0` missing parameters keep the initial values. Returns `hh = nothing` when the file has none.
"""
function match_mode_file(est::Estimation, mf; mode_compute=0)
    names = est.names
    n = length(names)
    fnames = mf.parameter_names === nothing ? copy(names) : mf.parameter_names
    nf = length(mf.xparam1)
    function find(nm, list)
        i = findfirst(==(nm), list)
        i !== nothing && return i
        if length(nm) > 7 && startswith(nm, "stderr ")
            return findfirst(==(strip(nm[8:end])), list)
        end
        nothing
    end
    x = copy(est.x0)
    hh = mf.hh
    if nf == n && fnames == names
        return copy(mf.xparam1), hh === nothing ? nothing : copy(hh)
    end
    md = Int[]; xd = Int[]
    for (i, nm) in enumerate(names)
        id = find(nm, fnames)
        id === nothing ? @info("mode_file: estimated parameter $nm is not present in the mode file") : (push!(xd, i); push!(md, id))
    end
    for nm in fnames
        find(nm, names) === nothing && @info("mode_file: parameter $nm of the mode file is not estimated in the current mod file")
    end
    if mode_compute == 0
        length(xd) == n || throw(EstimationError("Please change the 'mode_file' option, the list of estimated parameters or set 'mode_compute'>0."))
        return mf.xparam1[md], hh === nothing ? nothing : hh[md, md]
    end
    isempty(xd) && throw(EstimationError("Please remove the 'mode_file' option."))
    x[xd] = mf.xparam1[md]
    x, nothing
end

# ---------------------------------------------------------------------------------------------------- MH files
"Directory of the MH files: `<mod dir>/<fname>/metropolis`."
mh_directory(m::Model) = joinpath(m.mod_dir, m.name, "metropolis")

"Octave's generator state vectors of one chain (Dynare's `Unifor`, `Normal`), 625 elements each."
struct ChainSeeds
    unifor::Vector{Float64}
    normal::Vector{Float64}
end

"Generator whose uniform and normal streams are in the states `s`."
function rng_from_seeds(s::ChainSeeds)
    r = OctaveRNG(0)
    octave_set_state!(r.u, s.unifor)
    octave_set_state!(r.n, s.normal)
    r
end
seeds_of(r::OctaveRNG) = ChainSeeds(octave_state(r.u), octave_state(r.n))

"""
Dynare's `record` of an MH run: the last history file. `draws_per_session` is the `MhDraws` matrix (draws, files, lines in
the last file) of every session.
"""
struct MHRecord
    sampler::String
    nblck::Int
    mhdraws::Matrix{Float64}
    max_nruns::Vector{Float64}
    last_parameters::Matrix{Float64}
    last_logpost::Vector{Float64}
    initial_parameters::Matrix{Float64}
    initial_logpost::Vector{Float64}
    last_seeds::Union{Nothing,Vector{ChainSeeds}}
    initial_seeds::Union{Nothing,Vector{ChainSeeds}}
    proposal_covariance::Matrix{Float64}     # d = chol(invhess), empty for the slice sampler
    proposal_scale::Vector{Float64}
    acceptance::Vector{Float64}
    feval::Vector{Float64}
    concluded::Float64
    last_file::Int
    last_line::Int
    id::Int
end

_seedstruct(x, nb) = begin
    x === nothing && return nothing
    out = ChainSeeds[]
    if x isa MAT.MatlabStructArray
        iu, inn = findfirst(==("Unifor"), x.names), findfirst(==("Normal"), x.names)
        (iu === nothing || inn === nothing) && return nothing
        for j in 1:length(x.values[iu])
            push!(out, ChainSeeds(Float64.(vec(x.values[iu][j])), Float64.(vec(x.values[inn][j]))))
        end
    elseif x isa AbstractDict && haskey(x, "Unifor")
        push!(out, ChainSeeds(Float64.(vec(x["Unifor"])), Float64.(vec(x["Normal"]))))
    elseif x isa AbstractVector && all(e -> e isa AbstractDict, x)
        for e in x
            push!(out, ChainSeeds(Float64.(vec(e["Unifor"])), Float64.(vec(e["Normal"]))))
        end
    else
        return nothing
    end
    out
end

_matmat(x) = x isa AbstractMatrix ? Matrix{Float64}(x) : x isa Number ? fill(Float64(x), 1, 1) : x isa AbstractVector ? reshape(Float64.(x), :, 1) : zeros(0, 0)

"The history files `<fname>_mh_history_<id>.mat` of the directory, by id."
function _history_files(dir, fname)
    ids = Int[]
    isdir(dir) || return ids
    for f in readdir(dir)
        mt = match(Regex("^" * replace(fname, r"[^A-Za-z0-9_]" => s -> "\\" * s) * "_mh_history_(\\d+)\\.mat\$"), f)
        mt === nothing || push!(ids, parse(Int, mt.captures[1]))
    end
    sort!(ids)
end

"The last MH history file (`load_last_mh_history_file.m`)."
function load_last_mh_history(m::Model; dir=mh_directory(m))
    ids = _history_files(dir, m.name)
    isempty(ids) && throw(EstimationError("load_mh_file: I cannot find any MH history file in $dir"))
    id = ids[end]
    d = MAT.matread(joinpath(dir, "$(m.name)_mh_history_$(id).mat"))
    r = d["record"]
    nb = Int(only(r["Nblck"]))
    getv(k, default=nothing) = haskey(r, k) ? r[k] : default
    MHRecord(String(getv("Sampler", "")), nb, _matmat(r["MhDraws"]), vec(Float64.(r["MAX_nruns"])),
             _matmat(r["LastParameters"]), vec(Float64.(r["LastLogPost"])),
             _matmat(getv("InitialParameters", zeros(0, 0))), vec(Float64.(getv("InitialLogPost", Float64[]))),
             _seedstruct(getv("LastSeeds"), nb), _seedstruct(getv("InitialSeeds"), nb),
             _matmat(getv("ProposalCovariance", zeros(0, 0))), vec(Float64.(getv("ProposalScaleVec", Float64[]))),
             vec(Float64.(getv("AcceptanceRatio", Float64[]))), vec(Float64.(getv("FunctionEvalPerIteration", Float64[]))),
             Float64(only(getv("MCMCConcludedSuccessfully", NaN))), Int(only(r["LastFileNumber"])), Int(only(r["LastLineNumber"])), id)
end

"""
    load_mh_draws(m, nblck, nfiles; dir) -> (draws (nblck x T x npar), logpo (nblck x T))

All draws of the files `<fname>_mh<k>_blck<j>.mat`, k = 1..nfiles, in order.
"""
function load_mh_draws(m::Model, nblck::Int, nfiles::Int; dir=mh_directory(m))
    X = Vector{Matrix{Float64}}[]
    L = Vector{Vector{Float64}}[]
    for j in 1:nblck
        xs = Matrix{Float64}[]; ls = Vector{Float64}[]
        for k in 1:nfiles
            f = joinpath(dir, "$(m.name)_mh$(k)_blck$(j).mat")
            isfile(f) || throw(EstimationError("load_mh_file: the file $f is missing (use mh_recover?)"))
            d = MAT.matread(f)
            push!(xs, Matrix{Float64}(d["x2"])); push!(ls, vec(Float64.(d["logpo2"])))
        end
        push!(X, xs); push!(L, ls)
    end
    T = sum(size(x, 1) for x in X[1])
    npar = size(X[1][1], 2)
    draws = zeros(nblck, T, npar); lp = zeros(nblck, T)
    for j in 1:nblck
        draws[j, :, :] = reduce(vcat, X[j]); lp[j, :] = reduce(vcat, L[j])
    end
    draws, lp
end

_seed_dict(s::ChainSeeds, algo="mt19937ar") =
    Dict{String,Any}("Unifor" => UInt32.(s.unifor), "Normal" => UInt32.(s.normal), "algo" => algo, "global_stream" => zeros(0, 0))

"""
    write_mh_files(m, mc::MCMCResult, info; previous, new_draws, dir)

Writes one session as Dynare does (posterior_sampler_core.m): files `<fname>_mh<k>_blck<j>.mat` (x2, logpo2, LastSeeds and the
acceptance counters) of at most `MAX_nruns` draws each, the draws of a continued run (`previous`, the record of the loaded run)
appended to the last file of the earlier one when it is not full, and a new history file with the updated `record`.
`mc` holds all draws (earlier and new); `new_draws` of them are new.
"""
function write_mh_files(m::Model, mc, info; dir=mh_directory(m), previous::Union{Nothing,MHRecord}=nothing, new_draws::Int=size(mc.draws, 2))
    mkpath(dir)
    nb, T, npar = size(mc.draws)
    max_nruns = ceil(1e8 / (npar + 2) / 8)
    maxn = Int(max_nruns)
    nprev = T - new_draws
    if previous === nothing
        first_file, first_line = 1, 1
        for i in _history_files(dir, m.name)
            rm(joinpath(dir, "$(m.name)_mh_history_$(i).mat"); force=true)
        end
        for f in readdir(dir)
            occursin(Regex("^" * m.name * "_mh\\d+_blck\\d+\\.mat\$"), f) && rm(joinpath(dir, f); force=true)
        end
    elseif previous.last_line < maxn
        first_file, first_line = previous.last_file, previous.last_line + 1
    else
        first_file, first_line = previous.last_file + 1, 1
    end
    feval = Float64.(get(info, :feval, fill(1.0, nb)))
    lastfile = first_file; lastline = 0
    for j in 1:nb
        pos = nprev + 1; fl = first_line; f = first_file
        while pos <= T
            take = min(maxn - fl + 1, T - pos + 1)
            rows = (pos-(fl-1)):(pos+take-1)
            sd = Dict{String,Any}()
            if pos + take > T && haskey(info, :last_seeds)
                sd["file$(f)"] = Dict{String,Any}("Unifor" => UInt32.(info[:last_seeds][j].unifor), "Normal" => UInt32.(info[:last_seeds][j].normal),
                                                  "algo" => "mt19937ar", "global_stream" => zeros(0, 0))
            end
            nacc = sum(mc.accepted[j, pos:pos+take-1])
            MAT.matwrite(joinpath(dir, "$(m.name)_mh$(f)_blck$(j).mat"),
                         Dict{String,Any}("x2" => mc.draws[j, rows, :], "logpo2" => reshape(mc.log_posterior[j, rows], :, 1), "LastSeeds" => sd,
                                          "accepted_draws_this_chain" => Float64(sum(mc.accepted[j, nprev+1:T])),
                                          "accepted_draws_this_file" => Float64(nacc),
                                          "feval_this_chain" => feval[j] * new_draws, "feval_this_file" => feval[j] * take))
            lastfile = f; lastline = fl - 1 + take
            pos += take; fl = 1; f += 1
        end
    end
    mhdraws = previous === nothing ? zeros(0, 3) : previous.mhdraws
    nfiles_new = lastfile - first_file + (first_line == 1 ? 1 : 0)
    nfiles_new = previous === nothing ? lastfile : (previous.last_line < maxn ? lastfile - previous.last_file : lastfile - previous.last_file)
    mhdraws = vcat(mhdraws, [Float64(new_draws) Float64(nfiles_new) Float64(lastline)])
    maxv = previous === nothing ? Float64[] : previous.max_nruns
    id = previous === nothing ? 0 : previous.id + 1
    rec = Dict{String,Any}(
        "Sampler" => String(get(info, :method, "random_walk_metropolis_hastings")),
        "Nblck" => Float64(nb), "MhDraws" => mhdraws, "MAX_nruns" => vcat(maxv, max_nruns),
        "AcceptanceRatio" => reshape(Float64.(mc.acceptance_rate), 1, :),
        "FunctionEvalPerIteration" => reshape(feval, 1, :),
        "InitialParameters" => Matrix{Float64}(mc.x0), "InitialLogPost" => Float64.(get(info, :initial_logpost, zeros(nb))),
        "LastParameters" => Matrix{Float64}(mc.draws[:, end, :]), "LastLogPost" => Float64.(mc.log_posterior[:, end]),
        "LastFileNumber" => Float64(lastfile), "LastLineNumber" => Float64(lastline),
        "MCMCConcludedSuccessfully" => 1.0,
        "ProposalScaleVec" => Float64.(get(info, :jscale_vector, Float64[])),
        "ProposalCovariance" => Matrix{Float64}(get(info, :proposal_chol, zeros(0, 0))))
    haskey(info, :last_seeds) && (rec["LastSeeds"] = [_seed_dict(s) for s in info[:last_seeds]])
    haskey(info, :initial_seeds) && (rec["InitialSeeds"] = [_seed_dict(s) for s in info[:initial_seeds]])
    MAT.matwrite(joinpath(dir, "$(m.name)_mh_history_$(id).mat"), Dict{String,Any}("record" => rec))
    nothing
end
