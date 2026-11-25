# JDyn's side of the cross-port acceptance protocol: a long-lived Julia worker that runs one test file at a time through
# JDyn.dynare() and writes <id>.mat (oo_, M_, options_) plus the result record.  Driven by test/acceptance/accept_jdyn.py
# (docs/ACCEPTANCE_RUNNER.md).  Protocol (files, so that nothing JDyn or the preprocessor prints can corrupt it):
#
#   julia --project=<jdyn> accept_worker.jl <workdir>
#   worker writes  <workdir>/ready            (once `using JDyn` and the warm-up are done; holds the warm-up time)
#   driver writes  <workdir>/cmd.<n>.json     {"rel","path","mat","dynare_args","second_limit_s"}   (atomically, by rename)
#   worker writes  <workdir>/reply.<n>.json   as soon as the first dynare() call and the .mat are done (second_s pending)
#                  <workdir>/done.<n>         when the whole command (incl. the second call) is finished; reply.<n>.json is final
#   driver writes  <workdir>/quit             to stop the worker
using JDyn, Logging, JSON, SparseArrays, Zlib_jll
# JDYN_PATCH=file1.jl:file2.jl: definitions loaded into JDyn after `using JDyn` (development only: avoids a precompile per edit)
for f in split(get(ENV, "JDYN_PATCH", ""), ":"; keepempty=false)
    Base.include(JDyn, f)
end

const JD = JDyn
const WORK = isempty(ARGS) ? mktempdir() : ARGS[1]
const REFUSAL = (:ModelError, :Unsupported, :EstimationError, :SmootherError, :NonlinearFilterError, :OptimalPolicyError, :OSRError,
                 :DiscretionaryPolicyError, :LoglinearError, :KOrderError, :ParseError)

# ----------------------------------------------------------------------------------- JDyn value -> MAT-file value
const KEYRE = r"^[A-Za-z][A-Za-z0-9_]{0,62}$"

"A scalar struct as converted: ordered field names and values."
struct SOut; keys::Vector{String}; vals::Vector{Any}; end
Base.isempty(s::SOut) = isempty(s.keys)

# ----------------------------------------------------------------------------------- MAT-file level 5 writer
# MAT.jl writes only v4 and v7.3 (HDF5); the shared comparator reads with scipy.io.loadmat (v5/v7), so the converted values are
# written here: each top-level variable as one zlib-compressed miMATRIX element.  Types: double (real/complex), logical,
# char (UTF-16), cell, scalar struct.
const mxCELL, mxSTRUCT, mxCHAR, mxDOUBLE, mxUINT8 = 1, 2, 4, 6, 9
const miINT8, miUINT8, miINT32, miUINT32, miUINT16, miDOUBLE, miMATRIX, miCOMPRESSED = 1, 2, 5, 6, 4, 9, 14, 15

function put_elem(io::IO, mi::Integer, bytes::AbstractVector{UInt8}; pad::Bool=true)
    write(io, UInt32(mi), UInt32(length(bytes)))
    write(io, bytes)
    pad && (r = length(bytes) % 8; r == 0 || write(io, zeros(UInt8, 8 - r)))
end
bytes_of(a) = collect(reinterpret(UInt8, vec(a)))

function matrix_elem(name::String, class::Int, dims, parts::Function; flags::Integer=0)
    b = IOBuffer()
    put_elem(b, miUINT32, bytes_of(UInt32[class | (Int(flags) << 8), 0]))
    put_elem(b, miINT32, bytes_of(Int32[dims...]))
    put_elem(b, miINT8, collect(codeunits(name)))
    parts(b)
    out = IOBuffer(); put_elem(out, miMATRIX, take!(b)); take!(out)
end

dims_of(x::AbstractArray) = ndims(x) == 1 ? (length(x), 1) : size(x)

function mat_value(name::String, x)::Vector{UInt8}
    if x isa SOut
        n = length(x.keys)
        return matrix_elem(name, mxSTRUCT, (1, 1), b -> begin
            put_elem(b, miINT32, bytes_of(Int32[64]))
            nm = zeros(UInt8, 64 * n)
            for (i, k) in enumerate(x.keys); nm[64(i-1)+1:64(i-1)+ncodeunits(k)] .= codeunits(k); end
            put_elem(b, miINT8, nm)
            for (k, v) in zip(x.keys, x.vals); write(b, mat_value("", v)); end
        end)
    elseif x isa Array{Any}
        return matrix_elem(name, mxCELL, dims_of(x), b -> for v in x; write(b, mat_value("", v)); end)
    elseif x isa AbstractString
        u = isempty(x) ? UInt16[] : transcode(UInt16, String(x))
        return matrix_elem(name, mxCHAR, isempty(u) ? (0, 0) : (1, length(u)), b -> put_elem(b, miUINT16, bytes_of(u)))
    elseif x isa Bool
        return mat_value(name, fill(x, 1, 1))
    elseif x isa AbstractArray{Bool}
        return matrix_elem(name, mxUINT8, dims_of(x), b -> put_elem(b, miUINT8, UInt8.(vec(x))); flags=0x02)
    elseif x isa Float64
        return mat_value(name, fill(x, 1, 1))
    elseif x isa ComplexF64
        return mat_value(name, fill(x, 1, 1))
    elseif x isa AbstractArray{Float64}
        return matrix_elem(name, mxDOUBLE, dims_of(x), b -> put_elem(b, miDOUBLE, bytes_of(x)))
    elseif x isa AbstractArray{ComplexF64}
        return matrix_elem(name, mxDOUBLE, dims_of(x), b -> (put_elem(b, miDOUBLE, bytes_of(real.(x))); put_elem(b, miDOUBLE, bytes_of(imag.(x)))); flags=0x08)
    end
    error("mat_value: cannot write a $(typeof(x))")
end

function zcompress(bytes::Vector{UInt8})
    n = length(bytes)
    cap = ccall((:compressBound, libz), Culong, (Culong,), n)
    out = Vector{UInt8}(undef, cap); len = Ref(Culong(cap))
    r = ccall((:compress2, libz), Cint, (Ptr{UInt8}, Ref{Culong}, Ptr{UInt8}, Culong, Cint), out, len, bytes, n, 6)
    r == 0 || error("zlib compress2 failed ($r)")
    resize!(out, len[])
end

function write_mat5(path::AbstractString, vars::AbstractVector{<:Pair})
    open(path, "w") do io
        hdr = rpad("MATLAB 5.0 MAT-file, written by JDyn test/acceptance/accept_worker.jl", 116)
        write(io, codeunits(hdr)[1:116], zeros(UInt8, 8), UInt16(0x0100), UInt8('I'), UInt8('M'))
        for (n, v) in vars
            put_elem(io, miCOMPRESSED, zcompress(mat_value(String(n), v)); pad=false)
        end
    end
end

"JDyn results-layer value -> something MAT.jl writes as Dynare would store it; `nothing`: not stored (Octave's accept_save drops objects too)."
function conv(x, path::String, refused::Dict{String,String}, depth::Int=0)
    (x === nothing || depth > 14) && return nothing
    if x isa JD.MStruct
        d = SOut(String[], Any[])
        for k in x._keys
            p = isempty(path) ? k : string(path, ".", k)
            v = try; x[k]; catch e; refused[p] = first(sprint(showerror, e), 200); continue; end
            c = conv(v, p, refused, depth + 1)
            (c === nothing || !occursin(KEYRE, k)) && continue
            push!(d.keys, k); push!(d.vals, c)
        end
        return d
    elseif x isa JD.Cell
        a = Array{Any}(undef, x.shape...)
        for (i, it) in enumerate(x.items)
            c = conv(it, string(path, "{", i, "}"), refused, depth + 1)
            a[i] = c === nothing ? zeros(0, 0) : c
        end
        return a
    elseif x isa JD.Tainted
        refused[path] = first(x.why, 200)
        return nothing
    elseif x isa Bool
        return x
    elseif x isa Real
        return Float64(x)
    elseif x isa Complex
        return ComplexF64(x)
    elseif x isa AbstractString
        return String(x)
    elseif x isa AbstractSparseArray
        return conv(Array(x), path, refused, depth)
    elseif x isa AbstractArray{Bool}
        return Array{Bool}(x)
    elseif x isa AbstractArray{<:Real}
        length(x) > 5_000_000 && return nothing
        return Array{Float64}(x)
    elseif x isa AbstractArray{<:Complex}
        return Array{ComplexF64}(x)
    elseif x isa AbstractArray && !isempty(x) && all(v -> v isa AbstractString, x)
        return Array{Any}(String.(x))
    elseif x isa AbstractArray && eltype(x) <: JD.MStruct          # struct array: a cell of structs, as accept_save.m
        a = Array{Any}(undef, size(x)...)
        for (i, it) in enumerate(x)
            c = conv(it, string(path, "(", i, ")"), refused, depth + 1)
            a[i] = c === nothing ? zeros(0, 0) : c
        end
        return a
    elseif x isa AbstractArray && eltype(x) === Any
        a = Array{Any}(undef, size(x)...)
        for (i, it) in enumerate(x)
            c = conv(it, string(path, "{", i, "}"), refused, depth + 1)
            a[i] = c === nothing ? zeros(0, 0) : c
        end
        return a
    end
    return nothing                                                   # function handles, objects (dseries, dates ...)
end

function write_mat(ws, matpath)
    refused = Dict{String,String}()
    S = Pair{String,Any}[]
    for (n, v) in (("oo_", ws.oo_), ("M_", ws.M_), ("options_", ws.options_))
        c = try; conv(v, "", n == "oo_" ? refused : Dict{String,String}()); catch e; @warn "conv $n failed" e; nothing; end
        c === nothing || isempty(c) || push!(S, n => c)
    end
    any(p -> first(p) == "oo_", S) || pushfirst!(S, "oo_" => SOut(String[], Any[]))
    mkpath(dirname(matpath))
    write_mat5(matpath, S)
    refused
end

# ----------------------------------------------------------------------------------- one command
mutable struct Collect <: Base.CoreLogging.AbstractLogger; n::Int; end
Base.CoreLogging.min_enabled_level(::Collect) = Base.CoreLogging.Warn
Base.CoreLogging.shouldlog(::Collect, args...) = true
Base.CoreLogging.catch_exceptions(::Collect) = true
Base.CoreLogging.handle_message(l::Collect, args...; kwargs...) = (l.n += 1; nothing)

const UNSUP = r"not (yet )?(supported|implemented)|unsupported|cannot yet|does not support"i

function classify(e)
    t = Symbol(nameof(typeof(e)))
    msg = first(replace(sprint(showerror, e), r"\s+" => " "), 500)
    if e isa JD.MatlabError
        return "error", "MatlabError: " * msg            # the file's own error()/assert(), or an interpreter error
    elseif e isa JD.ModelError && hasfield(JD.ModelError, :cause) && e.cause isa JD.MatlabError
        return "error", "ModelError(MatlabError): " * msg
    elseif t in REFUSAL
        return "refused", string(t) * ": " * msg
    elseif occursin(UNSUP, msg)
        return "refused", string(t) * ": " * msg
    end
    "error", string(t) * ": " * msg
end

# JDYN_ACCEPT_RNG=octave: the workspace's stream, the particle filters and the MCMC samplers draw Octave's numbers bit for bit
# (`dynare(file; random=:octave)`), one stream in file order as Dynare under Octave does, and the result json says "rng": "octave"
# so that the comparator compares the random-number dependent fields too.
const OCTAVE_RNG = get(ENV, "JDYN_ACCEPT_RNG", "") == "octave"
function rng_kwargs()
    OCTAVE_RNG ? (; random=:octave, particle_rng=:octave, mcmc_rng=:octave) : (;)
end

function run_cmd(c::AbstractDict, n::Int)
    path = c["path"]
    args = String.(get(c, "dynare_args", String[]))
    cd(dirname(path))
    if get(c, "setup", false)             # an earlier file of the entry whose outputs a later one reads: run, record nothing
        st = try; Logging.with_logger(Collect(0)) do; JD.dynare(path); end; "ok"; catch e; e isa InterruptException && rethrow(); "failed"; end
        put(n, Dict("status" => st)); touch(joinpath(WORK, "done.$n")); return
    end
    matp = c["mat"]
    kw = merge(rng_kwargs(), isempty(args) ? (;) : (; extra_args=args))
    info = Dict{String,Any}("status" => "ok", "message" => "", "load_s" => LOAD_S[], "warmup_s" => WARMUP_S[])
    OCTAVE_RNG && (info["rng"] = "octave")
    lg = Collect(0)
    ws = nothing
    try
        t = time()
        ws = Logging.with_logger(lg) do; JD.dynare(path; kw...); end
        info["first_s"] = round(time() - t; digits=3)
    catch e
        e isa InterruptException && rethrow()
        info["status"], info["message"] = classify(e)
        if info["status"] == "error"
            info["trace"] = first(sprint(Base.show_backtrace, catch_backtrace()), 1200)
        end
        info["first_s"] = nothing
    end
    info["warnings"] = lg.n
    if ws !== nothing
        try
            info["refused_fields"] = write_mat(ws, matp)
        catch e
            info["status"] = "error"; info["message"] = "writing the .mat: " * first(sprint(showerror, e), 300)
            isfile(matp) && rm(matp)
        end
    end
    ws = nothing
    info["first_s"] === nothing && delete!(info, "first_s")
    put(n, info)                                   # stage 1: result + .mat are final
    if info["status"] == "ok"
        lim = get(c, "second_limit_s", 120.0)
        if info["first_s"] > lim
            info["second_skipped"] = "first_s over $(Int(lim)) s"
        else
            try
                t = time()
                Logging.with_logger(Collect(0)) do; JD.dynare(path; kw...); end
                info["second_s"] = round(time() - t; digits=3)
            catch e
                e isa InterruptException && rethrow()
                info["second_skipped"] = "second call failed: " * first(sprint(showerror, e), 150)
            end
        end
    end
    GC.gc()
    put(n, info)
    touch(joinpath(WORK, "done.$n"))
end

function put(n, info)
    f = joinpath(WORK, "reply.$n.json"); tmp = f * ".tmp"
    open(tmp, "w") do io; JSON.print(io, info, 1); end
    mv(tmp, f; force=true)
end

# ----------------------------------------------------------------------------------- main loop
const LOAD_S = Ref(NaN)
const WARMUP_S = Ref(NaN)

function proc_age()
    try
        ticks = parse(Int, split(rsplit(read("/proc/self/stat", String), ")"; limit=2)[2])[20])
        boot = parse(Int, split(first(filter(l -> startswith(l, "btime"), readlines("/proc/stat"))))[2])
        time() - (boot + ticks / 100)
    catch
        NaN
    end
end

function warm_up()
    d = joinpath(WORK, "warmup"); mkpath(d)
    write(joinpath(d, "w.mod"), """
var y c k z; varexo e; parameters alpha beta delta rho;
alpha = 0.36; beta = 0.99; delta = 0.025; rho = 0.9;
model;
c^(-1) = beta*c(+1)^(-1)*(alpha*exp(z(+1))*k^(alpha-1)+1-delta);
k = exp(z)*k(-1)^alpha - c + (1-delta)*k(-1);
z = rho*z(-1) + e;
y = exp(z)*k(-1)^alpha;
end;
initval; k = 10; c = 1.5; z = 0; y = 2; e = 0; end;
shocks; var e; stderr 0.01; end;
steady; check;
stoch_simul(order=1, irf=5, nograph, noprint);
""")
    cd(d)
    try
        Logging.with_logger(Logging.NullLogger()) do; JD.dynare(joinpath(d, "w.mod")); end
    catch e
        @warn "warm-up failed" e
    end
end

function main()
    LOAD_S[] = round(proc_age(); digits=2)       # process start -> `using JDyn` done (the age of the process now)
    t = time(); warm_up(); WARMUP_S[] = round(time() - t; digits=2)
    open(joinpath(WORK, "ready"), "w") do io; println(io, LOAD_S[], " ", WARMUP_S[]); end
    n = 0
    while true
        isfile(joinpath(WORK, "quit")) && break
        cf = joinpath(WORK, "cmd.$(n + 1).json")
        if !isfile(cf)
            sleep(0.05); continue
        end
        n += 1
        c = JSON.parsefile(cf)
        try
            run_cmd(c, n)
        catch e
            e isa InterruptException && break
            put(n, Dict("status" => "error", "message" => "worker: " * first(sprint(showerror, e), 300), "load_s" => LOAD_S[]))
            touch(joinpath(WORK, "done.$n"))
        end
    end
end

abspath(PROGRAM_FILE) == @__FILE__() && main()
