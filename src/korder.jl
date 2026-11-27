# Higher-order perturbation (orders 1 to 3) through Dynare's own C++ k-order library (libkorder), called with
# `ccall` (port of pydyn/korder.py). The library is built as a plain shared library by deps/korder/build.sh;
# Dynare's model derivatives (the preprocessor's G1-G3, sparse, symmetric entries once) are passed in.
using Libdl

struct KOrderError <: Exception
    msg::String
end
Base.showerror(io::IO, e::KOrderError) = print(io, "KOrderError: ", e.msg)

"Names of the folded tensors of the C API and Dynare's names for them in `oo_.dr`."
const _KORDER_NAMES = ["gy" => "ghx", "gu" => "ghu", "gyy" => "ghxx", "gyu" => "ghxu", "guu" => "ghuu",
                       "gss" => "ghs2", "gyyy" => "ghxxx", "gyyu" => "ghxxu", "gyuu" => "ghxuu",
                       "guuu" => "ghuuu", "gyss" => "ghxss", "guss" => "ghuss"]

const _korder_lib = Ref{Ptr{Cvoid}}(C_NULL)
const _korder_path = Ref{String}("")
const _korder_lock = ReentrantLock()      # the C++ library keeps process-wide state

"Default location of the library: `deps/korder/libjdyn_korder.so` in the package folder."
default_korder_library() = joinpath(dirname(@__DIR__), "deps", "korder", "libjdyn_korder." * Libdl.dlext)

const _ARTIFACTS_TOML = joinpath(dirname(@__DIR__), "Artifacts.toml")

"""
Path of the prebuilt k-order library that `Pkg` downloaded with JDyn (artifact `korder` in `Artifacts.toml`;
x86_64 Linux and Windows), or `nothing` if there is no build for this platform or it was not downloaded.
"""
function artifact_korder_library()
    isfile(_ARTIFACTS_TOML) || return nothing
    meta = Artifacts.artifact_meta("korder", _ARTIFACTS_TOML)
    meta === nothing && return nothing
    h = Base.SHA1(meta["git-tree-sha1"])
    Artifacts.artifact_exists(h) || return nothing
    joinpath(Artifacts.artifact_path(h), "libjdyn_korder." * Libdl.dlext)
end

# On Windows, load the BLAS/runtime DLLs that sit beside the library first, so the loader finds them by name.
function _preload_dependencies(lib)
    Sys.iswindows() || return
    for d in ("vcruntime140.dll", "openblas.dll", "libopenblas.dll")
        f = joinpath(dirname(lib), d)
        isfile(f) && Libdl.dlopen(f; throw_error=false)
    end
end

"""
    korder_library(path=nothing) -> String

Path of Dynare's k-order library that JDyn uses: `path`, else the environment variable `JDYN_KORDER_LIB`, else
`deps/korder/libjdyn_korder.so` (`.dll` on Windows) in the package folder (built with `deps/korder/build.sh` or
`build_windows.sh`), else the prebuilt library `Pkg` downloads with JDyn on x86_64 Linux and Windows. Loads it on
first use; throws `KOrderError` with build instructions if none is found.
"""
function korder_library(path=nothing)
    lock(_korder_lock) do
        if path === nothing && _korder_lib[] != C_NULL
            return _korder_path[]
        end
        path !== nothing && !isfile(path) && throw(KOrderError("k-order library $path does not exist"))
        cands = String[]
        path !== nothing && push!(cands, String(path))
        haskey(ENV, "JDYN_KORDER_LIB") && push!(cands, ENV["JDYN_KORDER_LIB"])
        push!(cands, default_korder_library())
        art = artifact_korder_library()
        art === nothing || push!(cands, art)
        for c in cands
            if isfile(c)
                if _korder_lib[] != C_NULL && _korder_path[] == abspath(c)
                    return _korder_path[]
                end
                _preload_dependencies(c)
                _korder_lib[] = Libdl.dlopen(c, Libdl.RTLD_LAZY | Libdl.RTLD_LOCAL)
                _korder_path[] = abspath(c)
                return _korder_path[]
            end
        end
        throw(KOrderError("Dynare's k-order library was not found (looked for $(join(cands, ", "))). JDyn ships a " *
                          "prebuilt one for x86_64 Linux (glibc >= 2.38) and Windows, downloaded by Pkg; otherwise build " *
                          "it with `deps/korder/build.sh /path/to/dynare-7.2` (needs g++ >= 13, BLAS and LAPACK; " *
                          "`build_windows.sh` on Windows) or set the environment variable JDYN_KORDER_LIB to the built library."))
    end
end

"Whether the k-order library can be found (for tests that skip without it)."
korder_available() = try; korder_library(); true; catch e; e isa KOrderError ? false : rethrow(); end

_ksym(name) = Libdl.dlsym(_korder_lib[], name)

"""
    k_order(m; order=nothing, ys=nothing, params=nothing, pruning=nothing, threads=1, library=nothing,
            qz_criterium=nothing) -> Dict{String,Any}

Solve the model by perturbation at `order` (1 to 3) with Dynare's C++ solver (`order` and `pruning` default to the
`stoch_simul` options of the `.mod` file, Dynare's default order being 2). Returns Dynare's `oo_.dr` fields as a Dict:
`ghx`, `ghu`, `ghxx`, `ghxu`, `ghuu`, `ghs2` (order 2) and `ghxxx`, `ghxxu`, `ghxuu`, `ghuuu`, `ghxss`, `ghuss`
(order 3), with rows in decision-rule order and state columns (state variables, then shocks) as in Dynare, plus
the folded tensors `g_0` ... `g_k` (`pruning_g_i` with pruning), `ys`, `order_var`, `state_var` (1-based),
`order`, `pruning`, `params`. The Dict goes straight into `simulate` and `irf`.
"""
function k_order(m::Model; order=nothing, ys=nothing, params=nothing, pruning=nothing, threads=1, library=nothing,
                 qz_criterium=nothing, loglinear=nothing, Sigma_e=nothing)
    so = stoch_options(m)
    order === nothing && (order = Int(get(so, "order", 2)))
    pruning === nothing && (pruning = get(so, "pruning", false) == true)
    loglinear === nothing && (loglinear = get(so, "loglinear", false) == true)
    loglinear && throw(KOrderError(order > 1 ? "Loglinear options currently only works at order 1" :   # Dynare's message
                                   "k_order does not do loglinear models: use first_order"))
    order in 1:3 || throw(KOrderError("order must be 1, 2 or 3"))
    qz = JDyn.qz_criterium(m, qz_criterium)
    order > m.max_deriv_order && throw(KOrderError("the preprocessor output only has derivatives up to order " *
        "$(m.max_deriv_order); load the model with `JDyn.load(file; output=\"third\")` (or \"second\") or set order in the .mod file"))
    korder_library(library)
    if ys === nothing
        ys, params = steady_state(m; params=params)
    end
    params = Vector{Float64}(params === nothing ? m.params : params)
    ys = Vector{Float64}(ys)
    # As in Dynare, only what the derivatives depend on matters: a parameter that was never set (NaN) but appears in no
    # equation is harmless; one that does appear is caught by the derivative check below.
    sigma = Matrix{Float64}(Sigma_e === nothing ? m.Sigma_e : Sigma_e)      # `Sigma_e`: the shock covariance to use instead of the model's (estimation)
    (all(isfinite, ys) && all(isfinite, sigma)) ||
        throw(KOrderError("steady state or Sigma_e contain NaN or Inf"))
    n, nx = m.n_endo, m.n_exo
    order_var, state_var, cnt = dr_ordering(m.lead_lag_incidence)
    yy, x0 = repeat(ys, 3), exo_steady(m)
    nnz = Int32[]; rows = Int32[]; cols = Int32[]; vals = Float64[]
    for o in 1:order
        r, c, v = dynamic_sparse_derivs(m, o, yy, x0, params, ys)
        if o > 1 && !isempty(r)
            key = [(r[k], Tuple(sort(c[k, :]))) for k in eachindex(r)]
            length(unique(key)) == length(key) || throw(KOrderError("duplicate symmetric entries in the derivative list"))
        end
        push!(nnz, length(r))
        append!(rows, Int32.(r .- 1))
        # row-major (nnz, o), 0-based; columns in the preprocessor's layout [y(-1), y, y(+1), u]
        append!(cols, Int32.(vec(permutedims(c .- 1))))
        append!(vals, v)
    end
    if !all(isfinite, vals)
        bad = [m.param_names[i] for i in eachindex(params) if !isfinite(params[i])]
        throw(KOrderError("the derivatives of the model contain NaN or Inf" *
                          (isempty(bad) ? "" : " (parameters without a value: $(join(bad, ", ")))")))
    end
    params_c = replace(params, NaN => 0.0, Inf => 0.0, -Inf => 0.0)       # not used by the solver itself
    dr = Int32.(order_var .- 1)
    h = lock(_korder_lock) do
        ccall(_ksym(:jdyn_korder_solve), Ptr{Cvoid},
              (Cint, Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint},
               Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Float64}, Float64, Cint, Cint),
              order, cnt.nstatic, cnt.npred, cnt.nboth, cnt.nfwrd, nx, length(params), ys, sigma, params_c, dr,
              nnz, rows, cols, vals, qz, Int(pruning), threads)
    end
    try
        err = unsafe_string(ccall(_ksym(:jdyn_korder_error), Cstring, (Ptr{Cvoid},), h))
        isempty(err) || throw(KOrderError(err))
        fetch(name) = begin
            nr, nc = Ref{Cint}(0), Ref{Cint}(0)
            ok = ccall(_ksym(:jdyn_korder_dims), Cint, (Ptr{Cvoid}, Cstring, Ptr{Cint}, Ptr{Cint}), h, name, nr, nc)
            ok == 0 && return nothing
            out = Matrix{Float64}(undef, nr[], nc[])
            ccall(_ksym(:jdyn_korder_copy), Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Float64}), h, name, out)
            out
        end
        res = Dict{String,Any}("ys" => ys, "order_var" => order_var, "state_var" => state_var, "order" => order,
                               "pruning" => pruning, "params" => copy(params))
        for (nm, dn) in _KORDER_NAMES
            a = fetch(nm)
            a === nothing && continue
            res[dn] = dn == "ghs2" ? vec(a) : a
        end
        for i in 0:order
            res["g_$i"] = fetch("g_$i")
            pruning && (res["pruning_g_$i"] = fetch("pruning_g_$i"))
        end
        return res
    finally
        ccall(_ksym(:jdyn_korder_free), Cvoid, (Ptr{Cvoid},), h)
    end
end
