# The `epilogue` block: variables computed from simulated or smoothed series after the fact (Dynare's epilogue_dynamic.m and
# epilogue_static.m, which the preprocessor writes for MATLAB; port of pydyn/epilogue.py). The preprocessor's JSON has no trace of the
# block, so it is read from the macro-expanded .mod file. Dynare's generated dynamic file runs dseries' `from ... to ... do` loop, which
# writes every assigned series back through sprintf('%f'), i.e. rounded to 6 decimals, and takes the leads of the vector case from
# t - lead: a Dynare 7.2 defect, corrected by default; `replicate_dynare_bugs=true` reproduces Dynare's numbers.

struct EpilogueDef
    name::String
    expr::String
    parsed::Any                                 # parsed MATLAB expression
    uses::Vector{Tuple{String,Int,String}}       # (variable, shift, placeholder)
end

struct Epilogue
    model::Model
    defs::Vector{EpilogueDef}
end
Base.show(io::IO, e::Epilogue) = print(io, "Epilogue(", join((d.name for d in e.defs), ", "), ")")
epilogue_names(e::Epilogue) = String[d.name for d in e.defs]

const _EPI_BLOCK = r"(?ims)^\s*epilogue\s*;(.*?)^\s*end\s*;"
function _epi_strip_comments(text)
    text = replace(text, r"/\*.*?\*/"s => " ")
    replace(text, r"//[^\n]*|%[^\n]*" => "")
end

"`[(name, expression)]` of the epilogue blocks of a (macro-expanded) .mod text."
function parse_epilogue(text::AbstractString)
    out = Tuple{String,String}[]
    for mt in eachmatch(_EPI_BLOCK, _epi_strip_comments(text))
        for stmt in split(mt.captures[1], ";")
            stmt = strip(stmt)
            isempty(stmt) && continue
            mm = match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$"s, stmt)
            mm === nothing && throw(ModelError("epilogue: cannot read $(repr(stmt)) (expected NAME = expression)"))
            push!(out, (String(mm.captures[1]), join(split(mm.captures[2]), " ")))
        end
    end
    out
end

"The .mod file after macro processing (`dynare-preprocessor file.mod onlymacro savemacro=...`)."
function macro_expanded_text(modfile::AbstractString; defines=String[], preprocessor=nothing)
    modfile = abspath(modfile)
    text = read(modfile, String)
    (!occursin("@#", text) && isempty(defines)) && return text
    exe = find_preprocessor(preprocessor)
    tmp = tempname() * ".mod"
    try
        out = IOBuffer(); err = IOBuffer()
        proc = run(pipeline(ignorestatus(Cmd(`$exe $(basename(modfile)) onlymacro savemacro=$tmp $defines`; dir=dirname(modfile))); stdout=out, stderr=err))
        proc.exitcode == 0 || error("preprocessor failed:\n$(String(take!(out)))\n$(String(take!(err)))")
        read(tmp, String)
    finally
        rm(tmp; force=true)
    end
end

"""
    epilogue(m; modfile=nothing, defines=String[], equations=nothing) -> Epilogue

The `epilogue` block of the model's .mod file (or the given `(name, expression)` pairs). Then
[`epilogue_dynamic`](@ref) and [`epilogue_static`](@ref) apply it to series.
"""
function epilogue(m::Model; modfile=nothing, defines=String[], equations=nothing)
    if equations === nothing
        modfile === nothing && (modfile = joinpath(m.mod_dir, m.name * ".mod"))
        equations = parse_epilogue(macro_expanded_text(modfile; defines=defines))
    end
    names = Set(String(n) for (n, _) in equations)
    known = union(Set(m.endo_names), Set(m.exo_names), names)
    defs = EpilogueDef[]
    for (name, expr) in equations
        uses = Tuple{String,Int,String}[]
        text = replace(expr, r"\b([A-Za-z_]\w*)\b(?:\s*\(\s*([+-]?\d+)\s*\))?" => s -> begin
            mt = match(r"^([A-Za-z_]\w*)(?:\s*\(\s*([+-]?\d+)\s*\))?$", s)
            v = String(mt.captures[1])
            v in known || return s
            shift = mt.captures[2] === nothing ? 0 : parse(Int, mt.captures[2])
            ph = "epi__$(v)__$(shift < 0 ? "m$(-shift)" : "p$shift")"
            any(u -> u[3] == ph, uses) || push!(uses, (v, shift, ph))
            ph
        end)
        body, _ = mparse(text)
        (length(body) == 1 && body[1] isa ExprStmt) || throw(ModelError("epilogue: cannot parse $(repr(expr))"))
        push!(defs, EpilogueDef(String(name), String(expr), body[1].e, uses))
    end
    Epilogue(m, defs)
end

function _epi_eval(I, def::EpilogueDef, params::Dict{String,Float64}, lookup)
    sc = Scope(; funcs=Dict{String,Any}())
    for (k, v) in params
        sc.vars[k] = v
    end
    for (v, shift, ph) in def.uses
        sc.vars[ph] = lookup(v, shift)
    end
    push!(I.scope_stack, sc)
    val = try
        ev(I, def.parsed, sc)
    finally
        pop!(I.scope_stack)
    end
    val isa Real ? Float64(val) : (val isa Complex ? (imag(val) == 0 ? Float64(real(val)) : NaN) : scalar(val, "epilogue value"))
end

_epi_params(m::Model, params) = Dict{String,Float64}(p => Float64((params === nothing ? m.params : params)[i]) for (i, p) in enumerate(m.param_names))

"""
    epilogue_dynamic(e, series; params=nothing, replicate_dynare_bugs=false) -> Dict

`epilogue_dynamic(params, ds)`: `series` maps names to equally long vectors (NaN where not observed). The definitions are applied in order;
each runs from the first period at which every variable it uses has been observed for at least its maximum lag to the last period
(recursively when the variable appears on the right-hand side); a definition that fails (a lead beyond the sample, a variable never
observed) leaves the series untouched (NaN for a new variable). Returns a new Dict with the epilogue variables added or updated.
"""
function epilogue_dynamic(e::Epilogue, series::AbstractDict; params=nothing, replicate_dynare_bugs::Bool=false)
    m = e.model
    S = Dict{String,Vector{Float64}}(String(k) => Float64.(vec(collect(v))) for (k, v) in series)
    T = length(first(values(S)))
    pr = _epi_params(m, params)
    I = _ssm_interp(m, params === nothing ? m.params : Vector{Float64}(params))
    for def in e.defs
        name = def.name
        haskey(S, name) || (S[name] = fill(NaN, T))
        vars = unique(u[1] for u in def.uses)
        lags = Dict(v => maximum([max(0, -s) for (w, s, _) in def.uses if w == v]) for v in vars)
        leads = Dict(v => maximum([max(0, s) for (w, s, _) in def.uses if w == v]) for v in vars)
        begins = Int[]
        ok = true
        for v in vars
            if !haskey(S, v)
                ok = false; break
            end
            f = findfirst(!isnan, S[v])
            f === nothing && (ok = false; break)
            push!(begins, f - 1 + lags[v])
        end
        (ok && !isempty(begins)) || continue
        t1, t2 = maximum(begins), T - 1
        t1 > t2 && continue
        any(v != name && t2 + leads[v] > T - 1 for v in vars) && continue
        recursive = name in vars
        recursive && leads[name] > 0 && continue
        data = Dict{String,Vector{Float64}}(v => copy(S[v]) for v in vars)
        haskey(data, name) || (data[name] = copy(S[name]))
        lead_sign = (recursive || !replicate_dynare_bugs) ? 1 : -1
        target = data[name]
        failed = false
        for t in t1:t2
            lookup = function (v, shift)
                k = t + (shift <= 0 ? shift : lead_sign * shift)
                (k < 0 || k >= T) && throw(BoundsError())
                data[v][k+1]
            end
            val = try
                _epi_eval(I, def, pr, lookup)
            catch ex
                (ex isa BoundsError || ex isa DomainError || ex isa MatlabError || ex isa ArgumentError || ex isa Unsupported) || rethrow()
                failed = true
                break
            end
            target[t+1] = val
        end
        failed && continue
        replicate_dynare_bugs && (target = Float64[isfinite(x) ? parse(Float64, @sprintf("%f", x)) : x for x in target])
        S[name] = target
    end
    S
end

"`epilogue_static(params, ds)`: every definition evaluated with lags and leads removed, on whole series (or scalars)."
function epilogue_static(e::Epilogue, values_::AbstractDict; params=nothing)
    m = e.model
    S = Dict{String,Vector{Float64}}(String(k) => Float64.(vec(collect(v isa Number ? [v] : v))) for (k, v) in values_)
    n = maximum(length, values(S); init=1)
    pr = _epi_params(m, params)
    I = _ssm_interp(m, params === nothing ? m.params : Vector{Float64}(params))
    for def in e.defs
        out = zeros(n)
        for t in 1:n
            lookup = (v, shift) -> (x = S[v]; length(x) == 1 ? x[1] : x[t])
            out[t] = _epi_eval(I, def, pr, lookup)
        end
        S[def.name] = out
    end
    S
end

"The epilogue variables of simulated or smoothed paths `endo` (T x n_endo, declaration order) and optional `exo` (T x n_exo)."
function epilogue_from_paths(e::Epilogue, endo::AbstractMatrix, exo=nothing; params=nothing, replicate_dynare_bugs::Bool=false)
    m = e.model
    S = Dict{String,Vector{Float64}}(v => Float64.(endo[:, i]) for (i, v) in enumerate(m.endo_names))
    exo === nothing || for (i, v) in enumerate(m.exo_names)
        S[v] = Float64.(exo[:, i])
    end
    epilogue_dynamic(e, S; params=params, replicate_dynare_bugs=replicate_dynare_bugs)
end
