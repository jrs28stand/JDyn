# Load Dynare's shipped examples unedited and compare with Dynare 7.2's results (port of pydyn's
# tools/sweep_examples.py `check`, first order only).
#
#   julia --project=. test/sweep_examples.jl            # prints a table for every example
#
# References: pydyn's reference/examples/<group>__<name>.mat (Dynare 7.2 under Octave).
# Examples run at order 2 are compared at order 2 too (ghxx, ghxu, ghuu, ghs2 from Dynare's k-order library through
# JDyn.k_order); without the library (deps/korder/build.sh) they report "order 1 matches (order 2 not compared)".
isdefined(@__MODULE__, :REF) || include("common.jl")
using Logging, Printf

const EXAMPLE_REFS = joinpath(REF, "examples")
const MATCH = "matches Dynare"
const MATCH1 = "order 1 matches (order 2 not compared)"

"Logger that keeps the messages of warnings."
mutable struct CollectLogger <: AbstractLogger
    msgs::Vector{String}
end
Logging.min_enabled_level(::CollectLogger) = Logging.Warn
Logging.shouldlog(::CollectLogger, args...) = true
Logging.handle_message(l::CollectLogger, level, msg, args...; kw...) = (push!(l.msgs, string(msg)); nothing)

_short(e) = (s = sprint(showerror, e); first(replace(s, '\n' => ' '), 140))
_f64(x) = x isa Number ? [Float64(x)] : vec(Float64.(x))
_mat(x, shape) = reshape(Float64.(x isa Number ? [x] : x), shape)

"""
    check_example(modpath; ref_file) -> Dict

Load `modpath` (a copy, in a folder holding the example's companion files), solve it at first order and
compare the decision rules with Dynare's. Keys: "name", "status", "detail", "skipped".
"""
function check_example(mod::AbstractString; ref_file=nothing, label=nothing)
    name = splitext(basename(mod))[1]
    out = Dict{String,Any}("name" => something(label, "$(basename(dirname(mod)))/$name"), "detail" => "", "skipped" => "")
    ref_file = ref_file === nothing ? joinpath(dirname(mod), "ref_$name.mat") : ref_file
    ref = isfile(ref_file) ? matread(ref_file) : Dict{String,Any}()
    lg = CollectLogger(String[])
    local m
    try
        m = with_logger(lg) do
            JDyn.load(mod)
        end
        out["skipped"] = join(sort(unique([first(s, 140) for s in lg.msgs if occursin("not handled", s)])), "; ")
    catch e
        msg = sprint(showerror, e)
        pac = occursin("dseries", msg) || occursin("pac", name)
        het = occursin("heterogeneous-agent", msg)
        out["status"] = pac || het ? "not supported" : "load failed"
        out["detail"] = pac ? "PAC / semi-structural models (dseries)" : het ? "heterogeneous agents" : _short(e)
        return out
    end
    kinds = Set(String(s["statementName"]) for s in m.statements)
    if !isempty(intersect(kinds, ["ramsey_model", "ramsey_policy", "discretionary_policy", "osr"]))
        out["status"], out["detail"] = "not supported", "optimal policy"
        return out
    end
    if any(k -> startswith(k, "heterogeneity"), kinds) || occursin("heterogeneity", mod)
        out["status"], out["detail"] = "not supported", "heterogeneous agents"
        return out
    end
    local ys, p
    try
        params = haskey(ref, "params") && all(isfinite, _f64(ref["params"])) ? _f64(ref["params"]) : nothing
        ys, p = with_logger(lg) do
            JDyn.steady_state(m; params=params)
        end
    catch e
        out["status"], out["detail"] = "steady state failed", _short(e)
        return out
    end
    notes = String[]
    opts = merge(get(m.options, "stoch_simul", Dict{String,Any}()), get(m.options, "estimation", Dict{String,Any}()))
    ref_ys = haskey(ref, "dr_ys") ? ref["dr_ys"] : get(ref, "ys", nothing)
    if ref_ys !== nothing && length(ref_ys) == length(ys)
        ref_ys = _f64(ref_ys)
        d = maximum(abs, ys .- ref_ys)
        if d > 1e-3 && all(ys .> 0) && maximum(abs, log.(ys) .- ref_ys) < 1e-6
            d = maximum(abs, log.(ys) .- ref_ys)                 # Dynare stored logs (loglinear)
        end
        push!(notes, @sprintf("ys %.1e", d))
        numerical = m.steady_state_model === nothing && m.steady_state_fn === nothing && m.steady_state_file === nothing
        if numerical && 1e-10 < d < 1e-4
            ys = ref_ys                                           # Dynare's solver stops early
            push!(notes, "rules at Dynare's ys")
        end
    end
    if !isempty(intersect(kinds, ["perfect_foresight_solver", "perfect_foresight_with_expectation_errors_solver"]))
        out["status"], out["detail"] = "not implemented", "perfect foresight (outside the first-order stream)"
        return out
    end
    local sol
    try
        sol = with_logger(lg) do
            JDyn.first_order(m; ys=ys, params=p, loglinear=false)
        end
    catch e
        out["status"], out["detail"] = "solution failed", _short(e)
        return out
    end
    ok = nothing
    if haskey(ref, "ghx")
        gx, gu = JDyn.ghx(sol), JDyn.ghu(sol)
        if get(opts, "loglinear", false) == true
            gx = gx .* ys[sol.state_var]' ./ ys[sol.order_var]
            gu = gu ./ ys[sol.order_var]
        end
        rx, ru = _mat(ref["ghx"], size(gx)), _mat(ref["ghu"], size(gu))
        d = max(maximum(abs, gx .- rx; init=0.0), maximum(abs, gu .- ru; init=0.0))
        haskey(ref, "order_var") && vec(Int.(ref["order_var"])) != sol.order_var && (d = Inf; push!(notes, "order_var differs"))
        push!(notes, @sprintf("rules %.1e", d))
        ok = d < 1e-6
    end
    order = Int(get(get(m.options, "stoch_simul", Dict()), "order", 1))
    compare2 = order >= 2 && haskey(ref, "ghxx") && m.max_deriv_order >= 2
    if compare2 && ok !== false && JDyn.korder_available()
        try
            dr = with_logger(lg) do
                JDyn.k_order(m; order=2, ys=ys, params=p, loglinear=false)
            end
            d2 = maximum(maximum(abs, dr[k] .- _mat(ref[k], size(dr[k]))) for k in ("ghxx", "ghxu", "ghuu", "ghs2"))
            push!(notes, @sprintf("order 2 %.1e", d2))
            ok = ok === true && d2 < 1e-6
            compare2 = false                                   # compared
        catch e
            push!(notes, "order 2 failed: " * first(_short(e), 60))
            ok = false
        end
    end
    out["status"] = ok === nothing ? "solved, no reference" : !ok ? "MISMATCH" : compare2 ? MATCH1 : MATCH
    out["detail"] = join(notes, ", ")
    out
end

"Copy an example's folder to a temporary directory and check it."
function check_shipped_example(group, name)
    d = mktempdir()
    for f in readdir(joinpath(EXAMPLES, group); join=true)
        isfile(f) && cp(f, joinpath(d, basename(f)); force=true)
    end
    check_example(joinpath(d, name * ".mod"); ref_file=joinpath(EXAMPLE_REFS, "$(group)__$(name).mat"), label="$group/$name")
end

if abspath(PROGRAM_FILE) == @__FILE__
    for group in sort(readdir(EXAMPLES)), f in sort(readdir(joinpath(EXAMPLES, group)))
        endswith(f, ".mod") || continue
        r = check_shipped_example(group, f[1:end-4])
        @printf("%-58s | %-38s | %s\n", r["name"], r["status"], r["detail"])
        isempty(r["skipped"]) || @printf("%-58s |   %s\n", "", r["skipped"])
    end
end
