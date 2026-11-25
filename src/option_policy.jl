# Which options of each Dynare statement JDyn acts on: nothing is silently ignored.
#
# Every option the Dynare 7.2 preprocessor can emit for a statement (src/option_tables.jl, read from DynareBison.yy by
# tools/gen_option_policy.py) is in exactly one of three sets of `OPTION_SPECS[statement]`:
#   honoured: JDyn reads it and the results depend on it (tools/options_probe*.jl, the sweep, or review);
#   harmless: it cannot change a number or a field JDyn computes (printing, figures, the name of an output folder, an algorithm that
#             gives the same solution), with the reason in `HARMLESS_REASONS`;
#   refused:  Dynare acts on it and JDyn does not; `dynare()` stops with an error naming the statement and the option.
# An option in none of them (unknown to Dynare 7.2's grammar for that statement, e.g. from a later Dynare) is refused as well.
# Statements in `UNIMPLEMENTED_STATEMENTS` are not run at all (JDyn says so when one is reached); their options are not classified.
# The plain Model API warns (once) about refused options of the file's statements instead of stopping, since it may be used for the
# parts of the file JDyn supports.

include("option_tables.jl")

_num_of(v) = v isa Real ? Float64(v) : v isa AbstractString ? something(tryparse(Float64, strip(v)), NaN) :
             (v isa AbstractVector && length(v) == 1) ? _num_of(v[1]) : NaN
_at_most(limit) = v -> (x = _num_of(v); !isnan(x) && x <= limit)
_at_least(limit) = v -> (x = _num_of(v); !isnan(x) && x >= limit)

"fsolve_options = ('Display','iter','AutoScaling','on'): only options of fsolve that do not move its solution."
function _fsolve_options_harmless(v)
    toks = String[strip(t) for t in split(string(v), ",")]
    isodd(length(toks)) && return false
    names = String[]
    for t in toks[1:2:end]
        mt = match(r"^''(\w+)''$", t)
        mt === nothing && return false
        push!(names, mt.captures[1])
    end
    !isempty(names) && all(n -> n in ("Display", "AutoScaling", "Diagnostics", "FunValCheck", "OutputFcn", "PlotFcns"), names)
end

"""
Options refused unless their value cannot change the result. Dynare's defaults (default_option_values.m): a looser tolerance (or
fewer iterations) changes the stored solution (decision_rules/first_order/walsh.mod: cycle reduction at tol 1e-1 differs from QZ); at
the default or tighter, cycle and logarithmic reduction agree with QZ to 4e-15 (pydyn checked under Octave).
"""
const VALUE_OK = Dict{String,Function}(
    "fsolve_options" => _fsolve_options_harmless,
    "dr_cycle_reduction_tol" => _at_most(1e-7), "dr_cycle_reduction_maxiter" => _at_least(100),
    "dr_logarithmic_reduction_tol" => _at_most(1e-12), "dr_logarithmic_reduction_maxiter" => _at_least(100))
"lyapunov=fixed_point at its default tolerance is 1.6e-8 away from the doubling algorithm (pydyn, Octave): refused. doubling and square_root_solver agree to 6e-13."
const ALWAYS_REFUSED = ("lyapunov_fp", "lyapunov_fixed_point_tol")

const _REFUSAL_REASONS = (
    ("particle.", "particle filters need order=2 (nonlinear filters); at order 1 they have no meaning"),
    ("skewed_kalman.", "the skewed Kalman filter is not implemented"),
    ("irf_opt.", "generalised impulse responses (EM / analytical GIRF) are not implemented"),
    ("mh_initialize_from_previous_mcmc.", "initialising the chains from a previous MCMC is not implemented"),
    ("shock_decomp.", "this realtime shock decomposition option is not implemented"),
    ("posterior_importance_sampling.", "importance sampling of the posterior is not implemented"))
_reason(key) = (for (p, why) in _REFUSAL_REASONS; startswith(key, p) && return why; end; "JDyn does not implement it")

# the tables, with the value-aware options and the always-refused ones moved out of `harmless` (see above)
for (st, s) in collect(OPTION_SPECS)
    h, r = copy(s.harmless), copy(s.refused)
    for k in (ALWAYS_REFUSED..., keys(VALUE_OK)...)
        if k in h; delete!(h, k); push!(r, k); end
    end
    OPTION_SPECS[st] = (honoured=s.honoured, harmless=h, refused=r)
end

"`:honoured`, `:harmless` or `:refused` for option `key` of `statement` (`nothing`: JDyn does not run that statement or it is not in the table)."
function classify_option(statement::AbstractString, key::AbstractString)
    s = get(OPTION_SPECS, statement, nothing)
    s === nothing && return nothing
    key in s.honoured && return :honoured
    (key in s.harmless || any(p -> startswith(key, p), keys(HARMLESS_PREFIXES))) && return :harmless
    :refused
end

_unset_value(v) = v === false || v == 0 && !(v isa AbstractString) || v === nothing || v == "false" || v == "0" || v == "" || (v isa AbstractVector && isempty(v))

"""
`[(option, reason)]` for the options of `statement` that JDyn does not act on (empty for a statement outside the table). A refused
option set to false/0 is the same as not set and is let through; `particle.*` options of `estimation` are honoured at order 2 only.
"""
function refused_options(statement::AbstractString, options)
    out = Tuple{String,String}[]
    options isa AbstractDict || return out
    haskey(OPTION_SPECS, statement) || return out
    for key in sort!(String.(collect(keys(options))))
        v = options[key]
        c = classify_option(statement, key)
        if c === :honoured && statement == "estimation" && (startswith(key, "particle.") || key == "pruning")
            o = _num_of(get(options, "order", 1))
            o == 2 && continue
            push!(out, (key, "particle filters belong to order=2 estimation; at order $(isnan(o) ? 1 : Int(o)) the option has no effect and is refused"))
        elseif c === :refused
            haskey(VALUE_OK, key) && VALUE_OK[key](v) && continue
            known = key in OPTION_SPECS[statement].refused
            if haskey(VALUE_OK, key)
                why = key == "fsolve_options" ? "fsolve options other than Display, AutoScaling, ... can move the solution" :
                      "JDyn solves with QZ; a value looser than Dynare's default would make Dynare's stored solution differ from it"
            elseif known
                why = _reason(key)
                _unset_value(v) && continue
            else
                why = "not an option of this statement in Dynare 7.2"
            end
            push!(out, (key, why))
        end
    end
    out
end

"Message refusing the statement because of its options (`nothing`: all of them are acted on)."
function option_policy_message(statement::AbstractString, options)
    bad = refused_options(statement, options)
    isempty(bad) && return nothing
    "JDyn does not act on " * join(["$k ($why)" for (k, why) in bad], ", ") * "; the $statement statement is refused rather than run without it"
end

"Plain Model API: warn (once per model) about the refused options of the file's statements; `dynare()` refuses the statement."
function warn_refused_options(m)
    bad = Dict{String,Set{String}}()
    for st in m.statements
        st isa AbstractDict || continue
        name = String(get(st, "statementName", ""))
        for (k, _) in refused_options(name, get(st, "options", nothing))
            push!(get!(bad, name, Set{String}()), k)
        end
    end
    for name in sort!(collect(keys(bad)))
        warn_once(m, "JDyn does not act on these $name options of the .mod file: $(join(sort!(collect(bad[name])), ", ")) (dynare() refuses the statement)")
    end
    bad
end
