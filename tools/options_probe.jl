# Differential probe for the options audit: run a statement of a small model with and without an option and report whether
# oo_ changes (differs), does not change (SAME) or the run stops (ERR).
#     julia --project=. tools/options_probe.jl [cases.jl]
# A case is (base_statements, [variant_statements...]) and an optional preamble; SAME is not an error (an option may legitimately not
# matter in the configuration probed); what matters is an option Dynare acts on that is SAME where it must matter: that option is
# silently ignored and belongs in the refused set of src/option_policy.jl. The preprocessor tells which option keys each variant adds.
using JDyn
import JSON

const PROBE_MOD = """var y c k a; varexo e u;
parameters alpha beta delta rho sig;
alpha=0.36;beta=0.99;delta=0.025;rho=0.9;sig=0.01;
model;
c^(-1)=beta*c(+1)^(-1)*(alpha*a(+1)*k^(alpha-1)+1-delta);
k=a*k(-1)^alpha+(1-delta)*k(-1)-c;
y=a*k(-1)^alpha;
log(a)=rho*log(a(-1))+e+u;
end;
initval; k=30;c=2.5;a=1;y=3;e=0;u=0;end;
steady;
shocks; var e; stderr 0.01; var u; stderr 0.007; end;
"""

function _flat!(out, o, path)
    if o isa JDyn.MStruct
        for k in keys(o); _flat!(out, o[k], path * "." * k); end
    elseif o isa JDyn.Cell
        for (i, x) in enumerate(o.items); _flat!(out, x, path * "{$i}"); end
    elseif o isa AbstractArray || o isa Number || o isa AbstractString || o isa Bool
        out[path] = o
    else
        out[path] = string(typeof(o))
    end
    out
end

function probe_run(stmt; pre=PROBE_MOD, tail="", files=Dict{String,String}())
    d = joinpath(tempdir(), "jdyn_probe"); rm(d; recursive=true, force=true); mkpath(d)       # a fixed folder: paths inside oo_ must not differ
    for (f, t) in files; write(joinpath(d, f), t); end
    write(joinpath(d, "m.mod"), pre * stmt * "\n" * tail * "\n")
    try
        ws = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                JDyn.dynare(joinpath(d, "m.mod"); seed=1)
            end
        end
        return _flat!(Dict{String,Any}(), ws.oo_, "oo_")
    catch e
        return (:ERR, first(sprint(showerror, e), 160))
    end
end
function same(a, b)
    keys(a) == keys(b) || return false
    all(k -> occursin("runtime", k) || isequal(a[k], b[k]), keys(a))      # run times differ between identical runs
end
function probe(cases; pre=PROBE_MOD, files=Dict{String,String}())
    for (base, variants) in cases
        b = probe_run(base; pre=pre, files=files)
        b2 = b isa Tuple ? b : probe_run(base; pre=pre, files=files)
        if b isa Tuple; println("BASE ERR  ", base, "  ", b[2]); continue; end
        same(b, b2) || println("NOTE: base run is not reproducible: ", base)
        for v in variants
            r = probe_run(v; pre=pre, files=files)
            println(r isa Tuple ? "ERR       " : same(b, r) ? "SAME      " : "differs   ", v, r isa Tuple ? "   " * r[2] : "")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    include(isempty(ARGS) ? joinpath(@__DIR__, "options_probe_cases.jl") : ARGS[1])
end
