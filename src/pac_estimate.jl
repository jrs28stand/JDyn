# Estimation of PAC equations: `pac.estimate.nls` and `pac.estimate.iterative_ols` (port of pydyn's pac_estimate.py / results_pacest.py).
#
# Behaviour follows Dynare 7.2: matlab/+pac/+estimate/{init, nls, iterative_ols}.m, matlab/optimization/{gauss_newton, jacobian}.m,
# matlab/utilities/estimation/{opt4nls, rewrite_equation_with_tables, write_residuals_routine, write_ssr_routine,
# get_estimated_parameters_indices}.m, matlab/{islagof, get_lhs_and_rhs, get_variables_and_parameters_in_equation}.m
# (Copyright (C) 2018-2026 Dynare Team, GPL-3.0-or-later).
#
# Dynare rewrites the equation as MATLAB text and writes residual functions to disk; here the equation (from the JSON of the
# preprocessor, with the PAC expectation replaced by its defining equation) is evaluated by a small recursive evaluator on
# arrays. Where the minimiser is deterministic and ported (csminwel, simplex, Gauss-Newton, annealing from Octave's stream) the
# iterates are Dynare's; `fmincon`, `fminunc`, `fminsearch` and `lsqnonlin` are MATLAB Optimization Toolbox (Octave optim)
# routines that JDyn does not port: they are run with own routines to a tight tolerance and agree with the minimiser of the sum
# of squares, not with the iterates of those routines. `particleswarm` (MATLAB's Global Optimization Toolbox) is refused.

# ---------------------------------------------------------------------------------------------- expression evaluator
const _PE_FUNCS = Dict{Symbol,Function}(:log => log, :exp => exp, :sqrt => sqrt, :abs => abs, :sin => sin, :cos => cos, :tan => tan,
    :sinh => sinh, :cosh => cosh, :tanh => tanh, :log10 => log10, :asin => asin, :acos => acos, :atan => atan, :sign => sign,
    :max => max, :min => min)
_pe_identifiers(text::AbstractString) = Set(String(m.match) for m in eachmatch(r"[A-Za-z_]\w*", text))

function _pe_lagarg(a)
    a isa Integer && return Int(a)
    if a isa Expr && a.head === :call && length(a.args) == 2 && a.args[2] isa Integer
        a.args[1] === :- && return -Int(a.args[2])
        a.args[1] === :+ && return Int(a.args[2])
    end
    nothing
end

"Evaluate the Dynare expression `ex` (a parsed Julia Expr); `res(name, shift)` gives the value of a symbol (nothing if unknown)."
function _pe_ev(ex, res)
    if ex isa Number
        return Float64(ex)
    elseif ex isa Symbol
        v = res(String(ex), 0)
        v === nothing && merr("'$(ex)' undefined")
        return v
    elseif ex isa Expr && ex.head === :call
        f = ex.args[1]
        if f isa Symbol && length(ex.args) == 2
            sh = _pe_lagarg(ex.args[2])
            if sh !== nothing
                v = res(String(f), sh)
                v !== nothing && return v
            end
        end
        args = Any[_pe_ev(a, res) for a in ex.args[2:end]]
        if f === :+ ; return length(args) == 1 ? args[1] : reduce((a, b) -> a .+ b, args)
        elseif f === :- ; return length(args) == 1 ? -args[1] : args[1] .- args[2]
        elseif f === :* ; return reduce((a, b) -> a .* b, args)
        elseif f === :/ ; return args[1] ./ args[2]
        elseif f === :^ ; return args[1] .^ args[2]
        elseif f isa Symbol && haskey(_PE_FUNCS, f)
            return length(args) == 1 ? _PE_FUNCS[f].(args[1]) : broadcast(_PE_FUNCS[f], args...)
        end
        throw(Unsupported("pac.estimate: the function $(f) in the equation", "function"))
    elseif ex isa Expr && ex.head === :tuple && length(ex.args) == 1
        return _pe_ev(ex.args[1], res)
    end
    throw(Unsupported("pac.estimate: the expression $(ex) in the equation", "function"))
end
_pe_parse(text::AbstractString) = Meta.parse(replace(text, r"(?<![\w.])\.(?=\d)" => "0."))

# ---------------------------------------------------------------------------------------------- series helpers
"The column shifted by `shift` periods (negative: lag), NaN where the period is not known."
function _pe_shift(x::AbstractVector, shift::Int)
    n = length(x)
    out = fill(NaN, n)
    k = -shift
    if k > 0
        k < n && (out[k+1:end] = x[1:n-k])
    elseif -k < n
        out[1:n+k] = x[1-k:end]
    end
    out
end

_pe_col(ds::MDseries, name) = (j = findfirst(==(name), ds.name); j === nothing ? nothing : view(ds.data, :, j))

"Rows (1-based range) of consecutive date codes in a series (Dynare's dseries subsref errors outside the sample)."
function _pe_rows(ds::MDseries, dates::AbstractVector)
    first = ds.dates.time[1]
    k0 = Int(round(dates[1] - first)); k1 = Int(round(dates[end] - first))
    k0 < 0 && merr("dseries::subsref: Indices are out of bounds! Subsample cannot start before $(date2string(first, ds.dates.freq)).")
    k1 >= nobs(ds) && merr("dseries::subsref: Indices are out of bounds! Subsample cannot end after $(date2string(first + nobs(ds) - 1, ds.dates.freq)).")
    k0+1:k1+1
end

function _pe_new_ds(data::AbstractMatrix, names, init::Float64, freq::Int)
    o = MDseries()
    o.data = Matrix{Float64}(data)
    o.name = String[names...]
    o.tex = [name2tex(s) for s in o.name]
    o.dates = mkdates(freq, init .+ (0:size(data, 1)-1))
    o.ops = fill("", length(o.name))
    o
end

"`dynamic_set_auxiliary_series`: the auxiliary variables of `aux_eqs` ([(name, rhs)] in order) from the series of `ds`."
function _pe_set_auxiliary_series(ds::MDseries, aux_eqs, param_names, params)
    cols = Dict{String,Vector{Float64}}(nm => Vector{Float64}(ds.data[:, j]) for (j, nm) in enumerate(ds.name))
    n = nobs(ds)
    names = copy(ds.name)
    pidx = Dict(nm => i for (i, nm) in enumerate(param_names))
    shiftable = Set(keys(cols)) ∪ Set(a for (a, _) in aux_eqs)
    defined = Set(ds.name) \ Set(a for (a, _) in aux_eqs)
    pending = collect(aux_eqs)
    while !isempty(pending)
        progress = false
        for item in copy(pending)
            name, rhs = item
            deps = Set(x for x in _pe_identifiers(rhs) if x in shiftable && x != name)
            issubset(deps, defined) || continue
            res = (nm, sh) -> haskey(cols, nm) || nm in shiftable ? _pe_shift(get(cols, nm, fill(NaN, n)), sh) :
                              (haskey(pidx, nm) ? params[pidx[nm]] : nothing)
            v = _pe_ev(_pe_parse(rhs), res)
            v = v isa Number ? fill(Float64(v), n) : Vector{Float64}(v)
            name in names || push!(names, name)
            cols[name] = v
            push!(defined, name)
            filter!(!=(item), pending)
            progress = true
        end
        if !progress
            miss = sort([x for (_, r) in pending for x in _pe_identifiers(r) if x in shiftable && !(x in defined)])
            merr("Variable $(first(miss)) is missing in the database.")
        end
    end
    _pe_new_ds(hcat((cols[nm] for nm in names)...), names, ds.dates.time[1], ds.dates.freq)
end

# ---------------------------------------------------------------------------------------------- initialisation
struct _PacPrep
    pacmodl::String
    lhs::String
    rhs::String
    rname::String
    pnames::Vector{String}
    enames::Vector{String}
    xnames::Vector{String}
    ipnames::Vector{Int}                          # 1-based positions in the parameter vector, in the order of `params`
    params::Vector{Tuple{String,Float64}}
    data::MDseries
    range::Vector{Float64}                        # date codes
    islagged::Bool
end

"The residual disappears with one of its neighbouring plus signs (Dynare leaves the sign and relies on MATLAB accepting it)."
function _pe_remove_residual(rhs::AbstractString, rname::AbstractString)
    pat = Regex("\\b" * rname * "\\b")
    ms = collect(eachmatch(pat, rhs))
    length(ms) == 1 || throw(Unsupported("pac.estimate: the residual $rname must appear exactly once in the equation", "function"))
    m = ms[1]
    before, after = rhs[1:m.offset-1], rhs[m.offset+length(m.match):end]
    startswith(after, "+") && return before * after[2:end]
    (endswith(before, "+") || (isempty(before) && isempty(after))) && return (isempty(before) ? before : before[1:end-1]) * after
    throw(Unsupported("pac.estimate: the residual $rname is not an additive term of the equation (Dynare removes its name and evaluates what is left, which is a MATLAB syntax error)", "function"))
end

function _pe_check_range(rng)
    (isempty(rng) || any(rng[i+1] - rng[i] != 1 for i in 1:length(rng)-1)) &&
        throw(Unsupported("pac.estimate: the range of dates of the estimation must be a sequence of consecutive periods", "function"))
end

_pe_f(s, k) = s isa MStruct && haskey(s, k) ? s[k] : merr("Reference to non-existent field '$k'.")
_pe_vec(x) = x isa Bool ? Float64[x] : x isa Real ? Float64[x] : x isa AbstractArray ? vec(Float64.(x)) : Float64[]
_pe_int1(x) = Int(round(first(_pe_vec(x))))

"""
`pac.estimate.init`: `kind` is `:nls` or `:iterative_ols`; `params` is a vector of `(name, value)` in the user's order; `jm`/`jo` the
equations of modfile.json / modfile-original.json; `pm` the model's `M_.pac.<name>` (an MStruct); `aux_eqs` the auxiliary variables'
defining equations.
"""
function _pe_init(m::Model, kind::Symbol, eqname, params, data::MDseries, rng, pm::MStruct, jm, jo, aux_eqs, params_vector)
    findeq(eqs) = (i = findfirst(e -> get(get(e, "tags", Dict()), "name", nothing) == eqname, eqs); i === nothing ? merr("Equation $eqname is not defined.") : eqs[i])
    RHS = findeq(jo)["rhs"]
    occursin("pac_expectation", lowercase(RHS)) || merr("This is not a PAC equation.")
    mm = match(r"\(model_name\s*=\s*(\w+)\)", RHS)
    pacmodl = String(mm.captures[1])
    e = findeq(jm)
    lhs, rhs = String(e["lhs"]), String(e["rhs"])
    auxrhs, paren = if _has(pm, "components")
        [(String(jm[_pe_int1(c.aux_id)]["lhs"]), String(jm[_pe_int1(c.aux_id)]["rhs"])) for c in pm.components], true
    elseif _has(pm, "aux_id")
        i = _pe_int1(pm.aux_id)
        [(String(jm[i]["lhs"]), String(jm[i]["rhs"]))], false
    else
        merr("Cannot find auxiliary variables for PAC expectation.")
    end
    for (a, r) in auxrhs
        rhs = replace(rhs, Regex("\\b" * a * "\\b") => (paren ? "(" * r * ")" : r))
    end
    ids = _pe_identifiers(lhs) ∪ _pe_identifiers(rhs)
    pnames = sort([x for x in m.param_names if x in ids])
    enames = sort([x for x in m.endo_names if x in ids])
    xnames = sort([x for x in m.exo_names if x in ids])
    missing_p = [p for (p, _) in params if !(p in pnames)]
    isempty(missing_p) || merr("The estimated parameters must be used in equation $eqname.")
    ipn_user = [findfirst(==(p), m.param_names) for (p, _) in params]
    ipn = collect(Union{Int,Nothing}, ipn_user)
    if kind === :iterative_ols
        ipn = Union{Int,Nothing}[_pe_int.(_pe_vec(pm.ec.params))..., _pe_int.(_pe_vec(pm.ar.params))...]
        if _has(pm, "optim_additive")
            v = _pe_vec(pm.optim_additive.params); append!(ipn, Int.(v[.!isnan.(v)]))
        end
        if _has(pm, "additive")
            v = _pe_vec(pm.additive.params); append!(ipn, Int.(v[.!isnan.(v)]))
        end
        _has(pm, "share_of_optimizing_agents_index") && !isempty(_pe_vec(pm.share_of_optimizing_agents_index)) &&
            push!(ipn, _pe_int1(pm.share_of_optimizing_agents_index))
        ipn = Union{Int,Nothing}[x in ipn_user ? x : nothing for x in ipn]
    end
    ipn = Int[x for x in ipn if x !== nothing]
    perm = [x in ipn ? findfirst(==(x), ipn) : 0 for x in ipn_user]
    params = collect(params)
    if kind === :iterative_ols
        sort(perm) == collect(1:length(perm)) ||
            throw(Unsupported("pac.estimate.iterative_ols: the estimated parameters are not all parameters of the PAC equation's error correction, autoregressive or share terms", "function"))
        params = [params[k] for k in perm]
    end
    norig = orig_endo_nbr(m)
    length(m.endo_names) > norig && (data = _pe_set_auxiliary_series(data, aux_eqs, m.param_names, params_vector))
    for n in enames
        n in data.name || merr("Variable $n is missing in the database.")
    end
    _pe_check_range(rng)
    rows = _pe_rows(data, rng)
    cols = [findfirst(==(n), data.name) for n in enames]
    any(isnan, data.data[rows, cols]) && merr("Some variable values are missing in the database.")
    for n in xnames
        n in data.name || merr("Variable $n is missing in the database.")
    end
    local rname
    if length(xnames) == 1
        rname = xnames[1]
        all(isnan, data.data[:, findfirst(==(rname), data.name)]) || merr("The residual ($rname) must have NaN values in the provided database.")
    else
        tmp = data.data[rows, [findfirst(==(n), data.name) for n in xnames]]
        idx = [j for j in 1:length(xnames) if !any(isnan, tmp[:, j])]
        length(idx) == length(xnames) && merr("There is no residual in this equation, all the exogenous variables are observed.")
        length(idx) < length(xnames) - 1 && merr("It is not allowed to have more than one residual in a PAC equation")
        rname = xnames[first(setdiff(1:length(xnames), idx))]
    end
    rhs = _pe_remove_residual(rhs, rname)
    _PacPrep(pacmodl, lhs, rhs, rname, pnames, enames, xnames, ipn, [(p, Float64(v)) for (p, v) in params], data, Float64.(rng), occursin("(-1)", rhs))
end
_pe_int(x) = Int(round(x))

# ---------------------------------------------------------------------------------------------- residual functions
"""
`r(p)`, `ssr(p)`, `full(p)`, `lhs_values(p)` of Dynare's written routines: the estimated parameters `p` are put in the parameter vector, the
PAC parameters recomputed (`update`), the residual `lhs - rhs` evaluated on the data of the range (and the period before).
"""
function _pe_residual_function(m::Model, prep::_PacPrep, update, base_params::Vector{Float64})
    data = prep.data
    rows = _pe_rows(data, vcat(prep.range[1] - 1, prep.range))
    D = data.data[rows, :]
    pidx = Dict(n => i for (i, n) in enumerate(m.param_names))
    colidx = Dict(n => j for (j, n) in enumerate(data.name))
    shiftable = Set(m.endo_names) ∪ Set(m.exo_names)
    lagged = prep.islagged
    function resolver(P)
        (name, shift) -> begin
            if haskey(pidx, name) && !(name in shiftable)
                return P[pidx[name]]
            elseif name in shiftable
                haskey(colidx, name) || merr("Variable $name is missing in the database.")
                k = colidx[name]
                if lagged
                    shift == 0 && return D[2:end, k]
                    shift == -1 && return D[1:end-1, k]
                    throw(Unsupported("pac.estimate: $name($shift): only lags of one period are rewritten by Dynare (rewrite_equation_with_tables.m); it then evaluates data(2:end,k)($shift), a MATLAB index error", "function"))
                end
                shift == 0 || throw(Unsupported("pac.estimate: $name($shift) in an equation without lags", "function"))
                return D[:, k]
            end
            nothing
        end
    end
    fl, fr = _pe_parse(prep.lhs), _pe_parse(prep.rhs)
    ip = prep.ipnames
    full(p) = (P = copy(base_params); P[ip] = p; update(P))
    function r(p)
        res = resolver(full(p))
        a, b = _pe_ev(fl, res), _pe_ev(fr, res)
        Vector{Float64}(vec(a .- b))
    end
    ssr(p) = (x = r(p); dot(x, x))
    lhs_values(p) = Vector{Float64}(vec(_pe_ev(fl, resolver(full(p)))))
    r, ssr, full, lhs_values
end

# ---------------------------------------------------------------------------------------------- optimisation
"opt4nls: `(noprint, options)` from key/value pairs; 'noprint' is consumed here."
function _pe_opt4nls(varargin)
    isodd(length(varargin)) && merr("Options must come by key/value pairs.")
    noprint = false
    opt = Pair{String,Any}[]
    for i in 1:2:length(varargin)-1
        k, v = String(varargin[i]), varargin[i+1]
        if k == "noprint"
            noprint = v isa AbstractString ? true : (v isa Bool ? v : (v isa Real ? v != 0 : true))
        else
            push!(opt, k => v)
        end
    end
    noprint, opt
end

"Dynare's jacobian.m: `J(:, i) = (r - fun(x + h e_i)) / h` (the opposite sign of the derivative)."
function _pe_jacobian(fun, x, h)
    r = fun(x)
    J = zeros(length(r), length(x))
    for i in eachindex(x)
        z = copy(x); z[i] += h
        J[:, i] = (r .- fun(z)) ./ h
    end
    r, J
end

"gauss_newton.m"
function _pe_gauss_newton(fun, x0)
    max_iter, h, xtol = 100, sqrt(eps()), 1e-6
    phi = 0.5 * (sqrt(5.0) + 1.0)
    x0 = Float64.(x0)
    counter = 0
    ssr(x) = (r = fun(x); dot(r, r))
    s1 = 0.0
    while true
        r0, J = _pe_jacobian(fun, x0, h)
        d = pinv(J) * r0
        x1 = x0 .+ d
        r1 = fun(x1)
        s0, s1 = dot(r0, r0), dot(r1, r1)
        if s1 > s0
            l1, l2 = 0.0, 1.0
            L1 = l2 - (l2 - l1) / phi
            L2 = l1 + (l2 - l1) / phi
            while abs(L1 - L2) > 1e-6
                if ssr(x0 .+ L1 .* d) < ssr(x0 .+ L2 .* d)
                    l2 = L2
                else
                    l1 = L1
                end
                L1 = l2 - (l2 - l1) / phi
                L2 = l1 + (l2 - l1) / phi
            end
            x1 = x0 .+ 0.5 * (l1 + l2) .* d
        end
        noconv = maximum(abs.(x1 .- x0)) > xtol
        counter += 1
        x0 = x1
        (!noconv || counter > max_iter) && break
    end
    x0, s1
end

"Dynare's simulated_annealing.m (mode_compute=2): draws its uniforms from `rng`, so that its path is Dynare's under Octave."
const SAOPT_DEFAULTS = Dict{String,Float64}("neps" => 10, "maximizer_indicator" => 0, "rt" => 0.10, "MaxIter" => 100000, "verbosity" => 1,
    "TolFun" => 1.0e-8, "initial_temperature" => 15, "ns" => 10, "nt" => 10, "step_length_c" => 0.1, "initial_step_length" => 1)

function simulated_annealing(fcn, x, optim::Dict{String,Float64}, lb, ub, rng)
    x = Float64.(vec(x)); lb = Float64.(vec(lb)); ub = Float64.(vec(ub))
    n = length(x)
    u() = first(rand(rng, 1))
    c = optim["step_length_c"] .* ones(n)
    t = optim["initial_temperature"]
    vm = optim["initial_step_length"] .* ones(n)
    xp = zeros(n)
    n_acc = n_oob = n_tot = 0
    xopt = copy(x)
    nacp = zeros(n)
    neps, ns, nt = Int(optim["neps"]), Int(optim["ns"]), Int(optim["nt"])
    fstar = 1e20 .* ones(neps)
    sign = optim["maximizer_indicator"] == 0 ? -1.0 : 1.0
    t <= 0.0 && return xopt, 0.0, 3, 0, 0, 0, "The initial temperature is not positive. Reset the variable t"
    (sum(x .> ub) + sum(x .< lb) > 0) && return xopt, 0.0, 2, 0, 0, 0, "Initial condition out of bounds"
    f = sign * fcn(x)
    n_tot += 1
    fopt = f
    fstar[1] = f
    max_iter, tolfun = optim["MaxIter"], optim["TolFun"]
    while true
        for _ in 1:nt
            for _ in 1:ns
                for h in 1:n
                    for i in 1:n
                        if i == h
                            xp[i] = x[i] + (u() * 2.0 - 1.0) * vm[i]
                        else
                            xp[i] = x[i]
                        end
                        if xp[i] < lb[i] || xp[i] > ub[i]
                            xp[i] = lb[i] + (ub[i] - lb[i]) * u()
                            n_oob += 1
                        end
                    end
                    fp = sign * fcn(xp)
                    n_tot += 1
                    if n_tot >= max_iter
                        return xopt, sign * fopt, 1, n_acc, n_tot, n_oob,
                               "Too many function evaluations; consider increasing optim.MaxIter or optim.TolFun or decreasing optim.nt or optim.rt. These results are likely to be poor"
                    end
                    if fp >= f
                        x = copy(xp); f = fp
                        n_acc += 1; nacp[h] += 1
                        if fp > fopt
                            xopt = copy(xp); fopt = fp
                        end
                    else
                        p = exp((fp - f) / t)
                        pp = u()
                        if pp < p
                            x = copy(xp); f = fp
                            n_acc += 1; nacp[h] += 1
                        end
                    end
                end
            end
            for i in 1:n
                ratio = nacp[i] / ns
                if ratio > .6
                    vm[i] = vm[i] * (1.0 + c[i] * (ratio - .6) / .4)
                elseif ratio < .4
                    vm[i] = vm[i] / (1.0 + c[i] * ((.4 - ratio) / .4))
                end
                vm[i] > (ub[i] - lb[i]) && (vm[i] = ub[i] - lb[i])
            end
            nacp = zeros(n)
        end
        fstar[1] = f
        quit_ = (fopt - fstar[1]) <= tolfun
        sum(abs.(f .- fstar) .> tolfun) > 0 && (quit_ = false)
        quit_ && return xopt, sign * fopt, 0, n_acc, n_tot, n_oob, "SA achieved termination criteria."
        t = optim["rt"] * t
        for i in neps:-1:2
            fstar[i] = fstar[i-1]
        end
        f = fopt
        x = copy(xopt)
    end
end

"""
Projected Levenberg-Marquardt on the residuals `res(p)` with box constraints, run to a tight tolerance. Stands in for MATLAB's
`fmincon`, `fminunc`, `fminsearch` and `lsqnonlin` (Optimization Toolbox / Octave optim, not ported): it finds the minimiser of the
sum of squares, not the iterates of those routines.
"""
function _pe_lm(res, x0, lo, hi; maxiter=500, tol=1e-12)
    x = clamp.(Float64.(x0), lo, hi)
    r = res(x)
    f = dot(r, r)
    lam = 1e-3
    n = length(x)
    for _ in 1:maxiter
        J = zeros(length(r), n)
        for i in 1:n
            h = 1e-7 * max(1.0, abs(x[i]))
            xp = copy(x); xp[i] += h
            xm = copy(x); xm[i] -= h
            if xp[i] > hi[i]; xp[i] = x[i]; end
            if xm[i] < lo[i]; xm[i] = x[i]; end
            J[:, i] = (res(xp) .- res(xm)) ./ (xp[i] - xm[i])
        end
        g = J' * r
        free = trues(n)
        for i in 1:n
            (x[i] <= lo[i] && g[i] > 0) && (free[i] = false)
            (x[i] >= hi[i] && g[i] < 0) && (free[i] = false)
        end
        any(free) || break
        Jf = J[:, free]
        A = Jf' * Jf
        improved = false
        for _ in 1:30
            d = -((A + lam .* Diagonal(max.(diag(A), 1e-12))) \ (Jf' * r))
            xn = copy(x); xn[free] .+= d
            xn = clamp.(xn, lo, hi)
            rn = res(xn)
            fn = dot(rn, rn)
            if isfinite(fn) && fn < f
                dx = maximum(abs.(xn .- x)); df = f - fn
                x, r, f = xn, rn, fn
                lam = max(lam / 10, 1e-15)
                improved = true
                (dx < tol * (1 + maximum(abs.(x))) && df < tol * (1 + f)) && return x, f
                break
            end
            lam *= 10
        end
        improved || break
    end
    x, f
end

function _pe_minimise(ssrfun, resfun, x0, optimizer, bounds, opt::Vector{Pair{String,Any}}, cs, rng)
    x0 = Float64.(x0)
    algo = optimizer === nothing ? "csminwel" : optimizer
    if algo == "GaussNewton"
        return _pe_gauss_newton(resfun, x0)
    elseif algo == "csminwel"
        crit, nit, method, eps_ = cs
        H0 = 1e-4 * Matrix{Float64}(I, length(x0), length(x0))
        for (k, v) in opt
            if k == "MaxIter"; nit = Int(round(_asnum(v)))
            elseif k == "InitialInverseHessian"; H0 = v isa AbstractMatrix ? Matrix{Float64}(v) : throw(Unsupported("pac.estimate.nls: the option InitialInverseHessian given as a string", "function"))
            elseif k == "TolFun"; crit = Float64(_asnum(v))
            elseif k == "NumgradAlgorithm"; method = Int(round(_asnum(v)))
            elseif k == "NumgradEpsilon"; eps_ = Float64(_asnum(v))
            elseif k in ("verbosity", "SaveFiles")
            else; @warn "csminwel: Unknown option ($k)!"
            end
        end
        fh, xh = csminwel(x -> (ssrfun(Float64.(x)), 0, 0.0), x0; H0=H0, crit=crit, nit=nit, method=method, epsilon=eps_)[1:2]
        return Float64.(xh), Float64(fh)
    elseif algo == "simplex"
        for (k, v) in opt
            k in ("verbosity", "SaveFiles") || @warn "simplex: Unknown option ($k)!"
        end
        r = simplex_optimization(z -> (ssrfun(Float64.(z)), true), x0)
        return Float64.(r[1]), Float64(r[2])
    elseif algo == "particleswarm"
        throw(Unsupported("pac.estimate.nls: the optimiser 'particleswarm' is MATLAB's Global Optimization Toolbox routine (Dynare itself stops under Octave: it is not available there), a random search whose result cannot be reproduced; Dynare 7.2 also sets the lower bound of the error-correction parameter to 1 instead of 0 for it", "function"))
    elseif algo == "annealing"
        rng === nothing && throw(Unsupported("pac.estimate.nls: the optimiser 'annealing' needs Dynare's random stream", "function"))
        so = copy(SAOPT_DEFAULTS)
        for (k, v) in opt
            if haskey(so, k)
                so[k] = Float64(_asnum(v))
            else
                @warn "solveopt: Unknown option ($k)!"
            end
        end
        huge = 1e7
        lo = [isinf(b) ? -huge : b for b in bounds[:, 1]]
        hi = [isinf(b) ? huge : b for b in bounds[:, 2]]
        xo, fo = simulated_annealing(z -> ssrfun(Float64.(z)), x0, so, lo, hi, rng)
        return Float64.(xo), Float64(fo)
    elseif algo in ("fmincon", "fminunc", "fminsearch", "lsqnonlin")
        lo, hi = bounds === nothing || algo in ("fminunc", "fminsearch") ? (fill(-Inf, length(x0)), fill(Inf, length(x0))) : (bounds[:, 1], bounds[:, 2])
        if algo == "lsqnonlin" && any(v == "levenberg-marquardt" for (_, v) in opt if v isa AbstractString)
            lo, hi = fill(-Inf, length(x0)), fill(Inf, length(x0))      # Levenberg-Marquardt does not handle bounds
        end
        @warn "pac.estimate.nls: '$algo' is run with JDyn's projected Levenberg-Marquardt to a tight tolerance: the minimiser of the sum of squares is that of Dynare's routine, its iterates are not"
        x, f = _pe_lm(resfun, x0, lo, hi)
        return x, f
    end
    merr("$algo is not an implemented optimization routine.\n")
end

"The covariance matrix of nls.m (White and Domowitz, theorem 3.2) and the t-statistics."
function _pe_nls_statistics(resfun, x)
    r, J = _pe_jacobian(resfun, x, 1e-6)
    T = length(r)
    A = 2.0 .* (J' * J) ./ T
    J = J .* r
    B = J' * J
    l = Int(floor(T^0.25 + 0.5))
    for tau in 1:l
        B = B + (1 - tau / (l + 1)) * (J[tau+1:end, :]' * J[1:end-tau, :] + J[1:end-tau, :]' * J[tau+1:end, :])
    end
    B = (4.0 / T) .* B
    Ai = inv(A)
    C = Ai * B * Ai ./ T
    r, C, x ./ sqrt.(diag(C))
end

# ---------------------------------------------------------------------------------------------- iterative OLS
_asnum(v) = first(_pe_vec(v))
_pe_ai(a, k) = (v = get(a, k, nothing); v === nothing ? nothing : Int(round(Float64(v))))

"matlab/islagof.m on 1-based indices of endogenous variables (including its test of `auxinfo1` twice)."
function _pe_islagof(m::Model, v1::Int, v2::Int)
    v1 == v2 && return false
    nor = orig_endo_nbr(m)
    a1, a2 = v1 > nor, v2 > nor
    (!a1 && !a2) && return false
    if a1 && a2
        i1, i2 = _aux_info(m, v1), _aux_info(m, v2)
        (i1 === nothing || i2 === nothing) && return false
        t1, t2 = _pe_ai(i1, "type"), _pe_ai(i2, "type")
        if t1 in (1, 0) && t2 in (1, 0)
            o1, o2 = get(i1, "orig_index", 0), get(i2, "orig_index", 0)
            return o1 != 0 && o2 != 0 && Int(i1["orig_lead_lag"]) < Int(i2["orig_lead_lag"])
        end
        if t1 in (8, 9)
            b = false
            if t1 == 9 && t2 == 8
                while _pe_ai(i1, "type") == 9
                    Int(i1["orig_index"]) == v2 && (b = true)
                    i1 = _aux_info(m, Int(i1["orig_index"]))
                    i1 === nothing && break
                end
            end
            return b
        end
        return false
    end
    if a1
        i1 = _aux_info(m, v1)
        return i1 !== nothing && _pe_ai(i1, "type") == 1 && Int(i1["orig_index"]) == v2
    end
    i2 = _aux_info(m, v2)
    i2 !== nothing && _pe_ai(i2, "type") == 0 && Int(i2["orig_index"]) == v1
end

"`[~, i] = setdiff(a, b)`: positions in `a` of the values not in `b`, by increasing value (last occurrence, as MATLAB)."
function _pe_setdiff_idx(a, b)
    bs = Set(x for x in b if !isnan(x))
    vals = sort(unique(x for x in a if !(x in bs)))
    [findlast(==(v), a) for v in vals]
end

_pe_nan_to_int_filter(v) = Int[Int(x) for x in v if !isnan(x)]

function _pe_iterative_ols(m::Model, prep::_PacPrep, pm::MStruct, P0::Vector{Float64}, update, share_bounds)
    data, rng = prep.data, prep.range
    rows = _pe_rows(data, rng)
    params0 = Float64[v for (_, v) in prep.params]
    ip = copy(prep.ipnames)
    P = copy(P0)
    pnames_est = Set(p for (p, _) in prep.params)
    endo, exo, pn = m.endo_names, m.exo_names, m.param_names
    col(name; lag=0) = begin
        j = findfirst(==(name), data.name)
        j === nothing && merr("dseries::extract: Variable $name is unknown in dseries object!")
        lag == 0 ? data.data[:, j] : _pe_shift(data.data[:, j], -lag)
    end
    nob = _has(pm, "non_optimizing_behaviour")
    is_exo = _has(pm, "additive") && length(_vec(pm.additive.vars)) > 1
    is_optim_exo = _has(pm, "optim_additive") && length(_vec(pm.optim_additive.vars)) > 0
    share_index = 0; estimate_share = false; share = 1.0
    nob_params = Float64[]
    nbp = nothing
    if nob
        nbp = pm.non_optimizing_behaviour
        npar = _vec(nbp.params)
        noparams = isnan.(npar)
        nob_params = fill(NaN, length(npar))
        if !all(noparams)
            any(pn[Int(i)] in pnames_est for i in npar[.!noparams]) && merr("The estimation of non optimizing behaviour parameters is not yet allowed.")
            nob_params[noparams] .= 1.0
            nob_params[.!noparams] = P[Int.(npar[.!noparams])]
        else
            nob_params[noparams] .= 1.0
        end
        nob_params = nob_params .* _vec(nbp.scaling_factor)
        share_index = Int(_pe_int1(pm.share_of_optimizing_agents_index))
        estimate_share = pn[share_index] in pnames_est
        if !estimate_share
            share = P[share_index]
            (share > 1 || share < 0) && merr("The share of optimizing agents shoud be in (0,1).")
        end
    end
    mk(v) = (params=_vec(v.params), vars=_pe_ivec(v.vars), isendo=_bvec(v.isendo), lags=_vec(v.lags), sf=_vec(v.scaling_factor))
    empty_spec = (params=Float64[], vars=Int[], isendo=Bool[], lags=Float64[], sf=Float64[])
    additive = empty_spec; add_est = Bool[]
    if is_exo
        a = mk(pm.additive)
        rid = findfirst(==(prep.rname), exo)
        keep = [j for j in eachindex(a.vars) if !(a.vars[j] == rid && !a.isendo[j])]
        additive = (params=a.params[keep], vars=a.vars[keep], isendo=a.isendo[keep], lags=a.lags[keep], sf=a.sf[keep])
        add_est = [!isnan(x) && Int(x) in ip for x in additive.params]
    end
    optim_additive = empty_spec; opt_est = Bool[]
    if is_optim_exo
        optim_additive = mk(pm.optim_additive)
        opt_est = [!isnan(x) && Int(x) in ip for x in optim_additive.params]
    end
    # data of the PAC expectation
    hpi = _ivec(pm.h_param_indices)
    listofvariables = String[]; blocks = Vector{Float64}[]
    isconstant = false
    for (i, hp) in enumerate(hpi)
        nm = replace(pn[hp], r"([\\.^$|?*+()\[\]{}])" => s"\\\1")
        mt = match(Regex("(?<var>((\\w*)|\\w*\\(-1\\)))\\*" * nm), prep.rhs)
        mt === nothing && (mt = match(Regex(nm * "\\*(?<var>((\\w*\\(-1\\))|(\\w*)))"), prep.rhs))
        if mt !== nothing
            v = String(mt[:var])
            if !occursin("(-1)", v)
                push!(listofvariables, v)
                i > length(listofvariables) && throw(Unsupported("pac.estimate.iterative_ols: Dynare itself fails here (the constant of the PAC expectation is not the last h parameter)", "function"))
                push!(blocks, col(listofvariables[i]))
            else
                push!(listofvariables, v[1:end-4])
                i > length(listofvariables) && throw(Unsupported("pac.estimate.iterative_ols: Dynare itself fails here (the constant of the PAC expectation is not the last h parameter)", "function"))
                push!(blocks, col(v[1:end-4]; lag=1))
            end
        elseif pn[hp] == "h_$(prep.pacmodl)_constant"
            isconstant = true
        end
    end
    length(unique(listofvariables)) == length(listofvariables) ||
        merr("dseries::horzcat: I cannot concatenate dseries objects with common variable names!")
    dataPAC = isempty(blocks) ? zeros(length(rows), 0) : hcat((b[rows] for b in blocks)...)
    isconstant && (dataPAC = hcat(ones(size(dataPAC, 1)), dataPAC))
    T = length(rows)
    dataNOB = zeros(T, 0)
    if nob
        cs = Vector{Float64}[]
        vars, isendo, lags = _pe_ivec(nbp.vars), _bvec(nbp.isendo), _vec(nbp.lags)
        for i in eachindex(vars)
            nme = isendo[i] ? endo[vars[i]] : exo[vars[i]]
            push!(cs, col(nme; lag=Int(lags[i]))[rows])
        end
        isempty(cs) || (dataNOB = hcat(cs...))
    end
    function additive_data(spec, est)
        estc = Vector{Float64}[]; cal = zeros(T); any_cal = any_est = false
        for i in eachindex(spec.vars)
            nme = spec.isendo[i] ? endo[spec.vars[i]] : exo[spec.vars[i]]
            lag = -Int(spec.lags[i])
            if est[i]
                any_est = true
                push!(estc, spec.sf[i] .* col(nme; lag=lag)[rows])
            else
                any_cal = true
                tmp = col(nme; lag=lag)[rows]
                isnan(spec.params[i]) || (tmp = P[Int(spec.params[i])] .* tmp)
                cal = cal .+ spec.sf[i] .* tmp
            end
        end
        (isempty(estc) ? nothing : hcat(estc...)), (any_cal ? cal : nothing), any_est
    end
    xdata_add, cal_add, any_est_x = is_exo ? additive_data(additive, add_est) : (nothing, nothing, false)
    xdata_opt, cal_opt, any_est_optim = is_optim_exo ? additive_data(optim_additive, opt_est) : (nothing, nothing, false)
    ec = pm.ec
    ecv = _ivec(ec.vars)
    ecvars = _bvec(ec.isendo)[2] ? ecv : reverse(ecv)
    ecterm = col(endo[ecvars[1]]; lag=1) .- col(endo[ecvars[2]]; lag=1)
    cols_x = Vector{Float64}[ecterm[rows]]
    arp = _vec(pm.ar.params); arv = _ivec(pm.ar.vars)
    lhs_var = Int(_pe_int1(pm.lhs_var))
    for i in eachindex(arp)
        _pe_islagof(m, arv[i], lhs_var) && push!(cols_x, col(endo[arv[i]])[rows])
    end
    XDATA = hcat(cols_x...)
    is_optim_exo && any_est_optim && (XDATA = hcat(XDATA, xdata_opt))
    is_exo && any_est_x && (XDATA = hcat(XDATA, xdata_add))
    params_id_0 = (nob && share_index in ip) ? [findfirst(==(share_index), ip)] : Int[]
    ipf = Float64.(ip)
    excl1 = Float64[nob ? [Float64(share_index)] : Float64[]; optim_additive.params; additive.params]
    excl5 = Float64[nob ? [Float64(share_index)] : Float64[]; additive.params]
    id1 = _pe_setdiff_idx(ipf, excl1)
    id5 = _pe_setdiff_idx(ipf, excl5)
    id2 = setdiff(1:length(ip), id1)
    id3 = setdiff(id2, params_id_0)
    id4 = vcat(id1, id3)
    params0_ = params0[id4]
    estimate_share && (share = params0[params_id_0[1]])
    (share > 1 || share < 0) && merr("Initial value for the share of optimizing agents shoud be in (0,1).")
    P[ip] = params0
    n5 = length(id5)
    hidx = hpi
    gl = _has(pm, "growth_linear_comb") ? pm.growth_linear_comb : nothing
    function pac_expectations(P)
        P = update(P)
        pe = dataPAC * P[hidx]
        corr = 0.0
        if gl !== nothing
            for g in (gl isa Cell ? gl.items : gl isa AbstractVector ? gl : [gl])
                gv = Float64(_asnum(g.constant))
                pid = Int(_asnum(g.param_id)); eid = Int(_asnum(g.exo_id)); nid = Int(_asnum(g.endo_id))
                pid > 0 && (gv = gv * P[pid])
                if eid > 0
                    gv = gv .* col(exo[eid]; lag=abs(Int(_asnum(g.lag))))[rows]
                elseif nid > 0
                    gv = gv .* col(endo[nid]; lag=abs(Int(_asnum(g.lag))))[rows]
                end
                corr = corr .+ gv
            end
            corr = corr .* P[Int(_pe_int1(pm.growth_neutrality_param_index))]
        end
        pe .+ corr, P
    end
    PacE, P = pac_expectations(P)
    Y0 = col(endo[lhs_var])[rows]
    noconv = true; counter = 0
    ssr = 0.0; r = Float64[]; log = Tuple{Int,Float64,Float64}[]
    while noconv
        counter += 1
        Y = copy(Y0)
        if nob
            Y = Y .- share .* PacE
            Y = Y .- (1 - share) .* (dataNOB * nob_params)
        else
            Y = Y .- PacE
        end
        is_exo && cal_add !== nothing && (Y = Y .- cal_add)
        is_optim_exo && cal_opt !== nothing && (Y = Y .- share .* cal_opt)
        p1 = XDATA \ Y
        p1[1:n5] = p1[1:n5] ./ share
        r = Y .- (XDATA[:, 1:n5] * p1[1:n5]) .* share
        is_optim_exo && any_est_optim && (r = r .- XDATA[:, n5+1:end] * p1[n5+1:end])
        ssr = dot(r, r)
        length(params0_) == length(p1) || merr("=: nonconformant arguments")
        crit = maximum(abs.(params0_ .- p1))
        noconv = crit > 1e-6
        push!(log, (counter, crit, ssr))
        params0_ = p1
        if estimate_share
            P[ip[id4]] = params0_
            PacE, P = pac_expectations(P)
            Y = Y0 .- dataNOB * nob_params
            if is_exo
                cal_add !== nothing && (Y = Y .- cal_add)
                any_est_x && (Y = Y .- XDATA[:, length(id1):end] * params0_[length(id1):end])
            end
            Z = XDATA[:, 1:n5] * params0_[1:n5] .+ PacE .- dataNOB * nob_params
            is_optim_exo && cal_opt !== nothing && (Z = Z .+ cal_opt)
            share = (reshape(Z, :, 1) \ Y)[1]
            share = max(min(share, share_bounds[2]), share_bounds[1])
            share < eps() && merr("On iteration $counter the share of optimizing agents is found to be zero. Please increase the default lower bound for this parameter.")
            P[ip[params_id_0[1]]] = share
        else
            length(params0_) == length(ip) || merr("=: nonconformant arguments")
            P[ip] = params0_
            PacE, P = pac_expectations(P)
        end
    end
    (ssr=ssr, residual=r, estimator=params0_, log=log, P=P)
end
_pe_ivec(x) = Int.(round.(_pe_vec(x)))

# ---------------------------------------------------------------------------------------------- the functions
function _pe_inputs(I, kind::String, a)
    h = _need_host(I)
    length(a) >= 4 || merr("Not enough input arguments.")
    eqname, params, data, rngv = a[1], a[2], a[3], a[4]
    eqname isa AbstractString || merr("pac.estimate: the first argument must be the name of the equation")
    params isa MStruct || merr("pac.estimate: the second argument must be a structure")
    data isa MDseries || merr("pac.estimate: the third argument must be a dseries object")
    rngv isa MDates || merr("pac.estimate: the fourth argument must be a range of dates")
    plist = Tuple{String,Float64}[]
    for k in params._keys
        v = _pe_vec(params[k])
        length(v) == 1 || merr("pac.estimate: the initial value of a parameter must be a scalar")
        push!(plist, (k, v[1]))
    end
    m = h.model
    jd = m.json_dir
    local jm, jo
    try
        jm = JSON.parsefile(joinpath(jd, "modfile.json"))["model"]
        jo = JSON.parsefile(joinpath(jd, "modfile-original.json"))["model"]
    catch e
        throw(Unsupported("pac.estimate: the JSON output of the preprocessor is not available ($(sprint(showerror, e)))", "file"))
    end
    jm isa AbstractVector || (jm = [jm]); jo isa AbstractVector || (jo = [jo])
    auxn = Set(m.endo_names[Int(x["endo_index"])] for x in get_or(m.raw, "aux_vars", []) if Int(x["type"]) != 6)
    aux_eqs = Tuple{String,String}[(String(e["lhs"]), String(e["rhs"])) for e in jm if isempty(get(e, "tags", Dict())) && String(e["lhs"]) in auxn]
    h, String(eqname), plist, data, Float64.(rngv.time), jm, jo, aux_eqs
end

function _pe_pm(h::Host, jo, eqname)
    S = semi_M(h.model)
    i = findfirst(e -> get(get(e, "tags", Dict()), "name", nothing) == eqname, jo)
    i === nothing && merr("Equation $eqname is not defined.")
    RHS = String(jo[i]["rhs"])
    mm = match(r"\(model_name\s*=\s*(\w+)\)", RHS)
    (occursin("pac_expectation", lowercase(RHS)) && mm !== nothing) || merr("This is not a PAC equation.")
    name = String(mm.captures[1])
    (_has(S, "pac") && _has(S.pac, name)) || merr("PAC model $name is not defined in the model block!")
    name, S.pac[name]
end

function _pe_updater(h::Host, name::String)
    function upd(P)
        h.params[:] = P
        pac_update_parameters!(h, name)
        copy(h.params)
    end
end

function _pe_store!(h::Host, name::String, fields::Vector{Pair{String,Any}})
    for (k, v) in fields
        _oo_put!(h, "pac", name, k, v)
    end
end

_pe_col(x::AbstractVector) = reshape(Float64.(x), :, 1)

function _pe_nls(I, a, nout)
    h, eqname, plist, data, rng, jm, jo, aux_eqs = _pe_inputs(I, "nls", a)
    m = h.model
    optimizer = length(a) > 4 && !(a[5] isa AbstractMatrix && isempty(a[5])) ? a[5] : nothing
    optimizer !== nothing && !(optimizer isa AbstractString) && merr("pac.estimate.nls: the optimizer must be a character array")
    rest = length(a) > 5 ? a[6:end] : Any[]
    name, pm = _pe_pm(h, jo, eqname)
    P0 = Float64.(copy(_semi_params(h)))
    h0 = copy(h.params)
    prep = _pe_init(m, :nls, eqname, plist, data, rng, pm, jm, jo, aux_eqs, P0)
    _bvec(pm.ec.istarget)[2] && merr("\nThe error correction term in PAC equation ($name) is not correct.\nThe error correction term should be the difference between a trend\nand the level of the endogenous variable.")
    update = _pe_updater(h, name)
    local x1, SSR, r, C, student, lhs, full
    try
        rfun, ssrfun, full, lhs_values = _pe_residual_function(m, prep, update, P0)
        ecname = m.param_names[Int(_pe_int1(pm.ec.params))]
        names = [p for (p, _) in prep.params]
        bounds = nothing
        if optimizer in ("lsqnonlin", "fmincon", "annealing", "particleswarm")
            bounds = repeat([-Inf Inf], length(names), 1)
            for (i, nm) in enumerate(names)
                nm == ecname && (bounds[i, :] = [0.0, 1.0])
            end
        end
        noprint, opt = _pe_opt4nls(rest)
        x0 = Float64[v for (_, v) in prep.params]
        ssr0 = ssrfun(x0)
        isfinite(ssr0) || merr("Cannot evaluate the Sum of Squared Residuals on the initial guess.")
        cs = (1e-7, 1000, 2, 1e-6)
        ws = get(_PF_HOST_WS, h, nothing)
        rng_ = ws === nothing ? nothing : dynare_rng(ws)
        x1, SSR = _pe_minimise(ssrfun, rfun, x0, optimizer, bounds, opt, cs, rng_)
        r, C, student = _pe_nls_statistics(rfun, x1)
        lhs = lhs_values(x1)
    catch e
        copyto!(h.params, h0)
        rethrow()
    end
    T = length(r)
    first_ = rng[1]
    freq = data.dates.freq
    s = Pair{String,Any}[
        "lhs" => _pe_new_ds(_pe_col(lhs), ["lhs"], first_, freq),
        "fit" => _pe_new_ds(_pe_col(lhs .- r), ["fit"], first_, freq),
        "residual" => _pe_new_ds(_pe_col(r), ["residual"], first_, freq),
        "ssr" => SSR, "s2" => SSR / T, "R2" => 1 - _pe_var(r) / _pe_var(lhs),
        "parnames" => Cell(Any[nm for (nm, _) in prep.params], (length(prep.params), 1)),
        "estimator" => _pe_col(x1), "covariance" => Matrix{Float64}(C), "student" => _pe_col(student)]
    h.params[:] = full(x1)
    h.nwrites += 1
    _pe_store!(h, name, s)
    Any[]
end

function _pe_iols(I, a, nout)
    length(a) > 4 && merr("Too many input arguments.")
    h, eqname, plist, data, rng, jm, jo, aux_eqs = _pe_inputs(I, "iterative_ols", a)
    m = h.model
    name, pm = _pe_pm(h, jo, eqname)
    P0 = Float64.(copy(_semi_params(h)))
    h0 = copy(h.params)
    prep = _pe_init(m, :iterative_ols, eqname, plist, data, rng, pm, jm, jo, aux_eqs, P0)
    ws = get(_PF_HOST_WS, h, nothing)
    lb, ub = 0.0, 1.0
    if ws !== nothing
        try
            o = ws.options_["pac"]["estimation"]["ols"]["share_of_optimizing_agents"]
            lb, ub = Float64(_asnum(o["lb"])), Float64(_asnum(o["ub"]))
        catch
        end
    end
    local res
    try
        res = _pe_iterative_ols(m, prep, pm, P0, _pe_updater(h, name), (lb, ub))
    catch e
        copyto!(h.params, h0)
        rethrow()
    end
    for (k, crit, ssr) in res.log
        println(@sprintf("Iter. %d,\t crit: %10.5f\t ssr: %10.8f", k, crit, ssr))
    end
    h.params[:] = res.P
    h.nwrites += 1
    n = length(res.estimator)
    _pe_store!(h, name, Pair{String,Any}["ssr" => res.ssr, "residual" => _pe_col(res.residual), "estimator" => _pe_col(res.estimator),
        "parnames" => Cell(Any[nm for (nm, _) in prep.params], (length(prep.params), 1)),
        "covariance" => fill(NaN, n, n), "student" => fill(NaN, n, 1)])
    Any[]
end

function _register_pac_estimate!()
    PKG_FUNCS["pac.estimate.nls"] = _pe_nls
    PKG_FUNCS["pac.estimate.iterative_ols"] = _pe_iols
    nothing
end

_register_pac_estimate!()
_pe_var(x) = (mu = sum(x) / length(x); sum(abs2, x .- mu) / (length(x) - 1))
