# Alternative solution methods for occasionally binding constraints, selected with `obc_method` (docs/OBC_PROGRESS.md):
#
#   occbin   (default) the piecewise-linear guess-and-verify solver of src/occbin.jl, untouched
#   minnorm  MacroModelling.jl's approach: the smallest anticipated wedges on the constraint equation that enforce the
#            constraint over a horizon (src/obc_solvers.jl)
#   news     Holden's DynareOBC (the same wedges, the LCP as a mixed-integer programme, HiGHS) with diagnostics on M, for
#            constraints that are all complementary pairs; the regime formulation of src/obc_regime.jl for the others
#
# All methods read the same specification as OccBin: the occbin_constraints block and the bind/relax equation tags.
# The 7.2 preprocessor merges the two tagged versions of an equation into one, multiplied by the parameter
# occbin_<name>_bind, so the residual of the reference ("relax") and alternative ("bind") versions are the model's
# residual with that parameter at 0 and at 1. This file
#   1. builds one constraint object per constraint and classifies it (the post-processor),
#   2. builds the stacked complementarity problem g = q + M y over a horizon of T periods,
#   3. runs OccBin's simulation loop with the regime iteration replaced by an LCP solver.

# ---------------------------------------------------------------------- DEBUG / BENCHMARK ONLY: timing hook
# bench/obc_bench.jl sets `OBC_TICKS[]` to a vector: the simulation loops (OccBin's in src/occbin.jl, the `obc_method` ones here) then
# append `time_ns()` at the start of every shock period and once after the loop. Off (`nothing`) by default: the cost is one load of
# the Ref and a branch per shock period (not timed separately), and results are never affected.
const OBC_TICKS = Ref{Union{Nothing,Vector{UInt64}}}(nothing)
@inline function _obc_tick()
    t = OBC_TICKS[]
    t === nothing || push!(t, time_ns())
    nothing
end

# ---------------------------------------------------------------------- errors
"""
    ObcError

Failure of an `obc_method` solver in a simulation: `e.status` is `:no_solution`, `:horizon_too_short` or
`:not_converged`, `e.period` the 1-based shock period. A violation of the constraint beyond the horizon is reported
as `:horizon_too_short` (increase `obc_horizon`), not as the absence of a solution.
"""
struct ObcError <: Exception
    status::Symbol
    period::Int
    msg::String
end
Base.showerror(io::IO, e::ObcError) = print(io, "ObcError (", e.status, ", period ", e.period, "): ", e.msg)

const OBC_METHODS = (:occbin, :minnorm, :news)

"The `obc_method` value as a Symbol (`occbin`, `minnorm`, `news`); anything else is an error."
function _obc_method(x)
    s = Symbol(lowercase(strip(string(x isa AbstractArray && length(x) == 1 ? x[1] : x))))
    s in OBC_METHODS || throw(ArgumentError("unknown obc_method '$x': use one of $(join(OBC_METHODS, ", "))"))
    s
end
function _obc_horizon(x)
    v = x isa AbstractArray ? (length(x) == 1 ? x[1] : throw(ArgumentError("obc_horizon must be a scalar"))) : x
    v isa AbstractString && (v = parse(Float64, v))
    (v isa Real && isfinite(v) && v >= 1 && v == round(v)) || throw(ArgumentError("obc_horizon must be a positive integer (got $x)"))
    Int(v)
end

# ---------------------------------------------------------------------- the constraint object
"""
    ObcAffine(c, g)

An affine function of the model's variables near the reference steady state: `c + g' * [y(-1); y; y(+1); u]` with
`y` in deviations from the steady state (declaration order, `n` entries per block) and `u` the exogenous variables
(`n_exo` entries), the column order of the dynamic Jacobian.
"""
struct ObcAffine
    c::Float64
    g::Vector{Float64}
end
_vec(a::ObcAffine) = vcat(a.c, a.g)

"""
    ObcConstraint

One constraint of the `occbin_constraints` block, as the post-processor sees it (docs/OBC_PROGRESS.md, contract
section 3):

- `name`, `param` (index of `occbin_<name>_bind`), `equations` (indices of the equations whose linearisation at the
  steady state changes with it: what a first-order method switches) and `eq_names` (their `name` tags);
  `inert_equations`: equations that contain the parameter but whose linearisation does not change (switching them
  does nothing at order 1);
- `ref_resid`, `alt_resid`: for each such equation, its residual with the parameter at 0 (reference regime) and at 1
  (alternative regime), as `ObcAffine` functions at the reference steady state;
- `s_g`, `s_y`: the signs that make `g = s_g * alt_resid` and `y = s_y * ref_resid` non-negative with `g * y = 0`
  (the `bind` condition is `g < 0`, leaving the alternative regime is `y < 0`); 0 when undetermined;
- `entry`, `exit`: the fitted `(a, b)` with `d_b = a*alt_resid + b*ref_resid` and `(c, e)` with
  `d_r = c*ref_resid + e*alt_resid`, and `fit_err` the relative residuals of the two fits; `approx` the larger of the
  two when it is above `OBC_FIT_TOL` (an approximate complementarity constraint), 0 otherwise;
- `g_ss`: the gap at the steady state; `class`: `:complementarity`, `:multi_equation`, `:shared_equation` or
  `:inconsistent`; `why`: the reason for the last three.
"""
struct ObcConstraint
    name::String
    param::Int
    equations::Vector{Int}
    eq_names::Vector{String}
    inert_equations::Vector{Int}
    ref_resid::Vector{ObcAffine}
    alt_resid::Vector{ObcAffine}
    bind::String
    relax::String
    s_g::Int
    s_y::Int
    entry::Tuple{Float64,Float64}
    exit::Tuple{Float64,Float64}
    fit_err::Tuple{Float64,Float64}
    approx::Float64
    g_ss::Float64
    class::Symbol
    why::String
end
Base.show(io::IO, c::ObcConstraint) = print(io, "ObcConstraint(", c.name, ": ", c.class, isempty(c.why) ? "" : " - " * c.why, ")")

const OBC_FIT_TOL = 1e-6          # entry / exit fits are exact below this relative residual (contract rule 6)
const OBC_APPROX_TOL = 0.05       # ... and still accepted, as approximate, up to this one

# a comparison L op R as "d < 0 means the condition holds": d = L - R for < and <=, R - L for > and >=
function _negative_side(ex)
    (ex isa Expr && ex.head == :call && length(ex.args) == 3 && ex.args[1] in (:<, :<=, :>, :>=)) ||
        return nothing
    ex.args[1] in (:<, :<=) ? Expr(:call, :-, ex.args[2], ex.args[3]) : Expr(:call, :-, ex.args[3], ex.args[2])
end

# value and gradient (over the current-period variables, levels) of an expression of the constraints block at the
# steady state: complex step (exact to rounding for analytic expressions), central differences otherwise
function _linearise_condition(m::Model, ex, ys, p)
    f = _compiled_condition(m, m.fns.mod, ex)
    val = Float64(real(Base.invokelatest(f, ys, p, ys)))
    n = length(ys)
    grad = zeros(n)
    h = 1e-30
    try
        for k in 1:n
            z = complex.(ys)
            z[k] += im * h
            grad[k] = imag(Base.invokelatest(f, z, p, ys)) / h
        end
    catch
        for k in 1:n
            hk = 1e-6 * max(1.0, abs(ys[k]))
            zp = copy(ys); zp[k] += hk
            zm = copy(ys); zm[k] -= hk
            grad[k] = (Base.invokelatest(f, zp, p, ys) - Base.invokelatest(f, zm, p, ys)) / (2hk)
        end
    end
    val, grad
end

# least-squares fit target ≈ a*u + b*v; returns (a, b, relative residual)
function _fit2(target, u, v)
    A = hcat(u, v)
    coef = A \ target
    r = norm(target - A * coef) / max(norm(target), 1e-300)
    coef[1], coef[2], r
end

"Indices of the equations whose text (JSON model) contains parameter `ip`."
function _equations_naming_param(m::Model, ip::Int)
    rx = Regex("\\b" * m.param_names[ip] * "\\b")
    [e for (e, eq) in enumerate(get_or(m.raw, "model", Any[]))
     if occursin(rx, string(get(eq, "lhs", ""))) || occursin(rx, string(get(eq, "rhs", "")))]
end

"Rows whose residual or Jacobian at the steady state changes between two parameter vectors (what order 1 sees)."
function _changed_rows(r0, J0, r1, J1)
    n = length(r0)
    [e for e in 1:n if begin
        sc = max(1.0, abs(r0[e]), abs(r1[e]), maximum(abs, view(J0, e, :); init=0.0), maximum(abs, view(J1, e, :); init=0.0))
        abs(r1[e] - r0[e]) > 1e-12 * sc || maximum(abs, view(J1, e, :) .- view(J0, e, :); init=0.0) > 1e-12 * sc
    end]
end

_eq_name(m::Model, e::Int) = (model = get_or(m.raw, "model", Any[]);
    e <= length(model) ? string(get(get(model[e], "tags", Dict()), "name", e)) : "auxiliary $e")

"""
    obc_constraints(m::Model; params=nothing, ys=nothing) -> Vector{ObcConstraint}
    obc_constraints(ob::Occbin) -> Vector{ObcConstraint}

The post-processor: one `ObcConstraint` per constraint of the `occbin_constraints` block, with its residuals in the
reference and alternative regimes, the sign normalisation of gap and wedge and its class. The reference regime is the
one with every `occbin_<name>_bind` at 0 (OccBin's), at the reference steady state (computed unless `ys` is given).

Classification (contract section 3 as amended 2026-11-24, plus `shared_equation` and the orientation test, see
docs/OBC_PROGRESS.md). "Equations" are those whose linearisation at the steady state changes with the parameter.
1. more than one equation: `multi_equation`;
2. the equation also changes with another constraint's parameter: `shared_equation`;
3. entry test: the linearised `bind` condition, written `d_b < 0`, must be `a*alt_resid + b*ref_resid` with `a != 0`;
   `s_g = sign(a)`;
4. exit test: the linearised `relax` condition, written `d_r > 0` ("leave"; `d_r = d_b` when there is no `relax`), must
   be `c*ref_resid + e*alt_resid` with `c != 0`; `s_y = -sign(c)`;
5. the gap at the steady state must be strictly positive;
6. fits are exact below a relative residual of `OBC_FIT_TOL` (1e-6) and accepted as approximate up to `OBC_APPROX_TOL`
   (0.05), recorded in `approx`;
7. orientation: the bind condition carried into the alternative regime must not say "leave" while the wedge is
   positive, nor the relax condition carried into the reference regime say "stay" while the gap is positive
   (`b*s_y <= 0` and `e*s_g >= 0`): otherwise entering and leaving contradict each other;
failures of 3 to 7 (and conditions that are not a single `<`, `<=`, `>` or `>=` comparison): `inconsistent`.
"""
function obc_constraints(m::Model; params=nothing, ys=nothing)
    isempty(m.occbin_constraints) && throw(ModelError("the model has no occbin_constraints block"))
    names = String[c["name"] for c in m.occbin_constraints]
    switch = Int[]
    for nm in names
        pn = "occbin_$(nm)_bind"
        haskey(m.ip, pn) || throw(ModelError("parameter $pn not found: no equation tagged for constraint '$nm' in the model"))
        push!(switch, m.ip[pn])
    end
    base = Vector{Float64}(params === nothing ? m.params : params)
    base[switch] .= 0.0
    if ys === nothing
        ys, base = steady_state(m; params=base)
    end
    x0 = exo_steady(m)
    yy = vcat(ys, ys, ys)
    n = m.n_endo
    r0, J0 = dynamic_resid(m, yy, x0, base, ys), dynamic_jac(m, yy, x0, base, ys)
    lin = map(switch) do ip
        p1 = copy(base); p1[ip] = 1.0
        (dynamic_resid(m, yy, x0, p1, ys), dynamic_jac(m, yy, x0, p1, ys))
    end
    eqsets = [_changed_rows(r0, J0, r1, J1) for (r1, J1) in lin]
    out = ObcConstraint[]
    for (j, c) in enumerate(m.occbin_constraints)
        nm, ip, eqs = names[j], switch[j], eqsets[j]
        r1, J1 = lin[j]
        inert = setdiff(_equations_naming_param(m, ip), eqs)
        ref = [ObcAffine(r0[e], J0[e, :]) for e in eqs]
        alt = [ObcAffine(r1[e], J1[e, :]) for e in eqs]
        bindtxt = string(get(c, "bind", ""))
        relaxtxt = strip(string(something(get(c, "relax", ""), "")))
        mk(cls, why; s_g=0, s_y=0, entry=(NaN, NaN), exit=(NaN, NaN), fit=(NaN, NaN), approx=0.0, g_ss=NaN) =
            ObcConstraint(nm, ip, eqs, [_eq_name(m, e) for e in eqs], inert, ref, alt, bindtxt, relaxtxt, s_g, s_y, entry, exit, fit,
                          approx, g_ss, cls, why)
        if isempty(eqs)
            push!(out, mk(:inconsistent, "no equation of the model changes (at order 1) with $(m.param_names[ip])"))
            continue
        elseif length(eqs) > 1
            push!(out, mk(:multi_equation, "the constraint switches $(length(eqs)) equations (" *
                                          join(["'" * _eq_name(m, e) * "'" for e in eqs], ", ") * ")"))
            continue
        end
        e = eqs[1]
        others = [names[k] for k in eachindex(names) if k != j && e in eqsets[k]]
        # the two conditions, linearised at the steady state, as vectors comparable with the residuals
        why = ""
        local db, dr
        try
            bex = _parse_constraint(bindtxt)
            nb = _negative_side(bex)
            nb === nothing && throw(ModelError("the bind condition '$bindtxt' is not a single <, <=, > or >= comparison"))
            vb, gb = _linearise_condition(m, nb, ys, base)
            db = vcat(vb, zeros(n), gb, zeros(n), zeros(m.n_exo))
            if isempty(relaxtxt)
                dr = db                 # OccBin leaves when the bind condition fails: d_b >= 0
            else
                rex = _parse_constraint(relaxtxt)
                nr = _negative_side(rex)
                nr === nothing && throw(ModelError("the relax condition '$relaxtxt' is not a single <, <=, > or >= comparison"))
                vr, gr = _linearise_condition(m, nr, ys, base)
                dr = -vcat(vr, zeros(n), gr, zeros(n), zeros(m.n_exo))      # relax holds  <=>  d_r > 0
            end
        catch err
            err isa Union{ModelError,ErrorException,MethodError,DomainError,UndefVarError,ArgumentError} || rethrow()
            push!(out, mk(:inconsistent, "the bind/relax condition could not be linearised: " * first(sprint(showerror, err), 200)))
            continue
        end
        va, vr_ = _vec(alt[1]), _vec(ref[1])
        a, b, rb = _fit2(db, va, vr_)
        cc, ee, rr = _fit2(dr, vr_, va)
        s_g = abs(a) * norm(va) > OBC_FIT_TOL * norm(db) ? Int(sign(a)) : 0
        s_y = abs(cc) * norm(vr_) > OBC_FIT_TOL * norm(dr) ? -Int(sign(cc)) : 0     # leave  <=>  d_r > 0  <=>  y < 0
        g_ss = s_g * alt[1].c
        reasons = String[]
        if rb > OBC_APPROX_TOL || s_g == 0
            push!(reasons, "entry test: the linearised bind condition is not a multiple of the alternative residual " *
                           "(plus a multiple of the reference one) [relative residual $(@sprintf("%.1e", rb))" *
                           _offending_terms(m, db - a * va - b * vr_) * "]")
        end
        if rr > OBC_APPROX_TOL || s_y == 0
            push!(reasons, "exit test: the linearised " * (isempty(relaxtxt) ? "bind (no relax given)" : "relax") *
                           " condition is not a multiple of the reference residual (plus a multiple of the alternative one) " *
                           "[relative residual $(@sprintf("%.1e", rr))" * _offending_terms(m, dr - cc * vr_ - ee * va) * "]")
        end
        if isempty(reasons)
            tolo = 1e-6 * max(abs(a), abs(cc))
            (b * s_y > tolo || ee * s_g < -tolo) &&
                push!(reasons, "orientation: the relax condition points the wrong way, it is not the complement of the bind condition (b*s_y = $(@sprintf("%.3g", b * s_y)), " *
                               "e*s_g = $(@sprintf("%.3g", ee * s_g)); in the alternative regime the bind condition would fail while the " *
                               "relax condition says stay, or the reverse)")
        end
        if isempty(reasons) && !(g_ss > 1e-12)
            push!(reasons, "the gap is not strictly positive at the steady state (g_ss = $(@sprintf("%.3g", g_ss)))")
        end
        if abs(ref[1].c) > 1e-8 * max(1.0, norm(ref[1].g, Inf))
            push!(reasons, "the reference residual is not zero at the steady state ($(ref[1].c))")
        end
        cls = !isempty(others) ? :shared_equation : isempty(reasons) ? :complementarity : :inconsistent
        approx = cls == :complementarity && max(rb, rr) > OBC_FIT_TOL ? max(rb, rr) : 0.0
        why = !isempty(others) ? "equation '$(_eq_name(m, e))' is also switched by constraint " * join(others, ", ") *
                                 (isempty(reasons) ? " (alone it would pass the entry and exit tests)" : "; also " * join(reasons, "; ")) :
              join(reasons, "; ")
        cls == :complementarity && approx > 0 &&
            (why = "approximate: the linearised conditions match the residuals to a relative $(@sprintf("%.1e", approx))" *
                   _offending_terms(m, (rb >= rr ? db - a * va - b * vr_ : dr - cc * vr_ - ee * va)))
        push!(out, mk(cls, why; s_g=s_g, s_y=s_y, entry=(a, b), exit=(cc, ee), fit=(rb, rr), approx=approx, g_ss=g_ss))
    end
    out
end

# the variables (with lag/lead) carrying the largest misfit, for the user-facing reason
function _offending_terms(m::Model, r::AbstractVector)
    n = m.n_endo
    labels = vcat("constant", [v * "(-1)" for v in m.endo_names], m.endo_names, [v * "(+1)" for v in m.endo_names], m.exo_names)
    big = sortperm(abs.(r); rev=true)
    s = [labels[i] for i in big[1:min(3, end)] if abs(r[i]) > OBC_FIT_TOL * max(norm(r, Inf), 1e-300) * 1e3 && abs(r[i]) > 1e-12]
    isempty(s) ? "" : "; misfit on " * join(s, ", ")
end

obc_constraints(ob::Occbin) = obc_constraints(ob.model; params=ob.params, ys=ob.ys)

"""
    obc_classification(m::Model) -> rows

The classification table of a model: one named tuple per constraint with `constraint, class, equation, s_g, s_y,
approx, bind, relax, why`.
"""
function obc_classification(m::Model)
    [(constraint=c.name, class=c.class, equation=join(c.eq_names, "; "), s_g=c.s_g, s_y=c.s_y, approx=c.approx, bind=c.bind,
      relax=c.relax, why=c.why) for c in obc_constraints(m)]
end

"Refuse a model whose constraints are not all `complementarity` for the non-OccBin methods."
function _require_complementarity(cons::Vector{ObcConstraint}, method::Symbol)
    bad = [c for c in cons if c.class != :complementarity]
    isempty(bad) && return nothing
    throw(ModelError("obc_method = $method needs every occasionally binding constraint to switch one equation between two " *
                     "regimes consistent with its bind/relax conditions (class 'complementarity'); " *
                     join(["constraint '$(c.name)' is $(c.class): $(c.why)" for c in bad], "; ") *
                     ". Use obc_method = occbin for this model."))
end

# ---------------------------------------------------------------------- M and q
"""
    ObcSystem

The stacked complementarity problem of a model over `T` periods (contract section 4). Wedges `y` and gaps `g` are
stacked by constraint, then by period: entry `(i - 1) * T + t` is constraint `i` in period `t - 1`. With the wedge path
known from period 0 (zero from `T` on) and everything else in the reference regime, the gaps over the horizon are
`g = q + M y` with `q = q0 + Qx * [y_init; u]` (`y_init`: the state, deviations from the steady state in declaration
order; `u`: this period's shocks).

Construction: the first-order solution with anticipated shocks. With the reference rule `y_t = P y_{t-1} + Q u_t` and
the linearised reference model `A y_{t+1} + B y_t + C y_{t-1} + J u_t + f_t = 0`, a known forcing path `f` gives
`y_t = P y_{t-1} + h_t` with `h_t = F h_{t+1} - (B + A P)^{-1} f_t`, `F = -(B + A P)^{-1} A`. The wedge of constraint
`j` is the forcing `f_t = -s_y y_j(t) e_eq`, i.e. its reference equation reads `ref_resid = s_y y_j(t)`.
"""
struct ObcSystem
    T::Int
    cons::Vector{ObcConstraint}
    eq::Vector{Int}
    n::Int
    n_exo::Int
    P::Matrix{Float64}
    Q::Matrix{Float64}
    F::Matrix{Float64}
    V::Matrix{Float64}          # n x nc: response of h_t to a unit wedge of constraint j in period t
    G::Matrix{Float64}          # nc x (3n + n_exo): s_g * gradient of the alternative residual
    g0::Vector{Float64}         # nc: s_g * alternative residual at the steady state (the gap there)
    M::Matrix{Float64}
    q0::Vector{Float64}
    Qx::Matrix{Float64}
    Wn::Matrix{Float64}         # n x N: response of the next state (period 1) to the wedges
    ext::Dict{Int,Any}          # lazily built ObcExt per number of checked periods Lg
end
"Gaps beyond the horizon (periods T+1..Lg) as affine functions of the wedges and of [state; shocks]: g_b = q0b + Qxb [init; u] + Mb y."
struct ObcExt
    Lg::Int
    Mb::Matrix{Float64}
    Qxb::Matrix{Float64}
    q0b::Vector{Float64}
end
Base.show(io::IO, s::ObcSystem) = print(io, "ObcSystem(T = ", s.T, ", ", length(s.cons), " constraint(s), M ", size(s.M, 1), "x", size(s.M, 2), ")")

const _OBC_CACHE = WeakKeyDict{Occbin,Dict{Int,ObcSystem}}()
const _OBC_CONS_CACHE = WeakKeyDict{Occbin,Vector{ObcConstraint}}()

"""
    obc_system(ob::Occbin; horizon=40, method=:minnorm) -> ObcSystem

`M`, `q0` and `Qx` for the model of `ob` (its parameters at construction) over `horizon` periods; cached per `Occbin`
object and horizon (they depend only on the parameters). Refuses models whose constraints are not all
`complementarity`.
"""
function obc_system(ob::Occbin; horizon::Integer=40, method::Symbol=:minnorm)
    cache = get!(() -> Dict{Int,ObcSystem}(), _OBC_CACHE, ob)
    haskey(cache, horizon) && return cache[horizon]
    horizon >= 1 || throw(ArgumentError("obc_horizon must be at least 1"))
    cons = get!(() -> obc_constraints(ob), _OBC_CONS_CACHE, ob)
    _require_complementarity(cons, method)
    n, nx, T = ob.n, ob.n_exo, Int(horizon)
    C, B, A, J, _ = regime_matrices(ob, ntuple(_ -> false, ob.k))
    BAP = lu(B + A * ob.P)
    F = -(BAP \ A)
    nc = length(cons)
    eq = [c.equations[1] for c in cons]
    V = zeros(n, nc)
    for j in 1:nc
        ej = zeros(n); ej[eq[j]] = 1.0
        V[:, j] = cons[j].s_y .* (BAP \ ej)             # h = -(B+AP)^{-1} f with f = -s_y e_eq
    end
    G = zeros(nc, 3n + nx)
    g0 = zeros(nc)
    for j in 1:nc
        G[j, :] = cons[j].s_g .* cons[j].alt_resid[1].g
        g0[j] = cons[j].s_g * cons[j].alt_resid[1].c
    end
    q0 = repeat(g0; inner=T)
    N = nc * T
    Wn = zeros(n, N)
    # C_r = F^r V (n x nc): the anticipation of a wedge r periods ahead; the next state responds to wedge s with C_{s-1}
    Cr = Matrix{Float64}(V)
    for r in 0:T-1
        for j in 1:nc
            Wn[:, (j-1)*T+r+1] = view(Cr, :, j)
        end
        r < T - 1 && (Cr = F * Cr)
    end
    sys0 = ObcSystem(T, cons, eq, n, nx, ob.P, ob.Q, F, V, G, g0, zeros(0, 0), q0, zeros(0, 0), Wn, Dict{Int,Any}())
    M, Qx = _obc_blocks(sys0, 1, T)
    sys = ObcSystem(T, cons, eq, n, nx, ob.P, ob.Q, F, V, G, g0, M, q0, Qx, Wn, sys0.ext)
    cache[horizon] = sys
end

"""
Rows for periods `t0..t1` of the gaps' response to the wedges (`M`-like, stacked by constraint then period) and to `[state; shocks]`.
With `a_m = Gf P^(m+1) + Gc P^m + Gl P^(m-1)` (terms with a negative power dropped) and `c_r = F^r V`, the response of gap `i` in
period `t` to a unit wedge `j` in period `s` is `sum_{r=0}^{s-1} a_(t-s+r) c_r`: one matrix product for all the terms and a running sum
along each diagonal, O(T^2 n) instead of one path per column.
"""
function _obc_blocks(sys::ObcSystem, t0::Int, t1::Int)
    n, nx, T, nc = sys.n, sys.n_exo, sys.T, length(sys.cons)
    P, G = sys.P, sys.G
    Gl, Gc, Gf, Gu = view(G, :, 1:n), view(G, :, n+1:2n), view(G, :, 2n+1:3n), view(G, :, 3n+1:3n+nx)
    mmax = t1                                   # a_m for m = -1 .. mmax
    # a[m+2] is nc x n
    A = Vector{Matrix{Float64}}(undef, mmax + 2)
    Pm = Matrix{Float64}(I, n, n)               # P^m
    Pm1 = Matrix{Float64}(I, n, n)              # P^(m-1) (valid for m >= 1)
    GfP = Gf * P                                # Gf P^(m+1)
    A[1] = Matrix(Gf)                           # m = -1
    GfPm = Matrix(GfP)
    GcPm = Matrix(Gc)
    GlPm1 = zeros(nc, n)
    for m in 0:mmax
        A[m+2] = GfPm .+ GcPm .+ (m >= 1 ? GlPm1 : 0.0)
        # advance to m + 1
        GfPm = GfPm * P
        GcPm = GcPm * P
        GlPm1 = m == 0 ? Matrix(Gl) : GlPm1 * P
    end
    # c_r = F^r V
    Cs = Vector{Matrix{Float64}}(undef, T)
    Cs[1] = Matrix(sys.V)
    for r in 2:T
        Cs[r] = sys.F * Cs[r-1]
    end
    Cmat = reduce(hcat, Cs)                     # n x (nc*T), column (r, j) at (r-1)*nc + j
    Lr = t1 - t0 + 1
    M = zeros(nc * Lr, nc * T)
    # all the terms a_m c_r in one product: E[(m+1)*nc + i, r*nc + j]
    Amat = reduce(vcat, A)                      # (mmax+2)*nc x n, row block m+2
    E = Amat * Cmat
    # response of gap i in period t to wedge j in period s: sum_{r=0}^{s-1} a_(t-s+r) c_r, accumulated along the diagonal d = t - s
    for i in 1:nc, j in 1:nc, d in (t0-T):(t1-1)
        acc = 0.0
        for s in 1:min(T, t1 - d)
            m = d + s - 1                       # term r = s - 1
            m >= -1 && (acc += E[(m+1)*nc+i, (s-1)*nc+j])
            t = s + d
            t >= t0 && (M[(i-1)*Lr+(t-t0+1), (j-1)*T+s] = acc)
        end
    end
    Qx = zeros(nc * Lr, n + nx)
    for i in 1:nc, t in t0:t1
        row = (i-1)*Lr + (t - t0 + 1)
        Qx[row, 1:n] = view(A[t+2], i, :)                        # init: a_t
        Qx[row, n+1:n+nx] = view(A[t+1], i, :)' * sys.Q          # shocks: a_(t-1) Q
        t == 1 && (Qx[row, n+1:n+nx] .+= view(Gu, i, :))
    end
    M, Qx
end

"The gaps beyond the horizon, periods T+1..Lg (cached per Lg)."
function _obc_ext(sys::ObcSystem, Lg::Int)
    get!(sys.ext, Lg) do
        nc, T = length(sys.cons), sys.T
        Lg > T || return ObcExt(Lg, zeros(0, nc * T), zeros(0, sys.n + sys.n_exo), zeros(0))
        Mb, Qb = _obc_blocks(sys, T + 1, Lg)
        ObcExt(Lg, Mb, Qb, repeat(sys.g0; inner=Lg - T))
    end::ObcExt
end

"`q` of the problem for state `init` (deviations) and shocks `u`."
obc_q(sys::ObcSystem, init::AbstractVector, u::AbstractVector) = sys.q0 .+ sys.Qx * vcat(init, u)

# the path (max(L, T) + 1 rows, deviations, periods 0..) under wedges y from state init with shocks u in period 0
function _obc_path_full(sys::ObcSystem, y::AbstractVector, init::AbstractVector, u::AbstractVector, L::Integer)
    n, T, nc = sys.n, sys.T, length(sys.cons)
    Lp = max(L, T) + 1
    h = zeros(n, T)
    hn = zeros(n)
    w = zeros(nc)
    for t in T:-1:1
        for j in 1:nc
            w[j] = y[(j-1)*T+t]
        end
        hn = sys.F * hn .+ sys.V * w
        h[:, t] = hn
    end
    Z = zeros(Lp, n)
    x = Vector{Float64}(init)
    for t in 1:Lp
        x = sys.P * x
        t <= T && (x .+= view(h, :, t))
        t == 1 && (x .+= sys.Q * u)
        Z[t, :] = x
    end
    Z
end

"""
    obc_gaps(sys, y, init, u, L; homogeneous=false) -> (Z, g)

The path `Z` (`L x n`, deviations, periods 0..L-1) under wedges `y` (stacked, `T` per constraint, zero afterwards)
from state `init` with shocks `u` in period 0, and the gaps `g` (`nc x L`) along it. `homogeneous` drops the gaps'
constant (the response to `y`, `init` and `u` alone).
"""
function obc_gaps(sys::ObcSystem, y::AbstractVector, init::AbstractVector, u::AbstractVector, L::Integer; homogeneous::Bool=false)
    n, T, nc = sys.n, sys.T, length(sys.cons)
    Z = _obc_path_full(sys, y, init, u, L)
    g = zeros(nc, L)
    Gl, Gc, Gf, Gu = (view(sys.G, :, 1:n), view(sys.G, :, n+1:2n), view(sys.G, :, 2n+1:3n), view(sys.G, :, 3n+1:3n+sys.n_exo))
    for t in 1:L
        prev = t == 1 ? init : view(Z, t - 1, :)
        gt = Gl * prev .+ Gc * view(Z, t, :) .+ Gf * view(Z, t + 1, :)
        t == 1 && (gt .+= Gu * u)
        homogeneous || (gt .+= sys.g0)
        g[:, t] = gt
    end
    Z[1:L, :], g
end

"""
    obc_diagnostics(sys::ObcSystem; max_exact=16) -> NamedTuple
    obc_diagnostics(ob::Occbin; horizon=40, max_exact=16)

Existence and uniqueness diagnostics (`m_diagnostics`) on the `M` of the stacked complementarity problem: S-matrix test (a linear
programme), exact P-matrix test (Tsatsomeros-Li recursion) up to `max_exact` rows, positive definite symmetric part (sufficient for P),
`largest_p`. Cached per `ObcSystem` (they depend on the parameters only). For the regime formulation see the method on `RegimeOBC`.
"""
function obc_diagnostics(sys::ObcSystem; max_exact::Integer=16)
    get!(sys.ext, -Int(max_exact)) do
        m_diagnostics(sys.M; n_c=length(sys.cons), max_exact=max_exact)
    end
end
obc_diagnostics(ob::Occbin; horizon::Integer=40, max_exact::Integer=16) = obc_diagnostics(obc_system(ob; horizon=horizon, method=:news); max_exact=max_exact)

# ---------------------------------------------------------------------- simulation
"""
    ObcNews

What the `minnorm` / `news` methods add to a simulation (Dynare-style `oo_.occbin.news`): `method`, `horizon`,
`constraints` (names), `wedges[t, :]` the wedge path used in shock period `t` (stacked by constraint then period,
`horizon` entries per constraint; in the units of `s_y * ref_resid`; the regime formulation has one block of `horizon` wedges per
switched equation), `status[t]` and the solver's `info[t]`. `formulation` is `:lcp` (every constraint a complementary pair: Holden's
LCP, `M` and `q`) or `:regime` (the regime formulation, src/obc_regime.jl); `diagnostics` is the `m_diagnostics` of `M` for
`method = :news` (`nothing` otherwise, and for the regime formulation unless the tied-wedge reduction applies).
"""
struct ObcNews
    method::Symbol
    horizon::Int
    constraints::Vector{String}
    wedges::Matrix{Float64}
    status::Vector{Symbol}
    info::Vector{Any}
    formulation::Symbol
    diagnostics::Any
end
ObcNews(a1, a2, a3, a4, a5, a6) = ObcNews(a1, a2, a3, a4, a5, a6, :lcp, nothing)

# OccBin options that do not apply to the LCP methods (guesses and iteration controls of the regime iteration)
const _OBC_IGNORED_DEFAULTS = (maxit=30, curb_retrench=false, periodic_solution=false, periodic_solution_threshold=1,
                               periodic_solution_strict=true, init_regime=nothing, init_binding_indicator=nothing,
                               reset_check_ahead_periods_in_new_period=false, reset_regime_in_new_period=false,
                               algo_truncation=1)

"""
OccBin's simulation loop (`simulate(ob, shocks)`) with the regime iteration replaced by the `obc_method` solver:
in each period with a shock, the path from the current state is the solution of the complementarity problem over
`obc_horizon` periods (first by the shared active-set fast path unless `fast_path=false`, then by the method's solver), verified beyond the horizon over `check_ahead_periods` more periods; periods without shocks
continue the last path, as OccBin does.
"""
function _simulate_obc(ob::Occbin, shocks_in::AbstractMatrix; method::Symbol, horizon::Integer, tol::Real, raise::Bool,
                       periods::Integer, check_ahead_periods::Integer, max_check_ahead_periods::Real, init, exo_pos,
                       ignored::NamedTuple, solver_options=NamedTuple(), fast_path::Bool=true, selection::Symbol=:occbin)
    k, n, n_exo, m = ob.k, ob.n, ob.n_exo, ob.model
    if method == :news                      # constraints that are not all complementary pairs: the regime formulation (src/obc_regime.jl);
                                            # obc_solver_options = (formulation = :regime,) asks for it on a complementarity model too
        cons = get!(() -> obc_constraints(ob), _OBC_CONS_CACHE, ob)
        (all(c -> c.class == :complementarity, cons) && get(solver_options, :formulation, :auto) != :regime) ||
            return selection == :occbin ?
                   _simulate_regime(ob, shocks_in; method=method, horizon=horizon, tol=tol, raise=raise, periods=periods,
                                    check_ahead_periods=check_ahead_periods, max_check_ahead_periods=max_check_ahead_periods,
                                    init=init, exo_pos=exo_pos, ignored=ignored, solver_options=solver_options, fast_path=fast_path) :
                   throw(ArgumentError("obc_selection = $selection is not implemented for the regime formulation (constraints that are not " *
                                       "one complementary pair each); use obc_selection = occbin"))
    end
    selection != :occbin && (method == :news && length(obc_constraints(ob)) == 1 ||
        throw(ArgumentError("obc_selection = $selection needs obc_method = news and a model with one constraint")))
    changed = [String(key) for (key, v) in pairs(ignored) if !isequal(v, getfield(_OBC_IGNORED_DEFAULTS, key))]
    isempty(changed) || warn_once(m, "obc_method = $method: OccBin options of the regime iteration are not used (ignored): " * join(sort(changed), ", "))
    solve = OBC_SOLVERS[method]
    is_news = method == :news
    sys = obc_system(ob; horizon=horizon, method=method)
    nc, T = length(sys.cons), sys.T
    solve(sys.M, sys.q0; tol=tol)                 # probe: q0 > 0, so a solver returns at once; a placeholder throws here
    exo_idx = exo_pos === nothing ? collect(1:n_exo) : Int[_exo_index(m, p) for p in (exo_pos isa Union{AbstractString,Integer} ? [exo_pos] : exo_pos)]
    size(shocks_in, 2) == length(exo_idx) ||
        throw(ArgumentError("shocks have $(size(shocks_in, 2)) columns but exo_pos selects $(length(exo_idx)) shocks"))
    S = size(shocks_in, 1)
    S >= 1 || throw(ArgumentError("at least one shock period is needed"))
    shocks = zeros(S, n_exo)
    shocks[:, exo_idx] = Float64.(shocks_in)
    n_periods = max(Int(periods), S)
    check = Int(min(check_ahead_periods, max_check_ahead_periods))
    n0 = max(check, n_periods - S)
    L = n0 + 1                                    # rows of OccBin's path in each period
    Lg = max(L, T + check)                        # periods over which the gap is checked (beyond the horizon too)
    y0 = init === nothing || isempty(init) ? zeros(n) : Float64.(vec(collect(init)))
    length(y0) == n || throw(ArgumentError("init needs $n values (declaration order)"))
    init_orig = copy(y0)

    piecewise = zeros(n_periods, n)
    history = Vector{Vector{Tuple{Vector{Int},Vector{Int}}}}()
    iters, diag = Int[], _Diag[]
    ss_out = Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}[]
    wedges = zeros(S, nc * T)
    status = Symbol[]
    infos = Any[]
    y = zeros(nc * T)
    Z = zeros(0, n)                               # the full path of the last solve: built only when needed (shift periods, the tail)
    Zvalid = false
    init_last, u_last = zeros(n), zeros(n_exo)
    ext = _obc_ext(sys, Lg)
    xin = zeros(n + n_exo)
    for sp in 1:S
        _obc_tick()
        u = shocks[sp, :]
        if sp == 1 || any(!iszero, u)
            xin[1:n] = y0; xin[n+1:end] = u
            q = sys.q0 .+ sys.Qx * xin
            st = :not_converged
            if fast_path                           # contract section 5: the shared active-set fast path first
                y, st, info = solve_active_set(sys.M, q; tol=tol)
                # news imposes the gaps beyond the horizon in its own programme: a fast-path answer that violates them goes there too
                if st == :ok && is_news && size(ext.Mb, 1) > 0 && minimum(ext.q0b .+ ext.Qxb * xin .+ ext.Mb * y) < -tol
                    st = :not_converged
                end
            end
            if st != :ok
                extra = is_news ? (; Mb=ext.Mb, qb=ext.q0b .+ ext.Qxb * xin, n_c=nc) : (;)
                (y, st, info) = solve(sys.M, q; tol=tol, extra..., solver_options...)
            end
            st in OBC_STATUSES || error("OBC solver returned an unknown status $st")
            piv = hasproperty(info, :pivots) ? info.pivots : 0
            if st == :ok
                hasproperty(info, :max_violation) || (okT = obc_verify(sys.M, q, y; tol=tol)[1]; okT || (st = :not_converged))
                if st == :ok && size(ext.Mb, 1) > 0
                    gb = ext.q0b .+ ext.Qxb * xin .+ ext.Mb * y        # gaps beyond the horizon, periods T+1..Lg
                    beyond = minimum(gb)
                    if beyond < -tol
                        st = :horizon_too_short
                        info = merge(info, (message="the gap is negative beyond the horizon (min $(beyond) in periods $(T)..$(Lg-1)); increase obc_horizon",))
                    end
                end
            end
            if st == :ok && selection != :occbin && any(!iszero, y)         # obc_selection: choose among the single-spell solutions (src/obc_filter.jl)
                Mfull = size(ext.Mb, 1) > 0 ? vcat(sys.M, ext.Mb) : sys.M
                qfull = size(ext.Mb, 1) > 0 ? vcat(q, ext.q0b .+ ext.Qxb * xin) : q
                y, _ = _select_spell(Mfull, qfull, nc * T, y, selection, max(tol, 1e-8))
            end
            init_last, u_last, Zvalid = copy(y0), copy(u), false
            ynext = ob.P * y0 .+ ob.Q * u .+ sys.Wn * y     # only the next state; the full path is built when it is needed
        else                                       # no new shock: the path solved last continues (OccBin's shift)
            Zvalid || (Z = _obc_path_full(sys, y, init_last, u_last, Lg)[1:Lg, :]; Zvalid = true)
            y = vcat([vcat(y[(j-1)*T+2:j*T], 0.0) for j in 1:nc]...)
            Z = vcat(Z[2:end, :], (ob.P * Z[end, :])')
            st, info, piv = :ok, (pivots=0, message="no shock: previous path continued"), 0
            ynext = Z[1, :]
        end
        if st != :ok && raise
            msg = hasproperty(info, :message) ? info.message : ""
            throw(ObcError(st, sp, "obc_method = $method, shock period $sp: $st. $msg"))
        end
        wedges[sp, :] = y
        push!(status, st)
        push!(infos, info)
        bi = falses(max(L, T), k)                   # rows of the regime history: the whole spell, also when the horizon exceeds OccBin's path length
        for j in 1:nc, t in 1:T
            bi[t, j] = y[(j-1)*T+t] > tol
        end
        push!(history, [_map_regime(view(bi, :, j)) for j in 1:k])
        push!(iters, piv)
        Tmax = _tmax(bi)
        push!(ss_out, Tmax >= 1 ? _rules(ob, bi, Tmax)[1] : (ob.P, ob.Q, zeros(n)))
        push!(diag, (max_err=[hasproperty(info, :max_violation) ? Float64(info.max_violation) : 0.0], violates=[false], n_viol=[0.0], selected=nothing))
        y0 = ynext
        piecewise[sp, :] = y0
    end
    _obc_tick()
    if n_periods > S
        Zvalid || (Z = _obc_path_full(sys, y, init_last, u_last, Lg)[1:Lg, :])
        piecewise[S+1:end, :] = Z[2:n_periods-S+1, :]
    end
    linear = Matrix{Float64}(undef, n_periods, n)
    yl = init_orig
    for t in 1:n_periods
        yl = ob.P * yl + (t <= S ? ob.Q * shocks[t, :] : zeros(n))
        linear[t, :] = yl
    end
    news = ObcNews(method, T, [c.name for c in sys.cons], wedges, status, infos, :lcp, is_news ? obc_diagnostics(sys) : nothing)
    OccbinResult(piecewise .+ ob.ys', linear .+ ob.ys', Matrix{Float64}(shocks_in), history, ob.ys, iters, exo_idx, falses(S),
                 diag, m.endo_names, m.exo_names, ss_out, news)
end

"`obc_*` options set by native MATLAB lines of the .mod file (`options_.occbin.obc_method = 'minnorm';`, `options_.occbin.obc_order = 2;`, ...)."
function _obc_mod_options(m::Model)
    out = Dict{Symbol,Any}()
    o = get(m.matlab_options, "occbin", nothing)
    o isa MStruct || return out
    _obc_flat_options(_flat_options(o))              # obc_method, obc_horizon, obc_selection, obc_order, obc_cubature, ... (src/obc_filter.jl)
end

"`oo_.occbin.news` from an `ObcNews`."
function _news_struct(nw::ObcNews)
    s = MStruct()
    s.method = String(nw.method)
    s.horizon = Float64(nw.horizon)
    s.constraints = Cell(Any[c for c in nw.constraints], (1, length(nw.constraints)))
    s.wedges = Matrix{Float64}(nw.wedges)
    s.status = Cell(Any[String(x) for x in nw.status], (length(nw.status), 1))
    s.formulation = String(nw.formulation)
    d = nw.diagnostics
    if d !== nothing
        ds = MStruct()
        ds.is_s = d.is_s
        ds.is_p = d.is_p === nothing ? "undecided" : d.is_p
        ds.symmetric_part_positive_definite = d.symmetric_part_positive_definite
        ds.largest_p = Float64(d.largest_p)
        ds.min_diag = d.min_diag
        ds.max_exact = Float64(d.max_exact)
        s.diagnostics = ds
    end
    s
end
