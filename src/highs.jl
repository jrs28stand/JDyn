# A thin wrapper over the HiGHS C API (HiGHS_jll, MIT licence; https://highs.dev) for the OBC solvers: one linear and one
# mixed-integer programme entry point taking dense matrices. No Julia modelling layer (HiGHS.jl, JuMP, MathOptInterface)
# is used, by the owner's decision (docs/OBC_PROGRESS.md). HiGHS's console output is switched off.
#
#     highs_lp(c, A, rlo, rhi, lb, ub)                         -> (status, x)
#     highs_milp(c, A, rlo, rhi, lb, ub, integer; time_limit)  -> (status, x)
#
# minimise c'x subject to rlo <= A x <= rhi and lb <= x <= ub (infinite bounds as +-Inf), x[j] integer where
# integer[j]. `status` is :optimal, :infeasible, :unbounded, :infeasible_or_unbounded, :time_limit, :error or :other
# (HiGHS's model status in `highs_status_code`); `x` is the solution, or empty when HiGHS has no primal feasible
# point (:time_limit with an incumbent returns the incumbent: callers must verify what they use).

using HiGHS_jll: libhighs

const _HIGHS_STATUS = Dict{Int,Symbol}(7 => :optimal, 8 => :infeasible, 9 => :infeasible_or_unbounded, 10 => :unbounded,
                                       13 => :time_limit, 1 => :error, 2 => :error, 3 => :error, 4 => :error, 5 => :error)

"HiGHS's model status code of the last `highs_lp` / `highs_milp` call in this task (7 = optimal, 8 = infeasible)."
const highs_status_code = Ref{Int}(0)

function _highs_solve(c::AbstractVector, A::AbstractMatrix, rlo::AbstractVector, rhi::AbstractVector, lb::AbstractVector,
                      ub::AbstractVector, integer::Union{Nothing,AbstractVector}, time_limit, options=NamedTuple())
    m, n = size(A)
    (length(c) == n && length(lb) == n && length(ub) == n && length(rlo) == m && length(rhi) == m) ||
        throw(DimensionMismatch("highs: A is $m x $n but c, bounds have lengths $(length(c)), $(length(lb)), $(length(ub)), $(length(rlo)), $(length(rhi))"))
    # column-wise sparse copy of the dense matrix
    start = Cint[0]
    idx = Cint[]
    val = Float64[]
    for j in 1:n
        for i in 1:m
            a = A[i, j]
            if a != 0
                push!(idx, i - 1)
                push!(val, a)
            end
        end
        push!(start, length(idx))
    end
    cc, lbc, ubc, rl, ru = Float64.(c), Float64.(lb), Float64.(ub), Float64.(rlo), Float64.(rhi)
    h = ccall((:Highs_create, libhighs), Ptr{Cvoid}, ())
    h == C_NULL && error("HiGHS: Highs_create failed")
    try
        ccall((:Highs_setBoolOptionValue, libhighs), Cint, (Ptr{Cvoid}, Cstring, Cint), h, "output_flag", 0)
        ccall((:Highs_setBoolOptionValue, libhighs), Cint, (Ptr{Cvoid}, Cstring, Cint), h, "log_to_console", 0)
        if time_limit !== nothing
            ccall((:Highs_setDoubleOptionValue, libhighs), Cint, (Ptr{Cvoid}, Cstring, Cdouble), h, "time_limit", Float64(time_limit))
        end
        for (k, v) in pairs(options)                       # HiGHS option values by type: Bool, Integer, Float64
            name = String(k)
            v isa Bool ? ccall((:Highs_setBoolOptionValue, libhighs), Cint, (Ptr{Cvoid}, Cstring, Cint), h, name, v) :
            v isa Integer ? ccall((:Highs_setIntOptionValue, libhighs), Cint, (Ptr{Cvoid}, Cstring, Cint), h, name, v) :
            ccall((:Highs_setDoubleOptionValue, libhighs), Cint, (Ptr{Cvoid}, Cstring, Cdouble), h, name, Float64(v))
        end
        if integer === nothing
            r = ccall((:Highs_passLp, libhighs), Cint,
                      (Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                       Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
                      h, n, m, length(val), 1, 1, 0.0, cc, lbc, ubc, rl, ru, start, idx, val)
        else
            integ = Cint[integer[j] ? 1 : 0 for j in 1:n]
            r = ccall((:Highs_passMip, libhighs), Cint,
                      (Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                       Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}),
                      h, n, m, length(val), 1, 1, 0.0, cc, lbc, ubc, rl, ru, start, idx, val, integ)
        end
                r < 0 && (highs_status_code[] = 1; return (:error, Float64[]))           # HiGHS: -1 error, 0 ok, 1 warning (accepted)
        ccall((:Highs_run, libhighs), Cint, (Ptr{Cvoid},), h)
        code = Int(ccall((:Highs_getModelStatus, libhighs), Cint, (Ptr{Cvoid},), h))
        highs_status_code[] = code
        status = get(_HIGHS_STATUS, code, :other)
        pss = Ref{Cint}(0)                                  # 2 = a primal feasible solution is available
        ccall((:Highs_getIntInfoValue, libhighs), Cint, (Ptr{Cvoid}, Cstring, Ptr{Cint}), h, "primal_solution_status", pss)
        x = Float64[]
        if status == :optimal || (pss[] == 2 && status in (:time_limit, :other))
            x = zeros(n)
            ccall((:Highs_getSolution, libhighs), Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                  h, x, C_NULL, C_NULL, C_NULL)
        end
        (status, x)
    finally
        ccall((:Highs_destroy, libhighs), Cvoid, (Ptr{Cvoid},), h)
    end
end

"""
    highs_lp(c, A, rlo, rhi, lb, ub; options=NamedTuple()) -> (status, x)

Linear programme `min c'x` s.t. `rlo <= A x <= rhi`, `lb <= x <= ub` with HiGHS (dense `A`). See the file header for `status`.
"""
highs_lp(c, A, rlo, rhi, lb, ub; options=NamedTuple()) = _highs_solve(c, A, rlo, rhi, lb, ub, nothing, nothing, options)

"""
    highs_milp(c, A, rlo, rhi, lb, ub, integer; time_limit=nothing, options=NamedTuple()) -> (status, x)

Mixed-integer programme: as `highs_lp`, with `x[j]` integer where `integer[j]` is true. `time_limit` in seconds; when
it is hit the status is `:time_limit` and `x` is the incumbent if HiGHS has one (not necessarily optimal), else empty.
`options` sets HiGHS options by name (e.g. `(mip_rel_gap=0.0,)`).
"""
highs_milp(c, A, rlo, rhi, lb, ub, integer; time_limit=nothing, options=NamedTuple()) =
    _highs_solve(c, A, rlo, rhi, lb, ub, integer, time_limit, options)
