# `heteroskedastic_shocks` blocks and the `heteroskedastic_filter` estimation option (port of pydyn/heteroskedastic.py; Dynare's
# get_Qvec_heteroskedastic_filter.m and the "set heteroskedastic shocks" section of dynare_estimation_init.m).
# `values` replace the variance of a shock (given as standard deviations) in the periods, `scales` multiply its standard deviation.
# Periods are those of the .mod file, counted from first_obs; the arrays have nobs + 1 columns. Periods given as dates are refused.

"`(Qvalue_orig, Qscale_orig)`: `(exo index, first period, last period, value)` in the order of the .mod file; `overwrite` discards earlier blocks."
function heteroskedastic_blocks(m::Model)
    values = Tuple{Int,Int,Int,Float64}[]
    scales = Tuple{Int,Int,Int,Float64}[]
    for st in m.statements
        get_or(st, "statementName", "") == "heteroskedastic_shocks" || continue
        if get_or(st, "overwrite", false) == true
            empty!(values); empty!(scales)
        end
        for (key, sub, out) in (("shocks_values", "values", values), ("shocks_scales", "scales", scales))
            for d in get_or(st, key, [])
                haskey(m.ix, d["var"]) || throw(ModelError("heteroskedastic_shocks: $(d["var"]) is not a stochastic shock"))
                for v in d[sub]
                    (v["period1"] isa AbstractString && occursin(r"[A-Za-z]", v["period1"])) &&
                        throw(ModelError("heteroskedastic_shocks: periods given as dates are not supported"))
                    push!(out, (m.ix[d["var"]], Int(_numval(v["period1"])), Int(_numval(v["period2"])), Float64(_value(m, v["value"]))))
                end
            end
        end
    end
    values, scales
end

has_heteroskedastic_blocks(m::Model) = any(st -> get_or(st, "statementName", "") == "heteroskedastic_shocks", m.statements)

"`M_.heteroskedastic_shocks.Qvalue` and `.Qscale`: (n_exo, nobs + 1), NaN where not set (variances / squared scales)."
function heteroskedastic_arrays(m::Model, nobs::Integer, first_obs::Integer=1)
    values, scales = heteroskedastic_blocks(m)
    Qvalue = fill(NaN, m.n_exo, nobs + 1)
    Qscale = fill(NaN, m.n_exo, nobs + 1)
    for (out, entries) in ((Qvalue, values), (Qscale, scales))
        for (i, p1, p2, v) in entries
            for p in max(p1, first_obs):min(p2, nobs + first_obs)
                out[i, p-first_obs+1] = v^2
            end
        end
    end
    any(.!isnan.(Qvalue) .& .!isnan.(Qscale)) &&
        throw(ModelError("Scale and value defined for the same shock in the same period with \"heteroskedastic_shocks\"."))
    Qvalue, Qscale
end

"`get_Qvec_heteroskedastic_filter.m`: the (n_exo, n_exo, nobs + 1) path of the shock covariance matrix."
function heteroskedastic_qvec(Q::AbstractMatrix, Qvalue::AbstractMatrix, Qscale::AbstractMatrix)
    Q = Matrix{Float64}(Q)
    kn = size(Qvalue, 2)
    isqdiag = all(abs.(Q .- Diagonal(diag(Q))) .< 1e-14)
    out = repeat(Q, 1, 1, kn)
    for k in 1:kn
        Qk = @view out[:, :, k]
        inx = findall(!isnan, Qvalue[:, k])
        if !isempty(inx)
            if isqdiag
                Qk[inx, inx] = Matrix(Diagonal(Qvalue[inx, k]))
            else
                for s in inx
                    if Q[s, s] > 1e-14
                        sc = sqrt(Qvalue[s, k] / Q[s, s])
                        Qk[s, :] .*= sc
                        Qk[:, s] .*= sc
                    else
                        Qk[s, s] = Qvalue[s, k]
                    end
                end
            end
        end
        inx = findall(!isnan, Qscale[:, k])
        if !isempty(inx)
            if isqdiag
                Qk[inx, inx] = Qk[inx, inx] .* Matrix(Diagonal(Qscale[inx, k]))
            else
                for s in inx
                    sc = sqrt(Qscale[s, k])
                    Qk[s, :] .*= sc
                    Qk[:, s] .*= sc
                end
            end
        end
    end
    out
end

"`R Q_t R'` for every slice of `Qv`: (n_states, n_states, nobs + 1)."
function heteroskedastic_rqr(R::AbstractMatrix, Qv::AbstractArray{<:Real,3})
    out = Array{Float64}(undef, size(R, 1), size(R, 1), size(Qv, 3))
    for k in axes(Qv, 3)
        out[:, :, k] = R * Qv[:, :, k] * R'
    end
    out
end
