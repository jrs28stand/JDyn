function SparseDynamicG1TT!(T::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    SparseDynamicResidTT!(T, y, x, params, steady_state)
@inbounds begin
T[7] = exp(y[13])/(exp(y[20])*y[16])
T[8] = exp(y[20])*y[16]*exp(y[20])*y[16]
T[9] = (-(y[9]*exp(y[13])*exp(y[20])))/T[8]
T[10] = exp(y[11])*get_power_deriv(y[3],params[4],1)
T[11] = (-(T[5]*T[10]))
T[12] = get_power_deriv(y[12],1+params[7],1)
T[13] = get_power_deriv(y[12],1-params[4],1)
T[14] = (-(y[9]*exp(y[13])*exp(y[20])*y[16]))/T[8]
end
    return nothing
end

