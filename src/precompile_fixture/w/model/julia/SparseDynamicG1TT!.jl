function SparseDynamicG1TT!(T::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    SparseDynamicResidTT!(T, y, x, params, steady_state)
@inbounds begin
T[5] = params[2]*get_power_deriv(y[7],(-1),1)
T[6] = exp(y[6])*get_power_deriv(y[2],params[1],1)
T[7] = params[1]*exp(y[9])*get_power_deriv(y[5],params[1]-1,1)
end
    return nothing
end

