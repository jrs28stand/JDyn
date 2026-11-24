function SparseDynamicResidTT!(T::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
@inbounds begin
T[1] = y[12]^(1+params[7])
T[2] = y[9]*exp(y[13])/(exp(y[20])*y[16])
T[3] = params[4]*exp(y[20])*y[15]+y[10]*(1-params[5])
T[4] = exp(y[11])*y[3]^params[4]
T[5] = y[12]^(1-params[4])
T[6] = T[4]*T[5]
end
    return nothing
end

