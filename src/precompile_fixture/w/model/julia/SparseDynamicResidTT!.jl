function SparseDynamicResidTT!(T::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
@inbounds begin
T[1] = params[2]*y[7]^(-1)
T[2] = params[1]*exp(y[9])*y[5]^(params[1]-1)
T[3] = 1+T[2]-params[4]
T[4] = exp(y[6])*y[2]^params[1]
end
    return nothing
end

