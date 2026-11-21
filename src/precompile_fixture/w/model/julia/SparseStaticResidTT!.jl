function SparseStaticResidTT!(T::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
@inbounds begin
T[1] = y[1]^(-1)
T[2] = params[1]*exp(y[3])*y[2]^(params[1]-1)
T[3] = 1+T[2]-params[4]
T[4] = exp(y[3])*y[2]^params[1]
end
    return nothing
end

