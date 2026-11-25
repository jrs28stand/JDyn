function SparseDynamicResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 6
    @assert length(residual) == 7
    @assert length(y) == 21
    @assert length(x) == 4
    @assert length(params) == 8
@inbounds begin
    residual[1] = (y[9]*params[6]*T[1]) - ((1-params[4])*y[8]);
    residual[2] = (y[10]) - (params[1]*T[2]*T[3]);
    residual[3] = (y[8]) - (T[6]);
    residual[4] = (y[10]) - (exp(y[13])*(y[8]-y[9])+(1-params[5])*y[3]);
    residual[5] = (y[11]) - (x[1]+params[8]*y[6]+x[3]+params[2]*y[4]+params[3]*y[7]);
    residual[6] = (y[13]) - (x[4]+y[4]*params[8]+params[2]*y[6]+x[2]);
    residual[7] = (y[14]) - (y[4]);
end
    return nothing
end

