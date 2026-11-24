function SparseStaticResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 4
    @assert length(residual) == 7
    @assert length(y) == 7
    @assert length(x) == 4
    @assert length(params) == 8
@inbounds begin
    residual[1] = (y[2]*params[6]*T[1]) - ((1-params[4])*y[1]);
    residual[2] = (y[3]) - (params[1]*(y[1]*params[4]*exp(y[6])+y[3]*(1-params[5])));
    residual[3] = (y[1]) - (T[4]);
    residual[4] = (y[3]) - (y[3]*(1-params[5])+exp(y[6])*(y[1]-y[2]));
    residual[5] = (y[4]) - (x[1]+y[6]*params[8]+x[3]+y[4]*params[2]+params[3]*y[7]);
    residual[6] = (y[6]) - (x[4]+y[4]*params[8]+y[6]*params[2]+x[2]);
    residual[7] = (y[7]) - (y[4]);
end
    return nothing
end

