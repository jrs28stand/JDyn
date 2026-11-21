function SparseStaticResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 4
    @assert length(residual) == 3
    @assert length(y) == 3
    @assert length(x) == 1
    @assert length(params) == 5
@inbounds begin
    residual[1] = (T[1]) - (T[1]*params[2]*T[3]);
    residual[2] = (y[2]) - (T[4]-y[1]+y[2]*(1-params[4]));
    residual[3] = (y[3]) - (y[3]*params[3]+x[1]);
end
    return nothing
end

