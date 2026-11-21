function SparseStaticG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 5
    @assert length(g1_v) == 7
    @assert length(y) == 3
    @assert length(x) == 1
    @assert length(params) == 5
@inbounds begin
g1_v[1]=T[5]-T[3]*params[2]*T[5];
g1_v[2]=1;
g1_v[3]=(-(T[1]*params[2]*params[1]*exp(y[3])*get_power_deriv(y[2],params[1]-1,1)));
g1_v[4]=1-(1-params[4]+exp(y[3])*get_power_deriv(y[2],params[1],1));
g1_v[5]=(-(T[1]*params[2]*T[2]));
g1_v[6]=(-T[4]);
g1_v[7]=1-params[3];
end
    return nothing
end

