function SparseDynamicG2!(T::Vector{<: Real}, g2_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 7
    @assert length(g2_v) == 10
    @assert length(y) == 9
    @assert length(x) == 1
    @assert length(params) == 5
@inbounds begin
g2_v[1]=get_power_deriv(y[4],(-1),2);
g2_v[2]=(-(T[3]*params[2]*get_power_deriv(y[7],(-1),2)));
g2_v[3]=(-(T[5]*T[7]));
g2_v[4]=(-(T[2]*T[5]));
g2_v[5]=(-(T[1]*params[1]*exp(y[9])*get_power_deriv(y[5],params[1]-1,2)));
g2_v[6]=(-(T[1]*T[7]));
g2_v[7]=(-(T[1]*T[2]));
g2_v[8]=(-(exp(y[6])*get_power_deriv(y[2],params[1],2)));
g2_v[9]=(-T[6]);
g2_v[10]=(-T[4]);
end
    return nothing
end

