function SparseDynamicG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 7
    @assert length(g1_v) == 11
    @assert length(y) == 9
    @assert length(x) == 1
    @assert length(params) == 5
@inbounds begin
g1_v[1]=(-(1-params[4]+T[6]));
g1_v[2]=(-params[3]);
g1_v[3]=get_power_deriv(y[4],(-1),1);
g1_v[4]=1;
g1_v[5]=(-(T[1]*T[7]));
g1_v[6]=1;
g1_v[7]=(-T[4]);
g1_v[8]=1;
g1_v[9]=(-(T[3]*T[5]));
g1_v[10]=(-(T[1]*T[2]));
g1_v[11]=(-1);
end
    return nothing
end

