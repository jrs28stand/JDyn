function SparseStaticG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 4
    @assert length(g1_v) == 21
    @assert length(y) == 7
    @assert length(x) == 4
    @assert length(params) == 8
@inbounds begin
g1_v[1]=(-(1-params[4]));
g1_v[2]=(-(params[1]*params[4]*exp(y[6])));
g1_v[3]=1;
g1_v[4]=(-exp(y[6]));
g1_v[5]=params[6]*T[1];
g1_v[6]=exp(y[6]);
g1_v[7]=1-params[1]*(1-params[5]);
g1_v[8]=(-(T[3]*exp(y[4])*get_power_deriv(y[3],params[4],1)));
g1_v[9]=1-(1-params[5]);
g1_v[10]=(-T[4]);
g1_v[11]=1-params[2];
g1_v[12]=(-params[8]);
g1_v[13]=(-1);
g1_v[14]=y[2]*params[6]*get_power_deriv(y[5],1+params[7],1);
g1_v[15]=(-(T[2]*get_power_deriv(y[5],1-params[4],1)));
g1_v[16]=(-(params[1]*y[1]*params[4]*exp(y[6])));
g1_v[17]=(-(exp(y[6])*(y[1]-y[2])));
g1_v[18]=(-params[8]);
g1_v[19]=1-params[2];
g1_v[20]=(-params[3]);
g1_v[21]=1;
end
    return nothing
end

