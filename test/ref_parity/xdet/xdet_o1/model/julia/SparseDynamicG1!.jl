function SparseDynamicG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 14
    @assert length(g1_v) == 31
    @assert length(y) == 21
    @assert length(x) == 4
    @assert length(params) == 8
@inbounds begin
g1_v[1]=T[11];
g1_v[2]=(-(1-params[5]));
g1_v[3]=(-params[2]);
g1_v[4]=(-params[8]);
g1_v[5]=(-1);
g1_v[6]=(-params[8]);
g1_v[7]=(-params[2]);
g1_v[8]=(-params[3]);
g1_v[9]=(-(1-params[4]));
g1_v[10]=1;
g1_v[11]=(-exp(y[13]));
g1_v[12]=params[6]*T[1];
g1_v[13]=(-(params[1]*T[3]*T[7]));
g1_v[14]=exp(y[13]);
g1_v[15]=1-params[1]*T[2]*(1-params[5]);
g1_v[16]=1;
g1_v[17]=(-T[6]);
g1_v[18]=1;
g1_v[19]=y[9]*params[6]*T[12];
g1_v[20]=(-(T[4]*T[13]));
g1_v[21]=(-(params[1]*T[2]*T[3]));
g1_v[22]=(-(exp(y[13])*(y[8]-y[9])));
g1_v[23]=1;
g1_v[24]=1;
g1_v[25]=(-(params[1]*T[2]*params[4]*exp(y[20])));
g1_v[26]=(-(params[1]*T[3]*T[9]));
g1_v[27]=(-(params[1]*(T[3]*T[14]+T[2]*params[4]*exp(y[20])*y[15])));
g1_v[28]=(-1);
g1_v[29]=(-1);
g1_v[30]=(-1);
g1_v[31]=(-1);
end
    return nothing
end

