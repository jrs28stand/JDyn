function SparseDynamicG2!(T::Vector{<: Real}, g2_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 14
    @assert length(g2_v) == 28
    @assert length(y) == 21
    @assert length(x) == 4
    @assert length(params) == 8
@inbounds begin
g2_v[1]=params[6]*T[12];
g2_v[2]=y[9]*params[6]*get_power_deriv(y[12],1+params[7],2);
g2_v[3]=(-(params[1]*params[4]*exp(y[20])*T[7]));
g2_v[4]=(-(params[1]*params[4]*exp(y[20])*T[9]));
g2_v[5]=(-(params[1]*T[2]*params[4]*exp(y[20])));
g2_v[6]=(-(params[1]*(T[2]*params[4]*exp(y[20])+params[4]*exp(y[20])*T[14])));
g2_v[7]=(-(params[1]*T[3]*(-(exp(y[13])*exp(y[20])))/T[8]));
g2_v[8]=(-(params[1]*(1-params[5])*T[7]));
g2_v[9]=(-(params[1]*T[3]*T[7]));
g2_v[10]=(-(params[1]*(params[4]*exp(y[20])*y[15]*T[7]+T[3]*(-(exp(y[13])*exp(y[20])*y[16]))/T[8])));
g2_v[11]=(-(params[1]*T[3]*(-((-(y[9]*exp(y[13])*exp(y[20])))*(exp(y[20])*exp(y[20])*y[16]+exp(y[20])*exp(y[20])*y[16])))/(T[8]*T[8])));
g2_v[12]=(-(params[1]*(1-params[5])*T[9]));
g2_v[13]=(-(params[1]*T[3]*T[9]));
g2_v[14]=(-(params[1]*(params[4]*exp(y[20])*y[15]*T[9]+T[3]*((-(y[9]*exp(y[13])*exp(y[20])))*T[8]-(-(y[9]*exp(y[13])*exp(y[20])))*(T[8]+T[8]))/(T[8]*T[8]))));
g2_v[15]=(-(params[1]*T[2]*(1-params[5])));
g2_v[16]=(-(params[1]*(1-params[5])*T[14]));
g2_v[17]=(-(params[1]*T[2]*T[3]));
g2_v[18]=(-(params[1]*(T[3]*T[14]+T[2]*params[4]*exp(y[20])*y[15])));
g2_v[19]=(-(params[1]*(params[4]*exp(y[20])*y[15]*T[14]+T[3]*(T[8]*(-(y[9]*exp(y[13])*exp(y[20])*y[16]))-(-(y[9]*exp(y[13])*exp(y[20])*y[16]))*(T[8]+T[8]))/(T[8]*T[8])+T[2]*params[4]*exp(y[20])*y[15]+params[4]*exp(y[20])*y[15]*T[14])));
g2_v[20]=(-(T[5]*exp(y[11])*get_power_deriv(y[3],params[4],2)));
g2_v[21]=T[11];
g2_v[22]=(-(T[10]*T[13]));
g2_v[23]=(-T[6]);
g2_v[24]=(-(T[4]*T[13]));
g2_v[25]=(-(T[4]*get_power_deriv(y[12],1-params[4],2)));
g2_v[26]=(-exp(y[13]));
g2_v[27]=exp(y[13]);
g2_v[28]=(-(exp(y[13])*(y[8]-y[9])));
end
    return nothing
end

