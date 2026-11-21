function D = fc_draws(n, T, R)
% the standard normal draws Dynare's forecast/conditional_forecast make after set_dynare_seed (one n x T block per replication)
D = zeros(n, T, R);
for b = 1:R
    D(:,:,b) = randn(n, T);
end
end
