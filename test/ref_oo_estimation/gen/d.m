addpath /root/ref/dynare-7.2/matlab
addpath /root/ref/dynare-7.2/matlab/estimation
addpath /root/ref/stubs
x = [1 + 0.3*sin((1:3000)'*0.37) + 0.1*cos((1:3000)'*1.91); repeat_ = 0.5*ones(150,1); 0.2*sin((1:350)'*0.11)];
x = x(1:3500);
[post_mean, post_median, post_var, hpd_interval, post_deciles, density] = posterior_moments(x, 0.9);
r = struct('x', x, 'post_mean', post_mean, 'post_median', post_median, 'post_var', post_var, 'hpd', hpd_interval, 'deciles', post_deciles, 'density', density);
save('-mat7-binary', 'dens_ref.mat', '-struct', 'r');
disp('OCT_OK')
