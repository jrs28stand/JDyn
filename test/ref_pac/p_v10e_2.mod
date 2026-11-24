// --+ options: json=compute, stochastic +--

var y x z v;

varexo ex ey ez ;

parameters a_y_1 a_y_2 b_y_1 b_y_2 b_x_1 b_x_2 d_y; // VAR parameters

parameters beta e_c_m c_z_1 c_z_2;               // PAC equation parameters

a_y_1 =  .2;
a_y_2 =  .3;
b_y_1 =  .1;
b_y_2 =  .4;
b_x_1 = -.1;
b_x_2 = -.2;
d_y = .5;


beta  =  .9;
e_c_m =  .1;
c_z_1 =  .7;
c_z_2 = -.3;

var_model(model_name=toto, structural, eqtags=['eq:x', 'eq:y', 'eq:v']);

pac_model(auxiliary_model_name=toto, discount=beta, kind=dd, growth=diff(v(-1)), model_name=pacman);

model(use_dll);

  [name='eq:y']
  diff(y) = .01 + a_y_1*diff(y(-1)) + a_y_2*diff(x(-1)) + b_y_1*diff(y(-2)) + b_y_2*diff(x(-2)) + ey ;


  [name='eq:x']
  diff(x) = .05 + b_x_1*diff(y(-2)) + b_x_2*diff(x(-1)) + ex ;

  [name='eq:v']
  diff(v) = diff(x) + d_y*diff(y) ;

  [name='zpac']
  diff(z) = e_c_m*(v(-1)-z(-1)) + c_z_1*diff(z(-1))  + c_z_2*diff(z(-2)) + pac_expectation(pacman) + ez;

end;

shocks;
  var ex = 1.0;
  var ey = 1.0;
  var ez = 1.0;
end;

// Initialize the PAC model (build the Companion VAR representation for the auxiliary model).
pac.initialize('pacman');

// Update the parameters of the PAC expectation model (h0 and h1 vectors).
pac.update.expectation('pacman');

// Set initial conditions to zero. Please use more sensible values if any...
