// --+ options: stochastic,json=compute +--

var foo z x y;
varexo e_x e_y e_z;
parameters a b c d e f beta ;

a =  .9;
b = -.2;
c =  .3;
f =  .8;
d =  .5;
e =  .4;

beta = 1/(1+.02);

// Define a VAR model from a subset of equations in the model block.
var_model(structural, model_name = toto, eqtags = [ 'X' 'Y' 'Z' ]);

// Define a VAR_EXPECTATION_MODEL
var_expectation_model(model_name = varexp, expression = diff(log(x)), auxiliary_model_name = toto, horizon = 1, discount = beta)  ;

model;
[ name = 'X' ]
diff(log(x)) = b*diff(z) + a*diff(log(x(-1))) + (1-a)*diff(log(x(-2))) + c*diff(z(-2)) + e_x;
[ name = 'Z' ]
diff(z) = f*(diff(z(-1))-diff(log(x)))+c*diff(z(-2)) + e_z;
[ name = 'Y' ]
log(y) = diff(log(x)) + d*log(y(-2)) + e*diff(z(-1)) + e_y;

foo = var_expectation(varexp);
end;

// Initialize the VAR expectation model, will build the companion matrix of the VAR.
var_expectation.initialize('varexp')

// Update VAR_EXPECTATION reduced form parameters
var_expectation.update('varexp');

// Print expanded VAR_EXPECTATION expression in a file (to be included in substitution.mod).
