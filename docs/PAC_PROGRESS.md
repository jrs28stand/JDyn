# PAC / VAR / trend-component progress log

Status (WIP): src/pac.jl written (driver.m reader via a second preprocessor run in MATLAB mode, get_companion_matrix,
var_expectation.initialize/update, pac.initialize/update.expectation/mce.parameters/bgp.*). Hooks: PKG_FUNCS (builtins.jl, interp.jl),
M_/oo_ fields (session.jl), build_M (results.jl), run_preprocessor/Model (model.jl).

Next: make it compile, run the three acceptance folders, validate against Octave, 3-D array indexing (AR(:,:,i)), test file, docs.
