# Optimal policy (Ramsey / evaluate_planner_objective / discretionary_policy / osr): progress

Status (WIP): src/optimal_policy.jl written (port of pydyn/optimal_policy.py) but NOT yet included in src/JDyn.jl or compiled.
References and generators copied to test/ref_optimal_policy/.

Next steps:
1. include("optimal_policy.jl") in JDyn.jl (after perfect_foresight.jl, before results.jl); export API.
2. steady.jl: route Ramsey models in `_steady_state` to `ramsey_steady_state`; `fill_auxiliary` skip type-6 aux and use `_aux_rows`;
   `_dynare_solve` gets `jacflag` keyword (no-Jacobian randomisation check).
3. model.jl: lift the ramsey refusal (set options_.instruments in matlab_options).
4. results.jl: ramsey_model/ramsey_policy/discretionary_policy/evaluate_planner_objective/osr in `_run_in_place!`, `ramsey_policy!`, `osr!`, ...
5. Tests test/test_optimal_policy.jl against test/ref_optimal_policy (add to runtests ALL_TESTS), Octave check of the Wyu ordering defect, docs.
