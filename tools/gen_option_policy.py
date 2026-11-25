import sys, json
sys.path.insert(0,'/home/claude/work/pydyn-rl'); sys.path.insert(0,'/home/claude/work/pydyn-rl/tools')
import options_audit as A
from pydyn import option_policy as P
gt = A.statements('/root/ref/dynare-7.2')
# statements whose JSON name differs from the grammar rule name (the driver's names)
for a, b in (('estimation_data','data'),('ms_sbvar_estimation','ms_estimation'),('ms_sbvar_compute_mdd','ms_compute_mdd'),('ms_sbvar_compute_probabilities','ms_compute_probabilities'),
             ('ms_sbvar_forecast','ms_forecast'),('ms_sbvar_irf','ms_irf'),('ms_sbvar_simulation','ms_simulation'),('ms_sbvar_variance_decomposition','ms_variance_decomposition'),
             ('dynare_sensitivity','sensitivity')):
    gt[a] = gt.pop(b)
gt['estimation_data'] = ['file', 'first_obs', 'last_obs', 'nobs', 'series']      # the `data` command's options as the JSON carries them (not o_ rules)
# JDyn runs these statements and acts on options (baseline: pydyn's classification, adjusted below)
IMPL = (set(P.SPECS) - {'identification'}) | {'estimation_data'}
UNIMPL_EXTRA = set()
specs = {st: {k: set(v) for k, v in P.SPECS[st].items()} for st in IMPL if st in P.SPECS}
specs['estimation_data'] = dict(honoured={'file','first_obs','last_obs','nobs','series'}, harmless=set(), refused=set())
# --- JDyn deltas
specs['initial_condition_decomposition']['refused'].discard('initial_condition_decomp.write_xls'); specs['initial_condition_decomposition']['harmless'].add('initial_condition_decomp.write_xls')
def move(st, key, to):
    for k in ('honoured','harmless','refused'): specs[st][k].discard(key)
    specs[st][to].add(key)
for _st in ('osr','ramsey_policy','discretionary_policy'):
    for _k in ('simul_replic','SpectralDensity.trigger'):
        if _k in gt[_st]: move(_st,_k,'refused')
SIM = ['stoch_simul','osr','ramsey_policy','discretionary_policy']
for st in specs:
    if 'partial_information' in gt[st]: move(st,'partial_information','refused')
move('stoch_simul','nocorr','honoured') if 'nocorr' in gt['stoch_simul'] else None
# estimation: JDyn's own sets
H = set("""datafile first_obs nobs loglinear logdata prefilter presample lik_init diffuse_filter kalman_algo kalman_tol diffuse_kalman_tol riccati_tol prior_trunc mode_compute mh_replic mh_nblck mh_jscale mh_drop mh_init_scale_factor order use_univariate_filters_if_singularity_is_detected rescale_prediction_error_covariance cova_compute mh_conf_sig optim_opt mode_check.status nodiagnostic smoother filtered_vars filter_step_ahead filter_covariance smoothed_state_uncertainty filter_decomposition forecast forecasts.conf_sig sub_draws nodecomposition ar conditional_variance_decomposition contemporaneous_correlation moments_varendo bayesian_irf irf irf_shocks relative_irf posterior_max_subsample_draws heteroskedastic_filter mode_file load_mh_file mh_recover load_results_after_load_mh dsge_var dsge_varlag endogenous_prior mh_posterior_mode_estimation selected_variables_only analytic_derivation endo_vars_for_moment_computations_in_estimation mh_tune_jscale.status mh_tune_jscale.target mh_tune_jscale.guess MCMC_jumping_covariance posterior_sampler_options.posterior_sampling_method posterior_sampler_options.sampling_opt frequentist_smoother estimation.moments_posterior_density.indicator pruning""".split())
est = specs['estimation']
allk = set(gt['estimation'])
cos = set(x for x in est['harmless'])
for k in sorted(allk):
    if k.startswith('particle.'): to='honoured'      # order 2 only: enforced by context in option_policy.jl
    elif k in H: to='honoured'
    elif k in P.HARMLESS_REASONS or k in ('mode_check.neighbourhood_size','mode_check.symmetric_plots','mode_check.number_of_points','fast_kalman_filter','xls_sheet','xls_range','lyapunov_db','lyapunov_srs','dr_cycle_reduction','dr_logarithmic_reduction') or k.startswith('irf_opt.'): to='harmless'
    elif k in est['harmless']: to='harmless'
    else: to='refused'
    move('estimation',k,to)
def fmt(s): return "Set(["+", ".join('"%s"'%k for k in sorted(s))+"])"
out=['# Generated from Dynare 7.2 preprocessor grammar (DynareBison.yy) by tools/gen_option_policy.py; classification reviewed by hand.',
     '# Do not edit the sets by hand without updating the generator inputs.','']
out.append('const OPTION_SPECS = Dict{String,NamedTuple{(:honoured, :harmless, :refused),NTuple{3,Set{String}}}}(')
for st in sorted(specs):
    s=specs[st]
    # every grammar option in exactly one set
    allg=set(gt[st]); cl=s['honoured']|s['harmless']|s['refused']
    # grammar options missing from pydyn's table -> refused
    for k in allg-cl: s['refused'].add(k)
    s['honoured']&=allg; s['harmless']&=allg; s['refused']&=allg
    out.append('    "%s" => (honoured=%s,\n        harmless=%s,\n        refused=%s),'%(st,fmt(s['honoured']),fmt(s['harmless']),fmt(s['refused'])))
out.append(')')
out.append('const HARMLESS_REASONS = Dict{String,String}(')
for k,v in sorted(P.HARMLESS_REASONS.items()): out.append('    "%s" => "%s",'%(k,v.replace('"','\\"')))
out.append(')')
out.append('const HARMLESS_PREFIXES = Dict{String,String}(' + ', '.join('"%s" => "%s"'%(k,v.replace('"','\\"')) for k,v in P.HARMLESS_PREFIXES.items()) + ')')
unimpl=sorted(set(gt)-set(specs))
out.append('')
out.append('"Statements of the grammar JDyn does not execute (it says so when one runs; their options are not classified one by one)."')
out.append('const UNIMPLEMENTED_STATEMENTS = Set(%s)'%('["'+'", "'.join(unimpl)+'"]'))
out.append('const GRAMMAR_OPTIONS = Dict{String,Vector{String}}(')
for st in sorted(gt): out.append('    "%s" => %s,'%(st,'["'+'", "'.join(gt[st])+'"]' if gt[st] else 'String[]'))
out.append(')')
open('/home/claude/work/jdyn-parity4/src/option_tables.jl','w').write("\n".join(out)+"\n")
print(len(specs),unimpl)
for st in ('estimation','stoch_simul'):
    print(st,{k:len(v) for k,v in specs[st].items()})
print(sorted(specs['estimation']['refused']))
json.dump({st:{k:sorted(v) for k,v in s.items()} for st,s in specs.items()}, open('/tmp/p4/specs.json','w'))
