# stoch_simul probes (order 1, 2; irfs; simulated periods)
let S(o, order=1, irf=0, extra="") = "stoch_simul(order=$order,irf=$irf,nograph$(isempty(o) ? "" : "," * o))$extra;"
    probe([
     (S("", 1), [S(x) for x in ["hp_filter=1600", "hp_filter=1600,filtered_theoretical_moments_grid=128", "bandpass_filter", "bandpass_filter=[6,32]", "ar=2",
                  "nocorr", "nodecomposition", "nomoments", "conditional_variance_decomposition=[1,5]", "contemporaneous_correlation",
                  "one_sided_hp_filter=1600", "qz_criterium=1.5", "solve_algo=1", "loglinear", "partial_information", "aim_solver",
                  "dr=cycle_reduction", "dr=logarithmic_reduction", "dr_cycle_reduction_tol=1e-9", "SpectralDensity=1",
                  "qz_zero_threshold=1e-5", "noprint", "nofunctions", "nomodelsummary", "dr_display_tol=1e-3"]]),
     (S("", 2), [S(x, 2) for x in ["hp_filter=1600", "nocorr", "pruning", "pruning,k_order_solver", "k_order_solver", "ar=2", "nodecomposition", "SpectralDensity=1"]]),
     (S("", 1, 10), [S(x, 1, 10) for x in ["irf_shocks=(e)", "relative_irf", "irf=5", "replic=3", "loglinear", "drop=10", "irf_in_percent"]]),
     (S("", 2, 10), [S(x, 2, 10) for x in ["irf_shocks=(e)", "relative_irf", "replic=3", "drop=10", "pruning,replic=3"]]),
     ("stoch_simul(order=1,periods=200,irf=0,nograph);", ["stoch_simul(order=1,periods=200,irf=0,nograph,$x);" for x in ["drop=10", "hp_filter=1600", "ar=2", "bandpass_filter", "nocorr", "nodecomposition", "contemporaneous_correlation", "simul_replic=3"]]),
    ])
end
