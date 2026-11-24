# Plots: status (branch `plots`)

Design: `src/figures.jl` is the shared figure model (`Figure` / `Panel` / `Curve`), SVG writer (built in) and `save_figures` (graph_format list, `none`, nograph, `<fname>/graphs`);
PNG/PDF/EPS through ext/JDynPlotsExt.jl (needs `using Plots`, `backend=:plots`). Every producer returns `Figure`s carrying the arrays Dynare draws; `figure_table(figs)` gives a tidy
long table for `write_csv` (ggplot2). Tests compare the data of the figures with Dynare's drawing calls (panel lists, titles, file stems, arrays).

| Area | Producer | Status |
|---|---|---|
| IRFs orders 1-3 | `irf_figures` (`order`, `pruning`, `replic`, `drop`, `rng`), `plot_irfs`, `to_figure` | in progress |
| OccBin IRFs / graph | `occbin_irf_figures`, `occbin_graph_figures` | in progress |
| Shock decomposition | `decomposition_figures`, `plot_shock_decomposition` | in progress |
| Estimation | `prior_figures`, `mode_check_figures`, `prior_posterior_figures`, `trace_figures`, `brooks_gelman_figures`, `bayesian_irf_figures`, `posterior_series_figures`, `smoother_figures` | in progress |
| Forecasts, rplot | | not started |
