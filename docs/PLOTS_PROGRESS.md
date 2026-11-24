# Plots: status (branch `plots`)

Design: `src/figures.jl` is the shared figure model (`Figure` / `Panel` / `Curve`), the built-in SVG writer and `save_figures` (graph_format list, `none`, nograph,
`<fname>/graphs`). PNG/PDF/EPS go through ext/JDynPlotsExt.jl (needs `using Plots` and `backend=:plots`; `_render_figure_plots`). Every producer returns `Figure`s carrying the
arrays Dynare draws; `figure_table(figs)` gives a tidy long table (figure, panel, series, kind, x, y, y2) for `write_csv` / ggplot2. Tests (test/test_plots.jl) compare the data of
the figures with Dynare's drawing calls (panel lists, titles, file stems, arrays).

| Area | Producer (file) | Status / validation |
|---|---|---|
| IRFs orders 1-3 | `irf_figures(order, pruning, replic, drop, rng)`, `plot_irfs`, `to_figure` (src/plotting.jl) | done; order>1 equals `irf(k_order...)`; names/titles/nograph/format tested |
| OccBin IRFs / graph | `occbin_irf_figures`, `occbin_graph_figures` (src/plots_occbin.jl) | done; arrays equal `irf_occbin` / `simulate`, names `<f>_irf_occbin_<shock>_<sign>_<k>`, `<f>_occbin_<k>` |
| Shock decomposition | `decomposition_figures`, `plot_shock_decomposition` (src/plots_decomp.jl) | done; bars/stacks/line/legend equal Dynare's recorded fill()/plot() calls (sd_nk_case1_drawn.mat), max diff 4.9e-15, for grp, plain, partial, ss+flip, full, icd |
| Estimation | `prior_figures`, `mode_check_figures`, `prior_posterior_figures`, `trace_figures`, `brooks_gelman_figures`, `bayesian_irf_figures`, `posterior_series_figures`, `smoother_figures` (src/plots_estimation.jl) | done against the layout/naming/grid rules of the MATLAB sources (no recorded Dynare arrays for these) |
| Forecasts, rplot | `forecast_figures`, `conditional_forecast_figures`, `rplot_figures` (src/plots_forecast.jl) | done, same level |
| Statement hooks | `dynare(file; plots=true, plots_backend=:svg)` writes IRF / forecast / rplot / plot_conditional_forecast figures to `<fname>/graphs` (src/plots_hooks.jl, one line in results.jl `_run_in_place!`) | off by default; honours `nograph`, `graph_format` |

## Follow-up (branch `plots-2`)

- **Automatic figures**: `dynare(file; plots=true)` (default false) writes, following Dynare's triggers and options (`nograph`, `graph_format`, `plot_priors` (default on), `mode_check`, `nodiagnostic`,
  `smoother`, `bayesian_irf`, `mh_replic`), via `_fig_est!` / `_plot_statement!` / `_statement_figures!` in src/plots_hooks.jl: priors, mode check, prior/posterior densities and Brooks-Gelman
  diagnostics after MH, Bayesian IRFs (`<fname>/Output`), posterior distributions of smoothed variables/shocks (bayesian smoother), classical smoother figures, `stoch_simul` IRFs, `forecast`,
  `plot_conditional_forecast`, `rplot`, `occbin_graph`, `plot_shock_decomposition` (groups, type, realtime/vintage, init2shocks, ...), `initial_condition_decomposition`.
  `shock_decomposition`, `realtime_shock_decomposition` and `squeeze_shock_decomposition` statements are only run (smoother re-run) when `plots=true`; they fill `ws.state[:sd]`, `[:rt]`.
  The re-run commands take `plots=true` too: `stoch_simul!`, `estimation!`, `calib_smoother!`, `forecast!`, `conditional_forecast!`. Paths are listed in `ws.state[:figure_files]`.
  `graph_format` formats other than SVG use the Plots extension automatically when Plots is loaded (otherwise an ArgumentError names the fix); `plots_backend=:plots` forces it.
  Not drawn: MCMC trace plots (Dynare draws them only on request through `trace_plot`), `irf_occbin` (a function, not a statement), calib_smoother without an estimation object.
- **Plots.jl extension**: all kinds (line/dashed/dotted/band/stacked bar/vline/scatter), grids, legends, titles; png, pdf, eps (GR PostScript), svg. Tested in test/test_plots.jl and
  test/test_plots_auto.jl, which load Plots from a stacked environment: `JDYN_PLOTS_ENV` (default `/tmp/plotsenv`, any environment with Plots installed; the tests skip if absent).
Not done / caveats: realtime/squeeze variants use the same `decomposition_figures` on their `DecompositionPlotData`.
