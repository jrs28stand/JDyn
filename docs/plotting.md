# Plotting in JDyn

Scope: the `stoch_simul` impulse-response figures at first order. Matching Dynare's look is not a goal; the
*content* is: which shocks, which variables, Dynare's selection rules, horizon, figure split and names.
Final graphics are expected to be made elsewhere (ggplot2), so the plotted data can be exported as tidy CSV.

## What exists (`src/plotting.jl`, `ext/JDynPlotsExt.jl`)

| function | purpose |
|---|---|
| `irf_figures(m, sol=first_order(m); var_list, shocks, periods, threshold, relative)` | the content: an `IRFResult` (figures, series, layout, skipped responses). No rendering. |
| `plot_irfs(m, sol; outdir, format="svg", backend=:svg, nograph)` | `irf_figures` plus files; honours `stoch_simul(nograph)`; fills `res.files` |
| `irf_table(res; all=false)`, `write_csv(path, rows)` | long format `shock,variable,period,value,order`; `all=true` includes responses below the threshold |
| `pltorg(n)` | Dynare's layout rule |

Dynare rules reproduced (from `stoch_simul.m`, `pltorg.m`): shocks in `M_.exo_names` order or `irf_shocks`; impulse =
column of the lower Cholesky factor of `Sigma_e` (one standard deviation, correlation-aware, `get_lower_cholesky_covariance`);
a shock whose Cholesky diagonal is <= 5e-7 gets no figure; a variable is plotted only if `max|irf| >= irf_plot_threshold`
(default 1e-10; the preprocessor stores it as `impulse_responses.plot_threshold`); horizon `irf` (default 40), x from 1
(impact period); variable list from `stoch_simul(...)` or all original endogenous variables; at most 9 panels per figure
(3x3; 1x1, 2x1, 3x1, 2x2, 3x2, 3x2, 3x3 ... for fewer), the last figure of a shock uses `pltorg`'s `lr x lc`;
titles "Orthogonalized shock to e" (+ " figure k"); files `<fname>_IRF_<shock>[k].<ext>` in `<fname>/graphs`;
`relative_irf` gives percent of one standard deviation. Note: `example1.mod` as shipped has perfectly correlated
shocks, so Dynare's Cholesky factor gives `u` a diagonal of about 1e-7 and, as in Dynare, draws no figure for `u`.

Only SVG is written without dependencies. PNG/PDF need Plots.jl: `using Plots` activates the extension
(`backend=:plots, format="png"|"pdf"|"svg"`). `using JDyn` itself loads no plotting library (`Plots` is a weak dependency).

```r
irf <- read.csv("example1/graphs/example1_IRF.csv")   # after write_csv(that_path, irf_table(res))
ggplot(irf, aes(period, value)) + geom_line() + facet_grid(variable ~ shock, scales = "free_y")
```

## Options compared (measured here: 2 shared cores, other jobs running, Julia 1.12.7, artifacts of Plots/GR already in the depot)

| | install + precompile | `using` | first plot (build + save png, pdf, svg of 9 panels) | PNG / PDF / SVG headless | extra deps |
|---|---|---|---|---|---|
| built-in SVG writer | none | 0.7 s (`using JDyn`) | 3.3 s first call (compiling), then ms | SVG only (convert with `rsvg-convert`) | 0 |
| Plots.jl + GR | 3 min 13 s (Plots precompile 156 s) | 1.8 s | 0.4 s + 0.9 s; whole script 3.5 s | all three; GKSwstype=100 for headless | ~160 packages |
| CairoMakie | 15 min 23 s (Makie 306 s, CairoMakie 80 s, 113 packages precompiled) | 8.4 s | 3.5 s + 4.9 s; whole script 18.7 s | all three | ~300 packages |

Through JDyn's extension the first PNG pair costs about 5 s of `using Plots` (includes the one-time extension
precompile) plus 4 s for the first render. Both libraries rendered readable output (grid of panels, titles, zero line).

## Recommendation

Keep the core free of plotting libraries; ship the SVG writer and the CSV export in core (zero dependencies, fast, enough
for a quick look and for ggplot2), and Plots.jl+GR as the optional extension for PNG/PDF. CairoMakie offers nicer typography but costs
5x the install time, 4x the first-plot time and many more dependencies; not worth it here. The figure content is a plain
struct, so another backend can be added by one method.

## Left to do

- Higher orders (`order >= 2` GIRFs, pruning): responses are first order only; `irf_figures` ignores `order`.
- `irf_shocks` blocks and `irf_opt` options other than the `stoch_simul` option; `irf_plot_threshold` set through `options_` lines.
- TeX output (`tex` option, `.tex` loader), `graph_format` mapping (`eps`, `fig`), `nodisplay`.
- OccBin graphs (`occbin.graph`, `plot_irfs` for OccBin: piecewise vs linear, with a `regime` column in the table), `rplot`,
  smoother/forecast/shock-decomposition/prior-posterior/mode-check figures; pydyn has the OccBin graph layout.
- Pixel comparison with Dynare is impossible here (no Octave graphics toolkit); only data and layout are tested.
