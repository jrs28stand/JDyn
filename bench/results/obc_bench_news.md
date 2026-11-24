# OBC benchmark run `news`

- date: 2026-11-24 12:44:27; julia 1.12.7; seed 20261124; backends: after (before = baseline source commit 567053915444825dc76c3dc9acf660d418910338; after = d4972af + working tree)
- regime formulation: integer programmes of the fallbacks limited to 0.5 s; methods: all
- load average at start: 1.51 1.84 1.74 2/108 4793; nproc 2; BLAS threads 1; repeats 5

## Model `zlb` (methods: occbin, minnorm, minnorm_pure, news, news_pure, regime)

n_endo = 5, n_exo = 1, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=2% redraws=0 | s=0.5 share=17% redraws=0 | s=0.75 share=26% redraws=0 | s=1 share=30% redraws=0

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | after | 47.1 / 66.9 | 37.8 / 53.4 | 66.8 / 95.4 | 27.7 | 0.0e+00 | true |
| minnorm | after | 7.1 / 7.2 | 5.6 / 5.8 | 8.6 / 9.1 | 7.4 | 2.5e-16 | true |
| minnorm_pure | after | 28.5 / 30.2 | 6.5 / 6.9 | 77.1 / 82.0 | 44.2 | 2.5e-16 | true |
| news | after | 7.7 / 8.3 | 6.1 / 6.6 | 9.3 / 10.0 | 8.7 | 2.5e-16 | true |
| news_pure | after | 1753.2 / 1835.1 | 9.0 / 9.4 | 5744.4 / 6013.1 | 141.2 | 2.5e-16 | true |
| regime | after | 26.2 / 42.0 | 19.2 / 31.2 | 40.1 / 63.7 | 6.5 | 2.5e-16 | true |

Shock scale s = 1.0, horizon 40, 1000 shock periods. Binding in the current period in 30.4% of periods; something binds somewhere on the path in 30.4% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | RegimeOBC set-up (fresh Occbin, N = 2H) | diagnostics on M (S test, P test up to 16 rows) | ready: occbin | ready: minnorm | ready: news (LCP, with diagnostics) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| after | 0.05 / 0.06 | 0.18 / 0.19 | 0.19 / 0.19 | 0.02 / 0.02 | 0.39 / 0.43 | 0.29 / 0.34 | 0.45 / 0.53 | 11.08 / 11.49 | 0.21 / 0.21 | 0.48 / 0.54 | 11.68 / 11.96 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phi, rho); horizon 40.

load average at end of model: 1.40 1.76 1.72 1/106 4807

## Model `zlb2` (methods: occbin, minnorm, minnorm_pure, news, news_pure, regime)

n_endo = 7, n_exo = 1, constraints = 2.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=4% redraws=0 | s=0.5 share=22% redraws=0 | s=0.75 share=28% redraws=0 | s=1 share=34% redraws=0

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | after | 126.9 / 153.4 | 90.8 / 111.1 | 191.5 / 233.9 | 46.4 | 0.0e+00 | true |
| minnorm | after | 12.8 / 14.4 | 9.3 / 10.6 | 17.1 / 18.7 | 12.8 | 3.9e-16 | true |
| minnorm_pure | after | 127.0 / 135.7 | 11.4 / 11.8 | 351.6 / 375.6 | 153.9 | 3.9e-16 | true |
| news | after | 13.8 / 14.8 | 10.7 / 11.5 | 18.1 / 19.3 | 14.9 | 3.9e-16 | true |
| news_pure | after | 31815.7 / 32714.6 | 14.6 / 17.1 | 94098.6 / 96753.1 | 601.9 | 3.9e-16 | true |
| regime | after | 86.5 / 94.6 | 53.6 / 57.5 | 147.0 / 164.3 | 10.9 | 3.9e-16 | true |

Shock scale s = 1.0, horizon 40, 1000 shock periods. Binding in the current period in 33.8% of periods; something binds somewhere on the path in 33.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | RegimeOBC set-up (fresh Occbin, N = 2H) | diagnostics on M (S test, P test up to 16 rows) | ready: occbin | ready: minnorm | ready: news (LCP, with diagnostics) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| after | 0.05 / 0.05 | 0.19 / 0.22 | 0.22 / 0.26 | 0.03 / 0.03 | 0.42 / 0.60 | 0.47 / 0.54 | 0.83 / 0.86 | 12.25 / 13.27 | 0.24 / 0.29 | 0.70 / 0.78 | 12.95 / 14.05 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phi, rho); horizon 40.

load average at end of model: 2.57 2.36 2.02 2/109 4927

## Model `nkm` (methods: occbin, minnorm, minnorm_pure, news, news_pure, regime)

n_endo = 24, n_exo = 5, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=0% redraws=0 | s=0.5 share=0% redraws=0 | s=0.75 share=2% redraws=0 | s=1 share=6% redraws=0 | s=1.5 share=14% redraws=0 | s=2 share=20% redraws=0 | s=3 share=31% redraws=0

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | after | 80.5 / 93.2 | 56.5 / 66.0 | 123.3 / 140.5 | 49.6 | 0.0e+00 | true |
| minnorm | after | 9.7 / 10.9 | 6.5 / 7.7 | 12.5 / 14.4 | 10.8 | 2.0e-05 | false |
| minnorm_pure | after | 41.5 / 45.0 | 7.4 / 8.1 | 103.0 / 111.4 | 61.3 | 2.0e-05 | false |
| news | after | 9.9 / 10.5 | 7.2 / 7.5 | 12.8 / 13.3 | 12.0 | 2.0e-05 | false |
| news_pure | after | 7660.2 / 8157.9 | 9.1 / 10.7 | 21992.5 / 23418.8 | 196.5 | 2.0e-05 | false |
| regime | after | 50.3 / 53.6 | 31.1 / 32.2 | 83.5 / 90.5 | 10.3 | 2.5e-12 | true |

Shock scale s = 3.0, horizon 40, 1000 shock periods. Binding in the current period in 31.4% of periods; something binds somewhere on the path in 34.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | RegimeOBC set-up (fresh Occbin, N = 2H) | diagnostics on M (S test, P test up to 16 rows) | ready: occbin | ready: minnorm | ready: news (LCP, with diagnostics) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| after | 0.28 / 0.33 | 0.41 / 0.46 | 0.43 / 0.47 | 0.04 / 0.05 | 0.41 / 0.46 | 0.52 / 0.56 | 0.52 / 0.60 | 11.75 / 12.24 | 0.47 / 0.51 | 0.95 / 1.02 | 12.77 / 13.19 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phipi, phiy); horizon 40.

load average at end of model: 2.78 2.47 2.09 2/108 4955

## Model `sw` (methods: occbin, regime; multi_equation constraint: minnorm and the LCP formulation refuse it)

n_endo = 32, n_exo = 7, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 fixed (no scale gives 30-60% binding without absurd shocks: see docs/OBC_PERF.md); share=2.6% redraws=1

**Failing draws were replaced: 1 of the 1000 shock periods were redrawn (fresh normal draw, same scale) until the simulation ran through with every method; the sequence is therefore not a pure iid normal sample.**

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | after | 52.1 / 82.9 | 50.0 / 79.6 | 92.4 / 145.6 | 39.4 | 0.0e+00 | true |
| regime | after | 25.4 / 29.6 | 21.6 / 24.7 | 47.3 / 54.1 | 10.9 | 3.8e-13 | true |

Shock scale s = 0.25, horizon 40, 1000 shock periods. Binding in the current period in 2.6% of periods; something binds somewhere on the path in 2.7% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | RegimeOBC set-up (fresh Occbin, N = 2H) | diagnostics on M (S test, P test up to 16 rows) | ready: occbin | ready: minnorm | ready: news (LCP, with diagnostics) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| after | 0.24 / 0.26 | 0.50 / 0.53 | 0.54 / 0.56 | 0.04 / 0.05 | - | - | 0.77 / 0.78 | - | 0.59 / 0.61 | - | - |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: crpi, cry); horizon 40.

load average at end of model: 2.66 2.46 2.09 2/110 4958

