# OBC benchmark run `smoke_sw`

- date: 2026-11-24 08:55:29; julia 1.12.7; seed 20261124; backends: before, after (before = baseline source commit 567053915444825dc76c3dc9acf660d418910338; after = 5670539 + working tree)
- load average at start: 1.94 2.56 1.97 3/111 1510; nproc 2; BLAS threads 1; repeats 2

## Model `sw` (OccBin only: multi_equation constraint, the news methods refuse it)

n_endo = 32, n_exo = 7, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=3% redraws=1 | s=0.5 share=16% redraws=6 | s=0.75 share=20% redraws=24 | s=1 share=21% redraws=50 | s=1.5 share=17% redraws=68 | s=2 share=13% redraws=93 | s=3 share=9% redraws=140 | s=4 share=8% redraws=156 | s=5 share=8% redraws=153 | s=6 share=5% redraws=169 | s=8 share=5% redraws=185 | s=10 share=5% redraws=185 | s=15 share=3% redraws=218 | s=20 share=3% redraws=242 | s=30 share=3% redraws=212 | s=50 share=5% redraws=218 | s=80 share=48% redraws=264 | s=100 share=5% redraws=218

**Failing draws were replaced: 264 of the 1000 shock periods were redrawn (fresh normal draw, same scale) until the simulation ran through with every method; the sequence is therefore not a pure iid normal sample.**

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 822.2 / 832.8 | 85.7 / 93.2 | 1508.0 / 1535.7 | 1150.8 | 0.0e+00 | true |
| occbin | after | 875.9 / 889.1 | 102.0 / 108.7 | 1611.5 / 1631.1 | 1150.8 | 0.0e+00 | true |

Shock scale s = 80.0, horizon 40, 1000 shock periods. Binding in the current period in 48.0% of periods; something binds somewhere on the path in 51.2% (the split above is by that second indicator). Timings: 2 repeats, interleaved, min / median.

load average at end of model: 1.84 1.87 1.88 2/107 2127

