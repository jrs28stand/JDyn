# OBC benchmark run `before`

- date: 2026-11-24 09:03:48; julia 1.12.7; seed 20261124; backends: before, after (before = baseline source commit 567053915444825dc76c3dc9acf660d418910338; after = 5670539 + working tree)
- load average at start: 1.70 1.83 1.86 2/107 2175; nproc 2; BLAS threads 1; repeats 5

## Model `zlb` (OccBin, minnorm, minnorm without the fast path)

n_endo = 5, n_exo = 1, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=2% redraws=0 | s=0.5 share=17% redraws=0 | s=0.75 share=26% redraws=0 | s=1 share=30% redraws=0

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 100.1 / 117.1 | 74.7 / 88.7 | 153.6 / 178.8 | 70.4 | 0.0e+00 | true |
| minnorm | before | 50.3 / 56.7 | 42.8 / 48.0 | 64.4 / 74.3 | 62.0 | 2.5e-16 | true |
| minnorm_pure | before | 68.8 / 76.8 | 42.7 / 47.6 | 126.3 / 141.7 | 98.9 | 2.5e-16 | true |
| occbin | after | 95.7 / 108.1 | 72.0 / 84.6 | 147.7 / 162.1 | 70.4 | 0.0e+00 | true |
| minnorm | after | 47.8 / 53.2 | 41.7 / 46.6 | 60.2 / 67.9 | 62.0 | 2.5e-16 | true |
| minnorm_pure | after | 70.6 / 72.7 | 43.2 / 44.4 | 129.6 / 134.8 | 98.9 | 2.5e-16 | true |

Shock scale s = 1.0, horizon 40, 1000 shock periods. Binding in the current period in 30.4% of periods; something binds somewhere on the path in 30.4% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.05 / 0.05 | 0.18 / 0.19 | 2.70 / 3.03 | 0.02 / 0.03 | 25.40 / 27.80 | 25.58 / 26.45 | 2.72 / 3.05 | 28.61 / 29.44 |
| after | 0.05 / 0.05 | 0.18 / 0.22 | 2.80 / 3.86 | 0.03 / 0.03 | 27.63 / 34.76 | 25.95 / 28.86 | 2.83 / 3.88 | 28.75 / 32.38 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phi, rho); horizon 40.

### P3 likelihood-like loop (50 iterations: perturb two parameters, redo P2, simulate 200 periods; H = 40)

| method | backend | seconds per loop min / median | ms per iteration (min) | failed iterations |
|---|---|---|---|---|
| occbin | before | 2.830 / 3.042 | 56.6 | 0 |
| minnorm | before | 2.074 / 2.354 | 41.5 | 0 |
| minnorm_pure | before | 2.333 / 2.509 | 46.7 | 0 |
| occbin | after | 2.946 / 3.119 | 58.9 | 0 |
| minnorm | after | 2.180 / 2.342 | 43.6 | 0 |
| minnorm_pure | after | 2.506 / 2.546 | 50.1 | 0 |

Correctness over the 50 parameter sets: minnorm: max path diff vs OccBin 1.5e-15, 0 iterations with different regime histories; minnorm_pure: max path diff vs OccBin 1.5e-15, 0 iterations with different regime histories; OccBin before vs after identical (paths bit for bit, regime histories): true.

### R1 reliability (2000 states from a long unconstrained simulation, one period solved from each, H = 40)

No method has a time limit; OccBin's own limit is maxit = 30 iterations (counted as its error 311). Different regime sequences are counted among the states both methods solve.

| s | method | solved | failed | failure types | solved by both it and OccBin | different regime sequences vs OccBin | max |path diff| vs OccBin (solved by both) |
|---|---|---|---|---|---|---|---|
| 0.5 | occbin | 2000 | 0 | - | - | - | - |
| 0.5 | minnorm | 2000 | 0 | - | 2000 | 0 | 1.9e-16 |
| 0.5 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 1.9e-16 |
| 1.0 | occbin | 2000 | 0 | - | - | - | - |
| 1.0 | minnorm | 2000 | 0 | - | 2000 | 0 | 2.2e-16 |
| 1.0 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 2.2e-16 |
| 1.5 | occbin | 2000 | 0 | - | - | - | - |
| 1.5 | minnorm | 2000 | 0 | - | 2000 | 0 | 3.3e-16 |
| 1.5 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 3.3e-16 |
| 2.0 | occbin | 2000 | 0 | - | - | - | - |
| 2.0 | minnorm | 2000 | 0 | - | 2000 | 0 | 3.3e-16 |
| 2.0 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 3.3e-16 |

s = 0.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 0.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 2.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 2.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.

load average at end of model: 2.08 1.94 1.90 2/108 2271

## Model `zlb2` (OccBin, minnorm, minnorm without the fast path)

n_endo = 7, n_exo = 1, constraints = 2.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=4% redraws=0 | s=0.5 share=22% redraws=0 | s=0.75 share=28% redraws=0 | s=1 share=34% redraws=0

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 322.2 / 340.7 | 215.9 / 229.2 | 526.3 / 572.0 | 130.4 | 0.0e+00 | true |
| minnorm | before | 61.4 / 72.0 | 47.9 / 56.4 | 84.4 / 98.6 | 82.2 | 5.6e-16 | true |
| minnorm_pure | before | 181.1 / 183.0 | 54.8 / 56.6 | 424.7 / 428.5 | 223.4 | 5.6e-16 | true |
| occbin | after | 295.1 / 324.6 | 199.8 / 220.5 | 477.8 / 524.3 | 130.4 | 0.0e+00 | true |
| minnorm | after | 68.4 / 89.6 | 53.4 / 71.7 | 90.3 / 120.7 | 82.2 | 5.6e-16 | true |
| minnorm_pure | after | 184.0 / 201.1 | 53.0 / 64.7 | 436.6 / 466.4 | 223.4 | 5.6e-16 | true |

Shock scale s = 1.0, horizon 40, 1000 shock periods. Binding in the current period in 33.8% of periods; something binds somewhere on the path in 33.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.06 / 0.06 | 0.19 / 0.24 | 4.68 / 6.51 | 0.03 / 0.05 | 44.63 / 51.61 | 42.56 / 58.08 | 4.71 / 6.61 | 47.24 / 64.60 |
| after | 0.06 / 0.06 | 0.21 / 0.24 | 4.79 / 5.30 | 0.03 / 0.05 | 43.34 / 58.48 | 44.52 / 57.29 | 4.85 / 5.34 | 50.90 / 62.09 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phi, rho); horizon 40.

### P3 likelihood-like loop (50 iterations: perturb two parameters, redo P2, simulate 200 periods; H = 40)

| method | backend | seconds per loop min / median | ms per iteration (min) | failed iterations |
|---|---|---|---|---|
| occbin | before | 6.117 / 6.990 | 122.3 | 0 |
| minnorm | before | 3.665 / 3.888 | 73.3 | 0 |
| minnorm_pure | before | 4.847 / 5.445 | 96.9 | 0 |
| occbin | after | 5.843 / 6.661 | 116.9 | 0 |
| minnorm | after | 3.734 / 3.923 | 74.7 | 0 |
| minnorm_pure | after | 4.939 / 5.439 | 98.8 | 0 |

Correctness over the 50 parameter sets: minnorm: max path diff vs OccBin 2.3e-15, 0 iterations with different regime histories; minnorm_pure: max path diff vs OccBin 2.3e-15, 0 iterations with different regime histories; OccBin before vs after identical (paths bit for bit, regime histories): true.

### R1 reliability (2000 states from a long unconstrained simulation, one period solved from each, H = 40)

No method has a time limit; OccBin's own limit is maxit = 30 iterations (counted as its error 311). Different regime sequences are counted among the states both methods solve.

| s | method | solved | failed | failure types | solved by both it and OccBin | different regime sequences vs OccBin | max |path diff| vs OccBin (solved by both) |
|---|---|---|---|---|---|---|---|
| 0.5 | occbin | 2000 | 0 | - | - | - | - |
| 0.5 | minnorm | 2000 | 0 | - | 2000 | 0 | 2.2e-16 |
| 0.5 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 2.2e-16 |
| 1.0 | occbin | 2000 | 0 | - | - | - | - |
| 1.0 | minnorm | 2000 | 0 | - | 2000 | 0 | 4.4e-16 |
| 1.0 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 4.4e-16 |
| 1.5 | occbin | 2000 | 0 | - | - | - | - |
| 1.5 | minnorm | 2000 | 0 | - | 2000 | 0 | 8.9e-16 |
| 1.5 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 8.9e-16 |
| 2.0 | occbin | 2000 | 0 | - | - | - | - |
| 2.0 | minnorm | 2000 | 0 | - | 2000 | 0 | 8.9e-16 |
| 2.0 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 8.9e-16 |

s = 0.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 0.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 2.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 2.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.

load average at end of model: 1.97 1.87 1.88 2/106 2500

## Model `nkm` (OccBin, minnorm, minnorm without the fast path)

n_endo = 24, n_exo = 5, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=0% redraws=0 | s=0.5 share=0% redraws=0 | s=0.75 share=2% redraws=0 | s=1 share=6% redraws=0 | s=1.5 share=14% redraws=0 | s=2 share=20% redraws=0 | s=3 share=31% redraws=0

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 223.0 / 257.4 | 83.5 / 97.6 | 479.1 / 547.6 | 272.4 | 0.0e+00 | true |
| minnorm | before | 119.8 / 150.3 | 60.8 / 81.7 | 224.9 / 273.0 | 206.1 | 2.0e-05 | false |
| minnorm_pure | before | 162.3 / 209.2 | 67.8 / 88.2 | 334.4 / 429.4 | 256.6 | 2.0e-05 | false |
| occbin | after | 224.4 / 273.2 | 84.3 / 104.4 | 482.0 / 584.0 | 272.4 | 0.0e+00 | true |
| minnorm | after | 122.6 / 138.1 | 62.6 / 72.5 | 229.4 / 254.8 | 206.1 | 2.0e-05 | false |
| minnorm_pure | after | 151.9 / 198.7 | 62.1 / 74.8 | 315.2 / 424.8 | 256.6 | 2.0e-05 | false |

Shock scale s = 3.0, horizon 40, 1000 shock periods. Binding in the current period in 31.4% of periods; something binds somewhere on the path in 34.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.28 / 0.31 | 0.39 / 0.44 | 3.11 / 3.79 | 0.05 / 0.06 | 31.83 / 38.54 | 29.55 / 37.25 | 3.16 / 3.86 | 32.78 / 41.92 |
| after | 0.29 / 0.34 | 0.39 / 0.41 | 3.07 / 3.34 | 0.05 / 0.08 | 30.74 / 35.94 | 30.58 / 31.43 | 3.12 / 3.43 | 33.92 / 34.50 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phipi, phiy); horizon 40.

### P3 likelihood-like loop (50 iterations: perturb two parameters, redo P2, simulate 200 periods; H = 40)

| method | backend | seconds per loop min / median | ms per iteration (min) | failed iterations |
|---|---|---|---|---|
| occbin | before | 5.600 / 5.772 | 112.0 | 0 |
| minnorm | before | 3.853 / 4.254 | 77.1 | 0 |
| minnorm_pure | before | 4.339 / 4.493 | 86.8 | 0 |
| occbin | after | 5.707 / 6.066 | 114.1 | 0 |
| minnorm | after | 3.722 / 4.257 | 74.4 | 0 |
| minnorm_pure | after | 4.464 / 4.570 | 89.3 | 0 |

Correctness over the 50 parameter sets: minnorm: max path diff vs OccBin 2.9e-05, 42 iterations with different regime histories; minnorm_pure: max path diff vs OccBin 2.9e-05, 42 iterations with different regime histories; OccBin before vs after identical (paths bit for bit, regime histories): true.

### P4 scaling with the horizon (s = 3.0)

#### H = 20

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 196.4 / 210.5 | 55.5 / 59.1 | 452.3 / 485.9 | 230.3 | 0.0e+00 | true |
| minnorm | before | 99.8 / 117.6 | 35.9 / 43.4 | 208.0 / 243.7 | 148.2 | 2.0e-05 | false |
| minnorm_pure | before | 108.1 / 114.3 | 35.0 / 35.8 | 234.9 / 250.6 | 167.3 | 2.0e-05 | false |
| occbin | after | 184.6 / 243.5 | 51.2 / 72.9 | 425.2 / 548.2 | 230.3 | 0.0e+00 | true |
| minnorm | after | 95.7 / 106.9 | 34.7 / 37.6 | 198.6 / 225.7 | 148.2 | 2.0e-05 | false |
| minnorm_pure | after | 107.1 / 126.9 | 34.4 / 39.8 | 233.9 / 279.0 | 167.3 | 2.0e-05 | false |

Shock scale s = 3.0, horizon 20, 1000 shock periods. Binding in the current period in 31.4% of periods; something binds somewhere on the path in 34.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.29 / 0.31 | 0.43 / 0.46 | 3.09 / 3.35 | 0.05 / 0.06 | 30.87 / 33.17 | 29.37 / 29.83 | 3.14 / 3.42 | 32.50 / 33.30 |
| after | 0.29 / 0.32 | 0.43 / 0.47 | 3.15 / 4.05 | 0.05 / 0.07 | 32.62 / 33.96 | 31.17 / 33.48 | 3.22 / 4.12 | 34.32 / 37.71 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phipi, phiy); horizon 20.

#### H = 40

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 232.7 / 272.2 | 87.2 / 108.8 | 495.5 / 569.2 | 272.4 | 0.0e+00 | true |
| minnorm | before | 149.1 / 174.6 | 76.4 / 89.4 | 272.3 / 322.6 | 206.1 | 2.0e-05 | false |
| minnorm_pure | before | 158.7 / 191.4 | 63.7 / 74.7 | 326.6 / 400.5 | 256.6 | 2.0e-05 | false |
| occbin | after | 238.1 / 243.3 | 88.7 / 90.1 | 506.8 / 521.8 | 272.4 | 0.0e+00 | true |
| minnorm | after | 124.2 / 134.6 | 61.5 / 68.7 | 231.8 / 248.4 | 206.1 | 2.0e-05 | false |
| minnorm_pure | after | 160.1 / 175.6 | 64.2 / 70.3 | 330.0 / 361.8 | 256.6 | 2.0e-05 | false |

Shock scale s = 3.0, horizon 40, 1000 shock periods. Binding in the current period in 31.4% of periods; something binds somewhere on the path in 34.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.32 / 0.41 | 0.40 / 0.42 | 3.17 / 3.84 | 0.06 / 0.06 | 31.51 / 34.34 | 30.52 / 33.79 | 3.23 / 3.90 | 33.69 / 39.36 |
| after | 0.31 / 0.37 | 0.37 / 0.45 | 3.16 / 3.55 | 0.06 / 0.08 | 30.27 / 34.47 | 30.14 / 35.17 | 3.22 / 3.60 | 33.30 / 38.66 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phipi, phiy); horizon 40.

#### H = 80

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 336.9 / 433.7 | 157.8 / 208.4 | 662.2 / 844.7 | 356.5 | 0.0e+00 | true |
| minnorm | before | 183.5 / 232.0 | 119.3 / 132.6 | 293.2 / 343.5 | 321.8 | 2.0e-05 | false |
| minnorm_pure | before | 308.4 / 366.2 | 133.2 / 160.2 | 626.0 / 680.3 | 465.7 | 2.0e-05 | false |
| occbin | after | 343.0 / 377.2 | 162.2 / 172.5 | 671.2 / 730.6 | 356.5 | 0.0e+00 | true |
| minnorm | after | 214.2 / 269.2 | 136.2 / 177.6 | 349.5 / 429.7 | 321.8 | 2.0e-05 | false |
| minnorm_pure | after | 340.9 / 385.8 | 149.3 / 158.5 | 685.0 / 825.1 | 465.7 | 2.0e-05 | false |

Shock scale s = 3.0, horizon 80, 1000 shock periods. Binding in the current period in 31.4% of periods; something binds somewhere on the path in 34.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.35 / 0.40 | 0.41 / 0.45 | 3.25 / 4.34 | 0.06 / 0.08 | 31.73 / 36.73 | 36.51 / 46.51 | 3.32 / 4.42 | 39.76 / 50.85 |
| after | 0.36 / 0.40 | 0.42 / 0.43 | 3.11 / 3.36 | 0.06 / 0.08 | 32.09 / 34.93 | 39.65 / 42.75 | 3.18 / 3.42 | 42.93 / 47.02 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phipi, phiy); horizon 80.

### R1 reliability (2000 states from a long unconstrained simulation, one period solved from each, H = 40)

No method has a time limit; OccBin's own limit is maxit = 30 iterations (counted as its error 311). Different regime sequences are counted among the states both methods solve.

| s | method | solved | failed | failure types | solved by both it and OccBin | different regime sequences vs OccBin | max |path diff| vs OccBin (solved by both) |
|---|---|---|---|---|---|---|---|
| 0.5 | occbin | 2000 | 0 | - | - | - | - |
| 0.5 | minnorm | 2000 | 0 | - | 2000 | 0 | 6.0e-15 |
| 0.5 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 6.0e-15 |
| 1.0 | occbin | 2000 | 0 | - | - | - | - |
| 1.0 | minnorm | 2000 | 0 | - | 2000 | 0 | 5.1e-15 |
| 1.0 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 5.1e-15 |
| 1.5 | occbin | 2000 | 0 | - | - | - | - |
| 1.5 | minnorm | 2000 | 0 | - | 2000 | 1 | 2.1e-05 |
| 1.5 | minnorm_pure | 2000 | 0 | - | 2000 | 1 | 2.1e-05 |
| 2.0 | occbin | 2000 | 0 | - | - | - | - |
| 2.0 | minnorm | 2000 | 0 | - | 2000 | 1 | 1.1e-05 |
| 2.0 | minnorm_pure | 2000 | 0 | - | 2000 | 1 | 1.1e-05 |

s = 0.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 0.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 2.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 2.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.

load average at end of model: 2.28 2.02 1.95 2/113 2819

## Model `sw` (OccBin only: multi_equation constraint, the news methods refuse it)

n_endo = 32, n_exo = 7, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=1.0 fixed (no scale gives 30-60% binding without absurd shocks: see docs/OBC_PERF.md); share=20.8% redraws=50

**Failing draws were replaced: 50 of the 1000 shock periods were redrawn (fresh normal draw, same scale) until the simulation ran through with every method; the sequence is therefore not a pure iid normal sample.**

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 189.5 / 210.4 | 76.6 / 83.6 | 579.0 / 649.8 | 252.8 | 0.0e+00 | true |
| occbin | after | 186.4 / 216.0 | 74.2 / 89.8 | 575.1 / 652.2 | 252.8 | 0.0e+00 | true |

Shock scale s = 1.0, horizon 40, 1000 shock periods. Binding in the current period in 20.8% of periods; something binds somewhere on the path in 22.0% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

