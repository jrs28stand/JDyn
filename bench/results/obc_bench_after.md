# OBC benchmark run `after`

- date: 2026-11-24 11:40:53; julia 1.12.7; seed 20261124; backends: before, after (before = baseline source commit 567053915444825dc76c3dc9acf660d418910338; after = f40251e + working tree)
- load average at start: 1.27 1.53 1.30 1/104 1964; nproc 2; BLAS threads 1; repeats 5

## Model `zlb` (OccBin, minnorm, minnorm without the fast path)

n_endo = 5, n_exo = 1, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=2% redraws=0 | s=0.5 share=17% redraws=0 | s=0.75 share=26% redraws=0 | s=1 share=30% redraws=0

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 89.3 / 97.1 | 67.0 / 71.6 | 138.5 / 153.8 | 70.4 | 0.0e+00 | true |
| minnorm | before | 47.6 / 51.6 | 41.3 / 43.1 | 60.0 / 63.6 | 62.0 | 2.5e-16 | true |
| minnorm_pure | before | 67.4 / 71.5 | 41.7 / 44.2 | 124.3 / 131.7 | 98.9 | 2.5e-16 | true |
| occbin | after | 39.2 / 41.1 | 31.2 / 33.4 | 55.4 / 57.2 | 27.7 | 0.0e+00 | true |
| minnorm | after | 6.5 / 6.7 | 5.2 / 5.4 | 8.0 / 8.2 | 7.4 | 2.5e-16 | true |
| minnorm_pure | after | 26.3 / 27.6 | 5.8 / 6.3 | 71.7 / 74.6 | 44.2 | 2.5e-16 | true |

Shock scale s = 1.0, horizon 40, 1000 shock periods. Binding in the current period in 30.4% of periods; something binds somewhere on the path in 30.4% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.05 / 0.05 | 0.17 / 0.18 | 2.79 / 3.53 | 0.02 / 0.03 | 24.94 / 27.87 | 25.07 / 25.58 | 2.82 / 3.56 | 27.90 / 29.81 |
| after | 0.05 / 0.06 | 0.18 / 0.19 | 0.17 / 0.19 | 0.02 / 0.02 | 0.28 / 0.32 | 0.27 / 0.30 | 0.19 / 0.21 | 0.48 / 0.50 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phi, rho); horizon 40.

### P3 likelihood-like loop (50 iterations: perturb two parameters, redo P2, simulate 200 periods; H = 40)

| method | backend | seconds per loop min / median | ms per iteration (min) | failed iterations |
|---|---|---|---|---|
| occbin | before | 2.776 / 2.842 | 55.5 | 0 |
| minnorm | before | 2.028 / 2.172 | 40.6 | 0 |
| minnorm_pure | before | 2.364 / 2.471 | 47.3 | 0 |
| occbin | after | 0.430 / 0.467 | 8.6 | 0 |
| minnorm | after | 0.085 / 0.093 | 1.7 | 0 |
| minnorm_pure | after | 0.287 / 0.337 | 5.7 | 0 |

Correctness over the 50 parameter sets: minnorm: max path diff vs OccBin 1.4e-15, 0 iterations with different regime histories; minnorm_pure: max path diff vs OccBin 1.4e-15, 0 iterations with different regime histories; OccBin before vs after identical (paths bit for bit, regime histories): true.

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
| 2.0 | minnorm | 2000 | 0 | - | 2000 | 0 | 4.4e-16 |
| 2.0 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 4.4e-16 |

s = 0.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 0.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 2.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 2.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.

load average at end of model: 1.60 1.54 1.32 2/108 2070

## Model `zlb2` (OccBin, minnorm, minnorm without the fast path)

n_endo = 7, n_exo = 1, constraints = 2.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=4% redraws=0 | s=0.5 share=22% redraws=0 | s=0.75 share=28% redraws=0 | s=1 share=34% redraws=0

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 234.2 / 238.3 | 155.9 / 158.7 | 381.4 / 391.9 | 130.4 | 0.0e+00 | true |
| minnorm | before | 59.2 / 67.6 | 47.6 / 55.6 | 78.0 / 87.6 | 82.2 | 5.6e-16 | true |
| minnorm_pure | before | 174.7 / 184.6 | 52.1 / 58.9 | 408.9 / 427.3 | 223.4 | 5.6e-16 | true |
| occbin | after | 111.6 / 115.6 | 80.3 / 83.5 | 169.9 / 178.0 | 46.4 | 0.0e+00 | true |
| minnorm | after | 13.0 / 13.3 | 9.3 / 9.5 | 17.1 / 17.3 | 12.8 | 3.9e-16 | true |
| minnorm_pure | after | 123.8 / 124.9 | 11.3 / 11.6 | 342.5 / 345.1 | 153.9 | 3.9e-16 | true |

Shock scale s = 1.0, horizon 40, 1000 shock periods. Binding in the current period in 33.8% of periods; something binds somewhere on the path in 33.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.05 / 0.05 | 0.19 / 0.21 | 4.58 / 4.71 | 0.03 / 0.04 | 40.61 / 42.81 | 40.51 / 42.32 | 4.62 / 4.75 | 45.13 / 47.07 |
| after | 0.05 / 0.05 | 0.19 / 0.21 | 0.21 / 0.30 | 0.03 / 0.03 | 0.38 / 0.50 | 0.49 / 0.54 | 0.24 / 0.33 | 0.71 / 0.81 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phi, rho); horizon 40.

### P3 likelihood-like loop (50 iterations: perturb two parameters, redo P2, simulate 200 periods; H = 40)

| method | backend | seconds per loop min / median | ms per iteration (min) | failed iterations |
|---|---|---|---|---|
| occbin | before | 5.977 / 6.318 | 119.5 | 0 |
| minnorm | before | 3.326 / 3.525 | 66.5 | 0 |
| minnorm_pure | before | 4.661 / 4.736 | 93.2 | 0 |
| occbin | after | 1.090 / 1.288 | 21.8 | 0 |
| minnorm | after | 0.176 / 0.204 | 3.5 | 0 |
| minnorm_pure | after | 1.310 / 1.386 | 26.2 | 0 |

Correctness over the 50 parameter sets: minnorm: max path diff vs OccBin 2.2e-15, 0 iterations with different regime histories; minnorm_pure: max path diff vs OccBin 2.2e-15, 0 iterations with different regime histories; OccBin before vs after identical (paths bit for bit, regime histories): true.

### R1 reliability (2000 states from a long unconstrained simulation, one period solved from each, H = 40)

No method has a time limit; OccBin's own limit is maxit = 30 iterations (counted as its error 311). Different regime sequences are counted among the states both methods solve.

| s | method | solved | failed | failure types | solved by both it and OccBin | different regime sequences vs OccBin | max |path diff| vs OccBin (solved by both) |
|---|---|---|---|---|---|---|---|
| 0.5 | occbin | 2000 | 0 | - | - | - | - |
| 0.5 | minnorm | 2000 | 0 | - | 2000 | 0 | 3.9e-16 |
| 0.5 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 3.9e-16 |
| 1.0 | occbin | 2000 | 0 | - | - | - | - |
| 1.0 | minnorm | 2000 | 0 | - | 2000 | 0 | 5.6e-16 |
| 1.0 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 5.6e-16 |
| 1.5 | occbin | 2000 | 0 | - | - | - | - |
| 1.5 | minnorm | 2000 | 0 | - | 2000 | 0 | 1.0e-15 |
| 1.5 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 1.0e-15 |
| 2.0 | occbin | 2000 | 0 | - | - | - | - |
| 2.0 | minnorm | 2000 | 0 | - | 2000 | 0 | 1.3e-15 |
| 2.0 | minnorm_pure | 2000 | 0 | - | 2000 | 0 | 1.3e-15 |

s = 0.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 0.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 1.5: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 1.5: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.
s = 2.0: minnorm solves 0 states where OccBin fails; OccBin solves 0 states where minnorm fails.
s = 2.0: minnorm_pure solves 0 states where OccBin fails; OccBin solves 0 states where minnorm_pure fails.

load average at end of model: 1.81 1.62 1.39 2/108 2199

## Model `nkm` (OccBin, minnorm, minnorm without the fast path)

n_endo = 24, n_exo = 5, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=0.25 share=0% redraws=0 | s=0.5 share=0% redraws=0 | s=0.75 share=2% redraws=0 | s=1 share=6% redraws=0 | s=1.5 share=14% redraws=0 | s=2 share=20% redraws=0 | s=3 share=31% redraws=0

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 260.9 / 283.8 | 89.0 / 98.2 | 580.3 / 627.9 | 272.4 | 0.0e+00 | true |
| minnorm | before | 111.7 / 120.2 | 57.8 / 62.0 | 209.7 / 225.8 | 206.1 | 2.0e-05 | false |
| minnorm_pure | before | 146.5 / 158.8 | 60.8 / 68.9 | 303.8 / 328.3 | 256.6 | 2.0e-05 | false |
| occbin | after | 72.5 / 76.1 | 50.8 / 52.1 | 109.7 / 116.7 | 49.6 | 0.0e+00 | true |
| minnorm | after | 9.1 / 9.9 | 6.3 / 6.7 | 11.8 / 12.7 | 10.8 | 2.0e-05 | false |
| minnorm_pure | after | 40.7 / 42.3 | 7.2 / 7.6 | 101.1 / 104.9 | 61.3 | 2.0e-05 | false |

Shock scale s = 3.0, horizon 40, 1000 shock periods. Binding in the current period in 31.4% of periods; something binds somewhere on the path in 34.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.23 / 0.31 | 0.44 / 0.44 | 3.16 / 3.39 | 0.05 / 0.06 | 29.08 / 30.69 | 29.02 / 30.76 | 3.22 / 3.45 | 32.41 / 33.92 |
| after | 0.27 / 0.30 | 0.37 / 0.42 | 0.41 / 0.47 | 0.04 / 0.05 | 0.36 / 0.40 | 0.50 / 0.57 | 0.45 / 0.51 | 0.96 / 1.00 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phipi, phiy); horizon 40.

### P3 likelihood-like loop (50 iterations: perturb two parameters, redo P2, simulate 200 periods; H = 40)

| method | backend | seconds per loop min / median | ms per iteration (min) | failed iterations |
|---|---|---|---|---|
| occbin | before | 4.626 / 5.847 | 92.5 | 0 |
| minnorm | before | 3.323 / 4.011 | 66.5 | 0 |
| minnorm_pure | before | 4.046 / 4.641 | 80.9 | 0 |
| occbin | after | 0.895 / 1.024 | 17.9 | 0 |
| minnorm | after | 0.185 / 0.205 | 3.7 | 0 |
| minnorm_pure | after | 0.641 / 0.673 | 12.8 | 0 |

Correctness over the 50 parameter sets: minnorm: max path diff vs OccBin 2.9e-05, 42 iterations with different regime histories; minnorm_pure: max path diff vs OccBin 2.9e-05, 42 iterations with different regime histories; OccBin before vs after identical (paths bit for bit, regime histories): true.

### P4 scaling with the horizon (s = 3.0)

#### H = 20

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 182.9 / 189.9 | 51.2 / 53.1 | 424.4 / 443.0 | 230.3 | 0.0e+00 | true |
| minnorm | before | 94.1 / 97.5 | 34.8 / 35.7 | 196.6 / 210.6 | 148.2 | 2.0e-05 | false |
| minnorm_pure | before | 110.9 / 118.6 | 35.9 / 39.6 | 236.7 / 263.1 | 167.3 | 2.0e-05 | false |
| occbin | after | 48.4 / 49.5 | 32.2 / 33.2 | 70.1 / 72.4 | 31.3 | 0.0e+00 | true |
| minnorm | after | 8.8 / 9.4 | 4.7 / 5.2 | 9.9 / 11.6 | 8.6 | 2.0e-05 | false |
| minnorm_pure | after | 21.2 / 21.4 | 5.5 / 5.5 | 48.1 / 49.2 | 27.7 | 2.0e-05 | false |

Shock scale s = 3.0, horizon 20, 1000 shock periods. Binding in the current period in 31.4% of periods; something binds somewhere on the path in 34.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.28 / 0.31 | 0.41 / 0.46 | 3.15 / 3.41 | 0.06 / 0.07 | 30.87 / 33.36 | 29.23 / 30.96 | 3.21 / 3.49 | 32.38 / 34.37 |
| after | 0.29 / 0.33 | 0.43 / 0.45 | 0.43 / 0.62 | 0.04 / 0.05 | 0.39 / 0.44 | 0.46 / 0.53 | 0.47 / 0.66 | 0.89 / 1.14 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phipi, phiy); horizon 20.

#### H = 40

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 273.9 / 328.7 | 87.5 / 109.2 | 620.2 / 708.7 | 272.4 | 0.0e+00 | true |
| minnorm | before | 125.1 / 132.0 | 64.2 / 66.6 | 235.6 / 250.3 | 206.1 | 2.0e-05 | false |
| minnorm_pure | before | 165.6 / 167.0 | 66.7 / 68.6 | 342.6 / 343.0 | 256.6 | 2.0e-05 | false |
| occbin | after | 78.4 / 84.7 | 53.3 / 58.0 | 118.2 / 126.4 | 49.6 | 0.0e+00 | true |
| minnorm | after | 10.7 / 12.5 | 7.0 / 7.8 | 12.9 / 15.2 | 10.8 | 2.0e-05 | false |
| minnorm_pure | after | 45.0 / 46.9 | 7.9 / 8.4 | 111.8 / 116.0 | 61.3 | 2.0e-05 | false |

Shock scale s = 3.0, horizon 40, 1000 shock periods. Binding in the current period in 31.4% of periods; something binds somewhere on the path in 34.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.32 / 0.37 | 0.43 / 0.45 | 3.31 / 4.56 | 0.07 / 0.08 | 35.02 / 44.46 | 34.39 / 43.47 | 3.38 / 4.65 | 38.48 / 48.03 |
| after | 0.31 / 0.47 | 0.41 / 0.56 | 0.52 / 0.74 | 0.05 / 0.06 | 0.41 / 0.59 | 0.55 / 0.86 | 0.56 / 0.83 | 1.07 / 1.59 |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: phipi, phiy); horizon 40.

#### H = 80

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 383.5 / 504.2 | 165.2 / 216.0 | 789.4 / 1041.0 | 356.5 | 0.0e+00 | true |
| minnorm | before | 206.6 / 268.0 | 117.6 / 155.1 | 367.8 / 472.6 | 321.8 | 2.0e-05 | false |
| minnorm_pure | before | 339.9 / 452.0 | 162.8 / 191.2 | 630.2 / 838.1 | 465.7 | 2.0e-05 | false |
| occbin | after | 127.9 / 186.5 | 90.1 / 124.3 | 192.2 / 298.4 | 86.2 | 0.0e+00 | true |
| minnorm | after | 14.7 / 21.5 | 10.6 / 15.4 | 18.8 / 28.1 | 14.8 | 2.0e-05 | false |
| minnorm_pure | after | 134.1 / 137.3 | 13.0 / 13.6 | 357.4 / 365.2 | 158.7 | 2.0e-05 | false |

Shock scale s = 3.0, horizon 80, 1000 shock periods. Binding in the current period in 31.4% of periods; something binds somewhere on the path in 34.8% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.26 / 0.33 | 0.39 / 0.41 | 3.04 / 3.10 | 0.05 / 0.06 | 29.67 / 30.80 | 34.59 / 34.77 | 3.10 / 3.15 | 37.69 / 38.10 |
| after | 0.29 / 0.33 | 0.37 / 0.40 | 0.40 / 0.44 | 0.04 / 0.04 | 0.34 / 0.38 | 0.71 / 0.74 | 0.43 / 0.50 | 1.15 / 1.17 |

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

load average at end of model: 1.94 1.73 1.48 2/106 2256

## Model `sw` (OccBin only: multi_equation constraint, the news methods refuse it)

n_endo = 32, n_exo = 7, constraints = 1.

Scale search (first grid point with the constraint binding in 30-60% of periods and no method failing; if none exists without failures, the first one with failing draws replaced): s=1.0 fixed (no scale gives 30-60% binding without absurd shocks: see docs/OBC_PERF.md); share=20.8% redraws=50

**Failing draws were replaced: 50 of the 1000 shock periods were redrawn (fresh normal draw, same scale) until the simulation ran through with every method; the sequence is therefore not a pure iid normal sample.**

### P1 per period (1000 shock periods, H = 40)

| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 201.7 / 205.5 | 76.0 / 101.1 | 557.4 / 576.9 | 252.8 | 0.0e+00 | true |
| occbin | after | 68.5 / 89.1 | 51.3 / 66.7 | 123.1 / 161.9 | 51.6 | 0.0e+00 | true |

Shock scale s = 1.0, horizon 40, 1000 shock periods. Binding in the current period in 20.8% of periods; something binds somewhere on the path in 22.0% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

### P2 per parameter set (H = 40)

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.25 / 0.25 | 0.48 / 0.53 | 2.67 / 2.76 | 0.05 / 0.06 | - | - | 2.72 / 2.82 | - |
| after | 0.24 / 0.26 | 0.53 / 0.53 | 0.54 / 0.58 | 0.04 / 0.04 | - | - | 0.60 / 0.62 | - |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: crpi, cry); horizon 40.

### P3 likelihood-like loop (50 iterations: perturb two parameters, redo P2, simulate 200 periods; H = 40)

| method | backend | seconds per loop min / median | ms per iteration (min) | failed iterations |
|---|---|---|---|---|
| occbin | before | 3.651 / 4.375 | 73.0 | 0 |
| occbin | after | 0.805 / 1.011 | 16.1 | 0 |

Correctness over the 50 parameter sets: ; OccBin before vs after identical (paths bit for bit, regime histories): true.

### P4 scaling with the horizon (s = 1.0)

#### H = 20

Failing draws replaced at this horizon: 50 of 1000.
| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 144.2 / 169.6 | 44.3 / 53.8 | 491.5 / 562.7 | 213.6 | 0.0e+00 | true |
| occbin | after | 40.0 / 41.4 | 28.9 / 30.1 | 69.0 / 69.8 | 32.4 | 0.0e+00 | true |

Shock scale s = 1.0, horizon 20, 1000 shock periods. Binding in the current period in 20.8% of periods; something binds somewhere on the path in 22.0% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.25 / 0.27 | 0.50 / 0.53 | 2.66 / 2.72 | 0.05 / 0.05 | - | - | 2.71 / 2.77 | - |
| after | 0.24 / 0.26 | 0.49 / 0.55 | 0.54 / 0.59 | 0.05 / 0.05 | - | - | 0.59 / 0.63 | - |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: crpi, cry); horizon 20.

#### H = 40

Failing draws replaced at this horizon: 50 of 1000.
| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 224.9 / 280.7 | 78.5 / 140.0 | 644.0 / 773.1 | 252.8 | 0.0e+00 | true |
| occbin | after | 67.4 / 68.0 | 50.1 / 51.1 | 122.0 / 123.2 | 51.6 | 0.0e+00 | true |

Shock scale s = 1.0, horizon 40, 1000 shock periods. Binding in the current period in 20.8% of periods; something binds somewhere on the path in 22.0% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.23 / 0.26 | 0.50 / 0.56 | 2.68 / 3.02 | 0.05 / 0.06 | - | - | 2.74 / 3.08 | - |
| after | 0.24 / 0.27 | 0.51 / 0.53 | 0.54 / 0.59 | 0.04 / 0.05 | - | - | 0.59 / 0.64 | - |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: crpi, cry); horizon 40.

#### H = 80

Failing draws replaced at this horizon: 50 of 1000.
| method | backend | µs/period min / median | µs/period, nothing binds | µs/period, something binds | KiB allocated/period | max path diff vs OccBin (after) | regimes equal |
|---|---|---|---|---|---|---|---|
| occbin | before | 285.2 / 295.2 | 138.0 / 148.8 | 709.9 / 846.7 | 331.3 | 0.0e+00 | true |
| occbin | after | 127.3 / 136.4 | 96.7 / 103.0 | 227.6 / 244.2 | 90.0 | 0.0e+00 | true |

Shock scale s = 1.0, horizon 80, 1000 shock periods. Binding in the current period in 20.8% of periods; something binds somewhere on the path in 22.0% (the split above is by that second indicator). Timings: 5 repeats, interleaved, min / median.

| backend | steady_state | first_order | Occbin() total | regime matrices (all 2^k) | obc_constraints (post-processor) | obc_system (M, q; fresh Occbin, includes its constraints) | ready: occbin | ready: minnorm |
|---|---|---|---|---|---|---|---|---|
| before | 0.24 / 0.28 | 0.54 / 0.57 | 2.67 / 2.75 | 0.05 / 0.05 | - | - | 2.72 / 2.80 | - |
| after | 0.24 / 0.27 | 0.50 / 0.54 | 0.55 / 0.59 | 0.04 / 0.05 | - | - | 0.59 / 0.63 | - |

Milliseconds, min / median over 5 interleaved repeats, a different perturbed parameter vector in each (two parameters: crpi, cry); horizon 80.

### R1 reliability (2000 states from a long unconstrained simulation, one period solved from each, H = 40)

No method has a time limit; OccBin's own limit is maxit = 30 iterations (counted as its error 311). Different regime sequences are counted among the states both methods solve.

| s | method | solved | failed | failure types | solved by both it and OccBin | different regime sequences vs OccBin | max |path diff| vs OccBin (solved by both) |
|---|---|---|---|---|---|---|---|
| 0.5 | occbin | 1810 | 190 | occbin 310: 8, occbin 311: 51, occbin 313: 131 | - | - | - |
| 1.0 | occbin | 1735 | 265 | occbin 310: 15, occbin 311: 69, occbin 313: 181 | - | - | - |
| 1.5 | occbin | 1647 | 353 | occbin 310: 18, occbin 311: 86, occbin 313: 249 | - | - | - |
| 2.0 | occbin | 1585 | 415 | occbin 310: 27, occbin 311: 139, occbin 313: 249 | - | - | - |


load average at end of model: 2.07 1.83 1.55 2/105 2291

