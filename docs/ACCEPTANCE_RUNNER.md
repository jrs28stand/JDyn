# JDyn's side of the cross-port acceptance sweep

This runs Dynare 7.2's test suite through `JDyn.dynare()` and writes, per test file, the results in the format the shared
comparator reads (the "Combined acceptance and timing sweep" section of the capability tracker). It does not replace
`test/acceptance/run_dynare_tests.jl`, which stays for each file's own checks (see `docs/ACCEPTANCE.md`).

The Python side reuses pydyn's tools rather than porting them: `accept_manifest.py` (entries and file order from Dynare's
`meson.build`), `accept_octave.py` (`relid`, `prepare_scratch`, `cost`) and, for the verdicts, `accept_compare.py`. They are read from
`/home/claude/work/pydyn-rl/tools` (override with `PYDYN_ACCEPT_TOOLS`).

## Files

| file | role |
|---|---|
| `test/acceptance/accept_jdyn.py` | driver: entries, test ids, scratch folders, 2 worker processes, timeouts, restarts, resume |
| `test/acceptance/accept_worker.jl` | long-lived Julia worker: runs one file at a time through `dynare()`, converts `ws.oo_/M_/options_`, writes the MAT-file |
| `test/acceptance/run_sweep.sh` | starts (or resumes) the driver detached |

## Running

```
export PATH=/root/jl/env/bin:$PATH JULIA_PKG_SERVER="" GKSwstype=100
export DYNARE_PREPROCESSOR=/root/ref/pp-standalone/src/dynare-preprocessor     # run_sweep.sh sets these itself
test/acceptance/run_sweep.sh                       # full sweep, detached; log and pid file in /home/claude/accept_jdyn
python3 test/acceptance/accept_jdyn.py --ids fs2000/fs2000 --out /tmp/x   # a few files, in the foreground
```

Options of `accept_jdyn.py`: `--out` (default `/home/claude/accept_jdyn`), `--ref` (the Octave store, default
`/home/claude/accept_ref/accept_ref`), `--jobs 2`, `--timeout 900`, `--only substr ...` (substring of the test path), `--ids id ...`
(exact ids), `--force` (re-run ids that have a `.json`), `--all` (also run tests whose Octave run failed), `--limit N` entries,
`--no-ref-artifacts`, `--julia-arg ARG` (e.g. `--sysimage=...`). `run_sweep.sh` passes its arguments on.
Environment: `DYNARE_SRC` (default `/root/ref/dynare-7.2`), `DYNARE_PREPROCESSOR`, `JDYN_ACCEPT_RNG=octave` (see "Random numbers").

Resuming: the driver skips every id whose `<id>.json` exists, so after a crash or a machine restart run `test/acceptance/run_sweep.sh`
again. A second start while a sweep runs is refused (a lock on `<out>/.lock`, and `run_sweep.sh` checks `<out>/sweep.pid`). To stop:
`kill $(cat /home/claude/accept_jdyn/sweep.pid)` (the workers are killed with it). To redo some files:
`run_sweep.sh --force --only occbin/`.

## What it does for each file

* Entries, ids and file order are pydyn's: the files of one `meson.build` entry run in order in one scratch folder
  (`/var/tmp/accj_*`), laid out by `prepare_scratch` (the folder's files plus the entry's extra files, `../examples/...` landing inside
  the scratch tree). Entries run cheapest first (`cost`).
* Before each file, the files Dynare's own Octave run left next to it are copied in from `<ref>/_files/<id>/` (as `accept_pydyn.py`
  does). On a resumed run, the same is done for earlier files of the entry that are not re-run, and an earlier file whose name the
  later pending files mention is re-run unrecorded ("setup" line in the log), so that e.g. `example1long/Output/example1long_results.mat`
  exists for `example2long_use_dll`.
* The file is run with `JDyn.dynare(path; extra_args = <the entry's -D arguments>)` from the file's folder (no figures).
* Skipped: tests whose Octave reference status is not `ok` (dynare-error, timeout, xfail, m-runner, no-result; as `accept_pydyn.py`),
  the `.m` runners, and the whole `heterogeneity/` folder (heterogeneous-agent models are outside both correctness and timing until a
  port supports them). That left 735 .mod tests on the first full run (the log's first line gives the count).

## Output formats

`<out>/<id>.mat`: MAT-file level 5 (zlib-compressed variables), read by `scipy.io.loadmat`. Variables `oo_`, `M_`, `options_`
holding the values of the results layer under Dynare's names and shapes (numbers as doubles, logicals, char, cells, nested structs,
complex arrays; Octave's `accept_save.m` drops objects and so does the converter). Reasons: MAT.jl only writes v4 and v7.3 (HDF5), which
scipy cannot read, so `accept_worker.jl` has its own small v5 writer (double, logical, char, cell, scalar struct). Values JDyn
does not hold are absent; a `oo_` entry that JDyn marks as unavailable (a `Tainted` value) is listed in `refused_fields`.
Struct arrays are stored as cells of structs, as `accept_save.m`. Field names that are not MATLAB identifiers are dropped.
The `.mat` is deleted unless the status is `ok`.

`<out>/<id>.json`:

| key | meaning |
|---|---|
| `status` | `ok`, `refused`, `error` or `timeout` |
| `message` | the error text (empty for `ok`) |
| `refused_fields` | `{oo_ path: reason}` for fields JDyn declines to fill (optional) |
| `load_s` | worker start (process start) to `using JDyn` done; once per worker, repeated in each of its files |
| `warmup_s` | the worker's warm-up run of a small model before its first file (not part of `load_s`; its compilation is what later files reuse) |
| `first_s` | the first `dynare()` call of the file: preprocessor, the file's run, MATLAB lines |
| `second_s` | the same call again in the same worker; left out (and `second_skipped` says why) when `first_s` is over 120 s, when the second call fails, or when the file's 900 s ran out |
| `warnings` | number of warnings JDyn logged in the first call |
| `trace` | Julia backtrace tail for `error` |

Classification: `refused` = JDyn's clear refusals (`ModelError`, `Unsupported`, `EstimationError`, `SmootherError`,
`NonlinearFilterError`, `OptimalPolicyError`, `OSRError`, `DiscretionaryPolicyError`, `LoglinearError`, `KOrderError`, `ParseError`,
or an error whose text says not supported/implemented). A `MatlabError` (the file's own `error()`/`assert()` or an interpreter error,
also when wrapped in a `ModelError`) is `error`. `timeout`: no result after `--timeout` seconds (900, Octave's); the worker is
killed (process group, preprocessor included) and restarted. A worker that dies is recorded as `error` ("worker process died") and
restarted; workers are also restarted after 60 files or above 3.5 GB resident size.

## Random numbers

The json carries no `"rng": "octave"`: JDyn draws its stochastic-simulation numbers from its own generator, not Octave's, so the
comparator skips the random-number fields (`endo_simul`, `exo_simul`, `irfs`, MCMC output, ...). With `JDYN_ACCEPT_RNG=octave` the
worker passes `particle_rng=:octave, mcmc_rng=:octave` to `dynare()`; the `"rng"` key is still not written, because simulations
and `rand`/`randn` in MATLAB code do not use Octave's stream yet. Set it in the driver only when that holds for every number.

## Comparing

```
python3 /home/claude/work/pydyn-rl/tools/accept_compare.py /home/claude/accept_jdyn \
    --ref /home/claude/accept_ref/accept_ref --known /home/claude/work/pydyn-rl/tools/accept_known.json \
    --name jdyn --report /home/claude/accept_jdyn/report.md
```

`--ref` is the folder that holds the `<id>.json/.mat` of the Octave runs (the `accept_ref` folder inside the unpacked store, not the
store's top folder). The comparator walks every id of the reference, so an id the sweep has not reached yet is a `port-error` ("wrote no
result") until the sweep is complete; run it on a partial sweep with `--only` (substring of the ids that exist). It writes
`<out>/_verdicts.json` too. Report sections: verdicts and by-folder tables, wrong results, port errors, commonest refusals, timing
(`load_s`, `first_s`, `second_s`). The Octave `elapsed` of the store is whole-process wall time; the figure comparable with it is
`load_s + first_s` (shared workers reuse code compiled for earlier files, so this understates a truly cold run).

## Notes on the comparison (not runner faults)

* Fields JDyn does not hold at all appear as `not-produced` (so `partial`), e.g. `oo_.exo_simul`, `dr.sdim`, `dr.edim`, `dr.gx` after estimation.
* The comparator compares simulated moments (`oo_.mean`, `var`, `autocorr`, ... of `periods>0` runs) and `posterior.optimization.*.runtime`
  without treating them as random-number or bookkeeping fields.
* Numerically solved steady states agree with Dynare's to about 1e-5, not the 1e-8 the comparator applies (the tracker's protocol allows 1e-5 for those).
