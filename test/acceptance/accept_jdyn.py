#!/usr/bin/env python3
"""JDyn's side of the cross-port acceptance protocol (docs/ACCEPTANCE_RUNNER.md).

    python3 test/acceptance/accept_jdyn.py [--out /home/claude/accept_jdyn] [--ref /home/claude/accept_ref/accept_ref]
            [--jobs 2] [--timeout 900] [--only substr ...] [--ids id ...] [--force] [--all] [--limit N]

Entries, test ids, file order and scratch-folder layout are pydyn's (tools/accept_manifest.py, tools/accept_octave.py: `relid`,
`prepare_scratch`, `cost`), and a test never starts with the files Dynare wrote when running that same file (see --ref-artifacts), as
tools/accept_pydyn.py does.  The files of an entry run one after the other in one scratch folder, through
`JDyn.dynare()` in a long-lived Julia worker (test/acceptance/accept_worker.jl).  Per test id it writes
`<out>/<id>.mat` and `<out>/<id>.json` (format: tools/accept_compare.py).  Resumable (ids with a .json are skipped unless --force);
a file over --timeout, or a worker that crashes, is recorded and the worker restarted.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PYDYN_TOOLS = os.environ.get("PYDYN_ACCEPT_TOOLS", "/home/claude/work/pydyn-rl/tools")
sys.path.insert(0, PYDYN_TOOLS)
from accept_manifest import load_manifest  # noqa: E402
from accept_octave import cost, prepare_scratch, relid  # noqa: E402

SRC = Path(os.environ.get("DYNARE_SRC", "/root/ref/dynare-7.2"))
TAG = os.environ.get("ACCEPT_TAG", "accj")      # distinct per clone when several run at once (work folders are shared under /var/tmp)
WORKER = HERE / "accept_worker.jl"
SKIP_FOLDERS = ("heterogeneity/",)          # heterogeneous-agent models: out of both correctness and timing until a port supports them
MAX_RSS_GB = 2.5                              # restart a worker above this resident size (Julia keeps memory)
MAX_TESTS_PER_WORKER = 60
_stop = threading.Event()
_workers: list = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Worker:
    """One long-lived `julia accept_worker.jl` process, driven through files in its work directory."""

    def __init__(self, idx, jl_args):
        self.idx, self.jl_args = idx, jl_args
        self.dir = Path("/var/tmp") / f"{TAG}_worker{idx}"
        self.proc = None
        self.n = 0                      # commands sent to the current process
        self.tests = 0
        self.load_s = None
        _workers.append(self)

    # -- life cycle
    def start(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        env = dict(os.environ)
        env.setdefault("DYNARE_PREPROCESSOR", "/root/ref/pp-standalone/src/dynare-preprocessor")
        env.setdefault("JULIA_PKG_SERVER", "")
        env.setdefault("GKSwstype", "100")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        cmd = ["nice", "-n", "10", "julia", f"--project={ROOT}", *self.jl_args, str(WORKER), str(self.dir)]
        logf = open(self.dir.parent / f"{TAG}_worker{self.idx}.log", "ab")
        logf.write(f"\n=== {time.ctime()} starting worker\n".encode())
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        self.n = self.tests = 0
        while not (self.dir / "ready").exists():
            if _stop.is_set() or self.proc.poll() is not None or time.time() - t0 > 1500:
                self.kill()
                raise RuntimeError("worker did not start; see " + str(self.dir.parent / f"{TAG}_worker{self.idx}.log"))
            time.sleep(0.2)
        self.load_s = float((self.dir / "ready").read_text().split()[0])
        log(f"worker {self.idx} ready (load_s={self.load_s}, pid {self.proc.pid})")

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        self.proc = None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            (self.dir / "quit").write_text("")
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                pass
        self.kill()

    def rss_gb(self):
        try:
            for ln in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
                if ln.startswith("VmRSS"):
                    return int(ln.split()[1]) / 1e6
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def ensure(self):
        if self.proc is None or self.proc.poll() is not None or self.tests >= MAX_TESTS_PER_WORKER or self.rss_gb() > MAX_RSS_GB:
            self.kill()
            self.start()

    # -- one command
    def run(self, cmd, timeout):
        """Run a command; returns (info dict, how) with how in ok/timeout/crash."""
        self.ensure()
        self.n += 1
        self.tests += 1
        n = self.n
        tmp = self.dir / f"cmd.{n}.tmp"
        tmp.write_text(json.dumps(cmd))
        os.replace(tmp, self.dir / f"cmd.{n}.json")
        t0 = time.time()
        reply, done = self.dir / f"reply.{n}.json", self.dir / f"done.{n}"
        while not done.exists():
            if _stop.is_set():
                raise KeyboardInterrupt
            if self.proc.poll() is not None:
                how = "crash"
                break
            if time.time() - t0 > timeout:
                how = "timeout"
                break
            time.sleep(0.2)
        else:
            how = "ok"
        info = json.loads(reply.read_text()) if reply.exists() else None
        if how != "ok":
            tail = ""
            try:
                tail = (self.dir.parent / f"{TAG}_worker{self.idx}.log").read_bytes()[-600:].decode(errors="replace").replace("\n", " | ")
            except OSError:
                pass
            self.kill()
            if info is not None and info.get("status") == "ok":      # the first call and the .mat were complete; only the second call was cut
                info["second_skipped"] = {"timeout": f"time limit {timeout}s reached during the second call",
                                          "crash": "worker died during the second call"}[how]
                return info, how
            if how == "timeout":
                return {"status": "timeout", "message": f"timeout after {timeout}s", "load_s": self.load_s}, how
            return {"status": "error", "message": "worker process died (crash/OOM) in this test; log tail: " + tail[-350:],
                    "load_s": self.load_s}, how
        return info, how


def copy_ref_artifacts(ref, mod, scratch, overwrite=True):
    art = Path(ref) / "_files" / relid(mod)
    if art.is_dir():
        dest = scratch / "tests" / Path(mod).parent
        for f in art.rglob("*"):
            if f.is_file():
                t = dest / f.relative_to(art)
                if t.exists() and not overwrite:
                    continue
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, t)


def mod_text(m):
    try:
        return (SRC / "tests" / m).read_text(errors="replace")
    except OSError:
        return ""


def setup_mods(entry, mods):
    """Files of the entry that are already done (resume) or not selected, but whose outputs a later pending file reads
    (its text mentions their name: example1long -> example2long_use_dll reads example1long/Output/example1long_results.mat).
    They are re-run without recording anything, in file order, so that the scratch folder holds what Dynare's run had."""
    todo = []
    for i, m in enumerate(entry["test"]):
        if m in mods or not m.endswith(".mod"):
            continue
        stem = Path(m).stem
        if any(mod_text(later).find(stem) >= 0 for later in entry["test"][i + 1:] if later in mods):
            todo.append(m)
    return todo


def run_setup(w, m, scratch, timeout, ref):
    path = os.path.normpath(scratch / "tests" / m)
    try:
        info, how = w.run({"setup": True, "path": path, "dynare_args": []}, timeout)
    except KeyboardInterrupt:
        return
    if how != "ok" or info.get("status") != "ok":          # the earlier file did not complete: Dynare's outputs of THAT file
        copy_ref_artifacts(ref, m, scratch, overwrite=False)
    log(f"setup    {relid(m)} (re-run unrecorded: a later file of the entry reads its outputs)")


def run_entry(w, entry, mods, out, ref, timeout, ref_artifacts):
    scratch = Path(tempfile.mkdtemp(prefix=f"{TAG}_", dir="/var/tmp"))
    setups = set(setup_mods(entry, mods))
    try:
        prepare_scratch(entry, SRC, scratch)
        last = max(i for i, m in enumerate(entry["test"]) if m in mods)
        if ref_artifacts:                  # legacy diagnosis mode (--ref-artifacts): Dynare's outputs of every earlier file
            for m in entry["test"][:last]:
                if m.endswith(".mod") and m not in mods:
                    copy_ref_artifacts(ref, m, scratch)
        for m in entry["test"]:
            if _stop.is_set():
                return
            if m in setups:
                run_setup(w, m, scratch, timeout, ref)
                continue
            if m not in mods:
                continue
            rel = relid(m)
            mat, js = Path(out) / (rel + ".mat"), Path(out) / (rel + ".json")
            mat.parent.mkdir(parents=True, exist_ok=True)
            if mat.exists():
                mat.unlink()
            if ref_artifacts:                  # legacy: the file's OWN Dynare outputs (diagnosis only)
                copy_ref_artifacts(ref, m, scratch)
            path = os.path.normpath(scratch / "tests" / m)
            t = time.time()
            try:
                info, how = w.run({"rel": rel, "path": path, "mat": str(mat), "dynare_args": entry.get("dynare_args", []),
                                   "second_limit_s": 120.0}, timeout)
            except KeyboardInterrupt:
                return
            except Exception as e:  # noqa: BLE001
                info, how = {"status": "error", "message": f"driver: {type(e).__name__}: {e}"}, "driver"
            if info.get("status") != "ok" and mat.exists():
                mat.unlink()
            if info.get("status") == "ok" and not mat.exists():
                info.update(status="error", message="no .mat written")
            if info.get("status") != "ok" and not ref_artifacts:     # a later file of the entry gets what Dynare's run of this one wrote
                copy_ref_artifacts(ref, m, scratch, overwrite=False)
            tmp = js.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(info, indent=1))
            os.replace(tmp, js)
            log(f"{info['status']:8s} {time.time() - t:7.1f}s {rel}  {info.get('message', '')[:110]}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def build_jobs(a, out, ref):
    entries = sorted(load_manifest(SRC), key=lambda e: cost(e, SRC))
    jobs = []
    for e in entries:
        mods = []
        for m in e["test"]:
            rel = relid(m)
            if not m.endswith(".mod") or rel.startswith(SKIP_FOLDERS):
                continue
            if a.only and not any(o in m for o in a.only):
                continue
            if a.ids and rel not in a.ids:
                continue
            rj = ref / (rel + ".json")
            if not rj.exists() or (not a.all and json.loads(rj.read_text())["status"] != "ok"):
                continue
            if (out / (rel + ".json")).exists() and not a.force:
                continue
            mods.append(m)
        if mods:
            jobs.append((e, mods))
    if a.limit:
        jobs = jobs[:a.limit]
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/claude/accept_jdyn")
    ap.add_argument("--ref", default="/home/claude/accept_ref/accept_ref")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", nargs="*", help="substrings of the test path")
    ap.add_argument("--ids", nargs="*", help="exact test ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all", action="store_true", help="also run tests whose Dynare reference run failed")
    ap.add_argument("--limit", type=int, help="at most N entries")
    ap.add_argument("--ref-artifacts", action="store_true",
                    help="legacy: pre-copy each file's own Dynare outputs (diagnosis only; made 46 files refuse for the wrong reason)")
    ap.add_argument("--julia-arg", action="append", default=[], help="extra argument for julia (e.g. --sysimage=...)")
    a = ap.parse_args(argv)
    out, ref = Path(a.out), Path(a.ref)
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another sweep is already writing to {out}")
    jobs = build_jobs(a, out, ref)
    log(sum(len(m) for _, m in jobs), "tests in", len(jobs), "entries; output", out)
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def shutdown(*_):
        _stop.set()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    def loop(idx):
        w = Worker(idx, a.julia_arg)
        try:
            while not _stop.is_set():
                try:
                    e, mods = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    run_entry(w, e, mods, out, ref, a.timeout, a.ref_artifacts)
                except Exception as ex:  # noqa: BLE001
                    log("entry failed:", mods, repr(ex))
                    time.sleep(5)
        finally:
            w.stop()
    threads = [threading.Thread(target=loop, args=(i,)) for i in range(a.jobs)]
    for i, t in enumerate(threads):
        t.start()
        time.sleep(1)
    for t in threads:
        while t.is_alive():
            t.join(1)
    for w in _workers:
        w.kill()
    log("INTERRUPTED" if _stop.is_set() else "DONE")


if __name__ == "__main__":
    main()
