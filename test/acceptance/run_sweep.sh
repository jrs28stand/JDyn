#!/usr/bin/env bash
# Launch (or resume) JDyn's full acceptance sweep detached.  Safe to re-run: if a sweep is running it says so and exits; otherwise
# it removes the scratch folders and workers a dead run left behind and continues where the output folder stands (ids with a
# .json are skipped).  Extra arguments go to accept_jdyn.py (e.g. --only occbin, --force).
#   log: $OUT/sweep.log    pid file: $OUT/sweep.pid     stop: kill $(cat $OUT/sweep.pid)
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ACCEPT_OUT:-/home/claude/accept_jdyn}"
mkdir -p "$OUT"
if [ -f "$OUT/sweep.pid" ] && kill -0 "$(cat "$OUT/sweep.pid")" 2>/dev/null; then
    echo "sweep already running (pid $(cat "$OUT/sweep.pid")); log $OUT/sweep.log"
    exit 0
fi
pkill -f "test/acceptance/accept_worker.jl" 2>/dev/null || true     # workers orphaned by a killed driver
rm -rf /var/tmp/accj_* 2>/dev/null || true
export PATH=/root/jl/env/bin:$PATH JULIA_PKG_SERVER="" GKSwstype=100
export DYNARE_PREPROCESSOR="${DYNARE_PREPROCESSOR:-/root/ref/pp-standalone/src/dynare-preprocessor}"
export DYNARE_SRC="${DYNARE_SRC:-/root/ref/dynare-7.2}"
cd "$HERE/../.."
setsid nohup python3 "$HERE/accept_jdyn.py" --out "$OUT" "$@" >> "$OUT/sweep.log" 2>&1 < /dev/null &
echo $! > "$OUT/sweep.pid"
echo "sweep started (pid $!); log $OUT/sweep.log"
