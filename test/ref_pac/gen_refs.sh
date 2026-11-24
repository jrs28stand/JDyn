#!/bin/sh
# Regenerates the Octave references of test_pac.jl: for each fixture (a Dynare 7.2 test .mod of tests/pac or tests/var-expectations,
# cut before its first dseries / simulation line) run Dynare 7.2 under Octave and save M_.params, the companion matrices and
# (when a stoch_simul ran) the moduli of the eigenvalues in <fixture>.ref.mat.   usage: sh gen_refs.sh [fixture ...]
HERE=$(cd "$(dirname "$0")" && pwd)
TESTS=${DYNARE_TESTS:-/root/ref/dynare-7.2/tests}
OCT=${OCT_SH:-/home/claude/work/r2tools/oct.sh}
for f in ${@:-$(cd $HERE && ls *.mod)}; do
  n=${f%.mod}
  w=$(mktemp -d)
  cp "$HERE/$f" "$w/$n.mod"
  cp "$HERE/extract.m" "$w/extract.m"
  [ -f "$HERE/$n.data" ] && cp "$HERE/$n.data"/* "$w/" 2>/dev/null
  sh $OCT "$w" "$n" extract.m > "$w/oct.out" 2>&1
  if [ -f "$w/ref.mat" ]; then cp "$w/ref.mat" "$HERE/$n.ref.mat"; echo "ok $n"; else echo "FAILED $n"; tail -5 "$w/oct.out"; fi
  rm -rf "$w"
done
