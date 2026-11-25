#!/bin/sh
# Generates save_params_ref.txt with Dynare 7.2 / Octave 8.4 (save_params_and_steady_state after steady(solve_algo=9))
OUT=$(cd "$(dirname "$0")" && pwd)/save_params_ref.txt
W=$(mktemp -d)
cat > $W/s.mod <<EOT
var c k; varexo x; parameters alph;
alph=0.5;
model; c + k - x*k(-1)^alph - 0.98*k(-1); c^(-0.5) - (1/1.05)*(0.5*alph*x(+1)*k^(alph-1) + 0.98)*c(+1)^(-0.5); end;
initval; x=1; k=10; c=2; end;
steady(solve_algo=9);
save_params_and_steady_state('pp.txt');
EOT
sh /home/claude/work/r2tools/oct.sh $W s > /dev/null 2>&1
cp $W/pp.txt "$OUT"; cat "$OUT"
