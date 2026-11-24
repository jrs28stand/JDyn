#!/bin/sh
# Generates homotopy_ref.txt: Dynare 7.2 (Octave 8.4) steady(homotopy_mode=m, homotopy_steps=n) on tests/steady_state/homotopy/common.mod
# (bet 0.05 -> bet1 and x 1 -> x1) with solve_algo 9. One line per case: case mode steps nsolves nfailed | ys(c k) | exo | params(alph gam delt bet aa)
OUT=$(cd "$(dirname "$0")" && pwd)/homotopy_ref.txt
: > "$OUT"
W=$(mktemp -d)
run() {  # name mode steps bet1 x1 variant
  d=$W/$1; mkdir -p $d
  if [ "$6" = endval ]; then SETUP="steady(solve_algo=9);
endval; x = $5; end;
homotopy_setup(from_initval_to_endval); end;"; else SETUP="homotopy_setup; bet, 0.05, $4; x, $5; end;"; fi
cat > $d/m.mod <<EOT
var c k; varexo x; parameters alph gam delt bet aa;
alph=0.5; gam=0.5; delt=0.02; aa=0.5; bet=0.05;
model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;
initval;
x = 1;
k = ((delt+bet)/(aa*x*alph))^(1/(alph-1));
c = aa*x*k^alph-delt*k;
end;
$SETUP
steady(homotopy_mode=$2, homotopy_steps=$3, solve_algo=9);
EOT
cat > $d/x.m <<EOT
fid = fopen('ss.txt','w'); fprintf(fid, '%.17g ', oo_.steady_state, -999, oo_.exo_steady_state, -999, M_.params); fclose(fid);
EOT
  sh /home/claude/work/r2tools/oct.sh $d m $d/x.m > /dev/null 2>&1
  ns=$(grep -c "successful step\|computing step\|lauching solver" $d/oct.log)
  nf=$(grep -c "failed step" $d/oct.log)
  echo "$1 $2 $3 $ns $nf $(cat $d/ss.txt)" >> "$OUT"
}
run a1 1 50 0.1 2 par
run a2 2 20 0.1 2 par
run a3 3 50 0.1 2 par
run b1 1 50 0.1 2 endval
run b2 2 50 0.1 2 endval
run b3 3 50 0.1 2 endval
run c3 3 50 0.9 20 par
run c1 1 8 0.3 4 par
run d3 3 50 3 500 par
run e3 3 50 -0.95 0.01 par
run f3 3 50 200 0.0001 par
run g3 3 12 -0.99 30 par
cat "$OUT"
