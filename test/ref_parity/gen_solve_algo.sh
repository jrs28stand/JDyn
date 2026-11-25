#!/bin/sh
# Generates solve_algo_ref.txt: oo_.steady_state of Dynare 7.2 (Octave 8.4) for solve_algo 4, 9, 13, 14 on two models.
# usage: sh gen_solve_algo.sh   (needs /home/claude/work/r2tools/oct.sh)
OUT=$(cd "$(dirname "$0")" && pwd)/solve_algo_ref.txt
: > "$OUT"
W=$(mktemp -d)
for model in rbc blocks rbcfar; do
for algo in 4 9 13 14; do
  d=$W/${model}_$algo; mkdir -p $d
  if [ $model = rbc ] || [ $model = rbcfar ]; then
    if [ $model = rbc ]; then IV='k = 20; c = 1.5; y = 2.5;'; else IV='k = 300; c = 0.2; y = 0.3;'; fi
cat > $d/m.mod <<EOT
var y c k; varexo e; parameters alpha beta delta A;
model; 1/c = beta/c(+1)*(alpha*A*k^(alpha-1)+1-delta); y = A*exp(e)*k(-1)^alpha; k = y - c + (1-delta)*k(-1); end;
alpha = 0.33; beta = 0.99; delta = 0.025; A = 1;
initval; $IV end;
steady(solve_algo=$algo);
EOT
  else
cat > $d/m.mod <<EOT
var a b c d f; varexo e;
model;
log(a) = 0.5 + 0.3*log(a(-1));
b*exp(b(-1)) = a(-1);
c = 0.3*d(-1)^0.5 + b(-1);
d = c(-1)^2/(1+c(-1)) + 0.1;
f = a*c + d + e;
end;
initval; a = 3; b = 0.5; c = 1; d = 1; f = 1; end;
steady(solve_algo=$algo);
EOT
  fi
  sh /home/claude/work/r2tools/oct.sh $d m "$(echo $d/x.m)" >/dev/null 2>&1
  cat > $d/x.m <<EOT
fid = fopen('ss.txt','w'); fprintf(fid, '%.17g ', oo_.steady_state); fclose(fid);
EOT
  sh /home/claude/work/r2tools/oct.sh $d m $d/x.m > /dev/null 2>&1
  echo "$model $algo $(cat $d/ss.txt)" >> "$OUT"
done
done
cat "$OUT"
