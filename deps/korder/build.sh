#!/bin/sh
# Build Dynare's k-order perturbation library (libkorder) plus JDyn's C API as one shared library,
# without MATLAB or Octave.
# Usage: deps/korder/build.sh [/path/to/dynare-7.2]     (default: $JDYN_DYNARE_SRC or /root/ref/dynare-7.2)
# Needs g++ >= 13 (C++20, OpenMP), BLAS and LAPACK (-llapack -lblas). Output: deps/korder/libjdyn_korder.so
set -e
DYNARE=${1:-${JDYN_DYNARE_SRC:-/root/ref/dynare-7.2}}
HERE=$(cd "$(dirname "$0")" && pwd)
S=$DYNARE/mex/sources; L=$S/libkorder
[ -d "$L" ] || { echo "no $L: pass the Dynare 7.2 source tree" >&2; exit 1; }
FLAGS="-std=gnu++20 -O2 -fPIC -fopenmp -DOCTAVE_MEX_FILE -DPACKAGE_VERSION=\"7.2\" -I$HERE/stub -I$S -I$L -I$L/kord -I$L/sylv -I$L/tl -I$L/utils"
OBJ=$HERE/build
mkdir -p "$OBJ"
JOBS=${JDYN_BUILD_JOBS:-$(nproc)}
ls $L/kord/*.cc $L/sylv/*.cc $L/tl/*.cc $L/utils/*.cc $L/k_ord_dynare.cc "$HERE/jdyn_korder_capi.cc" \
  | xargs -P "$JOBS" -I{} sh -c "g++ $FLAGS -c {} -o $OBJ/\$(basename {} .cc).o"
g++ -shared -fopenmp "$OBJ"/*.o -o "$HERE/libjdyn_korder.so" -llapack -lblas -lpthread
echo "built $HERE/libjdyn_korder.so"
