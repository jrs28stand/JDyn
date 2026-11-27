#!/bin/sh
# Build Dynare's k-order perturbation library plus JDyn's C API as a Windows DLL (deps/korder/libjdyn_korder.dll),
# needed for orders 2-3 on Windows. Two ways to run it:
#
#  1. Natively, in an MSYS2 "MINGW64" shell (https://www.msys2.org):
#       pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-openblas
#       deps/korder/build_windows.sh /c/path/to/dynare-7.2
#     Links MSYS2's OpenBLAS (which includes LAPACK).
#
#  2. Cross-compiled on Linux with MinGW-w64 (Ubuntu: apt install g++-mingw-w64-x86-64-posix), against a Windows
#     OpenBLAS DLL, e.g. conda-forge's: micromamba create --platform win-64 -p /tmp/winblas -c conda-forge libopenblas
#       JDYN_WIN_BLAS=/tmp/winblas/Library/bin/openblas.dll deps/korder/build_windows.sh /path/to/dynare-7.2
#     The BLAS DLL (and, if present beside it, vcruntime140.dll) is copied next to the output.
#
# The first argument is Dynare 7.2's source tree (default: $JDYN_DYNARE_SRC or /root/ref/dynare-7.2); only its
# mex/sources folder is used (the release tarball has it; the Windows installer may not).
# Cross-compiled, the GCC runtime is linked statically, so the DLL needs only the BLAS/LAPACK DLL beside it; built
# under MSYS2, the MinGW DLLs it needs (OpenBLAS, gfortran, winpthread, ...) are copied beside it. JDyn finds deps/korder/libjdyn_korder.dll by itself; elsewhere, set JDYN_KORDER_LIB or call
# korder_library(path).
set -e
DYNARE=${1:-${JDYN_DYNARE_SRC:-/root/ref/dynare-7.2}}
HERE=$(cd "$(dirname "$0")" && pwd)
S=$DYNARE/mex/sources; L=$S/libkorder
[ -d "$L" ] || { echo "no $L: pass the Dynare 7.2 source tree" >&2; exit 1; }

if [ -n "$JDYN_WIN_BLAS" ]; then                      # cross-compiling from Linux
    CXX=${CXX:-x86_64-w64-mingw32-g++-posix}
    BLAS="$JDYN_WIN_BLAS"
    [ -f "$BLAS" ] || { echo "JDYN_WIN_BLAS=$BLAS does not exist" >&2; exit 1; }
else                                                  # MSYS2 MINGW64 shell
    CXX=${CXX:-g++}
    BLAS="-lopenblas"
fi
FLAGS="-std=gnu++20 -O2 -fopenmp -DOCTAVE_MEX_FILE -DPACKAGE_VERSION=\"7.2\" -I$HERE/stub -I$S -I$L -I$L/kord -I$L/sylv -I$L/tl -I$L/utils"
OBJ=$HERE/build-windows
mkdir -p "$OBJ"
JOBS=${JDYN_BUILD_JOBS:-$(nproc 2>/dev/null || echo 2)}
ls $L/kord/*.cc $L/sylv/*.cc $L/tl/*.cc $L/utils/*.cc $L/k_ord_dynare.cc "$HERE/jdyn_korder_capi.cc" \
  | xargs -P "$JOBS" -I{} sh -c "$CXX $FLAGS -c {} -o $OBJ/\$(basename {} .cc).o"
OUT=$HERE/libjdyn_korder.dll
if [ -n "$JDYN_WIN_BLAS" ]; then
    $CXX -shared -static -fopenmp "$OBJ"/*.o -o "$OUT" "$BLAS"     # GCC runtime fully static; BLAS DLL linked directly
else
    $CXX -shared -fopenmp -static-libgcc -static-libstdc++ "$OBJ"/*.o -o "$OUT" $BLAS
fi
if [ -n "$JDYN_WIN_BLAS" ]; then
    cp "$JDYN_WIN_BLAS" "$HERE/"
    VCRT="$(dirname "$JDYN_WIN_BLAS")/vcruntime140.dll"
    [ -f "$VCRT" ] && cp "$VCRT" "$HERE/"
else
    # copy the MinGW runtime DLLs the output still needs (normally only libopenblas.dll) beside it
    for d in $(ldd "$OUT" | awk '/mingw64/ {print $3}'); do cp "$d" "$HERE/"; done
fi
echo "built $OUT"
