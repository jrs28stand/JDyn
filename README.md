# JDyn prebuilt binaries

Tarballs downloaded by `Pkg` through JDyn's `Artifacts.toml` (artifact `korder`): Dynare 7.2's k-order library with
JDyn's C API (`deps/korder/`), plus the OpenBLAS and runtime libraries it needs.
Built by `deps/korder/build.sh` (Linux, then relinked against conda-forge OpenBLAS with rpath `$ORIGIN`) and
`deps/korder/build_windows.sh` (cross-compiled with MinGW-w64 against conda-forge OpenBLAS), packaged by
`tools/make_korder_artifacts.jl`. Linux build needs glibc >= 2.38. Licences: Dynare GPL v3+; OpenBLAS BSD-3;
GCC runtime libraries GPL with the runtime exception; vcruntime140.dll Microsoft Visual C++ redistributable.
Do not rewrite this branch: published artifact URLs point at these files.
