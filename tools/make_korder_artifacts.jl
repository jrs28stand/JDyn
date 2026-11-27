# Package prebuilt k-order libraries as Julia artifacts and bind them in Artifacts.toml.
#   julia tools/make_korder_artifacts.jl <version> <linux-stage-dir> <windows-stage-dir> <outdir>
# Each stage dir holds libjdyn_korder.{so,dll} and the BLAS/runtime libraries it needs. The tarballs in <outdir> go on
# the `binaries` branch of the JDyn repository (download URLs below); `Pkg.add` then fetches the right one.
using Pkg.Artifacts, Pkg.BinaryPlatforms
ver, lin, win, out = ARGS
mkpath(out)
toml = joinpath(@__DIR__, "..", "Artifacts.toml")
for (dir, plat, tag) in ((lin, Platform("x86_64", "linux"; libc="glibc"), "x86_64-linux-gnu"),
                         (win, Platform("x86_64", "windows"), "x86_64-w64-mingw32"))
    h = create_artifact() do art
        for f in readdir(dir); cp(joinpath(dir, f), joinpath(art, f)); end
        Sys.iswindows() || foreach(f -> chmod(joinpath(art, f), 0o755), readdir(art))
    end
    name = "korder-$ver-$tag.tar.gz"
    tarball = joinpath(out, name)
    sha = archive_artifact(h, tarball)
    url = "https://github.com/jrs28stand/JDyn/raw/binaries/$name"
    bind_artifact!(toml, "korder", h; platform=plat, download_info=[(url, sha)], force=true)
    println(name, "  ", filesize(tarball) ÷ 2^20, " MB")
end
