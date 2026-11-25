"""Driver of bench/obc_cross_regime_ho.jl: runs pydyn's higher-order OBC solver on a model and a shock sequence written by JDyn.

    python3 obc_cross_regime_ho_pydyn.py MODFILE INDIR OUTDIR

Environment (the cross-port runs): DYNARE_PREPROCESSOR, PYDYN_EXAMPLES, PYTHONPATH=<pydyn checkout>. INDIR holds `shocks.csv` (S x n_exo), `cfg.txt` (lines key=value: `order`, `horizon`, `check_ahead`, `cubature`, `qmc_level`, `regions` (all, 1, 4), `formulation` (regime or lcp), `failure_tolerance`,
`undecided_tolerance`, `milp_time_limit`) and, with cubature, JDyn's root `R.csv` (rows in NewsOBC order) and points `xi.csv` (dim x points), which replace pydyn's own so
that both ports average over the same points (plain text, 17 significant digits: exact).  OUTDIR gets the paths, wedges and the per-period counts as csv files.  The checkout is read only.
"""
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

import pydyn
from pydyn.obc_ho import HigherOrderNews
from pydyn.obc_regime_ho import HigherOrderRegime


def main(mod, indir, outdir):
    indir, outdir = Path(indir), Path(outdir)
    c = dict(line.strip().split("=", 1) for line in (indir / "cfg.txt").read_text().splitlines() if "=" in line)
    mod = Path(mod)
    d = Path(tempfile.mkdtemp(prefix="xport_"))
    for f in mod.parent.iterdir():
        if f.is_file() and f.suffix in (".mod", ".inc", ".mat", ".m", ".csv"):
            shutil.copy(f, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(d / mod.name, extra_args=("output=second",)))
    regions = c["regions"]
    regions = regions if regions == "all" else int(regions)
    cub = c["cubature"] == "true"
    kw = dict(order=int(c["order"]), horizon=int(c["horizon"]), check_ahead=int(c["check_ahead"]), cubature=cub, qmc_level=int(c["qmc_level"]),
              regions=regions, cubature_failure_tolerance=float(c["failure_tolerance"]), cubature_undecided_tolerance=float(c["undecided_tolerance"]),
              cubature_milp_time_limit=float(c["milp_time_limit"]))
    if c["formulation"] == "regime":
        h = HigherOrderRegime(m, **kw)
    else:
        h = HigherOrderNews(m, obc_formulation="lcp", **kw)
    if cub and (indir / "R.csv").exists():
        h.R, h.xi = np.loadtxt(indir / "R.csv", delimiter=",", ndmin=2), np.loadtxt(indir / "xi.csv", delimiter=",", ndmin=2)
        if hasattr(h, "_ro_cache"):
            h._ro_cache = None
    S = np.loadtxt(indir / "shocks.csv", delimiter=",", ndmin=2)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = h.simulate(S)
    sec = time.perf_counter() - t0
    outdir.mkdir(parents=True, exist_ok=True)
    for k in ("with_bound", "without_bound", "wedges"):
        np.savetxt(outdir / (k + ".csv"), np.asarray(r[k]), delimiter=",", fmt="%.17g")
    cnt = np.column_stack([np.asarray(r[k]) for k in ("relevant", "dropped", "dropped_a", "dropped_b", "dropped_c")])
    np.savetxt(outdir / "counts.csv", cnt, delimiter=",", fmt="%d")
    (outdir / "meta.txt").write_text(f"seconds={sec}\nwarnings={len(rec)}\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
