"""Driver of bench/obc_cross_regime_ho.jl: runs pydyn's higher-order OBC solver on a model and a shock sequence written by JDyn.

    python3 obc_cross_regime_ho_pydyn.py MODFILE INPUT.mat OUTPUT.mat

Environment (the cross-port runs): DYNARE_PREPROCESSOR, PYDYN_EXAMPLES, PYTHONPATH=<pydyn checkout>. INPUT.mat holds `shocks` (S x n_exo), the options
(`order`, `horizon`, `check_ahead`, `cubature`, `qmc_level`, `regions` ('all', '1', '4'), `formulation` ('regime' or 'lcp'), `failure_tolerance`,
`undecided_tolerance`, `milp_time_limit`) and, with cubature, JDyn's root `R` (rows in NewsOBC order) and points `xi` (dim x points), which replace pydyn's own so
that both ports average over the same points.  OUTPUT.mat gets the paths, wedges and the per-period counts.  The checkout is read only.
"""
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

import pydyn
from pydyn.obc_ho import HigherOrderNews
from pydyn.obc_regime_ho import HigherOrderRegime


def main(mod, inp, out):
    c = loadmat(inp, squeeze_me=True)
    mod = Path(mod)
    d = Path(tempfile.mkdtemp(prefix="xport_"))
    for f in mod.parent.iterdir():
        if f.is_file() and f.suffix in (".mod", ".inc", ".mat", ".m", ".csv"):
            shutil.copy(f, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(d / mod.name, extra_args=("output=second",)))
    regions = str(c["regions"])
    regions = regions if regions == "all" else int(regions)
    kw = dict(order=int(c["order"]), horizon=int(c["horizon"]), check_ahead=int(c["check_ahead"]), cubature=bool(c["cubature"]), qmc_level=int(c["qmc_level"]),
              regions=regions, cubature_failure_tolerance=float(c["failure_tolerance"]), cubature_undecided_tolerance=float(c["undecided_tolerance"]),
              cubature_milp_time_limit=float(c["milp_time_limit"]))
    if str(c["formulation"]) == "regime":
        h = HigherOrderRegime(m, **kw)
    else:
        h = HigherOrderNews(m, obc_formulation="lcp", **kw)
    if kw["cubature"] and "R" in c and np.size(c["R"]):
        h.R, h.xi = np.asarray(c["R"], float), np.asarray(c["xi"], float)
        if hasattr(h, "_ro_cache"):
            h._ro_cache = None
    S = np.atleast_2d(np.asarray(c["shocks"], float))
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = h.simulate(S)
    sec = time.perf_counter() - t0
    res = {k: np.asarray(r[k]) for k in ("with_bound", "without_bound", "wedges", "relevant", "dropped", "dropped_a", "dropped_b", "dropped_c")}
    res["seconds"] = sec
    res["warnings"] = "\n".join(str(w.message) for w in rec)
    savemat(out, res)


if __name__ == "__main__":
    main(*sys.argv[1:4])
